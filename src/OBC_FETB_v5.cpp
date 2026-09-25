// [[Rcpp::depends(Rcpp)]]
#include <Rcpp.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <limits>
#include <stdexcept>


// Include shared headers
#include "common/optimal_binning_common.h"
#include "common/bin_structures.h"

using namespace Rcpp;
using namespace OptimalBinning;

namespace {

// Category names that contain bin_separator. A bin label is its categories
// joined with the separator, and obwoe_apply() recovers the categories by
// splitting the label on it, so such a name is cut into pieces that are not
// categories (and a piece equal to another category claims it too). The
// binning itself is unaffected; the caller is warned that its labels are
// ambiguous. Checked over the distinct categories only, so the cost is O(k).
template <typename T>
inline const std::string& separator_key(const std::pair<const std::string, T>& kv) {
  return kv.first;
}

template <typename Container>
std::size_t count_separator_hits(const Container& categories, const std::string& sep,
                                 std::string& example) {
  std::size_t hits = 0;
  if (sep.empty()) return 0;
  for (const auto& item : categories) {
    const std::string& cat = separator_key(item);
    if (cat.find(sep) != std::string::npos) {
      if (hits == 0) example = cat;
      ++hits;
    }
  }
  return hits;
}

inline void warn_separator_hits(const std::string& sep, std::size_t hits,
                                const std::string& example) {
  if (hits == 0) return;
  Rcpp::warning("bin_separator \"%s\" occurs inside %d categor%s of the feature "
                "(e.g. \"%s\"): the bin labels are ambiguous and cannot be split "
                "back into categories. Choose a bin_separator that does not occur "
                "in the category names.",
                sep, hits, (hits == 1 ? "y" : "ies"), example);
}

} // namespace


/********************************************************************
 *  Categorical Optimal Binning with Fisher's Exact Test (v2)
 *  Author : OB - 2025-04-17
 *  Licence: MIT
 *
 *  Key changes relative to v1
 *    - Merge-selection criterion uses the HIGHEST p-value
 *    - WoE-monotonicity checked after *every* merge
 *    - Safer factorial cache & NA handling
 ********************************************************************/

namespace {

// -----------------------------------------------------------------
// Fisher's exact test cache (unordered_map keyed by 4-tuple)
// -----------------------------------------------------------------
class FisherTestCache {
private:
  struct Key {
    size_t a, b, c, d;
    bool operator==(const Key& o) const {
      return a == o.a && b == o.b && c == o.c && d == o.d;
    }
  };
  struct KeyHash {
    size_t operator()(const Key& k) const {
      size_t h1 = k.a ^ (k.b << 16) ^ (k.b >> 16);
      size_t h2 = k.c ^ (k.d << 16) ^ (k.d >> 16);
      return h1 ^ (h2 << 16) ^ (h2 >> 16);
    }
  };
  std::unordered_map<Key, double, KeyHash> cache;

public:
  FisherTestCache() { cache.reserve(1024); }

  // Returns false when the table is not cached.
  inline bool get(size_t a, size_t b, size_t c, size_t d, double& out) const {
    auto it = cache.find({a, b, c, d});
    if (it == cache.end()) return false;
    out = it->second;
    return true;
  }
  inline void set(size_t a, size_t b, size_t c, size_t d, double v) {
    cache[{a, b, c, d}] = v;
  }
};

} // namespace

// -----------------------------------------------------------------
// Main class
// -----------------------------------------------------------------
class OBC_FETB {
private:
  // ---------------- Input & hyper-parameters --------------------
  const std::vector<std::string>& feature;
  const std::vector<int>&         target;
  const size_t  min_bins;
  size_t        max_bins;
  const double  bin_cutoff;
  const double  convergence_threshold;
  const size_t  max_iterations;
  const std::string bin_separator;

  // ---------------- Internals -----------------------------------
  std::unordered_map<std::string, size_t> category_counts;
  std::unordered_map<std::string, size_t> category_pos_counts;
  std::vector<CategoricalBin> bins;
  size_t total_pos = 0, total_neg = 0;

  std::vector<double> log_factorials;
  FisherTestCache     fisher_cache;

  bool   converged  = false;
  bool   exhausted  = false; // max_iterations reached before the IV tolerance
  size_t iterations = 0;

  // ------------ Utility: extend factorial cache -----------------
  inline void extendLogFactorials(size_t n) {
    size_t old = log_factorials.size();
    log_factorials.resize(n + 1);
    for (size_t i = old; i <= n; ++i)
      log_factorials[i] = log_factorials[i - 1] + std::log(static_cast<double>(i));
  }

  // ------------ Fisher's exact test (two-sided) -----------------
  //
  // Natural log of the two-sided p-value of Fisher's exact test for the 2x2
  // table
  //
  //                 positives   negatives
  //     bin 1           a           b
  //     bin 2           c           d
  //
  // Under independence with all margins fixed, the positives of bin 1 follow
  // a hypergeometric law, P(X = x) = C(r1, x) C(r2, c1 - x) / C(n, c1), with
  // r1 = a + b, r2 = c + d, c1 = a + c, n = r1 + r2. The two-sided p-value is
  // the total probability of every table with those margins that is no more
  // likely than the observed one, sum_{x : P(x) <= P(a)} P(x); like R's
  // fisher.test(), "no more likely" allows a relative tolerance of 1e-7 so
  // that ties are not lost to rounding.
  //
  // This used to return P(a) itself -- the point probability of the observed
  // table -- as if it were the p-value. The two are not interchangeable: the
  // point probability shrinks with the sample size even for bins with
  // identical event rates, so the "highest p-value" rule favoured merging the
  // smallest bins rather than the most similar ones.
  //
  // The value is returned on the log scale (and capped at 0, i.e. p <= 1) so
  // that very small p-values stay comparable instead of underflowing to 0 and
  // tying.
  double fisherLogPValue(size_t a, size_t b, size_t c, size_t d) {
    double cached;
    if (fisher_cache.get(a, b, c, d, cached)) return cached;

    const size_t r1 = a + b, r2 = c + d, c1 = a + c;
    const size_t n = r1 + r2;
    if (n >= log_factorials.size()) extendLogFactorials(n);
    const std::vector<double>& lf = log_factorials;

    // Support of X
    const size_t lo = (c1 > r2) ? c1 - r2 : 0;
    const size_t hi = std::min(r1, c1);

    // log P(x) up to the additive constant log_const
    auto log_kernel = [&](size_t x) {
      return -(lf[x] + lf[r1 - x] + lf[c1 - x] + lf[(r2 + x) - c1]);
    };
    const double log_const = lf[r1] + lf[r2] + lf[c1] + lf[n - c1] - lf[n];

    const double lk_obs = log_kernel(a);
    const double lk_thr = lk_obs + std::log1p(1e-7);

    // sum_{x : P(x) <= P(a)(1 + 1e-7)} P(x) / P(a); the observed table is
    // included, so the sum is >= 1 and its log is well defined.
    double rel_sum = 0.0;
    for (size_t x = lo; x <= hi; ++x) {
      const double lk = log_kernel(x);
      if (lk <= lk_thr) rel_sum += std::exp(lk - lk_obs);
    }

    double log_p = log_const + lk_obs + std::log(rel_sum);
    if (!(log_p < 0.0)) log_p = 0.0;
    fisher_cache.set(a, b, c, d, log_p);
    return log_p;
  }

  // ------------ WoE & IV ----------------------------------------
  inline void computeWoeIv(CategoricalBin& bin) const {
    double pos = static_cast<double>(bin.count_pos);
    double neg = static_cast<double>(bin.count_neg);
    if (pos <= EPSILON || neg <= EPSILON) { bin.woe = bin.iv = 0.0; return; }

    double dist_pos = pos / static_cast<double>(total_pos);
    double dist_neg = neg / static_cast<double>(total_neg);
    bin.woe = std::log(dist_pos / dist_neg);
    bin.iv  = (dist_pos - dist_neg) * bin.woe;
  }

  // ------------ Pre-processing ----------------------------------
  void preprocess() {
    category_counts.reserve(std::min(feature.size() / 4, size_t(1024)));
    category_pos_counts.reserve(category_counts.bucket_count());

    for (size_t i = 0; i < feature.size(); ++i) {
      const std::string& cat = feature[i];
      category_counts[cat]++;
      if (target[i] == 1) { category_pos_counts[cat]++; total_pos++; }
      else                 { total_neg++; }
    }
  }

  // Single-category bin with its WoE/IV
  CategoricalBin makeBin(const std::string& cat, size_t cnt, size_t cntp) const {
    CategoricalBin bin;
    bin.categories.push_back(cat);
    bin.count_pos = static_cast<int>(cntp);
    bin.count_neg = static_cast<int>(cnt - cntp);
    bin.update_count();
    computeWoeIv(bin);
    return bin;
  }

  // ------------ Initialise bins (rare categories together) ------
  void initialiseBins() {
    bins.clear();
    const double cutoff_cnt = bin_cutoff * static_cast<double>(feature.size());

    std::vector<std::pair<std::string,size_t>> sorted;
    sorted.reserve(category_counts.size());
    for (auto& kv : category_counts) sorted.emplace_back(kv.first, kv.second);
    std::sort(sorted.begin(), sorted.end(),
              [](auto& a, auto& b){ return a.second > b.second; });

    // (category, count, positives) of the categories below the cutoff
    struct Rare { const std::string* cat; size_t cnt; size_t cntp; };
    std::vector<Rare> rare_cats;

    for (auto& kv : sorted) {
      const std::string& cat = kv.first;
      size_t cnt  = kv.second;
      size_t cntp = category_pos_counts[cat];

      if (static_cast<double>(cnt) < cutoff_cnt) {
        rare_cats.push_back({&cat, cnt, cntp});
      } else {
        bins.push_back(makeBin(cat, cnt, cntp));
      }
    }

    if (!rare_cats.empty()) {
      if (bins.size() + 1 >= min_bins) {
        // All rare categories pooled into one bin.
        CategoricalBin rare;
        for (const Rare& r : rare_cats) {
          rare.categories.push_back(*r.cat);
          rare.count_pos += static_cast<int>(r.cntp);
          rare.count_neg += static_cast<int>(r.cnt - r.cntp);
        }
        rare.update_count();
        computeWoeIv(rare);
        bins.push_back(std::move(rare));
      } else {
        // Pooling every rare category into one bin would leave fewer than
        // min_bins bins -- on a high-cardinality feature where every level
        // is below bin_cutoff (e.g. 500 levels at 0.2% each) the result was a
        // single bin holding the whole sample, with WoE = IV = 0. Instead,
        // order the rare categories by event rate and pool neighbours only
        // until each pool reaches bin_cutoff, so similar categories end up
        // together and every pool but the last meets the cutoff.
        std::vector<Rare> by_rate = rare_cats;
        std::stable_sort(by_rate.begin(), by_rate.end(),
                         [](const Rare& x, const Rare& y) {
                           return static_cast<double>(x.cntp) / static_cast<double>(x.cnt) <
                             static_cast<double>(y.cntp) / static_cast<double>(y.cnt);
                         });
        std::vector<CategoricalBin> pools;
        CategoricalBin cur;
        for (const Rare& r : by_rate) {
          cur.categories.push_back(*r.cat);
          cur.count_pos += static_cast<int>(r.cntp);
          cur.count_neg += static_cast<int>(r.cnt - r.cntp);
          cur.update_count();
          if (static_cast<double>(cur.count) >= cutoff_cnt) {
            computeWoeIv(cur);
            pools.push_back(std::move(cur));
            cur = CategoricalBin();
          }
        }
        if (!cur.categories.empty()) {
          computeWoeIv(cur);
          pools.push_back(std::move(cur));
        }

        if (bins.size() + pools.size() >= min_bins) {
          for (auto& p : pools) bins.push_back(std::move(p));
        } else {
          // Even the pools are too few (a very large bin_cutoff): start from
          // one bin per category and let the Fisher merge do the grouping.
          for (const Rare& r : rare_cats) bins.push_back(makeBin(*r.cat, r.cnt, r.cntp));
        }
      }
    }

    std::sort(bins.begin(), bins.end(),
              [](const CategoricalBin& a, const CategoricalBin& b){ return a.woe < b.woe; });
  }

  // ------------ Merge two bins ----------------------------------
  inline void mergeBins(size_t i) {
    // merge bins[i] with bins[i+1]
    if (i+1 >= bins.size()) return;
    bins[i].merge_with(bins[i+1]);
    computeWoeIv(bins[i]);
    bins.erase(bins.begin() + static_cast<std::ptrdiff_t>(i) + 1);
  }

  // ------------ Local monotonicity fix --------------------------
  //
  // Merges adjacent bins whose WoE decreases, scanning from start_idx. After
  // a merge the merged bin is compared with its LEFT neighbour as well: its
  // WoE lies between the two it replaced, so it can now be below the bin
  // before it. The loop used to step back to the same index instead (the
  // "re-check backward" comment notwithstanding), which left a decreasing
  // pair to the left unfixed.
  //
  // min_bins takes precedence, as documented ("the algorithm will not merge
  // below this threshold"): the repair stops once min_bins bins remain.
  void enforceLocalMonotonicity(size_t start_idx=0) {
    size_t i = start_idx;
    while (bins.size() > min_bins && i + 1 < bins.size()) {
      if (bins[i].woe > bins[i+1].woe + EPSILON) {
        mergeBins(i);
        if (i > 0) --i;
      } else {
        ++i;
      }
    }
  }

  // ------------ Core merge loop ---------------------------------
  //
  // max_bins is a hard constraint: merging continues until it is met. It
  // used to stop at max_iterations instead, returning e.g. 89 bins for
  // max_bins = 5 when max_iterations was small. Every merge removes a bin, so
  // the loop ends after at most bins.size() - max_bins merges regardless;
  // max_iterations now decides only the `converged` flag, which is FALSE when
  // the cap was reached before the IV tolerance was met (as before).
  void mergeLoop() {
    iterations = 0;
    double prev_iv = -1.0;

    while (bins.size() > max_bins) {
      if (iterations == max_iterations && !converged) exhausted = true;

      // choose the adjacent pair with the HIGHEST p-value (= most similar)
      double best_lp = -std::numeric_limits<double>::infinity();
      size_t best_i  = 0;

      for (size_t i = 0; i+1 < bins.size(); ++i) {
        double lp = fisherLogPValue(static_cast<size_t>(bins[i].count_pos),
                                    static_cast<size_t>(bins[i].count_neg),
                                    static_cast<size_t>(bins[i+1].count_pos),
                                    static_cast<size_t>(bins[i+1].count_neg));
        if (lp > best_lp) { best_lp = lp; best_i = i; }
      }
      mergeBins(best_i);
      enforceLocalMonotonicity(best_i == 0 ? 0 : best_i-1);

      // convergence (IV change)
      double iv_total = 0.0;
      for (const auto& b : bins) iv_total += b.iv;
      if (std::fabs(iv_total - prev_iv) < convergence_threshold) {
        // The IV has settled. This used to break out of the loop, which
        // abandoned the descent to max_bins; max_bins is a documented user
        // constraint, so we record the convergence and keep merging by the
        // same criterion.
        converged = true;
      }
      prev_iv = iv_total;
      ++iterations;
    }
  }

  // ------------ Join categories for R output --------------------
  std::string joinCats(const std::vector<std::string>& cats) const {
    if (cats.empty()) return "";
    if (cats.size() == 1) return cats[0];
    size_t len=0; for (auto& c:cats) len+=c.size();
    len += bin_separator.size() * (cats.size()-1);
    std::string out; out.reserve(len);
    out = cats[0];
    for (size_t i=1;i<cats.size();++i){ out+=bin_separator; out+=cats[i]; }
    return out;
  }

public:
  OBC_FETB(const std::vector<std::string>& feature_,
           const std::vector<int>&         target_,
           size_t  min_bins_=3,
           size_t  max_bins_=5,
           double  bin_cutoff_=0.05,
           size_t  /* max_n_prebins: unused */ = 20,
           double  convergence_threshold_=1e-6,
           size_t  max_iterations_=1000,
           const std::string& bin_sep="%;%")
    : feature(feature_), target(target_),
      min_bins(min_bins_), max_bins(max_bins_),
      bin_cutoff(bin_cutoff_),
      convergence_threshold(convergence_threshold_),
      max_iterations(max_iterations_),
      bin_separator(bin_sep)
  {
    // quick input sanity
    if (feature.empty() || target.empty())
      throw std::invalid_argument("Feature/target cannot be empty.");
    if (feature.size()!=target.size())
      throw std::invalid_argument("Feature and target must match in length.");
    if (min_bins<2 || max_bins<min_bins)
      throw std::invalid_argument("Invalid min_bins / max_bins.");
    if (!(bin_cutoff>0.0 && bin_cutoff<1.0))
      throw std::invalid_argument("bin_cutoff must be in (0,1).");

    // factorial cache init (0! ... 1000!)
    log_factorials.resize(1001);
    log_factorials[0]=0.0;
    for (size_t i=1;i<log_factorials.size();++i)
      log_factorials[i]=log_factorials[i-1]+std::log(static_cast<double>(i));
  }

  // ---------------- Public API ----------------------------------
  /// Number of categories whose name contains bin_separator (and one of them)
  std::size_t separatorHits(std::string& example) const {
    return count_separator_hits(category_counts, bin_separator, example);
  }

  void fit() {
    preprocess();
    initialiseBins();
    max_bins = std::min(max_bins, bins.size());
    if (bins.size() > max_bins) mergeLoop();
    enforceLocalMonotonicity();
    // Reaching max_bins is a valid stopping state; only running out of
    // max_iterations before the IV tolerance was met leaves it FALSE.
    converged = !exhausted;
  }

  List results() const {
    size_t n = bins.size();
    NumericVector id(n), w(n), iv(n);
    CharacterVector lbl(n);
    IntegerVector cnt(n), cntp(n), cntn(n);

    for (size_t i=0;i<n;++i) {
      id[i]   = static_cast<double>(i + 1);
      lbl[i]  = joinCats(bins[i].categories);
      w[i]    = bins[i].woe;
      iv[i]   = bins[i].iv;
      cnt[i]  = bins[i].count;
      cntp[i] = bins[i].count_pos;
      cntn[i] = bins[i].count_neg;
    }
    return List::create(
      _["id"]=id, _["bin"]=lbl, _["woe"]=w, _["iv"]=iv,
        _["count"]=cnt, _["count_pos"]=cntp, _["count_neg"]=cntn,
          _["converged"]=converged, _["iterations"]=static_cast<int>(iterations)
    );
  }
};

// -----------------------------------------------------------------
// R interface
// -----------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_fetb(
   Rcpp::IntegerVector   target,
   Rcpp::CharacterVector feature,
   int    min_bins              = 3,
   int    max_bins              = 5,
   double bin_cutoff            = 0.05,
   int    max_n_prebins         = 20,
   double convergence_threshold = 1e-6,
   int    max_iterations        = 1000,
   std::string bin_separator    = "%;%")
{
 // ----------- Fast early checks --------------------------------
 if (feature.size() == 0 || target.size() == 0)
   Rcpp::stop("Feature and target cannot be empty.");
 if (feature.size() != target.size())
   Rcpp::stop("Feature and target must have the same length.");
 // Negative values would wrap around in the size_t parameters below.
 if (min_bins < 2 || max_bins < min_bins)
   Rcpp::stop("Invalid min_bins / max_bins.");
 if (max_iterations <= 0)
   Rcpp::stop("max_iterations must be positive.");

 // ----------- Convert to STL containers ------------------------
 std::vector<std::string> feat; feat.reserve(static_cast<size_t>(feature.size()));
 std::vector<int>         tar;  tar.reserve(static_cast<size_t>(target.size()));

 for (R_xlen_t i = 0; i < feature.size(); ++i) {
   feat.push_back( (feature[i]==NA_STRING) ? "__NA__"
                     : Rcpp::as<std::string>(feature[i]) );

   if (IntegerVector::is_na(target[i]))
     Rcpp::stop("Target cannot contain NA.");
   // Anything but 0/1 used to be silently counted as a negative.
   if (target[i] != 0 && target[i] != 1)
     Rcpp::stop("Target must be binary (0/1).");
   tar.push_back(target[i]);
 }
 // A single-class target used to return bins with WoE = IV = 0; every other
 // categorical algorithm rejects it.
 if (std::find(tar.begin(), tar.end(), 0) == tar.end() ||
     std::find(tar.begin(), tar.end(), 1) == tar.end())
   Rcpp::stop("Target must contain both 0 and 1 values.");

 // ----------- Run algorithm ------------------------------------
 try {
   OBC_FETB ob(
       feat, tar,
       static_cast<size_t>(min_bins),
       static_cast<size_t>(max_bins),
       bin_cutoff,
       static_cast<size_t>(std::max(max_n_prebins, 0)),
       convergence_threshold,
       static_cast<size_t>(max_iterations),
       bin_separator
   );
   ob.fit();
   Rcpp::List res = ob.results();
   std::string example;
   warn_separator_hits(bin_separator, ob.separatorHits(example), example);
   return res;
 } catch (const std::exception& e) {
   Rcpp::stop("Optimal binning failed: %s", e.what());
 }
}
