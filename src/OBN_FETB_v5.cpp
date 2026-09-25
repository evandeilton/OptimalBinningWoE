// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>
#include <sstream>
#include <cstdint>


// Include shared headers
#include "common/optimal_binning_common.h"
#include "common/bin_structures.h"

using namespace Rcpp;
using namespace OptimalBinning;


// -----------------------------------------------------------------------------
// Numerical Optimal Binning with Fisher's Exact Test (FETB)
//   * merges the pair of adjacent bins with the HIGHEST two-sided Fisher
//     p-value (the pair that is statistically most indistinguishable)
//   * keeps the WoE monotonic (ascending or descending) after every merge
//   * pure C++ (no calls back into R's fisher.test)
//   * keeps the public interface and output names identical to v0
// -----------------------------------------------------------------------------

namespace {

constexpr double FETB_EPS = 1e-12;   // numeric safety for WoE / monotonicity

// Relative tolerance R's fisher.test() uses when deciding which tables are
// "as or more extreme" than the observed one (relErr = 1 + 1e-7). In log space
// log(1 + 1e-7) is 1e-7 to double precision.
constexpr double FISHER_LOG_REL_TOL = 1e-7;

// Terms whose log-probability sits this far below the observed table's
// log-probability contribute less than exp(-50) ~ 2e-22 each; because the
// hypergeometric pmf is log-concave the tail beyond them is bounded by a short
// geometric series, so summation stops there with a relative error far below
// double precision on the final p-value.
constexpr double FISHER_LOG_NEGLIGIBLE = 50.0;

// -----------------------------------------------------------------------------
// log-factorial cache
// -----------------------------------------------------------------------------
class LogFactCache {
private:
  std::vector<double> lf_;                // log(i!) i = 0..N
public:
  explicit LogFactCache(size_t n = 1000) {
    lf_.resize(n + 1);
    lf_[0] = 0.0;
    for (size_t i = 1; i < lf_.size(); ++i)
      lf_[i] = lf_[i - 1] + std::log(static_cast<double>(i));
  }
  void ensure(size_t n) {
    if (n < lf_.size()) return;
    size_t old = lf_.size();
    lf_.resize(n + 1);
    for (size_t i = old; i <= n; ++i)
      lf_[i] = lf_[i - 1] + std::log(static_cast<double>(i));
  }
  double operator[](size_t i) const { return lf_[i]; }
};

// -----------------------------------------------------------------------------
// Two-sided Fisher exact test for the 2x2 table
//
//               target=1  target=0
//     bin i        a         b        r1 = a + b
//     bin i+1      c         d        r2 = c + d
//                  c1        c2       N
//
// Returns log(p) where p is the two-sided p-value with the same definition as
// R's fisher.test(): the total probability of all tables with the observed
// margins whose point probability does not exceed that of the observed table
// (up to a relative tolerance of 1e-7).
//
// The previous implementation returned the hypergeometric POINT probability
// of the observed table. That is not a p-value: it shrinks as the bins grow
// even when the two bins have identical event rates, so the merge criterion
// systematically preferred small bins over similar ones.
//
// Working in log space keeps the criterion meaningful for large tables, where
// every p-value below ~1e-308 would otherwise underflow to the same 0.
// -----------------------------------------------------------------------------
double fisher_two_sided_log_p(std::int64_t a, std::int64_t b,
                              std::int64_t c, std::int64_t d, LogFactCache& lf) {
  typedef std::int64_t i64;
  const i64 r1 = a + b, r2 = c + d, c1 = a + c, c2 = b + d;
  const i64 N = r1 + r2;
  if (N <= 0 || r1 == 0 || r2 == 0 || c1 == 0 || c2 == 0) return 0.0; // p = 1
  lf.ensure(static_cast<size_t>(N));

  const i64 lo = std::max<i64>(0, c1 - r2);
  const i64 hi = std::min<i64>(r1, c1);
  const double konst = lf[static_cast<size_t>(r1)] + lf[static_cast<size_t>(r2)] +
                       lf[static_cast<size_t>(c1)] + lf[static_cast<size_t>(c2)] -
                       lf[static_cast<size_t>(N)];
  auto logp = [&](i64 x) {
    return konst - lf[static_cast<size_t>(x)] - lf[static_cast<size_t>(r1 - x)] -
           lf[static_cast<size_t>(c1 - x)] - lf[static_cast<size_t>(r2 - c1 + x)];
  };

  const double base = logp(a);
  const double limit = base + FISHER_LOG_REL_TOL;

  // A mode of the hypergeometric distribution, clamped to the support. The pmf
  // is non-decreasing on [lo, mode] and non-increasing on [mode, hi], so the
  // qualifying tables form a prefix [lo, L] and a suffix [U, hi]. L and U are
  // located by binary search and each tail is summed outward from its largest
  // term until the terms become negligible.
  i64 mode = static_cast<i64>(
    std::floor((static_cast<double>(r1) + 1.0) * (static_cast<double>(c1) + 1.0) /
               (static_cast<double>(N) + 2.0)));
  mode = std::max(lo, std::min(hi, mode));

  double sum = 0.0;          // sum of exp(logp(x) - base) over qualifying x
  i64 L = lo - 1;            // last qualifying point of the left side
  if (logp(lo) <= limit) {
    i64 l = lo, r = mode;    // invariant: logp(l) <= limit
    while (l < r) {
      const i64 m = l + (r - l + 1) / 2;
      if (logp(m) <= limit) l = m; else r = m - 1;
    }
    L = l;
    for (i64 x = L; x >= lo; --x) {
      const double t = logp(x) - base;
      if (t < -FISHER_LOG_NEGLIGIBLE) break;
      sum += std::exp(t);
    }
  }

  const i64 start = std::max(mode, L + 1);
  if (start <= hi && logp(hi) <= limit) {
    i64 l = start, r = hi;   // invariant: logp(r) <= limit
    while (l < r) {
      const i64 m = l + (r - l) / 2;
      if (logp(m) <= limit) r = m; else l = m + 1;
    }
    for (i64 x = r; x <= hi; ++x) {
      const double t = logp(x) - base;
      if (t < -FISHER_LOG_NEGLIGIBLE) break;
      sum += std::exp(t);
    }
  }

  // The observed table always qualifies, so sum >= 1 and the log is finite.
  return std::min(base + std::log(sum), 0.0);
}

// -----------------------------------------------------------------------------
// Helper - compute WoE + IV in place
// -----------------------------------------------------------------------------
void calc_woe_iv(const std::vector<int>& pos,
                 const std::vector<int>& neg,
                 std::vector<double>& woe,
                 std::vector<double>& iv) {

  const double totPos = std::accumulate(pos.begin(), pos.end(), 0.0);
  const double totNeg = std::accumulate(neg.begin(), neg.end(), 0.0);
  const double nb = static_cast<double>(pos.size());

  woe.resize(pos.size());
  iv .resize(pos.size());

  for (size_t i = 0; i < pos.size(); ++i) {
    double dp = (pos[i] + 0.5) / (totPos + 0.5 * nb);  // add-0.5 smoothing
    double dn = (neg[i] + 0.5) / (totNeg + 0.5 * nb);
    dp = std::max(dp, FETB_EPS); dn = std::max(dn, FETB_EPS);

    woe[i] = std::log(dp / dn);
    iv [i] = (dp - dn) * woe[i];
  }
}

} // namespace

// -----------------------------------------------------------------------------
// Class encapsulating the numeric FETB
// -----------------------------------------------------------------------------
class OptimalBinningNumericFETB {
private:
  // Input (NaN observations removed)
  std::vector<int>    y_;
  std::vector<double> x_;
  const int    min_bins_, max_bins_;
  const size_t max_prebins_;
  const size_t max_iter_;

  // Working
  std::vector<double> edges_;       // -inf ... +inf
  std::vector<int>    cnt_, pos_, neg_;
  std::vector<double> woe_, iv_;
  std::vector<double> pair_logp_;   // log two-sided p-value of (i, i+1)
  size_t iterations_ = 0;
  bool   converged_  = true;

  LogFactCache lf_;

  // ---------------------------------------------------------------------------
  // build initial pre-bins (equal-frequency quantiles, at most max_prebins_)
  //
  // The k-th candidate cut is the order statistic at position floor(k*n/P),
  // k = 1..P-1, which yields at most P bins. The former step of
  // floor(n / P) produced up to 2P-1 pre-bins whenever n was not a multiple
  // of P (e.g. 39 pre-bins for n = 39, P = 20). A cut equal to the sample
  // maximum is skipped: it would only create an empty last bin.
  // ---------------------------------------------------------------------------
  void make_prebins() {
    std::vector<double> xs = x_;
    std::sort(xs.begin(), xs.end());

    edges_.clear();
    edges_.push_back(-std::numeric_limits<double>::infinity());

    const size_t n = xs.size();
    const size_t P = max_prebins_;
    const size_t q = n / P, r = n % P;
    const double xmax = xs.back();
    for (size_t k = 1; k < P; ++k) {
      const size_t i = k * q + (k * r) / P;       // floor(k * n / P), no overflow
      if (i >= n) break;
      const double e = xs[i];
      if (e < xmax && e > edges_.back()) edges_.push_back(e);
    }
    edges_.push_back(std::numeric_limits<double>::infinity());
  }

  // ---------------------------------------------------------------------------
  // fill counts for current edge set
  // ---------------------------------------------------------------------------
  void fill_counts() {
    const size_t B = edges_.size() - 1;
    cnt_.assign(B, 0); pos_.assign(B, 0); neg_.assign(B, 0);

    for (size_t i = 0; i < x_.size(); ++i) {
      double v = x_[i];
      // Bins are right-closed (a, b], as the emitted labels state, so a value
      // sitting exactly on an edge belongs to the bin BELOW it: lower_bound
      // (first edge >= v), not upper_bound. The index is computed as a signed
      // value first: subtracting 1 from position 0 would wrap a size_t around
      // and then clamp to the LAST bin instead of the first.
      std::ptrdiff_t pos =
        std::lower_bound(edges_.begin(), edges_.end(), v) - edges_.begin() - 1;
      if (pos < 0) pos = 0;
      size_t b = std::min(static_cast<size_t>(pos), B - 1);
      ++cnt_[b];
      if (y_[i]) ++pos_[b]; else ++neg_[b];
    }
  }

  double pair_log_p(size_t i) {
    return fisher_two_sided_log_p(pos_[i], neg_[i], pos_[i + 1], neg_[i + 1], lf_);
  }

  void build_pair_cache() {
    pair_logp_.assign(cnt_.size() > 0 ? cnt_.size() - 1 : 0, 0.0);
    for (size_t i = 0; i + 1 < cnt_.size(); ++i) pair_logp_[i] = pair_log_p(i);
  }

  // ---------------------------------------------------------------------------
  // merge two adjacent bins (i and i+1); only the pairs touching the merged
  // bin change, so the p-value cache is updated for those two pairs only.
  // ---------------------------------------------------------------------------
  void merge_bins(size_t i) {
    edges_.erase(edges_.begin() + static_cast<std::ptrdiff_t>(i) + 1);
    cnt_[i] += cnt_[i + 1]; pos_[i] += pos_[i + 1]; neg_[i] += neg_[i + 1];
    cnt_.erase(cnt_.begin() + static_cast<std::ptrdiff_t>(i) + 1);
    pos_.erase(pos_.begin() + static_cast<std::ptrdiff_t>(i) + 1);
    neg_.erase(neg_.begin() + static_cast<std::ptrdiff_t>(i) + 1);

    if (pair_logp_.size() == cnt_.size()) {       // cache is live
      pair_logp_.erase(pair_logp_.begin() + static_cast<std::ptrdiff_t>(i));
      if (i > 0) pair_logp_[i - 1] = pair_log_p(i - 1);
      if (i + 1 < cnt_.size()) pair_logp_[i] = pair_log_p(i);
    }
  }

  // ---------------------------------------------------------------------------
  // enforce monotone WoE by local merges
  // ---------------------------------------------------------------------------
  void enforce_monotone() {
    bool changed = true;
    while (changed && cnt_.size() > static_cast<size_t>(min_bins_)) {
      calc_woe_iv(pos_, neg_, woe_, iv_);
      changed = false;
      // The trend is that of the first adjacent pair whose WoE differs by
      // more than FETB_EPS; any later pair moving the other way is merged.
      // It used to be read from bins 0 and 1 unconditionally, so when those
      // two had EQUAL WoE (e.g. identical event rates) the trend defaulted to
      // "descending" and every increase after them was merged away. The trend
      // is re-read on every pass; the loop terminates because every pass
      // either merges or stops.
      int trend = 0;
      for (size_t i = 0; i + 1 < woe_.size(); ++i) {
        if (!((woe_[i] > woe_[i + 1] + FETB_EPS) || (woe_[i] < woe_[i + 1] - FETB_EPS)))
          continue;
        if (trend == 0) {
          trend = (woe_[i + 1] > woe_[i]) ? 1 : -1;
          continue;
        }
        if ((trend > 0 && woe_[i] > woe_[i + 1]) ||
            (trend < 0 && woe_[i] < woe_[i + 1])) {
          merge_bins(i);
          changed = true;
          break;
        }
      }
    }
  }

  // ---------------------------------------------------------------------------
  // main Fisher merge loop
  //
  // Merging continues until the bin count reaches max_bins (or max_iterations
  // is exhausted). The loop used to stop as soon as the total IV changed by
  // less than convergence_threshold, which returned MORE than max_bins bins
  // whenever two consecutive merges happened to leave the IV unchanged.
  // ---------------------------------------------------------------------------
  void fisher_merge_loop() {
    build_pair_cache();
    while (cnt_.size() > static_cast<size_t>(max_bins_) &&
           iterations_ < max_iter_) {

      // choose pair with highest p-value (most similar); first one on ties
      size_t best_i = 0;
      double best_lp = -std::numeric_limits<double>::infinity();
      for (size_t i = 0; i < pair_logp_.size(); ++i) {
        if (pair_logp_[i] > best_lp) { best_lp = pair_logp_[i]; best_i = i; }
      }
      merge_bins(best_i);

      enforce_monotone();  // keep WoE monotone throughout
      ++iterations_;
    }
    converged_ = cnt_.size() <= static_cast<size_t>(max_bins_);
  }

public:
  OptimalBinningNumericFETB(const NumericVector& y,
                            const NumericVector& x,
                            int  min_bins, int  max_bins,
                            size_t max_prebins,
                            size_t max_iter)
    : min_bins_(min_bins), max_bins_(max_bins),
      max_prebins_(max_prebins), max_iter_(max_iter) {

    if (y.size() != x.size())
      stop("target and feature must have equal length.");
    if (min_bins_ < 2 || max_bins_ < min_bins_)
      stop("invalid min_bins / max_bins.");

    // Validate the target on the doubles R handed over: converting NaN (an NA
    // target) to int is undefined behaviour.
    const R_xlen_t n = x.size();
    y_.reserve(static_cast<size_t>(n));
    x_.reserve(static_cast<size_t>(n));
    for (R_xlen_t i = 0; i < n; ++i) {
      const double yi = y[i];
      if (!(yi == 0.0 || yi == 1.0))
        stop("target must be binary (0/1).");
      // Missing feature values are excluded, as in the other numerical
      // engines (bb, dmiv, ewb). They used to be sorted together with the
      // data -- undefined behaviour for std::sort -- and then counted in the
      // first bin.
      if (std::isnan(x[i])) continue;
      x_.push_back(x[i]);
      y_.push_back(yi == 1.0 ? 1 : 0);
    }
    if (x_.empty())
      stop("feature has no non-missing values.");

    make_prebins();
    fill_counts();
    enforce_monotone();    // first pass (rare in numeric, but safe)
    fisher_merge_loop();   // main optimisation
    calc_woe_iv(pos_, neg_, woe_, iv_);  // final stats
  }

  List results() const {
    const size_t B = cnt_.size();
    CharacterVector bins(B);
    NumericVector   id  (B), woe(B), iv(B), cnt(B), pos(B), neg(B);
    for (size_t i=0;i<B;++i) {
      std::ostringstream oss;
      oss << "(" << edges_[i] << "; " << edges_[i+1] << "]";
      bins[i] = oss.str();
      id  [i] = static_cast<double>(i + 1);
      woe [i] = woe_[i];
      iv  [i] = iv_[i];
      cnt [i] = cnt_[i];
      pos [i] = pos_[i];
      neg [i] = neg_[i];
    }
    NumericVector cut(edges_.size()-2);
    std::copy(edges_.begin()+1, edges_.end()-1, cut.begin());

    return List::create(
      _["id"]        = id,
      _["bin"]       = bins,
      _["woe"]       = woe,
      _["iv"]        = iv,
      _["count"]     = cnt,
      _["count_pos"] = pos,
      _["count_neg"] = neg,
      _["cutpoints"] = cut,
      _["converged"] = converged_,
      _["iterations"]= static_cast<int>(iterations_)
    );
  }
};

// -----------------------------------------------------------------------------
// R interface
// -----------------------------------------------------------------------------

// [[Rcpp::export]]
List optimal_binning_numerical_fetb(NumericVector target,
                                   NumericVector feature,
                                   int    min_bins              = 3,
                                   int    max_bins              = 5,
                                   int    max_n_prebins         = 20,
                                   double convergence_threshold = 1e-6,
                                   int    max_iterations        = 1000) {
 // convergence_threshold is accepted for interface compatibility. The merge
 // loop only runs while there are more than max_bins bins, so stopping it on
 // an IV tolerance could only ever return too many bins.
 (void)convergence_threshold;
 if (max_n_prebins < 2)
   stop("max_n_prebins must be >= 2.");
 if (max_iterations < 0)
   stop("max_iterations must be >= 0.");
 OptimalBinningNumericFETB ob(target, feature,
                              min_bins, max_bins,
                              static_cast<size_t>(max_n_prebins),
                              static_cast<size_t>(max_iterations));
 return ob.results();
}
