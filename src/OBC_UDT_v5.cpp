// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <numeric>
#include <functional>
#include <sstream>


// Include shared headers
#include "common/optimal_binning_common.h"
#include "common/bin_structures.h"

using namespace Rcpp;
using namespace OptimalBinning;


// Global constants for better readability and consistency
// Constant removed (uses shared definition)
static constexpr double LAPLACE_ALPHA = 0.5;  // Laplace smoothing parameter
// Category used for NA when this entry point is called directly; the R wrapper
// already maps NA to "NA", the token every categorical binner and
// ob_apply_woe_cat()/obwoe_sql() use (this used to be "N/A").
static constexpr const char* MISSING_VALUE = "NA";

// Namespace for utility functions
namespace utils {
// Safe logarithm function to avoid -Inf
inline double safe_log(double x) {
  return x > EPSILON ? std::log(x) : std::log(EPSILON);
}

// Laplace smoothing for more robust probability estimates
inline std::pair<double, double> smoothed_proportions(
    int positive_count, 
    int negative_count, 
    int total_positive, 
    int total_negative, 
    double alpha = LAPLACE_ALPHA) {
  
  // Apply Laplace (add-alpha) smoothing
  double smoothed_pos_rate = (positive_count + alpha) / (total_positive + alpha * 2);
  double smoothed_neg_rate = (negative_count + alpha) / (total_negative + alpha * 2);
  
  return {smoothed_pos_rate, smoothed_neg_rate};
}

// Calculate Weight of Evidence with Laplace smoothing
inline double calculate_woe(
    int positive_count, 
    int negative_count, 
    int total_positive, 
    int total_negative, 
    double alpha = LAPLACE_ALPHA) {
  
  auto [smoothed_pos_rate, smoothed_neg_rate] = smoothed_proportions(
    positive_count, negative_count, total_positive, total_negative, alpha);
  
  return safe_log(smoothed_pos_rate / smoothed_neg_rate);
}

// Calculate Information Value with Laplace smoothing
inline double calculate_iv(
    int positive_count, 
    int negative_count, 
    int total_positive, 
    int total_negative, 
    double alpha = LAPLACE_ALPHA) {
  
  auto [smoothed_pos_rate, smoothed_neg_rate] = smoothed_proportions(
    positive_count, negative_count, total_positive, total_negative, alpha);
  
  double woe = safe_log(smoothed_pos_rate / smoothed_neg_rate);
  return (smoothed_pos_rate - smoothed_neg_rate) * woe;
}

// Calculate Jensen-Shannon divergence between two bins
inline double calculate_divergence(
    int bin1_pos, int bin1_neg, 
    int bin2_pos, int bin2_neg, 
    int total_pos, int total_neg) {
  
  // Jensen-Shannon divergence (symmetric KL divergence)
  auto [p1, n1] = smoothed_proportions(bin1_pos, bin1_neg, total_pos, total_neg);
  auto [p2, n2] = smoothed_proportions(bin2_pos, bin2_neg, total_pos, total_neg);
  
  // Average proportions
  double p_avg = (p1 + p2) / 2;
  double n_avg = (n1 + n2) / 2;
  
  // KL(P1 || P_avg) + KL(P2 || P_avg)
  double div_p1 = p1 > EPSILON ? p1 * safe_log(p1 / p_avg) : 0;
  double div_n1 = n1 > EPSILON ? n1 * safe_log(n1 / n_avg) : 0;
  double div_p2 = p2 > EPSILON ? p2 * safe_log(p2 / p_avg) : 0;
  double div_n2 = n2 > EPSILON ? n2 * safe_log(n2 / n_avg) : 0;
  
  return (div_p1 + div_n1 + div_p2 + div_n2) / 2;
}

// Join vector of categories with uniqueness checking
inline std::string join_categories(const std::vector<std::string>& categories, 
                                   const std::string& separator) {
  if (categories.empty()) return "";
  if (categories.size() == 1) return categories[0];
  
  // Ensure uniqueness
  std::unordered_set<std::string> unique_cats;
  std::vector<std::string> unique_vec;
  unique_vec.reserve(categories.size());
  
  for (const auto& cat : categories) {
    if (unique_cats.insert(cat).second) {
      unique_vec.push_back(cat);
    }
  }
  
  // Join with separator
  std::ostringstream result;
  result << unique_vec[0];
  for (size_t i = 1; i < unique_vec.size(); ++i) {
    result << separator << unique_vec[i];
  }
  
  return result.str();
}
}

// Greedy similarity merging shared by the optimisation loops.
//
// Repeatedly merges the pair of bins (i before j in the current order) that
// minimises  divergence_from(i, j) * (j directly after i ? 0.95 : 1),  the
// first such pair in (i, j) order on ties -- the rule of the former full
// scan -- until only 'target' bins remain. The former code rescanned all
// O(B^2) pairs (four logarithms each) after every merge, O(B^3) in total:
// minutes to hours once a few thousand categories passed bin_cutoff. Here
// every bin keeps its best partner among the bins after it; after a merge
// only the merged bin, the bins whose partner or neighbour changed, and the
// candidate pairs with the merged bin are re-evaluated. The pairs chosen are
// exactly those of the full scan.
//
// keep_woe_order: after each merge the bins are re-sorted by WoE with
// std::sort, as the former code did (same permutation, ties included).
static void greedy_similarity_merge(std::vector<CategoricalBin>& bins, size_t target,
                                    int total_pos, int total_neg, bool keep_woe_order) {
  const size_t B = bins.size();
  if (B <= target || B < 2) return;
  const size_t NONE = static_cast<size_t>(-1);
  const double NO_SCORE = std::numeric_limits<double>::max();

  std::vector<size_t> order(B), pos(B);
  for (size_t i = 0; i < B; ++i) { order[i] = i; pos[i] = i; }
  std::vector<double> best_score(B, NO_SCORE);
  std::vector<size_t> best_j(B, NONE);

  auto score = [&](size_t x, size_t y) {  // requires pos[x] < pos[y]
    double div = bins[x].divergence_from(bins[y], total_pos, total_neg);
    if (pos[y] == pos[x] + 1) {
      div *= 0.95;
    }
    return div;
  };
  auto rescan = [&](size_t x) {
    best_score[x] = NO_SCORE;
    best_j[x] = NONE;
    for (size_t p = pos[x] + 1; p < order.size(); ++p) {
      const size_t y = order[p];
      const double s = score(x, y);
      if (s < best_score[x]) {
        best_score[x] = s;
        best_j[x] = y;
      }
    }
  };
  for (size_t i = 0; i < B; ++i) rescan(i);

  std::vector<char> dirty(B, 0);
  while (order.size() > target && order.size() >= 2) {
    // Global choice: first minimum in position order
    size_t a = NONE, b = NONE;
    double best = NO_SCORE;
    for (size_t p = 0; p < order.size(); ++p) {
      const size_t x = order[p];
      if (best_j[x] != NONE && best_score[x] < best) {
        best = best_score[x];
        a = x;
        b = best_j[x];
      }
    }
    if (a == NONE) {  // no finite score: the full scan merged positions 0 and 1
      a = order[0];
      b = order[1];
    }

    const size_t pa = pos[a], pb = pos[b];
    std::vector<size_t> touched;
    if (pa > 0) touched.push_back(order[pa - 1]);
    if (pb > 0) touched.push_back(order[pb - 1]);

    bins[a].merge_with(bins[b]);
    bins[a].calculate_metrics(total_pos, total_neg);

    order.erase(order.begin() + static_cast<std::ptrdiff_t>(pb));
    std::vector<size_t> permuted;
    if (keep_woe_order) {
      // Re-sort exactly as the former code did (std::sort on the bins in
      // their current order): sorting the ids with the same comparator
      // performs the same comparisons and hence the same permutation, ties
      // included. Bins other than the merged one whose relative order
      // changed (tied bins that std::sort permuted) are re-evaluated.
      std::vector<size_t> before;
      before.reserve(order.size());
      for (size_t x : order) if (x != a) before.push_back(x);
      std::sort(order.begin(), order.end(), [&bins](size_t x, size_t y) {
        return bins[x].woe < bins[y].woe;
      });
      std::vector<size_t> after;
      after.reserve(order.size());
      for (size_t x : order) if (x != a) after.push_back(x);
      size_t f = 0;
      while (f < after.size() && after[f] == before[f]) ++f;
      if (f < after.size()) {
        size_t l = after.size() - 1;
        while (l > f && after[l] == before[l]) --l;
        // Tied bins almost always have identical class counts, and the
        // scores depend on the counts and positions only: swapping such
        // bins is a pure relabelling of the positions, so their partner
        // data is carried over instead of being recomputed.
        bool relabel = true;
        for (size_t p = f; p <= l && relabel; ++p) {
          const CategoricalBin& u = bins[before[p]];
          const CategoricalBin& v = bins[after[p]];
          relabel = (u.count_pos == v.count_pos && u.count_neg == v.count_neg);
        }
        if (relabel) {
          std::vector<size_t> rel(B, NONE);
          std::vector<double> old_score(l - f + 1);
          std::vector<size_t> old_j(l - f + 1);
          for (size_t p = f; p <= l; ++p) {
            rel[before[p]] = after[p];
            old_score[p - f] = best_score[before[p]];
            old_j[p - f] = best_j[before[p]];
          }
          for (size_t p = f; p <= l; ++p) {
            best_score[after[p]] = old_score[p - f];
            best_j[after[p]] = old_j[p - f];
          }
          for (size_t x : order) {
            if (best_j[x] != NONE && rel[best_j[x]] != NONE) best_j[x] = rel[best_j[x]];
          }
          for (size_t& x : touched) {
            if (rel[x] != NONE) x = rel[x];
          }
        } else {
          permuted.assign(after.begin() + static_cast<std::ptrdiff_t>(f),
                          after.begin() + static_cast<std::ptrdiff_t>(l + 1));
        }
      }
    }
    for (size_t p = 0; p < order.size(); ++p) pos[order[p]] = p;
    best_j[b] = NONE;
    best_score[b] = NO_SCORE;

    if (pos[a] > 0) touched.push_back(order[pos[a] - 1]);
    touched.push_back(a);
    for (size_t p = 0; p < order.size(); ++p) {
      const size_t x = order[p];
      if (best_j[x] == a || best_j[x] == b) touched.push_back(x);
    }
    if (!permuted.empty()) {
      std::vector<char> in_perm(B, 0);
      for (size_t x : permuted) {
        in_perm[x] = 1;
        touched.push_back(x);
        if (pos[x] > 0) touched.push_back(order[pos[x] - 1]);
      }
      for (size_t p = 0; p < order.size(); ++p) {
        const size_t x = order[p];
        if (best_j[x] != NONE && in_perm[best_j[x]]) touched.push_back(x);
      }
    }
    for (size_t x : touched) dirty[x] = 1;
    dirty[b] = 0;

    for (size_t p = 0; p < order.size(); ++p) {
      const size_t x = order[p];
      if (dirty[x]) {
        rescan(x);
        dirty[x] = 0;
      } else if (p < pos[a]) {
        const double s = score(x, a);
        if (s < best_score[x] || (s == best_score[x] && best_j[x] != NONE && pos[a] < pos[best_j[x]])) {
          best_score[x] = s;
          best_j[x] = a;
        }
      }
    }
  }

  std::vector<CategoricalBin> kept;
  kept.reserve(order.size());
  for (size_t x : order) kept.push_back(std::move(bins[x]));
  bins = std::move(kept);
}

class OBC_UDT {
private:
  // Enhanced bin structure with uniqueness guarantee
  // Local CategoricalBin definition removed

  
  // Class parameters
  int min_bins_;
  int max_bins_;
  double bin_cutoff_;
  int max_n_prebins_;
  std::string bin_separator_;
  double convergence_threshold_;
  int max_iterations_;
  
  // Internal state
  std::vector<CategoricalBin> bins_;
  bool converged_;
  int iterations_;
  int total_pos_;
  int total_neg_;
  
  // Input validation with improved error messages
  void validate_inputs(const std::vector<std::string>& categories,
                       const std::vector<int>& pos,
                       const std::vector<int>& neg,
                       size_t n_obs) {
    if (n_obs == 0 || categories.empty()) {
      throw std::invalid_argument("Input vectors cannot be empty.");
    }

    int all_pos = 0, all_neg = 0;
    for (size_t c = 0; c < categories.size(); ++c) {
      all_pos += pos[c];
      all_neg += neg[c];
    }
    if (all_pos == 0 || all_neg == 0) {
      throw std::invalid_argument("Target must contain both 0 and 1 values.");
    }

    // Validate parameter ranges
    if (min_bins_ < 1) {
      throw std::invalid_argument("min_bins must be at least 1.");
    }
    if (max_bins_ < min_bins_) {
      throw std::invalid_argument("max_bins must be greater than or equal to min_bins.");
    }
    if (bin_cutoff_ <= 0 || bin_cutoff_ >= 1) {
      throw std::invalid_argument("bin_cutoff must be between 0 and 1 (exclusive).");
    }
    if (max_n_prebins_ < min_bins_) {
      throw std::invalid_argument("max_n_prebins must be at least min_bins.");
    }
  }

  // Initial binning with one bin per unique category. Keys are inserted in
  // order of first appearance, exactly as the former per-observation loop
  // did, so the map iteration order (and the resulting bin order) is kept.
  void initial_binning(const std::vector<std::string>& categories,
                       const std::vector<int>& pos,
                       const std::vector<int>& neg) {
    std::unordered_map<std::string, CategoricalBin> bin_map;
    total_pos_ = 0;
    total_neg_ = 0;

    for (size_t c = 0; c < categories.size(); ++c) {
      auto& bin = bin_map[categories[c]];
      bin.categories.push_back(categories[c]);
      bin.count_pos = pos[c];
      bin.count_neg = neg[c];
      bin.count = pos[c] + neg[c];
      total_pos_ += pos[c];
      total_neg_ += neg[c];
    }

    bins_.clear();
    bins_.reserve(bin_map.size());
    for (auto& pair : bin_map) {
      bins_.push_back(std::move(pair.second));
    }
  }

  // Pool the categories below bin_cutoff into a single "rare" bin.
  //
  // A rare category is kept as its own bin only when that is needed to reach
  // min_bins, and then the largest rare categories are kept. The old loop
  // walked the bins in ascending count order and kept every bin while fewer
  // than min_bins had been kept, i.e. it always kept the min_bins RAREST
  // categories as separate bins (even when there were plenty of frequent
  // ones), contradicting the documented pooling.
  void merge_low_frequency_bins() {
    int total_count = std::accumulate(bins_.begin(), bins_.end(), 0,
                                      [](int sum, const CategoricalBin& bin) { return sum + bin.count; });
    double cutoff_count = total_count * bin_cutoff_;

    // Sort bins by count (ascending): the rare bins form a prefix
    std::sort(bins_.begin(), bins_.end(), [](const CategoricalBin& a, const CategoricalBin& b) {
      return a.count < b.count;
    });

    size_t n_rare = 0;
    while (n_rare < bins_.size() && bins_[n_rare].count < cutoff_count) ++n_rare;
    const size_t n_freq = bins_.size() - n_rare;
    const size_t target_bins = static_cast<size_t>(std::max(min_bins_, 0));
    size_t keep_rare = 0;
    while (keep_rare < n_rare &&
           n_freq + keep_rare + (n_rare > keep_rare ? 1 : 0) < target_bins) {
      ++keep_rare;
    }
    const size_t first_kept = n_rare - keep_rare;

    std::vector<CategoricalBin> new_bins;
    new_bins.reserve(bins_.size());
    CategoricalBin low_freq_bin;

    for (size_t i = 0; i < bins_.size(); ++i) {
      if (i >= first_kept) {
        new_bins.push_back(std::move(bins_[i]));
      } else {
        low_freq_bin.merge_with(bins_[i]);
      }
    }

    if (low_freq_bin.count > 0) {
      new_bins.push_back(std::move(low_freq_bin));
    }

    bins_ = std::move(new_bins);
  }

  // Calculate WoE and IV for all bins with Laplace smoothing
  void calculate_woe_iv() {
    for (auto& bin : bins_) {
      bin.calculate_metrics(total_pos_, total_neg_);
    }
  }
  
  // Calculate total IV across all bins
  double calculate_total_iv() const {
    return std::accumulate(bins_.begin(), bins_.end(), 0.0,
                           [](double sum, const CategoricalBin& bin) { return sum + std::fabs(bin.iv); });
  }
  
  // Ensure monotonicity by sorting bins by WoE
  void ensure_monotonicity() {
    std::sort(bins_.begin(), bins_.end(), [](const CategoricalBin& a, const CategoricalBin& b) {
      return a.woe < b.woe;
    });
  }
  
  // Merge bins with improved strategy using statistical similarity
  void merge_bins() {
    if ((int)bins_.size() > max_bins_) {
      greedy_similarity_merge(bins_, static_cast<size_t>(std::max(max_bins_, 1)),
                              total_pos_, total_neg_, false);
    }
  }
  
public:
  // Constructor with improved defaults and documentation
  OBC_UDT(
    int min_bins = 3,
    int max_bins = 5,
    double bin_cutoff = 0.05,
    int max_n_prebins = 20,
    std::string bin_separator = "%;%",
    double convergence_threshold = 1e-6,
    int max_iterations = 1000
  ) : min_bins_(min_bins), max_bins_(max_bins), bin_cutoff_(bin_cutoff),
  max_n_prebins_(max_n_prebins), bin_separator_(bin_separator),
  convergence_threshold_(convergence_threshold), max_iterations_(max_iterations),
  converged_(false), iterations_(0), total_pos_(0), total_neg_(0) {}
  
  // Main fitting method. 'categories' holds the distinct categories in order
  // of first appearance, 'pos'/'neg' their class counts.
  void fit(const std::vector<std::string>& categories,
           const std::vector<int>& pos,
           const std::vector<int>& neg,
           size_t n_obs) {
    validate_inputs(categories, pos, neg, n_obs);

    const int ncat = static_cast<int>(categories.size());

    // Adjust min_bins and max_bins based on unique categories
    max_bins_ = std::min(max_bins_, ncat);
    min_bins_ = std::min(min_bins_, max_bins_);

    // Initial binning (one bin per category)
    initial_binning(categories, pos, neg);

    // Special case: 1 or 2 unique levels
    if (ncat <= 2) {
      // max_bins = 1 with two categories: a single bin (it used to return
      // two bins, above max_bins)
      while (static_cast<int>(bins_.size()) > max_bins_ && bins_.size() > 1) {
        bins_[0].merge_with(bins_[1]);
        bins_.erase(bins_.begin() + 1);
      }
      calculate_woe_iv();
      converged_ = true;
      iterations_ = 0;
      return;
    }
    
    // Merge low frequency bins
    merge_low_frequency_bins();
    calculate_woe_iv();
    ensure_monotonicity();
    
    // Main optimization loop
    double prev_total_iv = calculate_total_iv();
    converged_ = false;
    iterations_ = 0;
    
    while (!converged_ && iterations_ < max_iterations_) {
      // Merge bins if needed
      if ((int)bins_.size() > max_bins_) {
        merge_bins();
      } else {
        // If within min_bins and max_bins range, we're done
        if ((int)bins_.size() >= min_bins_) {
          converged_ = true;
          break;
        }
        
        // If we can't increase the number of bins, we're done
        // No splitting is performed to avoid artificial categories
        converged_ = true;
        break;
      }
      
      // Ensure monotonicity
      ensure_monotonicity();
      
      // Check convergence
      double total_iv = calculate_total_iv();
      if (std::abs(total_iv - prev_total_iv) < convergence_threshold_) {
        converged_ = true;
      }
      
      prev_total_iv = total_iv;
      iterations_++;
    }
    
    // Final calculations
    calculate_woe_iv();
    ensure_monotonicity();
  }
  
  // Get results as Rcpp List with improved structure
  Rcpp::List get_woe_bin() const {
    const R_xlen_t nb = static_cast<R_xlen_t>(bins_.size());
    Rcpp::CharacterVector bin_names(nb);
    Rcpp::NumericVector woe_values(nb), iv_values(nb), event_rates(nb);
    Rcpp::IntegerVector counts(nb), counts_pos(nb), counts_neg(nb);

    for (R_xlen_t i = 0; i < nb; ++i) {
      const CategoricalBin& bin = bins_[static_cast<size_t>(i)];
      bin_names[i] = utils::join_categories(bin.categories, bin_separator_);
      woe_values[i] = bin.woe;
      iv_values[i] = bin.iv;
      counts[i] = bin.count;
      counts_pos[i] = bin.count_pos;
      counts_neg[i] = bin.count_neg;
      event_rates[i] = bin.event_rate();
    }

    // Calculate total IV
    double total_iv = calculate_total_iv();
    
    // Create sequential IDs
    Rcpp::NumericVector ids(nb);
    for (R_xlen_t i = 0; i < nb; i++) {
      ids[i] = static_cast<double>(i + 1);
    }
    
    // Return results
    return Rcpp::List::create(
      Rcpp::Named("id") = ids,
      Rcpp::Named("bin") = bin_names,
      Rcpp::Named("woe") = woe_values,
      Rcpp::Named("iv") = iv_values,
      Rcpp::Named("count") = counts,
      Rcpp::Named("count_pos") = counts_pos,
      Rcpp::Named("count_neg") = counts_neg,
      Rcpp::Named("event_rate") = event_rates,
      Rcpp::Named("converged") = converged_,
      Rcpp::Named("iterations") = iterations_,
      Rcpp::Named("total_iv") = total_iv
    );
  }
};

// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_udt(
   Rcpp::IntegerVector target,
   Rcpp::CharacterVector feature,
   int min_bins = 3,
   int max_bins = 5,
   double bin_cutoff = 0.05,
   int max_n_prebins = 20,
   std::string bin_separator = "%;%",
   double convergence_threshold = 1e-6,
   int max_iterations = 1000
) {
 try {
   const R_xlen_t n = feature.size();
   if (n != target.size()) {
     throw std::invalid_argument("Feature and target vectors must have the same length.");
   }

   // Aggregate per distinct category (first-appearance order). Each distinct
   // CHARSXP is resolved once; the string map merges equal byte strings
   // stored under different encodings.
   std::vector<std::string> categories;
   std::vector<int> pos, neg;
   std::unordered_map<std::string, size_t> str_index;
   std::unordered_map<SEXP, size_t> ptr_index;
   const int* tg = INTEGER(target);

   for (R_xlen_t i = 0; i < n; ++i) {
     const int t = tg[i];
     if (t == NA_INTEGER) {
       Rcpp::stop("Target cannot contain missing values");
     }
     if (t != 0 && t != 1) {
       throw std::invalid_argument("Target vector must contain only 0 and 1.");
     }
     SEXP cs = STRING_ELT(feature, i);
     size_t idx;
     auto pit = ptr_index.find(cs);
     if (pit != ptr_index.end()) {
       idx = pit->second;
     } else {
       std::string cat = (cs == NA_STRING) ? std::string(MISSING_VALUE) : std::string(CHAR(cs));
       auto ins = str_index.emplace(cat, categories.size());
       if (ins.second) {
         categories.push_back(std::move(cat));
         pos.push_back(0);
         neg.push_back(0);
       }
       idx = ins.first->second;
       ptr_index.emplace(cs, idx);
     }
     if (t == 1) pos[idx]++; else neg[idx]++;
   }

   OBC_UDT binning(
       min_bins, max_bins, bin_cutoff, max_n_prebins,
       bin_separator, convergence_threshold, max_iterations
   );

   binning.fit(categories, pos, neg, static_cast<size_t>(n));
   return binning.get_woe_bin();
 } catch (const std::exception& e) {
   Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
 }
}
