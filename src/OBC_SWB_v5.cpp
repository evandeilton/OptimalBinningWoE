// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <numeric>
#include <functional>


// Include shared headers
#include "common/optimal_binning_common.h"
#include "common/bin_structures.h"

using namespace Rcpp;
using namespace OptimalBinning;


// Global constants for better consistency and clarity
// Constant removed (uses shared definition)
static constexpr double LAPLACE_ALPHA = 0.5;  // Laplace smoothing parameter
// Category used for NA when this entry point is called directly; the R wrapper
// already maps NA to "NA", the token every categorical binner and
// ob_apply_woe_cat()/obwoe_sql() use (this used to be "__MISSING__").
static constexpr const char* MISSING_VALUE = "NA";

// Namespace for utility functions
namespace utils {
// Safe logarithm function to avoid -Inf
inline double safe_log(double x) {
  return x > EPSILON ? std::log(x) : std::log(EPSILON);
}

// Calculate Weight of Evidence with Laplace smoothing
inline double calculate_woe(int pos, int neg, int total_pos, int total_neg, double alpha = LAPLACE_ALPHA) {
  // Apply Laplace (add-alpha) smoothing
  double pos_rate = (pos + alpha) / (total_pos + alpha * 2);
  double neg_rate = (neg + alpha) / (total_neg + alpha * 2);
  
  return safe_log(pos_rate / neg_rate);
}

// Calculate Information Value with Laplace smoothing
inline double calculate_iv(int pos, int neg, int total_pos, int total_neg, double alpha = LAPLACE_ALPHA) {
  // Apply Laplace smoothing
  double pos_rate = (pos + alpha) / (total_pos + alpha * 2);
  double neg_rate = (neg + alpha) / (total_neg + alpha * 2);
  
  double woe = safe_log(pos_rate / neg_rate);
  return (pos_rate - neg_rate) * woe;
}

// Calculate total IV for a vector of bins
template <typename BinType>
inline double calculate_total_iv(const std::vector<BinType>& bins, int total_pos, int total_neg) {
  double total_iv = 0.0;
  
  for (const auto& bin : bins) {
    total_iv += calculate_iv(bin.count_pos, bin.count_neg, total_pos, total_neg);
  }
  
  return total_iv;
}

// Join vector of strings ensuring uniqueness
inline std::string join_categories(const std::vector<std::string>& categories, const std::string& delimiter) {
  if (categories.empty()) return "";
  if (categories.size() == 1) return categories[0];
  
  // Create a set for uniqueness check
  std::unordered_set<std::string> unique_categories;
  std::vector<std::string> unique_vector;
  unique_vector.reserve(categories.size());
  
  for (const auto& cat : categories) {
    if (unique_categories.insert(cat).second) {
      unique_vector.push_back(cat);
    }
  }
  
  // Estimate result size for pre-allocation
  size_t total_length = 0;
  for (const auto& cat : unique_vector) {
    total_length += cat.length();
  }
  total_length += delimiter.length() * (unique_vector.size() - 1);
  
  // Build result string
  std::string result;
  result.reserve(total_length);
  
  result = unique_vector[0];
  for (size_t i = 1; i < unique_vector.size(); ++i) {
    result += delimiter;
    result += unique_vector[i];
  }
  
  return result;
}

// Calculate Jensen-Shannon divergence between two bins
inline double calculate_divergence(int bin1_pos, int bin1_neg, int bin2_pos, int bin2_neg, 
                                   int total_pos, int total_neg, double alpha = LAPLACE_ALPHA) {
  // Smoothed proportions for bin 1
  double p1 = (bin1_pos + alpha) / (total_pos + alpha * 2);
  double n1 = (bin1_neg + alpha) / (total_neg + alpha * 2);
  
  // Smoothed proportions for bin 2
  double p2 = (bin2_pos + alpha) / (total_pos + alpha * 2);
  double n2 = (bin2_neg + alpha) / (total_neg + alpha * 2);
  
  // Average proportions
  double p_avg = (p1 + p2) / 2;
  double n_avg = (n1 + n2) / 2;
  
  // KL divergence components
  double div_p1 = p1 > EPSILON ? p1 * safe_log(p1 / p_avg) : 0;
  double div_n1 = n1 > EPSILON ? n1 * safe_log(n1 / n_avg) : 0;
  double div_p2 = p2 > EPSILON ? p2 * safe_log(p2 / p_avg) : 0;
  double div_n2 = n2 > EPSILON ? n2 * safe_log(n2 / n_avg) : 0;
  
  // Jensen-Shannon divergence (symmetric)
  return (div_p1 + div_n1 + div_p2 + div_n2) / 2;
}
}

// Improved Categorical Binning with Sliding Window Binning (SWB)
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

class OBC_SWB {
private:
  // Enhanced bin statistics structure with uniqueness guarantee
  // Local CategoricalBin definition removed

  
  // Input data (aggregated per distinct category, in order of first
  // appearance) and parameters
  std::vector<std::string> categories;
  std::vector<int> cat_pos;
  std::vector<int> cat_neg;
  size_t n_obs;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  std::string bin_separator;
  double convergence_threshold;
  int max_iterations;
  
  // Internal state
  std::vector<CategoricalBin> bins;
  int total_pos;
  int total_neg;
  bool converged;
  int iterations_run;
  
  // Initialize bins from raw data
  void initialize_bins() {
    std::unordered_map<std::string, CategoricalBin> initial_bins;
    total_pos = 0;
    total_neg = 0;
    
    // First pass: statistics for each unique category. Keys are inserted in
    // order of first appearance, as the former per-observation loop did, so
    // the map (and everything derived from its iteration order) is unchanged.
    for (size_t c = 0; c < categories.size(); ++c) {
      auto& b = initial_bins[categories[c]];
      b.categories.push_back(categories[c]);
      b.count_pos = cat_pos[c];
      b.count_neg = cat_neg[c];
      b.count = cat_pos[c] + cat_neg[c];
      total_pos += cat_pos[c];
      total_neg += cat_neg[c];
    }
    
    // Calculate minimum count threshold for a separate bin
    double count_threshold = bin_cutoff * static_cast<double>(n_obs);
    
    // Second pass: separate frequent and rare categories. When the frequent
    // categories plus the pooled rare bin cannot reach min_bins, the largest
    // rare categories keep their own bins (only as many as needed): the
    // result used to fall below min_bins -- down to a single bin with zero
    // IV when no category reached bin_cutoff.
    std::vector<CategoricalBin> temp_bins;
    CategoricalBin low_freq_bin;

    std::vector<const CategoricalBin*> rare;
    size_t n_freq = 0;
    for (const auto& pair : initial_bins) {
      if (pair.second.count >= count_threshold) ++n_freq; else rare.push_back(&pair.second);
    }
    const size_t target = static_cast<size_t>(std::max(min_bins, 1));
    size_t keep = 0;
    while (keep < rare.size() &&
           n_freq + keep + (rare.size() > keep ? 1 : 0) < target) {
      ++keep;
    }
    std::unordered_set<const CategoricalBin*> promoted;
    if (keep > 0) {
      std::vector<const CategoricalBin*> by_size(rare);
      std::stable_sort(by_size.begin(), by_size.end(),
                       [](const CategoricalBin* x, const CategoricalBin* y) { return x->count > y->count; });
      promoted.insert(by_size.begin(), by_size.begin() + static_cast<std::ptrdiff_t>(keep));
    }

    for (auto& pair : initial_bins) {
      if (pair.second.count >= count_threshold || promoted.count(&pair.second)) {
        pair.second.calculate_metrics(total_pos, total_neg);
        temp_bins.push_back(std::move(pair.second));
      } else {
        low_freq_bin.merge_with(pair.second);
      }
    }
    
    // Add the rare categories bin if it's not empty
    if (low_freq_bin.count > 0) {
      low_freq_bin.calculate_metrics(total_pos, total_neg);
      temp_bins.push_back(std::move(low_freq_bin));
    }
    
    bins = std::move(temp_bins);
    
    // Sort bins by WoE for better merging strategy
    std::sort(bins.begin(), bins.end(), [](const CategoricalBin& a, const CategoricalBin& b) {
      return a.woe < b.woe;
    });
    
    // Initial consolidation to max_n_prebins if needed (the incremental
    // merger picks the same pairs as merge_most_similar_bins(); see
    // greedy_similarity_merge()).
    const size_t prebin_target = static_cast<size_t>(std::max(std::max(max_n_prebins, min_bins), 1));
    if (bins.size() > prebin_target) {
      greedy_similarity_merge(bins, prebin_target, total_pos, total_neg, true);
    }
    while (bins.size() > (size_t)max_n_prebins && bins.size() > (size_t)min_bins) {
      merge_most_similar_bins();
    }
  }
  
  // Find and merge the most similar bins based on statistical divergence
  void merge_most_similar_bins() {
    if (bins.size() <= (size_t)min_bins) return;
    
    double min_divergence = std::numeric_limits<double>::max();
    size_t merge_idx1 = 0;
    size_t merge_idx2 = 1;
    
    // Find the pair of bins with minimal Jensen-Shannon divergence
    for (size_t i = 0; i < bins.size(); ++i) {
      for (size_t j = i + 1; j < bins.size(); ++j) {
        double div = bins[i].divergence_from(bins[j], total_pos, total_neg);
        
        // Prefer adjacent bins when divergence is similar
        if (j == i + 1) {
          div *= 0.95;  // Small bias towards adjacent bins
        }
        
        if (div < min_divergence) {
          min_divergence = div;
          merge_idx1 = i;
          merge_idx2 = j;
        }
      }
    }
    
    // Perform the merge
    if (merge_idx2 < merge_idx1) std::swap(merge_idx1, merge_idx2);
    
    bins[merge_idx1].merge_with(bins[merge_idx2]);
    bins[merge_idx1].calculate_metrics(total_pos, total_neg);
    bins.erase(bins.begin() + merge_idx2);
    
    // Re-sort bins by WoE after merge
    std::sort(bins.begin(), bins.end(), [](const CategoricalBin& a, const CategoricalBin& b) {
      return a.woe < b.woe;
    });
  }
  
  // Optimize bins for monotonicity and IV
  void optimize_bins() {
    double prev_iv = utils::calculate_total_iv(bins, total_pos, total_neg);
    converged = false;
    iterations_run = 0;
    
    while (iterations_run < max_iterations) {
      // Check if current binning satisfies all constraints
      if (is_monotonic() && bins.size() <= (size_t)max_bins && bins.size() >= (size_t)min_bins) {
        converged = true;
        break;
      }
      
      // Decide appropriate action based on current state
      if (bins.size() > (size_t)max_bins) {
        // Too many bins - merge the most similar pair
        merge_most_similar_bins();
      } else if (bins.size() < (size_t)min_bins) {
        // Not enough bins - would need splitting but we avoid it for stability
        // Better to stop here than risk creating unstable bins
        break;
      } else if (!is_monotonic()) {
        // Non-monotonic - try to fix monotonicity violations
        improve_monotonicity();
      } else {
        // No issues found but we should never reach here
        converged = true;
        break;
      }
      
      // Check for convergence
      double current_iv = utils::calculate_total_iv(bins, total_pos, total_neg);
      if (std::abs(current_iv - prev_iv) < convergence_threshold) {
        converged = true;
        break;
      }
      prev_iv = current_iv;
      iterations_run++;
    }
    
    // Final consolidation if needed
    while (bins.size() > (size_t)max_bins) {
      merge_most_similar_bins();
    }
    
    // Final metrics calculation
    for (auto& bin : bins) {
      bin.calculate_metrics(total_pos, total_neg);
    }
  }
  
  // Check if current binning is monotonic in WoE
  bool is_monotonic() const {
    if (bins.size() <= 2) return true;  // 1 or 2 bins are always monotonic
    
    bool increasing = true;
    bool decreasing = true;
    
    for (size_t i = 1; i < bins.size(); ++i) {
      if (bins[i].woe < bins[i - 1].woe - EPSILON) {
        increasing = false;
      }
      if (bins[i].woe > bins[i - 1].woe + EPSILON) {
        decreasing = false;
      }
      // If neither pattern holds, binning is not monotonic
      if (!increasing && !decreasing) return false;
    }
    
    return true;
  }
  
  // Fix monotonicity violations
  void improve_monotonicity() {
    // Identify the most serious monotonicity violation
    double max_violation = 0.0;
    size_t violation_idx = 0;
    bool found_violation = false;
    
    // Find if we're generally increasing or decreasing
    bool should_increase = true;
    if (bins.size() >= 3) {
      // Check first few bins to determine overall trend
      int increasing_count = 0;
      int decreasing_count = 0;
      
      for (size_t i = 1; i < std::min(bins.size(), size_t(5)); ++i) {
        if (bins[i].woe > bins[i-1].woe) increasing_count++;
        else if (bins[i].woe < bins[i-1].woe) decreasing_count++;
      }
      
      should_increase = (increasing_count >= decreasing_count);
    }
    
    // Find the most severe violation
    for (size_t i = 1; i < bins.size(); ++i) {
      double violation = 0.0;
      
      if (should_increase && bins[i].woe < bins[i-1].woe) {
        violation = bins[i-1].woe - bins[i].woe;
      } else if (!should_increase && bins[i].woe > bins[i-1].woe) {
        violation = bins[i].woe - bins[i-1].woe;
      }
      
      if (violation > max_violation) {
        max_violation = violation;
        violation_idx = i - 1; // Index of first bin in the violating pair
        found_violation = true;
      }
    }
    
    // Fix the violation by merging
    if (found_violation) {
      merge_bins(violation_idx, violation_idx + 1);
    }
  }
  
  // Merge two bins
  void merge_bins(size_t index1, size_t index2) {
    if (index1 >= bins.size() || index2 >= bins.size() || index1 == index2) {
      return;
    }
    
    bins[index1].merge_with(bins[index2]);
    bins[index1].calculate_metrics(total_pos, total_neg);
    bins.erase(bins.begin() + index2);
  }
  
public:
  // Constructor with validation. 'categories' holds the distinct categories
  // in order of first appearance and 'pos'/'neg' their class counts.
  OBC_SWB(std::vector<std::string> categories_,
          std::vector<int> pos_,
          std::vector<int> neg_,
          size_t n_obs_,
          int min_bins_ = 3,
          int max_bins_ = 5,
          double bin_cutoff_ = 0.05,
          int max_n_prebins_ = 20,
          std::string bin_separator_ = "%;%",
          double convergence_threshold_ = 1e-6,
          int max_iterations_ = 1000)
    : categories(std::move(categories_)),
      cat_pos(std::move(pos_)),
      cat_neg(std::move(neg_)),
      n_obs(n_obs_),
      min_bins(min_bins_),
      max_bins(max_bins_),
      bin_cutoff(bin_cutoff_),
      max_n_prebins(max_n_prebins_),
      bin_separator(std::move(bin_separator_)),
      convergence_threshold(convergence_threshold_),
      max_iterations(max_iterations_),
      total_pos(0),
      total_neg(0),
      converged(false),
      iterations_run(0) {
    
    if (n_obs == 0) {
      Rcpp::stop("Feature and target vectors cannot be empty");
    }
    
    // Target must contain both classes
    int all_pos = 0, all_neg = 0;
    for (size_t c = 0; c < categories.size(); ++c) {
      all_pos += cat_pos[c];
      all_neg += cat_neg[c];
    }
    if (all_pos == 0 || all_neg == 0) {
      Rcpp::stop("Target must contain both 0 and 1 values");
    }
    
    // min_bins > max_bins used to reach the final "merge while above
    // max_bins" loop with merging disabled by min_bins: an endless loop.
    if (min_bins < 1) {
      Rcpp::stop("min_bins must be at least 1");
    }
    if (max_bins < min_bins) {
      Rcpp::stop("max_bins must be greater than or equal to min_bins");
    }
    
    // Cap the bin limits at the number of categories. The constructor
    // parameters used to shadow the members here, so these adjustments were
    // silently lost (min_bins above the number of categories then stopped the
    // optimisation before monotonicity was checked).
    const int ncat = static_cast<int>(categories.size());
    max_bins = std::min(max_bins, ncat);
    min_bins = std::max(1, std::min(min_bins, max_bins));
    
    if (bin_cutoff <= 0 || bin_cutoff >= 1) {
      Rcpp::stop("bin_cutoff must be between 0 and 1");
    }
    
    if (max_n_prebins < min_bins) {
      Rcpp::stop("max_n_prebins must be at least min_bins");
    }
  }
  
  // Main fitting function
  void fit() {
    const int ncat = static_cast<int>(categories.size());
    
    // Handle special case of very few categories
    if (ncat <= 2) {
      // Process each unique category as a separate bin
      std::unordered_map<std::string, CategoricalBin> bin_map;
      total_pos = 0;
      total_neg = 0;
      
      for (size_t c = 0; c < categories.size(); ++c) {
        auto& bin = bin_map[categories[c]];
        bin.categories.push_back(categories[c]);
        bin.count_pos = cat_pos[c];
        bin.count_neg = cat_neg[c];
        bin.count = cat_pos[c] + cat_neg[c];
        total_pos += cat_pos[c];
        total_neg += cat_neg[c];
      }
      
      // Transfer to bins vector
      bins.clear();
      for (auto& kv : bin_map) {
        bins.push_back(std::move(kv.second));
      }
      
      // max_bins = 1 with two categories: a single bin (it used to return
      // two bins, above max_bins)
      while (static_cast<int>(bins.size()) > max_bins && bins.size() > 1) {
        bins[0].merge_with(bins[1]);
        bins.erase(bins.begin() + 1);
      }
      
      // Calculate final metrics
      for (auto& bin : bins) {
        bin.calculate_metrics(total_pos, total_neg);
      }
      
      converged = true;
      iterations_run = 0;
      return;
    }
    
    // Normal processing for more than 2 categories
    initialize_bins();
    optimize_bins();
  }
  
  // Get results as Rcpp List
  Rcpp::List get_results() const {
    // Prepare result vectors
    std::vector<std::string> bin_categories;
    std::vector<double> woes;
    std::vector<double> ivs;
    std::vector<int> counts;
    std::vector<int> counts_pos;
    std::vector<int> counts_neg;
    std::vector<double> event_rates;
    
    // Fill result vectors
    for (const auto& bin : bins) {
      std::string bin_name = utils::join_categories(bin.categories, bin_separator);
      bin_categories.push_back(bin_name);
      woes.push_back(bin.woe);
      ivs.push_back(bin.iv);
      counts.push_back(bin.count);
      counts_pos.push_back(bin.count_pos);
      counts_neg.push_back(bin.count_neg);
      event_rates.push_back(bin.event_rate());
    }
    
    // Calculate total IV
    double total_iv = 0.0;
    for (const auto& iv : ivs) {
      total_iv += std::fabs(iv);
    }
    
    // Create sequential IDs
    Rcpp::NumericVector ids(static_cast<R_xlen_t>(bin_categories.size()));
    for (size_t i = 0; i < bin_categories.size(); i++) {
      ids[static_cast<R_xlen_t>(i)] = static_cast<double>(i + 1);
    }
    
    // Return results
    return Rcpp::List::create(
      Rcpp::Named("id") = ids,
      Rcpp::Named("bin") = bin_categories,
      Rcpp::Named("woe") = woes,
      Rcpp::Named("iv") = ivs,
      Rcpp::Named("count") = counts,
      Rcpp::Named("count_pos") = counts_pos,
      Rcpp::Named("count_neg") = counts_neg,
      Rcpp::Named("event_rate") = event_rates,
      Rcpp::Named("converged") = converged,
      Rcpp::Named("iterations") = iterations_run,
      Rcpp::Named("total_iv") = total_iv
    );
  }
};

// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_swb(Rcpp::IntegerVector target,
                                           Rcpp::CharacterVector feature,
                                          int min_bins = 3,
                                          int max_bins = 5,
                                          double bin_cutoff = 0.05,
                                          int max_n_prebins = 20,
                                          std::string bin_separator = "%;%",
                                          double convergence_threshold = 1e-6,
                                          int max_iterations = 1000) {
 try {
   const R_xlen_t n = feature.size();
   if (n != target.size()) {
     Rcpp::stop("Feature and target vectors must have the same length");
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
       Rcpp::stop("Target must contain only binary values (0 or 1)");
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
   
   OBC_SWB binner(std::move(categories), std::move(pos), std::move(neg),
                  static_cast<size_t>(n), min_bins, max_bins,
                  bin_cutoff, max_n_prebins, bin_separator,
                  convergence_threshold, max_iterations);
   binner.fit();
   return binner.get_results();
 } catch (const std::exception& e) {
   Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
 }
}
