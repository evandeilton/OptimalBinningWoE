// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <numeric>
#include <utility>
#include <cstdint>
#include <queue>
#include <tuple>


// Include shared headers
#include "common/optimal_binning_common.h"
#include "common/bin_structures.h"
#include "common/ob_core.h"

using namespace Rcpp;
using namespace OptimalBinning;


// Constants for better readability and numerical stability
// Constant removed (uses shared definition)
// Bayesian smoothing parameter (adjustable prior strength)
// Constant removed (uses shared definition)

namespace {
// Strength (pseudo-observations) of the empirical-Bayes beta prior, centred on
// the overall event rate, that orders the categories and defines the DP
// objective (minimum evidence, audit Section 15.5); same value as OBC_MBA and
// OBC_Sketch. The reported WoE/IV keep the estimator of bin_iv().
constexpr double kEvidenceM = 20.0;
} // namespace

// Class for Optimal Binning using Simulated Annealing with enhanced algorithms
class OBC_SAB {
private:
  // Input data
  std::vector<std::string> feature;
  std::vector<int> target;
  
  // Binning parameters
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  std::string bin_separator;
  
  // Simulated Annealing parameters
  double initial_temperature;
  double cooling_rate;
  int max_iterations;
  double convergence_threshold;
  bool adaptive_cooling;
  
  // Category statistics
  std::vector<std::string> unique_categories;
  // Observations and events per category, indexed like unique_categories.
  // (Were two string-keyed hash maps queried with .at() for every category
  // on every iteration of the search.)
  std::vector<int> cat_count_;
  std::vector<int> cat_pos_;
  int total_count;
  int total_positive;
  int total_negative;
  
  // Binning results
  int actual_bins;
  std::vector<int> best_solution;
  std::vector<int> current_solution;
  double best_iv;
  double current_iv;
  bool converged;
  int iterations_run;
  int iterations_without_improvement;
  
  // Initialize the algorithm with comprehensive error checking
  void initialize() {
    // Count per category (one hash lookup per row)
    std::unordered_map<std::string, std::pair<int, int>> counts;
    for (size_t i = 0; i < feature.size(); ++i) {
      std::pair<int, int>& c = counts[feature[i]];
      c.first++;
      c.second += (target[i] == 1) ? 1 : 0;
    }

    // Hash-set iteration order differs between standard libraries (and with
    // the bucket count), and it decides which category every random move
    // touches, so the same data and seed gave different binnings on
    // different platforms. Sorting makes the order a function of the labels.
    unique_categories.clear();
    unique_categories.reserve(counts.size());
    for (const auto& kv : counts) {
      unique_categories.push_back(kv.first);
    }
    std::sort(unique_categories.begin(), unique_categories.end());
    cat_count_.resize(unique_categories.size());
    cat_pos_.resize(unique_categories.size());
    for (size_t i = 0; i < unique_categories.size(); ++i) {
      const std::pair<int, int>& c = counts[unique_categories[i]];
      cat_count_[i] = c.first;
      cat_pos_[i] = c.second;
    }
    
    // Count totals
    total_count = static_cast<int>(feature.size());
    total_positive = static_cast<int>(std::count(target.begin(), target.end(), 1));
    total_negative = total_count - total_positive;
    
    // Check for extremely imbalanced datasets
    if (total_positive < 5 || total_negative < 5) {
      Rcpp::warning("Dataset has fewer than 5 samples in one class. Results may be unstable.");
    }
    
    int n_categories = static_cast<int>(unique_categories.size());
    actual_bins = std::min(std::max(min_bins, std::min(max_bins, n_categories)), n_categories);
    
    best_solution.assign(unique_categories.size(), 0);
    best_iv = 0.0;
    converged = false;
    iterations_run = 0;
    iterations_without_improvement = 0;
  }
  
  // IV of one bin with the Bayesian smoothing that get_results() reports
  // (same expression, so the optimised value is the reported total IV).
  double bin_iv(int count, int positives) const {
    const int negatives = count - positives;
    const double prior_weight = BAYESIAN_PRIOR_STRENGTH;
    const double overall_event_rate = static_cast<double>(total_positive) / total_count;
    const double prior_pos = prior_weight * overall_event_rate;
    const double prior_neg = prior_weight * (1.0 - overall_event_rate);
    const double pos_dist = static_cast<double>(positives + prior_pos) /
      (total_positive + prior_weight);
    const double neg_dist = static_cast<double>(negatives + prior_neg) /
      (total_negative + prior_weight);
    if (pos_dist > EPSILON && neg_dist > EPSILON) {
      const double woe = std::log(pos_dist / neg_dist);
      const double iv = (pos_dist - neg_dist) * woe;
      if (std::isfinite(woe) && std::isfinite(iv)) return iv;
    }
    return 0.0;
  }

  // Exact partition of the ordered categories into exactly K contiguous
  // groups that maximizes the empirical-Bayes IV (ob::eb_woe_iv, kEvidenceM
  // pseudo-observations per bin, shares normalized over the K bins). With
  // `use_cutoff`, every group must hold at least bin_cutoff of the
  // observations. Fills f[k][i] (best value for the first i categories in k
  // groups, k <= K) and the back-pointers.
  void run_dp(const std::vector<int64_t>& C, const std::vector<int64_t>& P,
              int K, bool use_cutoff,
              std::vector<std::vector<double>>& f,
              std::vector<std::vector<int>>& bp) const {
    const int k_hi = K;
    const double Tp = static_cast<double>(total_positive);
    const double Tn = static_cast<double>(total_negative);
    const int m = static_cast<int>(C.size()) - 1;
    const double NEG = -std::numeric_limits<double>::infinity();
    f.assign(k_hi + 1, std::vector<double>(m + 1, NEG));
    bp.assign(k_hi + 1, std::vector<int>(m + 1, -1));
    f[0][0] = 0.0;
    auto ok = [&](int j, int i) {
      if (!use_cutoff) return true;
      return !(static_cast<double>(C[i] - C[j]) / total_count < bin_cutoff);
    };
    for (int k = 1; k <= k_hi; ++k) {
      for (int i = k; i <= m; ++i) {
        if ((i & 255) == 0) Rcpp::checkUserInterrupt();
        double best = NEG;
        int arg = -1;
        for (int j = k - 1; j < i; ++j) {
          if (f[k - 1][j] == NEG || !ok(j, i)) continue;
          const double pos = static_cast<double>(P[i] - P[j]);
          const double neg = static_cast<double>(C[i] - C[j]) - pos;
          const double v = f[k - 1][j] +
            ob::eb_woe_iv(pos, neg, Tp, Tn, K, kEvidenceM).iv;
          if (v > best) { best = v; arg = j; }
        }
        f[k][i] = best;
        bp[k][i] = arg;
      }
    }
  }

public:
  OBC_SAB(const std::vector<std::string>& feature_,
                               const std::vector<int>& target_,
                               int min_bins_ = 3,
                               int max_bins_ = 5,
                               double bin_cutoff_ = 0.05,
                               int max_n_prebins_ = 20,
                               std::string bin_separator_ = "%;%",
                               double initial_temperature_ = 1.0,
                               double cooling_rate_ = 0.995,
                               int max_iterations_ = 1000,
                               double convergence_threshold_ = 1e-6,
                               bool adaptive_cooling_ = true)
    : feature(feature_), target(target_), min_bins(min_bins_), max_bins(max_bins_),
      bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_), bin_separator(bin_separator_),
      initial_temperature(initial_temperature_), cooling_rate(cooling_rate_),
      max_iterations(max_iterations_), convergence_threshold(convergence_threshold_),
      adaptive_cooling(adaptive_cooling_), iterations_without_improvement(0) {
    
    // Enhanced validation
    if (feature.size() != target.size()) {
      throw std::invalid_argument("Feature and target vectors must have the same length");
    }
    
    if (feature.empty() || target.empty()) {
      throw std::invalid_argument("Feature and target vectors cannot be empty");
    }
    
    if (min_bins < 2) {
      throw std::invalid_argument("min_bins must be at least 2");
    }
    
    if (max_bins < min_bins) {
      throw std::invalid_argument("max_bins must be >= min_bins");
    }
    
    if (bin_cutoff <= 0 || bin_cutoff >= 1) {
      throw std::invalid_argument("bin_cutoff must be between 0 and 1");
    }
    
    if (max_n_prebins < max_bins) {
      throw std::invalid_argument("max_n_prebins must be >= max_bins");
    }
    
    // Check for empty strings in feature
    if (std::any_of(feature.begin(), feature.end(), [](const std::string& s) { 
      return s.empty(); 
    })) {
      throw std::invalid_argument("Feature cannot contain empty strings. Consider preprocessing your data.");
    }
    
    // Check for binary target
    bool has_zero = false;
    bool has_one = false;
    
    for (int val : target) {
      if (val == 0) has_zero = true;
      else if (val == 1) has_one = true;
      else throw std::invalid_argument("Target must be binary (0 or 1)");
      
      // No early exit: every value must be 0 or 1.
    }
    
    if (!has_zero || !has_one) {
      throw std::invalid_argument("Target must contain both 0 and 1 values");
    }
    
    if (!(initial_temperature > 0.0) || !std::isfinite(initial_temperature)) {
      throw std::invalid_argument("initial_temperature must be a positive finite number");
    }
    if (!(cooling_rate > 0.0 && cooling_rate <= 1.0)) {
      throw std::invalid_argument("cooling_rate must be in (0, 1]");
    }

    // The annealing parameters (initial_temperature, cooling_rate,
    // max_iterations, convergence_threshold, adaptive_cooling) are still
    // validated so the interface is unchanged, but fit() is now an exact DP
    // that uses no random numbers (audit C-02).
    initialize();
  }
  
  // Exact optimisation (audit C-02, C-03, C-04 / X-12), with minimum
  // evidence (audit Section 15.5). For a binary target and an IV that is a
  // sum of convex per-bin terms, the optimal grouping of categories into k
  // bins is contiguous in event-rate order (Fisher 1958; Breiman et al. 1984,
  // Thm 4.5), so a DP over the ordered categories finds it in O(k m^2). The
  // simulated annealing it replaces stopped after ~200 iterations without
  // improvement (it returned its random initial assignment on 150 categories,
  // 11-17% below the optimum while reporting converged = TRUE), subtracted
  // 500 from any solution whose arbitrary bin *labels* were not in rate
  // order, and re-heated to T0/2 every 100 rejections.
  //
  // With the *raw* rate order and the reported IV as objective, the exact DP
  // overfitted sparse high-cardinality features: with 200 categories of
  // about 10 rows, the categories that had no event by chance sit together
  // at the bottom of the raw order and the DP isolated them (winner's curse;
  // hold-out log-loss +5% to +8% in 20/20 seeds). Therefore:
  //  - categories are ordered by their posterior-mean rate
  //    (pos + m r0) / (n + m) under a beta prior centred on the overall rate
  //    r0 with m = kEvidenceM pseudo-observations (Gelman et al., BDA ch. 5;
  //    the Bayesian form of the contiguity argument with estimated rates):
  //    large categories keep their raw-rate rank, a tiny one is ranked near
  //    r0 unless its data say otherwise;
  //  - for every k in [min_bins, max_bins] the DP maximizes the IV under the
  //    same prior (ob::eb_woe_iv), so a bin earns IV in proportion to its
  //    evidence;
  //  - among those k partitions the one with the largest *reported* IV is
  //    kept (smallest k on ties), as before.
  // bin_cutoff is a hard constraint when feasible. No random numbers.
  void fit() {
    const int m = static_cast<int>(unique_categories.size());
    const double r0 = static_cast<double>(total_positive) / total_count;
    std::vector<double> shrunk(m);
    for (int t = 0; t < m; ++t) {
      shrunk[t] = (cat_pos_[t] + kEvidenceM * r0) / (cat_count_[t] + kEvidenceM);
    }
    // Shrunk rate; ties by ascending count, then name (the categories are
    // already name-sorted and the sort is stable).
    std::vector<int> ord(m);
    std::iota(ord.begin(), ord.end(), 0);
    std::stable_sort(ord.begin(), ord.end(), [this, &shrunk](int a, int b) {
      if (shrunk[a] != shrunk[b]) return shrunk[a] < shrunk[b];
      return cat_count_[a] < cat_count_[b];
    });
    // Pre-bins: contiguous runs of the ordered categories. With more than
    // max_n_prebins categories, the smallest run is merged into the
    // neighbour with the closer shrunk rate (the left one on ties) until
    // max_n_prebins remain -- the pre-binning of OBC_MBA. It bounds the DP
    // at O(k max_n_prebins^2): on all m categories it cost O(k^2 m^2) over
    // the k range (15 s for 20 000 categories). A heap keyed by (size,
    // leftmost category) keeps the merge sequence identical to repeatedly
    // scanning for the smallest run, in O(m log m).
    const std::vector<int> start = prebin_runs(ord, r0);
    const int g = static_cast<int>(start.size());
    std::vector<int64_t> C(g + 1, 0), P(g + 1, 0);
    for (int q = 0; q < g; ++q) {
      const int end = (q + 1 < g) ? start[q + 1] : m;
      int64_t c = 0, pp = 0;
      for (int t = start[q]; t < end; ++t) {
        c += cat_count_[ord[t]];
        pp += cat_pos_[ord[t]];
      }
      C[q + 1] = C[q] + c;
      P[q + 1] = P[q] + pp;
    }

    const int k_hi = std::max(1, std::min(actual_bins, g));
    const int k_lo = std::max(1, std::min(min_bins, k_hi));
    const double NEG = -std::numeric_limits<double>::infinity();
    std::vector<std::vector<double>> f;
    std::vector<std::vector<int>> bp;
    int best_k = -1;
    std::vector<int> best;
    for (int pass = 0; pass < 2 && best_k < 0; ++pass) {
      // Pass 0 enforces bin_cutoff; pass 1 (only when no k is feasible
      // under it) drops the constraint rather than failing.
      double best_rep = NEG;
      for (int k = k_lo; k <= k_hi; ++k) {
        run_dp(C, P, k, pass == 0, f, bp);
        if (f[k][g] == NEG) continue;
        std::vector<int> sol(m, 0);
        double rep = 0.0;
        for (int kk = k, i = g; kk >= 1; --kk) {
          const int j = bp[kk][i];
          const int t_end = (i < g) ? start[i] : m;
          for (int t = start[j]; t < t_end; ++t) sol[ord[t]] = kk - 1;
          rep += bin_iv(static_cast<int>(C[i] - C[j]),
                        static_cast<int>(P[i] - P[j]));
          i = j;
        }
        if (rep > best_rep) {
          best_rep = rep;
          best_k = k;
          best = std::move(sol);
        }
      }
      if (best_k >= 0) best_iv = best_rep;
    }

    best_solution = best_k >= 0 ? best : std::vector<int>(m, 0);
    converged = true;      // exact: no iteration budget can cut it short
    iterations_run = 0;    // no iterative search runs
  }

  // Start positions (in `ord`) of the pre-bin runs; see fit().
  std::vector<int> prebin_runs(const std::vector<int>& ord,
                               double r0) const {
    const int m = static_cast<int>(ord.size());
    std::vector<int> start(m);
    std::iota(start.begin(), start.end(), 0);
    if (m <= max_n_prebins || m <= min_bins) return start;
    const int target = std::max(max_n_prebins, min_bins);
    // Runs as a doubly linked list over their leftmost position.
    std::vector<int> prv(m), nxt(m);
    std::vector<int64_t> cnt(m), pos(m);
    std::vector<char> alive(m, 1);
    for (int t = 0; t < m; ++t) {
      prv[t] = t - 1;
      nxt[t] = (t + 1 < m) ? t + 1 : -1;
      cnt[t] = cat_count_[ord[t]];
      pos[t] = cat_pos_[ord[t]];
    }
    auto rate = [&](int t) {
      return (static_cast<double>(pos[t]) + kEvidenceM * r0) /
             (static_cast<double>(cnt[t]) + kEvidenceM);
    };
    using Item = std::tuple<int64_t, int>;  // (size, leftmost position)
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> heap;
    for (int t = 0; t < m; ++t) heap.emplace(cnt[t], t);
    int alive_n = m;
    while (alive_n > target && !heap.empty()) {
      const int64_t c = std::get<0>(heap.top());
      const int i = std::get<1>(heap.top());
      heap.pop();
      if (!alive[i] || cnt[i] != c) continue;  // stale entry
      int left;
      if (prv[i] < 0) {
        left = i;
      } else if (nxt[i] < 0) {
        left = prv[i];
      } else {
        const double s = rate(i);
        left = (std::fabs(s - rate(prv[i])) <= std::fabs(s - rate(nxt[i])))
                   ? prv[i] : i;
      }
      const int right = nxt[left];
      cnt[left] += cnt[right];
      pos[left] += pos[right];
      alive[right] = 0;
      nxt[left] = nxt[right];
      if (nxt[right] >= 0) prv[nxt[right]] = left;
      --alive_n;
      heap.emplace(cnt[left], left);
    }
    start.clear();
    for (int t = 0; t >= 0; t = nxt[t]) start.push_back(t);
    return start;
  }

  // Get final binning results with enhanced statistics
  List get_results() const {
    // Create bins with correct statistics
    std::vector<std::string> bins;
    std::vector<double> woe;
    std::vector<double> iv;
    std::vector<int> count;
    std::vector<int> count_pos;
    std::vector<int> count_neg;
    
    // Process categories into bins
    std::vector<std::unordered_set<std::string>> bin_category_sets(actual_bins);
    std::vector<std::vector<std::string>> bin_categories(actual_bins);
    std::vector<int> bin_counts(actual_bins, 0);
    std::vector<int> bin_positives(actual_bins, 0);
    
    for (size_t i = 0; i < unique_categories.size(); ++i) {
      const std::string& category = unique_categories[i];
      int bin = best_solution[i];
      
      // Add category to bin ensuring uniqueness
      if (bin_category_sets[bin].insert(category).second) {
        bin_categories[bin].push_back(category);
      }
      
      bin_counts[bin] += cat_count_[i];
      bin_positives[bin] += cat_pos_[i];
    }
    
    // Calculate total IV with statistics
    double total_iv = 0.0;
    
    for (int i = 0; i < actual_bins; ++i) {
      if (bin_counts[i] > 0) {
        // Sort categories within bin for consistent output
        std::sort(bin_categories[i].begin(), bin_categories[i].end());
        
        // Create bin name
        std::string bin_name = "";
        for (size_t j = 0; j < bin_categories[i].size(); ++j) {
          if (j > 0) bin_name += bin_separator;
          bin_name += bin_categories[i][j];
        }
        
        // Store bin statistics
        bins.push_back(bin_name);
        count.push_back(bin_counts[i]);
        count_pos.push_back(bin_positives[i]);
        int bin_negatives = bin_counts[i] - bin_positives[i];
        count_neg.push_back(bin_negatives);
        
        // Calculate WoE and IV with Bayesian smoothing
        double prior_weight = BAYESIAN_PRIOR_STRENGTH;
        double overall_event_rate = static_cast<double>(total_positive) / total_count;
        
        double prior_pos = prior_weight * overall_event_rate;
        double prior_neg = prior_weight * (1.0 - overall_event_rate);
        
        double pos_dist = static_cast<double>(bin_positives[i] + prior_pos) / 
          (total_positive + prior_weight);
        double neg_dist = static_cast<double>(bin_negatives + prior_neg) / 
          (total_negative + prior_weight);
        
        double woe_value = 0.0;
        double iv_value = 0.0;
        
        if (pos_dist > EPSILON && neg_dist > EPSILON) {
          woe_value = std::log(pos_dist / neg_dist);
          iv_value = (pos_dist - neg_dist) * woe_value;
          
          // Handle non-finite values
          if (!std::isfinite(woe_value)) woe_value = 0.0;
          if (!std::isfinite(iv_value)) iv_value = 0.0;
          
          total_iv += iv_value;
        }
        
        woe.push_back(woe_value);
        iv.push_back(iv_value);
      }
    }
    
    // The bins are contiguous in the raw event rate; the smoothed WoE of two
    // pure bins of different sizes can still come out in the other order, so
    // the bins are reported in WoE order (a stable sort, which leaves an
    // already monotone result untouched).
    std::vector<size_t> order(bins.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&woe](size_t a, size_t b) { return woe[a] < woe[b]; });
    auto permute = [&order](auto& v) {
      auto tmp = v;
      for (size_t k = 0; k < order.size(); ++k) v[k] = tmp[order[k]];
    };
    permute(bins);
    permute(woe);
    permute(iv);
    permute(count);
    permute(count_pos);
    permute(count_neg);

    // Create numeric IDs for bins
    Rcpp::NumericVector ids(bins.size());
    for(int i = 0; i < static_cast<int>(bins.size()); i++) {
      ids[i] = i + 1;
    }
    
    // Return comprehensive results
    return Rcpp::List::create(
      Named("id") = ids,
      Named("bin") = bins,
      Named("woe") = woe,
      Named("iv") = iv,
      Named("count") = count,
      Named("count_pos") = count_pos,
      Named("count_neg") = count_neg,
      Named("total_iv") = total_iv, 
      Named("converged") = converged,
      Named("iterations") = iterations_run
    );
  }
};


// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_sab(Rcpp::IntegerVector target,
                                          Rcpp::StringVector feature,
                                          int min_bins = 3,
                                          int max_bins = 5,
                                          double bin_cutoff = 0.05,
                                          int max_n_prebins = 20,
                                          std::string bin_separator = "%;%",
                                          double initial_temperature = 1.0,
                                          double cooling_rate = 0.995,
                                          int max_iterations = 1000,
                                          double convergence_threshold = 1e-6,
                                          bool adaptive_cooling = true) {
 try {
   // Basic parameter validation
   if (target.size() != feature.size()) {
     Rcpp::stop("Target and feature vectors must have the same length");
   }
   
   if (feature.size() == 0 || target.size() == 0) {
     Rcpp::stop("Feature and target vectors cannot be empty");
   }
   
   if (min_bins < 2) {
     Rcpp::stop("min_bins must be at least 2");
   }
   
   if (max_bins < min_bins) {
     Rcpp::stop("max_bins must be >= min_bins");
   }
   
   if (bin_cutoff <= 0 || bin_cutoff >= 1) {
     Rcpp::stop("bin_cutoff must be between 0 and 1");
   }
   
   if (max_n_prebins < max_bins) {
     Rcpp::stop("max_n_prebins must be >= max_bins");
   }
   
   // Convert data with NA handling
   std::vector<std::string> feature_vec;
   std::vector<int> target_vec;
   
   feature_vec.reserve(feature.size());
   target_vec.reserve(target.size());
   
   int na_feature_count = 0;
   
   for (R_xlen_t i = 0; i < feature.size(); ++i) {
     // Handle NA in feature
     SEXP s = STRING_ELT(feature, i);
     if (s == NA_STRING) {
       feature_vec.emplace_back("NA");
       na_feature_count++;
     } else {
       feature_vec.emplace_back(CHAR(s));
     }
     
     // Check for NA in target
     if (IntegerVector::is_na(target[i])) {
       Rcpp::stop("Target cannot contain missing values at position %d.", i+1);
     } else {
       target_vec.push_back(target[i]);
     }
   }
   
   // Warn about NA values in feature
   if (na_feature_count > 0) {
     Rcpp::warning("%d missing values found in feature and converted to \"NA\" category.", 
                   na_feature_count);
   }
   
   // The fit is deterministic (exact DP): no RNG state is read or written.

   // Create and run the binning algorithm
   OBC_SAB binner(feature_vec, target_vec,
                                       min_bins, max_bins, bin_cutoff,
                                       max_n_prebins, bin_separator,
                                       initial_temperature, cooling_rate,
                                       max_iterations, convergence_threshold,
                                       adaptive_cooling);
   binner.fit();
   
   return binner.get_results();
 } catch (const std::exception& e) {
   Rcpp::stop("Error in optimal binning: %s", e.what());
 }
}
