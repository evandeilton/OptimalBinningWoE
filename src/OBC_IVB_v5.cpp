// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

using namespace Rcpp;

// Include shared headers
#include "common/bin_structures.h"
#include "common/optimal_binning_common.h"

using namespace Rcpp;
using namespace OptimalBinning;

// Constants for better readability and precision
// Local constant removed (uses shared definition)
constexpr double NEG_INFINITY = -std::numeric_limits<double>::infinity();
// Bayesian smoothing parameter (prior strength)
// Local constant removed (uses shared definition)

// Optimized structure for category statistics
// Local CategoryStats definition removed

// Static separator removed%";

namespace {
// Cache for cumulative statistics - optimized for dynamic programming
class CumulativeStatsCache {
private:
  std::vector<int> cum_pos;
  std::vector<int> cum_neg;
  std::vector<int> cum_total;
  int total_pos = 0;
  int total_neg = 0;

public:
  CumulativeStatsCache(const std::vector<CategoryStats> &stats) {
    const size_t n = stats.size();
    cum_pos.resize(n + 1, 0);
    cum_neg.resize(n + 1, 0);
    cum_total.resize(n + 1, 0);

    for (size_t i = 0; i < n; ++i) {
      cum_pos[i + 1] = cum_pos[i] + stats[i].count_pos;
      cum_neg[i + 1] = cum_neg[i] + stats[i].count_neg;
      cum_total[i + 1] = cum_total[i] + stats[i].count;
    }

    total_pos = cum_pos[n];
    total_neg = cum_neg[n];
  }

  inline int get_pos(int start, int end) const {
    return cum_pos[end] - cum_pos[start];
  }

  inline int get_neg(int start, int end) const {
    return cum_neg[end] - cum_neg[start];
  }

  inline int get_total(int start, int end) const {
    return cum_total[end] - cum_total[start];
  }

  inline int get_total_pos() const { return total_pos; }

  inline int get_total_neg() const { return total_neg; }

  inline double get_event_rate(int start, int end) const {
    int total = get_total(start, end);
    if (total <= 0)
      return 0.0;
    return static_cast<double>(get_pos(start, end)) /
           static_cast<double>(total);
  }
};

// Cache for IV calculations with Bayesian smoothing
class IVCache {
private:
  std::vector<std::vector<double>> cache;
  std::shared_ptr<CumulativeStatsCache> stats_cache;
  bool enabled;

public:
  IVCache(size_t size, std::shared_ptr<CumulativeStatsCache> stats_cache,
          bool use_cache = true)
      : stats_cache(stats_cache), enabled(use_cache) {
    if (enabled) {
      cache.resize(size + 1);
      for (auto &row : cache) {
        row.resize(size + 1, -1.0);
      }
    }
  }

  double get(int start, int end) {
    if (!enabled || start >= static_cast<int>(cache.size()) ||
        end >= static_cast<int>(cache[0].size())) {
      return -1.0;
    }
    return cache[start][end];
  }

  void set(int start, int end, double value) {
    if (!enabled || start >= static_cast<int>(cache.size()) ||
        end >= static_cast<int>(cache[0].size())) {
      return;
    }
    cache[start][end] = value;
  }

  double calculate_and_cache(int start, int end) {
    double cached = get(start, end);
    if (cached >= 0.0) {
      return cached;
    }

    int pos = stats_cache->get_pos(start, end);
    int neg = stats_cache->get_neg(start, end);
    int total_pos = stats_cache->get_total_pos();
    int total_neg = stats_cache->get_total_neg();

    // Calculate Bayesian smoothed metrics
    double prior_pos = BAYESIAN_PRIOR_STRENGTH *
                       static_cast<double>(total_pos) / (total_pos + total_neg);
    double prior_neg = BAYESIAN_PRIOR_STRENGTH - prior_pos;

    double pos_rate = static_cast<double>(pos + prior_pos) /
                      static_cast<double>(total_pos + BAYESIAN_PRIOR_STRENGTH);
    double neg_rate = static_cast<double>(neg + prior_neg) /
                      static_cast<double>(total_neg + BAYESIAN_PRIOR_STRENGTH);

    // Calculate WoE and IV with improved numerical stability
    double woe = 0.0;
    double iv = 0.0;

    if (pos_rate > EPSILON && neg_rate > EPSILON) {
      woe = std::log(pos_rate / neg_rate);
      iv = (pos_rate - neg_rate) * woe;

      // Ensure finite values
      if (!std::isfinite(iv)) {
        iv = 0.0;
      }
    }

    set(start, end, iv);
    return iv;
  }
};
} // namespace

// Main class for IVB (Information Value-based Binning) with dynamic programming
// optimization
class OBC_IVB {
private:
  std::vector<std::string> feature;
  std::vector<int> target;
  double bin_cutoff;
  int min_bins;
  int max_bins;
  int max_n_prebins;
  std::string bin_separator;
  double convergence_threshold;
  int max_iterations;

  std::vector<CategoryStats> category_stats;
  std::vector<std::vector<double>> dp;
  std::vector<std::vector<int>> split_points;
  std::shared_ptr<CumulativeStatsCache> stats_cache;
  std::unique_ptr<IVCache> iv_cache;
  bool converged;
  int iterations_run;

  // Enhanced input validation with comprehensive checks
  void validate_input() {
    if (feature.size() != target.size()) {
      throw std::invalid_argument(
          "Feature and target vectors must have the same length");
    }
    if (feature.empty()) {
      throw std::invalid_argument("Feature and target vectors cannot be empty");
    }
    if (min_bins < 2) {
      throw std::invalid_argument("min_bins must be at least 2");
    }
    if (max_bins < min_bins) {
      throw std::invalid_argument(
          "max_bins must be greater than or equal to min_bins");
    }
    if (bin_cutoff < 0 || bin_cutoff > 1) {
      throw std::invalid_argument("bin_cutoff must be between 0 and 1");
    }

    // Check for empty strings in feature
    if (std::any_of(feature.begin(), feature.end(),
                    [](const std::string &s) { return s.empty(); })) {
      throw std::invalid_argument("Feature cannot contain empty strings. "
                                  "Consider preprocessing your data.");
    }

    // Efficient check for binary target
    bool has_zero = false, has_one = false;
    for (int t : target) {
      if (t == 0)
        has_zero = true;
      else if (t == 1)
        has_one = true;
      else
        throw std::invalid_argument("Target must be binary (0 or 1)");

      if (has_zero && has_one)
        break;
    }

    if (!has_zero || !has_one) {
      throw std::invalid_argument("Target must contain both 0 and 1 values");
    }
  }

  // Enhanced data preprocessing with optimized counting
  void preprocess_data() {
    // Estimate number of categories for pre-allocation
    size_t est_categories =
        std::min(feature.size() / 4, static_cast<size_t>(1024));
    std::unordered_map<std::string, CategoryStats> stats_map;
    stats_map.reserve(est_categories);

    // Set global separator for the structure
    // CategoryStats::bin_separator assignment removed

    // Efficient single-pass counting
    for (size_t i = 0; i < feature.size(); ++i) {
      auto &stats = stats_map[feature[i]];
      if (stats.category.empty()) {
        stats.category = feature[i];
      }
      stats.update(target[i]);
    }

    // Transfer to vector and calculate event rates
    category_stats.reserve(stats_map.size());
    for (auto &pair : stats_map) {
      // pair.second.compute_event_rate(); (auto-handled)
      category_stats.push_back(std::move(pair.second));
    }

    // Check for extremely imbalanced datasets
    int total_pos = 0, total_neg = 0;
    for (const auto &stats : category_stats) {
      total_pos += stats.count_pos;
      total_neg += stats.count_neg;
    }

    if (total_pos < 5 || total_neg < 5) {
      Rcpp::warning("Dataset has fewer than 5 samples in one class. Results "
                    "may be unstable.");
    }
  }

  // Enhanced rare category merging with improved handling
  void merge_rare_categories() {
    // Calculate total count efficiently
    int total_count = 0;
    for (const auto &stats : category_stats) {
      total_count += stats.count;
    }

    // Separate rare and normal categories
    std::vector<CategoryStats> merged_stats;
    std::vector<CategoryStats> rare_stats;

    merged_stats.reserve(category_stats.size());
    rare_stats.reserve(category_stats.size());

    for (auto &stats : category_stats) {
      if (static_cast<double>(stats.count) / static_cast<double>(total_count) >=
          bin_cutoff) {
        merged_stats.push_back(std::move(stats));
      } else {
        rare_stats.push_back(std::move(stats));
      }
    }

    // Merge rare categories into a single bin
    if (!rare_stats.empty()) {
      CategoryStats merged_rare;
      for (auto &rare : rare_stats) {
        merged_rare.merge_with(rare);
      }
      // merged_rare.compute_event_rate(); (auto-handled)
      merged_stats.push_back(std::move(merged_rare));
    }
    category_stats = std::move(merged_stats);
  }

  // Ensure maximum number of pre-bins
  void ensure_max_prebins() {
    if (static_cast<int>(category_stats.size()) > max_n_prebins) {
      // Sort by count and keep only the max_n_prebins most frequent
      std::partial_sort(category_stats.begin(),
                        category_stats.begin() + max_n_prebins,
                        category_stats.end(),
                        [](const CategoryStats &a, const CategoryStats &b) {
                          return a.count > b.count;
                        });
      category_stats.resize(max_n_prebins);
    }
  }

  // Compute and sort by event rates
  void compute_and_sort_event_rates() {
    // Event rates are calculated dynamically via event_rate() method
    // No need to iterate through category_stats here

    // Sort by event rate for monotonicity
    std::sort(category_stats.begin(), category_stats.end(),
              [](const CategoryStats &a, const CategoryStats &b) {
                return a.event_rate < b.event_rate;
              });
  }

  // Initialize dynamic programming structures with optimized memory usage
  void initialize_dp_structures() {
    int n = static_cast<int>(category_stats.size());

    // Initialize caches for efficient calculations
    stats_cache = std::make_shared<CumulativeStatsCache>(category_stats);
    iv_cache = std::make_unique<IVCache>(n, stats_cache, n > 20);

    // Initialize DP tables with pre-allocation
    dp.resize(n + 1);
    split_points.resize(n + 1);

    for (int i = 0; i <= n; ++i) {
      dp[i].resize(max_bins + 1, NEG_INFINITY);
      split_points[i].resize(max_bins + 1, 0);
    }

    // Fill base cases for 1 bin
    for (int i = 1; i <= n; ++i) {
      dp[i][1] = iv_cache->calculate_and_cache(0, i);
    }
  }

  // Enhanced dynamic programming algorithm with better efficiency
  void perform_dynamic_programming() {
    int n = static_cast<int>(category_stats.size());

    // Optimized DP algorithm to find optimal binning
    for (int k = 2; k <= max_bins; ++k) {
      // Use banded optimization: we only need to consider splits that would
      // result in at least min_bins bins at the end
      int min_required_per_bin = n / max_bins;

      for (int i = k; i <= n; ++i) {
        // Start from k-1 as we need at least k-1 previous bins
        // Use bounds to skip unnecessary calculations
        int j_start = std::max(k - 1, (i - 1) - (max_bins - k + 1) *
                                                    min_required_per_bin);
        for (int j = j_start; j < i; ++j) {
          double iv_left = dp[j][k - 1];

          // Skip if left side is invalid
          if (iv_left <= NEG_INFINITY + EPSILON)
            continue;

          double iv_right = iv_cache->calculate_and_cache(j, i);
          double iv_val = iv_left + iv_right;

          if (iv_val > dp[i][k]) {
            dp[i][k] = iv_val;
            split_points[i][k] = j;
          }
        }
      }
    }
  }

  // Backtrack to find optimal bins with improved handling of edge cases
  std::vector<int> backtrack_optimal_bins() {
    int n = static_cast<int>(category_stats.size());

    // Find best number of bins within allowed range
    double best_iv = NEG_INFINITY;
    int best_k = min_bins;

    for (int k = min_bins; k <= std::min(max_bins, n); ++k) {
      if (dp[n][k] > best_iv) {
        best_iv = dp[n][k];
        best_k = k;
      }
    }

    // Handle edge case where no valid solution was found
    if (best_iv <= NEG_INFINITY + EPSILON) {
      Rcpp::warning(
          "No valid binning solution found. Using equal-width binning.");
      // Fall back to equal-width binning
      std::vector<int> equal_bins;
      equal_bins.reserve(min_bins);
      int bin_size = n / min_bins;
      for (int i = 1; i <= min_bins; ++i) {
        equal_bins.push_back(std::min(i * bin_size, n));
      }
      return equal_bins;
    }

    // Optimized backtracking
    std::vector<int> bins;
    bins.reserve(best_k);

    int curr_n = n;
    int curr_k = best_k;

    while (curr_k > 0) {
      bins.push_back(curr_n);
      curr_n = split_points[curr_n][curr_k];
      curr_k--;
    }

    std::reverse(bins.begin(), bins.end());
    return bins;
  }

  // Enhanced monotonicity check with adaptive threshold
  bool check_monotonicity(const std::vector<int> &bins) {
    // Calculate average WoE gap for context-aware check
    std::vector<double> woe_values;
    woe_values.reserve(bins.size());

    int start = 0;
    for (int end : bins) {
      int pos = stats_cache->get_pos(start, end);
      int neg = stats_cache->get_neg(start, end);
      int total_pos = stats_cache->get_total_pos();
      int total_neg = stats_cache->get_total_neg();

      // Calculate Bayesian smoothed WoE
      double prior_pos = BAYESIAN_PRIOR_STRENGTH *
                         static_cast<double>(total_pos) /
                         (total_pos + total_neg);
      double prior_neg = BAYESIAN_PRIOR_STRENGTH - prior_pos;

      double pos_rate =
          static_cast<double>(pos + prior_pos) /
          static_cast<double>(total_pos + BAYESIAN_PRIOR_STRENGTH);
      double neg_rate =
          static_cast<double>(neg + prior_neg) /
          static_cast<double>(total_neg + BAYESIAN_PRIOR_STRENGTH);

      double woe = 0.0;
      if (pos_rate > EPSILON && neg_rate > EPSILON) {
        woe = std::log(pos_rate / neg_rate);
      }

      woe_values.push_back(woe);
      start = end;
    }

    // Calculate average gap
    double total_gap = 0.0;
    for (size_t i = 1; i < woe_values.size(); ++i) {
      total_gap += std::abs(woe_values[i] - woe_values[i - 1]);
    }

    double avg_gap =
        woe_values.size() > 1 ? total_gap / (woe_values.size() - 1) : 0.0;

    // Adaptive threshold based on average gap
    double monotonicity_threshold = std::min(EPSILON, avg_gap * 0.01);

    // Check monotonicity with adaptive threshold
    for (size_t i = 1; i < woe_values.size(); ++i) {
      if (woe_values[i] < woe_values[i - 1] - monotonicity_threshold) {
        return false;
      }
    }

    return true;
  }

  // Enhanced monotonicity enforcement with smarter bin merging
  void enforce_monotonicity(std::vector<int> &bins) {
    // Early exit if already monotonic or too few bins
    if (bins.size() <= 2 || check_monotonicity(bins)) {
      return;
    }

    const int max_attempts = static_cast<int>(bins.size() * 3);
    int attempts = 0;

    while (!check_monotonicity(bins) &&
           static_cast<int>(bins.size()) > min_bins &&
           attempts < max_attempts) {
      // Calculate WoE values for all bins
      std::vector<double> woe_values;
      woe_values.reserve(bins.size());

      int start = 0;
      for (int end : bins) {
        int pos = stats_cache->get_pos(start, end);
        int neg = stats_cache->get_neg(start, end);
        int total_pos = stats_cache->get_total_pos();
        int total_neg = stats_cache->get_total_neg();

        // Calculate Bayesian smoothed WoE
        double prior_pos = BAYESIAN_PRIOR_STRENGTH *
                           static_cast<double>(total_pos) /
                           (total_pos + total_neg);
        double prior_neg = BAYESIAN_PRIOR_STRENGTH - prior_pos;

        double pos_rate =
            static_cast<double>(pos + prior_pos) /
            static_cast<double>(total_pos + BAYESIAN_PRIOR_STRENGTH);
        double neg_rate =
            static_cast<double>(neg + prior_neg) /
            static_cast<double>(total_neg + BAYESIAN_PRIOR_STRENGTH);

        double woe = 0.0;
        if (pos_rate > EPSILON && neg_rate > EPSILON) {
          woe = std::log(pos_rate / neg_rate);
        }

        woe_values.push_back(woe);
        start = end;
      }

      // Find the worst violation and fix it
      double worst_violation = 0.0;
      size_t worst_idx = 0;

      for (size_t i = 1; i < woe_values.size(); ++i) {
        double violation = woe_values[i - 1] - woe_values[i];
        if (violation > worst_violation) {
          worst_violation = violation;
          worst_idx = i;
        }
      }

      // Choose which way to merge the violating bins
      // Consider both forward and backward merges and choose the one with
      // higher IV
      double forward_iv = 0.0;
      double backward_iv = 0.0;

      if (worst_idx < bins.size() - 1) {
        // Evaluate forward merge
        std::vector<int> forward_bins = bins;
        forward_bins.erase(forward_bins.begin() + worst_idx + 1);

        int start = 0;
        double total_iv = 0.0;
        for (size_t i = 0; i < forward_bins.size(); ++i) {
          total_iv += iv_cache->calculate_and_cache(start, forward_bins[i]);
          start = forward_bins[i];
        }
        forward_iv = total_iv;
      }

      if (worst_idx > 0) {
        // Evaluate backward merge
        std::vector<int> backward_bins = bins;
        backward_bins.erase(backward_bins.begin() + worst_idx);

        int start = 0;
        double total_iv = 0.0;
        for (size_t i = 0; i < backward_bins.size(); ++i) {
          total_iv += iv_cache->calculate_and_cache(start, backward_bins[i]);
          start = backward_bins[i];
        }
        backward_iv = total_iv;
      }

      // Choose the merge with higher IV
      if (backward_iv > forward_iv && worst_idx > 0) {
        bins.erase(bins.begin() + worst_idx);
      } else if (worst_idx < bins.size() - 1) {
        bins.erase(bins.begin() + worst_idx + 1);
      } else {
        // Fallback if we can't merge either way
        bins.erase(bins.begin() + worst_idx);
      }

      attempts++;
    }

    if (attempts >= max_attempts) {
      Rcpp::warning("Could not ensure monotonicity in %d attempts. Using best "
                    "solution found.",
                    max_attempts);
    }
  }

  // Efficient bin name generation
  std::string join_bin_names(int start, int end) const {
    std::string bin_name;
    // Estimate size to avoid reallocations
    bin_name.reserve((end - start) * 16);

    for (int i = start; i < end; ++i) {
      if (i > start)
        bin_name += bin_separator;
      bin_name += category_stats[i].category;
    }

    return bin_name;
  }

public:
  OBC_IVB(std::vector<std::string> feature, std::vector<int> target,
          double bin_cutoff, int min_bins, int max_bins, int max_n_prebins,
          std::string bin_separator, double convergence_threshold,
          int max_iterations)
      : feature(std::move(feature)), target(std::move(target)),
        bin_cutoff(bin_cutoff), min_bins(min_bins), max_bins(max_bins),
        max_n_prebins(max_n_prebins), bin_separator(std::move(bin_separator)),
        convergence_threshold(convergence_threshold),
        max_iterations(max_iterations), converged(false), iterations_run(0) {}

  List perform_binning() {
    try {
      // Processing steps
      validate_input();
      preprocess_data();
      merge_rare_categories();
      ensure_max_prebins();
      compute_and_sort_event_rates();

      // Adjust parameters based on the dataset
      int ncat = static_cast<int>(category_stats.size());
      min_bins = std::min(min_bins, ncat);
      max_bins = std::min(max_bins, ncat);
      if (max_bins < min_bins)
        max_bins = min_bins;

      std::vector<int> optimal_bins;

      // Special case: already have few enough bins
      if (ncat <= max_bins) {
        converged = true;
        iterations_run = 1;
        optimal_bins.resize(static_cast<size_t>(ncat));
        std::iota(optimal_bins.begin(), optimal_bins.end(), 1);
      } else {
        // Execute dynamic programming algorithm
        initialize_dp_structures();
        perform_dynamic_programming();

        optimal_bins = backtrack_optimal_bins();
        enforce_monotonicity(optimal_bins);

        // Check convergence
        double prev_iv = NEG_INFINITY;
        for (iterations_run = 0; iterations_run < max_iterations;
             ++iterations_run) {
          double current_iv = dp[ncat][static_cast<int>(optimal_bins.size())];
          if (std::fabs(current_iv - prev_iv) < convergence_threshold) {
            converged = true;
            break;
          }
          prev_iv = current_iv;
        }
      }

      // Optimized result preparation
      const size_t n_bins = optimal_bins.size();

      Rcpp::NumericVector ids(n_bins);
      Rcpp::CharacterVector bin_names(n_bins);
      Rcpp::NumericVector woe_values(n_bins);
      Rcpp::NumericVector iv_values(n_bins);
      Rcpp::IntegerVector count_values(n_bins);
      Rcpp::IntegerVector count_pos_values(n_bins);
      Rcpp::IntegerVector count_neg_values(n_bins);

      double total_iv = 0.0;

      int start = 0;
      for (size_t i = 0; i < n_bins; ++i) {
        int end = optimal_bins[i];

        ids[i] = i + 1;
        bin_names[i] = join_bin_names(start, end);

        // Use cache for statistics
        int pos_count = stats_cache->get_pos(start, end);
        int neg_count = stats_cache->get_neg(start, end);
        int total_count = pos_count + neg_count;
        int total_pos = stats_cache->get_total_pos();
        int total_neg = stats_cache->get_total_neg();

        // Calculate WoE and IV with Bayesian smoothing
        double prior_pos = BAYESIAN_PRIOR_STRENGTH *
                           static_cast<double>(total_pos) /
                           (total_pos + total_neg);
        double prior_neg = BAYESIAN_PRIOR_STRENGTH - prior_pos;

        double pos_rate =
            static_cast<double>(pos_count + prior_pos) /
            static_cast<double>(total_pos + BAYESIAN_PRIOR_STRENGTH);
        double neg_rate =
            static_cast<double>(neg_count + prior_neg) /
            static_cast<double>(total_neg + BAYESIAN_PRIOR_STRENGTH);

        double woe = 0.0;
        double iv_val = 0.0;

        if (pos_rate > EPSILON && neg_rate > EPSILON) {
          woe = std::log(pos_rate / neg_rate);
          iv_val = (pos_rate - neg_rate) * woe;

          // Protect against non-finite values
          if (!std::isfinite(woe))
            woe = 0.0;
          if (!std::isfinite(iv_val))
            iv_val = 0.0;
        }

        woe_values[i] = woe;
        iv_values[i] = iv_val;
        count_values[i] = total_count;
        count_pos_values[i] = pos_count;
        count_neg_values[i] = neg_count;

        total_iv += iv_val;
        start = end;
      }

      return Rcpp::List::create(
          Named("id") = ids, Named("bin") = bin_names,
          Named("woe") = woe_values, Named("iv") = iv_values,
          Named("count") = count_values, Named("count_pos") = count_pos_values,
          Named("count_neg") = count_neg_values, Named("total_iv") = total_iv,
          Named("converged") = converged, Named("iterations") = iterations_run);
    } catch (const std::exception &e) {
      Rcpp::stop("Error in optimal binning: %s", e.what());
    }
  }
};

// [[Rcpp::export]]
List optimal_binning_categorical_ivb(IntegerVector target, SEXP feature,
                                     int min_bins = 3, int max_bins = 5,
                                     double bin_cutoff = 0.05,
                                     int max_n_prebins = 20,
                                     std::string bin_separator = "%;%",
                                     double convergence_threshold = 1e-6,
                                     int max_iterations = 1000) {

  // Quick input validation
  if (target.size() == 0) {
    stop("Target vector cannot be empty");
  }

  // Optimized target conversion to std::vector
  std::vector<int> target_vec;
  target_vec.reserve(target.size());

  int na_count = 0;
  for (int t : target) {
    if (IntegerVector::is_na(t)) {
      na_count++;
      continue; // Skip NA values
    }
    target_vec.push_back(t);
  }

  if (na_count > 0) {
    Rcpp::warning("%d missing values found in target and removed.", na_count);
  }

  // Efficient feature conversion to std::vector<std::string>
  std::vector<std::string> feature_vec;
  feature_vec.reserve(target_vec.size());

  int feature_na_count = 0;

  if (Rf_isFactor(feature)) {
    IntegerVector levels = as<IntegerVector>(feature);
    CharacterVector level_names = levels.attr("levels");

    for (int i = 0; i < levels.size(); ++i) {
      if (IntegerVector::is_na(levels[i])) {
        feature_vec.push_back("NA");
        feature_na_count++;
      } else {
        feature_vec.push_back(as<std::string>(level_names[levels[i] - 1]));
      }
    }
  } else if (TYPEOF(feature) == STRSXP) {
    CharacterVector chars = as<CharacterVector>(feature);
    for (R_xlen_t i = 0; i < chars.size(); ++i) {
      if (chars[i] == NA_STRING) {
        feature_vec.push_back("NA");
        feature_na_count++;
      } else {
        feature_vec.push_back(as<std::string>(chars[i]));
      }
    }
  } else {
    stop("Feature must be a factor or character vector");
  }

  if (feature_na_count > 0) {
    Rcpp::warning(
        "%d missing values found in feature and converted to \"NA\" category.",
        feature_na_count);
  }

  // Remove observations with NA in target
  if (na_count > 0) {
    std::vector<std::string> filtered_feature;
    std::vector<int> filtered_target;

    filtered_feature.reserve(feature_vec.size() - na_count);
    filtered_target.reserve(target_vec.size());

    int j = 0;
    for (int i = 0; i < target.size(); ++i) {
      if (!IntegerVector::is_na(target[i])) {
        filtered_feature.push_back(feature_vec[j]);
        filtered_target.push_back(target[i]);
      }
      j++;
    }

    feature_vec = std::move(filtered_feature);
    target_vec = std::move(filtered_target);
  }

  // Quick dimension check
  if (feature_vec.size() != target_vec.size()) {
    stop("Feature and target vectors must have the same length after NA "
         "handling");
  }

  // Handle empty dataset after NA removal
  if (feature_vec.empty()) {
    stop("No valid observations after removing missing values");
  }

  // Adjust parameters based on dataset
  std::set<std::string> unique_categories(feature_vec.begin(),
                                          feature_vec.end());
  int ncat = static_cast<int>(unique_categories.size());

  min_bins = std::min(min_bins, ncat);
  max_bins = std::min(max_bins, ncat);
  if (max_bins < min_bins) {
    max_bins = min_bins;
  }

  // Execute optimized algorithm
  OBC_IVB binner(std::move(feature_vec), std::move(target_vec), bin_cutoff,
                 min_bins, max_bins, max_n_prebins, bin_separator,
                 convergence_threshold, max_iterations);

  return binner.perform_binning();
}