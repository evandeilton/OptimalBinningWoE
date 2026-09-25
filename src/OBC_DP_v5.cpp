// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <unordered_map>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <chrono>

// Include shared headers
#include "common/optimal_binning_common.h"
#include "common/bin_structures.h"
#include "common/monotonicity_utils.h"

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

namespace {

constexpr double NEGATIVE_INFINITY = -std::numeric_limits<double>::infinity();

/**
 * @brief Calculate Weight of Evidence (WoE) and Information Value (IV)
 *
 * @param count_pos Positive events in the bin
 * @param count_neg Negative events in the bin
 * @param total_pos Total positive events in the dataset
 * @param total_neg Total negative events in the dataset
 * @param woe Output parameter for Weight of Evidence
 * @param iv Output parameter for Information Value
 */
inline void compute_woe_iv(double count_pos, double count_neg, double total_pos, double total_neg,
                           double &woe, double &iv) {
  // Prevent division by zero and log of zero
  if (count_pos <= EPSILON || count_neg <= EPSILON) {
    woe = 0.0;
    iv = 0.0;
    return;
  }

  // Calculate distribution ratios
  const double dist_pos = count_pos / total_pos;
  const double dist_neg = count_neg / total_neg;

  // Calculate WoE and IV
  woe = std::log(dist_pos / dist_neg);
  iv = (dist_pos - dist_neg) * woe;
}

/**
 * @brief A pre-bin: one or more categories, their label and their counts.
 *
 * Pre-bins used to be handled as bare label strings ("a%;%b%;%c") whose counts
 * were recovered by splitting the label on bin_separator and looking every
 * piece up again. A category whose name contains the separator was therefore
 * split into pieces that either do not exist (its rows were dropped) or name
 * another category (whose rows were counted twice): counts no longer summed to
 * n. Carrying the counts with the label removes the round trip through the
 * string, and with it the parsing, the per-label hash cache and the
 * category-to-label map that was written but never read.
 */
struct PreBin {
  std::string name;
  int count = 0;
  int count_pos = 0;
};

} // namespace

/**
 * @class OBC_DP
 * @brief Optimal binning for categorical variables using dynamic programming
 *
 * This class implements an algorithm for optimal binning of categorical variables
 * using dynamic programming with linear constraints (e.g., monotonicity).
 * The algorithm aims to maximize the total Information Value (IV) while
 * respecting constraints on the number of bins and other requirements.
 *
 * Based on the methodology described in:
 * - Navas-Palencia, G. (2022). OptBinning: Mathematical Optimization for Optimal Binning.
 * - Siddiqi, N. (2017). Intelligent Credit Scoring: Building and Implementing Better Credit Risk Scorecards.
 * - Thomas, L.C., Edelman, D.B., & Crook, J.N. (2017). Credit Scoring and Its Applications.
 */
class OBC_DP {
public:
  /**
   * @brief Constructor for OBC_DP
   *
   * @param feature_ Vector of categorical feature values
   * @param target_ Vector of binary target values (0/1)
   * @param min_bins_ Minimum number of bins to create
   * @param max_bins_ Maximum number of bins to create
   * @param bin_cutoff_ Minimum proportion of observations for a bin
   * @param max_n_prebins_ Maximum number of pre-bins before final optimization
   * @param convergence_threshold_ Threshold for algorithm convergence
   * @param max_iterations_ Maximum number of iterations
   * @param bin_separator_ String separator for concatenating category names
   * @param monotonic_trend_ Force monotonic trend ('auto', 'ascending', 'descending', 'none')
   */
  OBC_DP(const std::vector<std::string> &feature_,
         const std::vector<int> &target_,
         int min_bins_,
         int max_bins_,
         double bin_cutoff_,
         int max_n_prebins_,
         double convergence_threshold_,
         int max_iterations_,
         const std::string &bin_separator_,
         const std::string &monotonic_trend_ = "auto") :
  feature(feature_),
  target(target_),
  min_bins(min_bins_),
  max_bins(max_bins_),
  bin_cutoff(bin_cutoff_),
  max_n_prebins(max_n_prebins_),
  convergence_threshold(convergence_threshold_),
  max_iterations(max_iterations_),
  bin_separator(bin_separator_),
  monotonic_trend(monotonic_trend_),
  total_count(0.0),
  total_pos(0.0),
  total_neg(0.0),
  converged(false),
  iterations_run(0),
  total_iv(0.0),
  execution_time_ms(0) {
    // The reserve() below fixes the bucket layout, and with it the iteration
    // order of category_info, which decides the order of equal-rate categories
    // after sorting. Keep it unchanged.
    const size_t est_categories = std::min(feature.size() / 4, static_cast<size_t>(1024));
    category_info.reserve(est_categories);
    if (max_bins > 0) bin_results.reserve(static_cast<size_t>(max_bins));
  }

  /**
   * @brief Perform the optimal binning algorithm
   *
   * @return Rcpp::List List containing the binning results
   */
  Rcpp::List perform_binning() {
    try {
      auto start_time = std::chrono::high_resolution_clock::now();

      // Step 1: Validate input parameters
      validate_input();

      // Step 2: Preprocess data (counts and statistics)
      preprocess_data();

      // Check if we already have fewer categories than max_bins
      size_t ncat = category_info.size();
      if (ncat <= static_cast<size_t>(max_bins)) {
        compute_bins_no_optimization();
      } else {
        // Step 3: Merge rare categories
        merge_rare_categories();

        // Step 4: Limit the number of pre-bins
        ensure_max_prebins();

        // Step 5: Calculate event rates and sort categories
        compute_and_sort_event_rates();

        // Step 6: Initialize DP structures
        initialize_dp_structures();

        // Step 7: Perform dynamic programming optimization
        perform_dynamic_programming();

        // Step 8: Backtrack to find optimal bins
        backtrack_optimal_bins();
      }

      // Calculate total IV across all bins
      calculate_total_iv();

      auto end_time = std::chrono::high_resolution_clock::now();
      execution_time_ms = static_cast<long>(std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count());

      // Step 9: Return results
      return prepare_output();

    } catch (const std::exception &e) {
      Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
    }
  }

  /// Number of categories whose name contains bin_separator (and one of them)
  std::size_t separator_hits(std::string& example) const {
    return count_separator_hits(category_info, bin_separator, example);
  }

private:
  // Input parameters
  const std::vector<std::string> &feature;
  const std::vector<int> &target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  double convergence_threshold;
  int max_iterations;
  std::string bin_separator;
  std::string monotonic_trend;

  // Auxiliary variables
  std::unordered_map<std::string, CategoryStats> category_info;
  double total_count;
  double total_pos;
  double total_neg;

  // Pre-bins after rare-category merging, and the same pre-bins in DP order
  std::vector<PreBin> merged_bins;
  std::vector<PreBin> sorted_bins;

  // DP structures
  std::vector<std::vector<double>> dp;
  std::vector<std::vector<int>> prev_bin;
  std::vector<double> cum_count_pos;
  std::vector<double> cum_count_neg;

  // Final results
  std::vector<CategoricalBin> bin_results;

  // Execution results
  bool converged;
  int iterations_run;
  double total_iv;
  long execution_time_ms;

  /**
   * @brief Validate input parameters
   */
  void validate_input() {
    if (min_bins < 2) {
      throw std::invalid_argument("min_bins must be >= 2.");
    }
    if (max_bins < min_bins) {
      throw std::invalid_argument("max_bins must be >= min_bins.");
    }
    if (feature.size() != target.size()) {
      throw std::invalid_argument("feature and target must have the same size.");
    }
    if (feature.empty()) {
      throw std::invalid_argument("Input vectors cannot be empty.");
    }
    if (bin_cutoff <= 0 || bin_cutoff >= 1) {
      throw std::invalid_argument("bin_cutoff must be between 0 and 1 (exclusive).");
    }
    // Documented as ">= 2". It was not checked, and max_n_prebins = 0 made
    // ensure_max_prebins() read past the end of a one-element vector.
    if (max_n_prebins < 2) {
      throw std::invalid_argument("max_n_prebins must be >= 2.");
    }
    if (convergence_threshold <= 0) {
      throw std::invalid_argument("convergence_threshold must be positive.");
    }
    if (max_iterations <= 0) {
      throw std::invalid_argument("max_iterations must be positive.");
    }

    // Validate monotonic_trend parameter
    if (monotonic_trend != "auto" && monotonic_trend != "ascending" &&
        monotonic_trend != "descending" && monotonic_trend != "none") {
      throw std::invalid_argument("monotonic_trend must be one of: 'auto', 'ascending', 'descending', 'none'.");
    }

    // Check if target is binary. Every value is checked: stopping at the first
    // 0 and 1 let a later 2 (or -1) through, to be counted as a positive.
    bool has_zero = false;
    bool has_one = false;
    for (int t : target) {
      if (t == 0) has_zero = true;
      else if (t == 1) has_one = true;
      else throw std::invalid_argument("Target must contain only values 0 and 1.");
    }

    if (!has_zero || !has_one) {
      throw std::invalid_argument("Target must contain both values 0 and 1.");
    }
  }

  /**
   * @brief Count occurrences and positives per category in a single pass.
   */
  void preprocess_data() {
    total_count = static_cast<double>(feature.size());
    total_pos = 0.0;

    for (size_t i = 0; i < feature.size(); ++i) {
      const int is_positive = target[i];
      category_info[feature[i]].update(is_positive);
      if (is_positive) total_pos += 1.0;
    }

    total_neg = total_count - total_pos;
  }

  /**
   * @brief One bin per category, used when there are at most max_bins categories.
   */
  void compute_bins_no_optimization() {
    for (const auto& pair : category_info) {
      const std::string& cat = pair.first;
      const CategoryStats& info = pair.second;

      double woe, iv;
      compute_woe_iv(info.count_pos, info.count_neg, total_pos, total_neg, woe, iv);

      bin_results.emplace_back(cat, woe, iv, info.count, info.count_pos, info.count_neg);
    }

    // Sort bins by event rate if monotonicity is required
    if (monotonic_trend != "none") {
      std::sort(bin_results.begin(), bin_results.end(),
                [](const CategoricalBin& a, const CategoricalBin& b) {
                  return a.event_rate() < b.event_rate();
                });
    }

    // One bin per category, already within max_bins: an exact, final result.
    converged = true;
    iterations_run = 1;
  }

  /**
   * @brief Group categories with fewer observations than the bin_cutoff threshold.
   *
   * Categories are visited in ascending order of count; rare ones are
   * accumulated into a group until the group reaches the cutoff.
   */
  void merge_rare_categories() {
    const double cutoff_count = bin_cutoff * total_count;

    std::vector<std::pair<std::string, const CategoryStats*>> sorted_cats;
    sorted_cats.reserve(category_info.size());
    for (const auto& pair : category_info) {
      sorted_cats.emplace_back(pair.first, &pair.second);
    }

    std::sort(sorted_cats.begin(), sorted_cats.end(),
              [](const auto& a, const auto& b) {
                return a.second->count < b.second->count;
              });

    merged_bins.clear();
    merged_bins.reserve(sorted_cats.size());

    PreBin current;
    bool open = false;

    for (const auto& [cat, info] : sorted_cats) {
      if (info->count < cutoff_count) {
        // Rare category, add to current group
        if (!open) {
          current.name = cat;
          open = true;
        } else {
          current.name += bin_separator;
          current.name += cat;
        }
        current.count += info->count;
        current.count_pos += info->count_pos;

        // Close the group once it reaches the threshold
        if (current.count >= cutoff_count) {
          merged_bins.push_back(std::move(current));
          current = PreBin();
          open = false;
        }
      } else {
        // Non-rare category: flush the open group first
        if (open) {
          merged_bins.push_back(std::move(current));
          current = PreBin();
          open = false;
        }
        merged_bins.push_back(PreBin{cat, info->count, info->count_pos});
      }
    }

    if (open) {
      merged_bins.push_back(std::move(current));
    }

    // Grouping left fewer pre-bins than min_bins (a large bin_cutoff pools
    // most categories together). No partition could then reach min_bins and
    // the fit used to abort with "Failed to find optimal binning with the
    // given constraints." Fall back to one pre-bin per category: this path
    // runs only when there are more categories than max_bins >= min_bins,
    // and ensure_max_prebins() and the DP do the grouping.
    if (merged_bins.size() < static_cast<size_t>(min_bins)) {
      merged_bins.clear();
      for (const auto& [cat, info] : sorted_cats) {
        merged_bins.push_back(PreBin{cat, info->count, info->count_pos});
      }
    }
  }

  /**
   * @brief Reduce the number of pre-bins to max_n_prebins by repeatedly
   * merging the two smallest ones.
   */
  void ensure_max_prebins() {
    const size_t limit = static_cast<size_t>(max_n_prebins);
    if (merged_bins.size() <= limit) {
      return;
    }

    auto by_count = [](const PreBin& a, const PreBin& b) {
      return a.count < b.count;
    };
    std::sort(merged_bins.begin(), merged_bins.end(), by_count);

    while (merged_bins.size() > limit) {
      PreBin merged;
      merged.name = merged_bins[0].name + bin_separator + merged_bins[1].name;
      merged.count = merged_bins[0].count + merged_bins[1].count;
      merged.count_pos = merged_bins[0].count_pos + merged_bins[1].count_pos;

      merged_bins.erase(merged_bins.begin(), merged_bins.begin() + 2);

      // Insert at the sorted position (binary search) instead of re-sorting.
      auto ins_pos = std::lower_bound(merged_bins.begin(), merged_bins.end(),
                                      merged, by_count);
      merged_bins.insert(ins_pos, std::move(merged));
    }
  }

  /**
   * @brief Sort the pre-bins by event rate (the DP order).
   */
  void compute_and_sort_event_rates() {
    std::vector<std::pair<size_t, double>> rates;
    rates.reserve(merged_bins.size());
    for (size_t i = 0; i < merged_bins.size(); ++i) {
      const PreBin& b = merged_bins[i];
      rates.emplace_back(i, b.count_pos / static_cast<double>(b.count));
    }

    std::sort(rates.begin(), rates.end(),
              [](const auto& a, const auto& b) {
                return a.second < b.second;
              });

    sorted_bins.clear();
    sorted_bins.reserve(rates.size());
    for (const auto& r : rates) {
      sorted_bins.push_back(merged_bins[r.first]);
    }

    if (monotonic_trend == "descending") {
      std::reverse(sorted_bins.begin(), sorted_bins.end());
    } else if (monotonic_trend == "auto") {
      // Detect the trend from the event rates in the sorted order.
      std::vector<double> event_rates;
      event_rates.reserve(rates.size());
      for (const auto& r : rates) {
        event_rates.push_back(r.second);
      }
      MonotonicTrend detected = detect_trend_welford_woe(event_rates);
      if (detected == MonotonicTrend::DESCENDING) { // # nocov start (rates are sorted ascending, so the slope is never negative)
        std::reverse(sorted_bins.begin(), sorted_bins.end());
        monotonic_trend = "descending";
      } else { // # nocov end
        monotonic_trend = "ascending";
      }
    }
  }

  /**
   * @brief Set up the DP tables and the cumulative counts.
   */
  void initialize_dp_structures() {
    const size_t n = sorted_bins.size();
    const size_t kmax = static_cast<size_t>(max_bins) + 1;

    dp.assign(n + 1, std::vector<double>(kmax, NEGATIVE_INFINITY));
    prev_bin.assign(n + 1, std::vector<int>(kmax, -1));
    dp[0][0] = 0.0; // Base case

    cum_count_pos.assign(n + 1, 0.0);
    cum_count_neg.assign(n + 1, 0.0);
    for (size_t i = 0; i < n; ++i) {
      const PreBin& b = sorted_bins[i];
      cum_count_pos[i + 1] = cum_count_pos[i] + b.count_pos;
      cum_count_neg[i + 1] = cum_count_neg[i] + (b.count - b.count_pos);
    }
  }

  /**
   * @brief Fill the DP table: dp[i][k] = best total IV of the first i
   * pre-bins in k bins.
   */
  void perform_dynamic_programming() {
    const size_t n = sorted_bins.size();

    converged = true;   // DP is exact: one pass
    iterations_run = 1;

    // Monotonicity needs no test inside the recurrence. The pre-bins are
    // sorted by event rate (reversed for "descending"), and the pooled event
    // rate of a run of consecutive pre-bins lies between those of its first
    // and last members, so every contiguous partition has monotone event
    // rates -- and WoE is an increasing function of the event rate.
    //
    // The test that used to sit here compared the WoE of the single pre-bin
    // j - 1 (not of the previous bin) with that of the candidate bin, using
    // compute_woe_iv(), which reports WoE = 0 for a bin without positives or
    // without negatives. It could therefore only fire on that artefact: a
    // pure pre-bin "violated" the order against its neighbour, valid
    // partitions were discarded, and the fit returned fewer bins than
    // min_bins or failed with "Failed to find optimal binning".

    // Precompute the IV of every (j, i) segment -- O(n^2), done once
    std::vector<std::vector<double>> seg_iv(n + 1);
    for (size_t i = 1; i <= n; ++i) {
      seg_iv[i].resize(i);
      for (size_t j = 0; j < i; ++j) {
        double cp = cum_count_pos[i] - cum_count_pos[j];
        double cn = cum_count_neg[i] - cum_count_neg[j];
        double woe, iv;
        compute_woe_iv(cp, cn, total_pos, total_neg, woe, iv);
        seg_iv[i][j] = iv;
      }
    }

    // Main DP pass -- O(n^2 * k)
    for (size_t i = 1; i <= n; ++i) {
      for (int k = 1; k <= max_bins && k <= static_cast<int>(i); ++k) {
        const size_t ku = static_cast<size_t>(k);
        for (size_t j = (k > 1 ? ku - 1 : 0); j < i; ++j) {
          double candidate = dp[j][ku - 1] + seg_iv[i][j];
          if (candidate > dp[i][ku]) {
            dp[i][ku] = candidate;
            prev_bin[i][ku] = static_cast<int>(j);
          }
        }
      }
    }
  }

  /**
   * @brief Pick the best number of bins and trace the optimal partition back.
   */
  void backtrack_optimal_bins() {
    const size_t n = sorted_bins.size();
    double max_total_iv = NEGATIVE_INFINITY;
    int best_k = -1;

    // A partition of n pre-bins has at most n bins. When max_n_prebins is
    // below min_bins, no k in [min_bins, max_bins] is feasible; that used to
    // abort with "Failed to find optimal binning with the given constraints."
    // The search now starts at min(min_bins, n), i.e. it returns every
    // pre-bin as a bin. Every dp[n][k] with k <= n is finite, so a k is
    // always found.
    const int k_lo = std::min(min_bins, static_cast<int>(n));
    for (int k = k_lo; k <= max_bins; ++k) {
      const size_t ku = static_cast<size_t>(k);
      if (dp[n][ku] > max_total_iv) {
        max_total_iv = dp[n][ku];
        best_k = k;
      }
    }

    if (best_k == -1) { // # nocov start (unreachable, see above)
      throw std::runtime_error("Failed to find optimal binning with the given constraints.");
    } // # nocov end

    // Determine bin edges using backtracking
    std::vector<size_t> bin_edges;
    bin_edges.reserve(static_cast<size_t>(best_k));

    size_t idx = n;
    int k = best_k;

    // Guard against the -1 sentinel in prev_bin before casting to size_t.
    while (k > 0) {
      int prev_j = prev_bin[idx][static_cast<size_t>(k)];
      if (prev_j < 0 || static_cast<size_t>(prev_j) > n) { // # nocov start (defensive: every dp[n][k] used here is finite)
        throw std::runtime_error(
          "DP backtracking failed: invalid predecessor index (" +
          std::to_string(prev_j) + "). The DP table may be incomplete."
        );
      } // # nocov end
      bin_edges.push_back(static_cast<size_t>(prev_j));
      idx = static_cast<size_t>(prev_j);
      k -= 1;
    }

    std::reverse(bin_edges.begin(), bin_edges.end());

    // Build final bins
    bin_results.clear();
    bin_results.reserve(static_cast<size_t>(best_k));

    size_t start = 0;
    for (size_t edge_idx = 0; edge_idx <= bin_edges.size(); ++edge_idx) {
      size_t end = (edge_idx < bin_edges.size()) ? bin_edges[edge_idx] : n;

      if (start >= end) continue;

      int bin_count = 0;
      int bin_count_pos = 0;
      std::string bin_name;

      for (size_t i = start; i < end; ++i) {
        if (i > start) bin_name += bin_separator;
        bin_name += sorted_bins[i].name;
        bin_count += sorted_bins[i].count;
        bin_count_pos += sorted_bins[i].count_pos;
      }

      int bin_count_neg = bin_count - bin_count_pos;

      if (bin_count > 0) {
        double woe, iv;
        compute_woe_iv(bin_count_pos, bin_count_neg, total_pos, total_neg, woe, iv);
        bin_results.emplace_back(bin_name, woe, iv, bin_count, bin_count_pos, bin_count_neg);
      }

      start = end;
    }
  }

  /**
   * @brief Sum the IV of the final bins.
   */
  void calculate_total_iv() {
    total_iv = 0.0;
    for (const auto& bin : bin_results) {
      total_iv += bin.iv;
    }
  }

  /**
   * @brief Organize the binning results into an R list.
   */
  Rcpp::List prepare_output() const {
    const size_t n_bins = bin_results.size();

    Rcpp::NumericVector ids(n_bins);
    Rcpp::CharacterVector bin_names(n_bins);
    Rcpp::NumericVector woe_values(n_bins);
    Rcpp::NumericVector iv_values(n_bins);
    Rcpp::IntegerVector count_values(n_bins);
    Rcpp::IntegerVector pos_count_values(n_bins);
    Rcpp::IntegerVector neg_count_values(n_bins);
    Rcpp::NumericVector event_rate_values(n_bins);

    for (size_t i = 0; i < n_bins; ++i) {
      ids[i] = static_cast<double>(i + 1);
      // Each result bin holds its full label as its single "category".
      bin_names[i] = bin_results[i].categories.front();
      woe_values[i] = bin_results[i].woe;
      iv_values[i] = bin_results[i].iv;
      count_values[i] = bin_results[i].count;
      pos_count_values[i] = bin_results[i].count_pos;
      neg_count_values[i] = bin_results[i].count_neg;
      event_rate_values[i] = bin_results[i].event_rate();
    }

    return Rcpp::List::create(
      Rcpp::Named("id") = ids,
      Rcpp::Named("bin") = bin_names,
      Rcpp::Named("woe") = woe_values,
      Rcpp::Named("iv") = iv_values,
      Rcpp::Named("count") = count_values,
      Rcpp::Named("count_pos") = pos_count_values,
      Rcpp::Named("count_neg") = neg_count_values,
      Rcpp::Named("event_rate") = event_rate_values,
      Rcpp::Named("total_iv") = total_iv,
      Rcpp::Named("converged") = converged,
      Rcpp::Named("iterations") = iterations_run,
      Rcpp::Named("execution_time_ms") = execution_time_ms
    );
  }
};


// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_dp(
   Rcpp::IntegerVector target,
   Rcpp::CharacterVector feature,
   int min_bins = 3,
   int max_bins = 5,
   double bin_cutoff = 0.05,
   int max_n_prebins = 20,
   double convergence_threshold = 1e-6,
   int max_iterations = 1000,
   std::string bin_separator = "%;%",
   std::string monotonic_trend = "auto"
) {
 if (feature.size() == 0 || target.size() == 0) {
   Rcpp::stop("Input vectors cannot be empty.");
 }

 if (feature.size() != target.size()) {
   Rcpp::stop("feature and target must have the same size.");
 }

 // Convert R vectors to C++
 std::vector<std::string> feature_vec;
 std::vector<int> target_vec;

 feature_vec.reserve(static_cast<size_t>(feature.size()));
 target_vec.reserve(static_cast<size_t>(target.size()));

 for (R_xlen_t i = 0; i < feature.size(); ++i) {
   if (feature[i] == NA_STRING) {
     feature_vec.push_back("NA");
   } else {
     feature_vec.push_back(Rcpp::as<std::string>(feature[i]));
   }

   if (IntegerVector::is_na(target[i])) {
     Rcpp::stop("Target cannot contain missing values.");
   } else {
     target_vec.push_back(target[i]);
   }
 }

 OBC_DP binning(
     feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins,
     convergence_threshold, max_iterations, bin_separator, monotonic_trend
 );

 Rcpp::List res = binning.perform_binning();
 std::string example;
 warn_separator_hits(bin_separator, binning.separator_hits(example), example);
 return res;
}
