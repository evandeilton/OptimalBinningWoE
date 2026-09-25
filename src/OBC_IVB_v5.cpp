// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <string>
#include <unordered_map>
#include <vector>


// Include shared headers
#include "common/bin_structures.h"
#include "common/optimal_binning_common.h"

using namespace Rcpp;
using namespace OptimalBinning;

namespace {

// Category names that contain bin_separator. A bin label is its categories
// joined with the separator, and obwoe_apply() recovers the categories by
// splitting the label on it, so such a name is cut into pieces that are not
// categories (and a piece equal to another category claims it too). The
// binning itself is unaffected; the caller is warned that its labels are
// ambiguous. Checked over the distinct categories only, so the cost is O(k).
inline const std::string& separator_key(const std::string& s) { return s; }
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
constexpr double NEG_INFINITY = -std::numeric_limits<double>::infinity();

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
  IVCache(size_t size, std::shared_ptr<CumulativeStatsCache> stats,
          bool use_cache = true)
      : stats_cache(std::move(stats)), enabled(use_cache) {
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

      // Ensure finite values (both rates are > EPSILON here, so this is a
      // safeguard only)
      if (!std::isfinite(iv)) { // # nocov start
        iv = 0.0;
      } // # nocov end
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
      // Every value is checked: stopping at the first 0 and 1 let a later 2
      // through, to be counted as a positive.
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

    if (rare_stats.empty()) {
      category_stats = std::move(merged_stats);
      return;
    }

    // The label of a pooled bin is built with the caller's bin_separator.
    // It used to be built with CategoryStats::merge_with()'s default "%;%"
    // whatever bin_separator was, so with a custom separator the pooled
    // categories could not be recovered from the label (obwoe_apply() splits
    // labels on the model's separator) and were never mapped to a WoE.
    if (static_cast<int>(merged_stats.size()) + 1 >= min_bins) {
      // Merge rare categories into a single bin
      CategoryStats merged_rare;
      for (auto &rare : rare_stats) {
        merged_rare.merge_with(rare, bin_separator);
      }
      merged_stats.push_back(std::move(merged_rare));
    } else {
      // Pooling every rare category into one bin would leave fewer than
      // min_bins bins: on a high-cardinality feature whose levels are all
      // below bin_cutoff (500 levels at 0.2% each) the whole sample came back
      // as one bin with IV = 0. Pool rare categories in event-rate order
      // instead, closing each pool once it reaches bin_cutoff, so similar
      // categories stay together and every pool but the last meets the cutoff.
      std::vector<CategoryStats> by_rate = rare_stats;
      std::stable_sort(by_rate.begin(), by_rate.end(),
                       [](const CategoryStats &a, const CategoryStats &b) {
                         return a.event_rate < b.event_rate;
                       });
      std::vector<CategoryStats> pools;
      CategoryStats cur;
      for (auto &rare : by_rate) {
        cur.merge_with(rare, bin_separator);
        if (static_cast<double>(cur.count) / static_cast<double>(total_count) >= bin_cutoff) {
          pools.push_back(std::move(cur));
          cur = CategoryStats();
        }
      }
      if (cur.count > 0) {
        pools.push_back(std::move(cur));
      }
      if (static_cast<int>(merged_stats.size() + pools.size()) >= min_bins) {
        for (auto &pool : pools) merged_stats.push_back(std::move(pool));
      } else {
        // Even the pools are too few (a very large bin_cutoff): keep every
        // category and let max_n_prebins and the DP do the grouping.
        for (auto &rare : rare_stats) merged_stats.push_back(std::move(rare));
      }
    }
    category_stats = std::move(merged_stats);
  }

  // Ensure maximum number of pre-bins
  //
  // The excess categories are folded into the smallest of the retained ones.
  // They used to be dropped instead: the vector was simply resized, which
  // removed those categories' observations from the binning altogether. The
  // loss was silent -- no warning, no error flag, and `converged` still true --
  // and it is reachable whenever more than `max_n_prebins` categories survive
  // merge_rare_categories(), which a bin_cutoff below 1/max_n_prebins allows.
  //
  // Folding rather than dropping keeps every observation represented, so the
  // reported counts, WoE and IV describe the whole sample as they claim to.
  // Which categories are kept as separate identities is unchanged: still the
  // max_n_prebins most frequent.
  void ensure_max_prebins() {
    // max_n_prebins is not range-validated for this algorithm, so the number of
    // bins to keep is floored at one; keeping zero would discard everything,
    // which is the very failure this function exists to avoid.
    const int keep = std::max(max_n_prebins, 1);
    const int n_stats = static_cast<int>(category_stats.size());

    if (n_stats > keep) {
      // Sort by count and keep only the `keep` most frequent
      std::partial_sort(category_stats.begin(), category_stats.begin() + keep,
                        category_stats.end(),
                        [](const CategoryStats &a, const CategoryStats &b) {
                          return a.count > b.count;
                        });

      // Fold the remainder into the smallest retained category. partial_sort
      // orders the retained block by descending count, so that is the last one.
      CategoryStats &absorber = category_stats[keep - 1];
      for (int i = keep; i < n_stats; ++i) {
        absorber.merge_with(category_stats[i], bin_separator);
      }

      category_stats.resize(keep);
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

  // Build the cumulative statistics cache if it does not exist yet.
  // The cache is consumed unconditionally when the result is assembled, so it
  // must also be available on the "few enough categories" fast path that skips
  // the dynamic programming stage entirely.
  void ensure_stats_cache() {
    if (!stats_cache) {
      stats_cache = std::make_shared<CumulativeStatsCache>(category_stats);
    }
  }

  // Initialize dynamic programming structures with optimized memory usage
  void initialize_dp_structures() {
    int n = static_cast<int>(category_stats.size());

    // Initialize caches for efficient calculations
    ensure_stats_cache();
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

    // DP[i][k] = max_{k-1 <= j < i} DP[j][k-1] + IV(j, i), as documented.
    //
    // The j range used to be "banded" to
    //   j >= (i - 1) - (max_bins - k + 1) * floor(n / max_bins),
    // which is not a feasibility bound: it capped the size of the last bin
    // (at floor(n / max_bins) + 1 pre-bins for k = max_bins), so partitions
    // with one large bin -- often the IV-optimal ones when several pre-bins
    // share an event rate -- were never evaluated and the result was not the
    // optimum the algorithm promises. The full range is O(n^2 k) with
    // n <= max_n_prebins, which is negligible.
    for (int k = 2; k <= max_bins; ++k) {
      for (int i = k; i <= n; ++i) {
        for (int j = k - 1; j < i; ++j) {
          // dp[j][k-1] is finite for every j >= k-1 (base case: all
          // dp[i][1] are finite IVs), so no transition needs skipping.
          double iv_left = dp[j][k - 1];

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

    // Every dp[n][k] with k <= n is finite now that the full recurrence is
    // evaluated, so a solution always exists. The "equal-width" fallback that
    // stood here was only reachable through the old banded j range, and it
    // lost categories itself (its last boundary was min_bins * (n / min_bins),
    // not n).
    if (best_iv <= NEG_INFINITY + EPSILON) { // # nocov start
      throw std::runtime_error("No valid binning solution found.");
    } // # nocov end

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
        woe_values.size() > 1
            ? total_gap / static_cast<double>(woe_values.size() - 1)
            : 0.0;

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

    // Every pass removes one boundary, so the loop ends after at most
    // bins.size() - min_bins passes.
    //
    // The DP optimum has not been seen to need this repair (no violation in
    // thousands of randomized inputs, including ones built to invert the
    // smoothed WoE order), so the loop is kept as a safeguard.
    // # nocov start
    while (!check_monotonicity(bins) &&
           static_cast<int>(bins.size()) > min_bins) {
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

      // The violation is between bins worst_idx - 1 and worst_idx
      // (worst_idx >= 1: check_monotonicity() failed, so some violation is
      // positive). bins[] holds each bin's END boundary, so merging bin j
      // with bin j + 1 means erasing bins[j]. Two repairs are possible:
      //   left : merge worst_idx - 1 with worst_idx  -> erase bins[worst_idx - 1]
      //   right: merge worst_idx with worst_idx + 1  -> erase bins[worst_idx]
      //          (only when bin worst_idx + 1 exists)
      // and the one leaving the higher total IV is applied.
      //
      // The indices used to be off by one: the "forward" repair erased
      // bins[worst_idx + 1] and the "backward" one bins[worst_idx]. The
      // former merged two bins that were not in violation at all, and
      // whenever the erased boundary was the last one -- the end of the data,
      // n -- the categories of the final bin silently disappeared from the
      // output (counts no longer summed to the number of observations).
      auto total_iv_without = [&](size_t erase_idx) {
        int seg_start = 0;
        double total_iv = 0.0;
        for (size_t i = 0; i < bins.size(); ++i) {
          if (i == erase_idx) continue;
          total_iv += iv_cache->calculate_and_cache(seg_start, bins[i]);
          seg_start = bins[i];
        }
        return total_iv;
      };

      const size_t left_idx = worst_idx - 1;
      const bool right_ok = (worst_idx + 1 < bins.size());
      const double left_iv = total_iv_without(left_idx);
      if (right_ok && total_iv_without(worst_idx) > left_iv) {
        bins.erase(bins.begin() + static_cast<std::ptrdiff_t>(worst_idx));
      } else {
        bins.erase(bins.begin() + static_cast<std::ptrdiff_t>(left_idx));
      }
    }
    // # nocov end
  }

  // Efficient bin name generation
  std::string join_bin_names(int start, int end) const {
    std::string bin_name;
    // Estimate size to avoid reallocations
    bin_name.reserve(static_cast<size_t>(end - start) * 16);

    for (int i = start; i < end; ++i) {
      if (i > start)
        bin_name += bin_separator;
      bin_name += category_stats[i].category;
    }

    return bin_name;
  }

public:
  OBC_IVB(std::vector<std::string> feature_, std::vector<int> target_,
          double bin_cutoff_, int min_bins_, int max_bins_, int max_n_prebins_,
          std::string bin_separator_, double convergence_threshold_,
          int max_iterations_)
      : feature(std::move(feature_)), target(std::move(target_)),
        bin_cutoff(bin_cutoff_), min_bins(min_bins_), max_bins(max_bins_),
        max_n_prebins(max_n_prebins_), bin_separator(std::move(bin_separator_)),
        convergence_threshold(convergence_threshold_),
        max_iterations(max_iterations_), converged(false), iterations_run(0) {}

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

      // The result assembly below reads from stats_cache on every path, so the
      // cache must be built before branching (the fast path never runs
      // initialize_dp_structures()). category_stats is final at this point.
      ensure_stats_cache();

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

        ids[i] = static_cast<double>(i + 1);
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
  // The NA-target filter below indexes the converted feature by the target's
  // positions, so a shorter feature was read out of bounds.
  if (Rf_xlength(feature) != target.size()) {
    stop("Feature and target vectors must have the same length");
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
    CharacterVector level_names = as<CharacterVector>(levels.attr("levels"));

    for (R_xlen_t i = 0; i < levels.size(); ++i) {
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

    filtered_feature.reserve(target_vec.size());
    filtered_target.reserve(target_vec.size());

    for (R_xlen_t i = 0; i < target.size(); ++i) {
      if (!IntegerVector::is_na(target[i])) {
        filtered_feature.push_back(feature_vec[static_cast<size_t>(i)]);
        filtered_target.push_back(target[i]);
      }
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
  // Only the count is needed; a hash set avoids the O(n log k) string
  // comparisons of the ordered std::set that was used here.
  std::unordered_set<std::string> unique_categories(feature_vec.begin(),
                                                    feature_vec.end());
  int ncat = static_cast<int>(unique_categories.size());
  // A single category used to surface as "min_bins must be at least 2",
  // after min_bins had been clamped to the category count below.
  if (ncat < 2) {
    stop("Feature must have at least 2 distinct categories");
  }

  min_bins = std::min(min_bins, ncat);
  max_bins = std::min(max_bins, ncat);
  if (max_bins < min_bins) {
    max_bins = min_bins;
  }

  // Execute optimized algorithm
  OBC_IVB binner(std::move(feature_vec), std::move(target_vec), bin_cutoff,
                 min_bins, max_bins, max_n_prebins, bin_separator,
                 convergence_threshold, max_iterations);

  std::string sep_example;
  const std::size_t sep_hits =
      count_separator_hits(unique_categories, bin_separator, sep_example);
  List res = binner.perform_binning();
  warn_separator_hits(bin_separator, sep_hits, sep_example);
  return res;
}
