// [[Rcpp::depends(Rcpp)]]
#include <Rcpp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>


// Include shared headers
#include "common/bin_structures.h"
#include "common/optimal_binning_common.h"

using namespace Rcpp;
using namespace OptimalBinning;

// =============================================================================
// NUMERICAL STABILITY UTILITIES
// =============================================================================

/**
 * @brief Safe logarithm with underflow protection
 * @param x Input value
 * @param epsilon Minimum value before log (default: EPSILON, 1e-10)
 * @return log(max(x, epsilon))
 *
 * [D5] Was hardcoded to 1e-12, diverging from OptimalBinning::EPSILON
 * (1e-10, common/optimal_binning_common.h) that most other algorithms'
 * local safe_log() copies already use. Unified to EPSILON so this
 * algorithm's WoE/IV underflow floor matches the rest of the package.
 */
inline double safe_log(double x, double epsilon = EPSILON) {
  return std::log(std::max(x, epsilon));
}

/**
 * @brief Safe division with zero denominator protection
 * @param num Numerator
 * @param denom Denominator
 * @param epsilon Minimum denominator magnitude (default: 1e-12)
 * @return num/denom with protection against division by zero
 */
inline double safe_divide(double num, double denom, double epsilon = 1e-12) {
  if (std::abs(denom) < epsilon) {
    return 0.0;
  }
  return num / denom;
}

/**
 * @brief Clamp value to range [min_val, max_val]
 * @param value Value to clamp
 * @param min_val Minimum allowed value
 * @param max_val Maximum allowed value
 * @return Clamped value
 */
inline double clamp(double value, double min_val, double max_val) {
  return std::max(min_val, std::min(value, max_val));
}

// =============================================================================
// BIN STRUCTURE
// =============================================================================

/**
 * @brief Structure representing a bin for numerical variable binning
 *
 * Contains the boundaries and statistics for a bin in the Chi-Merge algorithm
 */
// Local NumericalBin definition removed

// =============================================================================
// OPTIMAL BINNING NUMERICAL CLASS (ENHANCED)
// =============================================================================

/**
 * @brief Optimal Binning for Numerical Variables using Chi-Merge
 *
 * Implementation of the Chi-Merge and Chi2 algorithms for numerical variables,
 * based on Kerber (1992) and Liu & Setiono (1995)
 *
 * Time Complexity: O(n log n + k^2m) where n = observations, k = initial bins,
 * m = iterations Space Complexity: O(k) for bins and cache
 */
class OBN_CM {
private:
  // Input parameters (const references)
  const std::vector<double> &feature;
  const std::vector<int> &target;
  int min_bins;
  int max_bins;
  const double bin_cutoff;
  int max_n_prebins;
  const double convergence_threshold;
  const int max_iterations;
  const std::string init_method;
  const double chi_merge_threshold;
  const bool use_chi2_algorithm;

  // Internal state
  std::vector<NumericalBin> bins;
  bool converged;
  int iterations_run;
  int total_pos;
  int total_neg;
  bool is_increasing;
  std::vector<std::string> warnings;

  // Chi-square statistic of every adjacent pair: chi_adj[i] belongs to
  // (bins[i], bins[i+1]). Merging bins i and i+1 only changes the two pairs
  // that touch the merged bin, so merge_bins() drops one entry and recomputes
  // at most two -- O(B) per merge instead of recomputing every pair.
  //
  // This replaces a triangular ChiSquareCache that returned STALE values:
  // resize() changed the row stride of the triangular index without moving
  // the stored entries, and invalidate_bin() wrote a negative sentinel through
  // set(), which ignores negative values, so nothing was ever invalidated.
  // After a merge, find_min_chi_square_pair() could therefore read the
  // statistic of a different (or no longer existing) pair and merge the wrong
  // bins.
  std::vector<double> chi_adj;

  /**
   * @brief Calculate chi-square statistic between two bins with continuity
   * correction
   *
   * Calculates the chi-square statistic as per Kerber (1992)
   *
   * @param bin1 First bin
   * @param bin2 Second bin
   * @return double Chi-square statistic
   */
  double calculate_chi_square(const NumericalBin &bin1,
                              const NumericalBin &bin2) const {
    const int pair_pos = bin1.count_pos + bin2.count_pos;
    const int pair_neg = bin1.count_neg + bin2.count_neg;
    const int pair_total = pair_pos + pair_neg;

    if (pair_total == 0 || pair_pos == 0 || pair_neg == 0) {
      return 0.0; // No chi-square if any category has zero count
    }

    // Expected counts are formed in double: the former int product
    // count * pair_pos overflowed once it exceeded 2^31 (e.g. two bins of
    // 50,000 observations each), which is undefined behaviour and produced
    // garbage statistics for large samples.
    const double tot = static_cast<double>(pair_total);
    const double expected_pos1 = static_cast<double>(bin1.count) * pair_pos / tot;
    const double expected_neg1 = static_cast<double>(bin1.count) * pair_neg / tot;
    const double expected_pos2 = static_cast<double>(bin2.count) * pair_pos / tot;
    const double expected_neg2 = static_cast<double>(bin2.count) * pair_neg / tot;

    // Yates continuity correction as in R's chisq.test(correct = TRUE): the
    // absolute deviation is reduced by at most 0.5 and never below zero. The
    // unclamped form (|O - E| - 0.5)^2 grew again as |O - E| fell below 0.5,
    // so two bins with IDENTICAL event rates (|O - E| = 0) scored as badly as a
    // pair one full count apart and were not merged first.
    auto cell = [](double observed, double expected) {
      if (!(expected > EPSILON)) return 0.0;
      double dev = std::fabs(observed - expected) - 0.5;
      if (dev < 0.0) dev = 0.0;
      return dev * dev / expected;
    };

    return cell(bin1.count_pos, expected_pos1) + cell(bin1.count_neg, expected_neg1) +
           cell(bin2.count_pos, expected_pos2) + cell(bin2.count_neg, expected_neg2);
  }

  /**
   * @brief Merge adjacent bins
   *
   * @param index Index of the first bin to merge
   */
  void merge_bins(size_t index) {
    if (index >= bins.size() - 1)
      return;

    NumericalBin &left = bins[index];
    const NumericalBin &right = bins[index + 1];

    left.upper_bound = right.upper_bound;
    left.count += right.count;
    left.count_pos += right.count_pos;
    left.count_neg += right.count_neg;

    bins.erase(bins.begin() + static_cast<std::ptrdiff_t>(index) + 1);

    // Keep the adjacent-pair statistics in sync (see chi_adj).
    if (chi_adj.size() == bins.size()) {
      chi_adj.erase(chi_adj.begin() + static_cast<std::ptrdiff_t>(index));
      if (index > 0)
        chi_adj[index - 1] = calculate_chi_square(bins[index - 1], bins[index]);
      if (index + 1 < bins.size())
        chi_adj[index] = calculate_chi_square(bins[index], bins[index + 1]);
    } else {
      rebuild_chi_adj();
    }
  }

  /**
   * @brief Recompute the chi-square statistic of every adjacent pair
   */
  void rebuild_chi_adj() {
    chi_adj.assign(bins.size() > 0 ? bins.size() - 1 : 0, 0.0);
    for (size_t i = 0; i + 1 < bins.size(); ++i)
      chi_adj[i] = calculate_chi_square(bins[i], bins[i + 1]);
  }

  /**
   * @brief Merge bins with zero counts in any class
   *
   * This step is important to avoid invalid WoE/IV calculations
   */
  void merge_zero_bins() {
    bool merged = true;
    int merge_attempts = 0;
    const int max_merge_attempts = static_cast<int>(bins.size());

    while (merged && bins.size() > 1 && merge_attempts < max_merge_attempts) {
      merged = false;

      for (size_t i = 0; i < bins.size(); ++i) {
        if (bins[i].count_pos == 0 || bins[i].count_neg == 0) {
          // For edge bins, merge with adjacent bin
          if (i == 0) {
            merge_bins(i);
          } else if (i == bins.size() - 1) {
            merge_bins(i - 1);
          } else {
            // Otherwise, merge with more similar bin based on chi-square
            const double chi_left = calculate_chi_square(bins[i - 1], bins[i]);
            const double chi_right = calculate_chi_square(bins[i], bins[i + 1]);
            merge_bins(chi_left < chi_right ? i - 1 : i);
          }

          merged = true;
          merge_attempts++;
          break;
        }
      }
    }
  }

  /**
   * @brief Calculate Weight of Evidence and Information Value for all bins
   */
  void calculate_woe_iv() {
    if (total_pos < 1 || total_neg < 1) {
      // Recalculate totals if needed
      total_pos = 0;
      total_neg = 0;
      for (const auto &bin : bins) {
        total_pos += bin.count_pos;
        total_neg += bin.count_neg;
      }
    }

    // Avoid division by zero
    if (total_pos < 1 || total_neg < 1) {
      for (auto &bin : bins) {
        bin.woe = 0.0;
        bin.iv = 0.0;
      }
      return;
    }

    for (auto &bin : bins) {
      // Calculate proportions with Laplace smoothing
      const double pos_rate =
          safe_divide(bin.count_pos + 0.5, total_pos + 1.0, EPSILON);
      const double neg_rate =
          safe_divide(bin.count_neg + 0.5, total_neg + 1.0, EPSILON);

      // Calculate WoE with safeguards
      bin.woe = safe_log(pos_rate / neg_rate);
      bin.woe = clamp(bin.woe, MIN_WOE, MAX_WOE);

      bin.iv = (pos_rate - neg_rate) * bin.woe;
    }
  }

  /**
   * @brief Record a low/high total-IV note for the FINAL binning
   *
   * This used to run inside calculate_woe_iv(), which is called after every
   * merge, so the returned `warnings` repeated the note once per iteration and
   * quoted the IV of intermediate binnings rather than of the result.
   */
  void add_iv_warning() {
    double total_iv = 0.0;
    for (const auto &bin : bins) total_iv += bin.iv;
    if (total_iv < 0.02) {
      warnings.push_back("Low total IV (" + std::to_string(total_iv) +
                         ") - feature may have low predictive power");
    } else if (total_iv > 0.5) {
      warnings.push_back("High total IV (" + std::to_string(total_iv) +
                         ") - check for overfitting");
    }
  }

  /**
   * @brief Determine if WoE is monotonically increasing or decreasing
   *
   * @return true if monotonic
   */
  bool is_monotonic() const {
    if (bins.size() <= 2)
      return true;

    // Determine direction (increasing or decreasing) from first two bins
    const bool increasing = (bins[1].woe >= bins[0].woe - EPSILON);

    // Check the entire sequence for monotonicity
    for (size_t i = 2; i < bins.size(); ++i) {
      if ((increasing && bins[i].woe < bins[i - 1].woe - EPSILON) ||
          (!increasing && bins[i].woe > bins[i - 1].woe + EPSILON)) {
        return false;
      }
    }

    return true;
  }

  /**
   * @brief Determine monotonicity direction using linear regression
   */
  void determine_monotonicity_direction() {
    if (bins.size() < 3) {
      is_increasing = true;
      return;
    }

    // Calculate trend using linear regression
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
    int n = static_cast<int>(bins.size());

    for (int i = 0; i < n; ++i) {
      double x = static_cast<double>(i);
      double y = bins[i].woe;
      sum_x += x;
      sum_y += y;
      sum_xy += x * y;
      sum_x2 += x * x;
    }

    double slope_numerator = n * sum_xy - sum_x * sum_y;
    double slope_denominator = n * sum_x2 - sum_x * sum_x;

    if (std::abs(slope_denominator) < EPSILON) {
      is_increasing = true; // Default to increasing if trend is flat
    } else {
      double slope = slope_numerator / slope_denominator;
      is_increasing = (slope >= 0);
    }
  }

  /**
   * @brief Initial binning via equal frequency (quantiles)
   *
   * Chunks of records_per_bin sorted observations; a chunk that ends inside a
   * run of tied values is extended to the end of the run, so no value is ever
   * split across two bins. Each bin is labelled (previous upper; own upper].
   *
   * Two defects of the former loop are fixed here:
   *  - ties were absorbed into the previous bin only when the next chunk was
   *    not the last one, so a run of ties reaching into the final chunk was
   *    split: part of it was counted in a new last bin although the cutpoint
   *    (equal to the tied value) assigns all of it to the bin below;
   *  - the lower label bound was the smallest value inside the bin instead of
   *    the previous cutpoint, leaving gaps between consecutive labels
   *    ("(-Inf;-0.61]", "(-0.607;-0.33]", ...).
   */
  void initial_binning_equal_frequency() {
    // Sort data points
    std::vector<std::pair<double, int>> sorted_data(feature.size());
    for (size_t i = 0; i < feature.size(); ++i) {
      sorted_data[i] = {feature[i], target[i]};
    }

    std::sort(
        sorted_data.begin(), sorted_data.end(),
        [](const std::pair<double, int> &a, const std::pair<double, int> &b) {
          return a.first < b.first;
        });

    bins.clear();
    const size_t total_records = sorted_data.size();

    // Calculate number of records per bin - adjusted to respect max_bins
    int initial_bins = std::min(
        max_n_prebins,
        std::max(max_bins, static_cast<int>(std::sqrt(static_cast<double>(total_records)))));
    initial_bins = std::max(2, initial_bins); // Ensure at least 2 bins

    const size_t records_per_bin =
        std::max<size_t>(1, total_records / static_cast<size_t>(initial_bins));

    size_t start = 0;
    while (start < total_records) {
      size_t end = std::min(start + records_per_bin, total_records);
      // Never split a run of tied values across two bins, and never close a
      // bin on -Inf: its upper bound becomes a cut point, and cut points must
      // be finite (-Inf observations simply belong to the first bin).
      while (end < total_records &&
             (sorted_data[end].first == sorted_data[end - 1].first ||
              !std::isfinite(sorted_data[end - 1].first))) {
        ++end;
      }

      NumericalBin bin;
      bin.lower_bound = bins.empty() ? -std::numeric_limits<double>::infinity()
                                     : bins.back().upper_bound;
      bin.upper_bound = (end == total_records)
                            ? std::numeric_limits<double>::infinity()
                            : sorted_data[end - 1].first;
      bin.count = static_cast<int>(end - start);
      for (size_t i = start; i < end; ++i) {
        if (sorted_data[i].second == 1) {
          bin.count_pos++;
        } else {
          bin.count_neg++;
        }
      }

      bins.push_back(std::move(bin));
      start = end;
    }
  }

  /**
   * @brief Initial binning via equal width
   *
   * Observations are assigned with the same right-closed rule that the emitted
   * cutpoints imply: a value lies in the first bin whose upper bound is >= it.
   * The former floor((x - min) / width) placed a value sitting exactly on a
   * boundary in the bin ABOVE it ([a, b) semantics, contradicting the "(a;b]"
   * labels and the cutpoints -- common with integer-valued features), and when
   * max - min overflowed to Inf (e.g. -1e308 and 1e308) it evaluated
   * Inf / Inf = NaN, cast that to int and indexed the bin vector with it,
   * crashing R.
   */
  void initial_binning_equal_width() {
    // Range of the FINITE values: -Inf/+Inf observations fall in the
    // first/last bin and must not make the width infinite.
    double min_val = std::numeric_limits<double>::infinity();
    double max_val = -std::numeric_limits<double>::infinity();
    bool any_nonfinite = false;
    for (double v : feature) {
      if (!std::isfinite(v)) { any_nonfinite = true; continue; }
      if (v < min_val) min_val = v;
      if (v > max_val) max_val = v;
    }
    if (any_nonfinite && !(max_val > min_val)) {
      // fewer than two distinct finite values: no width to divide
      initial_binning_unique_values();
      return;
    }

    // Handle case where all values are the same
    if (!(max_val > min_val)) {
      // Create a single bin
      NumericalBin bin;
      bin.lower_bound = -std::numeric_limits<double>::infinity();
      bin.upper_bound = std::numeric_limits<double>::infinity();
      bin.count = static_cast<int>(feature.size());

      for (size_t i = 0; i < feature.size(); ++i) {
        if (target[i] == 1) {
          bin.count_pos++;
        } else {
          bin.count_neg++;
        }
      }

      bins.clear();
      bins.push_back(std::move(bin));
      return;
    }

    // Calculate bin width - adjust to respect max_bins
    int n_prebins = std::min(
        max_n_prebins,
        std::max(max_bins, static_cast<int>(std::sqrt(static_cast<double>(feature.size())))));
    n_prebins = std::max(2, n_prebins); // Ensure at least 2 bins

    // Interior boundaries min + k * width, k = 1 .. n_prebins - 1. When the
    // range itself overflows, the same boundaries are formed from half-values.
    std::vector<double> uppers(static_cast<size_t>(n_prebins));
    const double range = max_val - min_val;
    if (std::isfinite(range)) {
      const double bin_width = range / n_prebins;
      for (int i = 0; i < n_prebins - 1; ++i)
        uppers[static_cast<size_t>(i)] = min_val + (i + 1) * bin_width;
    } else {
      const double half_width = (max_val * 0.5 - min_val * 0.5) / n_prebins;
      for (int i = 0; i < n_prebins - 1; ++i)
        uppers[static_cast<size_t>(i)] = 2.0 * (min_val * 0.5 + (i + 1) * half_width);
    }
    uppers[static_cast<size_t>(n_prebins - 1)] = std::numeric_limits<double>::infinity();

    // Create empty bins
    bins.clear();
    bins.resize(static_cast<size_t>(n_prebins));
    for (size_t i = 0; i < bins.size(); ++i) {
      bins[i].lower_bound = (i == 0) ? -std::numeric_limits<double>::infinity()
                                     : uppers[i - 1];
      bins[i].upper_bound = uppers[i];
    }

    // Fill bins with data: first bin whose upper bound is >= the value
    for (size_t i = 0; i < feature.size(); ++i) {
      const size_t bin_idx = static_cast<size_t>(
          std::lower_bound(uppers.begin(), uppers.end(), feature[i]) - uppers.begin());
      NumericalBin &b = bins[std::min(bin_idx, bins.size() - 1)];
      b.count++;
      if (target[i] == 1) {
        b.count_pos++;
      } else {
        b.count_neg++;
      }
    }

    // Remove empty bins; the range of a removed bin joins the next bin, which
    // is what the remaining cutpoints imply, so relabel lower bounds to keep
    // the labels contiguous.
    bins.erase(
        std::remove_if(bins.begin(), bins.end(),
                       [](const NumericalBin &b) { return b.count == 0; }),
        bins.end());
    for (size_t i = 0; i < bins.size(); ++i) {
      bins[i].lower_bound = (i == 0) ? -std::numeric_limits<double>::infinity()
                                     : bins[i - 1].upper_bound;
    }
    if (!bins.empty())
      bins.back().upper_bound = std::numeric_limits<double>::infinity();
  }

  /**
   * @brief Initial binning based on unique values
   *
   * Used when the number of unique values is small
   */
  void initial_binning_unique_values() {
    // Get unique sorted values (exact equality: an absolute tolerance merged
    // every distinct value of a feature measured on a scale below 1e-10)
    std::vector<double> unique_values(feature);
    std::sort(unique_values.begin(), unique_values.end());
    unique_values.erase(std::unique(unique_values.begin(), unique_values.end()),
                        unique_values.end());

    // Cut points: every distinct value except the largest, finite only (a
    // -Inf cut would create the bin "(-Inf;-Inf]").
    std::vector<double> cuts;
    for (size_t i = 0; i + 1 < unique_values.size(); ++i)
      if (std::isfinite(unique_values[i])) cuts.push_back(unique_values[i]);

    bins.clear();
    bins.reserve(cuts.size() + 1);
    double lower = -std::numeric_limits<double>::infinity();
    for (double c : cuts) {
      NumericalBin bin;
      bin.lower_bound = lower;
      bin.upper_bound = c;
      bins.push_back(std::move(bin));
      lower = c;
    }
    NumericalBin last;
    last.lower_bound = lower;
    last.upper_bound = std::numeric_limits<double>::infinity();
    bins.push_back(std::move(last));

    // Fill bins with data: first bin whose upper bound (cut) is >= the value
    for (size_t i = 0; i < feature.size(); ++i) {
      size_t bin_idx = static_cast<size_t>(
          std::lower_bound(cuts.begin(), cuts.end(), feature[i]) - cuts.begin());
      if (bin_idx >= bins.size()) bin_idx = bins.size() - 1;

      bins[bin_idx].count++;
      if (target[i] == 1) {
        bins[bin_idx].count_pos++;
      } else {
        bins[bin_idx].count_neg++;
      }
    }
  }

  /**
   * @brief Enforce bin limit constraints
   *
   * Make sure we respect max_bins by merging bins with lowest chi-square
   */
  void enforce_bin_limits() {
    // If we have more than max_bins, continue merging until we reach max_bins
    while (bins.size() > static_cast<size_t>(max_bins)) {
      // Find pair with minimum chi-square
      std::pair<double, size_t> min_chi_pair = find_min_chi_square_pair();
      merge_bins(min_chi_pair.second);

      // Handle bins with zero counts
      merge_zero_bins();
    }

    // If we somehow end up with fewer than min_bins, reset min_bins
    if (bins.size() < static_cast<size_t>(min_bins)) {
      min_bins = static_cast<int>(bins.size());
    }

    // Update WoE/IV after bin limit enforcement
    calculate_woe_iv();
  }

  /**
   * @brief Main Chi-Merge algorithm implementation
   *
   * Based on Kerber (1992)
   */
  void chi_merge() {
    // Get chi-square critical value based on threshold
    double critical_value = chi_square_critical_value(chi_merge_threshold);

    double prev_total_iv = 0.0;
    converged = false;

    for (iterations_run = 0; iterations_run < max_iterations;
         ++iterations_run) {
      // Stop if we've reached minimum bins
      if (bins.size() <= static_cast<size_t>(min_bins)) {
        converged = true;
        break;
      }

      // Find pair with minimum chi-square
      std::pair<double, size_t> min_chi_pair = find_min_chi_square_pair();
      double min_chi_square = min_chi_pair.first;
      size_t merge_index = min_chi_pair.second;

      // Check if we should stop based on statistical significance OR bin count
      // This prioritizes bin count over significance
      if (bins.size() <= static_cast<size_t>(max_bins)) {
        converged = true;
        break;
      }

      // If chi-square exceeds critical value and we're within bin count limits,
      // we can stop
      if (min_chi_square > critical_value &&
          bins.size() <= static_cast<size_t>(max_bins + 3)) {
        converged = true;
        break;
      }

      // Merge bins with lowest chi-square
      merge_bins(merge_index);

      // Handle bins with zero counts
      merge_zero_bins();

      // Calculate WoE and IV
      calculate_woe_iv();

      // Calculate total IV for convergence check
      double total_iv = std::accumulate(
          bins.begin(), bins.end(), 0.0,
          [](double sum, const NumericalBin &b) { return sum + b.iv; });

      // Check for convergence based on IV change
      if (std::abs(total_iv - prev_total_iv) < convergence_threshold) {
        converged = true;
        break;
      }

      prev_total_iv = total_iv;
    }

    // Mark as converged if we've reached max iterations
    if (iterations_run >= max_iterations) {
      converged = true;
    }

    // Enforce max_bins constraint regardless of other criteria
    enforce_bin_limits();
  }

  /**
   * @brief Implementation of Chi2 algorithm
   *
   * Based on Liu & Setiono (1995)
   */
  void chi2_algorithm() {
    // Chi2 uses multiple phases with decreasing significance levels
    const std::vector<double> significance_levels = {0.5,  0.1,   0.05,
                                                     0.01, 0.005, 0.001};

    converged = false;
    iterations_run = 0;

    // Multiple phases with different significance levels
    for (double significance : significance_levels) {
      // Set current significance level
      double current_critical_value =
          chi_square_critical_value(significance);

      // Continue merging until no more bins can be merged
      bool continue_merging = true;
      int phase_iterations = 0;

      while (continue_merging && phase_iterations < max_iterations) {
        // Stop if we've reached minimum bins
        if (bins.size() <= static_cast<size_t>(min_bins)) {
          converged = true;
          break;
        }

        // Check if we're already at max_bins
        if (bins.size() <= static_cast<size_t>(max_bins)) {
          converged = true;
          break;
        }

        // Find pair with minimum chi-square
        std::pair<double, size_t> min_chi_pair = find_min_chi_square_pair();
        double min_chi_square = min_chi_pair.first;
        size_t merge_index = min_chi_pair.second;

        // Stop if chi-square exceeds threshold for this phase
        if (min_chi_square > current_critical_value) {
          continue_merging = false;
          break;
        }

        // Merge bins with lowest chi-square
        merge_bins(merge_index);

        // Handle bins with zero counts
        merge_zero_bins();

        // Calculate WoE and IV
        calculate_woe_iv();

        // Track iterations
        phase_iterations++;
        iterations_run++;
      }

      // Check inconsistency rate (Chi2 feature selection)
      if (calculate_inconsistency_rate() < 0.05) {
        break; // Feature is discriminative enough
      }

      // Check if we've reached target bins
      if (bins.size() <= static_cast<size_t>(max_bins)) {
        converged = true;
        break;
      }
    }

    // Enforce bin limits regardless of previous steps
    enforce_bin_limits();

    // Final adjustments to enforce monotonicity
    enforce_monotonicity();
  }

  /**
   * @brief Calculate inconsistency rate for Chi2 algorithm
   *
   * @return double Inconsistency rate (0-1)
   */
  double calculate_inconsistency_rate() const {
    // Count inconsistent instances
    int inconsistent_count = 0;

    // Using majority class per bin as a reference
    std::vector<bool> bin_majority_positive(bins.size());
    for (size_t i = 0; i < bins.size(); ++i) {
      bin_majority_positive[i] = bins[i].count_pos > bins[i].count_neg;
    }

    // Count inconsistencies over the dataset. Bin j holds the values in
    // (upper[j-1], upper[j]] and the last bin everything above, so the bin of
    // a value is the first j < B-1 with val <= upper[j] (binary search over
    // the ascending upper bounds), else B-1.
    std::vector<double> uppers;
    uppers.reserve(bins.size());
    for (size_t j = 0; j + 1 < bins.size(); ++j) uppers.push_back(bins[j].upper_bound);

    for (size_t i = 0; i < feature.size(); ++i) {
      const size_t bin_idx = static_cast<size_t>(
          std::lower_bound(uppers.begin(), uppers.end(), feature[i]) - uppers.begin());

      // Check if instance matches majority class
      bool is_positive = target[i] == 1;
      if (is_positive != bin_majority_positive[bin_idx]) {
        inconsistent_count++;
      }
    }

    return safe_divide(inconsistent_count, static_cast<int>(feature.size()));
  }

  /**
   * @brief Merge rare bins
   *
   * Merges bins with frequency less than bin_cutoff
   */
  void merge_rare_bins() {
    const double total_count = static_cast<double>(feature.size());
    bool merged_bins = true;

    while (merged_bins && bins.size() > static_cast<size_t>(min_bins)) {
      merged_bins = false;

      for (size_t i = 0; i < bins.size();) {
        const double freq = safe_divide(bins[i].count, total_count);

        if (freq < bin_cutoff) {
          // At the start, merge with the next bin
          if (i == 0) {
            if (bins.size() > 1) {
              merge_bins(0);
              merged_bins = true;

              // Reset and check from beginning
              i = 0;
              continue;
            }
          }
          // At the end, merge with the previous bin
          else if (i == bins.size() - 1) {
            merge_bins(i - 1);
            merged_bins = true;

            // Reset and check from beginning
            i = 0;
            continue;
          }
          // In the middle, merge with the more similar bin
          else {
            double chi_left = calculate_chi_square(bins[i - 1], bins[i]);
            double chi_right = calculate_chi_square(bins[i], bins[i + 1]);

            merge_bins(chi_left < chi_right ? i - 1 : i);
            merged_bins = true;

            // Reset and check from beginning
            i = 0;
            continue;
          }
        }

        // Move to next bin if no merge happened
        ++i;
      }

      if (merged_bins) {
        merge_zero_bins();
        calculate_woe_iv();
      }
    }

    // Enforce bin limits after merging rare bins
    enforce_bin_limits();
  }

  /**
   * @brief Enforce monotonicity of Weight of Evidence
   *
   * Merges bins to create monotonic WoE pattern
   */
  void enforce_monotonicity() {
    if (bins.size() <= 2)
      return;

    // Determine monotonicity direction
    determine_monotonicity_direction();

    int monotonicity_iterations = 0;
    const int max_monotonicity_iterations = 100; // Avoid infinite loops

    while (!is_monotonic() && bins.size() > static_cast<size_t>(min_bins) &&
           monotonicity_iterations < max_monotonicity_iterations) {

      // Find first violation of monotonicity
      size_t violation_index = 0;

      for (size_t i = 1; i < bins.size(); ++i) {
        bool is_violation = is_increasing
                                ? (bins[i].woe < bins[i - 1].woe - EPSILON)
                                : (bins[i].woe > bins[i - 1].woe + EPSILON);

        if (is_violation) {
          violation_index = i - 1;
          break;
        }
      }

      // Merge bins at violation point
      merge_bins(violation_index);
      merge_zero_bins();
      calculate_woe_iv();

      monotonicity_iterations++;
    }

    // If we can't achieve monotonicity through merging, enforce it by sorting
    if (!is_monotonic() &&
        monotonicity_iterations >= max_monotonicity_iterations) {
      // Sort bins by their boundaries (they should already be in this order)
      std::sort(bins.begin(), bins.end(),
                [](const NumericalBin &a, const NumericalBin &b) {
                  return a.lower_bound < b.lower_bound;
                });

      // Recalculate WoE and IV
      calculate_woe_iv();
    }

    // Make sure we still respect bin limits
    enforce_bin_limits();
  }

  /**
   * @brief Find the pair of adjacent bins with lowest chi-square
   *
   * @return std::pair<double, size_t> Minimum chi-square value and index
   */
  std::pair<double, size_t> find_min_chi_square_pair() {
    double min_chi_square = std::numeric_limits<double>::max();
    size_t min_index = 0;

    if (chi_adj.size() + 1 != bins.size())
      rebuild_chi_adj();

    for (size_t i = 0; i < chi_adj.size(); ++i) {
      const double chi_square = chi_adj[i];

      if (chi_square < min_chi_square) {
        min_chi_square = chi_square;
        min_index = i;
      }
    }

    return {min_chi_square, min_index};
  }

  /**
   * @brief Chi-square critical value (df = 1) for a significance level
   *
   * The documented meaning of chi_merge_threshold is a significance level
   * alpha: a pair is significantly different when its p-value is below alpha,
   * i.e. when chi2 > qchisq(1 - alpha, 1) (3.841 for alpha = 0.05). The former
   * lookup table was keyed the other way round and returned qchisq(alpha, 1)
   * (0.004 for alpha = 0.05; 0.455 -> 0.00001 over the Chi2 schedule, which
   * therefore relaxed instead of tightening), and silently snapped any other
   * alpha to the nearest of 14 hard-coded levels.
   *
   * @param significance Significance level in (0, 1)
   * @return double Critical chi-square value
   */
  static double chi_square_critical_value(double significance) {
    return R::qchisq(1.0 - significance, 1.0, 1, 0);
  }

public:
  /**
   * @brief Construct a new Optimal Binning Numerical CM object
   *
   * @param feature_ Feature vector
   * @param target_ Binary target vector (0/1)
   * @param min_bins_ Minimum number of bins
   * @param max_bins_ Maximum number of bins
   * @param bin_cutoff_ Minimum frequency for a bin
   * @param max_n_prebins_ Maximum number of initial bins
   * @param convergence_threshold_ Convergence threshold for IV
   * @param max_iterations_ Maximum number of iterations
   * @param init_method_ Initialization method ("equal_width",
   * "equal_frequency")
   * @param chi_merge_threshold_ Significance level for chi-square test
   * @param use_chi2_algorithm_ Whether to use Chi2 algorithm
   */
  OBN_CM(const std::vector<double> &feature_, const std::vector<int> &target_,
         int min_bins_, int max_bins_, double bin_cutoff_, int max_n_prebins_,
         double convergence_threshold_, int max_iterations_,
         std::string init_method_ = "equal_frequency",
         double chi_merge_threshold_ = 0.05, bool use_chi2_algorithm_ = false)
      : feature(feature_), target(target_), min_bins(min_bins_),
        max_bins(max_bins_), bin_cutoff(bin_cutoff_),
        max_n_prebins(max_n_prebins_),
        convergence_threshold(convergence_threshold_),
        max_iterations(max_iterations_), init_method(init_method_),
        chi_merge_threshold(chi_merge_threshold_),
        use_chi2_algorithm(use_chi2_algorithm_), converged(false),
        iterations_run(0), total_pos(0), total_neg(0), is_increasing(true),
        warnings() {

    // Validate inputs (basic validation, additional validation in fit())
    if (feature.size() != target.size()) {
      throw std::invalid_argument(
          "Feature and target must have the same size.");
    }
    if (feature.empty() || target.empty()) {
      throw std::invalid_argument("Input vectors cannot be empty.");
    }
    if (min_bins < 2) {
      throw std::invalid_argument("min_bins must be >= 2.");
    }
    if (max_bins < min_bins) {
      Rcpp::warning("max_bins (%d) is less than min_bins (%d), setting "
                    "max_bins = min_bins",
                    max_bins, min_bins);
      max_bins = min_bins;
    }
    if (bin_cutoff <= 0 || bin_cutoff >= 1) {
      throw std::invalid_argument("bin_cutoff must be between 0 and 1.");
    }
    if (max_n_prebins < max_bins) {
      Rcpp::warning("max_n_prebins (%d) is less than max_bins (%d), setting "
                    "max_n_prebins = max_bins",
                    max_n_prebins, max_bins);
      max_n_prebins = max_bins;
    }
    if (convergence_threshold <= 0) {
      throw std::invalid_argument("convergence_threshold must be positive.");
    }
    if (max_iterations <= 0) {
      throw std::invalid_argument("max_iterations must be positive.");
    }
    if (chi_merge_threshold <= 0 || chi_merge_threshold >= 1) {
      throw std::invalid_argument(
          "chi_merge_threshold must be between 0 and 1.");
    }
  }

  /**
   * @brief Perform binning optimization
   */
  void fit() {
    auto start_time = std::chrono::high_resolution_clock::now();

    // Count positive and negative cases
    total_pos = std::accumulate(target.begin(), target.end(), 0);
    total_neg = static_cast<int>(target.size()) - total_pos;

    // Validate target composition
    if (total_pos == 0 || total_neg == 0) {
      throw std::runtime_error(
          "Target is constant (all 0s or all 1s), making binning impossible.");
    }

    // Validate feature values (missing values are removed by the caller)
    for (double val : feature) {
      if (std::isnan(val)) {
        throw std::invalid_argument("Feature contains NaN values.");
      }
    }

    // Count unique values
    std::vector<double> unique_values(feature);
    std::sort(unique_values.begin(), unique_values.end());
    unique_values.erase(std::unique(unique_values.begin(), unique_values.end()),
                        unique_values.end());

    // If we have very few unique values, adjust bin limits
    if (unique_values.size() < static_cast<size_t>(min_bins)) {
      warnings.push_back("Feature has only " +
                         std::to_string(unique_values.size()) +
                         " unique values, setting min_bins = " +
                         std::to_string(unique_values.size()));
      min_bins = static_cast<int>(unique_values.size());
      if (max_bins < min_bins) {
        max_bins = min_bins;
      }
    }

    // Special case for very few unique values
    if (unique_values.size() <= 2) {
      initial_binning_unique_values();
      merge_zero_bins();
      calculate_woe_iv();
      add_iv_warning();
      converged = true;
      iterations_run = 0;
      return;
    }

    // Initial binning based on method
    if (init_method == "equal_width") {
      initial_binning_equal_width();
    } else {
      // Default to equal frequency
      initial_binning_equal_frequency();
    }

    // Immediately enforce max_bins constraint if we have too many initial bins
    if (bins.size() > static_cast<size_t>(max_bins)) {
      enforce_bin_limits();
    }

    // Merge bins with zero counts and calculate initial WoE/IV
    merge_zero_bins();
    calculate_woe_iv();

    // Apply Chi-Merge or Chi2 algorithm
    if (use_chi2_algorithm) {
      chi2_algorithm();
    } else {
      chi_merge();
      merge_rare_bins();
      enforce_monotonicity();

      // Final enforcement of bin limits
      enforce_bin_limits();
    }

    // Final WoE/IV calculation
    calculate_woe_iv();
    add_iv_warning();

    // Verify bin count constraints
    if (bins.size() > static_cast<size_t>(max_bins)) {
      warnings.push_back("Failed to respect max_bins constraint. Requested " +
                         std::to_string(max_bins) + " bins, got " +
                         std::to_string(bins.size()) + " bins.");
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time - start_time)
                        .count();
    warnings.push_back("Execution time: " + std::to_string(duration) + " ms");
  }

  /**
   * @brief Get binning results as R list
   *
   * @return Rcpp::List Results of binning
   */
  Rcpp::List get_results() const {
    // Initialize vectors for results
    Rcpp::StringVector bin_names(bins.size());
    Rcpp::NumericVector bin_woe(bins.size());
    Rcpp::NumericVector bin_iv(bins.size());
    Rcpp::IntegerVector bin_count(bins.size());
    Rcpp::IntegerVector bin_count_pos(bins.size());
    Rcpp::IntegerVector bin_count_neg(bins.size());
    Rcpp::NumericVector bin_cutpoints;

    // Format output with appropriate precision
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);

    for (size_t i = 0; i < bins.size(); ++i) {
      const auto &bin = bins[i];
      oss.str("");

      // Format bin name with infinity handling
      if (bin.lower_bound == -std::numeric_limits<double>::infinity()) {
        oss << "(-Inf";
      } else {
        oss << "(" << bin.lower_bound;
      }

      oss << ";";

      if (bin.upper_bound == std::numeric_limits<double>::infinity()) {
        oss << "+Inf]";
      } else {
        oss << bin.upper_bound << "]";
      }

      bin_names[i] = oss.str();
      bin_woe[i] = bin.woe;
      bin_iv[i] = bin.iv;
      bin_count[i] = bin.count;
      bin_count_pos[i] = bin.count_pos;
      bin_count_neg[i] = bin.count_neg;

      // Save cutpoints (for use in prediction)
      if (bin.upper_bound != std::numeric_limits<double>::infinity() &&
          i != bins.size() - 1) {
        bin_cutpoints.push_back(bin.upper_bound);
      }
    }

    // Create 1-based indices for R
    Rcpp::NumericVector ids(bin_names.size());
    for (int i = 0; i < bin_names.size(); i++) {
      ids[i] = i + 1;
    }

    // Calculate total IV
    double total_iv = std::accumulate(
        bins.begin(), bins.end(), 0.0,
        [](double sum, const NumericalBin &b) { return sum + b.iv; });

    // Build result list
    return Rcpp::List::create(
        Rcpp::Named("id") = ids, Rcpp::Named("bin") = bin_names,
        Rcpp::Named("woe") = bin_woe, Rcpp::Named("iv") = bin_iv,
        Rcpp::Named("count") = bin_count,
        Rcpp::Named("count_pos") = bin_count_pos,
        Rcpp::Named("count_neg") = bin_count_neg,
        Rcpp::Named("cutpoints") = bin_cutpoints,
        Rcpp::Named("converged") = converged,
        Rcpp::Named("iterations") = iterations_run,
        Rcpp::Named("total_iv") = total_iv,
        Rcpp::Named("monotonic") = is_monotonic(),
        Rcpp::Named("algorithm") = use_chi2_algorithm ? "Chi2" : "ChiMerge",
        Rcpp::Named("requested_min_bins") = min_bins,
        Rcpp::Named("requested_max_bins") = max_bins,
        Rcpp::Named("warnings") = warnings);
  }
};

// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_cm(
    Rcpp::IntegerVector target, Rcpp::NumericVector feature, int min_bins = 3,
    int max_bins = 5, double bin_cutoff = 0.05, int max_n_prebins = 20,
    double convergence_threshold = 1e-6, int max_iterations = 1000,
    std::string init_method = "equal_frequency",
    double chi_merge_threshold = 0.05, bool use_chi2_algorithm = false) {
  // Validate inputs with clear error messages
  if (target.size() != feature.size()) {
    Rcpp::stop(
        "Target and feature must have the same size (target: %d, feature: %d)",
        target.size(), feature.size());
  }
  if (target.size() == 0) {
    Rcpp::stop("Input vectors cannot be empty.");
  }
  if (min_bins < 2) {
    Rcpp::stop("min_bins must be >= 2 (received: %d)", min_bins);
  }
  if (max_bins < min_bins) {
    Rcpp::warning(
        "max_bins (%d) is less than min_bins (%d), setting max_bins = min_bins",
        max_bins, min_bins);
    max_bins = min_bins;
  }
  if (bin_cutoff <= 0 || bin_cutoff >= 1) {
    Rcpp::stop("bin_cutoff must be between 0 and 1 (received: %f)", bin_cutoff);
  }
  if (max_n_prebins < max_bins) {
    Rcpp::warning("max_n_prebins (%d) is less than max_bins (%d), setting "
                  "max_n_prebins = max_bins",
                  max_n_prebins, max_bins);
    max_n_prebins = max_bins;
  }
  if (convergence_threshold <= 0) {
    Rcpp::stop("convergence_threshold must be positive (received: %f)",
               convergence_threshold);
  }
  if (max_iterations <= 0) {
    Rcpp::stop("max_iterations must be positive (received: %d)",
               max_iterations);
  }
  if (chi_merge_threshold <= 0 || chi_merge_threshold >= 1) {
    Rcpp::stop("chi_merge_threshold must be between 0 and 1 (received: %f)",
               chi_merge_threshold);
  }
  if (init_method != "equal_width" && init_method != "equal_frequency") {
    Rcpp::warning("Unknown init_method '%s', defaulting to 'equal_frequency'",
                  init_method.c_str());
    init_method = "equal_frequency";
  }

  // Validate target values
  for (int i = 0; i < target.size(); ++i) {
    if (IntegerVector::is_na(target[i])) {
      Rcpp::stop("Target contains NA values.");
    }
    if (target[i] != 0 && target[i] != 1) {
      Rcpp::stop("Target must contain only 0 or 1 values.");
    }
  }

  // Rows with a missing feature (NA/NaN) are excluded from the fit, as in
  // every other numerical engine; -Inf/+Inf are kept as extreme values and
  // fall in the first/last bin. Both used to stop with an error.
  std::vector<double> feature_vec;
  std::vector<int> target_vec;
  feature_vec.reserve(static_cast<size_t>(feature.size()));
  target_vec.reserve(static_cast<size_t>(feature.size()));
  for (R_xlen_t i = 0; i < feature.size(); ++i) {
    if (std::isnan(feature[i])) continue;
    feature_vec.push_back(feature[i]);
    target_vec.push_back(target[i]);
  }
  if (feature_vec.empty()) {
    Rcpp::stop("Feature has no non-missing values.");
  }

  try {
    // Create binner object
    OBN_CM binner(feature_vec, target_vec, min_bins, max_bins, bin_cutoff,
                  max_n_prebins, convergence_threshold, max_iterations,
                  init_method, chi_merge_threshold, use_chi2_algorithm);

    // Perform binning
    binner.fit();

    // Return results
    return binner.get_results();
  } catch (const std::exception &e) {
    Rcpp::stop("Error in optimal binning: %s", e.what());
  }
}
