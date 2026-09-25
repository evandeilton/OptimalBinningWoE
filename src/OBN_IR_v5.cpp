// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <numeric>
#include <unordered_set>

/**
 * @file OBN_IR.cpp
 * @brief Optimal Binning for Numerical Variables using Isotonic Regression
 * 
 * This implementation provides supervised discretization of numerical variables
 * using isotonic regression to ensure monotonicity in event rates across bins.
 * The algorithm is particularly useful for risk modeling and credit scoring
 * applications where monotonicity is a desirable property.
 */


// Include shared headers
#include "common/optimal_binning_common.h"
#include "common/bin_structures.h"

using namespace Rcpp;
using namespace OptimalBinning;


// Class for Optimal Binning using Isotonic Regression
class OBN_IR {
private:
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  double convergence_threshold;
  int max_iterations;
  bool auto_monotonicity;  // Automatically determine monotonicity direction

  const std::vector<double>& feature;
  const std::vector<int>& target;

  std::vector<double> bin_edges;
  double total_iv;
  bool converged;
  int iterations_run;
  bool monotone_increasing;  // Direction of monotonicity

  // Structure to represent a bin and its statistics
  // Local NumericalBin definition removed


  std::vector<NumericalBin> bin_info;
  bool is_simple;           // Flag for simple binning (few unique values)

  // Finite feature values of each class, ascending. Every bin count is a
  // binary search in these arrays: #positives in (lo, hi] is
  // upper_bound(pos_sorted, hi) - upper_bound(pos_sorted, lo).
  std::vector<double> pos_sorted;
  std::vector<double> neg_sorted;

  // Small constant for numerical stability
  // Constant removed (uses shared definition)
  // Laplace smoothing factor
  static constexpr double ALPHA = 0.5;

public:
  /**
   * Constructor for the OBN_IR class
   * 
   * @param min_bins_ Minimum number of bins
   * @param max_bins_ Maximum number of bins
   * @param bin_cutoff_ Minimum proportion of observations per bin
   * @param max_n_prebins_ Maximum number of pre-bins before optimization
   * @param convergence_threshold_ Convergence threshold for algorithm
   * @param max_iterations_ Maximum number of iterations
   * @param feature_ Feature vector to be binned
   * @param target_ Binary target vector (0/1)
   * @param auto_monotonicity_ Whether to automatically determine monotonicity direction
   */
  OBN_IR(int min_bins_, int max_bins_,
                            double bin_cutoff_, int max_n_prebins_,
                            double convergence_threshold_, int max_iterations_,
                            const std::vector<double>& feature_,
                            const std::vector<int>& target_,
                            bool auto_monotonicity_ = true)
    : min_bins(min_bins_), max_bins(max_bins_),
      bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_),
      convergence_threshold(convergence_threshold_), max_iterations(max_iterations_),
      auto_monotonicity(auto_monotonicity_),
      feature(feature_), target(target_), total_iv(0.0),
      converged(false), iterations_run(0), monotone_increasing(true), is_simple(false) {
    validateInputs();
  }

  /**
   * Execute the binning algorithm
   */
  void fit() {
    // Step 1: Create initial bins
    createInitialBins();

    if (!is_simple) {
      // Step 2: Merge low frequency bins
      mergeLowFrequencyBins();

      // Step 3: Ensure min and max bin constraints
      ensureMinMaxBins();

      // Step 4: Determine monotonicity direction if auto_monotonicity is enabled
      if (auto_monotonicity) {
        determineMonotonicityDirection();
      }

      // Step 5: Apply isotonic regression to enforce monotonicity
      applyIsotonicRegression();
    }

    // Step 6: Calculate final WOE and IV
    calculateWOEandIV();
  }

  /**
   * Get the results of the binning process
   * 
   * @return A list containing bin information and metrics
   */
  Rcpp::List getResults() const {
    return createWOEBinList();
  }

private:
  /**
   * Validate input parameters and data
   * Throws exception if invalid
   */
  void validateInputs() const {
    if (feature.size() != target.size()) {
      throw std::invalid_argument("Feature and target must have the same length.");
    }

    if (target.empty()) {
      throw std::invalid_argument("Feature and target must not be empty.");
    }

    if (min_bins < 2) {
      throw std::invalid_argument("min_bins must be at least 2.");
    }

    if (max_bins < min_bins) {
      throw std::invalid_argument("max_bins must be greater than or equal to min_bins.");
    }

    if (bin_cutoff <= 0.0 || bin_cutoff >= 1.0) {
      throw std::invalid_argument("bin_cutoff must be between 0 and 1.");
    }

    if (max_n_prebins < min_bins) {
      throw std::invalid_argument("max_n_prebins must be at least min_bins.");
    }

    if (max_iterations <= 0) {
      throw std::invalid_argument("max_iterations must be positive.");
    }

    if (convergence_threshold <= 0.0) {
      throw std::invalid_argument("convergence_threshold must be positive.");
    }

    // A missing target is an error (numerical NA contract)
    for (int t : target) {
      if (t == NA_INTEGER) {
        throw std::invalid_argument("Target contains missing values (NA).");
      }
    }

    // Check that target is binary (0 or 1)
    auto [min_it, max_it] = std::minmax_element(target.begin(), target.end());
    if (*min_it < 0 || *max_it > 1) {
      throw std::invalid_argument("Target must be binary (0 or 1).");
    }

    // Check that both classes are present
    int sum_target = std::accumulate(target.begin(), target.end(), 0);
    if (sum_target == 0 || sum_target == static_cast<int>(target.size())) {
      throw std::invalid_argument("Target must contain both classes (0 and 1).");
    }

    // Count non-missing values (NA / NaN rows are excluded from the fit)
    int valid_count = 0;
    for (const auto& value : feature) {
      if (!std::isnan(value)) {
        valid_count++;
      }
    }

    if (valid_count == 0) {
      throw std::invalid_argument("Feature has no non-missing values.");
    }
  }

  /**
   * Create initial bins based on unique values or quantiles
   */
  void createInitialBins() {
    // Split the non-missing feature values by class and sort each class.
    // -Inf / +Inf are kept: they are counted in the first / last bin, but only
    // finite values ever become a cutpoint.
    pos_sorted.clear();
    neg_sorted.clear();
    for (size_t i = 0; i < feature.size(); ++i) {
      if (!std::isnan(feature[i])) {
        if (target[i] == 1) pos_sorted.push_back(feature[i]);
        else neg_sorted.push_back(feature[i]);
      }
    }
    std::sort(pos_sorted.begin(), pos_sorted.end());
    std::sort(neg_sorted.begin(), neg_sorted.end());

    // Sorted distinct values: merge of the two sorted class arrays.
    std::vector<double> sorted_feature;
    {
      size_t i = 0, k = 0;
      const size_t np = pos_sorted.size(), nn = neg_sorted.size();
      while (i < np || k < nn) {
        double v;
        if (k >= nn || (i < np && pos_sorted[i] < neg_sorted[k])) v = pos_sorted[i];
        else v = neg_sorted[k];
        while (i < np && pos_sorted[i] == v) ++i;
        while (k < nn && neg_sorted[k] == v) ++k;
        if (std::isfinite(v)) sorted_feature.push_back(v + 0.0);  // -0.0 == +0.0
      }
    }

    int unique_vals = static_cast<int>(sorted_feature.size());

    // Special case for few unique values
    if (unique_vals <= 2) {
      handleFewUniqueValues(sorted_feature);
    } else {
      // Regular binning for more unique values
      createRegularBins(sorted_feature, unique_vals);
    }
  }

  // Number of elements of the sorted array v that are <= x.
  static int countLE(const std::vector<double>& v, double x) {
    return static_cast<int>(std::upper_bound(v.begin(), v.end(), x) - v.begin());
  }

  // Counts of the observations in (lower, upper]; the first bin (lower = -Inf)
  // and the last (upper = +Inf) are unbounded on that side.
  void countInterval(NumericalBin& bin) const {
    const bool lo_inf = std::isinf(bin.lower_bound) && bin.lower_bound < 0;
    const bool hi_inf = std::isinf(bin.upper_bound) && bin.upper_bound > 0;
    const int p_hi = hi_inf ? static_cast<int>(pos_sorted.size()) : countLE(pos_sorted, bin.upper_bound);
    const int n_hi = hi_inf ? static_cast<int>(neg_sorted.size()) : countLE(neg_sorted, bin.upper_bound);
    const int p_lo = lo_inf ? 0 : countLE(pos_sorted, bin.lower_bound);
    const int n_lo = lo_inf ? 0 : countLE(neg_sorted, bin.lower_bound);
    bin.count_pos = p_hi - p_lo;
    bin.count_neg = n_hi - n_lo;
    bin.count = bin.count_pos + bin.count_neg;
  }

  /**
   * Handle the special case where there are few (<= 2) unique values
   *
   * @param sorted_feature Vector of sorted unique feature values
   */
  void handleFewUniqueValues(const std::vector<double>& sorted_feature) {
    is_simple = true;
    bin_edges.clear();
    bin_info.clear();

    int unique_vals = static_cast<int>(sorted_feature.size());

    bin_edges.push_back(-std::numeric_limits<double>::infinity());
    if (unique_vals == 2) {
      bin_edges.push_back(sorted_feature[0]);
    }
    bin_edges.push_back(std::numeric_limits<double>::infinity());

    for (size_t i = 0; i + 1 < bin_edges.size(); ++i) {
      NumericalBin bin;
      bin.lower_bound = bin_edges[i];
      bin.upper_bound = bin_edges[i + 1];
      countInterval(bin);
      bin_info.push_back(bin);
    }

    // One bin per distinct value is an exact, final binning. It sets is_simple,
    // which makes fit() skip applyIsotonicRegression() -- the only other place
    // that sets this flag -- so without this line every 0/1 feature reported
    // converged = FALSE despite a perfectly correct result.
    converged = true;
  }

  /**
   * Create regular bins for normal case (more than 2 unique values)
   * 
   * @param sorted_feature Vector of sorted unique feature values
   * @param unique_vals Number of unique values
   */
  void createRegularBins(const std::vector<double>& sorted_feature, int unique_vals) {
    is_simple = false;

    // Determine number of pre-bins
    int n_prebins = std::min({max_n_prebins, unique_vals, max_bins});
    n_prebins = std::max(n_prebins, min_bins);

    // Create bin edges
    bin_edges.resize(static_cast<size_t>(n_prebins + 1));
    bin_edges[0] = -std::numeric_limits<double>::infinity();
    bin_edges[n_prebins] = std::numeric_limits<double>::infinity();

    // Use quantiles for bin edges
    for (int i = 1; i < n_prebins; ++i) {
      // Calculate index for quantile
      double q = static_cast<double>(i) / n_prebins;
      int idx = static_cast<int>(std::round(q * unique_vals));
      idx = std::max(1, std::min(idx, unique_vals - 1));
      bin_edges[i] = sorted_feature[static_cast<size_t>(idx - 1)];
    }

    // Ensure uniqueness of bin edges (can happen with skewed distributions)
    std::sort(bin_edges.begin(), bin_edges.end());
    bin_edges.erase(std::unique(bin_edges.begin(), bin_edges.end()), bin_edges.end());

    // If we lost some bin edges due to duplicates, adjust
    if (bin_edges.size() < 3) {
      // Fall back to min/max approach
      bin_edges.clear();
      bin_edges.push_back(-std::numeric_limits<double>::infinity());

      double min_val = sorted_feature.front();
      double max_val = sorted_feature.back();
      double middle = (min_val + max_val) / 2.0;

      bin_edges.push_back(middle);
      bin_edges.push_back(std::numeric_limits<double>::infinity());
    }
  }

  /**
   * Merge bins with frequency below the cutoff threshold
   * This ensures statistical reliability of each bin
   */
  void mergeLowFrequencyBins() {
    // Initialize bin_info from bin_edges
    initializeBinsFromEdges();

    int total_count = 0;
    for (const auto& bin : bin_info) {
      total_count += bin.count;
    }

    double min_count = bin_cutoff * total_count;

    // Iteratively merge small bins
    bool merged = true;
    int iterations = 0;

    while (merged && iterations < max_iterations && bin_info.size() > static_cast<size_t>(min_bins)) {
      merged = false;

      for (size_t i = 0; i < bin_info.size(); ++i) {
        if (bin_info[i].count < min_count) {
          // Find optimal merge direction
          if (i == 0 && bin_info.size() > 1) {
            // First bin - merge with next
            mergeBins(0, 1);
          } else if (i == bin_info.size() - 1 && i > 0) {
            // Last bin - merge with previous
            mergeBins(i - 1, i);
          } else if (i > 0 && i < bin_info.size() - 1) {
            // Middle bin - compare event rates
            double rate_diff_prev = std::fabs(bin_info[i].event_rate() - bin_info[i-1].event_rate());
            double rate_diff_next = std::fabs(bin_info[i].event_rate() - bin_info[i+1].event_rate());

            if (rate_diff_prev <= rate_diff_next) {
              // Merge with previous
              mergeBins(i - 1, i);
            } else {
              // Merge with next
              mergeBins(i, i + 1);
            }
          }

          merged = true;
          break;
        }
      }

      iterations++;
    }

    iterations_run += iterations;
  }

  /**
   * Initialize bin information from bin edges
   * Assigns observations to bins and calculates initial statistics
   */
  void initializeBinsFromEdges() {
    bin_info.clear();
    for (size_t i = 0; i + 1 < bin_edges.size(); ++i) {
      NumericalBin bin;
      bin.lower_bound = bin_edges[i];
      bin.upper_bound = bin_edges[i + 1];
      countInterval(bin);
      bin_info.push_back(bin);
    }
  }

  /**
   * Merge two adjacent bins
   * 
   * @param idx1 Index of first bin
   * @param idx2 Index of second bin (must be adjacent to idx1)
   */
  void mergeBins(size_t idx1, size_t idx2) {
    if (idx1 > idx2) {
      std::swap(idx1, idx2);
    }

    if (idx2 != idx1 + 1) {
      throw std::invalid_argument("Can only merge adjacent bins");
    }

    // Merge statistics
    bin_info[idx1].upper_bound = bin_info[idx2].upper_bound;
    bin_info[idx1].count += bin_info[idx2].count;
    bin_info[idx1].count_pos += bin_info[idx2].count_pos;
    bin_info[idx1].count_neg += bin_info[idx2].count_neg;

    // Recalculate event rate
    // bin_info[idx1].event_rate() assignment removed (calculated dynamically)

    // Remove second bin
    bin_info.erase(bin_info.begin() + idx2);
  }

  /**
   * Ensure number of bins is within [min_bins, max_bins]
   * Either split large bins or merge similar bins
   */
  void ensureMinMaxBins() {
    // Add bins if below min_bins.
    // splitLargestBin() is a silent no-op whenever no bin can actually be
    // divided (every bin holds a single observation, or the candidate split
    // point coincides with a boundary). Without a progress check this loop spun
    // forever -- and with no R_CheckUserInterrupt() in it, the session could not
    // even be interrupted -- for any feature whose distinct-value count is below
    // min_bins. Stop as soon as a pass fails to add a bin.
    while (bin_info.size() < static_cast<size_t>(min_bins) && bin_info.size() > 1) {
      size_t before = bin_info.size();
      splitLargestBin();
      if (bin_info.size() == before) break;
    }

    // Merge bins if above max_bins (same progress guard, for symmetry)
    while (bin_info.size() > static_cast<size_t>(max_bins)) {
      size_t before = bin_info.size();
      mergeSimilarBins();
      if (bin_info.size() == before) break;
    }
  }

  /**
   * Split the largest bin that holds at least two distinct values.
   *
   * The cut is placed at the distinct value of the bin's median observation, or
   * at the next smaller distinct value when the median is the bin's largest
   * value, so both halves are non-empty and their counts are exactly the
   * observations between the new cutpoints. A bin holding a single distinct
   * value cannot be split; when no bin can, this is a no-op.
   */
  void splitLargestBin() {
    size_t best = bin_info.size();
    double best_split = 0.0;
    for (size_t idx = 0; idx < bin_info.size(); ++idx) {
      const NumericalBin& b = bin_info[idx];
      if (best < bin_info.size() && b.count <= bin_info[best].count) continue;
      double split_value;
      if (findSplitValue(b, split_value)) {
        best = idx;
        best_split = split_value;
      }
    }
    if (best >= bin_info.size()) return;

    NumericalBin bin1 = bin_info[best];
    NumericalBin bin2 = bin_info[best];
    bin1.upper_bound = best_split;
    bin2.lower_bound = best_split;
    countInterval(bin1);
    countInterval(bin2);
    bin_info[best] = bin1;
    bin_info.insert(bin_info.begin() + static_cast<std::ptrdiff_t>(best) + 1, bin2);
  }

  // Cut value inside bin b leaving both halves non-empty; false if none exists.
  // Prefers the value of the bin's median observation; when that is the bin's
  // largest value, the next smaller distinct value. O(bin size).
  bool findSplitValue(const NumericalBin& b, double& split_value) const {
    if (b.count < 2) return false;
    const bool lo_inf = std::isinf(b.lower_bound) && b.lower_bound < 0;
    const bool hi_inf = std::isinf(b.upper_bound) && b.upper_bound > 0;
    auto slice = [&](const std::vector<double>& v, size_t& from, size_t& to) {
      from = lo_inf ? 0 : static_cast<size_t>(countLE(v, b.lower_bound));
      to = hi_inf ? v.size() : static_cast<size_t>(countLE(v, b.upper_bound));
    };
    size_t p0, p1, n0, n1;
    slice(pos_sorted, p0, p1);
    slice(neg_sorted, n0, n1);

    // The bin's finite values in ascending order (a cutpoint is always finite;
    // -Inf / +Inf stay with the outer halves).
    std::vector<double> vals;
    vals.reserve(static_cast<size_t>(b.count));
    size_t i = p0, k = n0;
    while (i < p1 || k < n1) {
      double v;
      if (k >= n1 || (i < p1 && pos_sorted[i] <= neg_sorted[k])) v = pos_sorted[i++];
      else v = neg_sorted[k++];
      if (std::isfinite(v)) vals.push_back(v);
    }
    if (vals.size() < 2 || !(vals.front() < vals.back())) return false;  // one distinct value
    const double v_max = vals.back();
    const double v_med = vals[vals.size() / 2];
    if (v_med < v_max) {
      split_value = v_med;
    } else {
      // largest value below the maximum
      split_value = *(std::lower_bound(vals.begin(), vals.end(), v_max) - 1);
    }
    return true;
  }

  /**
   * Merge the two most similar adjacent bins
   * Similarity is based on event rates
   */
  void mergeSimilarBins() {
    if (bin_info.size() <= 2) return;

    double min_diff = std::numeric_limits<double>::max();
    size_t merge_idx = 0;

    // Find most similar adjacent bins
    for (size_t i = 0; i < bin_info.size() - 1; ++i) {
      double diff = std::fabs(bin_info[i].event_rate() - bin_info[i+1].event_rate());
      if (diff < min_diff) {
        min_diff = diff;
        merge_idx = i;
      }
    }

    // Merge bins
    mergeBins(merge_idx, merge_idx + 1);
  }

  /**
   * Determine the optimal monotonicity direction (increasing or decreasing)
   * based on correlation between bin midpoints and event rates
   */
  void determineMonotonicityDirection() {
    if (bin_info.size() <= 1) return;

    // Calculate bin midpoints
    std::vector<double> midpoints;
    std::vector<double> rates;

    for (const auto& bin : bin_info) {
      // For infinite bounds, use the next/previous finite bound
      double lower = bin.lower_bound;
      double upper = bin.upper_bound;

      if (std::isinf(lower) && bin_info.size() > 1) {
        lower = bin_info[1].lower_bound - 1.0;
      }

      if (std::isinf(upper) && bin_info.size() > 1) {
        upper = bin_info[bin_info.size() - 2].upper_bound + 1.0;
      }

      double midpoint = (lower + upper) / 2.0;

      midpoints.push_back(midpoint);
      rates.push_back(bin.event_rate());
    }

    // Calculate correlation
    double correlation = calculateCorrelation(midpoints, rates);

    // Determine direction based on correlation
    monotone_increasing = (correlation >= 0.0);
  }

  /**
   * Calculate Pearson correlation coefficient between two vectors
   * 
   * @param x First vector
   * @param y Second vector
   * @return Correlation coefficient
   */
  double calculateCorrelation(const std::vector<double>& x, const std::vector<double>& y) const {
    if (x.size() != y.size() || x.size() < 2) {
      return 0.0;
    }

    // Calculate means
    double mean_x = std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(x.size());
    double mean_y = std::accumulate(y.begin(), y.end(), 0.0) / static_cast<double>(y.size());

    // Calculate correlation coefficient
    double numerator = 0.0;
    double sum_sq_x = 0.0;
    double sum_sq_y = 0.0;

    for (size_t i = 0; i < x.size(); ++i) {
      double x_diff = x[i] - mean_x;
      double y_diff = y[i] - mean_y;
      numerator += x_diff * y_diff;
      sum_sq_x += x_diff * x_diff;
      sum_sq_y += y_diff * y_diff;
    }

    if (sum_sq_x < EPSILON || sum_sq_y < EPSILON) {
      return 0.0;
    }

    return numerator / std::sqrt(sum_sq_x * sum_sq_y);
  }

  /**
   * Apply isotonic regression to enforce monotonicity in event rates, with the
   * Pool Adjacent Violators Algorithm (PAVA; Barlow et al., 1972; Best &
   * Chakravarti, 1990), and pool the bins accordingly.
   *
   * Stack-based PAVA: the bins are scanned once in the direction of the trend;
   * each bin is pushed as a block and, while the block below the top has a
   * strictly higher event rate than the top, the two are pooled. Every bin is
   * pushed once and pooled at most once, so the pass is O(bins). Event rates
   * are compared exactly on the integer counts, a/b > c/d <=> a*d > c*b, so a
   * pooled block is never split or joined by rounding.
   *
   * "Pool adjacent violators" means the violating bins form one block. With
   * the bin counts as weights, the fitted rate of a block is by construction
   * the weighted mean of its members' event rates, that is exactly
   *     sum(count_pos) / sum(count)
   * over the block. Merging the block's bins therefore reproduces the isotonic
   * fit exactly, while count, count_pos and count_neg stay equal to what is
   * actually observed between the reported cutpoints. (Overwriting the counts
   * with round(fitted_rate * count), as this routine once did, described a
   * distribution that does not exist.)
   */
  void applyIsotonicRegression() {
    const size_t n = bin_info.size();
    if (n <= 1) {
      // A single bin is trivially monotone: nothing to pool, and the result is
      // final. That is a successful termination, not a failure to converge.
      converged = true;
      return;
    }

    struct Block {
      long long pos;
      long long cnt;
      size_t size;  // number of bins pooled into this block
    };
    // rate(a) > rate(b); an empty block has rate 0, as NumericalBin::event_rate().
    auto rate_greater = [](const Block& a, const Block& b) {
      if (a.cnt > 0 && b.cnt > 0) return a.pos * b.cnt > b.pos * a.cnt;
      const double ra = a.cnt > 0 ? static_cast<double>(a.pos) / static_cast<double>(a.cnt) : 0.0;
      const double rb = b.cnt > 0 ? static_cast<double>(b.pos) / static_cast<double>(b.cnt) : 0.0;
      return ra > rb;
    };

    // For a decreasing trend, run the increasing PAVA on the reversed sequence.
    std::vector<Block> st;
    st.reserve(n);
    for (size_t t = 0; t < n; ++t) {
      const NumericalBin& b = bin_info[monotone_increasing ? t : n - 1 - t];
      st.push_back(Block{b.count_pos, b.count, 1});
      while (st.size() >= 2 && rate_greater(st[st.size() - 2], st.back())) {
        Block top = st.back();
        st.pop_back();
        st.back().pos += top.pos;
        st.back().cnt += top.cnt;
        st.back().size += top.size;
      }
    }
    if (!monotone_increasing) std::reverse(st.begin(), st.end());

    // Realise the pooling on the bins, in one pass.
    std::vector<NumericalBin> pooled;
    pooled.reserve(st.size());
    size_t start = 0;
    for (const Block& blk : st) {
      NumericalBin merged = bin_info[start];
      for (size_t k = 1; k < blk.size; ++k) {
        const NumericalBin& nxt = bin_info[start + k];
        merged.upper_bound = nxt.upper_bound;
        merged.count += nxt.count;
        merged.count_pos += nxt.count_pos;
        merged.count_neg += nxt.count_neg;
      }
      pooled.push_back(merged);
      start += blk.size;
    }
    bin_info.swap(pooled);

    converged = true;
    iterations_run += 1;
  }

  /**
   * Calculate Weight of Evidence (WoE) and Information Value (IV)
   * for each bin and the total binning solution
   */
  void calculateWOEandIV() {
    // Calculate totals
    double total_pos = 0.0, total_neg = 0.0;
    for (const auto& bin : bin_info) {
      total_pos += bin.count_pos;
      total_neg += bin.count_neg;
    }

    // Apply Laplace smoothing to handle zero counts
    double pos_denominator = total_pos + static_cast<double>(bin_info.size()) * ALPHA;
    double neg_denominator = total_neg + static_cast<double>(bin_info.size()) * ALPHA;

    if (pos_denominator < EPSILON || neg_denominator < EPSILON) {
      throw std::runtime_error("Insufficient positive or negative cases for WoE and IV calculations.");
    }

    total_iv = 0.0;
    for (auto& bin : bin_info) {
      // Calculate rates with smoothing
      double pos_rate = (bin.count_pos + ALPHA) / pos_denominator;
      double neg_rate = (bin.count_neg + ALPHA) / neg_denominator;

      // Calculate WoE
      bin.woe = std::log(pos_rate / neg_rate);

      // Calculate IV contribution
      bin.iv = (pos_rate - neg_rate) * bin.woe;
      total_iv += bin.iv;
    }
  }

  /**
   * Create bin labels for output
   * 
   * @param bin The bin information
   * @param is_first Whether this is the first bin
   * @param is_last Whether this is the last bin
   * @return Formatted bin label string
   */
  std::string createBinLabel(const NumericalBin& bin, bool is_first, bool is_last) const {
    std::ostringstream oss;
    oss.precision(6);
    oss << std::fixed;

    if (is_first) {
      oss << "(-Inf;" << bin.upper_bound << "]";
    } else if (is_last) {
      oss << "(" << bin.lower_bound << ";+Inf]";
    } else {
      oss << "(" << bin.lower_bound << ";" << bin.upper_bound << "]";
    }

    return oss.str();
  }

  /**
   * Create final WOE bin list for output
   * 
   * @return List with bin information and metrics
   */
  Rcpp::List createWOEBinList() const {
    int n_bins = static_cast<int>(bin_info.size());
    Rcpp::CharacterVector bin_labels(n_bins);
    Rcpp::NumericVector woe_vec(n_bins), iv_vec(n_bins);
    Rcpp::IntegerVector count_vec(n_bins), count_pos_vec(n_bins), count_neg_vec(n_bins);
    Rcpp::NumericVector cutpoints(std::max(n_bins - 1, 0));

    for (int i = 0; i < n_bins; ++i) {
      const auto& b = bin_info[static_cast<size_t>(i)];
      std::string label = createBinLabel(b, i == 0, i == n_bins - 1);

      bin_labels[i] = label;
      woe_vec[i] = b.woe;
      iv_vec[i] = b.iv;
      count_vec[i] = b.count;
      count_pos_vec[i] = b.count_pos;
      count_neg_vec[i] = b.count_neg;

      if (i < n_bins - 1) {
        cutpoints[i] = b.upper_bound;
      }
    }

    // Create bin IDs (1-based indexing for R)
    Rcpp::NumericVector ids(bin_labels.size());
    for(int i = 0; i < bin_labels.size(); i++) {
      ids[i] = i + 1;
    }

    return Rcpp::List::create(
      Named("id") = ids,
      Rcpp::Named("bin") = bin_labels,
      Rcpp::Named("woe") = woe_vec,
      Rcpp::Named("iv") = iv_vec,
      Rcpp::Named("count") = count_vec,
      Rcpp::Named("count_pos") = count_pos_vec,
      Rcpp::Named("count_neg") = count_neg_vec,
      Rcpp::Named("cutpoints") = cutpoints,
      Rcpp::Named("converged") = converged,
      Rcpp::Named("iterations") = iterations_run,
      Rcpp::Named("total_iv") = total_iv,
      Rcpp::Named("monotone_increasing") = monotone_increasing
    );
  }
};

// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_ir(
   Rcpp::IntegerVector target,
   Rcpp::NumericVector feature,
   int min_bins = 3,
   int max_bins = 5,
   double bin_cutoff = 0.05,
   int max_n_prebins = 20,
   bool auto_monotonicity = true,
   double convergence_threshold = 1e-6,
   int max_iterations = 1000) {

 try {
   // Convert R vectors to STL containers
   std::vector<int> target_std = Rcpp::as<std::vector<int>>(target);
   std::vector<double> feature_std = Rcpp::as<std::vector<double>>(feature);

   // Create and execute binning algorithm
   OBN_IR binner(
       min_bins, max_bins, 
       bin_cutoff, max_n_prebins,
       convergence_threshold, max_iterations,
       feature_std, target_std,
       auto_monotonicity
   );

   binner.fit();
   return binner.getResults();
 } catch (const std::exception& e) {
   Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
 }
}
