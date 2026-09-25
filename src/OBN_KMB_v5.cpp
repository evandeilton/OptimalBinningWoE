// [[Rcpp::plugins(cpp11)]]

#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#include <string>
#include <sstream>
#include <numeric>
#include <functional>

/**
 * @file OBN_KMB.cpp
 * @brief Implementation of K-means Binning (KMB) algorithm for optimal binning of numerical variables
 * 
 * This implementation provides methods for supervised discretization of numerical variables
 * using a hybrid approach inspired by K-means clustering and information theory metrics.
 */

// Include shared headers
#include "common/optimal_binning_common.h"
#include "common/bin_structures.h"

using namespace Rcpp;
using namespace OptimalBinning;


/**
 * Class for Optimal Binning using K-means Binning (KMB)
 * 
 * This class implements a numerical binning algorithm that partitions a continuous
 * feature into optimal bins based on its relationship with a binary target variable.
 * The approach combines elements of clustering (like K-means) with information theory
 * to maximize predictive power.
 */
class OBN_KMB {
private:
  std::vector<double> feature;
  std::vector<int> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  int max_iterations;
  bool enforce_monotonic;
  
  bool converged;
  int iterations_run;
  double total_iv;
  
  // Flag to indicate special handling for few unique values
  bool is_unique_two_or_less;
  
  // Constant for floating point comparison
  // Constant removed (uses shared definition)
  
  /**
   * Structure representing a bin and its statistics
   */
  // Local NumericalBin definition removed
  
  
  std::vector<NumericalBin> bins;
  
  /**
   * Calculate Weight of Evidence (WoE) with Laplace smoothing
   * 
   * WoE = ln((% positive + smoothing) / (% negative + smoothing))
   * 
   * @param pos Count of positive cases in bin
   * @param neg Count of negative cases in bin
   * @param total_pos Total positive cases across all bins
   * @param total_neg Total negative cases across all bins
   * @return Weight of Evidence value
   */
  double calculateWOE(int pos, int neg, int total_pos, int total_neg) const {
    // Apply Laplace smoothing to handle zero counts
    double alpha = 0.5;  // Smoothing parameter
    double num_bins = static_cast<double>(bins.size());
    
    double pos_rate = (static_cast<double>(pos) + alpha) / (static_cast<double>(total_pos) + num_bins * alpha);
    double neg_rate = (static_cast<double>(neg) + alpha) / (static_cast<double>(total_neg) + num_bins * alpha);
    
    return std::log(pos_rate / neg_rate);
  }
  
  /**
   * Calculate Information Value (IV) contribution for a bin
   * 
   * IV = (% positive - % negative) * WoE
   * 
   * @param woe Weight of Evidence value
   * @param pos Count of positive cases in bin
   * @param neg Count of negative cases in bin
   * @param total_pos Total positive cases across all bins
   * @param total_neg Total negative cases across all bins
   * @return Information Value contribution
   */
  double calculateIV(double woe, int pos, int neg, int total_pos, int total_neg) const {
    if (total_pos <= 0 || total_neg <= 0) {
      return 0.0;
    }
    
    double pos_dist = static_cast<double>(pos) / static_cast<double>(total_pos);
    double neg_dist = static_cast<double>(neg) / static_cast<double>(total_neg);
    
    return (pos_dist - neg_dist) * woe;
  }
  
  /**
   * Validate input data and parameters
   * Throws exception if validation fails
   */
  void validateInputs() const {
    if (feature.empty() || target.empty()) {
      Rcpp::stop("Feature and target vectors must not be empty.");
    }
    
    if (feature.size() != target.size()) {
      Rcpp::stop("Feature and target vectors must have the same length.");
    }
    
    // Count valid (non-NaN, non-Inf) values
    int valid_count = 0;
    for (const auto& val : feature) {
      if (!std::isnan(val) && !std::isinf(val)) {
        valid_count++;
      }
    }
    
    if (valid_count == 0) {
      Rcpp::stop("Feature vector must contain at least one valid (non-NaN, non-Inf) value.");
    }
    
    // Check that target contains only 0 and 1 (NA is "anything else")
    int pos_count = 0;
    for (const auto& t : target) {
      if (t != 0 && t != 1) {
        Rcpp::stop("Target vector must contain only binary values 0 and 1.");
      }
      pos_count += t;
    }

    if (pos_count == 0 || pos_count == static_cast<int>(target.size())) {
      Rcpp::stop("Target vector must contain both positive (1) and negative (0) cases.");
    }

    // Validate parameters
    if (min_bins < 2) {
      Rcpp::stop("min_bins must be at least 2.");
    }
    
    if (max_bins < min_bins) {
      Rcpp::stop("max_bins must be greater than or equal to min_bins.");
    }
    
    if (bin_cutoff <= 0 || bin_cutoff >= 1) {
      Rcpp::stop("bin_cutoff must be between 0 and 1.");
    }
    
    if (max_n_prebins <= 0) {
      Rcpp::stop("max_n_prebins must be positive.");
    }
    
    if (max_iterations <= 0) {
      Rcpp::stop("max_iterations must be positive.");
    }
  }
  
  /**
   * Perform initial binning based on unique values or quantiles
   */
  void initialBinning() {
    // Only three facts about the finite feature values are needed here: their
    // minimum, their maximum and their number of distinct values -- and the
    // latter only up to the point where it can still change the number of
    // initial bins, min(max_n_prebins, #distinct), or the "at most two
    // distinct values" special case. That needs no sort of the sample: one pass
    // keeps min / max and a short list of the distinct values seen so far,
    // which stops growing once it exceeds that cap. Large caps
    // (max_n_prebins > 64) fall back to sort + unique.
    const size_t cap = static_cast<size_t>(std::max(max_n_prebins, 2)) + 1;
    const bool small_cap = cap <= 65;
    double min_val = std::numeric_limits<double>::infinity();
    double max_val = -std::numeric_limits<double>::infinity();
    std::vector<double> distinct;   // distinct finite values (up to `cap`)
    bool capped = false;
    for (size_t i = 0; i < feature.size(); ++i) {
      const double v = feature[i];
      if (std::isnan(v) || std::isinf(v)) continue;
      if (v < min_val) min_val = v;
      if (v > max_val) max_val = v;
      if (!small_cap) {
        distinct.push_back(v);
      } else if (!capped) {
        bool seen = false;
        for (double d : distinct) {
          if (d == v) { seen = true; break; }
        }
        if (!seen) {
          distinct.push_back(v);
          if (distinct.size() >= cap) capped = true;
        }
      }
    }
    std::sort(distinct.begin(), distinct.end());
    if (!small_cap) {
      distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
    }
    // -0.0 and +0.0 are one value; report it unsigned.
    min_val += 0.0;
    max_val += 0.0;
    for (double& d : distinct) d += 0.0;

    const int num_unique_values = static_cast<int>(distinct.size());

    // Special case: very few unique values
    if (num_unique_values <= 2) {
      handleFewUniqueValues(distinct);
    } else {
      // Regular binning for more unique values
      createRegularBins(min_val, max_val, num_unique_values);
    }
  }

  /**
   * Handle the special case of very few (<=2) unique values. The counts are
   * filled in by assignDataToBins(), like those of the regular bins.
   *
   * @param unique_values Sorted unique finite feature values
   */
  void handleFewUniqueValues(const std::vector<double>& unique_values) {
    is_unique_two_or_less = true;
    bins.clear();

    const int num_unique_values = static_cast<int>(unique_values.size());

    if (num_unique_values == 0) {
      // No valid values - shouldn't happen due to validation, but handle anyway
      Rcpp::stop("No valid feature values to bin.");
    } else if (num_unique_values == 1) {
      // Single unique value - create one bin
      NumericalBin bin;
      bin.lower_bound = -std::numeric_limits<double>::infinity();
      bin.upper_bound = std::numeric_limits<double>::infinity();
      bin.centroid = unique_values[0];
      bins.push_back(bin);
    } else { // num_unique_values == 2
      // Two unique values - create two bins split at the smaller one
      NumericalBin bin1, bin2;
      bin1.lower_bound = -std::numeric_limits<double>::infinity();
      bin1.upper_bound = unique_values[0];
      bin2.lower_bound = unique_values[0];
      bin2.upper_bound = std::numeric_limits<double>::infinity();
      bin1.centroid = unique_values[0];
      bin2.centroid = unique_values[1];
      bins.push_back(bin1);
      bins.push_back(bin2);
    }
  }

  /**
   * Create regular bins for the normal case (more than 2 unique values)
   * Uses a K-means inspired approach for initial bin boundaries
   *
   * @param min_val Smallest finite feature value
   * @param max_val Largest finite feature value
   * @param num_unique_values Number of unique values (possibly capped)
   */
  void createRegularBins(double min_val, double max_val, int num_unique_values) {
    is_unique_two_or_less = false;

    // Determine number of initial bins
    int n_bins = std::min(max_n_prebins, num_unique_values);
    n_bins = std::max(n_bins, min_bins);
    n_bins = std::min(n_bins, max_bins);

    // K-means inspired approach: equally spaced centroids over [min, max].
    //
    // With three or more distinct values the range is positive, whatever its
    // magnitude (the previous absolute test `range < 1e-10` collapsed any
    // feature measured on a small scale into a single bin). The range, or its
    // product with (i + 0.5), can overflow to +Inf for extreme values such as
    // -1e308 and 1e308; the spacing is then computed from the halves.
    const double range = max_val - min_val;
    std::vector<double> centroids;
    centroids.reserve(static_cast<size_t>(n_bins));
    for (int i = 0; i < n_bins; ++i) {
      double centroid = min_val + (i + 0.5) * range / n_bins;
      if (!std::isfinite(centroid)) {
        const double t = (i + 0.5) / n_bins;
        centroid = min_val * (1.0 - t) + max_val * t;
      }
      centroids.push_back(centroid);
    }

    // Determine bin boundaries as midpoints between centroids
    bins.clear();
    bins.reserve(static_cast<size_t>(n_bins));

    for (int i = 0; i < n_bins; ++i) {
      NumericalBin bin;
      const size_t ui = static_cast<size_t>(i);
      bin.centroid = centroids[ui];

      if (i == 0) {
        bin.lower_bound = -std::numeric_limits<double>::infinity();
      } else {
        bin.lower_bound = midpoint(centroids[ui - 1], centroids[ui]);
      }

      if (i == n_bins - 1) {
        bin.upper_bound = std::numeric_limits<double>::infinity();
      } else {
        bin.upper_bound = midpoint(centroids[ui], centroids[ui + 1]);
      }

      bins.push_back(bin);
    }
  }

  // (a + b) / 2 for finite a, b, without overflow.
  static double midpoint(double a, double b) {
    const double m = (a + b) / 2.0;
    return std::isfinite(m) ? m : a / 2.0 + b / 2.0;
  }

  // Upper bounds of the bins, the last one forced to +Inf, for lower_bound().
  std::vector<double> binUppers() const {
    std::vector<double> uppers(bins.size());
    for (size_t b = 0; b < bins.size(); ++b) uppers[b] = bins[b].upper_bound;
    uppers.back() = std::numeric_limits<double>::infinity();
    return uppers;
  }

  /**
   * Assign data points to bins and calculate bin statistics
   *
   * A value goes to the first bin whose upper bound is >= the value: the bins
   * are right-closed intervals (lower, upper] covering the whole real line
   * (the first one also holds -Inf, the last one +Inf). NaN values are skipped.
   */
  void assignDataToBins() {
    const std::vector<double> uppers = binUppers();
    for (auto& bin : bins) {
      bin.count = 0;
      bin.count_pos = 0;
      bin.count_neg = 0;
    }

    for (size_t i = 0; i < feature.size(); ++i) {
      const double value = feature[i];
      if (std::isnan(value)) continue;
      NumericalBin& bin = bins[static_cast<size_t>(
        std::lower_bound(uppers.begin(), uppers.end(), value) - uppers.begin())];
      bin.count++;
      if (target[i] == 1) bin.count_pos++;
    }

    for (auto& bin : bins) {
      bin.count_neg = bin.count - bin.count_pos;
    }
  }

  /**
   * Set each bin's centroid to the mean of its finite feature values, as the
   * documentation of the `centroids` output states. (It used to report the
   * evenly spaced initial centroid, count-weighted through the merges, which
   * is not the mean of anything in the data.) A bin holding no finite value
   * keeps that initial centroid.
   */
  void computeCentroids() {
    const std::vector<double> uppers = binUppers();
    std::vector<double> mean(bins.size(), 0.0);
    std::vector<double> k(bins.size(), 0.0);
    for (size_t i = 0; i < feature.size(); ++i) {
      const double value = feature[i];
      if (!std::isfinite(value)) continue;
      const size_t b = static_cast<size_t>(
        std::lower_bound(uppers.begin(), uppers.end(), value) - uppers.begin());
      // Running mean: cannot overflow, unlike a plain sum of extreme values.
      k[b] += 1.0;
      mean[b] += (value - mean[b]) / k[b];
    }
    for (size_t b = 0; b < bins.size(); ++b) {
      if (k[b] > 0.0) bins[b].centroid = mean[b];
    }
  }

  /**
   * Merge bins with frequency below the cutoff threshold
   * This ensures statistical reliability of each bin
   */
  void mergeLowFrequencyBins() {
    int total_count = 0;
    for (const auto& bin : bins) {
      total_count += bin.count;
    }
    
    double cutoff_count = bin_cutoff * total_count;
    
    int iterations = 0;
    bool merged = true;
    
    while (merged && iterations < max_iterations && static_cast<int>(bins.size()) > min_bins) {
      merged = false;
      
      for (size_t i = 0; i < bins.size(); ++i) {
        if (bins[i].count < cutoff_count) {
          // Determine optimal merge direction
          if (i == 0 && bins.size() > 1) {
            // First bin - merge with next
            mergeBins(0, 1);
          } else if (i == bins.size() - 1 && i > 0) {
            // Last bin - merge with previous
            mergeBins(i - 1, i);
          } else if (i > 0 && i < bins.size() - 1) {
            // Middle bin - choose based on event rate similarity
            double diff_prev = std::fabs(bins[i].event_rate() - bins[i-1].event_rate());
            double diff_next = std::fabs(bins[i].event_rate() - bins[i+1].event_rate());
            
            if (diff_prev <= diff_next) {
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
    if (iterations >= max_iterations) {
      converged = false;
    }
  }
  
  /**
   * Remove the bins that hold no observation, then restore min_bins.
   *
   * Equally spaced intervals over a skewed or outlier-laden feature (and, for
   * features spanning only a few ulps, intervals whose bounds coincide) can be
   * empty. mergeLowFrequencyBins() absorbs them only while more than min_bins
   * bins remain; one left over is not a bin of the data -- it has no event
   * rate, its WoE is an artefact of the smoothing and its bounds are
   * arbitrary. Each is merged into its lower neighbour (the first bin into the
   * next one). If that leaves fewer than min_bins bins, the most populous bin
   * whose values are not all equal is bisected at the midpoint of its value
   * range -- the equal-width rule applied locally, which always leaves both
   * halves non-empty -- until min_bins bins exist or no bin can be split
   * (fewer distinct values than min_bins; adjustBinCount() reports that).
   */
  void removeEmptyBins() {
    size_t i = 0;
    bool removed = false;
    while (i < bins.size() && bins.size() > 1) {
      if (bins[i].count == 0) {
        if (i == 0) mergeBins(0, 1);
        else mergeBins(i - 1, i);
        removed = true;
      } else {
        ++i;
      }
    }
    if (!removed) return;

    while (static_cast<int>(bins.size()) < min_bins) {
      // Smallest and largest finite value in each bin.
      const std::vector<double> uppers = binUppers();
      std::vector<double> vmin(bins.size(), std::numeric_limits<double>::infinity());
      std::vector<double> vmax(bins.size(), -std::numeric_limits<double>::infinity());
      for (size_t k = 0; k < feature.size(); ++k) {
        const double value = feature[k];
        if (!std::isfinite(value)) continue;
        const size_t b = static_cast<size_t>(
          std::lower_bound(uppers.begin(), uppers.end(), value) - uppers.begin());
        if (value < vmin[b]) vmin[b] = value;
        if (value > vmax[b]) vmax[b] = value;
      }
      size_t best = bins.size();
      for (size_t b = 0; b < bins.size(); ++b) {
        if (!(vmin[b] < vmax[b])) continue;
        if (best == bins.size() || bins[b].count > bins[best].count) best = b;
      }
      if (best == bins.size()) break;

      double cut = vmin[best] + (vmax[best] / 2.0 - vmin[best] / 2.0);
      if (!(cut < vmax[best]) || !(cut >= vmin[best])) cut = vmin[best];
      NumericalBin upper_half = bins[best];
      upper_half.lower_bound = cut;
      upper_half.centroid = vmax[best];
      bins[best].upper_bound = cut;
      bins[best].centroid = vmin[best];
      bins.insert(bins.begin() + static_cast<std::ptrdiff_t>(best) + 1, upper_half);
      assignDataToBins();
    }
  }

  /**
   * Merge two adjacent bins
   * 
   * @param idx1 Index of first bin to merge
   * @param idx2 Index of second bin to merge (must be adjacent to idx1)
   */
  void mergeBins(size_t idx1, size_t idx2) {
    if (idx1 > idx2) {
      std::swap(idx1, idx2);
    }
    
    if (idx2 != idx1 + 1) {
      Rcpp::stop("Can only merge adjacent bins");
    }
    
    NumericalBin& bin1 = bins[idx1];
    NumericalBin& bin2 = bins[idx2];
    
    // Calculate weighted centroid
    if (bin1.count + bin2.count > 0) {
      bin1.centroid = (bin1.centroid * bin1.count + bin2.centroid * bin2.count) / 
        (bin1.count + bin2.count);
    }
    
    // Merge boundaries and counts
    bin1.upper_bound = bin2.upper_bound;
    bin1.count += bin2.count;
    bin1.count_pos += bin2.count_pos;
    bin1.count_neg += bin2.count_neg;
    
    // Recalculate event rate
    // bin1.event_rate() assignment removed (calculated dynamically)
    
    // Remove second bin
    bins.erase(bins.begin() + idx2);
  }
  
  /**
   * Enforce monotonicity in WoE values by merging violating bins
   * This improves interpretability and model stability
   */
  void enforceMonotonicity() {
    if (!enforce_monotonic || bins.size() <= 2) {
      // Skip monotonicity enforcement if disabled or too few bins
      return;
    }
    
    // Calculate initial WoE values
    calculateBinStatistics();
    
    int iterations = 0;
    bool is_monotonic = false;
    
    // Determine monotonicity direction (increasing or decreasing)
    // based on the first two bins
    bool increasing = true;
    if (bins.size() >= 2) {
      increasing = (bins[1].woe >= bins[0].woe);
    }
    
    // Iteratively merge bins until monotonicity is achieved
    while (!is_monotonic && static_cast<int>(bins.size()) > min_bins && iterations < max_iterations) {
      is_monotonic = true;
      
      for (size_t i = 1; i < bins.size(); ++i) {
        // Check for monotonicity violation
        if ((increasing && bins[i].woe < bins[i - 1].woe) ||
            (!increasing && bins[i].woe > bins[i - 1].woe)) {
          
          // Merge bins to fix violation
          mergeBins(i - 1, i);
          
          // Recalculate WoE values
          calculateBinStatistics();
          
          is_monotonic = false;
          break;
        }
      }
      
      iterations++;
      
      // Stop if minimum bins reached
      if (static_cast<int>(bins.size()) <= min_bins) {
        break;
      }
    }
    
    iterations_run += iterations;
    if (iterations >= max_iterations) {
      converged = false;
    }
  }
  
  /**
   * Adjust number of bins to be within [min_bins, max_bins]
   * Uses optimal merge strategy to minimize information loss
   */
  void adjustBinCount() {
    int iterations = 0;
    
    // Calculate current WoE and IV values
    calculateBinStatistics();
    
    // If too many bins, merge until max_bins reached
    while (static_cast<int>(bins.size()) > max_bins && iterations < max_iterations) {
      // Find pair of adjacent bins with the smallest IV difference
      double min_iv_diff = std::numeric_limits<double>::max();
      int merge_index = -1;
      
      for (size_t i = 0; i < bins.size() - 1; ++i) {
        double diff = std::fabs(bins[i].iv - bins[i + 1].iv);
        if (diff < min_iv_diff) {
          min_iv_diff = diff;
          merge_index = static_cast<int>(i);
        }
      }
      
      if (merge_index != -1) {
        // Merge bins at merge_index and merge_index + 1
        mergeBins(merge_index, merge_index + 1);
        
        // Recalculate statistics
        calculateBinStatistics();
      } else {
        // No more bins can be merged
        break;
      }
      
      iterations++;
    }
    
    // If too few bins, split largest bins until min_bins reached
    while (static_cast<int>(bins.size()) < min_bins && iterations < max_iterations) {
      // For now, we'll just error out since splitting is complex
      // In a more advanced implementation, we could add bin splitting
      Rcpp::warning("Could not achieve minimum number of bins. Consider reducing min_bins parameter.");
      break;
    }
    
    iterations_run += iterations;
    if (iterations >= max_iterations) {
      converged = false;
    }
  }
  
  /**
   * Calculate WoE and IV for each bin
   * Updates the bin.woe and bin.iv properties
   */
  void calculateBinStatistics() {
    // Calculate total positives and negatives
    int total_pos = 0;
    int total_neg = 0;
    
    for (const auto& bin : bins) {
      total_pos += bin.count_pos;
      total_neg += bin.count_neg;
    }
    
    // Handle cases where total_pos or total_neg is zero
    if (total_pos == 0 || total_neg == 0) {
      Rcpp::stop("Target vector must contain both positive and negative cases.");
    }
    
    // Calculate WoE and IV for each bin
    total_iv = 0.0;
    for (auto& bin : bins) {
      bin.woe = calculateWOE(bin.count_pos, bin.count_neg, total_pos, total_neg);
      bin.iv = calculateIV(bin.woe, bin.count_pos, bin.count_neg, total_pos, total_neg);
      total_iv += bin.iv;
    }
  }
  
  /**
   * Format bin interval as a string
   * 
   * @param lower Lower bound of bin
   * @param upper Upper bound of bin
   * @return Formatted string representation of bin interval
   */
  std::string formatBinInterval(double lower, double upper) const {
    std::ostringstream oss;
    oss.precision(6);
    oss << std::fixed;
    
    oss << "(";
    if (std::isinf(lower) && lower < 0) {
      oss << "-Inf";
    } else {
      oss << lower;
    }
    
    oss << ";";
    
    if (std::isinf(upper) && upper > 0) {
      oss << "+Inf";
    } else {
      oss << upper;
    }
    
    oss << "]";
    return oss.str();
  }
  
public:
  /**
   * Constructor for OBN_KMB
   * 
   * @param feature_ Feature vector to be binned
   * @param target_ Binary target vector (0/1)
   * @param min_bins_ Minimum number of bins
   * @param max_bins_ Maximum number of bins
   * @param bin_cutoff_ Minimum frequency fraction for each bin
   * @param max_n_prebins_ Maximum number of pre-bins
   * @param enforce_monotonic_ Whether to enforce monotonicity in WoE
   * @param convergence_threshold_ Convergence threshold
   * @param max_iterations_ Maximum iterations allowed
   */
  OBN_KMB(
    const std::vector<double>& feature_, 
    const std::vector<int>& target_,
    int min_bins_, 
    int max_bins_, 
    double bin_cutoff_, 
    int max_n_prebins_,
    bool enforce_monotonic_ = true,
    double convergence_threshold_ = 1e-6, 
    int max_iterations_ = 1000)
    : feature(feature_), target(target_), 
      min_bins(min_bins_), max_bins(max_bins_),
      bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_),
      max_iterations(max_iterations_),
      enforce_monotonic(enforce_monotonic_),
      converged(true), iterations_run(0), total_iv(0.0),
      is_unique_two_or_less(false) {
    // Accepted for interface compatibility; this engine has no tolerance-based
    // stopping rule (every loop ends on a bin-count or monotonicity condition).
    (void) convergence_threshold_;
  }
  
  /**
   * Execute the binning algorithm and return results
   * 
   * @return List containing bin information and metrics
   */
  Rcpp::List fit() {
    // Step 1: Validate inputs
    validateInputs();
    
    // Step 2: Perform initial binning
    initialBinning();
    
    // Step 3: Assign data to bins
    assignDataToBins();
    
    // Step 4: Continue with optimization steps if not simple case
    if (!is_unique_two_or_less) {
      // Step 4a: Merge low frequency bins
      mergeLowFrequencyBins();

      // Step 4a': equal-width intervals holding no observation at all
      removeEmptyBins();
      
      // Step 4b: Enforce monotonicity if requested
      if (enforce_monotonic) {
        enforceMonotonicity();
      }
      
      // Step 4c: Adjust bin count to be within [min_bins, max_bins]
      adjustBinCount();
    }
    
    // Step 5: Final calculation of WoE and IV, and of the bin means
    calculateBinStatistics();
    computeCentroids();
    
    // Step 6: Prepare output
    return createResultList();
  }
  
  /**
   * Create the final result list
   * 
   * @return List with bin information and metrics
   */
  Rcpp::List createResultList() const {
    // Prepare output vectors
    std::vector<std::string> bin_labels;
    std::vector<double> woe_values;
    std::vector<double> iv_values;
    std::vector<int> counts;
    std::vector<int> counts_pos;
    std::vector<int> counts_neg;
    std::vector<double> cutpoints;
    std::vector<double> centroids;
    
    bin_labels.reserve(bins.size());
    woe_values.reserve(bins.size());
    iv_values.reserve(bins.size());
    counts.reserve(bins.size());
    counts_pos.reserve(bins.size());
    counts_neg.reserve(bins.size());
    cutpoints.reserve(bins.size() > 0 ? bins.size() - 1 : 0);
    centroids.reserve(bins.size());
    
    // Populate output vectors
    for (size_t i = 0; i < bins.size(); ++i) {
      const auto& bin = bins[i];
      bin_labels.push_back(formatBinInterval(bin.lower_bound, bin.upper_bound));
      woe_values.push_back(bin.woe);
      iv_values.push_back(bin.iv);
      counts.push_back(bin.count);
      counts_pos.push_back(bin.count_pos);
      counts_neg.push_back(bin.count_neg);
      centroids.push_back(bin.centroid);
      
      // Add cutpoints (exclude the last bin's upper bound)
      if (i < bins.size() - 1) {
        cutpoints.push_back(bin.upper_bound);
      }
    }
    
    // Create bin IDs (1-based indexing for R)
    Rcpp::NumericVector ids(bin_labels.size());
    for (int i = 0; i < static_cast<int>(bin_labels.size()); i++) {
      ids[i] = i + 1;
    }
    
    // Create and return final list
    return Rcpp::List::create(
      Rcpp::Named("id") = ids,
      Rcpp::Named("bin") = bin_labels,
      Rcpp::Named("woe") = woe_values,
      Rcpp::Named("iv") = iv_values,
      Rcpp::Named("count") = counts,
      Rcpp::Named("count_pos") = counts_pos,
      Rcpp::Named("count_neg") = counts_neg,
      Rcpp::Named("centroids") = centroids,
      Rcpp::Named("cutpoints") = cutpoints,
      Rcpp::Named("converged") = converged,
      Rcpp::Named("iterations") = iterations_run,
      Rcpp::Named("total_iv") = total_iv
    );
  }
};

// [[Rcpp::export]]
List optimal_binning_numerical_kmb(
    IntegerVector target,
    NumericVector feature,
    int min_bins = 3, 
    int max_bins = 5,
    double bin_cutoff = 0.05, 
    int max_n_prebins = 20,
    bool enforce_monotonic = true,
    double convergence_threshold = 1e-6, 
    int max_iterations = 1000) {
  
  try {
    // Convert R vectors to C++ vectors
    std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
    std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
    
    // Initialize binning class
    OBN_KMB binner(
        feature_vec, target_vec, 
        min_bins, max_bins, 
        bin_cutoff, max_n_prebins,
        enforce_monotonic,
        convergence_threshold, max_iterations);
    
    // Perform binning and return results
    return binner.fit();
  } catch(std::exception &e) {
    forward_exception_to_r(e);
  } catch(...) {
    Rcpp::stop("Unknown C++ exception in optimal_binning_numerical_kmb");
  }
  
  // Should never reach here
  return R_NilValue;
}
