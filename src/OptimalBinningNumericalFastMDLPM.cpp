#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <string>
#include <limits>
#include <stdexcept>

/**
 * @title Optimal Binning using Minimum Description Length Principle with Monotonicity
 * @description Implementation of numerical variable binning using the Minimum Description Length 
 * Principle (MDLP) with optional monotonicity constraints on the Weight of Evidence (WoE).
 * 
 * The algorithm recursively partitions the feature space by finding cut points that 
 * maximize information gain, subject to the MDLP criterion that determines whether a cut is 
 * justified based on information theory principles. The monotonicity constraint ensures that
 * the WoE values across bins follow a monotonic (strictly increasing or decreasing) pattern,
 * which is often desirable in credit risk modeling applications.
 * 
 * @references 
 * - Fayyad, U., & Irani, K. (1993). Multi-interval discretization of continuous-valued 
 *   attributes for classification learning. Proceedings of the 13th International 
 *   Joint Conference on Artificial Intelligence, 1022-1027.
 * - Pfahringer, B. (1995). Supervised and unsupervised discretization of continuous 
 *   features. Machine Learning: Proceedings of the Twelfth International Conference, 456-463.
 * - Kotsiantis, S., & Kanellopoulos, D. (2006). Discretization techniques: A recent 
 *   survey. GESTS International Transactions on Computer Science and Engineering, 32(1), 47-58.
 * - Belson, W. A. (1959). Matching and prediction on the principle of biological 
 *   classification. Journal of the Royal Statistical Society: Series C (Applied Statistics), 8(2), 65-75.
 */

// ----------------------------------------------------
// Helper functions
// ----------------------------------------------------

/**
 * Check if a vector is strictly increasing
 * 
 * Tests whether each element in the vector is strictly greater than the previous element.
 * 
 * @param x Vector to check
 * @return true if strictly increasing, false otherwise
 */
bool is_strictly_increasing(const std::vector<double> &x) {
  if (x.size() <= 1) return true;  // Empty or single-element vector is trivially increasing
  
  for (size_t i = 1; i < x.size(); i++) {
    if (x[i] <= x[i-1]) return false;
  }
  return true;
}

/**
 * Check if a vector is strictly decreasing
 * 
 * Tests whether each element in the vector is strictly less than the previous element.
 * 
 * @param x Vector to check
 * @return true if strictly decreasing, false otherwise
 */
bool is_strictly_decreasing(const std::vector<double> &x) {
  if (x.size() <= 1) return true;  // Empty or single-element vector is trivially decreasing
  
  for (size_t i = 1; i < x.size(); i++) {
    if (x[i] >= x[i-1]) return false;
  }
  return true;
}

/**
 * Calculate binary entropy for a classification split
 * 
 * The entropy formula used is: 
 * E = -p_pos * log2(p_pos) - p_neg * log2(p_neg)
 * 
 * where p_pos and p_neg are proportions of positive and negative cases.
 * If either proportion is zero, that term is treated as zero (lim x→0+ x*log(x) = 0).
 * 
 * @param count_pos Count of positive cases
 * @param count_neg Count of negative cases
 * @return Entropy value in the range [0,1]
 */
double entropy(int count_pos, int count_neg) {
  int total = count_pos + count_neg;
  if (total == 0) return 0.0;  // Handle empty set case
  
  double p_pos = static_cast<double>(count_pos) / static_cast<double>(total);
  double p_neg = static_cast<double>(count_neg) / static_cast<double>(total);
  
  double E = 0.0;
  // Handle the limit case: lim x→0+ x*log(x) = 0
  if (p_pos > 0) E -= p_pos * std::log2(p_pos);
  if (p_neg > 0) E -= p_neg * std::log2(p_neg);
  
  return E;
}

/**
 * Calculate conditional entropy after a binary split
 * 
 * The conditional entropy is the weighted average of entropies in each partition:
 * E_cond = (n_left/n_total)*E_left + (n_right/n_total)*E_right
 * 
 * where n_left and n_right are the sizes of the left and right partitions,
 * n_total is the total size, and E_left and E_right are the entropies of 
 * the left and right partitions, respectively.
 * 
 * @param pos_left Positive cases in left partition
 * @param neg_left Negative cases in left partition
 * @param pos_right Positive cases in right partition
 * @param neg_right Negative cases in right partition
 * @return Conditional entropy value
 */
double conditional_entropy(int pos_left, int neg_left, int pos_right, int neg_right) {
  int total_left = pos_left + neg_left;
  int total_right = pos_right + neg_right;
  int total = total_left + total_right;
  
  if (total == 0) return 0.0;  // Handle empty set case
  
  double E_left = entropy(pos_left, neg_left);
  double E_right = entropy(pos_right, neg_right);
  
  double weight_left = static_cast<double>(total_left) / static_cast<double>(total);
  double weight_right = static_cast<double>(total_right) / static_cast<double>(total);
  
  double E_cond = weight_left * E_left + weight_right * E_right;
  
  return E_cond;
}

/**
 * MDLP stopping criterion (Fayyad & Irani, 1993)
 * 
 * The criterion determines whether a split is justified based on the
 * Minimum Description Length Principle. A split is accepted if the information
 * gain from the split exceeds a certain threshold determined by the MDL principle.
 * 
 * Specifically, a split is accepted if:
 * 
 * Gain(S,T;X) > (log2(N-1) + Delta) / N
 * 
 * where:
 * - Gain(S,T;X) is the information gain from the split
 * - N is the total number of instances
 * - Delta = log2(3^k - 2) - k*E(S) for k classes (k=2 for binary classification)
 * 
 * This criterion balances the reduction in entropy against the complexity of
 * adding a new decision boundary, following the principle of parsimony.
 * 
 * @param pos_left Positive cases in left partition
 * @param neg_left Negative cases in left partition
 * @param pos_right Positive cases in right partition
 * @param neg_right Negative cases in right partition
 * @param total_pos Total positive cases
 * @param total_neg Total negative cases
 * @return true if the criterion suggests stopping (not making the split), false if the split should be made
 */
bool mdlp_stop_criterion(int pos_left, int neg_left, int pos_right, int neg_right, int total_pos, int total_neg) {
  // Calculate entropies
  double E_parent = entropy(total_pos, total_neg);
  double E_child = conditional_entropy(pos_left, neg_left, pos_right, neg_right);
  
  // Information gain
  double IG = E_parent - E_child;
  
  // For binary classification (k=2):
  // Delta = log2(3^2 - 2) - 2*E_parent = log2(7) - 2*E_parent
  double Delta = std::log2(7.0) - 2.0 * E_parent;
  
  int N = total_pos + total_neg;
  
  // MDLP criterion threshold: only split if IG is greater than this threshold
  double threshold = (std::log2(static_cast<double>(N - 1)) / N) + (Delta / N);
  
  // Return true if we should stop (don't split), false if we should split
  return (IG <= threshold);
}

/**
 * Recursive implementation of the MDLP algorithm
 * 
 * Finds optimal cut points by recursively applying the MDLP criterion
 * to partitions of the data. For each partition, it finds the best cut point
 * that maximizes information gain, then tests if the split meets the MDLP criterion.
 * If it does, the split is made, and the algorithm recurses on the resulting sub-partitions.
 * 
 * @param target_sorted Sorted target variable (0/1)
 * @param feature_sorted Sorted feature variable
 * @param start Start index of current partition
 * @param end End index of current partition
 * @param prefix_pos Cumulative sum of positive cases
 * @param prefix_neg Cumulative sum of negative cases
 * @param splits Vector to store cut point indices
 */
void mdlp_recursion(
    const std::vector<int> &target_sorted,
    const std::vector<double> &feature_sorted,
    int start,
    int end,
    const std::vector<int> &prefix_pos,
    const std::vector<int> &prefix_neg,
    std::vector<int> &splits
) {
  // Check if partition is too small for further splitting
  if ((end - start) <= 1) {
    return;
  }
  
  // Get total counts for current partition
  int pos_total = prefix_pos[end] - prefix_pos[start];
  int neg_total = prefix_neg[end] - prefix_neg[start];
  
  // Pure partition (all positive or all negative) - no need to split
  if (pos_total == 0 || neg_total == 0) {
    return;
  }
  
  // Try to find the best cut point that maximizes information gain
  double best_IG = -std::numeric_limits<double>::infinity();
  int best_split = -1;
  
  double E_parent = entropy(pos_total, neg_total);
  
  // Search for best split
  for (int i = start; i < end - 1; i++) {
    // Only consider splits between different feature values
    if (feature_sorted[i] == feature_sorted[i+1]) continue;
    
    // Calculate counts for left and right partitions
    int pos_left = prefix_pos[i+1] - prefix_pos[start];
    int neg_left = prefix_neg[i+1] - prefix_neg[start];
    int pos_right = pos_total - pos_left;
    int neg_right = neg_total - neg_left;
    
    // Calculate conditional entropy and information gain
    double E_child = conditional_entropy(pos_left, neg_left, pos_right, neg_right);
    double IG = E_parent - E_child;
    
    // Update best split if current is better
    if (IG > best_IG) {
      best_IG = IG;
      best_split = i;
    }
  }
  
  // If no valid split found, return
  if (best_split == -1) {
    return;
  }
  
  // Calculate counts for best split
  int pos_left = prefix_pos[best_split+1] - prefix_pos[start];
  int neg_left = prefix_neg[best_split+1] - prefix_neg[start];
  int pos_right = pos_total - pos_left;
  int neg_right = neg_total - neg_left;
  
  // Check if the split passes the MDLP criterion
  if (mdlp_stop_criterion(pos_left, neg_left, pos_right, neg_right, pos_total, neg_total)) {
    // Criterion suggests to stop splitting
    return;
  }
  
  // Accept the split and add it to the result
  splits.push_back(best_split);
  
  // Recursively process the left partition
  mdlp_recursion(target_sorted, feature_sorted, start, best_split+1, prefix_pos, prefix_neg, splits);
  
  // Recursively process the right partition
  mdlp_recursion(target_sorted, feature_sorted, best_split+1, end, prefix_pos, prefix_neg, splits);
}

/**
 * Calculate metrics for bins resulting from the splits
 * 
 * Computes counts, Weight of Evidence (WoE), Information Value (IV), and cutpoints
 * for each bin defined by the split points. Applies Laplace smoothing to avoid extreme
 * WoE values when a bin contains only positive or only negative cases.
 * 
 * The Weight of Evidence for a bin is defined as:
 * WoE = ln(p_pos / p_neg)
 * 
 * where p_pos is the proportion of positive cases in the bin relative to all positive cases,
 * and p_neg is the proportion of negative cases in the bin relative to all negative cases.
 * 
 * The Information Value component for a bin is:
 * IV_bin = (p_pos - p_neg) * WoE
 * 
 * @param feature_sorted Sorted feature values
 * @param target_sorted Sorted target values
 * @param prefix_pos Cumulative sum of positive cases
 * @param prefix_neg Cumulative sum of negative cases
 * @param splits Vector of split indices
 * @param counts Output vector for bin counts
 * @param pos_counts Output vector for positive counts in each bin
 * @param neg_counts Output vector for negative counts in each bin
 * @param woe Output vector for Weight of Evidence values
 * @param iv Output vector for Information Value components
 * @param cutpoints Output vector for bin cutpoints
 */
void calc_bins_metrics(
    const std::vector<double> &feature_sorted,
    const std::vector<int> &target_sorted,
    const std::vector<int> &prefix_pos,
    const std::vector<int> &prefix_neg,
    const std::vector<int> &splits,
    std::vector<int> &counts,
    std::vector<int> &pos_counts,
    std::vector<int> &neg_counts,
    std::vector<double> &woe,
    std::vector<double> &iv,
    std::vector<double> &cutpoints
) {
  // Constants for Laplace smoothing
  const double ALPHA = 0.5;  // Smoothing parameter
  
  // Clear output vectors
  counts.clear();
  pos_counts.clear();
  neg_counts.clear();
  woe.clear();
  iv.clear();
  cutpoints.clear();
  
  int total_pos = prefix_pos.back();
  int total_neg = prefix_neg.back();
  
  // Create bin boundaries from splits
  std::vector<int> boundaries;
  boundaries.push_back(0);
  for (size_t i = 0; i < splits.size(); i++) {
    boundaries.push_back(splits[i] + 1);
  }
  boundaries.push_back(static_cast<int>(feature_sorted.size()));
  
  // Process each bin
  for (size_t i = 0; i < boundaries.size() - 1; i++) {
    int s = boundaries[i];
    int e = boundaries[i + 1];
    
    // Calculate raw counts
    int c_pos = prefix_pos[e] - prefix_pos[s];
    int c_neg = prefix_neg[e] - prefix_neg[s];
    int c_total = c_pos + c_neg;
    
    // Store counts
    counts.push_back(c_total);
    pos_counts.push_back(c_pos);
    neg_counts.push_back(c_neg);
    
    // Apply Laplace smoothing to proportions
    double smoothed_pos = static_cast<double>(c_pos + ALPHA);
    double smoothed_neg = static_cast<double>(c_neg + ALPHA);
    double smoothed_total_pos = static_cast<double>(total_pos + ALPHA * boundaries.size());
    double smoothed_total_neg = static_cast<double>(total_neg + ALPHA * boundaries.size());
    
    // Calculate proportions with smoothing
    double pct_pos = smoothed_pos / smoothed_total_pos;
    double pct_neg = smoothed_neg / smoothed_total_neg;
    
    // Calculate Weight of Evidence
    double w = std::log(pct_pos / pct_neg);
    woe.push_back(w);
    
    // Calculate Information Value component
    double iv_part = (pct_pos - pct_neg) * w;
    iv.push_back(iv_part);
    
    // Store cutpoint (except for the last bin)
    if (i < boundaries.size() - 2) {
      double cp = feature_sorted[e - 1];
      cutpoints.push_back(cp);
    }
  }
}

/**
 * Force a minimum number of bins using equal frequency binning
 * 
 * If MDLP produces fewer bins than min_bins, this function ensures
 * at least min_bins by creating approximately equal frequency bins
 * or by splitting existing bins.
 * 
 * @param feature_sorted Sorted feature values
 * @param target_sorted Sorted target values
 * @param min_bins Minimum number of bins
 * @param prefix_pos Cumulative sum of positive cases
 * @param prefix_neg Cumulative sum of negative cases
 * @param splits Vector of split indices (to be modified)
 */
void force_min_bins(
    const std::vector<double> &feature_sorted,
    const std::vector<int> &target_sorted,
    int min_bins,
    const std::vector<int> &prefix_pos,
    const std::vector<int> &prefix_neg,
    std::vector<int> &splits
) {
  // Check if we need to force min_bins
  int current_bins = static_cast<int>(splits.size()) + 1;
  if (current_bins >= min_bins) return;
  
  // Clear current splits
  splits.clear();
  
  // Get total size
  int N = static_cast<int>(feature_sorted.size());
  
  // First count unique values
  std::vector<double> unique_values;
  for (size_t i = 0; i < feature_sorted.size(); i++) {
    if (i == 0 || feature_sorted[i] != feature_sorted[i-1]) {
      unique_values.push_back(feature_sorted[i]);
    }
  }
  
  int n_unique = static_cast<int>(unique_values.size());
  
  // If there are fewer unique values than min_bins - 1 (which would result in min_bins),
  // we'll have to duplicate some splits to achieve the desired min_bins
  if (n_unique < min_bins) {
    // In this extreme case, we'll use equal frequency binning regardless of unique values
    int bin_size = N / min_bins;
    for (int i = 1; i < min_bins; i++) {
      int idx = i * bin_size - 1;
      if (idx >= 0 && idx < N - 1) {
        splits.push_back(idx);
      }
    }
  } else {
    // We have enough unique values, try to find optimal split points
    // Determine the step size in the unique values
    double step = static_cast<double>(n_unique - 1) / static_cast<double>(min_bins - 1);
    
    // Create splits at approximately equal intervals in the unique values
    for (int i = 1; i < min_bins; i++) {
      double pos = i * step;
      int unique_idx = static_cast<int>(std::floor(pos));
      
      if (unique_idx < n_unique - 1) {
        double value = unique_values[unique_idx];
        
        // Find the last occurrence of this value in feature_sorted
        int idx = 0;
        for (int j = 0; j < N; j++) {
          if (feature_sorted[j] <= value) {
            idx = j;
          } else {
            break;
          }
        }
        
        // Add this as a split point if valid
        if (idx > 0 && idx < N - 1) {
          splits.push_back(idx);
        }
      }
    }
  }
  
  // Ensure uniqueness and proper ordering of splits
  std::sort(splits.begin(), splits.end());
  splits.erase(std::unique(splits.begin(), splits.end()), splits.end());
  
  // If we still don't have enough splits, we'll add splits at equal frequency intervals
  while (splits.size() + 1 < static_cast<size_t>(min_bins)) {
    // Find the largest gap between consecutive splits
    int largest_gap = 0;
    int gap_start = 0;
    
    // Include the implicit boundaries at 0 and N-1
    std::vector<int> all_boundaries;
    all_boundaries.push_back(0);
    all_boundaries.insert(all_boundaries.end(), splits.begin(), splits.end());
    all_boundaries.push_back(N - 1);
    
    for (size_t i = 0; i < all_boundaries.size() - 1; i++) {
      int gap = all_boundaries[i+1] - all_boundaries[i];
      if (gap > largest_gap) {
        largest_gap = gap;
        gap_start = all_boundaries[i];
      }
    }
    
    // Add a split in the middle of the largest gap
    if (largest_gap > 1) {
      int new_split = gap_start + largest_gap / 2;
      if (new_split > 0 && new_split < N - 1) {
        splits.push_back(new_split);
        std::sort(splits.begin(), splits.end());
      } else {
        break; // Can't add more splits
      }
    } else {
      break; // Can't add more splits
    }
  }
}

/**
 * Enforce a maximum number of bins by merging adjacent bins
 * 
 * If the number of bins exceeds max_bins, this function reduces the number
 * of bins by merging adjacent bins. Ideally, bins with similar WoE values
 * would be merged, but this implementation uses a simple approach by removing
 * the last splits.
 * 
 * @param max_bins Maximum number of bins
 * @param feature_sorted Sorted feature values
 * @param target_sorted Sorted target values
 * @param prefix_pos Cumulative sum of positive cases
 * @param prefix_neg Cumulative sum of negative cases
 * @param splits Vector of split indices (to be modified)
 */
void enforce_max_bins(
    int max_bins,
    const std::vector<double> &feature_sorted,
    const std::vector<int> &target_sorted,
    const std::vector<int> &prefix_pos,
    const std::vector<int> &prefix_neg,
    std::vector<int> &splits
) {
  // Check if we need to enforce max_bins
  int current_bins = static_cast<int>(splits.size()) + 1;
  if (current_bins <= max_bins) return;
  
  // Sort splits to ensure they're in ascending order
  std::sort(splits.begin(), splits.end());
  
  // If we have too many bins, we need to merge some
  while (splits.size() + 1 > static_cast<size_t>(max_bins)) {
    // A more sophisticated approach would calculate WoE for adjacent bins and merge
    // those with the most similar values. However, for simplicity, we'll remove
    // the last split to reduce the bin count.
    if (!splits.empty()) {
      splits.pop_back();
    } else {
      // No more splits to remove
      break;
    }
  }
}

/**
 * Enforce monotonicity of Weight of Evidence across bins
 * 
 * Attempts to merge adjacent bins to achieve monotonicity in WoE values.
 * If force_monotonicity is true, this function will prioritize monotonicity
 * even if it means reducing the number of bins below min_bins (unless min_bins=2).
 * 
 * @param splits Vector of split indices (to be modified)
 * @param counts Bin counts (to be modified)
 * @param pos_counts Positive counts in each bin (to be modified)
 * @param neg_counts Negative counts in each bin (to be modified)
 * @param woe Weight of Evidence values (to be modified)
 * @param iv Information Value components (to be modified)
 * @param cutpoints Bin cutpoints (to be modified)
 * @param force_monotonicity Whether to force monotonicity
 * @param min_bins Minimum number of bins
 * @return true if monotonicity was achieved, false otherwise
 */
bool enforce_monotonicity(
    std::vector<int> &splits,
    std::vector<int> &counts,
    std::vector<int> &pos_counts,
    std::vector<int> &neg_counts,
    std::vector<double> &woe,
    std::vector<double> &iv,
    std::vector<double> &cutpoints,
    bool force_monotonicity,
    int min_bins
) {
  // Check if monotonicity enforcement is needed
  if (woe.size() <= 1) {
    // With 0 or 1 bin, monotonicity is trivially satisfied
    return true;
  }
  
  // Check if WoE values are already monotonic
  bool is_mono_inc = is_strictly_increasing(woe);
  bool is_mono_dec = is_strictly_decreasing(woe);
  
  if (is_mono_inc || is_mono_dec) {
    return true; // Already monotonic
  }
  
  // If not forcing monotonicity, we're done
  if (!force_monotonicity) {
    return true;
  }
  
  // Function to recompute metrics after merging bins
  auto recompute_metrics = [&](const std::vector<int> &boundaries) {
    // Clear current metrics
    std::vector<int> new_counts;
    std::vector<int> new_pos_counts;
    std::vector<int> new_neg_counts;
    std::vector<double> new_woe;
    std::vector<double> new_iv;
    std::vector<double> new_cutpoints;
    
    // Constants for Laplace smoothing
    const double ALPHA = 0.5;
    
    // Calculate total counts for normalization
    int total_pos = 0;
    int total_neg = 0;
    for (size_t i = 0; i < pos_counts.size(); i++) {
      total_pos += pos_counts[i];
      total_neg += neg_counts[i];
    }
    
    // Process each merged bin
    for (size_t i = 0; i < boundaries.size() - 1; i++) {
      int c_pos = 0;
      int c_neg = 0;
      
      // Sum counts across original bins that are now merged
      for (int b = boundaries[i]; b < boundaries[i + 1]; b++) {
        c_pos += pos_counts[b];
        c_neg += neg_counts[b];
      }
      
      int c_total = c_pos + c_neg;
      
      // Store counts
      new_counts.push_back(c_total);
      new_pos_counts.push_back(c_pos);
      new_neg_counts.push_back(c_neg);
      
      // Apply Laplace smoothing
      double smoothed_pos = static_cast<double>(c_pos + ALPHA);
      double smoothed_neg = static_cast<double>(c_neg + ALPHA);
      double smoothed_total_pos = static_cast<double>(total_pos + ALPHA * boundaries.size());
      double smoothed_total_neg = static_cast<double>(total_neg + ALPHA * boundaries.size());
      
      // Calculate proportions with smoothing
      double pct_pos = smoothed_pos / smoothed_total_pos;
      double pct_neg = smoothed_neg / smoothed_total_neg;
      
      // Calculate WoE
      double w = std::log(pct_pos / pct_neg);
      new_woe.push_back(w);
      
      // Calculate IV component
      double iv_part = (pct_pos - pct_neg) * w;
      new_iv.push_back(iv_part);
      
      // Store cutpoint (except for the last bin)
      if (i < boundaries.size() - 2) {
        // Get the cutpoint from the last bin in this group
        int last_bin_idx = boundaries[i + 1] - 1;
        if (last_bin_idx >= 0 && last_bin_idx < static_cast<int>(cutpoints.size())) {
          new_cutpoints.push_back(cutpoints[last_bin_idx]);
        } else if (!cutpoints.empty()) {
          // Fallback to the last available cutpoint
          new_cutpoints.push_back(cutpoints.back());
        }
      }
    }
    
    // Update the metrics
    counts = new_counts;
    pos_counts = new_pos_counts;
    neg_counts = new_neg_counts;
    woe = new_woe;
    iv = new_iv;
    cutpoints = new_cutpoints;
  };
  
  // Create a mapping of original bins (before merging)
  int n_bins = static_cast<int>(woe.size());
  std::vector<int> bin_map(n_bins);
  for (int i = 0; i < n_bins; i++) {
    bin_map[i] = i;
  }
  
  // Initialize boundaries for original bins
  std::vector<int> boundaries;
  for (int i = 0; i <= n_bins; i++) {
    boundaries.push_back(i);
  }
  
  // Iteratively merge bins until monotonicity is achieved or convergence criteria are met
  const int MAX_ITERATIONS = 1000;
  for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
    // Check if we've reached a single bin (trivially monotonic)
    if (woe.size() <= 1) {
      return true;
    }
    
    // Check if current WoE values are monotonic
    bool inc = is_strictly_increasing(woe);
    bool dec = is_strictly_decreasing(woe);
    
    if (inc || dec) {
      // Monotonicity achieved
      return true;
    }
    
    // Not monotonic, try to merge a pair of adjacent bins
    int current_bins = static_cast<int>(woe.size());
    
    // If we're down to min_bins, we can't merge further
    if (current_bins <= min_bins) {
      return true;
    }
    
    // Find the pair of adjacent bins with most similar WoE values
    double min_diff = std::numeric_limits<double>::infinity();
    int merge_pos = -1;
    
    for (int i = 0; i < current_bins - 1; i++) {
      double diff = std::fabs(woe[i + 1] - woe[i]);
      if (diff < min_diff) {
        min_diff = diff;
        merge_pos = i;
      }
    }
    
    // Merge bins at merge_pos and merge_pos+1
    if (merge_pos >= 0) {
      boundaries.erase(boundaries.begin() + merge_pos + 1);
      recompute_metrics(boundaries);
    } else {
      // No valid merge position found
      return false;
    }
    
    // Check if we've violated min_bins constraint
    if (static_cast<int>(woe.size()) < min_bins) {
      // We need at least min_bins bins and we've gone below that
      return true;
    }
  }
  
  // If we reach here, we didn't converge within MAX_ITERATIONS
  return false;
}

//' Optimal Binning for Numerical Variables using MDLP with Monotonicity
//'
//' This function implements optimal binning for numerical variables using the Minimum
//' Description Length Principle (MDLP) with optional monotonicity constraints on the
//' Weight of Evidence (WoE).
//'
//' The algorithm recursively partitions the feature space by finding cut points that 
//' maximize information gain, subject to the MDLP criterion that determines whether a cut is 
//' justified based on information theory principles. The monotonicity constraint ensures that
//' the WoE values across bins follow a monotonic (strictly increasing or decreasing) pattern,
//' which is often desirable in credit risk modeling applications.
//'
//' @param target Binary target variable (0/1)
//' @param feature Numerical feature to be binned
//' @param min_bins Minimum number of bins (default: 2)
//' @param max_bins Maximum number of bins (default: 5)
//' @param bin_cutoff Minimum relative frequency for a bin (not fully implemented, for future extensions)
//' @param max_n_prebins Maximum number of pre-bins (not fully implemented, for future extensions) 
//' @param convergence_threshold Convergence threshold for monotonicity enforcement
//' @param max_iterations Maximum number of iterations for monotonicity enforcement
//' @param force_monotonicity Whether to enforce monotonicity of Weight of Evidence
//'
//' @return A list containing:
//'   \item{id}{Bin identifiers}
//'   \item{bin}{Bin interval representations}
//'   \item{woe}{Weight of Evidence values for each bin}
//'   \item{iv}{Information Value components for each bin}
//'   \item{count}{Total count in each bin}
//'   \item{count_pos}{Positive count in each bin}
//'   \item{count_neg}{Negative count in each bin}
//'   \item{cutpoints}{Cut points between bins}
//'   \item{converged}{Whether the algorithm converged}
//'   \item{iterations}{Number of iterations performed}
//'
//' @examples
//' \dontrun{
//' # Generate sample data
//' set.seed(123)
//' feature <- rnorm(1000)
//' target <- as.integer(feature + rnorm(1000) > 0)
//'
//' # Apply optimal binning
//' result <- optimal_binning_numerical_fast_mdlpm(target, feature, min_bins = 3, max_bins = 5)
//' 
//' # Print results
//' print(result)
//' 
//' # Create WoE transformation
//' woe_transform <- function(x, bins, woe_values) {
//'   result <- rep(NA, length(x))
//'   for(i in seq_along(bins)) {
//'     idx <- eval(parse(text = paste0("x", bins[i])))
//'     result[idx] <- woe_values[i]
//'   }
//'   return(result)
//' }
//' }
//'
//' @references
//' Fayyad, U., & Irani, K. (1993). Multi-interval discretization of continuous-valued 
//' attributes for classification learning. Proceedings of the 13th International 
//' Joint Conference on Artificial Intelligence, 1022-1027.
//'
//' Kotsiantis, S., & Kanellopoulos, D. (2006). Discretization techniques: A recent 
//' survey. GESTS International Transactions on Computer Science and Engineering, 32(1), 47-58.
//' 
// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_fast_mdlpm(
   Rcpp::IntegerVector target,
   Rcpp::NumericVector feature,
   int min_bins = 2,
   int max_bins = 5,
   double bin_cutoff = 0.05,
   int max_n_prebins = 100,
   double convergence_threshold = 1e-6,
   int max_iterations = 1000,
   bool force_monotonicity = true
) {
 // Validate input parameters
 if (target.size() != feature.size()) {
   Rcpp::stop("Target and feature must have the same length.");
 }
 
 if (min_bins < 2) {
   Rcpp::stop("min_bins must be >= 2.");
 }
 
 if (max_bins < min_bins) {
   Rcpp::stop("max_bins must be >= min_bins.");
 }
 
 // Check if target is binary (0/1)
 bool is_binary = true;
 for (int i = 0; i < target.size(); i++) {
   if (!Rcpp::IntegerVector::is_na(target[i]) && target[i] != 0 && target[i] != 1) {
     is_binary = false;
     break;
   }
 }
 
 if (!is_binary) {
   Rcpp::warning("Target variable should be binary (0/1). Non-binary values detected.");
 }
 
 // Remove NA values from feature and target
 std::vector<double> feat_tmp;
 std::vector<int> targ_tmp;
 
 for (int i = 0; i < target.size(); i++) {
   if (!Rcpp::NumericVector::is_na(feature[i]) && !Rcpp::IntegerVector::is_na(target[i])) {
     feat_tmp.push_back(feature[i]);
     targ_tmp.push_back(target[i]);
   }
 }
 
 // Check if we have enough data after removing NAs
 int N = static_cast<int>(feat_tmp.size());
 if (N == 0) {
   Rcpp::warning("No valid data after removing NA values.");
   // Return empty result
   return Rcpp::List::create(
     Rcpp::Named("id") = Rcpp::NumericVector(),
     Rcpp::Named("bin") = Rcpp::CharacterVector(),
     Rcpp::Named("woe") = Rcpp::NumericVector(),
     Rcpp::Named("iv") = Rcpp::NumericVector(),
     Rcpp::Named("count") = Rcpp::IntegerVector(),
     Rcpp::Named("count_pos") = Rcpp::IntegerVector(),
     Rcpp::Named("count_neg") = Rcpp::IntegerVector(),
     Rcpp::Named("cutpoints") = Rcpp::NumericVector(),
     Rcpp::Named("converged") = false,
     Rcpp::Named("iterations") = 0
   );
 }
 
 // Sort data by feature value
 std::vector<int> idx(N);
 for (int i = 0; i < N; i++) {
   idx[i] = i;
 }
 
 std::sort(idx.begin(), idx.end(), [&](int a, int b) {
   return feat_tmp[a] < feat_tmp[b];
 });
 
 // Reorder feature and target vectors based on sorted indices
 std::vector<double> feature_sorted(N);
 std::vector<int> target_sorted(N);
 
 for (int i = 0; i < N; i++) {
   feature_sorted[i] = feat_tmp[idx[i]];
   target_sorted[i] = targ_tmp[idx[i]];
 }
 
 // Calculate cumulative sums (prefix sums) for efficient counting
 std::vector<int> prefix_pos(N + 1, 0);
 std::vector<int> prefix_neg(N + 1, 0);
 
 for (int i = 0; i < N; i++) {
   prefix_pos[i + 1] = prefix_pos[i] + (target_sorted[i] == 1 ? 1 : 0);
   prefix_neg[i + 1] = prefix_neg[i] + (target_sorted[i] == 0 ? 1 : 0);
 }
 
 // Check if all feature values are the same
 bool all_equal = true;
 for (int i = 1; i < N; i++) {
   if (feature_sorted[i] != feature_sorted[0]) {
     all_equal = false;
     break;
   }
 }
 
 // Handle the case where all feature values are the same
 if (all_equal) {
   Rcpp::warning("All feature values are identical. Creating artificial bins.");
   
   // We can only have one bin for identical values
   // If min_bins > 1, we'll divide evenly (though this doesn't make statistical sense)
   std::vector<int> splits;
   
   if (min_bins > 1) {
     for (int i = 1; i < min_bins; i++) {
       int idx_cut = static_cast<int>(floor(static_cast<double>(i) * N / static_cast<double>(min_bins)));
       if (idx_cut > 0 && idx_cut < N) {
         splits.push_back(idx_cut - 1);
       }
     }
   }
   
   // Calculate bin metrics
   std::vector<int> counts, pos_counts, neg_counts;
   std::vector<double> woe, iv, cutpoints;
   calc_bins_metrics(feature_sorted, target_sorted, prefix_pos, prefix_neg, splits,
                     counts, pos_counts, neg_counts, woe, iv, cutpoints);
   
   // Create bin names
   Rcpp::CharacterVector bin_names;
   {
     int nb = static_cast<int>(woe.size());
     double lower = -std::numeric_limits<double>::infinity();
     
     for (int b = 0; b < nb; b++) {
       double upper = (b < static_cast<int>(cutpoints.size())) 
       ? cutpoints[b] 
       : std::numeric_limits<double>::infinity();
       
       std::string interval = "(" 
       + (std::isinf(lower) ? "-Inf" : std::to_string(lower)) 
         + ";" 
         + (std::isinf(upper) ? "+Inf" : std::to_string(upper)) 
         + "]";
         bin_names.push_back(interval);
         lower = upper;
     }
   }
   
   // Calculate total IV
   double total_iv = 0.0;
   for (auto &x : iv) {
     total_iv += x;
   }
   
   // Create bin IDs
   Rcpp::NumericVector ids(bin_names.size());
   for (int i = 0; i < bin_names.size(); i++) {
     ids[i] = i + 1;
   }
   
   // Return result
   return Rcpp::List::create(
     Rcpp::Named("id") = ids,
     Rcpp::Named("bin") = bin_names,
     Rcpp::Named("woe") = Rcpp::wrap(woe),
     Rcpp::Named("iv") = Rcpp::wrap(iv),
     Rcpp::Named("count") = Rcpp::wrap(counts),
     Rcpp::Named("count_pos") = Rcpp::wrap(pos_counts),
     Rcpp::Named("count_neg") = Rcpp::wrap(neg_counts),
     Rcpp::Named("cutpoints") = Rcpp::wrap(cutpoints),
     Rcpp::Named("converged") = true,
     Rcpp::Named("iterations") = 0
   );
 }
 
 // Apply MDLP algorithm to find optimal cut points
 std::vector<int> splits;
 mdlp_recursion(target_sorted, feature_sorted, 0, N, prefix_pos, prefix_neg, splits);
 
 // Sort the splits in ascending order
 std::sort(splits.begin(), splits.end());
 
 // Enforce min_bins constraint
 {
   int current_bins = static_cast<int>(splits.size()) + 1;
   if (current_bins < min_bins) {
     force_min_bins(feature_sorted, target_sorted, min_bins, prefix_pos, prefix_neg, splits);
   }
 }
 
 // Enforce max_bins constraint
 {
   int current_bins = static_cast<int>(splits.size()) + 1;
   if (current_bins > max_bins) {
     enforce_max_bins(max_bins, feature_sorted, target_sorted, prefix_pos, prefix_neg, splits);
   }
 }
 
 // Calculate initial bin metrics
 std::vector<int> counts, pos_counts, neg_counts;
 std::vector<double> woe, iv, cutpoints;
 calc_bins_metrics(feature_sorted, target_sorted, prefix_pos, prefix_neg, splits,
                   counts, pos_counts, neg_counts, woe, iv, cutpoints);
 
 // Try to enforce monotonicity if requested
 bool converged = false;
 int iterations = 0;
 
 // Store previous WoE for convergence check
 std::vector<double> old_woe = woe;
 
 // Iteratively enforce monotonicity
 for (iterations = 0; iterations < max_iterations; iterations++) {
   bool mono_res = enforce_monotonicity(splits, counts, pos_counts, neg_counts, woe, iv, cutpoints, force_monotonicity, min_bins);
   
   // Calculate average absolute difference in WoE
   double diff = 0.0;
   {
     size_t len = std::min(old_woe.size(), woe.size());
     for (size_t i = 0; i < len; i++) {
       diff += std::fabs(old_woe[i] - woe[i]);
     }
     if (len > 0) {
       diff /= static_cast<double>(len);
     }
   }
   
   // Check for convergence
   if (mono_res && diff < convergence_threshold) {
     converged = true;
     break;
   }
   
   // Update old WoE for next iteration
   old_woe = woe;
 }
 
 // Check if min_bins is still respected after monotonicity enforcement
 int final_bins = static_cast<int>(woe.size());
 if (final_bins < min_bins) {
   // We need to increase the number of bins again
   // Rebuild splits from cutpoints
   splits.clear();
   for (size_t i = 0; i < feature_sorted.size() - 1; i++) {
     if (feature_sorted[i] != feature_sorted[i+1]) {
       splits.push_back(i);
       if (splits.size() >= static_cast<size_t>(min_bins - 1)) {
         break;
       }
     }
   }
   
   // If we still don't have enough splits, use force_min_bins
   if (splits.size() + 1 < static_cast<size_t>(min_bins)) {
     force_min_bins(feature_sorted, target_sorted, min_bins, prefix_pos, prefix_neg, splits);
   }
   
   // Recalculate metrics
   counts.clear();
   pos_counts.clear();
   neg_counts.clear();
   woe.clear();
   iv.clear();
   cutpoints.clear();
   
   calc_bins_metrics(feature_sorted, target_sorted, prefix_pos, prefix_neg, splits,
                     counts, pos_counts, neg_counts, woe, iv, cutpoints);
 }
 
 // Create bin names
 Rcpp::CharacterVector bin_names;
 {
   int nb = static_cast<int>(woe.size());
   double lower = -std::numeric_limits<double>::infinity();
   
   for (int b = 0; b < nb; b++) {
     double upper = (b < static_cast<int>(cutpoints.size())) 
     ? cutpoints[b] 
     : std::numeric_limits<double>::infinity();
     
     std::string interval = "(" 
     + (std::isinf(lower) ? "-Inf" : std::to_string(lower)) 
       + ";" 
       + (std::isinf(upper) ? "+Inf" : std::to_string(upper)) 
       + "]";
       bin_names.push_back(interval);
       lower = upper;
   }
 }
 
 // Calculate total Information Value
 double total_iv = 0.0;
 for (auto &x : iv) {
   total_iv += x;
 }
 
 // Create bin IDs
 Rcpp::NumericVector ids(bin_names.size());
 for (int i = 0; i < bin_names.size(); i++) {
   ids[i] = i + 1;
 }
 
 // Final verification of min_bins and max_bins constraints
 final_bins = static_cast<int>(woe.size());
 
 if (final_bins < min_bins || final_bins > max_bins) {
   Rcpp::warning("The algorithm failed to respect min_bins/max_bins constraints. Resulted in %d bins instead of [%d,%d].",
                 final_bins, min_bins, max_bins);
 }
 
 // Return result
 return Rcpp::List::create(
   Rcpp::Named("id") = ids,
   Rcpp::Named("bin") = bin_names,
   Rcpp::Named("woe") = Rcpp::wrap(woe),
   Rcpp::Named("iv") = Rcpp::wrap(iv),
   Rcpp::Named("count") = Rcpp::wrap(counts),
   Rcpp::Named("count_pos") = Rcpp::wrap(pos_counts),
   Rcpp::Named("count_neg") = Rcpp::wrap(neg_counts),
   Rcpp::Named("cutpoints") = Rcpp::wrap(cutpoints),
   Rcpp::Named("converged") = converged,
   Rcpp::Named("iterations") = iterations
 );
}
