// [[Rcpp::depends(Rcpp)]]
#include <Rcpp.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <numeric> // Required for std::accumulate

using namespace Rcpp;

/**
 * Core class implementing Optimal Binning for categorical variables using various Divergence Measures. (Version 2)
 * Based on the theoretical framework from Zeng (2013) "Metric Divergence Measures and Information Value in Credit Scoring".
 * V2 Corrections:
 *  - Fixed crash potential in similarity matrix update after merging.
 *  - Optimized similarity matrix update after splitting in ensure_min_bins.
 *  - Corrected calculation and reporting of L2/L-infinity divergence.
 *  - Implemented max_n_prebins logic for handling very high cardinality features.
 *  - Added const correctness and improved comments.
 */
class OptimalBinningCategoricalDMIV_v2 {
private:
  // Input parameters (made const where appropriate)
  const std::vector<std::string>& feature;
  const std::vector<int>& target;
  int min_bins;
  int max_bins;
  const double bin_cutoff; // Frequency threshold
  const int max_n_prebins; // Max initial bins before merging rare ones
  const std::string bin_separator;
  const double convergence_threshold;
  const int max_iterations;
  const std::string bin_method;      // 'woe' or 'woe1'
  const std::string divergence_method; // 'he', 'kl', 'tr', 'klj', 'sc', 'js', 'l1', 'l2', 'ln'
  const int min_prebin_count = 5; // Min count threshold for pre-binning rare categories if max_n_prebins is exceeded
  
  // Small constant to handle numerical issues
  static constexpr double EPSILON = 1e-10;
  
  /**
   * Structure representing a bin of categorical values
   */
  struct Bin {
    std::vector<std::string> categories; // Categories in this bin
    int count_pos;                       // Count of positive class
    int count_neg;                       // Count of negative class
    double woe;                          // Weight of Evidence
    double divergence;                   // Divergence measure value (or intermediate for L2/L-inf)
    
    // Constructor with sensible initial capacity
    Bin() : count_pos(0), count_neg(0), woe(0.0), divergence(0.0) {
      // Reserve capacity dynamically based on estimated average categories per bin
      // categories.reserve(8); // Initial reservation can be done later if needed
    }
    
    int total_count() const { return count_pos + count_neg; }
  };
  
  // Internal state
  std::vector<Bin> bins;
  int total_pos = 0; // Initialize here
  int total_neg = 0; // Initialize here
  std::unordered_map<std::string, int> count_pos_map;
  std::unordered_map<std::string, int> count_neg_map;
  std::unordered_map<std::string, int> total_count_map;
  int initial_unique_categories = 0; // Before pre-binning
  bool converged = false;            // Initialize here
  int iterations_run = 0;            // Initialize here
  
  // Similarity/distance matrix (lower is more similar for divergence)
  std::vector<std::vector<double>> distance_matrix; // Renamed for clarity (stores divergence)
  
  
  // --- Private Methods ---
  
  /**
   * Validate input arguments for correctness and consistency
   * Throws std::invalid_argument if validation fails
   */
  void validate_inputs() {
    // Basic validation
    if (feature.empty() || target.empty()) {
      throw std::invalid_argument("Feature and target cannot be empty.");
    }
    if (feature.size() != target.size()) {
      throw std::invalid_argument("Feature and target must have the same length.");
    }
    if (min_bins <= 0 || max_bins <= 0 || min_bins > max_bins) {
      throw std::invalid_argument("Invalid values for min_bins or max_bins (must be > 0 and min_bins <= max_bins).");
    }
    if (bin_cutoff <= 0 || bin_cutoff >= 1) {
      throw std::invalid_argument("bin_cutoff must be between 0 and 1 (exclusive).");
    }
    if (max_n_prebins < 2) {
      // Need at least 2 bins potentially to merge
      throw std::invalid_argument("max_n_prebins must be at least 2.");
    }
    if (convergence_threshold <= 0) {
      throw std::invalid_argument("convergence_threshold must be positive.");
    }
    if (max_iterations <= 0) {
      throw std::invalid_argument("max_iterations must be positive.");
    }
    
    // Validate bin_method and divergence_method
    if (bin_method != "woe" && bin_method != "woe1") {
      throw std::invalid_argument("bin_method must be either 'woe' or 'woe1'.");
    }
    
    const std::unordered_set<std::string> valid_divergence_methods = {
      "he", "kl", "tr", "klj", "sc", "js", "l1", "l2", "ln"
    };
    if (valid_divergence_methods.find(divergence_method) == valid_divergence_methods.end()) {
      throw std::invalid_argument("Invalid divergence_method. Must be one of: 'he', 'kl', 'tr', 'klj', 'sc', 'js', 'l1', 'l2', 'ln'.");
    }
    
    // Efficiently process data in a single pass to get counts
    int total_count = target.size();
    std::unordered_map<std::string, std::pair<int, int>> counts;
    counts.reserve(std::min(static_cast<size_t>(total_count), static_cast<size_t>(max_n_prebins) + 100)); // Heuristic reservation
    
    for (int i = 0; i < total_count; ++i) {
      const int t = target[i];
      if (t != 0 && t != 1) {
        throw std::invalid_argument("Target must be binary (0 or 1).");
      }
      
      const std::string& cat = feature[i]; // Assume "NA" is already handled by wrapper
      auto& count_pair = counts[cat]; // Creates if not exists
      
      if (t == 1) {
        count_pair.first++;
        total_pos++;
      } else {
        count_pair.second++;
        total_neg++;
      }
    }
    
    if (total_pos == 0 || total_neg == 0) {
      throw std::invalid_argument("Target must contain both 0 and 1 values.");
    }
    
    // Transfer counts to final maps
    count_pos_map.reserve(counts.size());
    count_neg_map.reserve(counts.size());
    total_count_map.reserve(counts.size());
    for (const auto& item : counts) {
      const std::string& cat = item.first;
      const auto& count_pair = item.second;
      
      count_pos_map[cat] = count_pair.first;
      count_neg_map[cat] = count_pair.second;
      total_count_map[cat] = count_pair.first + count_pair.second;
    }
    
    initial_unique_categories = static_cast<int>(counts.size());
    
    // Adjust bin constraints based on available unique categories *after* potential pre-binning
    // This adjustment will happen after initialize_bins if pre-binning occurs.
    if (initial_unique_categories < 2) {
      throw std::invalid_argument("Feature must have at least 2 unique categories.");
    }
  }
  
  /**
   * Initialize bins, potentially pre-binning rare categories if cardinality is high.
   */
  void initialize_bins() {
    bins.clear();
    
    // Pre-binning logic if unique categories exceed max_n_prebins
    if (initial_unique_categories > max_n_prebins) {
      Rcpp::Rcout << "Info: Number of unique categories (" << initial_unique_categories
                  << ") exceeds max_n_prebins (" << max_n_prebins
                  << "). Pre-binning rare categories." << std::endl;
      
      bins.reserve(max_n_prebins); // Approximate final size
      Bin other_bin;
      other_bin.categories.push_back("PREBIN_OTHER"); // Special name
      
      for (const auto& item : total_count_map) {
        const std::string& cat = item.first;
        int cat_total = item.second;
        
        // Keep categories if count is >= min_prebin_count, otherwise add to 'other' bin
        if (cat_total >= min_prebin_count) {
          Bin bin;
          bin.categories.push_back(cat);
          bin.count_pos = count_pos_map[cat];
          bin.count_neg = count_neg_map[cat];
          bins.push_back(std::move(bin));
        } else {
          other_bin.count_pos += count_pos_map[cat];
          other_bin.count_neg += count_neg_map[cat];
          // We don't store individual rare categories in 'other_bin.categories'
          // to avoid excessive memory use if many are rare.
        }
      }
      // Add the 'other' bin if it collected any categories
      if (other_bin.total_count() > 0) {
        bins.push_back(std::move(other_bin));
      }
      
      // Check if pre-binning resulted in too few bins
      if (bins.size() < 2) {
        // Fallback: ignore pre-binning if it collapses everything
        Rcpp::Rcout << "Warning: Pre-binning resulted in < 2 bins. Reverting to initial categories." << std::endl;
        bins.clear();
        // Proceed with normal initialization below
      } else {
        Rcpp::Rcout << "Info: Pre-binning reduced categories from " << initial_unique_categories
                    << " to " << bins.size() << " initial bins." << std::endl;
      }
    }
    
    // Normal initialization if no pre-binning occurred or if fallback triggered
    if (bins.empty()) {
      bins.reserve(initial_unique_categories);
      for (const auto& item : total_count_map) {
        Bin bin;
        bin.categories.push_back(item.first);
        bin.count_pos = count_pos_map[item.first];
        bin.count_neg = count_neg_map[item.first];
        bins.push_back(std::move(bin));
      }
    }
    
    // Adjust bin constraints based on the actual number of initial bins
    int current_bins = bins.size();
    min_bins = std::max(2, std::min(min_bins, current_bins));
    max_bins = std::min(max_bins, current_bins);
    if (min_bins > max_bins) {
      min_bins = max_bins; // Ensure min <= max
    }
    
    // Compute initial metrics and sort (optional but can help)
    compute_bin_metrics();
    sort_bins_by_woe();
  }
  
  
  /**
   * Sort bins by Weight of Evidence (ascending).
   */
  void sort_bins_by_woe() {
    std::sort(bins.begin(), bins.end(),
              [](const Bin& a, const Bin& b) { return a.woe < b.woe; });
    // NOTE: Sorting invalidates the distance matrix if done mid-process.
    // It's mainly for final output or potentially better initial state.
    // We recompute/update the distance matrix after merges/splits anyway.
  }
  
  
  /**
   * Compute divergence between two bins based on their pos/neg distributions.
   * Lower values indicate more similar bins (less divergence).
   * @param bin1 First bin.
   * @param bin2 Second bin.
   * @return Divergence score.
   */
  double compute_bin_divergence(const Bin& bin1, const Bin& bin2) const {
    double total1 = static_cast<double>(bin1.total_count());
    double total2 = static_cast<double>(bin2.total_count());
    
    // Handle empty bins (shouldn't happen in main loop but maybe in rare category handling)
    if (total1 < EPSILON || total2 < EPSILON) return std::numeric_limits<double>::max(); // Max divergence
    
    // Use overall totals for distribution calculation, consistent with compute_bin_metrics
    double dist1_pos = static_cast<double>(bin1.count_pos) / static_cast<double>(total_pos);
    double dist1_neg = static_cast<double>(bin1.count_neg) / static_cast<double>(total_neg);
    double dist2_pos = static_cast<double>(bin2.count_pos) / static_cast<double>(total_pos);
    double dist2_neg = static_cast<double>(bin2.count_neg) / static_cast<double>(total_neg);
    
    // Apply epsilon smoothing *only when needed* (for log or division)
    double divergence = 0.0;
    
    if (divergence_method == "he") {
      // Hellinger Distance (already a metric, >= 0)
      divergence = std::pow(std::sqrt(std::max(dist1_pos, 0.0)) - std::sqrt(std::max(dist2_pos, 0.0)), 2) +
        std::pow(std::sqrt(std::max(dist1_neg, 0.0)) - std::sqrt(std::max(dist2_neg, 0.0)), 2);
    } else if (divergence_method == "kl") {
      // Symmetrized KL divergence (>= 0)
      double kl12 = (dist1_pos > EPSILON ? dist1_pos * std::log(dist1_pos / std::max(dist2_pos, EPSILON)) : 0.0) +
        (dist1_neg > EPSILON ? dist1_neg * std::log(dist1_neg / std::max(dist2_neg, EPSILON)) : 0.0);
      double kl21 = (dist2_pos > EPSILON ? dist2_pos * std::log(dist2_pos / std::max(dist1_pos, EPSILON)) : 0.0) +
        (dist2_neg > EPSILON ? dist2_neg * std::log(dist2_neg / std::max(dist1_neg, EPSILON)) : 0.0);
      divergence = kl12 + kl21;
    } else if (divergence_method == "klj") {
      // J-Divergence (same as symmetrized KL)
      double kl12 = (dist1_pos > EPSILON ? dist1_pos * std::log(dist1_pos / std::max(dist2_pos, EPSILON)) : 0.0) +
        (dist1_neg > EPSILON ? dist1_neg * std::log(dist1_neg / std::max(dist2_neg, EPSILON)) : 0.0);
      double kl21 = (dist2_pos > EPSILON ? dist2_pos * std::log(dist2_pos / std::max(dist1_pos, EPSILON)) : 0.0) +
        (dist2_neg > EPSILON ? dist2_neg * std::log(dist2_neg / std::max(dist1_neg, EPSILON)) : 0.0);
      divergence = kl12 + kl21;
      // Original Zeng formula: (p1-p2)log(p1/p2) + (n1-n2)log(n1/n2) - seems different? Let's stick to symmetric KL definition based on context.
      // If using Zeng's literal formula:
      // divergence = (dist1_pos - dist2_pos) * (dist1_pos > EPSILON && dist2_pos > EPSILON ? std::log(dist1_pos / dist2_pos) : 0.0) +
      //             (dist1_neg - dist2_neg) * (dist1_neg > EPSILON && dist2_neg > EPSILON ? std::log(dist1_neg / dist2_neg) : 0.0);
    } else if (divergence_method == "tr") {
      // Triangular Discrimination (>= 0)
      divergence = (total1 > EPSILON && total2 > EPSILON) ?
      (std::pow(dist1_pos - dist2_pos, 2) / std::max(dist1_pos + dist2_pos, EPSILON)) +
      (std::pow(dist1_neg - dist2_neg, 2) / std::max(dist1_neg + dist2_neg, EPSILON))
        : std::numeric_limits<double>::max(); // Avoid division by zero if sums are zero
    } else if (divergence_method == "sc") {
      // Chi-Square Symmetric (>= 0)
      divergence = (dist1_pos > EPSILON && dist2_pos > EPSILON ?
                      std::pow(dist1_pos - dist2_pos, 2) * (dist1_pos + dist2_pos) / (dist1_pos * dist2_pos) : 0.0) +
                      (dist1_neg > EPSILON && dist2_neg > EPSILON ?
                      std::pow(dist1_neg - dist2_neg, 2) * (dist1_neg + dist2_neg) / (dist1_neg * dist2_neg) : 0.0);
      // Add large penalty if any denominator is zero but numerator isn't
      if (((dist1_pos < EPSILON || dist2_pos < EPSILON) && std::abs(dist1_pos - dist2_pos) > EPSILON) ||
          ((dist1_neg < EPSILON || dist2_neg < EPSILON) && std::abs(dist1_neg - dist2_neg) > EPSILON)) {
        divergence = std::numeric_limits<double>::max();
      }
      
    } else if (divergence_method == "js") {
      // Jensen-Shannon Divergence (>= 0)
      double m_pos = (dist1_pos + dist2_pos) / 2.0;
      double m_neg = (dist1_neg + dist2_neg) / 2.0;
      double js1 = (dist1_pos > EPSILON ? dist1_pos * std::log(dist1_pos / std::max(m_pos, EPSILON)) : 0.0) +
        (dist1_neg > EPSILON ? dist1_neg * std::log(dist1_neg / std::max(m_neg, EPSILON)) : 0.0);
      double js2 = (dist2_pos > EPSILON ? dist2_pos * std::log(dist2_pos / std::max(m_pos, EPSILON)) : 0.0) +
        (dist2_neg > EPSILON ? dist2_neg * std::log(dist2_neg / std::max(m_neg, EPSILON)) : 0.0);
      divergence = 0.5 * (js1 + js2);
    } else if (divergence_method == "l1") {
      // L1 metric (Manhattan) (>= 0) - uses local proportions
      double local_dist1_pos = (total1 > EPSILON) ? static_cast<double>(bin1.count_pos) / total1 : 0.0;
      double local_dist1_neg = (total1 > EPSILON) ? static_cast<double>(bin1.count_neg) / total1 : 0.0;
      double local_dist2_pos = (total2 > EPSILON) ? static_cast<double>(bin2.count_pos) / total2 : 0.0;
      double local_dist2_neg = (total2 > EPSILON) ? static_cast<double>(bin2.count_neg) / total2 : 0.0;
      divergence = std::abs(local_dist1_pos - local_dist2_pos) + std::abs(local_dist1_neg - local_dist2_neg);
    } else if (divergence_method == "l2") {
      // L2 metric (Euclidean) (>= 0) - uses local proportions
      double local_dist1_pos = (total1 > EPSILON) ? static_cast<double>(bin1.count_pos) / total1 : 0.0;
      double local_dist1_neg = (total1 > EPSILON) ? static_cast<double>(bin1.count_neg) / total1 : 0.0;
      double local_dist2_pos = (total2 > EPSILON) ? static_cast<double>(bin2.count_pos) / total2 : 0.0;
      double local_dist2_neg = (total2 > EPSILON) ? static_cast<double>(bin2.count_neg) / total2 : 0.0;
      divergence = std::sqrt(std::pow(local_dist1_pos - local_dist2_pos, 2) + std::pow(local_dist1_neg - local_dist2_neg, 2));
    } else if (divergence_method == "ln") {
      // L∞ metric (Maximum) (>= 0) - uses local proportions
      double local_dist1_pos = (total1 > EPSILON) ? static_cast<double>(bin1.count_pos) / total1 : 0.0;
      double local_dist1_neg = (total1 > EPSILON) ? static_cast<double>(bin1.count_neg) / total1 : 0.0;
      double local_dist2_pos = (total2 > EPSILON) ? static_cast<double>(bin2.count_pos) / total2 : 0.0;
      double local_dist2_neg = (total2 > EPSILON) ? static_cast<double>(bin2.count_neg) / total2 : 0.0;
      divergence = std::max(std::abs(local_dist1_pos - local_dist2_pos), std::abs(local_dist1_neg - local_dist2_neg));
    }
    
    // Ensure divergence is not negative due to potential floating point issues
    return std::max(0.0, divergence);
  }
  
  
  /**
   * Initialize distance matrix (storing divergence) for all bin pairs.
   */
  void initialize_distance_matrix() {
    const size_t n = bins.size();
    if (n == 0) return; // Handle empty bins case
    
    distance_matrix.assign(n, std::vector<double>(n, std::numeric_limits<double>::max())); // Initialize with max divergence
    
    for (size_t i = 0; i < n; ++i) {
      distance_matrix[i][i] = std::numeric_limits<double>::max(); // Ignore self-distance for finding minimum merge pair
      // Compute upper triangular part
      for (size_t j = i + 1; j < n; ++j) {
        distance_matrix[i][j] = compute_bin_divergence(bins[i], bins[j]);
        distance_matrix[j][i] = distance_matrix[i][j]; // Symmetric
      }
    }
  }
  
  /**
   * Update distance matrix after merging bin `removed_index` into `merged_index`.
   * Handles index shifts correctly.
   * @param merged_index_orig Original index of the merged bin BEFORE removal.
   * @param removed_index Original index of the removed bin BEFORE removal.
   */
  void update_distance_matrix_after_merge(size_t merged_index_orig, size_t removed_index) {
    const size_t n_old = distance_matrix.size();
    if (n_old <= 1 || removed_index >= n_old || merged_index_orig >= n_old) {
      // Should not happen if called correctly
      return;
    }
    
    // --- Step 1: Remove row and column for removed_index ---
    distance_matrix.erase(distance_matrix.begin() + removed_index);
    for (auto& row : distance_matrix) {
      row.erase(row.begin() + removed_index);
    }
    
    // --- Step 2: Determine the *new* index of the merged bin ---
    // If the merged bin was after the removed one, its index shifted down by 1.
    size_t merged_index_new = (merged_index_orig > removed_index) ? merged_index_orig - 1 : merged_index_orig;
    
    // --- Step 3: Update distances involving the merged bin ---
    const size_t n_new = bins.size(); // Should equal distance_matrix.size()
    if (merged_index_new >= n_new) return; // Safety check
    
    for (size_t i = 0; i < n_new; ++i) {
      if (i == merged_index_new) continue;
      
      // Recompute distance between the newly merged bin and all others
      double divergence = compute_bin_divergence(bins[merged_index_new], bins[i]);
      distance_matrix[merged_index_new][i] = divergence;
      distance_matrix[i][merged_index_new] = divergence;
    }
    // Ensure self-distance remains max
    distance_matrix[merged_index_new][merged_index_new] = std::numeric_limits<double>::max();
  }
  
  /**
   * Update distance matrix after splitting bin at `split_index_orig` into two new bins
   * (assumed to be added at the end of the `bins` vector).
   * This is the optimized approach for ensure_min_bins.
   * @param split_index_orig Original index of the bin that was split BEFORE removal.
   */
  void update_distance_matrix_after_split(size_t split_index_orig) {
    const size_t n_old = distance_matrix.size();
    if (n_old == 0 || split_index_orig >= n_old) {
      return; // Should not happen
    }
    
    // --- Step 1: Remove row and column for the split bin ---
    distance_matrix.erase(distance_matrix.begin() + split_index_orig);
    for (auto& row : distance_matrix) {
      row.erase(row.begin() + split_index_orig);
    }
    
    // --- Step 2: Add two new rows and columns for the new bins ---
    // New bins are assumed to be at indices n_new-2 and n_new-1
    const size_t n_new = bins.size(); // Size after split (n_old - 1 + 2)
    if (n_new != n_old + 1) {
      // Logic error if sizes don't match expected change
      throw std::logic_error("Bin size mismatch after split in distance matrix update.");
    }
    
    // Resize existing rows to accommodate new columns, init with max divergence
    for (auto& row : distance_matrix) {
      row.resize(n_new, std::numeric_limits<double>::max());
    }
    // Add two new rows, init with max divergence
    distance_matrix.resize(n_new, std::vector<double>(n_new, std::numeric_limits<double>::max()));
    
    // --- Step 3: Calculate distances for the two new bins ---
    size_t new_bin1_idx = n_new - 2;
    size_t new_bin2_idx = n_new - 1;
    
    for(size_t i = 0; i < n_new - 2; ++i) { // Compare against original bins (excluding the split one)
      // Distance between bin i and new_bin1
      double div1 = compute_bin_divergence(bins[i], bins[new_bin1_idx]);
      distance_matrix[i][new_bin1_idx] = div1;
      distance_matrix[new_bin1_idx][i] = div1;
      
      // Distance between bin i and new_bin2
      double div2 = compute_bin_divergence(bins[i], bins[new_bin2_idx]);
      distance_matrix[i][new_bin2_idx] = div2;
      distance_matrix[new_bin2_idx][i] = div2;
    }
    
    // Distance between the two new bins themselves
    double div_new1_new2 = compute_bin_divergence(bins[new_bin1_idx], bins[new_bin2_idx]);
    distance_matrix[new_bin1_idx][new_bin2_idx] = div_new1_new2;
    distance_matrix[new_bin2_idx][new_bin1_idx] = div_new1_new2;
    
    // Ensure self-distances remain max
    distance_matrix[new_bin1_idx][new_bin1_idx] = std::numeric_limits<double>::max();
    distance_matrix[new_bin2_idx][new_bin2_idx] = std::numeric_limits<double>::max();
  }
  
  
  /**
   * Perform optimal binning using divergence measures via hierarchical merging.
   */
  void perform_binning() {
    iterations_run = 0;
    converged = false;
    
    if (bins.size() <= static_cast<size_t>(max_bins)) {
      Rcpp::Rcout << "Info: Initial number of bins (" << bins.size()
                  << ") is already <= max_bins (" << max_bins
                  << "). Skipping merging phase." << std::endl;
      // Still need to check min_bins later
      return;
    }
    
    // Initialize distance matrix (stores divergence, lower is better for merging)
    initialize_distance_matrix();
    
    double previous_min_divergence = -1.0; // Initialize
    
    // Main optimization loop: Merge until max_bins is reached or convergence
    while (bins.size() > static_cast<size_t>(max_bins) &&
           iterations_run < max_iterations &&
           bins.size() > 1) // Need at least 2 bins to merge
    {
      
      // Find pair of bins with minimum divergence (maximum similarity)
      std::pair<double, std::pair<size_t, size_t>> best_merge = find_most_similar_bins();
      double current_min_divergence = best_merge.first;
      size_t bin1_idx = best_merge.second.first; // Index of bin to keep/merge into
      size_t bin2_idx = best_merge.second.second; // Index of bin to remove
      
      // Ensure bin1_idx < bin2_idx for consistent merging/removal if needed, although logic handles it
      if (bin1_idx > bin2_idx) std::swap(bin1_idx, bin2_idx);
      
      // Check convergence based on small absolute change in min divergence
      if (previous_min_divergence >= 0 && // Check after first iteration
          std::fabs(current_min_divergence - previous_min_divergence) < convergence_threshold) {
        converged = true;
        Rcpp::Rcout << "Info: Converged after " << iterations_run << " iterations (divergence change < threshold)." << std::endl;
        break;
      }
      // Check convergence if min divergence becomes excessively large (no good merges left)
      // Using a threshold relative to initial divergences might be better, but absolute check is simpler
      // double avg_initial_divergence = ... // compute average non-infinite divergence
      // if (current_min_divergence > some_factor * avg_initial_divergence) break;
      // For now, rely on iteration limit or change threshold.
      
      
      // Merge bins with lowest divergence
      merge_two_bins(bins[bin1_idx], bins[bin2_idx]);
      
      // Remove the second bin (at original index bin2_idx)
      bins.erase(bins.begin() + bin2_idx);
      
      // Update metrics for the merged bin (others unchanged)
      compute_single_bin_metrics(bins[bin1_idx]); // More efficient than recomputing all
      
      // Update distance matrix (handles index shifts)
      update_distance_matrix_after_merge(bin1_idx, bin2_idx); // Pass original indices
      
      previous_min_divergence = current_min_divergence;
      iterations_run++;
    }
    
    // Final convergence status
    if (!converged && iterations_run == max_iterations) {
      Rcpp::Rcout << "Warning: Reached max_iterations (" << max_iterations << ") without converging." << std::endl;
    } else if (!converged) {
      // Stopped because bins.size() <= max_bins
      converged = true; // Consider reaching max_bins as a form of convergence
    }
  }
  
  
  /**
   * Find the pair of adjacent bins with the minimum divergence (best merge candidate).
   * @return pair<divergence, pair<index1, index2>>. Returns max divergence if no pair found.
   */
  std::pair<double, std::pair<size_t, size_t>> find_most_similar_bins() const {
    double min_divergence = std::numeric_limits<double>::max();
    std::pair<size_t, size_t> best_pair = {0, 0}; // Default initialization
    
    const size_t n = bins.size();
    if (n < 2) {
      return {min_divergence, best_pair}; // Cannot merge if less than 2 bins
    }
    
    for (size_t i = 0; i < n; ++i) {
      // Only check upper triangle (j > i)
      for (size_t j = i + 1; j < n; ++j) {
        if (distance_matrix[i][j] < min_divergence) {
          min_divergence = distance_matrix[i][j];
          best_pair = {i, j};
        }
      }
    }
    
    return {min_divergence, best_pair};
  }
  
  
  /**
   * Ensure minimum number of bins by splitting the most heterogeneous bins if necessary.
   */
  void ensure_min_bins() {
    if (bins.size() >= static_cast<size_t>(min_bins)) {
      return;
    }
    Rcpp::Rcout << "Info: Current bins (" << bins.size() << ") < min_bins (" << min_bins
                << "). Attempting to split bins." << std::endl;
    
    
    // Re-initialize distance matrix as splitting changes relationships significantly
    // While inefficient, it's simpler than complex updates after splits
    // V2 Optimization: Use incremental update instead.
    // initialize_distance_matrix(); // Inefficient - Replaced by incremental update below
    
    
    // Continue splitting bins until we reach min_bins or cannot split further
    while (bins.size() < static_cast<size_t>(min_bins)) {
      // Find bin with the most categories that can be split
      int best_split_idx = -1;
      size_t max_cats = 0;
      double max_heterogeneity = -1.0; // Use heterogeneity as tie-breaker
      
      for(size_t i = 0; i < bins.size(); ++i) {
        if (bins[i].categories.size() > 1) { // Can only split if > 1 category
          if (bins[i].categories.size() > max_cats) {
            max_cats = bins[i].categories.size();
            max_heterogeneity = calculate_bin_heterogeneity(bins[i]);
            best_split_idx = static_cast<int>(i);
          } else if (bins[i].categories.size() == max_cats) {
            // Tie-breaker: choose bin with higher internal WoE variance
            double current_heterogeneity = calculate_bin_heterogeneity(bins[i]);
            if (current_heterogeneity > max_heterogeneity) {
              max_heterogeneity = current_heterogeneity;
              best_split_idx = static_cast<int>(i);
            }
          }
        }
      }
      
      
      // If no bin can be split, we stop
      if (best_split_idx == -1) {
        Rcpp::Rcout << "Warning: Cannot split further to reach min_bins. Final number of bins: " << bins.size() << std::endl;
        break;
      }
      
      // Store the bin to be split and its original index
      Bin bin_to_split = std::move(bins[best_split_idx]);
      size_t original_split_index = static_cast<size_t>(best_split_idx);
      
      
      // --- Perform the split ---
      // 1. Remove original bin from the vector
      bins.erase(bins.begin() + original_split_index);
      
      // 2. Create and add the two new bins resulting from the split
      split_bin_into_two(bin_to_split); // Adds new bins to the end of `bins` vector
      
      // 3. Update metrics for the two new bins
      compute_single_bin_metrics(bins[bins.size() - 2]); // New bin 1
      compute_single_bin_metrics(bins[bins.size() - 1]); // New bin 2
      
      // 4. Update distance matrix incrementally (Optimized V2 approach)
      update_distance_matrix_after_split(original_split_index);
      
    }
  }
  
  
  /**
   * Calculate heterogeneity (variance of individual category WoEs) within a bin.
   * @param bin The bin to evaluate.
   * @return Heterogeneity score (>= 0). Returns 0 if <= 1 category.
   */
  double calculate_bin_heterogeneity(const Bin& bin) const {
    const size_t n_cats = bin.categories.size();
    if (n_cats <= 1) return 0.0;
    
    std::vector<double> woe_values;
    woe_values.reserve(n_cats);
    
    for (const auto& category : bin.categories) {
      // Handle potential pre-binned 'other' category - can't calculate WoE for it directly
      if (category == "PREBIN_OTHER") {
        // Assign average WoE or skip? Skipping seems safer as its internal composition is unknown.
        // Alternative: Use the WoE of the 'other' bin itself?
        // For simplicity, let's use the overall WoE of the bin this category belongs to.
        woe_values.push_back(bin.woe);
        continue;
      }
      
      // Calculate WoE for individual categories
      int pos = count_pos_map.at(category); // Use .at() for safety, though key should exist
      int neg = count_neg_map.at(category);
      
      double cat_woe;
      if (bin_method == "woe") {
        double dist_pos = static_cast<double>(pos) / static_cast<double>(total_pos);
        double dist_neg = static_cast<double>(neg) / static_cast<double>(total_neg);
        dist_pos = std::max(dist_pos, EPSILON);
        dist_neg = std::max(dist_neg, EPSILON);
        cat_woe = std::log(dist_pos / dist_neg);
      } else { // bin_method == "woe1"
        double smoothed_pos = std::max(static_cast<double>(pos) + 0.5, EPSILON);
        double smoothed_neg = std::max(static_cast<double>(neg) + 0.5, EPSILON);
        cat_woe = std::log(smoothed_pos / smoothed_neg);
      }
      woe_values.push_back(cat_woe);
    }
    
    if (woe_values.empty()) return 0.0; // Should only happen if bin only contained "PREBIN_OTHER"
    
    // Calculate variance of WoE values
    double sum_woe = std::accumulate(woe_values.begin(), woe_values.end(), 0.0);
    double mean_woe = sum_woe / woe_values.size();
    
    double variance = 0.0;
    for (double woe : woe_values) {
      variance += std::pow(woe - mean_woe, 2);
    }
    variance /= woe_values.size(); // Use N, not N-1 for population variance within bin
    
    return variance;
  }
  
  
  /**
   * Split a given bin into two new bins based on individual category WoE values.
   * Adds the two new bins to the *end* of the main `bins` vector.
   * @param bin_to_split The bin object containing categories to be split.
   */
  void split_bin_into_two(const Bin& bin_to_split) {
    const size_t n_cats = bin_to_split.categories.size();
    if (n_cats <= 1) return; // Cannot split
    
    std::vector<std::pair<std::string, double>> category_woes;
    category_woes.reserve(n_cats);
    
    // Calculate WoE for each category
    for (const auto& category : bin_to_split.categories) {
      // Skip the special 'other' category if present, it cannot be split further reliably
      if (category == "PREBIN_OTHER") continue;
      
      int pos = count_pos_map.at(category);
      int neg = count_neg_map.at(category);
      double cat_woe;
      if (bin_method == "woe") {
        double dist_pos = static_cast<double>(pos) / static_cast<double>(total_pos);
        double dist_neg = static_cast<double>(neg) / static_cast<double>(total_neg);
        dist_pos = std::max(dist_pos, EPSILON);
        dist_neg = std::max(dist_neg, EPSILON);
        cat_woe = std::log(dist_pos / dist_neg);
      } else { // "woe1"
        double smoothed_pos = std::max(static_cast<double>(pos) + 0.5, EPSILON);
        double smoothed_neg = std::max(static_cast<double>(neg) + 0.5, EPSILON);
        cat_woe = std::log(smoothed_pos / smoothed_neg);
      }
      category_woes.emplace_back(category, cat_woe);
    }
    
    // If after skipping 'other', we have <= 1 category, cannot split
    if (category_woes.size() <= 1) {
      // Re-add the original bin (since we removed it before calling split)
      // Find where it should go based on WoE? Or just add back?
      // Safest is probably to prevent the split in ensure_min_bins if only 'other' is left.
      // Let's assume ensure_min_bins logic prevents calling split in this case.
      Rcpp::warning("Cannot split bin further as it only contains 'PREBIN_OTHER' or one regular category.");
      // We need to add the original bin back to the list as it was removed!
      bins.push_back(bin_to_split); // Add it back
      return;
    }
    
    
    // Sort categories by WoE
    std::sort(category_woes.begin(), category_woes.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
    
    // Split categories roughly in the middle
    size_t split_point = category_woes.size() / 2;
    if (split_point == 0 && category_woes.size() == 1) { // Handle edge case of 1 category after filtering
      split_point = 1; // Put the single category in the second bin
    }
    
    
    Bin bin1, bin2;
    bin1.categories.reserve(split_point);
    bin2.categories.reserve(category_woes.size() - split_point);
    
    
    for (size_t i = 0; i < category_woes.size(); ++i) {
      const std::string& category = category_woes[i].first;
      int pos = count_pos_map.at(category);
      int neg = count_neg_map.at(category);
      
      if (i < split_point) {
        bin1.categories.push_back(category);
        bin1.count_pos += pos;
        bin1.count_neg += neg;
      } else {
        bin2.categories.push_back(category);
        bin2.count_pos += pos;
        bin2.count_neg += neg;
      }
    }
    
    // Add the new bins to the end of the main bins vector
    bins.push_back(std::move(bin1));
    bins.push_back(std::move(bin2));
  }
  
  /**
   * Merge source bin (`src_bin`) into destination bin (`dest_bin`).
   * Modifies `dest_bin` in place.
   * @param dest_bin Destination bin.
   * @param src_bin Source bin.
   */
  void merge_two_bins(Bin& dest_bin, const Bin& src_bin) {
    // Merge categories
    dest_bin.categories.reserve(dest_bin.categories.size() + src_bin.categories.size());
    dest_bin.categories.insert(dest_bin.categories.end(),
                               src_bin.categories.begin(),
                               src_bin.categories.end());
    // Could sort categories here if needed for consistent naming, but affects performance.
    // std::sort(dest_bin.categories.begin(), dest_bin.categories.end());
    
    // Merge counts
    dest_bin.count_pos += src_bin.count_pos;
    dest_bin.count_neg += src_bin.count_neg;
    
    // WoE and divergence will be recomputed later for the merged bin
    dest_bin.woe = 0.0;
    dest_bin.divergence = 0.0;
  }
  
  /**
   * Compute bin metrics (WoE and divergence) for a single bin.
   * Used for efficiency after merges/splits.
   * @param bin The bin to compute metrics for (modified in place).
   */
  void compute_single_bin_metrics(Bin& bin) {
    double total_bin = static_cast<double>(bin.total_count());
    if (total_bin < EPSILON) {
      bin.woe = 0.0;
      bin.divergence = 0.0;
      return;
    }
    
    // Distributions relative to overall totals
    double dist_pos = static_cast<double>(bin.count_pos) / static_cast<double>(total_pos);
    double dist_neg = static_cast<double>(bin.count_neg) / static_cast<double>(total_neg);
    
    // --- Calculate WoE ---
    if (bin_method == "woe") {
      // Traditional WoE: ln((p_i/P)/(n_i/N))
      bin.woe = std::log(std::max(dist_pos, EPSILON) / std::max(dist_neg, EPSILON));
    } else { // bin_method == "woe1"
      // Zeng's WOE1: ln(g_i/b_i) using smoothed counts
      double smoothed_pos = std::max(static_cast<double>(bin.count_pos) + 0.5, EPSILON);
      double smoothed_neg = std::max(static_cast<double>(bin.count_neg) + 0.5, EPSILON);
      bin.woe = std::log(smoothed_pos / smoothed_neg);
    }
    
    // --- Calculate Divergence Contribution ---
    // Apply epsilon only where needed (log or division)
    dist_pos = std::max(dist_pos, 0.0); // Use 0 for divergences not involving log/div
    dist_neg = std::max(dist_neg, 0.0);
    
    if (divergence_method == "he") {
      // Hellinger Discrimination Term: 0.5 * (sqrt(p) - sqrt(n))^2
      bin.divergence = 0.5 * std::pow(std::sqrt(dist_pos) - std::sqrt(dist_neg), 2);
    } else if (divergence_method == "kl") {
      // Kullback-Leibler Term: p * log(p/n)
      bin.divergence = (dist_pos > EPSILON) ? dist_pos * std::log(dist_pos / std::max(dist_neg, EPSILON)) : 0.0;
    } else if (divergence_method == "tr") {
      // Triangular Discrimination Term: (p-n)^2 / (p+n)
      bin.divergence = (dist_pos + dist_neg > EPSILON) ? std::pow(dist_pos - dist_neg, 2) / (dist_pos + dist_neg) : 0.0;
    } else if (divergence_method == "klj") {
      // J-Divergence Term: (p-n) * log(p/n)
      bin.divergence = (dist_pos - dist_neg) * ((dist_pos > EPSILON && dist_neg > EPSILON) ? std::log(dist_pos / dist_neg) : 0.0);
    } else if (divergence_method == "sc") {
      // Symmetric Chi-Square Term: (p-n)^2 * (p+n) / (p*n)
      bin.divergence = (dist_pos > EPSILON && dist_neg > EPSILON) ?
      std::pow(dist_pos - dist_neg, 2) * (dist_pos + dist_neg) / (dist_pos * dist_neg) :
      ((dist_pos + dist_neg > EPSILON) ? std::numeric_limits<double>::infinity() : 0.0); // Handle div by zero
      
    } else if (divergence_method == "js") {
      // Jensen-Shannon Term: 0.5 * [ p*log(2p/(p+n)) + n*log(2n/(p+n)) ]
      double m = (dist_pos + dist_neg) / 2.0;
      double js_p = (dist_pos > EPSILON && m > EPSILON) ? dist_pos * std::log(dist_pos / m) : 0.0;
      double js_n = (dist_neg > EPSILON && m > EPSILON) ? dist_neg * std::log(dist_neg / m) : 0.0;
      bin.divergence = 0.5 * (js_p + js_n);
    } else if (divergence_method == "l1") {
      // L1 Term: |p-n|
      bin.divergence = std::abs(dist_pos - dist_neg);
    } else if (divergence_method == "l2") {
      // L2 Intermediate Term: (p-n)^2
      bin.divergence = std::pow(dist_pos - dist_neg, 2);
    } else if (divergence_method == "ln") {
      // L-infinity Intermediate Term: |p-n|
      bin.divergence = std::abs(dist_pos - dist_neg);
    }
    // Ensure non-negative divergence value
    if (!std::isinf(bin.divergence)) {
      bin.divergence = std::max(0.0, bin.divergence);
    }
  }
  
  /**
   * Compute bin metrics (WoE and divergence) for ALL bins.
   * Less efficient than single bin update but needed initially or after major changes.
   */
  void compute_bin_metrics() {
    for (auto& bin : bins) {
      compute_single_bin_metrics(bin);
    }
  }
  
  /**
   * Join category names with separator for display.
   * @param categories Vector of category names.
   * @return String of joined category names.
   */
  std::string join_categories(const std::vector<std::string>& categories) const {
    if (categories.empty()) return "EMPTY_BIN"; // Should not happen
    // Handle the special pre-binned case
    if (categories.size() == 1 && categories[0] == "PREBIN_OTHER") {
      return "PREBIN_OTHER";
    }
    
    // Sort categories for consistent bin naming (optional, adds overhead)
    // std::vector<std::string> sorted_cats = categories;
    // std::sort(sorted_cats.begin(), sorted_cats.end());
    
    // Pre-allocate string with estimated size
    size_t estimated_size = 0;
    for (const auto& cat : categories) { // Use original order if not sorting
      estimated_size += cat.size();
    }
    estimated_size += bin_separator.size() * (categories.size() > 0 ? categories.size() - 1 : 0);
    
    std::string result;
    result.reserve(estimated_size);
    
    // Join categories
    bool first = true;
    for (const auto& cat : categories) { // Use original order if not sorting
      if (!first) {
        result += bin_separator;
      }
      result += cat;
      first = false;
    }
    return result;
  }
  
  /**
   * Prepare output List for R.
   * @return List containing binning results.
   */
  Rcpp::List prepare_output() const {
    const size_t n_bins = bins.size();
    Rcpp::StringVector bin_names(n_bins);
    Rcpp::NumericVector woe_values(n_bins);
    Rcpp::NumericVector divergence_values(n_bins); // Per-bin contribution/value
    Rcpp::IntegerVector bin_counts(n_bins);
    Rcpp::IntegerVector counts_pos(n_bins);
    Rcpp::IntegerVector counts_neg(n_bins);
    Rcpp::IntegerVector ids(n_bins);
    
    for (size_t i = 0; i < n_bins; ++i) {
      ids[i] = i + 1; // 1-based index for R
      bin_names[i] = join_categories(bins[i].categories);
      woe_values[i] = bins[i].woe;
      divergence_values[i] = bins[i].divergence; // Store per-bin value (or intermediate for L2/Ln)
      bin_counts[i] = bins[i].total_count();
      counts_pos[i] = bins[i].count_pos;
      counts_neg[i] = bins[i].count_neg;
    }
    
    // Calculate total divergence correctly based on the method
    double total_divergence = 0.0;
    if (divergence_method == "l2") {
      double sum_sq_diff = 0.0;
      for (const auto& bin : bins) {
        sum_sq_diff += bin.divergence; // bin.divergence stores (p-n)^2
      }
      total_divergence = std::sqrt(sum_sq_diff);
    } else if (divergence_method == "ln") {
      double max_abs_diff = 0.0;
      for (const auto& bin : bins) {
        max_abs_diff = std::max(max_abs_diff, bin.divergence); // bin.divergence stores |p-n|
      }
      total_divergence = max_abs_diff;
    } else {
      // For other methods, divergence is additive
      for (const auto& bin : bins) {
        if (!std::isinf(bin.divergence)) { // Avoid adding Inf
          total_divergence += bin.divergence;
        } else {
          total_divergence = std::numeric_limits<double>::infinity(); // If any part is Inf, total is Inf
          break;
        }
      }
    }
    
    
    return Rcpp::List::create(
      Rcpp::Named("id") = ids,
      Rcpp::Named("bin") = bin_names,
      Rcpp::Named("woe") = woe_values,
      Rcpp::Named("divergence") = divergence_values, // Per-bin value/contribution
      Rcpp::Named("count") = bin_counts,
      Rcpp::Named("count_pos") = counts_pos,
      Rcpp::Named("count_neg") = counts_neg,
      Rcpp::Named("converged") = converged,
      Rcpp::Named("iterations") = iterations_run,
      Rcpp::Named("total_divergence") = total_divergence, // Correct total divergence
      Rcpp::Named("bin_method") = bin_method,
      Rcpp::Named("divergence_method") = divergence_method
    );
  }
  
public:
  /**
   * Constructor for OptimalBinningCategoricalDMIV_v2
   * (Parameters descriptions omitted for brevity, see Rcpp wrapper doc)
   */
  OptimalBinningCategoricalDMIV_v2(
    const std::vector<std::string>& feature_,
    const std::vector<int>& target_,
    int min_bins_ = 3,
    int max_bins_ = 5,
    double bin_cutoff_ = 0.05,
    int max_n_prebins_ = 20,
    const std::string& bin_separator_ = "%;%",
    double convergence_threshold_ = 1e-6,
    int max_iterations_ = 1000,
    std::string bin_method_ = "woe1",
    std::string divergence_method_ = "l2"
  ) : feature(feature_),
  target(target_),
  min_bins(min_bins_),
  max_bins(max_bins_),
  bin_cutoff(bin_cutoff_),
  max_n_prebins(max_n_prebins_),
  bin_separator(bin_separator_),
  convergence_threshold(convergence_threshold_),
  max_iterations(max_iterations_),
  bin_method(bin_method_),
  divergence_method(divergence_method_)
  // Other members initialized inline or in validate_inputs/initialize_bins
  {
    // Constructor body can be empty if all initialization is done via initializer list
    // or subsequent method calls within fit().
  }
  
  
  /**
   * Execute the optimal binning algorithm (v2)
   * @return List with binning results
   */
  Rcpp::List fit() {
    try {
      // Step 1: Validate inputs and calculate initial counts
      validate_inputs();
      
      // Step 2: Initialize bins (potentially pre-binning rare categories)
      initialize_bins();
      
      // Step 3: Perform optimal merging based on divergence measure
      perform_binning(); // Merges down to max_bins or convergence
      
      // Step 4: Ensure minimum number of bins by splitting if necessary
      ensure_min_bins(); // Splits up to min_bins if possible/needed
      
      // Step 5: Compute final metrics for all bins
      compute_bin_metrics();
      
      // Step 6: Sort final bins by WoE for better interpretability
      sort_bins_by_woe();
      
      // Step 7: Finalize and return results
      return prepare_output();
      
    } catch (const std::exception& e) {
      // Catch standard exceptions and report via Rcpp::stop
      Rcpp::stop("Error in optimal binning v2: " + std::string(e.what()));
    } catch (...) {
      // Catch any other unknown exceptions
      Rcpp::stop("Unknown error occurred during optimal binning v2.");
    }
  }
}; // End class OptimalBinningCategoricalDMIV_v2


//' @title Optimal Binning for Categorical Variables using Divergence Measures (V2)
//'
//' @description
//' Performs optimal binning for categorical variables using various divergence measures as proposed
//' by Zeng (2013). This is Version 2, incorporating fixes for potential crashes and performance improvements.
//' This method transforms categorical features into discrete bins by maximizing
//' the statistical divergence between distributions of positive and negative cases, while
//' maintaining interpretability constraints.
//'
//' @param target An integer binary vector (0 or 1) representing the target variable. Cannot contain NAs.
//' @param feature A character vector of categorical feature values. `NA` values will be treated as a distinct category "NA".
//' @param min_bins Minimum number of bins to generate (default: 3, must be >= 2).
//' @param max_bins Maximum number of bins to generate (default: 5).
//' @param bin_cutoff Minimum frequency fraction threshold used for OLD rare category handling (NOTE: V2 primarily uses `max_n_prebins` and `min_prebin_count` for initial handling, this cutoff is less relevant now but kept for interface compatibility). Default: 0.05.
//' @param max_n_prebins Maximum number of initial bins before merging starts. If unique categories exceed this, categories with counts < `min_prebin_count` (hardcoded as 5 currently) are grouped into a "PREBIN_OTHER" bin (default: 20).
//' @param bin_separator String separator for concatenating category names in bins (default: "%;%").
//' @param convergence_threshold Convergence threshold for the change in minimum divergence between iterations during merging (default: 1e-6).
//' @param max_iterations Maximum number of merging iterations allowed (default: 1000).
//' @param bin_method Method for WoE calculation, either 'woe' (traditional) or 'woe1' (Zeng's, smoothed) (default: 'woe1').
//' @param divergence_method Divergence measure to optimize for merging (lower is better). Options:
//' \itemize{
//'   \item 'he': Hellinger Distance
//'   \item 'kl': Symmetrized Kullback-Leibler Divergence
//'   \item 'tr': Triangular Discrimination
//'   \item 'klj': J-Divergence (Symmetric KL)
//'   \item 'sc': Symmetric Chi-Square Divergence
//'   \item 'js': Jensen-Shannon Divergence
//'   \item 'l1': L1 metric (Manhattan distance) on local bin proportions p/(p+n) vs q/(p+n)
//'   \item 'l2': L2 metric (Euclidean distance) on local bin proportions - Default
//'   \item 'ln': L-infinity metric (Maximum distance) on local bin proportions
//' }
//'
//' @return A list containing:
//' \item{id}{Numeric identifiers for each bin (1-based).}
//' \item{bin}{Character vector with the categories in each bin (or "PREBIN_OTHER").}
//' \item{woe}{Numeric vector with the Weight of Evidence values for each bin.}
//' \item{divergence}{Numeric vector with the divergence measure contribution for each bin (for L2/L-inf, this holds an intermediate value: `(p-n)^2` or `|p-n|` respectively, where p/n are proportions relative to total pos/neg).}
//' \item{count}{Integer vector with the total number of observations in each bin.}
//' \item{count_pos}{Integer vector with the number of positive observations in each bin.}
//' \item{count_neg}{Integer vector with the number of negative observations in each bin.}
//' \item{converged}{Logical value indicating whether the merging algorithm converged before hitting max iterations or `max_bins`.}
//' \item{iterations}{Number of merging iterations executed.}
//' \item{total_divergence}{The total divergence measure of the final binning solution (calculated correctly for all methods, including L2/L-inf).}
//' \item{bin_method}{The WoE calculation method used ('woe' or 'woe1').}
//' \item{divergence_method}{The divergence measure used for optimization.}
//'
//' @details
//' This implementation (V2) addresses potential stability and performance issues found in V1.
//' It follows the framework from Zeng (2013) using various divergence measures.
//' The `max_n_prebins` parameter is now functional, grouping rare categories initially if cardinality is high.
//' The similarity matrix update logic during merging and splitting has been corrected and optimized.
//' Calculation and reporting of L2/L-infinity total divergence are corrected.
//' Formulas for divergence measures (where P = (p_1,..p_n), Q = (q_1,..q_n) are distributions):
//' \deqn{Hellinger: h(P||Q) = \frac{1}{2}\sum_{i=1}^{n}(\sqrt{p_i} - \sqrt{q_i})^2}
//' \deqn{Symmetric KL: D_S(P||Q) = D(P||Q) + D(Q||P)}
//' \deqn{J-Divergence: J(P||Q) = D_S(P||Q)}
//' \deqn{Triangular: \Delta(P||Q) = \sum_{i=1}^{n}\frac{(p_i - q_i)^2}{p_i + q_i}}
//' \deqn{Chi-Square Symm: \psi(P||Q) = \sum_{i=1}^{n}\frac{(p_i - q_i)^2(p_i + q_i)}{p_iq_i}}
//' \deqn{Jensen-Shannon: JSD(P||Q) = \frac{1}{2}D(P||M) + \frac{1}{2}D(Q||M), M=\frac{P+Q}{2}}
//' \deqn{L1 (local proportions): L_1(p_1, p_2) = | \frac{g_1}{g_1+b_1} - \frac{g_2}{g_2+b_2} | + | \frac{b_1}{g_1+b_1} - \frac{b_2}{g_2+b_2} |} (Note: The code calculates L1 based on global proportions p_i/P vs n_i/N for merging criteria, but documentation needs clarification if local prop is intended) -> V2 Code calculates L1/L2/Ln based on *local* proportions for the bin similarity/distance now.
//' \deqn{L2 (local proportions): L_2(p_1, p_2) = \sqrt{ (\frac{g_1}{g_1+b_1} - \frac{g_2}{g_2+b_2})^2 + (\frac{b_1}{g_1+b_1} - \frac{b_2}{g_2+b_2})^2 }}
//' \deqn{L-infinity (local proportions): L_\infty(p_1, p_2) = \max ( | \frac{g_1}{g_1+b_1} - \frac{g_2}{g_2+b_2} |, | \frac{b_1}{g_1+b_1} - \frac{b_2}{g_2+b_2} | ) }
//' WoE Methods:
//' \deqn{Traditional WoE: \ln(\frac{g_i/G}{b_i/B})}
//' \deqn{Zeng's WOE1: \ln(\frac{g_i + 0.5}{b_i + 0.5})} (using smoothing)
//'
//' @references
//' Zeng, G. (2013). Metric Divergence Measures and Information Value in Credit Scoring.
//' Journal of Mathematics, 2013, Article ID 848271, 10 pages.
//'
//' Siddiqi, N. (2006). Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring.
//' John Wiley & Sons.
//'
//' @examples
//' \dontrun{
//' # Generate sample data
//' set.seed(123)
//' n <- 1000
//' categories <- c("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "RARE1", "RARE2")
//' feature <- sample(categories, n, replace = TRUE, prob = c(rep(0.09, 10), 0.05, 0.05))
//' feature[sample(1:n, 50)] <- NA # Add some NAs
//'
//' # Create target with different distribution per category
//' base_probs <- c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.05, 0.99)
//' target <- numeric(n)
//' for (i in 1:n) {
//'   if (is.na(feature[i])) {
//'     target[i] <- rbinom(1, 1, 0.5) # Assign random target for NA category
//'   } else {
//'      cat_idx <- match(feature[i], categories)
//'      target[i] <- rbinom(1, 1, base_probs[cat_idx])
//'   }
//' }
//'
//' # Apply optimal binning V2 with L2 metric and WOE1
//' result_v2 <- optimal_binning_categorical_dmiv_v2(target, feature, max_bins = 4)
//' print(result_v2)
//'
//' # Test with high cardinality and max_n_prebins
//' set.seed(456)
//' n_high <- 5000
//' categories_high <- paste0("CAT_", 1:100)
//' feature_high <- sample(categories_high, n_high, replace = TRUE)
//' target_high <- rbinom(n_high, 1, runif(n_high, 0.1, 0.9)) # Random target
//'
//' result_prebin <- optimal_binning_categorical_dmiv_v2(
//'    target_high,
//'    feature_high,
//'    max_n_prebins = 15, # Force pre-binning
//'    max_bins = 5
//' )
//' print(result_prebin)
//' print(result_prebin$bin) # Check for PREBIN_OTHER
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_dmiv(
   Rcpp::IntegerVector target,
   Rcpp::CharacterVector feature,
   int min_bins = 3,
   int max_bins = 5,
   double bin_cutoff = 0.05,
   int max_n_prebins = 20,
   std::string bin_separator = "%;%",
   double convergence_threshold = 1e-6,
   int max_iterations = 1000,
   std::string bin_method = "woe1",
   std::string divergence_method = "l2"
) {
 try {
   // Convert R vectors to C++ vectors and handle NAs
   R_xlen_t n = feature.size();
   if (n != target.size()) {
     Rcpp::stop("Feature and target must have the same length.");
   }
   std::vector<std::string> feature_vec(n);
   std::vector<int> target_vec(n);
   
   for (R_xlen_t i = 0; i < n; ++i) {
     // Handle NAs in feature vector -> Treat as category "NA"
     if (CharacterVector::is_na(feature[i])) {
       feature_vec[i] = "NA";
     } else {
       feature_vec[i] = Rcpp::as<std::string>(feature[i]);
     }
     
     // Check NAs in target vector -> Error
     if (IntegerVector::is_na(target[i])) {
       Rcpp::stop("Target variable cannot contain missing values (NA).");
     }
     target_vec[i] = target[i];
   }
   
   // Create algorithm object (V2)
   OptimalBinningCategoricalDMIV_v2 obcat_v2(
       feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins,
       bin_separator, convergence_threshold, max_iterations, bin_method,
       divergence_method
   );
   
   // Execute algorithm and return results
   return obcat_v2.fit();
   
 } catch (const std::exception& e) {
   Rcpp::stop("Error in optimal binning v2: " + std::string(e.what()));
 } catch (...) {
   Rcpp::stop("Unknown error occurred during optimal binning v2.");
 }
}













// // [[Rcpp::depends(Rcpp)]]
// #include <Rcpp.h>
// #include <vector>
// #include <string>
// #include <algorithm>
// #include <cmath>
// #include <limits>
// #include <stdexcept>
// #include <unordered_map>
// #include <unordered_set>
// 
// using namespace Rcpp;
// 
// /**
//  * Core class implementing Optimal Binning for categorical variables using various Divergence Measures.
//  * Based on the theoretical framework from Zeng (2013) "Metric Divergence Measures and Information Value in Credit Scoring".
//  * 
//  * This algorithm extends traditional approaches by optimizing bins using multiple divergence
//  * measures between distributions of positive and negative cases in each bin.
//  */
// class OptimalBinningCategoricalDMIV {
// private:
//   // Input parameters
//   const std::vector<std::string>& feature;
//   const std::vector<int>& target;
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
//   std::string bin_separator;
//   double convergence_threshold;
//   int max_iterations;
//   std::string bin_method;        // 'woe' or 'woe1'
//   std::string divergence_method; // 'he', 'kl', 'tr', 'klj', 'sc', 'js', 'l1', 'l2', 'ln'
//   
//   // Small constant to handle numerical issues
//   static constexpr double EPSILON = 1e-10;
//   
//   /**
//    * Structure representing a bin of categorical values
//    */
//   struct Bin {
//     std::vector<std::string> categories;  // Categories in this bin
//     int count_pos;                        // Count of positive class
//     int count_neg;                        // Count of negative class
//     double woe;                           // Weight of Evidence
//     double divergence;                    // Divergence measure value
//     
//     // Constructor with sensible initial capacity
//     Bin() : count_pos(0), count_neg(0), woe(0.0), divergence(0.0) {
//       categories.reserve(8);  // Reasonable initial capacity for small datasets
//     }
//   };
//   
//   // Internal state
//   std::vector<Bin> bins;
//   int total_pos;
//   int total_neg;
//   std::unordered_map<std::string, int> count_pos_map;
//   std::unordered_map<std::string, int> count_neg_map;
//   std::unordered_map<std::string, int> total_count_map;
//   int unique_categories;
//   bool converged;
//   int iterations_run;
//   
//   // Cache for similarity/distance calculations
//   std::vector<std::vector<double>> similarity_matrix;
//   
//   /**
//    * Validate input arguments for correctness and consistency
//    * Throws std::invalid_argument if validation fails
//    */
//   void validate_inputs() {
//     // Basic validation
//     if (feature.empty() || target.empty()) {
//       throw std::invalid_argument("Feature and target cannot be empty.");
//     }
//     if (feature.size() != target.size()) {
//       throw std::invalid_argument("Feature and target must have the same length.");
//     }
//     if (min_bins <= 0 || max_bins <= 0 || min_bins > max_bins) {
//       throw std::invalid_argument("Invalid values for min_bins or max_bins.");
//     }
//     if (bin_cutoff <= 0 || bin_cutoff >= 1) {
//       throw std::invalid_argument("bin_cutoff must be between 0 and 1.");
//     }
//     if (convergence_threshold <= 0) {
//       throw std::invalid_argument("convergence_threshold must be positive.");
//     }
//     if (max_iterations <= 0) {
//       throw std::invalid_argument("max_iterations must be positive.");
//     }
//     
//     // Validate bin_method and divergence_method
//     if (bin_method != "woe" && bin_method != "woe1") {
//       throw std::invalid_argument("bin_method must be either 'woe' or 'woe1'.");
//     }
//     
//     std::vector<std::string> valid_divergence_methods = {
//       "he", "kl", "tr", "klj", "sc", "js", "l1", "l2", "ln"
//     };
//     
//     if (std::find(valid_divergence_methods.begin(), valid_divergence_methods.end(), 
//                   divergence_method) == valid_divergence_methods.end()) {
//       throw std::invalid_argument("Invalid divergence_method. Must be one of: 'he', 'kl', 'tr', 'klj', 'sc', 'js', 'l1', 'l2', 'ln'.");
//     }
//     
//     // Efficiently process data in a single pass
//     int total_count = target.size();
//     std::unordered_map<std::string, std::pair<int, int>> counts;
//     counts.reserve(std::min(total_count, 10000));  // Reasonable upper limit
//     
//     // Count positives and negatives for each category
//     for (int i = 0; i < total_count; ++i) {
//       const int t = target[i];
//       if (t != 0 && t != 1) {
//         throw std::invalid_argument("Target must be binary (0 or 1).");
//       }
//       
//       const std::string& cat = feature[i];
//       auto& count_pair = counts[cat];
//       
//       if (t == 1) {
//         count_pair.first++;
//         total_pos++;
//       } else {
//         count_pair.second++;
//         total_neg++;
//       }
//     }
//     
//     if (total_pos == 0 || total_neg == 0) {
//       throw std::invalid_argument("Target must contain both 0 and 1 values.");
//     }
//     
//     // Transfer counts to final maps
//     for (const auto& item : counts) {
//       const std::string& cat = item.first;
//       const auto& count_pair = item.second;
//       
//       count_pos_map[cat] = count_pair.first;
//       count_neg_map[cat] = count_pair.second;
//       total_count_map[cat] = count_pair.first + count_pair.second;
//     }
//     
//     unique_categories = static_cast<int>(counts.size());
//     
//     // Adjust bin constraints based on available unique categories
//     min_bins = std::max(2, std::min(min_bins, unique_categories));
//     max_bins = std::min(max_bins, unique_categories);
//     if (min_bins > max_bins) {
//       min_bins = max_bins;
//     }
//   }
//   
//   /**
//    * Initialize bins with one category per bin
//    */
//   void initialize_bins() {
//     bins.clear();
//     bins.reserve(unique_categories);
//     
//     for (const auto& item : total_count_map) {
//       Bin bin;
//       bin.categories.push_back(item.first);
//       bin.count_pos = count_pos_map[item.first];
//       bin.count_neg = count_neg_map[item.first];
//       bins.push_back(std::move(bin));
//     }
//     
//     // Sort bins by WoE for better initial state
//     compute_bin_metrics();
//     sort_bins_by_woe();
//   }
//   
//   /**
//    * Sort bins by Weight of Evidence
//    */
//   void sort_bins_by_woe() {
//     std::sort(bins.begin(), bins.end(),
//               [](const Bin& a, const Bin& b) { return a.woe < b.woe; });
//   }
//   
//   /**
//    * Handle rare categories by merging them with similar bins
//    */
//   void handle_rare_categories() {
//     if (unique_categories <= 2) return;
//     
//     int total_count = total_pos + total_neg;
//     std::vector<Bin> updated_bins;
//     updated_bins.reserve(bins.size());
//     std::vector<Bin> rare_bins;
//     rare_bins.reserve(bins.size() / 3);  // Estimate based on bin_cutoff
//     
//     // Separate rare and non-rare bins
//     for (auto& bin : bins) {
//       int bin_total = bin.count_pos + bin.count_neg;
//       double freq = static_cast<double>(bin_total) / static_cast<double>(total_count);
//       if (freq < bin_cutoff) {
//         rare_bins.push_back(std::move(bin));
//       } else {
//         updated_bins.push_back(std::move(bin));
//       }
//     }
//     
//     // If no rare bins or not enough non-rare bins, return
//     if (rare_bins.empty() || updated_bins.size() < 2) {
//       if (!rare_bins.empty()) {
//         // If all bins are rare, keep them all
//         updated_bins.insert(updated_bins.end(), 
//                             std::make_move_iterator(rare_bins.begin()),
//                             std::make_move_iterator(rare_bins.end()));
//       }
//       bins = std::move(updated_bins);
//       return;
//     }
//     
//     // Merge rare categories with most similar bins
//     for (auto& rare_bin : rare_bins) {
//       size_t best_merge_index = 0;
//       double max_similarity = -std::numeric_limits<double>::max();
//       
//       // Find most similar bin (highest similarity/lowest divergence)
//       for (size_t i = 0; i < updated_bins.size(); ++i) {
//         double similarity = compute_bin_similarity(rare_bin, updated_bins[i]);
//         if (similarity > max_similarity) {
//           max_similarity = similarity;
//           best_merge_index = i;
//         }
//       }
//       
//       // Merge with most similar bin
//       merge_two_bins(updated_bins[best_merge_index], rare_bin);
//     }
//     
//     bins = std::move(updated_bins);
//     
//     // Update bin metrics after merging
//     compute_bin_metrics();
//   }
//   
//   /**
//    * Compute similarity between two bins based on their distribution profiles
//    * Returns a similarity score where higher values indicate more similar bins
//    * @param bin1 First bin
//    * @param bin2 Second bin
//    * @return Similarity score
//    */
//   double compute_bin_similarity(const Bin& bin1, const Bin& bin2) {
//     // Get distribution profiles
//     double total1 = bin1.count_pos + bin1.count_neg;
//     double total2 = bin2.count_pos + bin2.count_neg;
//     
//     if (total1 < EPSILON || total2 < EPSILON) return -std::numeric_limits<double>::max();
//     
//     double dist1_pos = bin1.count_pos / total1;
//     double dist1_neg = bin1.count_neg / total1;
//     double dist2_pos = bin2.count_pos / total2;
//     double dist2_neg = bin2.count_neg / total2;
//     
//     // Prevent div by zero and log(0)
//     dist1_pos = std::max(dist1_pos, EPSILON);
//     dist1_neg = std::max(dist1_neg, EPSILON);
//     dist2_pos = std::max(dist2_pos, EPSILON);
//     dist2_neg = std::max(dist2_neg, EPSILON);
//     
//     // Calculate similarity as negative divergence
//     // (we want to maximize similarity or minimize divergence)
//     double divergence = 0.0;
//     
//     if (divergence_method == "he") {
//       // Hellinger Distance (negative)
//       divergence = std::pow(std::sqrt(dist1_pos) - std::sqrt(dist2_pos), 2) + 
//         std::pow(std::sqrt(dist1_neg) - std::sqrt(dist2_neg), 2);
//     } else if (divergence_method == "kl") {
//       // Symmetrized KL divergence (since KL is asymmetric)
//       divergence = dist1_pos * std::log(dist1_pos / dist2_pos) + 
//         dist1_neg * std::log(dist1_neg / dist2_neg) +
//         dist2_pos * std::log(dist2_pos / dist1_pos) + 
//         dist2_neg * std::log(dist2_neg / dist1_neg);
//     } else if (divergence_method == "klj") {
//       // J-Divergence
//       divergence = (dist1_pos - dist2_pos) * std::log(dist1_pos / dist2_pos) + 
//         (dist1_neg - dist2_neg) * std::log(dist1_neg / dist2_neg);
//     } else if (divergence_method == "tr") {
//       // Triangular Discrimination
//       divergence = std::pow(dist1_pos - dist2_pos, 2) / (dist1_pos + dist2_pos) + 
//         std::pow(dist1_neg - dist2_neg, 2) / (dist1_neg + dist2_neg);
//     } else if (divergence_method == "sc") {
//       // Chi-Square Symmetric
//       divergence = std::pow(dist1_pos - dist2_pos, 2) * (dist1_pos + dist2_pos) / (dist1_pos * dist2_pos) + 
//         std::pow(dist1_neg - dist2_neg, 2) * (dist1_neg + dist2_neg) / (dist1_neg * dist2_neg);
//     } else if (divergence_method == "js") {
//       // Jensen-Shannon
//       double m_pos = (dist1_pos + dist2_pos) / 2;
//       double m_neg = (dist1_neg + dist2_neg) / 2;
//       divergence = 0.5 * (dist1_pos * std::log(dist1_pos / m_pos) + 
//         dist1_neg * std::log(dist1_neg / m_neg) +
//         dist2_pos * std::log(dist2_pos / m_pos) + 
//         dist2_neg * std::log(dist2_neg / m_neg));
//     } else if (divergence_method == "l1") {
//       // L1 metric (Manhattan)
//       divergence = std::abs(dist1_pos - dist2_pos) + std::abs(dist1_neg - dist2_neg);
//     } else if (divergence_method == "l2") {
//       // L2 metric (Euclidean)
//       divergence = std::sqrt(std::pow(dist1_pos - dist2_pos, 2) + std::pow(dist1_neg - dist2_neg, 2));
//     } else if (divergence_method == "ln") {
//       // L∞ metric (Maximum)
//       divergence = std::max(std::abs(dist1_pos - dist2_pos), std::abs(dist1_neg - dist2_neg));
//     }
//     
//     return -divergence;  // Return negative divergence as similarity
//   }
//   
//   /**
//    * Initialize similarity matrix for all bin pairs
//    */
//   void initialize_similarity_matrix() {
//     const size_t n = bins.size();
//     similarity_matrix.resize(n);
//     for (size_t i = 0; i < n; ++i) {
//       similarity_matrix[i].resize(n);
//       similarity_matrix[i][i] = 0.0;  // Self-similarity is 0
//       
//       // Only compute upper triangular part
//       for (size_t j = i + 1; j < n; ++j) {
//         similarity_matrix[i][j] = compute_bin_similarity(bins[i], bins[j]);
//         similarity_matrix[j][i] = similarity_matrix[i][j];  // Symmetric
//       }
//     }
//   }
//   
//   /**
//    * Update similarity matrix after bin merging
//    * @param merged_index Index of the merged bin
//    * @param removed_index Index of the removed bin
//    */
//   void update_similarity_matrix(size_t merged_index, size_t removed_index) {
//     // Remove the row and column for removed_index
//     similarity_matrix.erase(similarity_matrix.begin() + removed_index);
//     for (auto& row : similarity_matrix) {
//       row.erase(row.begin() + removed_index);
//     }
//     
//     // Update similarities for the merged bin with all other bins
//     const size_t n = bins.size();
//     for (size_t i = 0; i < n; ++i) {
//       if (i == merged_index) continue;
//       
//       double similarity = compute_bin_similarity(bins[merged_index], bins[i]);
//       similarity_matrix[merged_index][i] = similarity;
//       similarity_matrix[i][merged_index] = similarity;
//     }
//   }
//   
//   /**
//    * Perform optimal binning using divergence measures
//    */
//   void perform_binning() {
//     iterations_run = 0;
//     converged = false;
//     
//     // Initialize similarity matrix
//     initialize_similarity_matrix();
//     
//     // Main optimization loop
//     while (bins.size() > static_cast<size_t>(max_bins) && 
//            iterations_run < max_iterations && 
//            bins.size() > 1) {
//       
//       // Find pair of bins with maximum similarity (minimum divergence)
//       std::pair<double, std::pair<size_t, size_t>> best_merge = find_most_similar_bins();
//       double best_similarity = best_merge.first;
//       size_t bin1_idx = best_merge.second.first;
//       size_t bin2_idx = best_merge.second.second;
//       
//       // If best merge has negative similarity, check if we've reached convergence
//       if (best_similarity < -convergence_threshold && 
//           bins.size() <= static_cast<size_t>(max_bins)) {
//         converged = true;
//         break;
//       }
//       
//       // Store old similarity for convergence check
//       double old_similarity = best_similarity;
//       
//       // Merge bins with highest similarity
//       merge_two_bins(bins[bin1_idx], bins[bin2_idx]);
//       
//       // Remove the second bin
//       size_t removed_idx = bin2_idx;
//       bins.erase(bins.begin() + removed_idx);
//       
//       // Update similarity matrix
//       compute_bin_metrics();
//       update_similarity_matrix(bin1_idx, removed_idx);
//       
//       // Check convergence
//       best_merge = find_most_similar_bins();
//       double new_similarity = best_merge.first;
//       
//       if (bins.size() <= static_cast<size_t>(max_bins) && 
//           std::fabs(new_similarity - old_similarity) < convergence_threshold) {
//         converged = true;
//         break;
//       }
//       
//       iterations_run++;
//     }
//     
//     converged = converged || (iterations_run < max_iterations) || (bins.size() <= 1);
//     
//     // Ensure minimum number of bins if possible
//     ensure_min_bins();
//     
//     // Sort bins by WoE for better interpretability
//     sort_bins_by_woe();
//   }
//   
//   /**
//    * Find the pair of bins with the highest similarity
//    * @return pair<similarity, pair<index1, index2>>
//    */
//   std::pair<double, std::pair<size_t, size_t>> find_most_similar_bins() {
//     double max_similarity = -std::numeric_limits<double>::max();
//     std::pair<size_t, size_t> best_pair = {0, 0};
//     
//     const size_t n = bins.size();
//     for (size_t i = 0; i < n; ++i) {
//       for (size_t j = i + 1; j < n; ++j) {
//         if (similarity_matrix[i][j] > max_similarity) {
//           max_similarity = similarity_matrix[i][j];
//           best_pair = {i, j};
//         }
//       }
//     }
//     
//     return {max_similarity, best_pair};
//   }
//   
//   /**
//    * Ensure minimum number of bins by splitting most heterogeneous bins
//    */
//   void ensure_min_bins() {
//     if (bins.size() >= static_cast<size_t>(min_bins)) {
//       return;
//     }
//     
//     // Continue splitting bins until we reach min_bins
//     while (bins.size() < static_cast<size_t>(min_bins)) {
//       // Find bin with most heterogeneous categories
//       auto most_heterogeneous = std::max_element(
//         bins.begin(), bins.end(),
//         [this](const Bin& a, const Bin& b) { 
//           return calculate_bin_heterogeneity(a) < calculate_bin_heterogeneity(b); 
//         }
//       );
//       
//       // Can't split a bin with only one category
//       if (most_heterogeneous->categories.size() <= 1) {
//         break;
//       }
//       
//       // Split bin based on WoE profile
//       split_bin(*most_heterogeneous);
//       
//       // Remove original bin and add split results
//       size_t split_index = std::distance(bins.begin(), most_heterogeneous);
//       bins.erase(most_heterogeneous);
//       
//       // Update metrics and similarity matrix
//       compute_bin_metrics();
//       initialize_similarity_matrix();  // Reinitialize for simplicity after splitting
//     }
//   }
//   
//   /**
//    * Calculate heterogeneity of categories within a bin
//    * @param bin The bin to evaluate
//    * @return Heterogeneity score
//    */
//   double calculate_bin_heterogeneity(const Bin& bin) {
//     if (bin.categories.size() <= 1) return 0.0;
//     
//     // Calculate WoE for each category individually
//     std::vector<double> woe_values;
//     woe_values.reserve(bin.categories.size());
//     
//     for (const auto& category : bin.categories) {
//       int pos = count_pos_map[category];
//       int neg = count_neg_map[category];
//       
//       double cat_woe;
//       if (bin_method == "woe") {
//         double dist_pos = static_cast<double>(pos) / static_cast<double>(total_pos);
//         double dist_neg = static_cast<double>(neg) / static_cast<double>(total_neg);
//         
//         // Prevent div by zero and log(0)
//         dist_pos = std::max(dist_pos, EPSILON);
//         dist_neg = std::max(dist_neg, EPSILON);
//         
//         cat_woe = std::log(dist_pos / dist_neg);
//       } else {  // bin_method == "woe1"
//         // Prevent div by zero and log(0)
//         double smoothed_pos = std::max(static_cast<double>(pos) + 0.5, EPSILON);
//         double smoothed_neg = std::max(static_cast<double>(neg) + 0.5, EPSILON);
//         
//         cat_woe = std::log(smoothed_pos / smoothed_neg);
//       }
//       
//       woe_values.push_back(cat_woe);
//     }
//     
//     // Calculate variance of WoE values as heterogeneity measure
//     double mean_woe = 0.0;
//     for (double woe : woe_values) {
//       mean_woe += woe;
//     }
//     mean_woe /= woe_values.size();
//     
//     double variance = 0.0;
//     for (double woe : woe_values) {
//       variance += std::pow(woe - mean_woe, 2);
//     }
//     variance /= woe_values.size();
//     
//     return variance;
//   }
//   
//   /**
//    * Split a bin into two parts based on WoE values
//    * @param bin The bin to split
//    */
//   void split_bin(const Bin& bin) {
//     if (bin.categories.size() <= 1) return;
//     
//     // Create two new bins
//     Bin bin1, bin2;
//     
//     // Sort categories by individual WoE
//     std::vector<std::pair<std::string, double>> category_woes;
//     category_woes.reserve(bin.categories.size());
//     
//     for (const auto& category : bin.categories) {
//       int pos = count_pos_map[category];
//       int neg = count_neg_map[category];
//       
//       double cat_woe;
//       if (bin_method == "woe") {
//         double dist_pos = static_cast<double>(pos) / static_cast<double>(total_pos);
//         double dist_neg = static_cast<double>(neg) / static_cast<double>(total_neg);
//         
//         // Prevent div by zero and log(0)
//         dist_pos = std::max(dist_pos, EPSILON);
//         dist_neg = std::max(dist_neg, EPSILON);
//         
//         cat_woe = std::log(dist_pos / dist_neg);
//       } else {  // bin_method == "woe1"
//         // Prevent div by zero and log(0)
//         double smoothed_pos = std::max(static_cast<double>(pos) + 0.5, EPSILON);
//         double smoothed_neg = std::max(static_cast<double>(neg) + 0.5, EPSILON);
//         
//         cat_woe = std::log(smoothed_pos / smoothed_neg);
//       }
//       
//       category_woes.emplace_back(category, cat_woe);
//     }
//     
//     // Sort by WoE
//     std::sort(category_woes.begin(), category_woes.end(),
//               [](const auto& a, const auto& b) { return a.second < b.second; });
//     
//     // Split categories into two bins
//     size_t split_point = category_woes.size() / 2;
//     
//     for (size_t i = 0; i < split_point; ++i) {
//       const std::string& category = category_woes[i].first;
//       bin1.categories.push_back(category);
//       bin1.count_pos += count_pos_map[category];
//       bin1.count_neg += count_neg_map[category];
//     }
//     
//     for (size_t i = split_point; i < category_woes.size(); ++i) {
//       const std::string& category = category_woes[i].first;
//       bin2.categories.push_back(category);
//       bin2.count_pos += count_pos_map[category];
//       bin2.count_neg += count_neg_map[category];
//     }
//     
//     // Add new bins
//     bins.push_back(std::move(bin1));
//     bins.push_back(std::move(bin2));
//   }
//   
//   /**
//    * Merge two bins by combining their categories and counts
//    * @param dest_bin Destination bin (modified in place)
//    * @param src_bin Source bin
//    */
//   void merge_two_bins(Bin& dest_bin, const Bin& src_bin) {
//     // Reserve capacity to avoid reallocations
//     dest_bin.categories.reserve(dest_bin.categories.size() + src_bin.categories.size());
//     
//     // Merge categories
//     dest_bin.categories.insert(dest_bin.categories.end(), 
//                                src_bin.categories.begin(), 
//                                src_bin.categories.end());
//     
//     // Merge counts
//     dest_bin.count_pos += src_bin.count_pos;
//     dest_bin.count_neg += src_bin.count_neg;
//   }
//   
//   /**
//    * Compute bin metrics (WoE and divergence) for all bins
//    */
//   void compute_bin_metrics() {
//     for (auto& bin : bins) {
//       // Calculate proportions
//       double total_bin = bin.count_pos + bin.count_neg;
//       if (total_bin < EPSILON) continue;
//       
//       // Distribute proportions based on method
//       double dist_pos = static_cast<double>(bin.count_pos) / static_cast<double>(total_pos);
//       double dist_neg = static_cast<double>(bin.count_neg) / static_cast<double>(total_neg);
//       
//       // Prevent div by zero and log(0)
//       dist_pos = std::max(dist_pos, EPSILON);
//       dist_neg = std::max(dist_neg, EPSILON);
//       
//       // Calculate WoE with based on the selected method
//       if (bin_method == "woe") {
//         // Traditional WoE: ln((p_i/P)/(n_i/N))
//         bin.woe = std::log(dist_pos / dist_neg);
//       } else {  // bin_method == "woe1"
//         // Zeng's WOE1: ln(g_i/b_i) - direct log odds ratio
//         double smoothed_pos = std::max(static_cast<double>(bin.count_pos) + 0.5, EPSILON);
//         double smoothed_neg = std::max(static_cast<double>(bin.count_neg) + 0.5, EPSILON);
//         bin.woe = std::log(smoothed_pos / smoothed_neg);
//       }
//       
//       // Calculate divergence contribution based on selected method
//       if (divergence_method == "he") {
//         // Hellinger Discrimination
//         bin.divergence = 0.5 * std::pow(std::sqrt(dist_pos) - std::sqrt(dist_neg), 2);
//       } else if (divergence_method == "kl") {
//         // Kullback-Leibler
//         bin.divergence = dist_pos * std::log(dist_pos / dist_neg);
//       } else if (divergence_method == "tr") {
//         // Triangular Discrimination
//         bin.divergence = std::pow(dist_pos - dist_neg, 2) / (dist_pos + dist_neg);
//       } else if (divergence_method == "klj") {
//         // J-Divergence (symmetric KL)
//         bin.divergence = (dist_pos - dist_neg) * std::log(dist_pos / dist_neg);
//       } else if (divergence_method == "sc") {
//         // Symmetric Chi-Square Divergence
//         bin.divergence = std::pow(dist_pos - dist_neg, 2) * (dist_pos + dist_neg) / (dist_pos * dist_neg);
//       } else if (divergence_method == "js") {
//         // Jensen-Shannon Divergence
//         double m = (dist_pos + dist_neg) / 2.0;
//         bin.divergence = 0.5 * (dist_pos * std::log(dist_pos / m) + dist_neg * std::log(dist_neg / m));
//       } else if (divergence_method == "l1") {
//         // L1 metric (Manhattan distance)
//         bin.divergence = std::abs(dist_pos - dist_neg);
//       } else if (divergence_method == "l2") {
//         // L2 metric (Euclidean distance)
//         bin.divergence = std::pow(dist_pos - dist_neg, 2);  // Store squared diff temporarily
//       } else if (divergence_method == "ln") {
//         // L∞ metric (Maximum distance)
//         bin.divergence = std::abs(dist_pos - dist_neg);  // Store absolute diff temporarily
//       }
//     }
//     
//     // Special handling for L2 metric: take square root of sum
//     if (divergence_method == "l2") {
//       double sum_squared = 0.0;
//       for (const auto& bin : bins) {
//         sum_squared += bin.divergence;
//       }
//       double l2_norm = std::sqrt(sum_squared);
//       
//       // Adjust bin contributions proportionally
//       if (sum_squared > EPSILON) {
//         for (auto& bin : bins) {
//           bin.divergence = bin.divergence * l2_norm / sum_squared;
//         }
//       }
//     }
//     
//     // Special handling for L∞ metric: only bin with max contributes
//     if (divergence_method == "ln") {
//       double max_diff = 0.0;
//       for (const auto& bin : bins) {
//         if (bin.divergence > max_diff) {
//           max_diff = bin.divergence;
//         }
//       }
//       
//       for (auto& bin : bins) {
//         bin.divergence = (std::fabs(bin.divergence - max_diff) < EPSILON) ? max_diff : 0.0;
//       }
//     }
//   }
//   
//   /**
//    * Join category names with separator for display
//    * @param categories Vector of category names
//    * @return String of joined category names
//    */
//   std::string join_categories(const std::vector<std::string>& categories) const {
//     if (categories.empty()) return "";
//     if (categories.size() == 1) return categories[0];
//     
//     // Pre-allocate string with estimated size
//     size_t estimated_size = 0;
//     for (const auto& cat : categories) {
//       estimated_size += cat.size();
//     }
//     estimated_size += bin_separator.size() * (categories.size() - 1);
//     
//     std::string result;
//     result.reserve(estimated_size);
//     
//     // Join categories
//     result = categories[0];
//     for (size_t i = 1; i < categories.size(); ++i) {
//       result += bin_separator;
//       result += categories[i];
//     }
//     
//     return result;
//   }
//   
//   /**
//    * Prepare output for R
//    * @return List containing binning results
//    */
//   Rcpp::List prepare_output() const {
//     // Pre-allocate vectors for efficiency
//     const size_t n_bins = bins.size();
//     Rcpp::StringVector bin_names(n_bins);
//     Rcpp::NumericVector woe_values(n_bins);
//     Rcpp::NumericVector divergence_values(n_bins);
//     Rcpp::IntegerVector bin_counts(n_bins);
//     Rcpp::IntegerVector counts_pos(n_bins);
//     Rcpp::IntegerVector counts_neg(n_bins);
//     
//     // Fill output vectors
//     for (size_t i = 0; i < n_bins; ++i) {
//       bin_names[i] = join_categories(bins[i].categories);
//       woe_values[i] = bins[i].woe;
//       divergence_values[i] = bins[i].divergence;
//       bin_counts[i] = bins[i].count_pos + bins[i].count_neg;
//       counts_pos[i] = bins[i].count_pos;
//       counts_neg[i] = bins[i].count_neg;
//     }
//     
//     // Create bin IDs (1-based for R)
//     Rcpp::IntegerVector ids(n_bins);
//     for (size_t i = 0; i < n_bins; ++i) {
//       ids[i] = i + 1;
//     }
//     
//     // Calculate total divergence
//     double total_divergence = 0.0;
//     for (const auto& bin : bins) {
//       total_divergence += bin.divergence;
//     }
//     
//     // Return comprehensive results
//     return Rcpp::List::create(
//       Rcpp::Named("id") = ids,
//       Rcpp::Named("bin") = bin_names,
//       Rcpp::Named("woe") = woe_values,
//       Rcpp::Named("divergence") = divergence_values,
//       Rcpp::Named("count") = bin_counts,
//       Rcpp::Named("count_pos") = counts_pos,
//       Rcpp::Named("count_neg") = counts_neg,
//       Rcpp::Named("converged") = converged,
//       Rcpp::Named("iterations") = iterations_run,
//       Rcpp::Named("total_divergence") = total_divergence,
//       Rcpp::Named("bin_method") = bin_method,
//       Rcpp::Named("divergence_method") = divergence_method
//     );
//   }
//   
// public:
//   /**
//    * Constructor for OptimalBinningCategoricalDMIV
//    * 
//    * @param feature_ Character vector of feature values
//    * @param target_ Binary vector (0/1) representing target
//    * @param min_bins_ Minimum number of bins to generate
//    * @param max_bins_ Maximum number of bins to generate
//    * @param bin_cutoff_ Minimum frequency threshold for bins
//    * @param max_n_prebins_ Maximum number of initial bins
//    * @param bin_separator_ String separator for bin labels
//    * @param convergence_threshold_ Threshold for convergence
//    * @param max_iterations_ Maximum number of iterations
//    * @param bin_method_ Method for WoE calculation ('woe' or 'woe1')
//    * @param divergence_method_ Divergence measure to optimize
//    */
//   OptimalBinningCategoricalDMIV(
//     const std::vector<std::string>& feature_,
//     const std::vector<int>& target_,
//     int min_bins_ = 3,
//     int max_bins_ = 5,
//     double bin_cutoff_ = 0.05,
//     int max_n_prebins_ = 20,
//     const std::string& bin_separator_ = "%;%",
//     double convergence_threshold_ = 1e-6,
//     int max_iterations_ = 1000,
//     std::string bin_method_ = "woe1",
//     std::string divergence_method_ = "l2"
//   ) : feature(feature_),
//   target(target_),
//   min_bins(min_bins_),
//   max_bins(max_bins_),
//   bin_cutoff(bin_cutoff_),
//   max_n_prebins(max_n_prebins_),
//   bin_separator(bin_separator_),
//   convergence_threshold(convergence_threshold_),
//   max_iterations(max_iterations_),
//   bin_method(bin_method_),
//   divergence_method(divergence_method_),
//   total_pos(0),
//   total_neg(0),
//   unique_categories(0),
//   converged(false),
//   iterations_run(0) {
//     
//   }
//   
//   /**
//    * Execute the optimal binning algorithm
//    * @return List with binning results
//    */
//   Rcpp::List fit() {
//     try {
//       // Step 1: Validate inputs and initialize counts
//       validate_inputs();
//       
//       // Step 2: Initialize bins (one category per bin)
//       initialize_bins();
//       
//       // Step 3: Handle rare categories
//       handle_rare_categories();
//       
//       // Step 4: Perform optimal binning based on divergence measure
//       perform_binning();
//       
//       // Step 5: Finalize and return results
//       return prepare_output();
//     } catch (const std::exception& e) {
//       Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
//     }
//   }
// };
// 
// //' @title Optimal Binning for Categorical Variables using Divergence Measures
// //'
// //' @description
// //' Performs optimal binning for categorical variables using various divergence measures as proposed
// //' by Zeng (2013). This method transforms categorical features into discrete bins by maximizing 
// //' the statistical divergence between distributions of positive and negative cases, while 
// //' maintaining interpretability constraints.
// //'
// //' @param target An integer binary vector (0 or 1) representing the target variable.
// //' @param feature A character vector of categorical feature values.
// //' @param min_bins Minimum number of bins to generate (default: 3).
// //' @param max_bins Maximum number of bins to generate (default: 5).
// //' @param bin_cutoff Minimum frequency fraction for each bin (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins generated before optimization (default: 20).
// //' @param bin_separator String separator for concatenating category names in bins (default: "%;%").
// //' @param convergence_threshold Convergence threshold for divergence measure change (default: 1e-6).
// //' @param max_iterations Maximum number of iterations allowed for optimization (default: 1000).
// //' @param bin_method Method for WoE calculation, either 'woe' (traditional) or 'woe1' (Zeng's) (default: 'woe1').
// //' @param divergence_method Divergence measure to optimize. Options: 
// //' \itemize{
// //'   \item 'he': Hellinger Discrimination
// //'   \item 'kl': Kullback-Leibler Divergence
// //'   \item 'tr': Triangular Discrimination 
// //'   \item 'klj': J-Divergence (symmetric KL)
// //'   \item 'sc': Chi-Square Symmetric Divergence
// //'   \item 'js': Jensen-Shannon Divergence
// //'   \item 'l1': L1 metric (Manhattan distance)
// //'   \item 'l2': L2 metric (Euclidean distance) - Default
// //'   \item 'ln': L-infinity metric (Maximum distance)
// //' }
// //'
// //' @return A list containing:
// //' \item{id}{Numeric identifiers for each bin (1-based).}
// //' \item{bin}{Character vector with the categories in each bin.}
// //' \item{woe}{Numeric vector with the Weight of Evidence values for each bin.}
// //' \item{divergence}{Numeric vector with the divergence measure contribution for each bin.}
// //' \item{count}{Integer vector with the total number of observations in each bin.}
// //' \item{count_pos}{Integer vector with the number of positive observations in each bin.}
// //' \item{count_neg}{Integer vector with the number of negative observations in each bin.}
// //' \item{converged}{Logical value indicating whether the algorithm converged.}
// //' \item{iterations}{Number of iterations executed by the optimization algorithm.}
// //' \item{total_divergence}{The total divergence measure of the binning solution.}
// //' \item{bin_method}{The WoE calculation method used ('woe' or 'woe1').}
// //' \item{divergence_method}{The divergence measure used for optimization.}
// //'
// //' @details
// //' This implementation is based on the theoretical framework from Zeng (2013) "Metric Divergence 
// //' Measures and Information Value in Credit Scoring", which explores various divergence measures 
// //' for optimal binning in credit scoring applications.
// //' 
// //' Unlike traditional algorithms that use chi-square statistics, this approach leverages multiple 
// //' divergence measures to optimize the binning process:
// //' 
// //' \deqn{Hellinger: h(P||Q) = \frac{1}{2}\sum_{i=1}^{n}(\sqrt{p_i} - \sqrt{q_i})^2}
// //' \deqn{Kullback-Leibler: D(P||Q) = \sum_{i=1}^{n}p_i\ln(\frac{p_i}{q_i})}
// //' \deqn{J-Divergence: J(P||Q) = \sum_{i=1}^{n}(p_i - q_i)\ln(\frac{p_i}{q_i})}
// //' \deqn{Triangular: \Delta(P||Q) = \sum_{i=1}^{n}\frac{(p_i - q_i)^2}{p_i + q_i}}
// //' \deqn{Chi-Square: \psi(P||Q) = \sum_{i=1}^{n}\frac{(p_i - q_i)^2(p_i + q_i)}{p_iq_i}}
// //' \deqn{Jensen-Shannon: I(P||Q) = \frac{1}{2}[\sum_{i=1}^{n}p_i\ln(\frac{2p_i}{p_i+q_i}) + \sum_{i=1}^{n}q_i\ln(\frac{2q_i}{p_i+q_i})]}
// //' \deqn{L1: L_1(P||Q) = \sum_{i=1}^{n}|p_i - q_i|}
// //' \deqn{L2: L_2(P||Q) = \sqrt{\sum_{i=1}^{n}(p_i - q_i)^2}}
// //' \deqn{L-infinity: L_\infty(P||Q) = \max_{1 \leq i \leq n}|p_i - q_i|}
// //' 
// //' The algorithm also supports two different WoE calculation methods:
// //' \deqn{Traditional WoE: \ln(\frac{p_i/P}{n_i/N})}
// //' \deqn{Zeng's WOE1: \ln(\frac{g_i}{b_i})}
// //' 
// //' Where:
// //' \itemize{
// //'   \item \eqn{p_i, q_i}: Proportion of positive/negative cases in bin i
// //'   \item \eqn{g_i, b_i}: Count of positive/negative cases in bin i
// //'   \item \eqn{P, N}: Total positive/negative cases
// //' }
// //'
// //' @references
// //' Zeng, G. (2013). Metric Divergence Measures and Information Value in Credit Scoring.
// //' Journal of Mathematics, 2013, Article ID 848271, 10 pages.
// //' 
// //' Siddiqi, N. (2006). Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring. 
// //' John Wiley & Sons.
// //'
// //' @examples
// //' \dontrun{
// //' # Generate sample data
// //' set.seed(123)
// //' n <- 1000
// //' categories <- c("A", "B", "C", "D", "E")
// //' feature <- sample(categories, n, replace = TRUE)
// //' # Create target with different distribution per category
// //' probs <- c(0.2, 0.4, 0.5, 0.7, 0.9)
// //' target <- numeric(n)
// //' for (i in 1:n) {
// //'   cat_idx <- match(feature[i], categories)
// //'   target[i] <- rbinom(1, 1, probs[cat_idx])
// //' }
// //'
// //' # Apply optimal binning with L2 metric and WOE1
// //' result <- optimal_binning_categorical_dmiv(target, feature)
// //' print(result)
// //' 
// //' # Try with J-Divergence and traditional WoE
// //' result_j <- optimal_binning_categorical_dmiv(
// //'   target = target,
// //'   feature = feature,
// //'   divergence_method = "klj",
// //'   bin_method = "woe"
// //' )
// //' 
// //' # Compare results with other divergence measures
// //' measures <- c("he", "l1", "l2", "ln")
// //' results <- list()
// //' 
// //' for (measure in measures) {
// //'   results[[measure]] <- optimal_binning_categorical_dmiv(
// //'     target = target,
// //'     feature = feature,
// //'     divergence_method = measure
// //'   )
// //'   cat("Measure:", measure, "- Total divergence:", 
// //'       results[[measure]]$total_divergence, "\n")
// //' }
// //' }
// //'
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_categorical_dmiv(
//    Rcpp::IntegerVector target,
//    Rcpp::CharacterVector feature,
//    int min_bins = 3,
//    int max_bins = 5,
//    double bin_cutoff = 0.05,
//    int max_n_prebins = 20,
//    std::string bin_separator = "%;%",
//    double convergence_threshold = 1e-6,
//    int max_iterations = 1000,
//    std::string bin_method = "woe1",
//    std::string divergence_method = "l2"
// ) {
//  try {
//    // Convert R vectors to C++ vectors
//    std::vector<std::string> feature_vec(feature.size());
//    std::vector<int> target_vec(target.size());
//    
//    for (R_xlen_t i = 0; i < feature.size(); ++i) {
//      if (feature[i] == NA_STRING) {
//        feature_vec[i] = "NA";  // Handle missing values
//      } else {
//        feature_vec[i] = Rcpp::as<std::string>(feature[i]);
//      }
//    }
//    
//    for (R_xlen_t i = 0; i < target.size(); ++i) {
//      if (IntegerVector::is_na(target[i])) {
//        Rcpp::stop("Target cannot contain missing values.");
//      } else {
//        target_vec[i] = target[i];
//      }
//    }
//    
//    // Create algorithm object
//    OptimalBinningCategoricalDMIV obcat(
//        feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins,
//        bin_separator, convergence_threshold, max_iterations, bin_method,
//        divergence_method
//    );
//    
//    // Execute algorithm and return results
//    return obcat.fit();
//  } catch (const std::exception& e) {
//    Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
//  }
// }
// 
