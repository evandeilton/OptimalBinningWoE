// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>
#include <stdexcept>
#include <memory>
#include <random>

using namespace Rcpp;

// Include shared headers
#include "common/optimal_binning_common.h"
#include "common/bin_structures.h"

using namespace Rcpp;
using namespace OptimalBinning;


// Global constants for better readability and consistency
// Constant removed (uses shared definition)
static constexpr double NEG_INFINITY = -std::numeric_limits<double>::infinity();
static constexpr double LAPLACE_ALPHA = 0.5;  // Laplace smoothing parameter
static constexpr const char* MISSING_VALUE = "N/A";  // Special category for missing values

// Namespace with optimized utility functions
namespace utils {
// Safe and optimized logarithm function
inline double safe_log(double x) {
  return x > EPSILON ? std::log(x) : std::log(EPSILON);
}

// Optimized string joining function, ensuring unique values
inline std::string join(const std::vector<std::string>& v, const std::string& delimiter) {
  if (v.empty()) return "";
  if (v.size() == 1) return v[0];
  
  // Create a set to ensure unique values
  std::unordered_set<std::string> unique_values;
  std::vector<std::string> unique_vector;
  unique_vector.reserve(v.size());
  
  for (const auto& s : v) {
    if (unique_values.insert(s).second) {
      unique_vector.push_back(s);
    }
  }
  
  // Estimate size for pre-allocation
  size_t total_length = 0;
  for (const auto& s : unique_vector) {
    total_length += s.length();
  }
  total_length += delimiter.length() * (unique_vector.size() - 1);
  
  std::string result;
  result.reserve(total_length);
  
  result = unique_vector[0];
  for (size_t i = 1; i < unique_vector.size(); ++i) {
    result += delimiter;
    result += unique_vector[i];
  }
  return result;
}

// Universal hash function for strings with improved distribution
inline size_t string_hash(const std::string& str, size_t seed) {
  // FNV-1a hash algorithm: better distribution than simple multiplication
  constexpr size_t FNV_PRIME = 1099511628211ULL;
  constexpr size_t FNV_OFFSET_BASIS = 14695981039346656037ULL;
  
  size_t hash = FNV_OFFSET_BASIS ^ seed;
  for (char c : str) {
    hash ^= static_cast<size_t>(c);
    hash *= FNV_PRIME;
  }
  return hash;
}

// Laplace smoothing for more robust probability estimates
inline std::pair<double, double> smoothed_proportions(
    int positive_count, 
    int negative_count, 
    int total_positive, 
    int total_negative, 
    double alpha = LAPLACE_ALPHA) {
  
  // Apply Laplace (add-alpha) smoothing
  double smoothed_pos_rate = (positive_count + alpha) / (total_positive + alpha * 2);
  double smoothed_neg_rate = (negative_count + alpha) / (total_negative + alpha * 2);
  
  return {smoothed_pos_rate, smoothed_neg_rate};
}

// Calculate Weight of Evidence with Laplace smoothing
inline double calculate_woe(
    int positive_count, 
    int negative_count, 
    int total_positive, 
    int total_negative, 
    double alpha = LAPLACE_ALPHA) {
  
  auto [smoothed_pos_rate, smoothed_neg_rate] = smoothed_proportions(
    positive_count, negative_count, total_positive, total_negative, alpha);
  
  return safe_log(smoothed_pos_rate / smoothed_neg_rate);
}

// Calculate Information Value with Laplace smoothing
inline double calculate_iv(
    int positive_count, 
    int negative_count, 
    int total_positive, 
    int total_negative, 
    double alpha = LAPLACE_ALPHA) {
  
  auto [smoothed_pos_rate, smoothed_neg_rate] = smoothed_proportions(
    positive_count, negative_count, total_positive, total_negative, alpha);
  
  double woe = safe_log(smoothed_pos_rate / smoothed_neg_rate);
  return (smoothed_pos_rate - smoothed_neg_rate) * woe;
}

// Calculate statistical divergence between two bins
inline double bin_divergence(
    int bin1_pos, int bin1_neg, 
    int bin2_pos, int bin2_neg, 
    int total_pos, int total_neg) {
  
  // Jensen-Shannon divergence (symmetric KL divergence)
  auto [p1, n1] = smoothed_proportions(bin1_pos, bin1_neg, total_pos, total_neg);
  auto [p2, n2] = smoothed_proportions(bin2_pos, bin2_neg, total_pos, total_neg);
  
  // Average proportions
  double p_avg = (p1 + p2) / 2;
  double n_avg = (n1 + n2) / 2;
  
  // KL(P1 || P_avg) + KL(P2 || P_avg)
  double div_p1 = p1 > EPSILON ? p1 * safe_log(p1 / p_avg) : 0;
  double div_n1 = n1 > EPSILON ? n1 * safe_log(n1 / n_avg) : 0;
  double div_p2 = p2 > EPSILON ? p2 * safe_log(p2 / p_avg) : 0;
  double div_n2 = n2 > EPSILON ? n2 * safe_log(n2 / n_avg) : 0;
  
  return (div_p1 + div_n1 + div_p2 + div_n2) / 2;
}
}

// Count-Min Sketch structure for frequency estimation
class CountMinSketch {
private:
  std::vector<std::vector<int>> table;
  std::vector<size_t> seeds;
  size_t width;
  size_t depth;
  
public:
  CountMinSketch(size_t width_param = 2000, size_t depth_param = 5) 
    : width(width_param), depth(depth_param) {
    // Initialize counting table
    table.resize(depth);
    for (auto& row : table) {
      row.resize(width, 0);
    }
    
    // Initialize seeds for hash functions
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_int_distribution<size_t> dist(1, std::numeric_limits<size_t>::max());
    
    seeds.resize(depth);
    for (size_t i = 0; i < depth; ++i) {
      seeds[i] = dist(gen);
    }
  }
  
  // Increment counter for an item
  void update(const std::string& item, int count = 1) {
    for (size_t i = 0; i < depth; ++i) {
      size_t hash = utils::string_hash(item, seeds[i]) % width;
      table[i][hash] += count;
    }
  }
  
  // Estimate frequency of an item
  int estimate(const std::string& item) const {
    int min_count = std::numeric_limits<int>::max();
    
    for (size_t i = 0; i < depth; ++i) {
      size_t hash = utils::string_hash(item, seeds[i]) % width;
      min_count = std::min(min_count, table[i][hash]);
    }
    
    return min_count;
  }
  
  // Estimate error bounds for an estimate
  std::pair<int, int> estimate_with_bounds(const std::string& item) const {
    std::vector<int> counts(depth);
    
    for (size_t i = 0; i < depth; ++i) {
      size_t hash = utils::string_hash(item, seeds[i]) % width;
      counts[i] = table[i][hash];
    }
    
    // Sort counts for percentile-based bounds
    std::sort(counts.begin(), counts.end());
    
    // Return {lower_bound, upper_bound} using median and min as conservative estimates
    return {counts[0], counts[depth/2]};
  }
  
  // Method to detect frequent categories (heavy hitters)
  std::vector<std::string> heavy_hitters(
      const std::vector<std::string>& candidates, 
      double threshold_ratio) const {
    
    // Improved total count estimation: average across all rows for better accuracy
    int64_t total_count = 0;
    for (size_t i = 0; i < depth; ++i) {
      int64_t row_sum = std::accumulate(table[i].begin(), table[i].end(), 0LL);
      total_count += row_sum;
    }
    total_count /= depth;  // Average across rows
    
    // Absolute threshold based on ratio
    int threshold = static_cast<int>(total_count * threshold_ratio);
    
    // Filter candidates by threshold
    std::vector<std::string> result;
    result.reserve(candidates.size() / 4);
    
    for (const auto& candidate : candidates) {
      if (estimate(candidate) >= threshold) {
        result.push_back(candidate);
      }
    }
    
    return result;
  }
  
  // Estimate total elements in the sketch
  int64_t estimate_total_elements() const {
    // Average total count across all rows
    int64_t total_count = 0;
    for (size_t i = 0; i < depth; ++i) {
      int64_t row_sum = std::accumulate(table[i].begin(), table[i].end(), 0LL);
      total_count += row_sum;
    }
    return total_count / depth;
  }
};

// Optimized CategoricalBin structure with guaranteed category uniqueness
// Local CategoricalBin definition removed


// Cache for potential merges
class MergeCache {
private:
  std::vector<std::vector<double>> iv_loss_cache;
  std::vector<std::vector<double>> divergence_cache;
  bool enabled;
  
public:
  MergeCache(size_t max_size, bool use_cache = true) : enabled(use_cache) {
    if (enabled && max_size > 0) {
      iv_loss_cache.resize(max_size);
      divergence_cache.resize(max_size);
      for (auto& row : iv_loss_cache) {
        row.resize(max_size, -1.0);
      }
      for (auto& row : divergence_cache) {
        row.resize(max_size, -1.0);
      }
    }
  }
  
  inline double get_iv_loss(size_t bin1, size_t bin2) {
    if (!enabled || bin1 >= iv_loss_cache.size() || bin2 >= iv_loss_cache[bin1].size()) {
      return -1.0;
    }
    return iv_loss_cache[bin1][bin2];
  }
  
  inline void set_iv_loss(size_t bin1, size_t bin2, double value) {
    if (!enabled || bin1 >= iv_loss_cache.size() || bin2 >= iv_loss_cache[bin1].size()) {
      return;
    }
    iv_loss_cache[bin1][bin2] = value;
  }
  
  inline double get_divergence(size_t bin1, size_t bin2) {
    if (!enabled || bin1 >= divergence_cache.size() || bin2 >= divergence_cache[bin1].size()) {
      return -1.0;
    }
    return divergence_cache[bin1][bin2];
  }
  
  inline void set_divergence(size_t bin1, size_t bin2, double value) {
    if (!enabled || bin1 >= divergence_cache.size() || bin2 >= divergence_cache[bin1].size()) {
      return;
    }
    divergence_cache[bin1][bin2] = value;
  }
  
  inline void invalidate_bin(size_t bin_idx) {
    if (!enabled || bin_idx >= iv_loss_cache.size()) {
      return;
    }
    
    for (size_t i = 0; i < iv_loss_cache.size(); ++i) {
      if (i < iv_loss_cache[bin_idx].size()) {
        iv_loss_cache[bin_idx][i] = -1.0;
        divergence_cache[bin_idx][i] = -1.0;
      }
      if (bin_idx < iv_loss_cache[i].size()) {
        iv_loss_cache[i][bin_idx] = -1.0;
        divergence_cache[i][bin_idx] = -1.0;
      }
    }
  }
  
  inline void resize(size_t new_size) {
    if (!enabled) return;
    
    iv_loss_cache.resize(new_size);
    divergence_cache.resize(new_size);
    for (auto& row : iv_loss_cache) {
      row.resize(new_size, -1.0);
    }
    for (auto& row : divergence_cache) {
      row.resize(new_size, -1.0);
    }
  }
};

// Main class for Categorical Sketch Binning
class OBC_Sketch {
private:
  std::vector<std::string> feature;
  std::vector<int> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  std::string bin_separator;
  double convergence_threshold;
  int max_iterations;
  size_t sketch_width;
  size_t sketch_depth;
  bool use_divergence;
  
  int total_good;
  int total_bad;
  
  std::vector<CategoricalBin> bins;
  std::unique_ptr<MergeCache> merge_cache;
  std::unique_ptr<CountMinSketch> sketch;
  std::unique_ptr<CountMinSketch> sketch_pos;  // For positive events
  std::unique_ptr<CountMinSketch> sketch_neg;  // For negative events
  
  // Optimized input validation
  void validate_inputs() {
    if (feature.size() != target.size()) {
      throw std::invalid_argument("Feature and target must have the same size.");
    }
    if (feature.empty()) {
      throw std::invalid_argument("Feature and target cannot be empty.");
    }
    if (min_bins < 2) {
      throw std::invalid_argument("min_bins must be >= 2.");
    }
    if (max_bins < min_bins) {
      throw std::invalid_argument("max_bins must be >= min_bins.");
    }
    if (bin_cutoff <= 0 || bin_cutoff >= 1) {
      throw std::invalid_argument("bin_cutoff must be between 0 and 1.");
    }
    if (max_n_prebins < max_bins) {
      throw std::invalid_argument("max_n_prebins must be >= max_bins.");
    }
    if (sketch_width < 100) {
      throw std::invalid_argument("sketch_width must be >= 100 for reasonable accuracy.");
    }
    if (sketch_depth < 3) {
      throw std::invalid_argument("sketch_depth must be >= 3 for reasonable accuracy.");
    }
    
    // Efficient target value verification
    bool has_zero = false;
    bool has_one = false;
    
    for (int val : target) {
      if (val == 0) has_zero = true;
      else if (val == 1) has_one = true;
      else throw std::invalid_argument("Target must contain only 0 and 1.");
      
      // Early termination
      if (has_zero && has_one) break;
    }
    
    if (!has_zero || !has_one) {
      throw std::invalid_argument("Target must contain both 0 and 1.");
    }
  }
  
  // Build sketches from the data
  void build_sketches() {
    // Initialize sketches
    sketch = std::make_unique<CountMinSketch>(sketch_width, sketch_depth);
    sketch_pos = std::make_unique<CountMinSketch>(sketch_width, sketch_depth);
    sketch_neg = std::make_unique<CountMinSketch>(sketch_width, sketch_depth);
    
    total_good = 0;
    total_bad = 0;
    
    // Set to track unique categories
    std::unordered_set<std::string> unique_categories;
    
    // Fill sketches in a single pass
    for (size_t i = 0; i < feature.size(); ++i) {
      const auto& cat = feature[i];
      int is_positive = target[i];
      
      unique_categories.insert(cat);
      
      // Update general sketch
      sketch->update(cat, 1);
      
      // Update sketch for positives/negatives
      if (is_positive) {
        sketch_pos->update(cat, 1);
        total_bad++;
      } else {
        sketch_neg->update(cat, 1);
        total_good++;
      }
    }
    
    // Adjust max_bins based on unique categories
    int ncat = static_cast<int>(unique_categories.size());
    if (max_bins > ncat) {
      max_bins = ncat;
    }
    
    // Adjust min_bins if necessary
    min_bins = std::min(min_bins, max_bins);
  }
  
  // Prebinning using sketch
  void prebinning() {
    // Identify unique categories
    std::unordered_set<std::string> unique_categories_set;
    for (const auto& cat : feature) {
      unique_categories_set.insert(cat);
    }
    
    // Convert to vector for easier processing
    std::vector<std::string> unique_categories(unique_categories_set.begin(), unique_categories_set.end());
    
    // Identify heavy hitters using Count-Min Sketch
    double heavy_hitter_threshold = bin_cutoff / 2.0;  // More permissive threshold for initial detection
    std::vector<std::string> heavy_hitters = sketch->heavy_hitters(unique_categories, heavy_hitter_threshold);
    
    // Create initial bins from heavy hitters
    bins.clear();
    bins.reserve(heavy_hitters.size());
    
    for (const auto& cat : heavy_hitters) {
      CategoricalBin bin;
      int pos_count = sketch_pos->estimate(cat);
      int neg_count = sketch_neg->estimate(cat);
      bin.categories.push_back(cat); bin.count_pos += pos_count; bin.count_neg += neg_count; bin.update_count();
      bins.push_back(bin);
    }
    
    // Add rare categories to a separate bin
    CategoricalBin rare_bin;
    for (const auto& cat : unique_categories) {
      if (std::find(heavy_hitters.begin(), heavy_hitters.end(), cat) == heavy_hitters.end()) {
        int pos_count = sketch_pos->estimate(cat);
        int neg_count = sketch_neg->estimate(cat);
        rare_bin.categories.push_back(cat); rare_bin.count_pos += pos_count; rare_bin.count_neg += neg_count; rare_bin.update_count();
      }
    }
    
    // Add rare categories bin if not empty
    if (rare_bin.count > 0) {
      bins.push_back(rare_bin);
    }
    
    // Sort bins by count (descending)
    std::sort(bins.begin(), bins.end(),
              [](const CategoricalBin& a, const CategoricalBin& b) { return a.count > b.count; });
    
    // Initialize merge cache
    merge_cache = std::make_unique<MergeCache>(bins.size(), bins.size() > 10);
    
    // Reduce to max_n_prebins if necessary
    while (static_cast<int>(bins.size()) > max_n_prebins && static_cast<int>(bins.size()) > min_bins) {
      // Find smallest bins to merge based on statistical similarity
      size_t min_idx1 = 0;
      size_t min_idx2 = 1;
      double min_divergence = std::numeric_limits<double>::max();
      
      for (size_t i = 0; i < bins.size(); ++i) {
        for (size_t j = i + 1; j < bins.size(); ++j) {
          // Prioritize merging small bins
          int combined_count = bins[i].count + bins[j].count;
          double size_penalty = 1.0 + std::log(1.0 + combined_count);
          
          double div = bins[i].divergence_from(bins[j], total_good, total_bad) * size_penalty;
          
          if (div < min_divergence) {
            min_divergence = div;
            min_idx1 = i;
            min_idx2 = j;
          }
        }
      }
      
      // Try to merge bins with minimal divergence
      if (!try_merge_bins(min_idx1, min_idx2)) break;
    }
  }
  
  // Optimized bin_cutoff enforcement
  void enforce_bin_cutoff() {
    int min_count = static_cast<int>(std::ceil(bin_cutoff * static_cast<double>(feature.size())));
    int min_count_pos = static_cast<int>(std::ceil(bin_cutoff * static_cast<double>(total_bad)));
    
    // Identify all low-frequency bins at once
    std::vector<size_t> low_freq_bins;
    
    for (size_t i = 0; i < bins.size(); ++i) {
      if (bins[i].count < min_count || bins[i].count_pos < min_count_pos) {
        low_freq_bins.push_back(i);
      }
    }
    
    // Process low-frequency bins efficiently
    for (size_t idx : low_freq_bins) {
      if (static_cast<int>(bins.size()) <= min_bins) {
        break; // Never go below min_bins
      }
      
      // CategoricalBin still exists and still below cutoff?
      if (idx >= bins.size() || (bins[idx].count >= min_count && bins[idx].count_pos >= min_count_pos)) {
        continue;
      }
      
      // Find most similar bin to merge with based on statistical similarity
      size_t merge_idx = idx;
      double min_divergence = std::numeric_limits<double>::max();
      
      for (size_t i = 0; i < bins.size(); ++i) {
        if (i == idx) continue;
        
        double div = bins[idx].divergence_from(bins[i], total_good, total_bad);
        if (div < min_divergence) {
          min_divergence = div;
          merge_idx = i;
        }
      }
      
      if (idx == merge_idx) {
        // Fallback to nearest neighbor if no suitable match found
        if (idx > 0) {
          merge_idx = idx - 1;
        } else if (idx + 1 < bins.size()) {
          merge_idx = idx + 1;
        } else {
          continue; // No neighbors
        }
      }
      
      if (!try_merge_bins(std::min(idx, merge_idx), std::max(idx, merge_idx))) {
        // If couldn't merge without violating min_bins, try another bin
        continue;
      }
      
      // Adjust indices of remaining bins, since we removed one
      for (auto& remaining_idx : low_freq_bins) {
        if (remaining_idx > merge_idx) {
          remaining_idx--;
        }
      }
    }
  }
  
  // Initial WoE calculation
  void calculate_initial_woe() {
    for (auto& bin : bins) {
      bin.calculate_metrics(total_good, total_bad);
    }
  }
  
  // Optimized monotonicity enforcement
  void enforce_monotonicity() {
    if (bins.empty()) {
      throw std::runtime_error("No bins available to enforce monotonicity.");
    }
    
    // Sort bins by WoE
    std::sort(bins.begin(), bins.end(),
              [](const CategoricalBin& a, const CategoricalBin& b) { return a.woe < b.woe; });
    
    // Determine monotonicity direction
    bool increasing = true;
    if (bins.size() > 1) {
      for (size_t i = 1; i < bins.size(); ++i) {
        if (bins[i].woe < bins[i-1].woe - EPSILON) {
          increasing = false;
          break;
        }
      }
    }
    
    // Optimized loop to enforce monotonicity
    bool any_merge;
    do {
      any_merge = false;
      
      // Find the most serious violation first
      double max_violation = 0.0;
      size_t violation_idx = 0;
      
      for (size_t i = 0; i + 1 < bins.size(); ++i) {
        if (static_cast<int>(bins.size()) <= min_bins) {
          break; // Don't reduce below min_bins
        }
        
        double violation_amount = 0.0;
        bool is_violation = false;
        
        if (increasing && bins[i].woe > bins[i+1].woe + EPSILON) {
          violation_amount = bins[i].woe - bins[i+1].woe;
          is_violation = true;
        } else if (!increasing && bins[i].woe < bins[i+1].woe - EPSILON) {
          violation_amount = bins[i+1].woe - bins[i].woe;
          is_violation = true;
        }
        
        if (is_violation && violation_amount > max_violation) {
          max_violation = violation_amount;
          violation_idx = i;
        }
      }
      
      // Fix the most serious violation first
      if (max_violation > EPSILON) {
        if (try_merge_bins(violation_idx, violation_idx + 1)) {
          any_merge = true;
        }
      }
      
    } while (any_merge && static_cast<int>(bins.size()) > min_bins);
  }
  
  // Enhanced bin optimization
  void optimize_bins() {
    if (static_cast<int>(bins.size()) <= max_bins) {
      return; // Already within max limit
    }
    
    int iterations = 0;
    double prev_total_iv = 0.0;
    
    for (const auto& bin : bins) {
      prev_total_iv += std::fabs(bin.iv);
    }
    
    while (static_cast<int>(bins.size()) > max_bins && iterations < max_iterations) {
      if (static_cast<int>(bins.size()) <= min_bins) {
        break; // Don't reduce below min_bins
      }
      
      // Find pair of bins with lowest combined IV or highest similarity
      double min_score = std::numeric_limits<double>::max();
      size_t min_score_idx1 = 0;
      size_t min_score_idx2 = 0;
      
      // Optimized search with cache
      for (size_t i = 0; i < bins.size(); ++i) {
        for (size_t j = i + 1; j < bins.size(); ++j) {
          double score;
          
          if (use_divergence) {
            // Use statistical divergence for merging decisions
            double cached_div = merge_cache->get_divergence(i, j);
            if (cached_div >= 0.0) {
              score = cached_div;
            } else {
              score = bins[i].divergence_from(bins[j], total_good, total_bad);
              merge_cache->set_divergence(i, j, score);
            }
          } else {
            // Use IV loss for merging decisions
            double cached_iv = merge_cache->get_iv_loss(i, j);
            if (cached_iv >= 0.0) {
              score = cached_iv;
            } else {
              score = std::fabs(bins[i].iv) + std::fabs(bins[j].iv);
              merge_cache->set_iv_loss(i, j, score);
            }
          }
          
          if (score < min_score) {
            min_score = score;
            min_score_idx1 = i;
            min_score_idx2 = j;
          }
        }
      }
      
      // Try to merge bins with minimal score
      if (!try_merge_bins(min_score_idx1, min_score_idx2)) {
        break; // Couldn't merge, stop
      }
      
      // Calculate new total IV and check convergence
      double total_iv = 0.0;
      for (const auto& bin : bins) {
        total_iv += std::fabs(bin.iv);
      }
      
      if (std::fabs(total_iv - prev_total_iv) < convergence_threshold) {
        break; // Convergence reached
      }
      
      prev_total_iv = total_iv;
      iterations++;
      
      // Adaptive strategy: switch between divergence and IV loss
      if (iterations % 5 == 0) {
        use_divergence = !use_divergence;
      }
    }
    
    if (static_cast<int>(bins.size()) > max_bins) {
      Rcpp::warning(
        "Could not reduce the number of bins to max_bins without violating min_bins or convergence criteria. "
        "Current bins: " + std::to_string(bins.size()) + ", max_bins: " + std::to_string(max_bins)
      );
    }
  }
  
  // Optimized bin merging attempt
  bool try_merge_bins(size_t index1, size_t index2) {
    // Safety checks
    if (static_cast<int>(bins.size()) <= min_bins) {
      return false; // Already at minimum, don't merge
    }
    
    if (index1 >= bins.size() || index2 >= bins.size() || index1 == index2) {
      return false;
    }
    
    if (index2 < index1) std::swap(index1, index2);
    
    // Efficient merging
    bins[index1].merge_with(bins[index2]);
    bins[index1].calculate_metrics(total_good, total_bad);
    
    bins.erase(bins.begin() + index2);
    
    // Update cache
    merge_cache->invalidate_bin(index1);
    merge_cache->resize(bins.size());
    
    return true;
  }
  
  // Optimized consistency check
  void check_consistency() const {
    int total_count = 0;
    int total_count_pos = 0;
    int total_count_neg = 0;
    
    for (const auto& bin : bins) {
      total_count += bin.count;
      total_count_pos += bin.count_pos;
      total_count_neg += bin.count_neg;
    }
    
    // More permissive validation due to sketch approximation
    double count_ratio = static_cast<double>(total_count) / feature.size();
    if (count_ratio < 0.95 || count_ratio > 1.05) {
      Rcpp::warning(
        "Possible inconsistency after binning due to sketch approximation. "
        "Total count: " + std::to_string(total_count) + ", expected: " + 
          std::to_string(feature.size()) + ". Ratio: " + std::to_string(count_ratio)
      );
    }
    
    double pos_ratio = static_cast<double>(total_count_pos) / total_bad;
    double neg_ratio = static_cast<double>(total_count_neg) / total_good;
    if (pos_ratio < 0.95 || pos_ratio > 1.05 || neg_ratio < 0.95 || neg_ratio > 1.05) {
      Rcpp::warning(
        "Possible inconsistency in positive/negative counts after binning due to sketch approximation. "
        "Positives: " + std::to_string(total_count_pos) + " vs " + std::to_string(total_bad) + 
          ", Negatives: " + std::to_string(total_count_neg) + " vs " + std::to_string(total_good)
      );
    }
  }
  
public:
  // Optimized constructor
  OBC_Sketch(
    const std::vector<std::string>& feature_,
    const Rcpp::IntegerVector& target_,
    int min_bins_ = 3,
    int max_bins_ = 5,
    double bin_cutoff_ = 0.05,
    int max_n_prebins_ = 20,
    std::string bin_separator_ = "%;%",
    double convergence_threshold_ = 1e-6,
    int max_iterations_ = 1000,
    size_t sketch_width_ = 2000,
    size_t sketch_depth_ = 5
  ) : feature(feature_), target(Rcpp::as<std::vector<int>>(target_)), 
  min_bins(min_bins_), max_bins(max_bins_), 
  bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_),
  bin_separator(bin_separator_),
  convergence_threshold(convergence_threshold_),
  max_iterations(max_iterations_),
  sketch_width(sketch_width_),
  sketch_depth(sketch_depth_),
  use_divergence(true),
  total_good(0), total_bad(0) {
    
    // Pre-allocation for better performance
    bins.reserve(std::min(max_n_prebins_, 1000));
  }
  
  // Optimized fit method
  Rcpp::List fit() {
    try {
      // Binning process with performance checks
      validate_inputs();
      build_sketches();
      prebinning();
      enforce_bin_cutoff();
      calculate_initial_woe();
      enforce_monotonicity();
      
      // CategoricalBin optimization if necessary
      bool converged_flag = false;
      int iterations_done = 0;
      
      if (static_cast<int>(bins.size()) <= max_bins) {
        converged_flag = true;
      } else {
        double prev_total_iv = 0.0;
        for (const auto& bin : bins) {
          prev_total_iv += std::fabs(bin.iv);
        }
        
        for (int i = 0; i < max_iterations; ++i) {
          // Starting point for this iteration
          size_t start_bins = bins.size();
          
          optimize_bins();
          
          // If didn't reduce bins or reached max_bins, check convergence
          if (bins.size() == start_bins || static_cast<int>(bins.size()) <= max_bins) {
            double total_iv = 0.0;
            for (const auto& bin : bins) {
              total_iv += std::fabs(bin.iv);
            }
            
            if (std::fabs(total_iv - prev_total_iv) < convergence_threshold) {
              converged_flag = true;
              iterations_done = i + 1;
              break;
            }
            
            prev_total_iv = total_iv;
          }
          
          iterations_done = i + 1;
          
          if (static_cast<int>(bins.size()) <= max_bins) {
            break;
          }
        }
      }
      
      // Final consistency check
      check_consistency();
      
      // Optimized results preparation with uniqueness guarantee
      const size_t n_bins = bins.size();
      
      CharacterVector bin_names(n_bins);
      NumericVector bin_woe(n_bins);
      NumericVector bin_iv(n_bins);
      IntegerVector bin_count(n_bins);
      IntegerVector bin_count_pos(n_bins);
      IntegerVector bin_count_neg(n_bins);
      NumericVector ids(n_bins);
      NumericVector event_rates(n_bins);
      
      for (size_t i = 0; i < n_bins; ++i) {
        // Uses utils::join which now guarantees unique values
        bin_names[i] = utils::join(bins[i].categories, bin_separator);
        bin_woe[i] = bins[i].woe;
        bin_iv[i] = bins[i].iv;
        bin_count[i] = bins[i].count;
        bin_count_pos[i] = bins[i].count_pos;
        bin_count_neg[i] = bins[i].count_neg;
        event_rates[i] = bins[i].event_rate();
        ids[i] = i + 1;
      }
      
      // Calculate total IV
      double total_iv = 0.0;
      for (size_t i = 0; i < n_bins; ++i) {
        total_iv += std::fabs(bin_iv[i]);
      }
      
      return Rcpp::List::create(
        Named("id") = ids,
        Named("bin") = bin_names,
        Named("woe") = bin_woe,
        Named("iv") = bin_iv,
        Named("count") = bin_count,
        Named("count_pos") = bin_count_pos,
        Named("count_neg") = bin_count_neg,
        Named("event_rate") = event_rates,
        Named("converged") = converged_flag,
        Named("iterations") = iterations_done,
        Named("total_iv") = total_iv
      );
    } catch (const std::exception& e) {
      Rcpp::stop("Error in optimal binning with sketch: " + std::string(e.what()));
    }
  }
};

// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_sketch(
   Rcpp::IntegerVector target,
   Rcpp::CharacterVector feature,
   int min_bins = 3,
   int max_bins = 5,
   double bin_cutoff = 0.05,
   int max_n_prebins = 20,
   std::string bin_separator = "%;%",
   double convergence_threshold = 1e-6,
   int max_iterations = 1000,
   int sketch_width = 2000,
   int sketch_depth = 5
) {
 // Preliminary checks
 if (feature.size() == 0 || target.size() == 0) {
   Rcpp::stop("Feature and target cannot be empty.");
 }
 
 if (feature.size() != target.size()) {
   Rcpp::stop("Feature and target must have the same size.");
 }
 
 // Optimized conversion from R to C++
 std::vector<std::string> feature_vec;
 feature_vec.reserve(feature.size());
 
 for (R_xlen_t i = 0; i < feature.size(); ++i) {
   // Handle NAs in feature
   if (feature[i] == NA_STRING) {
     feature_vec.push_back(MISSING_VALUE);
   } else {
     feature_vec.push_back(Rcpp::as<std::string>(feature[i]));
   }
 }
 
 // Validate NAs in target
 for (R_xlen_t i = 0; i < target.size(); ++i) {
   if (IntegerVector::is_na(target[i])) {
     Rcpp::stop("Target cannot contain missing values.");
   }
 }
 
 // Run the optimized sketch algorithm
 OBC_Sketch sketch_binner(
     feature_vec, target, min_bins, max_bins, bin_cutoff, max_n_prebins,
     bin_separator, convergence_threshold, max_iterations,
     sketch_width, sketch_depth
 );
 
 return sketch_binner.fit();
}