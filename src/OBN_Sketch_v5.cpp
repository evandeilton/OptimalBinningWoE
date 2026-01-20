// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// Include shared headers
#include "common/bin_structures.h"
#include "common/optimal_binning_common.h"

using namespace Rcpp;
using namespace OptimalBinning;

// ============================================================================
// Namespace with optimized utility functions
// ============================================================================
namespace utils {
// Safe and optimized logarithm function
inline double safe_log(double x) {
  return x > EPSILON ? std::log(x) : std::log(EPSILON);
}

// Safe comparison of doubles
inline bool are_equal(double a, double b, double tolerance = EPSILON) {
  return std::fabs(a - b) < tolerance;
}

// Check if value is finite
inline bool is_finite_safe(double x) {
  return std::isfinite(x) && !std::isnan(x);
}
} // namespace utils

// ============================================================================
// KLL Sketch structure for approximate quantile computation in streams
// ============================================================================
class KLLSketch {
private:
  struct Item {
    double value;
    int weight;
    
    Item(double v, int w = 1) : value(v), weight(w) {}
    
    bool operator<(const Item &other) const { return value < other.value; }
  };
  
  using Compactor = std::vector<Item>;
  std::vector<Compactor> compactors;
  int k;          // Parameter controlling accuracy
  int n;          // Number of items processed
  double min_value;
  double max_value;
  int max_level;  // Level limit to prevent stack overflow
  
  // Non-recursive function to compact a sketch level
  void compact_level(size_t level) {
    std::queue<size_t> levels_to_compact;
    levels_to_compact.push(level);
    
    while (!levels_to_compact.empty()) {
      size_t current_level = levels_to_compact.front();
      levels_to_compact.pop();
      
      // Check level limit to prevent infinite expansion
      if (current_level >= static_cast<size_t>(max_level)) {
        continue;
      }
      
      // If current level doesn't need compaction, continue
      if (current_level >= compactors.size() ||
          compactors[current_level].size() <= static_cast<size_t>(k)) {
        continue;
      }
      
      // Sort compactor
      std::sort(compactors[current_level].begin(),
                compactors[current_level].end());
      
      // CRITICAL FIX: Ensure next level exists BEFORE taking a reference
      if (current_level + 1 >= compactors.size()) {
        compactors.push_back(Compactor());
      }
      
      // NOW it's safe to take a reference
      Compactor &compactor = compactors[current_level];
      
      // Guard against underflow
      if (compactor.size() < 2) {
        continue;
      }
      
      std::vector<Item> next_level;
      next_level.reserve(compactor.size() / 2 + 1);
      
      // Compaction: merge adjacent pairs
      for (size_t i = 0; i + 1 < compactor.size(); i += 2) {
        bool include = ((current_level % 2 == 0 && i % 2 == 0) ||
                        (current_level % 2 == 1 && i % 2 == 1));
        
        if (include) {
          next_level.push_back(
            Item(compactor[i].value,
                 compactor[i].weight + compactor[i + 1].weight));
        } else {
          next_level.push_back(
            Item(compactor[i + 1].value,
                 compactor[i].weight + compactor[i + 1].weight));
        }
      }
      
      // If there's a remaining element (odd size)
      if (compactor.size() % 2 == 1) {
        next_level.push_back(compactor.back());
      }
      
      // Clear current level
      compactor.clear();
      
      // Add items to next level
      for (const auto &item : next_level) {
        compactors[current_level + 1].push_back(item);
      }
      
      // Add next level to queue if needed
      if (compactors[current_level + 1].size() > static_cast<size_t>(k)) {
        levels_to_compact.push(current_level + 1);
      }
    }
  }
  
public:
  KLLSketch(int k_param = 200)
    : k(std::max(k_param, 10)), // Ensure reasonable minimum k
      n(0),
      min_value(std::numeric_limits<double>::max()),
      max_value(std::numeric_limits<double>::lowest()),
      max_level(20) {
    if (k_param < 10) {
      Rcpp::warning("sketch_k too small, using k=10");
    }
    compactors.push_back(Compactor());
    compactors[0].reserve(k + 1);
  }
  
  // Add a value to the sketch
  void update(double value) {
    // Input validation
    if (!utils::is_finite_safe(value)) {
      Rcpp::warning("Non-finite value ignored in sketch");
      return;
    }
    
    // Update exact min/max
    min_value = std::min(min_value, value);
    max_value = std::max(max_value, value);
    
    // Add to first compactor
    compactors[0].push_back(Item(value));
    n++;
    
    // Compact if necessary
    if (compactors[0].size() > static_cast<size_t>(k)) {
      compact_level(0);
    }
  }
  
  // Estimate the q-th quantile (0 <= q <= 1)
  double get_quantile(double q) const {
    if (n == 0) {
      return 0.0;
    }
    
    // Input validation with safe correction
    if (q <= 0.0) return min_value;
    if (q >= 1.0) return max_value;
    
    // Build flattened representation of sketch for query
    std::vector<Item> flattened;
    flattened.reserve(n / 2);
    
    for (const auto &compactor : compactors) {
      flattened.insert(flattened.end(), compactor.begin(), compactor.end());
    }
    
    // Safety check
    if (flattened.empty()) {
      return min_value;
    }
    
    // Sort items
    std::sort(flattened.begin(), flattened.end());
    
    // Calculate total weight
    int total_weight = 0;
    for (const auto &item : flattened) {
      total_weight += item.weight;
    }
    
    // Safety check
    if (total_weight <= 0) {
      return min_value;
    }
    
    // Find item corresponding to the quantile
    int target_weight = static_cast<int>(q * total_weight);
    int cumulative_weight = 0;
    
    for (const auto &item : flattened) {
      cumulative_weight += item.weight;
      if (cumulative_weight >= target_weight) {
        return item.value;
      }
    }
    
    // Fallback to last item
    return flattened.back().value;
  }
  
  // Returns the number of items seen
  int count() const { return n; }
  
  // Returns the exact minimum value
  double get_min() const { 
    return n > 0 ? min_value : 0.0; 
  }
  
  // Returns the exact maximum value
  double get_max() const { 
    return n > 0 ? max_value : 0.0; 
  }
  
  // Returns compacted items for inspection
  std::vector<double> get_items() const {
    std::vector<double> result;
    result.reserve(n);
    for (const auto &compactor : compactors) {
      for (const auto &item : compactor) {
        result.push_back(item.value);
      }
    }
    std::sort(result.begin(), result.end());
    return result;
  }
};

// ============================================================================
// Structure to store target statistics per cutpoint
// ============================================================================
struct CutpointStats {
  double cutpoint;
  int count_below;
  int count_pos_below;
  int count_neg_below;
  int count_above;
  int count_pos_above;
  int count_neg_above;
  double iv;
  
  CutpointStats(double cp = 0.0)
    : cutpoint(cp), count_below(0), count_pos_below(0), count_neg_below(0),
      count_above(0), count_pos_above(0), count_neg_above(0), iv(0.0) {}
};

// ============================================================================
// Class for optimizing binning with dynamic programming
// ============================================================================
class DynamicProgramming {
private:
  std::vector<double> sorted_values;
  std::vector<int> target_values;
  std::vector<std::vector<double>> dp_table;
  std::vector<std::vector<int>> split_points;
  int total_pos;  // Total events (target=1)
  int total_neg;  // Total non-events (target=0)
  std::unordered_map<std::string, double> iv_cache;
  
  // Function to calculate cache key
  std::string get_cache_key(int i, int j) const {
    return std::to_string(i) + "_" + std::to_string(j);
  }
  
  // Calculate IV of a bin between indices i and j
  double calculate_bin_iv(int i, int j) {
    // Use cache to avoid recalculations
    std::string key = get_cache_key(i, j);
    auto cache_it = iv_cache.find(key);
    if (cache_it != iv_cache.end()) {
      return cache_it->second;
    }
    
    // CRITICAL FIX: Improved bounds checking
    const int n = static_cast<int>(sorted_values.size());
    if (n == 0 || i < 0 || j < i || j >= n) {
      iv_cache[key] = 0.0;
      return 0.0;
    }
    
    int bin_count_pos = 0;
    int bin_count_neg = 0;
    
    for (int k = i; k <= j; ++k) {
      if (target_values[k] == 1) {
        bin_count_pos++;
      } else {
        bin_count_neg++;
      }
    }
    
    // Check for empty bins
    if (bin_count_pos == 0 && bin_count_neg == 0) {
      iv_cache[key] = 0.0;
      return 0.0;
    }
    
    // Calculate proportions (events = target=1, non-events = target=0)
    double prop_event = static_cast<double>(bin_count_pos) / std::max(total_pos, 1);
    double prop_non_event = static_cast<double>(bin_count_neg) / std::max(total_neg, 1);
    
    prop_event = std::max(prop_event, EPSILON);
    prop_non_event = std::max(prop_non_event, EPSILON);
    
    double woe = utils::safe_log(prop_event / prop_non_event);
    double iv = (prop_event - prop_non_event) * woe;
    
    // Store in cache
    iv_cache[key] = std::fabs(iv);
    return iv_cache[key];
  }
  
public:
  DynamicProgramming(const std::vector<double> &values,
                     const std::vector<int> &targets)
    : total_pos(0), total_neg(0) {
    
    // Check for empty vectors
    if (values.empty() || targets.empty()) {
      return;
    }
    
    if (values.size() != targets.size()) {
      throw std::invalid_argument("values and targets must have the same size");
    }
    
    // Create temporary vector of sortable pairs
    std::vector<std::pair<double, int>> paired_data;
    paired_data.reserve(values.size());
    
    for (size_t i = 0; i < values.size(); ++i) {
      paired_data.push_back(std::make_pair(values[i], targets[i]));
    }
    
    // Sort pairs
    std::sort(paired_data.begin(), paired_data.end());
    
    // Fill sorted vectors
    sorted_values.reserve(paired_data.size());
    target_values.reserve(paired_data.size());
    
    for (const auto &pair : paired_data) {
      sorted_values.push_back(pair.first);
      target_values.push_back(pair.second);
      
      if (pair.second == 1) {
        total_pos++;
      } else {
        total_neg++;
      }
    }
  }
  
  // Find the k-1 optimal cutpoints for k bins
  std::vector<double> optimize(int k) {
    const int n = static_cast<int>(sorted_values.size());
    
    // CRITICAL FIX: Stricter safety checks
    if (n <= 1 || k <= 1) {
      return std::vector<double>();
    }
    
    // CRITICAL FIX: Ensure k is properly bounded
    k = std::max(2, std::min(k, std::min(n - 1, 50)));
    
    // Initialize DP table and split points
    try {
      dp_table.assign(n + 1, std::vector<double>(k + 1, -1.0));
      split_points.assign(n + 1, std::vector<int>(k + 1, -1));
    } catch (const std::bad_alloc &e) {
      Rcpp::warning("Memory allocation issue in DP. Using fallback.");
      return fallback_optimize(k);
    }
    
    // Base case: 1 bin
    for (int i = 0; i <= n; ++i) {
      dp_table[i][1] = (i > 0) ? calculate_bin_iv(0, i - 1) : 0.0;
    }
    
    // Fill DP table
    for (int j = 2; j <= k; ++j) {
      for (int i = j; i <= n; ++i) {
        dp_table[i][j] = -1.0;
        
        for (int l = j - 1; l < i; ++l) {
          if (dp_table[l][j - 1] < 0.0) continue; // Skip invalid states
          
          double current_iv = dp_table[l][j - 1] + calculate_bin_iv(l, i - 1);
          
          if (current_iv > dp_table[i][j]) {
            dp_table[i][j] = current_iv;
            split_points[i][j] = l;
          }
        }
      }
    }
    
    // Recover optimal cutpoints
    std::vector<int> optimal_splits;
    int i = n;
    int j = k;
    
    while (j > 1 && i > 0) {
      if (split_points[i][j] < 0) break;
      
      optimal_splits.push_back(split_points[i][j]);
      i = split_points[i][j];
      j--;
    }
    
    // Convert indices to actual cutpoint values
    std::vector<double> cutpoints;
    cutpoints.reserve(optimal_splits.size());
    
    for (int split : optimal_splits) {
      if (split > 0 && split < n) {
        cutpoints.push_back((sorted_values[split - 1] + sorted_values[split]) / 2.0);
      }
    }
    
    std::sort(cutpoints.begin(), cutpoints.end());
    return cutpoints;
  }
  
  // Alternative fallback method when DP fails
  std::vector<double> fallback_optimize(int k) {
    std::vector<double> cutpoints;
    const int n = static_cast<int>(sorted_values.size());
    
    if (n <= 1 || k <= 1) {
      return cutpoints;
    }
    
    k = std::min(k, n - 1);
    const int step = std::max(1, n / k);
    
    for (int i = 1; i < k && (i * step) < n; ++i) {
      cutpoints.push_back(sorted_values[i * step]);
    }
    
    return cutpoints;
  }
};

// ============================================================================
// Main class for optimal numerical binning with sketch
// ============================================================================
class OBN_Sketch {
private:
  std::vector<double> feature;
  std::vector<int> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  bool monotonic;
  double convergence_threshold;
  int max_iterations;
  int sketch_k;
  int dp_size_limit;
  
  int total_pos;  // Total events (target=1)
  int total_neg;  // Total non-events (target=0)
  
  std::unique_ptr<KLLSketch> sketch;
  std::vector<NumericalBin> bins;
  std::vector<double> cutpoints;
  
  // Input validation
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
    if (bin_cutoff <= 0.0 || bin_cutoff >= 1.0) {
      throw std::invalid_argument("bin_cutoff must be between 0 and 1.");
    }
    if (sketch_k < 10) {
      throw std::invalid_argument("sketch_k must be >= 10.");
    }
    if (max_iterations < 1) {
      throw std::invalid_argument("max_iterations must be >= 1.");
    }
    
    // Efficient validation of target values
    bool has_zero = false;
    bool has_one = false;
    
    for (int val : target) {
      if (val == 0) {
        has_zero = true;
      } else if (val == 1) {
        has_one = true;
      } else {
        throw std::invalid_argument("Target must contain only 0 and 1.");
      }
      
      if (has_zero && has_one) break;
    }
    
    if (!has_zero || !has_one) {
      throw std::invalid_argument("Target must contain both 0 and 1.");
    }
  }
  
  // Build KLL sketch
  void build_sketch() {
    sketch = std::make_unique<KLLSketch>(sketch_k);
    
    total_pos = 0;
    total_neg = 0;
    
    for (size_t i = 0; i < feature.size(); ++i) {
      sketch->update(feature[i]);
      
      if (target[i] == 1) {
        total_pos++;
      } else {
        total_neg++;
      }
    }
    
    if (total_pos == 0 || total_neg == 0) {
      throw std::runtime_error("Target must contain both 0 and 1.");
    }
  }
  
  // Extract cutpoint candidates from sketch
  std::vector<double> extract_candidates() {
    if (!sketch || sketch->count() == 0) {
      throw std::runtime_error("Sketch not initialized or empty.");
    }
    
    std::vector<double> candidates;
    candidates.reserve(100);
    
    // Extract quantiles on a finer grid at the extremes
    for (double q = 0.01; q <= 0.1; q += 0.01) {
      candidates.push_back(sketch->get_quantile(q));
      candidates.push_back(sketch->get_quantile(1.0 - q));
    }
    
    // Extract quantiles in the middle of the distribution
    for (double q = 0.1; q <= 0.9; q += 0.05) {
      candidates.push_back(sketch->get_quantile(q));
    }
    
    // Remove duplicates and sort
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    
    // Safety check
    if (candidates.empty()) {
      double min_val = sketch->get_min();
      double max_val = sketch->get_max();
      double range = max_val - min_val;
      
      if (range > EPSILON) {
        for (int i = 1; i < 10; ++i) {
          candidates.push_back(min_val + (range * i / 10.0));
        }
      }
    }
    
    return candidates;
  }
  
  // Calculate target statistics for each cutpoint candidate
  std::vector<CutpointStats> calculate_cutpoint_stats(const std::vector<double> &candidates) {
    std::vector<CutpointStats> stats;
    stats.reserve(candidates.size());
    
    for (double cutpoint : candidates) {
      stats.push_back(CutpointStats(cutpoint));
    }
    
    // Single pass over data
    for (size_t i = 0; i < feature.size(); ++i) {
      double value = feature[i];
      int is_positive = target[i];
      
      for (auto &cs : stats) {
        if (value <= cs.cutpoint) {
          cs.count_below++;
          if (is_positive) {
            cs.count_pos_below++;
          } else {
            cs.count_neg_below++;
          }
        } else {
          cs.count_above++;
          if (is_positive) {
            cs.count_pos_above++;
          } else {
            cs.count_neg_above++;
          }
        }
      }
    }
    
    // Calculate IV for each cutpoint
    for (auto &cs : stats) {
      if (cs.count_below == 0 || cs.count_above == 0) {
        cs.iv = 0.0;
        continue;
      }
      
      // Proportions (events = target=1, non-events = target=0)
      double prop_event_below = static_cast<double>(cs.count_pos_below) / std::max(total_pos, 1);
      double prop_non_event_below = static_cast<double>(cs.count_neg_below) / std::max(total_neg, 1);
      
      prop_event_below = std::max(prop_event_below, EPSILON);
      prop_non_event_below = std::max(prop_non_event_below, EPSILON);
      
      double woe_below = utils::safe_log(prop_event_below / prop_non_event_below);
      double iv_below = (prop_event_below - prop_non_event_below) * woe_below;
      
      double prop_event_above = static_cast<double>(cs.count_pos_above) / std::max(total_pos, 1);
      double prop_non_event_above = static_cast<double>(cs.count_neg_above) / std::max(total_neg, 1);
      
      prop_event_above = std::max(prop_event_above, EPSILON);
      prop_non_event_above = std::max(prop_non_event_above, EPSILON);
      
      double woe_above = utils::safe_log(prop_event_above / prop_non_event_above);
      double iv_above = (prop_event_above - prop_non_event_above) * woe_above;
      
      cs.iv = std::fabs(iv_below) + std::fabs(iv_above);
    }
    
    return stats;
  }
  
  // Select optimal cutpoints using greedy or DP
  void select_optimal_cutpoints(const std::vector<double> &candidates) {
    if (candidates.empty()) {
      Rcpp::warning("No cutpoint candidates available.");
      cutpoints.clear();
      create_initial_bins();
      return;
    }
    
    // For small datasets, use exact dynamic programming
    if (feature.size() <= static_cast<size_t>(dp_size_limit)) {
      try {
        DynamicProgramming dp(feature, target);
        cutpoints = dp.optimize(max_bins);
      } catch (const std::exception &e) {
        Rcpp::warning(std::string("DP error: ") + e.what() + ". Using greedy.");
        cutpoints.clear();
      }
    }
    
    // If DP failed or dataset is large, use greedy approach
    if (cutpoints.empty()) {
      std::vector<CutpointStats> stats = calculate_cutpoint_stats(candidates);
      
      // Sort by IV (descending)
      std::sort(stats.begin(), stats.end(),
                [](const CutpointStats &a, const CutpointStats &b) {
                  return a.iv > b.iv;
                });
      
      // Select max_bins-1 best points
      cutpoints.clear();
      cutpoints.reserve(max_bins - 1);
      
      for (size_t i = 0; i < std::min(stats.size(), static_cast<size_t>(max_bins - 1)); ++i) {
        cutpoints.push_back(stats[i].cutpoint);
      }
      
      std::sort(cutpoints.begin(), cutpoints.end());
    }
    
    create_initial_bins();
  }
  
  // Create initial bins from cutpoints
  void create_initial_bins() {
    bins.clear();
    
    if (!sketch) {
      throw std::runtime_error("Sketch not initialized.");
    }
    
    double min_val = sketch->get_min();
    double max_val = sketch->get_max();
    
    if (cutpoints.empty()) {
      // Single bin case
      NumericalBin bin(min_val, max_val);
      bins.push_back(bin);
    } else {
      // First bin
      bins.push_back(NumericalBin(min_val, cutpoints[0]));
      
      // Intermediate bins
      for (size_t i = 0; i < cutpoints.size() - 1; ++i) {
        bins.push_back(NumericalBin(cutpoints[i], cutpoints[i + 1]));
      }
      
      // Last bin
      bins.push_back(NumericalBin(cutpoints.back(), max_val));
    }
    
    // Fill counts for each bin
    for (size_t i = 0; i < feature.size(); ++i) {
      double value = feature[i];
      int is_positive = target[i];
      
      bool assigned = false;
      
      // CRITICAL FIX: Correct handling of boundary values
      for (size_t b = 0; b < bins.size(); ++b) {
        bool in_bin = false;
        
        if (b == bins.size() - 1) {
          // Last bin: closed interval [lower, upper]
          in_bin = (value >= bins[b].lower_bound && value <= bins[b].upper_bound);
        } else {
          // Other bins: half-open interval [lower, upper)
          in_bin = bins[b].contains(value);
        }
        
        if (in_bin) {
          bins[b].add_value(is_positive);
          assigned = true;
          break;
        }
      }
      
      // Safety fallback
      if (!assigned && !bins.empty()) {
        bins.back().add_value(is_positive);
      }
    }
  }
  
  // Enforce bin_cutoff
  void enforce_bin_cutoff() {
    if (bins.empty()) {
      return;
    }
    
    const int min_count = static_cast<int>(std::ceil(bin_cutoff * static_cast<double>(feature.size())));
    
    bool any_change = true;
    while (any_change && bins.size() > static_cast<size_t>(min_bins)) {
      any_change = false;
      
      for (size_t i = 0; i < bins.size(); ++i) {
        if (bins[i].count < min_count) {
          size_t best_neighbor = SIZE_MAX;
          double min_diff = std::numeric_limits<double>::max();
          
          // Check previous neighbor
          if (i > 0) {
            double diff = std::fabs(bins[i].event_rate() - bins[i - 1].event_rate());
            if (diff < min_diff) {
              min_diff = diff;
              best_neighbor = i - 1;
            }
          }
          
          // Check next neighbor
          if (i + 1 < bins.size()) {
            double diff = std::fabs(bins[i].event_rate() - bins[i + 1].event_rate());
            if (diff < min_diff) {
              min_diff = diff;
              best_neighbor = i + 1;
            }
          }
          
          if (best_neighbor != SIZE_MAX) {
            // CRITICAL FIX: Always merge into the lower index
            if (best_neighbor < i) {
              bins[best_neighbor].merge_with(bins[i]);
              bins.erase(bins.begin() + i);
            } else {
              bins[i].merge_with(bins[best_neighbor]);
              bins.erase(bins.begin() + best_neighbor);
            }
            any_change = true;
            break;
          }
        }
      }
    }
    
    update_cutpoints_from_bins();
  }
  
  // CRITICAL FIX: calculate_metrics expects (total_pos, total_neg)
  // where pos = events (target=1) and neg = non-events (target=0)
  void calculate_initial_woe() {
    for (auto &bin : bins) {
      bin.calculate_metrics(total_pos, total_neg);
    }
  }
  
  // Enforce monotonicity
  void enforce_monotonicity() {
    if (!monotonic || bins.size() <= 1) {
      return;
    }
    
    // Determine monotonicity direction
    bool increasing = bins.back().woe >= bins.front().woe;
    
    // PAVA (Pool Adjacent Violators Algorithm)
    bool any_change = true;
    while (any_change && bins.size() > static_cast<size_t>(min_bins)) {
      any_change = false;
      
      for (size_t i = 0; i + 1 < bins.size(); ++i) {
        bool violation = (increasing && bins[i].woe > bins[i + 1].woe + EPSILON) ||
          (!increasing && bins[i].woe < bins[i + 1].woe - EPSILON);
        
        if (violation) {
          bins[i].merge_with(bins[i + 1]);
          bins.erase(bins.begin() + i + 1);
          bins[i].calculate_metrics(total_pos, total_neg);
          any_change = true;
          break;
        }
      }
    }
    
    update_cutpoints_from_bins();
  }
  
  // Update cutpoints based on current bins
  void update_cutpoints_from_bins() {
    cutpoints.clear();
    cutpoints.reserve(bins.size() - 1);
    
    for (size_t i = 1; i < bins.size(); ++i) {
      cutpoints.push_back(bins[i].lower_bound);
    }
  }
  
  // Optimize bins
  void optimize_bins() {
    if (static_cast<int>(bins.size()) <= max_bins || bins.size() <= 1) {
      return;
    }
    
    int iterations = 0;
    while (static_cast<int>(bins.size()) > max_bins && iterations < max_iterations) {
      if (static_cast<int>(bins.size()) <= min_bins) {
        break;
      }
      
      // Find adjacent bin pair with minimum IV loss
      double min_iv_loss = std::numeric_limits<double>::max();
      size_t merge_idx = 0;
      
      for (size_t i = 0; i + 1 < bins.size(); ++i) {
        double current_iv = std::fabs(bins[i].iv) + std::fabs(bins[i + 1].iv);
        
        // Simulate merge
        NumericalBin merged = bins[i];
        merged.merge_with(bins[i + 1]);
        merged.calculate_metrics(total_pos, total_neg);
        
        double merged_iv = std::fabs(merged.iv);
        double iv_loss = current_iv - merged_iv;
        
        if (iv_loss < min_iv_loss) {
          min_iv_loss = iv_loss;
          merge_idx = i;
        }
      }
      
      // Merge bins
      bins[merge_idx].merge_with(bins[merge_idx + 1]);
      bins[merge_idx].calculate_metrics(total_pos, total_neg);
      bins.erase(bins.begin() + merge_idx + 1);
      
      iterations++;
      
      if (static_cast<int>(bins.size()) <= max_bins) {
        break;
      }
    }
    
    update_cutpoints_from_bins();
  }
  
  // Consistency check
  void check_consistency() const {
    if (bins.empty()) {
      Rcpp::warning("No bins created.");
      return;
    }
    
    int total_count = 0;
    int total_count_pos = 0;
    int total_count_neg = 0;
    
    for (const auto &bin : bins) {
      total_count += bin.count;
      total_count_pos += bin.count_pos;
      total_count_neg += bin.count_neg;
    }
    
    const double count_tolerance = 0.05;
    
    double count_ratio = static_cast<double>(total_count) / static_cast<double>(feature.size());
    if (std::fabs(count_ratio - 1.0) > count_tolerance) {
      Rcpp::warning("Inconsistency in total count. Ratio: " + std::to_string(count_ratio));
    }
    
    double pos_ratio = static_cast<double>(total_count_pos) / static_cast<double>(total_pos);
    double neg_ratio = static_cast<double>(total_count_neg) / static_cast<double>(total_neg);
    
    if (std::fabs(pos_ratio - 1.0) > count_tolerance || std::fabs(neg_ratio - 1.0) > count_tolerance) {
      Rcpp::warning("Inconsistency in pos/neg counts. Pos ratio: " + 
        std::to_string(pos_ratio) + ", Neg ratio: " + std::to_string(neg_ratio));
    }
  }
  
public:
  // Constructor
  OBN_Sketch(const std::vector<double> &feature_,
             const std::vector<int> &target_,
             int min_bins_ = 3,
             int max_bins_ = 5,
             double bin_cutoff_ = 0.05,
             int max_n_prebins_ = 20,
             bool monotonic_ = true,
             double convergence_threshold_ = 1e-6,
             int max_iterations_ = 1000,
             int sketch_k_ = 200)
    : feature(feature_),
      target(target_),
      min_bins(min_bins_),
      max_bins(max_bins_),
      bin_cutoff(bin_cutoff_),
      max_n_prebins(max_n_prebins_),
      monotonic(monotonic_),
      convergence_threshold(convergence_threshold_),
      max_iterations(max_iterations_),
      sketch_k(sketch_k_),
      dp_size_limit(50),
      total_pos(0),
      total_neg(0) {
    
    bins.reserve(max_bins_);
    cutpoints.reserve(max_bins_ - 1);
  }
  
  // Fit method
  Rcpp::List fit() {
    try {
      // Binning pipeline
      validate_inputs();
      build_sketch();
      std::vector<double> candidates = extract_candidates();
      select_optimal_cutpoints(candidates);
      enforce_bin_cutoff();
      calculate_initial_woe();
      enforce_monotonicity();
      
      // Optimization
      bool converged_flag = false;
      int iterations_done = 0;
      
      if (static_cast<int>(bins.size()) <= max_bins) {
        converged_flag = true;
      } else {
        double prev_total_iv = 0.0;
        for (const auto &bin : bins) {
          prev_total_iv += std::fabs(bin.iv);
        }
        
        for (int i = 0; i < max_iterations; ++i) {
          size_t start_bins = bins.size();
          
          optimize_bins();
          
          if (bins.size() == start_bins || static_cast<int>(bins.size()) <= max_bins) {
            double total_iv = 0.0;
            for (const auto &bin : bins) {
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
            converged_flag = true;
            break;
          }
        }
      }
      
      // Final consistency check
      check_consistency();
      
      if (bins.empty()) {
        return Rcpp::List::create(Named("error") = "Failed to create bins.");
      }
      
      // Prepare results
      const size_t n_bins = bins.size();
      
      NumericVector bin_lower(n_bins);
      NumericVector bin_upper(n_bins);
      NumericVector bin_woe(n_bins);
      NumericVector bin_iv(n_bins);
      IntegerVector bin_count(n_bins);
      IntegerVector bin_count_pos(n_bins);
      IntegerVector bin_count_neg(n_bins);
      NumericVector ids(n_bins);
      
      for (size_t i = 0; i < n_bins; ++i) {
        bin_lower[i] = bins[i].lower_bound;
        bin_upper[i] = bins[i].upper_bound;
        bin_woe[i] = bins[i].woe;
        bin_iv[i] = bins[i].iv;
        bin_count[i] = bins[i].count;
        bin_count_pos[i] = bins[i].count_pos;
        bin_count_neg[i] = bins[i].count_neg;
        ids[i] = static_cast<double>(i + 1);
      }
      
      return Rcpp::List::create(
        Named("id") = ids,
        Named("bin_lower") = bin_lower,
        Named("bin_upper") = bin_upper,
        Named("woe") = bin_woe,
        Named("iv") = bin_iv,
        Named("count") = bin_count,
        Named("count_pos") = bin_count_pos,
        Named("count_neg") = bin_count_neg,
        Named("cutpoints") = cutpoints,
        Named("converged") = converged_flag,
        Named("iterations") = iterations_done);
      
    } catch (const std::exception &e) {
      Rcpp::stop("Binning error: " + std::string(e.what()));
    }
  }
};

// ============================================================================
// Exported function for R
// ============================================================================
// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_sketch(
    Rcpp::IntegerVector target,
    Rcpp::NumericVector feature,
    int min_bins = 3,
    int max_bins = 5,
    double bin_cutoff = 0.05,
    int max_n_prebins = 20,
    bool monotonic = true,
    double convergence_threshold = 1e-6,
    int max_iterations = 1000,
    int sketch_k = 200) {
  
  // Preliminary checks
  if (feature.size() == 0 || target.size() == 0) {
    Rcpp::stop("Feature and target cannot be empty.");
  }
  
  if (feature.size() != target.size()) {
    Rcpp::stop("Feature and target must have the same size.");
  }
  
  // Conversion and validation
  std::vector<double> feature_vec;
  std::vector<int> target_vec;
  
  feature_vec.reserve(feature.size());
  target_vec.reserve(target.size());
  
  for (R_xlen_t i = 0; i < feature.size(); ++i) {
    if (NumericVector::is_na(feature[i])) {
      Rcpp::stop("Feature cannot contain missing values (NA).");
    }
    if (!utils::is_finite_safe(feature[i])) {
      Rcpp::stop("Feature contains non-finite values (Inf/NaN).");
    }
    feature_vec.push_back(feature[i]);
  }
  
  for (R_xlen_t i = 0; i < target.size(); ++i) {
    if (IntegerVector::is_na(target[i])) {
      Rcpp::stop("Target cannot contain missing values (NA).");
    }
    if (target[i] != 0 && target[i] != 1) {
      Rcpp::stop("Target must contain only 0 and 1.");
    }
    target_vec.push_back(target[i]);
  }
  
  // Check for constant values
  double min_val = *std::min_element(feature_vec.begin(), feature_vec.end());
  double max_val = *std::max_element(feature_vec.begin(), feature_vec.end());
  
  if (utils::are_equal(min_val, max_val)) {
    Rcpp::warning("Feature has constant value, creating a single bin.");
    
    int count_pos = 0;
    int count_neg = 0;
    for (int t : target_vec) {
      if (t == 1) count_pos++;
      else count_neg++;
    }
    
    const int total_count = static_cast<int>(feature_vec.size());
    
    // Calculate WoE and IV
    double prop_event = static_cast<double>(count_pos) / std::max(count_pos, 1);
    double prop_non_event = static_cast<double>(count_neg) / std::max(count_neg, 1);
    
    prop_event = std::max(prop_event, EPSILON);
    prop_non_event = std::max(prop_non_event, EPSILON);
    
    double woe = utils::safe_log(prop_event / prop_non_event);
    double iv = (prop_event - prop_non_event) * woe;
    
    return Rcpp::List::create(
      Named("id") = NumericVector::create(1),
      Named("bin_lower") = NumericVector::create(min_val),
      Named("bin_upper") = NumericVector::create(max_val),
      Named("woe") = NumericVector::create(woe),
      Named("iv") = NumericVector::create(iv),
      Named("count") = IntegerVector::create(total_count),
      Named("count_pos") = IntegerVector::create(count_pos),
      Named("count_neg") = IntegerVector::create(count_neg),
      Named("cutpoints") = NumericVector::create(),
      Named("converged") = true,
      Named("iterations") = 0);
  }
  
  // Execute algorithm
  OBN_Sketch sketch_binner(feature_vec, target_vec, min_bins, max_bins,
                           bin_cutoff, max_n_prebins, monotonic,
                           convergence_threshold, max_iterations, sketch_k);
  
  return sketch_binner.fit();
}
