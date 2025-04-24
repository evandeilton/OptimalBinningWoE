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

using namespace Rcpp;

// Global constants for better readability and consistency
static constexpr double EPSILON = 1e-10;
static constexpr double NEG_INFINITY = -std::numeric_limits<double>::infinity();
// Bayesian smoothing parameter (adjustable prior strength)
static constexpr double BAYESIAN_PRIOR_STRENGTH = 0.5;

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
}

// Enhanced Bin structure with Bayesian smoothing and uniqueness guarantee
struct Bin {
  std::unordered_set<std::string> category_set; // Set to ensure uniqueness
  std::vector<std::string> categories;          // Vector to maintain order
  double woe;
  double iv;
  int count;
  int count_pos;
  int count_neg;
  double event_rate;
  
  Bin() : woe(0), iv(0), count(0), count_pos(0), count_neg(0), event_rate(0) {
    categories.reserve(8);
  }
  
  // Method to add category, ensuring uniqueness
  inline void add_category(const std::string& cat, int is_positive) {
    // Add only if it's a new category
    if (category_set.insert(cat).second) {
      categories.push_back(cat);
    }
    
    count++;
    if (is_positive) {
      count_pos++;
    } else {
      count_neg++;
    }
    update_event_rate();
  }
  
  // Method to merge with another bin, ensuring uniqueness
  inline void merge_with(const Bin& other) {
    // Add only categories that don't already exist
    for (const auto& cat : other.categories) {
      if (category_set.insert(cat).second) {
        categories.push_back(cat);
      }
    }
    
    count += other.count;
    count_pos += other.count_pos;
    count_neg += other.count_neg;
    update_event_rate();
  }
  
  // Update event rate
  inline void update_event_rate() {
    event_rate = count > 0 ? static_cast<double>(count_pos) / count : 0.0;
  }
  
  // Calculate WoE and IV with Bayesian smoothing
  inline void calculate_metrics(int total_good, int total_bad) {
    // Calculate Bayesian prior based on overall prevalence
    double prior_weight = BAYESIAN_PRIOR_STRENGTH;
    double overall_event_rate = static_cast<double>(total_bad) / 
      (total_bad + total_good);
    
    double prior_pos = prior_weight * overall_event_rate;
    double prior_neg = prior_weight * (1.0 - overall_event_rate);
    
    // Apply Bayesian smoothing to proportions
    double prop_event = static_cast<double>(count_pos + prior_pos) / 
      static_cast<double>(total_bad + prior_weight);
    double prop_non_event = static_cast<double>(count_neg + prior_neg) / 
      static_cast<double>(total_good + prior_weight);
    
    // Ensure numerical stability
    prop_event = std::max(prop_event, EPSILON);
    prop_non_event = std::max(prop_non_event, EPSILON);
    
    // Calculate WoE and IV
    woe = utils::safe_log(prop_event / prop_non_event);
    iv = (prop_event - prop_non_event) * woe;
    
    // Handle non-finite values
    if (!std::isfinite(woe)) woe = 0.0;
    if (!std::isfinite(iv)) iv = 0.0;
  }
};

// Enhanced cache for potential merges
class MergeCache {
private:
  std::vector<std::vector<double>> iv_loss_cache;
  bool enabled;
  
public:
  MergeCache(size_t max_size, bool use_cache = true) : enabled(use_cache) {
    if (enabled && max_size > 0) {
      iv_loss_cache.resize(max_size);
      for (auto& row : iv_loss_cache) {
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
  
  inline void invalidate_bin(size_t bin_idx) {
    if (!enabled || bin_idx >= iv_loss_cache.size()) {
      return;
    }
    
    for (size_t i = 0; i < iv_loss_cache.size(); ++i) {
      if (i < iv_loss_cache[bin_idx].size()) {
        iv_loss_cache[bin_idx][i] = -1.0;
      }
      if (bin_idx < iv_loss_cache[i].size()) {
        iv_loss_cache[i][bin_idx] = -1.0;
      }
    }
  }
  
  inline void resize(size_t new_size) {
    if (!enabled) return;
    
    iv_loss_cache.resize(new_size);
    for (auto& row : iv_loss_cache) {
      row.resize(new_size, -1.0);
    }
  }
};

// Enhanced main class with improved optimization strategies
class OptimalBinningCategoricalMBA {
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
  
  int total_good;
  int total_bad;
  
  std::vector<Bin> bins;
  std::unique_ptr<MergeCache> merge_cache;
  
  // Enhanced input validation with comprehensive checks
  void validate_inputs() {
    if (feature.size() != target.size()) {
      throw std::invalid_argument("Feature and target must have the same length.");
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
    
    // Check for empty strings in feature
    if (std::any_of(feature.begin(), feature.end(), [](const std::string& s) { 
      return s.empty(); 
    })) {
      throw std::invalid_argument("Feature cannot contain empty strings. Consider preprocessing your data.");
    }
    
    // Efficient check for binary target values
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
      throw std::invalid_argument("Target must contain both 0 and 1 values.");
    }
    
    // Adjust max_bins based on unique categories
    std::unordered_set<std::string> unique_categories(feature.begin(), feature.end());
    int ncat = static_cast<int>(unique_categories.size());
    if (max_bins > ncat) {
      max_bins = ncat;
    }
    
    // Adjust min_bins if necessary
    min_bins = std::min(min_bins, max_bins);
  }
  
  // Enhanced prebinning with better statistical handling
  void prebinning() {
    // Efficient single-pass counting
    std::unordered_map<std::string, Bin> category_bins;
    category_bins.reserve(std::min(feature.size() / 4, static_cast<size_t>(1024)));
    
    total_good = 0;
    total_bad = 0;
    
    for (size_t i = 0; i < feature.size(); ++i) {
      const auto& cat = feature[i];
      int is_positive = target[i];
      
      if (category_bins.find(cat) == category_bins.end()) {
        category_bins[cat] = Bin();
      }
      
      category_bins[cat].add_category(cat, is_positive);
      
      if (is_positive) {
        total_bad++;
      } else {
        total_good++;
      }
    }
    
    // Check for extremely imbalanced datasets
    if (total_good < 5 || total_bad < 5) {
      Rcpp::warning("Dataset has fewer than 5 samples in one class. Results may be unstable.");
    }
    
    // Transfer to vector and sort by count (descending)
    bins.clear();
    bins.reserve(category_bins.size());
    
    for (auto& pair : category_bins) {
      bins.push_back(std::move(pair.second));
    }
    
    std::sort(bins.begin(), bins.end(),
              [](const Bin& a, const Bin& b) { return a.count > b.count; });
    
    // Initialize merge cache
    merge_cache = std::make_unique<MergeCache>(bins.size(), bins.size() > 10);
    
    // Reduce to max_n_prebins if necessary with improved strategy
    if (static_cast<int>(bins.size()) > max_n_prebins && static_cast<int>(bins.size()) > min_bins) {
      // First try merging smallest bins to preserve information
      std::vector<size_t> small_bin_indices;
      small_bin_indices.reserve(bins.size() - max_n_prebins);
      
      // Identify bins to potentially merge
      for (size_t i = max_n_prebins; i < bins.size(); ++i) {
        small_bin_indices.push_back(i);
      }
      
      // Sort by count (ascending)
      std::sort(small_bin_indices.begin(), small_bin_indices.end(),
                [this](size_t a, size_t b) { 
                  return bins[a].count < bins[b].count; 
                });
      
      // Iteratively merge smallest bins
      for (size_t idx : small_bin_indices) {
        if (static_cast<int>(bins.size()) <= max_n_prebins || 
            static_cast<int>(bins.size()) <= min_bins) {
          break;
        }
        
        // Find best merge candidate based on similarity
        double best_similarity = -1.0;
        size_t best_candidate = 0;
        
        for (size_t j = 0; j < bins.size(); ++j) {
          if (j == idx) continue;
          
          // Calculate event rate similarity as a merge criterion
          double rate_diff = std::fabs(bins[idx].event_rate - bins[j].event_rate);
          double similarity = 1.0 / (rate_diff + EPSILON);
          
          if (similarity > best_similarity) {
            best_similarity = similarity;
            best_candidate = j;
          }
        }
        
        // Try to merge with best candidate
        if (best_candidate < bins.size()) {
          size_t merge_idx1 = std::min(idx, best_candidate);
          size_t merge_idx2 = std::max(idx, best_candidate);
          
          // Adjust index if it's out of bounds after previous merges
          if (merge_idx1 >= bins.size()) merge_idx1 = bins.size() - 1;
          if (merge_idx2 >= bins.size()) merge_idx2 = bins.size() - 1;
          
          if (merge_idx1 != merge_idx2) {
            try_merge_bins(merge_idx1, merge_idx2);
          }
        }
      }
    }
  }
  
  // Enhanced bin_cutoff enforcement with improved handling
  void enforce_bin_cutoff() {
    // Calculate minimum counts based on bin_cutoff
    int min_count = static_cast<int>(std::ceil(bin_cutoff * static_cast<double>(feature.size())));
    int min_count_pos = static_cast<int>(std::ceil(bin_cutoff * static_cast<double>(total_bad)));
    
    // Identify all low-frequency bins at once
    std::vector<size_t> low_freq_bins;
    
    for (size_t i = 0; i < bins.size(); ++i) {
      if (bins[i].count < min_count || bins[i].count_pos < min_count_pos) {
        low_freq_bins.push_back(i);
      }
    }
    
    // Sort bins by frequency (ascending) for better merging strategy
    std::sort(low_freq_bins.begin(), low_freq_bins.end(),
              [this](size_t a, size_t b) { 
                return bins[a].count < bins[b].count; 
              });
    
    // Process low-frequency bins efficiently
    for (size_t idx : low_freq_bins) {
      if (static_cast<int>(bins.size()) <= min_bins) {
        break; // Never go below min_bins
      }
      
      // Check if bin still exists and is still below cutoff
      if (idx >= bins.size() || (bins[idx].count >= min_count && bins[idx].count_pos >= min_count_pos)) {
        continue;
      }
      
      // Find best merge candidate based on event rate similarity
      double best_similarity = -1.0;
      size_t best_candidate = idx;
      
      for (size_t j = 0; j < bins.size(); ++j) {
        if (j == idx) continue;
        
        // Calculate event rate similarity
        double rate_diff = std::fabs(bins[idx].event_rate - bins[j].event_rate);
        double similarity = 1.0 / (rate_diff + EPSILON);
        
        if (similarity > best_similarity) {
          best_similarity = similarity;
          best_candidate = j;
        }
      }
      
      // Try to merge with best candidate
      if (best_candidate != idx && best_candidate < bins.size()) {
        if (!try_merge_bins(std::min(idx, best_candidate), std::max(idx, best_candidate))) {
          // If unable to merge with best candidate, try adjacent bins
          if (idx > 0) {
            try_merge_bins(idx - 1, idx);
          } else if (idx + 1 < bins.size()) {
            try_merge_bins(idx, idx + 1);
          }
        }
      }
      
      // Adjust indices for remaining bins, since we removed one
      for (auto& remaining_idx : low_freq_bins) {
        if (remaining_idx > std::min(idx, best_candidate)) {
          remaining_idx--;
        }
      }
    }
  }
  
  // Enhanced initial WoE calculation
  void calculate_initial_woe() {
    for (auto& bin : bins) {
      bin.calculate_metrics(total_good, total_bad);
    }
  }
  
  // Enhanced monotonicity enforcement with adaptive thresholds
  void enforce_monotonicity() {
    if (bins.empty()) {
      throw std::runtime_error("No bins available to enforce monotonicity.");
    }
    
    // Sort bins by WoE
    std::sort(bins.begin(), bins.end(),
              [](const Bin& a, const Bin& b) { return a.woe < b.woe; });
    
    // Calculate average WoE gap for adaptive threshold
    double total_woe_gap = 0.0;
    if (bins.size() > 1) {
      for (size_t i = 1; i < bins.size(); ++i) {
        total_woe_gap += std::fabs(bins[i].woe - bins[i-1].woe);
      }
    }
    
    double avg_gap = bins.size() > 1 ? total_woe_gap / (bins.size() - 1) : 0.0;
    double monotonicity_threshold = std::min(EPSILON, avg_gap * 0.01);
    
    // Determine monotonicity direction
    bool increasing = true;
    if (bins.size() > 1) {
      for (size_t i = 1; i < bins.size(); ++i) {
        if (bins[i].woe < bins[i-1].woe - monotonicity_threshold) {
          increasing = false;
          break;
        }
      }
    }
    
    // Optimized loop to enforce monotonicity
    bool any_merge;
    int attempts = 0;
    const int max_attempts = static_cast<int>(bins.size() * 3); // Safety limit
    
    do {
      any_merge = false;
      
      // Find all violations and their severity
      std::vector<std::pair<size_t, double>> violations;
      
      for (size_t i = 0; i + 1 < bins.size(); ++i) {
        bool violation = (increasing && bins[i].woe > bins[i+1].woe + monotonicity_threshold) ||
          (!increasing && bins[i].woe < bins[i+1].woe - monotonicity_threshold);
        
        if (violation) {
          double severity = std::fabs(bins[i].woe - bins[i+1].woe);
          violations.push_back({i, severity});
        }
      }
      
      // If no violations, we're done
      if (violations.empty()) break;
      
      // Sort violations by severity (descending)
      std::sort(violations.begin(), violations.end(),
                [](const auto& a, const auto& b) { return a.second > b.second; });
      
      // Fix the most severe violation
      if (!violations.empty() && static_cast<int>(bins.size()) > min_bins) {
        size_t idx = violations[0].first;
        if (try_merge_bins(idx, idx + 1)) {
          any_merge = true;
        }
      }
      
      attempts++;
      if (attempts >= max_attempts) {
        Rcpp::warning("Could not ensure monotonicity in %d attempts. Using best solution found.", max_attempts);
        break;
      }
      
    } while (any_merge && static_cast<int>(bins.size()) > min_bins);
    
    // Final sort by WoE to ensure correct order
    std::sort(bins.begin(), bins.end(),
              [](const Bin& a, const Bin& b) { return a.woe < b.woe; });
  }
  
  // Enhanced bin optimization with improved IV-based strategy
  void optimize_bins() {
    if (static_cast<int>(bins.size()) <= max_bins) {
      return; // Already within maximum limit
    }
    
    // Track best solution seen so far
    double best_total_iv = 0.0;
    std::vector<Bin> best_bins = bins;
    
    for (const auto& bin : bins) {
      best_total_iv += std::fabs(bin.iv);
    }
    
    int iterations = 0;
    double prev_total_iv = best_total_iv;
    
    while (static_cast<int>(bins.size()) > max_bins && iterations < max_iterations) {
      if (static_cast<int>(bins.size()) <= min_bins) {
        break; // Don't reduce below min_bins
      }
      
      // Find pair of bins with minimum combined IV or minimum IV loss
      double min_iv_loss = std::numeric_limits<double>::max();
      size_t min_iv_index = 0;
      
      // Optimized search with cache
      for (size_t i = 0; i < bins.size() - 1; ++i) {
        // Try to merge each adjacent pair
        double cached_iv_loss = merge_cache->get_iv_loss(i, i+1);
        double iv_loss;
        
        if (cached_iv_loss >= 0.0) {
          iv_loss = cached_iv_loss;
        } else {
          // Simulate the merge to calculate IV loss
          Bin merged_bin = bins[i];
          merged_bin.merge_with(bins[i+1]);
          merged_bin.calculate_metrics(total_good, total_bad);
          
          double original_iv = std::fabs(bins[i].iv) + std::fabs(bins[i+1].iv);
          double new_iv = std::fabs(merged_bin.iv);
          iv_loss = original_iv - new_iv;
          
          merge_cache->set_iv_loss(i, i+1, iv_loss);
        }
        
        if (iv_loss < min_iv_loss) {
          min_iv_loss = iv_loss;
          min_iv_index = i;
        }
      }
      
      // Try to merge bins with minimum IV loss
      if (!try_merge_bins(min_iv_index, min_iv_index + 1)) {
        break; // Unable to merge, stop
      }
      
      // Calculate new total IV
      double total_iv = 0.0;
      for (const auto& bin : bins) {
        total_iv += std::fabs(bin.iv);
      }
      
      // Track best solution
      if (static_cast<int>(bins.size()) <= max_bins && total_iv > best_total_iv) {
        best_total_iv = total_iv;
        best_bins = bins;
      }
      
      // Check for convergence
      if (std::fabs(total_iv - prev_total_iv) < convergence_threshold) {
        break; // Convergence achieved
      }
      
      prev_total_iv = total_iv;
      iterations++;
    }
    
    // Restore best solution if we found a better one
    if (best_total_iv > 0.0 && 
        static_cast<int>(best_bins.size()) <= max_bins && 
        static_cast<int>(best_bins.size()) >= min_bins) {
      bins = std::move(best_bins);
    }
    
    if (static_cast<int>(bins.size()) > max_bins) {
      Rcpp::warning("Could not reduce number of bins to max_bins without violating min_bins or convergence constraints.");
    }
  }
  
  // Enhanced bin merging with improved checks
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
  
  // Enhanced consistency check
  void check_consistency() const {
    int total_count = 0;
    int total_count_pos = 0;
    int total_count_neg = 0;
    
    for (const auto& bin : bins) {
      total_count += bin.count;
      total_count_pos += bin.count_pos;
      total_count_neg += bin.count_neg;
    }
    
    if (static_cast<size_t>(total_count) != feature.size()) {
      throw std::runtime_error("Inconsistent total count after binning.");
    }
    
    if (total_count_pos != total_bad || total_count_neg != total_good) {
      throw std::runtime_error("Inconsistent positive/negative counts after binning.");
    }
  }
  
public:
  // Enhanced constructor with improved parameter handling
  OptimalBinningCategoricalMBA(
    const std::vector<std::string>& feature_,
    const Rcpp::IntegerVector& target_,
    int min_bins_ = 3,
    int max_bins_ = 5,
    double bin_cutoff_ = 0.05,
    int max_n_prebins_ = 20,
    std::string bin_separator_ = "%;%",
    double convergence_threshold_ = 1e-6,
    int max_iterations_ = 1000
  ) : feature(feature_), target(Rcpp::as<std::vector<int>>(target_)), 
  min_bins(min_bins_), max_bins(max_bins_), 
  bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_),
  bin_separator(bin_separator_),
  convergence_threshold(convergence_threshold_),
  max_iterations(max_iterations_),
  total_good(0), total_bad(0) {
    
    // Pre-allocation for better performance
    bins.reserve(std::min(feature.size() / 4, static_cast<size_t>(1024)));
  }
  
  // Enhanced fit method with better algorithm flow
  Rcpp::List fit() {
    try {
      // Binning process with performance checks
      validate_inputs();
      prebinning();
      enforce_bin_cutoff();
      calculate_initial_woe();
      enforce_monotonicity();
      
      // Bin optimization if necessary
      bool converged_flag = false;
      int iterations_done = 0;
      
      if (static_cast<int>(bins.size()) <= max_bins) {
        converged_flag = true;
      } else {
        double prev_total_iv = 0.0;
        for (const auto& bin : bins) {
          prev_total_iv += std::fabs(bin.iv);
        }
        
        // Track best solution
        double best_total_iv = prev_total_iv;
        std::vector<Bin> best_bins = bins;
        
        for (int i = 0; i < max_iterations; ++i) {
          // Starting point for this iteration
          size_t start_bins = bins.size();
          
          optimize_bins();
          
          // Calculate current IV
          double total_iv = 0.0;
          for (const auto& bin : bins) {
            total_iv += std::fabs(bin.iv);
          }
          
          // Track best solution
          if (static_cast<int>(bins.size()) <= max_bins && 
              static_cast<int>(bins.size()) >= min_bins && 
              total_iv > best_total_iv) {
            best_total_iv = total_iv;
            best_bins = bins;
          }
          
          // Check for convergence or no change
          if (bins.size() == start_bins || static_cast<int>(bins.size()) <= max_bins) {
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
        
        // Restore best solution if found
        if (best_total_iv > 0.0 && 
            static_cast<int>(best_bins.size()) <= max_bins && 
            static_cast<int>(best_bins.size()) >= min_bins) {
          bins = std::move(best_bins);
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
      
      double total_iv = 0.0;
      
      for (size_t i = 0; i < n_bins; ++i) {
        // Use utils::join which ensures unique values
        bin_names[i] = utils::join(bins[i].categories, bin_separator);
        bin_woe[i] = bins[i].woe;
        bin_iv[i] = bins[i].iv;
        bin_count[i] = bins[i].count;
        bin_count_pos[i] = bins[i].count_pos;
        bin_count_neg[i] = bins[i].count_neg;
        ids[i] = i + 1;
        
        total_iv += std::fabs(bins[i].iv);
      }
      
      return Rcpp::List::create(
        Named("id") = ids,
        Named("bin") = bin_names,
        Named("woe") = bin_woe,
        Named("iv") = bin_iv,
        Named("count") = bin_count,
        Named("count_pos") = bin_count_pos,
        Named("count_neg") = bin_count_neg,
        Named("total_iv") = total_iv,
        Named("converged") = converged_flag,
        Named("iterations") = iterations_done
      );
    } catch (const std::exception& e) {
      Rcpp::stop("Error in optimal binning: %s", e.what());
    }
  }
};

//' @title Optimal Binning for Categorical Variables using Monotonic Binning Algorithm (MBA)
//'
//' @description
//' Performs optimal binning for categorical variables using a Monotonic Binning Algorithm (MBA),
//' which combines Weight of Evidence (WOE) and Information Value (IV) methods with monotonicity
//' constraints. This implementation includes Bayesian smoothing for robust estimation with small
//' samples, adaptive monotonicity enforcement, and efficient handling of rare categories.
//'
//' @param feature A character vector of categorical feature values.
//' @param target An integer vector of binary target values (0 or 1).
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum frequency for a category to be considered as a separate bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins before merging (default: 20).
//' @param bin_separator String used to separate category names when merging bins (default: "%;%").
//' @param convergence_threshold Threshold for convergence in optimization (default: 1e-6).
//' @param max_iterations Maximum number of iterations for optimization (default: 1000).
//'
//' @return A list containing:
//' \itemize{
//'   \item id: Numeric vector of bin identifiers.
//'   \item bin: Character vector of bin labels.
//'   \item woe: Numeric vector of Weight of Evidence values for each bin.
//'   \item iv: Numeric vector of Information Value for each bin.
//'   \item count: Integer vector of total counts for each bin.
//'   \item count_pos: Integer vector of positive target counts for each bin.
//'   \item count_neg: Integer vector of negative target counts for each bin.
//'   \item total_iv: Total Information Value of the binning.
//'   \item converged: Logical value indicating whether the algorithm converged.
//'   \item iterations: Integer indicating the number of iterations run.
//' }
//'
//' @details
//' This algorithm implements an enhanced version of the monotonic binning approach with several
//' key features:
//' 
//' \enumerate{
//'   \item \strong{Bayesian Smoothing:} Applies prior pseudo-counts proportional to the overall class
//'         prevalence to improve stability for small bins and rare categories.
//'   \item \strong{Adaptive Monotonicity:} Uses context-aware thresholds based on the average WoE
//'         difference between bins to better handle datasets with varying scales.
//'   \item \strong{Similarity-Based Merging:} Merges bins based on event rate similarity rather than
//'         just adjacency, which better preserves information content.
//'   \item \strong{Best Solution Tracking:} Maintains the best solution found during optimization,
//'         even if the algorithm doesn't formally converge.
//' }
//'
//' The mathematical foundation of the algorithm is based on the following concepts:
//' 
//' The Weight of Evidence (WoE) with Bayesian smoothing is calculated as:
//' 
//' \deqn{WoE_i = \ln\left(\frac{p_i^*}{q_i^*}\right)}
//' 
//' where:
//' \itemize{
//'   \item \eqn{p_i^* = \frac{n_i^+ + \alpha \cdot \pi}{N^+ + \alpha}} is the smoothed proportion of
//'         positive cases in bin i
//'   \item \eqn{q_i^* = \frac{n_i^- + \alpha \cdot (1-\pi)}{N^- + \alpha}} is the smoothed proportion of
//'         negative cases in bin i
//'   \item \eqn{\pi = \frac{N^+}{N^+ + N^-}} is the overall positive rate
//'   \item \eqn{\alpha} is the prior strength parameter (default: 0.5)
//'   \item \eqn{n_i^+} is the count of positive cases in bin i
//'   \item \eqn{n_i^-} is the count of negative cases in bin i
//'   \item \eqn{N^+} is the total number of positive cases
//'   \item \eqn{N^-} is the total number of negative cases
//' }
//'
//' The Information Value (IV) for each bin is calculated as:
//' 
//' \deqn{IV_i = (p_i^* - q_i^*) \times WoE_i}
//'
//' And the total IV is:
//' 
//' \deqn{IV_{total} = \sum_{i=1}^{k} |IV_i|}
//'
//' The algorithm performs the following steps:
//' \enumerate{
//'   \item Input validation and preprocessing
//'   \item Initial pre-binning based on frequency
//'   \item Merging of rare categories based on bin_cutoff
//'   \item Calculation of WoE and IV with Bayesian smoothing
//'   \item Enforcement of monotonicity constraints
//'   \item Optimization of bin count through iterative merging
//' }
//'
//' @examples
//' \dontrun{
//' # Create sample data
//' set.seed(123)
//' target <- sample(0:1, 1000, replace = TRUE)
//' feature <- sample(LETTERS[1:5], 1000, replace = TRUE)
//'
//' # Run optimal binning
//' result <- optimal_binning_categorical_mba(feature, target)
//'
//' # View results
//' print(result)
//'
//' # Handle rare categories more aggressively
//' result2 <- optimal_binning_categorical_mba(
//'   feature, target, 
//'   bin_cutoff = 0.1, 
//'   min_bins = 2, 
//'   max_bins = 4
//' )
//' }
//'
//' @references
//' \itemize{
//'   \item Beltrami, M., Mach, M., & Dall'Aglio, M. (2021). Monotonic Optimal Binning Algorithm for Credit Risk Modeling. Risks, 9(3), 58.
//'   \item Siddiqi, N. (2006). Credit risk scorecards: developing and implementing intelligent credit scoring (Vol. 3). John Wiley & Sons.
//'   \item Mironchyk, P., & Tchistiakov, V. (2017). Monotone Optimal Binning Algorithm for Credit Risk Modeling. Working Paper.
//'   \item Gelman, A., Jakulin, A., Pittau, M. G., & Su, Y. S. (2008). A weakly informative default prior distribution for logistic and other regression models. The annals of applied statistics, 2(4), 1360-1383.
//'   \item Thomas, L.C., Edelman, D.B., & Crook, J.N. (2002). Credit Scoring and its Applications. SIAM.
//'   \item Navas-Palencia, G. (2020). Optimal binning: mathematical programming formulations for binary classification. arXiv preprint arXiv:2001.08025.
//'   \item Lin, X., Wang, G., & Zhang, T. (2022). Efficient monotonic binning for predictive modeling in high-dimensional spaces. Knowledge-Based Systems, 235, 107629.
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_mba(
   Rcpp::IntegerVector target,
   Rcpp::CharacterVector feature,
   int min_bins = 3,
   int max_bins = 5,
   double bin_cutoff = 0.05,
   int max_n_prebins = 20,
   std::string bin_separator = "%;%",
   double convergence_threshold = 1e-6,
   int max_iterations = 1000
) {
 // Preliminary checks
 if (feature.size() == 0 || target.size() == 0) {
   Rcpp::stop("Feature and target cannot be empty.");
 }
 
 if (feature.size() != target.size()) {
   Rcpp::stop("Feature and target must have the same length.");
 }
 
 // Optimized R to C++ conversion
 std::vector<std::string> feature_vec;
 feature_vec.reserve(feature.size());
 
 int na_feature_count = 0;
 int na_target_count = 0;
 
 for (R_xlen_t i = 0; i < feature.size(); ++i) {
   // Handle NA in feature
   if (feature[i] == NA_STRING) {
     feature_vec.push_back("NA");
     na_feature_count++;
   } else {
     feature_vec.push_back(Rcpp::as<std::string>(feature[i]));
   }
   
   // Check for NA in target
   if (IntegerVector::is_na(target[i])) {
     na_target_count++;
     Rcpp::stop("Target cannot contain missing values at position %d.", i+1);
   }
 }
 
 // Warn about NA values in feature
 if (na_feature_count > 0) {
   Rcpp::warning("%d missing values found in feature and converted to \"NA\" category.", 
                 na_feature_count);
 }
 
 // Execute optimized algorithm
 OptimalBinningCategoricalMBA mba(
     feature_vec, target, min_bins, max_bins, bin_cutoff, max_n_prebins,
     bin_separator, convergence_threshold, max_iterations
 );
 
 return mba.fit();
}




// // [[Rcpp::depends(Rcpp)]]
// 
// #include <Rcpp.h>
// #include <vector>
// #include <string>
// #include <unordered_map>
// #include <unordered_set>
// #include <algorithm>
// #include <cmath>
// #include <numeric>
// #include <limits>
// #include <stdexcept>
// #include <memory>
// 
// using namespace Rcpp;
// 
// // Constantes globais para melhor legibilidade e consistência
// static constexpr double EPSILON = 1e-10;
// static constexpr double NEG_INFINITY = -std::numeric_limits<double>::infinity();
// 
// // Namespace com funções auxiliares otimizadas
// namespace utils {
// // Função logaritmo segura e otimizada
// inline double safe_log(double x) {
//   return x > EPSILON ? std::log(x) : std::log(EPSILON);
// }
// 
// // Função de junção otimizada para strings, garantindo valores únicos
// inline std::string join(const std::vector<std::string>& v, const std::string& delimiter) {
//   if (v.empty()) return "";
//   if (v.size() == 1) return v[0];
//   
//   // Cria um conjunto para garantir valores únicos
//   std::unordered_set<std::string> unique_values;
//   std::vector<std::string> unique_vector;
//   unique_vector.reserve(v.size());
//   
//   for (const auto& s : v) {
//     if (unique_values.insert(s).second) {
//       unique_vector.push_back(s);
//     }
//   }
//   
//   // Estima tamanho para pré-alocação
//   size_t total_length = 0;
//   for (const auto& s : unique_vector) {
//     total_length += s.length();
//   }
//   total_length += delimiter.length() * (unique_vector.size() - 1);
//   
//   std::string result;
//   result.reserve(total_length);
//   
//   result = unique_vector[0];
//   for (size_t i = 1; i < unique_vector.size(); ++i) {
//     result += delimiter;
//     result += unique_vector[i];
//   }
//   return result;
// }
// }
// 
// // Estrutura Bin otimizada com garantia de unicidade de categorias
// struct Bin {
//   std::unordered_set<std::string> category_set; // Conjunto para garantir unicidade
//   std::vector<std::string> categories;          // Vetor para manter ordem
//   double woe;
//   double iv;
//   int count;
//   int count_pos;
//   int count_neg;
//   double event_rate;
//   
//   Bin() : woe(0), iv(0), count(0), count_pos(0), count_neg(0), event_rate(0) {
//     categories.reserve(8);
//   }
//   
//   // Método para adicionar categoria, garantindo unicidade
//   inline void add_category(const std::string& cat, int is_positive) {
//     // Adiciona apenas se é uma categoria nova
//     if (category_set.insert(cat).second) {
//       categories.push_back(cat);
//     }
//     
//     count++;
//     if (is_positive) {
//       count_pos++;
//     } else {
//       count_neg++;
//     }
//     update_event_rate();
//   }
//   
//   // Método para mesclar com outro bin, garantindo unicidade
//   inline void merge_with(const Bin& other) {
//     // Adiciona apenas categorias que ainda não existem
//     for (const auto& cat : other.categories) {
//       if (category_set.insert(cat).second) {
//         categories.push_back(cat);
//       }
//     }
//     
//     count += other.count;
//     count_pos += other.count_pos;
//     count_neg += other.count_neg;
//     update_event_rate();
//   }
//   
//   // Atualizar taxa de eventos
//   inline void update_event_rate() {
//     event_rate = count > 0 ? static_cast<double>(count_pos) / count : 0.0;
//   }
//   
//   // Cálculo de WoE e IV
//   inline void calculate_metrics(int total_good, int total_bad) {
//     double prop_event = static_cast<double>(count_pos) / std::max(total_bad, 1);
//     double prop_non_event = static_cast<double>(count_neg) / std::max(total_good, 1);
//     
//     prop_event = std::max(prop_event, EPSILON);
//     prop_non_event = std::max(prop_non_event, EPSILON);
//     
//     woe = utils::safe_log(prop_event / prop_non_event);
//     iv = (prop_event - prop_non_event) * woe;
//   }
// };
// 
// // Cache para mesclagens potenciais
// class MergeCache {
// private:
//   std::vector<std::vector<double>> iv_loss_cache;
//   bool enabled;
//   
// public:
//   MergeCache(size_t max_size, bool use_cache = true) : enabled(use_cache) {
//     if (enabled && max_size > 0) {
//       iv_loss_cache.resize(max_size);
//       for (auto& row : iv_loss_cache) {
//         row.resize(max_size, -1.0);
//       }
//     }
//   }
//   
//   inline double get_iv_loss(size_t bin1, size_t bin2) {
//     if (!enabled || bin1 >= iv_loss_cache.size() || bin2 >= iv_loss_cache[bin1].size()) {
//       return -1.0;
//     }
//     return iv_loss_cache[bin1][bin2];
//   }
//   
//   inline void set_iv_loss(size_t bin1, size_t bin2, double value) {
//     if (!enabled || bin1 >= iv_loss_cache.size() || bin2 >= iv_loss_cache[bin1].size()) {
//       return;
//     }
//     iv_loss_cache[bin1][bin2] = value;
//   }
//   
//   inline void invalidate_bin(size_t bin_idx) {
//     if (!enabled || bin_idx >= iv_loss_cache.size()) {
//       return;
//     }
//     
//     for (size_t i = 0; i < iv_loss_cache.size(); ++i) {
//       if (i < iv_loss_cache[bin_idx].size()) {
//         iv_loss_cache[bin_idx][i] = -1.0;
//       }
//       if (bin_idx < iv_loss_cache[i].size()) {
//         iv_loss_cache[i][bin_idx] = -1.0;
//       }
//     }
//   }
//   
//   inline void resize(size_t new_size) {
//     if (!enabled) return;
//     
//     iv_loss_cache.resize(new_size);
//     for (auto& row : iv_loss_cache) {
//       row.resize(new_size, -1.0);
//     }
//   }
// };
// 
// // Classe principal otimizada
// class OptimalBinningCategoricalMBA {
// private:
//   std::vector<std::string> feature;
//   std::vector<int> target;
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
//   std::string bin_separator;
//   double convergence_threshold;
//   int max_iterations;
//   
//   int total_good;
//   int total_bad;
//   
//   std::vector<Bin> bins;
//   std::unique_ptr<MergeCache> merge_cache;
//   
//   // Validação de entradas otimizada
//   void validate_inputs() {
//     if (feature.size() != target.size()) {
//       throw std::invalid_argument("Feature e target devem ter o mesmo tamanho.");
//     }
//     if (feature.empty()) {
//       throw std::invalid_argument("Feature e target não podem ser vazios.");
//     }
//     if (min_bins < 2) {
//       throw std::invalid_argument("min_bins deve ser >= 2.");
//     }
//     if (max_bins < min_bins) {
//       throw std::invalid_argument("max_bins deve ser >= min_bins.");
//     }
//     if (bin_cutoff <= 0 || bin_cutoff >= 1) {
//       throw std::invalid_argument("bin_cutoff deve estar entre 0 e 1.");
//     }
//     if (max_n_prebins < max_bins) {
//       throw std::invalid_argument("max_n_prebins >= max_bins.");
//     }
//     
//     // Verificação eficiente dos valores de target
//     bool has_zero = false;
//     bool has_one = false;
//     
//     for (int val : target) {
//       if (val == 0) has_zero = true;
//       else if (val == 1) has_one = true;
//       else throw std::invalid_argument("Target deve conter apenas 0 e 1.");
//       
//       // Early termination
//       if (has_zero && has_one) break;
//     }
//     
//     if (!has_zero || !has_one) {
//       throw std::invalid_argument("Target deve conter tanto 0 quanto 1.");
//     }
//     
//     // Ajuste de max_bins baseado em categorias únicas
//     std::unordered_set<std::string> unique_categories(feature.begin(), feature.end());
//     int ncat = static_cast<int>(unique_categories.size());
//     if (max_bins > ncat) {
//       max_bins = ncat;
//     }
//     
//     // Ajuste de min_bins se necessário
//     min_bins = std::min(min_bins, max_bins);
//   }
//   
//   // Prebinning otimizado com garantia de unicidade
//   void prebinning() {
//     // Contagem eficiente em uma única passagem
//     std::unordered_map<std::string, Bin> category_bins;
//     category_bins.reserve(std::min(feature.size() / 4, static_cast<size_t>(1024)));
//     
//     total_good = 0;
//     total_bad = 0;
//     
//     for (size_t i = 0; i < feature.size(); ++i) {
//       const auto& cat = feature[i];
//       int is_positive = target[i];
//       
//       if (category_bins.find(cat) == category_bins.end()) {
//         category_bins[cat] = Bin();
//       }
//       
//       category_bins[cat].add_category(cat, is_positive);
//       
//       if (is_positive) {
//         total_bad++;
//       } else {
//         total_good++;
//       }
//     }
//     
//     // Transfere para vetor e ordena por contagem (decrescente)
//     bins.clear();
//     bins.reserve(category_bins.size());
//     
//     for (auto& pair : category_bins) {
//       bins.push_back(std::move(pair.second));
//     }
//     
//     std::sort(bins.begin(), bins.end(),
//               [](const Bin& a, const Bin& b) { return a.count > b.count; });
//     
//     // Inicializa cache de mesclagem
//     merge_cache = std::make_unique<MergeCache>(bins.size(), bins.size() > 10);
//     
//     // Reduz para max_n_prebins se necessário
//     while (static_cast<int>(bins.size()) > max_n_prebins && static_cast<int>(bins.size()) > min_bins) {
//       // Encontra bin com menor contagem
//       auto min_it = std::min_element(bins.begin(), bins.end(),
//                                      [](const Bin& a, const Bin& b) {
//                                        return a.count < b.count;
//                                      });
//       
//       size_t idx = static_cast<size_t>(min_it - bins.begin());
//       
//       // Tenta mesclar com bin adjacente
//       if (idx > 0) {
//         if (!try_merge_bins(idx - 1, idx)) break;
//       } else if (bins.size() > 1) {
//         if (!try_merge_bins(0, 1)) break;
//       } else {
//         break;
//       }
//     }
//   }
//   
//   // Enforcement de bin_cutoff otimizado
//   void enforce_bin_cutoff() {
//     int min_count = static_cast<int>(std::ceil(bin_cutoff * static_cast<double>(feature.size())));
//     int min_count_pos = static_cast<int>(std::ceil(bin_cutoff * static_cast<double>(total_bad)));
//     
//     // Identifica todos os bins de baixa frequência de uma vez
//     std::vector<size_t> low_freq_bins;
//     
//     for (size_t i = 0; i < bins.size(); ++i) {
//       if (bins[i].count < min_count || bins[i].count_pos < min_count_pos) {
//         low_freq_bins.push_back(i);
//       }
//     }
//     
//     // Processa bins de baixa frequência de forma eficiente
//     for (size_t idx : low_freq_bins) {
//       if (static_cast<int>(bins.size()) <= min_bins) {
//         break; // Nunca descer abaixo de min_bins
//       }
//       
//       // Bin ainda existe e ainda está abaixo do cutoff?
//       if (idx >= bins.size() || (bins[idx].count >= min_count && bins[idx].count_pos >= min_count_pos)) {
//         continue;
//       }
//       
//       // Encontra vizinho mais próximo para mesclar
//       size_t merge_index;
//       if (idx > 0) {
//         merge_index = idx - 1;
//       } else if (idx + 1 < bins.size()) {
//         merge_index = idx + 1;
//       } else {
//         continue; // Não há vizinhos
//       }
//       
//       if (!try_merge_bins(std::min(idx, merge_index), std::max(idx, merge_index))) {
//         // Se não foi possível mesclar sem violar min_bins, tentamos outro bin
//         continue;
//       }
//       
//       // Ajustamos os índices dos bins restantes, já que removemos um
//       for (auto& remaining_idx : low_freq_bins) {
//         if (remaining_idx > merge_index) {
//           remaining_idx--;
//         }
//       }
//     }
//   }
//   
//   // Cálculo inicial de WoE otimizado
//   void calculate_initial_woe() {
//     for (auto& bin : bins) {
//       bin.calculate_metrics(total_good, total_bad);
//     }
//   }
//   
//   // Enforcement de monotonicidade otimizado
//   void enforce_monotonicity() {
//     if (bins.empty()) {
//       throw std::runtime_error("Nenhum bin disponível para impor monotonicidade.");
//     }
//     
//     // Ordena bins por WoE
//     std::sort(bins.begin(), bins.end(),
//               [](const Bin& a, const Bin& b) { return a.woe < b.woe; });
//     
//     // Determina direção da monotonicidade
//     bool increasing = true;
//     if (bins.size() > 1) {
//       for (size_t i = 1; i < bins.size(); ++i) {
//         if (bins[i].woe < bins[i-1].woe - EPSILON) {
//           increasing = false;
//           break;
//         }
//       }
//     }
//     
//     // Loop otimizado para enforçar monotonicidade
//     bool any_merge;
//     do {
//       any_merge = false;
//       
//       for (size_t i = 0; i + 1 < bins.size(); ++i) {
//         if (static_cast<int>(bins.size()) <= min_bins) {
//           break; // Não reduzir abaixo de min_bins
//         }
//         
//         bool violation = (increasing && bins[i].woe > bins[i+1].woe + EPSILON) ||
//           (!increasing && bins[i].woe < bins[i+1].woe - EPSILON);
//         
//         if (violation) {
//           if (try_merge_bins(i, i+1)) {
//             any_merge = true;
//             break;
//           }
//         }
//       }
//       
//     } while (any_merge && static_cast<int>(bins.size()) > min_bins);
//   }
//   
//   // Otimização de bins melhorada
//   void optimize_bins() {
//     if (static_cast<int>(bins.size()) <= max_bins) {
//       return; // Já estamos dentro do limite máximo
//     }
//     
//     int iterations = 0;
//     double prev_total_iv = 0.0;
//     
//     for (const auto& bin : bins) {
//       prev_total_iv += std::fabs(bin.iv);
//     }
//     
//     while (static_cast<int>(bins.size()) > max_bins && iterations < max_iterations) {
//       if (static_cast<int>(bins.size()) <= min_bins) {
//         break; // Não reduzir abaixo de min_bins
//       }
//       
//       // Encontra o par de bins com menor IV combinado
//       double min_combined_iv = std::numeric_limits<double>::max();
//       size_t min_iv_index = 0;
//       
//       // Busca otimizada com cache
//       for (size_t i = 0; i < bins.size() - 1; ++i) {
//         double cached_iv = merge_cache->get_iv_loss(i, i+1);
//         double combined_iv;
//         
//         if (cached_iv >= 0.0) {
//           combined_iv = cached_iv;
//         } else {
//           combined_iv = std::fabs(bins[i].iv) + std::fabs(bins[i+1].iv);
//           merge_cache->set_iv_loss(i, i+1, combined_iv);
//         }
//         
//         if (combined_iv < min_combined_iv) {
//           min_combined_iv = combined_iv;
//           min_iv_index = i;
//         }
//       }
//       
//       // Tenta mesclar os bins com menor IV combinado
//       if (!try_merge_bins(min_iv_index, min_iv_index + 1)) {
//         break; // Não foi possível mesclar, paramos
//       }
//       
//       // Calcula novo IV total e verifica convergência
//       double total_iv = 0.0;
//       for (const auto& bin : bins) {
//         total_iv += std::fabs(bin.iv);
//       }
//       
//       if (std::fabs(total_iv - prev_total_iv) < convergence_threshold) {
//         break; // Convergência atingida
//       }
//       
//       prev_total_iv = total_iv;
//       iterations++;
//     }
//     
//     if (static_cast<int>(bins.size()) > max_bins) {
//       Rcpp::warning("Não foi possível reduzir o número de bins até max_bins sem violar min_bins ou convergência.");
//     }
//   }
//   
//   // Tentativa de mesclagem otimizada
//   bool try_merge_bins(size_t index1, size_t index2) {
//     // Verificações de segurança
//     if (static_cast<int>(bins.size()) <= min_bins) {
//       return false; // Já estamos no mínimo, não mesclar
//     }
//     
//     if (index1 >= bins.size() || index2 >= bins.size() || index1 == index2) {
//       return false;
//     }
//     
//     if (index2 < index1) std::swap(index1, index2);
//     
//     // Mesclagem eficiente
//     bins[index1].merge_with(bins[index2]);
//     bins[index1].calculate_metrics(total_good, total_bad);
//     
//     bins.erase(bins.begin() + index2);
//     
//     // Atualiza cache
//     merge_cache->invalidate_bin(index1);
//     merge_cache->resize(bins.size());
//     
//     return true;
//   }
//   
//   // Verificação de consistência otimizada
//   void check_consistency() const {
//     int total_count = 0;
//     int total_count_pos = 0;
//     int total_count_neg = 0;
//     
//     for (const auto& bin : bins) {
//       total_count += bin.count;
//       total_count_pos += bin.count_pos;
//       total_count_neg += bin.count_neg;
//     }
//     
//     if (static_cast<size_t>(total_count) != feature.size()) {
//       throw std::runtime_error("Contagem total inconsistente após binagem.");
//     }
//     
//     if (total_count_pos != total_bad || total_count_neg != total_good) {
//       throw std::runtime_error("Contagens positivas/negativas inconsistentes após binagem.");
//     }
//   }
//   
// public:
//   // Construtor otimizado
//   OptimalBinningCategoricalMBA(
//     const std::vector<std::string>& feature_,
//     const Rcpp::IntegerVector& target_,
//     int min_bins_ = 3,
//     int max_bins_ = 5,
//     double bin_cutoff_ = 0.05,
//     int max_n_prebins_ = 20,
//     std::string bin_separator_ = "%;%",
//     double convergence_threshold_ = 1e-6,
//     int max_iterations_ = 1000
//   ) : feature(feature_), target(Rcpp::as<std::vector<int>>(target_)), 
//   min_bins(min_bins_), max_bins(max_bins_), 
//   bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_),
//   bin_separator(bin_separator_),
//   convergence_threshold(convergence_threshold_),
//   max_iterations(max_iterations_),
//   total_good(0), total_bad(0) {
//     
//     // Pré-alocação para melhor desempenho
//     bins.reserve(std::min(feature.size() / 4, static_cast<size_t>(1024)));
//   }
//   
//   // Método fit otimizado
//   Rcpp::List fit() {
//     try {
//       // Processo de binagem com verificações de desempenho
//       validate_inputs();
//       prebinning();
//       enforce_bin_cutoff();
//       calculate_initial_woe();
//       enforce_monotonicity();
//       
//       // Otimização de bins se necessário
//       bool converged_flag = false;
//       int iterations_done = 0;
//       
//       if (static_cast<int>(bins.size()) <= max_bins) {
//         converged_flag = true;
//       } else {
//         double prev_total_iv = 0.0;
//         for (const auto& bin : bins) {
//           prev_total_iv += std::fabs(bin.iv);
//         }
//         
//         for (int i = 0; i < max_iterations; ++i) {
//           // Ponto de partida para esta iteração
//           size_t start_bins = bins.size();
//           
//           optimize_bins();
//           
//           // Se não reduziu bins ou atingiu max_bins, verificar convergência
//           if (bins.size() == start_bins || static_cast<int>(bins.size()) <= max_bins) {
//             double total_iv = 0.0;
//             for (const auto& bin : bins) {
//               total_iv += std::fabs(bin.iv);
//             }
//             
//             if (std::fabs(total_iv - prev_total_iv) < convergence_threshold) {
//               converged_flag = true;
//               iterations_done = i + 1;
//               break;
//             }
//             
//             prev_total_iv = total_iv;
//           }
//           
//           iterations_done = i + 1;
//           
//           if (static_cast<int>(bins.size()) <= max_bins) {
//             break;
//           }
//         }
//       }
//       
//       // Verificação final de consistência
//       check_consistency();
//       
//       // Preparação de resultados otimizada com garantia de unicidade
//       const size_t n_bins = bins.size();
//       
//       CharacterVector bin_names(n_bins);
//       NumericVector bin_woe(n_bins);
//       NumericVector bin_iv(n_bins);
//       IntegerVector bin_count(n_bins);
//       IntegerVector bin_count_pos(n_bins);
//       IntegerVector bin_count_neg(n_bins);
//       NumericVector ids(n_bins);
//       
//       for (size_t i = 0; i < n_bins; ++i) {
//         // Usa utils::join que agora garante valores únicos
//         bin_names[i] = utils::join(bins[i].categories, bin_separator);
//         bin_woe[i] = bins[i].woe;
//         bin_iv[i] = bins[i].iv;
//         bin_count[i] = bins[i].count;
//         bin_count_pos[i] = bins[i].count_pos;
//         bin_count_neg[i] = bins[i].count_neg;
//         ids[i] = i + 1;
//       }
//       
//       return Rcpp::List::create(
//         Named("id") = ids,
//         Named("bin") = bin_names,
//         Named("woe") = bin_woe,
//         Named("iv") = bin_iv,
//         Named("count") = bin_count,
//         Named("count_pos") = bin_count_pos,
//         Named("count_neg") = bin_count_neg,
//         Named("converged") = converged_flag,
//         Named("iterations") = iterations_done
//       );
//     } catch (const std::exception& e) {
//       Rcpp::stop("Erro no binning ótimo: " + std::string(e.what()));
//     }
//   }
// };
// 
// //' @title Optimal Binning for Categorical Variables using Monotonic Binning Algorithm (MBA)
// //'
// //' @description
// //' This function performs optimal binning for categorical variables using a Monotonic Binning Algorithm (MBA) approach,
// //' which combines Weight of Evidence (WOE) and Information Value (IV) methods with monotonicity constraints.
// //'
// //' @param feature A character vector of categorical feature values.
// //' @param target An integer vector of binary target values (0 or 1).
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff Minimum frequency for a category to be considered as a separate bin (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins before merging (default: 20).
// //' @param bin_separator String used to separate category names when merging bins (default: "%;%").
// //' @param convergence_threshold Threshold for convergence in optimization (default: 1e-6).
// //' @param max_iterations Maximum number of iterations for optimization (default: 1000).
// //'
// //' @return A list containing:
// //' \itemize{
// //'   \item bins: A character vector of bin labels
// //'   \item woe: A numeric vector of Weight of Evidence values for each bin
// //'   \item iv: A numeric vector of Information Value for each bin
// //'   \item count: An integer vector of total counts for each bin
// //'   \item count_pos: An integer vector of positive target counts for each bin
// //'   \item count_neg: An integer vector of negative target counts for each bin
// //'   \item converged: A logical value indicating whether the algorithm converged
// //'   \item iterations: An integer indicating the number of iterations run
// //' }
// //'
// //' @details
// //' The algorithm performs the following steps:
// //' \enumerate{
// //'   \item Input validation and preprocessing
// //'   \item Initial pre-binning based on frequency
// //'   \item Enforcing minimum bin size (bin_cutoff)
// //'   \item Calculating initial Weight of Evidence (WOE) and Information Value (IV)
// //'   \item Enforcing monotonicity of WOE across bins
// //'   \item Optimizing the number of bins through iterative merging
// //' }
// //'
// //' The Weight of Evidence (WOE) is calculated as:
// //' \deqn{WOE = \ln\left(\frac{\text{Proportion of Events}}{\text{Proportion of Non-Events}}\right)}
// //'
// //' The Information Value (IV) for each bin is calculated as:
// //' \deqn{IV = (\text{Proportion of Events} - \text{Proportion of Non-Events}) \times WOE}
// //'
// //' @examples
// //' \dontrun{
// //' # Create sample data
// //' set.seed(123)
// //' target <- sample(0:1, 1000, replace = TRUE)
// //' feature <- sample(LETTERS[1:5], 1000, replace = TRUE)
// //'
// //' # Run optimal binning
// //' result <- optimal_binning_categorical_mba(feature, target)
// //'
// //' # View results
// //' print(result)
// //' }
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_categorical_mba(
//    Rcpp::IntegerVector target,
//    Rcpp::CharacterVector feature,
//    int min_bins = 3,
//    int max_bins = 5,
//    double bin_cutoff = 0.05,
//    int max_n_prebins = 20,
//    std::string bin_separator = "%;%",
//    double convergence_threshold = 1e-6,
//    int max_iterations = 1000
// ) {
//  // Verificações preliminares
//  if (feature.size() == 0 || target.size() == 0) {
//    Rcpp::stop("Feature e target não podem ser vazios.");
//  }
//  
//  if (feature.size() != target.size()) {
//    Rcpp::stop("Feature e target devem ter o mesmo tamanho.");
//  }
//  
//  // Conversão otimizada de R para C++
//  std::vector<std::string> feature_vec;
//  feature_vec.reserve(feature.size());
//  
//  for (R_xlen_t i = 0; i < feature.size(); ++i) {
//    // Tratamento de NAs em feature
//    if (feature[i] == NA_STRING) {
//      feature_vec.push_back("NA");
//    } else {
//      feature_vec.push_back(Rcpp::as<std::string>(feature[i]));
//    }
//  }
//  
//  // Validação de NAs em target
//  for (R_xlen_t i = 0; i < target.size(); ++i) {
//    if (IntegerVector::is_na(target[i])) {
//      Rcpp::stop("Target não pode conter valores ausentes.");
//    }
//  }
//  
//  // Executa o algoritmo otimizado
//  OptimalBinningCategoricalMBA mba(
//      feature_vec, target, min_bins, max_bins, bin_cutoff, max_n_prebins,
//      bin_separator, convergence_threshold, max_iterations
//  );
//  
//  return mba.fit();
// }
