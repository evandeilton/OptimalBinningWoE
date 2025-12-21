// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <memory>

using namespace Rcpp;

// Include shared headers
#include "common/optimal_binning_common.h"
#include "common/bin_structures.h"

using namespace Rcpp;
using namespace OptimalBinning;


// Constants for better readability and precision
// Local constant removed (uses shared definition)
constexpr double NEG_INFINITY = -std::numeric_limits<double>::infinity();
// Local constant removed (uses shared definition) // Adjustable prior strength for smoothing

// Optimized structure for storing bin information
// Local CategoricalBin definition removed


// Class for incremental IV calculation caching
class IVCache {
private:
  std::vector<std::vector<double>> cache;
  bool enabled;
  
public:
  IVCache(size_t size, bool use_cache = true) : enabled(use_cache) {
    if (enabled) {
      cache.resize(size);
      for (auto& row : cache) {
        row.resize(size, -1.0);
      }
    }
  }
  
  double get(size_t i, size_t j) {
    if (!enabled || i >= cache.size() || j >= cache.size()) return -1.0;
    return cache[i][j];
  }
  
  void set(size_t i, size_t j, double value) {
    if (!enabled || i >= cache.size() || j >= cache.size()) return;
    cache[i][j] = value;
  }
  
  void invalidate_row(size_t i) {
    if (!enabled || i >= cache.size()) return;
    for (size_t j = 0; j < cache.size(); j++) {
      cache[i][j] = -1.0;
      if (i != j) cache[j][i] = -1.0;
    }
  }
  
  void resize(size_t new_size) {
    if (!enabled) return;
    cache.resize(new_size);
    for (auto& row : cache) {
      row.resize(new_size, -1.0);
    }
  }
};

class OBC_GMB {
private:
  const std::vector<std::string>& feature;
  const std::vector<int>& target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  std::string bin_separator;
  double convergence_threshold;
  int max_iterations;
  
  std::vector<CategoricalBin> bins;
  std::unique_ptr<IVCache> iv_cache;
  int total_pos = 0;
  int total_neg = 0;
  bool converged = false;
  int iterations_run = 0;
  
  // Enhanced input validation with more comprehensive checks
  void validateInput() const {
    if (feature.size() != target.size()) {
      throw std::invalid_argument("Feature and target must have the same length.");
    }
    if (feature.empty()) {
      throw std::invalid_argument("Feature cannot be empty.");
    }
    if (min_bins < 2) {
      throw std::invalid_argument("min_bins must be >= 2.");
    }
    if (max_bins < min_bins) {
      throw std::invalid_argument("max_bins must be >= min_bins.");
    }
    if (bin_cutoff <= 0 || bin_cutoff >= 1) {
      throw std::invalid_argument("bin_cutoff must be between 0 and 1 (exclusive).");
    }
    if (max_n_prebins < min_bins) {
      throw std::invalid_argument("max_n_prebins must be >= min_bins.");
    }
    
    // Check for empty strings in feature
    if (std::any_of(feature.begin(), feature.end(), [](const std::string& s) { 
      return s.empty(); 
    })) {
      throw std::invalid_argument("Feature cannot contain empty strings. Consider preprocessing your data.");
    }
    
    // Efficient check for binary target
    bool has_zero = false;
    bool has_one = false;
    for (int t : target) {
      if (t == 0) has_zero = true;
      else if (t == 1) has_one = true;
      else throw std::invalid_argument("Target must be binary (0 or 1).");
      
      // Early termination once we've seen both values
      if (has_zero && has_one) break;
    }
    
    if (!has_zero || !has_one) {
      throw std::invalid_argument("Target must contain both 0 and 1 values.");
    }
  }
  
  // Calculate WoE with protection against log(0) and improved smoothing
  inline double calculateWOE(int pos, int neg) const {
    // Calculate with Bayesian smoothing
    double prior_pos = BAYESIAN_PRIOR_STRENGTH * static_cast<double>(total_pos) / 
      (total_pos + total_neg);
    double prior_neg = BAYESIAN_PRIOR_STRENGTH - prior_pos;
    
    double pos_rate = static_cast<double>(pos + prior_pos) / 
      static_cast<double>(total_pos + BAYESIAN_PRIOR_STRENGTH);
    double neg_rate = static_cast<double>(neg + prior_neg) / 
      static_cast<double>(total_neg + BAYESIAN_PRIOR_STRENGTH);
    
    return std::log(pos_rate / neg_rate);
  }
  
  // Calculate total IV for a set of bins with improved numerical stability
  double calculateIV(const std::vector<CategoricalBin>& bins_to_check) const {
    double iv = 0.0;
    for (const auto& bin : bins_to_check) {
      // Use pre-calculated value if available
      if (bin.iv != 0.0 && std::isfinite(bin.iv)) {
        iv += bin.iv;
      } else {
        // Calculate with Bayesian smoothing
        double prior_pos = BAYESIAN_PRIOR_STRENGTH * static_cast<double>(total_pos) / 
          (total_pos + total_neg);
        double prior_neg = BAYESIAN_PRIOR_STRENGTH - prior_pos;
        
        double pos_rate = static_cast<double>(bin.count_pos + prior_pos) / 
          static_cast<double>(total_pos + BAYESIAN_PRIOR_STRENGTH);
        double neg_rate = static_cast<double>(bin.count_neg + prior_neg) / 
          static_cast<double>(total_neg + BAYESIAN_PRIOR_STRENGTH);
        
        double woe = std::log(pos_rate / neg_rate);
        double local_iv = (pos_rate - neg_rate) * woe;
        
        if (std::isfinite(local_iv)) {
          iv += local_iv;
        }
      }
    }
    return iv;
  }
  
  // Enhanced bin initialization with optimized counting
  void initializeBins() {
    // Efficient single-pass counting
    std::unordered_map<std::string, std::pair<int, int>> category_stats;
    category_stats.reserve(std::min(static_cast<size_t>(feature.size() / 4), static_cast<size_t>(1024)));
    
    total_pos = 0;
    total_neg = 0;
    
    for (size_t i = 0; i < feature.size(); ++i) {
      const std::string& cat = feature[i];
      auto& stats = category_stats[cat];
      
      if (target[i] == 1) {
        stats.first++;  // pos_count
        total_pos++;
      } else {
        stats.second++;  // neg_count
        total_neg++;
      }
    }
    
    // Check for extremely imbalanced datasets
    if (total_pos < 5 || total_neg < 5) {
      Rcpp::warning("Dataset has fewer than 5 samples in one class. Results may be unstable.");
    }
    
    // Optimized bin creation
    bins.clear();
    bins.reserve(category_stats.size());
    
    for (const auto& [cat, stats] : category_stats) {
      CategoricalBin bin;
      bin.categories.push_back(cat); bin.count_pos += stats.first; bin.count_neg += (stats.first + stats.second - stats.first); bin.update_count();
      bins.push_back(std::move(bin));
    }
    
    // Sort by positive rate for consistent ordering
    for (const auto& bin : bins) {
      (void)bin; // Suppress unused variable warning - event_rate() calculated dynamically
    }
    
    std::sort(bins.begin(), bins.end(), [](const CategoricalBin& a, const CategoricalBin& b) {
      return a.event_rate() < b.event_rate();
    });
    
    // Enhanced rare category handling
    int total_count = std::accumulate(bins.begin(), bins.end(), 0,
                                      [](int sum, const CategoricalBin& bin) { return sum + bin.count; });
    
    std::vector<CategoricalBin> merged_bins;
    merged_bins.reserve(bins.size());
    
    CategoricalBin current_rare_bin;
    bool has_rare_bin = false;
    
    for (auto& bin : bins) {
      double freq = static_cast<double>(bin.count) / static_cast<double>(total_count);
      
      if (freq < bin_cutoff) {
        // Merge rare bin into current_rare_bin
        current_rare_bin.merge_with(bin);
        has_rare_bin = true;
      } else {
        // Add accumulated rare bin if it exists
        if (has_rare_bin) {
          merged_bins.push_back(std::move(current_rare_bin));
          current_rare_bin = CategoricalBin();
          has_rare_bin = false;
        }
        merged_bins.push_back(std::move(bin));
      }
    }
    
    // Add final rare bin if it exists
    if (has_rare_bin) {
      merged_bins.push_back(std::move(current_rare_bin));
    }
    
    bins = std::move(merged_bins);
    
    // Limit number of pre-bins if necessary
    if (static_cast<int>(bins.size()) > max_n_prebins) {
      // Sort by bin size before limiting
      std::sort(bins.begin(), bins.end(), [](const CategoricalBin& a, const CategoricalBin& b) {
        return a.count > b.count;  // Descending order
      });
      bins.resize(max_n_prebins);
      
      // Resort by positive rate
      for (const auto& bin : bins) {
        (void)bin; // Suppress unused variable warning - event_rate() calculated dynamically
      }
      
      std::sort(bins.begin(), bins.end(), [](const CategoricalBin& a, const CategoricalBin& b) {
        return a.event_rate() < b.event_rate();
      });
    }
    
    // Calculate metrics for all bins
    for (auto& bin : bins) {
      bin.calculate_metrics(total_pos, total_neg);
    }
    
    // Initialize IV cache
    iv_cache = std::make_unique<IVCache>(bins.size(), bins.size() > 10);
  }
  
  // Enhanced greedy merge with improved tie handling
  void greedyMerge() {
    // Early exit if we already have few enough bins
    if (static_cast<int>(bins.size()) <= max_bins) {
      converged = true;
      return;
    }
    
    double prev_iv = calculateIV(bins);
    double current_iv = prev_iv;
    
    while (static_cast<int>(bins.size()) > min_bins && iterations_run < max_iterations) {
      double best_merge_score = NEG_INFINITY;
      double second_best_score = NEG_INFINITY;
      size_t best_merge_index = 0;
      size_t second_best_index = 0;
      
      // Enhanced evaluation of merge options
      for (size_t i = 0; i < bins.size() - 1; ++i) {
        // Check cache first
        double cached_score = iv_cache->get(i, i+1);
        if (cached_score >= 0.0) {
          if (cached_score > best_merge_score) {
            second_best_score = best_merge_score;
            second_best_index = best_merge_index;
            best_merge_score = cached_score;
            best_merge_index = i;
          } else if (cached_score > second_best_score) {
            second_best_score = cached_score;
            second_best_index = i;
          }
          continue;
        }
        
        // Simulate merging
        CategoricalBin merged_bin;
        merged_bin.merge_with(bins[i]);
        merged_bin.merge_with(bins[i+1]);
        merged_bin.calculate_metrics(total_pos, total_neg);
        
        // Create simulated bin configuration
        std::vector<CategoricalBin> temp_bins;
        temp_bins.reserve(bins.size() - 1);
        
        // Add bins before the merge
        temp_bins.insert(temp_bins.end(), bins.begin(), bins.begin() + i);
        
        // Add merged bin
        temp_bins.push_back(merged_bin);
        
        // Add bins after the merge
        if (i + 2 < bins.size()) {
          temp_bins.insert(temp_bins.end(), bins.begin() + i + 2, bins.end());
        }
        
        // Calculate merge score
        double merge_score = calculateIV(temp_bins);
        iv_cache->set(i, i+1, merge_score);
        
        // Early exit if we find an excellent merge (5% improvement)
        if (merge_score > current_iv * 1.05 && std::isfinite(merge_score)) {
          best_merge_score = merge_score;
          best_merge_index = i;
          break;  // Early exit for significant improvement
        }
        
        // Track best and second best options
        if (merge_score > best_merge_score && std::isfinite(merge_score)) {
          second_best_score = best_merge_score;
          second_best_index = best_merge_index;
          best_merge_score = merge_score;
          best_merge_index = i;
        } else if (merge_score > second_best_score && std::isfinite(merge_score)) {
          second_best_score = merge_score;
          second_best_index = i;
        }
      }
      
      // Tie handling: If best and second best are very close, prefer more balanced bins
      if (std::abs(best_merge_score - second_best_score) < convergence_threshold * 10) {
        int size_diff_best = std::abs(bins[best_merge_index].count - bins[best_merge_index+1].count);
        int size_diff_second = std::abs(bins[second_best_index].count - bins[second_best_index+1].count);
        
        if (size_diff_second < size_diff_best * 0.8) {  // Second option is significantly more balanced
          best_merge_index = second_best_index;
          best_merge_score = second_best_score;
        }
      }
      
      // Execute the best merge
      CategoricalBin& bin1 = bins[best_merge_index];
      CategoricalBin& bin2 = bins[best_merge_index + 1];
      
      bin1.merge_with(bin2);
      bin1.calculate_metrics(total_pos, total_neg);
      
      bins.erase(bins.begin() + best_merge_index + 1);
      
      // Invalidate cache for affected rows
      iv_cache->invalidate_row(best_merge_index);
      iv_cache->resize(bins.size());
      
      // Recalculate IV after merging
      current_iv = calculateIV(bins);
      
      // Check convergence
      if (std::fabs(current_iv - prev_iv) < convergence_threshold) {
        converged = true;
        break;
      }
      
      prev_iv = current_iv;
      iterations_run++;
      
      // Stop if we've reached max_bins
      if (static_cast<int>(bins.size()) <= max_bins) {
        break;
      }
    }
  }
  
  // Enhanced monotonicity enforcement with gradient relaxation
  void ensureMonotonicity() {
    if (bins.size() <= 1) return;
    
    bool monotonic = false;
    const int max_attempts = static_cast<int>(bins.size() * 3); // Safe limit
    int attempts = 0;
    
    // Calculate average bin WoE gap for context-aware monotonicity check
    double avg_woe_gap = 0.0;
    if (bins.size() > 1) {
      double total_gap = 0.0;
      for (size_t i = 1; i < bins.size(); i++) {
        total_gap += std::abs(bins[i].woe - bins[i-1].woe);
      }
      avg_woe_gap = total_gap / (bins.size() - 1);
    }
    
    // Adaptive threshold based on average gap
    double monotonicity_threshold = std::min(EPSILON, avg_woe_gap * 0.01);
    
    while (!monotonic && static_cast<int>(bins.size()) > min_bins && attempts < max_attempts) {
      monotonic = true;
      
      for (size_t i = 1; i < bins.size(); ++i) {
        // Check if monotonicity is violated with context-aware threshold
        if (bins[i].woe < bins[i-1].woe - monotonicity_threshold) {
          // Merge bins i-1 and i
          bins[i-1].merge_with(bins[i]);
          bins[i-1].calculate_metrics(total_pos, total_neg);
          bins.erase(bins.begin() + i);
          
          // Recalculate adaptive threshold
          if (bins.size() > 1) {
            double total_gap = 0.0;
            for (size_t j = 1; j < bins.size(); j++) {
              total_gap += std::abs(bins[j].woe - bins[j-1].woe);
            }
            avg_woe_gap = total_gap / (bins.size() - 1);
            monotonicity_threshold = std::min(EPSILON, avg_woe_gap * 0.01);
          }
          
          monotonic = false;
          break;
        }
      }
      
      attempts++;
    }
    
    if (attempts >= max_attempts) {
      Rcpp::warning("Could not ensure monotonicity in %d attempts", max_attempts);
    }
  }
  
  // Efficient category name joining for bin representation
  std::string joinCategoryNames(const std::vector<std::string>& categories) const {
    if (categories.empty()) return "";
    if (categories.size() == 1) return categories[0];
    
    // Estimate total size for pre-allocation
    size_t total_size = 0;
    for (const auto& cat : categories) {
      total_size += cat.size();
    }
    total_size += bin_separator.size() * (categories.size() - 1);
    
    std::string result;
    result.reserve(total_size);
    
    result = categories[0];
    for (size_t i = 1; i < categories.size(); ++i) {
      result += bin_separator;
      result += categories[i];
    }
    
    return result;
  }
  
public:
  OBC_GMB(const std::vector<std::string>& feature,
                               const std::vector<int>& target,
                               int min_bins = 3,
                               int max_bins = 5,
                               double bin_cutoff = 0.05,
                               int max_n_prebins = 20,
                               std::string bin_separator = "%;%",
                               double convergence_threshold = 1e-6,
                               int max_iterations = 1000)
    : feature(feature), target(target), min_bins(min_bins), max_bins(max_bins),
      bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins), bin_separator(bin_separator),
      convergence_threshold(convergence_threshold), max_iterations(max_iterations) {
    
    validateInput();
    
    // Adjust max_bins if necessary, efficiently
    std::unordered_set<std::string> unique_cats;
    unique_cats.reserve(std::min(feature.size(), static_cast<size_t>(1024)));
    
    for (const auto& cat : feature) {
      unique_cats.insert(cat);
    }
    
    int ncat = static_cast<int>(unique_cats.size());
    max_bins = std::min(max_bins, ncat);
    min_bins = std::min(min_bins, max_bins);
  }
  
  Rcpp::List fit() {
    // Initialization
    initializeBins();
    
    // Greedy merging
    greedyMerge();
    
    // Monotonicity enforcement
    ensureMonotonicity();
    
    // Prepare result efficiently
    const size_t n_bins = bins.size();
    
    Rcpp::NumericVector ids(n_bins);
    Rcpp::CharacterVector bin_names(n_bins);
    Rcpp::NumericVector woe_values(n_bins);
    Rcpp::NumericVector iv_values(n_bins);
    Rcpp::IntegerVector count_values(n_bins);
    Rcpp::IntegerVector count_pos_values(n_bins);
    Rcpp::IntegerVector count_neg_values(n_bins);
    
    for (size_t i = 0; i < n_bins; ++i) {
      ids[i] = i + 1;
      bin_names[i] = joinCategoryNames(bins[i].categories);
      woe_values[i] = bins[i].woe;
      iv_values[i] = bins[i].iv;
      count_values[i] = bins[i].count;
      count_pos_values[i] = bins[i].count_pos;
      count_neg_values[i] = bins[i].count_neg;
    }
    
    // Calculate total IV for the binning
    double total_iv = std::accumulate(iv_values.begin(), iv_values.end(), 0.0);
    
    return Rcpp::List::create(
      Rcpp::Named("id") = ids,
      Rcpp::Named("bin") = bin_names,
      Rcpp::Named("woe") = woe_values,
      Rcpp::Named("iv") = iv_values,
      Rcpp::Named("count") = count_values,
      Rcpp::Named("count_pos") = count_pos_values,
      Rcpp::Named("count_neg") = count_neg_values,
      Rcpp::Named("total_iv") = total_iv,
      Rcpp::Named("converged") = converged,
      Rcpp::Named("iterations") = iterations_run
    );
  }
};

// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_gmb(Rcpp::IntegerVector target,
                                          Rcpp::StringVector feature,
                                          int min_bins = 3,
                                          int max_bins = 5,
                                          double bin_cutoff = 0.05,
                                          int max_n_prebins = 20,
                                          std::string bin_separator = "%;%",
                                          double convergence_threshold = 1e-6,
                                          int max_iterations = 1000) {
 // Preliminary validations with improved error messages
 if (feature.size() == 0 || target.size() == 0) {
   Rcpp::stop("Input vectors cannot be empty.");
 }
 
 if (feature.size() != target.size()) {
   Rcpp::stop("Feature and target must have the same length (got %d and %d).", 
              feature.size(), target.size());
 }
 
 // Optimized conversion of R vectors to C++
 std::vector<std::string> feature_vec;
 std::vector<int> target_vec;
 
 feature_vec.reserve(feature.size());
 target_vec.reserve(target.size());
 
 // Count NAs for more informative error messages
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
   
   // Handle NA in target
   if (IntegerVector::is_na(target[i])) {
     na_target_count++;
     Rcpp::stop("Target cannot contain missing values. Found NA at position %d.", i+1);
   } else {
     target_vec.push_back(target[i]);
   }
 }
 
 // Warn about NA values in feature
 if (na_feature_count > 0) {
   Rcpp::warning("%d missing values found in feature and converted to \"NA\" category.", 
                 na_feature_count);
 }
 
 try {
   OBC_GMB binner(feature_vec, target_vec, min_bins, max_bins, 
                                       bin_cutoff, max_n_prebins, bin_separator, 
                                       convergence_threshold, max_iterations);
   return binner.fit();
 } catch (const std::exception& e) {
   Rcpp::stop("Error in optimal binning: %s", e.what());
 }
}