// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <numeric>
#include <functional>

using namespace Rcpp;

// Include shared headers
#include "common/optimal_binning_common.h"
#include "common/bin_structures.h"

using namespace Rcpp;
using namespace OptimalBinning;


// Global constants for better consistency and clarity
// Constant removed (uses shared definition)
static constexpr double LAPLACE_ALPHA = 0.5;  // Laplace smoothing parameter
static constexpr const char* MISSING_VALUE = "__MISSING__";  // Special category for missing values

// Namespace for utility functions
namespace utils {
// Safe logarithm function to avoid -Inf
inline double safe_log(double x) {
  return x > EPSILON ? std::log(x) : std::log(EPSILON);
}

// Calculate Weight of Evidence with Laplace smoothing
inline double calculate_woe(int pos, int neg, int total_pos, int total_neg, double alpha = LAPLACE_ALPHA) {
  // Apply Laplace (add-alpha) smoothing
  double pos_rate = (pos + alpha) / (total_pos + alpha * 2);
  double neg_rate = (neg + alpha) / (total_neg + alpha * 2);
  
  return safe_log(pos_rate / neg_rate);
}

// Calculate Information Value with Laplace smoothing
inline double calculate_iv(int pos, int neg, int total_pos, int total_neg, double alpha = LAPLACE_ALPHA) {
  // Apply Laplace smoothing
  double pos_rate = (pos + alpha) / (total_pos + alpha * 2);
  double neg_rate = (neg + alpha) / (total_neg + alpha * 2);
  
  double woe = safe_log(pos_rate / neg_rate);
  return (pos_rate - neg_rate) * woe;
}

// Calculate total IV for a vector of bins
template <typename BinType>
inline double calculate_total_iv(const std::vector<BinType>& bins, int total_pos, int total_neg) {
  double total_iv = 0.0;
  
  for (const auto& bin : bins) {
    total_iv += calculate_iv(bin.count_pos, bin.count_neg, total_pos, total_neg);
  }
  
  return total_iv;
}

// Join vector of strings ensuring uniqueness
inline std::string join_categories(const std::vector<std::string>& categories, const std::string& delimiter) {
  if (categories.empty()) return "";
  if (categories.size() == 1) return categories[0];
  
  // Create a set for uniqueness check
  std::unordered_set<std::string> unique_categories;
  std::vector<std::string> unique_vector;
  unique_vector.reserve(categories.size());
  
  for (const auto& cat : categories) {
    if (unique_categories.insert(cat).second) {
      unique_vector.push_back(cat);
    }
  }
  
  // Estimate result size for pre-allocation
  size_t total_length = 0;
  for (const auto& cat : unique_vector) {
    total_length += cat.length();
  }
  total_length += delimiter.length() * (unique_vector.size() - 1);
  
  // Build result string
  std::string result;
  result.reserve(total_length);
  
  result = unique_vector[0];
  for (size_t i = 1; i < unique_vector.size(); ++i) {
    result += delimiter;
    result += unique_vector[i];
  }
  
  return result;
}

// Calculate Jensen-Shannon divergence between two bins
inline double calculate_divergence(int bin1_pos, int bin1_neg, int bin2_pos, int bin2_neg, 
                                   int total_pos, int total_neg, double alpha = LAPLACE_ALPHA) {
  // Smoothed proportions for bin 1
  double p1 = (bin1_pos + alpha) / (total_pos + alpha * 2);
  double n1 = (bin1_neg + alpha) / (total_neg + alpha * 2);
  
  // Smoothed proportions for bin 2
  double p2 = (bin2_pos + alpha) / (total_pos + alpha * 2);
  double n2 = (bin2_neg + alpha) / (total_neg + alpha * 2);
  
  // Average proportions
  double p_avg = (p1 + p2) / 2;
  double n_avg = (n1 + n2) / 2;
  
  // KL divergence components
  double div_p1 = p1 > EPSILON ? p1 * safe_log(p1 / p_avg) : 0;
  double div_n1 = n1 > EPSILON ? n1 * safe_log(n1 / n_avg) : 0;
  double div_p2 = p2 > EPSILON ? p2 * safe_log(p2 / p_avg) : 0;
  double div_n2 = n2 > EPSILON ? n2 * safe_log(n2 / n_avg) : 0;
  
  // Jensen-Shannon divergence (symmetric)
  return (div_p1 + div_n1 + div_p2 + div_n2) / 2;
}
}

// Improved Categorical Binning with Sliding Window Binning (SWB)
class OBC_SWB {
private:
  // Enhanced bin statistics structure with uniqueness guarantee
  // Local CategoricalBin definition removed

  
  // Input data and parameters
  std::vector<std::string> feature;
  std::vector<int> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  std::string bin_separator;
  double convergence_threshold;
  int max_iterations;
  
  // Internal state
  std::vector<CategoricalBin> bins;
  int total_pos;
  int total_neg;
  bool converged;
  int iterations_run;
  
  // Initialize bins from raw data
  void initialize_bins() {
    std::unordered_map<std::string, CategoricalBin> initial_bins;
    total_pos = 0;
    total_neg = 0;
    
    // First pass: collect statistics for each unique category
    for (size_t i = 0; i < feature.size(); ++i) {
      const std::string& cat = feature[i];
      int target_val = target[i];
      
      { auto& b = initial_bins[cat]; if(b.categories.empty()) b.categories.push_back(cat); b.count++; if(target_val) b.count_pos++; else b.count_neg++; }
      
      total_pos += target_val;
      total_neg += (1 - target_val);
    }
    
    // Calculate minimum count threshold for a separate bin
    double count_threshold = bin_cutoff * feature.size();
    
    // Second pass: separate frequent and rare categories
    std::vector<CategoricalBin> temp_bins;
    CategoricalBin low_freq_bin;
    
    for (auto& pair : initial_bins) {
      if (pair.second.count >= count_threshold) {
        pair.second.calculate_metrics(total_pos, total_neg);
        temp_bins.push_back(std::move(pair.second));
      } else {
        low_freq_bin.merge_with(pair.second);
      }
    }
    
    // Add the rare categories bin if it's not empty
    if (low_freq_bin.count > 0) {
      low_freq_bin.calculate_metrics(total_pos, total_neg);
      temp_bins.push_back(std::move(low_freq_bin));
    }
    
    bins = std::move(temp_bins);
    
    // Sort bins by WoE for better merging strategy
    std::sort(bins.begin(), bins.end(), [](const CategoricalBin& a, const CategoricalBin& b) {
      return a.woe < b.woe;
    });
    
    // Initial consolidation to max_n_prebins if needed
    while (bins.size() > (size_t)max_n_prebins && bins.size() > (size_t)min_bins) {
      merge_most_similar_bins();
    }
  }
  
  // Find and merge the most similar adjacent bins based on information loss
  void merge_adjacent_bins() {
    if (bins.size() <= (size_t)min_bins) return;
    
    double min_iv_loss = std::numeric_limits<double>::max();
    size_t merge_index = 0;
    
    double original_iv = utils::calculate_total_iv(bins, total_pos, total_neg);
    
    // Find adjacent bins with minimal IV loss when merged
    for (size_t i = 0; i < bins.size() - 1; ++i) {
      CategoricalBin merged_bin = bins[i];
      merged_bin.merge_with(bins[i + 1]);
      merged_bin.calculate_metrics(total_pos, total_neg);
      
      // Create temporary set of bins with the merge applied
      std::vector<CategoricalBin> temp_bins = bins;
      temp_bins[i] = merged_bin;
      temp_bins.erase(temp_bins.begin() + i + 1);
      
      // Calculate IV after merge
      double new_iv = utils::calculate_total_iv(temp_bins, total_pos, total_neg);
      double iv_loss = original_iv - new_iv;
      
      if (iv_loss < min_iv_loss) {
        min_iv_loss = iv_loss;
        merge_index = i;
      }
    }
    
    // Perform the merge with minimal IV loss
    bins[merge_index].merge_with(bins[merge_index + 1]);
    bins[merge_index].calculate_metrics(total_pos, total_neg);
    bins.erase(bins.begin() + merge_index + 1);
  }
  
  // Find and merge the most similar bins based on statistical divergence
  void merge_most_similar_bins() {
    if (bins.size() <= (size_t)min_bins) return;
    
    double min_divergence = std::numeric_limits<double>::max();
    size_t merge_idx1 = 0;
    size_t merge_idx2 = 1;
    
    // Find the pair of bins with minimal Jensen-Shannon divergence
    for (size_t i = 0; i < bins.size(); ++i) {
      for (size_t j = i + 1; j < bins.size(); ++j) {
        double div = bins[i].divergence_from(bins[j], total_pos, total_neg);
        
        // Prefer adjacent bins when divergence is similar
        if (j == i + 1) {
          div *= 0.95;  // Small bias towards adjacent bins
        }
        
        if (div < min_divergence) {
          min_divergence = div;
          merge_idx1 = i;
          merge_idx2 = j;
        }
      }
    }
    
    // Perform the merge
    if (merge_idx2 < merge_idx1) std::swap(merge_idx1, merge_idx2);
    
    bins[merge_idx1].merge_with(bins[merge_idx2]);
    bins[merge_idx1].calculate_metrics(total_pos, total_neg);
    bins.erase(bins.begin() + merge_idx2);
    
    // Re-sort bins by WoE after merge
    std::sort(bins.begin(), bins.end(), [](const CategoricalBin& a, const CategoricalBin& b) {
      return a.woe < b.woe;
    });
  }
  
  // Optimize bins for monotonicity and IV
  void optimize_bins() {
    double prev_iv = utils::calculate_total_iv(bins, total_pos, total_neg);
    converged = false;
    iterations_run = 0;
    
    while (iterations_run < max_iterations) {
      // Check if current binning satisfies all constraints
      if (is_monotonic() && bins.size() <= (size_t)max_bins && bins.size() >= (size_t)min_bins) {
        converged = true;
        break;
      }
      
      // Decide appropriate action based on current state
      if (bins.size() > (size_t)max_bins) {
        // Too many bins - merge the most similar pair
        merge_most_similar_bins();
      } else if (bins.size() < (size_t)min_bins) {
        // Not enough bins - would need splitting but we avoid it for stability
        // Better to stop here than risk creating unstable bins
        break;
      } else if (!is_monotonic()) {
        // Non-monotonic - try to fix monotonicity violations
        improve_monotonicity();
      } else {
        // No issues found but we should never reach here
        converged = true;
        break;
      }
      
      // Check for convergence
      double current_iv = utils::calculate_total_iv(bins, total_pos, total_neg);
      if (std::abs(current_iv - prev_iv) < convergence_threshold) {
        converged = true;
        break;
      }
      prev_iv = current_iv;
      iterations_run++;
    }
    
    // Final consolidation if needed
    while (bins.size() > (size_t)max_bins) {
      merge_most_similar_bins();
    }
    
    // Final metrics calculation
    for (auto& bin : bins) {
      bin.calculate_metrics(total_pos, total_neg);
    }
  }
  
  // Check if current binning is monotonic in WoE
  bool is_monotonic() const {
    if (bins.size() <= 2) return true;  // 1 or 2 bins are always monotonic
    
    bool increasing = true;
    bool decreasing = true;
    
    for (size_t i = 1; i < bins.size(); ++i) {
      if (bins[i].woe < bins[i - 1].woe - EPSILON) {
        increasing = false;
      }
      if (bins[i].woe > bins[i - 1].woe + EPSILON) {
        decreasing = false;
      }
      // If neither pattern holds, binning is not monotonic
      if (!increasing && !decreasing) return false;
    }
    
    return true;
  }
  
  // Fix monotonicity violations
  void improve_monotonicity() {
    // Identify the most serious monotonicity violation
    double max_violation = 0.0;
    size_t violation_idx = 0;
    bool found_violation = false;
    
    // Find if we're generally increasing or decreasing
    bool should_increase = true;
    if (bins.size() >= 3) {
      // Check first few bins to determine overall trend
      int increasing_count = 0;
      int decreasing_count = 0;
      
      for (size_t i = 1; i < std::min(bins.size(), size_t(5)); ++i) {
        if (bins[i].woe > bins[i-1].woe) increasing_count++;
        else if (bins[i].woe < bins[i-1].woe) decreasing_count++;
      }
      
      should_increase = (increasing_count >= decreasing_count);
    }
    
    // Find the most severe violation
    for (size_t i = 1; i < bins.size(); ++i) {
      double violation = 0.0;
      
      if (should_increase && bins[i].woe < bins[i-1].woe) {
        violation = bins[i-1].woe - bins[i].woe;
      } else if (!should_increase && bins[i].woe > bins[i-1].woe) {
        violation = bins[i].woe - bins[i-1].woe;
      }
      
      if (violation > max_violation) {
        max_violation = violation;
        violation_idx = i - 1; // Index of first bin in the violating pair
        found_violation = true;
      }
    }
    
    // Fix the violation by merging
    if (found_violation) {
      merge_bins(violation_idx, violation_idx + 1);
    }
  }
  
  // Merge two bins
  void merge_bins(size_t index1, size_t index2) {
    if (index1 >= bins.size() || index2 >= bins.size() || index1 == index2) {
      return;
    }
    
    bins[index1].merge_with(bins[index2]);
    bins[index1].calculate_metrics(total_pos, total_neg);
    bins.erase(bins.begin() + index2);
  }
  
public:
  // Constructor with improved validation
  OBC_SWB(const std::vector<std::string>& feature,
                               const std::vector<int>& target,
                               int min_bins = 3,
                               int max_bins = 5,
                               double bin_cutoff = 0.05,
                               int max_n_prebins = 20,
                               std::string bin_separator = "%;%",
                               double convergence_threshold = 1e-6,
                               int max_iterations = 1000)
    : feature(feature),
      target(target),
      min_bins(min_bins),
      max_bins(max_bins),
      bin_cutoff(bin_cutoff),
      max_n_prebins(max_n_prebins),
      bin_separator(bin_separator),
      convergence_threshold(convergence_threshold),
      max_iterations(max_iterations),
      converged(false),
      iterations_run(0) {
    
    // Validate inputs
    if (feature.size() != target.size()) {
      Rcpp::stop("Feature and target vectors must have the same length");
    }
    
    if (feature.empty()) {
      Rcpp::stop("Feature and target vectors cannot be empty");
    }
    
    // Validate target values (must be binary)
    for (int val : target) {
      if (val != 0 && val != 1) {
        Rcpp::stop("Target must contain only binary values (0 or 1)");
      }
    }
    
    // Check if target has both 0 and 1
    bool has_zero = false;
    bool has_one = false;
    
    for (int val : target) {
      if (val == 0) has_zero = true;
      else if (val == 1) has_one = true;
      
      if (has_zero && has_one) break;
    }
    
    if (!has_zero || !has_one) {
      Rcpp::stop("Target must contain both 0 and 1 values");
    }
    
    // Adjust min_bins and max_bins based on unique categories
    std::unordered_set<std::string> unique_cats(feature.begin(), feature.end());
    int ncat = static_cast<int>(unique_cats.size());
    
    // Cap max_bins at number of unique categories
    max_bins = std::min(max_bins, ncat);
    
    // Ensure min_bins is valid (at least 1 and not more than max_bins)
    min_bins = std::max(1, std::min(min_bins, max_bins));
    
    // Validate other parameters
    if (bin_cutoff <= 0 || bin_cutoff >= 1) {
      Rcpp::stop("bin_cutoff must be between 0 and 1");
    }
    
    if (max_n_prebins < min_bins) {
      Rcpp::stop("max_n_prebins must be at least min_bins");
    }
  }
  
  // Main fitting function
  void fit() {
    // Count unique categories
    std::unordered_set<std::string> unique_cats(feature.begin(), feature.end());
    int ncat = static_cast<int>(unique_cats.size());
    
    // Handle special case of very few categories
    if (ncat <= 2) {
      // Process each unique category as a separate bin
      std::unordered_map<std::string, CategoricalBin> bin_map;
      total_pos = 0;
      total_neg = 0;
      
      for (size_t i = 0; i < feature.size(); ++i) {
        auto& bin = bin_map[feature[i]];
        if (bin.categories.empty()) {
          bin.categories.push_back(feature[i]);
        }
        bin.count++;
        if (target[i] == 1) {
          bin.count_pos++;
        } else {
          bin.count_neg++;
        }
        total_pos += target[i];
        total_neg += (1 - target[i]);
      }
      
      // Transfer to bins vector
      bins.clear();
      for (auto& kv : bin_map) {
        bins.push_back(std::move(kv.second));
      }
      
      // Calculate final metrics
      for (auto& bin : bins) {
        bin.calculate_metrics(total_pos, total_neg);
      }
      
      converged = true;
      iterations_run = 0;
      return;
    }
    
    // Normal processing for more than 2 categories
    initialize_bins();
    optimize_bins();
  }
  
  // Get results as Rcpp List
  Rcpp::List get_results() const {
    // Prepare result vectors
    std::vector<std::string> bin_categories;
    std::vector<double> woes;
    std::vector<double> ivs;
    std::vector<int> counts;
    std::vector<int> counts_pos;
    std::vector<int> counts_neg;
    std::vector<double> event_rates;
    
    // Fill result vectors
    for (const auto& bin : bins) {
      std::string bin_name = utils::join_categories(bin.categories, bin_separator);
      bin_categories.push_back(bin_name);
      woes.push_back(bin.woe);
      ivs.push_back(bin.iv);
      counts.push_back(bin.count);
      counts_pos.push_back(bin.count_pos);
      counts_neg.push_back(bin.count_neg);
      event_rates.push_back(bin.event_rate());
    }
    
    // Calculate total IV
    double total_iv = 0.0;
    for (const auto& iv : ivs) {
      total_iv += std::fabs(iv);
    }
    
    // Create sequential IDs
    Rcpp::NumericVector ids(bin_categories.size());
    for (size_t i = 0; i < bin_categories.size(); i++) {
      ids[i] = i + 1;
    }
    
    // Return results
    return Rcpp::List::create(
      Rcpp::Named("id") = ids,
      Rcpp::Named("bin") = bin_categories,
      Rcpp::Named("woe") = woes,
      Rcpp::Named("iv") = ivs,
      Rcpp::Named("count") = counts,
      Rcpp::Named("count_pos") = counts_pos,
      Rcpp::Named("count_neg") = counts_neg,
      Rcpp::Named("event_rate") = event_rates,
      Rcpp::Named("converged") = converged,
      Rcpp::Named("iterations") = iterations_run,
      Rcpp::Named("total_iv") = total_iv
    );
  }
};

// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_swb(Rcpp::IntegerVector target,
                                           Rcpp::CharacterVector feature,
                                          int min_bins = 3,
                                          int max_bins = 5,
                                          double bin_cutoff = 0.05,
                                          int max_n_prebins = 20,
                                          std::string bin_separator = "%;%",
                                          double convergence_threshold = 1e-6,
                                          int max_iterations = 1000) {
 try {
   // Handle missing values in feature
   std::vector<std::string> feature_vec;
   feature_vec.reserve(feature.size());
   
   for (R_xlen_t i = 0; i < feature.size(); ++i) {
     if (feature[i] == NA_STRING) {
       feature_vec.push_back(MISSING_VALUE);
     } else {
       feature_vec.push_back(Rcpp::as<std::string>(feature[i]));
     }
   }
   
   std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
   
   // Check for missing values in target
   for (R_xlen_t i = 0; i < target.size(); ++i) {
     if (IntegerVector::is_na(target[i])) {
       Rcpp::stop("Target cannot contain missing values");
     }
   }
   
   OBC_SWB binner(feature_vec, target_vec, min_bins, max_bins,
                                       bin_cutoff, max_n_prebins, bin_separator,
                                       convergence_threshold, max_iterations);
   binner.fit();
   return binner.get_results();
 } catch (const std::exception& e) {
   Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
 }
}