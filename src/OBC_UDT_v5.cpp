// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <numeric>
#include <functional>
#include <sstream>

using namespace Rcpp;

// Include shared headers
#include "common/optimal_binning_common.h"
#include "common/bin_structures.h"

using namespace Rcpp;
using namespace OptimalBinning;


// Global constants for better readability and consistency
// Constant removed (uses shared definition)
static constexpr double LAPLACE_ALPHA = 0.5;  // Laplace smoothing parameter
static constexpr const char* MISSING_VALUE = "N/A";  // Special category for missing values

// Namespace for utility functions
namespace utils {
// Safe logarithm function to avoid -Inf
inline double safe_log(double x) {
  return x > EPSILON ? std::log(x) : std::log(EPSILON);
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

// Calculate Jensen-Shannon divergence between two bins
inline double calculate_divergence(
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

// Join vector of categories with uniqueness checking
inline std::string join_categories(const std::vector<std::string>& categories, 
                                   const std::string& separator) {
  if (categories.empty()) return "";
  if (categories.size() == 1) return categories[0];
  
  // Ensure uniqueness
  std::unordered_set<std::string> unique_cats;
  std::vector<std::string> unique_vec;
  unique_vec.reserve(categories.size());
  
  for (const auto& cat : categories) {
    if (unique_cats.insert(cat).second) {
      unique_vec.push_back(cat);
    }
  }
  
  // Join with separator
  std::ostringstream result;
  result << unique_vec[0];
  for (size_t i = 1; i < unique_vec.size(); ++i) {
    result << separator << unique_vec[i];
  }
  
  return result.str();
}
}

class OBC_UDT {
private:
  // Enhanced bin structure with uniqueness guarantee
  // Local CategoricalBin definition removed

  
  // Class parameters
  int min_bins_;
  int max_bins_;
  double bin_cutoff_;
  int max_n_prebins_;
  std::string bin_separator_;
  double convergence_threshold_;
  int max_iterations_;
  
  // Internal state
  std::vector<CategoricalBin> bins_;
  bool converged_;
  int iterations_;
  int total_pos_;
  int total_neg_;
  
  // Input validation with improved error messages
  void validate_inputs(const std::vector<std::string>& feature, const std::vector<int>& target) {
    if (feature.size() != target.size()) {
      throw std::invalid_argument("Feature and target vectors must have the same length.");
    }
    if (feature.empty()) {
      throw std::invalid_argument("Input vectors cannot be empty.");
    }
    
    // Check target values and count positives/negatives
    bool has_zero = false;
    bool has_one = false;
    
    for (int t : target) {
      if (t == 0) has_zero = true;
      else if (t == 1) has_one = true;
      else throw std::invalid_argument("Target vector must contain only 0 and 1.");
      
      // Early termination
      if (has_zero && has_one) break;
    }
    
    if (!has_zero || !has_one) {
      throw std::invalid_argument("Target must contain both 0 and 1 values.");
    }
    
    // Validate parameter ranges
    if (min_bins_ < 1) {
      throw std::invalid_argument("min_bins must be at least 1.");
    }
    if (max_bins_ < min_bins_) {
      throw std::invalid_argument("max_bins must be greater than or equal to min_bins.");
    }
    if (bin_cutoff_ <= 0 || bin_cutoff_ >= 1) {
      throw std::invalid_argument("bin_cutoff must be between 0 and 1 (exclusive).");
    }
    if (max_n_prebins_ < min_bins_) {
      throw std::invalid_argument("max_n_prebins must be at least min_bins.");
    }
  }
  
  // Initial binning with one bin per unique category
  void initial_binning(const std::vector<std::string>& feature, const std::vector<int>& target) {
    std::unordered_map<std::string, CategoricalBin> bin_map;
    total_pos_ = 0;
    total_neg_ = 0;
    
    // Process each observation
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
      total_pos_ += target[i];
      total_neg_ += (1 - target[i]);
    }
    
    // Transfer to bins vector
    bins_.clear();
    bins_.reserve(bin_map.size());
    for (auto& pair : bin_map) {
      bins_.push_back(std::move(pair.second));
    }
  }
  
  // Merge low-frequency bins with improved strategy
  void merge_low_frequency_bins() {
    // Calculate cutoff threshold
    int total_count = std::accumulate(bins_.begin(), bins_.end(), 0,
                                      [](int sum, const CategoricalBin& bin) { return sum + bin.count; });
    double cutoff_count = total_count * bin_cutoff_;
    
    // Sort bins by count (ascending)
    std::sort(bins_.begin(), bins_.end(), [](const CategoricalBin& a, const CategoricalBin& b) {
      return a.count < b.count;
    });
    
    // Process bins, keeping those above threshold
    std::vector<CategoricalBin> new_bins;
    CategoricalBin low_freq_bin;
    
    for (auto& bin : bins_) {
      if (bin.count >= cutoff_count || (int)new_bins.size() < min_bins_) {
        new_bins.push_back(bin);
      } else {
        // Merge into low frequency bin
        low_freq_bin.merge_with(bin);
      }
    }
    
    // Add low frequency bin if not empty
    if (low_freq_bin.count > 0) {
      new_bins.push_back(low_freq_bin);
    }
    
    bins_ = std::move(new_bins);
  }
  
  // Calculate WoE and IV for all bins with Laplace smoothing
  void calculate_woe_iv() {
    for (auto& bin : bins_) {
      bin.calculate_metrics(total_pos_, total_neg_);
    }
  }
  
  // Calculate total IV across all bins
  double calculate_total_iv() const {
    return std::accumulate(bins_.begin(), bins_.end(), 0.0,
                           [](double sum, const CategoricalBin& bin) { return sum + std::fabs(bin.iv); });
  }
  
  // Ensure monotonicity by sorting bins by WoE
  void ensure_monotonicity() {
    std::sort(bins_.begin(), bins_.end(), [](const CategoricalBin& a, const CategoricalBin& b) {
      return a.woe < b.woe;
    });
  }
  
  // Find the most similar bins for merging based on statistical divergence
  std::pair<size_t, size_t> find_most_similar_bins() const {
    double min_divergence = std::numeric_limits<double>::max();
    size_t idx1 = 0, idx2 = 1;
    
    for (size_t i = 0; i < bins_.size(); ++i) {
      for (size_t j = i + 1; j < bins_.size(); ++j) {
        double div = bins_[i].divergence_from(bins_[j], total_pos_, total_neg_);
        
        // Prefer adjacent bins if divergence is similar
        if (j == i + 1) {
          div *= 0.95;  // Slight bias towards adjacent bins
        }
        
        if (div < min_divergence) {
          min_divergence = div;
          idx1 = i;
          idx2 = j;
        }
      }
    }
    
    return {idx1, idx2};
  }
  
  // Merge bins with improved strategy using statistical similarity
  void merge_bins() {
    while ((int)bins_.size() > max_bins_) {
      // Find most statistically similar bins
      auto [idx1, idx2] = find_most_similar_bins();
      
      // Ensure lower index first
      if (idx2 < idx1) std::swap(idx1, idx2);
      
      // Merge bins
      bins_[idx1].merge_with(bins_[idx2]);
      bins_[idx1].calculate_metrics(total_pos_, total_neg_);
      bins_.erase(bins_.begin() + idx2);
      
      // Recalculate WoE/IV after merge
      calculate_woe_iv();
    }
  }
  
public:
  // Constructor with improved defaults and documentation
  OBC_UDT(
    int min_bins = 3,
    int max_bins = 5,
    double bin_cutoff = 0.05,
    int max_n_prebins = 20,
    std::string bin_separator = "%;%",
    double convergence_threshold = 1e-6,
    int max_iterations = 1000
  ) : min_bins_(min_bins), max_bins_(max_bins), bin_cutoff_(bin_cutoff),
  max_n_prebins_(max_n_prebins), bin_separator_(bin_separator),
  convergence_threshold_(convergence_threshold), max_iterations_(max_iterations),
  converged_(false), iterations_(0), total_pos_(0), total_neg_(0) {}
  
  // Main fitting method with improved logic
  void fit(const std::vector<std::string>& feature, const std::vector<int>& target) {
    validate_inputs(feature, target);
    
    // Count unique categories
    std::unordered_set<std::string> unique_cats(feature.begin(), feature.end());
    int ncat = static_cast<int>(unique_cats.size());
    
    // Adjust min_bins and max_bins based on unique categories
    max_bins_ = std::min(max_bins_, ncat);
    min_bins_ = std::min(min_bins_, max_bins_);
    
    // Initial binning (one bin per category)
    initial_binning(feature, target);
    
    // Special case: 1 or 2 unique levels
    if (ncat <= 2) {
      calculate_woe_iv();
      converged_ = true;
      iterations_ = 0;
      return;
    }
    
    // Merge low frequency bins
    merge_low_frequency_bins();
    calculate_woe_iv();
    ensure_monotonicity();
    
    // Main optimization loop
    double prev_total_iv = calculate_total_iv();
    converged_ = false;
    iterations_ = 0;
    
    while (!converged_ && iterations_ < max_iterations_) {
      // Merge bins if needed
      if ((int)bins_.size() > max_bins_) {
        merge_bins();
      } else {
        // If within min_bins and max_bins range, we're done
        if ((int)bins_.size() >= min_bins_) {
          converged_ = true;
          break;
        }
        
        // If we can't increase the number of bins, we're done
        // No splitting is performed to avoid artificial categories
        converged_ = true;
        break;
      }
      
      // Ensure monotonicity
      ensure_monotonicity();
      
      // Check convergence
      double total_iv = calculate_total_iv();
      if (std::abs(total_iv - prev_total_iv) < convergence_threshold_) {
        converged_ = true;
      }
      
      prev_total_iv = total_iv;
      iterations_++;
    }
    
    // Final calculations
    calculate_woe_iv();
    ensure_monotonicity();
  }
  
  // Get results as Rcpp List with improved structure
  Rcpp::List get_woe_bin() const {
    // Prepare result vectors
    Rcpp::CharacterVector bin_names;
    Rcpp::NumericVector woe_values, iv_values, event_rates;
    Rcpp::IntegerVector counts, counts_pos, counts_neg;
    
    // Fill result vectors
    for (const auto& bin : bins_) {
      bin_names.push_back(utils::join_categories(bin.categories, bin_separator_));
      woe_values.push_back(bin.woe);
      iv_values.push_back(bin.iv);
      counts.push_back(bin.count);
      counts_pos.push_back(bin.count_pos);
      counts_neg.push_back(bin.count_neg);
      event_rates.push_back(bin.event_rate());
    }
    
    // Calculate total IV
    double total_iv = calculate_total_iv();
    
    // Create sequential IDs
    Rcpp::NumericVector ids(bin_names.size());
    for (int i = 0; i < bin_names.size(); i++) {
      ids[i] = i + 1;
    }
    
    // Return results
    return Rcpp::List::create(
      Rcpp::Named("id") = ids,
      Rcpp::Named("bin") = bin_names,
      Rcpp::Named("woe") = woe_values,
      Rcpp::Named("iv") = iv_values,
      Rcpp::Named("count") = counts,
      Rcpp::Named("count_pos") = counts_pos,
      Rcpp::Named("count_neg") = counts_neg,
      Rcpp::Named("event_rate") = event_rates,
      Rcpp::Named("converged") = converged_,
      Rcpp::Named("iterations") = iterations_,
      Rcpp::Named("total_iv") = total_iv
    );
  }
};

// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_udt(
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
   
   // Check for missing values in target
   std::vector<int> target_vec;
   target_vec.reserve(target.size());
   
   for (R_xlen_t i = 0; i < target.size(); ++i) {
     if (IntegerVector::is_na(target[i])) {
       Rcpp::stop("Target cannot contain missing values");
     }
     target_vec.push_back(target[i]);
   }
   
   OBC_UDT binning(
       min_bins, max_bins, bin_cutoff, max_n_prebins,
       bin_separator, convergence_threshold, max_iterations
   );
   
   binning.fit(feature_vec, target_vec);
   return binning.get_woe_bin();
 } catch (const std::exception& e) {
   Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
 }
}