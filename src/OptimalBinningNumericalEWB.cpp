#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <iomanip>
#include <unordered_set>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

// [[Rcpp::plugins(openmp)]]
using namespace Rcpp;

class OptimalBinningNumericalEWB {
private:
  std::vector<double> feature;
  std::vector<int> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  
  struct Bin {
    double lower;
    double upper;
    int count;
    int count_pos;
    int count_neg;
    double woe;
    double iv;
    
    Bin(double lb = -std::numeric_limits<double>::infinity(),
        double ub = std::numeric_limits<double>::infinity(),
        int c = 0, int cp = 0, int cn = 0)
      : lower(lb), upper(ub), count(c), count_pos(cp), count_neg(cn), woe(0.0), iv(0.0) {}
  };
  
  std::vector<Bin> bins;
  
  // Total positives and negatives in the dataset
  int total_pos;
  int total_neg;
  
  // Helper function to convert double to string with proper formatting
  std::string double_to_string(double value) const {
    if (std::isinf(value)) {
      return value > 0 ? "+Inf" : "-Inf";
    }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6) << value;
    return ss.str();
  }
  
  // Calculate Weight of Evidence (WoE)
  double calculate_woe(int pos, int neg) const {
    if (pos == 0 || neg == 0) {
      // Handle division by zero or log(0)
      return 0.0;
    }
    double pos_rate = static_cast<double>(pos) / total_pos;
    double neg_rate = static_cast<double>(neg) / total_neg;
    return std::log(pos_rate / neg_rate);
  }
  
  // Calculate Information Value (IV)
  double calculate_iv(double woe, int pos, int neg) const {
    if (pos == 0 || neg == 0) {
      return 0.0;
    }
    double pos_rate = static_cast<double>(pos) / total_pos;
    double neg_rate = static_cast<double>(neg) / total_neg;
    return (pos_rate - neg_rate) * woe;
  }
  
  // Validate input parameters and data
  void validate_inputs() {
    if (feature.empty()) {
      Rcpp::stop("Feature vector is empty.");
    }
    if (feature.size() != target.size()) {
      Rcpp::stop("Feature and target vectors must have the same length.");
    }
    
    // Check target values
    for (const auto& t : target) {
      if (t != 0 && t != 1) {
        Rcpp::stop("Target vector must contain only 0 and 1.");
      }
    }
    
    // Determine unique feature values
    std::unordered_set<double> unique_values(feature.begin(), feature.end());
    int unique_count = unique_values.size();
    
    // Adjust min_bins and max_bins based on unique categories
    if (unique_count <= 2) {
      min_bins = unique_count;
      max_bins = unique_count;
      if (bin_cutoff > 1.0 || bin_cutoff <= 0.0) {
        bin_cutoff = 1.0; // All data in one bin
      }
    } else {
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
    }
  }
  
  // Create initial equal-width pre-bins
  void create_prebins() {
    double min_value = *std::min_element(feature.begin(), feature.end());
    double max_value = *std::max_element(feature.begin(), feature.end());
    
    // Handle case where all feature values are the same
    if (min_value == max_value) {
      bins.emplace_back(min_value, max_value, feature.size(),
                        std::count(target.begin(), target.end(), 1),
                        std::count(target.begin(), target.end(), 0));
      return;
    }
    
    int n_prebins = std::min(max_n_prebins, static_cast<int>(feature.size()));
    double bin_width = (max_value - min_value) / n_prebins;
    
    bins.clear();
    bins.reserve(n_prebins);
    
    for (int i = 0; i < n_prebins; ++i) {
      double lower = (i == 0) ? -std::numeric_limits<double>::infinity() : min_value + i * bin_width;
      double upper = (i == n_prebins - 1) ? std::numeric_limits<double>::infinity() : min_value + (i + 1) * bin_width;
      bins.emplace_back(lower, upper, 0, 0, 0);
    }
    
    // Assign data to bins
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < feature.size(); ++i) {
      double value = feature[i];
      int target_value = target[i];
      int bin_index = 0;
      
      // Binary search to find the appropriate bin
      for (size_t b = 0; b < bins.size(); ++b) {
        if (value <= bins[b].upper) {
          bin_index = b;
          break;
        }
      }
      
#ifdef _OPENMP
#pragma omp atomic
#endif
      bins[bin_index].count++;
      
      if (target_value == 1) {
#ifdef _OPENMP
#pragma omp atomic
#endif
        bins[bin_index].count_pos++;
      } else {
#ifdef _OPENMP
#pragma omp atomic
#endif
        bins[bin_index].count_neg++;
      }
    }
  }
  
  // Merge bins that are below the bin_cutoff threshold
  void merge_rare_bins() {
    int total_count = std::accumulate(bins.begin(), bins.end(), 0,
                                      [](int sum, const Bin& bin) { return sum + bin.count; });
    double cutoff_count = bin_cutoff * total_count;
    
    std::vector<Bin> merged_bins;
    merged_bins.reserve(bins.size());
    
    Bin current_bin = bins[0];
    
    for (size_t i = 1; i < bins.size(); ++i) {
      if (current_bin.count < cutoff_count) {
        // Merge with the next bin
        current_bin.upper = bins[i].upper;
        current_bin.count += bins[i].count;
        current_bin.count_pos += bins[i].count_pos;
        current_bin.count_neg += bins[i].count_neg;
      } else {
        merged_bins.emplace_back(current_bin);
        current_bin = bins[i];
      }
    }
    merged_bins.emplace_back(current_bin);
    
    // Iteratively merge until all bins meet the cutoff
    bool merged = true;
    while (merged && merged_bins.size() > 1) {
      merged = false;
      std::vector<Bin> temp_bins;
      temp_bins.reserve(merged_bins.size());
      
      current_bin = merged_bins[0];
      for (size_t i = 1; i < merged_bins.size(); ++i) {
        if (current_bin.count < cutoff_count) {
          // Merge with the next bin
          current_bin.upper = merged_bins[i].upper;
          current_bin.count += merged_bins[i].count;
          current_bin.count_pos += merged_bins[i].count_pos;
          current_bin.count_neg += merged_bins[i].count_neg;
          merged = true;
        } else {
          temp_bins.emplace_back(current_bin);
          current_bin = merged_bins[i];
        }
      }
      temp_bins.emplace_back(current_bin);
      merged_bins = temp_bins;
    }
    
    bins = merged_bins;
  }
  
  // Ensure minimum number of bins by merging the smallest bins
  void ensure_min_bins() {
    while (static_cast<int>(bins.size()) < min_bins && bins.size() > 1) {
      // Find the bin with the smallest count
      auto min_it = std::min_element(bins.begin(), bins.end(),
                                     [](const Bin& a, const Bin& b) { return a.count < b.count; });
      size_t index = std::distance(bins.begin(), min_it);
      
      if (index == 0) {
        // Merge with next bin
        bins[1].lower = bins[0].lower;
        bins[1].count += bins[0].count;
        bins[1].count_pos += bins[0].count_pos;
        bins[1].count_neg += bins[0].count_neg;
        bins.erase(bins.begin());
      } else {
        // Merge with previous bin
        bins[index - 1].upper = bins[index].upper;
        bins[index - 1].count += bins[index].count;
        bins[index - 1].count_pos += bins[index].count_pos;
        bins[index - 1].count_neg += bins[index].count_neg;
        bins.erase(bins.begin() + index);
      }
    }
  }
  
  // Optimize bins to not exceed max_bins by merging adjacent bins with smallest IV loss
  void optimize_bins() {
    if (bins.size() <= static_cast<size_t>(max_bins)) {
      return;
    }
    
    while (static_cast<int>(bins.size()) > max_bins) {
      double min_iv_loss = std::numeric_limits<double>::max();
      int merge_index = -1;
      
      // Precompute total_pos and total_neg
      // Already computed in total_pos and total_neg
      
      // Find the pair of adjacent bins with the smallest IV loss when merged
      for (size_t i = 0; i < bins.size() - 1; ++i) {
        // Calculate IV before merging
        double iv_before = calculate_iv(calculate_woe(bins[i].count_pos, bins[i].count_neg),
                                        bins[i].count_pos, bins[i].count_neg) +
                                          calculate_iv(calculate_woe(bins[i + 1].count_pos, bins[i + 1].count_neg),
                                                       bins[i + 1].count_pos, bins[i + 1].count_neg);
        
        // Calculate counts after merging
        int merged_pos = bins[i].count_pos + bins[i + 1].count_pos;
        int merged_neg = bins[i].count_neg + bins[i + 1].count_neg;
        
        // Calculate IV after merging
        double woe_merged = calculate_woe(merged_pos, merged_neg);
        double iv_after = calculate_iv(woe_merged, merged_pos, merged_neg);
        
        double iv_loss = iv_before - iv_after;
        
        if (iv_loss < min_iv_loss) {
          min_iv_loss = iv_loss;
          merge_index = static_cast<int>(i);
        }
      }
      
      if (merge_index == -1) {
        break; // No suitable pair found
      }
      
      // Merge the identified pair of bins
      bins[merge_index].upper = bins[merge_index + 1].upper;
      bins[merge_index].count += bins[merge_index + 1].count;
      bins[merge_index].count_pos += bins[merge_index + 1].count_pos;
      bins[merge_index].count_neg += bins[merge_index + 1].count_neg;
      
      // Remove the next bin
      bins.erase(bins.begin() + merge_index + 1);
    }
  }
  
  // Calculate WoE and IV for each bin
  void calculate_woe_iv() {
    for (auto& bin : bins) {
      bin.woe = calculate_woe(bin.count_pos, bin.count_neg);
      bin.iv = calculate_iv(bin.woe, bin.count_pos, bin.count_neg);
    }
  }
  
public:
  OptimalBinningNumericalEWB(const std::vector<double>& feature_, const std::vector<int>& target_,
                             int min_bins_ = 3, int max_bins_ = 5, double bin_cutoff_ = 0.05, int max_n_prebins_ = 20)
    : feature(feature_), target(target_), min_bins(min_bins_), max_bins(max_bins_),
      bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_), total_pos(0), total_neg(0) {}
  
  void fit() {
    validate_inputs();
    
    // Calculate total positives and negatives
    total_pos = std::accumulate(target.begin(), target.end(), 0);
    total_neg = target.size() - total_pos;
    
    if (total_pos == 0 || total_neg == 0) {
      Rcpp::stop("Target vector must contain at least one positive and one negative case.");
    }
    
    // Create initial pre-bins
    create_prebins();
    
    // Merge rare bins
    merge_rare_bins();
    
    // Ensure minimum number of bins
    ensure_min_bins();
    
    // Optimize bins to not exceed max_bins
    optimize_bins();
    
    // Calculate WoE and IV
    calculate_woe_iv();
  }
  
  // Get binning results
  List get_results() const {
    std::vector<std::string> bin_labels;
    std::vector<double> woe_values;
    std::vector<double> iv_values;
    std::vector<int> counts;
    std::vector<int> counts_pos;
    std::vector<int> counts_neg;
    
    for (const auto& bin : bins) {
      std::string label = "(" + (bin.lower == -std::numeric_limits<double>::infinity() ? "-Inf" : double_to_string(bin.lower)) + ";" +
        (bin.upper == std::numeric_limits<double>::infinity() ? "+Inf" : double_to_string(bin.upper)) + "]";
      bin_labels.push_back(label);
      woe_values.push_back(bin.woe);
      iv_values.push_back(bin.iv);
      counts.push_back(bin.count);
      counts_pos.push_back(bin.count_pos);
      counts_neg.push_back(bin.count_neg);
    }
    
    return List::create(
      Named("woebin") = DataFrame::create(
        Named("bin") = bin_labels,
        Named("woe") = woe_values,
        Named("iv") = iv_values,
        Named("count") = counts,
        Named("count_pos") = counts_pos,
        Named("count_neg") = counts_neg
      )
    );
  }
  
  // Transform new feature data using the calculated bins
  std::vector<double> transform(const std::vector<double>& new_feature) const {
    std::vector<double> woe_feature(new_feature.size(), 0.0);
    
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < new_feature.size(); ++i) {
      double value = new_feature[i];
      double woe = 0.0;
      
      for (const auto& bin : bins) {
        if (value > bin.lower && value <= bin.upper) {
          woe = bin.woe;
          break;
        }
      }
      
      // Handle values outside the bin ranges
      if (value <= bins.front().lower) {
        woe = bins.front().woe;
      } else if (value > bins.back().upper) {
        woe = bins.back().woe;
      }
      
      woe_feature[i] = woe;
    }
    
    return woe_feature;
  }
};


//' @title Optimal Binning for Numerical Variables using Equal-Width Binning
//'
//' @description
//' This function implements an optimal binning algorithm for numerical variables using an Equal-Width Binning approach with subsequent merging and adjustment. It aims to find a good binning strategy that balances interpretability and predictive power.
//'
//' @param target An integer vector of binary target values (0 or 1).
//' @param feature A numeric vector of feature values to be binned.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum fraction of total observations in each bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins (default: 20).
//'
//' @return A list containing:
//' \item{woefeature}{A numeric vector of Weight of Evidence (WoE) values for each observation}
//' \item{woebin}{A data frame with binning information, including bin ranges, WoE, IV, and counts}
//'
//' @details
//' The optimal binning algorithm using Equal-Width Binning consists of several steps:
//'
//' 1. Initial binning: The feature range is divided into \code{max_n_prebins} bins of equal width.
//' 2. Merging rare bins: Bins with a fraction of observations less than \code{bin_cutoff} are merged with adjacent bins.
//' 3. Adjusting number of bins: If the number of bins exceeds \code{max_bins}, adjacent bins with the most similar WoE values are merged until \code{max_bins} is reached.
//' 4. WoE and IV calculation: The Weight of Evidence (WoE) and Information Value (IV) are calculated for each bin.
//'
//' The Weight of Evidence (WoE) for each bin is calculated as:
//'
//' \deqn{WoE = \ln\left(\frac{P(X|Y=1)}{P(X|Y=0)}\right)}
//'
//' where \eqn{P(X|Y=1)} is the probability of the feature being in a particular bin given a positive target, and \eqn{P(X|Y=0)} is the probability given a negative target.
//'
//' The Information Value (IV) for each bin is calculated as:
//'
//' \deqn{IV = (P(X|Y=1) - P(X|Y=0)) * WoE}
//'
//' The total IV is the sum of IVs for all bins:
//'
//' \deqn{Total IV = \sum_{i=1}^{n} IV_i}
//'
//' This approach provides a balance between simplicity and effectiveness, creating bins of equal width initially and then adjusting them based on the data distribution and target variable relationship.
//'
//' @examples
//' \dontrun{
//' set.seed(123)
//' target <- sample(0:1, 1000, replace = TRUE)
//' feature <- rnorm(1000)
//' result <- optimal_binning_numerical_ewb(target, feature)
//' print(result$woebin)
//' }
//'
//' @references
//' \itemize{
//'   \item Dougherty, J., Kohavi, R., & Sahami, M. (1995). Supervised and unsupervised discretization of continuous features. In Machine Learning Proceedings 1995 (pp. 194-202). Morgan Kaufmann.
//'   \item Liu, H., Hussain, F., Tan, C. L., & Dash, M. (2002). Discretization: An enabling technique. Data mining and knowledge discovery, 6(4), 393-423.
//' }
//'
//' @author Lopes, J. E.
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_ewb(Rcpp::IntegerVector target, Rcpp::NumericVector feature,
                                        int min_bins = 3, int max_bins = 5, double bin_cutoff = 0.05,
                                        int max_n_prebins = 20) {
 // Convert R vectors to C++ vectors
 std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
 std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
 
 // Initialize the binning class
 OptimalBinningNumericalEWB binner(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins);
 
 // Fit the binning
 binner.fit();
 
 // Get binning results
 List binning_results = binner.get_results();
 
 // Transform the original feature using the bins to get WoE values
 std::vector<double> woe_feature = binner.transform(feature_vec);
 
 // Add woe_feature to the results
 binning_results["woefeature"] = woe_feature;
 
 return binning_results;
}






// #include <Rcpp.h>
// #include <vector>
// #include <algorithm>
// #include <cmath>
// #include <limits>
// 
// #ifdef _OPENMP
// #include <omp.h>
// #endif
// 
// 
// class OptimalBinningNumericalEWB {
// private:
//   std::vector<double> feature;
//   std::vector<int> target;
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
// 
//   struct Bin {
//     double lower;
//     double upper;
//     int count;
//     int count_pos;
//     int count_neg;
//     double woe;
//     double iv;
//   };
// 
//   std::vector<Bin> bins;
// 
//   double calculate_woe(int pos, int neg) {
//     double pos_rate = static_cast<double>(pos) / (pos + neg);
//     double neg_rate = static_cast<double>(neg) / (pos + neg);
//     return std::log(pos_rate / neg_rate);
//   }
// 
//   double calculate_iv(double woe, int pos, int neg, int total_pos, int total_neg) {
//     double pos_rate = static_cast<double>(pos) / total_pos;
//     double neg_rate = static_cast<double>(neg) / total_neg;
//     return (pos_rate - neg_rate) * woe;
//   }
// 
// public:
//   OptimalBinningNumericalEWB(const std::vector<double>& feature, const std::vector<int>& target,
//                              int min_bins = 3, int max_bins = 5, double bin_cutoff = 0.05, int max_n_prebins = 20)
//     : feature(feature), target(target), min_bins(min_bins), max_bins(max_bins),
//       bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins) {
//     if (min_bins < 2) {
//       Rcpp::stop("min_bins must be at least 2");
//     }
//     if (max_bins < min_bins) {
//       Rcpp::stop("max_bins must be greater than or equal to min_bins");
//     }
//   }
// 
//   void fit() {
//     // Sort feature and target together
//     std::vector<size_t> indices(feature.size());
//     std::iota(indices.begin(), indices.end(), 0);
//     std::sort(indices.begin(), indices.end(),
//               [this](size_t i1, size_t i2) { return feature[i1] < feature[i2]; });
// 
//     std::vector<double> sorted_feature(feature.size());
//     std::vector<int> sorted_target(target.size());
// 
// #pragma omp parallel for
//     for (size_t i = 0; i < indices.size(); ++i) {
//       sorted_feature[i] = feature[indices[i]];
//       sorted_target[i] = target[indices[i]];
//     }
// 
//     // Calculate initial equal-width bins
//     double min_value = sorted_feature.front();
//     double max_value = sorted_feature.back();
//     int n_prebins = std::min(max_n_prebins, static_cast<int>(sorted_feature.size()));
//     double bin_width = (max_value - min_value) / n_prebins;
// 
//     // Initialize bins
//     bins.clear();
//     for (int i = 0; i < n_prebins; ++i) {
//       double lower = (i == 0) ? std::numeric_limits<double>::lowest() : min_value + i * bin_width;
//       double upper = (i == n_prebins - 1) ? std::numeric_limits<double>::max() : min_value + (i + 1) * bin_width;
//       bins.push_back({lower, upper, 0, 0, 0, 0.0, 0.0});
//     }
// 
//     // Count samples in each bin
//     int total_pos = 0, total_neg = 0;
//     for (size_t i = 0; i < sorted_feature.size(); ++i) {
//       auto it = std::lower_bound(bins.begin(), bins.end(), sorted_feature[i],
//                                  [](const Bin& bin, double value) { return bin.upper < value; });
//       if (it != bins.end()) {
//         it->count++;
//         if (sorted_target[i] == 1) {
//           it->count_pos++;
//           total_pos++;
//         } else {
//           it->count_neg++;
//           total_neg++;
//         }
//       }
//     }
// 
//     // Merge rare bins
//     for (auto it = bins.begin(); it != bins.end(); ) {
//       if (static_cast<double>(it->count) / sorted_feature.size() < bin_cutoff) {
//         if (it != bins.begin()) {
//           auto prev = std::prev(it);
//           prev->upper = it->upper;
//           prev->count += it->count;
//           prev->count_pos += it->count_pos;
//           prev->count_neg += it->count_neg;
//           it = bins.erase(it);
//         } else {
//           auto next = std::next(it);
//           next->lower = it->lower;
//           next->count += it->count;
//           next->count_pos += it->count_pos;
//           next->count_neg += it->count_neg;
//           it = bins.erase(it);
//         }
//       } else {
//         ++it;
//       }
//     }
// 
//     // Ensure number of bins is within limits
//     while (static_cast<int>(bins.size()) > max_bins) {
//       // Find the pair of adjacent bins with the smallest difference in WoE
//       auto min_diff_it = bins.begin();
//       double min_diff = std::numeric_limits<double>::max();
// 
//       for (auto it = bins.begin(); it != std::prev(bins.end()); ++it) {
//         double woe1 = calculate_woe(it->count_pos, it->count_neg);
//         double woe2 = calculate_woe(std::next(it)->count_pos, std::next(it)->count_neg);
//         double diff = std::abs(woe1 - woe2);
// 
//         if (diff < min_diff) {
//           min_diff = diff;
//           min_diff_it = it;
//         }
//       }
// 
//       // Merge the bins
//       auto next_bin = std::next(min_diff_it);
//       min_diff_it->upper = next_bin->upper;
//       min_diff_it->count += next_bin->count;
//       min_diff_it->count_pos += next_bin->count_pos;
//       min_diff_it->count_neg += next_bin->count_neg;
//       bins.erase(next_bin);
//     }
// 
//     // Calculate WoE and IV for each bin
//     double total_iv = 0.0;
// #pragma omp parallel for reduction(+:total_iv)
//     for (auto& bin : bins) {
//       bin.woe = calculate_woe(bin.count_pos, bin.count_neg);
//       bin.iv = calculate_iv(bin.woe, bin.count_pos, bin.count_neg, total_pos, total_neg);
//       total_iv += bin.iv;
//     }
//   }
// 
//   Rcpp::List get_results() {
//     std::vector<std::string> bin_labels;
//     std::vector<double> woe_values;
//     std::vector<double> iv_values;
//     std::vector<int> counts;
//     std::vector<int> counts_pos;
//     std::vector<int> counts_neg;
// 
//     for (const auto& bin : bins) {
//       std::string label = "(" + (bin.lower == std::numeric_limits<double>::lowest() ? "-Inf" : std::to_string(bin.lower)) + ";" +
//         (bin.upper == std::numeric_limits<double>::max() ? "+Inf" : std::to_string(bin.upper)) + "]";
//       bin_labels.push_back(label);
//       woe_values.push_back(bin.woe);
//       iv_values.push_back(bin.iv);
//       counts.push_back(bin.count);
//       counts_pos.push_back(bin.count_pos);
//       counts_neg.push_back(bin.count_neg);
//     }
// 
//     return Rcpp::List::create(
//       Rcpp::Named("woebin") = Rcpp::DataFrame::create(
//         Rcpp::Named("bin") = bin_labels,
//         Rcpp::Named("woe") = woe_values,
//         Rcpp::Named("iv") = iv_values,
//         Rcpp::Named("count") = counts,
//         Rcpp::Named("count_pos") = counts_pos,
//         Rcpp::Named("count_neg") = counts_neg
//       )
//     );
//   }
// 
//   std::vector<double> transform(const std::vector<double>& new_feature) {
//     std::vector<double> woe_feature(new_feature.size());
// 
// #pragma omp parallel for
//     for (size_t i = 0; i < new_feature.size(); ++i) {
//       auto it = std::lower_bound(bins.begin(), bins.end(), new_feature[i],
//                                  [](const Bin& bin, double value) { return bin.upper < value; });
//       if (it != bins.end()) {
//         woe_feature[i] = it->woe;
//       } else {
//         woe_feature[i] = 0.0;  // or some other default value
//       }
//     }
// 
//     return woe_feature;
//   }
// };
// 
// 
// //' @title Optimal Binning for Numerical Variables using Equal-Width Binning
// //' 
// //' @description
// //' This function implements an optimal binning algorithm for numerical variables using an Equal-Width Binning approach with subsequent merging and adjustment. It aims to find a good binning strategy that balances interpretability and predictive power.
// //' 
// //' @param target An integer vector of binary target values (0 or 1).
// //' @param feature A numeric vector of feature values to be binned.
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff Minimum fraction of total observations in each bin (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins (default: 20).
// //' 
// //' @return A list containing:
// //' \item{woefeature}{A numeric vector of Weight of Evidence (WoE) values for each observation}
// //' \item{woebin}{A data frame with binning information, including bin ranges, WoE, IV, and counts}
// //' 
// //' @details
// //' The optimal binning algorithm using Equal-Width Binning consists of several steps:
// //' 
// //' 1. Initial binning: The feature range is divided into \code{max_n_prebins} bins of equal width.
// //' 2. Merging rare bins: Bins with a fraction of observations less than \code{bin_cutoff} are merged with adjacent bins.
// //' 3. Adjusting number of bins: If the number of bins exceeds \code{max_bins}, adjacent bins with the most similar WoE values are merged until \code{max_bins} is reached.
// //' 4. WoE and IV calculation: The Weight of Evidence (WoE) and Information Value (IV) are calculated for each bin.
// //' 
// //' The Weight of Evidence (WoE) for each bin is calculated as:
// //' 
// //' \deqn{WoE = \ln\left(\frac{P(X|Y=1)}{P(X|Y=0)}\right)}
// //' 
// //' where \eqn{P(X|Y=1)} is the probability of the feature being in a particular bin given a positive target, and \eqn{P(X|Y=0)} is the probability given a negative target.
// //' 
// //' The Information Value (IV) for each bin is calculated as:
// //' 
// //' \deqn{IV = (P(X|Y=1) - P(X|Y=0)) * WoE}
// //' 
// //' The total IV is the sum of IVs for all bins:
// //' 
// //' \deqn{Total IV = \sum_{i=1}^{n} IV_i}
// //' 
// //' This approach provides a balance between simplicity and effectiveness, creating bins of equal width initially and then adjusting them based on the data distribution and target variable relationship.
// //' 
// //' @examples
// //' \dontrun{
// //' set.seed(123)
// //' target <- sample(0:1, 1000, replace = TRUE)
// //' feature <- rnorm(1000)
// //' result <- optimal_binning_numerical_ewb(target, feature)
// //' print(result$woebin)
// //' }
// //' 
// //' @references
// //' \itemize{
// //'   \item Dougherty, J., Kohavi, R., & Sahami, M. (1995). Supervised and unsupervised discretization of continuous features. In Machine Learning Proceedings 1995 (pp. 194-202). Morgan Kaufmann.
// //'   \item Liu, H., Hussain, F., Tan, C. L., & Dash, M. (2002). Discretization: An enabling technique. Data mining and knowledge discovery, 6(4), 393-423.
// //' }
// //' 
// //' @author Lopes, J. E.
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_numerical_ewb(const std::vector<int>& target,
//                                          const std::vector<double>& feature,
//                                          int min_bins = 3, int max_bins = 5, double bin_cutoff = 0.05, int max_n_prebins = 20) {
//   OptimalBinningNumericalEWB binner(feature, target, min_bins, max_bins, bin_cutoff, max_n_prebins);
//   binner.fit();
//   Rcpp::List results = binner.get_results();
//   std::vector<double> woe_feature = binner.transform(feature);
//   results["woefeature"] = woe_feature;
//   return results;
// }

