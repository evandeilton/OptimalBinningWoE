// [[Rcpp::depends(Rcpp)]]
#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <cmath>
#include <limits>
#include <numeric>

using namespace Rcpp;

// Structure to hold bin information
struct Bin {
  double lower;
  double upper;
  int count;
  int count_pos;
  int count_neg;
  double woe;
  double iv;
  
  Bin(double l = 0.0, double u = 0.0) :
    lower(l), upper(u), count(0), count_pos(0), count_neg(0), woe(0.0), iv(0.0) {}
};

// Class for Optimal Binning using Standard Deviation-based Unsupervised Binning
class OptimalBinningNumericalUBSD {
private:
  std::vector<double> feature;
  std::vector<double> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  double convergence_threshold;
  int max_iterations;
  std::vector<Bin> bins;
  std::vector<double> cutpoints;
  bool converged;
  int iterations_run;
  
  // Validate inputs
  void validate_inputs() {
    if (feature.size() != target.size()) {
      Rcpp::stop("Feature and target must have the same length.");
    }
    if (min_bins < 2) {
      Rcpp::stop("min_bins must be at least 2.");
    }
    if (max_bins < min_bins) {
      Rcpp::stop("max_bins must be greater than or equal to min_bins.");
    }
    // Check for NAs and invalid values
    for(size_t i = 0; i < feature.size(); ++i) {
      if (std::isnan(feature[i]) || std::isinf(feature[i])) {
        Rcpp::stop("Feature contains NA or infinite values. Please handle them before binning.");
      }
      if (std::isnan(target[i]) || std::isinf(target[i])) {
        Rcpp::stop("Target contains NA or infinite values. Please handle them before binning.");
      }
      if (target[i] != 0 && target[i] != 1) {
        Rcpp::stop("Target must contain only 0 and 1 values.");
      }
    }
  }
  
  // Calculate mean of a vector
  double mean(const std::vector<double>& v) const {
    if(v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
  }
  
  // Calculate standard deviation of a vector
  double stddev(const std::vector<double>& v) const {
    if(v.size() < 2) return 0.0;
    double m = mean(v);
    double accum = 0.0;
    for(const double& d : v) {
      accum += (d - m) * (d - m);
    }
    return std::sqrt(accum / (v.size() - 1));
  }
  
  // Initial binning based on standard deviations
  void initial_binning() {
    double m = mean(feature);
    double sd = stddev(feature);
    
    std::vector<double> cut_points;
    cut_points.push_back(-std::numeric_limits<double>::infinity());
    
    if(sd == 0.0) {
      // All feature values are identical
      cut_points.push_back(std::numeric_limits<double>::infinity());
    }
    else {
      // Define bins based on mean and standard deviations
      for(int i = -2; i <= 2; ++i) {
        cut_points.push_back(m + i * sd);
      }
      cut_points.push_back(std::numeric_limits<double>::infinity());
      
      // Remove duplicates and sort
      std::sort(cut_points.begin(), cut_points.end());
      cut_points.erase(std::unique(cut_points.begin(), cut_points.end(),
                                   [](double a, double b) { return std::abs(a - b) < 1e-8; }),
                                                                                     cut_points.end());
      
      // Limit to max_n_prebins
      while(cut_points.size() - 1 > static_cast<size_t>(max_n_prebins)) {
        // Merge the two closest cut points
        double min_diff = std::numeric_limits<double>::infinity();
        size_t merge_index = 0;
        for(size_t i = 1; i < cut_points.size(); ++i) {
          double diff = cut_points[i] - cut_points[i-1];
          if(diff < min_diff) {
            min_diff = diff;
            merge_index = i;
          }
        }
        // Remove the cut point at merge_index
        cut_points.erase(cut_points.begin() + merge_index);
      }
    }
    
    // Initialize bins
    bins.clear();
    for(size_t i = 0; i < cut_points.size() - 1; ++i) {
      bins.emplace_back(cut_points[i], cut_points[i+1]);
    }
    
    // Ensure at least min_bins bins
    if(bins.size() < static_cast<size_t>(min_bins)) {
      double min_val = *std::min_element(feature.begin(), feature.end());
      double max_val = *std::max_element(feature.begin(), feature.end());
      double step = (max_val - min_val) / min_bins;
      bins.clear();
      bins.emplace_back(-std::numeric_limits<double>::infinity(), min_val + step);
      for(int i = 1; i < min_bins - 1; ++i) {
        bins.emplace_back(min_val + i * step, min_val + (i + 1) * step);
      }
      bins.emplace_back(min_val + (min_bins - 1) * step, std::numeric_limits<double>::infinity());
    }
  }
  
  // Assign data points to bins and calculate counts
  void assign_bins() {
    for(size_t i = 0; i < feature.size(); ++i) {
      double val = feature[i];
      double tar = target[i];
      // Find the bin where val belongs
      auto it = std::find_if(bins.begin(), bins.end(), [val](const Bin& bin) {
        return val > bin.lower && val <= bin.upper;
      });
      if(it == bins.end()) {
        // Assign to the last bin if not found (should not happen)
        it = std::prev(bins.end());
      }
      // Update bin counts
      it->count++;
      if(tar == 1) {
        it->count_pos++;
      } else {
        it->count_neg++;
      }
    }
  }
  
  // Merge bins with counts below bin_cutoff
  void merge_bins_by_cutoff() {
    double total_count = static_cast<double>(feature.size());
    
    bool merged = true;
    while(merged && bins.size() > static_cast<size_t>(min_bins)) {
      merged = false;
      for(size_t i = 0; i < bins.size(); ++i) {
        double bin_pct = static_cast<double>(bins[i].count) / total_count;
        if(bin_pct < bin_cutoff && bins.size() > static_cast<size_t>(min_bins)) {
          // Decide to merge with previous or next bin
          if(i == 0) {
            // Merge with next bin
            if(bins.size() < 2) break; // Prevent merging if only one bin
            bins[i+1].lower = bins[i].lower;
            bins[i+1].count += bins[i].count;
            bins[i+1].count_pos += bins[i].count_pos;
            bins[i+1].count_neg += bins[i].count_neg;
            bins.erase(bins.begin() + i);
          }
          else {
            // Merge with previous bin
            bins[i-1].upper = bins[i].upper;
            bins[i-1].count += bins[i].count;
            bins[i-1].count_pos += bins[i].count_pos;
            bins[i-1].count_neg += bins[i].count_neg;
            bins.erase(bins.begin() + i);
          }
          merged = true;
          break; // Restart after a merge
        }
      }
    }
  }
  
  // Calculate WOE and IV for each bin
  void calculate_woe_iv() {
    double total_pos = 0.0;
    double total_neg = 0.0;
    for(const double& tar : target) {
      if(tar == 1) {
        total_pos += 1.0;
      }
      else {
        total_neg += 1.0;
      }
    }
    
    if(total_pos == 0.0 || total_neg == 0.0) {
      Rcpp::stop("One of the target classes has zero instances. WoE and IV cannot be calculated.");
    }
    
    for(auto &bin : bins) {
      double dist_pos = static_cast<double>(bin.count_pos) / total_pos;
      double dist_neg = static_cast<double>(bin.count_neg) / total_neg;
      
      // Handle zero distributions by introducing a small constant
      if(dist_pos == 0.0) dist_pos = 1e-8;
      if(dist_neg == 0.0) dist_neg = 1e-8;
      
      bin.woe = std::log(dist_pos / dist_neg);
      bin.iv = (dist_pos - dist_neg) * bin.woe;
    }
  }
  
  // Enforce monotonicity of WOE
  void enforce_monotonicity() {
    if (bins.size() <= 2) {
      // Skip monotonicity enforcement if there are two or fewer bins
      return;
    }
    
    // Determine the direction of monotonicity
    std::vector<double> bin_midpoints;
    std::vector<double> woe_values;
    for (const auto& bin : bins) {
      double midpoint = (bin.lower + bin.upper) / 2.0;
      bin_midpoints.push_back(midpoint);
      woe_values.push_back(bin.woe);
    }
    
    double sum_x = std::accumulate(bin_midpoints.begin(), bin_midpoints.end(), 0.0);
    double sum_y = std::accumulate(woe_values.begin(), woe_values.end(), 0.0);
    double mean_x = sum_x / bin_midpoints.size();
    double mean_y = sum_y / woe_values.size();
    
    double numerator = 0.0;
    double denominator = 0.0;
    for (size_t i = 0; i < bin_midpoints.size(); ++i) {
      numerator += (bin_midpoints[i] - mean_x) * (woe_values[i] - mean_y);
      denominator += (bin_midpoints[i] - mean_x) * (bin_midpoints[i] - mean_x);
    }
    
    double slope = numerator / (denominator + 1e-8); // Add small constant to prevent division by zero
    bool increasing = (slope >= 0);
    
    // Enforce monotonicity
    bool merged = true;
    while (merged && bins.size() > static_cast<size_t>(min_bins)) {
      merged = false;
      for (size_t i = 1; i < bins.size(); ++i) {
        if ((increasing && bins[i].woe < bins[i - 1].woe) ||
            (!increasing && bins[i].woe > bins[i - 1].woe)) {
          // Merge bins[i - 1] and bins[i]
          bins[i - 1].upper = bins[i].upper;
          bins[i - 1].count += bins[i].count;
          bins[i - 1].count_pos += bins[i].count_pos;
          bins[i - 1].count_neg += bins[i].count_neg;
          bins.erase(bins.begin() + i);
          calculate_woe_iv();
          merged = true;
          break;
        }
      }
    }
  }
  
  // Further merge bins to ensure bin count does not exceed max_bins
  void merge_to_max_bins() {
    while(bins.size() > static_cast<size_t>(max_bins)) {
      // Find the pair of adjacent bins with the smallest total count
      int merge_index = -1;
      int min_total_count = std::numeric_limits<int>::max();
      
      for(size_t i = 0; i < bins.size()-1; ++i) {
        int combined_count = bins[i].count + bins[i+1].count;
        if(combined_count < min_total_count) {
          min_total_count = combined_count;
          merge_index = i;
        }
      }
      
      if(merge_index == -1) {
        break; // No more bins to merge
      }
      
      // Merge bins[merge_index] and bins[merge_index + 1]
      bins[merge_index].upper = bins[merge_index + 1].upper;
      bins[merge_index].count += bins[merge_index + 1].count;
      bins[merge_index].count_pos += bins[merge_index + 1].count_pos;
      bins[merge_index].count_neg += bins[merge_index + 1].count_neg;
      
      // Remove the merged bin
      bins.erase(bins.begin() + merge_index + 1);
      
      // Recalculate WOE and IV after merging
      calculate_woe_iv();
    }
  }
  
  // Update cutpoints based on current bins
  void update_cutpoints() {
    cutpoints.clear();
    for(size_t i = 1; i < bins.size(); ++i) {
      cutpoints.push_back(bins[i].lower);
    }
  }
  
public:
  // Constructor
  OptimalBinningNumericalUBSD(const std::vector<double>& feat, const std::vector<double>& targ,
                              int min_b, int max_b, double cutoff, int max_prebins,
                              double conv_threshold, int max_iter) :
  feature(feat), target(targ),
  min_bins(min_b), max_bins(max_b),
  bin_cutoff(cutoff), max_n_prebins(max_prebins),
  convergence_threshold(conv_threshold), max_iterations(max_iter),
  converged(false), iterations_run(0) {}
  
  // Fit the binning model
  void fit() {
    validate_inputs();
    
    // Check if the number of distinct feature values is less than or equal to 2
    std::vector<double> unique_values = feature;
    std::sort(unique_values.begin(), unique_values.end());
    unique_values.erase(std::unique(unique_values.begin(), unique_values.end()), unique_values.end());
    
    if (unique_values.size() <= 2) {
      // No optimization or extra bins, use min value as cutpoint
      double min_val = *std::min_element(feature.begin(), feature.end());
      bins.clear();
      bins.emplace_back(-std::numeric_limits<double>::infinity(), min_val);
      bins.emplace_back(min_val, std::numeric_limits<double>::infinity());
      assign_bins();
      calculate_woe_iv();
      update_cutpoints();
      converged = true;
      iterations_run = 1;
      return;
    }
    
    initial_binning();
    assign_bins();
    merge_bins_by_cutoff();
    calculate_woe_iv();
    
    double prev_total_iv = 0.0;
    for (int iter = 0; iter < max_iterations; ++iter) {
      enforce_monotonicity();
      merge_to_max_bins();
      calculate_woe_iv();
      
      double total_iv = 0.0;
      for (const auto& bin : bins) {
        total_iv += bin.iv;
      }
      
      if (std::abs(total_iv - prev_total_iv) < convergence_threshold) {
        converged = true;
        iterations_run = iter + 1;
        break;
      }
      
      prev_total_iv = total_iv;
    }
    
    if (!converged) {
      iterations_run = max_iterations;
    }
    
    update_cutpoints();
  }
  
  // Getters for output
  std::vector<std::string> get_bin_names() const {
    std::vector<std::string> bin_names;
    for (const auto& bin : bins) {
      std::ostringstream oss;
      oss << "(";
      if (std::isinf(bin.lower) && bin.lower < 0) {
        oss << "-Inf";
      } else {
        oss << bin.lower;
      }
      oss << ";";
      if (std::isinf(bin.upper)) {
        oss << "Inf";
      } else {
        oss << bin.upper;
      }
      oss << "]";
      bin_names.push_back(oss.str());
    }
    return bin_names;
  }
  
  std::vector<double> get_bin_woe() const {
    std::vector<double> woe_values;
    for (const auto& bin : bins) {
      woe_values.push_back(bin.woe);
    }
    return woe_values;
  }
  
  std::vector<double> get_bin_iv() const {
    std::vector<double> iv_values;
    for (const auto& bin : bins) {
      iv_values.push_back(bin.iv);
    }
    return iv_values;
  }
  
  std::vector<int> get_bin_count() const {
    std::vector<int> count_values;
    for (const auto& bin : bins) {
      count_values.push_back(bin.count);
    }
    return count_values;
  }
  
  std::vector<int> get_bin_count_pos() const {
    std::vector<int> count_pos_values;
    for (const auto& bin : bins) {
      count_pos_values.push_back(bin.count_pos);
    }
    return count_pos_values;
  }
  
  std::vector<int> get_bin_count_neg() const {
    std::vector<int> count_neg_values;
    for (const auto& bin : bins) {
      count_neg_values.push_back(bin.count_neg);
    }
    return count_neg_values;
  }
  
  std::vector<double> get_cutpoints() const {
    return cutpoints;
  }
  
  bool get_converged() const {
    return converged;
  }
  
  int get_iterations_run() const {
    return iterations_run;
  }
};

//' @title Optimal Binning for Numerical Variables using Unsupervised Binning with Standard Deviation
//' 
//' @description
//' This function implements an optimal binning algorithm for numerical variables using an 
//' Unsupervised Binning approach based on Standard Deviation (UBSD) with Weight of Evidence (WoE) 
//' and Information Value (IV) criteria.
//' 
//' @param target A numeric vector of binary target values (should contain exactly two unique values: 0 and 1).
//' @param feature A numeric vector of feature values to be binned.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum frequency of observations in each bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins for initial standard deviation-based discretization (default: 20).
//' @param convergence_threshold Threshold for convergence of the total IV (default: 1e-6).
//' @param max_iterations Maximum number of iterations for the algorithm (default: 1000).
//' 
//' @return A list containing the following elements:
//' \item{bins}{A character vector of bin names.}
//' \item{woe}{A numeric vector of Weight of Evidence values for each bin.}
//' \item{iv}{A numeric vector of Information Value for each bin.}
//' \item{count}{An integer vector of the total count of observations in each bin.}
//' \item{count_pos}{An integer vector of the count of positive observations in each bin.}
//' \item{count_neg}{An integer vector of the count of negative observations in each bin.}
//' \item{cutpoints}{A numeric vector of cut points used to generate the bins.}
//' \item{converged}{A logical value indicating whether the algorithm converged.}
//' \item{iterations}{An integer value indicating the number of iterations run.}
//' 
//' @details
//' The optimal binning algorithm for numerical variables uses an Unsupervised Binning approach 
//' based on Standard Deviation (UBSD) with Weight of Evidence (WoE) and Information Value (IV) 
//' to create bins that maximize the predictive power of the feature while maintaining interpretability.
//' 
//' The algorithm follows these steps:
//' 1. Initial binning based on standard deviations around the mean
//' 2. Assignment of data points to bins
//' 3. Merging of rare bins based on the bin_cutoff parameter
//' 4. Calculation of WoE and IV for each bin
//' 5. Enforcement of monotonicity in WoE across bins
//' 6. Further merging of bins to ensure the number of bins is within the specified range
//' 
//' The algorithm iterates until convergence is reached or the maximum number of iterations is hit.
//' 
//' @examples
//' \dontrun{
//' # Generate sample data
//' set.seed(123)
//' n <- 10000
//' feature <- rnorm(n)
//' target <- rbinom(n, 1, plogis(0.5 * feature))
//' 
//' # Apply optimal binning
//' result <- optimal_binning_numerical_ubsd(target, feature, min_bins = 3, max_bins = 5)
//' 
//' # View binning results
//' print(result)
//' }
//' 
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_ubsd(Rcpp::NumericVector target,
                                         Rcpp::NumericVector feature,
                                         int min_bins = 3,
                                         int max_bins = 5,
                                         double bin_cutoff = 0.05,
                                         int max_n_prebins = 20,
                                         double convergence_threshold = 1e-6,
                                         int max_iterations = 1000) {
 // Check for NA values
 if(Rcpp::any(Rcpp::is_na(target))) {
   Rcpp::stop("Target vector contains NA values. Please remove or impute them before binning.");
 }
 if(Rcpp::any(Rcpp::is_na(feature))) {
   Rcpp::stop("Feature vector contains NA values. Please remove or impute them before binning.");
 }
 
 // Convert R vectors to std::vector
 std::vector<double> std_feature = Rcpp::as<std::vector<double>>(feature);
 std::vector<double> std_target = Rcpp::as<std::vector<double>>(target);
 
 // Instantiate and fit the binning model
 OptimalBinningNumericalUBSD binning_model(std_feature, std_target,
                                           min_bins, max_bins,
                                           bin_cutoff, max_n_prebins,
                                           convergence_threshold, max_iterations);
 binning_model.fit();
 
 // Prepare the output list
 return Rcpp::List::create(
   Rcpp::Named("bins") = binning_model.get_bin_names(),
   Rcpp::Named("woe") = binning_model.get_bin_woe(),
   Rcpp::Named("iv") = binning_model.get_bin_iv(),
   Rcpp::Named("count") = binning_model.get_bin_count(),
   Rcpp::Named("count_pos") = binning_model.get_bin_count_pos(),
   Rcpp::Named("count_neg") = binning_model.get_bin_count_neg(),
   Rcpp::Named("cutpoints") = binning_model.get_cutpoints(),
   Rcpp::Named("converged") = binning_model.get_converged(),
   Rcpp::Named("iterations") = binning_model.get_iterations_run()
 );
}











// // [[Rcpp::depends(Rcpp)]]
// #include <Rcpp.h>
// #include <vector>
// #include <algorithm>
// #include <string>
// #include <sstream>
// #include <cmath>
// #include <limits>
// #include <numeric>
// 
// using namespace Rcpp;
// 
// // Structure to hold bin information
// struct Bin {
//   double lower;
//   double upper;
//   int count;
//   int count_pos;
//   int count_neg;
//   double woe;
//   double iv;
//   
//   Bin(double l = 0.0, double u = 0.0) :
//     lower(l), upper(u), count(0), count_pos(0), count_neg(0), woe(0.0), iv(0.0) {}
// };
// 
// // Class for Optimal Binning using Standard Deviation-based Unsupervised Binning
// class OptimalBinningNumericalUBSD {
// private:
//   std::vector<double> feature;
//   std::vector<double> target;
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
//   double convergence_threshold;
//   int max_iterations;
//   std::vector<Bin> bins;
//   std::vector<double> cutpoints;
//   bool converged;
//   int iterations_run;
//   
//   // Validate inputs
//   void validate_inputs() {
//     if (feature.size() != target.size()) {
//       Rcpp::stop("Feature and target must have the same length.");
//     }
//     if (min_bins < 2) {
//       Rcpp::stop("min_bins must be at least 2.");
//     }
//     if (max_bins < min_bins) {
//       Rcpp::stop("max_bins must be greater than or equal to min_bins.");
//     }
//     // Check for NAs and invalid values
//     for(size_t i = 0; i < feature.size(); ++i) {
//       if (std::isnan(feature[i]) || std::isinf(feature[i])) {
//         Rcpp::stop("Feature contains NA or infinite values. Please handle them before binning.");
//       }
//       if (std::isnan(target[i]) || std::isinf(target[i])) {
//         Rcpp::stop("Target contains NA or infinite values. Please handle them before binning.");
//       }
//       if (target[i] != 0 && target[i] != 1) {
//         Rcpp::stop("Target must contain only 0 and 1 values.");
//       }
//     }
//   }
//   
//   // Calculate mean of a vector
//   double mean(const std::vector<double>& v) const {
//     if(v.empty()) return 0.0;
//     return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
//   }
//   
//   // Calculate standard deviation of a vector
//   double stddev(const std::vector<double>& v) const {
//     if(v.size() < 2) return 0.0;
//     double m = mean(v);
//     double accum = 0.0;
//     for(const double& d : v) {
//       accum += (d - m) * (d - m);
//     }
//     return std::sqrt(accum / (v.size() - 1));
//   }
//   
//   // Initial binning based on standard deviations
//   void initial_binning() {
//     double m = mean(feature);
//     double sd = stddev(feature);
//     
//     std::vector<double> cut_points;
//     cut_points.push_back(-std::numeric_limits<double>::infinity());
//     
//     if(sd == 0.0) {
//       // All feature values are identical
//       cut_points.push_back(std::numeric_limits<double>::infinity());
//     }
//     else {
//       // Define bins based on mean and standard deviations
//       for(int i = -2; i <= 2; ++i) {
//         cut_points.push_back(m + i * sd);
//       }
//       cut_points.push_back(std::numeric_limits<double>::infinity());
//       
//       // Remove duplicates and sort
//       std::sort(cut_points.begin(), cut_points.end());
//       cut_points.erase(std::unique(cut_points.begin(), cut_points.end(),
//                                    [](double a, double b) { return std::abs(a - b) < 1e-8; }),
//                                                                                      cut_points.end());
//       
//       // Limit to max_n_prebins
//       while(cut_points.size() - 1 > static_cast<size_t>(max_n_prebins)) {
//         // Merge the two closest cut points
//         double min_diff = std::numeric_limits<double>::infinity();
//         size_t merge_index = 0;
//         for(size_t i = 1; i < cut_points.size(); ++i) {
//           double diff = cut_points[i] - cut_points[i-1];
//           if(diff < min_diff) {
//             min_diff = diff;
//             merge_index = i;
//           }
//         }
//         // Remove the cut point at merge_index
//         cut_points.erase(cut_points.begin() + merge_index);
//       }
//     }
//     
//     // Initialize bins
//     bins.clear();
//     for(size_t i = 0; i < cut_points.size() - 1; ++i) {
//       bins.emplace_back(cut_points[i], cut_points[i+1]);
//     }
//     
//     // Ensure at least min_bins bins
//     if(bins.size() < static_cast<size_t>(min_bins)) {
//       double min_val = *std::min_element(feature.begin(), feature.end());
//       double max_val = *std::max_element(feature.begin(), feature.end());
//       double step = (max_val - min_val) / min_bins;
//       bins.clear();
//       bins.emplace_back(-std::numeric_limits<double>::infinity(), min_val + step);
//       for(int i = 1; i < min_bins - 1; ++i) {
//         bins.emplace_back(min_val + i * step, min_val + (i + 1) * step);
//       }
//       bins.emplace_back(min_val + (min_bins - 1) * step, std::numeric_limits<double>::infinity());
//     }
//   }
//   
//   // Assign data points to bins and calculate counts
//   void assign_bins() {
//     for(size_t i = 0; i < feature.size(); ++i) {
//       double val = feature[i];
//       double tar = target[i];
//       // Find the bin where val belongs
//       auto it = std::find_if(bins.begin(), bins.end(), [val](const Bin& bin) {
//         return val > bin.lower && val <= bin.upper;
//       });
//       if(it == bins.end()) {
//         // Assign to the last bin if not found (should not happen)
//         it = std::prev(bins.end());
//       }
//       // Update bin counts
//       it->count++;
//       if(tar == 1) {
//         it->count_pos++;
//       } else {
//         it->count_neg++;
//       }
//     }
//   }
//   
//   // Merge bins with counts below bin_cutoff
//   void merge_bins_by_cutoff() {
//     double total_count = static_cast<double>(feature.size());
//     
//     bool merged = true;
//     while(merged && bins.size() > static_cast<size_t>(min_bins)) {
//       merged = false;
//       for(size_t i = 0; i < bins.size(); ++i) {
//         double bin_pct = static_cast<double>(bins[i].count) / total_count;
//         if(bin_pct < bin_cutoff && bins.size() > static_cast<size_t>(min_bins)) {
//           // Decide to merge with previous or next bin
//           if(i == 0) {
//             // Merge with next bin
//             if(bins.size() < 2) break; // Prevent merging if only one bin
//             bins[i+1].lower = bins[i].lower;
//             bins[i+1].count += bins[i].count;
//             bins[i+1].count_pos += bins[i].count_pos;
//             bins[i+1].count_neg += bins[i].count_neg;
//             bins.erase(bins.begin() + i);
//           }
//           else {
//             // Merge with previous bin
//             bins[i-1].upper = bins[i].upper;
//             bins[i-1].count += bins[i].count;
//             bins[i-1].count_pos += bins[i].count_pos;
//             bins[i-1].count_neg += bins[i].count_neg;
//             bins.erase(bins.begin() + i);
//           }
//           merged = true;
//           break; // Restart after a merge
//         }
//       }
//     }
//   }
//   
//   // Calculate WOE and IV for each bin
//   void calculate_woe_iv() {
//     double total_pos = 0.0;
//     double total_neg = 0.0;
//     for(const double& tar : target) {
//       if(tar == 1) {
//         total_pos += 1.0;
//       }
//       else {
//         total_neg += 1.0;
//       }
//     }
//     
//     if(total_pos == 0.0 || total_neg == 0.0) {
//       Rcpp::stop("One of the target classes has zero instances. WoE and IV cannot be calculated.");
//     }
//     
//     for(auto &bin : bins) {
//       double dist_pos = static_cast<double>(bin.count_pos) / total_pos;
//       double dist_neg = static_cast<double>(bin.count_neg) / total_neg;
//       
//       // Handle zero distributions by introducing a small constant
//       if(dist_pos == 0.0) dist_pos = 1e-8;
//       if(dist_neg == 0.0) dist_neg = 1e-8;
//       
//       bin.woe = std::log(dist_pos / dist_neg);
//       bin.iv = (dist_pos - dist_neg) * bin.woe;
//     }
//   }
//   
//   // Enforce monotonicity of WOE
//   void enforce_monotonicity() {
//     if (bins.size() <= 2) {
//       // Skip monotonicity enforcement if there are two or fewer bins
//       return;
//     }
//     
//     // Determine the direction of monotonicity
//     std::vector<double> bin_midpoints;
//     std::vector<double> woe_values;
//     for (const auto& bin : bins) {
//       double midpoint = (bin.lower + bin.upper) / 2.0;
//       bin_midpoints.push_back(midpoint);
//       woe_values.push_back(bin.woe);
//     }
//     
//     double sum_x = std::accumulate(bin_midpoints.begin(), bin_midpoints.end(), 0.0);
//     double sum_y = std::accumulate(woe_values.begin(), woe_values.end(), 0.0);
//     double mean_x = sum_x / bin_midpoints.size();
//     double mean_y = sum_y / woe_values.size();
//     
//     double numerator = 0.0;
//     double denominator = 0.0;
//     for (size_t i = 0; i < bin_midpoints.size(); ++i) {
//       numerator += (bin_midpoints[i] - mean_x) * (woe_values[i] - mean_y);
//       denominator += (bin_midpoints[i] - mean_x) * (bin_midpoints[i] - mean_x);
//     }
//     
//     double slope = numerator / (denominator + 1e-8); // Add small constant to prevent division by zero
//     bool increasing = (slope >= 0);
//     
//     // Enforce monotonicity
//     bool merged = true;
//     while (merged && bins.size() > static_cast<size_t>(min_bins)) {
//       merged = false;
//       for (size_t i = 1; i < bins.size(); ++i) {
//         if ((increasing && bins[i].woe < bins[i - 1].woe) ||
//             (!increasing && bins[i].woe > bins[i - 1].woe)) {
//           // Merge bins[i - 1] and bins[i]
//           bins[i - 1].upper = bins[i].upper;
//           bins[i - 1].count += bins[i].count;
//           bins[i - 1].count_pos += bins[i].count_pos;
//           bins[i - 1].count_neg += bins[i].count_neg;
//           bins.erase(bins.begin() + i);
//           calculate_woe_iv();
//           merged = true;
//           break;
//         }
//       }
//     }
//   }
//   
//   // Further merge bins to ensure bin count does not exceed max_bins
//   void merge_to_max_bins() {
//     while(bins.size() > static_cast<size_t>(max_bins)) {
//       // Find the pair of adjacent bins with the smallest total count
//       int merge_index = -1;
//       int min_total_count = std::numeric_limits<int>::max();
//       
//       for(size_t i = 0; i < bins.size()-1; ++i) {
//         int combined_count = bins[i].count + bins[i+1].count;
//         if(combined_count < min_total_count) {
//           min_total_count = combined_count;
//           merge_index = i;
//         }
//       }
//       
//       if(merge_index == -1) {
//         break; // No more bins to merge
//       }
//       
//       // Merge bins[merge_index] and bins[merge_index + 1]
//       bins[merge_index].upper = bins[merge_index + 1].upper;
//       bins[merge_index].count += bins[merge_index + 1].count;
//       bins[merge_index].count_pos += bins[merge_index + 1].count_pos;
//       bins[merge_index].count_neg += bins[merge_index + 1].count_neg;
//       
//       // Remove the merged bin
//       bins.erase(bins.begin() + merge_index + 1);
//       
//       // Recalculate WOE and IV after merging
//       calculate_woe_iv();
//     }
//   }
//   
//   // Update cutpoints based on current bins
//   void update_cutpoints() {
//     cutpoints.clear();
//     for(size_t i = 1; i < bins.size(); ++i) {
//       cutpoints.push_back(bins[i].lower);
//     }
//   }
//   
// public:
//   // Constructor
//   OptimalBinningNumericalUBSD(const std::vector<double>& feat, const std::vector<double>& targ,
//                               int min_b, int max_b, double cutoff, int max_prebins,
//                               double conv_threshold, int max_iter) :
//   feature(feat), target(targ),
//   min_bins(min_b), max_bins(max_b),
//   bin_cutoff(cutoff), max_n_prebins(max_prebins),
//   convergence_threshold(conv_threshold), max_iterations(max_iter),
//   converged(false), iterations_run(0) {}
//   
//   // Fit the binning model
//   void fit() {
//     validate_inputs();
//     
//     // Check if the number of distinct feature values is less than or equal to min_bins
//     std::vector<double> unique_values = feature;
//     std::sort(unique_values.begin(), unique_values.end());
//     unique_values.erase(std::unique(unique_values.begin(), unique_values.end()), unique_values.end());
//     
//     if (unique_values.size() <= static_cast<size_t>(min_bins)) {
//       // No need to optimize, use unique values as bin boundaries
//       bins.clear();
//       for (size_t i = 0; i < unique_values.size(); ++i) {
//         double lower = (i == 0) ? -std::numeric_limits<double>::infinity() : unique_values[i-1];
//         double upper = (i == unique_values.size() - 1) ? std::numeric_limits<double>::infinity() : unique_values[i];
//         bins.emplace_back(lower, upper);
//       }
//       assign_bins();
//       calculate_woe_iv();
//       update_cutpoints();
//       converged = true;
//       iterations_run = 1;
//       return;
//     }
//     
//     initial_binning();
//     assign_bins();
//     merge_bins_by_cutoff();
//     calculate_woe_iv();
//     
//     double prev_total_iv = 0.0;
//     for (int iter = 0; iter < max_iterations; ++iter) {
//       enforce_monotonicity();
//       merge_to_max_bins();
//       calculate_woe_iv();
//       
//       double total_iv = 0.0;
//       for (const auto& bin : bins) {
//         total_iv += bin.iv;
//       }
//       
//       if (std::abs(total_iv - prev_total_iv) < convergence_threshold) {
//         converged = true;
//         iterations_run = iter + 1;
//         break;
//       }
//       
//       prev_total_iv = total_iv;
//     }
//     
//     if (!converged) {
//       iterations_run = max_iterations;
//     }
//     
//     update_cutpoints();
//   }
//   
//   // Getters for output
//   std::vector<std::string> get_bin_names() const {
//     std::vector<std::string> bin_names;
//     for (const auto& bin : bins) {
//       std::ostringstream oss;
//       oss << "(";
//       if (std::isinf(bin.lower) && bin.lower < 0) {
//         oss << "-Inf";
//       } else {
//         oss << bin.lower;
//       }
//       oss << ";";
//       if (std::isinf(bin.upper)) {
//         oss << "Inf";
//       } else {
//         oss << bin.upper;
//       }
//       oss << "]";
//       bin_names.push_back(oss.str());
//     }
//     return bin_names;
//   }
//   
//   std::vector<double> get_bin_woe() const {
//     std::vector<double> woe_values;
//     for (const auto& bin : bins) {
//       woe_values.push_back(bin.woe);
//     }
//     return woe_values;
//   }
//   
//   std::vector<double> get_bin_iv() const {
//     std::vector<double> iv_values;
//     for (const auto& bin : bins) {
//       iv_values.push_back(bin.iv);
//     }
//     return iv_values;
//   }
//   
//   std::vector<int> get_bin_count() const {
//     std::vector<int> count_values;
//     for (const auto& bin : bins) {
//       count_values.push_back(bin.count);
//     }
//     return count_values;
//   }
//   
//   std::vector<int> get_bin_count_pos() const {
//     std::vector<int> count_pos_values;
//     for (const auto& bin : bins) {
//       count_pos_values.push_back(bin.count_pos);
//     }
//     return count_pos_values;
//   }
//   
//   std::vector<int> get_bin_count_neg() const {
//     std::vector<int> count_neg_values;
//     for (const auto& bin : bins) {
//       count_neg_values.push_back(bin.count_neg);
//     }
//     return count_neg_values;
//   }
//   
//   std::vector<double> get_cutpoints() const {
//     return cutpoints;
//   }
//   
//   bool get_converged() const {
//     return converged;
//   }
//   
//   int get_iterations_run() const {
//     return iterations_run;
//   }
// };
// 
// //' @title Optimal Binning for Numerical Variables using Unsupervised Binning with Standard Deviation
// //' 
// //' @description
// //' This function implements an optimal binning algorithm for numerical variables using an 
// //' Unsupervised Binning approach based on Standard Deviation (UBSD) with Weight of Evidence (WoE) 
// //' and Information Value (IV) criteria.
// //' 
// //' @param target A numeric vector of binary target values (should contain exactly two unique values: 0 and 1).
// //' @param feature A numeric vector of feature values to be binned.
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff Minimum frequency of observations in each bin (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins for initial standard deviation-based discretization (default: 20).
// //' @param convergence_threshold Threshold for convergence of the total IV (default: 1e-6).
// //' @param max_iterations Maximum number of iterations for the algorithm (default: 1000).
// //' 
// //' @return A list containing the following elements:
// //' \item{bins}{A character vector of bin names.}
// //' \item{woe}{A numeric vector of Weight of Evidence values for each bin.}
// //' \item{iv}{A numeric vector of Information Value for each bin.}
// //' \item{count}{An integer vector of the total count of observations in each bin.}
// //' \item{count_pos}{An integer vector of the count of positive observations in each bin.}
// //' \item{count_neg}{An integer vector of the count of negative observations in each bin.}
// //' \item{cutpoints}{A numeric vector of cut points used to generate the bins.}
// //' \item{converged}{A logical value indicating whether the algorithm converged.}
// //' \item{iterations}{An integer value indicating the number of iterations run.}
// //' 
// //' @details
// //' The optimal binning algorithm for numerical variables uses an Unsupervised Binning approach 
// //' based on Standard Deviation (UBSD) with Weight of Evidence (WoE) and Information Value (IV) 
// //' to create bins that maximize the predictive power of the feature while maintaining interpretability.
// //' 
// //' The algorithm follows these steps:
// //' 1. Initial binning based on standard deviations around the mean
// //' 2. Assignment of data points to bins
// //' 3. Merging of rare bins based on the bin_cutoff parameter
// //' 4. Calculation of WoE and IV for each bin
// //' 5. Enforcement of monotonicity in WoE across bins
// //' 6. Further merging of bins to ensure the number of bins is within the specified range
// //' 
// //' The algorithm iterates until convergence is reached or the maximum number of iterations is hit.
// //' 
// //' @examples
// //' \dontrun{
// //' # Generate sample data
// //' set.seed(123)
// //' n <- 10000
// //' feature <- rnorm(n)
// //' target <- rbinom(n, 1, plogis(0.5 * feature))
// //' 
// //' # Apply optimal binning
// //' result <- optimal_binning_numerical_ubsd(target, feature, min_bins = 3, max_bins = 5)
// //' 
// //' # View binning results
// //' print(result)
// //' }
// //' 
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_numerical_ubsd(Rcpp::NumericVector target,
//                                          Rcpp::NumericVector feature,
//                                          int min_bins = 3,
//                                          int max_bins = 5,
//                                          double bin_cutoff = 0.05,
//                                          int max_n_prebins = 20,
//                                          double convergence_threshold = 1e-6,
//                                          int max_iterations = 1000) {
//  // Check for NA values
//  if(Rcpp::any(Rcpp::is_na(target))) {
//    Rcpp::stop("Target vector contains NA values. Please remove or impute them before binning.");
//  }
//  if(Rcpp::any(Rcpp::is_na(feature))) {
//    Rcpp::stop("Feature vector contains NA values. Please remove or impute them before binning.");
//  }
//  
//  // Convert R vectors to std::vector
//  std::vector<double> std_feature = Rcpp::as<std::vector<double>>(feature);
//  std::vector<double> std_target = Rcpp::as<std::vector<double>>(target);
//  
//  // Instantiate and fit the binning model
//  OptimalBinningNumericalUBSD binning_model(std_feature, std_target,
//                                            min_bins, max_bins,
//                                            bin_cutoff, max_n_prebins,
//                                            convergence_threshold, max_iterations);
//  binning_model.fit();
//  
//  // Prepare the output list
//  return Rcpp::List::create(
//    Rcpp::Named("bin") = binning_model.get_bin_names(),
//    Rcpp::Named("woe") = binning_model.get_bin_woe(),
//    Rcpp::Named("iv") = binning_model.get_bin_iv(),
//    Rcpp::Named("count") = binning_model.get_bin_count(),
//    Rcpp::Named("count_pos") = binning_model.get_bin_count_pos(),
//    Rcpp::Named("count_neg") = binning_model.get_bin_count_neg(),
//    Rcpp::Named("cutpoints") = binning_model.get_cutpoints(),
//    Rcpp::Named("converged") = binning_model.get_converged(),
//    Rcpp::Named("iterations") = binning_model.get_iterations_run()
//  );
// }
