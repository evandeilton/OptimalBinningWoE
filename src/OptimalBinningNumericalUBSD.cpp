// [[Rcpp::plugins(openmp)]]
#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <cmath>
#include <limits>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

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
  std::vector<double> woefeature;
  Rcpp::DataFrame woebin;
  
  std::vector<Bin> bins;
  
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
    // Check for NAs
    for(size_t i = 0; i < feature.size(); ++i) {
      if (std::isnan(feature[i])) {
        Rcpp::stop("Feature contains NA values. Please handle them before binning.");
      }
      if (std::isnan(target[i])) {
        Rcpp::stop("Target contains NA values. Please handle them before binning.");
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
    int n = feature.size();
    int n_bins = bins.size();
    
    // Create vectors of bin lower and upper bounds
    std::vector<double> bin_lowers(n_bins);
    std::vector<double> bin_uppers(n_bins);
    for(int i = 0; i < n_bins; ++i) {
      bin_lowers[i] = bins[i].lower;
      bin_uppers[i] = bins[i].upper;
    }
    
    // Parallel assignment of data points to bins
#pragma omp parallel for schedule(static)
    for(int i = 0; i < n; ++i) {
      double val = feature[i];
      double tar = target[i];
      // Find the bin index where val belongs
      int bin_idx = -1;
      for(int j = 0; j < n_bins; ++j) {
        if(val > bin_lowers[j] && val <= bin_uppers[j]) {
          bin_idx = j;
          break;
        }
      }
      if(bin_idx == -1) {
        // Assign to the last bin if not found (should not happen)
        bin_idx = n_bins - 1;
      }
      // Update bin counts atomically
#pragma omp atomic
      bins[bin_idx].count++;
      if(tar == 1) {
#pragma omp atomic
        bins[bin_idx].count_pos++;
      } else {
#pragma omp atomic
        bins[bin_idx].count_neg++;
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
  
  // Apply WOE to the feature using correct bin assignment
  void apply_woe_to_feature() {
    woefeature.resize(feature.size());
    std::fill(woefeature.begin(), woefeature.end(), 0.0);
    
    int n = feature.size();
    int n_bins = bins.size();
    
    // Create vectors of bin lower and upper bounds
    std::vector<double> bin_lowers(n_bins);
    std::vector<double> bin_uppers(n_bins);
    for(int i = 0; i < n_bins; ++i) {
      bin_lowers[i] = bins[i].lower;
      bin_uppers[i] = bins[i].upper;
    }
    
    // Parallel assignment of WOE values
#pragma omp parallel for schedule(static)
    for(int i = 0; i < n; ++i) {
      double val = feature[i];
      // Find the bin index where val belongs
      int bin_idx = -1;
      for(int j = 0; j < n_bins; ++j) {
        if(val > bin_lowers[j] && val <= bin_uppers[j]) {
          bin_idx = j;
          break;
        }
      }
      if(bin_idx == -1) {
        // Assign to the last bin if not found (should not happen)
        bin_idx = n_bins - 1;
      }
      woefeature[i] = bins[bin_idx].woe;
    }
  }
  
  // Prepare the output DataFrame for WoE bins
  void prepare_output() {
    size_t n_bins = bins.size();
    Rcpp::CharacterVector bin_strings(n_bins);
    Rcpp::NumericVector woe_values(n_bins);
    Rcpp::NumericVector iv_values(n_bins);
    Rcpp::IntegerVector count_values(n_bins);
    Rcpp::IntegerVector count_pos_values(n_bins);
    Rcpp::IntegerVector count_neg_values(n_bins);
    
    for(size_t i = 0; i < n_bins; ++i) {
      std::ostringstream oss;
      oss << "(";
      if(std::isinf(bins[i].lower) && bins[i].lower < 0) {
        oss << "-Inf";
      } else {
        oss << bins[i].lower;
      }
      oss << ";";
      if(std::isinf(bins[i].upper)) {
        oss << "+Inf";
      } else {
        oss << bins[i].upper;
      }
      oss << "]";
      bin_strings[i] = oss.str();
      
      woe_values[i] = bins[i].woe;
      iv_values[i] = bins[i].iv;
      count_values[i] = bins[i].count;
      count_pos_values[i] = bins[i].count_pos;
      count_neg_values[i] = bins[i].count_neg;
    }
    
    woebin = Rcpp::DataFrame::create(
      Rcpp::Named("bin") = bin_strings,
      Rcpp::Named("woe") = woe_values,
      Rcpp::Named("iv") = iv_values,
      Rcpp::Named("count") = count_values,
      Rcpp::Named("count_pos") = count_pos_values,
      Rcpp::Named("count_neg") = count_neg_values,
      Rcpp::Named("stringsAsFactors") = false
    );
  }
  
public:
  // Constructor
  OptimalBinningNumericalUBSD(const std::vector<double>& feat, const std::vector<double>& targ,
                              int min_b, int max_b, double cutoff, int max_prebins) :
  feature(feat), target(targ),
  min_bins(min_b), max_bins(max_b),
  bin_cutoff(cutoff), max_n_prebins(max_prebins) {}
  
  // Fit the binning model
  void fit() {
    validate_inputs();
    initial_binning();
    assign_bins();
    merge_bins_by_cutoff();
    calculate_woe_iv();
    enforce_monotonicity();
    
    // After enforcing monotonicity, ensure bin count does not exceed max_bins
    if(static_cast<int>(bins.size()) > max_bins) {
      merge_to_max_bins();
    }
    
    apply_woe_to_feature();
    prepare_output();
  }
  
  // Getters for output
  std::vector<double> get_woefeature() const {
    return woefeature;
  }
  
  Rcpp::DataFrame get_woebin() const {
    return woebin;
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
//' 
//' @return A list containing two elements:
//' \item{woefeature}{A numeric vector of WoE-transformed feature values.}
//' \item{woebin}{A data frame with binning details, including bin boundaries, WoE, IV, and count statistics.}
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
//' 7. Application of WoE transformation to the original feature
//' 
//' Weight of Evidence (WoE) is calculated for each bin as:
//' 
//' \deqn{WoE_i = \ln\left(\frac{P(X_i|Y=1)}{P(X_i|Y=0)}\right)}
//' 
//' where \eqn{P(X_i|Y=1)} is the proportion of positive cases in bin i, and 
//' \eqn{P(X_i|Y=0)} is the proportion of negative cases in bin i.
//' 
//' Information Value (IV) for each bin is calculated as:
//' 
//' \deqn{IV_i = (P(X_i|Y=1) - P(X_i|Y=0)) * WoE_i}
//' 
//' The total IV for the feature is the sum of IVs across all bins:
//' 
//' \deqn{IV_{total} = \sum_{i=1}^{n} IV_i}
//' 
//' The UBSD approach ensures that the resulting binning maximizes the separation between 
//' classes while maintaining the desired number of bins and respecting the minimum bin 
//' frequency constraint.
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
//' print(result$woebin)
//' 
//' # Plot WoE transformation
//' plot(feature, result$woefeature, main = "WoE Transformation", 
//'      xlab = "Original Feature", ylab = "WoE")
//' }
//' 
//' @references
//' \itemize{
//' \item Kotsiantis, S., & Kanellopoulos, D. (2006). Discretization techniques: A recent survey. 
//'       GESTS International Transactions on Computer Science and Engineering, 32(1), 47-58.
//' \item Dougherty, J., Kohavi, R., & Sahami, M. (1995). Supervised and unsupervised 
//'       discretization of continuous features. In Machine Learning Proceedings 1995 
//'       (pp. 194-202). Morgan Kaufmann.
//' }
//' 
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_ubsd(Rcpp::NumericVector target,
                                         Rcpp::NumericVector feature,
                                         int min_bins = 3,
                                         int max_bins = 5,
                                         double bin_cutoff = 0.05,
                                         int max_n_prebins = 20) {
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
                                           bin_cutoff, max_n_prebins);
 binning_model.fit();
 
 // Prepare the output list
 return Rcpp::List::create(
   Rcpp::Named("woefeature") = binning_model.get_woefeature(),
   Rcpp::Named("woebin") = binning_model.get_woebin()
 );
}


// // [[Rcpp::plugins(openmp)]]
// #include <Rcpp.h>
// #include <vector>
// #include <algorithm>
// #include <string>
// #include <sstream>
// #include <cmath>
// #include <limits>
// #include <numeric>
// 
// #ifdef _OPENMP
// #include <omp.h>
// #endif
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
//   std::vector<double> woefeature;
//   Rcpp::DataFrame woebin;
//   
//   std::vector<Bin> bins;
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
//   }
//   
//   // Calculate mean of a vector
//   double mean(const std::vector<double>& v) {
//     return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
//   }
//   
//   // Calculate standard deviation of a vector
//   double stddev(const std::vector<double>& v) {
//     double m = mean(v);
//     double accum = 0.0;
//     std::for_each(v.begin(), v.end(), [&](const double d) {
//       accum += (d - m) * (d - m);
//     });
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
//     // Define bins based on mean and standard deviations
//     for(int i = -2; i <= 2; ++i) {
//       cut_points.push_back(m + i * sd);
//     }
//     cut_points.push_back(std::numeric_limits<double>::infinity());
//     
//     // Remove duplicates and sort
//     std::sort(cut_points.begin(), cut_points.end());
//     cut_points.erase(std::unique(cut_points.begin(), cut_points.end(),
//                                  [](double a, double b) { return std::abs(a - b) < 1e-8; }),
//                                                                                    cut_points.end());
//     
//     // Limit to max_n_prebins
//     while(cut_points.size() - 1 > static_cast<size_t>(max_n_prebins)) {
//       // Merge the two closest cut points
//       double min_diff = std::numeric_limits<double>::infinity();
//       size_t merge_index = 0;
//       for(size_t i = 1; i < cut_points.size(); ++i) {
//         double diff = cut_points[i] - cut_points[i-1];
//         if(diff < min_diff) {
//           min_diff = diff;
//           merge_index = i;
//         }
//       }
//       // Remove the cut point at merge_index
//       cut_points.erase(cut_points.begin() + merge_index);
//     }
//     
//     // Initialize bins
//     bins.clear();
//     for(size_t i = 0; i < cut_points.size() - 1; ++i) {
//       bins.emplace_back(cut_points[i], cut_points[i+1]);
//     }
//   }
//   
//   // Assign data points to bins and calculate counts
//   void assign_bins() {
//     int n = feature.size();
//     int n_bins = bins.size();
//     
//     // Parallel assignment of data points to bins
// #pragma omp parallel for schedule(static)
//     for(int i = 0; i < n; ++i) {
//       double val = feature[i];
//       double tar = target[i];
//       // Binary search to find the bin
//       int left = 0;
//       int right = n_bins - 1;
//       int bin_idx = n_bins - 1; // Default to last bin
//       while(left <= right) {
//         int mid = left + (right - left) / 2;
//         if(val > bins[mid].upper) {
//           left = mid + 1;
//         }
//         else if(val <= bins[mid].lower) {
//           right = mid - 1;
//         }
//         else {
//           bin_idx = mid;
//           break;
//         }
//       }
//       // Update bin counts atomically
// #pragma omp atomic
//       bins[bin_idx].count++;
//       if(tar == 1) {
// #pragma omp atomic
//         bins[bin_idx].count_pos++;
//       } else {
// #pragma omp atomic
//         bins[bin_idx].count_neg++;
//       }
//     }
//   }
//   
//   // Merge bins with counts below bin_cutoff
//   void merge_bins_by_cutoff() {
//     double total_count = feature.size();
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
//     double total_pos = std::accumulate(target.begin(), target.end(), 0.0);
//     double total_neg = target.size() - total_pos;
//     
//     for(auto &bin : bins) {
//       double dist_pos = static_cast<double>(bin.count_pos) / total_pos;
//       double dist_neg = static_cast<double>(bin.count_neg) / total_neg;
//       
//       // Handle zero distributions
//       if(dist_pos == 0) dist_pos = 0.0001;
//       if(dist_neg == 0) dist_neg = 0.0001;
//       
//       bin.woe = std::log(dist_pos / dist_neg);
//       bin.iv = (dist_pos - dist_neg) * bin.woe;
//     }
//   }
//   
//   // Enforce monotonicity of WOE
//   void enforce_monotonicity() {
//     bool monotonic = false;
//     while(!monotonic) {
//       monotonic = true;
//       // Determine the direction of monotonicity
//       bool increasing = true;
//       bool decreasing = true;
//       for(size_t i = 1; i < bins.size(); ++i) {
//         if(bins[i].woe < bins[i-1].woe) {
//           increasing = false;
//         }
//         if(bins[i].woe > bins[i-1].woe) {
//           decreasing = false;
//         }
//       }
//       
//       if(increasing || decreasing) {
//         // Monotonicity is satisfied
//         return;
//       }
//       
//       // Find the first pair that violates monotonicity and merge them
//       for(size_t i = 1; i < bins.size(); ++i) {
//         if(bins[i].woe < bins[i-1].woe && bins[i].woe < bins[i+1].woe && i < bins.size()-1) {
//           // Merge bins[i-1] and bins[i]
//           bins[i-1].upper = bins[i].upper;
//           bins[i-1].count += bins[i].count;
//           bins[i-1].count_pos += bins[i].count_pos;
//           bins[i-1].count_neg += bins[i].count_neg;
//           bins.erase(bins.begin() + i);
//           monotonic = false;
//           break;
//         }
//       }
//       
//       calculate_woe_iv();
//       
//       // If still not monotonic after merging, continue the loop
//     }
//   }
//   
//   // Further merge bins to ensure bin count does not exceed max_bins
//   void merge_to_max_bins() {
//     while(bins.size() > static_cast<size_t>(max_bins)) {
//       // Find the pair of adjacent bins with the smallest IV
//       double min_iv = std::numeric_limits<double>::infinity();
//       size_t merge_index = 0;
//       
//       for(size_t i = 0; i < bins.size()-1; ++i) {
//         double combined_iv = bins[i].iv + bins[i+1].iv;
//         if(combined_iv < min_iv) {
//           min_iv = combined_iv;
//           merge_index = i;
//         }
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
//   // Apply WOE to the feature
//   void apply_woe_to_feature() {
//     woefeature.resize(feature.size());
//     std::fill(woefeature.begin(), woefeature.end(), 0.0);
//     
//     int n = feature.size();
//     int n_bins = bins.size();
//     
//     // Parallel assignment of WOE values
// #pragma omp parallel for schedule(static)
//     for(int i = 0; i < n; ++i) {
//       double val = feature[i];
//       for(int j = 0; j < n_bins; ++j) {
//         if((val > bins[j].lower && val <= bins[j].upper) ||
//            (j == 0 && val == bins[j].lower)) { // Include lower bound for first bin
//           woefeature[i] = bins[j].woe;
//           break;
//         }
//       }
//     }
//   }
//   
//   // Prepare the output DataFrame for WoE bins
//   void prepare_output() {
//     size_t n_bins = bins.size();
//     Rcpp::CharacterVector bin_strings(n_bins);
//     Rcpp::NumericVector woe_values(n_bins);
//     Rcpp::NumericVector iv_values(n_bins);
//     Rcpp::IntegerVector count_values(n_bins);
//     Rcpp::IntegerVector count_pos_values(n_bins);
//     Rcpp::IntegerVector count_neg_values(n_bins);
//     
//     for(size_t i = 0; i < n_bins; ++i) {
//       std::ostringstream oss;
//       oss << "(";
//       if(std::isinf(bins[i].lower) && bins[i].lower < 0) {
//         oss << "-Inf";
//       } else {
//         oss << bins[i].lower;
//       }
//       oss << ";";
//       if(std::isinf(bins[i].upper)) {
//         oss << "+Inf";
//       } else {
//         oss << bins[i].upper;
//       }
//       oss << "]";
//       bin_strings[i] = oss.str();
//       
//       woe_values[i] = bins[i].woe;
//       iv_values[i] = bins[i].iv;
//       count_values[i] = bins[i].count;
//       count_pos_values[i] = bins[i].count_pos;
//       count_neg_values[i] = bins[i].count_neg;
//     }
//     
//     woebin = Rcpp::DataFrame::create(
//       Rcpp::Named("bin") = bin_strings,
//       Rcpp::Named("woe") = woe_values,
//       Rcpp::Named("iv") = iv_values,
//       Rcpp::Named("count") = count_values,
//       Rcpp::Named("count_pos") = count_pos_values,
//       Rcpp::Named("count_neg") = count_neg_values,
//       Rcpp::Named("stringsAsFactors") = false
//     );
//   }
//   
// public:
//   // Constructor
//   OptimalBinningNumericalUBSD(const std::vector<double>& feat, const std::vector<double>& targ,
//                               int min_b, int max_b, double cutoff, int max_prebins) :
//   feature(feat), target(targ),
//   min_bins(min_b), max_bins(max_b),
//   bin_cutoff(cutoff), max_n_prebins(max_prebins) {}
//   
//   // Fit the binning model
//   void fit() {
//     validate_inputs();
//     initial_binning();
//     assign_bins();
//     merge_bins_by_cutoff();
//     calculate_woe_iv();
//     enforce_monotonicity();
//     
//     // After enforcing monotonicity, ensure bin count does not exceed max_bins
//     if(bins.size() > static_cast<size_t>(max_bins)) {
//       merge_to_max_bins();
//     }
//     
//     apply_woe_to_feature();
//     prepare_output();
//   }
//   
//   // Getters for output
//   std::vector<double> get_woefeature() const {
//     return woefeature;
//   }
//   
//   Rcpp::DataFrame get_woebin() const {
//     return woebin;
//   }
// };
// 
// 
// //' @title Optimal Binning for Numerical Variables using Unsupervised Binning with Standard Deviation
// //' 
// //' @description
// //' This function implements an optimal binning algorithm for numerical variables using an 
// //' Unsupervised Binning approach based on Standard Deviation (UBSD) with Weight of Evidence (WoE) 
// //' and Information Value (IV) criteria.
// //' 
// //' @param target A numeric vector of binary target values (should contain exactly two unique values).
// //' @param feature A numeric vector of feature values to be binned.
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff Minimum frequency of observations in each bin (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins for initial standard deviation-based discretization (default: 20).
// //' 
// //' @return A list containing two elements:
// //' \item{woefeature}{A numeric vector of WoE-transformed feature values.}
// //' \item{woebin}{A data frame with binning details, including bin boundaries, WoE, IV, and count statistics.}
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
// //' 7. Application of WoE transformation to the original feature
// //' 
// //' Weight of Evidence (WoE) is calculated for each bin as:
// //' 
// //' \deqn{WoE_i = \ln\left(\frac{P(X_i|Y=1)}{P(X_i|Y=0)}\right)}
// //' 
// //' where \eqn{P(X_i|Y=1)} is the proportion of positive cases in bin i, and 
// //' \eqn{P(X_i|Y=0)} is the proportion of negative cases in bin i.
// //' 
// //' Information Value (IV) for each bin is calculated as:
// //' 
// //' \deqn{IV_i = (P(X_i|Y=1) - P(X_i|Y=0)) * WoE_i}
// //' 
// //' The total IV for the feature is the sum of IVs across all bins:
// //' 
// //' \deqn{IV_{total} = \sum_{i=1}^{n} IV_i}
// //' 
// //' The UBSD approach ensures that the resulting binning maximizes the separation between 
// //' classes while maintaining the desired number of bins and respecting the minimum bin 
// //' frequency constraint.
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
// //' print(result$woebin)
// //' 
// //' # Plot WoE transformation
// //' plot(feature, result$woefeature, main = "WoE Transformation", 
// //'      xlab = "Original Feature", ylab = "WoE")
// //' }
// //' 
// //' @references
// //' \itemize{
// //' \item Kotsiantis, S., & Kanellopoulos, D. (2006). Discretization techniques: A recent survey. 
// //'       GESTS International Transactions on Computer Science and Engineering, 32(1), 47-58.
// //' \item Dougherty, J., Kohavi, R., & Sahami, M. (1995). Supervised and unsupervised 
// //'       discretization of continuous features. In Machine Learning Proceedings 1995 
// //'       (pp. 194-202). Morgan Kaufmann.
// //' }
// //' 
// //' @author Lopes, J. E.
// //' 
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_numerical_ubsd(Rcpp::NumericVector target,
//                                           Rcpp::NumericVector feature,
//                                           int min_bins = 3,
//                                           int max_bins = 5,
//                                           double bin_cutoff = 0.05,
//                                           int max_n_prebins = 20) {
//   // Convert R vectors to std::vector
//   std::vector<double> std_feature(feature.begin(), feature.end());
//   std::vector<double> std_target(target.begin(), target.end());
//   
//   // Validate that target is binary
//   std::vector<double> unique_vals = std_target;
//   std::sort(unique_vals.begin(), unique_vals.end());
//   unique_vals.erase(std::unique(unique_vals.begin(), unique_vals.end()), unique_vals.end());
//   if(unique_vals.size() != 2) {
//     Rcpp::stop("Target must be binary (contain exactly two unique values).");
//   }
//   
//   // Instantiate and fit the binning model
//   OptimalBinningNumericalUBSD binning_model(std_feature, std_target,
//                                             min_bins, max_bins,
//                                             bin_cutoff, max_n_prebins);
//   binning_model.fit();
//   
//   // Prepare the output list
//   return Rcpp::List::create(
//     Rcpp::Named("woefeature") = binning_model.get_woefeature(),
//     Rcpp::Named("woebin") = binning_model.get_woebin()
//   );
// }
