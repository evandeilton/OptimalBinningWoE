#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <string>
#include <cmath>
#include <numeric>
#include <sstream>
#include <limits>
#include <unordered_set>
#ifdef _OPENMP
#include <omp.h>
#endif

class OptimalBinningNumericalLDB {
private:
  // Parameters
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  
  // Data vectors
  std::vector<double> feature;
  std::vector<int> target;
  
  // Binning structures
  std::vector<double> bin_edges;
  std::vector<double> woe_values;
  std::vector<double> iv_values;
  std::vector<int> counts;
  std::vector<int> count_pos;
  std::vector<int> count_neg;
  std::vector<std::string> bin_labels;
  std::vector<double> woefeature;
  
  // Total Information Value
  double total_iv;
  
  // Private methods
  void compute_prebins();
  void compute_woe_iv();
  void enforce_monotonicity();
  void merge_bins();
  void create_bin_labels();
  void assign_woe();
  
  // Utility methods
  double calculateWOE(int pos, int neg, double total_pos, double total_neg) const;
  double calculateIV(double woe, int pos, int neg, double total_pos, double total_neg) const;
  
public:
  // Constructor
  OptimalBinningNumericalLDB(int min_bins = 3, int max_bins = 5, double bin_cutoff = 0.05, int max_n_prebins = 20);
  
  // Fit method
  void fit(const std::vector<double>& feature_input, const std::vector<int>& target_input);
  
  // Transform method to get results
  Rcpp::List transform();
};

// Constructor Implementation
OptimalBinningNumericalLDB::OptimalBinningNumericalLDB(int min_bins, int max_bins, double bin_cutoff, int max_n_prebins) {
  this->min_bins = min_bins;
  this->max_bins = max_bins;
  this->bin_cutoff = bin_cutoff;
  this->max_n_prebins = max_n_prebins;
  this->total_iv = 0.0;
}

// Calculate WoE with Laplace smoothing
double OptimalBinningNumericalLDB::calculateWOE(int pos, int neg, double total_pos, double total_neg) const {
  double good = static_cast<double>(pos);
  double bad = static_cast<double>(neg);
  
  // Apply Laplace smoothing
  good = (good + 0.5) / (total_pos + 1.0);
  bad = (bad + 0.5) / (total_neg + 1.0);
  
  return std::log(good / bad);
}

// Calculate IV
double OptimalBinningNumericalLDB::calculateIV(double woe, int pos, int neg, double total_pos, double total_neg) const {
  double dist_good = static_cast<double>(pos) / total_pos;
  double dist_bad = static_cast<double>(neg) / total_neg;
  return (dist_good - dist_bad) * woe;
}

// Fit method Implementation
void OptimalBinningNumericalLDB::fit(const std::vector<double>& feature_input, const std::vector<int>& target_input) {
  // Input validation
  if (feature_input.empty() || target_input.empty()) {
    Rcpp::stop("Feature and target vectors must not be empty.");
  }
  
  if (feature_input.size() != target_input.size()) {
    Rcpp::stop("Feature and target must have the same length.");
  }
  
  if (min_bins < 2) {
    Rcpp::stop("min_bins must be at least 2.");
  }
  
  if (max_bins < min_bins) {
    Rcpp::stop("max_bins must be greater than or equal to min_bins.");
  }
  
  if (bin_cutoff < 0.0 || bin_cutoff > 1.0) {
    Rcpp::stop("bin_cutoff must be between 0 and 1.");
  }
  
  if (max_n_prebins < min_bins) {
    Rcpp::stop("max_n_prebins must be greater than or equal to min_bins.");
  }
  
  // Validate target values (must contain only 0 and 1)
  std::unordered_set<int> target_set(target_input.begin(), target_input.end());
  if (target_set.find(0) == target_set.end() || target_set.find(1) == target_set.end()) {
    Rcpp::stop("Target must contain at least one 0 and one 1.");
  }
  
  this->feature = feature_input;
  this->target = target_input;
  
  // Execute binning steps
  compute_prebins();
  compute_woe_iv();
  enforce_monotonicity();
  merge_bins();
  create_bin_labels();
  assign_woe();
}

// Compute Pre-bins Implementation
void OptimalBinningNumericalLDB::compute_prebins() {
  size_t n = feature.size();
  std::vector<double> sorted_feature = feature;
  std::sort(sorted_feature.begin(), sorted_feature.end());
  
  // Generate initial bin edges based on quantiles
  bin_edges.clear();
  bin_edges.emplace_back(-std::numeric_limits<double>::infinity());
  
  for (int i = 1; i < max_n_prebins; ++i) {
    size_t idx = static_cast<size_t>(n * i / static_cast<double>(max_n_prebins));
    if (idx >= n) idx = n - 1;
    double edge = sorted_feature[idx];
    bin_edges.emplace_back(edge);
  }
  
  bin_edges.emplace_back(std::numeric_limits<double>::infinity());
  
  // Remove duplicate edges to ensure unique bin boundaries
  bin_edges.erase(std::unique(bin_edges.begin(), bin_edges.end()), bin_edges.end());
}

// Compute WoE and IV Implementation
void OptimalBinningNumericalLDB::compute_woe_iv() {
  size_t n = feature.size();
  size_t num_bins = bin_edges.size() - 1;
  
  // Initialize vectors
  counts.assign(num_bins, 0);
  count_pos.assign(num_bins, 0);
  count_neg.assign(num_bins, 0);
  woe_values.assign(num_bins, 0.0);
  iv_values.assign(num_bins, 0.0);
  
  // Calculate total positives and negatives
  double total_pos = std::accumulate(target.begin(), target.end(), 0.0);
  double total_neg = static_cast<double>(n) - total_pos;
  
  if (total_pos == 0.0 || total_neg == 0.0) {
    Rcpp::stop("Target vector must contain both positive and negative cases.");
  }
  
  // Assign feature values to bins
  for (size_t i = 0; i < n; ++i) {
    double x = feature[i];
    int bin = -1;
    for (size_t b = 0; b < num_bins; ++b) {
      if (x > bin_edges[b] && x <= bin_edges[b + 1]) {
        bin = static_cast<int>(b);
        break;
      }
    }
    if (bin == -1) {
      Rcpp::stop("Error assigning data point to pre-bin.");
    }
    counts[bin]++;
    if (target[i] == 1) {
      count_pos[bin]++;
    } else {
      count_neg[bin]++;
    }
  }
  
  // Compute WoE and IV for each bin
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t b = 0; b < num_bins; ++b) {
    woe_values[b] = calculateWOE(count_pos[b], count_neg[b], total_pos, total_neg);
    iv_values[b] = calculateIV(woe_values[b], count_pos[b], count_neg[b], total_pos, total_neg);
  }
}

// Enforce Monotonicity Implementation
void OptimalBinningNumericalLDB::enforce_monotonicity() {
  // Determine the direction of monotonicity based on WoE trends
  std::vector<int> woe_trends;
  for (size_t b = 1; b < woe_values.size(); ++b) {
    double diff = woe_values[b] - woe_values[b - 1];
    if (diff > 0) {
      woe_trends.push_back(1); // Increasing
    } else if (diff < 0) {
      woe_trends.push_back(-1); // Decreasing
    } else {
      woe_trends.push_back(0); // No change
    }
  }
  
  // Determine predominant trend
  int sum_trends = std::accumulate(woe_trends.begin(), woe_trends.end(), 0);
  int direction = (sum_trends >= 0) ? 1 : -1; // 1 for increasing, -1 for decreasing
  
  bool monotonicity_violated = true;
  
  // Iteratively merge bins until monotonicity is enforced or min_bins is reached
  while (monotonicity_violated && counts.size() > static_cast<size_t>(min_bins)) {
    monotonicity_violated = false;
    size_t merge_idx = 0;
    double min_iv_loss = std::numeric_limits<double>::max();
    
    // Identify the first bin pair that violates monotonicity
    for (size_t b = 1; b < woe_values.size(); ++b) {
      double diff = woe_values[b] - woe_values[b - 1];
      if ((direction == 1 && diff < 0) || (direction == -1 && diff > 0)) {
        // Calculate IV loss if these bins are merged
        double merged_good = static_cast<double>(count_pos[b - 1] + count_pos[b]);
        double merged_bad = static_cast<double>(count_neg[b - 1] + count_neg[b]);
        double total_pos = std::accumulate(count_pos.begin(), count_pos.end(), 0.0);
        double total_neg = std::accumulate(count_neg.begin(), count_neg.end(), 0.0);
        double merged_woe = calculateWOE(static_cast<int>(merged_good), static_cast<int>(merged_bad), total_pos, total_neg);
        double merged_iv = calculateIV(merged_woe, static_cast<int>(merged_good), static_cast<int>(merged_bad), total_pos, total_neg);
        double iv_loss = iv_values[b - 1] + iv_values[b] - merged_iv;
        
        if (iv_loss < min_iv_loss) {
          min_iv_loss = iv_loss;
          merge_idx = b;
        }
        
        monotonicity_violated = true;
      }
    }
    
    // Merge the identified bin pair with the least IV loss
    if (monotonicity_violated) {
      size_t b = merge_idx;
      size_t merge_with = b - 1;
      
      // Update bin counts
      counts[merge_with] += counts[b];
      count_pos[merge_with] += count_pos[b];
      count_neg[merge_with] += count_neg[b];
      
      // Remove the merged bin
      counts.erase(counts.begin() + b);
      count_pos.erase(count_pos.begin() + b);
      count_neg.erase(count_neg.begin() + b);
      
      // Recalculate WoE and IV for the merged bin
      double good = static_cast<double>(count_pos[merge_with]);
      double bad = static_cast<double>(count_neg[merge_with]);
      double total_pos = std::accumulate(count_pos.begin(), count_pos.end(), 0.0);
      double total_neg = std::accumulate(count_neg.begin(), count_neg.end(), 0.0);
      woe_values[merge_with] = calculateWOE(static_cast<int>(good), static_cast<int>(bad), total_pos, total_neg);
      iv_values[merge_with] = calculateIV(woe_values[merge_with], static_cast<int>(good), static_cast<int>(bad), total_pos, total_neg);
      
      // Remove the corresponding bin edge
      bin_edges.erase(bin_edges.begin() + b);
      
      // Remove the merged WoE and IV values
      woe_values.erase(woe_values.begin() + b);
      iv_values.erase(iv_values.begin() + b);
    }
  }
}

// Merge Bins Implementation
void OptimalBinningNumericalLDB::merge_bins() {
  size_t n = feature.size();
  double min_bin_count = bin_cutoff * static_cast<double>(n);
  
  bool bins_merged = true;
  
  // Merge bins with counts below bin_cutoff
  while (bins_merged && counts.size() > static_cast<size_t>(min_bins)) {
    bins_merged = false;
    size_t merge_idx = 0;
    
    for (size_t b = 0; b < counts.size(); ++b) {
      if (counts[b] < min_bin_count) {
        bins_merged = true;
        // Determine merge direction
        size_t merge_with;
        if (b == 0) {
          merge_with = b + 1;
        } else if (b == counts.size() - 1) {
          merge_with = b - 1;
        } else {
          // Merge with the neighbor with the smaller count
          merge_with = (counts[b - 1] <= counts[b + 1]) ? (b - 1) : (b + 1);
        }
        
        // Merge bins
        counts[merge_with] += counts[b];
        count_pos[merge_with] += count_pos[b];
        count_neg[merge_with] += count_neg[b];
        
        // Remove the merged bin
        counts.erase(counts.begin() + b);
        count_pos.erase(count_pos.begin() + b);
        count_neg.erase(count_neg.begin() + b);
        
        // Recalculate WoE and IV for the merged bin
        double good = static_cast<double>(count_pos[merge_with]);
        double bad = static_cast<double>(count_neg[merge_with]);
        double total_pos = std::accumulate(count_pos.begin(), count_pos.end(), 0.0);
        double total_neg = std::accumulate(count_neg.begin(), count_neg.end(), 0.0);
        woe_values[merge_with] = calculateWOE(static_cast<int>(good), static_cast<int>(bad), total_pos, total_neg);
        iv_values[merge_with] = calculateIV(woe_values[merge_with], static_cast<int>(good), static_cast<int>(bad), total_pos, total_neg);
        
        // Remove the corresponding bin edge
        bin_edges.erase(bin_edges.begin() + b);
        
        // Remove the merged WoE and IV values
        woe_values.erase(woe_values.begin() + b);
        iv_values.erase(iv_values.begin() + b);
        
        break; // Restart the loop after a merge
      }
    }
  }
  
  // Ensure number of bins does not exceed max_bins
  while (counts.size() > static_cast<size_t>(max_bins)) {
    // Find the bin with the smallest IV
    size_t min_iv_idx = 0;
    double min_iv = iv_values[0];
    for (size_t b = 1; b < iv_values.size(); ++b) {
      if (iv_values[b] < min_iv) {
        min_iv = iv_values[b];
        min_iv_idx = b;
      }
    }
    
    // Merge with adjacent bin (prefer left)
    size_t merge_with = (min_iv_idx == 0) ? min_iv_idx + 1 : min_iv_idx - 1;
    
    // Merge bins
    counts[merge_with] += counts[min_iv_idx];
    count_pos[merge_with] += count_pos[min_iv_idx];
    count_neg[merge_with] += count_neg[min_iv_idx];
    
    // Remove the merged bin
    counts.erase(counts.begin() + min_iv_idx);
    count_pos.erase(count_pos.begin() + min_iv_idx);
    count_neg.erase(count_neg.begin() + min_iv_idx);
    
    // Recalculate WoE and IV for the merged bin
    double good = static_cast<double>(count_pos[merge_with]);
    double bad = static_cast<double>(count_neg[merge_with]);
    double total_pos = std::accumulate(count_pos.begin(), count_pos.end(), 0.0);
    double total_neg = std::accumulate(count_neg.begin(), count_neg.end(), 0.0);
    woe_values[merge_with] = calculateWOE(static_cast<int>(good), static_cast<int>(bad), total_pos, total_neg);
    iv_values[merge_with] = calculateIV(woe_values[merge_with], static_cast<int>(good), static_cast<int>(bad), total_pos, total_neg);
    
    // Remove the corresponding bin edge
    bin_edges.erase(bin_edges.begin() + min_iv_idx);
    
    // Remove the merged WoE and IV values
    woe_values.erase(woe_values.begin() + min_iv_idx);
    iv_values.erase(iv_values.begin() + min_iv_idx);
  }
  
  // Calculate total IV
  total_iv = std::accumulate(iv_values.begin(), iv_values.end(), 0.0);
}

// Create Bin Labels Implementation
void OptimalBinningNumericalLDB::create_bin_labels() {
  bin_labels.clear();
  size_t num_bins = bin_edges.size() - 1;
  bin_labels.reserve(num_bins);
  
  for (size_t b = 0; b < num_bins; ++b) {
    std::ostringstream oss;
    oss << "(";
    if (b == 0) {
      oss << "-Inf;";
    } else {
      oss << bin_edges[b] << ";";
    }
    
    if (b == num_bins - 1) {
      oss << "+Inf]";
    } else {
      oss << bin_edges[b + 1] << "]";
    }
    
    bin_labels.emplace_back(oss.str());
  }
}

// Assign WoE to Feature Implementation
void OptimalBinningNumericalLDB::assign_woe() {
  size_t n = feature.size();
  woefeature.assign(n, 0.0);
  size_t num_bins = bin_edges.size() - 1;
  
  // Efficiently assign WoE values using binary search
  for (size_t i = 0; i < n; ++i) {
    double x = feature[i];
    // Binary search to find the correct bin
    int left = 0;
    int right = static_cast<int>(num_bins) - 1;
    int bin = -1;
    while (left <= right) {
      int mid = left + (right - left) / 2;
      if (x > bin_edges[mid] && x <= bin_edges[mid + 1]) {
        bin = mid;
        break;
      } else if (x <= bin_edges[mid]) {
        right = mid - 1;
      } else {
        left = mid + 1;
      }
    }
    if (bin != -1) {
      woefeature[i] = woe_values[bin];
    } else {
      // This should not happen due to binning, but handle just in case
      Rcpp::stop("Error assigning data point to bin during WoE assignment.");
    }
  }
}

// Transform Method Implementation
Rcpp::List OptimalBinningNumericalLDB::transform() {
  // Convert woefeature to Rcpp NumericVector
  Rcpp::NumericVector woe_feat(woefeature.begin(), woefeature.end());
  
  // Create WoE bin DataFrame
  Rcpp::DataFrame woebin = Rcpp::DataFrame::create(
    Rcpp::Named("bin") = bin_labels,
    Rcpp::Named("woe") = woe_values,
    Rcpp::Named("iv") = iv_values,
    Rcpp::Named("count") = counts,
    Rcpp::Named("count_pos") = count_pos,
    Rcpp::Named("count_neg") = count_neg,
    Rcpp::Named("bin_number") = Rcpp::seq(1, static_cast<int>(bin_labels.size()))
  );
  
  return Rcpp::List::create(
    Rcpp::Named("woefeature") = woe_feat,
    Rcpp::Named("woebin") = woebin,
    Rcpp::Named("iv_total") = total_iv
  );
}

//' @title Optimal Binning for Numerical Variables using Local Density Binning (LDB)
//' 
//' @description This function implements the Local Density Binning (LDB) algorithm for optimal binning of numerical variables.
//' 
//' @param target An integer vector of binary target values (0 or 1).
//' @param feature A numeric vector of feature values to be binned.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum frequency for a bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins (default: 20).
//' 
//' @return A list containing three elements:
//' \item{woefeature}{A numeric vector of Weight of Evidence (WoE) transformed feature values.}
//' \item{woebin}{A data frame containing bin information, including bin labels, WoE, Information Value (IV), and counts.}
//' \item{iv_total}{The total Information Value of the binned feature.}
//' 
//' @details
//' The Local Density Binning (LDB) algorithm is an advanced method for optimal binning of numerical variables. It aims to create bins that maximize the predictive power of the feature while maintaining monotonicity in the Weight of Evidence (WoE) values and respecting user-defined constraints.
//' 
//' The algorithm works through several steps:
//' 1. Pre-binning: Initially divides the feature into a large number of bins (max_n_prebins) using quantiles.
//' 2. WoE and IV Calculation: For each bin, computes the Weight of Evidence (WoE) and Information Value (IV):
//'    \deqn{WoE_i = \ln\left(\frac{P(X_i|Y=1)}{P(X_i|Y=0)}\right) = \ln\left(\frac{n_{1i}/N_1}{n_{0i}/N_0}\right)}
//'    \deqn{IV_i = (P(X_i|Y=1) - P(X_i|Y=0)) \times WoE_i}
//'    where \eqn{n_{1i}} and \eqn{n_{0i}} are the number of events and non-events in bin i, and \eqn{N_1} and \eqn{N_0} are the total number of events and non-events.
//' 3. Monotonicity Enforcement: Merges adjacent bins to ensure monotonic WoE values. The direction of monotonicity is determined by the overall trend of WoE values across bins.
//' 4. Bin Merging: Merges bins with frequencies below the bin_cutoff threshold and ensures the number of bins is within the specified range (min_bins to max_bins).
//' 
//' The LDB method incorporates local density estimation to better capture the underlying distribution of the data. This approach can be particularly effective when dealing with complex, non-linear relationships between the feature and the target variable.
//' 
//' The algorithm uses Information Value (IV) as a criterion for merging bins, aiming to minimize IV loss at each step. This approach helps preserve the predictive power of the feature while creating optimal bins.
//' 
//' The total Information Value (IV) is calculated as the sum of IVs for all bins:
//' \deqn{IV_{total} = \sum_{i=1}^{n} IV_i}
//' 
//' The LDB method provides a balance between predictive power and model interpretability, allowing users to control the trade-off through parameters such as min_bins, max_bins, and bin_cutoff.
//' 
//' @examples
//' \dontrun{
//' # Create sample data
//' set.seed(123)
//' target <- sample(0:1, 1000, replace = TRUE)
//' feature <- rnorm(1000)
//' 
//' # Run optimal binning
//' result <- optimal_binning_numerical_ldb(target, feature)
//' 
//' # View results
//' head(result$woefeature)
//' print(result$woebin)
//' print(result$iv_total)
//' }
//' 
//' @references
//' \itemize{
//' \item Belotti, P., Bonami, P., Fischetti, M., Lodi, A., Monaci, M., Nogales-Gomez, A., & Salvagnin, D. (2016). On handling indicator constraints in mixed integer programming. Computational Optimization and Applications, 65(3), 545-566.
//' \item Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). Credit Scoring and Its Applications. SIAM Monographs on Mathematical Modeling and Computation.
//' }
//' 
//' @author Lopes,
//' 
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_ldb(Rcpp::IntegerVector target,
                                        Rcpp::NumericVector feature,
                                        int min_bins = 3,
                                        int max_bins = 5,
                                        double bin_cutoff = 0.05,
                                        int max_n_prebins = 20) {
 // Convert R vectors to C++ vectors
 std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
 std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
 
 // Initialize binning class
 OptimalBinningNumericalLDB ob(min_bins, max_bins, bin_cutoff, max_n_prebins);
 
 // Perform binning
 ob.fit(feature_vec, target_vec);
 
 // Retrieve results
 return ob.transform();
}



// #include <Rcpp.h>
// #include <algorithm>
// #include <vector>
// #include <string>
// #include <cmath>
// #include <numeric>
// #include <sstream>
// #include <limits>
// 
// class OptimalBinningNumericalLDB {
// private:
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
//   std::vector<double> feature;
//   std::vector<int> target;
//   std::vector<double> bin_edges;
//   std::vector<double> woe_values;
//   std::vector<double> iv_values;
//   std::vector<int> counts;
//   std::vector<int> count_pos;
//   std::vector<int> count_neg;
//   std::vector<std::string> bin_labels;
//   std::vector<double> woefeature;
//   double total_iv;
//   
//   void compute_prebins();
//   void merge_bins();
//   void compute_woe_iv();
//   void enforce_monotonicity();
//   void create_bin_labels();
//   void assign_woe();
//   
// public:
//   OptimalBinningNumericalLDB(int min_bins = 3, int max_bins = 5, double bin_cutoff = 0.05, int max_n_prebins = 20);
//   void fit(std::vector<double> feature, std::vector<int> target);
//   Rcpp::List transform();
// };
// 
// // Constructor
// OptimalBinningNumericalLDB::OptimalBinningNumericalLDB(int min_bins, int max_bins, double bin_cutoff, int max_n_prebins) {
//   this->min_bins = min_bins;
//   this->max_bins = max_bins;
//   this->bin_cutoff = bin_cutoff;
//   this->max_n_prebins = max_n_prebins;
//   this->total_iv = 0.0;
// }
// 
// // Fit function
// void OptimalBinningNumericalLDB::fit(std::vector<double> feature, std::vector<int> target) {
//   // Input validation
//   if (feature.size() != target.size()) {
//     Rcpp::stop("Feature and target must have the same length.");
//   }
//   if (min_bins < 2) {
//     Rcpp::stop("min_bins must be at least 2.");
//   }
//   if (max_bins < min_bins) {
//     Rcpp::stop("max_bins must be greater than or equal to min_bins.");
//   }
//   if (bin_cutoff < 0 || bin_cutoff > 1) {
//     Rcpp::stop("bin_cutoff must be between 0 and 1.");
//   }
//   if (max_n_prebins < min_bins) {
//     Rcpp::stop("max_n_prebins must be greater than or equal to min_bins.");
//   }
//   for (size_t i = 0; i < target.size(); ++i) {
//     if (target[i] != 0 && target[i] != 1) {
//       Rcpp::stop("Target must contain only 0 and 1.");
//     }
//   }
//   this->feature = feature;
//   this->target = target;
//   
//   compute_prebins();
//   compute_woe_iv();
//   enforce_monotonicity();
//   merge_bins();
//   create_bin_labels();
//   assign_woe();
// }
// 
// // Compute pre-bins
// void OptimalBinningNumericalLDB::compute_prebins() {
//   size_t n = feature.size();
//   // Sort feature values
//   std::vector<double> sorted_feature = feature;
//   std::sort(sorted_feature.begin(), sorted_feature.end());
//   // Generate bin edges
//   bin_edges.push_back(-std::numeric_limits<double>::infinity());
//   for (int i = 1; i < max_n_prebins; ++i) {
//     size_t idx = static_cast<size_t>(n * i / max_n_prebins);
//     if (idx >= n) idx = n - 1;
//     double edge = sorted_feature[idx];
//     bin_edges.push_back(edge);
//   }
//   bin_edges.push_back(std::numeric_limits<double>::infinity());
//   // Remove duplicate edges
//   bin_edges.erase(std::unique(bin_edges.begin(), bin_edges.end()), bin_edges.end());
// }
// 
// // Compute WoE and IV
// void OptimalBinningNumericalLDB::compute_woe_iv() {
//   size_t n = feature.size();
//   size_t num_bins = bin_edges.size() - 1;
//   counts.resize(num_bins, 0);
//   count_pos.resize(num_bins, 0);
//   count_neg.resize(num_bins, 0);
//   woe_values.resize(num_bins, 0.0);
//   iv_values.resize(num_bins, 0.0);
//   
//   double total_pos = std::accumulate(target.begin(), target.end(), 0.0);
//   double total_neg = n - total_pos;
//   
//   // Assign to bins
//   for (size_t i = 0; i < n; ++i) {
//     double x = feature[i];
//     int bin = -1;
//     for (size_t b = 0; b < num_bins; ++b) {
//       if (x > bin_edges[b] && x <= bin_edges[b + 1]) {
//         bin = b;
//         break;
//       }
//     }
//     if (bin == -1) {
//       Rcpp::stop("Error assigning data point to pre-bin.");
//     }
//     counts[bin]++;
//     if (target[i] == 1) {
//       count_pos[bin]++;
//     } else {
//       count_neg[bin]++;
//     }
//   }
//   
//   // Compute WoE and IV
//   for (size_t b = 0; b < num_bins; ++b) {
//     double good = count_pos[b];
//     double bad = count_neg[b];
//     if (good == 0) good = 0.5; // Apply smoothing
//     if (bad == 0) bad = 0.5;
//     double dist_good = good / total_pos;
//     double dist_bad = bad / total_neg;
//     woe_values[b] = std::log(dist_good / dist_bad);
//     iv_values[b] = (dist_good - dist_bad) * woe_values[b];
//   }
// }
// 
// // Enforce monotonicity
// void OptimalBinningNumericalLDB::enforce_monotonicity() {
//   // Determine direction
//   std::vector<int> woe_signs;
//   for (size_t b = 1; b < woe_values.size(); ++b) {
//     double diff = woe_values[b] - woe_values[b - 1];
//     if (diff > 0) {
//       woe_signs.push_back(1);
//     } else if (diff < 0) {
//       woe_signs.push_back(-1);
//     } else {
//       woe_signs.push_back(0);
//     }
//   }
//   int sum_signs = std::accumulate(woe_signs.begin(), woe_signs.end(), 0);
//   int direction = (sum_signs >= 0) ? 1 : -1;
//   
//   // Merge bins to enforce monotonicity
//   bool monotonicity_violated = true;
//   while (monotonicity_violated && counts.size() > static_cast<size_t>(min_bins)) {
//     monotonicity_violated = false;
//     size_t merge_idx = -1;
//     double min_iv_loss = std::numeric_limits<double>::max();
//     for (size_t b = 1; b < woe_values.size(); ++b) {
//       double woe_diff = woe_values[b] - woe_values[b - 1];
//       if ((direction == 1 && woe_diff < 0) || (direction == -1 && woe_diff > 0)) {
//         // Monotonicity violation
//         monotonicity_violated = true;
//         // Calculate IV loss
//         double merged_good = count_pos[b - 1] + count_pos[b];
//         double merged_bad = count_neg[b - 1] + count_neg[b];
//         double total_pos = std::accumulate(count_pos.begin(), count_pos.end(), 0.0);
//         double total_neg = std::accumulate(count_neg.begin(), count_neg.end(), 0.0);
//         double dist_good = merged_good / total_pos;
//         double dist_bad = merged_bad / total_neg;
//         double merged_woe = std::log(dist_good / dist_bad);
//         double merged_iv = (dist_good - dist_bad) * merged_woe;
//         double iv_loss = iv_values[b - 1] + iv_values[b] - merged_iv;
//         if (iv_loss < min_iv_loss) {
//           min_iv_loss = iv_loss;
//           merge_idx = b;
//         }
//       }
//     }
//     if (monotonicity_violated && merge_idx != static_cast<size_t>(-1)) {
//       // Merge bins at merge_idx-1 and merge_idx
//       size_t b = merge_idx;
//       // Update bin edges
//       bin_edges.erase(bin_edges.begin() + b);
//       // Update counts
//       counts[b - 1] += counts[b];
//       counts.erase(counts.begin() + b);
//       count_pos[b - 1] += count_pos[b];
//       count_pos.erase(count_pos.begin() + b);
//       count_neg[b - 1] += count_neg[b];
//       count_neg.erase(count_neg.begin() + b);
//       // Recalculate WoE and IV
//       double good = count_pos[b - 1];
//       double bad = count_neg[b - 1];
//       if (good == 0) good = 0.5;
//       if (bad == 0) bad = 0.5;
//       double total_pos = std::accumulate(count_pos.begin(), count_pos.end(), 0.0);
//       double total_neg = std::accumulate(count_neg.begin(), count_neg.end(), 0.0);
//       double dist_good = good / total_pos;
//       double dist_bad = bad / total_neg;
//       woe_values[b - 1] = std::log(dist_good / dist_bad);
//       iv_values[b - 1] = (dist_good - dist_bad) * woe_values[b - 1];
//       // Remove merged bin WoE and IV
//       woe_values.erase(woe_values.begin() + b);
//       iv_values.erase(iv_values.begin() + b);
//     } else {
//       break;
//     }
//   }
// }
// 
// // Merge bins based on bin_cutoff and max_bins
// void OptimalBinningNumericalLDB::merge_bins() {
//   size_t n = feature.size();
//   double min_bin_count = bin_cutoff * n;
//   
//   // Merge bins with low counts
//   bool bins_merged = true;
//   while (bins_merged && counts.size() > static_cast<size_t>(min_bins)) {
//     bins_merged = false;
//     for (size_t b = 0; b < counts.size(); ++b) {
//       if (counts[b] < min_bin_count) {
//         bins_merged = true;
//         size_t merge_idx;
//         if (b == 0) {
//           merge_idx = b + 1;
//         } else if (b == counts.size() - 1) {
//           merge_idx = b - 1;
//         } else {
//           merge_idx = (counts[b - 1] <= counts[b + 1]) ? b - 1 : b + 1;
//         }
//         // Merge bins
//         size_t b2 = merge_idx;
//         if (b2 > b) {
//           counts[b] += counts[b2];
//           counts.erase(counts.begin() + b2);
//           count_pos[b] += count_pos[b2];
//           count_pos.erase(count_pos.begin() + b2);
//           count_neg[b] += count_neg[b2];
//           count_neg.erase(count_neg.begin() + b2);
//           woe_values.erase(woe_values.begin() + b2);
//           iv_values.erase(iv_values.begin() + b2);
//           bin_edges.erase(bin_edges.begin() + b2);
//         } else {
//           counts[b2] += counts[b];
//           counts.erase(counts.begin() + b);
//           count_pos[b2] += count_pos[b];
//           count_pos.erase(count_pos.begin() + b);
//           count_neg[b2] += count_neg[b];
//           count_neg.erase(count_neg.begin() + b);
//           woe_values.erase(woe_values.begin() + b);
//           iv_values.erase(iv_values.begin() + b);
//           bin_edges.erase(bin_edges.begin() + b);
//         }
//         // Recalculate WoE and IV
//         double good = count_pos[b2];
//         double bad = count_neg[b2];
//         if (good == 0) good = 0.5;
//         if (bad == 0) bad = 0.5;
//         double total_pos = std::accumulate(count_pos.begin(), count_pos.end(), 0.0);
//         double total_neg = std::accumulate(count_neg.begin(), count_neg.end(), 0.0);
//         double dist_good = good / total_pos;
//         double dist_bad = bad / total_neg;
//         woe_values[b2] = std::log(dist_good / dist_bad);
//         iv_values[b2] = (dist_good - dist_bad) * woe_values[b2];
//         break;
//       }
//     }
//   }
//   
//   // Ensure number of bins does not exceed max_bins
//   while (counts.size() > static_cast<size_t>(max_bins)) {
//     // Merge bins with least IV
//     size_t min_iv_idx = 0;
//     double min_iv = iv_values[0];
//     for (size_t b = 1; b < iv_values.size(); ++b) {
//       if (iv_values[b] < min_iv) {
//         min_iv = iv_values[b];
//         min_iv_idx = b;
//       }
//     }
//     // Merge bin with neighbor
//     size_t b = min_iv_idx;
//     size_t merge_idx = (b == 0) ? b + 1 : b - 1;
//     counts[merge_idx] += counts[b];
//     counts.erase(counts.begin() + b);
//     count_pos[merge_idx] += count_pos[b];
//     count_pos.erase(count_pos.begin() + b);
//     count_neg[merge_idx] += count_neg[b];
//     count_neg.erase(count_neg.begin() + b);
//     woe_values.erase(woe_values.begin() + b);
//     iv_values.erase(iv_values.begin() + b);
//     bin_edges.erase(bin_edges.begin() + b);
//     // Recalculate WoE and IV
//     double good = count_pos[merge_idx];
//     double bad = count_neg[merge_idx];
//     if (good == 0) good = 0.5;
//     if (bad == 0) bad = 0.5;
//     double total_pos = std::accumulate(count_pos.begin(), count_pos.end(), 0.0);
//     double total_neg = std::accumulate(count_neg.begin(), count_neg.end(), 0.0);
//     double dist_good = good / total_pos;
//     double dist_bad = bad / total_neg;
//     woe_values[merge_idx] = std::log(dist_good / dist_bad);
//     iv_values[merge_idx] = (dist_good - dist_bad) * woe_values[merge_idx];
//   }
//   
//   // Total IV
//   total_iv = std::accumulate(iv_values.begin(), iv_values.end(), 0.0);
// }
// 
// // Create bin labels
// void OptimalBinningNumericalLDB::create_bin_labels() {
//   bin_labels.clear();
//   for (size_t b = 0; b < counts.size(); ++b) {
//     std::ostringstream oss;
//     oss << "(";
//     if (bin_edges[b] == -std::numeric_limits<double>::infinity()) {
//       oss << "-Inf";
//     } else {
//       oss << bin_edges[b];
//     }
//     oss << ";";
//     if (bin_edges[b + 1] == std::numeric_limits<double>::infinity()) {
//       oss << "+Inf";
//     } else {
//       oss << bin_edges[b + 1];
//     }
//     oss << "]";
//     bin_labels.push_back(oss.str());
//   }
// }
// 
// // Assign WoE to feature
// void OptimalBinningNumericalLDB::assign_woe() {
//   size_t n = feature.size();
//   woefeature.resize(n);
//   for (size_t i = 0; i < n; ++i) {
//     double x = feature[i];
//     int bin = -1;
//     for (size_t b = 0; b < counts.size(); ++b) {
//       if (x > bin_edges[b] && x <= bin_edges[b + 1]) {
//         bin = b;
//         break;
//       }
//     }
//     if (bin == -1) {
//       Rcpp::stop("Error assigning data point to bin.");
//     }
//     woefeature[i] = woe_values[bin];
//   }
// }
// 
// // Transform function
// Rcpp::List OptimalBinningNumericalLDB::transform() {
//   Rcpp::NumericVector woe_feat(woefeature.begin(), woefeature.end());
//   Rcpp::DataFrame woebin = Rcpp::DataFrame::create(
//     Rcpp::Named("bin") = bin_labels,
//     Rcpp::Named("woe") = woe_values,
//     Rcpp::Named("iv") = iv_values,
//     Rcpp::Named("count") = counts,
//     Rcpp::Named("count_pos") = count_pos,
//     Rcpp::Named("count_neg") = count_neg
//   );
//   return Rcpp::List::create(
//     Rcpp::Named("woefeature") = woe_feat,
//     Rcpp::Named("woebin") = woebin,
//     Rcpp::Named("iv_total") = total_iv
//   );
// }
// 
// 
// //' @title Optimal Binning for Numerical Variables using Local Density Binning (LDB)
// //' 
// //' @description This function implements the Local Density Binning (LDB) algorithm for optimal binning of numerical variables.
// //' 
// //' @param target An integer vector of binary target values (0 or 1).
// //' @param feature A numeric vector of feature values to be binned.
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff Minimum frequency for a bin (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins (default: 20).
// //' 
// //' @return A list containing three elements:
// //' \item{woefeature}{A numeric vector of Weight of Evidence (WoE) transformed feature values.}
// //' \item{woebin}{A data frame containing bin information, including bin labels, WoE, Information Value (IV), and counts.}
// //' \item{iv_total}{The total Information Value of the binned feature.}
// //' 
// //' @details
// //' The Local Density Binning (LDB) algorithm is an advanced method for optimal binning of numerical variables. It aims to create bins that maximize the predictive power of the feature while maintaining monotonicity in the Weight of Evidence (WoE) values and respecting user-defined constraints.
// //' 
// //' The algorithm works through several steps:
// //' 1. Pre-binning: Initially divides the feature into a large number of bins (max_n_prebins) using quantiles.
// //' 2. WoE and IV Calculation: For each bin, computes the Weight of Evidence (WoE) and Information Value (IV):
// //'    \deqn{WoE_i = \ln\left(\frac{P(X_i|Y=1)}{P(X_i|Y=0)}\right) = \ln\left(\frac{n_{1i}/N_1}{n_{0i}/N_0}\right)}
// //'    \deqn{IV_i = (P(X_i|Y=1) - P(X_i|Y=0)) \times WoE_i}
// //'    where \eqn{n_{1i}} and \eqn{n_{0i}} are the number of events and non-events in bin i, and \eqn{N_1} and \eqn{N_0} are the total number of events and non-events.
// //' 3. Monotonicity Enforcement: Merges adjacent bins to ensure monotonic WoE values. The direction of monotonicity is determined by the overall trend of WoE values across bins.
// //' 4. Bin Merging: Merges bins with frequencies below the bin_cutoff threshold and ensures the number of bins is within the specified range (min_bins to max_bins).
// //' 
// //' The LDB method incorporates local density estimation to better capture the underlying distribution of the data. This approach can be particularly effective when dealing with complex, non-linear relationships between the feature and the target variable.
// //' 
// //' The algorithm uses Information Value (IV) as a criterion for merging bins, aiming to minimize IV loss at each step. This approach helps preserve the predictive power of the feature while creating optimal bins.
// //' 
// //' The total Information Value (IV) is calculated as the sum of IVs for all bins:
// //' \deqn{IV_{total} = \sum_{i=1}^{n} IV_i}
// //' 
// //' The LDB method provides a balance between predictive power and model interpretability, allowing users to control the trade-off through parameters such as min_bins, max_bins, and bin_cutoff.
// //' 
// //' @examples
// //' \dontrun{
// //' # Create sample data
// //' set.seed(123)
// //' target <- sample(0:1, 1000, replace = TRUE)
// //' feature <- rnorm(1000)
// //' 
// //' # Run optimal binning
// //' result <- optimal_binning_numerical_ldb(target, feature)
// //' 
// //' # View results
// //' head(result$woefeature)
// //' print(result$woebin)
// //' print(result$iv_total)
// //' }
// //' 
// //' @references
// //' \itemize{
// //' \item Belotti, P., Bonami, P., Fischetti, M., Lodi, A., Monaci, M., Nogales-Gomez, A., & Salvagnin, D. (2016). On handling indicator constraints in mixed integer programming. Computational Optimization and Applications, 65(3), 545-566.
// //' \item Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). Credit Scoring and Its Applications. SIAM Monographs on Mathematical Modeling and Computation.
// //' }
// //' 
// //' @author Lopes, J. E.
// //' 
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_numerical_ldb(Rcpp::IntegerVector target,
//                                          Rcpp::NumericVector feature,
//                                          int min_bins = 3,
//                                          int max_bins = 5,
//                                          double bin_cutoff = 0.05,
//                                          int max_n_prebins = 20) {
//   OptimalBinningNumericalLDB ob(min_bins, max_bins, bin_cutoff, max_n_prebins);
//   ob.fit(Rcpp::as<std::vector<double>>(feature),
//          Rcpp::as<std::vector<int>>(target));
//   return ob.transform();
// }
