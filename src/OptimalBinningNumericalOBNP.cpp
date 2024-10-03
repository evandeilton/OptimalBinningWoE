// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::plugins(openmp)]]

#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cmath>
#include <sstream>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

class OptimalBinningNumericalOBNP {
private:
  std::vector<double> feature;
  std::vector<double> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  bool monotonic_trend; // True if increasing, false if decreasing
  
  std::vector<double> bin_edges;
  std::vector<double> woefeature;
  DataFrame woebin;
  
  const double EPSILON = 1e-10; // Small constant to prevent division by zero
  
public:
  // Constructor
  OptimalBinningNumericalOBNP(const std::vector<double>& feature,
                              const std::vector<double>& target,
                              int min_bins = 3,
                              int max_bins = 5,
                              double bin_cutoff = 0.05,
                              int max_n_prebins = 20)
    : feature(feature),
      target(target),
      min_bins(min_bins),
      max_bins(max_bins),
      bin_cutoff(bin_cutoff),
      max_n_prebins(max_n_prebins)
  {
    // Input validation
    if (min_bins < 2)
      stop("min_bins must be at least 2");
    if (max_bins < min_bins)
      stop("max_bins must be greater than or equal to min_bins");
    if (bin_cutoff < 0 || bin_cutoff > 1)
      stop("bin_cutoff must be between 0 and 1");
    if (max_n_prebins < max_bins)
      stop("max_n_prebins must be greater than or equal to max_bins");
    if (feature.size() != target.size())
      stop("feature and target must have the same length");
    
    // Ensure target is binary
    std::vector<double> unique_targets = target;
    std::sort(unique_targets.begin(), unique_targets.end());
    unique_targets.erase(std::unique(unique_targets.begin(), unique_targets.end()), unique_targets.end());
    if (unique_targets.size() != 2 || unique_targets[0] != 0 || unique_targets[1] != 1)
      stop("target must be binary (0 and 1)");
  }
  
  // Fit method
  void fit() {
    // Step 1: Pre-binning
    prebin();
    
    // Step 2: Determine monotonic trend
    determine_monotonic_trend();
    
    // Step 3: Optimization
    optimize_bins();
    
    // Step 4: Compute WoE and IV
    compute_woe_iv();
  }
  
  // Get methods
  std::vector<double> get_woefeature() {
    return woefeature;
  }
  
  DataFrame get_woebin() {
    return woebin;
  }
  
private:
  void prebin() {
    // Pre-bin the feature into max_n_prebins using quantiles
    std::vector<double> sorted_feature = feature;
    std::sort(sorted_feature.begin(), sorted_feature.end());
    
    bin_edges.clear();
    bin_edges.push_back(-std::numeric_limits<double>::infinity());
    
    size_t n = sorted_feature.size();
    std::set<double> unique_edges;
    
    for (int i = 1; i < max_n_prebins; ++i) {
      size_t index = i * n / max_n_prebins;
      if (index >= n) index = n - 1;
      double edge_value = sorted_feature[index];
      unique_edges.insert(edge_value);
    }
    
    for (double edge : unique_edges) {
      if (edge > bin_edges.back()) {
        bin_edges.push_back(edge);
      }
    }
    
    bin_edges.push_back(std::numeric_limits<double>::infinity());
    
    // Adjust min_bins if necessary
    size_t unique_values = unique_edges.size();
    if (unique_values < static_cast<size_t>(min_bins)) {
      min_bins = static_cast<int>(unique_values);
    }
  }
  
  void determine_monotonic_trend() {
    // Calculate the correlation between feature and target
    double mean_x = std::accumulate(feature.begin(), feature.end(), 0.0) / feature.size();
    double mean_y = std::accumulate(target.begin(), target.end(), 0.0) / target.size();
    
    double numerator = 0.0, denominator_x = 0.0, denominator_y = 0.0;
    for (size_t i = 0; i < feature.size(); ++i) {
      double dx = feature[i] - mean_x;
      double dy = target[i] - mean_y;
      numerator += dx * dy;
      denominator_x += dx * dx;
      denominator_y += dy * dy;
    }
    
    if (denominator_x > 0 && denominator_y > 0) {
      double corr = numerator / std::sqrt(denominator_x * denominator_y);
      monotonic_trend = (corr >= 0);
    } else {
      monotonic_trend = true; // Default to increasing trend if variance is zero
    }
  }
  
  void optimize_bins() {
    // Implement the optimization logic to combine pre-bins into optimal bins
    // respecting the monotonicity constraints and other parameters
    
    std::vector<double> optimized_bin_edges = bin_edges;
    
    // Merge bins to meet max_bins constraint while enforcing monotonicity
    while ((optimized_bin_edges.size() - 1) > max_bins) {
      // Merge bins while considering monotonicity
      merge_bins(optimized_bin_edges);
    }
    
    // Apply bin_cutoff to merge bins with low frequency
    apply_bin_cutoff(optimized_bin_edges);
    
    // Ensure that number of bins is at least min_bins
    while ((optimized_bin_edges.size() - 1) < min_bins) {
      // Merge bins to increase the number of observations per bin
      merge_bins(optimized_bin_edges);
    }
    
    bin_edges = optimized_bin_edges;
  }
  
  void merge_bins(std::vector<double>& edges) {
    // Merge adjacent bins to enforce monotonicity and respect constraints
    size_t n_bins = edges.size() - 1;
    std::vector<size_t> bin_assignments = assign_bins(feature, edges);
    std::vector<double> bin_target_rate(n_bins);
    std::vector<double> bin_counts(n_bins);
    
    // Compute target rate and counts per bin
    std::vector<double> bin_pos_counts(n_bins, 0.0);
    
    for (size_t i = 0; i < bin_assignments.size(); ++i) {
      size_t bin_idx = bin_assignments[i];
      bin_counts[bin_idx] += 1.0;
      bin_pos_counts[bin_idx] += target[i];
    }
    
    for (size_t i = 0; i < n_bins; ++i) {
      bin_target_rate[i] = bin_pos_counts[i] / bin_counts[i];
    }
    
    // Identify bins to merge based on monotonicity
    size_t merge_idx = n_bins; // Initialize with an invalid index
    for (size_t i = 0; i < n_bins - 1; ++i) {
      bool condition = monotonic_trend
      ? (bin_target_rate[i] > bin_target_rate[i + 1])
        : (bin_target_rate[i] < bin_target_rate[i + 1]);
      if (condition) {
        merge_idx = i;
        break;
      }
    }
    
    // If no bins violate monotonicity, merge bins with smallest difference in target rate
    if (merge_idx == n_bins) {
      double min_diff = std::numeric_limits<double>::infinity();
      for (size_t i = 0; i < n_bins - 1; ++i) {
        double diff = std::abs(bin_target_rate[i + 1] - bin_target_rate[i]);
        if (diff < min_diff) {
          min_diff = diff;
          merge_idx = i;
        }
      }
    }
    
    // Merge bins at merge_idx and merge_idx + 1
    if (merge_idx < n_bins) {
      edges.erase(edges.begin() + merge_idx + 1);
    }
  }
  
  void apply_bin_cutoff(std::vector<double>& edges) {
    // Merge bins where the proportion of observations is less than bin_cutoff
    bool bins_merged = true;
    while (bins_merged) {
      bins_merged = false;
      size_t n_bins = edges.size() - 1;
      std::vector<size_t> bin_assignments = assign_bins(feature, edges);
      std::vector<double> bin_counts(n_bins, 0.0);
      
      // Compute counts per bin
      for (size_t i = 0; i < bin_assignments.size(); ++i) {
        bin_counts[bin_assignments[i]] += 1.0;
      }
      
      double total_count = feature.size();
      
      for (size_t i = 0; i < n_bins; ++i) {
        if (bin_counts[i] / total_count < bin_cutoff) {
          // Merge with adjacent bin while preserving monotonicity
          if (i == 0) {
            // Merge with next bin
            edges.erase(edges.begin() + 1);
          } else {
            // Merge with previous bin
            edges.erase(edges.begin() + i);
          }
          bins_merged = true;
          break;
        }
      }
    }
  }
  
  std::vector<size_t> assign_bins(const std::vector<double>& x, const std::vector<double>& edges) {
    // Assign observations to bins using std::upper_bound
    std::vector<size_t> bin_assignments(x.size());
#pragma omp parallel for
    for (size_t i = 0; i < x.size(); ++i) {
      double xi = x[i];
      size_t bin_index = std::upper_bound(edges.begin(), edges.end(), xi) - edges.begin() - 1;
      if (bin_index >= edges.size() - 1) {
        bin_index = edges.size() - 2; // Assign to last bin if xi >= last edge
      }
      bin_assignments[i] = bin_index;
    }
    return bin_assignments;
  }
  
  void compute_woe_iv() {
    size_t n_bins = bin_edges.size() - 1;
    std::vector<size_t> bin_assignments = assign_bins(feature, bin_edges);
    std::vector<double> bin_counts(n_bins, 0.0);
    std::vector<double> bin_pos_counts(n_bins, 0.0);
    std::vector<double> bin_neg_counts(n_bins, 0.0);
    std::vector<double> bin_woe(n_bins, 0.0);
    std::vector<double> bin_iv(n_bins, 0.0);
    
    double total_pos = 0.0;
    double total_neg = 0.0;
    
    // Process all observations
    for (size_t i = 0; i < bin_assignments.size(); ++i) {
      size_t bin_idx = bin_assignments[i];
      bin_counts[bin_idx] += 1.0;
      if (target[i] == 1.0) {
        bin_pos_counts[bin_idx] += 1.0;
        total_pos += 1.0;
      } else {
        bin_neg_counts[bin_idx] += 1.0;
        total_neg += 1.0;
      }
    }
    
    // Handle cases where total_pos or total_neg is zero
    if (total_pos == 0) total_pos = EPSILON;
    if (total_neg == 0) total_neg = EPSILON;
    
    // Compute WoE and IV per bin
    for (size_t i = 0; i < n_bins; ++i) {
      // Handle cases where bin_pos_counts[i] or bin_neg_counts[i] is zero
      double dist_pos = bin_pos_counts[i] / total_pos;
      double dist_neg = bin_neg_counts[i] / total_neg;
      
      if (dist_pos == 0) dist_pos = EPSILON;
      if (dist_neg == 0) dist_neg = EPSILON;
      
      bin_woe[i] = std::log(dist_pos / dist_neg);
      bin_iv[i] = (dist_pos - dist_neg) * bin_woe[i];
    }
    
    // Assign WoE values to features
    woefeature.resize(bin_assignments.size());
#pragma omp parallel for
    for (size_t i = 0; i < bin_assignments.size(); ++i) {
      size_t bin_idx = bin_assignments[i];
      woefeature[i] = bin_woe[bin_idx];
    }
    
    // Prepare woebin DataFrame
    std::vector<std::string> bin_intervals(n_bins);
    for (size_t i = 0; i < n_bins; ++i) {
      std::ostringstream oss;
      oss << "(" << bin_edges[i] << ";" << bin_edges[i + 1] << "]";
      bin_intervals[i] = oss.str();
    }
    
    woebin = DataFrame::create(
      Named("bin") = bin_intervals,
      Named("woe") = bin_woe,
      Named("iv") = bin_iv,
      Named("count") = bin_counts,
      Named("count_pos") = bin_pos_counts,
      Named("count_neg") = bin_neg_counts
    );
  }
};


//' @title Optimal Binning for Numerical Variables using OBNP
//'
//' @description
//' This function performs optimal binning for numerical variables using the Optimal Binning for Non-Parametric Transformations (OBNP) approach. It creates optimal bins for a numerical feature based on its relationship with a binary target variable, maximizing the predictive power while respecting user-defined constraints and enforcing monotonicity.
//'
//' @param target A numeric vector of binary target values (0 or 1).
//' @param feature A numeric vector of feature values.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum proportion of total observations for a bin to avoid being merged (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins before the optimization process (default: 20).
//'
//' @return A list containing two elements:
//' \item{woefeature}{A numeric vector of Weight of Evidence (WoE) values for each observation.}
//' \item{woebin}{A data frame with the following columns:
//'   \itemize{
//'     \item bin: Character vector of bin ranges.
//'     \item woe: Numeric vector of WoE values for each bin.
//'     \item iv: Numeric vector of Information Value (IV) for each bin.
//'     \item count: Numeric vector of total observations in each bin.
//'     \item count_pos: Numeric vector of positive target observations in each bin.
//'     \item count_neg: Numeric vector of negative target observations in each bin.
//'   }
//' }
//'
//' @details
//' The Optimal Binning for Non-Parametric Transformations (OBNP) algorithm for numerical variables works as follows:
//' 1. Pre-bin the feature into max_n_prebins using quantiles.
//' 2. Determine the monotonic trend (increasing or decreasing) based on the correlation between the feature and target.
//' 3. Optimize bins by merging pre-bins, respecting the monotonicity constraint and other parameters.
//' 4. Apply the bin_cutoff to merge bins with low frequency.
//' 5. Ensure the number of bins is between min_bins and max_bins.
//' 6. Compute Weight of Evidence (WoE) and Information Value (IV) for each bin.
//'
//' The algorithm aims to create bins that maximize the predictive power of the numerical variable while adhering to the specified constraints. It enforces monotonicity of WoE values, which is particularly useful for credit scoring and risk modeling applications.
//'
//' Weight of Evidence (WoE) is calculated as:
//' \deqn{WoE = \ln(\frac{\text{Distribution of Positives}}{\text{Distribution of Negatives}})}
//'
//' Information Value (IV) is calculated as:
//' \deqn{IV = (\text{Distribution of Positives} - \text{Distribution of Negatives}) \times WoE}
//'
//' This implementation uses OpenMP for parallel processing when available, which can significantly speed up the computation for large datasets.
//'
//' @references
//' \itemize{
//'   \item Mironchyk, P., & Tchistiakov, V. (2017). Monotone optimal binning algorithm for credit risk modeling. SSRN Electronic Journal. doi:10.2139/ssrn.2978774
//'   \item Bellotti, T., & Crook, J. (2012). Loss given default models incorporating macroeconomic variables for credit cards. International Journal of Forecasting, 28(1), 171-182.
//' }
//'
//' @examples
//' \dontrun{
//' # Create sample data
//' set.seed(123)
//' n <- 1000
//' target <- sample(0:1, n, replace = TRUE)
//' feature <- rnorm(n)
//'
//' # Run optimal binning
//' result <- optimal_binning_numerical_obnp(target, feature, min_bins = 2, max_bins = 4)
//'
//' # Print results
//' print(result$woebin)
//'
//' # Plot WoE values
//' plot(result$woebin$woe, type = "s", xaxt = "n", xlab = "Bins", ylab = "WoE",
//'      main = "Weight of Evidence by Bin")
//' axis(1, at = 1:nrow(result$woebin), labels = result$woebin$bin, las = 2)
//' }
//'
//' @export
// [[Rcpp::export]]
List optimal_binning_numerical_obnp(NumericVector target,
                                   NumericVector feature,
                                   int min_bins = 3,
                                   int max_bins = 5,
                                   double bin_cutoff = 0.05,
                                   int max_n_prebins = 20)
{
 std::vector<double> feature_vec(feature.begin(), feature.end());
 std::vector<double> target_vec(target.begin(), target.end());
 
 OptimalBinningNumericalOBNP obnp(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins);
 obnp.fit();
 
 return List::create(
   Named("woefeature") = obnp.get_woefeature(),
   Named("woebin") = obnp.get_woebin()
 );
}



// // [[Rcpp::plugins(cpp11)]]
// // [[Rcpp::plugins(openmp)]]
// 
// #include <Rcpp.h>
// #ifdef _OPENMP
// #include <omp.h>
// #endif
// 
// #include <vector>
// #include <algorithm>
// #include <numeric>
// #include <limits>
// #include <cmath>
// 
// using namespace Rcpp;
// 
// class OptimalBinningNumericalOBNP {
// private:
//   std::vector<double> feature;
//   std::vector<double> target;
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
//   bool monotonic_trend; // True if increasing, false if decreasing
//   
//   std::vector<double> bin_edges;
//   std::vector<double> woefeature;
//   DataFrame woebin;
//   
// public:
//   // Constructor
//   OptimalBinningNumericalOBNP(const std::vector<double>& feature,
//                               const std::vector<double>& target,
//                               int min_bins = 3,
//                               int max_bins = 5,
//                               double bin_cutoff = 0.05,
//                               int max_n_prebins = 20)
//     : feature(feature),
//       target(target),
//       min_bins(min_bins),
//       max_bins(max_bins),
//       bin_cutoff(bin_cutoff),
//       max_n_prebins(max_n_prebins)
//   {
//     // Input validation
//     if (min_bins < 2)
//       stop("min_bins must be at least 2");
//     if (max_bins < min_bins)
//       stop("max_bins must be greater than or equal to min_bins");
//     if (bin_cutoff < 0 || bin_cutoff > 1)
//       stop("bin_cutoff must be between 0 and 1");
//     if (max_n_prebins < max_bins)
//       stop("max_n_prebins must be greater than or equal to max_bins");
//     if (feature.size() != target.size())
//       stop("feature and target must have the same length");
//     
//     // Ensure target is binary
//     std::vector<double> unique_targets = target;
//     std::sort(unique_targets.begin(), unique_targets.end());
//     unique_targets.erase(std::unique(unique_targets.begin(), unique_targets.end()), unique_targets.end());
//     if (unique_targets.size() != 2 || unique_targets[0] != 0 || unique_targets[1] != 1)
//       stop("target must be binary (0 and 1)");
//   }
//   
//   // Fit method
//   void fit() {
//     // Step 1: Pre-binning
//     prebin();
//     
//     // Step 2: Determine monotonic trend
//     determine_monotonic_trend();
//     
//     // Step 3: Optimization
//     optimize_bins();
//     
//     // Step 4: Compute WoE and IV
//     compute_woe_iv();
//   }
//   
//   // Get methods
//   std::vector<double> get_woefeature() {
//     return woefeature;
//   }
//   
//   DataFrame get_woebin() {
//     return woebin;
//   }
//   
// private:
//   void prebin() {
//     // Pre-bin the feature into max_n_prebins using quantiles
//     std::vector<double> sorted_feature = feature;
//     std::sort(sorted_feature.begin(), sorted_feature.end());
//     
//     bin_edges.clear();
//     bin_edges.push_back(-std::numeric_limits<double>::infinity());
//     
//     for (int i = 1; i < max_n_prebins; ++i) {
//       size_t index = i * sorted_feature.size() / max_n_prebins;
//       if (index > 0 && index < sorted_feature.size() && sorted_feature[index] > bin_edges.back()) {
//         bin_edges.push_back(sorted_feature[index]);
//       }
//     }
//     
//     bin_edges.push_back(std::numeric_limits<double>::infinity());
//   }
//   
//   void determine_monotonic_trend() {
//     // Calculate the correlation between feature and target
//     double mean_x = std::accumulate(feature.begin(), feature.end(), 0.0) / feature.size();
//     double mean_y = std::accumulate(target.begin(), target.end(), 0.0) / target.size();
//     
//     double numerator = 0.0, denominator_x = 0.0, denominator_y = 0.0;
//     for (size_t i = 0; i < feature.size(); ++i) {
//       double dx = feature[i] - mean_x;
//       double dy = target[i] - mean_y;
//       numerator += dx * dy;
//       denominator_x += dx * dx;
//       denominator_y += dy * dy;
//     }
//     
//     double corr = numerator / std::sqrt(denominator_x * denominator_y);
//     monotonic_trend = (corr >= 0);
//   }
//   
//   void optimize_bins() {
//     // Implement the optimization logic to combine pre-bins into optimal bins
//     // respecting the monotonicity constraints and other parameters
//     
//     // For simplicity, we'll use a heuristic approach here
//     // In practice, this step would involve solving an optimization problem
//     
//     // Initialize bins
//     std::vector<double> optimized_bin_edges = bin_edges;
//     
//     // Merge bins to meet min_bins and max_bins constraints
//     while ((optimized_bin_edges.size() - 1) > max_bins) {
//       // Merge bins with the smallest difference in target rate
//       merge_bins(optimized_bin_edges);
//     }
//     
//     // Apply bin_cutoff to merge bins with low frequency
//     apply_bin_cutoff(optimized_bin_edges);
//     
//     // Ensure that number of bins is at least min_bins
//     while ((optimized_bin_edges.size() - 1) < min_bins) {
//       // Merge bins to increase the number of observations per bin
//       merge_bins(optimized_bin_edges);
//     }
//     
//     bin_edges = optimized_bin_edges;
//   }
//   
//   void merge_bins(std::vector<double>& edges) {
//     // Merge adjacent bins with the smallest difference in target rate
//     size_t n_bins = edges.size() - 1;
//     std::vector<double> bin_assignments = assign_bins(feature, edges);
//     std::vector<double> bin_target_rate(n_bins);
//     std::vector<double> bin_counts(n_bins);
//     
//     // Compute target rate and counts per bin
//     for (size_t i = 0; i < n_bins; ++i) {
//       double sum = 0.0;
//       double count = 0.0;
//       for (size_t j = 0; j < bin_assignments.size(); ++j) {
//         if (bin_assignments[j] == i) {
//           sum += target[j];
//           count += 1.0;
//         }
//       }
//       bin_counts[i] = count;
//       bin_target_rate[i] = sum / count;
//     }
//     
//     // Find pair of bins with the smallest difference in target rate
//     double min_diff = std::numeric_limits<double>::infinity();
//     size_t min_idx = 0;
//     for (size_t i = 0; i < n_bins - 1; ++i) {
//       double diff = std::abs(bin_target_rate[i+1] - bin_target_rate[i]);
//       if (diff < min_diff) {
//         min_diff = diff;
//         min_idx = i;
//       }
//     }
//     
//     // Merge bins at min_idx and min_idx + 1
//     edges.erase(edges.begin() + min_idx + 1);
//   }
//   
//   void apply_bin_cutoff(std::vector<double>& edges) {
//     // Merge bins where the proportion of observations is less than bin_cutoff
//     bool bins_merged = true;
//     while (bins_merged) {
//       bins_merged = false;
//       size_t n_bins = edges.size() - 1;
//       std::vector<double> bin_assignments = assign_bins(feature, edges);
//       std::vector<double> bin_counts(n_bins, 0.0);
//       
//       // Compute counts per bin
//       for (double assignment : bin_assignments) {
//         bin_counts[static_cast<size_t>(assignment)] += 1.0;
//       }
//       
//       double total_count = feature.size();
//       
//       for (size_t i = 0; i < n_bins; ++i) {
//         if (bin_counts[i] / total_count < bin_cutoff) {
//           // Merge with adjacent bin
//           if (i == 0) {
//             // Merge with next bin
//             edges.erase(edges.begin() + 1);
//           } else {
//             // Merge with previous bin
//             edges.erase(edges.begin() + i);
//           }
//           bins_merged = true;
//           break;
//         }
//       }
//     }
//   }
//   
//   std::vector<double> assign_bins(const std::vector<double>& x, const std::vector<double>& edges) {
//     // Assign observations to bins
//     std::vector<double> bin_assignments(x.size());
// #pragma omp parallel for
//     for (size_t i = 0; i < x.size(); ++i) {
//       double xi = x[i];
//       for (size_t j = 0; j < edges.size() - 1; ++j) {
//         if (xi > edges[j] && xi <= edges[j+1]) {
//           bin_assignments[i] = j;
//           break;
//         }
//       }
//     }
//     return bin_assignments;
//   }
//   
//   void compute_woe_iv() {
//     size_t n_bins = bin_edges.size() - 1;
//     std::vector<double> bin_assignments = assign_bins(feature, bin_edges);
//     std::vector<double> bin_counts(n_bins, 0.0);
//     std::vector<double> bin_pos_counts(n_bins, 0.0);
//     std::vector<double> bin_neg_counts(n_bins, 0.0);
//     std::vector<double> bin_woe(n_bins, 0.0);
//     std::vector<double> bin_iv(n_bins, 0.0);
//     
//     double total_pos = std::accumulate(target.begin(), target.end(), 0.0);
//     double total_neg = target.size() - total_pos;
//     
//     // Compute counts and WoE per bin
// #pragma omp parallel for
//     for (size_t i = 0; i < n_bins; ++i) {
//       for (size_t j = 0; j < bin_assignments.size(); ++j) {
//         if (bin_assignments[j] == i) {
//           bin_counts[i] += 1.0;
//           bin_pos_counts[i] += target[j];
//         }
//       }
//       bin_neg_counts[i] = bin_counts[i] - bin_pos_counts[i];
//       
//       double dist_pos = bin_pos_counts[i] / total_pos;
//       double dist_neg = bin_neg_counts[i] / total_neg;
//       
//       if (dist_pos == 0)
//         dist_pos = 0.0001; // Avoid division by zero
//       if (dist_neg == 0)
//         dist_neg = 0.0001;
//       
//       bin_woe[i] = std::log(dist_pos / dist_neg);
//       bin_iv[i] = (dist_pos - dist_neg) * bin_woe[i];
//     }
//     
//     woefeature.resize(bin_assignments.size());
//     for (size_t i = 0; i < bin_assignments.size(); ++i) {
//       woefeature[i] = bin_woe[static_cast<size_t>(bin_assignments[i])];
//     }
//     
//     // Prepare woebin DataFrame
//     std::vector<std::string> bin_intervals(n_bins);
//     for (size_t i = 0; i < n_bins; ++i) {
//       std::ostringstream oss;
//       oss << "(" << bin_edges[i] << ";" << bin_edges[i+1] << "]";
//       bin_intervals[i] = oss.str();
//     }
//     
//     woebin = DataFrame::create(
//       Named("bin") = bin_intervals,
//       Named("woe") = bin_woe,
//       Named("iv") = bin_iv,
//       Named("count") = bin_counts,
//       Named("count_pos") = bin_pos_counts,
//       Named("count_neg") = bin_neg_counts
//     );
//   }
// };
// 
// 
// //' @title Optimal Binning for Numerical Variables using OBNP
// //'
// //' @description
// //' This function performs optimal binning for numerical variables using the Optimal Binning for Non-Parametric Transformations (OBNP) approach. It creates optimal bins for a numerical feature based on its relationship with a binary target variable, maximizing the predictive power while respecting user-defined constraints and enforcing monotonicity.
// //'
// //' @param target A numeric vector of binary target values (0 or 1).
// //' @param feature A numeric vector of feature values.
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff Minimum proportion of total observations for a bin to avoid being merged (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins before the optimization process (default: 20).
// //'
// //' @return A list containing two elements:
// //' \item{woefeature}{A numeric vector of Weight of Evidence (WoE) values for each observation.}
// //' \item{woebin}{A data frame with the following columns:
// //'   \itemize{
// //'     \item bin: Character vector of bin ranges.
// //'     \item woe: Numeric vector of WoE values for each bin.
// //'     \item iv: Numeric vector of Information Value (IV) for each bin.
// //'     \item count: Numeric vector of total observations in each bin.
// //'     \item count_pos: Numeric vector of positive target observations in each bin.
// //'     \item count_neg: Numeric vector of negative target observations in each bin.
// //'   }
// //' }
// //'
// //' @details
// //' The Optimal Binning for Non-Parametric Transformations (OBNP) algorithm for numerical variables works as follows:
// //' 1. Pre-bin the feature into max_n_prebins using quantiles.
// //' 2. Determine the monotonic trend (increasing or decreasing) based on the correlation between the feature and target.
// //' 3. Optimize bins by merging pre-bins, respecting the monotonicity constraint and other parameters.
// //' 4. Apply the bin_cutoff to merge bins with low frequency.
// //' 5. Ensure the number of bins is between min_bins and max_bins.
// //' 6. Compute Weight of Evidence (WoE) and Information Value (IV) for each bin.
// //'
// //' The algorithm aims to create bins that maximize the predictive power of the numerical variable while adhering to the specified constraints. It enforces monotonicity of WoE values, which is particularly useful for credit scoring and risk modeling applications.
// //'
// //' Weight of Evidence (WoE) is calculated as:
// //' \deqn{WoE = \ln(\frac{\text{Positive Rate}}{\text{Negative Rate}})}
// //'
// //' Information Value (IV) is calculated as:
// //' \deqn{IV = (\text{Positive Rate} - \text{Negative Rate}) \times WoE}
// //'
// //' This implementation uses OpenMP for parallel processing when available, which can significantly speed up the computation for large datasets.
// //'
// //' @references
// //' \itemize{
// //'   \item Mironchyk, P., & Tchistiakov, V. (2017). Monotone optimal binning algorithm for credit risk modeling. SSRN Electronic Journal. doi:10.2139/ssrn.2978774
// //'   \item Bellotti, T., & Crook, J. (2012). Loss given default models incorporating macroeconomic variables for credit cards. International Journal of Forecasting, 28(1), 171-182.
// //' }
// //'
// //' @examples
// //' \dontrun{
// //' # Create sample data
// //' set.seed(123)
// //' n <- 1000
// //' target <- sample(0:1, n, replace = TRUE)
// //' feature <- rnorm(n)
// //'
// //' # Run optimal binning
// //' result <- optimal_binning_numerical_obnp(target, feature, min_bins = 2, max_bins = 4)
// //'
// //' # Print results
// //' print(result$woebin)
// //'
// //' # Plot WoE values
// //' plot(result$woebin$woe, type = "s", xaxt = "n", xlab = "Bins", ylab = "WoE",
// //'      main = "Weight of Evidence by Bin")
// //' axis(1, at = 1:nrow(result$woebin), labels = result$woebin$bin, las = 2)
// //' }
// //'
// //' @author Lopes, J. E.
// //'
// //' @export
// // [[Rcpp::export]]
// List optimal_binning_numerical_obnp(NumericVector target,
//                                     NumericVector feature,
//                                     int min_bins = 3,
//                                     int max_bins = 5,
//                                     double bin_cutoff = 0.05,
//                                     int max_n_prebins = 20)
// {
//   std::vector<double> feature_vec(feature.begin(), feature.end());
//   std::vector<double> target_vec(target.begin(), target.end());
//   
//   OptimalBinningNumericalOBNP obnp(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins);
//   obnp.fit();
//   
//   return List::create(
//     Named("woefeature") = obnp.get_woefeature(),
//     Named("woebin") = obnp.get_woebin()
//   );
// }
