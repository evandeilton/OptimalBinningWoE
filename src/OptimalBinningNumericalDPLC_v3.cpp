// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>

using namespace Rcpp;

// Class for Optimal Binning using Dynamic Programming with Local Constraints
class OptimalBinningNumericalDPLC {
private:
  const std::vector<double>& feature;
  const std::vector<unsigned int>& target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  double convergence_threshold;
  int max_iterations;
  
  bool converged;
  int iterations_run;
  
  std::vector<double> bin_edges;
  std::vector<double> woe_values;
  std::vector<double> iv_values;
  std::vector<std::string> bin_labels;
  std::vector<double> count_pos;
  std::vector<double> count_neg;
  std::vector<double> counts;
  
  double total_pos;
  double total_neg;
  
  static constexpr double EPSILON = 1e-10;
  
public:
  OptimalBinningNumericalDPLC(const std::vector<double>& feature,
                              const std::vector<unsigned int>& target,
                              int min_bins,
                              int max_bins,
                              double bin_cutoff,
                              int max_n_prebins,
                              double convergence_threshold,
                              int max_iterations)
    : feature(feature),
      target(target),
      min_bins(min_bins),
      max_bins(max_bins),
      bin_cutoff(bin_cutoff),
      max_n_prebins(max_n_prebins),
      convergence_threshold(convergence_threshold),
      max_iterations(max_iterations),
      converged(true),
      iterations_run(0) {
    
    double sum_target = std::accumulate(target.begin(), target.end(), 0.0);
    total_pos = sum_target;
    total_neg = static_cast<double>(target.size()) - total_pos;
    
    // Additional checks to prevent unexpected behavior
    if (max_bins < min_bins) {
      Rcpp::stop("max_bins must be >= min_bins");
    }
    if (max_n_prebins < 1) {
      max_n_prebins = 1;
    }
    if (bin_cutoff < 0.0) {
      bin_cutoff = 0.0;
    }
  }
  
  void fit() {
    // Adjust min_bins if necessary
    if (min_bins < 2) {
      min_bins = 2; // ensure minimum
    }
    if (min_bins > max_bins) {
      min_bins = max_bins;
    }
    
    // Check the number of unique feature values
    std::vector<double> unique_feature_values = feature;
    std::sort(unique_feature_values.begin(), unique_feature_values.end());
    unique_feature_values.erase(std::unique(unique_feature_values.begin(), unique_feature_values.end()), unique_feature_values.end());
    int num_unique_values = static_cast<int>(unique_feature_values.size());
    
    // If <=2 unique values, just create trivial bins without further optimization
    if (num_unique_values <= 2) {
      bin_edges.clear();
      bin_edges.reserve(static_cast<size_t>(num_unique_values + 1));
      bin_edges.push_back(-std::numeric_limits<double>::infinity());
      if (num_unique_values == 2) {
        double midpoint = (unique_feature_values[0] + unique_feature_values[1]) / 2.0;
        bin_edges.push_back(midpoint);
      }
      bin_edges.push_back(std::numeric_limits<double>::infinity());
      calculate_counts_woe();
      calculate_iv();
      converged = true;
      iterations_run = 0;
      return;
    }
    
    // Pre-binning
    prebinning();
    
    // Calculate initial counts and WOE
    calculate_counts_woe();
    
    // If total_pos or total_neg is zero, stop to avoid instability
    if (total_pos < EPSILON || total_neg < EPSILON) {
      // Already computed counts and WOE, just finalize IV
      calculate_iv();
      converged = true;
      return;
    }
    
    // Enforce monotonicity
    enforce_monotonicity();
    
    // Ensure bin constraints and handle rare bins
    ensure_bin_constraints();
    
    // Final IV calculation
    calculate_iv();
  }
  
  List get_results() const {
    // Exclude -Inf and +Inf from cutpoints
    std::vector<double> cutpoints;
    if (bin_edges.size() > 2) {
      cutpoints.reserve(bin_edges.size() - 2);
      for (size_t i = 1; i < bin_edges.size() - 1; ++i) {
        cutpoints.push_back(bin_edges[i]);
      }
    }
    
    return List::create(
      Named("bin") = bin_labels,
      Named("woe") = woe_values,
      Named("iv") = iv_values,
      Named("count") = counts,
      Named("count_pos") = count_pos,
      Named("count_neg") = count_neg,
      Named("cutpoints") = cutpoints,
      Named("converged") = converged,
      Named("iterations") = iterations_run
    );
  }
  
private:
  void prebinning() {
    // Initial pre-binning based on approximate quantiles
    std::vector<size_t> sorted_indices(feature.size());
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [this](size_t i1, size_t i2) { return feature[i1] < feature[i2]; });
    
    std::vector<double> sorted_feature(feature.size());
    for (size_t i = 0; i < feature.size(); ++i) {
      sorted_feature[i] = feature[sorted_indices[i]];
    }
    
    int n = static_cast<int>(feature.size());
    int bin_size = std::max(1, n / std::max(1, max_n_prebins));
    
    std::vector<double> edges;
    edges.reserve(static_cast<size_t>(std::max(1, max_n_prebins - 1)));
    
    for (int i = 1; i < max_n_prebins; ++i) {
      int idx = i * bin_size;
      if (idx < n) {
        edges.push_back(sorted_feature[static_cast<size_t>(idx)]);
      }
    }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    
    bin_edges.clear();
    bin_edges.reserve(edges.size() + 2);
    bin_edges.push_back(-std::numeric_limits<double>::infinity());
    for (size_t i = 0; i < edges.size(); ++i) {
      bin_edges.push_back(edges[i]);
    }
    bin_edges.push_back(std::numeric_limits<double>::infinity());
  }
  
  void calculate_counts_woe() {
    int num_bins = static_cast<int>(bin_edges.size()) - 1;
    count_pos.assign(static_cast<size_t>(num_bins), 0.0);
    count_neg.assign(static_cast<size_t>(num_bins), 0.0);
    counts.assign(static_cast<size_t>(num_bins), 0.0);
    
    for (size_t i = 0; i < feature.size(); ++i) {
      int bin_idx = find_bin(feature[i]);
      if (bin_idx < 0 || bin_idx >= num_bins) {
        // Safety check, should not happen
        continue;
      }
      counts[static_cast<size_t>(bin_idx)] += 1.0;
      if (target[i] == 1U) {
        count_pos[static_cast<size_t>(bin_idx)] += 1.0;
      } else {
        count_neg[static_cast<size_t>(bin_idx)] += 1.0;
      }
    }
    
    calculate_woe();
    update_bin_labels();
  }
  
  void calculate_woe() {
    if (total_pos < EPSILON || total_neg < EPSILON) {
      // Avoid division by zero
      // If this happens, WOE might be unstable, just set small corrections
      total_pos = std::max(total_pos, EPSILON);
      total_neg = std::max(total_neg, EPSILON);
    }
    
    int num_bins = static_cast<int>(counts.size());
    woe_values.resize(static_cast<size_t>(num_bins));
    
    for (int i = 0; i < num_bins; ++i) {
      double rate_pos = (count_pos[static_cast<size_t>(i)] > 0.0) ? (count_pos[static_cast<size_t>(i)] / total_pos) : (EPSILON / total_pos);
      double rate_neg = (count_neg[static_cast<size_t>(i)] > 0.0) ? (count_neg[static_cast<size_t>(i)] / total_neg) : (EPSILON / total_neg);
      
      // Ensure rates are not zero
      rate_pos = std::max(rate_pos, EPSILON);
      rate_neg = std::max(rate_neg, EPSILON);
      
      woe_values[static_cast<size_t>(i)] = std::log(rate_pos / rate_neg);
    }
  }
  
  void update_bin_labels() {
    bin_labels.clear();
    bin_labels.reserve(bin_edges.size() - 1);
    for (size_t i = 0; i < bin_edges.size() - 1; ++i) {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(6);
      double left = bin_edges[i];
      double right = bin_edges[i+1];
      if (std::isinf(left) && left < 0) {
        oss << "(-Inf;" << right << "]";
      } else if (std::isinf(right) && right > 0) {
        oss << "(" << left << ";+Inf]";
      } else {
        oss << "(" << left << ";" << right << "]";
      }
      bin_labels.push_back(oss.str());
    }
  }
  
  int find_bin(double value) const {
    // Locate bin via upper_bound
    auto it = std::upper_bound(bin_edges.begin(), bin_edges.end(), value);
    int bin_idx = static_cast<int>(std::distance(bin_edges.begin(), it)) - 1;
    // Ensure bin_idx is within valid range
    if (bin_idx < 0) bin_idx = 0;
    if (bin_idx >= static_cast<int>(counts.size())) bin_idx = static_cast<int>(counts.size()) - 1;
    return bin_idx;
  }
  
  void enforce_monotonicity() {
    if (counts.size() <= 2) {
      // If already small number of bins, no monotonic enforcement needed
      return;
    }
    
    // Determine monotonic direction from first two WOE values
    bool increasing = true;
    if (woe_values.size() >= 2) {
      increasing = (woe_values[1] >= woe_values[0]);
    }
    
    bool is_monotonic = false;
    int iterations = 0;
    
    // Enforce monotonicity by merging bins that break monotonic order
    while (!is_monotonic && counts.size() > static_cast<size_t>(min_bins) && iterations < max_iterations) {
      is_monotonic = true;
      for (size_t i = 1; i < woe_values.size(); ++i) {
        if ((increasing && (woe_values[i] < woe_values[i - 1])) ||
            (!increasing && (woe_values[i] > woe_values[i - 1]))) {
          // Merge bin i-1 and i
          merge_bins(static_cast<int>(i - 1));
          is_monotonic = false;
          break;
        }
      }
      iterations++;
      if (counts.size() == static_cast<size_t>(min_bins)) {
        // Reached minimum number of bins, stop
        break;
      }
    }
    
    iterations_run += iterations;
    if (iterations >= max_iterations) {
      converged = false;
    }
  }
  
  void merge_bins(int idx) {
    // Safety checks on idx
    if (idx < 0 || idx >= static_cast<int>(counts.size()) - 1) {
      // Invalid index for merging
      return;
    }
    
    // Merge bin idx with idx+1
    bin_edges.erase(bin_edges.begin() + idx + 1);
    counts[idx] += counts[static_cast<size_t>(idx + 1)];
    counts.erase(counts.begin() + idx + 1);
    count_pos[idx] += count_pos[static_cast<size_t>(idx + 1)];
    count_pos.erase(count_pos.begin() + idx + 1);
    count_neg[idx] += count_neg[static_cast<size_t>(idx + 1)];
    count_neg.erase(count_neg.begin() + idx + 1);
    
    // Recalculate WOE and labels after merging
    calculate_woe();
    update_bin_labels();
  }
  
  void ensure_bin_constraints() {
    // Ensure counts.size() <= max_bins by merging if needed
    int iterations = 0;
    while (counts.size() > static_cast<size_t>(max_bins) && iterations < max_iterations) {
      int idx = find_smallest_woe_diff();
      if (idx == -1) {
        // No valid merge index found, break to avoid infinite loop
        break;
      }
      merge_bins(idx);
      iterations++;
    }
    iterations_run += iterations;
    if (iterations >= max_iterations) {
      converged = false;
    }
    
    // Handle rare bins
    handle_rare_bins();
  }
  
  int find_smallest_woe_diff() const {
    if (woe_values.size() <= 1) return -1;
    std::vector<double> woe_diffs(woe_values.size() - 1);
    for (size_t i = 0; i < woe_diffs.size(); ++i) {
      woe_diffs[i] = std::fabs(woe_values[i + 1] - woe_values[i]);
    }
    auto min_it = std::min_element(woe_diffs.begin(), woe_diffs.end());
    if (min_it == woe_diffs.end()) return -1;
    int idx = static_cast<int>(std::distance(woe_diffs.begin(), min_it));
    return idx; // This corresponds directly to the bin pair (idx, idx+1)
  }
  
  void handle_rare_bins() {
    double total_count = std::accumulate(counts.begin(), counts.end(), 0.0);
    bool merged = true;
    int iterations = 0;
    
    // Merge bins that are too small until no rare bins or min_bins reached
    while (merged && counts.size() > static_cast<size_t>(min_bins) && iterations < max_iterations) {
      merged = false;
      for (size_t i = 0; i < counts.size(); ++i) {
        double proportion = total_count > 0.0 ? (counts[i] / total_count) : 0.0;
        if (proportion < bin_cutoff && counts.size() > static_cast<size_t>(min_bins)) {
          int merge_idx;
          if (i == 0) {
            merge_idx = 0; 
          } else if (i == counts.size() - 1) {
            merge_idx = static_cast<int>(counts.size()) - 2;
          } else {
            double diff_prev = std::fabs(woe_values[i] - woe_values[i - 1]);
            double diff_next = std::fabs(woe_values[i] - woe_values[i + 1]);
            if (diff_prev <= diff_next) {
              merge_idx = static_cast<int>(i - 1);
            } else {
              merge_idx = static_cast<int>(i);
            }
          }
          merge_bins(merge_idx);
          merged = true;
          iterations++;
          break; // Restart loop after merge
        }
      }
    }
    
    iterations_run += iterations;
    if (iterations >= max_iterations) {
      converged = false;
    }
  }
  
  void calculate_iv() {
    iv_values.resize(woe_values.size());
    for (size_t i = 0; i < woe_values.size(); ++i) {
      double p_rate = (total_pos > 0.0) ? (count_pos[i] / total_pos) : EPSILON;
      double n_rate = (total_neg > 0.0) ? (count_neg[i] / total_neg) : EPSILON;
      // Ensure p_rate, n_rate > 0
      p_rate = std::max(p_rate, EPSILON);
      n_rate = std::max(n_rate, EPSILON);
      iv_values[i] = (p_rate - n_rate) * woe_values[i];
    }
  }
};


//' @title Optimal Binning for Numerical Variables using Dynamic Programming with Local Constraints (DPLC)
//'
//' @description
//' Performs optimal binning for numerical variables using a Dynamic Programming with Local Constraints (DPLC) approach.
//' It creates optimal bins for a numerical feature based on its relationship with a binary target variable, 
//' maximizing the predictive power while respecting user-defined constraints and enforcing monotonicity.
//'
//' @param target An integer vector of binary target values (0 or 1).
//' @param feature A numeric vector of feature values.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum proportion of total observations for a bin to avoid being merged (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins before the optimization process (default: 20).
//' @param convergence_threshold Convergence threshold for the algorithm (default: 1e-6).
//' @param max_iterations Maximum number of iterations allowed (default: 1000).
//'
//' @return A list containing the following elements:
//' \item{bin}{Character vector of bin ranges.}
//' \item{woe}{Numeric vector of WoE values for each bin.}
//' \item{iv}{Numeric vector of Information Value (IV) for each bin.}
//' \item{count}{Numeric vector of total observations in each bin.}
//' \item{count_pos}{Numeric vector of positive target observations in each bin.}
//' \item{count_neg}{Numeric vector of negative target observations in each bin.}
//' \item{cutpoints}{Numeric vector of cut points to generate the bins.}
//' \item{converged}{Logical indicating if the algorithm converged.}
//' \item{iterations}{Integer number of iterations run by the algorithm.}
//'
//' @details
//' The Dynamic Programming with Local Constraints (DPLC) algorithm for numerical variables works as follows:
//' 1. Perform initial pre-binning based on quantiles of the feature distribution.
//' 2. Calculate initial counts and Weight of Evidence (WoE) for each bin.
//' 3. Enforce monotonicity of WoE values across bins by merging adjacent non-monotonic bins.
//' 4. Ensure the number of bins is between \code{min_bins} and \code{max_bins}:
//'   - Merge bins with the smallest WoE difference if above \code{max_bins}.
//'   - Handle rare bins by merging those below the \code{bin_cutoff} threshold.
//' 5. Calculate final Information Value (IV) for each bin.
//'
//' The algorithm aims to create bins that maximize the predictive power of the numerical variable while adhering to the specified constraints. It enforces monotonicity of WoE values, which is particularly useful for credit scoring and risk modeling applications.
//'
//' Weight of Evidence (WoE) is calculated as:
//' \deqn{WoE = \ln\left(\frac{\text{Positive Rate}}{\text{Negative Rate}}\right)}
//'
//' Information Value (IV) is calculated as:
//' \deqn{IV = (\text{Positive Rate} - \text{Negative Rate}) \times WoE}
//'
//' @examples
//' # Create sample data
//' set.seed(123)
//' n <- 1000
//' target <- sample(0:1, n, replace = TRUE)
//' feature <- rnorm(n)
//'
//' # Run optimal binning
//' result <- optimal_binning_numerical_dplc(target, feature, min_bins = 2, max_bins = 4)
//'
//' # Print results
//' print(result)
//'
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_dplc(Rcpp::IntegerVector target,
                                          Rcpp::NumericVector feature,
                                          int min_bins = 3,
                                          int max_bins = 5,
                                          double bin_cutoff = 0.05,
                                          int max_n_prebins = 20,
                                          double convergence_threshold = 1e-6,
                                          int max_iterations = 1000) {
  if (min_bins < 2) {
    Rcpp::stop("min_bins must be at least 2");
  }
  if (max_bins < min_bins) {
    Rcpp::stop("max_bins must be greater than or equal to min_bins");
  }
  
  std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
  std::vector<unsigned int> target_vec = Rcpp::as<std::vector<unsigned int>>(target);
  
  // Check that target contains only 0s and 1s
  for (size_t i = 0; i < target_vec.size(); ++i) {
    if (target_vec[i] != 0U && target_vec[i] != 1U) {
      Rcpp::stop("Target variable must contain only 0 and 1");
    }
  }
  
  OptimalBinningNumericalDPLC ob(feature_vec, target_vec,
                                 min_bins, max_bins, bin_cutoff, max_n_prebins,
                                 convergence_threshold, max_iterations);
  ob.fit();
  
  return ob.get_results();
}

/*
 Improvements made:
 - Added extra checks for input validity and adjusted min_bins if needed.
 - Ensured no infinite loops by checking iterations against max_iterations in all merging loops.
 - Added safety checks when merging bins (index bounds).
 - Ensured no division by zero in WoE calculations by imposing EPSILON floors.
 - Removed potential instability by enforcing minimum rates for positive/negative counts.
 - Handled edge cases where total_pos or total_neg could be zero or very small.
 - Preserved function signatures and output structure.
 - All internal logs and messages kept in English.
*/



// // [[Rcpp::plugins(cpp11)]]
// // [[Rcpp::depends(Rcpp)]]
// 
// #include <Rcpp.h>
// #include <vector>
// #include <string>
// #include <sstream>
// #include <algorithm>
// #include <cmath>
// #include <iomanip>
// #include <limits>
// #include <numeric>
// 
// using namespace Rcpp;
// 
// // Class for Optimal Binning using Dynamic Programming with Local Constraints
// class OptimalBinningNumericalDPLC {
// private:
//   const std::vector<double>& feature;
//   const std::vector<unsigned int>& target;
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
//   double convergence_threshold;
//   int max_iterations;
//   
//   bool converged;
//   int iterations_run;
//   
//   std::vector<double> bin_edges;
//   std::vector<double> woe_values;
//   std::vector<double> iv_values;
//   std::vector<std::string> bin_labels;
//   std::vector<double> count_pos;
//   std::vector<double> count_neg;
//   std::vector<double> counts;
//   
//   double total_pos;
//   double total_neg;
//   
//   static constexpr double EPSILON = 1e-10;
//   
// public:
//   OptimalBinningNumericalDPLC(const std::vector<double>& feature,
//                               const std::vector<unsigned int>& target,
//                               int min_bins,
//                               int max_bins,
//                               double bin_cutoff,
//                               int max_n_prebins,
//                               double convergence_threshold,
//                               int max_iterations)
//     : feature(feature),
//       target(target),
//       min_bins(min_bins),
//       max_bins(max_bins),
//       bin_cutoff(bin_cutoff),
//       max_n_prebins(max_n_prebins),
//       convergence_threshold(convergence_threshold),
//       max_iterations(max_iterations),
//       converged(true),
//       iterations_run(0) {
//     total_pos = std::accumulate(target.begin(), target.end(), 0.0);
//     total_neg = target.size() - total_pos;
//   }
//   
//   void fit() {
//     // Adjust min_bins if necessary
//     if (min_bins < 2) {
//       min_bins = 2;
//     }
//     if (min_bins > max_bins) {
//       min_bins = max_bins;
//     }
//     
//     // Check the number of unique feature values
//     std::vector<double> unique_feature_values = feature;
//     std::sort(unique_feature_values.begin(), unique_feature_values.end());
//     unique_feature_values.erase(std::unique(unique_feature_values.begin(), unique_feature_values.end()), unique_feature_values.end());
//     int num_unique_values = unique_feature_values.size();
//     
//     if (num_unique_values <= 2) { // Changed condition to <= 2
//       // No need to optimize; create bins based on unique values
//       bin_edges.clear();
//       bin_edges.reserve(num_unique_values + 1);
//       bin_edges.push_back(-std::numeric_limits<double>::infinity());
//       if (num_unique_values == 2) {
//         double midpoint = (unique_feature_values[0] + unique_feature_values[1]) / 2.0;
//         bin_edges.push_back(midpoint);
//       }
//       bin_edges.push_back(std::numeric_limits<double>::infinity());
//       calculate_counts_woe();
//       calculate_iv();
//       converged = true;
//       iterations_run = 0;
//       return;
//     }
//     
//     prebinning();
//     calculate_counts_woe();
//     enforce_monotonicity();
//     ensure_bin_constraints();
//     calculate_iv();
//   }
//   
//   List get_results() const {
//     // Exclude -Inf and +Inf from cutpoints
//     std::vector<double> cutpoints;
//     for (size_t i = 1; i < bin_edges.size() - 1; ++i) {
//       cutpoints.push_back(bin_edges[i]);
//     }
//     
//     return List::create(
//       Named("bin") = bin_labels,
//       Named("woe") = woe_values,
//       Named("iv") = iv_values,
//       Named("count") = counts,
//       Named("count_pos") = count_pos,
//       Named("count_neg") = count_neg,
//       Named("cutpoints") = cutpoints,
//       Named("converged") = converged,
//       Named("iterations") = iterations_run
//     );
//   }
//   
// private:
//   void prebinning() {
//     // Initial pre-binning based on quantiles
//     std::vector<size_t> sorted_indices(feature.size());
//     std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
//     std::sort(sorted_indices.begin(), sorted_indices.end(),
//               [this](size_t i1, size_t i2) { return feature[i1] < feature[i2]; });
//     
//     std::vector<double> sorted_feature(feature.size());
//     for (size_t i = 0; i < feature.size(); ++i) {
//       sorted_feature[i] = feature[sorted_indices[i]];
//     }
//     
//     int n = feature.size();
//     int bin_size = std::max(1, n / max_n_prebins);
//     std::vector<double> edges;
//     edges.reserve(max_n_prebins - 1);
//     
//     for (int i = 1; i < max_n_prebins; ++i) {
//       int idx = i * bin_size;
//       if (idx < n) {
//         edges.push_back(sorted_feature[idx]);
//       }
//     }
//     std::sort(edges.begin(), edges.end());
//     edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
//     
//     bin_edges.clear();
//     bin_edges.reserve(edges.size() + 2);
//     bin_edges.push_back(-std::numeric_limits<double>::infinity());
//     for (size_t i = 0; i < edges.size(); ++i) {
//       bin_edges.push_back(edges[i]);
//     }
//     bin_edges.push_back(std::numeric_limits<double>::infinity());
//   }
//   
//   void calculate_counts_woe() {
//     int num_bins = bin_edges.size() - 1;
//     count_pos.assign(num_bins, 0);
//     count_neg.assign(num_bins, 0);
//     counts.assign(num_bins, 0);
//     
//     for (size_t i = 0; i < feature.size(); ++i) {
//       int bin_idx = find_bin(feature[i]);
//       counts[bin_idx] += 1;
//       if (target[i] == 1) {
//         count_pos[bin_idx] += 1;
//       } else {
//         count_neg[bin_idx] += 1;
//       }
//     }
//     
//     calculate_woe();
//     update_bin_labels();
//   }
//   
//   void calculate_woe() {
//     if (total_pos <= 0 || total_neg <= 0) {
//       Rcpp::stop("Total positives or negatives are zero. Cannot compute WoE.");
//     }
//     
//     int num_bins = counts.size();
//     woe_values.resize(num_bins);
//     
//     for (int i = 0; i < num_bins; ++i) {
//       double rate_pos = (count_pos[i] > 0) ? (count_pos[i] / total_pos) : (EPSILON / total_pos);
//       double rate_neg = (count_neg[i] > 0) ? (count_neg[i] / total_neg) : (EPSILON / total_neg);
//       rate_pos = std::max(rate_pos, EPSILON);
//       rate_neg = std::max(rate_neg, EPSILON);
//       
//       woe_values[i] = std::log(rate_pos / rate_neg);
//     }
//   }
//   
//   void update_bin_labels() {
//     bin_labels.clear();
//     bin_labels.reserve(bin_edges.size() - 1);
//     for (size_t i = 0; i < bin_edges.size() - 1; ++i) {
//       std::ostringstream oss;
//       oss << std::fixed << std::setprecision(6);
//       if (bin_edges[i] == -std::numeric_limits<double>::infinity()) {
//         oss << "(-Inf;" << bin_edges[i + 1] << "]";
//       } else if (bin_edges[i + 1] == std::numeric_limits<double>::infinity()) {
//         oss << "(" << bin_edges[i] << ";+Inf]";
//       } else {
//         oss << "(" << bin_edges[i] << ";" << bin_edges[i + 1] << "]";
//       }
//       bin_labels.push_back(oss.str());
//     }
//   }
//   
//   int find_bin(double value) const {
//     auto it = std::upper_bound(bin_edges.begin(), bin_edges.end(), value);
//     int bin_idx = std::distance(bin_edges.begin(), it) - 1;
//     // Ensure bin_idx is within valid range
//     if (bin_idx < 0) bin_idx = 0;
//     if (bin_idx >= (int)counts.size()) bin_idx = counts.size() - 1;
//     return bin_idx;
//   }
//   
//   void enforce_monotonicity() {
//     if (counts.size() <= 2) {
//       // If feature has two or fewer bins, ignore monotonicity enforcement
//       return;
//     }
//     
//     bool is_monotonic = false;
//     // Determine the direction of monotonicity
//     bool increasing = true;
//     if (woe_values.size() >= 2) {
//       increasing = (woe_values[1] >= woe_values[0]);
//     }
//     
//     int iterations = 0;
//     while (!is_monotonic && counts.size() > min_bins && iterations < max_iterations) {
//       is_monotonic = true;
//       for (size_t i = 1; i < woe_values.size(); ++i) {
//         if ((increasing && woe_values[i] < woe_values[i - 1]) ||
//             (!increasing && woe_values[i] > woe_values[i - 1])) {
//           merge_bins(i - 1);
//           is_monotonic = false;
//           break;
//         }
//       }
//       iterations++;
//       if (counts.size() == min_bins) {
//         // min_bins reached, stop merging even if monotonicity is not achieved
//         break;
//       }
//     }
//     iterations_run += iterations;
//     if (iterations >= max_iterations) {
//       converged = false;
//     }
//   }
//   
//   void merge_bins(int idx) {
//     if (idx < 0 || idx >= (int)counts.size() - 1) {
//       return; // Invalid index
//     }
//     // Merge bin idx and idx+1
//     bin_edges.erase(bin_edges.begin() + idx + 1);
//     counts[idx] += counts[idx + 1];
//     counts.erase(counts.begin() + idx + 1);
//     count_pos[idx] += count_pos[idx + 1];
//     count_pos.erase(count_pos.begin() + idx + 1);
//     count_neg[idx] += count_neg[idx + 1];
//     count_neg.erase(count_neg.begin() + idx + 1);
//     calculate_woe();
//     update_bin_labels();
//   }
//   
//   void ensure_bin_constraints() {
//     // Ensure counts.size() <= max_bins
//     int iterations = 0;
//     while (counts.size() > max_bins && iterations < max_iterations) {
//       int idx = find_smallest_woe_diff();
//       if (idx == -1) break;
//       merge_bins(idx);
//       iterations++;
//     }
//     iterations_run += iterations;
//     if (iterations >= max_iterations) {
//       converged = false;
//     }
//     
//     // Handle rare bins
//     handle_rare_bins();
//   }
//   
//   int find_smallest_woe_diff() const {
//     if (woe_values.size() <= 2) return -1;
//     std::vector<double> woe_diffs(woe_values.size() - 1);
//     for (size_t i = 0; i < woe_diffs.size(); ++i) {
//       woe_diffs[i] = std::abs(woe_values[i + 1] - woe_values[i]);
//     }
//     return std::distance(woe_diffs.begin(), std::min_element(woe_diffs.begin(), woe_diffs.end()));
//   }
//   
//   void handle_rare_bins() {
//     double total_count = std::accumulate(counts.begin(), counts.end(), 0.0);
//     bool merged;
//     int iterations = 0;
//     do {
//       merged = false;
//       for (size_t i = 0; i < counts.size(); ++i) {
//         if (counts[i] / total_count < bin_cutoff && counts.size() > min_bins) {
//           int merge_idx;
//           if (i == 0) {
//             merge_idx = 0;
//           } else if (i == counts.size() - 1) {
//             merge_idx = counts.size() - 2;
//           } else {
//             // Merge with neighboring bin that has the smallest WOE difference
//             double diff_prev = std::abs(woe_values[i] - woe_values[i - 1]);
//             double diff_next = std::abs(woe_values[i] - woe_values[i + 1]);
//             if (diff_prev <= diff_next) {
//               merge_idx = i - 1;
//             } else {
//               merge_idx = i;
//             }
//           }
//           merge_bins(merge_idx);
//           merged = true;
//           iterations++;
//           break;
//         }
//       }
//     } while (merged && counts.size() > min_bins && iterations < max_iterations);
//     iterations_run += iterations;
//     if (iterations >= max_iterations) {
//       converged = false;
//     }
//   }
//   
//   void calculate_iv() {
//     iv_values.resize(woe_values.size());
//     for (size_t i = 0; i < woe_values.size(); ++i) {
//       double rate_pos = count_pos[i] / total_pos;
//       double rate_neg = count_neg[i] / total_neg;
//       iv_values[i] = (rate_pos - rate_neg) * woe_values[i];
//     }
//   }
// };
// 
// //' @title Optimal Binning for Numerical Variables using Dynamic Programming with Local Constraints (DPLC)
// //'
// //' @description
// //' Performs optimal binning for numerical variables using a Dynamic Programming with Local Constraints (DPLC) approach. It creates optimal bins for a numerical feature based on its relationship with a binary target variable, maximizing the predictive power while respecting user-defined constraints and enforcing monotonicity.
// //'
// //' @param target An integer vector of binary target values (0 or 1).
// //' @param feature A numeric vector of feature values.
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff Minimum proportion of total observations for a bin to avoid being merged (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins before the optimization process (default: 20).
// //' @param convergence_threshold Convergence threshold for the algorithm (default: 1e-6).
// //' @param max_iterations Maximum number of iterations allowed (default: 1000).
// //'
// //' @return A list containing the following elements:
// //' \item{bins}{Character vector of bin ranges.}
// //' \item{woe}{Numeric vector of WoE values for each bin.}
// //' \item{iv}{Numeric vector of Information Value (IV) for each bin.}
// //' \item{count}{Numeric vector of total observations in each bin.}
// //' \item{count_pos}{Numeric vector of positive target observations in each bin.}
// //' \item{count_neg}{Numeric vector of negative target observations in each bin.}
// //' \item{cutpoints}{Numeric vector of cut points to generate the bins.}
// //' \item{converged}{Logical indicating if the algorithm converged.}
// //' \item{iterations}{Integer number of iterations run by the algorithm.}
// //'
// //' @details
// //' The Dynamic Programming with Local Constraints (DPLC) algorithm for numerical variables works as follows:
// //' 1. Perform initial pre-binning based on quantiles of the feature distribution.
// //' 2. Calculate initial counts and Weight of Evidence (WoE) for each bin.
// //' 3. Enforce monotonicity of WoE values across bins by merging adjacent non-monotonic bins.
// //' 4. Ensure the number of bins is between \code{min_bins} and \code{max_bins}:
// //'   - Merge bins with the smallest WoE difference if above \code{max_bins}.
// //'   - Handle rare bins by merging those below the \code{bin_cutoff} threshold.
// //' 5. Calculate final Information Value (IV) for each bin.
// //'
// //' The algorithm aims to create bins that maximize the predictive power of the numerical variable while adhering to the specified constraints. It enforces monotonicity of WoE values, which is particularly useful for credit scoring and risk modeling applications.
// //'
// //' Weight of Evidence (WoE) is calculated as:
// //' \deqn{WoE = \ln\left(\frac{\text{Positive Rate}}{\text{Negative Rate}}\right)}
// //'
// //' Information Value (IV) is calculated as:
// //' \deqn{IV = (\text{Positive Rate} - \text{Negative Rate}) \times WoE}
// //'
// //' @examples
// //' # Create sample data
// //' set.seed(123)
// //' n <- 1000
// //' target <- sample(0:1, n, replace = TRUE)
// //' feature <- rnorm(n)
// //'
// //' # Run optimal binning
// //' result <- optimal_binning_numerical_dplc(target, feature, min_bins = 2, max_bins = 4)
// //'
// //' # Print results
// //' print(result)
// //'
// //' @export
// // [[Rcpp::export]]
// List optimal_binning_numerical_dplc(IntegerVector target,
//                                    NumericVector feature,
//                                    int min_bins = 3,
//                                    int max_bins = 5,
//                                    double bin_cutoff = 0.05,
//                                    int max_n_prebins = 20,
//                                    double convergence_threshold = 1e-6,
//                                    int max_iterations = 1000) {
//  if (min_bins < 2) {
//    Rcpp::stop("min_bins must be at least 2");
//  }
//  if (max_bins < min_bins) {
//    Rcpp::stop("max_bins must be greater than or equal to min_bins");
//  }
//  
//  std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
//  std::vector<unsigned int> target_vec = Rcpp::as<std::vector<unsigned int>>(target);
//  
//  // Check that target contains only 0s and 1s
//  for (size_t i = 0; i < target_vec.size(); ++i) {
//    if (target_vec[i] != 0 && target_vec[i] != 1) {
//      Rcpp::stop("Target variable must contain only 0 and 1");
//    }
//  }
//  
//  OptimalBinningNumericalDPLC ob(feature_vec, target_vec,
//                                 min_bins, max_bins, bin_cutoff, max_n_prebins,
//                                 convergence_threshold, max_iterations);
//  ob.fit();
//  
//  return ob.get_results();
// }
