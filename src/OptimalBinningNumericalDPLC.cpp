// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::plugins(openmp)]]

#include <Rcpp.h>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

class OptimalBinningNumericalDPLC {
private:
  const std::vector<double>& feature;
  const std::vector<unsigned int>& target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  
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
                              int max_n_prebins)
    : feature(feature),
      target(target),
      min_bins(min_bins),
      max_bins(max_bins),
      bin_cutoff(bin_cutoff),
      max_n_prebins(max_n_prebins) {
    total_pos = std::accumulate(target.begin(), target.end(), 0.0);
    total_neg = target.size() - total_pos;
  }
  
  void fit() {
    // Adjust min_bins if necessary
    if (min_bins < 2) {
      min_bins = 2;
    }
    if (min_bins > max_bins) {
      min_bins = max_bins;
    }
    prebinning();
    calculate_counts_woe();
    enforce_monotonicity();
    ensure_bin_constraints();
    calculate_iv();
  }
  
  List get_results() const {
    std::vector<double> woefeature(feature.size());
    
#pragma omp parallel for
    for (size_t i = 0; i < feature.size(); ++i) {
      int bin_idx = find_bin(feature[i]);
      woefeature[i] = woe_values[bin_idx];
    }
    
    DataFrame woebin = DataFrame::create(
      Named("bin") = bin_labels,
      Named("woe") = woe_values,
      Named("iv") = iv_values,
      Named("count") = counts,
      Named("count_pos") = count_pos,
      Named("count_neg") = count_neg
    );
    
    return List::create(
      Named("woefeature") = woefeature,
      Named("woebin") = woebin
    );
  }
  
private:
  void prebinning() {
    std::vector<size_t> sorted_indices(feature.size());
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [this](size_t i1, size_t i2) { return feature[i1] < feature[i2]; });
    
    std::vector<double> sorted_feature(feature.size());
    for (size_t i = 0; i < feature.size(); ++i) {
      sorted_feature[i] = feature[sorted_indices[i]];
    }
    
    int n = feature.size();
    int bin_size = std::max(1, n / max_n_prebins);
    std::vector<double> edges;
    edges.reserve(max_n_prebins - 1);
    
    for (int i = 1; i < max_n_prebins; ++i) {
      int idx = i * bin_size;
      if (idx < n) {
        edges.push_back(sorted_feature[idx]);
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
    int num_bins = bin_edges.size() - 1;
    count_pos.assign(num_bins, 0);
    count_neg.assign(num_bins, 0);
    counts.assign(num_bins, 0);
    
#pragma omp parallel for
    for (size_t i = 0; i < feature.size(); ++i) {
      int bin_idx = find_bin(feature[i]);
#pragma omp atomic
      counts[bin_idx] += 1;
      if (target[i] == 1) {
#pragma omp atomic
        count_pos[bin_idx] += 1;
      } else {
#pragma omp atomic
        count_neg[bin_idx] += 1;
      }
    }
    
    calculate_woe();
    update_bin_labels();
  }
  
  void calculate_woe() {
    if (total_pos <= 0 || total_neg <= 0) {
      Rcpp::stop("Total positives or negatives are zero. Cannot compute WoE.");
    }
    
    int num_bins = counts.size();
    woe_values.resize(num_bins);
    
    for (int i = 0; i < num_bins; ++i) {
      double rate_pos, rate_neg;
      
      if (count_pos[i] <= 0) {
        rate_pos = EPSILON / total_pos;
      } else {
        rate_pos = count_pos[i] / total_pos;
      }
      
      if (count_neg[i] <= 0) {
        rate_neg = EPSILON / total_neg;
      } else {
        rate_neg = count_neg[i] / total_neg;
      }
      
      // Avoid division by zero and log(0)
      rate_pos = std::max(rate_pos, EPSILON);
      rate_neg = std::max(rate_neg, EPSILON);
      
      woe_values[i] = std::log(rate_pos / rate_neg);
    }
  }
  
  void update_bin_labels() {
    bin_labels.clear();
    bin_labels.reserve(bin_edges.size() - 1);
    for (size_t i = 0; i < bin_edges.size() - 1; ++i) {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(6);
      if (i == 0) {
        oss << "(-Inf;" << bin_edges[i + 1] << "]";
      } else if (i == bin_edges.size() - 2) {
        oss << "(" << bin_edges[i] << ";+Inf]";
      } else {
        oss << "(" << bin_edges[i] << ";" << bin_edges[i + 1] << "]";
      }
      bin_labels.push_back(oss.str());
    }
  }
  
  int find_bin(double value) const {
    auto it = std::upper_bound(bin_edges.begin(), bin_edges.end(), value);
    int bin_idx = std::distance(bin_edges.begin(), it) - 1;
    // Ensure bin_idx is within valid range
    if (bin_idx < 0) bin_idx = 0;
    if (bin_idx >= (int)counts.size()) bin_idx = counts.size() - 1;
    return bin_idx;
  }
  
  void enforce_monotonicity() {
    if (counts.size() <= 2) {
      // If feature has two or less bins, ignore monotonicity enforcement
      return;
    }
    
    bool is_monotonic = false;
    // Determine the direction of monotonicity
    bool increasing = true;
    if (woe_values.size() >= 2) {
      increasing = (woe_values[1] >= woe_values[0]);
    }
    
    while (!is_monotonic && counts.size() > min_bins) {
      is_monotonic = true;
      for (size_t i = 1; i < woe_values.size(); ++i) {
        if ( (increasing && woe_values[i] < woe_values[i - 1]) ||
             (!increasing && woe_values[i] > woe_values[i - 1]) ) {
          merge_bins(i - 1);
          is_monotonic = false;
          break;
        }
      }
      if (counts.size() == min_bins) {
        // min_bins reached, can detour monotonicity
        break;
      }
    }
  }
  
  void merge_bins(int idx) {
    if (idx < 0 || idx >= (int)counts.size() - 1) {
      return; // Invalid index
    }
    // Merge bin idx and idx+1
    bin_edges.erase(bin_edges.begin() + idx + 1);
    counts[idx] += counts[idx + 1];
    counts.erase(counts.begin() + idx + 1);
    count_pos[idx] += count_pos[idx + 1];
    count_pos.erase(count_pos.begin() + idx + 1);
    count_neg[idx] += count_neg[idx + 1];
    count_neg.erase(count_neg.begin() + idx + 1);
    woe_values.erase(woe_values.begin() + idx + 1);
    bin_labels.erase(bin_labels.begin() + idx + 1);
    update_bin_labels();
    calculate_woe();
  }
  
  void ensure_bin_constraints() {
    // Ensure counts.size() <= max_bins
    while (counts.size() > max_bins) {
      int idx = find_smallest_woe_diff();
      if (idx == -1) break;
      merge_bins(idx);
    }
    
    // Handle rare bins
    handle_rare_bins();
  }
  
  int find_smallest_woe_diff() const {
    if (woe_values.size() <= 2) return -1;
    std::vector<double> woe_diffs(woe_values.size() - 1);
    for (size_t i = 0; i < woe_diffs.size(); ++i) {
      woe_diffs[i] = std::abs(woe_values[i + 1] - woe_values[i]);
    }
    return std::distance(woe_diffs.begin(), std::min_element(woe_diffs.begin(), woe_diffs.end()));
  }
  
  void handle_rare_bins() {
    double total_count = std::accumulate(counts.begin(), counts.end(), 0.0);
    bool merged;
    do {
      merged = false;
      for (size_t i = 0; i < counts.size(); ++i) {
        if (counts[i] / total_count < bin_cutoff && counts.size() > min_bins) {
          int merge_idx;
          if (i == 0) {
            merge_idx = 0;
          } else if (i == counts.size() - 1) {
            merge_idx = counts.size() - 2;
          } else {
            // Merge with neighboring bin that has the smallest WOE difference
            double diff_prev = std::abs(woe_values[i] - woe_values[i - 1]);
            double diff_next = std::abs(woe_values[i] - woe_values[i + 1]);
            if (diff_prev <= diff_next) {
              merge_idx = i - 1;
            } else {
              merge_idx = i;
            }
          }
          merge_bins(merge_idx);
          merged = true;
          break;
        }
      }
    } while (merged && counts.size() > min_bins);
  }
  
  void calculate_iv() {
    iv_values.resize(woe_values.size());
    for (size_t i = 0; i < woe_values.size(); ++i) {
      double rate_pos = count_pos[i] / total_pos;
      double rate_neg = count_neg[i] / total_neg;
      iv_values[i] = (rate_pos - rate_neg) * woe_values[i];
    }
  }
};

//' @title Optimal Binning for Numerical Variables using Dynamic Programming with Local Constraints (DPLC)
//'
//' @description
//' This function performs optimal binning for numerical variables using a Dynamic Programming with Local Constraints (DPLC) approach. It creates optimal bins for a numerical feature based on its relationship with a binary target variable, maximizing the predictive power while respecting user-defined constraints and enforcing monotonicity.
//'
//' @param target An integer vector of binary target values (0 or 1).
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
//' The Dynamic Programming with Local Constraints (DPLC) algorithm for numerical variables works as follows:
//' 1. Perform initial pre-binning based on quantiles of the feature distribution.
//' 2. Calculate initial counts and Weight of Evidence (WoE) for each bin.
//' 3. Enforce monotonicity of WoE values across bins by merging adjacent non-monotonic bins.
//' 4. Ensure the number of bins is between \code{min_bins} and \code{max_bins}:
//'   - Merge bins with the smallest IV difference if above \code{max_bins}.
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
//' This implementation uses OpenMP for parallel processing when available, which can significantly speed up the computation for large datasets.
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
//' result <- optimal_binning_numerical_dplc(target, feature, min_bins = 2, max_bins = 4)
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
//' @references
//' \itemize{
//'   \item Mironchyk, P., & Tchistiakov, V. (2017). Monotone optimal binning algorithm for credit risk modeling. SSRN Electronic Journal. DOI: 10.2139/ssrn.2978774
//'   \item Bellman, R. (1952). On the theory of dynamic programming. Proceedings of the National Academy of Sciences, 38(8), 716-719.
//' }
//' @author Lopes, J. E.
//'
//' @export
// [[Rcpp::export]]
List optimal_binning_numerical_dplc(IntegerVector target,
                                   NumericVector feature,
                                   int min_bins = 3,
                                   int max_bins = 5,
                                   double bin_cutoff = 0.05,
                                   int max_n_prebins = 20) {
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
   if (target_vec[i] != 0 && target_vec[i] != 1) {
     Rcpp::stop("Target variable must contain only 0 and 1");
   }
 }
 
 OptimalBinningNumericalDPLC ob(feature_vec, target_vec,
                                min_bins, max_bins, bin_cutoff, max_n_prebins);
 ob.fit();
 
 return ob.get_results();
}



// // [[Rcpp::plugins(cpp11)]]
// // [[Rcpp::plugins(openmp)]]
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
// #ifdef _OPENMP
// #include <omp.h>
// #endif
// 
// using namespace Rcpp;
// 
// class OptimalBinningNumericalDPLC {
// private:
//   const std::vector<double>& feature;
//   const std::vector<unsigned int>& target;
//   const int min_bins;
//   const int max_bins;
//   const double bin_cutoff;
//   const int max_n_prebins;
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
//                               int max_n_prebins)
//     : feature(feature),
//       target(target),
//       min_bins(min_bins),
//       max_bins(max_bins),
//       bin_cutoff(bin_cutoff),
//       max_n_prebins(max_n_prebins) {
//     total_pos = std::accumulate(target.begin(), target.end(), 0.0);
//     total_neg = target.size() - total_pos;
//   }
// 
//   void fit() {
//     prebinning();
//     calculate_counts_woe();
//     enforce_monotonicity();
//     ensure_bin_constraints();
//     calculate_iv();
//   }
// 
//   List get_results() const {
//     std::vector<double> woefeature(feature.size());
// 
// #pragma omp parallel for
//     for (size_t i = 0; i < feature.size(); ++i) {
//       int bin_idx = find_bin(feature[i]);
//       woefeature[i] = woe_values[bin_idx];
//     }
// 
//     DataFrame woebin = DataFrame::create(
//       Named("bin") = bin_labels,
//       Named("woe") = woe_values,
//       Named("iv") = iv_values,
//       Named("count") = counts,
//       Named("count_pos") = count_pos,
//       Named("count_neg") = count_neg
//     );
// 
//     return List::create(
//       Named("woefeature") = woefeature,
//       Named("woebin") = woebin
//     );
//   }
// 
// private:
//   void prebinning() {
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
//     bin_edges.resize(edges.size() + 2);
//     bin_edges[0] = -std::numeric_limits<double>::infinity();
//     std::copy(edges.begin(), edges.end(), bin_edges.begin() + 1);
//     bin_edges[bin_edges.size() - 1] = std::numeric_limits<double>::infinity();
//   }
// 
//   void calculate_counts_woe() {
//     int num_bins = bin_edges.size() - 1;
//     count_pos.assign(num_bins, 0);
//     count_neg.assign(num_bins, 0);
//     counts.assign(num_bins, 0);
// 
// #pragma omp parallel for
//     for (size_t i = 0; i < feature.size(); ++i) {
//       int bin_idx = find_bin(feature[i]);
// #pragma omp atomic
//       counts[bin_idx] += 1;
//       if (target[i] == 1) {
// #pragma omp atomic
//         count_pos[bin_idx] += 1;
//       } else {
// #pragma omp atomic
//         count_neg[bin_idx] += 1;
//       }
//     }
// 
//     calculate_woe();
//     update_bin_labels();
//   }
// 
//   void calculate_woe() {
//     int num_bins = counts.size();
//     woe_values.resize(num_bins);
//     for (int i = 0; i < num_bins; ++i) {
//       double rate_pos = (count_pos[i] + 0.5) / (total_pos + 0.5 * num_bins);
//       double rate_neg = (count_neg[i] + 0.5) / (total_neg + 0.5 * num_bins);
//       woe_values[i] = std::log(std::max(rate_pos, EPSILON) / std::max(rate_neg, EPSILON));
//     }
//   }
// 
//   void update_bin_labels() {
//     bin_labels.clear();
//     bin_labels.reserve(bin_edges.size() - 1);
//     for (size_t i = 0; i < bin_edges.size() - 1; ++i) {
//       std::ostringstream oss;
//       oss << std::fixed << std::setprecision(6);
//       if (i == 0) {
//         oss << "(-Inf;" << bin_edges[i + 1] << "]";
//       } else if (i == bin_edges.size() - 2) {
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
//     return std::distance(bin_edges.begin(), it) - 1;
//   }
// 
//   void enforce_monotonicity() {
//     bool is_monotonic = false;
//     while (!is_monotonic && counts.size() > min_bins) {
//       is_monotonic = true;
//       for (size_t i = 1; i < woe_values.size(); ++i) {
//         if (woe_values[i] < woe_values[i - 1]) {
//           merge_bins(i - 1);
//           is_monotonic = false;
//           break;
//         }
//       }
//     }
//   }
// 
//   void merge_bins(int idx) {
//     bin_edges.erase(bin_edges.begin() + idx + 1);
//     counts[idx] += counts[idx + 1];
//     counts.erase(counts.begin() + idx + 1);
//     count_pos[idx] += count_pos[idx + 1];
//     count_pos.erase(count_pos.begin() + idx + 1);
//     count_neg[idx] += count_neg[idx + 1];
//     count_neg.erase(count_neg.begin() + idx + 1);
//     woe_values.erase(woe_values.begin() + idx + 1);
//     bin_labels.erase(bin_labels.begin() + idx + 1);
//     update_bin_labels();
//     calculate_woe();
//   }
// 
//   void ensure_bin_constraints() {
//     while (counts.size() > max_bins) {
//       int idx = find_smallest_iv_diff();
//       if (idx == -1) break;
//       merge_bins(idx);
//     }
// 
//     handle_rare_bins();
//   }
// 
//   int find_smallest_iv_diff() const {
//     if (woe_values.size() <= 2) return -1;
//     std::vector<double> iv_diffs(woe_values.size() - 1);
//     for (size_t i = 0; i < iv_diffs.size(); ++i) {
//       iv_diffs[i] = std::abs(woe_values[i + 1] - woe_values[i]);
//     }
//     return std::distance(iv_diffs.begin(), std::min_element(iv_diffs.begin(), iv_diffs.end()));
//   }
// 
//   void handle_rare_bins() {
//     double total_count = std::accumulate(counts.begin(), counts.end(), 0.0);
//     bool merged;
//     do {
//       merged = false;
//       for (size_t i = 0; i < counts.size(); ++i) {
//         if (counts[i] / total_count < bin_cutoff && counts.size() > min_bins) {
//           int merge_idx = (i == 0) ? 0 : i - 1;
//           merge_bins(merge_idx);
//           merged = true;
//           break;
//         }
//       }
//     } while (merged && counts.size() > min_bins);
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
// //' This function performs optimal binning for numerical variables using a Dynamic Programming with Local Constraints (DPLC) approach. It creates optimal bins for a numerical feature based on its relationship with a binary target variable, maximizing the predictive power while respecting user-defined constraints and enforcing monotonicity.
// //'
// //' @param target An integer vector of binary target values (0 or 1).
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
// //' The Dynamic Programming with Local Constraints (DPLC) algorithm for numerical variables works as follows:
// //' 1. Perform initial pre-binning based on quantiles of the feature distribution.
// //' 2. Calculate initial counts and Weight of Evidence (WoE) for each bin.
// //' 3. Enforce monotonicity of WoE values across bins by merging adjacent non-monotonic bins.
// //' 4. Ensure the number of bins is between \code{min_bins} and \code{max_bins}:
// //'   - Merge bins with the smallest IV difference if above \code{max_bins}.
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
// //' This implementation uses OpenMP for parallel processing when available, which can significantly speed up the computation for large datasets.
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
// //' result <- optimal_binning_numerical_dplc(target, feature, min_bins = 2, max_bins = 4)
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
// //' @references
// //' \itemize{
// //'   \item Mironchyk, P., & Tchistiakov, V. (2017). Monotone optimal binning algorithm for credit risk modeling. SSRN Electronic Journal. DOI: 10.2139/ssrn.2978774
// //'   \item Bellman, R. (1952). On the theory of dynamic programming. Proceedings of the National Academy of Sciences, 38(8), 716-719.
// //' }
// //' @author Lopes, J. E.
// //'
// //' @export
// // [[Rcpp::export]]
// List optimal_binning_numerical_dplc(IntegerVector target,
//                                    NumericVector feature,
//                                    int min_bins = 3,
//                                    int max_bins = 5,
//                                    double bin_cutoff = 0.05,
//                                    int max_n_prebins = 20) {
//  if (min_bins < 2) {
//    stop("min_bins must be at least 2");
//  }
//  if (max_bins < min_bins) {
//    stop("max_bins must be greater than or equal to min_bins");
//  }
// 
//  std::vector<double> feature_vec = as<std::vector<double>>(feature);
//  std::vector<unsigned int> target_vec = as<std::vector<unsigned int>>(target);
// 
//  OptimalBinningNumericalDPLC ob(feature_vec, target_vec,
//                                 min_bins, max_bins, bin_cutoff, max_n_prebins);
//  ob.fit();
// 
//  return ob.get_results();
// }
