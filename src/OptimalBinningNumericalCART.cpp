// [[Rcpp::plugins(openmp)]]
#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

class OptimalBinningNumericalCART {
private:
  std::vector<double> feature;
  std::vector<int> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  bool is_monotonic;
  std::vector<double> bin_edges;
  std::vector<double> woe_values;
  std::vector<double> iv_values;
  std::vector<int> counts;
  std::vector<int> count_pos;
  std::vector<int> count_neg;
  double total_iv;
  
  void validate_inputs() {
    if (feature.empty() || target.empty()) {
      throw std::runtime_error("Feature and target vectors must not be empty.");
    }
    if (feature.size() != target.size()) {
      throw std::runtime_error("Feature and target vectors must have the same length.");
    }
    if (!std::all_of(target.begin(), target.end(), [](int i){ return i == 0 || i == 1; })) {
      throw std::runtime_error("Target vector must be binary (0 and 1).");
    }
    if (min_bins < 2) {
      throw std::runtime_error("min_bins must be at least 2.");
    }
    if (max_bins < min_bins) {
      throw std::runtime_error("max_bins must be greater than or equal to min_bins.");
    }
    if (bin_cutoff <= 0 || bin_cutoff >= 1) {
      throw std::runtime_error("bin_cutoff must be between 0 and 1.");
    }
    if (max_n_prebins < max_bins) {
      throw std::runtime_error("max_n_prebins must be greater than or equal to max_bins.");
    }
  }
  
  void initialize_bins() {
    std::vector<double> sorted_feature = feature;
    std::sort(sorted_feature.begin(), sorted_feature.end());
    
    int n = sorted_feature.size();
    int n_prebins = std::min(max_n_prebins, n);
    bin_edges.resize(n_prebins + 1);
    bin_edges[0] = std::nextafter(sorted_feature.front(), -std::numeric_limits<double>::infinity());
    bin_edges[n_prebins] = std::nextafter(sorted_feature.back(), std::numeric_limits<double>::infinity());
    
    for (int i = 1; i < n_prebins; ++i) {
      int idx = i * n / n_prebins;
      bin_edges[i] = sorted_feature[idx];
    }
    
    auto last = std::unique(bin_edges.begin(), bin_edges.end());
    bin_edges.erase(last, bin_edges.end());
    
    while (bin_edges.size() < min_bins + 1) {
      double new_edge = sorted_feature[n * bin_edges.size() / (min_bins + 1)];
      auto it = std::lower_bound(bin_edges.begin(), bin_edges.end(), new_edge);
      if (it == bin_edges.end() || *it != new_edge) {
        bin_edges.insert(it, new_edge);
      }
    }
    
    counts.assign(bin_edges.size() - 1, 0);
    count_pos.assign(bin_edges.size() - 1, 0);
    count_neg.assign(bin_edges.size() - 1, 0);
  }
  
  void compute_woe_iv() {
    int n_bins = bin_edges.size() - 1;
    std::fill(counts.begin(), counts.end(), 0);
    std::fill(count_pos.begin(), count_pos.end(), 0);
    std::fill(count_neg.begin(), count_neg.end(), 0);
    
    int total_pos = std::accumulate(target.begin(), target.end(), 0);
    int total_neg = target.size() - total_pos;
    
    int n = feature.size();
    
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
      double val = feature[i];
      int bin_idx = find_bin(val);
#pragma omp atomic
      counts[bin_idx]++;
      if (target[i] == 1) {
#pragma omp atomic
        count_pos[bin_idx]++;
      } else {
#pragma omp atomic
        count_neg[bin_idx]++;
      }
    }
    
    woe_values.assign(n_bins, 0.0);
    iv_values.assign(n_bins, 0.0);
    total_iv = 0.0;
    
    const double smoothing = 0.5; // Laplace smoothing
    for (int i = 0; i < n_bins; ++i) {
      double pos = count_pos[i] + smoothing;
      double neg = count_neg[i] + smoothing;
      double dist_pos = pos / (total_pos + n_bins * smoothing);
      double dist_neg = neg / (total_neg + n_bins * smoothing);
      woe_values[i] = std::log(dist_pos / dist_neg);
      iv_values[i] = (dist_pos - dist_neg) * woe_values[i];
      total_iv += iv_values[i];
    }
  }
  
  int find_bin(double value) const {
    auto it = std::upper_bound(bin_edges.begin(), bin_edges.end(), value);
    return std::max(0, std::min((int)bin_edges.size() - 2, (int)std::distance(bin_edges.begin(), it) - 1));
  }
  
  void merge_bin_indices(size_t idx1, size_t idx2) {
    bin_edges.erase(bin_edges.begin() + idx2);
    counts[idx1] += counts[idx2];
    count_pos[idx1] += count_pos[idx2];
    count_neg[idx1] += count_neg[idx2];
    counts.erase(counts.begin() + idx2);
    count_pos.erase(count_pos.begin() + idx2);
    count_neg.erase(count_neg.begin() + idx2);
    woe_values.erase(woe_values.begin() + idx2);
    iv_values.erase(iv_values.begin() + idx2);
  }
  
  void merge_bins() {
    while (bin_edges.size() > max_bins + 1) {
      double min_iv_diff = std::numeric_limits<double>::max();
      size_t merge_idx = 0;
      for (size_t i = 0; i < woe_values.size() - 1; ++i) {
        double iv_diff = std::abs(iv_values[i] - iv_values[i+1]);
        if (iv_diff < min_iv_diff) {
          min_iv_diff = iv_diff;
          merge_idx = i;
        }
      }
      merge_bin_indices(merge_idx, merge_idx + 1);
      compute_woe_iv();
    }
    
    bool merged;
    do {
      merged = false;
      for (size_t i = 0; i < counts.size(); ++i) {
        double bin_pct = static_cast<double>(counts[i]) / feature.size();
        if (bin_pct < bin_cutoff && bin_edges.size() > min_bins + 1) {
          size_t merge_with = (i == 0) ? 1 : ((i == counts.size() - 1) ? counts.size() - 2 :
                                                (iv_values[i - 1] < iv_values[i + 1] ? i - 1 : i + 1));
          merge_bin_indices(std::min(i, merge_with), std::max(i, merge_with));
          compute_woe_iv();
          merged = true;
          break;
        }
      }
    } while (merged && bin_edges.size() > min_bins + 1);
  }
  
  void enforce_monotonicity() {
    if (!is_monotonic) return;
    
    bool adjusted;
    do {
      adjusted = false;
      for (size_t i = 1; i < woe_values.size(); ++i) {
        if (woe_values[i] < woe_values[i - 1] && bin_edges.size() > min_bins + 1) {
          merge_bin_indices(i - 1, i);
          compute_woe_iv();
          adjusted = true;
          break;
        }
      }
    } while (adjusted && bin_edges.size() > min_bins + 1);
  }
  
public:
  OptimalBinningNumericalCART(const std::vector<double>& feature, const std::vector<int>& target,
                              int min_bins = 3, int max_bins = 5, double bin_cutoff = 0.05,
                              int max_n_prebins = 20, bool is_monotonic = true)
    : feature(feature), target(target), min_bins(min_bins), max_bins(max_bins),
      bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins), is_monotonic(is_monotonic) {}
  
  void fit() {
    try {
      validate_inputs();
      initialize_bins();
      compute_woe_iv();
      merge_bins();
      enforce_monotonicity();
      compute_woe_iv();
    } catch (const std::exception& e) {
      Rcpp::stop("Error in OptimalBinningNumericalCART: " + std::string(e.what()));
    }
  }
  
  List get_result() {
    int n_bins = bin_edges.size() - 1;
    CharacterVector bins(n_bins);
    for (int i = 0; i < n_bins; ++i) {
      std::ostringstream oss;
      oss.precision(6);
      oss << std::fixed;
      if (i == 0) {
        oss << "(-Inf; " << bin_edges[i + 1] << "]";
      } else if (i == n_bins - 1) {
        oss << "(" << bin_edges[i] << "; Inf]";
      } else {
        oss << "(" << bin_edges[i] << "; " << bin_edges[i + 1] << "]";
      }
      bins[i] = oss.str();
    }
    
    return List::create(
      _["woefeature"] = assign_woe(),
      _["woebin"] = DataFrame::create(
        _["bin"] = bins,
        _["woe"] = woe_values,
        _["iv"] = iv_values,
        _["count"] = counts,
        _["count_pos"] = count_pos,
        _["count_neg"] = count_neg
      ),
      _["total_iv"] = total_iv
    );
  }
  
  NumericVector assign_woe() const {
    int n = feature.size();
    NumericVector woefeature(n);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
      int bin_idx = find_bin(feature[i]);
      woefeature[i] = woe_values[bin_idx];
    }
    return woefeature;
  }
};

//' @title Optimal Binning for Numerical Variables using CART-based approach
//'
//' @description
//' This function implements an optimal binning algorithm for numerical variables using a CART-based (Classification and Regression Trees) approach with Weight of Evidence (WoE) and Information Value (IV) criteria.
//'
//' @param target An integer vector of binary target values (0 or 1).
//' @param feature A numeric vector of feature values to be binned.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum frequency of observations in each bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins for initial quantile-based discretization (default: 20).
//' @param is_monotonic Boolean indicating whether to enforce monotonicity of WoE across bins (default: TRUE).
//'
//' @return A list containing three elements:
//' \item{woefeature}{A numeric vector of WoE-transformed feature values.}
//' \item{woebin}{A data frame with binning details, including bin boundaries, WoE, IV, and count statistics.}
//' \item{total_iv}{The total Information Value of the binned feature.}
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
//' result <- optimal_binning_numerical_cart(target, feature, min_bins = 3, max_bins = 5)
//'
//' # View binning results
//' print(result$woebin)
//'
//' # Plot WoE transformation
//' plot(feature, result$woefeature, main = "WoE Transformation",
//'  xlab = "Original Feature", ylab = "WoE")
//'
//' # Print total Information Value
//' cat("Total IV:", result$total_iv, "\n")
//' }
//'
//' @details
//' The optimal binning algorithm for numerical variables uses a CART-based approach with Weight of Evidence (WoE) and Information Value (IV) to create bins that maximize the predictive power of the feature while maintaining interpretability.
//'
//' The algorithm follows these steps:
//' 1. Initial discretization using quantile-based binning
//' 2. Calculation of WoE and IV for each bin
//' 3. Merging of bins based on minimizing IV differences
//' 4. Enforcing minimum bin frequency (bin_cutoff)
//' 5. Enforcing monotonicity of WoE across bins (if is_monotonic is TRUE)
//'
//' Weight of Evidence (WoE) is calculated for each bin as:
//'
//' \deqn{WoE_i = \ln\left(\frac{P(X_i|Y=1)}{P(X_i|Y=0)}\right)}
//'
//' where \eqn{P(X_i|Y=1)} is the proportion of positive cases in bin i, and \eqn{P(X_i|Y=0)} is the proportion of negative cases in bin i.
//'
//' Information Value (IV) for each bin is calculated as:
//'
//' \deqn{IV_i = (P(X_i|Y=1) - P(X_i|Y=0)) \times WoE_i}
//'
//' The total IV for the feature is the sum of IVs across all bins:
//'
//' \deqn{IV_{total} = \sum_{i=1}^{n} IV_i}
//'
//' The CART-based approach iteratively merges bins with the smallest IV difference, ensuring that the resulting binning maximizes the total IV while maintaining the desired number of bins and respecting the minimum bin frequency constraint.
//'
//' @references
//' \itemize{
//'   \item Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984). Classification and regression trees. CRC press.
//'   \item Zeng, G. (2014). A necessary condition for a good binning algorithm in credit scoring. Applied Mathematical Sciences, 8(65), 3229-3242.
//' }
//'
//' @author Lopes, J. E.
//'
//' @export
// [[Rcpp::export]]
List optimal_binning_numerical_cart(const IntegerVector& target, const NumericVector& feature, 
                                    int min_bins = 3, int max_bins = 5, double bin_cutoff = 0.05,
                                    int max_n_prebins = 20, bool is_monotonic = true) {
  std::vector<double> feature_vec(feature.begin(), feature.end());
  std::vector<int> target_vec(target.begin(), target.end());
  
  try {
    OptimalBinningNumericalCART cart(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins, is_monotonic);
    cart.fit();
    return cart.get_result();
  } catch (const std::exception& e) {
    Rcpp::stop("Error in optimal_binning_numerical_cart: " + std::string(e.what()));
  }
}


// // [[Rcpp::plugins(openmp)]]
// #include <Rcpp.h>
// #include <vector>
// #include <algorithm>
// #include <numeric>
// #include <limits>
// 
// #ifdef _OPENMP
// #include <omp.h>
// #endif
// 
// using namespace Rcpp;
// 
// class OptimalBinningNumericalCART {
// private:
//   std::vector<double> feature;
//   std::vector<int> target;
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
//   bool is_monotonic;
//   std::vector<double> bin_edges;
//   std::vector<double> woe_values;
//   std::vector<double> iv_values;
//   std::vector<int> counts;
//   std::vector<int> count_pos;
//   std::vector<int> count_neg;
//   double total_iv;
// 
//   void validate_inputs() {
//     if (feature.size() != target.size()) {
//       stop("Feature and target vectors must have the same length.");
//     }
//     if (!std::all_of(target.begin(), target.end(), [](int i){ return i == 0 || i == 1; })) {
//       stop("Target vector must be binary (0 and 1).");
//     }
//     if (min_bins < 2) {
//       stop("min_bins must be at least 2.");
//     }
//     if (max_bins < min_bins) {
//       stop("max_bins must be greater than or equal to min_bins.");
//     }
//     if (bin_cutoff <= 0 || bin_cutoff >= 1) {
//       stop("bin_cutoff must be between 0 and 1.");
//     }
//     if (max_n_prebins < max_bins) {
//       stop("max_n_prebins must be greater than or equal to max_bins.");
//     }
//   }
// 
//   void initialize_bins() {
//     int n = feature.size();
//     std::vector<double> sorted_feature = feature;
//     std::sort(sorted_feature.begin(), sorted_feature.end());
// 
//     int n_prebins = std::min(max_n_prebins, n);
//     bin_edges.resize(n_prebins + 1);
//     bin_edges[0] = -std::numeric_limits<double>::infinity();
//     bin_edges[n_prebins] = std::numeric_limits<double>::infinity();
// 
//     for (int i = 1; i < n_prebins; ++i) {
//       int idx = i * n / n_prebins;
//       bin_edges[i] = sorted_feature[idx];
//     }
// 
//     auto last = std::unique(bin_edges.begin(), bin_edges.end());
//     bin_edges.erase(last, bin_edges.end());
// 
//     // Ensure at least min_bins unique edges
//     while (bin_edges.size() < min_bins + 1) {
//       double new_edge = sorted_feature[n * bin_edges.size() / (min_bins + 1)];
//       auto it = std::lower_bound(bin_edges.begin(), bin_edges.end(), new_edge);
//       if (it == bin_edges.end() || *it != new_edge) {
//         bin_edges.insert(it, new_edge);
//       }
//     }
// 
//     counts.assign(bin_edges.size() - 1, 0);
//     count_pos.assign(bin_edges.size() - 1, 0);
//     count_neg.assign(bin_edges.size() - 1, 0);
//   }
// 
//   void compute_woe_iv() {
//     int n_bins = bin_edges.size() - 1;
//     std::fill(counts.begin(), counts.end(), 0);
//     std::fill(count_pos.begin(), count_pos.end(), 0);
//     std::fill(count_neg.begin(), count_neg.end(), 0);
// 
//     int total_pos = std::accumulate(target.begin(), target.end(), 0);
//     int total_neg = target.size() - total_pos;
// 
//     int n = feature.size();
// 
// #pragma omp parallel for schedule(static)
//     for (int i = 0; i < n; ++i) {
//       double val = feature[i];
//       int bin_idx = find_bin(val);
// #pragma omp atomic
//       counts[bin_idx]++;
//       if (target[i] == 1) {
// #pragma omp atomic
//         count_pos[bin_idx]++;
//       } else {
// #pragma omp atomic
//         count_neg[bin_idx]++;
//       }
//     }
// 
//     woe_values.assign(n_bins, 0.0);
//     iv_values.assign(n_bins, 0.0);
//     total_iv = 0.0;
// 
//     double smoothing = 0.5; // Laplace smoothing
//     for (int i = 0; i < n_bins; ++i) {
//       double pos = count_pos[i] + smoothing;
//       double neg = count_neg[i] + smoothing;
//       double dist_pos = pos / (total_pos + n_bins * smoothing);
//       double dist_neg = neg / (total_neg + n_bins * smoothing);
//       woe_values[i] = std::log(dist_pos / dist_neg);
//       iv_values[i] = (dist_pos - dist_neg) * woe_values[i];
//       total_iv += iv_values[i];
//     }
//   }
// 
//   int find_bin(double value) const {
//     auto it = std::upper_bound(bin_edges.begin(), bin_edges.end(), value);
//     return std::max(0, std::min((int)bin_edges.size() - 2, (int)std::distance(bin_edges.begin(), it) - 1));
//   }
// 
//   void merge_bin_indices(size_t idx1, size_t idx2) {
//     bin_edges.erase(bin_edges.begin() + idx2);
//     counts[idx1] += counts[idx2];
//     count_pos[idx1] += count_pos[idx2];
//     count_neg[idx1] += count_neg[idx2];
//     counts.erase(counts.begin() + idx2);
//     count_pos.erase(count_pos.begin() + idx2);
//     count_neg.erase(count_neg.begin() + idx2);
//     woe_values.erase(woe_values.begin() + idx2);
//     iv_values.erase(iv_values.begin() + idx2);
//   }
// 
//   void merge_bins() {
//     while (bin_edges.size() > max_bins + 1) {
//       double min_iv_diff = std::numeric_limits<double>::max();
//       size_t merge_idx = 0;
//       for (size_t i = 0; i < woe_values.size() - 1; ++i) {
//         double iv_diff = std::abs(iv_values[i] - iv_values[i+1]);
//         if (iv_diff < min_iv_diff) {
//           min_iv_diff = iv_diff;
//           merge_idx = i;
//         }
//       }
//       merge_bin_indices(merge_idx, merge_idx + 1);
//       compute_woe_iv();
//     }
// 
//     bool merged;
//     do {
//       merged = false;
//       for (size_t i = 0; i < counts.size(); ++i) {
//         double bin_pct = (double)counts[i] / feature.size();
//         if (bin_pct < bin_cutoff && bin_edges.size() > min_bins + 1) {
//           size_t merge_with = (i == 0) ? 1 : ((i == counts.size() - 1) ? counts.size() - 2 :
//                                                 (iv_values[i - 1] < iv_values[i + 1] ? i - 1 : i + 1));
//           merge_bin_indices(std::min(i, merge_with), std::max(i, merge_with));
//           compute_woe_iv();
//           merged = true;
//           break;
//         }
//       }
//     } while (merged && bin_edges.size() > min_bins + 1);
//   }
// 
//   void enforce_monotonicity() {
//     if (!is_monotonic) return;
// 
//     bool adjusted;
//     do {
//       adjusted = false;
//       for (size_t i = 1; i < woe_values.size(); ++i) {
//         if (woe_values[i] < woe_values[i - 1] && bin_edges.size() > min_bins + 1) {
//           merge_bin_indices(i - 1, i);
//           compute_woe_iv();
//           adjusted = true;
//           break;
//         }
//       }
//     } while (adjusted && bin_edges.size() > min_bins + 1);
//   }
// 
// public:
//   OptimalBinningNumericalCART(std::vector<double> feature, std::vector<int> target,
//                               int min_bins = 3, int max_bins = 5, double bin_cutoff = 0.05,
//                               int max_n_prebins = 20, bool is_monotonic = true)
//     : feature(feature), target(target), min_bins(min_bins), max_bins(max_bins),
//       bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins), is_monotonic(is_monotonic) {}
// 
//   void fit() {
//     validate_inputs();
//     initialize_bins();
//     compute_woe_iv();
//     merge_bins();
//     enforce_monotonicity();
//     compute_woe_iv();
//   }
// 
//   List get_result() {
//     int n_bins = bin_edges.size() - 1;
//     CharacterVector bins(n_bins);
//     for (int i = 0; i < n_bins; ++i) {
//       std::ostringstream oss;
//       if (i == 0) {
//         oss << "(-Inf; " << bin_edges[i + 1] << "]";
//       } else if (i == n_bins - 1) {
//         oss << "(" << bin_edges[i] << "; Inf]";
//       } else {
//         oss << "(" << bin_edges[i] << "; " << bin_edges[i + 1] << "]";
//       }
//       bins[i] = oss.str();
//     }
// 
//     return List::create(
//       _["woefeature"] = assign_woe(),
//       _["woebin"] = DataFrame::create(
//         _["bin"] = bins,
//         _["woe"] = woe_values,
//         _["iv"] = iv_values,
//         _["count"] = counts,
//         _["count_pos"] = count_pos,
//         _["count_neg"] = count_neg
//       ),
//       _["total_iv"] = total_iv
//     );
//   }
// 
//   NumericVector assign_woe() const {
//     int n = feature.size();
//     NumericVector woefeature(n);
// #pragma omp parallel for schedule(static)
//     for (int i = 0; i < n; ++i) {
//       int bin_idx = find_bin(feature[i]);
//       woefeature[i] = woe_values[bin_idx];
//     }
//     return woefeature;
//   }
// };
// 
// 
// //' @title Optimal Binning for Numerical Variables using CART-based approach
// //'
// //' @description
// //' This function implements an optimal binning algorithm for numerical variables using a CART-based (Classification and Regression Trees) approach with Weight of Evidence (WoE) and Information Value (IV) criteria.
// //'
// //' @param target An integer vector of binary target values (0 or 1).
// //' @param feature A numeric vector of feature values to be binned.
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff Minimum frequency of observations in each bin (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins for initial quantile-based discretization (default: 20).
// //' @param is_monotonic Boolean indicating whether to enforce monotonicity of WoE across bins (default: TRUE).
// //'
// //' @return A list containing three elements:
// //' \item{woefeature}{A numeric vector of WoE-transformed feature values.}
// //' \item{woebin}{A data frame with binning details, including bin boundaries, WoE, IV, and count statistics.}
// //' \item{total_iv}{The total Information Value of the binned feature.}
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
// //' result <- optimal_binning_numerical_cart(target, feature, min_bins = 3, max_bins = 5)
// //'
// //' # View binning results
// //' print(result$woebin)
// //'
// //' # Plot WoE transformation
// //' plot(feature, result$woefeature, main = "WoE Transformation",
// //'  xlab = "Original Feature", ylab = "WoE")
// //'
// //' # Print total Information Value
// //' cat("Total IV:", result$total_iv, "\n")
// //' }
// //'
// //' @details
// //' The optimal binning algorithm for numerical variables uses a CART-based approach with Weight of Evidence (WoE) and Information Value (IV) to create bins that maximize the predictive power of the feature while maintaining interpretability.
// //'
// //' The algorithm follows these steps:
// //' 1. Initial discretization using quantile-based binning
// //' 2. Calculation of WoE and IV for each bin
// //' 3. Merging of bins based on minimizing IV differences
// //' 4. Enforcing minimum bin frequency (bin_cutoff)
// //' 5. Enforcing monotonicity of WoE across bins (if is_monotonic is TRUE)
// //'
// //' Weight of Evidence (WoE) is calculated for each bin as:
// //'
// //' \deqn{WoE_i = \ln\left(\frac{P(X_i|Y=1)}{P(X_i|Y=0)}\right)}
// //'
// //' where \eqn{P(X_i|Y=1)} is the proportion of positive cases in bin i, and \eqn{P(X_i|Y=0)} is the proportion of negative cases in bin i.
// //'
// //' Information Value (IV) for each bin is calculated as:
// //'
// //' \deqn{IV_i = (P(X_i|Y=1) - P(X_i|Y=0)) \times WoE_i}
// //'
// //' The total IV for the feature is the sum of IVs across all bins:
// //'
// //' \deqn{IV_{total} = \sum_{i=1}^{n} IV_i}
// //'
// //' The CART-based approach iteratively merges bins with the smallest IV difference, ensuring that the resulting binning maximizes the total IV while maintaining the desired number of bins and respecting the minimum bin frequency constraint.
// //'
// //' @references
// //' \itemize{
// //'   \item Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984). Classification and regression trees. CRC press.
// //'   \item Zeng, G. (2014). A necessary condition for a good binning algorithm in credit scoring. Applied Mathematical Sciences, 8(65), 3229-3242.
// //' }
// //'
// //' @author Lopes, J. E.
// //'
// //' @export
// // [[Rcpp::export]]
// List optimal_binning_numerical_cart(IntegerVector target, NumericVector feature,
//                                     int min_bins = 3, int max_bins = 5, double bin_cutoff = 0.05,
//                                     int max_n_prebins = 20, bool is_monotonic = true) {
//   std::vector<double> feature_vec(feature.begin(), feature.end());
//   std::vector<int> target_vec(target.begin(), target.end());
// 
//   OptimalBinningNumericalCART cart(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins, is_monotonic);
//   cart.fit();
//   return cart.get_result();
// }
// 
