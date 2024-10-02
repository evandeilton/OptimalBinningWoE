// [[Rcpp::plugins(cpp11)]]
#ifdef _OPENMP
// [[Rcpp::plugins(openmp)]]
#include <omp.h>
#endif

#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <set>
#include <string>
#include <sstream>
#include <iomanip>

using namespace Rcpp;

// Structure to hold bin information
struct Bin {
  double lower_bound;
  double upper_bound;
  int count;
  int count_pos;
  int count_neg;
  double woe;
  double iv;
};

// Comparator for sorting feature values
bool compare_feature(const double a, const double b) {
  return a < b;
}

class OptimalBinningNumericalEB {
private:
  NumericVector feature;
  IntegerVector target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  int n_threads;
  
  std::vector<Bin> bins;
  
  // Validate input parameters and data
  void validate_inputs() {
    if (feature.size() != target.size()) {
      stop("Feature and target vectors must be of the same length.");
    }
    
    if (min_bins < 2) {
      stop("min_bins must be at least 2.");
    }
    
    if (max_bins < min_bins) {
      stop("max_bins must be greater than or equal to min_bins.");
    }
    
    if (bin_cutoff <= 0 || bin_cutoff >= 0.5) {
      stop("bin_cutoff must be between 0 and 0.5.");
    }
    
    if (max_n_prebins < max_bins) {
      stop("max_n_prebins must be greater than or equal to max_bins.");
    }
    
    // Check for NA values in feature
    for (int i = 0; i < feature.size(); ++i) {
      if (NumericVector::is_na(feature[i])) {
        stop("Feature vector contains NA values.");
      }
    }
    
    // Check for NA values in target
    for (int i = 0; i < target.size(); ++i) {
      if (IntegerVector::is_na(target[i])) {
        stop("Target vector contains NA values.");
      }
    }
    
    // Check that target contains only 0 and 1
    std::set<int> unique_targets;
    for (int i = 0; i < target.size(); ++i) {
      unique_targets.insert(target[i]);
    }
    if (unique_targets.size() != 2 || unique_targets.find(0) == unique_targets.end() || unique_targets.find(1) == unique_targets.end()) {
      stop("Target vector must be binary (0 and 1).");
    }
    
    // Additional check: Ensure there are both positives and negatives
    int total_pos = 0;
    int total_neg = 0;
    for (int i = 0; i < target.size(); ++i) {
      if (target[i] == 1) {
        total_pos++;
      } else {
        total_neg++;
      }
    }
    if (total_pos == 0 || total_neg == 0) {
      stop("Target vector must contain both positive (1) and negative (0) cases.");
    }
  }
  
  // Compute quantiles for pre-binning
  std::vector<double> compute_quantiles(const std::vector<double>& sorted_feature, const std::vector<double>& probs) {
    int n = sorted_feature.size();
    std::vector<double> quantiles(probs.size());
    
    for (size_t i = 0; i < probs.size(); ++i) {
      double p = probs[i];
      double h = (n - 1) * p;
      int h_int = static_cast<int>(std::floor(h));
      double h_frac = h - h_int;
      
      if (h_int + 1 < n) {
        quantiles[i] = sorted_feature[h_int] + h_frac * (sorted_feature[h_int + 1] - sorted_feature[h_int]);
      } else {
        quantiles[i] = sorted_feature[n - 1];
      }
    }
    return quantiles;
  }
  
  // Initialize pre-bins based on quantiles
  void initialize_prebins(std::vector<double>& cut_points) {
    // Create max_n_prebins equally spaced quantiles
    std::vector<double> probs(max_n_prebins - 1);
    for (int i = 1; i < max_n_prebins; ++i) {
      probs[i - 1] = static_cast<double>(i) / max_n_prebins;
    }
    
    // Sort feature values
    std::vector<double> sorted_feature = Rcpp::as<std::vector<double>>(feature);
    std::sort(sorted_feature.begin(), sorted_feature.end(), compare_feature);
    
    // Compute quantiles
    std::vector<double> cuts = compute_quantiles(sorted_feature, probs);
    
    // Remove duplicate cut points
    std::set<double> unique_cuts(cuts.begin(), cuts.end());
    cut_points.assign(unique_cuts.begin(), unique_cuts.end());
    std::sort(cut_points.begin(), cut_points.end());
  }
  
  // Create initial bins based on cut points
  void create_initial_bins(const std::vector<double>& cut_points) {
    bins.clear();
    std::vector<double> boundaries = cut_points;
    boundaries.insert(boundaries.begin(), -std::numeric_limits<double>::infinity());
    boundaries.push_back(std::numeric_limits<double>::infinity());
    
    int n_bins = boundaries.size() - 1;
    bins.resize(n_bins);
    
    // Initialize bin boundaries in parallel
#pragma omp parallel for num_threads(n_threads)
    for (int i = 0; i < n_bins; ++i) {
      bins[i].lower_bound = boundaries[i];
      bins[i].upper_bound = boundaries[i + 1];
      bins[i].count = 0;
      bins[i].count_pos = 0;
      bins[i].count_neg = 0;
      bins[i].woe = 0.0;
      bins[i].iv = 0.0;
    }
    
    // Assign observations to bins using binary search
    int n = feature.size();
#pragma omp parallel for num_threads(n_threads)
    for (int i = 0; i < n; ++i) {
      double val = feature[i];
      // Binary search to find the appropriate bin
      int bin_index = std::upper_bound(boundaries.begin(), boundaries.end(), val) - boundaries.begin() - 1;
      // Clamp bin_index to valid range
      if (bin_index < 0) bin_index = 0;
      if (bin_index >= n_bins) bin_index = n_bins - 1;
      
      // Atomically update bin counts
#pragma omp atomic
      bins[bin_index].count += 1;
      
      if (target[i] == 1) {
#pragma omp atomic
        bins[bin_index].count_pos += 1;
      } else {
#pragma omp atomic
        bins[bin_index].count_neg += 1;
      }
    }
  }
  
  // Merge bins with counts below the cutoff
  void merge_small_bins() {
    int total_count = feature.size();
    double min_bin_count = bin_cutoff * total_count;
    
    bool bins_merged = true;
    while (bins_merged) {
      bins_merged = false;
      for (size_t i = 0; i < bins.size(); ++i) {
        if (bins[i].count < min_bin_count) {
          if (bins.size() <= static_cast<size_t>(min_bins)) {
            // Cannot merge further without violating min_bins
            break;
          }
          
          if (i == 0) {
            // Merge with next bin
            if (bins.size() < 2) break; // Safety check
            bins[i + 1].lower_bound = bins[i].lower_bound;
            bins[i + 1].count += bins[i].count;
            bins[i + 1].count_pos += bins[i].count_pos;
            bins[i + 1].count_neg += bins[i].count_neg;
            bins.erase(bins.begin() + i);
          } else {
            // Merge with previous bin
            bins[i - 1].upper_bound = bins[i].upper_bound;
            bins[i - 1].count += bins[i].count;
            bins[i - 1].count_pos += bins[i].count_pos;
            bins[i - 1].count_neg += bins[i].count_neg;
            bins.erase(bins.begin() + i);
          }
          
          // Recalculate WoE and IV after merging
          calculate_woe_iv();
          bins_merged = true;
          break; // Restart the loop after a merge
        }
      }
    }
  }
  
  // Calculate WoE and IV for each bin
  void calculate_woe_iv() {
    int total_pos = 0;
    int total_neg = 0;
    for (size_t i = 0; i < target.size(); ++i) {
      if (target[i] == 1) {
        total_pos++;
      } else {
        total_neg++;
      }
    }
    
    // Avoid division by zero
    if (total_pos == 0 || total_neg == 0) {
      stop("Cannot calculate WoE and IV because there are no positives or no negatives.");
    }
    
#pragma omp parallel for num_threads(n_threads)
    for (size_t i = 0; i < bins.size(); ++i) {
      double dist_pos = static_cast<double>(bins[i].count_pos) / total_pos;
      double dist_neg = static_cast<double>(bins[i].count_neg) / total_neg;
      
      // Handle zero distributions
      if (dist_pos == 0) dist_pos = 1e-8;
      if (dist_neg == 0) dist_neg = 1e-8;
      
      bins[i].woe = std::log(dist_pos / dist_neg);
      bins[i].iv = (dist_pos - dist_neg) * bins[i].woe;
    }
  }
  
  // Enforce monotonicity of WoE values
  void enforce_monotonicity() {
    if (bins.size() <= 1) return; // No need to enforce monotonicity
    
    bool monotonic = false;
    while (!monotonic) {
      monotonic = true;
      bool increasing = true;
      
      if (bins.size() >= 2) {
        increasing = bins[1].woe >= bins[0].woe;
      }
      
      for (size_t i = 1; i < bins.size(); ++i) {
        if ((increasing && bins[i].woe < bins[i - 1].woe) ||
            (!increasing && bins[i].woe > bins[i - 1].woe)) {
          // Merge bins[i-1] and bins[i]
          bins[i - 1].upper_bound = bins[i].upper_bound;
          bins[i - 1].count += bins[i].count;
          bins[i - 1].count_pos += bins[i].count_pos;
          bins[i - 1].count_neg += bins[i].count_neg;
          bins.erase(bins.begin() + i);
          // Recalculate WoE and IV after merging
          calculate_woe_iv();
          monotonic = false;
          break; // Restart the loop after a merge
        }
      }
    }
  }
  
  // Adjust the number of bins to be within [min_bins, max_bins]
  void adjust_bins() {
    // Reduce the number of bins if it exceeds max_bins
    while (bins.size() > static_cast<size_t>(max_bins)) {
      // Find the pair of adjacent bins with the smallest IV difference
      double min_iv_diff = std::numeric_limits<double>::max();
      size_t merge_index = 0;
      
      for (size_t i = 0; i < bins.size() - 1; ++i) {
        double iv_diff = std::abs(bins[i].iv - bins[i + 1].iv);
        if (iv_diff < min_iv_diff) {
          min_iv_diff = iv_diff;
          merge_index = i;
        }
      }
      
      // Merge bins[merge_index] and bins[merge_index + 1]
      bins[merge_index].upper_bound = bins[merge_index + 1].upper_bound;
      bins[merge_index].count += bins[merge_index + 1].count;
      bins[merge_index].count_pos += bins[merge_index + 1].count_pos;
      bins[merge_index].count_neg += bins[merge_index + 1].count_neg;
      bins.erase(bins.begin() + merge_index + 1);
      
      // Recalculate WoE and IV after merging
      calculate_woe_iv();
      // Re-enforce monotonicity after merging
      enforce_monotonicity();
    }
    
    // Increase the number of bins if it falls below min_bins
    // Note: Splitting bins is not implemented in this algorithm
    // Therefore, we ensure that after merging small bins, the number of bins is at least min_bins
    if (bins.size() < static_cast<size_t>(min_bins)) {
      warning("Number of bins is less than min_bins after merging small bins.");
      // Optionally, implement bin splitting here
    }
  }
  
public:
  // Constructor
  OptimalBinningNumericalEB(NumericVector feature, IntegerVector target,
                            int min_bins = 3, int max_bins = 5,
                            double bin_cutoff = 0.05, int max_n_prebins = 20,
                            int n_threads = 1)
    : feature(feature), target(target), min_bins(min_bins),
      max_bins(max_bins), bin_cutoff(bin_cutoff),
      max_n_prebins(max_n_prebins), n_threads(n_threads) {
    validate_inputs();
  }
  
  // Fit the binning model
  List fit() {
    std::vector<double> cut_points;
    initialize_prebins(cut_points);
    create_initial_bins(cut_points);
    merge_small_bins();
    enforce_monotonicity();
    adjust_bins();
    
    // Prepare output vectors
    NumericVector woefeature(feature.size(), NA_REAL);
    NumericVector woevalues(bins.size());
    NumericVector ivvalues(bins.size());
    CharacterVector bin_names(bins.size());
    IntegerVector counts(bins.size());
    IntegerVector count_pos(bins.size());
    IntegerVector count_neg(bins.size());
    
    // Assign WoE to feature values using binary search
    std::vector<double> boundaries;
    for (size_t i = 0; i < bins.size(); ++i) {
      boundaries.push_back(bins[i].upper_bound);
    }
    
#pragma omp parallel for num_threads(n_threads)
    for (int i = 0; i < feature.size(); ++i) {
      double val = feature[i];
      // Binary search to find the bin index
      int bin_index = std::upper_bound(boundaries.begin(), boundaries.end(), val) - boundaries.begin();
      bin_index = std::max(0, std::min(bin_index, static_cast<int>(bins.size()) - 1));
      woefeature[i] = bins[bin_index].woe;
    }
    
    // Prepare binning details
    for (size_t i = 0; i < bins.size(); ++i) {
      std::ostringstream oss;
      oss << "(";
      if (std::isinf(bins[i].lower_bound) && bins[i].lower_bound < 0) {
        oss << "-Inf";
      } else {
        oss << std::fixed << std::setprecision(6) << bins[i].lower_bound;
      }
      oss << ";";
      if (std::isinf(bins[i].upper_bound) && bins[i].upper_bound > 0) {
        oss << "+Inf";
      } else {
        oss << std::fixed << std::setprecision(6) << bins[i].upper_bound;
      }
      oss << "]";
      bin_names[i] = oss.str();
      
      woevalues[i] = bins[i].woe;
      ivvalues[i] = bins[i].iv;
      counts[i] = bins[i].count;
      count_pos[i] = bins[i].count_pos;
      count_neg[i] = bins[i].count_neg;
    }
    
    // Create DataFrame for binning details
    DataFrame woebin = DataFrame::create(
      Named("bin") = bin_names,
      Named("woe") = woevalues,
      Named("iv") = ivvalues,
      Named("count") = counts,
      Named("count_pos") = count_pos,
      Named("count_neg") = count_neg
    );
    
    return List::create(
      Named("woefeature") = woefeature,
      Named("woebin") = woebin
    );
  }
};

//' @title Optimal Binning for Numerical Variables using Entropy-Based Approach
//'
//' @description This function implements an optimal binning algorithm for numerical variables using an entropy-based approach. It creates bins that maximize the predictive power of the feature with respect to a binary target variable, while ensuring monotonicity of the Weight of Evidence (WoE) values.
//'
//' @param target An integer vector of binary target values (0 or 1).
//' @param feature A numeric vector of feature values to be binned.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum frequency of observations in each bin as a proportion of total observations (default: 0.05).
//' @param max_n_prebins Maximum number of initial bins before merging (default: 20).
//' @param n_threads Number of threads for parallel processing (default: 1).
//'
//' @return A list containing two elements:
//' \item{woefeature}{A numeric vector of WoE values assigned to each observation in the input feature.}
//' \item{woebin}{A data frame containing binning details, including bin boundaries, WoE values, Information Value (IV), and counts.}
//'
//' @details
//' The optimal binning algorithm for numerical variables using an entropy-based approach works as follows:
//' 1. Initial Binning: The algorithm starts by creating \code{max_n_prebins} equally spaced quantiles of the input feature.
//' 2. Merging Small Bins: Bins with a frequency below \code{bin_cutoff} are merged with adjacent bins to ensure statistical significance.
//' 3. Calculating WoE and IV: For each bin, the Weight of Evidence (WoE) and Information Value (IV) are calculated using the following formulas:
//'
//' \deqn{WoE_i = \ln\left(\frac{P(X_i|Y=1)}{P(X_i|Y=0)}\right)}
//' \deqn{IV_i = (P(X_i|Y=1) - P(X_i|Y=0)) \times WoE_i}
//'
//' where \eqn{X_i} represents the i-th bin and \eqn{Y} is the binary target variable.
//' 4. Enforcing Monotonicity: The algorithm ensures that WoE values are monotonic across bins by merging adjacent bins that violate this condition.
//' 5. Adjusting Bin Count: If the number of bins exceeds \code{max_bins}, the algorithm merges bins with the smallest total count until the desired number of bins is achieved.
//' 6. Final Output: The algorithm assigns WoE values to each observation in the input feature and provides detailed binning information.
//'
//' This approach aims to maximize the predictive power of the feature while maintaining interpretability and robustness of the binning process.
//'
//' @examples
//' \dontrun{
//' # Generate sample data
//' set.seed(42)
//' target <- sample(0:1, 1000, replace = TRUE)
//' feature <- rnorm(1000)
//' # Run optimal binning
//' result <- optimal_binning_numerical_eb(target, feature)
//' # View WoE-transformed feature
//' head(result$woefeature)
//' # View binning details
//' print(result$woebin)
//' }
//'
//' @references
//' \itemize{
//'   \item Beltratti, A., Margarita, S., & Terna, P. (1996). Neural networks for economic and financial modelling. International Thomson Computer Press.
//'   \item Kotsiantis, S., & Kanellopoulos, D. (2006). Discretization techniques: A recent survey. GESTS International Transactions on Computer Science and Engineering, 32(1), 47-58.
//' }
//'
//' @author Lopes, J. E.
//' @export
// [[Rcpp::export]]
List optimal_binning_numerical_eb(IntegerVector target, NumericVector feature,
                                  int min_bins = 3, int max_bins = 5,
                                  double bin_cutoff = 0.05, int max_n_prebins = 20,
                                  int n_threads = 1) {
  // Ensure the number of threads is at least 1
  if (n_threads < 1) n_threads = 1;
  
  // Convert Rcpp vectors to standard C++ vectors for efficiency
  std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
  std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
  
  // Create an instance of the OptimalBinningNumericalEB class
  OptimalBinningNumericalEB ob(feature, target,
                               min_bins, max_bins, bin_cutoff, max_n_prebins,
                               n_threads);
  // Fit the binning model
  return ob.fit();
}





// // [[Rcpp::plugins(cpp11)]]
// // [[Rcpp::plugins(openmp)]]
// #include <Rcpp.h>
// #include <vector>
// #include <algorithm>
// #include <cmath>
// 
// #ifdef _OPENMP
// #include <omp.h>
// #endif
// 
// using namespace Rcpp;
// 
// class OptimalBinningNumericalEB {
// private:
//   NumericVector feature;
//   IntegerVector target;
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
//   int n_threads;
// 
//   struct Bin {
//     double lower_bound;
//     double upper_bound;
//     int count;
//     int count_pos;
//     int count_neg;
//     double woe;
//     double iv;
//   };
// 
//   std::vector<Bin> bins;
// 
//   void validate_inputs() {
//     if (feature.size() != target.size()) {
//       stop("Feature and target vectors must be of the same length.");
//     }
// 
//     if (min_bins < 2) {
//       stop("min_bins must be at least 2.");
//     }
// 
//     if (max_bins < min_bins) {
//       stop("max_bins must be greater than or equal to min_bins.");
//     }
// 
//     if (bin_cutoff <= 0 || bin_cutoff >= 0.5) {
//       stop("bin_cutoff must be between 0 and 0.5.");
//     }
// 
//     if (max_n_prebins < max_bins) {
//       stop("max_n_prebins must be greater than or equal to max_bins.");
//     }
// 
//     if (is_true(any(is_na(feature)))) {
//       stop("Feature vector contains NA values.");
//     }
// 
//     if (is_true(any(is_na(target)))) {
//       stop("Target vector contains NA values.");
//     }
// 
//     IntegerVector unique_targets = sort_unique(target);
//     if (unique_targets.size() != 2 || unique_targets[0] != 0 || unique_targets[1] != 1) {
//       stop("Target vector must be binary (0 and 1).");
//     }
//   }
// 
//   NumericVector compute_quantiles(NumericVector x, NumericVector probs) {
//     int n = x.size();
//     NumericVector x_sorted = clone(x).sort();
//     NumericVector quantiles(probs.size());
// 
//     for (int i = 0; i < probs.size(); ++i) {
//       double p = probs[i];
//       double h = (n - 1) * p;
//       int h_int = (int)std::floor(h);
//       double h_frac = h - h_int;
// 
//       if (h_int + 1 < n) {
//         quantiles[i] = x_sorted[h_int] + h_frac * (x_sorted[h_int + 1] - x_sorted[h_int]);
//       } else {
//         quantiles[i] = x_sorted[n - 1];
//       }
//     }
//     return quantiles;
//   }
// 
//   void initialize_prebins(std::vector<double>& cut_points) {
//     // Create max_n_prebins equally spaced quantiles
//     NumericVector probs(max_n_prebins - 1);
//     for (int i = 1; i < max_n_prebins; ++i) {
//       probs[i - 1] = (double)i / max_n_prebins;
//     }
// 
//     // Compute quantiles
//     NumericVector cuts = compute_quantiles(feature, probs);
// 
//     // Remove duplicate cut points
//     std::set<double> unique_cuts(cuts.begin(), cuts.end());
//     cut_points.assign(unique_cuts.begin(), unique_cuts.end());
//     std::sort(cut_points.begin(), cut_points.end());
//   }
// 
//   void create_initial_bins(const std::vector<double>& cut_points) {
//     // Initialize bins based on cut points
//     bins.clear();
//     std::vector<double> boundaries = cut_points;
//     boundaries.insert(boundaries.begin(), R_NegInf);
//     boundaries.push_back(R_PosInf);
// 
//     int n_bins = boundaries.size() - 1;
//     bins.resize(n_bins);
// 
//     // Initialize bin counts
// #pragma omp parallel for num_threads(n_threads)
//     for (int i = 0; i < n_bins; ++i) {
//       bins[i].lower_bound = boundaries[i];
//       bins[i].upper_bound = boundaries[i + 1];
//       bins[i].count = 0;
//       bins[i].count_pos = 0;
//       bins[i].count_neg = 0;
//       bins[i].woe = 0.0;
//       bins[i].iv = 0.0;
//     }
// 
//     // Assign observations to bins
//     int n = feature.size();
// #pragma omp parallel for num_threads(n_threads)
//     for (int i = 0; i < n; ++i) {
//       double val = feature[i];
//       int bin_index = -1;
//       // Find the bin for the current value
//       for (int j = 0; j < n_bins; ++j) {
//         if (val > bins[j].lower_bound && val <= bins[j].upper_bound) {
//           bin_index = j;
//           break;
//         }
//       }
//       if (bin_index != -1) {
// #pragma omp atomic
//         bins[bin_index].count += 1;
// 
//         if (target[i] == 1) {
// #pragma omp atomic
//           bins[bin_index].count_pos += 1;
//         } else {
// #pragma omp atomic
//           bins[bin_index].count_neg += 1;
//         }
//       }
//     }
//   }
// 
//   void merge_small_bins() {
//     // Merge bins with frequency below bin_cutoff
//     int total_count = feature.size();
//     double min_bin_count = bin_cutoff * total_count;
// 
//     bool bins_merged = true;
//     while (bins_merged) {
//       bins_merged = false;
//       for (size_t i = 0; i < bins.size(); ++i) {
//         if (bins[i].count < min_bin_count) {
//           if (i == 0) {
//             // Merge with next bin
//             bins[i + 1].lower_bound = bins[i].lower_bound;
//             bins[i + 1].count += bins[i].count;
//             bins[i + 1].count_pos += bins[i].count_pos;
//             bins[i + 1].count_neg += bins[i].count_neg;
//             bins.erase(bins.begin() + i);
//           } else {
//             // Merge with previous bin
//             bins[i - 1].upper_bound = bins[i].upper_bound;
//             bins[i - 1].count += bins[i].count;
//             bins[i - 1].count_pos += bins[i].count_pos;
//             bins[i - 1].count_neg += bins[i].count_neg;
//             bins.erase(bins.begin() + i);
//           }
//           bins_merged = true;
//           break;
//         }
//       }
//     }
//   }
// 
//   void calculate_woe_iv() {
//     int total_pos = std::accumulate(target.begin(), target.end(), 0);
//     int total_neg = target.size() - total_pos;
// 
// #pragma omp parallel for num_threads(n_threads)
//     for (size_t i = 0; i < bins.size(); ++i) {
//       double dist_pos = (double)bins[i].count_pos / total_pos;
//       double dist_neg = (double)bins[i].count_neg / total_neg;
// 
//       // Avoid division by zero
//       if (dist_pos == 0) dist_pos = 1e-8;
//       if (dist_neg == 0) dist_neg = 1e-8;
// 
//       bins[i].woe = std::log(dist_pos / dist_neg);
//       bins[i].iv = (dist_pos - dist_neg) * bins[i].woe;
//     }
//   }
// 
//   void enforce_monotonicity() {
//     // Ensure that WoE is monotonic across bins
//     bool monotonic = false;
//     while (!monotonic) {
//       monotonic = true;
//       for (size_t i = 1; i < bins.size(); ++i) {
//         if (bins[i - 1].woe > bins[i].woe) {
//           // Merge bins i-1 and i
//           bins[i - 1].upper_bound = bins[i].upper_bound;
//           bins[i - 1].count += bins[i].count;
//           bins[i - 1].count_pos += bins[i].count_pos;
//           bins[i - 1].count_neg += bins[i].count_neg;
//           bins.erase(bins.begin() + i);
//           calculate_woe_iv();
//           monotonic = false;
//           break;
//         }
//       }
//     }
//   }
// 
//   void adjust_bins() {
//     // Ensure number of bins is within min_bins and max_bins
//     while ((int)bins.size() > max_bins) {
//       // Merge bins with smallest total count
//       size_t merge_index = 0;
//       int min_count = bins[0].count + bins[1].count;
//       for (size_t i = 1; i < bins.size() - 1; ++i) {
//         int combined_count = bins[i].count + bins[i + 1].count;
//         if (combined_count < min_count) {
//           min_count = combined_count;
//           merge_index = i;
//         }
//       }
//       // Merge bins at merge_index and merge_index + 1
//       bins[merge_index].upper_bound = bins[merge_index + 1].upper_bound;
//       bins[merge_index].count += bins[merge_index + 1].count;
//       bins[merge_index].count_pos += bins[merge_index + 1].count_pos;
//       bins[merge_index].count_neg += bins[merge_index + 1].count_neg;
//       bins.erase(bins.begin() + merge_index + 1);
//       calculate_woe_iv();
//       enforce_monotonicity();
//     }
// 
//     while ((int)bins.size() < min_bins) {
//       // Cannot split bins further; break the loop
//       break;
//     }
//   }
// 
// public:
//   OptimalBinningNumericalEB(NumericVector feature, IntegerVector target,
//                             int min_bins = 3, int max_bins = 5,
//                             double bin_cutoff = 0.05, int max_n_prebins = 20,
//                             int n_threads = 1)
//     : feature(feature), target(target), min_bins(min_bins),
//       max_bins(max_bins), bin_cutoff(bin_cutoff),
//       max_n_prebins(max_n_prebins), n_threads(n_threads) {
//     validate_inputs();
//   }
// 
//   List fit() {
//     std::vector<double> cut_points;
//     initialize_prebins(cut_points);
//     create_initial_bins(cut_points);
//     merge_small_bins();
//     calculate_woe_iv();
//     enforce_monotonicity();
//     adjust_bins();
// 
//     // Prepare output
//     NumericVector woefeature(feature.size(), NA_REAL);
//     NumericVector woevalues(bins.size());
//     NumericVector ivvalues(bins.size());
//     CharacterVector bin_names(bins.size());
//     IntegerVector counts(bins.size());
//     IntegerVector count_pos(bins.size());
//     IntegerVector count_neg(bins.size());
// 
//     // Assign WoE to feature values
//     int n = feature.size();
// #pragma omp parallel for num_threads(n_threads)
//     for (int i = 0; i < n; ++i) {
//       double val = feature[i];
//       for (size_t j = 0; j < bins.size(); ++j) {
//         if (val > bins[j].lower_bound && val <= bins[j].upper_bound) {
//           woefeature[i] = bins[j].woe;
//           break;
//         }
//       }
//     }
// 
//     // Prepare binning details
//     for (size_t i = 0; i < bins.size(); ++i) {
//       std::string bin_name = "(";
//       if (std::isinf(bins[i].lower_bound)) {
//         bin_name += "-Inf";
//       } else {
//         bin_name += std::to_string(bins[i].lower_bound);
//       }
//       bin_name += ";";
//       if (std::isinf(bins[i].upper_bound)) {
//         bin_name += "+Inf";
//       } else {
//         bin_name += std::to_string(bins[i].upper_bound);
//       }
//       bin_name += "]";
//       bin_names[i] = bin_name;
//       woevalues[i] = bins[i].woe;
//       ivvalues[i] = bins[i].iv;
//       counts[i] = bins[i].count;
//       count_pos[i] = bins[i].count_pos;
//       count_neg[i] = bins[i].count_neg;
//     }
// 
//     DataFrame woebin = DataFrame::create(
//       Named("bin") = bin_names,
//       Named("woe") = woevalues,
//       Named("iv") = ivvalues,
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
// };
// 
// //' @title Optimal Binning for Numerical Variables using Entropy-Based Approach
// //' 
// //' @description This function implements an optimal binning algorithm for numerical variables using an entropy-based approach. It creates bins that maximize the predictive power of the feature with respect to a binary target variable, while ensuring monotonicity of the Weight of Evidence (WoE) values.
// //' 
// //' @param target An integer vector of binary target values (0 or 1).
// //' @param feature A numeric vector of feature values to be binned.
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff Minimum frequency of observations in each bin as a proportion of total observations (default: 0.05).
// //' @param max_n_prebins Maximum number of initial bins before merging (default: 20).
// //' @param n_threads Number of threads for parallel processing (default: 1).
// //' 
// //' @return A list containing two elements:
// //' \item{woefeature}{A numeric vector of WoE values assigned to each observation in the input feature.}
// //' \item{woebin}{A data frame containing binning details, including bin boundaries, WoE values, Information Value (IV), and counts.}
// //' 
// //' @details
// //' The optimal binning algorithm for numerical variables using an entropy-based approach works as follows:
// //' 1. Initial Binning: The algorithm starts by creating \code{max_n_prebins} equally spaced quantiles of the input feature.
// //' 2. Merging Small Bins: Bins with a frequency below \code{bin_cutoff} are merged with adjacent bins to ensure statistical significance.
// //' 3. Calculating WoE and IV: For each bin, the Weight of Evidence (WoE) and Information Value (IV) are calculated using the following formulas:
// //' 
// //' \deqn{WoE_i = \ln\left(\frac{P(X_i|Y=1)}{P(X_i|Y=0)}\right)}
// //' \deqn{IV_i = (P(X_i|Y=1) - P(X_i|Y=0)) \times WoE_i}
// //' 
// //' where \eqn{X_i} represents the i-th bin and \eqn{Y} is the binary target variable.
// //' 4. Enforcing Monotonicity: The algorithm ensures that WoE values are monotonic across bins by merging adjacent bins that violate this condition.
// //' 5. Adjusting Bin Count: If the number of bins exceeds \code{max_bins}, the algorithm merges bins with the smallest total count until the desired number of bins is achieved.
// //' 6. Final Output: The algorithm assigns WoE values to each observation in the input feature and provides detailed binning information.
// //' 
// //' This approach aims to maximize the predictive power of the feature while maintaining interpretability and robustness of the binning process.
// //' 
// //' @examples
// //' \dontrun{
// //' # Generate sample data
// //' set.seed(42)
// //' target <- sample(0:1, 1000, replace = TRUE)
// //' feature <- rnorm(1000)
// //' # Run optimal binning
// //' result <- optimal_binning_numerical_eb(target, feature)
// //' # View WoE-transformed feature
// //' head(result$woefeature)
// //' # View binning details
// //' print(result$woebin)
// //' }
// //' 
// //' @references
// //' \itemize{
// //'   \item Beltratti, A., Margarita, S., & Terna, P. (1996). Neural networks for economic and financial modelling. International Thomson Computer Press.
// //'   \item Kotsiantis, S., & Kanellopoulos, D. (2006). Discretization techniques: A recent survey. GESTS International Transactions on Computer Science and Engineering, 32(1), 47-58.
// //' }
// //' 
// //' @author Lopes, J. E.
// //' @export
// // [[Rcpp::export]]
// List optimal_binning_numerical_eb(IntegerVector target, NumericVector feature,
//                                   int min_bins = 3, int max_bins = 5,
//                                   double bin_cutoff = 0.05, int max_n_prebins = 20,
//                                   int n_threads = 1) {
//   OptimalBinningNumericalEB ob(feature, target, min_bins, max_bins, bin_cutoff, max_n_prebins, n_threads);
//   return ob.fit();
// }
