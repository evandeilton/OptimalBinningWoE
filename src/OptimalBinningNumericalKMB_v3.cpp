// [[Rcpp::plugins(cpp11)]]

#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#include <string>
#include <sstream>
#include <unordered_set>

using namespace Rcpp;

// Class for Optimal Binning using K-means Binning (KMB)
class OptimalBinningNumericalKMB {
private:
  std::vector<double> feature;
  std::vector<int> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  double convergence_threshold;
  int max_iterations;
  
  bool converged;
  int iterations_run;
  
  bool is_unique_two_or_less; // Flag to indicate if unique values <= 2
  
  struct Bin {
    double lower_bound;
    double upper_bound;
    int count;
    int count_pos;
    int count_neg;
    double woe;
    double iv;
  };
  
  std::vector<Bin> bins;
  
  // Calculate Weight of Evidence with Laplace smoothing
  double calculateWOE(int pos, int neg, int total_pos, int total_neg) const {
    double pos_rate = (static_cast<double>(pos) + 0.5) / (static_cast<double>(total_pos) + 1.0);
    double neg_rate = (static_cast<double>(neg) + 0.5) / (static_cast<double>(total_neg) + 1.0);
    return std::log(pos_rate / neg_rate);
  }
  
  // Calculate Information Value
  double calculateIV(double woe, int pos, int neg, int total_pos, int total_neg) const {
    double pos_dist = static_cast<double>(pos) / static_cast<double>(total_pos);
    double neg_dist = static_cast<double>(neg) / static_cast<double>(total_neg);
    return (pos_dist - neg_dist) * woe;
  }
  
  // Initial Binning based on unique values and pre-bins
  void initialBinning() {
    // Extract unique sorted values
    std::vector<double> unique_values = feature;
    std::sort(unique_values.begin(), unique_values.end());
    unique_values.erase(std::unique(unique_values.begin(), unique_values.end()), unique_values.end());
    
    int num_unique_values = (int)unique_values.size();
    
    if (num_unique_values <= 2) {
      // Do not optimize or create extra bins; create bins based on unique values
      is_unique_two_or_less = true;
      bins.clear();
      bins.reserve((size_t)num_unique_values);
      for (int i = 0; i < num_unique_values; ++i) {
        double lower = (i == 0) ? -std::numeric_limits<double>::infinity() : unique_values[(size_t)i - 1];
        double upper = unique_values[(size_t)i];
        bins.push_back(Bin{lower, upper, 0, 0, 0, 0.0, 0.0});
      }
      bins.back().upper_bound = std::numeric_limits<double>::infinity();
    } else {
      // Proceed with existing binning logic
      is_unique_two_or_less = false;
      // Determine number of initial bins
      int n_bins = std::min(max_n_prebins, num_unique_values);
      n_bins = std::max(n_bins, min_bins);
      n_bins = std::min(n_bins, max_bins);
      
      // Determine bin boundaries
      std::vector<double> boundaries;
      for (int i = 1; i < n_bins; ++i) {
        int index = i * (num_unique_values) / n_bins;
        boundaries.push_back(unique_values[(size_t)index]);
      }
      
      // Initialize bins
      bins.clear();
      bins.reserve((size_t)n_bins);
      double lower = -std::numeric_limits<double>::infinity();
      for (size_t i = 0; i <= boundaries.size(); ++i) {
        double upper = (i == boundaries.size()) ? std::numeric_limits<double>::infinity() : boundaries[i];
        bins.push_back(Bin{lower, upper, 0, 0, 0, 0.0, 0.0});
        lower = upper;
      }
    }
  }
  
  // Assign data points to bins
  void assignDataToBins() {
    for (size_t i = 0; i < feature.size(); ++i) {
      double value = feature[i];
      int target_value = target[i];
      bool assigned = false;
      for (auto& bin : bins) {
        if (value > bin.lower_bound && value <= bin.upper_bound) {
          bin.count++;
          if (target_value == 1) {
            bin.count_pos++;
          } else {
            bin.count_neg++;
          }
          assigned = true;
          break;
        }
      }
      if (!assigned) {
        // Handle edge cases
        if (value <= bins.front().lower_bound) {
          bins.front().count++;
          if (target_value == 1) {
            bins.front().count_pos++;
          } else {
            bins.front().count_neg++;
          }
        } else if (value > bins.back().upper_bound) {
          bins.back().count++;
          if (target_value == 1) {
            bins.back().count_pos++;
          } else {
            bins.back().count_neg++;
          }
        }
      }
    }
  }
  
  // Merge bins with low frequency based on bin_cutoff
  void mergeLowFrequencyBins() {
    int total_count = (int)feature.size();
    double cutoff_count = bin_cutoff * total_count;
    
    int iterations = 0;
    while (iterations < max_iterations) {
      bool merged = false;
      for (size_t i = 0; i < bins.size(); ++i) {
        if (bins[i].count < cutoff_count && (int)bins.size() > min_bins) {
          if (i == 0) {
            // Merge with next bin
            bins[i].upper_bound = bins[i + 1].upper_bound;
            bins[i].count += bins[i + 1].count;
            bins[i].count_pos += bins[i + 1].count_pos;
            bins[i].count_neg += bins[i + 1].count_neg;
            bins.erase(bins.begin() + i + 1);
          } else {
            // Merge with previous bin
            bins[i - 1].upper_bound = bins[i].upper_bound;
            bins[i - 1].count += bins[i].count;
            bins[i - 1].count_pos += bins[i].count_pos;
            bins[i - 1].count_neg += bins[i].count_neg;
            bins.erase(bins.begin() + i);
          }
          merged = true;
          break;
        }
      }
      if (!merged) {
        break;
      }
      iterations++;
    }
    iterations_run += iterations;
    if (iterations >= max_iterations) {
      converged = false;
    }
  }
  
  // Enforce monotonicity of WoE values
  void enforceMonotonicity() {
    if (bins.size() <= 2) {
      // If feature has two or fewer bins, ignore monotonicity enforcement
      return;
    }
    
    int iterations = 0;
    bool is_monotonic = false;
    bool increasing = true;
    if (bins.size() >= 2) {
      increasing = (bins[1].woe >= bins[0].woe);
    }
    while (!is_monotonic && (int)bins.size() > min_bins && iterations < max_iterations) {
      is_monotonic = true;
      for (size_t i = 1; i < bins.size(); ++i) {
        if ((increasing && bins[i].woe < bins[i - 1].woe) ||
            (!increasing && bins[i].woe > bins[i - 1].woe)) {
          // Merge bins[i - 1] and bins[i]
          bins[i - 1].upper_bound = bins[i].upper_bound;
          bins[i - 1].count += bins[i].count;
          bins[i - 1].count_pos += bins[i].count_pos;
          bins[i - 1].count_neg += bins[i].count_neg;
          bins.erase(bins.begin() + i);
          calculateBinStatistics();
          is_monotonic = false;
          break;
        }
      }
      iterations++;
      if (bins.size() == min_bins) {
        // min_bins reached, stop merging even if monotonicity is not achieved
        break;
      }
    }
    iterations_run += iterations;
    if (iterations >= max_iterations) {
      converged = false;
    }
  }
  
  // Adjust bin count to be within [min_bins, max_bins]
  void adjustBinCount() {
    int iterations = 0;
    while ((int)bins.size() > max_bins && iterations < max_iterations) {
      // Find the pair of adjacent bins with the smallest IV difference
      double min_iv_diff = std::numeric_limits<double>::max();
      int merge_index = -1;
      for (size_t i = 0; i < bins.size() - 1; ++i) {
        double diff = std::abs(bins[i].iv - bins[i + 1].iv);
        if (diff < min_iv_diff) {
          min_iv_diff = diff;
          merge_index = (int)i;
        }
      }
      if (merge_index != -1) {
        // Merge bins at merge_index and merge_index + 1
        bins[merge_index].upper_bound = bins[merge_index + 1].upper_bound;
        bins[merge_index].count += bins[merge_index + 1].count;
        bins[merge_index].count_pos += bins[merge_index + 1].count_pos;
        bins[merge_index].count_neg += bins[merge_index + 1].count_neg;
        bins.erase(bins.begin() + merge_index + 1);
        calculateBinStatistics();
      } else {
        break; // No more bins can be merged
      }
      iterations++;
    }
    iterations_run += iterations;
    if (iterations >= max_iterations) {
      converged = false;
    }
  }
  
  // Calculate WoE and IV for each bin
  void calculateBinStatistics() {
    int total_pos = 0;
    int total_neg = 0;
    
    for (const auto& bin : bins) {
      total_pos += bin.count_pos;
      total_neg += bin.count_neg;
    }
    
    // Handle cases where total_pos or total_neg is zero
    if (total_pos == 0 || total_neg == 0) {
      Rcpp::stop("Target vector must contain both positive and negative cases.");
    }
    
    for (auto& bin : bins) {
      bin.woe = calculateWOE(bin.count_pos, bin.count_neg, total_pos, total_neg);
      bin.iv = calculateIV(bin.woe, bin.count_pos, bin.count_neg, total_pos, total_neg);
    }
  }
  
  // Format bin intervals as strings
  std::string formatBinInterval(double lower, double upper) const {
    std::ostringstream oss;
    oss << "(";
    if (lower == -std::numeric_limits<double>::infinity()) {
      oss << "-Inf";
    } else {
      oss << lower;
    }
    oss << ";";
    if (upper == std::numeric_limits<double>::infinity()) {
      oss << "+Inf";
    } else {
      oss << upper;
    }
    oss << "]";
    return oss.str();
  }
  
public:
  OptimalBinningNumericalKMB(const std::vector<double>& feature_, const std::vector<int>& target_,
                             int min_bins_, int max_bins_, double bin_cutoff_, int max_n_prebins_,
                             double convergence_threshold_, int max_iterations_)
    : feature(feature_), target(target_), min_bins(min_bins_), max_bins(max_bins_),
      bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_),
      convergence_threshold(convergence_threshold_), max_iterations(max_iterations_),
      converged(true), iterations_run(0), is_unique_two_or_less(false) {}
  
  Rcpp::List fit() {
    // Input validation
    if (feature.empty() || target.empty()) {
      Rcpp::stop("Feature and target vectors must not be empty.");
    }
    
    if (feature.size() != target.size()) {
      Rcpp::stop("Feature and target vectors must have the same length.");
    }
    
    // Ensure target contains only 0 and 1
    std::unordered_set<int> target_set(target.begin(), target.end());
    if (target_set.size() > 2 || (target_set.find(0) == target_set.end() && target_set.find(1) == target_set.end())) {
      Rcpp::stop("Target vector must contain only binary values 0 and 1.");
    }
    
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
    
    if (max_iterations <= 0) {
      Rcpp::stop("max_iterations must be positive.");
    }
    
    // Initial Binning
    initialBinning();
    
    // Assign data to bins
    assignDataToBins();
    
    if (is_unique_two_or_less) {
      // Calculate WoE and IV
      calculateBinStatistics();
      
      // Prepare output vectors
      std::vector<std::string> bin_labels;
      std::vector<double> woe_values;
      std::vector<double> iv_values;
      std::vector<int> counts;
      std::vector<int> counts_pos;
      std::vector<int> counts_neg;
      std::vector<double> cutpoints;
      
      for (size_t i = 0; i < bins.size(); ++i) {
        const auto& bin = bins[i];
        bin_labels.push_back(formatBinInterval(bin.lower_bound, bin.upper_bound));
        woe_values.push_back(bin.woe);
        iv_values.push_back(bin.iv);
        counts.push_back(bin.count);
        counts_pos.push_back(bin.count_pos);
        counts_neg.push_back(bin.count_neg);
        if (i < bins.size() - 1) {
          cutpoints.push_back(bin.upper_bound);
        }
      }
      
      Rcpp::NumericVector ids(bin_labels.size());
      for(int i = 0; i < bin_labels.size(); i++) {
        ids[i] = i + 1;
      }
      
      return Rcpp::List::create(
        Rcpp::Named("id") = ids,
        Rcpp::Named("bin") = bin_labels,
        Rcpp::Named("woe") = woe_values,
        Rcpp::Named("iv") = iv_values,
        Rcpp::Named("count") = counts,
        Rcpp::Named("count_pos") = counts_pos,
        Rcpp::Named("count_neg") = counts_neg,
        Rcpp::Named("cutpoints") = cutpoints,
        Rcpp::Named("converged") = converged,
        Rcpp::Named("iterations") = iterations_run
      );
    }
    
    // Continue with optimization steps
    // Calculate initial WoE and IV
    calculateBinStatistics();
    
    // Merge low frequency bins
    mergeLowFrequencyBins();
    
    // Enforce monotonicity
    enforceMonotonicity();
    
    // Adjust bin count to be within [min_bins, max_bins]
    adjustBinCount();
    
    // Final calculation of WoE and IV
    calculateBinStatistics();
    
    // Prepare output vectors
    std::vector<std::string> bin_labels;
    std::vector<double> woe_values;
    std::vector<double> iv_values;
    std::vector<int> counts;
    std::vector<int> counts_pos;
    std::vector<int> counts_neg;
    std::vector<double> cutpoints;
    
    for (size_t i = 0; i < bins.size(); ++i) {
      const auto& bin = bins[i];
      bin_labels.push_back(formatBinInterval(bin.lower_bound, bin.upper_bound));
      woe_values.push_back(bin.woe);
      iv_values.push_back(bin.iv);
      counts.push_back(bin.count);
      counts_pos.push_back(bin.count_pos);
      counts_neg.push_back(bin.count_neg);
      if (i < bins.size() - 1) {
        cutpoints.push_back(bin.upper_bound);
      }
    }
    
    return Rcpp::List::create(
      Rcpp::Named("bin") = bin_labels,
      Rcpp::Named("woe") = woe_values,
      Rcpp::Named("iv") = iv_values,
      Rcpp::Named("count") = counts,
      Rcpp::Named("count_pos") = counts_pos,
      Rcpp::Named("count_neg") = counts_neg,
      Rcpp::Named("cutpoints") = cutpoints,
      Rcpp::Named("converged") = converged,
      Rcpp::Named("iterations") = iterations_run
    );
  }
};



//' @title Optimal Binning for Numerical Variables using K-means Binning (KMB)
//'
//' @description This function implements the K-means Binning (KMB) algorithm for optimal binning of numerical variables.
//'
//' @param target An integer vector of binary target values (0 or 1).
//' @param feature A numeric vector of feature values to be binned.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum frequency for a bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins (default: 20).
//' @param convergence_threshold Convergence threshold for the algorithm (default: 1e-6).
//' @param max_iterations Maximum number of iterations allowed (default: 1000).
//'
//' @return A list containing the following elements:
//' \item{bin}{Character vector of bin ranges.}
//' \item{woe}{Numeric vector of WoE values for each bin.}
//' \item{iv}{Numeric vector of Information Value (IV) for each bin.}
//' \item{count}{Integer vector of total observations in each bin.}
//' \item{count_pos}{Integer vector of positive target observations in each bin.}
//' \item{count_neg}{Integer vector of negative target observations in each bin.}
//' \item{cutpoints}{Numeric vector of cut points to generate the bins.}
//' \item{converged}{Logical indicating if the algorithm converged.}
//' \item{iterations}{Integer number of iterations run by the algorithm.}
//'
//' @details
//' The K-means Binning (KMB) algorithm is an advanced method for optimal binning of numerical variables.
//' It combines elements of k-means clustering with traditional binning techniques to create bins that maximize
//' the predictive power of the feature while respecting user-defined constraints.
//'
//' The algorithm works through several steps:
//' 1. Initial Binning: Creates initial bins based on the unique values of the feature, respecting the max_n_prebins constraint.
//' 2. Data Assignment: Assigns data points to the appropriate bins.
//' 3. Low Frequency Merging: Merges bins with frequencies below the bin_cutoff threshold.
//' 4. Enforce Monotonicity: Merges bins to ensure that the WoE values are monotonic.
//' 5. Bin Count Adjustment: Adjusts the number of bins to fall within the specified range (min_bins to max_bins).
//' 6. Statistics Calculation: Computes Weight of Evidence (WoE) and Information Value (IV) for each bin.
//'
//' The KMB method uses a modified version of the Weight of Evidence (WoE) calculation that incorporates Laplace smoothing
//' to handle cases with zero counts
//' \deqn{WoE_i = \ln\left(\frac{(n_{1i} + 0.5) / (N_1 + 1)}{(n_{0i} + 0.5) / (N_0 + 1)}\right)}
//' where \eqn{n_{1i}} and \eqn{n_{0i}} are the number of events and non-events in bin i,
//' and \eqn{N_1} and \eqn{N_0} are the total number of events and non-events.
//'
//' The Information Value (IV) for each bin is calculated as:
//' \deqn{IV_i = \left(\frac{n_{1i}}{N_1} - \frac{n_{0i}}{N_0}\right) \times WoE_i}
//'
//' The KMB method aims to create bins that maximize the overall IV while respecting the user-defined constraints.
//' It uses a greedy approach to merge bins when necessary, choosing to merge bins with the smallest difference in IV.
//'
//' When adjusting the number of bins, the algorithm either merges bins with the most similar IVs (if there are too many bins)
//' or stops merging when min_bins is reached, even if monotonicity is not achieved.
//'
//' @examples
//' \dontrun{
//'   # Create sample data
//'   set.seed(123)
//'   target <- sample(0:1, 1000, replace = TRUE)
//'   feature <- rnorm(1000)
//'
//'   # Run optimal binning
//'   result <- optimal_binning_numerical_kmb(target, feature)
//'
//'   # View results
//'   print(result)
//' }
//'
//' @references
//' \itemize{
//' \item Fayyad, U., & Irani, K. (1993). Multi-interval discretization of continuous-valued attributes for classification learning. In Proceedings of the 13th International Joint Conference on Artificial Intelligence (pp. 1022-1027).
//' \item Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). Credit Scoring and Its Applications. SIAM Monographs on Mathematical Modeling and Computation.
//' }
//'
//' @export
// [[Rcpp::export]]
List optimal_binning_numerical_kmb(IntegerVector target,
                                  NumericVector feature,
                                  int min_bins = 3, int max_bins = 5,
                                  double bin_cutoff = 0.05, int max_n_prebins = 20,
                                  double convergence_threshold = 1e-6, int max_iterations = 1000) {
 // Convert R vectors to C++ vectors
 std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
 std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
 
 // Initialize binning class
 OptimalBinningNumericalKMB binner(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins,
                                   convergence_threshold, max_iterations);
 
 // Perform binning
 Rcpp::List result = binner.fit();
 
 return result;
}



// #include <Rcpp.h>
// #include <algorithm>
// #include <vector>
// #include <cmath>
// #include <limits>
// #include <string>
// #include <sstream>
// #include <unordered_set>
// 
// // [[Rcpp::plugins(cpp11)]]
// using namespace Rcpp;
// 
// // Class for Optimal Binning using K-means Binning (KMB)
// class OptimalBinningNumericalKMB {
// private:
//   std::vector<double> feature;
//   std::vector<int> target;
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
//   bool is_unique_two_or_less; // Flag to indicate if unique values <= 2
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
//   // Calculate Weight of Evidence with Laplace smoothing
//   double calculateWOE(int pos, int neg, int total_pos, int total_neg) const {
//     double pos_rate = (static_cast<double>(pos) + 0.5) / (static_cast<double>(total_pos) + 1.0);
//     double neg_rate = (static_cast<double>(neg) + 0.5) / (static_cast<double>(total_neg) + 1.0);
//     return std::log(pos_rate / neg_rate);
//   }
//   
//   // Calculate Information Value
//   double calculateIV(double woe, int pos, int neg, int total_pos, int total_neg) const {
//     double pos_dist = static_cast<double>(pos) / static_cast<double>(total_pos);
//     double neg_dist = static_cast<double>(neg) / static_cast<double>(total_neg);
//     return (pos_dist - neg_dist) * woe;
//   }
//   
//   // Initial Binning based on unique values and pre-bins
//   void initialBinning() {
//     // Extract unique sorted values
//     std::vector<double> unique_values = feature;
//     std::sort(unique_values.begin(), unique_values.end());
//     unique_values.erase(std::unique(unique_values.begin(), unique_values.end()), unique_values.end());
//     
//     int num_unique_values = unique_values.size();
//     
//     if (num_unique_values <= 2) {
//       // Do not optimize or create extra bins; create bins based on unique values
//       is_unique_two_or_less = true;
//       bins.clear();
//       bins.reserve(num_unique_values);
//       for (int i = 0; i < num_unique_values; ++i) {
//         double lower = (i == 0) ? -std::numeric_limits<double>::infinity() : unique_values[i - 1];
//         double upper = unique_values[i];
//         bins.push_back(Bin{lower, upper, 0, 0, 0, 0.0, 0.0});
//       }
//       bins.back().upper_bound = std::numeric_limits<double>::infinity();
//     } else {
//       // Proceed with existing binning logic
//       is_unique_two_or_less = false;
//       // Determine number of initial bins
//       int n_bins = std::min(max_n_prebins, num_unique_values);
//       n_bins = std::max(n_bins, min_bins);
//       n_bins = std::min(n_bins, max_bins);
//       
//       // Determine bin boundaries
//       std::vector<double> boundaries;
//       for (int i = 1; i < n_bins; ++i) {
//         int index = i * (num_unique_values) / n_bins;
//         boundaries.push_back(unique_values[index]);
//       }
//       
//       // Initialize bins
//       bins.clear();
//       bins.reserve(n_bins);
//       double lower = -std::numeric_limits<double>::infinity();
//       for (size_t i = 0; i <= static_cast<size_t>(boundaries.size()); ++i) {
//         double upper = (i == static_cast<size_t>(boundaries.size())) ? std::numeric_limits<double>::infinity() : boundaries[i];
//         bins.push_back(Bin{lower, upper, 0, 0, 0, 0.0, 0.0});
//         lower = upper;
//       }
//     }
//   }
//   
//   // Assign data points to bins
//   void assignDataToBins() {
//     for (size_t i = 0; i < feature.size(); ++i) {
//       double value = feature[i];
//       int target_value = target[i];
//       bool assigned = false;
//       for (auto& bin : bins) {
//         if (value > bin.lower_bound && value <= bin.upper_bound) {
//           bin.count++;
//           if (target_value == 1) {
//             bin.count_pos++;
//           } else {
//             bin.count_neg++;
//           }
//           assigned = true;
//           break;
//         }
//       }
//       if (!assigned) {
//         // Handle edge cases
//         if (value <= bins.front().lower_bound) {
//           bins.front().count++;
//           if (target_value == 1) {
//             bins.front().count_pos++;
//           } else {
//             bins.front().count_neg++;
//           }
//         } else if (value > bins.back().upper_bound) {
//           bins.back().count++;
//           if (target_value == 1) {
//             bins.back().count_pos++;
//           } else {
//             bins.back().count_neg++;
//           }
//         }
//       }
//     }
//   }
//   
//   // Merge bins with low frequency based on bin_cutoff
//   void mergeLowFrequencyBins() {
//     int total_count = feature.size();
//     double cutoff_count = bin_cutoff * total_count;
//     
//     int iterations = 0;
//     while (iterations < max_iterations) {
//       bool merged = false;
//       for (size_t i = 0; i < bins.size(); ++i) {
//         if (bins[i].count < cutoff_count && static_cast<int>(bins.size()) > min_bins) {
//           if (i == 0) {
//             // Merge with next bin
//             bins[i].upper_bound = bins[i + 1].upper_bound;
//             bins[i].count += bins[i + 1].count;
//             bins[i].count_pos += bins[i + 1].count_pos;
//             bins[i].count_neg += bins[i + 1].count_neg;
//             bins.erase(bins.begin() + i + 1);
//           } else {
//             // Merge with previous bin
//             bins[i - 1].upper_bound = bins[i].upper_bound;
//             bins[i - 1].count += bins[i].count;
//             bins[i - 1].count_pos += bins[i].count_pos;
//             bins[i - 1].count_neg += bins[i].count_neg;
//             bins.erase(bins.begin() + i);
//           }
//           merged = true;
//           break;
//         }
//       }
//       if (!merged) {
//         break;
//       }
//       iterations++;
//     }
//     iterations_run += iterations;
//     if (iterations >= max_iterations) {
//       converged = false;
//     }
//   }
//   
//   // Enforce monotonicity of WoE values
//   void enforceMonotonicity() {
//     if (bins.size() <= 2) {
//       // If feature has two or fewer bins, ignore monotonicity enforcement
//       return;
//     }
//     
//     int iterations = 0;
//     bool is_monotonic = false;
//     bool increasing = true;
//     if (bins.size() >= 2) {
//       increasing = (bins[1].woe >= bins[0].woe);
//     }
//     while (!is_monotonic && static_cast<int>(bins.size()) > min_bins && iterations < max_iterations) {
//       is_monotonic = true;
//       for (size_t i = 1; i < bins.size(); ++i) {
//         if ((increasing && bins[i].woe < bins[i - 1].woe) ||
//             (!increasing && bins[i].woe > bins[i - 1].woe)) {
//           // Merge bins[i - 1] and bins[i]
//           bins[i - 1].upper_bound = bins[i].upper_bound;
//           bins[i - 1].count += bins[i].count;
//           bins[i - 1].count_pos += bins[i].count_pos;
//           bins[i - 1].count_neg += bins[i].count_neg;
//           bins.erase(bins.begin() + i);
//           calculateBinStatistics();
//           is_monotonic = false;
//           break;
//         }
//       }
//       iterations++;
//       if (bins.size() == min_bins) {
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
//   // Adjust bin count to be within [min_bins, max_bins]
//   void adjustBinCount() {
//     int iterations = 0;
//     while (static_cast<int>(bins.size()) > max_bins && iterations < max_iterations) {
//       // Find the pair of adjacent bins with the smallest IV difference
//       double min_iv_diff = std::numeric_limits<double>::max();
//       int merge_index = -1;
//       for (size_t i = 0; i < bins.size() - 1; ++i) {
//         double diff = std::abs(bins[i].iv - bins[i + 1].iv);
//         if (diff < min_iv_diff) {
//           min_iv_diff = diff;
//           merge_index = static_cast<int>(i);
//         }
//       }
//       if (merge_index != -1) {
//         // Merge bins at merge_index and merge_index + 1
//         bins[merge_index].upper_bound = bins[merge_index + 1].upper_bound;
//         bins[merge_index].count += bins[merge_index + 1].count;
//         bins[merge_index].count_pos += bins[merge_index + 1].count_pos;
//         bins[merge_index].count_neg += bins[merge_index + 1].count_neg;
//         bins.erase(bins.begin() + merge_index + 1);
//         calculateBinStatistics();
//       } else {
//         break; // No more bins can be merged
//       }
//       iterations++;
//     }
//     iterations_run += iterations;
//     if (iterations >= max_iterations) {
//       converged = false;
//     }
//   }
//   
//   // Calculate WoE and IV for each bin
//   void calculateBinStatistics() {
//     int total_pos = 0;
//     int total_neg = 0;
//     
//     for (const auto& bin : bins) {
//       total_pos += bin.count_pos;
//       total_neg += bin.count_neg;
//     }
//     
//     // Handle cases where total_pos or total_neg is zero
//     if (total_pos == 0 || total_neg == 0) {
//       Rcpp::stop("Target vector must contain both positive and negative cases.");
//     }
//     
//     for (auto& bin : bins) {
//       bin.woe = calculateWOE(bin.count_pos, bin.count_neg, total_pos, total_neg);
//       bin.iv = calculateIV(bin.woe, bin.count_pos, bin.count_neg, total_pos, total_neg);
//     }
//   }
//   
//   // Format bin intervals as strings
//   std::string formatBinInterval(double lower, double upper) const {
//     std::ostringstream oss;
//     oss << "(";
//     if (lower == -std::numeric_limits<double>::infinity()) {
//       oss << "-Inf";
//     } else {
//       oss << lower;
//     }
//     oss << ";";
//     if (upper == std::numeric_limits<double>::infinity()) {
//       oss << "+Inf";
//     } else {
//       oss << upper;
//     }
//     oss << "]";
//     return oss.str();
//   }
//   
// public:
//   OptimalBinningNumericalKMB(const std::vector<double>& feature_, const std::vector<int>& target_,
//                              int min_bins_, int max_bins_, double bin_cutoff_, int max_n_prebins_,
//                              double convergence_threshold_, int max_iterations_)
//     : feature(feature_), target(target_), min_bins(min_bins_), max_bins(max_bins_),
//       bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_),
//       convergence_threshold(convergence_threshold_), max_iterations(max_iterations_),
//       converged(true), iterations_run(0), is_unique_two_or_less(false) {}
//   
//   Rcpp::List fit() {
//     // Input validation
//     if (feature.empty() || target.empty()) {
//       Rcpp::stop("Feature and target vectors must not be empty.");
//     }
//     
//     if (feature.size() != target.size()) {
//       Rcpp::stop("Feature and target vectors must have the same length.");
//     }
//     
//     // Ensure target contains only 0 and 1
//     std::unordered_set<int> target_set(target.begin(), target.end());
//     if (target_set.size() > 2 || (target_set.find(0) == target_set.end() && target_set.find(1) == target_set.end())) {
//       Rcpp::stop("Target vector must contain only binary values 0 and 1.");
//     }
//     
//     if (min_bins < 2) {
//       Rcpp::stop("min_bins must be at least 2.");
//     }
//     
//     if (max_bins < min_bins) {
//       Rcpp::stop("max_bins must be greater than or equal to min_bins.");
//     }
//     
//     if (bin_cutoff <= 0 || bin_cutoff >= 1) {
//       Rcpp::stop("bin_cutoff must be between 0 and 1.");
//     }
//     
//     if (max_n_prebins <= 0) {
//       Rcpp::stop("max_n_prebins must be positive.");
//     }
//     
//     if (max_iterations <= 0) {
//       Rcpp::stop("max_iterations must be positive.");
//     }
//     
//     // Initial Binning
//     initialBinning();
//     
//     // Assign data to bins
//     assignDataToBins();
//     
//     if (is_unique_two_or_less) {
//       // Calculate WoE and IV
//       calculateBinStatistics();
//       
//       // Prepare output vectors
//       std::vector<std::string> bin_labels;
//       std::vector<double> woe_values;
//       std::vector<double> iv_values;
//       std::vector<int> counts;
//       std::vector<int> counts_pos;
//       std::vector<int> counts_neg;
//       std::vector<double> cutpoints;
//       
//       for (size_t i = 0; i < bins.size(); ++i) {
//         const auto& bin = bins[i];
//         bin_labels.push_back(formatBinInterval(bin.lower_bound, bin.upper_bound));
//         woe_values.push_back(bin.woe);
//         iv_values.push_back(bin.iv);
//         counts.push_back(bin.count);
//         counts_pos.push_back(bin.count_pos);
//         counts_neg.push_back(bin.count_neg);
//         if (i < bins.size() - 1) {
//           cutpoints.push_back(bin.upper_bound);
//         }
//       }
//       
//       return Rcpp::List::create(
//         Rcpp::Named("bin") = bin_labels,
//         Rcpp::Named("woe") = woe_values,
//         Rcpp::Named("iv") = iv_values,
//         Rcpp::Named("count") = counts,
//         Rcpp::Named("count_pos") = counts_pos,
//         Rcpp::Named("count_neg") = counts_neg,
//         Rcpp::Named("cutpoints") = cutpoints,
//         Rcpp::Named("converged") = converged,
//         Rcpp::Named("iterations") = iterations_run
//       );
//     }
//     
//     // Continue with optimization steps
//     // Calculate initial WoE and IV
//     calculateBinStatistics();
//     
//     // Merge low frequency bins
//     mergeLowFrequencyBins();
//     
//     // Enforce monotonicity
//     enforceMonotonicity();
//     
//     // Adjust bin count to be within [min_bins, max_bins]
//     adjustBinCount();
//     
//     // Final calculation of WoE and IV
//     calculateBinStatistics();
//     
//     // Prepare output vectors
//     std::vector<std::string> bin_labels;
//     std::vector<double> woe_values;
//     std::vector<double> iv_values;
//     std::vector<int> counts;
//     std::vector<int> counts_pos;
//     std::vector<int> counts_neg;
//     std::vector<double> cutpoints;
//     
//     for (size_t i = 0; i < bins.size(); ++i) {
//       const auto& bin = bins[i];
//       bin_labels.push_back(formatBinInterval(bin.lower_bound, bin.upper_bound));
//       woe_values.push_back(bin.woe);
//       iv_values.push_back(bin.iv);
//       counts.push_back(bin.count);
//       counts_pos.push_back(bin.count_pos);
//       counts_neg.push_back(bin.count_neg);
//       if (i < bins.size() - 1) {
//         cutpoints.push_back(bin.upper_bound);
//       }
//     }
//     
//     return Rcpp::List::create(
//       Rcpp::Named("bin") = bin_labels,
//       Rcpp::Named("woe") = woe_values,
//       Rcpp::Named("iv") = iv_values,
//       Rcpp::Named("count") = counts,
//       Rcpp::Named("count_pos") = counts_pos,
//       Rcpp::Named("count_neg") = counts_neg,
//       Rcpp::Named("cutpoints") = cutpoints,
//       Rcpp::Named("converged") = converged,
//       Rcpp::Named("iterations") = iterations_run
//     );
//   }
// };
// 
// 
// //' @title Optimal Binning for Numerical Variables using K-means Binning (KMB)
// //'
// //' @description This function implements the K-means Binning (KMB) algorithm for optimal binning of numerical variables.
// //'
// //' @param target An integer vector of binary target values (0 or 1).
// //' @param feature A numeric vector of feature values to be binned.
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff Minimum frequency for a bin (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins (default: 20).
// //' @param convergence_threshold Convergence threshold for the algorithm (default: 1e-6).
// //' @param max_iterations Maximum number of iterations allowed (default: 1000).
// //'
// //' @return A list containing the following elements:
// //' \item{bins}{Character vector of bin ranges.}
// //' \item{woe}{Numeric vector of WoE values for each bin.}
// //' \item{iv}{Numeric vector of Information Value (IV) for each bin.}
// //' \item{count}{Integer vector of total observations in each bin.}
// //' \item{count_pos}{Integer vector of positive target observations in each bin.}
// //' \item{count_neg}{Integer vector of negative target observations in each bin.}
// //' \item{cutpoints}{Numeric vector of cut points to generate the bins.}
// //' \item{converged}{Logical indicating if the algorithm converged.}
// //' \item{iterations}{Integer number of iterations run by the algorithm.}
// //'
// //' @details
// //' The K-means Binning (KMB) algorithm is an advanced method for optimal binning of numerical variables.
// //' It combines elements of k-means clustering with traditional binning techniques to create bins that maximize
// //' the predictive power of the feature while respecting user-defined constraints.
// //'
// //' The algorithm works through several steps:
// //' 1. Initial Binning: Creates initial bins based on the unique values of the feature, respecting the max_n_prebins constraint.
// //' 2. Data Assignment: Assigns data points to the appropriate bins.
// //' 3. Low Frequency Merging: Merges bins with frequencies below the bin_cutoff threshold.
// //' 4. Enforce Monotonicity: Merges bins to ensure that the WoE values are monotonic.
// //' 5. Bin Count Adjustment: Adjusts the number of bins to fall within the specified range (min_bins to max_bins).
// //' 6. Statistics Calculation: Computes Weight of Evidence (WoE) and Information Value (IV) for each bin.
// //'
// //' The KMB method uses a modified version of the Weight of Evidence (WoE) calculation that incorporates Laplace smoothing
// //' to handle cases with zero counts
// //' \deqn{WoE_i = \ln\left(\frac{(n_{1i} + 0.5) / (N_1 + 1)}{(n_{0i} + 0.5) / (N_0 + 1)}\right)}
// //' where \eqn{n_{1i}} and \eqn{n_{0i}} are the number of events and non-events in bin i,
// //' and \eqn{N_1} and \eqn{N_0} are the total number of events and non-events.
// //'
// //' The Information Value (IV) for each bin is calculated as:
// //' \deqn{IV_i = \left(\frac{n_{1i}}{N_1} - \frac{n_{0i}}{N_0}\right) \times WoE_i}
// //'
// //' The KMB method aims to create bins that maximize the overall IV while respecting the user-defined constraints.
// //' It uses a greedy approach to merge bins when necessary, choosing to merge bins with the smallest difference in IV.
// //'
// //' When adjusting the number of bins, the algorithm either merges bins with the most similar IVs (if there are too many bins)
// //' or stops merging when min_bins is reached, even if monotonicity is not achieved.
// //'
// //' @examples
// //' \dontrun{
// //'   # Create sample data
// //'   set.seed(123)
// //'   target <- sample(0:1, 1000, replace = TRUE)
// //'   feature <- rnorm(1000)
// //'
// //'   # Run optimal binning
// //'   result <- optimal_binning_numerical_kmb(target, feature)
// //'
// //'   # View results
// //'   print(result)
// //' }
// //'
// //' @references
// //' \itemize{
// //' \item Fayyad, U., & Irani, K. (1993). Multi-interval discretization of continuous-valued attributes for classification learning. In Proceedings of the 13th International Joint Conference on Artificial Intelligence (pp. 1022-1027).
// //' \item Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). Credit Scoring and Its Applications. SIAM Monographs on Mathematical Modeling and Computation.
// //' }
// //'
// //' @export
// // [[Rcpp::export]]
// List optimal_binning_numerical_kmb(IntegerVector target,
//                                   NumericVector feature,
//                                   int min_bins = 3, int max_bins = 5,
//                                   double bin_cutoff = 0.05, int max_n_prebins = 20,
//                                   double convergence_threshold = 1e-6, int max_iterations = 1000) {
//  // Convert R vectors to C++ vectors
//  std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
//  std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
//  
//  // Initialize binning class
//  OptimalBinningNumericalKMB binner(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins,
//                                    convergence_threshold, max_iterations);
//  
//  // Perform binning
//  Rcpp::List result = binner.fit();
//  
//  return result;
// }
