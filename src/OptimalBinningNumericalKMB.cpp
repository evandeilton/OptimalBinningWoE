#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#include <string>
#include <sstream>
#include <unordered_set>
#ifdef _OPENMP
#include <omp.h>
#endif

class OptimalBinningNumericalKMB {
private:
  std::vector<double> feature;
  std::vector<int> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  
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
    
    // Determine number of initial bins
    int n_bins = std::min(static_cast<int>(unique_values.size()), max_n_prebins);
    n_bins = std::max(n_bins, min_bins);
    n_bins = std::min(n_bins, max_bins);
    
    // Determine bin boundaries
    std::vector<double> boundaries;
    for (int i = 0; i <= n_bins; ++i) {
      int index = i * (unique_values.size() - 1) / n_bins;
      boundaries.push_back(unique_values[index]);
    }
    
    // Initialize bins
    bins.clear();
    bins.reserve(n_bins);
    for (int i = 0; i < n_bins; ++i) {
      bins.push_back(Bin{0.0, 0.0, 0, 0, 0, 0.0, 0.0});
      bins[i].lower_bound = (i == 0) ? -std::numeric_limits<double>::infinity() : boundaries[i];
      bins[i].upper_bound = (i == n_bins - 1) ? std::numeric_limits<double>::infinity() : boundaries[i + 1];
    }
  }
  
  // Assign data points to bins using binary search for efficiency
  void assignDataToBins() {
    for (size_t i = 0; i < feature.size(); ++i) {
      double value = feature[i];
      // Binary search to find the correct bin
      int left = 0;
      int right = static_cast<int>(bins.size()) - 1;
      int bin_index = -1;
      while (left <= right) {
        int mid = left + (right - left) / 2;
        if (value > bins[mid].upper_bound) {
          left = mid + 1;
        } else if (value <= bins[mid].upper_bound) {
          bin_index = mid;
          right = mid - 1;
        }
      }
      if (bin_index != -1) {
        bins[bin_index].count++;
        if (target[i] == 1) {
          bins[bin_index].count_pos++;
        } else {
          bins[bin_index].count_neg++;
        }
      }
    }
  }
  
  // Merge bins with low frequency based on bin_cutoff
  void mergeLowFrequencyBins() {
    int total_count = feature.size();
    std::vector<Bin> merged_bins;
    merged_bins.reserve(bins.size());
    
    for (const auto& bin : bins) {
      if (static_cast<double>(bin.count) / total_count >= bin_cutoff) {
        merged_bins.push_back(bin);
      } else {
        if (!merged_bins.empty()) {
          // Merge with the last bin
          Bin& last_bin = merged_bins.back();
          last_bin.upper_bound = bin.upper_bound;
          last_bin.count += bin.count;
          last_bin.count_pos += bin.count_pos;
          last_bin.count_neg += bin.count_neg;
        } else {
          // If merged_bins is empty, start with the current bin
          merged_bins.push_back(bin);
        }
      }
    }
    
    // Ensure at least min_bins after merging
    while (static_cast<int>(merged_bins.size()) < min_bins && merged_bins.size() > 1) {
      // Find the two adjacent bins with the smallest IV difference
      double min_iv_diff = std::numeric_limits<double>::max();
      int merge_index = 0;
      for (size_t i = 0; i < merged_bins.size() - 1; ++i) {
        double diff = std::abs(merged_bins[i].woe - merged_bins[i + 1].woe);
        if (diff < min_iv_diff) {
          min_iv_diff = diff;
          merge_index = static_cast<int>(i);
        }
      }
      // Merge bins at merge_index and merge_index + 1
      merged_bins[merge_index].upper_bound = merged_bins[merge_index + 1].upper_bound;
      merged_bins[merge_index].count += merged_bins[merge_index + 1].count;
      merged_bins[merge_index].count_pos += merged_bins[merge_index + 1].count_pos;
      merged_bins[merge_index].count_neg += merged_bins[merge_index + 1].count_neg;
      merged_bins.erase(merged_bins.begin() + merge_index + 1);
    }
    
    bins = merged_bins;
  }
  
  // Adjust bin count to be within [min_bins, max_bins]
  void adjustBinCount() {
    // If bins exceed max_bins, merge bins with smallest IV difference
    while (static_cast<int>(bins.size()) > max_bins) {
      double min_iv_diff = std::numeric_limits<double>::max();
      int merge_index = 0;
      for (size_t i = 0; i < bins.size() - 1; ++i) {
        double diff = std::abs(bins[i].iv - bins[i + 1].iv);
        if (diff < min_iv_diff) {
          min_iv_diff = diff;
          merge_index = static_cast<int>(i);
        }
      }
      // Merge bins at merge_index and merge_index + 1
      bins[merge_index].upper_bound = bins[merge_index + 1].upper_bound;
      bins[merge_index].count += bins[merge_index + 1].count;
      bins[merge_index].count_pos += bins[merge_index + 1].count_pos;
      bins[merge_index].count_neg += bins[merge_index + 1].count_neg;
      bins.erase(bins.begin() + merge_index + 1);
    }
    
    // If bins are fewer than min_bins, split the bin with the largest range
    while (static_cast<int>(bins.size()) < min_bins) {
      // Find the bin with the largest range
      auto max_range_it = std::max_element(bins.begin(), bins.end(),
                                           [](const Bin& a, const Bin& b) {
                                             return (a.upper_bound - a.lower_bound) < (b.upper_bound - b.lower_bound);
                                           });
      if (max_range_it == bins.end()) break; // Safety check
      
      double mid = (max_range_it->lower_bound + max_range_it->upper_bound) / 2.0;
      // Create two new bins by splitting the found bin
      Bin bin1 = {max_range_it->lower_bound, mid, 0, 0, 0, 0.0, 0.0};
      Bin bin2 = {mid, max_range_it->upper_bound, 0, 0, 0, 0.0, 0.0};
      
      // Redistribute counts
      for (size_t i = 0; i < feature.size(); ++i) {
        double value = feature[i];
        if (value > max_range_it->lower_bound && value <= mid) {
          bin1.count++;
          if (target[i] == 1) {
            bin1.count_pos++;
          } else {
            bin1.count_neg++;
          }
        } else if (value > mid && value <= max_range_it->upper_bound) {
          bin2.count++;
          if (target[i] == 1) {
            bin2.count_pos++;
          } else {
            bin2.count_neg++;
          }
        }
      }
      
      // Replace the original bin with the two new bins
      *max_range_it = bin1;
      bins.insert(max_range_it + 1, bin2);
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
    
    // Parallelize the calculation if OpenMP is available
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < bins.size(); ++i) {
      bins[i].woe = calculateWOE(bins[i].count_pos, bins[i].count_neg, total_pos, total_neg);
      bins[i].iv = calculateIV(bins[i].woe, bins[i].count_pos, bins[i].count_neg, total_pos, total_neg);
    }
  }
  
  // Format bin intervals as strings
  std::string formatBinInterval(double lower, double upper) const {
    std::ostringstream oss;
    if (lower == -std::numeric_limits<double>::infinity()) {
      oss << "(-Inf;";
    } else {
      oss << "(" << lower << ";";
    }
    
    if (upper == std::numeric_limits<double>::infinity()) {
      oss << "+Inf]";
    } else {
      oss << upper << "]";
    }
    return oss.str();
  }
  
public:
  OptimalBinningNumericalKMB(const std::vector<double>& feature, const std::vector<int>& target,
                             int min_bins, int max_bins, double bin_cutoff, int max_n_prebins)
    : feature(feature), target(target), min_bins(min_bins), max_bins(max_bins),
      bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins) {}
  
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
    
    // Initial Binning
    initialBinning();
    
    // Assign data to bins
    assignDataToBins();
    
    // Merge low frequency bins
    mergeLowFrequencyBins();
    
    // Adjust bin count to be within [min_bins, max_bins]
    adjustBinCount();
    
    // Calculate WoE and IV
    calculateBinStatistics();
    
    // Prepare output vectors
    Rcpp::NumericVector bin_vec(bins.size());
    Rcpp::NumericVector woe_vec(bins.size());
    Rcpp::NumericVector iv_vec(bins.size());
    Rcpp::IntegerVector count_vec(bins.size());
    Rcpp::IntegerVector count_pos_vec(bins.size());
    Rcpp::IntegerVector count_neg_vec(bins.size());
    Rcpp::StringVector bin_labels(bins.size());
    
    // Preallocate woefeature with WoE values
    std::vector<double> woefeature(feature.size(), 0.0);
    
    // Assign WoE to woefeature efficiently using binary search
    for (size_t i = 0; i < bins.size(); ++i) {
      bin_vec[i] = static_cast<double>(i + 1);
      woe_vec[i] = bins[i].woe;
      iv_vec[i] = bins[i].iv;
      count_vec[i] = bins[i].count;
      count_pos_vec[i] = bins[i].count_pos;
      count_neg_vec[i] = bins[i].count_neg;
      bin_labels[i] = formatBinInterval(bins[i].lower_bound, bins[i].upper_bound);
    }
    
    // Create a sorted vector of bin boundaries for assignment
    std::vector<std::pair<double, double>> bin_boundaries;
    bin_boundaries.reserve(bins.size());
    for (const auto& bin : bins) {
      bin_boundaries.emplace_back(bin.lower_bound, bin.upper_bound);
    }
    
    // Assign WoE to woefeature using a single pass
    for (size_t j = 0; j < feature.size(); ++j) {
      double value = feature[j];
      // Binary search to find the bin index
      int left = 0;
      int right = static_cast<int>(bin_boundaries.size()) - 1;
      int bin_idx = -1;
      while (left <= right) {
        int mid = left + (right - left) / 2;
        if (value > bin_boundaries[mid].second) {
          left = mid + 1;
        } else if (value <= bin_boundaries[mid].second) {
          bin_idx = mid;
          right = mid - 1;
        }
      }
      if (bin_idx != -1) {
        woefeature[j] = woe_vec[bin_idx];
      }
    }
    
    // Create WoE bin DataFrame
    Rcpp::DataFrame woebin = Rcpp::DataFrame::create(
      Rcpp::Named("bin") = bin_labels,
      Rcpp::Named("woe") = woe_vec,
      Rcpp::Named("iv") = iv_vec,
      Rcpp::Named("count") = count_vec,
      Rcpp::Named("count_pos") = count_pos_vec,
      Rcpp::Named("count_neg") = count_neg_vec,
      Rcpp::Named("bin_number") = bin_vec
    );
    
    return Rcpp::List::create(
      Rcpp::Named("woefeature") = woefeature,
      Rcpp::Named("woebin") = woebin
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
//' 
//' @return A list containing two elements:
//' \item{woefeature}{A numeric vector of Weight of Evidence (WoE) transformed feature values.}
//' \item{woebin}{A data frame containing bin information, including bin labels, WoE, Information Value (IV), and counts.}
//' 
//' @details
//' The K-means Binning (KMB) algorithm is an advanced method for optimal binning of numerical variables. It combines elements of k-means clustering with traditional binning techniques to create bins that maximize the predictive power of the feature while respecting user-defined constraints.
//' 
//' The algorithm works through several steps:
//' 1. Initial Binning: Creates initial bins based on the unique values of the feature, respecting the max_n_prebins constraint.
//' 2. Data Assignment: Assigns data points to the appropriate bins.
//' 3. Low Frequency Merging: Merges bins with frequencies below the bin_cutoff threshold.
//' 4. Bin Count Adjustment: Adjusts the number of bins to fall within the specified range (min_bins to max_bins).
//' 5. Statistics Calculation: Computes Weight of Evidence (WoE) and Information Value (IV) for each bin.
//' 
//' The KMB method uses a modified version of the Weight of Evidence (WoE) calculation that incorporates Laplace smoothing to handle cases with zero counts:
//' 
//' \deqn{WoE_i = \ln\left(\frac{(n_{1i} + 0.5) / (N_1 + 1)}{(n_{0i} + 0.5) / (N_0 + 1)}\right)}
//' 
//' where \eqn{n_{1i}} and \eqn{n_{0i}} are the number of events and non-events in bin i, and \eqn{N_1} and \eqn{N_0} are the total number of events and non-events.
//' 
//' The Information Value (IV) for each bin is calculated as:
//' 
//' \deqn{IV_i = \left(\frac{n_{1i}}{N_1} - \frac{n_{0i}}{N_0}\right) \times WoE_i}
//' 
//' The KMB method aims to create bins that maximize the overall IV while respecting the user-defined constraints. It uses a greedy approach to merge bins when necessary, choosing to merge bins with the smallest difference in IV.
//' 
//' When adjusting the number of bins, the algorithm either merges bins with the most similar IVs (if there are too many bins) or splits the bin with the largest range (if there are too few bins).
//' 
//' The KMB method provides a balance between predictive power and model interpretability, allowing users to control the trade-off through parameters such as min_bins, max_bins, and bin_cutoff.
//' 
//' @examples
//' \dontrun{
//' # Create sample data
//' set.seed(123)
//' target <- sample(0:1, 1000, replace = TRUE)
//' feature <- rnorm(1000)
//' 
//' # Run optimal binning
//' result <- optimal_binning_numerical_kmb(target, feature)
//' 
//' # View results
//' head(result$woefeature)
//' print(result$woebin)
//' }
//' 
//' @references
//' \itemize{
//' \item Fayyad, U., & Irani, K. (1993). Multi-interval discretization of continuous-valued attributes for classification learning. In Proceedings of the 13th International Joint Conference on Artificial Intelligence (pp. 1022-1027).
//' \item Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). Credit Scoring and Its Applications. SIAM Monographs on Mathematical Modeling and Computation.
//' }
//' 
//' @author Lopes,
//' 
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_kmb(Rcpp::IntegerVector target,
                                        Rcpp::NumericVector feature,
                                        int min_bins = 3, int max_bins = 5,
                                        double bin_cutoff = 0.05, int max_n_prebins = 20) {
 // Convert R vectors to C++ vectors
 std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
 std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
 
 // Initialize binning class
 OptimalBinningNumericalKMB binner(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins);
 
 // Perform binning
 return binner.fit();
}


// #include <Rcpp.h>
// #include <algorithm>
// #include <vector>
// #include <cmath>
// #include <limits>
// #include <string>
// #include <sstream>
// #ifdef _OPENMP
// #include <omp.h>
// #endif
// 
// class OptimalBinningNumericalKMB {
// private:
//   std::vector<double> feature;
//   std::vector<int> target;
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
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
//   double calculateWOE(int pos, int neg, int total_pos, int total_neg) {
//     double pos_rate = (pos + 0.5) / (total_pos + 1.0);
//     double neg_rate = (neg + 0.5) / (total_neg + 1.0);
//     return std::log(pos_rate / neg_rate);
//   }
// 
//   double calculateIV(double woe, int pos, int neg, int total_pos, int total_neg) {
//     double pos_rate = static_cast<double>(pos) / total_pos;
//     double neg_rate = static_cast<double>(neg) / total_neg;
//     return (pos_rate - neg_rate) * woe;
//   }
// 
//   void initialBinning() {
//     std::vector<double> unique_values = feature;
//     std::sort(unique_values.begin(), unique_values.end());
//     unique_values.erase(std::unique(unique_values.begin(), unique_values.end()), unique_values.end());
// 
//     int n_bins = std::min(static_cast<int>(unique_values.size()), max_n_prebins);
//     n_bins = std::max(n_bins, min_bins);
//     n_bins = std::min(n_bins, max_bins);
// 
//     std::vector<double> boundaries;
//     for (int i = 0; i <= n_bins; ++i) {
//       int index = i * (unique_values.size() - 1) / n_bins;
//       boundaries.push_back(unique_values[index]);
//     }
// 
//     bins.clear();
//     for (size_t i = 0; i < boundaries.size() - 1; ++i) {
//       bins.push_back({boundaries[i], boundaries[i + 1], 0, 0, 0, 0.0, 0.0});
//     }
//     bins.front().lower_bound = std::numeric_limits<double>::lowest();
//     bins.back().upper_bound = std::numeric_limits<double>::max();
//   }
// 
//   void assignDataToBins() {
//     for (size_t i = 0; i < feature.size(); ++i) {
//       auto it = std::lower_bound(bins.begin(), bins.end(), feature[i],
//                                  [](const Bin& bin, double value) { return bin.upper_bound <= value; });
// 
//       if (it != bins.end()) {
//         it->count++;
//         if (target[i] == 1) {
//           it->count_pos++;
//         } else {
//           it->count_neg++;
//         }
//       }
//     }
//   }
// 
//   void mergeLowFrequencyBins() {
//     int total_count = feature.size();
//     std::vector<Bin> merged_bins;
// 
//     for (const auto& bin : bins) {
//       if (static_cast<double>(bin.count) / total_count >= bin_cutoff) {
//         merged_bins.push_back(bin);
//       } else if (!merged_bins.empty()) {
//         auto& last_bin = merged_bins.back();
//         last_bin.upper_bound = bin.upper_bound;
//         last_bin.count += bin.count;
//         last_bin.count_pos += bin.count_pos;
//         last_bin.count_neg += bin.count_neg;
//       } else {
//         merged_bins.push_back(bin);
//       }
//     }
// 
//     bins = merged_bins;
//   }
// 
//   void adjustBinCount() {
//     while (static_cast<int>(bins.size()) > max_bins) {
//       auto min_iv_diff_it = std::min_element(bins.begin(), bins.end() - 1,
//                                              [this](const Bin& a, const Bin& b) {
//                                                size_t index_a = &a - &bins[0];
//                                                size_t index_b = &b - &bins[0];
//                                                return std::abs(a.iv - bins[index_a + 1].iv) < std::abs(b.iv - bins[index_b + 1].iv);
//                                              });
// 
//       auto next_bin = std::next(min_iv_diff_it);
//       min_iv_diff_it->upper_bound = next_bin->upper_bound;
//       min_iv_diff_it->count += next_bin->count;
//       min_iv_diff_it->count_pos += next_bin->count_pos;
//       min_iv_diff_it->count_neg += next_bin->count_neg;
//       bins.erase(next_bin);
//     }
// 
//     while (static_cast<int>(bins.size()) < min_bins) {
//       auto max_range_it = std::max_element(bins.begin(), bins.end(),
//                                            [](const Bin& a, const Bin& b) {
//                                              return (a.upper_bound - a.lower_bound) < (b.upper_bound - b.lower_bound);
//                                            });
// 
//       double mid = (max_range_it->lower_bound + max_range_it->upper_bound) / 2;
//       Bin new_bin = {mid, max_range_it->upper_bound, 0, 0, 0, 0.0, 0.0};
//       max_range_it->upper_bound = mid;
// 
//       // Redistribute counts
//       for (size_t i = 0; i < feature.size(); ++i) {
//         if (feature[i] > mid && feature[i] <= new_bin.upper_bound) {
//           new_bin.count++;
//           max_range_it->count--;
//           if (target[i] == 1) {
//             new_bin.count_pos++;
//             max_range_it->count_pos--;
//           } else {
//             new_bin.count_neg++;
//             max_range_it->count_neg--;
//           }
//         }
//       }
// 
//       bins.insert(max_range_it + 1, new_bin);
//     }
//   }
// 
//   void calculateBinStatistics() {
//     int total_pos = 0, total_neg = 0;
// 
//     for (const auto& bin : bins) {
//       total_pos += bin.count_pos;
//       total_neg += bin.count_neg;
//     }
// 
//     for (auto& bin : bins) {
//       bin.woe = calculateWOE(bin.count_pos, bin.count_neg, total_pos, total_neg);
//       bin.iv = calculateIV(bin.woe, bin.count_pos, bin.count_neg, total_pos, total_neg);
//     }
//   }
// 
//   std::string formatBinInterval(double lower, double upper) {
//     std::ostringstream oss;
//     if (lower == std::numeric_limits<double>::lowest()) {
//       oss << "(-Inf;";
//     } else {
//       oss << "(" << lower << ";";
//     }
// 
//     if (upper == std::numeric_limits<double>::max()) {
//       oss << "+Inf]";
//     } else {
//       oss << upper << "]";
//     }
//     return oss.str();
//   }
// 
// public:
//   OptimalBinningNumericalKMB(const std::vector<double>& feature, const std::vector<int>& target,
//                              int min_bins, int max_bins, double bin_cutoff, int max_n_prebins)
//     : feature(feature), target(target), min_bins(min_bins), max_bins(max_bins),
//       bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins) {}
// 
//   Rcpp::List fit() {
//     if (feature.size() != target.size()) {
//       Rcpp::stop("Feature and target vectors must have the same length");
//     }
// 
//     if (min_bins < 2 || max_bins < min_bins) {
//       Rcpp::stop("Invalid bin constraints");
//     }
// 
//     initialBinning();
//     assignDataToBins();
//     mergeLowFrequencyBins();
//     adjustBinCount();
//     calculateBinStatistics();
// 
//     // Prepare output
//     std::vector<double> woefeature(feature.size());
//     Rcpp::NumericVector bin_vec(bins.size());
//     Rcpp::NumericVector woe_vec(bins.size());
//     Rcpp::NumericVector iv_vec(bins.size());
//     Rcpp::IntegerVector count_vec(bins.size());
//     Rcpp::IntegerVector count_pos_vec(bins.size());
//     Rcpp::IntegerVector count_neg_vec(bins.size());
//     Rcpp::StringVector bin_labels(bins.size());
// 
//     for (size_t i = 0; i < bins.size(); ++i) {
//       const auto& bin = bins[i];
//       bin_vec[i] = i + 1;
//       woe_vec[i] = bin.woe;
//       iv_vec[i] = bin.iv;
//       count_vec[i] = bin.count;
//       count_pos_vec[i] = bin.count_pos;
//       count_neg_vec[i] = bin.count_neg;
//       bin_labels[i] = formatBinInterval(bin.lower_bound, bin.upper_bound);
// 
//       for (size_t j = 0; j < feature.size(); ++j) {
//         if (feature[j] > bin.lower_bound && feature[j] <= bin.upper_bound) {
//           woefeature[j] = bin.woe;
//         }
//       }
//     }
// 
//     Rcpp::DataFrame woebin = Rcpp::DataFrame::create(
//       Rcpp::Named("bin") = bin_labels,
//       Rcpp::Named("woe") = woe_vec,
//       Rcpp::Named("iv") = iv_vec,
//       Rcpp::Named("count") = count_vec,
//       Rcpp::Named("count_pos") = count_pos_vec,
//       Rcpp::Named("count_neg") = count_neg_vec
//     );
// 
//     return Rcpp::List::create(
//       Rcpp::Named("woefeature") = woefeature,
//       Rcpp::Named("woebin") = woebin
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
// //' 
// //' @return A list containing two elements:
// //' \item{woefeature}{A numeric vector of Weight of Evidence (WoE) transformed feature values.}
// //' \item{woebin}{A data frame containing bin information, including bin labels, WoE, Information Value (IV), and counts.}
// //' 
// //' @details
// //' The K-means Binning (KMB) algorithm is an advanced method for optimal binning of numerical variables. It combines elements of k-means clustering with traditional binning techniques to create bins that maximize the predictive power of the feature while respecting user-defined constraints.
// //' 
// //' The algorithm works through several steps:
// //' 1. Initial Binning: Creates initial bins based on the unique values of the feature, respecting the max_n_prebins constraint.
// //' 2. Data Assignment: Assigns data points to the appropriate bins.
// //' 3. Low Frequency Merging: Merges bins with frequencies below the bin_cutoff threshold.
// //' 4. Bin Count Adjustment: Adjusts the number of bins to fall within the specified range (min_bins to max_bins).
// //' 5. Statistics Calculation: Computes Weight of Evidence (WoE) and Information Value (IV) for each bin.
// //' 
// //' The KMB method uses a modified version of the Weight of Evidence (WoE) calculation that incorporates Laplace smoothing to handle cases with zero counts:
// //' 
// //' \deqn{WoE_i = \ln\left(\frac{(n_{1i} + 0.5) / (N_1 + 1)}{(n_{0i} + 0.5) / (N_0 + 1)}\right)}
// //' 
// //' where \eqn{n_{1i}} and \eqn{n_{0i}} are the number of events and non-events in bin i, and \eqn{N_1} and \eqn{N_0} are the total number of events and non-events.
// //' 
// //' The Information Value (IV) for each bin is calculated as:
// //' 
// //' \deqn{IV_i = \left(\frac{n_{1i}}{N_1} - \frac{n_{0i}}{N_0}\right) \times WoE_i}
// //' 
// //' The KMB method aims to create bins that maximize the overall IV while respecting the user-defined constraints. It uses a greedy approach to merge bins when necessary, choosing to merge bins with the smallest difference in IV.
// //' 
// //' When adjusting the number of bins, the algorithm either merges bins with the most similar IVs (if there are too many bins) or splits the bin with the largest range (if there are too few bins).
// //' 
// //' The KMB method provides a balance between predictive power and model interpretability, allowing users to control the trade-off through parameters such as min_bins, max_bins, and bin_cutoff.
// //' 
// //' @examples
// //' \dontrun{
// //' # Create sample data
// //' set.seed(123)
// //' target <- sample(0:1, 1000, replace = TRUE)
// //' feature <- rnorm(1000)
// //' 
// //' # Run optimal binning
// //' result <- optimal_binning_numerical_kmb(target, feature)
// //' 
// //' # View results
// //' head(result$woefeature)
// //' print(result$woebin)
// //' }
// //' 
// //' @references
// //' \itemize{
// //' \item Fayyad, U., & Irani, K. (1993). Multi-interval discretization of continuous-valued attributes for classification learning. In Proceedings of the 13th International Joint Conference on Artificial Intelligence (pp. 1022-1027).
// //' \item Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). Credit Scoring and Its Applications. SIAM Monographs on Mathematical Modeling and Computation.
// //' }
// //' 
// //' @author Lopes, J. E.
// //' 
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_numerical_kmb(Rcpp::IntegerVector target,
//                                          Rcpp::NumericVector feature,
//                                          int min_bins = 3, int max_bins = 5,
//                                          double bin_cutoff = 0.05, int max_n_prebins = 20) {
//   std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
//   std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
// 
//   OptimalBinningNumericalKMB binner(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins);
//   return binner.fit();
// }
