#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <sstream>
#include <iomanip>
#include <set>

// Enable OpenMP if available
#ifdef _OPENMP
#include <omp.h>
#endif

// [[Rcpp::plugins(openmp)]]

class OptimalBinningNumericalMDLP {
private:
  std::vector<double> feature;
  std::vector<int> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  
  struct Bin {
    double lower_bound; // Inclusive
    double upper_bound; // Exclusive
    int count;
    int count_pos;
    int count_neg;
    double woe;
    double iv;
  };
  
  std::vector<Bin> bins;
  
  // Calculate entropy for a bin
  double calculate_entropy(int pos, int neg) const {
    if (pos == 0 || neg == 0) return 0.0;
    double total = static_cast<double>(pos + neg);
    double p = pos / total;
    double q = neg / total;
    return -p * std::log2(p) - q * std::log2(q);
  }
  
  // Calculate MDL cost for current bins
  double calculate_mdl_cost(const std::vector<Bin>& current_bins) const {
    int total_count = 0;
    int total_pos = 0;
    for (const auto& bin : current_bins) {
      total_count += bin.count;
      total_pos += bin.count_pos;
    }
    int total_neg = total_count - total_pos;
    
    // Handle cases with all positives or all negatives
    if (total_pos == 0 || total_neg == 0) {
      return std::numeric_limits<double>::infinity();
    }
    
    double model_cost = std::log2(static_cast<double>(current_bins.size()) - 1.0);
    double data_cost = total_count * calculate_entropy(total_pos, total_neg);
    
    for (const auto& bin : current_bins) {
      data_cost -= bin.count * calculate_entropy(bin.count_pos, bin.count_neg);
    }
    
    return model_cost + data_cost;
  }
  
  // Merge bin at specified index with the next bin
  void merge_bins(size_t index) {
    Bin& left = bins[index];
    Bin& right = bins[index + 1];
    
    left.upper_bound = right.upper_bound;
    left.count += right.count;
    left.count_pos += right.count_pos;
    left.count_neg += right.count_neg;
    
    bins.erase(bins.begin() + index + 1);
  }
  
  // Calculate WoE and IV for each bin
  void calculate_woe_iv() {
    int total_pos = 0, total_neg = 0;
    for (const auto& bin : bins) {
      total_pos += bin.count_pos;
      total_neg += bin.count_neg;
    }
    
    // Check if total_pos and total_neg are zero
    if (total_pos == 0 || total_neg == 0) {
      Rcpp::stop("All target values are the same. WoE and IV cannot be calculated.");
    }
    
    // Calculate distributions
    std::vector<double> dist_pos(bins.size(), 0.0);
    std::vector<double> dist_neg(bins.size(), 0.0);
    double sum_dist_pos = 0.0;
    double sum_dist_neg = 0.0;
    
    for (size_t i = 0; i < bins.size(); ++i) {
      dist_pos[i] = static_cast<double>(bins[i].count_pos) / total_pos;
      dist_neg[i] = static_cast<double>(bins[i].count_neg) / total_neg;
      sum_dist_pos += dist_pos[i];
      sum_dist_neg += dist_neg[i];
    }
    
    // Validate if distributions sum to 1
    if (std::abs(sum_dist_pos - 1.0) > 1e-6 || std::abs(sum_dist_neg - 1.0) > 1e-6) {
      Rcpp::stop("Positive or negative distributions do not sum to 1.");
    }
    
    // Calculate WoE and IV
    for (size_t i = 0; i < bins.size(); ++i) {
      if (dist_pos[i] > 0.0 && dist_neg[i] > 0.0) {
        bins[i].woe = std::log(dist_pos[i] / dist_neg[i]);
      } else if (dist_pos[i] == 0.0 && dist_neg[i] > 0.0) {
        // No positives in bin, assign WoE as -Inf
        bins[i].woe = -std::numeric_limits<double>::infinity();
      } else if (dist_neg[i] == 0.0 && dist_pos[i] > 0.0) {
        // No negatives in bin, assign WoE as +Inf
        bins[i].woe = std::numeric_limits<double>::infinity();
      } else {
        // Both dist_pos and dist_neg are zero, assign WoE as 0
        bins[i].woe = 0.0;
      }
      
      // Calculate IV
      if (dist_pos[i] > 0.0 || dist_neg[i] > 0.0) {
        bins[i].iv = (dist_pos[i] - dist_neg[i]) * bins[i].woe;
      } else {
        bins[i].iv = 0.0;
      }
    }
  }
  
  // Check if WoE is monotonic (increasing)
  bool is_monotonic() const {
    if (bins.empty()) return true;
    double prev_woe = bins[0].woe;
    for (size_t i = 1; i < bins.size(); ++i) {
      if (bins[i].woe < prev_woe) {
        return false;
      }
      prev_woe = bins[i].woe;
    }
    return true;
  }
  
  // Enforce monotonicity by merging bins where WoE decreases
  void enforce_monotonicity() {
    bool monotonic = false;
    while (!monotonic) {
      monotonic = true;
      for (size_t i = 1; i < bins.size(); ++i) {
        if (bins[i].woe < bins[i - 1].woe) {
          merge_bins(i - 1);
          monotonic = false;
          if (bins.size() <= min_bins) {
            return;
          }
          break;
        }
      }
    }
  }
  
  // Validate if bins are ordered, non-overlapping, and cover all feature values
  void validate_bins() const {
    if (bins.empty()) {
      Rcpp::stop("No bins available after binning process.");
    }
    
    // Ensure bins are ordered by upper_bound
    for (size_t i = 1; i < bins.size(); ++i) {
      if (bins[i - 1].upper_bound > bins[i].upper_bound) {
        Rcpp::stop("Bins are not correctly ordered by upper_bound.");
      }
    }
    
    // Ensure bins cover the entire range with no gaps
    // First bin should start with -Inf
    if (bins.front().lower_bound != -std::numeric_limits<double>::infinity()) {
      Rcpp::stop("First bin doesn't start with -Inf.");
    }
    
    // Last bin should end with +Inf
    if (bins.back().upper_bound != std::numeric_limits<double>::infinity()) {
      Rcpp::stop("Last bin doesn't end with +Inf.");
    }
    
    // Ensure no gaps between bins
    for (size_t i = 1; i < bins.size(); ++i) {
      if (bins[i].lower_bound != bins[i - 1].upper_bound) {
        Rcpp::stop("There's a gap between bin " + std::to_string(i - 1) + " and bin " + std::to_string(i));
      }
    }
  }
  
  // New method to find the correct bin for a value
  size_t find_bin(double value) const {
    for (size_t i = 0; i < bins.size(); ++i) {
      if (i == 0 && value < bins[i].upper_bound) {
        return i;
      } else if (i == bins.size() - 1 || (value >= bins[i].lower_bound && value < bins[i].upper_bound)) {
        return i;
      }
    }
    return bins.size() - 1; // Return last bin if not found (shouldn't happen)
  }
  
public:
  // Constructor with input validation
  OptimalBinningNumericalMDLP(
    const std::vector<double>& feature,
    const std::vector<int>& target,
    int min_bins = 3,
    int max_bins = 5,
    double bin_cutoff = 0.05,
    int max_n_prebins = 20
  ) : feature(feature), target(target), min_bins(min_bins), max_bins(max_bins),
  bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins) {
    
    // Basic input validations
    if (feature.empty()) {
      Rcpp::stop("Feature vector is empty.");
    }
    
    if (feature.size() != target.size()) {
      Rcpp::stop("Feature and target vectors must have the same size.");
    }
    
    if (min_bins < 2) {
      Rcpp::stop("min_bins must be at least 2.");
    }
    
    if (max_bins < min_bins) {
      Rcpp::stop("max_bins must be greater than or equal to min_bins.");
    }
    
    if (bin_cutoff <= 0.0 || bin_cutoff >= 1.0) {
      Rcpp::stop("bin_cutoff must be between 0 and 1.");
    }
    
    if (max_n_prebins < 2) {
      Rcpp::stop("max_n_prebins must be at least 2.");
    }
    
    // Check if target contains only 0 and 1
    for (const int t : target) {
      if (t != 0 && t != 1) {
        Rcpp::stop("Target vector must contain only 0 and 1.");
      }
    }
    
    // Check for NaN or Inf in feature
    for (const double f : feature) {
      if (std::isnan(f) || std::isinf(f)) {
        Rcpp::stop("Feature vector contains NaN or infinite values.");
      }
    }
    
    // Count unique values
    std::set<double> unique_values(feature.begin(), feature.end());
    int n_unique = static_cast<int>(unique_values.size());
    
    // Adjust min_bins and max_bins based on unique values
    if (n_unique < min_bins) {
      min_bins = n_unique;
      if (max_bins < min_bins) {
        max_bins = min_bins;
      }
    }
    
    if (n_unique < max_bins) {
      max_bins = n_unique;
    }
  }
  
  // Fit the binning model
  void fit() {
    // Sort feature and target together
    std::vector<std::pair<double, int>> sorted_data;
    sorted_data.reserve(feature.size());
    for (size_t i = 0; i < feature.size(); ++i) {
      sorted_data.emplace_back(feature[i], target[i]);
    }
    std::sort(sorted_data.begin(), sorted_data.end(),
              [](const std::pair<double, int>& a, const std::pair<double, int>& b) -> bool {
                return a.first < b.first;
              });
    
    // Handle case with all feature values identical
    if (sorted_data.front().first == sorted_data.back().first) {
      Bin bin;
      bin.lower_bound = -std::numeric_limits<double>::infinity();
      bin.upper_bound = std::numeric_limits<double>::infinity();
      bin.count = static_cast<int>(sorted_data.size());
      bin.count_pos = 0;
      bin.count_neg = 0;
      for (const auto& pair : sorted_data) {
        if (pair.second == 1) {
          bin.count_pos++;
        } else {
          bin.count_neg++;
        }
      }
      bins.push_back(bin);
      calculate_woe_iv();
      return;
    }
    
    // Create initial bins using equal-frequency binning
    int records_per_bin = std::max(1, static_cast<int>(sorted_data.size() / max_n_prebins));
    for (size_t i = 0; i < sorted_data.size(); i += records_per_bin) {
      size_t end = std::min(i + records_per_bin, sorted_data.size());
      Bin bin;
      
      // Set lower_bound correctly to avoid gaps
      if (i == 0) {
        bin.lower_bound = -std::numeric_limits<double>::infinity();
      } else {
        bin.lower_bound = sorted_data[i].first;
      }
      
      // Set upper_bound
      if (end == sorted_data.size()) {
        bin.upper_bound = std::numeric_limits<double>::infinity();
      } else {
        bin.upper_bound = sorted_data[end].first;
      }
      
      bin.count = static_cast<int>(end - i);
      bin.count_pos = 0;
      bin.count_neg = 0;
      
      for (size_t j = i; j < end; ++j) {
        if (sorted_data[j].second == 1) {
          bin.count_pos++;
        } else {
          bin.count_neg++;
        }
      }
      
      bins.push_back(bin);
    }
    
    // MDLP algorithm for bin merging
    while (bins.size() > min_bins) {
      double current_mdl = calculate_mdl_cost(bins);
      double best_mdl = current_mdl;
      size_t best_merge_index = bins.size();
      
      // Iterate over adjacent bins to find the best pair to merge
      for (size_t i = 0; i < bins.size() - 1; ++i) {
        std::vector<Bin> temp_bins = bins;
        // Merge bins i and i+1
        Bin& left = temp_bins[i];
        Bin& right = temp_bins[i + 1];
        
        left.upper_bound = right.upper_bound;
        left.count += right.count;
        left.count_pos += right.count_pos;
        left.count_neg += right.count_neg;
        
        temp_bins.erase(temp_bins.begin() + i + 1);
        
        double new_mdl = calculate_mdl_cost(temp_bins);
        
        if (new_mdl < best_mdl) {
          best_mdl = new_mdl;
          best_merge_index = i;
        }
      }
      
      // If a better MDL cost was found, merge the corresponding bins
      if (best_merge_index < bins.size()) {
        merge_bins(best_merge_index);
      } else {
        break; // No further improvement
      }
      
      // Stop if max_bins is reached
      if (bins.size() <= max_bins) {
        break;
      }
    }
    
    // Merge rare bins based on bin_cutoff
    double total_count = static_cast<double>(feature.size());
    bool merged = true;
    while (merged) {
      merged = false;
      for (size_t i = 0; i < bins.size(); ++i) {
        double bin_proportion = static_cast<double>(bins[i].count) / total_count;
        if (bin_proportion < bin_cutoff) {
          if (bins.size() == 1) {
            break; // Cannot merge further
          }
          if (i == 0) {
            merge_bins(0);
          } else {
            merge_bins(i - 1);
          }
          merged = true;
          break; // Restart the loop after merging
        }
      }
    }
    
    calculate_woe_iv();
    
    // Enforce monotonicity if possible
    if (bins.size() > min_bins && !is_monotonic()) {
      enforce_monotonicity();
    }
    
    // Final validation to ensure bins cover all feature values
    validate_bins();
  }
  
  // Get the binning results
  Rcpp::List get_results() const {
    // Prepare vector for WoE assignment
    Rcpp::NumericVector woefeature(feature.size());
    
    // Assign WoE values based on bin intervals
    for (size_t i = 0; i < feature.size(); ++i) {
      size_t bin_index = find_bin(feature[i]);
      woefeature[i] = bins[bin_index].woe;
    }
    
    // Prepare WoE bin details
    Rcpp::StringVector bin_labels;
    Rcpp::NumericVector woe_values;
    Rcpp::NumericVector iv_values;
    Rcpp::IntegerVector count_values;
    Rcpp::IntegerVector count_pos_values;
    Rcpp::IntegerVector count_neg_values;
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    
    for (const auto& bin : bins) {
      oss.str("");
      oss.clear();
      
      // Define lower bound (inclusive)
      if (bin.lower_bound == -std::numeric_limits<double>::infinity()) {
        oss << "[-Inf";
      } else {
        oss << "[" << bin.lower_bound;
      }
      
      oss << ";";
      
      // Define upper bound (exclusive)
      if (bin.upper_bound == std::numeric_limits<double>::infinity()) {
        oss << "+Inf)";
      } else {
        oss << bin.upper_bound << ")";
      }
      
      bin_labels.push_back(oss.str());
      woe_values.push_back(bin.woe);
      iv_values.push_back(bin.iv);
      count_values.push_back(bin.count);
      count_pos_values.push_back(bin.count_pos);
      count_neg_values.push_back(bin.count_neg);
    }
    
    // Create WoE bins list
    Rcpp::List woebin;
    woebin["bin"] = bin_labels;
    woebin["woe"] = woe_values;
    woebin["iv"] = iv_values;
    woebin["count"] = count_values;
    woebin["count_pos"] = count_pos_values;
    woebin["count_neg"] = count_neg_values;
    
    // Verify total count
    int total_count = 0;
    for (const auto& bin : bins) {
      total_count += bin.count;
    }
    if (total_count != static_cast<int>(feature.size())) {
      Rcpp::warning("Total count in bins does not match feature size");
    }
    
    return Rcpp::List::create(
      Rcpp::Named("woefeature") = woefeature,
      Rcpp::Named("woebin") = woebin
    );
  }
};

// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_mdlp(
    Rcpp::IntegerVector target,
    Rcpp::NumericVector feature,
    int min_bins = 3,
    int max_bins = 5,
    double bin_cutoff = 0.05,
    int max_n_prebins = 20
) {
  // Input validation is handled within the class constructor
  
  // Convert Rcpp vectors to std::vector
  std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
  std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
  
  // Create and run the binning
  OptimalBinningNumericalMDLP binner(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins);
  binner.fit();
  return binner.get_results();
}


// #include <Rcpp.h>
// #include <vector>
// #include <algorithm>
// #include <cmath>
// #include <limits>
// 
// #ifdef _OPENMP
// #include <omp.h>
// #endif
// 
// // [[Rcpp::plugins(openmp)]]
// 
// class OptimalBinningNumericalMDLP {
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
//   double calculate_entropy(int pos, int neg) {
//     if (pos == 0 || neg == 0) return 0.0;
//     double p = static_cast<double>(pos) / (pos + neg);
//     return -p * std::log2(p) - (1 - p) * std::log2(1 - p);
//   }
// 
//   double calculate_mdl_cost(const std::vector<Bin>& current_bins) {
//     double total_count = 0;
//     double total_pos = 0;
//     for (const auto& bin : current_bins) {
//       total_count += bin.count;
//       total_pos += bin.count_pos;
//     }
//     double total_neg = total_count - total_pos;
// 
//     double model_cost = std::log2(current_bins.size() - 1);
//     double data_cost = total_count * calculate_entropy(total_pos, total_neg);
// 
//     for (const auto& bin : current_bins) {
//       data_cost -= bin.count * calculate_entropy(bin.count_pos, bin.count_neg);
//     }
// 
//     return model_cost + data_cost;
//   }
// 
//   void merge_bins(size_t index) {
//     Bin& left = bins[index];
//     Bin& right = bins[index + 1];
// 
//     left.upper_bound = right.upper_bound;
//     left.count += right.count;
//     left.count_pos += right.count_pos;
//     left.count_neg += right.count_neg;
// 
//     bins.erase(bins.begin() + index + 1);
//   }
// 
//   void calculate_woe_iv() {
//     double total_pos = 0, total_neg = 0;
//     for (const auto& bin : bins) {
//       total_pos += bin.count_pos;
//       total_neg += bin.count_neg;
//     }
// 
//     for (auto& bin : bins) {
//       double pos_rate = (double)bin.count_pos / total_pos;
//       double neg_rate = (double)bin.count_neg / total_neg;
//       bin.woe = std::log(pos_rate / neg_rate);
//       bin.iv = (pos_rate - neg_rate) * bin.woe;
//     }
//   }
// 
// public:
//   OptimalBinningNumericalMDLP(
//     const std::vector<double>& feature,
//     const std::vector<int>& target,
//     int min_bins = 3,
//     int max_bins = 5,
//     double bin_cutoff = 0.05,
//     int max_n_prebins = 20
//   ) : feature(feature), target(target), min_bins(min_bins), max_bins(max_bins),
//   bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins) {
// 
//     if (feature.size() != target.size()) {
//       Rcpp::stop("Feature and target vectors must have the same length");
//     }
// 
//     if (min_bins < 2 || max_bins < min_bins) {
//       Rcpp::stop("Invalid bin constraints");
//     }
//   }
// 
//   void fit() {
//     // Initial binning
//     std::vector<std::pair<double, int>> sorted_data;
//     for (size_t i = 0; i < feature.size(); ++i) {
//       sorted_data.push_back({feature[i], target[i]});
//     }
//     std::sort(sorted_data.begin(), sorted_data.end());
// 
//     // Create initial bins
//     int records_per_bin = std::max(1, (int)sorted_data.size() / max_n_prebins);
//     for (size_t i = 0; i < sorted_data.size(); i += records_per_bin) {
//       size_t end = std::min(i + records_per_bin, sorted_data.size());
//       Bin bin;
//       bin.lower_bound = (i == 0) ? -std::numeric_limits<double>::infinity() : sorted_data[i].first;
//       bin.upper_bound = (end == sorted_data.size()) ? std::numeric_limits<double>::infinity() : sorted_data[end - 1].first;
//       bin.count = end - i;
//       bin.count_pos = 0;
//       bin.count_neg = 0;
// 
//       for (size_t j = i; j < end; ++j) {
//         if (sorted_data[j].second == 1) {
//           bin.count_pos++;
//         } else {
//           bin.count_neg++;
//         }
//       }
// 
//       bins.push_back(bin);
//     }
// 
//     // MDLP algorithm
//     while (bins.size() > min_bins) {
//       double current_mdl = calculate_mdl_cost(bins);
//       double best_mdl = current_mdl;
//       size_t best_merge_index = bins.size();
// 
// #pragma omp parallel for
//       for (size_t i = 0; i < bins.size() - 1; ++i) {
//         std::vector<Bin> temp_bins = bins;
//         Bin& left = temp_bins[i];
//         Bin& right = temp_bins[i + 1];
// 
//         left.upper_bound = right.upper_bound;
//         left.count += right.count;
//         left.count_pos += right.count_pos;
//         left.count_neg += right.count_neg;
// 
//         temp_bins.erase(temp_bins.begin() + i + 1);
// 
//         double new_mdl = calculate_mdl_cost(temp_bins);
// 
// #pragma omp critical
// {
//   if (new_mdl < best_mdl) {
//     best_mdl = new_mdl;
//     best_merge_index = i;
//   }
// }
//       }
// 
//       if (best_merge_index < bins.size()) {
//         merge_bins(best_merge_index);
//       } else {
//         break;
//       }
// 
//       if (bins.size() <= max_bins) {
//         break;
//       }
//     }
// 
//     // Merge rare bins
//     for (auto it = bins.begin(); it != bins.end(); ) {
//       if ((double)it->count / feature.size() < bin_cutoff) {
//         if (it == bins.begin()) {
//           merge_bins(0);
//         } else {
//           merge_bins(std::distance(bins.begin(), it) - 1);
//         }
//       } else {
//         ++it;
//       }
//     }
// 
//     calculate_woe_iv();
//   }
// 
//   Rcpp::List get_results() {
//     Rcpp::NumericVector woefeature(feature.size());
//     Rcpp::List woebin;
//     Rcpp::StringVector bin_labels;
//     Rcpp::NumericVector woe_values;
//     Rcpp::NumericVector iv_values;
//     Rcpp::IntegerVector count_values;
//     Rcpp::IntegerVector count_pos_values;
//     Rcpp::IntegerVector count_neg_values;
// 
//     for (size_t i = 0; i < feature.size(); ++i) {
//       for (const auto& bin : bins) {
//         if (feature[i] <= bin.upper_bound) {
//           woefeature[i] = bin.woe;
//           break;
//         }
//       }
//     }
// 
//     for (const auto& bin : bins) {
//       std::string bin_label = (bin.lower_bound == -std::numeric_limits<double>::infinity() ? "(-Inf" : "(" + std::to_string(bin.lower_bound)) +
//         ";" + (bin.upper_bound == std::numeric_limits<double>::infinity() ? "+Inf]" : std::to_string(bin.upper_bound) + "]");
//       bin_labels.push_back(bin_label);
//       woe_values.push_back(bin.woe);
//       iv_values.push_back(bin.iv);
//       count_values.push_back(bin.count);
//       count_pos_values.push_back(bin.count_pos);
//       count_neg_values.push_back(bin.count_neg);
//     }
// 
//     woebin["bin"] = bin_labels;
//     woebin["woe"] = woe_values;
//     woebin["iv"] = iv_values;
//     woebin["count"] = count_values;
//     woebin["count_pos"] = count_pos_values;
//     woebin["count_neg"] = count_neg_values;
// 
//     return Rcpp::List::create(
//       Rcpp::Named("woefeature") = woefeature,
//       Rcpp::Named("woebin") = woebin
//     );
//   }
// };
// 
// 
// //' @title Optimal Binning for Numerical Variables using MDLP
// //' 
// //' @description
// //' This function performs optimal binning for numerical variables using the Minimum Description Length Principle (MDLP). It creates optimal bins for a numerical feature based on its relationship with a binary target variable, maximizing the predictive power while respecting user-defined constraints.
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
// //'     \item count: Integer vector of total observations in each bin.
// //'     \item count_pos: Integer vector of positive target observations in each bin.
// //'     \item count_neg: Integer vector of negative target observations in each bin.
// //'   }
// //' }
// //' 
// //' @details
// //' The Optimal Binning algorithm for numerical variables using MDLP works as follows:
// //' 1. Create initial bins using equal-frequency binning.
// //' 2. Apply the MDLP algorithm to merge bins:
// //'    - Calculate the current MDL cost.
// //'    - For each pair of adjacent bins, calculate the MDL cost if merged.
// //'    - Merge the pair with the lowest MDL cost.
// //'    - Repeat until no further merging reduces the MDL cost or the minimum number of bins is reached.
// //' 3. Merge rare bins (those with a proportion less than bin_cutoff).
// //' 4. Calculate Weight of Evidence (WoE) and Information Value (IV) for each bin:
// //'    \deqn{WoE = \ln\left(\frac{\text{Positive Rate}}{\text{Negative Rate}}\right)}
// //'    \deqn{IV = (\text{Positive Rate} - \text{Negative Rate}) \times WoE}
// //' 
// //' The MDLP algorithm aims to find the optimal trade-off between model complexity (number of bins) and goodness of fit. It uses the principle of minimum description length, which states that the best model is the one that provides the shortest description of the data.
// //' 
// //' The MDL cost is calculated as:
// //' \deqn{MDL = \log_2(k - 1) + n \times H(S) - \sum_{i=1}^k n_i \times H(S_i)}
// //' where k is the number of bins, n is the total number of instances, H(S) is the entropy of the entire dataset, and H(S_i) is the entropy of the i-th bin.
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
// //' result <- optimal_binning_numerical_mdlp(target, feature, min_bins = 2, max_bins = 4)
// //' 
// //' # Print results
// //' print(result$woebin)
// //' 
// //' # Plot WoE values
// //' plot(result$woebin$woe, type = "s", xaxt = "n", xlab = "Bins", ylab = "WoE",
// //'      main = "Weight of Evidence by Bin")
// //' axis(1, at = 1:nrow(result$woebin), labels = result$woebin$bin)
// //' }
// //' 
// //' @references
// //' \itemize{
// //' \item Fayyad, U. M., & Irani, K. B. (1993). Multi-interval discretization of continuous-valued attributes for classification learning. In Proceedings of the 13th International Joint Conference on Artificial Intelligence (pp. 1022-1027).
// //' \item Rissanen, J. (1978). Modeling by shortest data description. Automatica, 14(5), 465-471.
// //' }
// //' 
// //' @author Lopes, J. E.
// //' 
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_numerical_mdlp(
//     Rcpp::IntegerVector target,
//     Rcpp::NumericVector feature,
//     int min_bins = 3,
//     int max_bins = 5,
//     double bin_cutoff = 0.05,
//     int max_n_prebins = 20
// ) {
//   std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
//   std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
// 
//   OptimalBinningNumericalMDLP binner(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins);
//   binner.fit();
//   return binner.get_results();
// }
// 
