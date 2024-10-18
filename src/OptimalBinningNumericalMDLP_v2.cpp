#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <sstream>
#include <iomanip>
#include <set>

// No OpenMP dependencies as per instruction

class OptimalBinningNumericalMDLP {
private:
  std::vector<double> feature;
  std::vector<int> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  double convergence_threshold;
  int max_iterations;
  
  int iterations_run;
  bool converged;
  
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
    
    for (size_t i = 0; i < bins.size(); ++i) {
      dist_pos[i] = static_cast<double>(bins[i].count_pos) / total_pos;
      dist_neg[i] = static_cast<double>(bins[i].count_neg) / total_neg;
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
    while (!monotonic && iterations_run < max_iterations) {
      iterations_run++;
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
    if (iterations_run >= max_iterations) {
      converged = false;
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
  
  // Find the correct bin for a value
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
    int max_n_prebins = 20,
    double convergence_threshold = 1e-6,
    int max_iterations = 1000
  ) : feature(feature), target(target), min_bins(min_bins), max_bins(max_bins),
  bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins),
  convergence_threshold(convergence_threshold), max_iterations(max_iterations),
  iterations_run(0), converged(true) {
    
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
    while (bins.size() > min_bins && iterations_run < max_iterations) {
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
      
      iterations_run++;
    }
    
    if (iterations_run >= max_iterations) {
      converged = false;
    }
    
    // Merge rare bins based on bin_cutoff
    double total_count = static_cast<double>(feature.size());
    bool merged = true;
    while (merged && iterations_run < max_iterations) {
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
      iterations_run++;
    }
    
    if (iterations_run >= max_iterations) {
      converged = false;
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
    // Prepare WoE bin details
    Rcpp::StringVector bin_labels;
    Rcpp::NumericVector woe_values;
    Rcpp::NumericVector iv_values;
    Rcpp::IntegerVector count_values;
    Rcpp::IntegerVector count_pos_values;
    Rcpp::IntegerVector count_neg_values;
    Rcpp::NumericVector cutpoints;
    
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
        cutpoints.push_back(bin.upper_bound);
      }
      
      bin_labels.push_back(oss.str());
      woe_values.push_back(bin.woe);
      iv_values.push_back(bin.iv);
      count_values.push_back(bin.count);
      count_pos_values.push_back(bin.count_pos);
      count_neg_values.push_back(bin.count_neg);
    }
    
    // Verify total count
    int total_count = 0;
    for (const auto& bin : bins) {
      total_count += bin.count;
    }
    if (total_count != static_cast<int>(feature.size())) {
      Rcpp::warning("Total count in bins does not match feature size");
    }
    
    // Create WoE bins list
    Rcpp::List woebin = Rcpp::List::create(
      Rcpp::Named("bin") = bin_labels,
      Rcpp::Named("woe") = woe_values,
      Rcpp::Named("iv") = iv_values,
      Rcpp::Named("count") = count_values,
      Rcpp::Named("count_pos") = count_pos_values,
      Rcpp::Named("count_neg") = count_neg_values,
      Rcpp::Named("cutpoints") = cutpoints,
      Rcpp::Named("converged") = converged,
      Rcpp::Named("iterations") = iterations_run
    );
    
    return woebin;
  }
};

//' Optimal Binning for Numerical Features using MDLP
//'
//' This function performs optimal binning on a numerical feature using the Minimum Description Length Principle (MDLP).
//'
//' @param target An integer vector containing binary target values (0 or 1).
//' @param feature A numeric vector containing the feature values to be binned.
//' @param min_bins The minimum number of bins to produce.
//' @param max_bins The maximum number of bins to produce.
//' @param bin_cutoff The minimum proportion of records allowed in a bin.
//' @param max_n_prebins The maximum number of pre-bins to create.
//' @param convergence_threshold The convergence threshold for the algorithm.
//' @param max_iterations The maximum number of iterations allowed.
//' @return A list containing the binning results, including bins, WoE, IV, counts, cutpoints, convergence status, and iterations run.
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_mdlp(
   Rcpp::IntegerVector target,
   Rcpp::NumericVector feature,
   int min_bins = 3,
   int max_bins = 5,
   double bin_cutoff = 0.05,
   int max_n_prebins = 20,
   double convergence_threshold = 1e-6,
   int max_iterations = 1000
) {
 // Convert Rcpp vectors to std::vector
 std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
 std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
 
 // Create and run the binning
 OptimalBinningNumericalMDLP binner(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins, convergence_threshold, max_iterations);
 binner.fit();
 return binner.get_results();
}









// #include <Rcpp.h>
// #include <vector>
// #include <algorithm>
// #include <cmath>
// #include <limits>
// #include <string>
// #include <sstream>
// #include <iomanip>
// #include <set>
// 
// // Enable OpenMP if available
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
//     double lower_bound; // Inclusive
//     double upper_bound; // Exclusive
//     int count;
//     int count_pos;
//     int count_neg;
//     double woe;
//     double iv;
//   };
//   
//   std::vector<Bin> bins;
//   
//   // Calculate entropy for a bin
//   double calculate_entropy(int pos, int neg) const {
//     if (pos == 0 || neg == 0) return 0.0;
//     double total = static_cast<double>(pos + neg);
//     double p = pos / total;
//     double q = neg / total;
//     return -p * std::log2(p) - q * std::log2(q);
//   }
//   
//   // Calculate MDL cost for current bins
//   double calculate_mdl_cost(const std::vector<Bin>& current_bins) const {
//     int total_count = 0;
//     int total_pos = 0;
//     for (const auto& bin : current_bins) {
//       total_count += bin.count;
//       total_pos += bin.count_pos;
//     }
//     int total_neg = total_count - total_pos;
//     
//     // Handle cases with all positives or all negatives
//     if (total_pos == 0 || total_neg == 0) {
//       return std::numeric_limits<double>::infinity();
//     }
//     
//     double model_cost = std::log2(static_cast<double>(current_bins.size()) - 1.0);
//     double data_cost = total_count * calculate_entropy(total_pos, total_neg);
//     
//     for (const auto& bin : current_bins) {
//       data_cost -= bin.count * calculate_entropy(bin.count_pos, bin.count_neg);
//     }
//     
//     return model_cost + data_cost;
//   }
//   
//   // Merge bin at specified index with the next bin
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
//   // Calculate WoE and IV for each bin
//   void calculate_woe_iv() {
//     int total_pos = 0, total_neg = 0;
//     for (const auto& bin : bins) {
//       total_pos += bin.count_pos;
//       total_neg += bin.count_neg;
//     }
//     
//     // Check if total_pos and total_neg are zero
//     if (total_pos == 0 || total_neg == 0) {
//       Rcpp::stop("All target values are the same. WoE and IV cannot be calculated.");
//     }
//     
//     // Calculate distributions
//     std::vector<double> dist_pos(bins.size(), 0.0);
//     std::vector<double> dist_neg(bins.size(), 0.0);
//     double sum_dist_pos = 0.0;
//     double sum_dist_neg = 0.0;
//     
//     for (size_t i = 0; i < bins.size(); ++i) {
//       dist_pos[i] = static_cast<double>(bins[i].count_pos) / total_pos;
//       dist_neg[i] = static_cast<double>(bins[i].count_neg) / total_neg;
//       sum_dist_pos += dist_pos[i];
//       sum_dist_neg += dist_neg[i];
//     }
//     
//     // Validate if distributions sum to 1
//     if (std::abs(sum_dist_pos - 1.0) > 1e-6 || std::abs(sum_dist_neg - 1.0) > 1e-6) {
//       Rcpp::stop("Positive or negative distributions do not sum to 1.");
//     }
//     
//     // Calculate WoE and IV
//     for (size_t i = 0; i < bins.size(); ++i) {
//       if (dist_pos[i] > 0.0 && dist_neg[i] > 0.0) {
//         bins[i].woe = std::log(dist_pos[i] / dist_neg[i]);
//       } else if (dist_pos[i] == 0.0 && dist_neg[i] > 0.0) {
//         // No positives in bin, assign WoE as -Inf
//         bins[i].woe = -std::numeric_limits<double>::infinity();
//       } else if (dist_neg[i] == 0.0 && dist_pos[i] > 0.0) {
//         // No negatives in bin, assign WoE as +Inf
//         bins[i].woe = std::numeric_limits<double>::infinity();
//       } else {
//         // Both dist_pos and dist_neg are zero, assign WoE as 0
//         bins[i].woe = 0.0;
//       }
//       
//       // Calculate IV
//       if (dist_pos[i] > 0.0 || dist_neg[i] > 0.0) {
//         bins[i].iv = (dist_pos[i] - dist_neg[i]) * bins[i].woe;
//       } else {
//         bins[i].iv = 0.0;
//       }
//     }
//   }
//   
//   // Check if WoE is monotonic (increasing)
//   bool is_monotonic() const {
//     if (bins.empty()) return true;
//     double prev_woe = bins[0].woe;
//     for (size_t i = 1; i < bins.size(); ++i) {
//       if (bins[i].woe < prev_woe) {
//         return false;
//       }
//       prev_woe = bins[i].woe;
//     }
//     return true;
//   }
//   
//   // Enforce monotonicity by merging bins where WoE decreases
//   void enforce_monotonicity() {
//     bool monotonic = false;
//     while (!monotonic) {
//       monotonic = true;
//       for (size_t i = 1; i < bins.size(); ++i) {
//         if (bins[i].woe < bins[i - 1].woe) {
//           merge_bins(i - 1);
//           monotonic = false;
//           if (bins.size() <= min_bins) {
//             return;
//           }
//           break;
//         }
//       }
//     }
//   }
//   
//   // Validate if bins are ordered, non-overlapping, and cover all feature values
//   void validate_bins() const {
//     if (bins.empty()) {
//       Rcpp::stop("No bins available after binning process.");
//     }
//     
//     // Ensure bins are ordered by upper_bound
//     for (size_t i = 1; i < bins.size(); ++i) {
//       if (bins[i - 1].upper_bound > bins[i].upper_bound) {
//         Rcpp::stop("Bins are not correctly ordered by upper_bound.");
//       }
//     }
//     
//     // Ensure bins cover the entire range with no gaps
//     // First bin should start with -Inf
//     if (bins.front().lower_bound != -std::numeric_limits<double>::infinity()) {
//       Rcpp::stop("First bin doesn't start with -Inf.");
//     }
//     
//     // Last bin should end with +Inf
//     if (bins.back().upper_bound != std::numeric_limits<double>::infinity()) {
//       Rcpp::stop("Last bin doesn't end with +Inf.");
//     }
//     
//     // Ensure no gaps between bins
//     for (size_t i = 1; i < bins.size(); ++i) {
//       if (bins[i].lower_bound != bins[i - 1].upper_bound) {
//         Rcpp::stop("There's a gap between bin " + std::to_string(i - 1) + " and bin " + std::to_string(i));
//       }
//     }
//   }
//   
//   // New method to find the correct bin for a value
//   size_t find_bin(double value) const {
//     for (size_t i = 0; i < bins.size(); ++i) {
//       if (i == 0 && value < bins[i].upper_bound) {
//         return i;
//       } else if (i == bins.size() - 1 || (value >= bins[i].lower_bound && value < bins[i].upper_bound)) {
//         return i;
//       }
//     }
//     return bins.size() - 1; // Return last bin if not found (shouldn't happen)
//   }
//   
// public:
//   // Constructor with input validation
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
//     // Basic input validations
//     if (feature.empty()) {
//       Rcpp::stop("Feature vector is empty.");
//     }
//     
//     if (feature.size() != target.size()) {
//       Rcpp::stop("Feature and target vectors must have the same size.");
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
//     if (bin_cutoff <= 0.0 || bin_cutoff >= 1.0) {
//       Rcpp::stop("bin_cutoff must be between 0 and 1.");
//     }
//     
//     if (max_n_prebins < 2) {
//       Rcpp::stop("max_n_prebins must be at least 2.");
//     }
//     
//     // Check if target contains only 0 and 1
//     for (const int t : target) {
//       if (t != 0 && t != 1) {
//         Rcpp::stop("Target vector must contain only 0 and 1.");
//       }
//     }
//     
//     // Check for NaN or Inf in feature
//     for (const double f : feature) {
//       if (std::isnan(f) || std::isinf(f)) {
//         Rcpp::stop("Feature vector contains NaN or infinite values.");
//       }
//     }
//     
//     // Count unique values
//     std::set<double> unique_values(feature.begin(), feature.end());
//     int n_unique = static_cast<int>(unique_values.size());
//     
//     // Adjust min_bins and max_bins based on unique values
//     if (n_unique < min_bins) {
//       min_bins = n_unique;
//       if (max_bins < min_bins) {
//         max_bins = min_bins;
//       }
//     }
//     
//     if (n_unique < max_bins) {
//       max_bins = n_unique;
//     }
//   }
//   
//   // Fit the binning model
//   void fit() {
//     // Sort feature and target together
//     std::vector<std::pair<double, int>> sorted_data;
//     sorted_data.reserve(feature.size());
//     for (size_t i = 0; i < feature.size(); ++i) {
//       sorted_data.emplace_back(feature[i], target[i]);
//     }
//     std::sort(sorted_data.begin(), sorted_data.end(),
//               [](const std::pair<double, int>& a, const std::pair<double, int>& b) -> bool {
//                 return a.first < b.first;
//               });
//     
//     // Handle case with all feature values identical
//     if (sorted_data.front().first == sorted_data.back().first) {
//       Bin bin;
//       bin.lower_bound = -std::numeric_limits<double>::infinity();
//       bin.upper_bound = std::numeric_limits<double>::infinity();
//       bin.count = static_cast<int>(sorted_data.size());
//       bin.count_pos = 0;
//       bin.count_neg = 0;
//       for (const auto& pair : sorted_data) {
//         if (pair.second == 1) {
//           bin.count_pos++;
//         } else {
//           bin.count_neg++;
//         }
//       }
//       bins.push_back(bin);
//       calculate_woe_iv();
//       return;
//     }
//     
//     // Create initial bins using equal-frequency binning
//     int records_per_bin = std::max(1, static_cast<int>(sorted_data.size() / max_n_prebins));
//     for (size_t i = 0; i < sorted_data.size(); i += records_per_bin) {
//       size_t end = std::min(i + records_per_bin, sorted_data.size());
//       Bin bin;
//       
//       // Set lower_bound correctly to avoid gaps
//       if (i == 0) {
//         bin.lower_bound = -std::numeric_limits<double>::infinity();
//       } else {
//         bin.lower_bound = sorted_data[i].first;
//       }
//       
//       // Set upper_bound
//       if (end == sorted_data.size()) {
//         bin.upper_bound = std::numeric_limits<double>::infinity();
//       } else {
//         bin.upper_bound = sorted_data[end].first;
//       }
//       
//       bin.count = static_cast<int>(end - i);
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
//     // MDLP algorithm for bin merging
//     while (bins.size() > min_bins) {
//       double current_mdl = calculate_mdl_cost(bins);
//       double best_mdl = current_mdl;
//       size_t best_merge_index = bins.size();
//       
//       // Iterate over adjacent bins to find the best pair to merge
//       for (size_t i = 0; i < bins.size() - 1; ++i) {
//         std::vector<Bin> temp_bins = bins;
//         // Merge bins i and i+1
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
//         if (new_mdl < best_mdl) {
//           best_mdl = new_mdl;
//           best_merge_index = i;
//         }
//       }
//       
//       // If a better MDL cost was found, merge the corresponding bins
//       if (best_merge_index < bins.size()) {
//         merge_bins(best_merge_index);
//       } else {
//         break; // No further improvement
//       }
//       
//       // Stop if max_bins is reached
//       if (bins.size() <= max_bins) {
//         break;
//       }
//     }
//     
//     // Merge rare bins based on bin_cutoff
//     double total_count = static_cast<double>(feature.size());
//     bool merged = true;
//     while (merged) {
//       merged = false;
//       for (size_t i = 0; i < bins.size(); ++i) {
//         double bin_proportion = static_cast<double>(bins[i].count) / total_count;
//         if (bin_proportion < bin_cutoff) {
//           if (bins.size() == 1) {
//             break; // Cannot merge further
//           }
//           if (i == 0) {
//             merge_bins(0);
//           } else {
//             merge_bins(i - 1);
//           }
//           merged = true;
//           break; // Restart the loop after merging
//         }
//       }
//     }
//     
//     calculate_woe_iv();
//     
//     // Enforce monotonicity if possible
//     if (bins.size() > min_bins && !is_monotonic()) {
//       enforce_monotonicity();
//     }
//     
//     // Final validation to ensure bins cover all feature values
//     validate_bins();
//   }
//   
//   // Get the binning results
//   Rcpp::List get_results() const {
//     // Prepare vector for WoE assignment
//     Rcpp::NumericVector woefeature(feature.size());
//     
//     // Assign WoE values based on bin intervals
//     for (size_t i = 0; i < feature.size(); ++i) {
//       size_t bin_index = find_bin(feature[i]);
//       woefeature[i] = bins[bin_index].woe;
//     }
//     
//     // Prepare WoE bin details
//     Rcpp::StringVector bin_labels;
//     Rcpp::NumericVector woe_values;
//     Rcpp::NumericVector iv_values;
//     Rcpp::IntegerVector count_values;
//     Rcpp::IntegerVector count_pos_values;
//     Rcpp::IntegerVector count_neg_values;
//     
//     std::ostringstream oss;
//     oss << std::fixed << std::setprecision(6);
//     
//     for (const auto& bin : bins) {
//       oss.str("");
//       oss.clear();
//       
//       // Define lower bound (inclusive)
//       if (bin.lower_bound == -std::numeric_limits<double>::infinity()) {
//         oss << "[-Inf";
//       } else {
//         oss << "[" << bin.lower_bound;
//       }
//       
//       oss << ";";
//       
//       // Define upper bound (exclusive)
//       if (bin.upper_bound == std::numeric_limits<double>::infinity()) {
//         oss << "+Inf)";
//       } else {
//         oss << bin.upper_bound << ")";
//       }
//       
//       bin_labels.push_back(oss.str());
//       woe_values.push_back(bin.woe);
//       iv_values.push_back(bin.iv);
//       count_values.push_back(bin.count);
//       count_pos_values.push_back(bin.count_pos);
//       count_neg_values.push_back(bin.count_neg);
//     }
//     
//     // Create WoE bins list
//     Rcpp::List woebin;
//     woebin["bin"] = bin_labels;
//     woebin["woe"] = woe_values;
//     woebin["iv"] = iv_values;
//     woebin["count"] = count_values;
//     woebin["count_pos"] = count_pos_values;
//     woebin["count_neg"] = count_neg_values;
//     
//     // Verify total count
//     int total_count = 0;
//     for (const auto& bin : bins) {
//       total_count += bin.count;
//     }
//     if (total_count != static_cast<int>(feature.size())) {
//       Rcpp::warning("Total count in bins does not match feature size");
//     }
//     
//     return Rcpp::List::create(
//       Rcpp::Named("woefeature") = woefeature,
//       Rcpp::Named("woebin") = woebin
//     );
//   }
// };
// 
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_numerical_mdlp(
//     Rcpp::IntegerVector target,
//     Rcpp::NumericVector feature,
//     int min_bins = 3,
//     int max_bins = 5,
//     double bin_cutoff = 0.05,
//     int max_n_prebins = 20
// ) {
//   // Input validation is handled within the class constructor
//   
//   // Convert Rcpp vectors to std::vector
//   std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
//   std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
//   
//   // Create and run the binning
//   OptimalBinningNumericalMDLP binner(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins);
//   binner.fit();
//   return binner.get_results();
// }
