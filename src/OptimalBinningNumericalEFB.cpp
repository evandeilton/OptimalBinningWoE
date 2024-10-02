#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#include <sstream>
#include <iomanip>
#include <unordered_set>

#ifdef _OPENMP
#include <omp.h>
#endif

class OptimalBinningNumericalEFB {
private:
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  std::vector<double> feature;
  std::vector<int> target;
  std::vector<double> bin_edges;
  std::vector<double> woe_values;
  std::vector<double> iv_values;
  
  struct Bin {
    double lower_bound;
    double upper_bound;
    int count;
    int count_pos;
    int count_neg;
    double woe;
    double iv;
    
    Bin(double lb = -std::numeric_limits<double>::infinity(),
        double ub = std::numeric_limits<double>::infinity(),
        int c = 0, int cp = 0, int cn = 0)
      : lower_bound(lb), upper_bound(ub), count(c),
        count_pos(cp), count_neg(cn), woe(0.0), iv(0.0) {}
  };
  
  std::vector<Bin> bins;
  
  void validate_inputs() {
    // Check feature and target size
    if (feature.empty()) {
      Rcpp::stop("Feature vector is empty.");
    }
    if (feature.size() != target.size()) {
      Rcpp::stop("Feature and target must have the same length.");
    }
    
    // Check target values
    for (const auto& t : target) {
      if (t != 0 && t != 1) {
        Rcpp::stop("Target vector must contain only 0 and 1.");
      }
    }
    
    // Determine unique categories
    std::unordered_set<double> unique_values(feature.begin(), feature.end());
    int unique_count = unique_values.size();
    
    // Adjust min_bins if unique categories are less than min_bins
    if (unique_count <= 2) {
      min_bins = unique_count;
      max_bins = unique_count;
      if (bin_cutoff > 1.0 || bin_cutoff <= 0.0) {
        bin_cutoff = 1.0; // All data in one bin
      }
    } else {
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
    }
  }
  
  std::string double_to_string(double value) {
    if (std::isinf(value)) {
      return value > 0 ? "+Inf" : "-Inf";
    }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6) << value;
    return ss.str();
  }
  
  void create_prebins() {
    std::vector<std::pair<double, int>> feature_target;
    feature_target.reserve(feature.size());
    for (size_t i = 0; i < feature.size(); ++i) {
      feature_target.emplace_back(feature[i], target[i]);
    }
    
    // Sort by feature value
    std::sort(feature_target.begin(), feature_target.end(),
              [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                return a.first < b.first;
              });
    
    int n = feature_target.size();
    int prebins = std::min(max_n_prebins, n);
    
    // Calculate the number of observations per prebin
    int observations_per_bin = static_cast<int>(std::ceil(static_cast<double>(n) / prebins));
    
    bin_edges.clear();
    bin_edges.emplace_back(-std::numeric_limits<double>::infinity());
    
    for (int i = 1; i < prebins; ++i) {
      int index = i * observations_per_bin;
      if (index >= n) {
        index = n - 1;
      }
      // Ensure unique bin edges
      double edge = feature_target[index].first;
      if (edge <= bin_edges.back()) {
        edge = bin_edges.back() + 1e-6; // Add small epsilon to ensure uniqueness
      }
      bin_edges.emplace_back(edge);
    }
    
    bin_edges.emplace_back(std::numeric_limits<double>::infinity());
  }
  
  void calculate_bin_statistics() {
    bins.clear();
    bins.reserve(bin_edges.size() - 1);
    for (size_t i = 0; i < bin_edges.size() - 1; ++i) {
      bins.emplace_back(bin_edges[i], bin_edges[i + 1]);
    }
    
    // Parallel binning
#ifdef _OPENMP
#pragma omp parallel
#endif
{
#ifdef _OPENMP
#pragma omp for nowait
#endif
  for (size_t i = 0; i < feature.size(); ++i) {
    double value = feature[i];
    int target_value = target[i];
    // Binary search to find the bin
    int bin_index = std::upper_bound(bin_edges.begin(), bin_edges.end(), value) - bin_edges.begin() - 1;
    if (bin_index < 0) {
      bin_index = 0;
    } else if (bin_index >= static_cast<int>(bins.size())) {
      bin_index = bins.size() - 1;
    }
    
#ifdef _OPENMP
#pragma omp atomic
#endif
    bins[bin_index].count++;
    
    if (target_value == 1) {
#ifdef _OPENMP
#pragma omp atomic
#endif
      bins[bin_index].count_pos++;
    } else {
#ifdef _OPENMP
#pragma omp atomic
#endif
      bins[bin_index].count_neg++;
    }
  }
}
  }
  
  void merge_rare_bins() {
    int total_count = 0;
    for (const auto& bin : bins) {
      total_count += bin.count;
    }
    
    double cutoff_count = bin_cutoff * total_count;
    
    std::vector<Bin> merged_bins;
    merged_bins.reserve(bins.size());
    
    Bin current_bin = bins[0];
    
    for (size_t i = 1; i < bins.size(); ++i) {
      if (current_bin.count < cutoff_count) {
        // Merge with the next bin
        current_bin.upper_bound = bins[i].upper_bound;
        current_bin.count += bins[i].count;
        current_bin.count_pos += bins[i].count_pos;
        current_bin.count_neg += bins[i].count_neg;
      } else {
        merged_bins.emplace_back(current_bin);
        current_bin = bins[i];
      }
    }
    
    // Add the last bin
    merged_bins.emplace_back(current_bin);
    
    // If after merging, some bins still have counts < cutoff, merge them iteratively
    bool merged = true;
    while (merged && merged_bins.size() > 1) {
      merged = false;
      std::vector<Bin> temp_bins;
      temp_bins.reserve(merged_bins.size());
      
      Bin temp_bin = merged_bins[0];
      for (size_t i = 1; i < merged_bins.size(); ++i) {
        if (temp_bin.count < cutoff_count) {
          // Merge with next bin
          temp_bin.upper_bound = merged_bins[i].upper_bound;
          temp_bin.count += merged_bins[i].count;
          temp_bin.count_pos += merged_bins[i].count_pos;
          temp_bin.count_neg += merged_bins[i].count_neg;
          merged = true;
        } else {
          temp_bins.emplace_back(temp_bin);
          temp_bin = merged_bins[i];
        }
      }
      temp_bins.emplace_back(temp_bin);
      merged_bins = temp_bins;
    }
    
    bins = merged_bins;
  }
  
  void calculate_woe_and_iv() {
    int total_pos = 0;
    int total_neg = 0;
    
    for (const auto& bin : bins) {
      total_pos += bin.count_pos;
      total_neg += bin.count_neg;
    }
    
    // Handle cases where total_pos or total_neg is zero
    if (total_pos == 0 || total_neg == 0) {
      // Set WoE and IV to zero for all bins
      for (auto& bin : bins) {
        bin.woe = 0.0;
        bin.iv = 0.0;
      }
      woe_values.assign(bins.size(), 0.0);
      iv_values.assign(bins.size(), 0.0);
      return;
    }
    
    double total_iv = 0.0;
    woe_values.clear();
    iv_values.clear();
    woe_values.reserve(bins.size());
    iv_values.reserve(bins.size());
    
    for (auto& bin : bins) {
      double pos_rate = static_cast<double>(bin.count_pos) / total_pos;
      double neg_rate = static_cast<double>(bin.count_neg) / total_neg;
      
      // Avoid division by zero and log(0)
      if (pos_rate > 0 && neg_rate > 0) {
        bin.woe = std::log(pos_rate / neg_rate);
        bin.iv = (pos_rate - neg_rate) * bin.woe;
      } else {
        // Assign WoE as zero if either pos_rate or neg_rate is zero
        bin.woe = 0.0;
        bin.iv = 0.0;
      }
      
      total_iv += bin.iv;
      woe_values.push_back(bin.woe);
      iv_values.push_back(bin.iv);
    }
  }
  
  void optimize_bins() {
    // Recalculate total_pos and total_neg
    int total_pos = 0;
    int total_neg = 0;
    for (const auto& bin : bins) {
      total_pos += bin.count_pos;
      total_neg += bin.count_neg;
    }
    
    // If the number of bins is already within max_bins, no optimization needed
    while (static_cast<int>(bins.size()) > max_bins) {
      double min_iv_loss = std::numeric_limits<double>::max();
      int merge_index = -1;
      
      // Find the pair of adjacent bins with the smallest IV loss when merged
      for (size_t i = 0; i < bins.size() - 1; ++i) {
        // Calculate IV before merging
        double iv_before = bins[i].iv + bins[i + 1].iv;
        
        // Calculate counts after merging
        int merged_count = bins[i].count + bins[i + 1].count;
        int merged_pos = bins[i].count_pos + bins[i + 1].count_pos;
        int merged_neg = bins[i].count_neg + bins[i + 1].count_neg;
        
        // Calculate rates
        double pos_rate = static_cast<double>(merged_pos) / total_pos;
        double neg_rate = static_cast<double>(merged_neg) / total_neg;
        
        // Calculate IV after merging
        double iv_after = 0.0;
        if (pos_rate > 0 && neg_rate > 0) {
          double woe = std::log(pos_rate / neg_rate);
          iv_after = (pos_rate - neg_rate) * woe;
        }
        
        double iv_loss = iv_before - iv_after;
        
        if (iv_loss < min_iv_loss) {
          min_iv_loss = iv_loss;
          merge_index = static_cast<int>(i);
        }
      }
      
      if (merge_index == -1) {
        break; // No further merging possible
      }
      
      // Merge the identified pair of bins
      Bin merged_bin;
      merged_bin.lower_bound = bins[merge_index].lower_bound;
      merged_bin.upper_bound = bins[merge_index + 1].upper_bound;
      merged_bin.count = bins[merge_index].count + bins[merge_index + 1].count;
      merged_bin.count_pos = bins[merge_index].count_pos + bins[merge_index + 1].count_pos;
      merged_bin.count_neg = bins[merge_index].count_neg + bins[merge_index + 1].count_neg;
      
      // Calculate merged WoE and IV
      double pos_rate = static_cast<double>(merged_bin.count_pos) / total_pos;
      double neg_rate = static_cast<double>(merged_bin.count_neg) / total_neg;
      
      if (pos_rate > 0 && neg_rate > 0) {
        merged_bin.woe = std::log(pos_rate / neg_rate);
        merged_bin.iv = (pos_rate - neg_rate) * merged_bin.woe;
      } else {
        merged_bin.woe = 0.0;
        merged_bin.iv = 0.0;
      }
      
      // Replace the two bins with the merged bin
      bins.erase(bins.begin() + merge_index, bins.begin() + merge_index + 2);
      bins.insert(bins.begin() + merge_index, merged_bin);
    }
    
    // Recalculate WoE and IV after optimization
    woe_values.clear();
    iv_values.clear();
    woe_values.reserve(bins.size());
    iv_values.reserve(bins.size());
    
    double total_iv = 0.0;
    for (const auto& bin : bins) {
      woe_values.push_back(bin.woe);
      iv_values.push_back(bin.iv);
      total_iv += bin.iv;
    }
  }
  
public:
  OptimalBinningNumericalEFB(int min_bins_, int max_bins_, double bin_cutoff_, int max_n_prebins_)
    : min_bins(min_bins_), max_bins(max_bins_), bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_) {}
  
  void fit(const std::vector<double>& feature_, const std::vector<int>& target_) {
    feature = feature_;
    target = target_;
    
    validate_inputs();
    create_prebins();
    calculate_bin_statistics();
    merge_rare_bins();
    
    // Ensure minimum number of bins
    while (static_cast<int>(bins.size()) < min_bins && bins.size() > 1) {
      // Merge the two smallest bins
      double min_count = std::numeric_limits<double>::max();
      int merge_index = -1;
      for (size_t i = 0; i < bins.size() - 1; ++i) {
        if ((bins[i].count + bins[i + 1].count) < min_count) {
          min_count = bins[i].count + bins[i + 1].count;
          merge_index = static_cast<int>(i);
        }
      }
      if (merge_index != -1) {
        Bin merged_bin;
        merged_bin.lower_bound = bins[merge_index].lower_bound;
        merged_bin.upper_bound = bins[merge_index + 1].upper_bound;
        merged_bin.count = bins[merge_index].count + bins[merge_index + 1].count;
        merged_bin.count_pos = bins[merge_index].count_pos + bins[merge_index + 1].count_pos;
        merged_bin.count_neg = bins[merge_index].count_neg + bins[merge_index + 1].count_neg;
        
        bins.erase(bins.begin() + merge_index, bins.begin() + merge_index + 2);
        bins.insert(bins.begin() + merge_index, merged_bin);
      } else {
        break;
      }
    }
    
    optimize_bins();
    calculate_woe_and_iv();
  }
  
  Rcpp::List get_results() {
    std::vector<std::string> bin_labels;
    std::vector<int> counts;
    std::vector<int> counts_pos;
    std::vector<int> counts_neg;
    
    for (const auto& bin : bins) {
      std::string label = "(" + double_to_string(bin.lower_bound) + ";" + double_to_string(bin.upper_bound) + "]";
      bin_labels.push_back(label);
      counts.push_back(bin.count);
      counts_pos.push_back(bin.count_pos);
      counts_neg.push_back(bin.count_neg);
    }
    
    // Assign WoE values to each observation
    std::vector<double> woe_feature(feature.size(), 0.0);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < feature.size(); ++i) {
      double value = feature[i];
      int bin_index = 0;
      // Binary search to find the bin
      bin_index = std::upper_bound(bin_edges.begin(), bin_edges.end(), value) - bin_edges.begin() - 1;
      if (bin_index < 0) {
        bin_index = 0;
      } else if (bin_index >= static_cast<int>(bins.size())) {
        bin_index = bins.size() - 1;
      }
      woe_feature[i] = bins[bin_index].woe;
    }
    
    return Rcpp::List::create(
      Rcpp::Named("woefeature") = woe_feature,
      Rcpp::Named("woebin") = Rcpp::DataFrame::create(
        Rcpp::Named("bin") = bin_labels,
        Rcpp::Named("woe") = woe_values,
        Rcpp::Named("iv") = iv_values,
        Rcpp::Named("count") = counts,
        Rcpp::Named("count_pos") = counts_pos,
        Rcpp::Named("count_neg") = counts_neg
      )
    );
  }
  
};

//' @title Optimal Binning for Numerical Variables using Equal-Frequency Binning
//'
//' @description
//' This function implements an optimal binning algorithm for numerical variables using an Equal-Frequency Binning approach with subsequent optimization. It aims to find a good binning strategy that balances interpretability and predictive power.
//'
//' @param target An integer vector of binary target values (0 or 1).
//' @param feature A numeric vector of feature values to be binned.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum fraction of total observations in each bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins (default: 20).
//'
//' @return A list containing:
//' \item{woefeature}{A numeric vector of Weight of Evidence (WoE) values for each observation}
//' \item{woebin}{A data frame with binning information, including bin ranges, WoE, IV, and counts}
//'
//' @details
//' The optimal binning algorithm using Equal-Frequency Binning consists of several steps:
//'
//' 1. Initial binning: The feature is divided into \code{max_n_prebins} bins, each containing approximately the same number of observations.
//' 2. Merging rare bins: Bins with a fraction of observations less than \code{bin_cutoff} are merged with adjacent bins.
//' 3. Optimizing bins: If the number of bins exceeds \code{max_bins}, adjacent bins are merged iteratively to minimize the loss of Information Value (IV).
//' 4. WoE and IV calculation: The Weight of Evidence (WoE) and Information Value (IV) are calculated for each bin.
//'
//' The Weight of Evidence (WoE) for each bin is calculated as:
//'
//' \deqn{WoE = \ln\left(\frac{P(X|Y=1)}{P(X|Y=0)}\right)}
//'
//' where \eqn{P(X|Y=1)} is the probability of the feature being in a particular bin given a positive target, and \eqn{P(X|Y=0)} is the probability given a negative target.
//'
//' The Information Value (IV) for each bin is calculated as:
//'
//' \deqn{IV = (P(X|Y=1) - P(X|Y=0)) * WoE}
//'
//' The total IV is the sum of IVs for all bins:
//'
//' \deqn{Total IV = \sum_{i=1}^{n} IV_i}
//'
//' This approach provides a balance between simplicity and effectiveness, creating bins with equal frequency initially and then adjusting them based on the data distribution and target variable relationship. The optimization step ensures that the final binning maximizes the predictive power while respecting the specified constraints.
//'
//' @examples
//' \dontrun{
//' set.seed(123)
//' target <- sample(0:1, 1000, replace = TRUE)
//' feature <- rnorm(1000)
//' result <- optimal_binning_numerical_efb(target, feature)
//' print(result$woebin)
//' }
//'
//' @references
//' \itemize{
//'   \item Dougherty, J., Kohavi, R., & Sahami, M. (1995). Supervised and unsupervised discretization of continuous features. In Machine Learning Proceedings 1995 (pp. 194-202). Morgan Kaufmann.
//'   \item Liu, H., Hussain, F., Tan, C. L., & Dash, M. (2002). Discretization: An enabling technique. Data mining and knowledge discovery, 6(4), 393-423.
//' }
//'
//' @author Lopes, J. E.
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_efb(Rcpp::IntegerVector target, Rcpp::NumericVector feature,
                                        int min_bins = 3, int max_bins = 5, double bin_cutoff = 0.05,
                                        int max_n_prebins = 20) {
 // Convert R vectors to C++ vectors
 std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
 std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
 
 // Initialize the binning class
 OptimalBinningNumericalEFB binner(min_bins, max_bins, bin_cutoff, max_n_prebins);
 
 // Fit the binning
 binner.fit(feature_vec, target_vec);
 
 // Get and return the results
 return binner.get_results();
}




// #include <Rcpp.h>
// #include <algorithm>
// #include <vector>
// #include <cmath>
// #include <limits>
// 
// #ifdef _OPENMP
// #include <omp.h>
// #endif
// 
// class OptimalBinningNumericalEFB {
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
//     if (min_bins < 2) {
//       Rcpp::stop("min_bins must be at least 2");
//     }
//     if (max_bins < min_bins) {
//       Rcpp::stop("max_bins must be greater than or equal to min_bins");
//     }
//     if (bin_cutoff <= 0 || bin_cutoff >= 1) {
//       Rcpp::stop("bin_cutoff must be between 0 and 1");
//     }
//     if (max_n_prebins <= 0) {
//       Rcpp::stop("max_n_prebins must be positive");
//     }
//     if (feature.size() != target.size()) {
//       Rcpp::stop("feature and target must have the same length");
//     }
//     for (int t : target) {
//       if (t != 0 && t != 1) {
//         Rcpp::stop("target must contain only 0 and 1");
//       }
//     }
//   }
// 
//   std::string double_to_string(double value) {
//     if (std::isinf(value)) {
//       return value > 0 ? "+Inf" : "-Inf";
//     }
//     std::ostringstream ss;
//     ss << std::fixed << std::setprecision(6) << value;
//     return ss.str();
//   }
// 
//   void create_prebins() {
//     std::vector<std::pair<double, int>> feature_target;
//     for (size_t i = 0; i < feature.size(); ++i) {
//       feature_target.push_back({feature[i], target[i]});
//     }
//     std::sort(feature_target.begin(), feature_target.end());
// 
//     int n = feature_target.size();
//     int observations_per_bin = std::max(1, n / max_n_prebins);
// 
//     bin_edges.push_back(-std::numeric_limits<double>::infinity());
// 
//     for (int i = 1; i < max_n_prebins; ++i) {
//       int index = i * observations_per_bin;
//       if (index < n) {
//         bin_edges.push_back(feature_target[index].first);
//       }
//     }
// 
//     bin_edges.push_back(std::numeric_limits<double>::infinity());
//   }
// 
//   void calculate_bin_statistics() {
//     bins.clear();
// 
//     for (size_t i = 0; i < bin_edges.size() - 1; ++i) {
//       Bin bin;
//       bin.lower_bound = bin_edges[i];
//       bin.upper_bound = bin_edges[i + 1];
//       bin.count = 0;
//       bin.count_pos = 0;
//       bin.count_neg = 0;
// 
//       bins.push_back(bin);
//     }
// 
// #pragma omp parallel for
//     for (size_t i = 0; i < feature.size(); ++i) {
//       double value = feature[i];
//       int target_value = target[i];
// 
//       auto it = std::upper_bound(bin_edges.begin(), bin_edges.end(), value) - 1;
//       int bin_index = it - bin_edges.begin();
// 
// #pragma omp atomic
//       bins[bin_index].count++;
// 
//       if (target_value == 1) {
// #pragma omp atomic
//         bins[bin_index].count_pos++;
//       } else {
// #pragma omp atomic
//         bins[bin_index].count_neg++;
//       }
//     }
//   }
// 
//   void merge_rare_bins() {
//     int total_count = 0;
//     for (const auto& bin : bins) {
//       total_count += bin.count;
//     }
// 
//     double cutoff_count = bin_cutoff * total_count;
// 
//     std::vector<Bin> merged_bins;
//     Bin current_bin = bins[0];
// 
//     for (size_t i = 1; i < bins.size(); ++i) {
//       if (current_bin.count < cutoff_count) {
//         current_bin.upper_bound = bins[i].upper_bound;
//         current_bin.count += bins[i].count;
//         current_bin.count_pos += bins[i].count_pos;
//         current_bin.count_neg += bins[i].count_neg;
//       } else {
//         merged_bins.push_back(current_bin);
//         current_bin = bins[i];
//       }
//     }
// 
//     if (current_bin.count > 0) {
//       merged_bins.push_back(current_bin);
//     }
// 
//     bins = merged_bins;
//   }
// 
//   void calculate_woe_and_iv() {
//     int total_pos = 0;
//     int total_neg = 0;
// 
//     for (const auto& bin : bins) {
//       total_pos += bin.count_pos;
//       total_neg += bin.count_neg;
//     }
// 
//     double total_iv = 0.0;
// 
//     for (auto& bin : bins) {
//       double pos_rate = static_cast<double>(bin.count_pos) / total_pos;
//       double neg_rate = static_cast<double>(bin.count_neg) / total_neg;
// 
//       if (pos_rate > 0 && neg_rate > 0) {
//         bin.woe = std::log(pos_rate / neg_rate);
//         bin.iv = (pos_rate - neg_rate) * bin.woe;
//       } else {
//         bin.woe = 0.0;
//         bin.iv = 0.0;
//       }
// 
//       total_iv += bin.iv;
//     }
// 
//     woe_values.clear();
//     iv_values.clear();
// 
//     for (const auto& bin : bins) {
//       woe_values.push_back(bin.woe);
//       iv_values.push_back(bin.iv);
//     }
//   }
// 
//   void optimize_bins() {
//     while (bins.size() > max_bins) {
//       int merge_index = -1;
//       double min_iv_loss = std::numeric_limits<double>::max();
// 
//       for (size_t i = 0; i < bins.size() - 1; ++i) {
//         double iv_before = bins[i].iv + bins[i + 1].iv;
// 
//         Bin merged_bin;
//         merged_bin.lower_bound = bins[i].lower_bound;
//         merged_bin.upper_bound = bins[i + 1].upper_bound;
//         merged_bin.count = bins[i].count + bins[i + 1].count;
//         merged_bin.count_pos = bins[i].count_pos + bins[i + 1].count_pos;
//         merged_bin.count_neg = bins[i].count_neg + bins[i + 1].count_neg;
// 
//         int total_pos = 0;
//         int total_neg = 0;
//         for (const auto& bin : bins) {
//           total_pos += bin.count_pos;
//           total_neg += bin.count_neg;
//         }
// 
//         double pos_rate = static_cast<double>(merged_bin.count_pos) / total_pos;
//         double neg_rate = static_cast<double>(merged_bin.count_neg) / total_neg;
// 
//         if (pos_rate > 0 && neg_rate > 0) {
//           merged_bin.woe = std::log(pos_rate / neg_rate);
//           merged_bin.iv = (pos_rate - neg_rate) * merged_bin.woe;
//         } else {
//           merged_bin.woe = 0.0;
//           merged_bin.iv = 0.0;
//         }
// 
//         double iv_after = merged_bin.iv;
//         double iv_loss = iv_before - iv_after;
// 
//         if (iv_loss < min_iv_loss) {
//           min_iv_loss = iv_loss;
//           merge_index = i;
//         }
//       }
// 
//       if (merge_index != -1) {
//         Bin merged_bin;
//         merged_bin.lower_bound = bins[merge_index].lower_bound;
//         merged_bin.upper_bound = bins[merge_index + 1].upper_bound;
//         merged_bin.count = bins[merge_index].count + bins[merge_index + 1].count;
//         merged_bin.count_pos = bins[merge_index].count_pos + bins[merge_index + 1].count_pos;
//         merged_bin.count_neg = bins[merge_index].count_neg + bins[merge_index + 1].count_neg;
// 
//         int total_pos = 0;
//         int total_neg = 0;
//         for (const auto& bin : bins) {
//           total_pos += bin.count_pos;
//           total_neg += bin.count_neg;
//         }
// 
//         double pos_rate = static_cast<double>(merged_bin.count_pos) / total_pos;
//         double neg_rate = static_cast<double>(merged_bin.count_neg) / total_neg;
// 
//         if (pos_rate > 0 && neg_rate > 0) {
//           merged_bin.woe = std::log(pos_rate / neg_rate);
//           merged_bin.iv = (pos_rate - neg_rate) * merged_bin.woe;
//         } else {
//           merged_bin.woe = 0.0;
//           merged_bin.iv = 0.0;
//         }
// 
//         bins.erase(bins.begin() + merge_index, bins.begin() + merge_index + 2);
//         bins.insert(bins.begin() + merge_index, merged_bin);
//       } else {
//         break;
//       }
//     }
//   }
// 
// public:
//   OptimalBinningNumericalEFB(int min_bins_, int max_bins_, double bin_cutoff_, int max_n_prebins_)
//     : min_bins(min_bins_), max_bins(max_bins_), bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_) {}
// 
//   void fit(const std::vector<double>& feature_, const std::vector<int>& target_) {
//     feature = feature_;
//     target = target_;
// 
//     validate_inputs();
//     create_prebins();
//     calculate_bin_statistics();
//     merge_rare_bins();
//     optimize_bins();
//     calculate_woe_and_iv();
//   }
// 
//   Rcpp::List get_results() {
//     std::vector<std::string> bin_labels;
//     std::vector<int> counts;
//     std::vector<int> counts_pos;
//     std::vector<int> counts_neg;
// 
//     for (size_t i = 0; i < bins.size(); ++i) {
//       const auto& bin = bins[i];
//       std::string label = "(" + double_to_string(bin.lower_bound) + ";" + double_to_string(bin.upper_bound) + "]";
//       bin_labels.push_back(label);
//       counts.push_back(bin.count);
//       counts_pos.push_back(bin.count_pos);
//       counts_neg.push_back(bin.count_neg);
//     }
// 
//     return Rcpp::List::create(
//       Rcpp::Named("woefeature") = woe_values,
//       Rcpp::Named("woebin") = Rcpp::DataFrame::create(
//         Rcpp::Named("bin") = bin_labels,
//         Rcpp::Named("woe") = woe_values,
//         Rcpp::Named("iv") = iv_values,
//         Rcpp::Named("count") = counts,
//         Rcpp::Named("count_pos") = counts_pos,
//         Rcpp::Named("count_neg") = counts_neg
//       )
//     );
//   }
// 
// };
// 
// //' @title Optimal Binning for Numerical Variables using Equal-Frequency Binning
// //'
// //' @description
// //' This function implements an optimal binning algorithm for numerical variables using an Equal-Frequency Binning approach with subsequent optimization. It aims to find a good binning strategy that balances interpretability and predictive power.
// //'
// //' @param target An integer vector of binary target values (0 or 1).
// //' @param feature A numeric vector of feature values to be binned.
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff Minimum fraction of total observations in each bin (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins (default: 20).
// //'
// //' @return A list containing:
// //' \item{woefeature}{A numeric vector of Weight of Evidence (WoE) values for each observation}
// //' \item{woebin}{A data frame with binning information, including bin ranges, WoE, IV, and counts}
// //'
// //' @details
// //' The optimal binning algorithm using Equal-Frequency Binning consists of several steps:
// //'
// //' 1. Initial binning: The feature is divided into \code{max_n_prebins} bins, each containing approximately the same number of observations.
// //' 2. Merging rare bins: Bins with a fraction of observations less than \code{bin_cutoff} are merged with adjacent bins.
// //' 3. Optimizing bins: If the number of bins exceeds \code{max_bins}, adjacent bins are merged iteratively to minimize the loss of Information Value (IV).
// //' 4. WoE and IV calculation: The Weight of Evidence (WoE) and Information Value (IV) are calculated for each bin.
// //'
// //' The Weight of Evidence (WoE) for each bin is calculated as:
// //'
// //' \deqn{WoE = \ln\left(\frac{P(X|Y=1)}{P(X|Y=0)}\right)}
// //'
// //' where \eqn{P(X|Y=1)} is the probability of the feature being in a particular bin given a positive target, and \eqn{P(X|Y=0)} is the probability given a negative target.
// //'
// //' The Information Value (IV) for each bin is calculated as:
// //'
// //' \deqn{IV = (P(X|Y=1) - P(X|Y=0)) * WoE}
// //'
// //' The total IV is the sum of IVs for all bins:
// //'
// //' \deqn{Total IV = \sum_{i=1}^{n} IV_i}
// //'
// //' This approach provides a balance between simplicity and effectiveness, creating bins with equal frequency initially and then adjusting them based on the data distribution and target variable relationship. The optimization step ensures that the final binning maximizes the predictive power while respecting the specified constraints.
// //'
// //' @examples
// //' \dontrun{
// //' set.seed(123)
// //' target <- sample(0:1, 1000, replace = TRUE)
// //' feature <- rnorm(1000)
// //' result <- optimal_binning_numerical_efb(target, feature)
// //' print(result$woebin)
// //' }
// //'
// //' @references
// //' \itemize{
// //'   \item Dougherty, J., Kohavi, R., & Sahami, M. (1995). Supervised and unsupervised discretization of continuous features. In Machine Learning Proceedings 1995 (pp. 194-202). Morgan Kaufmann.
// //'   \item Liu, H., Hussain, F., Tan, C. L., & Dash, M. (2002). Discretization: An enabling technique. Data mining and knowledge discovery, 6(4), 393-423.
// //' }
// //'
// //' @author Lopes, J. E.
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_numerical_efb(Rcpp::IntegerVector target, Rcpp::NumericVector feature,
//                                          int min_bins = 3, int max_bins = 5, double bin_cutoff = 0.05,
//                                          int max_n_prebins = 20) {
//   OptimalBinningNumericalEFB binner(min_bins, max_bins, bin_cutoff, max_n_prebins);
// 
//   std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
//   std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
// 
//   binner.fit(feature_vec, target_vec);
//   return binner.get_results();
// }
