#include <Rcpp.h>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <cmath>
#include <limits>
#include <numeric>

using namespace Rcpp;

class OptimalBinningCategoricalGMB {
private:
  std::vector<std::string> feature;
  std::vector<int> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  
  struct BinInfo {
    std::vector<std::string> categories;
    int count = 0;
    int count_pos = 0;
    int count_neg = 0;
    double woe = 0.0;
    double iv = 0.0;
  };
  
  std::vector<BinInfo> bins;
  
  // Improved WOE calculation with better handling of edge cases
  double calculateWOE(int pos, int neg, int total_pos, int total_neg) {
    const double epsilon = 1e-10; // Smaller epsilon for better precision
    double pos_rate = (pos + epsilon) / (total_pos + epsilon);
    double neg_rate = (neg + epsilon) / (total_neg + epsilon);
    return std::log(pos_rate / neg_rate);
  }
  
  // Improved IV calculation
  double calculateIV(const std::vector<BinInfo>& bins, int total_pos, int total_neg) {
    double iv = 0.0;
    for (const auto& bin : bins) {
      double pos_rate = static_cast<double>(bin.count_pos) / total_pos;
      double neg_rate = static_cast<double>(bin.count_neg) / total_neg;
      iv += (pos_rate - neg_rate) * bin.woe;
    }
    return iv;
  }
  
  void initializeBins() {
    std::unordered_map<std::string, BinInfo> category_map;
    
    // Count occurrences and initialize bins
    for (size_t i = 0; i < feature.size(); ++i) {
      auto& bin = category_map[feature[i]];
      if (bin.categories.empty()) {
        bin.categories.push_back(feature[i]);
      }
      bin.count++;
      bin.count_pos += target[i];
      bin.count_neg += 1 - target[i];
    }
    
    // Convert map to vector and sort by positive rate
    bins.clear();
    bins.reserve(category_map.size());
    for (auto& pair : category_map) {
      bins.push_back(std::move(pair.second));
    }
    
    std::sort(bins.begin(), bins.end(), [](const BinInfo& a, const BinInfo& b) {
      return (static_cast<double>(a.count_pos) / a.count) < (static_cast<double>(b.count_pos) / b.count);
    });
    
    // Merge rare categories
    int total_count = std::accumulate(bins.begin(), bins.end(), 0,
                                      [](int sum, const BinInfo& bin) { return sum + bin.count; });
    
    std::vector<BinInfo> merged_bins;
    BinInfo current_bin;
    for (const auto& bin : bins) {
      if (static_cast<double>(bin.count) / total_count < bin_cutoff) {
        current_bin.categories.insert(current_bin.categories.end(), bin.categories.begin(), bin.categories.end());
        current_bin.count += bin.count;
        current_bin.count_pos += bin.count_pos;
        current_bin.count_neg += bin.count_neg;
      } else {
        if (!current_bin.categories.empty()) {
          merged_bins.push_back(std::move(current_bin));
          current_bin = BinInfo();
        }
        merged_bins.push_back(bin);
      }
    }
    
    if (!current_bin.categories.empty()) {
      merged_bins.push_back(std::move(current_bin));
    }
    
    bins = std::move(merged_bins);
    
    // Limit to max_n_prebins
    if (static_cast<int>(bins.size()) > max_n_prebins) {
      bins.resize(max_n_prebins);
    }
  }
  
  void greedyMerge() {
    int total_pos = 0, total_neg = 0;
    for (const auto& bin : bins) {
      total_pos += bin.count_pos;
      total_neg += bin.count_neg;
    }
    
    while (static_cast<int>(bins.size()) > min_bins) {
      double best_merge_score = -std::numeric_limits<double>::infinity();
      size_t best_merge_index = 0;
      
      for (size_t i = 0; i < bins.size() - 1; ++i) {
        BinInfo merged_bin;
        merged_bin.categories.insert(merged_bin.categories.end(), bins[i].categories.begin(), bins[i].categories.end());
        merged_bin.categories.insert(merged_bin.categories.end(), bins[i + 1].categories.begin(), bins[i + 1].categories.end());
        merged_bin.count = bins[i].count + bins[i + 1].count;
        merged_bin.count_pos = bins[i].count_pos + bins[i + 1].count_pos;
        merged_bin.count_neg = bins[i].count_neg + bins[i + 1].count_neg;
        
        std::vector<BinInfo> temp_bins = bins;
        temp_bins[i] = merged_bin;
        temp_bins.erase(temp_bins.begin() + i + 1);
        
        for (auto& bin : temp_bins) {
          bin.woe = calculateWOE(bin.count_pos, bin.count_neg, total_pos, total_neg);
        }
        double merge_score = calculateIV(temp_bins, total_pos, total_neg);
        
        if (merge_score > best_merge_score) {
          best_merge_score = merge_score;
          best_merge_index = i;
        }
      }
      
      // Perform the best merge
      bins[best_merge_index].categories.insert(bins[best_merge_index].categories.end(),
                                               bins[best_merge_index + 1].categories.begin(),
                                               bins[best_merge_index + 1].categories.end());
      bins[best_merge_index].count += bins[best_merge_index + 1].count;
      bins[best_merge_index].count_pos += bins[best_merge_index + 1].count_pos;
      bins[best_merge_index].count_neg += bins[best_merge_index + 1].count_neg;
      bins.erase(bins.begin() + best_merge_index + 1);
      
      if (static_cast<int>(bins.size()) <= max_bins) {
        break;
      }
    }
    
    // Calculate WOE and IV for final bins
    for (auto& bin : bins) {
      bin.woe = calculateWOE(bin.count_pos, bin.count_neg, total_pos, total_neg);
      double pos_rate = static_cast<double>(bin.count_pos) / total_pos;
      double neg_rate = static_cast<double>(bin.count_neg) / total_neg;
      bin.iv = (pos_rate - neg_rate) * bin.woe;
    }
  }
  
  // New function to check monotonicity
  bool checkMonotonicity() const {
    for (size_t i = 1; i < bins.size(); ++i) {
      if (bins[i].woe < bins[i-1].woe) {
        return false;
      }
    }
    return true;
  }
  
  // New function to ensure WoE consistency
  bool checkWoEConsistency(const std::vector<double>& woefeature) const {
    std::unordered_map<std::string, double> category_woe;
    for (const auto& bin : bins) {
      for (const auto& category : bin.categories) {
        category_woe[category] = bin.woe;
      }
    }
    
    for (size_t i = 0; i < feature.size(); ++i) {
      if (std::abs(woefeature[i] - category_woe[feature[i]]) > 1e-6) {
        return false;
      }
    }
    return true;
  }
  
public:
  OptimalBinningCategoricalGMB(const std::vector<std::string>& feature,
                               const std::vector<int>& target,
                               int min_bins = 3,
                               int max_bins = 5,
                               double bin_cutoff = 0.05,
                               int max_n_prebins = 20)
    : feature(feature), target(target), min_bins(min_bins), max_bins(max_bins),
      bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins) {
    
    // Input validation
    if (feature.size() != target.size()) {
      Rcpp::stop("Feature and target vectors must have the same length");
    }
    if (min_bins < 2) {
      Rcpp::stop("min_bins must be at least 2");
    }
    if (max_bins < min_bins) {
      Rcpp::stop("max_bins must be greater than or equal to min_bins");
    }
    if (bin_cutoff <= 0 || bin_cutoff >= 1) {
      Rcpp::stop("bin_cutoff must be between 0 and 1");
    }
    if (max_n_prebins < min_bins) {
      Rcpp::stop("max_n_prebins must be greater than or equal to min_bins");
    }
    
    // Check if target is binary
    for (int t : target) {
      if (t != 0 && t != 1) {
        Rcpp::stop("Target must be binary (0 or 1)");
      }
    }
  }
  
  Rcpp::List fit() {
    initializeBins();
    greedyMerge();
    
    // Prepare output
    std::vector<std::string> bin_names;
    std::vector<double> woe_values;
    std::vector<double> iv_values;
    std::vector<int> count_values;
    std::vector<int> count_pos_values;
    std::vector<int> count_neg_values;
    
    for (const auto& bin : bins) {
      std::string bin_name = bin.categories[0];
      for (size_t i = 1; i < bin.categories.size(); ++i) {
        bin_name += "%;%" + bin.categories[i];
      }
      
      bin_names.push_back(bin_name);
      woe_values.push_back(bin.woe);
      iv_values.push_back(bin.iv);
      count_values.push_back(bin.count);
      count_pos_values.push_back(bin.count_pos);
      count_neg_values.push_back(bin.count_neg);
    }
    
    // Create WOE feature
    std::vector<double> woefeature(feature.size());
    for (size_t i = 0; i < feature.size(); ++i) {
      for (const auto& bin : bins) {
        if (std::find(bin.categories.begin(), bin.categories.end(), feature[i]) != bin.categories.end()) {
          woefeature[i] = bin.woe;
          break;
        }
      }
    }
    
    // Check monotonicity
    bool is_monotonic = checkMonotonicity();
    
    // Check WoE consistency
    bool woe_consistent = checkWoEConsistency(woefeature);
    
    // Calculate total IV
    double total_iv = std::accumulate(iv_values.begin(), iv_values.end(), 0.0);
    
    return Rcpp::List::create(
      Rcpp::Named("woefeature") = woefeature,
      Rcpp::Named("woebin") = Rcpp::DataFrame::create(
        Rcpp::Named("bin") = bin_names,
        Rcpp::Named("woe") = woe_values,
        Rcpp::Named("iv") = iv_values,
        Rcpp::Named("count") = count_values,
        Rcpp::Named("count_pos") = count_pos_values,
        Rcpp::Named("count_neg") = count_neg_values
      ),
      Rcpp::Named("is_monotonic") = is_monotonic,
      Rcpp::Named("woe_consistent") = woe_consistent,
      Rcpp::Named("total_iv") = total_iv
    );
  }
};

//' @title Categorical Optimal Binning with Greedy Merge Binning
//'
//' @description
//' Implements optimal binning for categorical variables using a Greedy Merge approach,
//' calculating Weight of Evidence (WoE) and Information Value (IV).
//'
//' @param target Integer vector of binary target values (0 or 1).
//' @param feature Character vector of categorical feature values.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum frequency for a separate bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins before merging (default: 20).
//'
//' @return A list with two elements:
//' \itemize{
//'   \item woefeature: Numeric vector of WoE values for each input feature value.
//'   \item woebin: Data frame with binning results (bin names, WoE, IV, counts).
//' }
//'
//' @details
//' The algorithm uses a greedy merge approach to find an optimal binning solution.
//' It starts with each unique category as a separate bin and iteratively merges
//' bins to maximize the overall Information Value (IV) while respecting the
//' constraints on the number of bins.
//'
//' Weight of Evidence (WoE) for each bin is calculated as:
//'
//' \deqn{WoE = \ln\left(\frac{P(X|Y=1)}{P(X|Y=0)}\right)}
//'
//' Information Value (IV) for each bin is calculated as:
//'
//' \deqn{IV = (P(X|Y=1) - P(X|Y=0)) \times WoE}
//'
//' The algorithm includes the following key steps:
//' \enumerate{
//'   \item Initialize bins with each unique category.
//'   \item Merge rare categories based on bin_cutoff.
//'   \item Iteratively merge adjacent bins that result in the highest IV.
//'   \item Stop merging when the number of bins reaches min_bins or max_bins.
//'   \item Calculate final WoE and IV for each bin.
//' }
//'
//' The algorithm handles zero counts by using a small constant (epsilon) to avoid
//' undefined logarithms and division by zero.
//'
//' @examples
//' \dontrun{
//' # Sample data
//' target <- c(1, 0, 1, 1, 0, 1, 0, 0, 1, 1)
//' feature <- c("A", "B", "A", "C", "B", "D", "C", "A", "D", "B")
//'
//' # Run optimal binning
//' result <- optimal_binning_categorical_gmb(target, feature, min_bins = 2, max_bins = 4)
//'
//' # View results
//' print(result$woebin)
//' print(result$woefeature)
//' }
//'
//' @author Lopes, J. E.
//'
//' @references
//' \itemize{
//'   \item Beltrami, M., Mach, M., & Dall'Aglio, M. (2021). Monotonic Optimal Binning Algorithm for Credit Risk Modeling. Risks, 9(3), 58.
//'   \item Siddiqi, N. (2006). Credit risk scorecards: developing and implementing intelligent credit scoring (Vol. 3). John Wiley & Sons.
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_gmb(Rcpp::IntegerVector target,
                                           Rcpp::StringVector feature,
                                           int min_bins = 3,
                                           int max_bins = 5,
                                           double bin_cutoff = 0.05,
                                           int max_n_prebins = 20) {
  std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
  std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
  
  OptimalBinningCategoricalGMB binner(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins);
  return binner.fit();
}


// #include <Rcpp.h>
// #include <vector>
// #include <string>
// #include <algorithm>
// #include <unordered_map>
// #include <cmath>
// #include <limits>
// 
// class OptimalBinningCategoricalGMB {
// private:
//   std::vector<std::string> feature;
//   std::vector<int> target;
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
// 
//   struct BinInfo {
//     std::vector<std::string> categories;
//     int count = 0;
//     int count_pos = 0;
//     int count_neg = 0;
//     double woe = 0.0;
//     double iv = 0.0;
//   };
// 
//   std::vector<BinInfo> bins;
// 
//   double calculateWOE(int pos, int neg, int total_pos, int total_neg) {
//     const double epsilon = 0.5;
//     double adjusted_pos = (pos == 0) ? epsilon : static_cast<double>(pos);
//     double adjusted_neg = (neg == 0) ? epsilon : static_cast<double>(neg);
//     double pos_rate = adjusted_pos / total_pos;
//     double neg_rate = adjusted_neg / total_neg;
//     return std::log(pos_rate / neg_rate);
//   }
// 
//   double calculateIV(const std::vector<BinInfo>& bins, int total_pos, int total_neg) {
//     double iv = 0.0;
//     const double epsilon = 0.5;
//     for (const auto& bin : bins) {
//       double adjusted_pos = (bin.count_pos == 0) ? epsilon : static_cast<double>(bin.count_pos);
//       double adjusted_neg = (bin.count_neg == 0) ? epsilon : static_cast<double>(bin.count_neg);
//       double pos_rate = adjusted_pos / total_pos;
//       double neg_rate = adjusted_neg / total_neg;
//       double woe = std::log(pos_rate / neg_rate);
//       iv += (pos_rate - neg_rate) * woe;
//     }
//     return iv;
//   }
// 
//   void initializeBins() {
//     std::unordered_map<std::string, BinInfo> category_map;
// 
//     for (size_t i = 0; i < feature.size(); ++i) {
//       auto& bin = category_map[feature[i]];
//       if (bin.categories.empty()) {
//         bin.categories.push_back(feature[i]);
//       }
//       bin.count++;
//       if (target[i] == 1) {
//         bin.count_pos++;
//       } else {
//         bin.count_neg++;
//       }
//     }
// 
//     bins.clear();
//     bins.reserve(category_map.size());
//     for (auto& pair : category_map) {
//       bins.push_back(std::move(pair.second));
//     }
// 
//     // Sort bins by positive rate
//     std::sort(bins.begin(), bins.end(), [](const BinInfo& a, const BinInfo& b) {
//       return (static_cast<double>(a.count_pos) / a.count) < (static_cast<double>(b.count_pos) / b.count);
//     });
// 
//     // Merge rare categories
//     int total_count = 0;
//     for (const auto& bin : bins) {
//       total_count += bin.count;
//     }
// 
//     std::vector<BinInfo> merged_bins;
//     BinInfo current_bin;
//     for (const auto& bin : bins) {
//       if (static_cast<double>(bin.count) / total_count < bin_cutoff) {
//         current_bin.categories.insert(current_bin.categories.end(), bin.categories.begin(), bin.categories.end());
//         current_bin.count += bin.count;
//         current_bin.count_pos += bin.count_pos;
//         current_bin.count_neg += bin.count_neg;
//       } else {
//         if (!current_bin.categories.empty()) {
//           merged_bins.push_back(std::move(current_bin));
//           current_bin = BinInfo();
//         }
//         merged_bins.push_back(bin);
//       }
//     }
// 
//     if (!current_bin.categories.empty()) {
//       merged_bins.push_back(std::move(current_bin));
//     }
// 
//     bins = std::move(merged_bins);
// 
//     // Limit to max_n_prebins
//     if (static_cast<int>(bins.size()) > max_n_prebins) {
//       bins.resize(max_n_prebins);
//     }
//   }
// 
//   void greedyMerge() {
//     int total_pos = 0, total_neg = 0;
//     for (const auto& bin : bins) {
//       total_pos += bin.count_pos;
//       total_neg += bin.count_neg;
//     }
// 
//     while (static_cast<int>(bins.size()) > min_bins) {
//       double best_merge_score = -std::numeric_limits<double>::infinity();
//       size_t best_merge_index = 0;
// 
//       for (size_t i = 0; i < bins.size() - 1; ++i) {
//         BinInfo merged_bin;
//         merged_bin.categories = bins[i].categories;
//         merged_bin.categories.insert(merged_bin.categories.end(), bins[i + 1].categories.begin(), bins[i + 1].categories.end());
//         merged_bin.count = bins[i].count + bins[i + 1].count;
//         merged_bin.count_pos = bins[i].count_pos + bins[i + 1].count_pos;
//         merged_bin.count_neg = bins[i].count_neg + bins[i + 1].count_neg;
// 
//         std::vector<BinInfo> temp_bins = bins;
//         temp_bins[i] = merged_bin;
//         temp_bins.erase(temp_bins.begin() + i + 1);
// 
//         for (auto& bin : temp_bins) {
//           bin.woe = calculateWOE(bin.count_pos, bin.count_neg, total_pos, total_neg);
//         }
//         double merge_score = calculateIV(temp_bins, total_pos, total_neg);
// 
//         if (merge_score > best_merge_score) {
//           best_merge_score = merge_score;
//           best_merge_index = i;
//         }
//       }
// 
//       // Perform the best merge
//       bins[best_merge_index].categories.insert(bins[best_merge_index].categories.end(),
//                                                bins[best_merge_index + 1].categories.begin(),
//                                                bins[best_merge_index + 1].categories.end());
//       bins[best_merge_index].count = bins[best_merge_index].count + bins[best_merge_index + 1].count;
//       bins[best_merge_index].count_pos = bins[best_merge_index].count_pos + bins[best_merge_index + 1].count_pos;
//       bins[best_merge_index].count_neg = bins[best_merge_index].count_neg + bins[best_merge_index + 1].count_neg;
//       bins.erase(bins.begin() + best_merge_index + 1);
// 
//       if (static_cast<int>(bins.size()) <= max_bins) {
//         break;
//       }
//     }
// 
//     // Calculate WOE and IV for final bins
//     for (auto& bin : bins) {
//       bin.woe = calculateWOE(bin.count_pos, bin.count_neg, total_pos, total_neg);
//       double adjusted_pos = (bin.count_pos == 0) ? 0.5 : static_cast<double>(bin.count_pos);
//       double adjusted_neg = (bin.count_neg == 0) ? 0.5 : static_cast<double>(bin.count_neg);
//       double pos_rate = adjusted_pos / total_pos;
//       double neg_rate = adjusted_neg / total_neg;
//       bin.iv = (pos_rate - neg_rate) * bin.woe;
//     }
//   }
// 
// public:
//   OptimalBinningCategoricalGMB(const std::vector<std::string>& feature,
//                                const std::vector<int>& target,
//                                int min_bins = 3,
//                                int max_bins = 5,
//                                double bin_cutoff = 0.05,
//                                int max_n_prebins = 20)
//     : feature(feature), target(target), min_bins(min_bins), max_bins(max_bins),
//       bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins) {
// 
//     if (min_bins < 2) {
//       Rcpp::stop("min_bins must be at least 2");
//     }
//     if (max_bins < min_bins) {
//       Rcpp::stop("max_bins must be greater than or equal to min_bins");
//     }
//   }
// 
//   Rcpp::List fit() {
//     initializeBins();
//     greedyMerge();
// 
//     // Prepare output
//     std::vector<std::string> bin_names;
//     std::vector<double> woe_values;
//     std::vector<double> iv_values;
//     std::vector<int> count_values;
//     std::vector<int> count_pos_values;
//     std::vector<int> count_neg_values;
// 
//     for (const auto& bin : bins) {
//       std::string bin_name = bin.categories[0];
//       for (size_t i = 1; i < bin.categories.size(); ++i) {
//         bin_name += "+" + bin.categories[i];
//       }
// 
//       bin_names.push_back(bin_name);
//       woe_values.push_back(bin.woe);
//       iv_values.push_back(bin.iv);
//       count_values.push_back(bin.count);
//       count_pos_values.push_back(bin.count_pos);
//       count_neg_values.push_back(bin.count_neg);
//     }
// 
//     // Create WOE feature
//     std::vector<double> woefeature(feature.size());
//     for (size_t i = 0; i < feature.size(); ++i) {
//       for (const auto& bin : bins) {
//         if (std::find(bin.categories.begin(), bin.categories.end(), feature[i]) != bin.categories.end()) {
//           woefeature[i] = bin.woe;
//           break;
//         }
//       }
//     }
// 
//     return Rcpp::List::create(
//       Rcpp::Named("woefeature") = woefeature,
//       Rcpp::Named("woebin") = Rcpp::DataFrame::create(
//         Rcpp::Named("bin") = bin_names,
//         Rcpp::Named("woe") = woe_values,
//         Rcpp::Named("iv") = iv_values,
//         Rcpp::Named("count") = count_values,
//         Rcpp::Named("count_pos") = count_pos_values,
//         Rcpp::Named("count_neg") = count_neg_values
//       )
//     );
//   }
// };
// 
// 
// //' @title Categorical Optimal Binning with Greedy Merge Binning
// //'
// //' @description
// //' Implements optimal binning for categorical variables using a Greedy Merge approach,
// //' calculating Weight of Evidence (WoE) and Information Value (IV).
// //'
// //' @param target Integer vector of binary target values (0 or 1).
// //' @param feature Character vector of categorical feature values.
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff Minimum frequency for a separate bin (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins before merging (default: 20).
// //'
// //' @return A list with two elements:
// //' \itemize{
// //'   \item woefeature: Numeric vector of WoE values for each input feature value.
// //'   \item woebin: Data frame with binning results (bin names, WoE, IV, counts).
// //' }
// //'
// //' @details
// //' The algorithm uses a greedy merge approach to find an optimal binning solution.
// //' It starts with each unique category as a separate bin and iteratively merges
// //' bins to maximize the overall Information Value (IV) while respecting the
// //' constraints on the number of bins.
// //'
// //' Weight of Evidence (WoE) for each bin is calculated as:
// //'
// //' \deqn{WoE = \ln\left(\frac{P(X|Y=1)}{P(X|Y=0)}\right)}
// //'
// //' Information Value (IV) for each bin is calculated as:
// //'
// //' \deqn{IV = (P(X|Y=1) - P(X|Y=0)) \times WoE}
// //'
// //' The algorithm includes the following key steps:
// //' \enumerate{
// //'   \item Initialize bins with each unique category.
// //'   \item Merge rare categories based on bin_cutoff.
// //'   \item Iteratively merge adjacent bins that result in the highest IV.
// //'   \item Stop merging when the number of bins reaches min_bins or max_bins.
// //'   \item Calculate final WoE and IV for each bin.
// //' }
// //'
// //' The algorithm handles zero counts by using a small constant (epsilon) to avoid
// //' undefined logarithms and division by zero.
// //'
// //' @examples
// //' \dontrun{
// //' # Sample data
// //' target <- c(1, 0, 1, 1, 0, 1, 0, 0, 1, 1)
// //' feature <- c("A", "B", "A", "C", "B", "D", "C", "A", "D", "B")
// //'
// //' # Run optimal binning
// //' result <- optimal_binning_categorical_gmb(target, feature, min_bins = 2, max_bins = 4)
// //'
// //' # View results
// //' print(result$woebin)
// //' print(result$woefeature)
// //' }
// //'
// //' @author Lopes, J. E.
// //'
// //' @references
// //' \itemize{
// //'   \item Beltrami, M., Mach, M., & Dall'Aglio, M. (2021). Monotonic Optimal Binning Algorithm for Credit Risk Modeling. Risks, 9(3), 58.
// //'   \item Siddiqi, N. (2006). Credit risk scorecards: developing and implementing intelligent credit scoring (Vol. 3). John Wiley & Sons.
// //' }
// //'
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_categorical_gmb(Rcpp::IntegerVector target,
//                                          Rcpp::StringVector feature,
//                                          int min_bins = 3,
//                                          int max_bins = 5,
//                                          double bin_cutoff = 0.05,
//                                          int max_n_prebins = 20) {
// std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
// std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
// 
// OptimalBinningCategoricalGMB binner(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins);
// return binner.fit();
// }
