#include <Rcpp.h>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <numeric>

// For convenience
using namespace Rcpp;

// Define a structure to hold bin information
struct BinInfo {
  std::vector<std::string> categories;
  double woe;
  double iv;
  int count;
  int count_pos;
  int count_neg;
  
  // Constructor to initialize counts
  BinInfo() : woe(0.0), iv(0.0), count(0), count_pos(0), count_neg(0) {}
};

// OptimalBinningCategoricalUDT Class
class OptimalBinningCategoricalUDT {
private:
  // Parameters
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  
  // Data
  std::vector<std::string> feature;
  std::vector<int> target;
  std::unordered_map<std::string, BinInfo> bin_map;
  std::vector<double> woe_feature;
  std::vector<BinInfo> final_bins;
  
  // Helper Functions
  void validate_input_parameters(const size_t unique_categories) {
    min_bins = std::max(2, std::min(static_cast<int>(unique_categories), min_bins));
    max_bins = std::max(min_bins, std::min(static_cast<int>(unique_categories), max_bins));
    bin_cutoff = std::max(0.0, std::min(1.0, bin_cutoff));
    max_n_prebins = std::max(min_bins, max_n_prebins);
  }
  
  void validate_input_data() const {
    if (feature.size() != target.size()) {
      throw std::invalid_argument("Feature and target vectors must have the same length.");
    }
    std::unordered_set<int> unique_targets(target.begin(), target.end());
    if (unique_targets.size() != 2 || !unique_targets.count(0) || !unique_targets.count(1)) {
      throw std::invalid_argument("Target variable must be binary (0 and 1).");
    }
  }
  
  double calculate_woe(int count_pos, int count_neg, int total_pos, int total_neg) const {
    const double epsilon = 1e-10;
    double dist_pos = std::max(epsilon, static_cast<double>(count_pos) / static_cast<double>(total_pos));
    double dist_neg = std::max(epsilon, static_cast<double>(count_neg) / static_cast<double>(total_neg));
    return std::log(dist_pos / dist_neg);
  }
  
  double calculate_iv(int count_pos, int count_neg, int total_pos, int total_neg) const {
    const double epsilon = 1e-10;
    double dist_pos = std::max(epsilon, static_cast<double>(count_pos) / static_cast<double>(total_pos));
    double dist_neg = std::max(epsilon, static_cast<double>(count_neg) / static_cast<double>(total_neg));
    return (dist_pos - dist_neg) * std::log(dist_pos / dist_neg);
  }
  
  void compute_initial_bins(std::vector<BinInfo>& initial_bins, int& total_pos, int& total_neg) {
    std::unordered_map<std::string, BinInfo> temp_bins;
    total_pos = 0;
    total_neg = 0;
    for (size_t i = 0; i < feature.size(); ++i) {
      const std::string& cat = feature[i];
      int tgt = target[i];
      auto& bin = temp_bins[cat];
      if (bin.categories.empty()) {
        bin.categories.push_back(cat);
      }
      bin.count++;
      bin.count_pos += tgt;
      bin.count_neg += (1 - tgt);
      
      total_pos += tgt;
      total_neg += (1 - tgt);
    }
    
    initial_bins.clear();
    for (auto& pair : temp_bins) {
      initial_bins.push_back(std::move(pair.second));
    }
  }
  
  void calculate_woe_iv(std::vector<BinInfo>& bins, int total_pos, int total_neg) {
    for (auto& bin : bins) {
      bin.woe = calculate_woe(bin.count_pos, bin.count_neg, total_pos, total_neg);
      bin.iv = calculate_iv(bin.count_pos, bin.count_neg, total_pos, total_neg);
    }
  }
  
  bool is_monotonic(const std::vector<BinInfo>& bins) const {
    if (bins.size() <= 2) return true;
    bool increasing = bins[1].woe > bins[0].woe;
    for (size_t i = 2; i < bins.size(); ++i) {
      if ((increasing && bins[i].woe < bins[i-1].woe) || (!increasing && bins[i].woe > bins[i-1].woe)) {
        return false;
      }
    }
    return true;
  }
  
public:
  // Constructor
  OptimalBinningCategoricalUDT(int min_bins_ = 2, int max_bins_ = 5, double bin_cutoff_ = 0.05, int max_n_prebins_ = 20) :
  min_bins(min_bins_), max_bins(max_bins_), bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_) {}
  
  // Fit Function
  void fit(const std::vector<std::string>& feature_, const std::vector<int>& target_) {
    feature = feature_;
    target = target_;
    
    std::vector<BinInfo> initial_bins;
    int total_pos = 0, total_neg = 0;
    compute_initial_bins(initial_bins, total_pos, total_neg);
    size_t unique_categories = initial_bins.size();
    
    validate_input_parameters(unique_categories);
    validate_input_data();
    
    calculate_woe_iv(initial_bins, total_pos, total_neg);
    
    std::sort(initial_bins.begin(), initial_bins.end(), [](const BinInfo& a, const BinInfo& b) {
      return a.count < b.count;
    });
    
    while (initial_bins.size() > static_cast<size_t>(max_n_prebins)) {
      BinInfo& bin1 = initial_bins[0];
      BinInfo& bin2 = initial_bins[1];
      bin1.categories.insert(bin1.categories.end(), bin2.categories.begin(), bin2.categories.end());
      bin1.count += bin2.count;
      bin1.count_pos += bin2.count_pos;
      bin1.count_neg += bin2.count_neg;
      bin1.woe = calculate_woe(bin1.count_pos, bin1.count_neg, total_pos, total_neg);
      bin1.iv = calculate_iv(bin1.count_pos, bin1.count_neg, total_pos, total_neg);
      
      initial_bins.erase(initial_bins.begin() + 1);
      std::sort(initial_bins.begin(), initial_bins.end(), [](const BinInfo& a, const BinInfo& b) {
        return a.count < b.count;
      });
    }
    
    final_bins = std::move(initial_bins);
    
    std::sort(final_bins.begin(), final_bins.end(), [](const BinInfo& a, const BinInfo& b) {
      return a.woe < b.woe;
    });
    
    while (final_bins.size() > static_cast<size_t>(max_bins)) {
      double min_iv_diff = std::numeric_limits<double>::max();
      int merge_idx = -1;
      for (size_t i = 0; i < final_bins.size() - 1; ++i) {
        double iv_diff = std::abs(final_bins[i].iv - final_bins[i+1].iv);
        if (iv_diff < min_iv_diff) {
          min_iv_diff = iv_diff;
          merge_idx = static_cast<int>(i);
        }
      }
      
      if (merge_idx == -1) break;
      
      BinInfo& bin1 = final_bins[merge_idx];
      BinInfo& bin2 = final_bins[merge_idx + 1];
      bin1.categories.insert(bin1.categories.end(), bin2.categories.begin(), bin2.categories.end());
      bin1.count += bin2.count;
      bin1.count_pos += bin2.count_pos;
      bin1.count_neg += bin2.count_neg;
      bin1.woe = calculate_woe(bin1.count_pos, bin1.count_neg, total_pos, total_neg);
      bin1.iv = calculate_iv(bin1.count_pos, bin1.count_neg, total_pos, total_neg);
      
      final_bins.erase(final_bins.begin() + merge_idx + 1);
    }
    
    // Ensure monotonicity if possible
    if (!is_monotonic(final_bins) && final_bins.size() > static_cast<size_t>(min_bins)) {
      while (!is_monotonic(final_bins) && final_bins.size() > static_cast<size_t>(min_bins)) {
        double min_iv_diff = std::numeric_limits<double>::max();
        int merge_idx = -1;
        for (size_t i = 0; i < final_bins.size() - 1; ++i) {
          double iv_diff = std::abs(final_bins[i].iv - final_bins[i+1].iv);
          if (iv_diff < min_iv_diff) {
            min_iv_diff = iv_diff;
            merge_idx = static_cast<int>(i);
          }
        }
        
        if (merge_idx == -1) break;
        
        BinInfo& bin1 = final_bins[merge_idx];
        BinInfo& bin2 = final_bins[merge_idx + 1];
        bin1.categories.insert(bin1.categories.end(), bin2.categories.begin(), bin2.categories.end());
        bin1.count += bin2.count;
        bin1.count_pos += bin2.count_pos;
        bin1.count_neg += bin2.count_neg;
        bin1.woe = calculate_woe(bin1.count_pos, bin1.count_neg, total_pos, total_neg);
        bin1.iv = calculate_iv(bin1.count_pos, bin1.count_neg, total_pos, total_neg);
        
        final_bins.erase(final_bins.begin() + merge_idx + 1);
      }
    }
    
    bin_map.clear();
    for (const auto& bin : final_bins) {
      for (const auto& cat : bin.categories) {
        bin_map[cat] = bin;
      }
    }
    
    woe_feature.resize(feature.size());
    for (size_t i = 0; i < feature.size(); ++i) {
      auto it = bin_map.find(feature[i]);
      woe_feature[i] = (it != bin_map.end()) ? it->second.woe : 0.0;
    }
  }
  
  // Getters for output
  std::vector<double> get_woe_feature() const {
    return woe_feature;
  }
  
  DataFrame get_woe_bin() const {
    std::vector<std::string> bin_labels;
    std::vector<double> woe_vals;
    std::vector<double> iv_vals;
    std::vector<int> counts;
    std::vector<int> count_pos;
    std::vector<int> count_neg;
    
    for (const auto& bin : final_bins) {
      std::string label = std::accumulate(std::next(bin.categories.begin()), bin.categories.end(),
                                          bin.categories[0],
                                                        [](std::string a, std::string b) {
                                                          return std::move(a) + "%;%" + b;
                                                        });
      bin_labels.push_back(label);
      woe_vals.push_back(bin.woe);
      iv_vals.push_back(bin.iv);
      counts.push_back(bin.count);
      count_pos.push_back(bin.count_pos);
      count_neg.push_back(bin.count_neg);
    }
    
    return DataFrame::create(
      Named("bin") = bin_labels,
      Named("woe") = woe_vals,
      Named("iv") = iv_vals,
      Named("count") = counts,
      Named("count_pos") = count_pos,
      Named("count_neg") = count_neg
    );
  }
};

//' @title
//' Optimal Binning for Categorical Variables using Unsupervised Decision Tree (UDT)
//'
//' @description
//' This function performs optimal binning for categorical variables using an Unsupervised Decision Tree (UDT) approach,
//' which combines Weight of Evidence (WOE) and Information Value (IV) methods.
//'
//' @param target An integer vector of binary target values (0 or 1).
//' @param feature A character vector of categorical feature values.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum frequency for a category to be considered as a separate bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins before merging (default: 20).
//'
//' @return A list containing two elements:
//' \item{woefeature}{A numeric vector of WOE values for each input feature value}
//' \item{woebin}{A data frame containing bin information, including bin labels, WOE, IV, and counts}
//'
//' @details
//' The algorithm performs the following steps:
//' 1. Rare category handling: Categories with frequency below bin_cutoff are merged into a "Rare" bin.
//' 2. Pre-binning: If the number of bins exceeds max_n_prebins, the least frequent categories are merged into an "Other" bin.
//' 3. Calculate initial WOE and IV for each bin.
//' 4. Iterative merging: Bins are merged based on minimum combined IV until the number of bins reaches max_bins.
//'
//' The Weight of Evidence (WOE) is calculated as:
//'
//' WOE = ln((Distribution of Good) / (Distribution of Bad))
//'
//' The Information Value (IV) for each bin is calculated as:
//'
//' IV = (Distribution of Good - Distribution of Bad) * WOE
//'
//' @references
//' \itemize{
//'   \item Saleem, S. M., & Jain, A. K. (2017). A comprehensive review of supervised binning techniques for credit scoring. Journal of Risk Model Validation, 11(3), 1-35.
//'   \item Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). Credit scoring and its applications. SIAM.
//' }
//'
//' @examples
//' \dontrun{
//' # Create sample data
//' set.seed(123)
//' target <- sample(0:1, 1000, replace = TRUE)
//' feature <- sample(LETTERS[1:5], 1000, replace = TRUE)
//'
//' # Run optimal binning
//' result <- optimal_binning_categorical_udt(target, feature)
//'
//' # View results
//' print(result$woebin)
//' hist(result$woefeature)
//' }
//'
//' @author Lopes, J. E.
//'
//' @export
// [[Rcpp::export]]
List optimal_binning_categorical_udt(IntegerVector target,
                                     CharacterVector feature,
                                     int min_bins = 3, int max_bins = 5,
                                     double bin_cutoff = 0.05, int max_n_prebins = 20) {
  std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
  std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
  
  OptimalBinningCategoricalUDT binning(min_bins, max_bins, bin_cutoff, max_n_prebins);
  
  try {
    binning.fit(feature_vec, target_vec);
  } catch (const std::exception& e) {
    Rcpp::stop(e.what());
  }
  
  std::vector<double> woe_feature = binning.get_woe_feature();
  DataFrame woebin = binning.get_woe_bin();
  
  return List::create(
    Named("woefeature") = woe_feature,
    Named("woebin") = woebin
  );
}


// #include <Rcpp.h>
// #include <vector>
// #include <unordered_map>
// #include <algorithm>
// #include <cmath>
// #include <limits>
// #include <stdexcept>
// 
// // For convenience
// using namespace Rcpp;
// 
// // Define a structure to hold bin information
// struct BinInfo {
//   std::vector<std::string> categories;
//   double woe;
//   double iv;
//   int count;
//   int count_pos;
//   int count_neg;
//   
//   // Constructor to initialize counts
//   BinInfo() : woe(0.0), iv(0.0), count(0), count_pos(0), count_neg(0) {}
// };
// 
// // OptimalBinningCategoricalUDT Class
// class OptimalBinningCategoricalUDT {
// private:
//   // Parameters
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
//   
//   // Data
//   std::vector<std::string> feature;
//   std::vector<int> target;
//   std::unordered_map<std::string, BinInfo> bin_map;
//   std::vector<double> woe_feature;
//   std::vector<BinInfo> final_bins;
//   
//   // Helper Functions
//   void validate_input_parameters(const size_t unique_categories) {
//     if (min_bins < 2) {
//       min_bins = 2;
//     }
//     if (max_bins < min_bins) {
//       max_bins = min_bins;
//     }
//     if (bin_cutoff <= 0.0 || bin_cutoff >= 1.0) {
//       bin_cutoff = 0.05;
//     }
//     if (max_n_prebins < min_bins) {
//       max_n_prebins = min_bins;
//     }
//     if (static_cast<size_t>(min_bins) > unique_categories) {
//       min_bins = unique_categories;
//     }
//     if (static_cast<size_t>(max_bins) > unique_categories) {
//       max_bins = unique_categories;
//     }
//   }
//   
//   void validate_input_data() const {
//     std::unordered_set<int> unique_targets;
//     for (auto val : target) {
//       unique_targets.insert(val);
//       if (unique_targets.size() > 2) {
//         throw std::invalid_argument("Target variable must be binary.");
//       }
//     }
//     if (unique_targets.size() != 2) {
//       throw std::invalid_argument("Target variable must contain exactly two classes.");
//     }
//   }
//   
//   void merge_rare_categories(std::unordered_map<std::string, BinInfo>& initial_bins, size_t total) {
//     std::vector<std::string> to_merge;
//     for (auto& pair : initial_bins) {
//       double freq = static_cast<double>(pair.second.count) / static_cast<double>(total);
//       if (freq < bin_cutoff) {
//         to_merge.push_back(pair.first);
//       }
//     }
//     
//     if (!to_merge.empty()) {
//       BinInfo& rare_bin = initial_bins["Rare"];
//       for (auto& cat : to_merge) {
//         rare_bin.categories.push_back(cat);
//         rare_bin.count += initial_bins[cat].count;
//         rare_bin.count_pos += initial_bins[cat].count_pos;
//         rare_bin.count_neg += initial_bins[cat].count_neg;
//         initial_bins.erase(cat);
//       }
//     }
//   }
//   
//   double calculate_woe(int count_pos, int count_neg, int total_pos, int total_neg) const {
//     double dist_pos = (count_pos == 0) ? 1e-6 : static_cast<double>(count_pos) / static_cast<double>(total_pos);
//     double dist_neg = (count_neg == 0) ? 1e-6 : static_cast<double>(count_neg) / static_cast<double>(total_neg);
//     return std::log(dist_pos / dist_neg);
//   }
//   
//   double calculate_iv(int count_pos, int count_neg, int total_pos, int total_neg) const {
//     double dist_pos = (count_pos == 0) ? 1e-6 : static_cast<double>(count_pos) / static_cast<double>(total_pos);
//     double dist_neg = (count_neg == 0) ? 1e-6 : static_cast<double>(count_neg) / static_cast<double>(total_neg);
//     return (dist_pos - dist_neg) * std::log(dist_pos / dist_neg);
//   }
//   
//   void compute_initial_bins(std::unordered_map<std::string, BinInfo>& initial_bins, int& total_pos, int& total_neg) const {
//     total_pos = 0;
//     total_neg = 0;
//     for (size_t i = 0; i < feature.size(); ++i) {
//       const std::string& cat = feature[i];
//       int tgt = target[i];
//       if (initial_bins.find(cat) == initial_bins.end()) {
//         BinInfo bin;
//         bin.categories.push_back(cat);
//         bin.count = 1;
//         bin.count_pos = tgt;
//         bin.count_neg = 1 - tgt;
//         initial_bins[cat] = bin;
//       } else {
//         initial_bins[cat].count += 1;
//         initial_bins[cat].count_pos += tgt;
//         initial_bins[cat].count_neg += (1 - tgt);
//       }
//       
//       if (tgt == 1) {
//         total_pos += 1;
//       } else {
//         total_neg += 1;
//       }
//     }
//   }
//   
//   void calculate_woe_iv(std::unordered_map<std::string, BinInfo>& bins, int total_pos, int total_neg) const {
//     for (auto& pair : bins) {
//       BinInfo& bin = pair.second;
//       bin.woe = calculate_woe(bin.count_pos, bin.count_neg, total_pos, total_neg);
//       bin.iv = calculate_iv(bin.count_pos, bin.count_neg, total_pos, total_neg);
//     }
//   }
//   
// public:
//   // Constructor
//   OptimalBinningCategoricalUDT(int min_bins_ = 2, int max_bins_ = 5, double bin_cutoff_ = 0.05, int max_n_prebins_ = 20) :
//   min_bins(min_bins_), max_bins(max_bins_), bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_) {}
//   
//   // Fit Function
//   void fit(const std::vector<std::string>& feature_, const std::vector<int>& target_) {
//     feature = feature_;
//     target = target_;
//     
//     std::unordered_map<std::string, BinInfo> temp_bins;
//     int temp_total_pos = 0, temp_total_neg = 0;
//     compute_initial_bins(temp_bins, temp_total_pos, temp_total_neg);
//     size_t unique_categories = temp_bins.size();
//     
//     validate_input_parameters(unique_categories);
//     validate_input_data();
//     
//     std::unordered_map<std::string, BinInfo> initial_bins = temp_bins;
//     int total_pos = temp_total_pos;
//     int total_neg = temp_total_neg;
//     
//     merge_rare_categories(initial_bins, feature.size());
//     
//     while (initial_bins.size() > static_cast<size_t>(max_n_prebins)) {
//       auto it = std::min_element(initial_bins.begin(), initial_bins.end(),
//                                  [](const std::pair<std::string, BinInfo>& a, const std::pair<std::string, BinInfo>& b) {
//                                    return a.second.count < b.second.count;
//                                  });
//       if (it == initial_bins.end()) {
//         break;
//       }
//       BinInfo& other_bin = initial_bins["Other"];
//       other_bin.categories.insert(other_bin.categories.end(), it->second.categories.begin(), it->second.categories.end());
//       other_bin.count += it->second.count;
//       other_bin.count_pos += it->second.count_pos;
//       other_bin.count_neg += it->second.count_neg;
//       other_bin.woe = calculate_woe(other_bin.count_pos, other_bin.count_neg, total_pos, total_neg);
//       other_bin.iv = calculate_iv(other_bin.count_pos, other_bin.count_neg, total_pos, total_neg);
//       initial_bins.erase(it);
//     }
//     
//     calculate_woe_iv(initial_bins, total_pos, total_neg);
//     
//     final_bins.clear();
//     final_bins.reserve(initial_bins.size());
//     for (auto& pair : initial_bins) {
//       final_bins.push_back(pair.second);
//     }
//     
//     while (final_bins.size() > static_cast<size_t>(max_bins)) {
//       double min_iv = std::numeric_limits<double>::max();
//       int merge_idx1 = -1, merge_idx2 = -1;
//       for (size_t i = 0; i < final_bins.size(); ++i) {
//         for (size_t j = i + 1; j < final_bins.size(); ++j) {
//           double combined_iv = final_bins[i].iv + final_bins[j].iv;
//           if (combined_iv < min_iv) {
//             min_iv = combined_iv;
//             merge_idx1 = static_cast<int>(i);
//             merge_idx2 = static_cast<int>(j);
//           }
//         }
//       }
//       
//       if (merge_idx1 == -1 || merge_idx2 == -1) {
//         break;
//       }
//       
//       BinInfo merged_bin;
//       merged_bin.categories.insert(merged_bin.categories.end(),
//                                    final_bins[merge_idx1].categories.begin(),
//                                    final_bins[merge_idx1].categories.end());
//       merged_bin.categories.insert(merged_bin.categories.end(),
//                                    final_bins[merge_idx2].categories.begin(),
//                                    final_bins[merge_idx2].categories.end());
//       merged_bin.count = final_bins[merge_idx1].count + final_bins[merge_idx2].count;
//       merged_bin.count_pos = final_bins[merge_idx1].count_pos + final_bins[merge_idx2].count_pos;
//       merged_bin.count_neg = final_bins[merge_idx1].count_neg + final_bins[merge_idx2].count_neg;
//       merged_bin.woe = calculate_woe(merged_bin.count_pos, merged_bin.count_neg, total_pos, total_neg);
//       merged_bin.iv = calculate_iv(merged_bin.count_pos, merged_bin.count_neg, total_pos, total_neg);
//       
//       if (static_cast<size_t>(merge_idx1) > static_cast<size_t>(merge_idx2)) {
//         final_bins.erase(final_bins.begin() + merge_idx1);
//         final_bins.erase(final_bins.begin() + merge_idx2);
//       } else {
//         final_bins.erase(final_bins.begin() + merge_idx2);
//         final_bins.erase(final_bins.begin() + merge_idx1);
//       }
//       final_bins.push_back(merged_bin);
//     }
//     
//     bin_map.clear();
//     for (auto& bin : final_bins) {
//       for (auto& cat : bin.categories) {
//         bin_map[cat] = bin;
//       }
//     }
//     
//     woe_feature.resize(feature.size());
//     for (size_t i = 0; i < feature.size(); ++i) {
//       woe_feature[i] = bin_map[feature[i]].woe;
//     }
//   }
//   
//   // Getters for output
//   std::vector<double> get_woe_feature() const {
//     return woe_feature;
//   }
//   
//   DataFrame get_woe_bin() const {
//     std::vector<std::string> bin_labels;
//     std::vector<double> woe_vals;
//     std::vector<double> iv_vals;
//     std::vector<int> counts;
//     std::vector<int> count_pos;
//     std::vector<int> count_neg;
//     
//     for (const auto& bin : final_bins) {
//       std::string label = "";
//       for (size_t i = 0; i < bin.categories.size(); ++i) {
//         label += bin.categories[i];
//         if (i != bin.categories.size() - 1) {
//           label += "+";
//         }
//       }
//       bin_labels.push_back(label);
//       woe_vals.push_back(bin.woe);
//       iv_vals.push_back(bin.iv);
//       counts.push_back(bin.count);
//       count_pos.push_back(bin.count_pos);
//       count_neg.push_back(bin.count_neg);
//     }
//     
//     return DataFrame::create(
//       Named("bin") = bin_labels,
//       Named("woe") = woe_vals,
//       Named("iv") = iv_vals,
//       Named("count") = counts,
//       Named("count_pos") = count_pos,
//       Named("count_neg") = count_neg
//     );
//   }
// };
// 
// //' @title 
// //' Optimal Binning for Categorical Variables using Unsupervised Decision Tree (UDT)
// //'
// //' @description
// //' This function performs optimal binning for categorical variables using an Unsupervised Decision Tree (UDT) approach,
// //' which combines Weight of Evidence (WOE) and Information Value (IV) methods.
// //'
// //' @param target An integer vector of binary target values (0 or 1).
// //' @param feature A character vector of categorical feature values.
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff Minimum frequency for a category to be considered as a separate bin (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins before merging (default: 20).
// //'
// //' @return A list containing two elements:
// //' \item{woefeature}{A numeric vector of WOE values for each input feature value}
// //' \item{woebin}{A data frame containing bin information, including bin labels, WOE, IV, and counts}
// //'
// //' @details
// //' The algorithm performs the following steps:
// //' 1. Rare category handling: Categories with frequency below bin_cutoff are merged into a "Rare" bin.
// //' 2. Pre-binning: If the number of bins exceeds max_n_prebins, the least frequent categories are merged into an "Other" bin.
// //' 3. Calculate initial WOE and IV for each bin.
// //' 4. Iterative merging: Bins are merged based on minimum combined IV until the number of bins reaches max_bins.
// //'
// //' The Weight of Evidence (WOE) is calculated as:
// //'
// //' WOE = ln((Distribution of Good) / (Distribution of Bad))
// //'
// //' The Information Value (IV) for each bin is calculated as:
// //'
// //' IV = (Distribution of Good - Distribution of Bad) * WOE
// //'
// //' @references
// //' \itemize{
// //'   \item Saleem, S. M., & Jain, A. K. (2017). A comprehensive review of supervised binning techniques for credit scoring. Journal of Risk Model Validation, 11(3), 1-35.
// //'   \item Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). Credit scoring and its applications. SIAM.
// //' }
// //'
// //' @examples
// //' \dontrun{
// //' # Create sample data
// //' set.seed(123)
// //' target <- sample(0:1, 1000, replace = TRUE)
// //' feature <- sample(LETTERS[1:5], 1000, replace = TRUE)
// //'
// //' # Run optimal binning
// //' result <- optimal_binning_categorical_udt(target, feature)
// //'
// //' # View results
// //' print(result$woebin)
// //' hist(result$woefeature)
// //' }
// //'
// //' @author Lopes, J. E.
// //'
// //' @export
// // [[Rcpp::export]]
// List optimal_binning_categorical_udt(IntegerVector target,
//                                      CharacterVector feature,
//                                      int min_bins = 3, int max_bins = 5,
//                                      double bin_cutoff = 0.05, int max_n_prebins = 20) {
//   std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
//   std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
//   
//   OptimalBinningCategoricalUDT binning(min_bins, max_bins, bin_cutoff, max_n_prebins);
//   
//   try {
//     binning.fit(feature_vec, target_vec);
//   } catch (const std::exception& e) {
//     Rcpp::stop(e.what());
//   }
//   
//   std::vector<double> woe_feature = binning.get_woe_feature();
//   DataFrame woebin = binning.get_woe_bin();
//   
//   return List::create(
//     Named("woefeature") = woe_feature,
//     Named("woebin") = woebin
//   );
// }
