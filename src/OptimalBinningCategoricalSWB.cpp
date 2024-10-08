#include <Rcpp.h>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

class OptimalBinningCategoricalSWB {
private:
  struct BinStats {
    std::vector<std::string> categories;
    int count;
    int count_pos;
    int count_neg;
    double woe;
    double iv;
    
    BinStats() : count(0), count_pos(0), count_neg(0), woe(0.0), iv(0.0) {}
  };
  
  std::vector<std::string> feature;
  std::vector<int> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  
  std::vector<BinStats> bins;
  int total_pos;
  int total_neg;
  
  double calculate_woe(int pos, int neg) const {
    if (pos == 0 || neg == 0) return 0.0;  // Avoid division by zero and log(0)
    double pos_rate = static_cast<double>(pos) / total_pos;
    double neg_rate = static_cast<double>(neg) / total_neg;
    return std::log(pos_rate / neg_rate);
  }
  
  double calculate_iv(const std::vector<BinStats>& current_bins) const {
    double iv = 0.0;
    for (const auto& bin : current_bins) {
      double pos_rate = static_cast<double>(bin.count_pos) / total_pos;
      double neg_rate = static_cast<double>(bin.count_neg) / total_neg;
      if (pos_rate > 0 && neg_rate > 0) {
        iv += (pos_rate - neg_rate) * std::log(pos_rate / neg_rate);
      }
    }
    return iv;
  }
  
  bool is_monotonic(const std::vector<BinStats>& current_bins) const {
    if (current_bins.size() <= 2) return true;  // Always monotonic with 2 or fewer bins
    bool increasing = true;
    bool decreasing = true;
    
    for (size_t i = 1; i < current_bins.size(); ++i) {
      if (current_bins[i].woe < current_bins[i-1].woe) {
        increasing = false;
      }
      if (current_bins[i].woe > current_bins[i-1].woe) {
        decreasing = false;
      }
      if (!increasing && !decreasing) break;
    }
    
    return increasing || decreasing;
  }
  
  void initialize_bins() {
    std::unordered_map<std::string, BinStats> initial_bins;
    total_pos = 0;
    total_neg = 0;
    
#pragma omp parallel for reduction(+:total_pos,total_neg)
    for (size_t i = 0; i < feature.size(); ++i) {
      const std::string& cat = feature[i];
      int target_val = target[i];
      
#pragma omp critical
{
  auto& bin = initial_bins[cat];
  if (std::find(bin.categories.begin(), bin.categories.end(), cat) == bin.categories.end()) {
    bin.categories.push_back(cat);
  }
  bin.count++;
  bin.count_pos += target_val;
  bin.count_neg += 1 - target_val;
}
total_pos += target_val;
total_neg += 1 - target_val;
    }
    
    bins.reserve(initial_bins.size());
    for (auto& pair : initial_bins) {
      pair.second.woe = calculate_woe(pair.second.count_pos, pair.second.count_neg);
      bins.push_back(std::move(pair.second));
    }
    
    std::sort(bins.begin(), bins.end(), [](const BinStats& a, const BinStats& b) {
      return a.woe < b.woe;
    });
    
    while (bins.size() > max_n_prebins) {
      merge_adjacent_bins();
    }
  }
  
  void merge_adjacent_bins() {
    if (bins.size() <= 2) return;  // Cannot merge if there are 2 or fewer bins
    
    double min_iv_loss = std::numeric_limits<double>::max();
    size_t merge_index = 0;
    
    for (size_t i = 0; i < bins.size() - 1; ++i) {
      BinStats merged_bin = bins[i];
      merged_bin.count += bins[i+1].count;
      merged_bin.count_pos += bins[i+1].count_pos;
      merged_bin.count_neg += bins[i+1].count_neg;
      merged_bin.categories.insert(merged_bin.categories.end(),
                                   bins[i+1].categories.begin(),
                                   bins[i+1].categories.end());
      merged_bin.woe = calculate_woe(merged_bin.count_pos, merged_bin.count_neg);
      
      std::vector<BinStats> temp_bins = bins;
      temp_bins[i] = merged_bin;
      temp_bins.erase(temp_bins.begin() + i + 1);
      
      double new_iv = calculate_iv(temp_bins);
      double iv_loss = calculate_iv(bins) - new_iv;
      
      if (iv_loss < min_iv_loss) {
        min_iv_loss = iv_loss;
        merge_index = i;
      }
    }
    
    bins[merge_index].count += bins[merge_index+1].count;
    bins[merge_index].count_pos += bins[merge_index+1].count_pos;
    bins[merge_index].count_neg += bins[merge_index+1].count_neg;
    bins[merge_index].categories.insert(bins[merge_index].categories.end(),
                                        bins[merge_index+1].categories.begin(),
                                        bins[merge_index+1].categories.end());
    bins[merge_index].woe = calculate_woe(bins[merge_index].count_pos, bins[merge_index].count_neg);
    bins.erase(bins.begin() + merge_index + 1);
  }
  
  void optimize_bins() {
    while (bins.size() > min_bins) {
      if (is_monotonic(bins) && bins.size() <= max_bins) {
        break;
      }
      merge_adjacent_bins();
    }
    
    double total_iv = calculate_iv(bins);
    
    for (auto& bin : bins) {
      double pos_rate = static_cast<double>(bin.count_pos) / total_pos;
      double neg_rate = static_cast<double>(bin.count_neg) / total_neg;
      bin.iv = (pos_rate - neg_rate) * bin.woe;
    }
  }
  
  static std::string join_categories(const std::vector<std::string>& categories, const std::string& delimiter) {
    std::ostringstream result;
    for (size_t i = 0; i < categories.size(); ++i) {
      if (i > 0) result << delimiter;
      result << categories[i];
    }
    return result.str();
  }
  
public:
  OptimalBinningCategoricalSWB(const std::vector<std::string>& feature,
                               const std::vector<int>& target,
                               int min_bins = 3,
                               int max_bins = 5,
                               double bin_cutoff = 0.05,
                               int max_n_prebins = 20)
    : feature(feature), target(target), min_bins(min_bins), max_bins(max_bins),
      bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins) {
    if (feature.size() != target.size()) {
      Rcpp::stop("Feature and target vectors must have the same length");
    }
    if (min_bins < 2 || max_bins < min_bins) {
      Rcpp::stop("Invalid bin constraints");
    }
    min_bins = std::max(2, std::min(min_bins, static_cast<int>(feature.size())));
    max_bins = std::max(min_bins, std::min(max_bins, static_cast<int>(feature.size())));
  }
  
  void fit() {
    initialize_bins();
    optimize_bins();
  }
  
  Rcpp::List get_results() const {
    std::vector<std::string> bin_categories;
    std::vector<double> woes;
    std::vector<double> ivs;
    std::vector<int> counts;
    std::vector<int> counts_pos;
    std::vector<int> counts_neg;
    std::vector<double> woe_feature(feature.size());
    
    std::unordered_map<std::string, double> category_to_woe;
    for (const auto& bin : bins) {
      std::string bin_name = join_categories(bin.categories, "%;%");
      bin_categories.push_back(bin_name);
      woes.push_back(bin.woe);
      ivs.push_back(bin.iv);
      counts.push_back(bin.count);
      counts_pos.push_back(bin.count_pos);
      counts_neg.push_back(bin.count_neg);
      
      for (const auto& category : bin.categories) {
        category_to_woe[category] = bin.woe;
      }
    }
    
#pragma omp parallel for
    for (size_t i = 0; i < feature.size(); ++i) {
      woe_feature[i] = category_to_woe[feature[i]];
    }
    
    return Rcpp::List::create(
      Rcpp::Named("woefeature") = woe_feature,
      Rcpp::Named("woebin") = Rcpp::DataFrame::create(
        Rcpp::Named("bin") = bin_categories,
        Rcpp::Named("woe") = woes,
        Rcpp::Named("iv") = ivs,
        Rcpp::Named("count") = counts,
        Rcpp::Named("count_pos") = counts_pos,
        Rcpp::Named("count_neg") = counts_neg
      )
    );
  }
};

//' @title Optimal Binning for Categorical Variables using Sliding Window Binning (SWB)
//'
//' @description
//' This function performs optimal binning for categorical variables using a Sliding Window Binning (SWB) approach,
//' which combines Weight of Evidence (WOE) and Information Value (IV) methods with monotonicity constraints.
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
//' \enumerate{
//'   \item Initialize bins for each unique category
//'   \item Sort bins by their WOE values
//'   \item Merge adjacent bins iteratively, minimizing information loss
//'   \item Optimize the number of bins while maintaining monotonicity
//'   \item Calculate final WOE and IV values for each bin
//' }
//'
//' The Weight of Evidence (WOE) is calculated as:
//' \deqn{WOE = \ln\left(\frac{\text{Proportion of Events}}{\text{Proportion of Non-Events}}\right)}
//'
//' The Information Value (IV) for each bin is calculated as:
//' \deqn{IV = (\text{Proportion of Events} - \text{Proportion of Non-Events}) \times WOE}
//'
//' @references
//' \itemize{
//'   \item Saleem, S. M., & Jain, A. K. (2017). A comprehensive review of supervised binning techniques for credit scoring. Journal of Risk Model Validation, 11(3), 1-35.
//'   \item Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). Credit scoring and its applications. SIAM.
//' }
//'
//' @author Lopes, J. E.
//'
//' @examples
//' \dontrun{
//' # Create sample data
//' set.seed(123)
//' target <- sample(0:1, 1000, replace = TRUE)
//' feature <- sample(LETTERS[1:5], 1000, replace = TRUE)
//'
//' # Run optimal binning
//' result <- optimal_binning_categorical_swb(target, feature)
//'
//' # View results
//' print(result$woebin)
//' hist(result$woefeature)
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_swb(Rcpp::IntegerVector target,
                                          Rcpp::StringVector feature,
                                          int min_bins = 3,
                                          int max_bins = 5,
                                          double bin_cutoff = 0.05,
                                          int max_n_prebins = 20) {
 std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
 std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
 
 OptimalBinningCategoricalSWB binner(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins);
 binner.fit();
 return binner.get_results();
}

// #include <Rcpp.h>
// #include <unordered_map>
// #include <vector>
// #include <algorithm>
// #include <cmath>
// #include <limits>
// 
// #ifdef _OPENMP
// #include <omp.h>
// #endif
// 
// #include <sstream>
// 
// class OptimalBinningCategoricalSWB {
// private:
//   std::vector<std::string> feature;
//   std::vector<int> target;
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
// 
//   struct BinStats {
//     std::vector<std::string> categories;
//     int count;
//     int count_pos;
//     int count_neg;
//     double woe;
//     double iv;
//   };
// 
//   std::vector<BinStats> bins;
//   int total_pos;
//   int total_neg;
// 
//   double calculate_woe(int pos, int neg) {
//     double pos_rate = static_cast<double>(pos) / total_pos;
//     double neg_rate = static_cast<double>(neg) / total_neg;
//     if (pos_rate == 0 || neg_rate == 0) {
//       return 0.0;  // Avoid log(0)
//     }
//     return std::log(pos_rate / neg_rate);
//   }
// 
//   double calculate_iv(const std::vector<BinStats>& current_bins) {
//     double iv = 0.0;
//     for (const auto& bin : current_bins) {
//       double pos_rate = static_cast<double>(bin.count_pos) / total_pos;
//       double neg_rate = static_cast<double>(bin.count_neg) / total_neg;
//       if (pos_rate > 0 && neg_rate > 0) {
//         iv += (pos_rate - neg_rate) * std::log(pos_rate / neg_rate);
//       }
//     }
//     return iv;
//   }
// 
//   bool is_monotonic(const std::vector<BinStats>& current_bins) {
//     bool increasing = true;
//     bool decreasing = true;
// 
//     for (size_t i = 1; i < current_bins.size(); ++i) {
//       if (current_bins[i].woe < current_bins[i-1].woe) {
//         increasing = false;
//       }
//       if (current_bins[i].woe > current_bins[i-1].woe) {
//         decreasing = false;
//       }
//     }
// 
//     return increasing || decreasing;
//   }
// 
//   void initialize_bins() {
//     std::unordered_map<std::string, BinStats> initial_bins;
//     total_pos = 0;
//     total_neg = 0;
// 
// #pragma omp parallel for reduction(+:total_pos,total_neg)
//     for (size_t i = 0; i < feature.size(); ++i) {
//       const std::string& cat = feature[i];
//       int target_val = target[i];
// 
// #pragma omp critical
// {
//   auto& bin = initial_bins[cat];
//   if (std::find(bin.categories.begin(), bin.categories.end(), cat) == bin.categories.end()) {
//     bin.categories.push_back(cat);
//   }
//   bin.count++;
//   bin.count_pos += target_val;
//   bin.count_neg += 1 - target_val;
// }
// total_pos += target_val;
// total_neg += 1 - target_val;
//     }
// 
//     bins.reserve(initial_bins.size());
//     for (auto& pair : initial_bins) {
//       pair.second.woe = calculate_woe(pair.second.count_pos, pair.second.count_neg);
//       bins.push_back(std::move(pair.second));
//     }
// 
//     std::sort(bins.begin(), bins.end(), [](const BinStats& a, const BinStats& b) {
//       return a.woe < b.woe;
//     });
// 
//     while (bins.size() > max_n_prebins) {
//       merge_adjacent_bins();
//     }
//   }
// 
//   void merge_adjacent_bins() {
//     double min_iv_loss = std::numeric_limits<double>::max();
//     size_t merge_index = 0;
// 
//     for (size_t i = 0; i < bins.size() - 1; ++i) {
//       BinStats merged_bin = bins[i];
//       merged_bin.count += bins[i+1].count;
//       merged_bin.count_pos += bins[i+1].count_pos;
//       merged_bin.count_neg += bins[i+1].count_neg;
//       merged_bin.categories.insert(merged_bin.categories.end(),
//                                    bins[i+1].categories.begin(),
//                                    bins[i+1].categories.end());
//       merged_bin.woe = calculate_woe(merged_bin.count_pos, merged_bin.count_neg);
// 
//       std::vector<BinStats> temp_bins = bins;
//       temp_bins[i] = merged_bin;
//       temp_bins.erase(temp_bins.begin() + i + 1);
// 
//       double new_iv = calculate_iv(temp_bins);
//       double iv_loss = calculate_iv(bins) - new_iv;
// 
//       if (iv_loss < min_iv_loss) {
//         min_iv_loss = iv_loss;
//         merge_index = i;
//       }
//     }
// 
//     bins[merge_index].count += bins[merge_index+1].count;
//     bins[merge_index].count_pos += bins[merge_index+1].count_pos;
//     bins[merge_index].count_neg += bins[merge_index+1].count_neg;
//     bins[merge_index].categories.insert(bins[merge_index].categories.end(),
//                                         bins[merge_index+1].categories.begin(),
//                                         bins[merge_index+1].categories.end());
//     bins[merge_index].woe = calculate_woe(bins[merge_index].count_pos, bins[merge_index].count_neg);
//     bins.erase(bins.begin() + merge_index + 1);
//   }
// 
//   void optimize_bins() {
//     while (bins.size() > min_bins) {
//       if (is_monotonic(bins) && bins.size() <= max_bins) {
//         break;
//       }
// 
//       merge_adjacent_bins();
//     }
// 
//     // Calculate final IV for each bin
//     double total_iv = calculate_iv(bins);
// 
//     // Assign IV to each bin
//     for (auto& bin : bins) {
//       double pos_rate = static_cast<double>(bin.count_pos) / total_pos;
//       double neg_rate = static_cast<double>(bin.count_neg) / total_neg;
//       bin.iv = (pos_rate - neg_rate) * bin.woe / total_iv;
//     }
//   }
// 
//   std::string join_categories(const std::vector<std::string>& categories, const std::string& delimiter) {
//     std::ostringstream result;
//     for (size_t i = 0; i < categories.size(); ++i) {
//       if (i > 0) result << delimiter;
//       result << categories[i];
//     }
//     return result.str();
//   }
// 
// public:
//   OptimalBinningCategoricalSWB(const std::vector<std::string>& feature,
//                                const std::vector<int>& target,
//                                int min_bins = 3,
//                                int max_bins = 5,
//                                double bin_cutoff = 0.05,
//                                int max_n_prebins = 20)
//     : feature(feature), target(target), min_bins(min_bins), max_bins(max_bins),
//       bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins) {
//     if (feature.size() != target.size()) {
//       Rcpp::stop("Feature and target vectors must have the same length");
//     }
//     if (min_bins < 2 || max_bins < min_bins) {
//       Rcpp::stop("Invalid bin constraints");
//     }
//   }
// 
//   void fit() {
//     initialize_bins();
//     optimize_bins();
//   }
// 
//   Rcpp::List get_results() {
//     std::vector<std::string> bin_categories;
//     std::vector<double> woes;
//     std::vector<double> ivs;
//     std::vector<int> counts;
//     std::vector<int> counts_pos;
//     std::vector<int> counts_neg;
//     std::vector<double> woe_feature(feature.size());
// 
//     std::unordered_map<std::string, double> category_to_woe;
//     for (const auto& bin : bins) {
//       std::string bin_name = join_categories(bin.categories, ",");
//       bin_categories.push_back(bin_name);
//       woes.push_back(bin.woe);
//       ivs.push_back(bin.iv);
//       counts.push_back(bin.count);
//       counts_pos.push_back(bin.count_pos);
//       counts_neg.push_back(bin.count_neg);
// 
//       for (const auto& category : bin.categories) {
//         category_to_woe[category] = bin.woe;
//       }
//     }
// 
// #pragma omp parallel for
//     for (size_t i = 0; i < feature.size(); ++i) {
//       woe_feature[i] = category_to_woe[feature[i]];
//     }
// 
//     return Rcpp::List::create(
//       Rcpp::Named("woefeature") = woe_feature,
//       Rcpp::Named("woebin") = Rcpp::DataFrame::create(
//         Rcpp::Named("bin") = bin_categories,
//         Rcpp::Named("woe") = woes,
//         Rcpp::Named("iv") = ivs,
//         Rcpp::Named("count") = counts,
//         Rcpp::Named("count_pos") = counts_pos,
//         Rcpp::Named("count_neg") = counts_neg
//       )
//     );
//   }
// };
// 
// 
// //' @title Optimal Binning for Categorical Variables using Sliding Window Binning (SWB)
// //'
// //' @description
// //' This function performs optimal binning for categorical variables using a Sliding Window Binning (SWB) approach,
// //' which combines Weight of Evidence (WOE) and Information Value (IV) methods with monotonicity constraints.
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
// //' \enumerate{
// //'   \item Initialize bins for each unique category
// //'   \item Sort bins by their WOE values
// //'   \item Merge adjacent bins iteratively, minimizing information loss
// //'   \item Optimize the number of bins while maintaining monotonicity
// //'   \item Calculate final WOE and IV values for each bin
// //' }
// //'
// //' The Weight of Evidence (WOE) is calculated as:
// //' \deqn{WOE = \ln\left(\frac{\text{Proportion of Events}}{\text{Proportion of Non-Events}}\right)}
// //'
// //' The Information Value (IV) for each bin is calculated as:
// //' \deqn{IV = (\text{Proportion of Events} - \text{Proportion of Non-Events}) \times WOE}
// //'
// //' @references
// //' \itemize{
// //'   \item Saleem, S. M., & Jain, A. K. (2017). A comprehensive review of supervised binning techniques for credit scoring. Journal of Risk Model Validation, 11(3), 1-35.
// //'   \item Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). Credit scoring and its applications. SIAM.
// //' }
// //'
// //' @author Lopes, J. E.
// //'
// //' @examples
// //' \dontrun{
// //' # Create sample data
// //' set.seed(123)
// //' target <- sample(0:1, 1000, replace = TRUE)
// //' feature <- sample(LETTERS[1:5], 1000, replace = TRUE)
// //'
// //' # Run optimal binning
// //' result <- optimal_binning_categorical_swb(target, feature)
// //'
// //' # View results
// //' print(result$woebin)
// //' hist(result$woefeature)
// //' }
// //'
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_categorical_swb(Rcpp::IntegerVector target,
//                                            Rcpp::StringVector feature,
//                                            int min_bins = 3,
//                                            int max_bins = 5,
//                                            double bin_cutoff = 0.05,
//                                            int max_n_prebins = 20) {
//   std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
//   std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
// 
//   OptimalBinningCategoricalSWB binner(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins);
//   binner.fit();
//   return binner.get_results();
// }
