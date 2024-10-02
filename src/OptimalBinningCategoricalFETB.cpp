#include <Rcpp.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <sstream>

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::plugins(openmp)]]

class OptimalBinningCategoricalFETB {
private:
  const std::vector<std::string>& feature;
  const std::vector<int>& target;
  const size_t min_bins;
  size_t max_bins;
  const double bin_cutoff;
  const size_t max_n_prebins;
  
  std::unordered_map<std::string, std::string> original_to_merged_category;
  std::unordered_map<std::string, std::unordered_set<std::string>> merged_to_original_categories;
  
  struct BinInfo {
    std::unordered_set<std::string> categories;
    size_t count = 0;
    size_t count_pos = 0;
    size_t count_neg = 0;
    double woe = 0.0;
    double iv = 0.0;
  };
  
  std::vector<BinInfo> bins;
  size_t total_pos = 0;
  size_t total_neg = 0;
  
  static constexpr double EPSILON = 1e-10;
  std::vector<double> log_factorials;
  
  void calculateWoeIv(BinInfo& bin) const {
    if (bin.count_pos == 0 || bin.count_neg == 0 || total_pos == 0 || total_neg == 0) {
      bin.woe = 0.0;
      bin.iv = 0.0;
      return;
    }
    
    double dist_pos = static_cast<double>(bin.count_pos) / total_pos;
    double dist_neg = static_cast<double>(bin.count_neg) / total_neg;
    
    bin.woe = std::log((dist_pos + EPSILON) / (dist_neg + EPSILON));
    bin.iv = (dist_pos - dist_neg) * bin.woe;
  }
  
  double fisherExactTest(size_t a, size_t b, size_t c, size_t d) const {
    size_t n = a + b + c + d;
    size_t row1 = a + b;
    size_t row2 = c + d;
    size_t col1 = a + c;
    size_t col2 = b + d;
    
    double log_p = log_factorials[row1] + log_factorials[row2] + log_factorials[col1] + log_factorials[col2]
    - log_factorials[n] - log_factorials[a] - log_factorials[b] - log_factorials[c] - log_factorials[d];
    
    return std::exp(log_p);
  }
  
  void mergeRareCategories() {
    std::unordered_map<std::string, size_t> category_counts;
    size_t total_count = feature.size();
    
    for (const auto& cat : feature) {
      category_counts[cat]++;
    }
    
    std::vector<std::pair<std::string, size_t>> sorted_categories(category_counts.begin(), category_counts.end());
    std::sort(sorted_categories.begin(), sorted_categories.end(),
              [](const std::pair<std::string, size_t>& a, const std::pair<std::string, size_t>& b) {
                return a.second > b.second;
              });
    
    const std::string rare_category = "__RARE__";
    
    for (const auto& [cat, count] : sorted_categories) {
      double frequency = static_cast<double>(count) / total_count;
      if (frequency < bin_cutoff) {
        original_to_merged_category[cat] = rare_category;
      } else {
        original_to_merged_category[cat] = cat;
      }
    }
    
    for (const auto& [original_cat, merged_cat] : original_to_merged_category) {
      merged_to_original_categories[merged_cat].insert(original_cat);
    }
  }
  
  void initializeBins() {
    std::unordered_map<std::string, BinInfo> bin_map;
    
    for (size_t i = 0; i < feature.size(); ++i) {
      const std::string& original_cat = feature[i];
      auto it = original_to_merged_category.find(original_cat);
      if (it == original_to_merged_category.end()) {
        continue;
      }
      const std::string& merged_cat = it->second;
      
      BinInfo& bin = bin_map[merged_cat];
      bin.count++;
      if (target[i] == 1) {
        bin.count_pos++;
      } else {
        bin.count_neg++;
      }
    }
    
    for (auto& [merged_cat, bin_info] : bin_map) {
      bin_info.categories = merged_to_original_categories[merged_cat];
      calculateWoeIv(bin_info);
    }
    
    bins.reserve(bin_map.size());
    for (auto& [_, bin_info] : bin_map) {
      bins.emplace_back(std::move(bin_info));
    }
    
    std::sort(bins.begin(), bins.end(), [](const BinInfo& a, const BinInfo& b) {
      return a.woe < b.woe;
    });
  }
  
  void mergeBins() {
    while (bins.size() > max_bins) {
      if (bins.size() <= min_bins) {
        break;
      }
      
      double min_p_value = std::numeric_limits<double>::max();
      size_t merge_index = 0;
      
      for (size_t i = 0; i < bins.size() - 1; ++i) {
        size_t a = bins[i].count_pos;
        size_t b = bins[i].count_neg;
        size_t c = bins[i + 1].count_pos;
        size_t d = bins[i + 1].count_neg;
        
        double p_value = fisherExactTest(a, b, c, d);
        
        if (p_value < min_p_value) {
          min_p_value = p_value;
          merge_index = i;
        }
      }
      
      BinInfo& bin1 = bins[merge_index];
      BinInfo& bin2 = bins[merge_index + 1];
      
      bin1.count += bin2.count;
      bin1.count_pos += bin2.count_pos;
      bin1.count_neg += bin2.count_neg;
      bin1.categories.insert(bin2.categories.begin(), bin2.categories.end());
      
      calculateWoeIv(bin1);
      
      bins.erase(bins.begin() + merge_index + 1);
    }
    
    if (bins.size() < min_bins) {
      Rcpp::warning("Resulting number of bins (%d) is less than min_bins (%d)", static_cast<int>(bins.size()), static_cast<int>(min_bins));
    }
  }
  
  static std::string joinStrings(const std::unordered_set<std::string>& strings, const std::string& delimiter) {
    std::ostringstream oss;
    size_t i = 0;
    for (const auto& s : strings) {
      if (i > 0) oss << delimiter;
      oss << s;
      i++;
    }
    return oss.str();
  }
  
public:
  OptimalBinningCategoricalFETB(const std::vector<std::string>& feature,
                                const std::vector<int>& target,
                                size_t min_bins = 3,
                                size_t max_bins = 5,
                                double bin_cutoff = 0.05,
                                size_t max_n_prebins = 20)
    : feature(feature), target(target),
      min_bins(min_bins), max_bins(max_bins),
      bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins) {
    
    if (feature.empty() || target.empty()) {
      Rcpp::stop("Input vectors cannot be empty");
    }
    
    if (feature.size() != target.size()) {
      Rcpp::stop("Feature and target vectors must have the same length");
    }
    
    if (min_bins < 2) {
      Rcpp::stop("min_bins must be >= 2");
    }
    if (max_bins < min_bins) {
      Rcpp::stop("max_bins must be >= min_bins");
    }
    if (bin_cutoff <= 0.0 || bin_cutoff >= 1.0) {
      Rcpp::stop("bin_cutoff must be between 0 and 1");
    }
    if (max_n_prebins < min_bins) {
      Rcpp::stop("max_n_prebins must be >= min_bins");
    }
    
    for (int val : target) {
      if (val != 0 && val != 1) {
        Rcpp::stop("Target vector must contain only binary values (0 and 1)");
      }
      if (val == 1) {
        total_pos++;
      } else {
        total_neg++;
      }
    }
    
    // Precalculate logarithms of factorials
    size_t max_n = feature.size();
    log_factorials.resize(max_n + 1);
    log_factorials[0] = 0.0;
    for (size_t i = 1; i <= max_n; ++i) {
      log_factorials[i] = log_factorials[i - 1] + std::log(static_cast<double>(i));
    }
  }
  
  void fit() {
    mergeRareCategories();
    initializeBins();
    
    if (bins.size() > max_n_prebins) {
      max_bins = std::max(min_bins, std::min(max_bins, max_n_prebins));
    }
    
    mergeBins();
    
    if (bins.size() < min_bins) {
      Rcpp::warning("Final number of bins (%d) is less than min_bins (%d)", static_cast<int>(bins.size()), static_cast<int>(min_bins));
    }
  }
  
  Rcpp::List getResults() const {
    std::vector<std::string> bin_labels;
    std::vector<double> woe_values;
    std::vector<double> iv_values;
    std::vector<int> counts;
    std::vector<int> counts_pos;
    std::vector<int> counts_neg;
    std::vector<double> woe_feature(feature.size());
    
    std::unordered_map<std::string, double> category_woe_map;
    
    bin_labels.reserve(bins.size());
    woe_values.reserve(bins.size());
    iv_values.reserve(bins.size());
    counts.reserve(bins.size());
    counts_pos.reserve(bins.size());
    counts_neg.reserve(bins.size());
    
    for (const auto& bin : bins) {
      std::string label = joinStrings(bin.categories, "+");
      bin_labels.emplace_back(label);
      woe_values.emplace_back(bin.woe);
      iv_values.emplace_back(bin.iv);
      counts.emplace_back(static_cast<int>(bin.count));
      counts_pos.emplace_back(static_cast<int>(bin.count_pos));
      counts_neg.emplace_back(static_cast<int>(bin.count_neg));
      
      for (const auto& cat : bin.categories) {
        category_woe_map[cat] = bin.woe;
      }
    }
    
#pragma omp parallel for
    for (size_t i = 0; i < feature.size(); ++i) {
      auto it = category_woe_map.find(feature[i]);
      if (it != category_woe_map.end()) {
        woe_feature[i] = it->second;
      } else {
        woe_feature[i] = 0.0;
      }
    }
    
    Rcpp::DataFrame woebin = Rcpp::DataFrame::create(
      Rcpp::Named("bin") = bin_labels,
      Rcpp::Named("woe") = woe_values,
      Rcpp::Named("iv") = iv_values,
      Rcpp::Named("count") = counts,
      Rcpp::Named("count_pos") = counts_pos,
      Rcpp::Named("count_neg") = counts_neg
    );
    
    return Rcpp::List::create(
      Rcpp::Named("woefeature") = woe_feature,
      Rcpp::Named("woebin") = woebin
    );
  }
};

//' @title Categorical Optimal Binning with Fisher's Exact Test
//'
//' @description
//' Implements optimal binning for categorical variables using Fisher's Exact Test,
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
//' The algorithm uses Fisher's Exact Test to iteratively merge bins, maximizing
//' the statistical significance of the difference between adjacent bins.
//'
//' Weight of Evidence (WoE) for each bin is calculated as:
//'
//' \deqn{WoE = \ln\left(\frac{P(X|Y=1)}{P(X|Y=0)}\right)}
//'
//' Information Value (IV) for each bin is calculated as:
//'
//' \deqn{IV = (P(X|Y=1) - P(X|Y=0)) \times WoE}
//'
//' Fisher's Exact Test p-value is calculated using the hypergeometric distribution:
//'
//' \deqn{p = \frac{{a+b \choose a}{c+d \choose c}}{{n \choose a+c}}}
//'
//' where a, b, c, d are the elements of the 2x2 contingency table, and n is the total sample size.
//'
//' The algorithm first merges rare categories based on the bin_cutoff, then
//' iteratively merges bins with the lowest p-value from Fisher's Exact Test
//' until the desired number of bins is reached or further merging is not statistically significant.
//'
//' @examples
//' \dontrun{
//' # Sample data
//' target <- c(1, 0, 1, 1, 0, 1, 0, 0, 1, 1)
//' feature <- c("A", "B", "A", "C", "B", "D", "C", "A", "D", "B")
//'
//' # Run optimal binning
//' result <- optimal_binning_categorical_fetb(target, feature, min_bins = 2, max_bins = 4)
//'
//' # View results
//' print(result$woefeature)
//' print(result$woebin)
//' }
//'
//' @author Lopes, J. E.
//'
//' @references
//' \itemize{
//'   \item Agresti, A. (1992). A Survey of Exact Inference for Contingency Tables. 
//'         Statistical Science, 7(1), 131-153.
//'   \item Savage, L. J. (1956). On the Choice of a Classification Statistic. 
//'         In Contributions to Probability and Statistics: Essays in Honor of Harold Hotelling, 
//'         Stanford University Press, 139-161.
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_fetb(Rcpp::IntegerVector target,
                                           Rcpp::CharacterVector feature,
                                           int min_bins = 3,
                                           int max_bins = 5,
                                           double bin_cutoff = 0.05,
                                           int max_n_prebins = 20) {
 // Convert Rcpp vectors to std::vector
 std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
 std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
 
 // Create and run the binner
 OptimalBinningCategoricalFETB binner(feature_vec, target_vec,
                                      static_cast<size_t>(min_bins),
                                      static_cast<size_t>(max_bins),
                                      bin_cutoff,
                                      static_cast<size_t>(max_n_prebins));
 
 binner.fit();
 
 // Return the results
 return binner.getResults();
}

 
 
 
 
 
 
 
 
 
 
 
// #include <Rcpp.h>
// #include <vector>
// #include <string>
// #include <algorithm>
// #include <cmath>
// #include <unordered_map>
// #include <limits>
// #include <sstream>
// #include <set>
// 
// // [[Rcpp::plugins(cpp11)]]
// // [[Rcpp::plugins(openmp)]]
// 
// class OptimalBinningCategoricalFETB {
// private:
//   const std::vector<std::string>& feature;
//   const std::vector<int>& target;
//   const size_t min_bins;
//   size_t max_bins;
//   const double bin_cutoff;
//   const size_t max_n_prebins;
//   
//   std::unordered_map<std::string, std::string> original_to_merged_category;
//   std::unordered_map<std::string, std::set<std::string>> merged_to_original_categories;
//   
//   struct BinInfo {
//     std::set<std::string> categories;
//     size_t count = 0;
//     size_t count_pos = 0;
//     size_t count_neg = 0;
//     double woe = 0.0;
//     double iv = 0.0;
//   };
//   
//   std::vector<BinInfo> bins;
//   size_t total_pos = 0;
//   size_t total_neg = 0;
//   
//   static constexpr double EPSILON = 1e-10;
//   
//   void calculateWoeIv(BinInfo& bin) const {
//     if (bin.count_pos == 0 || bin.count_neg == 0 || total_pos == 0 || total_neg == 0) {
//       bin.woe = 0.0;
//       bin.iv = 0.0;
//       return;
//     }
//     
//     double dist_pos = static_cast<double>(bin.count_pos) / total_pos;
//     double dist_neg = static_cast<double>(bin.count_neg) / total_neg;
//     
//     bin.woe = std::log((dist_pos + EPSILON) / (dist_neg + EPSILON));
//     bin.iv = (dist_pos - dist_neg) * bin.woe;
//   }
//   
//   double fisherExactTest(size_t a, size_t b, size_t c, size_t d) const {
//     size_t n = a + b + c + d;
//     size_t row1 = a + b;
//     size_t row2 = c + d;
//     size_t col1 = a + c;
//     size_t col2 = b + d;
//     
//     auto logFactorial = [](size_t x) -> double {
//       return std::lgamma(static_cast<double>(x) + 1.0);
//     };
//     
//     double log_p = logFactorial(row1) + logFactorial(row2) + logFactorial(col1) + logFactorial(col2)
//       - logFactorial(n) - logFactorial(a) - logFactorial(b) - logFactorial(c) - logFactorial(d);
//     
//     return std::exp(log_p);
//   }
//   
//   void mergeRareCategories() {
//     std::unordered_map<std::string, size_t> category_counts;
//     size_t total_count = feature.size();
//     
//     for (const auto& cat : feature) {
//       category_counts[cat]++;
//     }
//     
//     std::vector<std::pair<std::string, size_t>> sorted_categories(category_counts.begin(), category_counts.end());
//     std::sort(sorted_categories.begin(), sorted_categories.end(),
//               [](const std::pair<std::string, size_t>& a, const std::pair<std::string, size_t>& b) {
//                 return a.second > b.second;
//               });
//     
//     const std::string rare_category = "__RARE__";
//     
//     for (const auto& [cat, count] : sorted_categories) {
//       double frequency = static_cast<double>(count) / total_count;
//       if (frequency < bin_cutoff) {
//         original_to_merged_category[cat] = rare_category;
//       } else {
//         original_to_merged_category[cat] = cat;
//       }
//     }
//     
//     for (const auto& [original_cat, merged_cat] : original_to_merged_category) {
//       merged_to_original_categories[merged_cat].insert(original_cat);
//     }
//   }
//   
//   void initializeBins() {
//     std::unordered_map<std::string, BinInfo> bin_map;
//     
//     for (size_t i = 0; i < feature.size(); ++i) {
//       const std::string& original_cat = feature[i];
//       auto it = original_to_merged_category.find(original_cat);
//       if (it == original_to_merged_category.end()) {
//         continue;
//       }
//       const std::string& merged_cat = it->second;
//       
//       BinInfo& bin = bin_map[merged_cat];
//       bin.count++;
//       if (target[i] == 1) {
//         bin.count_pos++;
//       } else {
//         bin.count_neg++;
//       }
//     }
//     
//     for (auto& [merged_cat, bin_info] : bin_map) {
//       bin_info.categories = merged_to_original_categories[merged_cat];
//       calculateWoeIv(bin_info);
//     }
//     
//     bins.reserve(bin_map.size());
//     for (auto& [_, bin_info] : bin_map) {
//       bins.emplace_back(std::move(bin_info));
//     }
//     
//     std::sort(bins.begin(), bins.end(), [](const BinInfo& a, const BinInfo& b) {
//       return a.woe < b.woe;
//     });
//   }
//   
//   void mergeBins() {
//     while (bins.size() > max_bins) {
//       if (bins.size() <= min_bins) {
//         break;
//       }
//       
//       double min_p_value = std::numeric_limits<double>::max();
//       size_t merge_index = 0;
//       
//       for (size_t i = 0; i < bins.size() - 1; ++i) {
//         size_t a = bins[i].count_pos;
//         size_t b = bins[i].count_neg;
//         size_t c = bins[i + 1].count_pos;
//         size_t d = bins[i + 1].count_neg;
//         
//         double p_value = fisherExactTest(a, b, c, d);
//         
//         if (p_value < min_p_value) {
//           min_p_value = p_value;
//           merge_index = i;
//         }
//       }
//       
//       BinInfo& bin1 = bins[merge_index];
//       BinInfo& bin2 = bins[merge_index + 1];
//       
//       bin1.count += bin2.count;
//       bin1.count_pos += bin2.count_pos;
//       bin1.count_neg += bin2.count_neg;
//       bin1.categories.insert(bin2.categories.begin(), bin2.categories.end());
//       
//       calculateWoeIv(bin1);
//       
//       bins.erase(bins.begin() + merge_index + 1);
//     }
//     
//     if (bins.size() < min_bins) {
//       Rcpp::warning("Resulting number of bins (%d) is less than min_bins (%d)", static_cast<int>(bins.size()), static_cast<int>(min_bins));
//     }
//   }
//   
//   static std::string joinStrings(const std::set<std::string>& strings, const std::string& delimiter) {
//     std::ostringstream oss;
//     size_t i = 0;
//     for (const auto& s : strings) {
//       if (i > 0) oss << delimiter;
//       oss << s;
//       i++;
//     }
//     return oss.str();
//   }
//   
// public:
//   OptimalBinningCategoricalFETB(const std::vector<std::string>& feature,
//                                 const std::vector<int>& target,
//                                 size_t min_bins = 3,
//                                 size_t max_bins = 5,
//                                 double bin_cutoff = 0.05,
//                                 size_t max_n_prebins = 20)
//     : feature(feature), target(target),
//       min_bins(min_bins), max_bins(max_bins),
//       bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins) {
//     
//     if (min_bins < 2) {
//       Rcpp::stop("min_bins must be >= 2");
//     }
//     if (max_bins < min_bins) {
//       Rcpp::stop("max_bins must be >= min_bins");
//     }
//     if (bin_cutoff <= 0.0 || bin_cutoff >= 1.0) {
//       Rcpp::stop("bin_cutoff must be between 0 and 1");
//     }
//     if (max_n_prebins < min_bins) {
//       Rcpp::stop("max_n_prebins must be >= min_bins");
//     }
//     
//     for (int val : target) {
//       if (val != 0 && val != 1) {
//         Rcpp::stop("Target vector must contain only binary values (0 and 1)");
//       }
//       if (val == 1) {
//         total_pos++;
//       } else {
//         total_neg++;
//       }
//     }
//   }
//   
//   void fit() {
//     mergeRareCategories();
//     initializeBins();
//     
//     if (bins.size() > max_n_prebins) {
//       max_bins = std::max(min_bins, std::min(max_bins, max_n_prebins));
//     }
//     
//     mergeBins();
//     
//     if (bins.size() < min_bins) {
//       Rcpp::warning("Final number of bins (%d) is less than min_bins (%d)", static_cast<int>(bins.size()), static_cast<int>(min_bins));
//     }
//   }
//   
//   Rcpp::List getResults() const {
//     std::vector<std::string> bin_labels;
//     std::vector<double> woe_values;
//     std::vector<double> iv_values;
//     std::vector<int> counts;
//     std::vector<int> counts_pos;
//     std::vector<int> counts_neg;
//     std::vector<double> woe_feature(feature.size());
//     
//     std::unordered_map<std::string, double> category_woe_map;
//     
//     bin_labels.reserve(bins.size());
//     woe_values.reserve(bins.size());
//     iv_values.reserve(bins.size());
//     counts.reserve(bins.size());
//     counts_pos.reserve(bins.size());
//     counts_neg.reserve(bins.size());
//     
//     for (const auto& bin : bins) {
//       std::string label = joinStrings(bin.categories, "+");
//       bin_labels.emplace_back(label);
//       woe_values.emplace_back(bin.woe);
//       iv_values.emplace_back(bin.iv);
//       counts.emplace_back(static_cast<int>(bin.count));
//       counts_pos.emplace_back(static_cast<int>(bin.count_pos));
//       counts_neg.emplace_back(static_cast<int>(bin.count_neg));
//       
//       for (const auto& cat : bin.categories) {
//         category_woe_map[cat] = bin.woe;
//       }
//     }
//     
// #pragma omp parallel for
//     for (size_t i = 0; i < feature.size(); ++i) {
//       auto it = category_woe_map.find(feature[i]);
//       if (it != category_woe_map.end()) {
//         woe_feature[i] = it->second;
//       } else {
//         woe_feature[i] = 0.0;
//       }
//     }
//     
//     Rcpp::DataFrame woebin = Rcpp::DataFrame::create(
//       Rcpp::Named("bin") = bin_labels,
//       Rcpp::Named("woe") = woe_values,
//       Rcpp::Named("iv") = iv_values,
//       Rcpp::Named("count") = counts,
//       Rcpp::Named("count_pos") = counts_pos,
//       Rcpp::Named("count_neg") = counts_neg
//     );
//     
//     return Rcpp::List::create(
//       Rcpp::Named("woefeature") = woe_feature,
//       Rcpp::Named("woebin") = woebin
//     );
//   }
// };
// 
// //' @title Categorical Optimal Binning with Fisher's Exact Test
// //'
// //' @description
// //' Implements optimal binning for categorical variables using Fisher's Exact Test,
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
// //' The algorithm uses Fisher's Exact Test to iteratively merge bins, maximizing
// //' the statistical significance of the difference between adjacent bins.
// //'
// //' Weight of Evidence (WoE) for each bin is calculated as:
// //'
// //' \deqn{WoE = \ln\left(\frac{P(X|Y=1)}{P(X|Y=0)}\right)}
// //'
// //' Information Value (IV) for each bin is calculated as:
// //'
// //' \deqn{IV = (P(X|Y=1) - P(X|Y=0)) \times WoE}
// //'
// //' Fisher's Exact Test p-value is calculated using the hypergeometric distribution:
// //'
// //' \deqn{p = \frac{{a+b \choose a}{c+d \choose c}}{{n \choose a+c}}}
// //'
// //' where a, b, c, d are the elements of the 2x2 contingency table, and n is the total sample size.
// //'
// //' The algorithm first merges rare categories based on the bin_cutoff, then
// //' iteratively merges bins with the lowest p-value from Fisher's Exact Test
// //' until the desired number of bins is reached or further merging is not statistically significant.
// //'
// //' @examples
// //' \dontrun{
// //' # Sample data
// //' target <- c(1, 0, 1, 1, 0, 1, 0, 0, 1, 1)
// //' feature <- c("A", "B", "A", "C", "B", "D", "C", "A", "D", "B")
// //'
// //' # Run optimal binning
// //' result <- optimal_binning_categorical_fetb(target, feature, min_bins = 2, max_bins = 4)
// //'
// //' # View results
// //' print(result$woefeature)
// //' }
// //'
// //' @author Lopes, J. E.
// //'
// //' @references
// //' \itemize{
// //'   \item Agresti, A. (1992). A Survey of Exact Inference for Contingency Tables. 
// //'         Statistical Science, 7(1), 131-153.
// //'   \item Savage, L. J. (1956). On the Choice of a Classification Statistic. 
// //'         In Contributions to Probability and Statistics: Essays in Honor of Harold Hotelling, 
// //'         Stanford University Press, 139-161.
// //' }
// //'
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_categorical_fetb(Rcpp::IntegerVector target,
//                                            Rcpp::CharacterVector feature,
//                                            int min_bins = 3,
//                                            int max_bins = 5,
//                                            double bin_cutoff = 0.05,
//                                            int max_n_prebins = 20) {
//  std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
//  std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
//  
//  OptimalBinningCategoricalFETB binner(feature_vec, target_vec,
//                                       static_cast<size_t>(min_bins),
//                                       static_cast<size_t>(max_bins),
//                                       bin_cutoff,
//                                       static_cast<size_t>(max_n_prebins));
//  
//  binner.fit();
//  
//  return binner.getResults();
// }


// #include <Rcpp.h>
// #include <vector>
// #include <string>
// #include <algorithm>
// #include <cmath>
// #include <unordered_map>
// #include <limits>
// #include <sstream>
// #include <set>
// 
// // [[Rcpp::plugins(cpp11)]]
// // [[Rcpp::plugins(openmp)]]
// 
// class OptimalBinningCategoricalFETB {
// private:
//   std::vector<std::string> feature;
//   const std::vector<int> target;
//   const size_t min_bins;
//   size_t max_bins;
//   const double bin_cutoff;
//   const size_t max_n_prebins;
// 
//   std::unordered_map<std::string, std::string> original_to_merged_category;
//   std::unordered_map<std::string, std::set<std::string>> merged_to_original_categories;
// 
//   struct BinInfo {
//     std::set<std::string> categories;
//     size_t count = 0;
//     size_t count_pos = 0;
//     size_t count_neg = 0;
//     double woe = 0.0;
//     double iv = 0.0;
//   };
// 
//   std::vector<BinInfo> bins;
//   size_t total_pos = 0;
//   size_t total_neg = 0;
// 
//   void calculateWoeIv(BinInfo& bin) const {
//     if (bin.count_pos == 0 || bin.count_neg == 0 || total_pos == 0 || total_neg == 0) {
//       bin.woe = 0.0;
//       bin.iv = 0.0;
//       return;
//     }
// 
//     double dist_pos = static_cast<double>(bin.count_pos) / total_pos;
//     double dist_neg = static_cast<double>(bin.count_neg) / total_neg;
// 
//     const double epsilon = 1e-10;
//     bin.woe = std::log((dist_pos + epsilon) / (dist_neg + epsilon));
//     bin.iv = (dist_pos - dist_neg) * bin.woe;
//   }
// 
//   double fisherExactTest(size_t a, size_t b, size_t c, size_t d) const {
//     size_t n = a + b + c + d;
//     size_t row1 = a + b;
//     size_t row2 = c + d;
//     size_t col1 = a + c;
//     size_t col2 = b + d;
// 
//     auto logFactorial = [&](size_t x) -> double {
//       return std::lgamma(static_cast<double>(x) + 1.0);
//     };
// 
//     double log_p = logFactorial(row1) + logFactorial(row2) + logFactorial(col1) + logFactorial(col2)
//       - logFactorial(n) - logFactorial(a) - logFactorial(b) - logFactorial(c) - logFactorial(d);
// 
//     return std::exp(log_p);
//   }
// 
//   void mergeRareCategories() {
//     std::unordered_map<std::string, size_t> category_counts;
//     size_t total_count = feature.size();
// 
//     for (const auto& cat : feature) {
//       category_counts[cat]++;
//     }
// 
//     std::vector<std::pair<std::string, size_t>> sorted_categories(category_counts.begin(), category_counts.end());
//     std::sort(sorted_categories.begin(), sorted_categories.end(),
//               [](const std::pair<std::string, size_t>& a, const std::pair<std::string, size_t>& b) {
//                 return a.second > b.second;
//               });
// 
//     const std::string rare_category = "__RARE__";
// 
//     for (const auto& [cat, count] : sorted_categories) {
//       double frequency = static_cast<double>(count) / total_count;
//       if (frequency < bin_cutoff) {
//         original_to_merged_category[cat] = rare_category;
//       } else {
//         original_to_merged_category[cat] = cat;
//       }
//     }
// 
//     for (const auto& [original_cat, merged_cat] : original_to_merged_category) {
//       merged_to_original_categories[merged_cat].insert(original_cat);
//     }
//   }
// 
//   void initializeBins() {
//     std::unordered_map<std::string, BinInfo> bin_map;
// 
//     for (size_t i = 0; i < feature.size(); ++i) {
//       const std::string& original_cat = feature[i];
//       auto it = original_to_merged_category.find(original_cat);
//       if (it == original_to_merged_category.end()) {
//         continue;
//       }
//       const std::string& merged_cat = it->second;
// 
//       BinInfo& bin = bin_map[merged_cat];
//       bin.count++;
//       if (target[i] == 1) {
//         bin.count_pos++;
//       } else {
//         bin.count_neg++;
//       }
//     }
// 
//     for (auto& [merged_cat, bin_info] : bin_map) {
//       bin_info.categories = merged_to_original_categories[merged_cat];
//       calculateWoeIv(bin_info);
//     }
// 
//     bins.reserve(bin_map.size());
//     for (auto& [_, bin_info] : bin_map) {
//       bins.emplace_back(std::move(bin_info));
//     }
// 
//     std::sort(bins.begin(), bins.end(), [](const BinInfo& a, const BinInfo& b) {
//       return a.woe < b.woe;
//     });
//   }
// 
//   void mergeBins() {
//     while (bins.size() > max_bins) {
//       if (bins.size() <= min_bins) {
//         break;
//       }
// 
//       double min_p_value = std::numeric_limits<double>::max();
//       size_t merge_index = 0;
// 
//       for (size_t i = 0; i < bins.size() - 1; ++i) {
//         size_t a = bins[i].count_pos;
//         size_t b = bins[i].count_neg;
//         size_t c = bins[i + 1].count_pos;
//         size_t d = bins[i + 1].count_neg;
// 
//         double p_value = fisherExactTest(a, b, c, d);
// 
//         if (p_value < min_p_value) {
//           min_p_value = p_value;
//           merge_index = i;
//         }
//       }
// 
//       BinInfo& bin1 = bins[merge_index];
//       BinInfo& bin2 = bins[merge_index + 1];
// 
//       bin1.count += bin2.count;
//       bin1.count_pos += bin2.count_pos;
//       bin1.count_neg += bin2.count_neg;
//       bin1.categories.insert(bin2.categories.begin(), bin2.categories.end());
// 
//       calculateWoeIv(bin1);
// 
//       bins.erase(bins.begin() + merge_index + 1);
//     }
// 
//     if (bins.size() < min_bins) {
//       Rcpp::warning("Resulting number of bins (%d) is less than min_bins (%d)", static_cast<int>(bins.size()), static_cast<int>(min_bins));
//     }
//   }
// 
//   std::string joinStrings(const std::set<std::string>& strings, const std::string& delimiter) const {
//     std::ostringstream oss;
//     size_t i = 0;
//     for (const auto& s : strings) {
//       if (i > 0) oss << delimiter;
//       oss << s;
//       i++;
//     }
//     return oss.str();
//   }
// 
// public:
//   OptimalBinningCategoricalFETB(const std::vector<std::string>& feature,
//                                 const std::vector<int>& target,
//                                 size_t min_bins = 3,
//                                 size_t max_bins = 5,
//                                 double bin_cutoff = 0.05,
//                                 size_t max_n_prebins = 20)
//     : feature(feature), target(target),
//       min_bins(min_bins), max_bins(max_bins),
//       bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins) {
// 
//     if (min_bins < 2) {
//       Rcpp::stop("min_bins must be >= 2");
//     }
//     if (max_bins < min_bins) {
//       Rcpp::stop("max_bins must be >= min_bins");
//     }
//     if (bin_cutoff <= 0.0 || bin_cutoff >= 1.0) {
//       Rcpp::stop("bin_cutoff must be between 0 and 1");
//     }
//     if (max_n_prebins < min_bins) {
//       Rcpp::stop("max_n_prebins must be >= min_bins");
//     }
// 
//     for (size_t i = 0; i < target.size(); ++i) {
//       int val = target[i];
//       if (val != 0 && val != 1) {
//         Rcpp::stop("Target vector must contain only binary values (0 and 1)");
//       }
//       if (val == 1) {
//         total_pos++;
//       } else {
//         total_neg++;
//       }
//     }
//   }
// 
//   void fit() {
//     mergeRareCategories();
//     initializeBins();
// 
//     if (bins.size() > max_n_prebins) {
//       max_bins = std::max(min_bins, std::min(max_bins, max_n_prebins));
//     }
// 
//     mergeBins();
// 
//     if (bins.size() < min_bins) {
//       Rcpp::warning("Final number of bins (%d) is less than min_bins (%d)", static_cast<int>(bins.size()), static_cast<int>(min_bins));
//     }
//   }
// 
//   Rcpp::List getResults() const {
//     std::vector<std::string> bin_labels;
//     std::vector<double> woe_values;
//     std::vector<double> iv_values;
//     std::vector<int> counts;
//     std::vector<int> counts_pos;
//     std::vector<int> counts_neg;
//     std::vector<double> woe_feature(feature.size());
// 
//     std::unordered_map<std::string, double> category_woe_map;
// 
//     bin_labels.reserve(bins.size());
//     woe_values.reserve(bins.size());
//     iv_values.reserve(bins.size());
//     counts.reserve(bins.size());
//     counts_pos.reserve(bins.size());
//     counts_neg.reserve(bins.size());
// 
//     for (const auto& bin : bins) {
//       std::string label = joinStrings(bin.categories, "+");
//       bin_labels.emplace_back(label);
//       woe_values.emplace_back(bin.woe);
//       iv_values.emplace_back(bin.iv);
//       counts.emplace_back(static_cast<int>(bin.count));
//       counts_pos.emplace_back(static_cast<int>(bin.count_pos));
//       counts_neg.emplace_back(static_cast<int>(bin.count_neg));
// 
//       for (const auto& cat : bin.categories) {
//         category_woe_map[cat] = bin.woe;
//       }
//     }
// 
// #pragma omp parallel for
//     for (size_t i = 0; i < feature.size(); ++i) {
//       auto it = category_woe_map.find(feature[i]);
//       if (it != category_woe_map.end()) {
//         woe_feature[i] = it->second;
//       } else {
//         woe_feature[i] = 0.0;
//       }
//     }
// 
//     Rcpp::DataFrame woebin = Rcpp::DataFrame::create(
//       Rcpp::Named("bin") = bin_labels,
//       Rcpp::Named("woe") = woe_values,
//       Rcpp::Named("iv") = iv_values,
//       Rcpp::Named("count") = counts,
//       Rcpp::Named("count_pos") = counts_pos,
//       Rcpp::Named("count_neg") = counts_neg
//     );
// 
//     return Rcpp::List::create(
//       Rcpp::Named("woefeature") = woe_feature,
//       Rcpp::Named("woebin") = woebin
//     );
//   }
// };
// 
// 
// //' @title Categorical Optimal Binning with Fisher's Exact Test
// //'
// //' @description
// //' Implements optimal binning for categorical variables using Fisher's Exact Test,
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
// //' The algorithm uses Fisher's Exact Test to iteratively merge bins, maximizing
// //' the statistical significance of the difference between adjacent bins.
// //'
// //' Weight of Evidence (WoE) for each bin is calculated as:
// //'
// //' \deqn{WoE = \ln\left(\frac{P(X|Y=1)}{P(X|Y=0)}\right)}
// //'
// //' Information Value (IV) for each bin is calculated as:
// //'
// //' \deqn{IV = (P(X|Y=1) - P(X|Y=0)) \times WoE}
// //'
// //' Fisher's Exact Test p-value is calculated using the hypergeometric distribution:
// //'
// //' \deqn{p = \frac{{a+b \choose a}{c+d \choose c}}{{n \choose a+c}}}
// //'
// //' where a, b, c, d are the elements of the 2x2 contingency table, and n is the total sample size.
// //'
// //' The algorithm first merges rare categories based on the bin_cutoff, then
// //' iteratively merges bins with the lowest p-value from Fisher's Exact Test
// //' until the desired number of bins is reached or further merging is not statistically significant.
// //'
// //' @examples
// //' \dontrun{
// //' # Sample data
// //' target <- c(1, 0, 1, 1, 0, 1, 0, 0, 1, 1)
// //' feature <- c("A", "B", "A", "C", "B", "D", "C", "A", "D", "B")
// //'
// //' # Run optimal binning
// //' result <- optimal_binning_categorical_fetb(target, feature, min_bins = 2, max_bins = 4)
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
// //'   \item Agresti, A. (1992). A Survey of Exact Inference for Contingency Tables. 
// //'         Statistical Science, 7(1), 131-153.
// //'   \item Savage, L. J. (1956). On the Choice of a Classification Statistic. 
// //'         In Contributions to Probability and Statistics: Essays in Honor of Harold Hotelling, 
// //'         Stanford University Press, 139-161.
// //' }
// //'
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_categorical_fetb(Rcpp::IntegerVector target,
//                                           Rcpp::CharacterVector feature,
//                                           int min_bins = 3,
//                                           int max_bins = 5,
//                                           double bin_cutoff = 0.05,
//                                           int max_n_prebins = 20) {
// std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
// std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
// 
// OptimalBinningCategoricalFETB binner(feature_vec, target_vec,
//                                      static_cast<size_t>(min_bins),
//                                      static_cast<size_t>(max_bins),
//                                      bin_cutoff,
//                                      static_cast<size_t>(max_n_prebins));
// 
// binner.fit();
// 
// return binner.getResults();
// }
