// [[Rcpp::plugins(cpp11)]]
#include <Rcpp.h>
#include <unordered_map>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

class OptimalBinningCategoricalIVB {
private:
  const std::vector<std::string>& feature;
  const std::vector<int>& target;
  double bin_cutoff;
  int min_bins;
  int max_bins;
  int max_n_prebins;
  
  std::unordered_map<std::string, int> category_counts;
  std::unordered_map<std::string, int> category_pos_counts;
  std::unordered_map<std::string, int> category_neg_counts;
  
  std::vector<std::string> merged_bins;
  std::vector<double> woe_values;
  std::vector<double> iv_values;
  std::vector<int> count_values;
  std::vector<int> count_pos_values;
  std::vector<int> count_neg_values;
  
  void validate_inputs() const;
  void compute_category_stats() noexcept;
  void optimize_binning() noexcept;
  void compute_woe_iv();
  double safe_log(double x) const noexcept;
  void handle_small_categories() noexcept;
  
public:
  OptimalBinningCategoricalIVB(const std::vector<std::string>& feature,
                               const std::vector<int>& target,
                               double bin_cutoff,
                               int min_bins,
                               int max_bins,
                               int max_n_prebins);
  
  Rcpp::List fit();
};

OptimalBinningCategoricalIVB::OptimalBinningCategoricalIVB(const std::vector<std::string>& feature,
                                                           const std::vector<int>& target,
                                                           double bin_cutoff,
                                                           int min_bins,
                                                           int max_bins,
                                                           int max_n_prebins)
  : feature(feature), target(target), bin_cutoff(bin_cutoff),
    min_bins(min_bins), max_bins(max_bins), max_n_prebins(max_n_prebins) {
  validate_inputs();
}

void OptimalBinningCategoricalIVB::validate_inputs() const {
  if (feature.empty() || target.empty()) {
    Rcpp::stop("Input vectors cannot be empty.");
  }
  if (feature.size() != target.size()) {
    Rcpp::stop("target and feature must have the same length.");
  }
  if (min_bins < 2) {
    Rcpp::stop("min_bins must be at least 2.");
  }
  if (max_bins < min_bins) {
    Rcpp::stop("max_bins must be greater than or equal to min_bins.");
  }
  if (bin_cutoff < 0 || bin_cutoff > 1) {
    Rcpp::stop("bin_cutoff must be between 0 and 1.");
  }
  for (auto& t : target) {
    if (t != 0 && t != 1) {
      Rcpp::stop("target must be binary (0 or 1).");
    }
  }
}

void OptimalBinningCategoricalIVB::compute_category_stats() noexcept {
  // Reserve space for the unordered_maps
  category_counts.reserve(feature.size());
  category_pos_counts.reserve(feature.size());
  category_neg_counts.reserve(feature.size());
  
  // Build category counts
  for (size_t i = 0; i < feature.size(); ++i) {
    category_counts[feature[i]]++;
    if (target[i] == 1) {
      category_pos_counts[feature[i]]++;
    } else {
      category_neg_counts[feature[i]]++;
    }
  }
}

void OptimalBinningCategoricalIVB::handle_small_categories() noexcept {
  std::string other_category = "Other";
  int other_count = 0;
  int other_pos_count = 0;
  int other_neg_count = 0;
  
  for (auto it = category_counts.begin(); it != category_counts.end();) {
    double category_freq = static_cast<double>(it->second) / feature.size();
    if (category_freq < bin_cutoff) {
      std::string key = it->first;  // Store key before erasing
      other_count += it->second;
      other_pos_count += category_pos_counts[key];
      other_neg_count += category_neg_counts[key];
      it = category_counts.erase(it);
      category_pos_counts.erase(key);
      category_neg_counts.erase(key);
    } else {
      ++it;
    }
  }
  
  if (other_count > 0) {
    category_counts[other_category] = other_count;
    category_pos_counts[other_category] = other_pos_count;
    category_neg_counts[other_category] = other_neg_count;
  }
}

void OptimalBinningCategoricalIVB::optimize_binning() noexcept {
  handle_small_categories();
  
  struct CategoryStats {
    std::string category;
    double event_rate;
    int count;
  };
  
  std::vector<CategoryStats> stats;
  stats.reserve(category_counts.size());
  for (const auto& pair : category_counts) {
    double event_rate = static_cast<double>(category_pos_counts[pair.first]) / pair.second;
    stats.emplace_back(CategoryStats{pair.first, event_rate, pair.second});
  }
  
  // Sort categories by event rate to ensure monotonicity
  std::sort(stats.begin(), stats.end(), [](const CategoryStats& a, const CategoryStats& b) {
    return a.event_rate < b.event_rate;
  });
  
  // Limit the number of pre-bins
  if (static_cast<int>(stats.size()) > max_n_prebins) {
    stats.resize(max_n_prebins);
  }
  
  size_t n_bins = std::max(static_cast<size_t>(min_bins),
                           std::min(static_cast<size_t>(max_bins), stats.size()));
  size_t bin_size = stats.size() / n_bins;
  size_t remainder = stats.size() % n_bins;
  
  merged_bins.clear();
  merged_bins.reserve(n_bins);
  size_t idx = 0;
  for (size_t i = 0; i < n_bins; ++i) {
    size_t current_bin_size = bin_size + (i < remainder ? 1 : 0);
    std::string bin_name;
    for (size_t j = 0; j < current_bin_size; ++j) {
      if (!bin_name.empty()) bin_name += "+";
      bin_name += stats[idx++].category;
    }
    merged_bins.emplace_back(std::move(bin_name));
  }
}

double OptimalBinningCategoricalIVB::safe_log(double x) const noexcept {
  return std::log(std::max(x, std::numeric_limits<double>::epsilon()));
}

void OptimalBinningCategoricalIVB::compute_woe_iv() {
  int total_pos = std::accumulate(target.begin(), target.end(), 0);
  int total_neg = target.size() - total_pos;
  
  if (total_pos == 0 || total_neg == 0) {
    Rcpp::stop("Target has only one class.");
  }
  
  woe_values.clear();
  iv_values.clear();
  count_values.clear();
  count_pos_values.clear();
  count_neg_values.clear();
  
  woe_values.reserve(merged_bins.size());
  iv_values.reserve(merged_bins.size());
  count_values.reserve(merged_bins.size());
  count_pos_values.reserve(merged_bins.size());
  count_neg_values.reserve(merged_bins.size());
  
  double total_iv = 0.0;
  
  for (const auto& bin : merged_bins) {
    int bin_pos = 0, bin_neg = 0;
    size_t start = 0, end = bin.find('+');
    while (end != std::string::npos) {
      std::string cat = bin.substr(start, end - start);
      bin_pos += category_pos_counts[cat];
      bin_neg += category_neg_counts[cat];
      start = end + 1;
      end = bin.find('+', start);
    }
    std::string cat = bin.substr(start);
    bin_pos += category_pos_counts[cat];
    bin_neg += category_neg_counts[cat];
    
    int bin_count = bin_pos + bin_neg;
    double dist_pos = static_cast<double>(bin_pos) / total_pos;
    double dist_neg = static_cast<double>(bin_neg) / total_neg;
    
    double woe = safe_log(dist_pos / dist_neg);
    double iv = (dist_pos - dist_neg) * woe;
    
    woe_values.push_back(woe);
    iv_values.push_back(iv);
    count_values.push_back(bin_count);
    count_pos_values.push_back(bin_pos);
    count_neg_values.push_back(bin_neg);
    
    total_iv += iv;
  }
  
  // Handle zero information case
  if (total_iv == 0.0) {
    Rcpp::warning("Zero information value. Check if target has sufficient variation.");
  }
}

Rcpp::List OptimalBinningCategoricalIVB::fit() {
  compute_category_stats();
  optimize_binning();
  compute_woe_iv();
  
  std::vector<double> woefeature(feature.size());
  std::unordered_map<std::string, double> category_to_woe;
  
  for (size_t i = 0; i < merged_bins.size(); ++i) {
    size_t start = 0, end = merged_bins[i].find('+');
    while (end != std::string::npos) {
      std::string cat = merged_bins[i].substr(start, end - start);
      category_to_woe[cat] = woe_values[i];
      start = end + 1;
      end = merged_bins[i].find('+', start);
    }
    std::string cat = merged_bins[i].substr(start);
    category_to_woe[cat] = woe_values[i];
  }
  
  // Apply WoE to the feature vector
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (size_t i = 0; i < feature.size(); ++i) {
    auto it = category_to_woe.find(feature[i]);
    if (it != category_to_woe.end()) {
      woefeature[i] = it->second;
    } else {
      // Handle unseen categories
      woefeature[i] = 0.0; // or use a special value to indicate unseen categories
    }
  }
  
  Rcpp::DataFrame woebin = Rcpp::DataFrame::create(
    Rcpp::Named("bin") = merged_bins,
    Rcpp::Named("woe") = woe_values,
    Rcpp::Named("iv") = iv_values,
    Rcpp::Named("count") = count_values,
    Rcpp::Named("count_pos") = count_pos_values,
    Rcpp::Named("count_neg") = count_neg_values
  );
  
  return Rcpp::List::create(
    Rcpp::Named("woefeature") = woefeature,
    Rcpp::Named("woebin") = woebin,
    Rcpp::Named("category_mapping") = Rcpp::wrap(category_to_woe)
  );
}

//' @title Categorical Optimal Binning with Information Value Binning
//'
//' @description
//' Implements optimal binning for categorical variables using Information Value (IV)
//' as the primary criterion, calculating Weight of Evidence (WoE) and IV for resulting bins.
//'
//' @param target Integer vector of binary target values (0 or 1).
//' @param feature Character vector or factor of categorical feature values.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum frequency for a separate bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins before merging (default: 20).
//'
//' @return A list with three elements:
//' \itemize{
//'   \item woefeature: Numeric vector of WoE values for each input feature value.
//'   \item woebin: Data frame with binning results (bin names, WoE, IV, counts).
//'   \item category_mapping: Named vector mapping original categories to their WoE values.
//' }
//'
//' @details
//' The algorithm uses Information Value (IV) to create optimal bins for categorical variables.
//' It starts by computing statistics for each category, then sorts categories by event rate
//' to ensure monotonicity. The algorithm then creates initial bins based on the specified
//' constraints and computes WoE and IV for each bin.
//'
//' Weight of Evidence (WoE) for each bin is calculated as:
//' \deqn{WoE_i = \ln(\frac{P(X|Y=1)}{P(X|Y=0)})}
//'
//' Information Value (IV) for each bin is calculated as:
//' \deqn{IV = \sum_{i=1}^{N} (P(X|Y=1) - P(X|Y=0)) \times WoE_i}
//'
//' @examples
//' \dontrun{
//' # Sample data
//' target <- c(1, 0, 1, 1, 0, 1, 0, 0, 1, 1)
//' feature <- c("A", "B", "A", "C", "B", "D", "C", "A", "D", "B")
//'
//' # Run optimal binning
//' result <- optimal_binning_categorical_ivb(target, feature, min_bins = 2, max_bins = 4)
//'
//' # View results
//' print(result$woebin)
//' print(result$woefeature)
//' print(result$category_mapping)
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_ivb(Rcpp::IntegerVector target,
                                          Rcpp::CharacterVector feature,
                                          int min_bins = 3,
                                          int max_bins = 5,
                                          double bin_cutoff = 0.05,
                                          int max_n_prebins = 20) {
 std::vector<std::string> feature_str;
 feature_str.reserve(feature.size());
 
 if (Rf_isFactor(feature)) {
   Rcpp::IntegerVector levels = Rcpp::as<Rcpp::IntegerVector>(feature);
   Rcpp::CharacterVector level_names = feature.attr("levels");
   for (int i = 0; i < levels.size(); ++i) {
     feature_str.emplace_back(Rcpp::as<std::string>(level_names[levels[i] - 1]));
   }
 } else if (TYPEOF(feature) == STRSXP) {
   feature_str = Rcpp::as<std::vector<std::string>>(feature);
 } else {
   Rcpp::stop("feature must be a factor or character vector");
 }
 
 std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
 
 // Handle special case for features with two or fewer categories
 std::set<std::string> unique_categories(feature_str.begin(), feature_str.end());
 if (unique_categories.size() <= 2) {
   min_bins = max_bins = unique_categories.size();
 }
 
 OptimalBinningCategoricalIVB obcivb(feature_str, target_vec, bin_cutoff, min_bins, max_bins, max_n_prebins);
 return obcivb.fit();
}


// // [[Rcpp::plugins(cpp11)]]
// #include <Rcpp.h>
// #include <unordered_map>
// #include <map>
// #include <vector>
// #include <string>
// #include <algorithm>
// #include <numeric>
// 
// #ifdef _OPENMP
// #include <omp.h>
// #endif
// 
// using namespace Rcpp;
// 
// class OptimalBinningCategoricalIVB {
// private:
//   std::vector<std::string> feature;
//   std::vector<int> target;
//   double bin_cutoff;
//   int min_bins;
//   int max_bins;
//   int max_n_prebins;
// 
//   std::map<std::string, int> category_counts;
//   std::map<std::string, int> category_pos_counts;
//   std::map<std::string, int> category_neg_counts;
// 
//   std::vector<std::string> merged_bins;
//   std::vector<double> woe_values;
//   std::vector<double> iv_values;
// 
//   void validate_inputs();
//   void compute_category_stats();
//   void optimize_binning();
//   void compute_woe_iv();
// 
// public:
//   OptimalBinningCategoricalIVB(std::vector<std::string> feature,
//                                std::vector<int> target,
//                                double bin_cutoff,
//                                int min_bins,
//                                int max_bins,
//                                int max_n_prebins);
// 
//   Rcpp::List fit();
// };
// OptimalBinningCategoricalIVB::OptimalBinningCategoricalIVB(std::vector<std::string> feature,
//                                                            std::vector<int> target,
//                                                            double bin_cutoff,
//                                                            int min_bins,
//                                                            int max_bins,
//                                                            int max_n_prebins) {
//   this->feature = feature;
//   this->target = target;
//   this->bin_cutoff = bin_cutoff;
//   this->min_bins = min_bins;
//   this->max_bins = max_bins;
//   this->max_n_prebins = max_n_prebins;
//   validate_inputs();
// }
// 
// void OptimalBinningCategoricalIVB::validate_inputs() {
//   if (min_bins < 2) {
//     Rcpp::stop("min_bins must be at least 2.");
//   }
//   if (max_bins < min_bins) {
//     Rcpp::stop("max_bins must be greater than or equal to min_bins.");
//   }
//   if (bin_cutoff < 0 || bin_cutoff > 1) {
//     Rcpp::stop("bin_cutoff must be between 0 and 1.");
//   }
//   if (target.size() != feature.size()) {
//     Rcpp::stop("target and feature must have the same length.");
//   }
//   for (auto& t : target) {
//     if (t != 0 && t != 1) {
//       Rcpp::stop("target must be binary (0 or 1).");
//     }
//   }
// }
// 
// void OptimalBinningCategoricalIVB::compute_category_stats() {
//   size_t n = target.size();
//   for (size_t i = 0; i < n; ++i) {
//     category_counts[feature[i]]++;
//     if (target[i] == 1) {
//       category_pos_counts[feature[i]]++;
//     } else {
//       category_neg_counts[feature[i]]++;
//     }
//   }
// }
// 
// void OptimalBinningCategoricalIVB::optimize_binning() {
//   struct CategoryStats {
//     std::string category;
//     double event_rate;
//     int count;
//   };
// 
//   std::vector<CategoryStats> stats;
//   for (auto& pair : category_counts) {
//     double event_rate = static_cast<double>(category_pos_counts[pair.first]) / pair.second;
//     stats.push_back({pair.first, event_rate, pair.second});
//   }
// 
//   // Sort categories by event rate to ensure monotonicity
//   std::sort(stats.begin(), stats.end(), [](const CategoryStats& a, const CategoryStats& b) {
//     return a.event_rate < b.event_rate;
//   });
// 
//   // Respect max_n_prebins
//   if (static_cast<int>(stats.size()) > max_n_prebins) {
//     stats.resize(max_n_prebins);
//   }
// 
//   // Initialize bins
//   size_t n_bins = std::max(static_cast<size_t>(min_bins),
//                            std::min(static_cast<size_t>(max_bins), stats.size()));
//   size_t bin_size = stats.size() / n_bins;
//   size_t remainder = stats.size() % n_bins;
// 
//   merged_bins.clear();
//   size_t idx = 0;
//   for (size_t i = 0; i < n_bins; ++i) {
//     size_t current_bin_size = bin_size + (i < remainder ? 1 : 0);
//     std::vector<std::string> bin_categories;
//     for (size_t j = 0; j < current_bin_size; ++j) {
//       bin_categories.push_back(stats[idx++].category);
//     }
//     std::string bin_name = bin_categories[0];
//     for (size_t k = 1; k < bin_categories.size(); ++k) {
//       bin_name += "+" + bin_categories[k];
//     }
//     merged_bins.push_back(bin_name);
//   }
// }
// 
// void OptimalBinningCategoricalIVB::compute_woe_iv() {
//   int total_pos = std::accumulate(target.begin(), target.end(), 0);
//   int total_neg = target.size() - total_pos;
// 
//   woe_values.clear();
//   iv_values.clear();
// 
//   for (auto& bin : merged_bins) {
//     std::vector<std::string> categories;
//     std::stringstream ss(bin);
//     std::string item;
//     while (std::getline(ss, item, '+')) {
//       categories.push_back(item);
//     }
// 
//     int bin_pos = 0, bin_neg = 0;
//     for (auto& cat : categories) {
//       bin_pos += category_pos_counts[cat];
//       bin_neg += category_neg_counts[cat];
//     }
// 
//     double dist_pos = static_cast<double>(bin_pos) / total_pos;
//     double dist_neg = static_cast<double>(bin_neg) / total_neg;
// 
//     // Avoid division by zero
//     if (dist_pos == 0) dist_pos = 1e-6;
//     if (dist_neg == 0) dist_neg = 1e-6;
// 
//     double woe = std::log(dist_pos / dist_neg);
//     double iv = (dist_pos - dist_neg) * woe;
// 
//     woe_values.push_back(woe);
//     iv_values.push_back(iv);
//   }
// }
// 
// Rcpp::List OptimalBinningCategoricalIVB::fit() {
//   compute_category_stats();
//   optimize_binning();
//   compute_woe_iv();
// 
//   // Create woefeature vector
//   std::vector<double> woefeature(target.size());
//   std::unordered_map<std::string, double> category_to_woe;
// 
//   for (size_t i = 0; i < merged_bins.size(); ++i) {
//     std::vector<std::string> categories;
//     std::stringstream ss(merged_bins[i]);
//     std::string item;
//     while (std::getline(ss, item, '+')) {
//       category_to_woe[item] = woe_values[i];
//     }
//   }
// 
// #pragma omp parallel for
//   for (size_t i = 0; i < feature.size(); ++i) {
//     if (category_to_woe.find(feature[i]) != category_to_woe.end()) {
//       woefeature[i] = category_to_woe[feature[i]];
//     } else {
//       // For unseen categories, use the WOE of the bin with the closest event rate
//       double event_rate = static_cast<double>(category_pos_counts[feature[i]]) / category_counts[feature[i]];
//       auto it = std::lower_bound(merged_bins.begin(), merged_bins.end(), event_rate,
//                                  [this](const std::string& bin, double rate) {
//                                    std::vector<std::string> categories;
//                                    std::stringstream ss(bin);
//                                    std::string item;
//                                    while (std::getline(ss, item, '+')) {
//                                      categories.push_back(item);
//                                    }
//                                    double bin_pos = 0, bin_total = 0;
//                                    for (auto& cat : categories) {
//                                      bin_pos += category_pos_counts[cat];
//                                      bin_total += category_counts[cat];
//                                    }
//                                    return (static_cast<double>(bin_pos) / bin_total) < rate;
//                                  });
//       size_t index = std::distance(merged_bins.begin(), it);
//       if (index == merged_bins.size()) index--;
//       woefeature[i] = woe_values[index];
//     }
//   }
// 
//   // Create woebin DataFrame
//   Rcpp::DataFrame woebin = Rcpp::DataFrame::create(
//     Rcpp::Named("bin") = merged_bins,
//     Rcpp::Named("woe") = woe_values,
//     Rcpp::Named("iv") = iv_values
//   );
// 
//   return Rcpp::List::create(
//     Rcpp::Named("woefeature") = woefeature,
//     Rcpp::Named("woebin") = woebin
//   );
// }
// 
// 
// //' @title Categorical Optimal Binning with Information Value Binning
// //'
// //' @description
// //' Implements optimal binning for categorical variables using Information Value (IV)
// //' as the primary criterion, calculating Weight of Evidence (WoE) and IV for resulting bins.
// //'
// //' @param target Integer vector of binary target values (0 or 1).
// //' @param feature Character vector or factor of categorical feature values.
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff Minimum frequency for a separate bin (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins before merging (default: 20).
// //'
// //' @return A list with two elements:
// //' \itemize{
// //'   \item woefeature: Numeric vector of WoE values for each input feature value.
// //'   \item woebin: Data frame with binning results (bin names, WoE, IV).
// //' }
// //'
// //' @details
// //' The algorithm uses Information Value (IV) to create optimal bins for categorical variables.
// //' It starts by computing statistics for each category, then sorts categories by event rate
// //' to ensure monotonicity. The algorithm then creates initial bins based on the specified
// //' constraints and computes WoE and IV for each bin.
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
// //'   \item Compute category statistics (counts, positive counts, negative counts).
// //'   \item Sort categories by event rate to ensure monotonicity.
// //'   \item Create initial bins based on sorted categories and specified constraints.
// //'   \item Compute WoE and IV for each bin.
// //'   \item Assign WoE values to the original feature, handling unseen categories.
// //' }
// //'
// //' @examples
// //' \dontrun{
// //' # Sample data
// //' target <- c(1, 0, 1, 1, 0, 1, 0, 0, 1, 1)
// //' feature <- c("A", "B", "A", "C", "B", "D", "C", "A", "D", "B")
// //'
// //' # Run optimal binning
// //' result <- optimal_binning_categorical_ivb(target, feature, min_bins = 2, max_bins = 4)
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
// //'   \item Siddiqi, N. (2006). Credit risk scorecards: developing and implementing intelligent credit scoring. John Wiley & Sons.
// //'   \item Thomas, L. C. (2009). Consumer credit models: Pricing, profit and portfolios. OUP Oxford.
// //' }
// //'
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_categorical_ivb(Rcpp::IntegerVector target,
//                                            Rcpp::CharacterVector feature,
//                                            int min_bins = 3,
//                                            int max_bins = 5,
//                                            double bin_cutoff = 0.05,
//                                            int max_n_prebins = 20) {
//   std::vector<std::string> feature_str;
//   if (Rf_isFactor(feature)) {
//     Rcpp::IntegerVector levels = Rcpp::as<Rcpp::IntegerVector>(feature);
//     Rcpp::CharacterVector level_names = levels.attr("levels");
//     feature_str.reserve(levels.size());
//     for (int i = 0; i < levels.size(); ++i) {
//       feature_str.push_back(Rcpp::as<std::string>(level_names[levels[i] - 1]));
//     }
//   } else if (TYPEOF(feature) == STRSXP) {
//     feature_str = Rcpp::as<std::vector<std::string>>(feature);
//   } else {
//     Rcpp::stop("feature must be a factor or character vector");
//   }
// 
//   // Convertendo Rcpp::IntegerVector para std::vector<int>
//   std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
// 
//   OptimalBinningCategoricalIVB obcivb(feature_str, target_vec, bin_cutoff, min_bins, max_bins, max_n_prebins);
//   return obcivb.fit();
// }
// 
