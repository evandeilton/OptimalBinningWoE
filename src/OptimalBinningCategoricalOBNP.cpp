#include <Rcpp.h>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

class OptimalBinningCategoricalMILP {
private:
  std::vector<int> target;
  std::vector<std::string> feature;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  
  struct Bin {
    std::set<std::string> categories;
    int count_pos;
    int count_neg;
    double woe;
    double iv;
  };
  
  std::vector<Bin> bins;
  int total_pos;
  int total_neg;
  Rcpp::NumericVector woefeature;
  Rcpp::DataFrame woebin;
  
  void validate_inputs() const;
  double safe_log(double x) const;
  
public:
  OptimalBinningCategoricalMILP(
    const std::vector<int>& target,
    const std::vector<std::string>& feature,
    int min_bins,
    int max_bins,
    double bin_cutoff,
    int max_n_prebins
  );
  
  void fit();
  Rcpp::List get_results() const;
  
private:
  void initialize_bins();
  void enforce_monotonicity();
  void limit_bins();
  void merge_bins_by_minimum_delta_iv();
  void calculate_woe_iv(Bin& bin) const;
  void calculate_all_woe_iv();
};

OptimalBinningCategoricalMILP::OptimalBinningCategoricalMILP(
  const std::vector<int>& target,
  const std::vector<std::string>& feature,
  int min_bins,
  int max_bins,
  double bin_cutoff,
  int max_n_prebins
) : target(target), feature(feature), min_bins(min_bins), max_bins(max_bins),
bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins)
{
  validate_inputs();
}

void OptimalBinningCategoricalMILP::validate_inputs() const {
  if (target.size() != feature.size()) {
    throw std::invalid_argument("Target and feature vectors must have the same length.");
  }
  if (min_bins < 2) {
    throw std::invalid_argument("min_bins must be at least 2.");
  }
  if (max_bins < min_bins) {
    throw std::invalid_argument("max_bins must be greater than or equal to min_bins.");
  }
  if (bin_cutoff < 0.0 || bin_cutoff > 1.0) {
    throw std::invalid_argument("bin_cutoff must be between 0 and 1.");
  }
  if (max_n_prebins < 1) {
    throw std::invalid_argument("max_n_prebins must be at least 1.");
  }
}

double OptimalBinningCategoricalMILP::safe_log(double x) const {
  const double epsilon = std::numeric_limits<double>::min();
  return std::log(std::max(x, epsilon));
}

void OptimalBinningCategoricalMILP::initialize_bins() {
  std::map<std::string, Bin> bin_map;
  total_pos = 0;
  total_neg = 0;
  
  // Criar bins iniciais para cada categoria original
  for (size_t i = 0; i < target.size(); ++i) {
    const std::string& cat = feature[i];
    int tar = target[i];
    
    if (tar != 0 && tar != 1) {
      throw std::invalid_argument("Target variable must be binary (0 or 1).");
    }
    
    if (bin_map.find(cat) == bin_map.end()) {
      bin_map[cat] = Bin{{cat}, 0, 0, 0.0, 0.0};
    }
    
    if (tar == 1) {
      bin_map[cat].count_pos++;
      total_pos++;
    } else {
      bin_map[cat].count_neg++;
      total_neg++;
    }
  }
  
  bins.clear();
  bins.reserve(bin_map.size());
  for (const auto& kv : bin_map) {
    bins.push_back(kv.second);
  }
  
  // Calcular WoE e IV iniciais
  calculate_all_woe_iv();
}

void OptimalBinningCategoricalMILP::calculate_woe_iv(Bin& bin) const {
  double dist_pos = static_cast<double>(bin.count_pos) / total_pos;
  double dist_neg = static_cast<double>(bin.count_neg) / total_neg;
  bin.woe = safe_log(dist_pos) - safe_log(dist_neg);
  bin.iv = (dist_pos - dist_neg) * bin.woe;
}

void OptimalBinningCategoricalMILP::calculate_all_woe_iv() {
#pragma omp parallel for
  for (size_t i = 0; i < bins.size(); ++i) {
    calculate_woe_iv(bins[i]);
  }
}

void OptimalBinningCategoricalMILP::enforce_monotonicity() {
  if (bins.size() <= 2) {
    // Não aplica monotonicidade para variáveis com duas ou menos categorias
    return;
  }
  
  // Determinar a tendência predominante
  int positive_diff = 0;
  int negative_diff = 0;
  for (size_t i = 0; i < bins.size() - 1; ++i) {
    double diff = bins[i + 1].woe - bins[i].woe;
    if (diff > 0) {
      positive_diff++;
    } else if (diff < 0) {
      negative_diff++;
    }
  }
  
  bool is_increasing = false;
  if (positive_diff > negative_diff) {
    is_increasing = true;
  } else {
    is_increasing = false;
  }
  
  bool is_monotonic = false;
  
  while (!is_monotonic && bins.size() > static_cast<size_t>(min_bins)) {
    is_monotonic = true;
    for (size_t i = 0; i < bins.size() - 1; ++i) {
      if ((is_increasing && bins[i + 1].woe < bins[i].woe) ||
          (!is_increasing && bins[i + 1].woe > bins[i].woe)) {
        // Mesclar bins que violam a monotonicidade
        bins[i].categories.insert(bins[i + 1].categories.begin(), bins[i + 1].categories.end());
        bins[i].count_pos += bins[i + 1].count_pos;
        bins[i].count_neg += bins[i + 1].count_neg;
        bins.erase(bins.begin() + i + 1);
        
        calculate_all_woe_iv();
        
        is_monotonic = false;
        break;
      }
    }
    
    // Permite desvio da monotonicidade se min_bins for atingido
    if (bins.size() <= static_cast<size_t>(min_bins)) {
      break;
    }
  }
}

void OptimalBinningCategoricalMILP::limit_bins() {
  // Mesclar bins para atender ao max_bins
  while (bins.size() > static_cast<size_t>(max_bins)) {
    merge_bins_by_minimum_delta_iv();
  }
  
  // Garante pelo menos min_bins
  while (bins.size() < static_cast<size_t>(min_bins) && bins.size() > 1) {
    merge_bins_by_minimum_delta_iv();
  }
}

void OptimalBinningCategoricalMILP::merge_bins_by_minimum_delta_iv() {
  if (bins.size() <= 1) return;
  
  double min_delta_iv = std::numeric_limits<double>::max();
  size_t merge_idx1 = 0;
  
  for (size_t i = 0; i < bins.size() - 1; ++i) {
    double iv_before = bins[i].iv + bins[i + 1].iv;
    
    Bin merged_bin;
    merged_bin.categories.insert(bins[i].categories.begin(), bins[i].categories.end());
    merged_bin.categories.insert(bins[i + 1].categories.begin(), bins[i + 1].categories.end());
    merged_bin.count_pos = bins[i].count_pos + bins[i + 1].count_pos;
    merged_bin.count_neg = bins[i].count_neg + bins[i + 1].count_neg;
    calculate_woe_iv(merged_bin);
    
    double delta_iv = iv_before - merged_bin.iv;
    
    if (delta_iv < min_delta_iv) {
      min_delta_iv = delta_iv;
      merge_idx1 = i;
    }
  }
  
  // Mesclar bins[merge_idx1] e bins[merge_idx1 + 1]
  bins[merge_idx1].categories.insert(bins[merge_idx1 + 1].categories.begin(), bins[merge_idx1 + 1].categories.end());
  bins[merge_idx1].count_pos += bins[merge_idx1 + 1].count_pos;
  bins[merge_idx1].count_neg += bins[merge_idx1 + 1].count_neg;
  
  bins.erase(bins.begin() + merge_idx1 + 1);
  
  calculate_all_woe_iv();
}

void OptimalBinningCategoricalMILP::fit() {
  initialize_bins();
  enforce_monotonicity();
  limit_bins();
  calculate_all_woe_iv();
  
  std::map<std::string, double> category_to_woe;
  for (const auto& bin : bins) {
    for (const auto& cat : bin.categories) {
      category_to_woe[cat] = bin.woe;
    }
  }
  
  woefeature = Rcpp::NumericVector(feature.size());
#pragma omp parallel for
  for (size_t i = 0; i < feature.size(); ++i) {
    const std::string& cat = feature[i];
    auto it = category_to_woe.find(cat);
    if (it != category_to_woe.end()) {
      woefeature[i] = it->second;
    } else {
      // Se a categoria não for encontrada (o que não deve ocorrer), atribui 0.0
      woefeature[i] = 0.0;
    }
  }
  
  size_t num_bins = bins.size();
  Rcpp::CharacterVector bin_vec(num_bins);
  Rcpp::NumericVector woe_vec(num_bins);
  Rcpp::NumericVector iv_vec(num_bins);
  Rcpp::IntegerVector count_vec(num_bins);
  Rcpp::IntegerVector count_pos_vec(num_bins);
  Rcpp::IntegerVector count_neg_vec(num_bins);
  
  for (size_t i = 0; i < num_bins; ++i) {
    const Bin& bin = bins[i];
    std::vector<std::string> sorted_categories(bin.categories.begin(), bin.categories.end());
    std::sort(sorted_categories.begin(), sorted_categories.end());
    
    std::string bin_name = sorted_categories[0];
    for (size_t j = 1; j < sorted_categories.size(); ++j) {
      bin_name += "%;%" + sorted_categories[j];
    }
    
    bin_vec[i] = bin_name;
    woe_vec[i] = bin.woe;
    iv_vec[i] = bin.iv;
    count_vec[i] = bin.count_pos + bin.count_neg;
    count_pos_vec[i] = bin.count_pos;
    count_neg_vec[i] = bin.count_neg;
  }
  
  woebin = Rcpp::DataFrame::create(
    Rcpp::Named("bin") = bin_vec,
    Rcpp::Named("woe") = woe_vec,
    Rcpp::Named("iv") = iv_vec,
    Rcpp::Named("count") = count_vec,
    Rcpp::Named("count_pos") = count_pos_vec,
    Rcpp::Named("count_neg") = count_neg_vec
  );
}

Rcpp::List OptimalBinningCategoricalMILP::get_results() const {
  return Rcpp::List::create(
    Rcpp::Named("woefeature") = woefeature,
    Rcpp::Named("woebin") = woebin
  );
}

//' @title Optimal Binning for Categorical Variables using OBNP
//'
//' @description This function performs optimal binning for categorical variables using the Optimal Binning Numerical Procedures (OBNP) approach.
//' The process aims to maximize the Information Value (IV) while maintaining a specified number of bins.
//'
//' @param target An integer vector of binary target values (0 or 1).
//' @param feature A character vector of categorical feature values.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum proportion of observations in a bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins (default: 20).
//'
//' @return A list containing two elements:
//' \itemize{
//'   \item woefeature: A numeric vector of Weight of Evidence (WoE) values for each observation
//'   \item woebin: A data frame containing binning information, including bin names, WoE, Information Value (IV), and counts
//' }
//'
//' @details
//' The algorithm works as follows:
//' 1. Merge rare categories: Categories with fewer observations than the specified bin_cutoff are merged into an "Other" category.
//' 2. Create initial bins: Each unique category is assigned to its own bin, up to max_n_prebins.
//' 3. Optimize bins:
//'    a. Calculate WoE and IV for each bin.
//'    b. Enforce monotonicity when possible, merging bins as needed, unless min_bins is reached.
//'    c. Limit the number of bins to be within min_bins and max_bins.
//' 4. Transform the feature: Assign WoE values to each observation based on its category.
//'
//' The Weight of Evidence (WoE) is calculated as:
//' \deqn{WoE = \ln\left(\frac{\text{% of events}}{\text{% of non-events}}\right)}
//'
//' The Information Value (IV) is calculated as:
//' \deqn{IV = (\text{% of events} - \text{% of non-events}) \times WoE}
//'
//' @examples
//' \dontrun{
//' # Create sample data
//' target <- sample(0:1, 1000, replace = TRUE)
//' feature <- sample(LETTERS[1:5], 1000, replace = TRUE)
//'
//' # Run optimal binning
//' result <- optimal_binning_categorical_obnp(target, feature)
//'
//' # View results
//' print(result$woebin)
//' }
//'
//' @references
//' \itemize{
//'    \item Belotti, T., Crook, J. (2009). Credit Scoring with Macroeconomic Variables Using Survival Analysis.
//'          Journal of the Operational Research Society, 60(12), 1699-1707.
//'    \item Thomas, L. C. (2000). A survey of credit and behavioural scoring: forecasting financial risk of lending to consumers.
//'          International Journal of Forecasting, 16(2), 149-172.
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_milp(
    Rcpp::IntegerVector target,
    Rcpp::CharacterVector feature,
    int min_bins = 3,
    int max_bins = 5,
    double bin_cutoff = 0.05,
    int max_n_prebins = 20
) {
  try {
    std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
    std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
    
    OptimalBinningCategoricalMILP obcm(
        target_vec,
        feature_vec,
        min_bins,
        max_bins,
        bin_cutoff,
        max_n_prebins
    );
    
    obcm.fit();
    
    return obcm.get_results();
  } catch (const std::exception& e) {
    Rcpp::stop("Error in optimal_binning_categorical_milp: " + std::string(e.what()));
  }
}




// #include <Rcpp.h>
// #include <vector>
// #include <string>
// #include <algorithm>
// #include <unordered_map>
// #include <cmath>
// 
// #ifdef _OPENMP
// #include <omp.h>
// #endif
// 
// class OptimalBinningCategoricalOBNP {
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
//     int count;
//     int count_pos;
//     int count_neg;
//     double woe;
//     double iv;
//   };
// 
//   std::vector<BinInfo> bins;
// 
//   void validate_inputs() {
//     if (feature.size() != target.size()) {
//       Rcpp::stop("Feature and target must have the same length");
//     }
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
//   }
// 
//   void merge_rare_categories() {
//     std::unordered_map<std::string, int> category_counts;
//     int total_count = feature.size();
// 
//     // Count occurrences of each category
//     for (const auto& cat : feature) {
//       category_counts[cat]++;
//     }
// 
//     // Identify rare categories
//     std::vector<std::string> rare_categories;
//     for (const auto& pair : category_counts) {
//       if (static_cast<double>(pair.second) / total_count < bin_cutoff) {
//         rare_categories.push_back(pair.first);
//       }
//     }
// 
//     // Merge rare categories
//     std::string merged_category = "Other";
//     for (auto& cat : feature) {
//       if (std::find(rare_categories.begin(), rare_categories.end(), cat) != rare_categories.end()) {
//         cat = merged_category;
//       }
//     }
//   }
// 
//   void create_initial_bins() {
//     std::unordered_map<std::string, BinInfo> bin_map;
// 
// #pragma omp parallel for
//     for (size_t i = 0; i < feature.size(); ++i) {
//       const std::string& cat = feature[i];
//       int t = target[i];
// 
// #pragma omp critical
// {
//   if (bin_map.find(cat) == bin_map.end()) {
//     bin_map[cat] = BinInfo{{cat}, 1, t, 1 - t, 0.0, 0.0};
//   } else {
//     bin_map[cat].count++;
//     bin_map[cat].count_pos += t;
//     bin_map[cat].count_neg += 1 - t;
//   }
// }
//     }
// 
//     bins.clear();
//     for (const auto& pair : bin_map) {
//       bins.push_back(pair.second);
//     }
// 
//     // Sort bins by count_pos / count ratio (descending)
//     std::sort(bins.begin(), bins.end(), [](const BinInfo& a, const BinInfo& b) {
//       return static_cast<double>(a.count_pos) / a.count > static_cast<double>(b.count_pos) / b.count;
//     });
// 
//     // Limit to max_n_prebins
//     if (bins.size() > static_cast<size_t>(max_n_prebins)) {
//       bins.resize(max_n_prebins);
//     }
//   }
// 
//   void optimize_bins() {
//     while (bins.size() > static_cast<size_t>(min_bins) && bins.size() > static_cast<size_t>(max_bins)) {
//       merge_least_significant_bins();
//     }
// 
//     calculate_woe_and_iv();
//   }
// 
//   void merge_least_significant_bins() {
//     auto min_iv_it = std::min_element(bins.begin(), bins.end(),
//                                       [](const BinInfo& a, const BinInfo& b) { return a.iv < b.iv; });
// 
//     if (min_iv_it != bins.end() && std::next(min_iv_it) != bins.end()) {
//       auto next_bin = std::next(min_iv_it);
//       min_iv_it->categories.insert(min_iv_it->categories.end(),
//                                    next_bin->categories.begin(), next_bin->categories.end());
//       min_iv_it->count += next_bin->count;
//       min_iv_it->count_pos += next_bin->count_pos;
//       min_iv_it->count_neg += next_bin->count_neg;
//       bins.erase(next_bin);
//     }
//   }
// 
//   void calculate_woe_and_iv() {
//     int total_pos = 0, total_neg = 0;
//     for (const auto& bin : bins) {
//       total_pos += bin.count_pos;
//       total_neg += bin.count_neg;
//     }
// 
//     double total_iv = 0.0;
//     for (auto& bin : bins) {
//       double pos_rate = static_cast<double>(bin.count_pos) / total_pos;
//       double neg_rate = static_cast<double>(bin.count_neg) / total_neg;
//       bin.woe = std::log(pos_rate / neg_rate);
//       bin.iv = (pos_rate - neg_rate) * bin.woe;
//       total_iv += bin.iv;
//     }
//   }
// 
// public:
//   OptimalBinningCategoricalOBNP(const std::vector<std::string>& feature,
//                                 const std::vector<int>& target,
//                                 int min_bins = 3,
//                                 int max_bins = 5,
//                                 double bin_cutoff = 0.05,
//                                 int max_n_prebins = 20)
//     : feature(feature), target(target), min_bins(min_bins), max_bins(max_bins),
//       bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins) {
//     validate_inputs();
//   }
// 
//   void fit() {
//     merge_rare_categories();
//     create_initial_bins();
//     optimize_bins();
//   }
// 
//   Rcpp::List get_results() {
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
//       bin_names.push_back(bin_name);
//       woe_values.push_back(bin.woe);
//       iv_values.push_back(bin.iv);
//       count_values.push_back(bin.count);
//       count_pos_values.push_back(bin.count_pos);
//       count_neg_values.push_back(bin.count_neg);
//     }
// 
//     return Rcpp::DataFrame::create(
//       Rcpp::Named("bin") = bin_names,
//       Rcpp::Named("woe") = woe_values,
//       Rcpp::Named("iv") = iv_values,
//       Rcpp::Named("count") = count_values,
//       Rcpp::Named("count_pos") = count_pos_values,
//       Rcpp::Named("count_neg") = count_neg_values
//     );
//   }
// 
//   std::vector<double> transform(const std::vector<std::string>& new_feature) {
//     std::vector<double> woe_feature(new_feature.size());
// 
// #pragma omp parallel for
//     for (size_t i = 0; i < new_feature.size(); ++i) {
//       const std::string& cat = new_feature[i];
//       auto it = std::find_if(bins.begin(), bins.end(), [&cat](const BinInfo& bin) {
//         return std::find(bin.categories.begin(), bin.categories.end(), cat) != bin.categories.end();
//       });
// 
//       if (it != bins.end()) {
//         woe_feature[i] = it->woe;
//       } else {
//         // Assign the WoE of the last bin (typically for unseen categories)
//         woe_feature[i] = bins.back().woe;
//       }
//     }
// 
//     return woe_feature;
//   }
// };
// 
// //' @title Optimal Binning for Categorical Variables using OBNP
// //'
// //' @description This function performs optimal binning for categorical variables using the Optimal Binning Numerical Procedures (OBNP) approach.
// //' The process aims to maximize the Information Value (IV) while maintaining a specified number of bins.
// //'
// //' @param target An integer vector of binary target values (0 or 1).
// //' @param feature A character vector of categorical feature values.
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff Minimum proportion of observations in a bin (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins (default: 20).
// //'
// //' @return A list containing two elements:
// //' \itemize{
// //'   \item woefeature: A numeric vector of Weight of Evidence (WoE) values for each observation
// //'   \item woebin: A data frame containing binning information, including bin names, WoE, Information Value (IV), and counts
// //' }
// //'
// //' @details
// //' The algorithm works as follows:
// //' 1. Merge rare categories: Categories with fewer observations than the specified bin_cutoff are merged into an "Other" category.
// //' 2. Create initial bins: Each unique category is assigned to its own bin, up to max_n_prebins.
// //' 3. Optimize bins:
// //'    a. While the number of bins exceeds max_bins, merge the two bins with the lowest IV.
// //'    b. Calculate WoE and IV for each bin.
// //' 4. Transform the feature: Assign WoE values to each observation based on its category.
// //'
// //' The Weight of Evidence (WoE) is calculated as:
// //' \deqn{WoE = \ln\left(\frac{\text{% of events}}{\text{% of non-events}}\right)}
// //'
// //' The Information Value (IV) is calculated as:
// //' \deqn{IV = (\text{% of events} - \text{% of non-events}) \times WoE}
// //'
// //' The algorithm uses OpenMP for parallel processing to improve performance.
// //'
// //' @examples
// //' \dontrun{
// //' # Create sample data
// //' target <- sample(0:1, 1000, replace = TRUE)
// //' feature <- sample(LETTERS[1:5], 1000, replace = TRUE)
// //'
// //' # Run optimal binning
// //' result <- optimal_binning_categorical_obnp(target, feature)
// //'
// //' # View results
// //' print(result$woebin)
// //' }
// //'
// //' @references
// //' \itemize{
// //'    \item Belotti, T., Crook, J. (2009). Credit Scoring with Macroeconomic Variables Using Survival Analysis. 
// //'          Journal of the Operational Research Society, 60(12), 1699-1707.
// //'    \item Thomas, L. C. (2000). A survey of credit and behavioural scoring: forecasting financial risk of lending to consumers. 
// //'          International Journal of Forecasting, 16(2), 149-172.
// //' }
// //'
// //' @author Lopes, J. E.
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_categorical_obnp(Rcpp::IntegerVector target,
//                                             Rcpp::CharacterVector feature,
//                                             int min_bins = 3,
//                                             int max_bins = 5,
//                                             double bin_cutoff = 0.05,
//                                             int max_n_prebins = 20) {
//   std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
//   std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
// 
//   OptimalBinningCategoricalOBNP binner(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins);
//   binner.fit();
//   Rcpp::List woebin = binner.get_results();
//   std::vector<double> woefeature = binner.transform(feature_vec);
// 
//   return Rcpp::List::create(
//     Rcpp::Named("woefeature") = woefeature,
//     Rcpp::Named("woebin") = woebin
//   );
// }
