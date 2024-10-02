#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_map>

#ifdef _OPENMP
#include <omp.h>
#endif

// [[Rcpp::plugins(openmp)]]

class OptimalBinningNumericalCM {
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
  
  static constexpr double EPSILON = 1e-10;
  
  double calculate_chi_square(const Bin& bin1, const Bin& bin2) const {
    int total_pos = bin1.count_pos + bin2.count_pos;
    int total_neg = bin1.count_neg + bin2.count_neg;
    int total = total_pos + total_neg;
    
    double expected_pos1 = std::max(static_cast<double>(bin1.count * total_pos) / total, EPSILON);
    double expected_neg1 = std::max(static_cast<double>(bin1.count * total_neg) / total, EPSILON);
    double expected_pos2 = std::max(static_cast<double>(bin2.count * total_pos) / total, EPSILON);
    double expected_neg2 = std::max(static_cast<double>(bin2.count * total_neg) / total, EPSILON);
    
    double chi_square =
      std::pow(bin1.count_pos - expected_pos1, 2) / expected_pos1 +
      std::pow(bin1.count_neg - expected_neg1, 2) / expected_neg1 +
      std::pow(bin2.count_pos - expected_pos2, 2) / expected_pos2 +
      std::pow(bin2.count_neg - expected_neg2, 2) / expected_neg2;
    
    return chi_square;
  }
  
  void merge_bins(size_t index) {
    Bin& left = bins[index];
    Bin& right = bins[index + 1];
    
    left.upper_bound = right.upper_bound;
    left.count += right.count;
    left.count_pos += right.count_pos;
    left.count_neg += right.count_neg;
    
    bins.erase(bins.begin() + index + 1);
  }
  
  void calculate_woe_iv() {
    double total_pos = 0, total_neg = 0;
    for (const auto& bin : bins) {
      total_pos += bin.count_pos;
      total_neg += bin.count_neg;
    }
    
    for (auto& bin : bins) {
      double pos_rate = std::max(static_cast<double>(bin.count_pos) / total_pos, EPSILON);
      double neg_rate = std::max(static_cast<double>(bin.count_neg) / total_neg, EPSILON);
      
      bin.woe = std::log(pos_rate / neg_rate);
      bin.iv = (pos_rate - neg_rate) * bin.woe;
    }
  }
  
  bool is_monotonic() const {
    if (bins.size() <= 2) return true;
    
    bool increasing = bins[1].woe > bins[0].woe;
    for (size_t i = 2; i < bins.size(); ++i) {
      if ((increasing && bins[i].woe < bins[i-1].woe) ||
          (!increasing && bins[i].woe > bins[i-1].woe)) {
        return false;
      }
    }
    return true;
  }
  
public:
  OptimalBinningNumericalCM(
    const std::vector<double>& feature,
    const std::vector<int>& target,
    int min_bins = 3,
    int max_bins = 5,
    double bin_cutoff = 0.05,
    int max_n_prebins = 20
  ) : feature(feature), target(target), min_bins(min_bins), max_bins(max_bins),
  bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins) {
    
    if (feature.size() != target.size()) {
      Rcpp::stop("Feature and target vectors must have the same length");
    }
    
    if (min_bins < 2 || max_bins < min_bins) {
      Rcpp::stop("Invalid bin constraints");
    }
    
    // Adjust min_bins if feature has fewer unique values
    std::unordered_map<double, int> unique_values;
    for (const auto& val : feature) {
      unique_values[val]++;
    }
    min_bins = std::min(min_bins, static_cast<int>(unique_values.size()));
    max_bins = std::min(max_bins, static_cast<int>(unique_values.size()));
  }
  
  void fit() {
    // Initial binning
    std::vector<std::pair<double, int>> sorted_data(feature.size());
    for (size_t i = 0; i < feature.size(); ++i) {
      sorted_data[i] = {feature[i], target[i]};
    }
    std::sort(sorted_data.begin(), sorted_data.end());
    
    // Create initial bins
    int records_per_bin = std::max(1, static_cast<int>(sorted_data.size()) / max_n_prebins);
    bins.reserve(max_n_prebins);
    
    for (size_t i = 0; i < sorted_data.size(); i += records_per_bin) {
      size_t end = std::min(i + records_per_bin, sorted_data.size());
      Bin bin;
      bin.lower_bound = (i == 0) ? -std::numeric_limits<double>::infinity() : sorted_data[i].first;
      bin.upper_bound = (end == sorted_data.size()) ? std::numeric_limits<double>::infinity() : sorted_data[end - 1].first;
      bin.count = end - i;
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
    
    // ChiMerge algorithm
    while (bins.size() > min_bins) {
      double min_chi_square = std::numeric_limits<double>::max();
      size_t merge_index = 0;
      
#pragma omp parallel
{
  double local_min_chi_square = std::numeric_limits<double>::max();
  size_t local_merge_index = 0;
  
#pragma omp for nowait
  for (size_t i = 0; i < bins.size() - 1; ++i) {
    double chi_square = calculate_chi_square(bins[i], bins[i + 1]);
    if (chi_square < local_min_chi_square) {
      local_min_chi_square = chi_square;
      local_merge_index = i;
    }
  }
  
#pragma omp critical
{
  if (local_min_chi_square < min_chi_square) {
    min_chi_square = local_min_chi_square;
    merge_index = local_merge_index;
  }
}
}

merge_bins(merge_index);

if (bins.size() <= max_bins) {
  break;
}
    }
    
    // Merge rare bins
    double total_count = static_cast<double>(feature.size());
    for (auto it = bins.begin(); it != bins.end(); ) {
      if (static_cast<double>(it->count) / total_count < bin_cutoff) {
        if (it == bins.begin()) {
          merge_bins(0);
        } else {
          merge_bins(std::distance(bins.begin(), it) - 1);
        }
      } else {
        ++it;
      }
    }
    
    calculate_woe_iv();
    
    // Ensure monotonicity if possible
    if (!is_monotonic() && bins.size() > min_bins) {
      while (!is_monotonic() && bins.size() > min_bins) {
        double min_chi_square = std::numeric_limits<double>::max();
        size_t merge_index = 0;
        
        for (size_t i = 0; i < bins.size() - 1; ++i) {
          double chi_square = calculate_chi_square(bins[i], bins[i + 1]);
          if (chi_square < min_chi_square) {
            min_chi_square = chi_square;
            merge_index = i;
          }
        }
        
        merge_bins(merge_index);
        calculate_woe_iv();
      }
    }
  }
  
  Rcpp::List get_results() const {
    Rcpp::NumericVector woefeature(feature.size());
    Rcpp::DataFrame woebin;
    Rcpp::StringVector bin_labels;
    Rcpp::NumericVector woe_values;
    Rcpp::NumericVector iv_values;
    Rcpp::IntegerVector count_values;
    Rcpp::IntegerVector count_pos_values;
    Rcpp::IntegerVector count_neg_values;
    
#pragma omp parallel for
    for (size_t i = 0; i < feature.size(); ++i) {
      for (const auto& bin : bins) {
        if (feature[i] <= bin.upper_bound) {
          woefeature[i] = bin.woe;
          break;
        }
      }
    }
    
    for (const auto& bin : bins) {
      std::string bin_label = (bin.lower_bound == -std::numeric_limits<double>::infinity() ? "(-Inf" : "(" + std::to_string(bin.lower_bound)) +
        ";" + (bin.upper_bound == std::numeric_limits<double>::infinity() ? "+Inf]" : std::to_string(bin.upper_bound) + "]");
      bin_labels.push_back(bin_label);
      woe_values.push_back(bin.woe);
      iv_values.push_back(bin.iv);
      count_values.push_back(bin.count);
      count_pos_values.push_back(bin.count_pos);
      count_neg_values.push_back(bin.count_neg);
    }
    
    woebin = Rcpp::DataFrame::create(
      Rcpp::Named("bin") = bin_labels,
      Rcpp::Named("woe") = woe_values,
      Rcpp::Named("iv") = iv_values,
      Rcpp::Named("count") = count_values,
      Rcpp::Named("count_pos") = count_pos_values,
      Rcpp::Named("count_neg") = count_neg_values
    );
    
    return Rcpp::List::create(
      Rcpp::Named("woefeature") = woefeature,
      Rcpp::Named("woebin") = woebin
    );
  }
};

//' @title Optimal Binning for Numerical Variables using ChiMerge
//'
//' @description
//' This function implements an optimal binning algorithm for numerical variables using the ChiMerge approach with Weight of Evidence (WoE) and Information Value (IV) criteria.
//'
//' @param target An integer vector of binary target values (0 or 1).
//' @param feature A numeric vector of feature values to be binned.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum frequency of observations in each bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins for initial discretization (default: 20).
//'
//' @return A list containing two elements:
//' \item{woefeature}{A numeric vector of WoE-transformed feature values.}
//' \item{woebin}{A data frame with binning details, including bin boundaries, WoE, IV, and count statistics.}
//'
//' @examples
//' \dontrun{
//' # Generate sample data
//' set.seed(123)
//' n <- 10000
//' feature <- rnorm(n)
//' target <- rbinom(n, 1, plogis(0.5 * feature))
//'
//' # Apply optimal binning
//' result <- optimal_binning_numerical_cm(target, feature, min_bins = 3, max_bins = 5)
//'
//' # View binning results
//' print(result$woebin)
//'
//' # Plot WoE transformation
//' plot(feature, result$woefeature, main = "WoE Transformation",
//' xlab = "Original Feature", ylab = "WoE")
//' }
//'
//' @details
//' The optimal binning algorithm for numerical variables uses the ChiMerge approach with Weight of Evidence (WoE) and Information Value (IV) to create bins that maximize the predictive power of the feature while maintaining interpretability.
//'
//' The algorithm follows these steps:
//' 1. Initial discretization into max_n_prebins
//' 2. Iterative merging of adjacent bins based on chi-square statistic
//' 3. Merging of rare bins based on the bin_cutoff parameter
//' 4. Calculation of WoE and IV for each final bin
//'
//' The chi-square statistic for two adjacent bins is calculated as:
//'
//' \deqn{\chi^2 = \sum_{i=1}^{2} \sum_{j=1}^{2} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}}
//'
//' where \eqn{O_{ij}} is the observed frequency and \eqn{E_{ij}} is the expected frequency for bin i and class j.
//'
//' Weight of Evidence (WoE) is calculated for each bin as:
//'
//' \deqn{WoE_i = \ln\left(\frac{P(X_i|Y=1)}{P(X_i|Y=0)}\right)}
//'
//' where \eqn{P(X_i|Y=1)} is the proportion of positive cases in bin i, and \eqn{P(X_i|Y=0)} is the proportion of negative cases in bin i.
//'
//' Information Value (IV) for each bin is calculated as:
//'
//' \deqn{IV_i = (P(X_i|Y=1) - P(X_i|Y=0)) \times WoE_i}
//'
//' The total IV for the feature is the sum of IVs across all bins:
//'
//' \deqn{IV_{total} = \sum_{i=1}^{n} IV_i}
//'
//' The ChiMerge approach ensures that the resulting binning maximizes the separation between classes while maintaining the desired number of bins and respecting the minimum bin frequency constraint.
//'
//' @references
//' \itemize{
//'   \item Kerber, R. (1992). ChiMerge: Discretization of Numeric Attributes. In Proceedings of the tenth national conference on Artificial intelligence (pp. 123-128). AAAI Press.
//'   \item Zeng, G. (2014). A necessary condition for a good binning algorithm in credit scoring. Applied Mathematical Sciences, 8(65), 3229-3242.
//' }
//'
//' @author Lopes, J. E.
//'
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_cm(
   Rcpp::IntegerVector target,
   Rcpp::NumericVector feature,
   int min_bins = 3,
   int max_bins = 5,
   double bin_cutoff = 0.05,
   int max_n_prebins = 20
) {
 // Input validation
 if (target.size() != feature.size()) {
   Rcpp::stop("Target and feature vectors must have the same length");
 }
 
 if (min_bins < 2 || max_bins < min_bins) {
   Rcpp::stop("Invalid bin constraints: min_bins must be at least 2 and max_bins must be greater than or equal to min_bins");
 }
 
 if (bin_cutoff <= 0 || bin_cutoff >= 1) {
   Rcpp::stop("bin_cutoff must be between 0 and 1");
 }
 
 if (max_n_prebins < max_bins) {
   Rcpp::stop("max_n_prebins must be greater than or equal to max_bins");
 }
 
 // Convert Rcpp vectors to std::vector
 std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
 std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
 
 // Create and run the binner
 OptimalBinningNumericalCM binner(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins);
 binner.fit();
 
 // Get and return the results
 return binner.get_results();
}



// #include <Rcpp.h>
// #include <vector>
// #include <algorithm>
// #include <cmath>
// #include <limits>
// #include <string>
// 
// #ifdef _OPENMP
// #include <omp.h>
// #endif
// 
// // [[Rcpp::plugins(openmp)]]
// 
// class OptimalBinningNumericalCM {
// private:
// std::vector<double> feature;
// std::vector<int> target;
// int min_bins;
// int max_bins;
// double bin_cutoff;
// int max_n_prebins;
// 
// struct Bin {
//   double lower_bound;
//   double upper_bound;
//   int count;
//   int count_pos;
//   int count_neg;
//   double woe;
//   double iv;
// };
// 
// std::vector<Bin> bins;
// 
// static constexpr double EPSILON = 1e-10;
// 
// double calculate_chi_square(const Bin& bin1, const Bin& bin2) const {
//   int total_pos = bin1.count_pos + bin2.count_pos;
//   int total_neg = bin1.count_neg + bin2.count_neg;
//   int total = total_pos + total_neg;
//   
//   double expected_pos1 = static_cast<double>(bin1.count * total_pos) / total;
//   double expected_neg1 = static_cast<double>(bin1.count * total_neg) / total;
//   double expected_pos2 = static_cast<double>(bin2.count * total_pos) / total;
//   double expected_neg2 = static_cast<double>(bin2.count * total_neg) / total;
//   
//   // Avoid division by zero
//   expected_pos1 = std::max(expected_pos1, EPSILON);
//   expected_neg1 = std::max(expected_neg1, EPSILON);
//   expected_pos2 = std::max(expected_pos2, EPSILON);
//   expected_neg2 = std::max(expected_neg2, EPSILON);
//   
//   double chi_square =
//     std::pow(bin1.count_pos - expected_pos1, 2) / expected_pos1 +
//     std::pow(bin1.count_neg - expected_neg1, 2) / expected_neg1 +
//     std::pow(bin2.count_pos - expected_pos2, 2) / expected_pos2 +
//     std::pow(bin2.count_neg - expected_neg2, 2) / expected_neg2;
//   
//   return chi_square;
// }
// 
// void merge_bins(size_t index) {
//   Bin& left = bins[index];
//   Bin& right = bins[index + 1];
//   
//   left.upper_bound = right.upper_bound;
//   left.count += right.count;
//   left.count_pos += right.count_pos;
//   left.count_neg += right.count_neg;
//   
//   bins.erase(bins.begin() + index + 1);
// }
// 
// void calculate_woe_iv() {
//   double total_pos = 0, total_neg = 0;
//   for (const auto& bin : bins) {
//     total_pos += bin.count_pos;
//     total_neg += bin.count_neg;
//   }
//   
//   for (auto& bin : bins) {
//     double pos_rate = static_cast<double>(bin.count_pos) / total_pos;
//     double neg_rate = static_cast<double>(bin.count_neg) / total_neg;
//     
//     // Avoid division by zero and log of zero
//     pos_rate = std::max(pos_rate, EPSILON);
//     neg_rate = std::max(neg_rate, EPSILON);
//     
//     bin.woe = std::log(pos_rate / neg_rate);
//     bin.iv = (pos_rate - neg_rate) * bin.woe;
//   }
// }
// 
// public:
// OptimalBinningNumericalCM(
//   const std::vector<double>& feature,
//   const std::vector<int>& target,
//   int min_bins = 3,
//   int max_bins = 5,
//   double bin_cutoff = 0.05,
//   int max_n_prebins = 20
// ) : feature(feature), target(target), min_bins(min_bins), max_bins(max_bins),
// bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins) {
//   
//   if (feature.size() != target.size()) {
//     Rcpp::stop("Feature and target vectors must have the same length");
//   }
//   
//   if (min_bins < 2 || max_bins < min_bins) {
//     Rcpp::stop("Invalid bin constraints");
//   }
// }
// 
// void fit() {
//   // Initial binning
//   std::vector<std::pair<double, int>> sorted_data(feature.size());
//   for (size_t i = 0; i < feature.size(); ++i) {
//     sorted_data[i] = {feature[i], target[i]};
//   }
//   std::sort(sorted_data.begin(), sorted_data.end());
//   
//   // Create initial bins
//   int records_per_bin = std::max(1, static_cast<int>(sorted_data.size()) / max_n_prebins);
//   bins.reserve(max_n_prebins);
//   
//   for (size_t i = 0; i < sorted_data.size(); i += records_per_bin) {
//     size_t end = std::min(i + records_per_bin, sorted_data.size());
//     Bin bin;
//     bin.lower_bound = (i == 0) ? -std::numeric_limits<double>::infinity() : sorted_data[i].first;
//     bin.upper_bound = (end == sorted_data.size()) ? std::numeric_limits<double>::infinity() : sorted_data[end - 1].first;
//     bin.count = end - i;
//     bin.count_pos = 0;
//     bin.count_neg = 0;
//     
//     for (size_t j = i; j < end; ++j) {
//       if (sorted_data[j].second == 1) {
//         bin.count_pos++;
//       } else {
//         bin.count_neg++;
//       }
//     }
//     
//     bins.push_back(bin);
//   }
//   
//   // ChiMerge algorithm
//   while (bins.size() > min_bins) {
//     double min_chi_square = std::numeric_limits<double>::max();
//     size_t merge_index = 0;
//     
// #pragma omp parallel
// {
// double local_min_chi_square = std::numeric_limits<double>::max();
// size_t local_merge_index = 0;
// 
// #pragma omp for nowait
// for (size_t i = 0; i < bins.size() - 1; ++i) {
//   double chi_square = calculate_chi_square(bins[i], bins[i + 1]);
//   if (chi_square < local_min_chi_square) {
//     local_min_chi_square = chi_square;
//     local_merge_index = i;
//   }
// }
// 
// #pragma omp critical
// {
// if (local_min_chi_square < min_chi_square) {
//   min_chi_square = local_min_chi_square;
//   merge_index = local_merge_index;
// }
// }
// }
// 
// merge_bins(merge_index);
// 
// if (bins.size() <= max_bins) {
// break;
// }
//   }
//   
//   // Merge rare bins
//   double total_count = static_cast<double>(feature.size());
//   for (auto it = bins.begin(); it != bins.end(); ) {
//     if (static_cast<double>(it->count) / total_count < bin_cutoff) {
//       if (it == bins.begin()) {
//         merge_bins(0);
//       } else {
//         merge_bins(std::distance(bins.begin(), it) - 1);
//       }
//     } else {
//       ++it;
//     }
//   }
//   
//   calculate_woe_iv();
// }
// 
// Rcpp::List get_results() const {
//   Rcpp::NumericVector woefeature(feature.size());
//   Rcpp::List woebin;
//   Rcpp::StringVector bin_labels;
//   Rcpp::NumericVector woe_values;
//   Rcpp::NumericVector iv_values;
//   Rcpp::IntegerVector count_values;
//   Rcpp::IntegerVector count_pos_values;
//   Rcpp::IntegerVector count_neg_values;
//   
// #pragma omp parallel for
//   for (size_t i = 0; i < feature.size(); ++i) {
//     for (const auto& bin : bins) {
//       if (feature[i] <= bin.upper_bound) {
//         woefeature[i] = bin.woe;
//         break;
//       }
//     }
//   }
//   
//   for (const auto& bin : bins) {
//     std::string bin_label = (bin.lower_bound == -std::numeric_limits<double>::infinity() ? "(-Inf" : "(" + std::to_string(bin.lower_bound)) +
//       ";" + (bin.upper_bound == std::numeric_limits<double>::infinity() ? "+Inf]" : std::to_string(bin.upper_bound) + "]");
//     bin_labels.push_back(bin_label);
//     woe_values.push_back(bin.woe);
//     iv_values.push_back(bin.iv);
//     count_values.push_back(bin.count);
//     count_pos_values.push_back(bin.count_pos);
//     count_neg_values.push_back(bin.count_neg);
//   }
//   
//   woebin["bin"] = bin_labels;
//   woebin["woe"] = woe_values;
//   woebin["iv"] = iv_values;
//   woebin["count"] = count_values;
//   woebin["count_pos"] = count_pos_values;
//   woebin["count_neg"] = count_neg_values;
//   
//   return Rcpp::List::create(
//     Rcpp::Named("woefeature") = woefeature,
//     Rcpp::Named("woebin") = woebin
//   );
// }
// };
// 
// //' @title Optimal Binning for Numerical Variables using ChiMerge
// //'
// //' @description
// //' This function implements an optimal binning algorithm for numerical variables using the ChiMerge approach with Weight of Evidence (WoE) and Information Value (IV) criteria.
// //'
// //' @param target An integer vector of binary target values (0 or 1).
// //' @param feature A numeric vector of feature values to be binned.
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff Minimum frequency of observations in each bin (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins for initial discretization (default: 20).
// //'
// //' @return A list containing two elements:
// //' \item{woefeature}{A numeric vector of WoE-transformed feature values.}
// //' \item{woebin}{A data frame with binning details, including bin boundaries, WoE, IV, and count statistics.}
// //'
// //' @examples
// //' \dontrun{
// //' # Generate sample data
// //' set.seed(123)
// //' n <- 10000
// //' feature <- rnorm(n)
// //' target <- rbinom(n, 1, plogis(0.5 * feature))
// //'
// //' # Apply optimal binning
// //' result <- optimal_binning_numerical_cm(target, feature, min_bins = 3, max_bins = 5)
// //'
// //' # View binning results
// //' print(result$woebin)
// //'
// //' # Plot WoE transformation
// //' plot(feature, result$woefeature, main = "WoE Transformation", 
// //' xlab = "Original Feature", ylab = "WoE")
// //' }
// //'
// //' @details
// //' The optimal binning algorithm for numerical variables uses the ChiMerge approach with Weight of Evidence (WoE) and Information Value (IV) to create bins that maximize the predictive power of the feature while maintaining interpretability.
// //'
// //' The algorithm follows these steps:
// //' 1. Initial discretization into max_n_prebins
// //' 2. Iterative merging of adjacent bins based on chi-square statistic
// //' 3. Merging of rare bins based on the bin_cutoff parameter
// //' 4. Calculation of WoE and IV for each final bin
// //'
// //' The chi-square statistic for two adjacent bins is calculated as:
// //'
// //' \deqn{\chi^2 = \sum_{i=1}^{2} \sum_{j=1}^{2} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}}
// //'
// //' where \eqn{O_{ij}} is the observed frequency and \eqn{E_{ij}} is the expected frequency for bin i and class j.
// //'
// //' Weight of Evidence (WoE) is calculated for each bin as:
// //'
// //' \deqn{WoE_i = \ln\left(\frac{P(X_i|Y=1)}{P(X_i|Y=0)}\right)}
// //'
// //' where \eqn{P(X_i|Y=1)} is the proportion of positive cases in bin i, and \eqn{P(X_i|Y=0)} is the proportion of negative cases in bin i.
// //'
// //' Information Value (IV) for each bin is calculated as:
// //'
// //' \deqn{IV_i = (P(X_i|Y=1) - P(X_i|Y=0)) \times WoE_i}
// //'
// //' The total IV for the feature is the sum of IVs across all bins:
// //'
// //' \deqn{IV_{total} = \sum_{i=1}^{n} IV_i}
// //'
// //' The ChiMerge approach ensures that the resulting binning maximizes the separation between classes while maintaining the desired number of bins and respecting the minimum bin frequency constraint.
// //'
// //' @references
// //' \itemize{
// //'   \item Kerber, R. (1992). ChiMerge: Discretization of Numeric Attributes. In Proceedings of the tenth national conference on Artificial intelligence (pp. 123-128). AAAI Press.
// //'   \item Zeng, G. (2014). A necessary condition for a good binning algorithm in credit scoring. Applied Mathematical Sciences, 8(65), 3229-3242.
// //' }
// //'
// //' @author Lopes, J. E.
// //'
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_numerical_cm(
//    Rcpp::IntegerVector target,
//    Rcpp::NumericVector feature,
//    int min_bins = 3,
//    int max_bins = 5,
//    double bin_cutoff = 0.05,
//    int max_n_prebins = 20
// ) {
//  std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
//  std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
//  
//  OptimalBinningNumericalCM binner(feature_vec, target_vec, min_bins,max_bins, bin_cutoff, max_n_prebins);
//  binner.fit();
//  return binner.get_results();
// }
