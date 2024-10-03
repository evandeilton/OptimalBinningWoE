// [[Rcpp::plugins(openmp)]]
#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <numeric>

// Enable OpenMP for parallel processing
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

class OptimalBinningNumericalIR {
private:
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  
  const std::vector<double>& feature;
  const std::vector<int>& target;
  
  std::vector<double> bin_edges;
  std::vector<double> woe_values;
  double total_iv;
  
  struct BinInfo {
    double lower;
    double upper;
    int count;
    int count_pos;
    int count_neg;
    double woe;
    double iv;
  };
  
  std::vector<BinInfo> bin_info;
  
public:
  OptimalBinningNumericalIR(int min_bins_, int max_bins_,
                            double bin_cutoff_, int max_n_prebins_,
                            const std::vector<double>& feature_,
                            const std::vector<int>& target_)
    : min_bins(min_bins_), max_bins(max_bins_),
      bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_),
      feature(feature_), target(target_), total_iv(0.0) {}
  
  void fit() {
    validateInputs();
    createInitialBins();
    mergeLowFrequencyBins();
    ensureMinMaxBins();
    applyIsotonicRegression();
    calculateWOEandIV();
  }
  
  Rcpp::List getResults() const {
    std::vector<double> woefeature = applyWOEToFeature();
    Rcpp::DataFrame woebin = createWOEBinDataFrame();
    
    return Rcpp::List::create(
      Rcpp::Named("woefeature") = woefeature,
      Rcpp::Named("woebin") = woebin,
      Rcpp::Named("iv") = total_iv
    );
  }
  
private:
  void validateInputs() const {
    if (feature.size() != target.size()) {
      Rcpp::stop("Feature and target must have the same length.");
    }
    if (min_bins < 2) {
      Rcpp::stop("min_bins must be at least 2.");
    }
    if (max_bins < min_bins) {
      Rcpp::stop("max_bins must be greater than or equal to min_bins.");
    }
    // Check if target contains only 0 and 1
    auto [min_it, max_it] = std::minmax_element(target.begin(), target.end());
    if (*min_it < 0 || *max_it > 1) {
      Rcpp::stop("Target must be binary (0 or 1).");
    }
    // Check if both classes are present
    int sum_target = std::accumulate(target.begin(), target.end(), 0);
    if (sum_target == 0 || sum_target == static_cast<int>(target.size())) {
      Rcpp::stop("Target must contain both classes (0 and 1).");
    }
    // Check for NaN and Inf in feature
    for (size_t i = 0; i < feature.size(); ++i) {
      if (std::isnan(feature[i]) || std::isinf(feature[i])) {
        Rcpp::stop("Feature contains NaN or Inf values.");
      }
    }
  }
  
  void createInitialBins() {
    std::vector<double> sorted_feature = feature;
    std::sort(sorted_feature.begin(), sorted_feature.end());
    
    // Remove duplicates
    sorted_feature.erase(std::unique(sorted_feature.begin(), sorted_feature.end()), sorted_feature.end());
    
    int unique_vals = sorted_feature.size();
    int n_prebins = std::min(max_n_prebins, unique_vals);
    n_prebins = std::max(n_prebins, min_bins); // Ensure at least min_bins
    
    bin_edges.resize(n_prebins + 1);
    bin_edges[0] = -std::numeric_limits<double>::infinity();
    bin_edges[n_prebins] = std::numeric_limits<double>::infinity();
    
    for (int i = 1; i < n_prebins; ++i) {
      int idx = std::round((static_cast<double>(i) / n_prebins) * unique_vals);
      // idx = std::clamp(idx, 1, unique_vals - 1);
      idx = std::clamp(idx, 1, unique_vals - 1);
      bin_edges[i] = sorted_feature[idx];
    }
  }
  
  void mergeLowFrequencyBins() {
    std::vector<BinInfo> temp_bins;
    int total_count = feature.size();
    
    for (size_t i = 0; i < bin_edges.size() - 1; ++i) {
      BinInfo bin;
      bin.lower = bin_edges[i];
      bin.upper = bin_edges[i + 1];
      bin.count = 0;
      bin.count_pos = 0;
      bin.count_neg = 0;
      bin.woe = 0.0;
      bin.iv = 0.0;
      
      for (size_t j = 0; j < feature.size(); ++j) {
        if ((feature[j] > bin.lower || (i == 0 && feature[j] == bin.lower)) && feature[j] <= bin.upper) {
          bin.count++;
          bin.count_pos += target[j];
        }
      }
      bin.count_neg = bin.count - bin.count_pos;
      
      double proportion = static_cast<double>(bin.count) / total_count;
      if (proportion >= bin_cutoff || temp_bins.empty()) {
        temp_bins.push_back(bin);
      } else {
        // Merge with the previous bin
        temp_bins.back().upper = bin.upper;
        temp_bins.back().count += bin.count;
        temp_bins.back().count_pos += bin.count_pos;
        temp_bins.back().count_neg += bin.count_neg;
      }
    }
    
    bin_info = temp_bins;
  }
  
  void ensureMinMaxBins() {
    // Merge bins if less than min_bins
    while (bin_info.size() < static_cast<size_t>(min_bins)) {
      // Find the bin with the smallest count to merge
      auto it = std::min_element(bin_info.begin(), bin_info.end(),
                                 [](const BinInfo& a, const BinInfo& b) {
                                   return a.count < b.count;
                                 });
      if (it == bin_info.end()) break; // Safety check
      
      size_t idx = std::distance(bin_info.begin(), it);
      if (bin_info.size() == 1) break; // Cannot merge further
      
      // Merge with the adjacent bin that has the closest count
      if (idx == 0) {
        // Merge with next bin
        mergeBins(idx, idx + 1);
      } else if (idx == bin_info.size() - 1) {
        // Merge with previous bin
        mergeBins(idx - 1, idx);
      } else {
        // Merge with the neighbor with the smaller count
        if (bin_info[idx - 1].count < bin_info[idx + 1].count) {
          mergeBins(idx - 1, idx);
        } else {
          mergeBins(idx, idx + 1);
        }
      }
    }
    
    // Merge bins if more than max_bins
    while (bin_info.size() > static_cast<size_t>(max_bins)) {
      // Find the pair of adjacent bins with the smallest WoE difference
      double min_diff = std::numeric_limits<double>::max();
      size_t merge_idx = 0;
      
      for (size_t i = 0; i < bin_info.size() - 1; ++i) {
        double diff = std::abs(bin_info[i].woe - bin_info[i + 1].woe);
        if (diff < min_diff) {
          min_diff = diff;
          merge_idx = i;
        }
      }
      
      mergeBins(merge_idx, merge_idx + 1);
    }
  }
  
  void mergeBins(size_t idx1, size_t idx2) {
    bin_info[idx1].upper = bin_info[idx2].upper;
    bin_info[idx1].count += bin_info[idx2].count;
    bin_info[idx1].count_pos += bin_info[idx2].count_pos;
    bin_info[idx1].count_neg += bin_info[idx2].count_neg;
    bin_info.erase(bin_info.begin() + idx2);
  }
  
  void applyIsotonicRegression() {
    int n = bin_info.size();
    std::vector<double> y(n), w(n);
    
    for (int i = 0; i < n; ++i) {
      y[i] = static_cast<double>(bin_info[i].count_pos) / bin_info[i].count;
      w[i] = static_cast<double>(bin_info[i].count);
    }
    
    std::vector<double> isotonic_y = isotonic_regression(y, w);
    
    for (int i = 0; i < n; ++i) {
      // Prevent division by zero
      double pos_rate = isotonic_y[i];
      double neg_rate = 1.0 - pos_rate;
      if (neg_rate <= 0.0) {
        bin_info[i].woe = std::log((pos_rate + 1e-10) / (neg_rate + 1e-10));
      } else {
        bin_info[i].woe = std::log(pos_rate / neg_rate);
      }
    }
  }
  
  std::vector<double> isotonic_regression(const std::vector<double>& y, const std::vector<double>& w) {
    int n = y.size();
    std::vector<double> result = y;
    std::vector<double> weights = w;
    std::vector<int> start(n);
    std::vector<int> end(n);
    
    for (int i = 0; i < n; ++i) {
      start[i] = i;
      end[i] = i;
    }
    
    int i = 0;
    while (i < n - 1) {
      if (result[i] > result[i + 1]) {
        double sum_yw = result[i] * weights[i] + result[i + 1] * weights[i + 1];
        double sum_w = weights[i] + weights[i + 1];
        double pooled = sum_yw / sum_w;
        result[i] = pooled;
        result[i + 1] = pooled;
        weights[i] = sum_w;
        weights[i + 1] = sum_w;
        
        // Merge blocks
        int j = i;
        while (j > 0 && result[j - 1] > result[j]) {
          sum_yw = result[j - 1] * weights[j - 1] + result[j] * weights[j];
          sum_w = weights[j - 1] + weights[j];
          pooled = sum_yw / sum_w;
          result[j - 1] = pooled;
          result[j] = pooled;
          weights[j - 1] = sum_w;
          weights[j] = sum_w;
          j--;
        }
        // Restart
        i = std::max(j, 0);
      } else {
        i++;
      }
    }
    
    return result;
  }
  
  void calculateWOEandIV() {
    double total_pos = 0.0, total_neg = 0.0;
    for (const auto& bin : bin_info) {
      total_pos += bin.count_pos;
      total_neg += bin.count_neg;
    }
    
    // Prevent division by zero
    if (total_pos == 0.0 || total_neg == 0.0) {
      Rcpp::stop("Insufficient positive or negative cases for WoE and IV calculations.");
    }
    
    total_iv = 0.0;
    for (auto& bin : bin_info) {
      double pos_rate = static_cast<double>(bin.count_pos) / total_pos;
      double neg_rate = static_cast<double>(bin.count_neg) / total_neg;
      
      // Handle cases where pos_rate or neg_rate is zero
      if (pos_rate == 0.0) pos_rate = 1e-10;
      if (neg_rate == 0.0) neg_rate = 1e-10;
      
      bin.woe = std::log(pos_rate / neg_rate);
      bin.iv = (pos_rate - neg_rate) * bin.woe;
      total_iv += bin.iv;
    }
  }
  
  std::vector<double> applyWOEToFeature() const {
    int n = feature.size();
    std::vector<double> woefeature(n, 0.0);
    
    // Create sorted bin edges for binary search
    std::vector<double> sorted_edges;
    for (const auto& bin : bin_info) {
      sorted_edges.push_back(bin.upper);
    }
    
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
      double x = feature[i];
      // Binary search to find the bin
      int bin_idx = std::upper_bound(sorted_edges.begin(), sorted_edges.end(), x) - sorted_edges.begin();
      if (bin_idx >= bin_info.size()) bin_idx = bin_info.size() - 1;
      woefeature[i] = bin_info[bin_idx].woe;
    }
    
    return woefeature;
  }
  
  Rcpp::DataFrame createWOEBinDataFrame() const {
    int n_bins = bin_info.size();
    Rcpp::CharacterVector bin_labels(n_bins);
    Rcpp::NumericVector woe(n_bins), iv(n_bins);
    Rcpp::IntegerVector count(n_bins), count_pos(n_bins), count_neg(n_bins);
    
    for (int i = 0; i < n_bins; ++i) {
      const auto& b = bin_info[i];
      std::string label;
      
      if (i == 0) {
        label = "(-Inf;" + std::to_string(b.upper) + "]";
      } else if (i == n_bins - 1) {
        label = "(" + std::to_string(b.lower) + ";Inf)";
      } else {
        label = "(" + std::to_string(b.lower) + ";" + std::to_string(b.upper) + "]";
      }
      
      bin_labels[i] = label;
      woe[i] = b.woe;
      iv[i] = b.iv;
      count[i] = b.count;
      count_pos[i] = b.count_pos;
      count_neg[i] = b.count_neg;
    }
    
    return Rcpp::DataFrame::create(
      Rcpp::Named("bin") = bin_labels,
      Rcpp::Named("woe") = woe,
      Rcpp::Named("iv") = iv,
      Rcpp::Named("count") = count,
      Rcpp::Named("count_pos") = count_pos,
      Rcpp::Named("count_neg") = count_neg,
      Rcpp::Named("iv_total") = total_iv
    );
  }
};

//' @title Optimal Binning for Numerical Variables using Isotonic Regression
//' 
//' @description
//' This function performs optimal binning for numerical variables using isotonic regression. 
//' It creates optimal bins for a numerical feature based on its relationship with a binary 
//' target variable, maximizing the predictive power while respecting user-defined constraints.
//' 
//' @param target An integer vector of binary target values (0 or 1).
//' @param feature A numeric vector of feature values.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum proportion of total observations for a bin to avoid being merged (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins before the optimization process (default: 20).
//' 
//' @return A list containing three elements:
//' \item{woefeature}{A numeric vector of Weight of Evidence (WoE) values for each observation.}
//' \item{woebin}{A data frame with the following columns:
//'   \itemize{
//'     \item bin: Character vector of bin ranges.
//'     \item woe: Numeric vector of WoE values for each bin.
//'     \item iv: Numeric vector of Information Value (IV) for each bin.
//'     \item count: Integer vector of total observations in each bin.
//'     \item count_pos: Integer vector of positive target observations in each bin.
//'     \item count_neg: Integer vector of negative target observations in each bin.
//'     \item iv_total: Total Information Value (IV) for the feature.
//'   }
//' }
//' 
//' @details
//' The Optimal Binning algorithm for numerical variables using isotonic regression works as follows:
//' 1. Create initial bins using equal-frequency binning.
//' 2. Merge low-frequency bins (those with a proportion less than \code{bin_cutoff}).
//' 3. Ensure the number of bins is between \code{min_bins} and \code{max_bins} by splitting or merging bins.
//' 4. Apply isotonic regression to smooth the positive rates across bins.
//' 5. Calculate Weight of Evidence (WoE) and Information Value (IV) for each bin.
//' 
//' Weight of Evidence (WoE) is calculated for each bin as:
//' 
//' \deqn{WoE_i = \ln\left(\frac{P(X_i|Y=1)}{P(X_i|Y=0)}\right)}
//' 
//' where \eqn{P(X_i|Y=1)} is the proportion of positive cases in bin i, and 
//' \eqn{P(X_i|Y=0)} is the proportion of negative cases in bin i.
//' 
//' Information Value (IV) for each bin is calculated as:
//' 
//' \deqn{IV_i = (P(X_i|Y=1) - P(X_i|Y=0)) * WoE_i}
//' 
//' The algorithm aims to create monotonic bins that maximize the predictive power of the 
//' numerical variable while adhering to the specified constraints. Isotonic regression ensures 
//' that the positive rates are non-decreasing across bins, which is particularly useful for 
//' credit scoring and risk modeling applications.
//' 
//' This implementation uses OpenMP for parallel processing when available, which can 
//' significantly speed up the computation for large datasets.
//' 
//' @examples
//' \dontrun{
//' # Create sample data
//' set.seed(123)
//' n <- 1000
//' target <- sample(0:1, n, replace = TRUE)
//' feature <- rnorm(n)
//' # Run optimal binning
//' result <- optimal_binning_numerical_ir(target, feature, min_bins = 2, max_bins = 4)
//' # Print results
//' print(result$woebin)
//' # Plot WoE values
//' plot(result$woebin$woe, type = "s", xaxt = "n", xlab = "Bins", ylab = "WoE",
//'      main = "Weight of Evidence by Bin")
//' axis(1, at = 1:nrow(result$woebin), labels = result$woebin$bin)
//' }
//' 
//' @references
//' \itemize{
//'   \item Barlow, R. E., Bartholomew, D. J., Bremner, J. M., & Brunk, H. D. (1972). 
//'         Statistical inference under order restrictions: The theory and application 
//'         of isotonic regression. Wiley.
//'   \item Mironchyk, P., & Tchistiakov, V. (2017). Monotone optimal binning algorithm 
//'         for credit risk modeling. SSRN Electronic Journal. DOI: 10.2139/ssrn.2978774
//' }
//' 
//' @author Lopes, J. E.
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_ir(Rcpp::IntegerVector target, Rcpp::NumericVector feature,
                                       int min_bins = 3, int max_bins = 5,
                                       double bin_cutoff = 0.05, int max_n_prebins = 20) {
 // Convert Rcpp vectors to std:: vectors
 std::vector<int> target_std = Rcpp::as<std::vector<int>>(target);
 std::vector<double> feature_std = Rcpp::as<std::vector<double>>(feature);
 
 // Handle features with fewer unique values than min_bins
 std::vector<double> sorted_unique = feature_std;
 std::sort(sorted_unique.begin(), sorted_unique.end());
 sorted_unique.erase(std::unique(sorted_unique.begin(), sorted_unique.end()), sorted_unique.end());
 int unique_vals = sorted_unique.size();
 if (unique_vals < min_bins) {
   min_bins = unique_vals;
   if (max_bins < min_bins) {
     max_bins = min_bins;
   }
 }
 
 OptimalBinningNumericalIR binner(min_bins, max_bins, bin_cutoff, max_n_prebins, feature_std, target_std);
 binner.fit();
 return binner.getResults();
}



// #include <Rcpp.h>
// 
// #ifdef _OPENMP
// #include <omp.h>
// #endif
// 
// #include <algorithm>
// #include <vector>
// #include <string>
// #include <cmath>
// #include <limits>
// #include <numeric>
// 
// // [[Rcpp::plugins(openmp)]]
// 
// class OptimalBinningNumericalIR {
// private:
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
//   
//   std::vector<double> feature;
//   std::vector<int> target;
//   
//   std::vector<double> bin_edges;
//   std::vector<double> woe_values;
//   double total_iv;
//   
//   struct BinInfo {
//     double lower;
//     double upper;
//     int count;
//     int count_pos;
//     int count_neg;
//     double woe;
//     double iv;
//   };
//   
//   std::vector<BinInfo> bin_info;
//   
// public:
//   OptimalBinningNumericalIR(int min_bins_ = 2, int max_bins_ = 5,
//                             double bin_cutoff_ = 0.05, int max_n_prebins_ = 20)
//     : min_bins(min_bins_), max_bins(max_bins_),
//       bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_) {}
//   
//   void fit(const std::vector<double>& feature_, const std::vector<int>& target_) {
//     feature = feature_;
//     target = target_;
//     
//     validateInputs();
//     createInitialBins();
//     mergeLowFrequencyBins();
//     ensureMinMaxBins();
//     applyIsotonicRegression();
//     calculateWOEandIV();
//   }
//   
//   Rcpp::List getResults() {
//     std::vector<double> woefeature = applyWOEToFeature();
//     Rcpp::DataFrame woebin = createWOEBinDataFrame();
//     
//     return Rcpp::List::create(
//       Rcpp::Named("woefeature") = woefeature,
//       Rcpp::Named("woebin") = woebin
//     );
//   }
//   
// private:
//   void validateInputs() {
//     if (feature.size() != target.size()) {
//       Rcpp::stop("Feature and target must have the same length");
//     }
//     if (min_bins < 2) {
//       Rcpp::stop("min_bins must be at least 2");
//     }
//     if (max_bins < min_bins) {
//       Rcpp::stop("max_bins must be greater than or equal to min_bins");
//     }
//     if (*std::min_element(target.begin(), target.end()) < 0 || 
//         *std::max_element(target.begin(), target.end()) > 1) {
//         Rcpp::stop("Target must be binary (0 or 1)");
//     }
//   }
//   
//   void createInitialBins() {
//     std::vector<double> sorted_feature = feature;
//     std::sort(sorted_feature.begin(), sorted_feature.end());
//     int n = sorted_feature.size();
//     int n_prebins = std::min(max_n_prebins, n);
//     
//     bin_edges.resize(n_prebins + 1);
//     bin_edges[0] = sorted_feature[0] - 1e-6;
//     bin_edges[n_prebins] = sorted_feature[n - 1] + 1e-6;
//     
//     for (int i = 1; i < n_prebins; ++i) {
//       int idx = (i * n) / n_prebins;
//       bin_edges[i] = (sorted_feature[idx - 1] + sorted_feature[idx]) / 2;
//     }
//   }
//   
//   void mergeLowFrequencyBins() {
//     std::vector<BinInfo> temp_bins;
//     int total_count = feature.size();
//     
//     for (size_t i = 0; i < bin_edges.size() - 1; ++i) {
//       BinInfo bin;
//       bin.lower = bin_edges[i];
//       bin.upper = bin_edges[i + 1];
//       
//       bin.count = 0;
//       bin.count_pos = 0;
//       bin.count_neg = 0;
//       
//       for (size_t j = 0; j < feature.size(); ++j) {
//         if (feature[j] > bin.lower && feature[j] <= bin.upper) {
//           bin.count++;
//           bin.count_pos += target[j];
//         }
//       }
//       bin.count_neg = bin.count - bin.count_pos;
//       
//       if (static_cast<double>(bin.count) / total_count >= bin_cutoff) {
//         temp_bins.push_back(bin);
//       } else if (!temp_bins.empty()) {
//         temp_bins.back().upper = bin.upper;
//         temp_bins.back().count += bin.count;
//         temp_bins.back().count_pos += bin.count_pos;
//         temp_bins.back().count_neg += bin.count_neg;
//       } else {
//         // If the first bin is below cutoff, we need to keep it to ensure we have at least one bin
//         temp_bins.push_back(bin);
//       }
//     }
//     
//     bin_info = temp_bins;
//   }
//   
//   void ensureMinMaxBins() {
//     // If we have fewer than min_bins, split the largest bins
//     while (bin_info.size() < min_bins) {
//       auto it = std::max_element(bin_info.begin(), bin_info.end(),
//                                  [](const BinInfo& a, const BinInfo& b) { return a.count < b.count; });
//       
//       size_t idx = std::distance(bin_info.begin(), it);
//       double mid = (it->lower + it->upper) / 2;
//       
//       BinInfo new_bin = *it;
//       it->upper = mid;
//       new_bin.lower = mid;
//       
//       // Recalculate counts for split bins
//       it->count = 0;
//       it->count_pos = 0;
//       new_bin.count = 0;
//       new_bin.count_pos = 0;
//       
//       for (size_t i = 0; i < feature.size(); ++i) {
//         if (feature[i] > it->lower && feature[i] <= it->upper) {
//           it->count++;
//           it->count_pos += target[i];
//         } else if (feature[i] > new_bin.lower && feature[i] <= new_bin.upper) {
//           new_bin.count++;
//           new_bin.count_pos += target[i];
//         }
//       }
//       
//       it->count_neg = it->count - it->count_pos;
//       new_bin.count_neg = new_bin.count - new_bin.count_pos;
//       
//       bin_info.insert(it + 1, new_bin);
//     }
//     
//     // If we have more than max_bins, merge the most similar adjacent bins
//     while (bin_info.size() > max_bins) {
//       auto it = std::min_element(bin_info.begin(), bin_info.end() - 1,
//                                  [](const BinInfo& a, const BinInfo& b) {
//                                    double rate_a = static_cast<double>(a.count_pos) / a.count;
//                                    double rate_b = static_cast<double>(b.count_pos) / b.count;
//                                    double next_rate_a = static_cast<double>(a.count_pos + (*(std::next(&a))).count_pos) /
//                                      (a.count + (*(std::next(&a))).count);
//                                    double next_rate_b = static_cast<double>(b.count_pos + (*(std::next(&b))).count_pos) /
//                                      (b.count + (*(std::next(&b))).count);
//                                    return std::abs(rate_a - next_rate_a) < std::abs(rate_b - next_rate_b);
//                                  });
//       
//       it->upper = (*(std::next(it))).upper;
//       it->count += (*(std::next(it))).count;
//       it->count_pos += (*(std::next(it))).count_pos;
//       it->count_neg += (*(std::next(it))).count_neg;
//       bin_info.erase(std::next(it));
//     }
//   }
//   
//   void applyIsotonicRegression() {
//     int n = bin_info.size();
//     std::vector<double> y(n), w(n);
//     
//     for (int i = 0; i < n; ++i) {
//       double pos_rate = static_cast<double>(bin_info[i].count_pos) / bin_info[i].count;
//       y[i] = pos_rate;
//       w[i] = bin_info[i].count;
//     }
//     
//     std::vector<double> isotonic_y = isotonic_regression(y, w);
//     
//     for (int i = 0; i < n; ++i) {
//       bin_info[i].woe = std::log(isotonic_y[i] / (1 - isotonic_y[i]));
//     }
//   }
//   
//   std::vector<double> isotonic_regression(const std::vector<double>& y, const std::vector<double>& w) {
//     int n = y.size();
//     std::vector<double> result = y;
//     
//     for (int i = 1; i < n; ++i) {
//       if (result[i] < result[i - 1]) {
//         double numerator = 0, denominator = 0;
//         int j = i;
//         
//         while (j > 0 && result[j] < result[j - 1]) {
//           numerator += w[j] * y[j] + w[j - 1] * y[j - 1];
//           denominator += w[j] + w[j - 1];
//           double pooled = numerator / denominator;
//           result[j] = result[j - 1] = pooled;
//           j--;
//         }
//       }
//     }
//     
//     return result;
//   }
//   
//   void calculateWOEandIV() {
//     double total_pos = 0, total_neg = 0;
//     for (const auto& bin : bin_info) {
//       total_pos += bin.count_pos;
//       total_neg += bin.count_neg;
//     }
//     
//     total_iv = 0;
//     for (auto& bin : bin_info) {
//       double pos_rate = bin.count_pos / total_pos;
//       double neg_rate = bin.count_neg / total_neg;
//       bin.woe = std::log(pos_rate / neg_rate);
//       bin.iv = (pos_rate - neg_rate) * bin.woe;
//       total_iv += bin.iv;
//     }
//   }
//   
//   std::vector<double> applyWOEToFeature() {
//     int n = feature.size();
//     std::vector<double> woefeature(n);
//     
// #pragma omp parallel for
//     for (int i = 0; i < n; ++i) {
//       double x = feature[i];
//       for (const auto& bin : bin_info) {
//         if (x > bin.lower && x <= bin.upper) {
//           woefeature[i] = bin.woe;
//           break;
//         }
//       }
//     }
//     
//     return woefeature;
//   }
//   
//   Rcpp::DataFrame createWOEBinDataFrame() {
//     int n_bins = bin_info.size();
//     Rcpp::CharacterVector bin(n_bins);
//     Rcpp::NumericVector woe(n_bins), iv(n_bins);
//     Rcpp::IntegerVector count(n_bins), count_pos(n_bins), count_neg(n_bins);
//     
//     for (int i = 0; i < n_bins; ++i) {
//       const auto& b = bin_info[i];
//       bin[i] = "(" + std::to_string(b.lower) + ";" + std::to_string(b.upper) + "]";
//       woe[i] = b.woe;
//       iv[i] = b.iv;
//       count[i] = b.count;
//       count_pos[i] = b.count_pos;
//       count_neg[i] = b.count_neg;
//     }
//     
//     return Rcpp::DataFrame::create(
//       Rcpp::Named("bin") = bin,
//       Rcpp::Named("woe") = woe,
//       Rcpp::Named("iv") = iv,
//       Rcpp::Named("count") = count,
//       Rcpp::Named("count_pos") = count_pos,
//       Rcpp::Named("count_neg") = count_neg
//     );
//   }
// };
// 
// 
// //' @title Optimal Binning for Numerical Variables using Isotonic Regression
// //' 
// //' @description
// //' This function performs optimal binning for numerical variables using isotonic regression. 
// //' It creates optimal bins for a numerical feature based on its relationship with a binary 
// //' target variable, maximizing the predictive power while respecting user-defined constraints.
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
// //' The Optimal Binning algorithm for numerical variables using isotonic regression works as follows:
// //' 1. Create initial bins using equal-frequency binning.
// //' 2. Merge low-frequency bins (those with a proportion less than \code{bin_cutoff}).
// //' 3. Ensure the number of bins is between \code{min_bins} and \code{max_bins} by splitting or merging bins.
// //' 4. Apply isotonic regression to smooth the positive rates across bins.
// //' 5. Calculate Weight of Evidence (WoE) and Information Value (IV) for each bin.
// //' 
// //' Weight of Evidence (WoE) is calculated for each bin as:
// //' 
// //' \deqn{WoE_i = \ln\left(\frac{P(X_i|Y=1)}{P(X_i|Y=0)}\right)}
// //' 
// //' where \eqn{P(X_i|Y=1)} is the proportion of positive cases in bin i, and 
// //' \eqn{P(X_i|Y=0)} is the proportion of negative cases in bin i.
// //' 
// //' Information Value (IV) for each bin is calculated as:
// //' 
// //' \deqn{IV_i = (P(X_i|Y=1) - P(X_i|Y=0)) * WoE_i}
// //' 
// //' The algorithm aims to create monotonic bins that maximize the predictive power of the 
// //' numerical variable while adhering to the specified constraints. Isotonic regression ensures 
// //' that the positive rates are non-decreasing across bins, which is particularly useful for 
// //' credit scoring and risk modeling applications.
// //' 
// //' This implementation uses OpenMP for parallel processing when available, which can 
// //' significantly speed up the computation for large datasets.
// //' 
// //' @examples
// //' \dontrun{
// //' # Create sample data
// //' set.seed(123)
// //' n <- 1000
// //' target <- sample(0:1, n, replace = TRUE)
// //' feature <- rnorm(n)
// //' # Run optimal binning
// //' result <- optimal_binning_numerical_ir(target, feature, min_bins = 2, max_bins = 4)
// //' # Print results
// //' print(result$woebin)
// //' # Plot WoE values
// //' plot(result$woebin$woe, type = "s", xaxt = "n", xlab = "Bins", ylab = "WoE",
// //'      main = "Weight of Evidence by Bin")
// //' axis(1, at = 1:nrow(result$woebin), labels = result$woebin$bin)
// //' }
// //' 
// //' @references
// //' \itemize{
// //'   \item Barlow, R. E., Bartholomew, D. J., Bremner, J. M., & Brunk, H. D. (1972). 
// //'         Statistical inference under order restrictions: The theory and application 
// //'         of isotonic regression. Wiley.
// //'   \item Mironchyk, P., & Tchistiakov, V. (2017). Monotone optimal binning algorithm 
// //'         for credit risk modeling. SSRN Electronic Journal. DOI: 10.2139/ssrn.2978774
// //' }
// //' 
// //' @author Lopes, J. E.
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_numerical_ir(std::vector<int> target, std::vector<double> feature,
//                                         int min_bins = 3, int max_bins = 5,
//                                         double bin_cutoff = 0.05, int max_n_prebins = 20) {
//   OptimalBinningNumericalIR binner(min_bins, max_bins, bin_cutoff, max_n_prebins);
//   binner.fit(feature, target);
//   return binner.getResults();
// }
