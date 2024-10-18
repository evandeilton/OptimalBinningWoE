#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <sstream>

class OptimalBinningNumericalIR {
private:
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  double convergence_threshold;
  int max_iterations;
  
  const std::vector<double>& feature;
  const std::vector<int>& target;
  
  std::vector<double> bin_edges;
  double total_iv;
  bool converged;
  int iterations_run;
  
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
                            double convergence_threshold_, int max_iterations_,
                            const std::vector<double>& feature_,
                            const std::vector<int>& target_)
    : min_bins(min_bins_), max_bins(max_bins_),
      bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_),
      convergence_threshold(convergence_threshold_), max_iterations(max_iterations_),
      feature(feature_), target(target_), total_iv(0.0),
      converged(false), iterations_run(0) {
    validateInputs();
  }
  
  void fit() {
    createInitialBins();
    mergeLowFrequencyBins();
    ensureMinMaxBins();
    applyIsotonicRegression();
    calculateWOEandIV();
  }
  
  Rcpp::List getResults() const {
    return createWOEBinList();
  }
  
private:
  void validateInputs() const {
    if (feature.size() != target.size()) {
      throw std::invalid_argument("Feature and target must have the same length.");
    }
    if (min_bins < 2) {
      throw std::invalid_argument("min_bins must be at least 2.");
    }
    if (max_bins < min_bins) {
      throw std::invalid_argument("max_bins must be greater than or equal to min_bins.");
    }
    // Check if target contains only 0 and 1
    auto [min_it, max_it] = std::minmax_element(target.begin(), target.end());
    if (*min_it < 0 || *max_it > 1) {
      throw std::invalid_argument("Target must be binary (0 or 1).");
    }
    // Check if both classes are present
    int sum_target = std::accumulate(target.begin(), target.end(), 0);
    if (sum_target == 0 || sum_target == static_cast<int>(target.size())) {
      throw std::invalid_argument("Target must contain both classes (0 and 1).");
    }
    // Check for NaN and Inf in feature
    for (const auto& value : feature) {
      if (std::isnan(value) || std::isinf(value)) {
        throw std::invalid_argument("Feature contains NaN or Inf values.");
      }
    }
  }
  
  void createInitialBins() {
    std::vector<double> sorted_feature = feature;
    std::sort(sorted_feature.begin(), sorted_feature.end());
    
    // Remove duplicates
    sorted_feature.erase(std::unique(sorted_feature.begin(), sorted_feature.end()), sorted_feature.end());
    
    int unique_vals = sorted_feature.size();
    
    // If the number of unique values is less than or equal to min_bins,
    // use those values directly as bin edges
    if (unique_vals <= min_bins) {
      bin_edges = sorted_feature;
      bin_edges.insert(bin_edges.begin(), -std::numeric_limits<double>::infinity());
      bin_edges.push_back(std::numeric_limits<double>::infinity());
    } else {
      int n_prebins = std::min({max_n_prebins, unique_vals, max_bins});
      n_prebins = std::max(n_prebins, min_bins);
      
      bin_edges.resize(n_prebins + 1);
      bin_edges[0] = -std::numeric_limits<double>::infinity();
      bin_edges[n_prebins] = std::numeric_limits<double>::infinity();
      
      for (int i = 1; i < n_prebins; ++i) {
        int idx = static_cast<int>(std::round((static_cast<double>(i) / n_prebins) * unique_vals));
        idx = std::max(1, std::min(idx, unique_vals - 1));
        bin_edges[i] = sorted_feature[idx - 1];
      }
    }
  }
  
  void mergeLowFrequencyBins() {
    std::vector<BinInfo> temp_bins;
    int total_count = feature.size();
    
    for (size_t i = 0; i < bin_edges.size() - 1; ++i) {
      BinInfo bin{bin_edges[i], bin_edges[i + 1], 0, 0, 0, 0.0, 0.0};
      
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
    while (bin_info.size() < static_cast<size_t>(min_bins) && bin_info.size() > 1) {
      splitLargestBin();
    }
    
    while (bin_info.size() > static_cast<size_t>(max_bins)) {
      mergeSimilarBins();
    }
  }
  
  void splitLargestBin() {
    auto it = std::max_element(bin_info.begin(), bin_info.end(),
                               [](const BinInfo& a, const BinInfo& b) {
                                 return a.count < b.count;
                               });
    
    if (it != bin_info.end()) {
      size_t idx = std::distance(bin_info.begin(), it);
      double mid = (it->lower + it->upper) / 2.0;
      
      BinInfo new_bin = *it;
      it->upper = mid;
      new_bin.lower = mid;
      
      // Recalculate counts for both bins
      it->count = 0;
      it->count_pos = 0;
      new_bin.count = 0;
      new_bin.count_pos = 0;
      
      for (size_t j = 0; j < feature.size(); ++j) {
        if (feature[j] > it->lower && feature[j] <= it->upper) {
          it->count++;
          it->count_pos += target[j];
        } else if (feature[j] > new_bin.lower && feature[j] <= new_bin.upper) {
          new_bin.count++;
          new_bin.count_pos += target[j];
        }
      }
      
      it->count_neg = it->count - it->count_pos;
      new_bin.count_neg = new_bin.count - new_bin.count_pos;
      
      bin_info.insert(it + 1, new_bin);
    }
  }
  
  void mergeSimilarBins() {
    double min_diff = std::numeric_limits<double>::max();
    size_t merge_idx = 0;
    
    for (size_t i = 0; i < bin_info.size() - 1; ++i) {
      double diff = std::abs(
        static_cast<double>(bin_info[i].count_pos) / bin_info[i].count -
          static_cast<double>(bin_info[i + 1].count_pos) / bin_info[i + 1].count
      );
      if (diff < min_diff) {
        min_diff = diff;
        merge_idx = i;
      }
    }
    
    mergeBins(merge_idx, merge_idx + 1);
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
      bin_info[i].woe = calculateWoE(isotonic_y[i], 1.0 - isotonic_y[i]);
    }
  }
  
  std::vector<double> isotonic_regression(const std::vector<double>& y, const std::vector<double>& w) {
    int n = y.size();
    std::vector<double> result = y;
    std::vector<double> active_set(n);
    std::vector<double> active_set_weights(n);
    
    int j = 0;
    for (int i = 0; i < n; ++i) {
      active_set[j] = y[i];
      active_set_weights[j] = w[i];
      
      while (j > 0 && active_set[j - 1] > active_set[j]) {
        double weighted_avg = (active_set[j - 1] * active_set_weights[j - 1] + 
                               active_set[j] * active_set_weights[j]) / 
                               (active_set_weights[j - 1] + active_set_weights[j]);
        active_set[j - 1] = weighted_avg;
        active_set_weights[j - 1] += active_set_weights[j];
        --j;
      }
      ++j;
    }
    
    for (int i = 0; i < n; ++i) {
      result[i] = active_set[0];
      for (int k = 1; k < j; ++k) {
        if (y[i] >= active_set[k]) {
          result[i] = active_set[k];
        } else {
          break;
        }
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
    
    if (total_pos == 0.0 || total_neg == 0.0) {
      throw std::runtime_error("Insufficient positive or negative cases for WoE and IV calculations.");
    }
    
    total_iv = 0.0;
    for (auto& bin : bin_info) {
      double pos_rate = static_cast<double>(bin.count_pos) / total_pos;
      double neg_rate = static_cast<double>(bin.count_neg) / total_neg;
      
      bin.woe = calculateWoE(pos_rate, neg_rate);
      bin.iv = calculateIV(pos_rate, neg_rate, bin.woe);
      total_iv += bin.iv;
    }
  }
  
  double calculateWoE(double pos_rate, double neg_rate) const {
    const double epsilon = 1e-10;
    return std::log((pos_rate + epsilon) / (neg_rate + epsilon));
  }
  
  double calculateIV(double pos_rate, double neg_rate, double woe) const {
    return (pos_rate - neg_rate) * woe;
  }
  
  Rcpp::List createWOEBinList() const {
    int n_bins = bin_info.size();
    Rcpp::CharacterVector bin_labels(n_bins);
    Rcpp::NumericVector woe(n_bins), iv(n_bins);
    Rcpp::IntegerVector count(n_bins), count_pos(n_bins), count_neg(n_bins);
    Rcpp::NumericVector cutpoints(n_bins - 1);
    
    for (int i = 0; i < n_bins; ++i) {
      const auto& b = bin_info[i];
      std::string label = createBinLabel(b, i == 0, i == n_bins - 1);
      
      bin_labels[i] = label;
      woe[i] = b.woe;
      iv[i] = b.iv;
      count[i] = b.count;
      count_pos[i] = b.count_pos;
      count_neg[i] = b.count_neg;
      
      if (i < n_bins - 1) {
        cutpoints[i] = b.upper;
      }
    }
    
    return Rcpp::List::create(
      Rcpp::Named("bins") = bin_labels,
      Rcpp::Named("woe") = woe,
      Rcpp::Named("iv") = iv,
      Rcpp::Named("count") = count,
      Rcpp::Named("count_pos") = count_pos,
      Rcpp::Named("count_neg") = count_neg,
      Rcpp::Named("cutpoints") = cutpoints,
      Rcpp::Named("converged") = converged,
      Rcpp::Named("iterations") = iterations_run
      // Rcpp::Named("total_iv") = total_iv
    );
  }
  
  std::string createBinLabel(const BinInfo& bin, bool is_first, bool is_last) const {
    std::ostringstream oss;
    oss.precision(6);
    
    if (is_first) {
      oss << "(-Inf;" << bin.upper << "]";
    } else if (is_last) {
      oss << "(" << bin.lower << ";+Inf]";
    } else {
      oss << "(" << bin.lower << ";" << bin.upper << "]";
    }
    
    return oss.str();
  }
};

//' Optimal Binning for Numerical Variables using Isotonic Regression
//' 
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
//' @param convergence_threshold Threshold for convergence in isotonic regression (default: 1e-6).
//' @param max_iterations Maximum number of iterations for isotonic regression (default: 1000).
//' 
//' @return A list containing the following elements:
//' \itemize{
//'   \item bins: Character vector of bin ranges.
//'   \item woe: Numeric vector of Weight of Evidence (WoE) values for each bin.
//'   \item iv: Numeric vector of Information Value (IV) for each bin.
//'   \item count: Integer vector of total observations in each bin.
//'   \item count_pos: Integer vector of positive target observations in each bin.
//'   \item count_neg: Integer vector of negative target observations in each bin.
//'   \item cutpoints: Numeric vector of cutpoints between bins.
//'   \item converged: Logical indicating whether the algorithm converged.
//'   \item iterations: Number of iterations run.
//'   \item total_iv: Total Information Value (IV) for the feature.
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
//' @examples
//' \dontrun{
//' set.seed(123)
//' n <- 1000
//' target <- sample(0:1, n, replace = TRUE)
//' feature <- rnorm(n)
//' result <- optimal_binning_numerical_ir(target, feature, min_bins = 2, max_bins = 4)
//' print(result)
//' }
//' 
//' @references
//' Barlow, R. E., Bartholomew, D. J., Bremner, J. M., & Brunk, H. D. (1972).
//' Statistical inference under order restrictions: The theory and application
//' of isotonic regression. Wiley.
//' 
//' Mironchyk, P., & Tchistiakov, V. (2017). Monotone optimal binning algorithm
//' for credit risk modeling. SSRN Electronic Journal. DOI: 10.2139/ssrn.2978774
//' 
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_ir(Rcpp::IntegerVector target,
                                       Rcpp::NumericVector feature,
                                       int min_bins = 3,
                                       int max_bins = 5,
                                       double bin_cutoff = 0.05,
                                       int max_n_prebins = 20,
                                       double convergence_threshold = 1e-6,
                                       int max_iterations = 1000) {
 // Convert Rcpp vectors to std::vectors
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
 
 try {
   OptimalBinningNumericalIR binner(min_bins, max_bins, bin_cutoff, max_n_prebins,
                                    convergence_threshold, max_iterations,
                                    feature_std, target_std);
   binner.fit();
   return binner.getResults();
 } catch (const std::exception& e) {
   Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
 }
}
 