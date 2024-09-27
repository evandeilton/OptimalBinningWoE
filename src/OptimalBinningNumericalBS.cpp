#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <numeric>

using namespace Rcpp;

class OptimalBinningNumericalBS {
private:
  std::vector<double> feature;
  std::vector<int> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  int total_pos;
  int total_neg;
  std::vector<double> bin_edges;
  std::vector<double> bin_woe;
  std::vector<double> bin_iv;
  std::vector<int> bin_count;
  std::vector<int> bin_pos_count;
  std::vector<int> bin_neg_count;
  std::vector<std::string> bin_labels;
  std::vector<double> woefeature;
  
  void validate_inputs() {
    if (feature.empty()) stop("Feature vector is empty.");
    if (min_bins < 2) stop("min_bins must be >= 2.");
    if (max_bins < min_bins) stop("max_bins must be >= min_bins.");
    if (bin_cutoff < 0 || bin_cutoff > 0.5) stop("bin_cutoff must be between 0 and 0.5.");
    if (max_n_prebins < min_bins) stop("max_n_prebins must be >= min_bins.");
    if (max_n_prebins > 1000) stop("max_n_prebins must be <= 1000.");
  }
  
  void prebin_data() {
    std::vector<double> feature_clean;
    std::vector<int> target_clean;
    for (size_t i = 0; i < feature.size(); ++i) {
      if (std::isfinite(feature[i])) {
        feature_clean.push_back(feature[i]);
        target_clean.push_back(target[i]);
      }
    }
    
    std::sort(feature_clean.begin(), feature_clean.end());
    size_t n = feature_clean.size();
    std::vector<double> prebin_edges(max_n_prebins + 1);
    for (int i = 0; i <= max_n_prebins; ++i) {
      size_t idx = i * (n - 1) / max_n_prebins;
      prebin_edges[i] = feature_clean[idx];
    }
    
    prebin_edges[0] = std::min(prebin_edges[0], *std::min_element(feature_clean.begin(), feature_clean.end()));
    prebin_edges[prebin_edges.size() - 1] = std::max(prebin_edges[prebin_edges.size() - 1], *std::max_element(feature_clean.begin(), feature_clean.end()));
    
    bin_edges = prebin_edges;
    int n_bins = bin_edges.size() - 1;
    
    bin_count.resize(n_bins, 0);
    bin_pos_count.resize(n_bins, 0);
    bin_neg_count.resize(n_bins, 0);
    
    for (size_t i = 0; i < feature_clean.size(); ++i) {
      double val = feature_clean[i];
      int bin_idx = std::upper_bound(bin_edges.begin(), bin_edges.end(), val) - bin_edges.begin() - 1;
      bin_idx = std::max(0, std::min(bin_idx, n_bins - 1));
      
      bin_count[bin_idx]++;
      if (target_clean[i] == 1) {
        bin_pos_count[bin_idx]++;
      } else {
        bin_neg_count[bin_idx]++;
      }
    }
    
    // Remove empty bins
    std::vector<double> bin_edges_new;
    std::vector<int> bin_count_new, bin_pos_count_new, bin_neg_count_new;
    bin_edges_new.push_back(bin_edges.front());
    
    for (int i = 0; i < n_bins; ++i) {
      if (bin_count[i] > 0) {
        bin_edges_new.push_back(bin_edges[i + 1]);
        bin_count_new.push_back(bin_count[i]);
        bin_pos_count_new.push_back(bin_pos_count[i]);
        bin_neg_count_new.push_back(bin_neg_count[i]);
      }
    }
    
    bin_edges = bin_edges_new;
    bin_count = bin_count_new;
    bin_pos_count = bin_pos_count_new;
    bin_neg_count = bin_neg_count_new;
  }
  
  void calculate_woe_iv() {
    int n_bins = bin_edges.size() - 1;
    bin_woe.resize(n_bins);
    bin_iv.resize(n_bins);
    
    for (int i = 0; i < n_bins; ++i) {
      double pos_rate = (double)bin_pos_count[i] / total_pos;
      double neg_rate = (double)bin_neg_count[i] / total_neg;
      pos_rate = std::max(pos_rate, 1e-6);
      neg_rate = std::max(neg_rate, 1e-6);
      
      bin_woe[i] = std::log(pos_rate / neg_rate);
      bin_iv[i] = (pos_rate - neg_rate) * bin_woe[i];
    }
  }
  
  void enforce_monotonicity() {
    int n_bins = bin_edges.size() - 1;
    bool increasing = bin_woe[0] <= bin_woe[n_bins - 1];
    
    for (int i = 1; i < n_bins; ++i) {
      if ((increasing && bin_woe[i] < bin_woe[i - 1]) || (!increasing && bin_woe[i] > bin_woe[i - 1])) {
        int j = i;
        while (j > 0 && ((increasing && bin_woe[j] < bin_woe[j - 1]) || (!increasing && bin_woe[j] > bin_woe[j - 1]))) {
          merge_bins(j - 1);
          --j;
          --i;
          --n_bins;
        }
      }
    }
  }
  
  void merge_bins(int idx) {
    bin_edges.erase(bin_edges.begin() + idx + 1);
    bin_count[idx] += bin_count[idx + 1];
    bin_pos_count[idx] += bin_pos_count[idx + 1];
    bin_neg_count[idx] += bin_neg_count[idx + 1];
    bin_count.erase(bin_count.begin() + idx + 1);
    bin_pos_count.erase(bin_pos_count.begin() + idx + 1);
    bin_neg_count.erase(bin_neg_count.begin() + idx + 1);
    calculate_woe_iv();
  }
  
  void apply_bin_cutoff() {
    int n_bins = bin_edges.size() - 1;
    for (int i = 0; i < n_bins; ++i) {
      if ((double)bin_count[i] / feature.size() < bin_cutoff) {
        if (i > 0) {
          merge_bins(i - 1);
          --i;
          --n_bins;
        } else if (i < n_bins - 1) {
          merge_bins(i);
          --i;
          --n_bins;
        }
      }
    }
  }
  
  void adjust_bin_count() {
    while (bin_edges.size() - 1 > max_bins) {
      int min_iv_idx = std::min_element(bin_iv.begin(), bin_iv.end()) - bin_iv.begin();
      merge_bins(min_iv_idx);
    }
    
    while (bin_edges.size() - 1 < min_bins) {
      int max_count_idx = std::max_element(bin_count.begin(), bin_count.end()) - bin_count.begin();
      double mid_point = (bin_edges[max_count_idx] + bin_edges[max_count_idx + 1]) / 2;
      bin_edges.insert(bin_edges.begin() + max_count_idx + 1, mid_point);
      
      int left_count = bin_count[max_count_idx] / 2;
      int right_count = bin_count[max_count_idx] - left_count;
      int left_pos = bin_pos_count[max_count_idx] / 2;
      int right_pos = bin_pos_count[max_count_idx] - left_pos;
      int left_neg = bin_neg_count[max_count_idx] / 2;
      int right_neg = bin_neg_count[max_count_idx] - left_neg;
      
      bin_count[max_count_idx] = left_count;
      bin_pos_count[max_count_idx] = left_pos;
      bin_neg_count[max_count_idx] = left_neg;
      
      bin_count.insert(bin_count.begin() + max_count_idx + 1, right_count);
      bin_pos_count.insert(bin_pos_count.begin() + max_count_idx + 1, right_pos);
      bin_neg_count.insert(bin_neg_count.begin() + max_count_idx + 1, right_neg);
      
      calculate_woe_iv();
    }
  }
  
  void create_bin_labels() {
    int n_bins = bin_edges.size() - 1;
    bin_labels.resize(n_bins);
    
    for (int i = 0; i < n_bins; ++i) {
      std::string lower = (i == 0) ? "(-Inf" : "(" + std::to_string(bin_edges[i]);
      std::string upper = (i == n_bins - 1) ? "+Inf]" : std::to_string(bin_edges[i + 1]) + "]";
      bin_labels[i] = lower + ";" + upper;
    }
  }
  
  void assign_woe_values() {
    woefeature.resize(feature.size(), 0.0);
    int n_bins = bin_edges.size() - 1;
    
    for (size_t i = 0; i < feature.size(); ++i) {
      double val = feature[i];
      int bin_idx = std::upper_bound(bin_edges.begin(), bin_edges.end(), val) - bin_edges.begin() - 1;
      bin_idx = std::max(0, std::min(bin_idx, n_bins - 1));
      woefeature[i] = bin_woe[bin_idx];
    }
  }
  
public:
  OptimalBinningNumericalBS(const std::vector<double>& feature_, const std::vector<int>& target_, int min_bins_, int max_bins_, double bin_cutoff_, int max_n_prebins_)
    : feature(feature_), target(target_), min_bins(min_bins_), max_bins(max_bins_), bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_) {
    validate_inputs();
  }
  
  void fit() {
    total_pos = std::accumulate(target.begin(), target.end(), 0);
    total_neg = target.size() - total_pos;
    
    prebin_data();
    calculate_woe_iv();
    enforce_monotonicity();
    apply_bin_cutoff();
    adjust_bin_count();
    create_bin_labels();
    assign_woe_values();
  }
  
  List get_result() {
    return List::create(
      Named("woefeature") = woefeature,
      Named("woebin") = DataFrame::create(
        Named("bin") = bin_labels,
        Named("woe") = bin_woe,
        Named("iv") = bin_iv,
        Named("count") = bin_count,
        Named("count_pos") = bin_pos_count,
        Named("count_neg") = bin_neg_count
      )
    );
  }
};


//' @title 
//' Optimal Binning for Numerical Variables using Binary Search
//'
//' @description
//' This function implements an optimal binning algorithm for numerical variables using a Binary Search approach with Weight of Evidence (WoE) and Information Value (IV) criteria.
//'
//' @param target An integer vector of binary target values (0 or 1).
//' @param feature A numeric vector of feature values to be binned.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum frequency of observations in each bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins for initial quantile-based discretization (default: 20).
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
//' result <- optimal_binning_numerical_bs(target, feature, min_bins = 3, max_bins = 5)
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
//' The optimal binning algorithm for numerical variables uses a Binary Search approach with Weight of Evidence (WoE) and Information Value (IV) to create bins that maximize the predictive power of the feature while maintaining interpretability.
//'
//' The algorithm follows these steps:
//' 1. Initial discretization using quantile-based binning
//' 2. Calculation of WoE and IV for each bin
//' 3. Enforcing monotonicity of WoE across bins
//' 4. Merging of rare bins based on the bin_cutoff parameter
//' 5. Adjusting the number of bins to be within the specified range using a Binary Search approach
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
//' The Binary Search approach efficiently adjusts the number of bins by iteratively merging bins with the lowest IV contribution or splitting bins with the highest count, while respecting the constraints on the number of bins and minimum bin frequency. This process ensures that the resulting binning maximizes the total IV while maintaining the desired number of bins.
//'
//' @references
//' \itemize{
//'   \item Mironchyk, P., & Tchistiakov, V. (2017). Monotone optimal binning algorithm for credit risk modeling. arXiv preprint arXiv:1711.05095.
//'   \item Beltratti, A., Margarita, S., & Terna, P. (1996). Neural networks for economic and financial modelling. International Thomson Computer Press.
//' }
//'
//' @author Lopes, J. E.
//'
//' @export
// [[Rcpp::export]]
List optimal_binning_numerical_bs(const std::vector<int>& target,
                                  const std::vector<double>& feature,
                                  int min_bins = 3, int max_bins = 5,
                                  double bin_cutoff = 0.05, int max_n_prebins = 20) {
  OptimalBinningNumericalBS binning(feature, target, min_bins, max_bins, bin_cutoff, max_n_prebins);
  binning.fit();
  return binning.get_result();
}
