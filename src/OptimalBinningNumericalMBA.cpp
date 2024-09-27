// [[Rcpp::plugins(openmp)]]
#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cmath>
#include <limits>

using namespace Rcpp;

class OptimalBinningNumericalMBA {
private:
  NumericVector feature;
  IntegerVector target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  int n_threads;

  // Binning results
  std::vector<double> bin_edges;
  std::vector<double> woe_values;
  std::vector<double> iv_values;
  std::vector<int> bin_counts;
  std::vector<int> bin_count_pos;
  std::vector<int> bin_count_neg;
  std::vector<std::string> bin_labels;

public:
  OptimalBinningNumericalMBA(NumericVector feature, IntegerVector target,
                             int min_bins, int max_bins, double bin_cutoff,
                             int max_n_prebins, int n_threads)
    : feature(feature), target(target),
      min_bins(min_bins), max_bins(max_bins),
      bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins),
      n_threads(n_threads) {}

  void fit() {
    // Input validation
    if (min_bins < 2) {
      stop("min_bins must be at least 2");
    }
    if (max_bins < min_bins) {
      stop("max_bins must be greater than or equal to min_bins");
    }
    if (bin_cutoff < 0 || bin_cutoff > 0.5) {
      stop("bin_cutoff must be between 0 and 0.5");
    }
    if (max_n_prebins < min_bins) {
      stop("max_n_prebins must be greater than or equal to min_bins");
    }
    if (feature.size() != target.size()) {
      stop("feature and target must have the same length");
    }

    // Ensure target is binary
    IntegerVector unique_targets = unique(target);
    if (unique_targets.size() != 2 || !(is_true(any(unique_targets == 0)) && is_true(any(unique_targets == 1)))) {
      stop("target must be binary with values 0 and 1");
    }

    // Pre-binning
    prebinning();

    // Merge bins with low counts
    mergeRareBins();

    // Monotonic binning
    monotonicBinning();

    // Compute WOE and IV
    computeWOEIV();
  }

  List transform() {
    // Map feature values to WOE values
    NumericVector woefeature(feature.size());

#pragma omp parallel for num_threads(n_threads)
    for (int i = 0; i < feature.size(); ++i) {
      double val = feature[i];
      int bin_idx = findBin(val);
      woefeature[i] = woe_values[bin_idx];
    }

    // Prepare woebin DataFrame
    DataFrame woebin = DataFrame::create(
      Named("bin") = bin_labels,
      Named("woe") = woe_values,
      Named("iv") = iv_values,
      Named("count") = bin_counts,
      Named("count_pos") = bin_count_pos,
      Named("count_neg") = bin_count_neg
    );

    return List::create(
      Named("woefeature") = woefeature,
      Named("woebin") = woebin
    );
  }

private:
  void prebinning() {
    // Create initial pre-bins
    NumericVector sorted_feature = clone(feature);
    std::sort(sorted_feature.begin(), sorted_feature.end());

    // Determine bin edges using quantiles
    int n = sorted_feature.size();
    int n_bins = std::min(max_n_prebins, n);

    // Use quantiles to get bin edges
    NumericVector quantiles = NumericVector(n_bins - 1);
    for (int i = 1; i < n_bins; ++i) {
      double q = (double)i / n_bins;
      quantiles[i - 1] = sorted_feature[std::floor(q * (n - 1))];
    }

    // Remove duplicate quantiles to avoid zero-width bins
    std::vector<double> unique_edges;
    unique_edges.push_back(-std::numeric_limits<double>::infinity());
    for (double q : quantiles) {
      if (q > unique_edges.back()) {
        unique_edges.push_back(q);
      }
    }
    unique_edges.push_back(std::numeric_limits<double>::infinity());

    bin_edges = unique_edges;

    // Initialize bin counts
    int n_bins_actual = bin_edges.size() - 1;
    bin_counts = std::vector<int>(n_bins_actual, 0);
    bin_count_pos = std::vector<int>(n_bins_actual, 0);
    bin_count_neg = std::vector<int>(n_bins_actual, 0);

    // Assign observations to bins
#pragma omp parallel for num_threads(n_threads)
    for (int i = 0; i < feature.size(); ++i) {
      double val = feature[i];
      int bin_idx = findBin(val);

#pragma omp atomic
      bin_counts[bin_idx] += 1;

      if (target[i] == 1) {
#pragma omp atomic
        bin_count_pos[bin_idx] += 1;
      } else {
#pragma omp atomic
        bin_count_neg[bin_idx] += 1;
      }
    }

    // Initialize bin labels
    updateBinLabels();
  }

  void mergeRareBins() {
    // Merge bins with counts less than bin_cutoff proportion
    int total_count = feature.size();
    int n_bins = bin_counts.size();
    bool merged = true;

    while (merged && n_bins > min_bins) {
      merged = false;
      for (int i = 0; i < n_bins; ++i) {
        double bin_prop = (double)bin_counts[i] / total_count;
        if (bin_prop < bin_cutoff) {
          // Merge with adjacent bin with smallest count
          int merge_with = -1;
          if (i == 0) {
            merge_with = i + 1;
          } else if (i == n_bins - 1) {
            merge_with = i - 1;
          } else {
            if (bin_counts[i - 1] <= bin_counts[i + 1]) {
              merge_with = i - 1;
            } else {
              merge_with = i + 1;
            }
          }

          // Merge bins
          mergeBins(i, merge_with);
          merged = true;
          n_bins = bin_counts.size();
          break; // Need to restart loop after merging
        }
      }
    }
  }

  void mergeBins(int idx1, int idx2) {
    if (idx1 == idx2) return;
    if (idx1 > idx2) std::swap(idx1, idx2);

    // Merge bins idx1 and idx2
    bin_edges.erase(bin_edges.begin() + idx2);
    bin_counts[idx1] += bin_counts[idx2];
    bin_count_pos[idx1] += bin_count_pos[idx2];
    bin_count_neg[idx1] += bin_count_neg[idx2];

    // Remove merged bin
    bin_counts.erase(bin_counts.begin() + idx2);
    bin_count_pos.erase(bin_count_pos.begin() + idx2);
    bin_count_neg.erase(bin_count_neg.begin() + idx2);

    // Update bin labels
    updateBinLabels();
  }

  void monotonicBinning() {
    // Compute initial WOE values
    computeWOE();

    // Check monotonicity and merge bins if necessary
    bool is_monotonic = checkMonotonicity();
    while (!is_monotonic && bin_counts.size() > min_bins) {
      // Find the pair of bins with the smallest difference in WOE to merge
      int idx_to_merge = findBinToMerge();

      // Merge bins idx_to_merge and idx_to_merge + 1
      mergeBins(idx_to_merge, idx_to_merge + 1);

      // Recompute WOE values
      computeWOE();

      // Check monotonicity again
      is_monotonic = checkMonotonicity();
    }

    // Ensure max_bins constraint
    while (bin_counts.size() > max_bins) {
      // Merge bins with smallest total count
      int idx_to_merge = findSmallestBin();
      if (idx_to_merge == bin_counts.size() - 1) {
        mergeBins(idx_to_merge - 1, idx_to_merge);
      } else {
        mergeBins(idx_to_merge, idx_to_merge + 1);
      }
      computeWOE();
    }
  }

  void computeWOE() {
    // Total counts
    double total_pos = std::accumulate(bin_count_pos.begin(), bin_count_pos.end(), 0.0);
    double total_neg = std::accumulate(bin_count_neg.begin(), bin_count_neg.end(), 0.0);

    woe_values = std::vector<double>(bin_counts.size());
    for (size_t i = 0; i < bin_counts.size(); ++i) {
      double dist_pos = bin_count_pos[i] / total_pos;
      double dist_neg = bin_count_neg[i] / total_neg;
      if (dist_pos == 0) dist_pos = 1e-8;
      if (dist_neg == 0) dist_neg = 1e-8;
      woe_values[i] = std::log(dist_pos / dist_neg);
    }
  }

  void computeWOEIV() {
    // Total counts
    double total_pos = std::accumulate(bin_count_pos.begin(), bin_count_pos.end(), 0.0);
    double total_neg = std::accumulate(bin_count_neg.begin(), bin_count_neg.end(), 0.0);

    woe_values = std::vector<double>(bin_counts.size());
    iv_values = std::vector<double>(bin_counts.size());

    for (size_t i = 0; i < bin_counts.size(); ++i) {
      double dist_pos = bin_count_pos[i] / total_pos;
      double dist_neg = bin_count_neg[i] / total_neg;
      if (dist_pos == 0) dist_pos = 1e-8;
      if (dist_neg == 0) dist_neg = 1e-8;
      woe_values[i] = std::log(dist_pos / dist_neg);
      iv_values[i] = (dist_pos - dist_neg) * woe_values[i];
    }
  }

  bool checkMonotonicity() {
    // Check if WOE values are monotonic
    bool increasing = true;
    bool decreasing = true;
    for (size_t i = 1; i < woe_values.size(); ++i) {
      if (woe_values[i] < woe_values[i - 1]) {
        increasing = false;
      }
      if (woe_values[i] > woe_values[i - 1]) {
        decreasing = false;
      }
    }
    return increasing || decreasing;
  }

  int findBinToMerge() {
    // Find the pair of bins with smallest difference in WOE
    double min_diff = std::numeric_limits<double>::infinity();
    int idx_to_merge = 0;
    for (size_t i = 0; i < woe_values.size() - 1; ++i) {
      double diff = std::abs(woe_values[i + 1] - woe_values[i]);
      if (diff < min_diff) {
        min_diff = diff;
        idx_to_merge = i;
      }
    }
    return idx_to_merge;
  }

  int findSmallestBin() {
    // Find the bin with the smallest count
    int min_count = bin_counts[0];
    int idx = 0;
    for (size_t i = 1; i < bin_counts.size(); ++i) {
      if (bin_counts[i] < min_count) {
        min_count = bin_counts[i];
        idx = i;
      }
    }
    return idx;
  }

  int findBin(double value) {
    // Find the bin index for a given value
    auto it = std::upper_bound(bin_edges.begin(), bin_edges.end(), value);
    return std::distance(bin_edges.begin(), it) - 1;
  }

  void updateBinLabels() {
    bin_labels.clear();
    for (size_t i = 0; i < bin_edges.size() - 1; ++i) {
      std::ostringstream oss;
      oss << "[" << (i == 0 ? "-Inf" : std::to_string(bin_edges[i])) << ", "
          << (i == bin_edges.size() - 2 ? "Inf" : std::to_string(bin_edges[i + 1])) << ")";
      bin_labels.push_back(oss.str());
    }
  }
};


//' @title Optimal Binning for Numerical Variables using Modified Binning Algorithm (MBA)
//' 
//' @description This function implements an optimal binning algorithm for numerical variables using a Modified Binning Algorithm (MBA) approach.
//' 
//' @param target An integer vector of binary target values (0 or 1).
//' @param feature A numeric vector of feature values to be binned.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum frequency for a bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins (default: 20).
//' @param n_threads Number of threads to use for parallel processing (default: 1).
//' 
//' @return A list containing two elements:
//' \item{woefeature}{A numeric vector of Weight of Evidence (WoE) transformed feature values.}
//' \item{woebin}{A data frame containing bin information, including bin labels, WoE, Information Value (IV), and counts.}
//' 
//' @details
//' The Modified Binning Algorithm (MBA) is an advanced method for optimal binning of numerical variables. It aims to create bins that maximize the predictive power of the feature while maintaining monotonicity in the Weight of Evidence (WoE) values and respecting user-defined constraints.
//' 
//' The algorithm works through several steps:
//' 1. Pre-binning: Initially divides the feature into a large number of bins (max_n_prebins) using quantiles.
//' 2. Merging rare bins: Combines bins with frequencies below the bin_cutoff threshold.
//' 3. Monotonic binning: Merges adjacent bins to ensure monotonic WoE values.
//' 4. Respecting bin constraints: Ensures the final number of bins is between min_bins and max_bins.
//' 
//' The algorithm uses the difference in Weight of Evidence (WoE) values as a criterion for merging bins, aiming to maintain monotonicity while preserving the predictive power of the feature.
//' 
//' Weight of Evidence (WoE) is calculated as:
//' \deqn{WoE = \ln\left(\frac{\text{% of positive cases}}{\text{% of negative cases}}\right)}
//' 
//' Information Value (IV) is calculated as:
//' \deqn{IV = (\text{% of positive cases} - \text{% of negative cases}) \times WoE}
//' 
//' The MBA method ensures that the resulting bins have monotonic WoE values, which is often desirable in credit scoring and risk modeling applications. It also provides flexibility in terms of the number of bins and the minimum bin frequency, allowing users to balance between predictive power and model interpretability.
//' 
//' @examples
//' \dontrun{
//' # Create sample data
//' set.seed(123)
//' target <- sample(0:1, 1000, replace = TRUE)
//' feature <- rnorm(1000)
//' 
//' # Run optimal binning
//' result <- optimal_binning_numerical_mba(target, feature)
//' 
//' # View results
//' head(result$woefeature)
//' print(result$woebin)
//' }
//' 
//' @references
//' \itemize{
//' \item Mironchyk, P., & Tchistiakov, V. (2017). Monotone optimal binning algorithm for credit scoring modeling. arXiv preprint arXiv:1711.07139.
//' \item Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). Credit scoring and its applications. SIAM.
//' }
//' 
//' @author Lopes, J. E.
//' 
//' @export
// [[Rcpp::export]]
List optimal_binning_numerical_mba(IntegerVector target,
                                   NumericVector feature,
                                   int min_bins = 3, int max_bins = 5,
                                   double bin_cutoff = 0.05, int max_n_prebins = 20,
                                   int n_threads = 1) {
  OptimalBinningNumericalMBA ob(feature, target, min_bins, max_bins, bin_cutoff, max_n_prebins, n_threads);
  ob.fit();
  return ob.transform();
}
