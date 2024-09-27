#include <Rcpp.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#include <set>
#include <string>
#include <numeric>

using namespace Rcpp;

// [[Rcpp::plugins(openmp)]]

class OptimalBinningNumericalSWB {
private:
  // Input vectors
  NumericVector feature;
  IntegerVector target;

  // Binning parameters
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;

  // Output vectors
  NumericVector woefeature;
  DataFrame woebin;

  // Variables for computation
  std::vector<double> bin_edges;
  std::vector<int> bin_counts;
  std::vector<int> bin_pos_counts;
  std::vector<int> bin_neg_counts;
  std::vector<double> bin_woe;
  std::vector<double> bin_iv;
  std::vector<std::string> bin_labels;

  // Helper functions
  bool is_monotonic(const std::vector<double>& woe, bool& increasing);
  void enforce_monotonicity(int max_iterations = 100);
  void merge_bins(int idx);
  void calculate_woe_iv();

public:
  // Constructor
  OptimalBinningNumericalSWB(NumericVector feature, IntegerVector target, int min_bins, int max_bins, double bin_cutoff, int max_n_prebins);

  // Main function to perform binning
  void fit();

  // Function to retrieve results
  List get_result();
};

OptimalBinningNumericalSWB::OptimalBinningNumericalSWB(NumericVector feature, IntegerVector target, int min_bins, int max_bins, double bin_cutoff, int max_n_prebins)
  : feature(feature), target(target), min_bins(min_bins), max_bins(max_bins), bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins) {

  // Input validation
  if (min_bins < 2) {
    stop("min_bins must be at least 2.");
  }
  if (max_bins < min_bins) {
    stop("max_bins must be greater than or equal to min_bins.");
  }
  if (bin_cutoff < 0 || bin_cutoff > 1) {
    stop("bin_cutoff must be between 0 and 1.");
  }
  if (max_n_prebins < min_bins) {
    stop("max_n_prebins must be greater than or equal to min_bins.");
  }
  if (feature.size() != target.size()) {
    stop("feature and target must have the same length.");
  }

  // Check for NA or infinite values in feature
  for (int i = 0; i < feature.size(); ++i) {
    if (NumericVector::is_na(feature[i]) || std::isinf(feature[i])) {
      stop("Feature contains NA or Inf values.");
    }
  }

  // Check that target contains only 0 and 1
  for (int i = 0; i < target.size(); ++i) {
    if (target[i] != 0 && target[i] != 1) {
      stop("Target variable must be binary (0 or 1).");
    }
  }
}

bool OptimalBinningNumericalSWB::is_monotonic(const std::vector<double>& woe, bool& increasing) {
  if (woe.size() < 2) return true; // A single bin is trivially monotonic

  bool inc = true, dec = true;
  for (size_t i = 1; i < woe.size(); ++i) {
    if (woe[i] < woe[i - 1]) {
      inc = false;
    }
    if (woe[i] > woe[i - 1]) {
      dec = false;
    }
  }

  if (inc) {
    increasing = true;
    return true;
  }
  if (dec) {
    increasing = false;
    return true;
  }
  return false;
}

void OptimalBinningNumericalSWB::merge_bins(int idx) {
  // Merge bin at idx with the next bin (idx + 1)
  if (idx < 0 || idx >= static_cast<int>(bin_counts.size()) - 1) {
    stop("Attempted to merge bins out of range.");
  }

  // Merge counts
  bin_counts[idx] += bin_counts[idx + 1];
  bin_pos_counts[idx] += bin_pos_counts[idx + 1];
  bin_neg_counts[idx] += bin_neg_counts[idx + 1];

  // Remove the next bin's counts
  bin_counts.erase(bin_counts.begin() + idx + 1);
  bin_pos_counts.erase(bin_pos_counts.begin() + idx + 1);
  bin_neg_counts.erase(bin_neg_counts.begin() + idx + 1);

  // Remove the corresponding bin edge
  bin_edges.erase(bin_edges.begin() + idx + 1);
}

void OptimalBinningNumericalSWB::calculate_woe_iv() {
  double total_pos = std::accumulate(bin_pos_counts.begin(), bin_pos_counts.end(), 0.0);
  double total_neg = std::accumulate(bin_neg_counts.begin(), bin_neg_counts.end(), 0.0);

  // Handle cases where total_pos or total_neg is zero
  if (total_pos == 0 || total_neg == 0) {
    stop("Total positive or negative counts are zero, cannot compute WOE/IV.");
  }

  bin_woe.resize(bin_counts.size());
  bin_iv.resize(bin_counts.size());
  bin_labels.resize(bin_counts.size());

  for (size_t i = 0; i < bin_counts.size(); ++i) {
    double dist_pos = bin_pos_counts[i] / total_pos;
    double dist_neg = bin_neg_counts[i] / total_neg;
    if (dist_pos == 0) dist_pos = 1e-10; // Avoid division by zero
    if (dist_neg == 0) dist_neg = 1e-10; // Avoid division by zero
    bin_woe[i] = std::log(dist_pos / dist_neg);
    bin_iv[i] = (dist_pos - dist_neg) * bin_woe[i];

    // Create bin labels
    std::string left = (bin_edges[i] == -std::numeric_limits<double>::infinity()) ? "(-Inf" : "(" + std::to_string(bin_edges[i]);
    std::string right = (bin_edges[i + 1] == std::numeric_limits<double>::infinity()) ? "+Inf]" : std::to_string(bin_edges[i + 1]) + "]";
    bin_labels[i] = left + ";" + right;
  }
}

void OptimalBinningNumericalSWB::enforce_monotonicity(int max_iterations) {
  int iteration = 0;
  bool is_mono = false;
  bool increasing = true;

  while (iteration < max_iterations) {
    calculate_woe_iv();
    is_mono = is_monotonic(bin_woe, increasing);
    if (is_mono) break;

    // Find the first pair that violates monotonicity
    int merge_idx = -1;
    for (size_t i = 1; i < bin_woe.size(); ++i) {
      if (increasing) {
        if (bin_woe[i] < bin_woe[i - 1]) {
          merge_idx = i - 1;
          break;
        }
      } else {
        if (bin_woe[i] > bin_woe[i - 1]) {
          merge_idx = i - 1;
          break;
        }
      }
    }

    if (merge_idx != -1) {
      merge_bins(merge_idx);
      iteration++;
    } else {
      // If no specific violation is found, merge the pair with the smallest WOE difference
      double min_diff = std::numeric_limits<double>::max();
      int min_idx = -1;
      for (size_t i = 0; i < bin_woe.size() - 1; ++i) {
        double diff = std::abs(bin_woe[i] - bin_woe[i + 1]);
        if (diff < min_diff) {
          min_diff = diff;
          min_idx = i;
        }
      }
      if (min_idx != -1) {
        merge_bins(min_idx);
        iteration++;
      } else {
        // Cannot merge further, break to prevent infinite loop
        break;
      }
    }
  }

  if (iteration == max_iterations) {
    Rcpp::warning("Maximum iterations reached while enforcing monotonicity.");
  }
}

void OptimalBinningNumericalSWB::fit() {
  int n = feature.size();

  // Create pre-bins using quantiles
  std::vector<double> sorted_feature = as<std::vector<double>>(feature);
  std::sort(sorted_feature.begin(), sorted_feature.end());

  std::set<double> cut_points;
  for (int i = 1; i < max_n_prebins; ++i) {
    double p = static_cast<double>(i) / max_n_prebins;
    size_t idx = static_cast<size_t>(std::floor(p * (n - 1)));
    cut_points.insert(sorted_feature[idx]);
  }

  // Construct bin edges with -Inf and +Inf
  bin_edges.clear();
  bin_edges.push_back(-std::numeric_limits<double>::infinity());
  bin_edges.insert(bin_edges.end(), cut_points.begin(), cut_points.end());
  bin_edges.push_back(std::numeric_limits<double>::infinity());

  int num_bins = bin_edges.size() - 1;
  bin_counts.resize(num_bins, 0);
  bin_pos_counts.resize(num_bins, 0);
  bin_neg_counts.resize(num_bins, 0);

  // Assign observations to bins using OpenMP for parallelism
  int nthreads = 1;
#ifdef _OPENMP
  nthreads = omp_get_max_threads();
#endif

  // Initialize thread-local storage for counts
  std::vector<std::vector<int>> thread_bin_counts(nthreads, std::vector<int>(num_bins, 0));
  std::vector<std::vector<int>> thread_bin_pos_counts(nthreads, std::vector<int>(num_bins, 0));
  std::vector<std::vector<int>> thread_bin_neg_counts(nthreads, std::vector<int>(num_bins, 0));

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < n; ++i) {
    double val = feature[i];
    int thread_num = 0;
#ifdef _OPENMP
    thread_num = omp_get_thread_num();
#endif
    // Find the bin index
    int bin_idx = std::upper_bound(bin_edges.begin(), bin_edges.end(), val) - bin_edges.begin() - 1;
    // Safety check for bin_idx
    if (bin_idx < 0) bin_idx = 0;
    if (bin_idx >= num_bins) bin_idx = num_bins - 1;

    thread_bin_counts[thread_num][bin_idx]++;
    if (target[i] == 1) {
      thread_bin_pos_counts[thread_num][bin_idx]++;
    } else { // target[i] == 0 (validated in constructor)
      thread_bin_neg_counts[thread_num][bin_idx]++;
    }
  }

  // Aggregate counts from all threads
  for (int i = 0; i < num_bins; ++i) {
    for (int t = 0; t < nthreads; ++t) {
      bin_counts[i] += thread_bin_counts[t][i];
      bin_pos_counts[i] += thread_bin_pos_counts[t][i];
      bin_neg_counts[i] += thread_bin_neg_counts[t][i];
    }
  }

  // Merge bins with counts less than bin_cutoff
  int total_counts = n;
  double cutoff_count = bin_cutoff * total_counts;

  std::vector<double> new_bin_edges;
  std::vector<int> new_bin_counts;
  std::vector<int> new_bin_pos_counts;
  std::vector<int> new_bin_neg_counts;

  new_bin_edges.push_back(bin_edges[0]);
  int i_bin = 0;
  while (i_bin < num_bins) {
    int current_count = bin_counts[i_bin];
    int current_pos_count = bin_pos_counts[i_bin];
    int current_neg_count = bin_neg_counts[i_bin];
    double current_right_edge = bin_edges[i_bin + 1];

    // Merge with subsequent bins if current count is below cutoff and more bins are available
    while (current_count < cutoff_count && i_bin + 1 < num_bins) {
      i_bin++;
      current_count += bin_counts[i_bin];
      current_pos_count += bin_pos_counts[i_bin];
      current_neg_count += bin_neg_counts[i_bin];
      current_right_edge = bin_edges[i_bin + 1];
    }

    new_bin_edges.push_back(current_right_edge);
    new_bin_counts.push_back(current_count);
    new_bin_pos_counts.push_back(current_pos_count);
    new_bin_neg_counts.push_back(current_neg_count);
    i_bin++;
  }

  // Update bin variables
  bin_edges = new_bin_edges;
  bin_counts = new_bin_counts;
  bin_pos_counts = new_bin_pos_counts;
  bin_neg_counts = new_bin_neg_counts;
  num_bins = bin_counts.size();

  // Ensure number of bins is between min_bins and max_bins
  // Merge the smallest bins first to reduce the number
  while (num_bins > max_bins) {
    // Find the pair of adjacent bins with the smallest combined count
    int min_count = bin_counts[0] + bin_counts[1];
    int merge_idx = 0;
    for (int i = 1; i < num_bins - 1; ++i) {
      int combined_count = bin_counts[i] + bin_counts[i + 1];
      if (combined_count < min_count) {
        min_count = combined_count;
        merge_idx = i;
      }
    }
    // Merge the identified pair
    merge_bins(merge_idx);
    num_bins--;
  }

  // Ensure at least min_bins by merging the largest bins first
  while (num_bins < min_bins) {
    // Find the pair of adjacent bins with the largest combined count
    if (bin_counts.size() < 2) break; // Cannot merge further
    int max_count = bin_counts[0] + bin_counts[1];
    int merge_idx = 0;
    for (int i = 1; i < num_bins - 1; ++i) {
      int combined_count = bin_counts[i] + bin_counts[i + 1];
      if (combined_count > max_count) {
        max_count = combined_count;
        merge_idx = i;
      }
    }
    // Merge the identified pair
    merge_bins(merge_idx);
    num_bins--;
  }

  // Recalculate WOE and IV after initial merging
  calculate_woe_iv();

  // Enforce monotonicity with a maximum number of iterations to prevent infinite loops
  enforce_monotonicity(100);

  // Final check to ensure the number of bins does not exceed max_bins after enforcing monotonicity
  while (bin_counts.size() > static_cast<size_t>(max_bins)) {
    // Merge the pair with the smallest WOE difference
    double min_diff = std::numeric_limits<double>::max();
    int merge_idx = -1;
    for (size_t i = 0; i < bin_woe.size() - 1; ++i) {
      double diff = std::abs(bin_woe[i] - bin_woe[i + 1]);
      if (diff < min_diff) {
        min_diff = diff;
        merge_idx = i;
      }
    }
    if (merge_idx != -1) {
      merge_bins(merge_idx);
      calculate_woe_iv();
      enforce_monotonicity(100);
    } else {
      break; // Cannot merge further
    }
  }

  // Assign WOE values to the feature
  woefeature = NumericVector(n);

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < n; ++i) {
    double val = feature[i];
    int bin_idx = std::upper_bound(bin_edges.begin(), bin_edges.end(), val) - bin_edges.begin() - 1;
    // Safety check for bin_idx
    if (bin_idx < 0) bin_idx = 0;
    if (bin_idx >= static_cast<int>(bin_woe.size())) bin_idx = bin_woe.size() - 1;
    woefeature[i] = bin_woe[bin_idx];
  }

  // Prepare woebin DataFrame
  woebin = DataFrame::create(
    Named("bin") = bin_labels,
    Named("woe") = bin_woe,
    Named("iv") = bin_iv,
    Named("count") = bin_counts,
    Named("count_pos") = bin_pos_counts,
    Named("count_neg") = bin_neg_counts
  );
}

List OptimalBinningNumericalSWB::get_result() {
  return List::create(
    Named("woefeature") = woefeature,
    Named("woebin") = woebin
  );
}


//' @title Optimal Binning for Numerical Variables using Sliding Window Binning
//'
//' @description
//' This function implements an optimal binning algorithm for numerical variables using 
//' a Sliding Window Binning (SWB) approach with Weight of Evidence (WoE) and 
//' Information Value (IV) criteria.
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
//' @details
//' The optimal binning algorithm for numerical variables uses a Sliding Window Binning 
//' approach with Weight of Evidence (WoE) and Information Value (IV) to create bins 
//' that maximize the predictive power of the feature while maintaining interpretability.
//'
//' The algorithm follows these steps:
//' 1. Initial discretization using quantile-based binning
//' 2. Merging of rare bins based on the bin_cutoff parameter
//' 3. Adjustment of bin count to be within the specified range (min_bins to max_bins)
//' 4. Calculation of WoE and IV for each bin
//' 5. Enforcement of monotonicity in WoE across bins
//' 6. Final adjustment to ensure the number of bins does not exceed max_bins
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
//' The total IV for the feature is the sum of IVs across all bins:
//'
//' \deqn{IV_{total} = \sum_{i=1}^{n} IV_i}
//'
//' The SWB approach ensures that the resulting binning maximizes the separation between 
//' classes while maintaining the desired number of bins and respecting the minimum bin 
//' frequency constraint.
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
//' result <- optimal_binning_numerical_swb(target, feature, min_bins = 3, max_bins = 5)
//'
//' # View binning results
//' print(result$woebin)
//'
//' # Plot WoE transformation
//' plot(feature, result$woefeature, main = "WoE Transformation", 
//'      xlab = "Original Feature", ylab = "WoE")
//' }
//'
//' @references
//' \itemize{
//' \item Mironchyk, P., & Tchistiakov, V. (2017). Monotone optimal binning algorithm 
//'       for credit risk modeling. arXiv preprint arXiv:1711.05095.
//' \item Beltratti, A., Margarita, S., & Terna, P. (1996). Neural networks for economic 
//'       and financial modelling. International Thomson Computer Press.
//' }
//'
//' @author Lopes, J. E.
//'
//' @export
// [[Rcpp::export]]
List optimal_binning_numerical_swb(IntegerVector target, NumericVector feature, int min_bins = 3, int max_bins = 5, double bin_cutoff = 0.05, int max_n_prebins = 20) {
  OptimalBinningNumericalSWB ob(feature, target, min_bins, max_bins, bin_cutoff, max_n_prebins);
  ob.fit();
  return ob.get_result();
}