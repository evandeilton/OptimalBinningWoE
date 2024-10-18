#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#include <sstream>
#include <numeric>

using namespace Rcpp;

// Fisher's Exact Test using R's built-in function
double fisher_exact_test(int a, int b, int c, int d) {
  // Construct the contingency table
  NumericMatrix table(2, 2);
  table(0, 0) = a;
  table(0, 1) = b;
  table(1, 0) = c;
  table(1, 1) = d;
  
  // Call R's fisher.test function
  Environment stats = Environment::namespace_env("stats");
  Function fisher_test = stats["fisher.test"];
  
  List result = fisher_test(table, Named("alternative") = "two.sided");
  return as<double>(result["p.value"]);
}

// Class for Optimal Binning Numerical using Fisher's Exact Test
class OptimalBinningNumericalFETB {
public:
  OptimalBinningNumericalFETB(NumericVector target, NumericVector feature,
                              int min_bins, int max_bins, double bin_cutoff, int max_n_prebins,
                              double convergence_threshold, int max_iterations);
  List performBinning();
  
private:
  std::vector<double> target;
  std::vector<double> feature;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  double convergence_threshold;
  int max_iterations;
  bool converged;
  int iterations_run;
  
  void validateInputs();
  void createPrebins();
  void calculateBinStats();
  void mergeBins();
  void enforceMonotonicity();
  void calculateWoE();
  double calculateIV();
  
  std::vector<double> binEdges;
  std::vector<int> binCounts;
  std::vector<int> binPosCounts;
  std::vector<int> binNegCounts;
  std::vector<double> binWoE;
  std::vector<double> binIV;
};

OptimalBinningNumericalFETB::OptimalBinningNumericalFETB(NumericVector target_, NumericVector feature_,
                                                         int min_bins_, int max_bins_, double bin_cutoff_, int max_n_prebins_,
                                                         double convergence_threshold_, int max_iterations_)
  : target(target_.begin(), target_.end()), feature(feature_.begin(), feature_.end()),
    min_bins(min_bins_), max_bins(max_bins_), bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_),
    convergence_threshold(convergence_threshold_), max_iterations(max_iterations_), converged(true), iterations_run(0) {
  validateInputs();
}

void OptimalBinningNumericalFETB::validateInputs() {
  // Check that target is binary
  for (size_t i = 0; i < target.size(); ++i) {
    if (target[i] != 0 && target[i] != 1) {
      stop("Target must be binary (0 or 1).");
    }
  }
  
  // Check min_bins
  if (min_bins < 2) {
    stop("min_bins must be at least 2.");
  }
  
  // Check max_bins
  if (max_bins < min_bins) {
    stop("max_bins must be greater than or equal to min_bins.");
  }
  
  // Check bin_cutoff
  if (bin_cutoff <= 0 || bin_cutoff >= 0.5) {
    stop("bin_cutoff must be between 0 and 0.5.");
  }
  
  // Check max_n_prebins
  if (max_n_prebins < max_bins) {
    stop("max_n_prebins must be greater than or equal to max_bins.");
  }
  
  // Check target and feature sizes
  if (target.size() != feature.size()) {
    stop("Target and feature must have the same length.");
  }
}

void OptimalBinningNumericalFETB::createPrebins() {
  std::vector<double> sorted_feature = feature;
  std::sort(sorted_feature.begin(), sorted_feature.end());
  
  binEdges.clear();
  
  // Start with -infinity
  binEdges.push_back(-std::numeric_limits<double>::infinity());
  
  // Determine step size
  int step = std::max(1, static_cast<int>(sorted_feature.size()) / max_n_prebins);
  
  // Add upper bounds for pre-bins
  for (size_t i = step; i < sorted_feature.size(); i += step) {
    double edge = sorted_feature[i];
    if (edge != binEdges.back()) {
      binEdges.push_back(edge);
    }
    if (binEdges.size() >= static_cast<size_t>(max_n_prebins)) {
      break;
    }
  }
  
  // Ensure the last edge is infinity
  if (binEdges.back() != std::numeric_limits<double>::infinity()) {
    binEdges.push_back(std::numeric_limits<double>::infinity());
  }
}

void OptimalBinningNumericalFETB::calculateBinStats() {
  int n_bins = binEdges.size() - 1;
  binCounts.assign(n_bins, 0);
  binPosCounts.assign(n_bins, 0);
  binNegCounts.assign(n_bins, 0);
  
  for (size_t i = 0; i < feature.size(); i++) {
    double x = feature[i];
    // Find the bin index
    int bin_index = std::upper_bound(binEdges.begin(), binEdges.end(), x) - binEdges.begin() - 1;
    bin_index = std::max(0, std::min(bin_index, n_bins - 1));
    
    // Update counts
    binCounts[bin_index]++;
    if (target[i] == 1) {
      binPosCounts[bin_index]++;
    } else {
      binNegCounts[bin_index]++;
    }
  }
}

void OptimalBinningNumericalFETB::mergeBins() {
  bool binsMerged = true;
  int iterations = 0;
  
  while (binsMerged && static_cast<int>(binEdges.size() - 1) > min_bins && iterations < max_iterations) {
    binsMerged = false;
    double max_p_value = -1.0;
    int merge_index = -1;
    
    // Find the pair of adjacent bins with the highest p-value
    for (size_t i = 0; i < binEdges.size() - 2; i++) {
      int a = binPosCounts[i];
      int b = binNegCounts[i];
      int c = binPosCounts[i + 1];
      int d = binNegCounts[i + 1];
      
      double p_value = fisher_exact_test(a, b, c, d);
      
      if (p_value > max_p_value) {
        max_p_value = p_value;
        merge_index = i;
      }
    }
    
    // If the highest p-value exceeds the cutoff, merge the bins
    if (max_p_value > bin_cutoff && merge_index != -1) {
      // Merge bin at merge_index and merge_index + 1
      binEdges.erase(binEdges.begin() + merge_index + 1);
      binCounts[merge_index] += binCounts[merge_index + 1];
      binPosCounts[merge_index] += binPosCounts[merge_index + 1];
      binNegCounts[merge_index] += binNegCounts[merge_index + 1];
      binCounts.erase(binCounts.begin() + merge_index + 1);
      binPosCounts.erase(binPosCounts.begin() + merge_index + 1);
      binNegCounts.erase(binNegCounts.begin() + merge_index + 1);
      binsMerged = true;
    }
    iterations++;
  }
  
  if (iterations >= max_iterations) {
    converged = false;
  }
  iterations_run += iterations;
}

void OptimalBinningNumericalFETB::calculateWoE() {
  binWoE.clear();
  binIV.clear();
  
  int totalPos = std::accumulate(binPosCounts.begin(), binPosCounts.end(), 0);
  int totalNeg = std::accumulate(binNegCounts.begin(), binNegCounts.end(), 0);
  
  for (size_t i = 0; i < binPosCounts.size(); i++) {
    // Apply continuity correction
    double distPos = (static_cast<double>(binPosCounts[i]) + 0.5) / (totalPos + 0.5 * binPosCounts.size());
    double distNeg = (static_cast<double>(binNegCounts[i]) + 0.5) / (totalNeg + 0.5 * binNegCounts.size());
    
    // Prevent division by zero and log of zero
    if (distPos <= 0) distPos = 1e-10;
    if (distNeg <= 0) distNeg = 1e-10;
    
    double woe = std::log(distPos / distNeg);
    binWoE.push_back(woe);
    binIV.push_back((distPos - distNeg) * woe);
  }
}

void OptimalBinningNumericalFETB::enforceMonotonicity() {
  calculateWoE();
  bool monotonic = false;
  bool increasing = true;
  bool decreasing = true;
  int iterations = 0;
  
  // Determine the overall trend
  for (size_t i = 0; i < binWoE.size() - 1; i++) {
    if (binWoE[i] < binWoE[i + 1]) {
      decreasing = false;
    }
    if (binWoE[i] > binWoE[i + 1]) {
      increasing = false;
    }
  }
  
  // If not monotonic, enforce monotonicity
  if (!increasing && !decreasing) {
    while (!monotonic && static_cast<int>(binEdges.size() - 1) > min_bins && iterations < max_iterations) {
      monotonic = true;
      double trend = binWoE[1] - binWoE[0];
      bool is_increasing = trend >= 0;
      
      for (size_t i = 0; i < binWoE.size() - 1; i++) {
        double current_trend = binWoE[i + 1] - binWoE[i];
        if ((is_increasing && current_trend < 0) || (!is_increasing && current_trend > 0)) {
          // Merge bins i and i+1
          binEdges.erase(binEdges.begin() + i + 1);
          binCounts[i] += binCounts[i + 1];
          binPosCounts[i] += binPosCounts[i + 1];
          binNegCounts[i] += binNegCounts[i + 1];
          binCounts.erase(binCounts.begin() + i + 1);
          binPosCounts.erase(binPosCounts.begin() + i + 1);
          binNegCounts.erase(binNegCounts.begin() + i + 1);
          monotonic = false;
          break;
        }
      }
      calculateWoE();
      iterations++;
    }
    
    if (iterations >= max_iterations) {
      converged = false;
    }
    iterations_run += iterations;
  }
}

double OptimalBinningNumericalFETB::calculateIV() {
  return std::accumulate(binIV.begin(), binIV.end(), 0.0);
}

List OptimalBinningNumericalFETB::performBinning() {
  // Get unique feature values
  std::vector<double> unique_feature = feature;
  std::sort(unique_feature.begin(), unique_feature.end());
  unique_feature.erase(std::unique(unique_feature.begin(), unique_feature.end(),
                                   [](double a, double b) { return std::abs(a - b) < 1e-9; }), unique_feature.end());
  
  // Check if optimization is needed
  if (static_cast<int>(unique_feature.size()) <= min_bins) {
    // No need to optimize; create bins based on unique values
    binEdges.clear();
    binCounts.clear();
    binPosCounts.clear();
    binNegCounts.clear();
    binWoE.clear();
    binIV.clear();
    
    // Set bin edges
    binEdges.push_back(-std::numeric_limits<double>::infinity());
    for (size_t i = 0; i < unique_feature.size() - 1; i++) {
      binEdges.push_back((unique_feature[i] + unique_feature[i + 1]) / 2.0);
    }
    binEdges.push_back(std::numeric_limits<double>::infinity());
    
    calculateBinStats();
    calculateWoE();
    double totalIV = calculateIV();
    
    // Create bin labels
    std::vector<std::string> bin_labels;
    for (size_t i = 0; i < binEdges.size() - 1; i++) {
      std::ostringstream oss;
      oss << "(" << binEdges[i] << "; " << binEdges[i + 1] << "]";
      bin_labels.push_back(oss.str());
    }
    
    // Prepare cutpoints (excluding -Inf and +Inf)
    std::vector<double> cutpoints(binEdges.begin() + 1, binEdges.end() - 1);
    
    // Create woebin List
    return List::create(
      Named("bin") = bin_labels,
      Named("woe") = binWoE,
      Named("iv") = binIV,
      Named("count") = binCounts,
      Named("count_pos") = binPosCounts,
      Named("count_neg") = binNegCounts,
      Named("cutpoints") = cutpoints,
      Named("converged") = converged,
      Named("iterations") = iterations_run
    );
  }
  
  // Proceed with binning
  createPrebins();
  calculateBinStats();
  mergeBins();
  enforceMonotonicity();
  calculateWoE();
  double totalIV = calculateIV();
  
  // Create bin labels
  std::vector<std::string> bin_labels;
  for (size_t i = 0; i < binEdges.size() - 1; i++) {
    std::ostringstream oss;
    oss << "(" << binEdges[i] << "; " << binEdges[i + 1] << "]";
    bin_labels.push_back(oss.str());
  }
  
  // Prepare cutpoints (excluding -Inf and +Inf)
  std::vector<double> cutpoints(binEdges.begin() + 1, binEdges.end() - 1);
  
  // Create woebin List
  return List::create(
    Named("bin") = bin_labels,
    Named("woe") = binWoE,
    Named("iv") = binIV,
    Named("count") = binCounts,
    Named("count_pos") = binPosCounts,
    Named("count_neg") = binNegCounts,
    Named("cutpoints") = cutpoints,
    Named("converged") = converged,
    Named("iterations") = iterations_run
  );
}

//' @title Optimal Binning for Numerical Variables using Fisher's Exact Test
//'
//' @description
//' This function implements an optimal binning algorithm for numerical variables using Fisher's Exact Test. It aims to find the best binning strategy that maximizes the predictive power while ensuring statistical significance between adjacent bins.
//'
//' @param target A numeric vector of binary target values (0 or 1).
//' @param feature A numeric vector of feature values to be binned.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff P-value threshold for merging bins (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins (default: 20).
//' @param convergence_threshold Threshold for convergence (default: 1e-6).
//' @param max_iterations Maximum number of iterations (default: 1000).
//'
//' @return A list containing:
//' \item{bins}{A vector of bin labels}
//' \item{woe}{A numeric vector of Weight of Evidence (WoE) values for each bin}
//' \item{iv}{A numeric vector of Information Value (IV) for each bin}
//' \item{count}{Total count of observations in each bin}
//' \item{count_pos}{Count of positive target observations in each bin}
//' \item{count_neg}{Count of negative target observations in each bin}
//' \item{cutpoints}{Numeric vector of cutpoints used to generate the bins}
//' \item{converged}{Logical value indicating if the algorithm converged}
//' \item{iterations}{Number of iterations run by the algorithm}
//'
//' @details
//' The optimal binning algorithm using Fisher's Exact Test consists of several steps:
//' 1. Pre-binning: The feature is initially divided into a maximum number of bins specified by \code{max_n_prebins}.
//' 2. Bin merging: Adjacent bins are iteratively merged based on the p-value of Fisher's Exact Test.
//' 3. Monotonicity enforcement: Ensures that the Weight of Evidence (WoE) values are monotonic across bins.
//' 4. WoE and IV calculation: Calculates the Weight of Evidence and Information Value for each bin.
//'
//' @examples
//' \dontrun{
//' set.seed(123)
//' target <- sample(0:1, 1000, replace = TRUE)
//' feature <- rnorm(1000)
//' result <- optimal_binning_numerical_fetb(target, feature)
//' print(result$bins)
//' print(result$woe)
//' print(result$iv)
//' }
//'
//' @export
// [[Rcpp::export]]
List optimal_binning_numerical_fetb(NumericVector target,
                                   NumericVector feature,
                                   int min_bins = 3, int max_bins = 5,
                                   double bin_cutoff = 0.05, int max_n_prebins = 20,
                                   double convergence_threshold = 1e-6, int max_iterations = 1000) {
 OptimalBinningNumericalFETB binning(target, feature, min_bins, max_bins, bin_cutoff, max_n_prebins,
                                     convergence_threshold, max_iterations);
 return binning.performBinning();
}

