#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#include <sstream>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

class OptimalBinningNumericalFETB {
public:
  OptimalBinningNumericalFETB(NumericVector target, NumericVector feature,
                              int min_bins, int max_bins, double bin_cutoff, int max_n_prebins);
  List performBinning();
  
private:
  std::vector<double> target;
  std::vector<double> feature;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  
  void validateInputs();
  void createPrebins();
  void calculateBinStats();
  void mergeBins();
  void enforceMonotonicity();
  void calculateWoE();
  double calculateIV();
  double fisherTest(int a, int b, int c, int d);
  
  std::vector<double> binEdges;
  std::vector<int> binCounts;
  std::vector<int> binPosCounts;
  std::vector<int> binNegCounts;
  std::vector<double> binWoE;
  std::vector<double> binIV;
};

OptimalBinningNumericalFETB::OptimalBinningNumericalFETB(NumericVector target, NumericVector feature,
                                                         int min_bins, int max_bins, double bin_cutoff, int max_n_prebins)
  : target(target.begin(), target.end()), feature(feature.begin(), feature.end()), min_bins(min_bins), max_bins(max_bins),
    bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins) {
  validateInputs();
}

void OptimalBinningNumericalFETB::validateInputs() {
  for (size_t i = 0; i < target.size(); ++i) {
    if (target[i] != 0 && target[i] != 1) {
      stop("Target must be binary (0 or 1).");
    }
  }
  if (min_bins < 2) {
    stop("min_bins must be at least 2.");
  }
  if (max_bins < min_bins) {
    stop("max_bins must be greater than or equal to min_bins.");
  }
  if (bin_cutoff <= 0 || bin_cutoff >= 0.5) {
    stop("bin_cutoff must be between 0 and 0.5.");
  }
  if (max_n_prebins < max_bins) {
    stop("max_n_prebins must be greater than or equal to max_bins.");
  }
  if (target.size() != feature.size()) {
    stop("Target and feature must have the same length.");
  }
}

void OptimalBinningNumericalFETB::createPrebins() {
  std::vector<double> sorted_feature = feature;
  std::sort(sorted_feature.begin(), sorted_feature.end());
  
  binEdges.clear();
  binEdges.push_back(-std::numeric_limits<double>::infinity());
  
  int step = std::max(1, (int)sorted_feature.size() / max_n_prebins);
  for (size_t i = step; i < sorted_feature.size(); i += step) {
    if (sorted_feature[i] != binEdges.back() && binEdges.size() < max_n_prebins) {
      binEdges.push_back(sorted_feature[i]);
    }
  }
  
  if (binEdges.back() != std::numeric_limits<double>::infinity()) {
    binEdges.push_back(std::numeric_limits<double>::infinity());
  }
}

void OptimalBinningNumericalFETB::calculateBinStats() {
  int n_bins = binEdges.size() - 1;
  binCounts.resize(n_bins, 0);
  binPosCounts.resize(n_bins, 0);
  binNegCounts.resize(n_bins, 0);
  
#pragma omp parallel for
  for (size_t i = 0; i < feature.size(); i++) {
    double x = feature[i];
    int bin_index = std::lower_bound(binEdges.begin(), binEdges.end(), x) - binEdges.begin() - 1;
    if (bin_index < 0) bin_index = 0;
    if (bin_index >= n_bins) bin_index = n_bins - 1;
    
#pragma omp atomic
    binCounts[bin_index]++;
    
    if (target[i] == 1) {
#pragma omp atomic
      binPosCounts[bin_index]++;
    } else {
#pragma omp atomic
      binNegCounts[bin_index]++;
    }
  }
}

double OptimalBinningNumericalFETB::fisherTest(int a, int b, int c, int d) {
  int n = a + b + c + d;
  double p = 1.0;
  double p_cutoff = 1.0;
  
  for (int i = 0; i <= std::min(a, b); i++) {
    double p_current = 1.0;
    for (int j = 0; j < a + b; j++) p_current *= (double)(a + b - j) / (n - j);
    for (int j = 0; j < a; j++) p_current /= (double)(a - j) / (a + c - j);
    for (int j = 0; j < b; j++) p_current /= (double)(b - j) / (b + d - j);
    
    if (i == 0) p_cutoff = p_current;
    if (p_current <= p_cutoff) p += p_current;
  }
  
  return p;
}

void OptimalBinningNumericalFETB::mergeBins() {
  bool binsMerged = true;
  
  while (binsMerged && binEdges.size() - 1 > min_bins) {
    binsMerged = false;
    double max_p_value = 0.0;
    int merge_index = -1;
    
    for (size_t i = 0; i < binEdges.size() - 2; i++) {
      int a = binPosCounts[i];
      int b = binNegCounts[i];
      int c = binPosCounts[i + 1];
      int d = binNegCounts[i + 1];
      
      double p_value = fisherTest(a, b, c, d);
      
      if (p_value > max_p_value) {
        max_p_value = p_value;
        merge_index = i;
      }
    }
    
    if (max_p_value > bin_cutoff) {
      binEdges.erase(binEdges.begin() + merge_index + 1);
      binCounts[merge_index] += binCounts[merge_index + 1];
      binPosCounts[merge_index] += binPosCounts[merge_index + 1];
      binNegCounts[merge_index] += binNegCounts[merge_index + 1];
      binCounts.erase(binCounts.begin() + merge_index + 1);
      binPosCounts.erase(binPosCounts.begin() + merge_index + 1);
      binNegCounts.erase(binNegCounts.begin() + merge_index + 1);
      binsMerged = true;
    } else {
      binsMerged = false;
    }
  }
}

void OptimalBinningNumericalFETB::calculateWoE() {
  int totalPos = std::accumulate(binPosCounts.begin(), binPosCounts.end(), 0);
  int totalNeg = std::accumulate(binNegCounts.begin(), binNegCounts.end(), 0);
  binWoE.clear();
  binIV.clear();
  
  for (size_t i = 0; i < binPosCounts.size(); i++) {
    double distPos = (double)(binPosCounts[i] + 0.5) / (totalPos + 0.5);
    double distNeg = (double)(binNegCounts[i] + 0.5) / (totalNeg + 0.5);
    
    double woe = std::log(distPos / distNeg);
    binWoE.push_back(woe);
    binIV.push_back((distPos - distNeg) * woe);
  }
}

void OptimalBinningNumericalFETB::enforceMonotonicity() {
  calculateWoE();
  bool monotonic = false;
  
  while (!monotonic && binEdges.size() - 1 > min_bins) {
    monotonic = true;
    for (size_t i = 0; i < binWoE.size() - 1; i++) {
      if (binWoE[i] > binWoE[i + 1]) {
        monotonic = false;
        binEdges.erase(binEdges.begin() + i + 1);
        binCounts[i] += binCounts[i + 1];
        binPosCounts[i] += binPosCounts[i + 1];
        binNegCounts[i] += binNegCounts[i + 1];
        binCounts.erase(binCounts.begin() + i + 1);
        binPosCounts.erase(binPosCounts.begin() + i + 1);
        binNegCounts.erase(binNegCounts.begin() + i + 1);
        break;
      }
    }
    calculateWoE();
  }
}

double OptimalBinningNumericalFETB::calculateIV() {
  return std::accumulate(binIV.begin(), binIV.end(), 0.0);
}

List OptimalBinningNumericalFETB::performBinning() {
  createPrebins();
  calculateBinStats();
  mergeBins();
  enforceMonotonicity();
  calculateWoE();
  double totalIV = calculateIV();
  
  NumericVector woefeature(feature.size());
  int n_bins = binEdges.size() - 1;
  
#pragma omp parallel for
  for (size_t i = 0; i < feature.size(); i++) {
    double x = feature[i];
    int bin_index = std::lower_bound(binEdges.begin(), binEdges.end(), x) - binEdges.begin() - 1;
    if (bin_index < 0) bin_index = 0;
    if (bin_index >= n_bins) bin_index = n_bins - 1;
    woefeature[i] = binWoE[bin_index];
  }
  
  CharacterVector bin_labels(n_bins);
  for (int i = 0; i < n_bins; i++) {
    std::ostringstream oss;
    oss << "(" << binEdges[i] << ";" << binEdges[i + 1] << "]";
    bin_labels[i] = oss.str();
  }
  
  DataFrame woebin = DataFrame::create(
    _["bin"] = bin_labels,
    _["woe"] = NumericVector(binWoE.begin(), binWoE.end()),
    _["iv"] = NumericVector(binIV.begin(), binIV.end()),
    _["count"] = IntegerVector(binCounts.begin(), binCounts.end()),
    _["count_pos"] = IntegerVector(binPosCounts.begin(), binPosCounts.end()),
    _["count_neg"] = IntegerVector(binNegCounts.begin(), binNegCounts.end())
  );
  
  return List::create(
    _["woefeature"] = woefeature,
    _["woebin"] = woebin,
    _["totalIV"] = totalIV
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
//' 
//' @return A list containing:
//' \item{woefeature}{A numeric vector of Weight of Evidence (WoE) values for each observation}
//' \item{woebin}{A data frame with binning information, including bin ranges, WoE, IV, and counts}
//' \item{totalIV}{The total Information Value for the binning}
//' 
//' @details
//' The optimal binning algorithm using Fisher's Exact Test consists of several steps:
//' 
//' 1. Pre-binning: The feature is initially divided into a maximum number of bins specified by \code{max_n_prebins}.
//' 2. Bin merging: Adjacent bins are iteratively merged based on the p-value of Fisher's Exact Test.
//' 3. Monotonicity enforcement: Ensures that the Weight of Evidence (WoE) values are monotonic across bins.
//' 4. WoE and IV calculation: Calculates the Weight of Evidence and Information Value for each bin.
//' 
//' Fisher's Exact Test is used to determine if there is a significant difference in the proportion of positive cases between adjacent bins. The test is performed on a 2x2 contingency table:
//' 
//' \deqn{
//' \begin{array}{|c|c|c|}
//' \hline
//'  & \text{Positive} & \text{Negative} \\
//' \hline
//' \text{Bin 1} & a & b \\
//' \hline
//' \text{Bin 2} & c & d \\
//' \hline
//' \end{array}
//' }
//' 
//' The p-value from this test is used to decide whether to merge adjacent bins.
//' 
//' The Weight of Evidence (WoE) for each bin is calculated as:
//' 
//' \deqn{WoE = \ln\left(\frac{P(X|Y=1)}{P(X|Y=0)}\right)}
//' 
//' where \eqn{P(X|Y=1)} is the probability of the feature being in a particular bin given a positive target, and \eqn{P(X|Y=0)} is the probability given a negative target.
//' 
//' The Information Value (IV) for each bin is calculated as:
//' 
//' \deqn{IV = (P(X|Y=1) - P(X|Y=0)) * WoE}
//' 
//' The total IV is the sum of IVs for all bins:
//' 
//' \deqn{Total IV = \sum_{i=1}^{n} IV_i}
//' 
//' This approach ensures that the resulting bins are statistically different from each other, potentially leading to more robust and meaningful binning.
//' 
//' @examples
//' \dontrun{
//' set.seed(123)
//' target <- sample(0:1, 1000, replace = TRUE)
//' feature <- rnorm(1000)
//' result <- optimal_binning_numerical_fetb(target, feature)
//' print(result$woebin)
//' print(result$totalIV)
//' }
//' 
//' @references
//' \itemize{
//'   \item Fisher, R. A. (1922). On the interpretation of χ2 from contingency tables, and the calculation of P. Journal of the Royal Statistical Society, 85(1), 87-94.
//'   \item Belotti, P., & Carrasco, M. (2017). Optimal binning: mathematical programming formulation and solution approach. arXiv preprint arXiv:1705.03287.
//' }
//' 
//' @author Lopes, J. E.
//' @export
// [[Rcpp::export]]
List optimal_binning_numerical_fetb(NumericVector target, NumericVector feature,
                                    int min_bins = 3, int max_bins = 5, double bin_cutoff = 0.05, int max_n_prebins = 20) {
  OptimalBinningNumericalFETB binning(target, feature, min_bins, max_bins, bin_cutoff, max_n_prebins);
  return binning.performBinning();
}
