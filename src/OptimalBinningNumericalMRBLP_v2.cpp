#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>

using namespace Rcpp;

/**
 * @class OptimalBinningNumericalMRBLP
 * @brief Performs optimal binning on numerical features using Monotonic Risk Binning with Likelihood Ratio Pre-binning (MRBLP).
 *
 * This class implements a binning algorithm that divides a numerical feature into bins,
 * ensuring monotonicity in the relationship between bins and the target variable.
 */
class OptimalBinningNumericalMRBLP {
private:
  // Feature and target vectors
  std::vector<double> feature;
  std::vector<int> target;
  
  // Binning parameters
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  double convergence_threshold;
  int max_iterations;
  
  /**
   * @struct Bin
   * @brief Represents a single bin with its boundaries and statistics.
   */
  struct Bin {
    double lower_bound;
    double upper_bound;
    double woe;
    double iv;
    int count;
    int count_pos;
    int count_neg;
  };
  
  std::vector<Bin> bins; // Vector of bins
  bool converged;
  int iterations_run;
  
public:
  /**
   * @brief Constructor for the OptimalBinningNumericalMRBLP class.
   *
   * @param feature NumericVector of feature values.
   * @param target IntegerVector of target values (0 or 1).
   * @param min_bins Minimum number of bins.
   * @param max_bins Maximum number of bins.
   * @param bin_cutoff Minimum bin size as a proportion of total data.
   * @param max_n_prebins Maximum number of pre-bins.
   * @param convergence_threshold Threshold for convergence.
   * @param max_iterations Maximum number of iterations for convergence.
   */
  OptimalBinningNumericalMRBLP(const NumericVector& feature,
                               const IntegerVector& target,
                               int min_bins,
                               int max_bins,
                               double bin_cutoff,
                               int max_n_prebins,
                               double convergence_threshold,
                               int max_iterations)
    : feature(feature.begin(), feature.end()),
      target(target.begin(), target.end()),
      min_bins(min_bins),
      max_bins(max_bins),
      bin_cutoff(bin_cutoff),
      max_n_prebins(max_n_prebins),
      convergence_threshold(convergence_threshold),
      max_iterations(max_iterations),
      converged(false),
      iterations_run(0) {
    
    // Input validation
    if (feature.size() != target.size()) {
      throw std::invalid_argument("Feature and target vectors must be of the same length.");
    }
    if (min_bins <= 0) {
      throw std::invalid_argument("min_bins must be greater than 0.");
    }
    if (max_bins < min_bins) {
      throw std::invalid_argument("max_bins must be greater than or equal to min_bins.");
    }
    if (max_n_prebins <= 0) {
      throw std::invalid_argument("max_n_prebins must be greater than 0.");
    }
    if (bin_cutoff <= 0 || bin_cutoff >= 1) {
      throw std::invalid_argument("bin_cutoff must be between 0 and 1.");
    }
    if (convergence_threshold <= 0) {
      throw std::invalid_argument("convergence_threshold must be greater than 0.");
    }
    if (max_iterations <= 0) {
      throw std::invalid_argument("max_iterations must be greater than 0.");
    }
  }
  
  /**
   * @brief Fits the binning model to the data.
   */
  void fit() {
    // Check if the number of unique values is less than or equal to min_bins
    std::vector<double> unique_features = feature;
    std::sort(unique_features.begin(), unique_features.end());
    auto last = std::unique(unique_features.begin(), unique_features.end());
    unique_features.erase(last, unique_features.end());
    
    if (unique_features.size() <= static_cast<size_t>(min_bins)) {
      // Use unique values as bin boundaries
      for (size_t i = 0; i < unique_features.size(); ++i) {
        Bin bin;
        bin.lower_bound = (i == 0) ? -std::numeric_limits<double>::infinity() : unique_features[i-1];
        bin.upper_bound = (i == unique_features.size() - 1) ? std::numeric_limits<double>::infinity() : unique_features[i];
        bins.push_back(bin);
      }
    } else {
      prebinning();
      mergeSmallBins();
      monotonicBinning();
    }
    
    // Compute final statistics
    computeWOEIV();
  }
  
  /**
   * @brief Retrieves the binning information as a List.
   *
   * @return List containing bin ranges, WoE, IV, counts, cutpoints, convergence status, and iterations run.
   */
  List getWoebin() const {
    size_t n_bins = bins.size();
    CharacterVector bin_names(n_bins);
    NumericVector bin_woe(n_bins);
    NumericVector bin_iv(n_bins);
    IntegerVector bin_count(n_bins);
    IntegerVector bin_count_pos(n_bins);
    IntegerVector bin_count_neg(n_bins);
    NumericVector bin_cutpoints(n_bins - 1);
    
    for (size_t i = 0; i < n_bins; ++i) {
      std::ostringstream oss;
      if (std::isinf(bins[i].lower_bound) && bins[i].lower_bound < 0) {
        oss << "(-Inf," << bins[i].upper_bound << "]";
      } else if (std::isinf(bins[i].upper_bound)) {
        oss << "(" << bins[i].lower_bound << ",+Inf]";
      } else {
        oss << "(" << bins[i].lower_bound << "," << bins[i].upper_bound << "]";
      }
      bin_names[i] = oss.str();
      bin_woe[i] = bins[i].woe;
      bin_iv[i] = bins[i].iv;
      bin_count[i] = bins[i].count;
      bin_count_pos[i] = bins[i].count_pos;
      bin_count_neg[i] = bins[i].count_neg;
      
      if (i < n_bins - 1) {
        bin_cutpoints[i] = bins[i].upper_bound;
      }
    }
    
    return List::create(
      Named("bins") = bin_names,
      Named("woe") = bin_woe,
      Named("iv") = bin_iv,
      Named("count") = bin_count,
      Named("count_pos") = bin_count_pos,
      Named("count_neg") = bin_count_neg,
      Named("cutpoints") = bin_cutpoints,
      Named("converged") = converged,
      Named("iterations") = iterations_run
    );
  }
  
private:
  /**
   * @brief Performs initial pre-binning using equal-frequency binning.
   */
  void prebinning() {
    // Clean feature and target by removing NA values
    std::vector<std::pair<double, int>> data;
    data.reserve(feature.size());
    
    for (size_t i = 0; i < feature.size(); ++i) {
      if (!NumericVector::is_na(feature[i])) {
        data.emplace_back(feature[i], target[i]);
      }
    }
    
    if (data.empty()) {
      throw std::runtime_error("All feature values are NA.");
    }
    
    // Sort data based on feature values
    std::sort(data.begin(), data.end(),
              [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                return a.first < b.first;
              });
    
    // Determine unique feature values
    std::vector<double> unique_features;
    unique_features.reserve(data.size());
    unique_features.push_back(data[0].first);
    for (size_t i = 1; i < data.size(); ++i) {
      if (data[i].first != data[i - 1].first) {
        unique_features.push_back(data[i].first);
      }
    }
    
    size_t unique_size = unique_features.size();
    
    // Adjust max_n_prebins based on unique feature values
    max_n_prebins = std::min(static_cast<int>(unique_size), max_n_prebins);
    
    // Determine pre-bin boundaries using equal-frequency binning
    std::vector<double> boundaries;
    boundaries.reserve(max_n_prebins + 1);
    boundaries.emplace_back(-std::numeric_limits<double>::infinity());
    
    size_t bin_size = std::max(static_cast<size_t>(1), static_cast<size_t>(data.size() / max_n_prebins));
    for (int i = 1; i < max_n_prebins; ++i) {
      size_t idx = i * bin_size;
      if (idx >= data.size()) {
        break;
      }
      // To ensure unique boundaries, pick the value where feature changes
      while (idx < data.size() && data[idx].first == data[idx - 1].first) {
        idx++;
      }
      if (idx < data.size()) {
        boundaries.emplace_back(data[idx].first);
      }
    }
    boundaries.emplace_back(std::numeric_limits<double>::infinity());
    
    size_t n_bins = boundaries.size() - 1;
    bins.clear();
    bins.resize(n_bins);
    
    for (size_t i = 0; i < n_bins; ++i) {
      bins[i].lower_bound = boundaries[i];
      bins[i].upper_bound = boundaries[i + 1];
      bins[i].count = 0;
      bins[i].count_pos = 0;
      bins[i].count_neg = 0;
    }
    
    // Assign observations to bins
    for (const auto& d : data) {
      double val = d.first;
      int tgt = d.second;
      int bin_idx = findBin(val);
      if (bin_idx != -1) {
        bins[bin_idx].count++;
        if (tgt == 1) {
          bins[bin_idx].count_pos++;
        } else if (tgt == 0) {
          bins[bin_idx].count_neg++;
        }
      }
    }
    
    // Remove empty bins
    bins.erase(std::remove_if(bins.begin(), bins.end(),
                              [](const Bin& bin) { return bin.count == 0; }),
                              bins.end());
    
    // Compute initial WoE and IV
    computeWOEIV();
  }
  
  /**
   * @brief Computes WoE and IV for each bin.
   */
  void computeWOEIV() {
    double total_pos = 0.0;
    double total_neg = 0.0;
    
    for (const auto& bin : bins) {
      total_pos += bin.count_pos;
      total_neg += bin.count_neg;
    }
    
    // Handle cases with no positives or no negatives
    if (total_pos == 0 || total_neg == 0) {
      for (auto& bin : bins) {
        bin.woe = 0.0;
        bin.iv = 0.0;
      }
      return;
    }
    
    for (auto& bin : bins) {
      double dist_pos = static_cast<double>(bin.count_pos) / total_pos;
      double dist_neg = static_cast<double>(bin.count_neg) / total_neg;
      
      // Avoid division by zero by adding a small constant
      if (dist_pos == 0.0) {
        dist_pos = 1e-10;
      }
      if (dist_neg == 0.0) {
        dist_neg = 1e-10;
      }
      
      bin.woe = std::log(dist_pos / dist_neg);
      bin.iv = (dist_pos - dist_neg) * bin.woe;
    }
  }
  
  /**
   * @brief Merges bins that do not meet the bin_cutoff threshold.
   */
  void mergeSmallBins() {
    bool merged = true;
    while (merged && bins.size() > static_cast<size_t>(min_bins)) {
      merged = false;
      
      // Find the bin with the smallest ratio
      size_t smallest_bin_idx = 0;
      double smallest_bin_ratio = std::numeric_limits<double>::max();
      
      for (size_t i = 0; i < bins.size(); ++i) {
        double bin_ratio = static_cast<double>(bins[i].count) / feature.size();
        if (bin_ratio < smallest_bin_ratio) {
          smallest_bin_ratio = bin_ratio;
          smallest_bin_idx = i;
        }
      }
      
      // If the smallest bin meets the cutoff, stop merging
      if (smallest_bin_ratio >= bin_cutoff) {
        break;
      }
      
      // Merge the smallest bin with its adjacent bin
      if (smallest_bin_idx == 0) {
        mergeBins(0, 1);
      } else if (smallest_bin_idx == bins.size() - 1) {
        mergeBins(bins.size() - 2, bins.size() - 1);
      } else {
        // Merge with the neighbor that has a smaller count
        if (bins[smallest_bin_idx - 1].count <= bins[smallest_bin_idx + 1].count) {
          mergeBins(smallest_bin_idx - 1, smallest_bin_idx);
        } else {
          mergeBins(smallest_bin_idx, smallest_bin_idx + 1);
        }
      }
      computeWOEIV();
      merged = true;
    }
  }
  
  /**
   * @brief Merges two adjacent bins specified by their indices.
   *
   * @param idx1 Index of the first bin.
   * @param idx2 Index of the second bin.
   */
  void mergeBins(size_t idx1, size_t idx2) {
    if (idx2 >= bins.size()) return;
    
    bins[idx1].upper_bound = bins[idx2].upper_bound;
    bins[idx1].count += bins[idx2].count;
    bins[idx1].count_pos += bins[idx2].count_pos;
    bins[idx1].count_neg += bins[idx2].count_neg;
    
    bins.erase(bins.begin() + idx2);
  }
  
  /**
   * @brief Determines if WoE values are increasing or decreasing.
   *
   * @return True if WoE is generally increasing, false otherwise.
   */
  bool isIncreasingWOE() const {
    int n_increasing = 0;
    int n_decreasing = 0;
    
    for (size_t i = 0; i < bins.size() - 1; ++i) {
      if (bins[i].woe < bins[i + 1].woe) {
        n_increasing++;
      } else if (bins[i].woe > bins[i + 1].woe) {
        n_decreasing++;
      }
    }
    return n_increasing >= n_decreasing;
  }
  
  /**
   * @brief Enforces monotonicity in WoE values across bins.
   */
  void monotonicBinning() {
    bool increasing = isIncreasingWOE();
    
    iterations_run = 0;
    converged = false;
    bool need_merge = true;
    while (need_merge && bins.size() > static_cast<size_t>(min_bins) && iterations_run < max_iterations) {
      need_merge = false;
      for (size_t i = 0; i < bins.size() - 1; ++i) {
        bool violation = false;
        if (increasing) {
          if (bins[i].woe > bins[i + 1].woe) {
            violation = true;
          }
        } else {
          if (bins[i].woe < bins[i + 1].woe) {
            violation = true;
          }
        }
        
        if (violation) {
          mergeBins(i, i + 1);
          computeWOEIV();
          need_merge = true;
          break;
        }
      }
      
      iterations_run++;
      
      // Check for convergence
      if (!need_merge) {
        converged = true;
        break;
      }
      
      // Check if we've reached the convergence threshold
      if (iterations_run > 1 && std::abs(bins[bins.size()-1].woe - bins[0].woe) < convergence_threshold) {
        converged = true;
        break;
      }
    }
    
    // Ensure the number of bins does not exceed max_bins
    while (bins.size() > static_cast<size_t>(max_bins)) {
      size_t merge_idx = findSmallestIVDiff();
      mergeBins(merge_idx, merge_idx + 1);
      computeWOEIV();
    }
    
    // Ensure minimum number of bins
    ensureMinBins();
  }
  
  /**
   * @brief Finds the index of the bin pair with the smallest absolute IV difference.
   *
   * @return Index of the first bin in the pair to merge.
   */
  size_t findSmallestIVDiff() const {
    double min_diff = std::numeric_limits<double>::infinity();
    size_t min_idx = 0;
    for (size_t i = 0; i < bins.size() - 1; ++i) {
      double iv_diff = std::abs(bins[i].iv - bins[i + 1].iv);
      if (iv_diff < min_diff) {
        min_diff = iv_diff;
        min_idx = i;
      }
    }
    return min_idx;
  }
  
  /**
   * @brief Ensures that the number of bins does not fall below min_bins by splitting the largest bin.
   */
  void ensureMinBins() {
    while (bins.size() < static_cast<size_t>(min_bins)) {
      size_t split_idx = findLargestBin();
      splitBin(split_idx);
      computeWOEIV();
    }
  }
  
  /**
   * @brief Finds the index of the bin with the largest count.
   *
   * @return Index of the largest bin.
   */
  size_t findLargestBin() const {
    size_t max_idx = 0;
    int max_count = bins[0].count;
    for (size_t i = 1; i < bins.size(); ++i) {
      if (bins[i].count > max_count) {
        max_count = bins[i].count;
        max_idx = i;
      }
    }
    return max_idx;
  }
  
  /**
   * @brief Splits a bin into two at the midpoint of its boundaries.
   *
   * @param idx Index of the bin to split.
   */
  void splitBin(size_t idx) {
    if (idx >= bins.size()) return;
    
    Bin& bin = bins[idx];
    double mid = (bin.lower_bound + bin.upper_bound) / 2.0;
    
    // Create a new bin
    Bin new_bin;
    new_bin.lower_bound = mid;
    new_bin.upper_bound = bin.upper_bound;
    new_bin.count = bin.count / 2;
    new_bin.count_pos = bin.count_pos / 2;
    new_bin.count_neg = bin.count_neg / 2;
    
    // Update the existing bin
    bin.upper_bound = mid;
    bin.count -= new_bin.count;
    bin.count_pos -= new_bin.count_pos;
    bin.count_neg -= new_bin.count_neg;
    
    // Insert the new bin after the current bin
    bins.insert(bins.begin() + idx + 1, new_bin);
  }
  
  /**
   * @brief Finds the bin index for a given feature value using binary search.
   *
   * @param val Feature value to assign.
   * @return Index of the bin, or -1 if not found.
   */
  int findBin(double val) const {
    int left = 0;
    int right = bins.size() - 1;
    int mid;
    
    while (left <= right) {
      mid = left + (right - left) / 2;
      if (val > bins[mid].lower_bound && val <= bins[mid].upper_bound) {
        return mid;
      } else if (val <= bins[mid].lower_bound) {
        right = mid - 1;
      } else { // val > bins[mid].upper_bound
        left = mid + 1;
      }
    }
    return -1; // Not found
  }
};

//' @title Optimal Binning for Numerical Variables using Monotonic Risk Binning with Likelihood Ratio Pre-binning (MRBLP)
//'
//' @description
//' This function implements an optimal binning algorithm for numerical variables using
//' Monotonic Risk Binning with Likelihood Ratio Pre-binning (MRBLP). It transforms a
//' continuous feature into discrete bins while preserving the monotonic relationship
//' with the target variable and maximizing the predictive power.
//'
//' @param target An integer vector of binary target values (0 or 1).
//' @param feature A numeric vector of the continuous feature to be binned.
//' @param min_bins Integer. The minimum number of bins to create (default: 3).
//' @param max_bins Integer. The maximum number of bins to create (default: 5).
//' @param bin_cutoff Numeric. The minimum proportion of observations in each bin (default: 0.05).
//' @param max_n_prebins Integer. The maximum number of pre-bins to create during the initial binning step (default: 20).
//' @param convergence_threshold Numeric. The threshold for convergence in the monotonic binning step (default: 1e-6).
//' @param max_iterations Integer. The maximum number of iterations for the monotonic binning step (default: 1000).
//'
//' @return A list containing the following elements:
//' \item{bins}{A character vector of bin ranges.}
//' \item{woe}{A numeric vector of Weight of Evidence (WoE) values for each bin.}
//' \item{iv}{A numeric vector of Information Value (IV) for each bin.}
//' \item{count}{An integer vector of the total count of observations in each bin.}
//' \item{count_pos}{An integer vector of the count of positive observations in each bin.}
//' \item{count_neg}{An integer vector of the count of negative observations in each bin.}
//' \item{cutpoints}{A numeric vector of cutpoints used to create the bins.}
//' \item{converged}{A logical value indicating whether the algorithm converged.}
//' \item{iterations}{An integer value indicating the number of iterations run.}
//'
//' @details
//' The MRBLP algorithm combines pre-binning, small bin merging, and monotonic binning to create an optimal binning solution for numerical variables. The process involves the following steps:
//'
//' 1. Pre-binning: The algorithm starts by creating initial bins using equal-frequency binning. The number of pre-bins is determined by the `max_n_prebins` parameter.
//' 2. Small bin merging: Bins with a proportion of observations less than `bin_cutoff` are merged with adjacent bins to ensure statistical significance.
//' 3. Monotonic binning: The algorithm enforces a monotonic relationship between the bin order and the Weight of Evidence (WoE) values. This step ensures that the binning preserves the original relationship between the feature and the target variable.
//' 4. Bin count adjustment: If the number of bins exceeds `max_bins`, the algorithm merges bins with the smallest difference in Information Value (IV). If the number of bins is less than `min_bins`, the largest bin is split.
//'
//' The algorithm includes additional controls to prevent instability and ensure convergence:
//' - A convergence threshold is used to determine when the algorithm should stop iterating.
//' - A maximum number of iterations is set to prevent infinite loops.
//' - If convergence is not reached within the specified time and standards, the function returns the best result obtained up to the last iteration.
//'
//' @examples
//' \dontrun{
//' # Generate sample data
//' set.seed(42)
//' n <- 10000
//' feature <- rnorm(n)
//' target <- rbinom(n, 1, plogis(0.5 + 0.5 * feature))
//'
//' # Run optimal binning
//' result <- optimal_binning_numerical_mrblp(target, feature)
//'
//' # View binning results
//' print(result)
//' }
//'
//' @references
//' \itemize{
//' \item Belcastro, L., Marozzo, F., Talia, D., & Trunfio, P. (2020). "Big Data Analytics on Clouds."
//'       In Handbook of Big Data Technologies (pp. 101-142). Springer, Cham.
//' \item Zeng, Y. (2014). "Optimal Binning for Scoring Modeling." Computational Economics, 44(1), 137-149.
//' }
//'
//' @author Lopes, J. E.
//'
//' @export
// [[Rcpp::export]]
List optimal_binning_numerical_mrblp(const IntegerVector& target,
                                    const NumericVector& feature,
                                    int min_bins = 3,
                                    int max_bins = 5,
                                    double bin_cutoff = 0.05,
                                    int max_n_prebins = 20,
                                    double convergence_threshold = 1e-6,
                                    int max_iterations = 1000) {
 try {
   // Instantiate the binning class
   OptimalBinningNumericalMRBLP binning(feature, target, min_bins, max_bins, bin_cutoff, max_n_prebins, convergence_threshold, max_iterations);
   
   // Fit the binning model
   binning.fit();
   
   // Get the binning information
   return binning.getWoebin();
 } catch (const std::exception& e) {
   Rcpp::stop("Error in optimal_binning_numerical_mrblp: " + std::string(e.what()));
 }
}
