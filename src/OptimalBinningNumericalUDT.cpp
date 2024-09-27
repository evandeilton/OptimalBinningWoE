#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

// Helper structure for bins
struct Bin {
  double lower;
  double upper;
  double woe;
  double iv;
  int count;
  int count_pos;
  int count_neg;
};

class OptimalBinningNumericalUDT {
private:
  std::vector<double> feature;
  std::vector<int> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  std::vector<Bin> bins;

public:
  OptimalBinningNumericalUDT(std::vector<double> feature,
                             std::vector<int> target,
                             int min_bins = 3,
                             int max_bins = 5,
                             double bin_cutoff = 0.05,
                             int max_n_prebins = 20) {
    this->feature = feature;
    this->target = target;
    this->min_bins = std::max(2, min_bins);
    this->max_bins = std::max(this->min_bins, max_bins);
    this->bin_cutoff = bin_cutoff;
    this->max_n_prebins = max_n_prebins;
  }

  void fit() {
    // Pre-binning using quantiles
    std::vector<double> cut_points = get_quantile_cutpoints(feature, max_n_prebins);

    // Initial binning
    bins = initial_binning(feature, target, cut_points);

    // Rare bin merging
    merge_rare_bins();

    // Bin optimization
    optimize_bins();
  }

  List get_result() {
    // Prepare output
    std::vector<double> woefeature(feature.size());
#pragma omp parallel for
    for (size_t i = 0; i < feature.size(); ++i) {
      woefeature[i] = get_woe_for_value(feature[i]);
    }

    // Prepare bin dataframe
    int n_bins = bins.size();
    CharacterVector bin_intervals(n_bins);
    NumericVector woe_values(n_bins), iv_values(n_bins);
    IntegerVector counts(n_bins), counts_pos(n_bins), counts_neg(n_bins);

    for (int i = 0; i < n_bins; ++i) {
      bin_intervals[i] = create_interval(bins[i].lower, bins[i].upper);
      woe_values[i] = bins[i].woe;
      iv_values[i] = bins[i].iv;
      counts[i] = bins[i].count;
      counts_pos[i] = bins[i].count_pos;
      counts_neg[i] = bins[i].count_neg;
    }

    DataFrame woebin = DataFrame::create(
      _["bin"] = bin_intervals,
      _["woe"] = woe_values,
      _["iv"] = iv_values,
      _["count"] = counts,
      _["count_pos"] = counts_pos,
      _["count_neg"] = counts_neg
    );

    return List::create(
      _["woefeature"] = woefeature,
      _["woebin"] = woebin
    );
  }

private:
  std::vector<double> get_quantile_cutpoints(std::vector<double> &data, int n_bins) {
    std::vector<double> quantiles(n_bins - 1);
    NumericVector data_vec = wrap(data);
    data_vec = na_omit(data_vec);
    data_vec.sort();
    int n = data_vec.size();

    for (int i = 1; i < n_bins; ++i) {
      double quantile = data_vec[int((double)i / n_bins * n)];
      quantiles[i - 1] = quantile;
    }
    std::sort(quantiles.begin(), quantiles.end());
    quantiles.erase(std::unique(quantiles.begin(), quantiles.end()), quantiles.end());
    return quantiles;
  }

  std::vector<Bin> initial_binning(std::vector<double> &feature,
                                   std::vector<int> &target,
                                   std::vector<double> &cut_points) {
    std::vector<Bin> initial_bins;
    std::vector<double> boundaries = cut_points;
    boundaries.insert(boundaries.begin(), -std::numeric_limits<double>::infinity());
    boundaries.push_back(std::numeric_limits<double>::infinity());
    int n_bins = boundaries.size() - 1;

    initial_bins.resize(n_bins);
    for (int i = 0; i < n_bins; ++i) {
      initial_bins[i].lower = boundaries[i];
      initial_bins[i].upper = boundaries[i + 1];
      initial_bins[i].count = 0;
      initial_bins[i].count_pos = 0;
      initial_bins[i].count_neg = 0;
    }

    // Assign data to bins
    for (size_t i = 0; i < feature.size(); ++i) {
      double val = feature[i];
      int bin_idx = find_bin_index(val, boundaries);
      if (bin_idx >= 0 && bin_idx < n_bins) {
        initial_bins[bin_idx].count++;
        if (target[i] == 1)
          initial_bins[bin_idx].count_pos++;
        else
          initial_bins[bin_idx].count_neg++;
      }
    }

    // Calculate WOE and IV
    calculate_woe_iv(initial_bins);

    return initial_bins;
  }

  int find_bin_index(double value, std::vector<double> &boundaries) {
    int left = 0;
    int right = boundaries.size() - 1;
    while (left <= right) {
      int mid = (left + right) / 2;
      if (value <= boundaries[mid]) {
        right = mid - 1;
      } else {
        left = mid + 1;
      }
    }
    return left - 1;
  }

  void calculate_woe_iv(std::vector<Bin> &bins) {
    int total_pos = 0, total_neg = 0;
    for (auto &bin : bins) {
      total_pos += bin.count_pos;
      total_neg += bin.count_neg;
    }

    for (auto &bin : bins) {
      double dist_pos = (double)bin.count_pos / total_pos;
      double dist_neg = (double)bin.count_neg / total_neg;
      if (dist_pos == 0)
        dist_pos = 0.0001;
      if (dist_neg == 0)
        dist_neg = 0.0001;
      bin.woe = std::log(dist_pos / dist_neg);
      bin.iv = (dist_pos - dist_neg) * bin.woe;
    }
  }

  void merge_rare_bins() {
    // Merge bins with frequency below bin_cutoff
    int total_count = feature.size();
    double cutoff_count = bin_cutoff * total_count;
    bool merged = false;

    do {
      merged = false;
      for (size_t i = 0; i < bins.size(); ++i) {
        if (bins[i].count < cutoff_count && bins.size() > min_bins) {
          if (i > 0) {
            // Merge with previous bin
            bins[i - 1] = merge_bins(bins[i - 1], bins[i]);
            bins.erase(bins.begin() + i);
          } else if (i < bins.size() - 1) {
            // Merge with next bin
            bins[i + 1] = merge_bins(bins[i], bins[i + 1]);
            bins.erase(bins.begin() + i);
          }
          merged = true;
          break;
        }
      }
    } while (merged && bins.size() > min_bins);

    // Recalculate WOE and IV
    calculate_woe_iv(bins);
  }

  Bin merge_bins(Bin &bin1, Bin &bin2) {
    Bin merged;
    merged.lower = bin1.lower;
    merged.upper = bin2.upper;
    merged.count = bin1.count + bin2.count;
    merged.count_pos = bin1.count_pos + bin2.count_pos;
    merged.count_neg = bin1.count_neg + bin2.count_neg;
    return merged;
  }

  void optimize_bins() {
    // Ensure number of bins within min_bins and max_bins
    while ((int)bins.size() > max_bins) {
      // Merge the two bins with the least IV gain
      double min_iv = std::numeric_limits<double>::max();
      size_t min_idx = 0;
      for (size_t i = 0; i < bins.size() - 1; ++i) {
        double iv_gain = bins[i].iv + bins[i + 1].iv;
        if (iv_gain < min_iv) {
          min_iv = iv_gain;
          min_idx = i;
        }
      }
      bins[min_idx] = merge_bins(bins[min_idx], bins[min_idx + 1]);
      bins.erase(bins.begin() + min_idx + 1);
      calculate_woe_iv(bins);
    }

    // Enforce monotonicity
    enforce_monotonicity();

    // Ensure minimum number of bins
    while ((int)bins.size() < min_bins && bins.size() > 1) {
      // Split the bin with the highest IV
      size_t max_iv_idx = 0;
      double max_iv = 0;
      for (size_t i = 0; i < bins.size(); ++i) {
        if (bins[i].iv > max_iv) {
          max_iv = bins[i].iv;
          max_iv_idx = i;
        }
      }

      // Split the bin at its midpoint
      Bin left_bin = bins[max_iv_idx];
      Bin right_bin = bins[max_iv_idx];
      double mid_point = (left_bin.lower + right_bin.upper) / 2;
      left_bin.upper = mid_point;
      right_bin.lower = mid_point;

      // Reassign data points to new bins
      int left_count = 0, left_pos = 0, left_neg = 0;
      for (size_t i = 0; i < feature.size(); ++i) {
        if (feature[i] > left_bin.lower && feature[i] <= left_bin.upper) {
          left_count++;
          if (target[i] == 1) left_pos++;
          else left_neg++;
        }
      }
      left_bin.count = left_count;
      left_bin.count_pos = left_pos;
      left_bin.count_neg = left_neg;
      right_bin.count -= left_count;
      right_bin.count_pos -= left_pos;
      right_bin.count_neg -= left_neg;

      // Insert the new bin
      bins[max_iv_idx] = left_bin;
      bins.insert(bins.begin() + max_iv_idx + 1, right_bin);

      // Recalculate WOE and IV
      calculate_woe_iv(bins);
    }
  }

  void enforce_monotonicity() {
    // Check if WOE is monotonic
    bool increasing = true, decreasing = true;
    for (size_t i = 1; i < bins.size(); ++i) {
      if (bins[i].woe < bins[i - 1].woe)
        increasing = false;
      if (bins[i].woe > bins[i - 1].woe)
        decreasing = false;
    }

    // If not monotonic, merge bins to enforce monotonicity
    while (!increasing && !decreasing && bins.size() > min_bins) {
      size_t merge_idx = 0;
      double min_diff = std::numeric_limits<double>::max();
      for (size_t i = 0; i < bins.size() - 1; ++i) {
        double diff = std::abs(bins[i].woe - bins[i + 1].woe);
        if (diff < min_diff) {
          min_diff = diff;
          merge_idx = i;
        }
      }
      bins[merge_idx] = merge_bins(bins[merge_idx], bins[merge_idx + 1]);
      bins.erase(bins.begin() + merge_idx + 1);
      calculate_woe_iv(bins);

      // Re-check monotonicity
      increasing = true;
      decreasing = true;
      for (size_t i = 1; i < bins.size(); ++i) {
        if (bins[i].woe < bins[i - 1].woe)
          increasing = false;
        if (bins[i].woe > bins[i - 1].woe)
          decreasing = false;
      }
    }
  }

  double get_woe_for_value(double value) {
    for (auto &bin : bins) {
      if (value > bin.lower && value <= bin.upper)
        return bin.woe;
    }
    return 0.0;
  }

  std::string create_interval(double lower, double upper) {
    std::stringstream ss;
    ss << "(";
    if (std::isinf(lower))
      ss << "-Inf";
    else
      ss << lower;
    ss << ";";
    if (std::isinf(upper))
      ss << "+Inf";
    else
      ss << upper;
    ss << "]";
    return ss.str();
  }
};


//' @title Optimal Binning for Numerical Variables using Unsupervised Decision Trees
//' 
//' @description
//' This function implements an optimal binning algorithm for numerical variables 
//' using an Unsupervised Decision Tree (UDT) approach with Weight of Evidence (WoE) 
//' and Information Value (IV) criteria.
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
//' The optimal binning algorithm for numerical variables uses an Unsupervised Decision Tree 
//' approach with Weight of Evidence (WoE) and Information Value (IV) to create bins that 
//' maximize the predictive power of the feature while maintaining interpretability.
//' 
//' The algorithm follows these steps:
//' 1. Initial discretization using quantile-based binning
//' 2. Merging of rare bins based on the bin_cutoff parameter
//' 3. Bin optimization using IV and WoE criteria
//' 4. Enforcement of monotonicity in WoE across bins
//' 5. Adjustment of the number of bins to be within the specified range
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
//' The UDT approach ensures that the resulting binning maximizes the separation between 
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
//' result <- optimal_binning_numerical_udt(target, feature, min_bins = 3, max_bins = 5)
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
//' \item Fayyad, U. M., & Irani, K. B. (1993). Multi-interval discretization of 
//'       continuous-valued attributes for classification learning. In Proceedings 
//'       of the 13th International Joint Conference on Artificial Intelligence (pp. 1022-1027).
//' \item Dougherty, J., Kohavi, R., & Sahami, M. (1995). Supervised and unsupervised 
//'       discretization of continuous features. In Machine Learning Proceedings 1995 
//'       (pp. 194-202). Morgan Kaufmann.
//' }
//' 
//' @author Lopes, J. E.
//' 
//' @export
// [[Rcpp::export]]
List optimal_binning_numerical_udt(IntegerVector target,
                                   NumericVector feature,
                                   int min_bins = 3,
                                   int max_bins = 5,
                                   double bin_cutoff = 0.05,
                                   int max_n_prebins = 20) {
  // Convert inputs to std::vector
  std::vector<double> feature_vec = as<std::vector<double>>(feature);
  std::vector<int> target_vec = as<std::vector<int>>(target);

  OptimalBinningNumericalUDT obnu(feature_vec, target_vec, min_bins,
                                  max_bins, bin_cutoff, max_n_prebins);
  obnu.fit();
  return obnu.get_result();
}

