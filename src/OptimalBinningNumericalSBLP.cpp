#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

class OptimalBinningNumericalMRBLP {
private:
  std::vector<double> feature;
  std::vector<int> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  int n_threads;

  struct Bin {
    double lower_bound;
    double upper_bound;
    double woe;
    double iv;
    int count;
    int count_pos;
    int count_neg;
  };

  std::vector<Bin> bins;

public:
  OptimalBinningNumericalMRBLP(const NumericVector& feature,
                               const IntegerVector& target,
                               int min_bins,
                               int max_bins,
                               double bin_cutoff,
                               int max_n_prebins,
                               int n_threads = 1)
    : feature(feature.begin(), feature.end()),
      target(target.begin(), target.end()),
      min_bins(min_bins),
      max_bins(max_bins),
      bin_cutoff(bin_cutoff),
      max_n_prebins(max_n_prebins),
      n_threads(n_threads) {}

  void fit() {
    prebinning();
    mergeSmallBins();
    monotonicBinning();
  }

  NumericVector transform() {
    NumericVector woefeature(feature.size());
#pragma omp parallel for num_threads(n_threads)
    for (size_t i = 0; i < feature.size(); ++i) {
      double val = feature[i];
      for (size_t j = 0; j < bins.size(); ++j) {
        if ((val > bins[j].lower_bound) && (val <= bins[j].upper_bound)) {
          woefeature[i] = bins[j].woe;
          break;
        }
      }
    }
    return woefeature;
  }

  DataFrame getWoebin() {
    size_t n_bins = bins.size();
    CharacterVector bin(n_bins);
    NumericVector woe(n_bins);
    NumericVector iv(n_bins);
    IntegerVector count(n_bins);
    IntegerVector count_pos(n_bins);
    IntegerVector count_neg(n_bins);

    for (size_t i = 0; i < n_bins; ++i) {
      std::string bin_str;
      if (std::isinf(bins[i].lower_bound)) {
        bin_str = "(-Inf;" + std::to_string(bins[i].upper_bound) + "]";
      } else if (std::isinf(bins[i].upper_bound)) {
        bin_str = "(" + std::to_string(bins[i].lower_bound) + ";+Inf]";
      } else {
        bin_str = "(" + std::to_string(bins[i].lower_bound) + ";" + std::to_string(bins[i].upper_bound) + "]";
      }
      bin[i] = bin_str;
      woe[i] = bins[i].woe;
      iv[i] = bins[i].iv;
      count[i] = bins[i].count;
      count_pos[i] = bins[i].count_pos;
      count_neg[i] = bins[i].count_neg;
    }

    return DataFrame::create(_["bin"] = bin,
                             _["woe"] = woe,
                             _["iv"] = iv,
                             _["count"] = count,
                             _["count_pos"] = count_pos,
                             _["count_neg"] = count_neg);
  }

private:
  void prebinning() {
    std::vector<double> feature_clean;
    std::vector<int> target_clean;
    for (size_t i = 0; i < feature.size(); ++i) {
      if (!NumericVector::is_na(feature[i])) {
        feature_clean.push_back(feature[i]);
        target_clean.push_back(target[i]);
      }
    }

    // Determine pre-bin boundaries using equal-frequency binning
    std::vector<double> sorted_feature = feature_clean;
    std::sort(sorted_feature.begin(), sorted_feature.end());

    std::vector<double> boundaries;
    boundaries.push_back(-std::numeric_limits<double>::infinity());

    int bin_size = std::max(1, static_cast<int>(sorted_feature.size() / max_n_prebins));
    for (size_t i = bin_size; i < sorted_feature.size(); i += bin_size) {
      if (boundaries.size() < static_cast<size_t>(max_n_prebins - 1)) {
        boundaries.push_back(sorted_feature[i]);
      }
    }
    boundaries.push_back(std::numeric_limits<double>::infinity());

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
#pragma omp parallel for num_threads(n_threads)
    for (size_t i = 0; i < feature_clean.size(); ++i) {
      double val = feature_clean[i];
      int tgt = target_clean[i];
      for (size_t j = 0; j < n_bins; ++j) {
        if ((val > bins[j].lower_bound) && (val <= bins[j].upper_bound)) {
#pragma omp atomic
          bins[j].count++;
          if (tgt == 1) {
#pragma omp atomic
            bins[j].count_pos++;
          } else if (tgt == 0) {
#pragma omp atomic
            bins[j].count_neg++;
          }
          break;
        }
      }
    }

    // Remove empty bins
    bins.erase(std::remove_if(bins.begin(), bins.end(),
                              [](const Bin& bin) { return bin.count == 0; }),
                              bins.end());

    computeWOEIV();
  }

  void computeWOEIV() {
    double total_pos = 0.0;
    double total_neg = 0.0;
    for (size_t i = 0; i < bins.size(); ++i) {
      total_pos += bins[i].count_pos;
      total_neg += bins[i].count_neg;
    }

    for (size_t i = 0; i < bins.size(); ++i) {
      double dist_pos = bins[i].count_pos / total_pos;
      double dist_neg = bins[i].count_neg / total_neg;

      if (dist_pos == 0)
        dist_pos = 1e-10;
      if (dist_neg == 0)
        dist_neg = 1e-10;

      bins[i].woe = std::log(dist_pos / dist_neg);
      bins[i].iv = (dist_pos - dist_neg) * bins[i].woe;
    }
  }

  void mergeSmallBins() {
    while (bins.size() > static_cast<size_t>(min_bins)) {
      size_t smallest_bin_idx = 0;
      double smallest_bin_ratio = std::numeric_limits<double>::max();

      for (size_t i = 0; i < bins.size(); ++i) {
        double bin_ratio = static_cast<double>(bins[i].count) / feature.size();
        if (bin_ratio < smallest_bin_ratio) {
          smallest_bin_ratio = bin_ratio;
          smallest_bin_idx = i;
        }
      }

      if (smallest_bin_ratio >= bin_cutoff) {
        break;
      }

      if (smallest_bin_idx == 0) {
        mergeBins(0, 1);
      } else if (smallest_bin_idx == bins.size() - 1) {
        mergeBins(bins.size() - 2, bins.size() - 1);
      } else {
        size_t merge_idx = (bins[smallest_bin_idx - 1].count < bins[smallest_bin_idx + 1].count)
        ? smallest_bin_idx - 1 : smallest_bin_idx;
        mergeBins(merge_idx, merge_idx + 1);
      }
    }
    computeWOEIV();
  }

  void mergeBins(size_t idx1, size_t idx2) {
    if (idx2 >= bins.size()) return;

    bins[idx1].upper_bound = bins[idx2].upper_bound;
    bins[idx1].count += bins[idx2].count;
    bins[idx1].count_pos += bins[idx2].count_pos;
    bins[idx1].count_neg += bins[idx2].count_neg;

    bins.erase(bins.begin() + idx2);
  }

  bool isIncreasingWOE() {
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

  void monotonicBinning() {
    bool increasing = isIncreasingWOE();

    bool need_merge = true;
    while (need_merge && bins.size() > static_cast<size_t>(min_bins)) {
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
    }

    while (bins.size() > static_cast<size_t>(max_bins)) {
      size_t merge_idx = findSmallestIVDiff();
      mergeBins(merge_idx, merge_idx + 1);
      computeWOEIV();
    }

    ensureMinBins();
  }

  size_t findSmallestIVDiff() {
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

  void ensureMinBins() {
    while (bins.size() < static_cast<size_t>(min_bins)) {
      size_t split_idx = findLargestBin();
      splitBin(split_idx);
      computeWOEIV();
    }
  }

  size_t findLargestBin() {
    size_t max_idx = 0;
    int max_count = 0;
    for (size_t i = 0; i < bins.size(); ++i) {
      if (bins[i].count > max_count) {
        max_count = bins[i].count;
        max_idx = i;
      }
    }
    return max_idx;
  }

  void splitBin(size_t idx) {
    Bin& bin = bins[idx];
    double mid = (bin.lower_bound + bin.upper_bound) / 2;

    Bin new_bin;
    new_bin.lower_bound = mid;
    new_bin.upper_bound = bin.upper_bound;
    bin.upper_bound = mid;

    new_bin.count = 0;
    new_bin.count_pos = 0;
    new_bin.count_neg = 0;

    // Reassign observations to the new bins
    for (size_t i = 0; i < feature.size(); ++i) {
      if (feature[i] > mid && feature[i] <= bin.upper_bound) {
        new_bin.count++;
        if (target[i] == 1) {
          new_bin.count_pos++;
        } else {
          new_bin.count_neg++;
        }
        bin.count--;
        if (target[i] == 1) {
          bin.count_pos--;
        } else {
          bin.count_neg--;
        }
      }
    }

    bins.insert(bins.begin() + idx + 1, new_bin);
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
//' @param n_threads Integer. The number of threads to use for parallel processing (default: 1).
//'
//' @return A list containing two elements:
//' \item{woefeature}{A numeric vector of Weight of Evidence (WoE) transformed values for the input feature.}
//' \item{woebin}{A data frame containing the binning information, including bin boundaries, WoE values, Information Value (IV), and count statistics.}
//'
//' @details
//' The MRBLP algorithm combines pre-binning, small bin merging, and monotonic binning to create an optimal binning solution for numerical variables. The process involves the following steps:
//'
//' 1. Pre-binning: The algorithm starts by creating initial bins using equal-frequency binning. The number of pre-bins is determined by the `max_n_prebins` parameter.
//' 2. Small bin merging: Bins with a proportion of observations less than `bin_cutoff` are merged with adjacent bins to ensure statistical significance.
//' 3. Monotonic binning: The algorithm enforces a monotonic relationship between the bin order and the Weight of Evidence (WoE) values. This step ensures that the binning preserves the original relationship between the feature and the target variable.
//' 4. Bin count adjustment: If the number of bins exceeds `max_bins`, the algorithm merges bins with the smallest difference in Information Value (IV). If the number of bins is less than `min_bins`, the largest bin is split.
//'
//' The Weight of Evidence (WoE) for each bin is calculated as:
//'
//' \deqn{WoE = \ln\left(\frac{P(X|Y=1)}{P(X|Y=0)}\right) = \ln\left(\frac{\frac{n_{1i}}{n_1}}{\frac{n_{0i}}{n_0}}\right)}
//'
//' where \eqn{n_{1i}} and \eqn{n_{0i}} are the number of events and non-events in bin i, respectively, and \eqn{n_1} and \eqn{n_0} are the total number of events and non-events.
//'
//' The Information Value (IV) for each bin is calculated as:
//'
//' \deqn{IV_i = \left(\frac{n_{1i}}{n_1} - \frac{n_{0i}}{n_0}\right) \times WoE_i}
//'
//' The total Information Value for the binning solution is the sum of IVs across all bins:
//'
//' \deqn{IV_{total} = \sum_{i=1}^{k} IV_i}
//'
//' where k is the number of bins.
//'
//' This implementation uses OpenMP for parallel processing to improve performance on multi-core systems.
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
//' print(result$woebin)
//'
//' # Use WoE-transformed feature
//' woe_feature <- result$woefeature
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
List optimal_binning_numerical_mrblp(IntegerVector target,
                                     NumericVector feature,
                                     int min_bins = 3,
                                     int max_bins = 5,
                                     double bin_cutoff = 0.05,
                                     int max_n_prebins = 20,
                                     int n_threads = 1) {

  if (min_bins < 2) {
    stop("min_bins must be >= 2.");
  }
  if (max_bins < min_bins) {
    stop("max_bins must be >= min_bins.");
  }
  if (bin_cutoff < 0 || bin_cutoff > 1) {
    stop("bin_cutoff must be between 0 and 1.");
  }
  if (max_n_prebins < min_bins) {
    stop("max_n_prebins must be >= min_bins.");
  }

  IntegerVector unique_targets = unique(target);
  if (unique_targets.size() != 2 || !(is_true(any(unique_targets == 0)) && is_true(any(unique_targets == 1)))) {
    stop("Target must be binary with values 0 and 1.");
  }

  OptimalBinningNumericalMRBLP binning(feature, target, min_bins, max_bins, bin_cutoff, max_n_prebins, n_threads);
  binning.fit();
  NumericVector woefeature = binning.transform();
  DataFrame woebin = binning.getWoebin();

  return List::create(_["woefeature"] = woefeature,
                      _["woebin"] = woebin);
}
