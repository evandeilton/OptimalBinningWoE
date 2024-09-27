#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif


class OptimalBinningNumericalEWB {
private:
  std::vector<double> feature;
  std::vector<int> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;

  struct Bin {
    double lower;
    double upper;
    int count;
    int count_pos;
    int count_neg;
    double woe;
    double iv;
  };

  std::vector<Bin> bins;

  double calculate_woe(int pos, int neg) {
    double pos_rate = static_cast<double>(pos) / (pos + neg);
    double neg_rate = static_cast<double>(neg) / (pos + neg);
    return std::log(pos_rate / neg_rate);
  }

  double calculate_iv(double woe, int pos, int neg, int total_pos, int total_neg) {
    double pos_rate = static_cast<double>(pos) / total_pos;
    double neg_rate = static_cast<double>(neg) / total_neg;
    return (pos_rate - neg_rate) * woe;
  }

public:
  OptimalBinningNumericalEWB(const std::vector<double>& feature, const std::vector<int>& target,
                             int min_bins = 3, int max_bins = 5, double bin_cutoff = 0.05, int max_n_prebins = 20)
    : feature(feature), target(target), min_bins(min_bins), max_bins(max_bins),
      bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins) {
    if (min_bins < 2) {
      Rcpp::stop("min_bins must be at least 2");
    }
    if (max_bins < min_bins) {
      Rcpp::stop("max_bins must be greater than or equal to min_bins");
    }
  }

  void fit() {
    // Sort feature and target together
    std::vector<size_t> indices(feature.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [this](size_t i1, size_t i2) { return feature[i1] < feature[i2]; });

    std::vector<double> sorted_feature(feature.size());
    std::vector<int> sorted_target(target.size());

#pragma omp parallel for
    for (size_t i = 0; i < indices.size(); ++i) {
      sorted_feature[i] = feature[indices[i]];
      sorted_target[i] = target[indices[i]];
    }

    // Calculate initial equal-width bins
    double min_value = sorted_feature.front();
    double max_value = sorted_feature.back();
    int n_prebins = std::min(max_n_prebins, static_cast<int>(sorted_feature.size()));
    double bin_width = (max_value - min_value) / n_prebins;

    // Initialize bins
    bins.clear();
    for (int i = 0; i < n_prebins; ++i) {
      double lower = (i == 0) ? std::numeric_limits<double>::lowest() : min_value + i * bin_width;
      double upper = (i == n_prebins - 1) ? std::numeric_limits<double>::max() : min_value + (i + 1) * bin_width;
      bins.push_back({lower, upper, 0, 0, 0, 0.0, 0.0});
    }

    // Count samples in each bin
    int total_pos = 0, total_neg = 0;
    for (size_t i = 0; i < sorted_feature.size(); ++i) {
      auto it = std::lower_bound(bins.begin(), bins.end(), sorted_feature[i],
                                 [](const Bin& bin, double value) { return bin.upper < value; });
      if (it != bins.end()) {
        it->count++;
        if (sorted_target[i] == 1) {
          it->count_pos++;
          total_pos++;
        } else {
          it->count_neg++;
          total_neg++;
        }
      }
    }

    // Merge rare bins
    for (auto it = bins.begin(); it != bins.end(); ) {
      if (static_cast<double>(it->count) / sorted_feature.size() < bin_cutoff) {
        if (it != bins.begin()) {
          auto prev = std::prev(it);
          prev->upper = it->upper;
          prev->count += it->count;
          prev->count_pos += it->count_pos;
          prev->count_neg += it->count_neg;
          it = bins.erase(it);
        } else {
          auto next = std::next(it);
          next->lower = it->lower;
          next->count += it->count;
          next->count_pos += it->count_pos;
          next->count_neg += it->count_neg;
          it = bins.erase(it);
        }
      } else {
        ++it;
      }
    }

    // Ensure number of bins is within limits
    while (static_cast<int>(bins.size()) > max_bins) {
      // Find the pair of adjacent bins with the smallest difference in WoE
      auto min_diff_it = bins.begin();
      double min_diff = std::numeric_limits<double>::max();

      for (auto it = bins.begin(); it != std::prev(bins.end()); ++it) {
        double woe1 = calculate_woe(it->count_pos, it->count_neg);
        double woe2 = calculate_woe(std::next(it)->count_pos, std::next(it)->count_neg);
        double diff = std::abs(woe1 - woe2);

        if (diff < min_diff) {
          min_diff = diff;
          min_diff_it = it;
        }
      }

      // Merge the bins
      auto next_bin = std::next(min_diff_it);
      min_diff_it->upper = next_bin->upper;
      min_diff_it->count += next_bin->count;
      min_diff_it->count_pos += next_bin->count_pos;
      min_diff_it->count_neg += next_bin->count_neg;
      bins.erase(next_bin);
    }

    // Calculate WoE and IV for each bin
    double total_iv = 0.0;
#pragma omp parallel for reduction(+:total_iv)
    for (auto& bin : bins) {
      bin.woe = calculate_woe(bin.count_pos, bin.count_neg);
      bin.iv = calculate_iv(bin.woe, bin.count_pos, bin.count_neg, total_pos, total_neg);
      total_iv += bin.iv;
    }
  }

  Rcpp::List get_results() {
    std::vector<std::string> bin_labels;
    std::vector<double> woe_values;
    std::vector<double> iv_values;
    std::vector<int> counts;
    std::vector<int> counts_pos;
    std::vector<int> counts_neg;

    for (const auto& bin : bins) {
      std::string label = "(" + (bin.lower == std::numeric_limits<double>::lowest() ? "-Inf" : std::to_string(bin.lower)) + ";" +
        (bin.upper == std::numeric_limits<double>::max() ? "+Inf" : std::to_string(bin.upper)) + "]";
      bin_labels.push_back(label);
      woe_values.push_back(bin.woe);
      iv_values.push_back(bin.iv);
      counts.push_back(bin.count);
      counts_pos.push_back(bin.count_pos);
      counts_neg.push_back(bin.count_neg);
    }

    return Rcpp::List::create(
      Rcpp::Named("woebin") = Rcpp::DataFrame::create(
        Rcpp::Named("bin") = bin_labels,
        Rcpp::Named("woe") = woe_values,
        Rcpp::Named("iv") = iv_values,
        Rcpp::Named("count") = counts,
        Rcpp::Named("count_pos") = counts_pos,
        Rcpp::Named("count_neg") = counts_neg
      )
    );
  }

  std::vector<double> transform(const std::vector<double>& new_feature) {
    std::vector<double> woe_feature(new_feature.size());

#pragma omp parallel for
    for (size_t i = 0; i < new_feature.size(); ++i) {
      auto it = std::lower_bound(bins.begin(), bins.end(), new_feature[i],
                                 [](const Bin& bin, double value) { return bin.upper < value; });
      if (it != bins.end()) {
        woe_feature[i] = it->woe;
      } else {
        woe_feature[i] = 0.0;  // or some other default value
      }
    }

    return woe_feature;
  }
};


//' @title Optimal Binning for Numerical Variables using Equal-Width Binning
//' 
//' @description
//' This function implements an optimal binning algorithm for numerical variables using an Equal-Width Binning approach with subsequent merging and adjustment. It aims to find a good binning strategy that balances interpretability and predictive power.
//' 
//' @param target An integer vector of binary target values (0 or 1).
//' @param feature A numeric vector of feature values to be binned.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum fraction of total observations in each bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins (default: 20).
//' 
//' @return A list containing:
//' \item{woefeature}{A numeric vector of Weight of Evidence (WoE) values for each observation}
//' \item{woebin}{A data frame with binning information, including bin ranges, WoE, IV, and counts}
//' 
//' @details
//' The optimal binning algorithm using Equal-Width Binning consists of several steps:
//' 
//' 1. Initial binning: The feature range is divided into \code{max_n_prebins} bins of equal width.
//' 2. Merging rare bins: Bins with a fraction of observations less than \code{bin_cutoff} are merged with adjacent bins.
//' 3. Adjusting number of bins: If the number of bins exceeds \code{max_bins}, adjacent bins with the most similar WoE values are merged until \code{max_bins} is reached.
//' 4. WoE and IV calculation: The Weight of Evidence (WoE) and Information Value (IV) are calculated for each bin.
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
//' This approach provides a balance between simplicity and effectiveness, creating bins of equal width initially and then adjusting them based on the data distribution and target variable relationship.
//' 
//' @examples
//' \dontrun{
//' set.seed(123)
//' target <- sample(0:1, 1000, replace = TRUE)
//' feature <- rnorm(1000)
//' result <- optimal_binning_numerical_ewb(target, feature)
//' print(result$woebin)
//' }
//' 
//' @references
//' \itemize{
//'   \item Dougherty, J., Kohavi, R., & Sahami, M. (1995). Supervised and unsupervised discretization of continuous features. In Machine Learning Proceedings 1995 (pp. 194-202). Morgan Kaufmann.
//'   \item Liu, H., Hussain, F., Tan, C. L., & Dash, M. (2002). Discretization: An enabling technique. Data mining and knowledge discovery, 6(4), 393-423.
//' }
//' 
//' @author Lopes, J. E.
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_ewb(const std::vector<int>& target,
                                         const std::vector<double>& feature,
                                         int min_bins = 3, int max_bins = 5, double bin_cutoff = 0.05, int max_n_prebins = 20) {
  OptimalBinningNumericalEWB binner(feature, target, min_bins, max_bins, bin_cutoff, max_n_prebins);
  binner.fit();
  Rcpp::List results = binner.get_results();
  std::vector<double> woe_feature = binner.transform(feature);
  results["woefeature"] = woe_feature;
  return results;
}
