#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

class OptimalBinningNumericalJNBO {
private:
  std::vector<double> feature;
  std::vector<int> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;

  struct Bin {
    double lower_bound;
    double upper_bound;
    int count;
    int count_pos;
    int count_neg;
    double woe;
    mutable double iv;  // Make iv mutable so it can be modified in const contexts
  };

  std::vector<Bin> bins;

  double calculate_woe(int pos, int neg, int total_pos, int total_neg) const {
    double pos_rate = pos / static_cast<double>(total_pos);
    double neg_rate = neg / static_cast<double>(total_neg);
    return std::log((pos_rate + 1e-6) / (neg_rate + 1e-6));
  }

  double calculate_iv(const std::vector<Bin>& bins, int total_pos, int total_neg) const {
    double iv = 0.0;
    for (auto& bin : bins) {
      double pos_rate = bin.count_pos / static_cast<double>(total_pos);
      double neg_rate = bin.count_neg / static_cast<double>(total_neg);
      bin.iv = (pos_rate - neg_rate) * bin.woe;
      iv += bin.iv;
    }
    return iv;
  }

  void pre_binning() {
    std::vector<std::pair<double, int>> sorted_data;
    sorted_data.reserve(feature.size());
    for (size_t i = 0; i < feature.size(); ++i) {
      sorted_data.emplace_back(feature[i], target[i]);
    }
    std::sort(sorted_data.begin(), sorted_data.end());

    int total_count = sorted_data.size();
    int bin_size = std::max(1, total_count / max_n_prebins);

    bins.clear();
    int current_count = 0;
    int current_pos = 0;
    int current_neg = 0;
    double current_lower = -std::numeric_limits<double>::infinity();

    for (size_t i = 0; i < sorted_data.size(); ++i) {
      current_count++;
      current_pos += sorted_data[i].second;
      current_neg += 1 - sorted_data[i].second;

      if (current_count >= bin_size || i == sorted_data.size() - 1 || sorted_data[i].first != sorted_data[i+1].first) {
        Bin bin;
        bin.lower_bound = current_lower;
        bin.upper_bound = sorted_data[i].first;
        bin.count = current_count;
        bin.count_pos = current_pos;
        bin.count_neg = current_neg;
        bins.push_back(bin);

        current_lower = sorted_data[i].first;
        current_count = 0;
        current_pos = 0;
        current_neg = 0;
      }
    }

    // Merge rare bins
    merge_rare_bins();
  }

  void merge_rare_bins() {
    std::vector<Bin> merged_bins;
    Bin current_bin = bins[0];

    for (size_t i = 1; i < bins.size(); ++i) {
      if (static_cast<double>(current_bin.count) / feature.size() < bin_cutoff) {
        current_bin.upper_bound = bins[i].upper_bound;
        current_bin.count += bins[i].count;
        current_bin.count_pos += bins[i].count_pos;
        current_bin.count_neg += bins[i].count_neg;
      } else {
        merged_bins.push_back(current_bin);
        current_bin = bins[i];
      }
    }

    merged_bins.push_back(current_bin);

    bins = merged_bins;
  }

  void optimize_bins() {
    int n = bins.size();
    std::vector<std::vector<double>> dp(n + 1, std::vector<double>(max_bins + 1, -std::numeric_limits<double>::infinity()));
    std::vector<std::vector<int>> split(n + 1, std::vector<int>(max_bins + 1, 0));

    int total_pos = 0, total_neg = 0;
    for (const auto& bin : bins) {
      total_pos += bin.count_pos;
      total_neg += bin.count_neg;
    }

    // Initialize base cases
    for (int i = 1; i <= n; ++i) {
      dp[i][1] = calculate_iv(std::vector<Bin>(bins.begin(), bins.begin() + i), total_pos, total_neg);
    }

    // Dynamic programming to find optimal binning
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int j = 2; j <= max_bins; ++j) {
      for (int i = j; i <= n; ++i) {
        for (int k = j - 1; k < i; ++k) {
          double current_iv = dp[k][j-1] + calculate_iv(std::vector<Bin>(bins.begin() + k, bins.begin() + i), total_pos, total_neg);
          if (current_iv > dp[i][j]) {
            dp[i][j] = current_iv;
            split[i][j] = k;
          }
        }
      }
    }

    // Find the optimal number of bins
    int optimal_bins = max_bins;
    for (int j = min_bins; j <= max_bins; ++j) {
      if (dp[n][j] > dp[n][optimal_bins]) {
        optimal_bins = j;
      }
    }

    // Backtrack to find the optimal solution
    std::vector<Bin> optimal_bins_vec;
    int i = n;
    int j = optimal_bins;
    while (j > 0) {
      int k = split[i][j];
      Bin merged_bin = bins[i-1];
      for (int l = k; l < i - 1; ++l) {
        merged_bin.lower_bound = bins[l].lower_bound;
        merged_bin.count += bins[l].count;
        merged_bin.count_pos += bins[l].count_pos;
        merged_bin.count_neg += bins[l].count_neg;
      }
      merged_bin.woe = calculate_woe(merged_bin.count_pos, merged_bin.count_neg, total_pos, total_neg);
      optimal_bins_vec.insert(optimal_bins_vec.begin(), merged_bin);
      i = k;
      --j;
    }

    bins = optimal_bins_vec;
    calculate_iv(bins, total_pos, total_neg); // Calculate IV for final bins
  }

public:
  OptimalBinningNumericalJNBO(const std::vector<double>& feature, const std::vector<int>& target,
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
    pre_binning();
    optimize_bins();
  }

  Rcpp::List get_result() const {
    std::vector<double> woefeature(feature.size());
    std::vector<std::string> bin_labels;
    std::vector<double> woe_values;
    std::vector<double> iv_values;
    std::vector<int> count_values;
    std::vector<int> count_pos_values;
    std::vector<int> count_neg_values;

    double total_iv = 0.0;
    for (const auto& bin : bins) {
      std::string bin_label = "(" + std::to_string(bin.lower_bound) + ";" + std::to_string(bin.upper_bound) + "]";
      bin_labels.push_back(bin_label);
      woe_values.push_back(bin.woe);
      iv_values.push_back(bin.iv);
      count_values.push_back(bin.count);
      count_pos_values.push_back(bin.count_pos);
      count_neg_values.push_back(bin.count_neg);
      total_iv += bin.iv;
    }

    // Aplicar WoE à feature
    for (size_t i = 0; i < feature.size(); ++i) {
      bool assigned = false;
      for (const auto& bin : bins) {
        if (feature[i] > bin.lower_bound && feature[i] <= bin.upper_bound) {
          woefeature[i] = bin.woe;
          assigned = true;
          break;
        }
      }
      if (!assigned) {
        // Se o valor não se encaixar em nenhum bin, atribuímos o WoE do primeiro ou último bin
        if (feature[i] <= bins.front().upper_bound) {
          woefeature[i] = bins.front().woe;
        } else {
          woefeature[i] = bins.back().woe;
        }
      }
    }

    Rcpp::DataFrame woebin = Rcpp::DataFrame::create(
      Rcpp::Named("bin") = bin_labels,
      Rcpp::Named("woe") = woe_values,
      Rcpp::Named("iv") = iv_values,
      Rcpp::Named("count") = count_values,
      Rcpp::Named("count_pos") = count_pos_values,
      Rcpp::Named("count_neg") = count_neg_values
    );

    return Rcpp::List::create(
      Rcpp::Named("woefeature") = woefeature,
      Rcpp::Named("woebin") = woebin,
      Rcpp::Named("total_iv") = total_iv
    );
  }

};


//' @title Optimal Binning for Numerical Variables using Dynamic Programming
//' 
//' @description
//' This function implements an optimal binning algorithm for numerical variables using dynamic programming. It aims to find the best binning strategy that maximizes the Information Value (IV) while respecting the specified constraints.
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
//' \item{total_iv}{The total Information Value for the binning}
//' 
//' @details
//' The optimal binning algorithm uses dynamic programming to find the best binning strategy that maximizes the Information Value (IV) while respecting the specified constraints. The algorithm consists of several steps:
//' 
//' 1. Pre-binning: The feature is initially divided into a maximum number of bins specified by \code{max_n_prebins}.
//' 2. Merging rare bins: Bins with a fraction of observations less than \code{bin_cutoff} are merged with adjacent bins.
//' 3. Dynamic programming optimization: The algorithm uses dynamic programming to find the optimal binning strategy that maximizes the total IV.
//' 
//' The Weight of Evidence (WoE) for each bin is calculated as:
//' 
//' \deqn{WoE = \ln\left(\frac{P(X|Y=1)}{P(X|Y=0)}\right)}
//' 
//' where \eqn{P(X|Y=1)} is the probability of the feature being in a particular bin given a positive target, and \eqn{P(X|Y=0)} is the probability given a negative target.
//' 
//' The Information Value (IV) for each bin is calculated as:
//' 
//' \deqn{IV = (P(X|Y=1) - P(X|Y=0)) \times WoE}
//' 
//' The total IV is the sum of IVs for all bins:
//' 
//' \deqn{\text{Total IV} = \sum_{i=1}^{n} IV_i}
//' 
//' The dynamic programming approach ensures that the global optimum is found within the constraints of the minimum and maximum number of bins.
//' 
//' @examples
//' \dontrun{
//' set.seed(123)
//' target <- sample(0:1, 1000, replace = TRUE)
//' feature <- rnorm(1000)
//' result <- optimal_binning_numerical_jnbo(target, feature)
//' print(result$woebin)
//' print(result$total_iv)
//' }
//' 
//' @references
//' \itemize{
//'   \item Belotti, P., & Carrasco, M. (2016). Optimal Binning: Mathematical Programming Formulation and Solution Approach. \emph{arXiv preprint arXiv:1605.05710}.
//'   \item Gutiérrez, P. A., Pérez-Ortiz, M., Sánchez-Monedero, J., Fernández-Navarro, F., & Hervás-Martínez, C. (2016). Ordinal regression methods: survey and experimental study. \emph{IEEE Transactions on Knowledge and Data Engineering}, 28(1), 127-146.
//' }
//' 
//' @author Lopes, J. E.
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_jnbo(Rcpp::IntegerVector target,
                                          Rcpp::NumericVector feature,
                                          int min_bins = 3,
                                          int max_bins = 5,
                                          double bin_cutoff = 0.05,
                                          int max_n_prebins = 20) {
  std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
  std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);

  OptimalBinningNumericalJNBO binning(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins);
  binning.fit();
  return binning.get_result();
}
