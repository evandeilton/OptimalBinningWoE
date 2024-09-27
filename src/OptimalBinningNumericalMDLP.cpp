#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

// [[Rcpp::plugins(openmp)]]

class OptimalBinningNumericalMDLP {
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
    double iv;
  };

  std::vector<Bin> bins;

  double calculate_entropy(int pos, int neg) {
    if (pos == 0 || neg == 0) return 0.0;
    double p = static_cast<double>(pos) / (pos + neg);
    return -p * std::log2(p) - (1 - p) * std::log2(1 - p);
  }

  double calculate_mdl_cost(const std::vector<Bin>& current_bins) {
    double total_count = 0;
    double total_pos = 0;
    for (const auto& bin : current_bins) {
      total_count += bin.count;
      total_pos += bin.count_pos;
    }
    double total_neg = total_count - total_pos;

    double model_cost = std::log2(current_bins.size() - 1);
    double data_cost = total_count * calculate_entropy(total_pos, total_neg);

    for (const auto& bin : current_bins) {
      data_cost -= bin.count * calculate_entropy(bin.count_pos, bin.count_neg);
    }

    return model_cost + data_cost;
  }

  void merge_bins(size_t index) {
    Bin& left = bins[index];
    Bin& right = bins[index + 1];

    left.upper_bound = right.upper_bound;
    left.count += right.count;
    left.count_pos += right.count_pos;
    left.count_neg += right.count_neg;

    bins.erase(bins.begin() + index + 1);
  }

  void calculate_woe_iv() {
    double total_pos = 0, total_neg = 0;
    for (const auto& bin : bins) {
      total_pos += bin.count_pos;
      total_neg += bin.count_neg;
    }

    for (auto& bin : bins) {
      double pos_rate = (double)bin.count_pos / total_pos;
      double neg_rate = (double)bin.count_neg / total_neg;
      bin.woe = std::log(pos_rate / neg_rate);
      bin.iv = (pos_rate - neg_rate) * bin.woe;
    }
  }

public:
  OptimalBinningNumericalMDLP(
    const std::vector<double>& feature,
    const std::vector<int>& target,
    int min_bins = 3,
    int max_bins = 5,
    double bin_cutoff = 0.05,
    int max_n_prebins = 20
  ) : feature(feature), target(target), min_bins(min_bins), max_bins(max_bins),
  bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins) {

    if (feature.size() != target.size()) {
      Rcpp::stop("Feature and target vectors must have the same length");
    }

    if (min_bins < 2 || max_bins < min_bins) {
      Rcpp::stop("Invalid bin constraints");
    }
  }

  void fit() {
    // Initial binning
    std::vector<std::pair<double, int>> sorted_data;
    for (size_t i = 0; i < feature.size(); ++i) {
      sorted_data.push_back({feature[i], target[i]});
    }
    std::sort(sorted_data.begin(), sorted_data.end());

    // Create initial bins
    int records_per_bin = std::max(1, (int)sorted_data.size() / max_n_prebins);
    for (size_t i = 0; i < sorted_data.size(); i += records_per_bin) {
      size_t end = std::min(i + records_per_bin, sorted_data.size());
      Bin bin;
      bin.lower_bound = (i == 0) ? -std::numeric_limits<double>::infinity() : sorted_data[i].first;
      bin.upper_bound = (end == sorted_data.size()) ? std::numeric_limits<double>::infinity() : sorted_data[end - 1].first;
      bin.count = end - i;
      bin.count_pos = 0;
      bin.count_neg = 0;

      for (size_t j = i; j < end; ++j) {
        if (sorted_data[j].second == 1) {
          bin.count_pos++;
        } else {
          bin.count_neg++;
        }
      }

      bins.push_back(bin);
    }

    // MDLP algorithm
    while (bins.size() > min_bins) {
      double current_mdl = calculate_mdl_cost(bins);
      double best_mdl = current_mdl;
      size_t best_merge_index = bins.size();

#pragma omp parallel for
      for (size_t i = 0; i < bins.size() - 1; ++i) {
        std::vector<Bin> temp_bins = bins;
        Bin& left = temp_bins[i];
        Bin& right = temp_bins[i + 1];

        left.upper_bound = right.upper_bound;
        left.count += right.count;
        left.count_pos += right.count_pos;
        left.count_neg += right.count_neg;

        temp_bins.erase(temp_bins.begin() + i + 1);

        double new_mdl = calculate_mdl_cost(temp_bins);

#pragma omp critical
{
  if (new_mdl < best_mdl) {
    best_mdl = new_mdl;
    best_merge_index = i;
  }
}
      }

      if (best_merge_index < bins.size()) {
        merge_bins(best_merge_index);
      } else {
        break;
      }

      if (bins.size() <= max_bins) {
        break;
      }
    }

    // Merge rare bins
    for (auto it = bins.begin(); it != bins.end(); ) {
      if ((double)it->count / feature.size() < bin_cutoff) {
        if (it == bins.begin()) {
          merge_bins(0);
        } else {
          merge_bins(std::distance(bins.begin(), it) - 1);
        }
      } else {
        ++it;
      }
    }

    calculate_woe_iv();
  }

  Rcpp::List get_results() {
    Rcpp::NumericVector woefeature(feature.size());
    Rcpp::List woebin;
    Rcpp::StringVector bin_labels;
    Rcpp::NumericVector woe_values;
    Rcpp::NumericVector iv_values;
    Rcpp::IntegerVector count_values;
    Rcpp::IntegerVector count_pos_values;
    Rcpp::IntegerVector count_neg_values;

    for (size_t i = 0; i < feature.size(); ++i) {
      for (const auto& bin : bins) {
        if (feature[i] <= bin.upper_bound) {
          woefeature[i] = bin.woe;
          break;
        }
      }
    }

    for (const auto& bin : bins) {
      std::string bin_label = (bin.lower_bound == -std::numeric_limits<double>::infinity() ? "(-Inf" : "(" + std::to_string(bin.lower_bound)) +
        ";" + (bin.upper_bound == std::numeric_limits<double>::infinity() ? "+Inf]" : std::to_string(bin.upper_bound) + "]");
      bin_labels.push_back(bin_label);
      woe_values.push_back(bin.woe);
      iv_values.push_back(bin.iv);
      count_values.push_back(bin.count);
      count_pos_values.push_back(bin.count_pos);
      count_neg_values.push_back(bin.count_neg);
    }

    woebin["bin"] = bin_labels;
    woebin["woe"] = woe_values;
    woebin["iv"] = iv_values;
    woebin["count"] = count_values;
    woebin["count_pos"] = count_pos_values;
    woebin["count_neg"] = count_neg_values;

    return Rcpp::List::create(
      Rcpp::Named("woefeature") = woefeature,
      Rcpp::Named("woebin") = woebin
    );
  }
};


//' @title Optimal Binning for Numerical Variables using MDLP
//' 
//' @description
//' This function performs optimal binning for numerical variables using the Minimum Description Length Principle (MDLP). It creates optimal bins for a numerical feature based on its relationship with a binary target variable, maximizing the predictive power while respecting user-defined constraints.
//' 
//' @param target An integer vector of binary target values (0 or 1).
//' @param feature A numeric vector of feature values.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum proportion of total observations for a bin to avoid being merged (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins before the optimization process (default: 20).
//' 
//' @return A list containing two elements:
//' \item{woefeature}{A numeric vector of Weight of Evidence (WoE) values for each observation.}
//' \item{woebin}{A data frame with the following columns:
//'   \itemize{
//'     \item bin: Character vector of bin ranges.
//'     \item woe: Numeric vector of WoE values for each bin.
//'     \item iv: Numeric vector of Information Value (IV) for each bin.
//'     \item count: Integer vector of total observations in each bin.
//'     \item count_pos: Integer vector of positive target observations in each bin.
//'     \item count_neg: Integer vector of negative target observations in each bin.
//'   }
//' }
//' 
//' @details
//' The Optimal Binning algorithm for numerical variables using MDLP works as follows:
//' 1. Create initial bins using equal-frequency binning.
//' 2. Apply the MDLP algorithm to merge bins:
//'    - Calculate the current MDL cost.
//'    - For each pair of adjacent bins, calculate the MDL cost if merged.
//'    - Merge the pair with the lowest MDL cost.
//'    - Repeat until no further merging reduces the MDL cost or the minimum number of bins is reached.
//' 3. Merge rare bins (those with a proportion less than bin_cutoff).
//' 4. Calculate Weight of Evidence (WoE) and Information Value (IV) for each bin:
//'    \deqn{WoE = \ln\left(\frac{\text{Positive Rate}}{\text{Negative Rate}}\right)}
//'    \deqn{IV = (\text{Positive Rate} - \text{Negative Rate}) \times WoE}
//' 
//' The MDLP algorithm aims to find the optimal trade-off between model complexity (number of bins) and goodness of fit. It uses the principle of minimum description length, which states that the best model is the one that provides the shortest description of the data.
//' 
//' The MDL cost is calculated as:
//' \deqn{MDL = \log_2(k - 1) + n \times H(S) - \sum_{i=1}^k n_i \times H(S_i)}
//' where k is the number of bins, n is the total number of instances, H(S) is the entropy of the entire dataset, and H(S_i) is the entropy of the i-th bin.
//' 
//' This implementation uses OpenMP for parallel processing when available, which can significantly speed up the computation for large datasets.
//' 
//' @examples
//' \dontrun{
//' # Create sample data
//' set.seed(123)
//' n <- 1000
//' target <- sample(0:1, n, replace = TRUE)
//' feature <- rnorm(n)
//' 
//' # Run optimal binning
//' result <- optimal_binning_numerical_mdlp(target, feature, min_bins = 2, max_bins = 4)
//' 
//' # Print results
//' print(result$woebin)
//' 
//' # Plot WoE values
//' plot(result$woebin$woe, type = "s", xaxt = "n", xlab = "Bins", ylab = "WoE",
//'      main = "Weight of Evidence by Bin")
//' axis(1, at = 1:nrow(result$woebin), labels = result$woebin$bin)
//' }
//' 
//' @references
//' \itemize{
//' \item Fayyad, U. M., & Irani, K. B. (1993). Multi-interval discretization of continuous-valued attributes for classification learning. In Proceedings of the 13th International Joint Conference on Artificial Intelligence (pp. 1022-1027).
//' \item Rissanen, J. (1978). Modeling by shortest data description. Automatica, 14(5), 465-471.
//' }
//' 
//' @author Lopes, J. E.
//' 
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_mdlp(
    Rcpp::IntegerVector target,
    Rcpp::NumericVector feature,
    int min_bins = 3,
    int max_bins = 5,
    double bin_cutoff = 0.05,
    int max_n_prebins = 20
) {
  std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
  std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);

  OptimalBinningNumericalMDLP binner(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins);
  binner.fit();
  return binner.get_results();
}

