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


// library(testthat)
//   library(Rcpp)
//
// # Assuming the C++ code is compiled and the optimal_binning_numerical_mdlp function is available
//
// # Helper function to generate test data
//   generate_test_data <- function(n = 1000, seed = 42) {
//     set.seed(seed)
//     feature <- runif(n, 0, 100)
//     target <- rbinom(n, 1, plogis(feature / 50 - 1))
//     list(feature = feature, target = target)
//   }
//
//   test_that("optimal_binning_numerical_mdlp handles basic case correctly", {
//     data <- generate_test_data()
//     result <- optimal_binning_numerical_mdlp(data$feature, data$target)
//
//     expect_is(result, "list")
//     expect_true(all(c("woefeature", "woebin") %in% names(result)))
//     expect_equal(length(result$woefeature), length(data$feature))
//     expect_true(all(c("bin", "woe", "iv", "count", "count_pos", "count_neg") %in% names(result$woebin)))
//   })
//
//     test_that("optimal_binning_numerical_mdlp respects min_bins and max_bins", {
//       data <- generate_test_data()
//
//       result_min <- optimal_binning_numerical_mdlp(data$feature, data$target, min_bins = 3, max_bins = 2)
//       expect_equal(length(result_min$woebin$bin), 2)
//
//       result_max <- optimal_binning_numerical_mdlp(data$feature, data$target, min_bins = 3, max_bins = 5)
//       expect_true(length(result_max$woebin$bin) >= 2 && length(result_max$woebin$bin) <= 5)
//     })
//
//     test_that("optimal_binning_numerical_mdlp handles edge cases", {
// # Test with all same values
//       feature_same <- rep(1, 100)
//       target_same <- rbinom(100, 1, 0.5)
//       result_same <- optimal_binning_numerical_mdlp(feature_same, target_same)
//       expect_equal(length(result_same$woebin$bin), 1)
//
// # Test with extreme outliers
//       data <- generate_test_data(1000)
//         data$feature[1] <- 1e6  # Add an extreme outlier
//       result_outlier <- optimal_binning_numerical_mdlp(data$feature, data$target)
//         expect_true(any(grepl("\\+Inf", result_outlier$woebin$bin)))
//     })
//
//     test_that("optimal_binning_numerical_mdlp handles rare bin merging", {
//       data <- generate_test_data(10000)
//       result <- optimal_binning_numerical_mdlp(data$feature, data$target, bin_cutoff = 0.1)
//       bin_proportions <- result$woebin$count / sum(result$woebin$count)
//       expect_true(all(bin_proportions >= 0.1))
//     })
//
//     test_that("optimal_binning_numerical_mdlp produces monotonic WoE", {
//       data <- generate_test_data()
//       result <- optimal_binning_numerical_mdlp(data$feature, data$target)
//       expect_true(is.unsorted(result$woebin$woe) || is.unsorted(rev(result$woebin$woe)))
//     })
//
//     test_that("optimal_binning_numerical_mdlp handles imbalanced target", {
//       data <- generate_test_data(1000)
//       data$target <- rbinom(1000, 1, 0.01)  # Highly imbalanced target
//       result <- optimal_binning_numerical_mdlp(data$feature, data$target)
//       expect_true(all(is.finite(result$woebin$woe)))
//     })
//
//     test_that("optimal_binning_numerical_mdlp is deterministic", {
//       data <- generate_test_data()
//       result1 <- optimal_binning_numerical_mdlp(data$feature, data$target)
//       result2 <- optimal_binning_numerical_mdlp(data$feature, data$target)
//       expect_equal(result1, result2)
//     })
//
//     test_that("optimal_binning_numerical_mdlp handles large datasets", {
//       data <- generate_test_data(1e5)  # 100,000 samples
//       expect_error(optimal_binning_numerical_mdlp(data$feature, data$target), NA)
//     })
//
//     test_that("optimal_binning_numerical_mdlp produces expected Information Value", {
//       data <- generate_test_data()
//       result <- optimal_binning_numerical_mdlp(data$feature, data$target)
//       total_iv <- sum(result$woebin$iv)
//       expect_true(total_iv > 0 && total_iv < 2)  # Typical range for a predictive feature
//     })
//
//     test_that("optimal_binning_numerical_mdlp handles constant features", {
//       feature_constant <- rep(5, 1000)
//       target_random <- rbinom(1000, 1, 0.5)
//       expect_warning(
//         result <- optimal_binning_numerical_mdlp(feature_constant, target_random),
//                 "Constant feature detected"
//       )
//       expect_equal(length(result$woebin$bin), 1)
//     })
//
//
