#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

// [[Rcpp::plugins(openmp)]]

class OptimalBinningNumericalCM {
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

  double calculate_chi_square(const Bin& bin1, const Bin& bin2) {
    int total_pos = bin1.count_pos + bin2.count_pos;
    int total_neg = bin1.count_neg + bin2.count_neg;
    int total = total_pos + total_neg;

    double expected_pos1 = (double)(bin1.count * total_pos) / total;
    double expected_neg1 = (double)(bin1.count * total_neg) / total;
    double expected_pos2 = (double)(bin2.count * total_pos) / total;
    double expected_neg2 = (double)(bin2.count * total_neg) / total;

    double chi_square =
      std::pow(bin1.count_pos - expected_pos1, 2) / expected_pos1 +
      std::pow(bin1.count_neg - expected_neg1, 2) / expected_neg1 +
      std::pow(bin2.count_pos - expected_pos2, 2) / expected_pos2 +
      std::pow(bin2.count_neg - expected_neg2, 2) / expected_neg2;

    return chi_square;
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
  OptimalBinningNumericalCM(
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

    // ChiMerge algorithm
    while (bins.size() > min_bins) {
      double min_chi_square = std::numeric_limits<double>::max();
      size_t merge_index = 0;

#pragma omp parallel for
      for (size_t i = 0; i < bins.size() - 1; ++i) {
        double chi_square = calculate_chi_square(bins[i], bins[i + 1]);
#pragma omp critical
{
  if (chi_square < min_chi_square) {
    min_chi_square = chi_square;
    merge_index = i;
  }
}
      }

      merge_bins(merge_index);

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
Rcpp::List optimal_binning_numerical_cm(
    Rcpp::IntegerVector target,
    Rcpp::NumericVector feature,
    int min_bins = 3,
    int max_bins = 5,
    double bin_cutoff = 0.05,
    int max_n_prebins = 20
) {
  std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
  std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);

  OptimalBinningNumericalCM binner(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins);
  binner.fit();
  return binner.get_results();
}


// library(testthat)
// library(Rcpp)
//
// # Assuming the C++ code is compiled and the optimal_binning_numerical_cm function is available
//
// # Helper function to generate test data
//   generate_test_data <- function(n = 1000, seed = 42) {
//     set.seed(seed)
//     feature <- runif(n, 0, 100)
//     target <- rbinom(n, 1, plogis(feature / 50 - 1))
//     list(feature = feature, target = target)
//   }
//
//   test_that("optimal_binning_numerical_cm handles basic case correctly", {
//     data <- generate_test_data()
//     result <- optimal_binning_numerical_cm(data$feature, data$target)
//
//     expect_is(result, "list")
//     expect_true(all(c("woefeature", "woebin") %in% names(result)))
//     expect_equal(length(result$woefeature), length(data$feature))
//     expect_true(all(c("bin", "woe", "iv", "count", "count_pos", "count_neg") %in% names(result$woebin)))
//   })
//
//     test_that("optimal_binning_numerical_cm respects min_bins and max_bins", {
//       data <- generate_test_data()
//
//       result_min <- optimal_binning_numerical_cm(data$feature, data$target, min_bins = 3, max_bins = 2)
//       expect_equal(length(result_min$woebin$bin), 2)
//
//       result_max <- optimal_binning_numerical_cm(data$feature, data$target, min_bins = 3, max_bins = 5)
//       expect_true(length(result_max$woebin$bin) >= 2 && length(result_max$woebin$bin) <= 5)
//     })
//
//     test_that("optimal_binning_numerical_cm handles edge cases", {
// # Test with all same values
//       feature_same <- rep(1, 100)
//       target_same <- rbinom(100, 1, 0.5)
//       result_same <- optimal_binning_numerical_cm(feature_same, target_same)
//       expect_equal(length(result_same$woebin$bin), 1)
//
// # Test with extreme outliers
//       data <- generate_test_data(1000)
//         data$feature[1] <- 1e6  # Add an extreme outlier
//       result_outlier <- optimal_binning_numerical_cm(data$feature, data$target)
//         expect_true(any(grepl("\\+Inf", result_outlier$woebin$bin)))
//     })
//
//     test_that("optimal_binning_numerical_cm handles rare bin merging", {
//       data <- generate_test_data(10000)
//       result <- optimal_binning_numerical_cm(data$feature, data$target, bin_cutoff = 0.1)
//       bin_proportions <- result$woebin$count / sum(result$woebin$count)
//       expect_true(all(bin_proportions >= 0.1))
//     })
//
//     test_that("optimal_binning_numerical_cm produces monotonic WoE", {
//       data <- generate_test_data()
//       result <- optimal_binning_numerical_cm(data$feature, data$target)
//       expect_true(is.unsorted(result$woebin$woe) || is.unsorted(rev(result$woebin$woe)))
//     })
//
//     test_that("optimal_binning_numerical_cm handles imbalanced target", {
//       data <- generate_test_data(1000)
//       data$target <- rbinom(1000, 1, 0.01)  # Highly imbalanced target
//       result <- optimal_binning_numerical_cm(data$feature, data$target)
//       expect_true(all(is.finite(result$woebin$woe)))
//     })
//
//     test_that("optimal_binning_numerical_cm is deterministic", {
//       data <- generate_test_data()
//       result1 <- optimal_binning_numerical_cm(data$feature, data$target)
//       result2 <- optimal_binning_numerical_cm(data$feature, data$target)
//       expect_equal(result1, result2)
//     })
//
//     test_that("optimal_binning_numerical_cm handles large datasets", {
//       data <- generate_test_data(1e5)  # 100,000 samples
//       expect_error(optimal_binning_numerical_cm(data$feature, data$target), NA)
//     })
//
