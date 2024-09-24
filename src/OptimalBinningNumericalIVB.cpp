// [[Rcpp::plugins(openmp)]]
#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <sstream>
#ifdef _OPENMP
#include <omp.h>
#endif

class OptimalBinningNumericalIVB {
private:
  int min_bins;
  int max_bins;
  const double bin_cutoff = 0.05;  // Fixed value as per requirement
  int max_n_prebins;
  std::vector<double> feature;
  std::vector<int> target;
  std::vector<double> unique_values;
  std::vector<int> value_counts;
  std::vector<int> pos_counts;
  std::vector<double> woe_values;
  std::vector<double> iv_values;
  std::vector<std::string> bin_intervals;

  void validate_inputs() {
    if (min_bins < 2) {
      Rcpp::stop("min_bins must be at least 2");
    }
    if (max_bins < min_bins) {
      Rcpp::stop("max_bins must be greater than or equal to min_bins");
    }
    if (max_n_prebins <= 0) {
      Rcpp::stop("max_n_prebins must be positive");
    }
    if (feature.size() != target.size()) {
      Rcpp::stop("feature and target must have the same length");
    }
    for (int t : target) {
      if (t != 0 && t != 1) {
        Rcpp::stop("target must contain only 0 and 1");
      }
    }
  }

  void preprocess() {
    std::vector<std::pair<double, int>> sorted_data;
    for (size_t i = 0; i < feature.size(); ++i) {
      sorted_data.push_back({feature[i], target[i]});
    }
    std::sort(sorted_data.begin(), sorted_data.end());

    unique_values.clear();
    value_counts.clear();
    pos_counts.clear();

    double current_value = sorted_data[0].first;
    int current_count = 1;
    int current_pos_count = sorted_data[0].second;

    for (size_t i = 1; i < sorted_data.size(); ++i) {
      if (sorted_data[i].first != current_value) {
        unique_values.push_back(current_value);
        value_counts.push_back(current_count);
        pos_counts.push_back(current_pos_count);

        current_value = sorted_data[i].first;
        current_count = 1;
        current_pos_count = sorted_data[i].second;
      } else {
        current_count++;
        current_pos_count += sorted_data[i].second;
      }
    }

    unique_values.push_back(current_value);
    value_counts.push_back(current_count);
    pos_counts.push_back(current_pos_count);

    // Apply max_n_prebins constraint
    if (static_cast<int>(unique_values.size()) > max_n_prebins) {
      int step = std::ceil(static_cast<double>(unique_values.size()) / max_n_prebins);
      std::vector<double> new_unique_values;
      std::vector<int> new_value_counts;
      std::vector<int> new_pos_counts;

      for (size_t i = 0; i < unique_values.size(); i += step) {
        new_unique_values.push_back(unique_values[i]);
        int total_count = 0;
        int total_pos_count = 0;
        for (size_t j = i; j < i + step && j < unique_values.size(); ++j) {
          total_count += value_counts[j];
          total_pos_count += pos_counts[j];
        }
        new_value_counts.push_back(total_count);
        new_pos_counts.push_back(total_pos_count);
      }

      unique_values = new_unique_values;
      value_counts = new_value_counts;
      pos_counts = new_pos_counts;
    }
  }

  void merge_rare_bins() {
    int total_count = 0;
    for (int count : value_counts) {
      total_count += count;
    }

    int min_count = static_cast<int>(bin_cutoff * total_count);

    std::vector<double> merged_unique_values;
    std::vector<int> merged_value_counts;
    std::vector<int> merged_pos_counts;

    int current_count = 0;
    int current_pos_count = 0;

    for (size_t i = 0; i < unique_values.size(); ++i) {
      current_count += value_counts[i];
      current_pos_count += pos_counts[i];

      if (current_count >= min_count || i == unique_values.size() - 1) {
        merged_unique_values.push_back(unique_values[i]);
        merged_value_counts.push_back(current_count);
        merged_pos_counts.push_back(current_pos_count);
        current_count = 0;
        current_pos_count = 0;
      }
    }

    unique_values = merged_unique_values;
    value_counts = merged_value_counts;
    pos_counts = merged_pos_counts;
  }

  void calculate_woe_iv() {
    int total_pos = 0;
    int total_neg = 0;
    for (size_t i = 0; i < pos_counts.size(); ++i) {
      total_pos += pos_counts[i];
      total_neg += value_counts[i] - pos_counts[i];
    }

    woe_values.clear();
    iv_values.clear();

    for (size_t i = 0; i < pos_counts.size(); ++i) {
      int neg_count = value_counts[i] - pos_counts[i];
      double pos_rate = static_cast<double>(pos_counts[i]) / total_pos;
      double neg_rate = static_cast<double>(neg_count) / total_neg;

      double woe = std::log((pos_rate + 1e-10) / (neg_rate + 1e-10));
      double iv = (pos_rate - neg_rate) * woe;

      woe_values.push_back(woe);
      iv_values.push_back(iv);
    }
  }

  void optimize_bins() {
    std::vector<int> best_split_points;
    double best_total_iv = 0.0;

    for (int num_bins = min_bins; num_bins <= max_bins; ++num_bins) {
      std::vector<int> split_points = find_optimal_split_points(num_bins);
      double total_iv = calculate_total_iv(split_points);

      if (total_iv > best_total_iv) {
        best_total_iv = total_iv;
        best_split_points = split_points;
      }
    }

    apply_split_points(best_split_points);
  }

  std::vector<int> find_optimal_split_points(int num_bins) {
    std::vector<std::vector<double>> dp(unique_values.size(), std::vector<double>(num_bins, -std::numeric_limits<double>::infinity()));
    std::vector<std::vector<int>> split_points(unique_values.size(), std::vector<int>(num_bins, -1));

    // Initialize first bin
    double cum_pos = 0, cum_neg = 0;
    for (size_t i = 0; i < unique_values.size(); ++i) {
      cum_pos += pos_counts[i];
      cum_neg += value_counts[i] - pos_counts[i];
      dp[i][0] = calculate_iv(0, i);
    }

    // Dynamic programming to find optimal split points
    for (int j = 1; j < num_bins; ++j) {
      for (int i = j; i < static_cast<int>(unique_values.size()); ++i) {
        for (int k = j - 1; k < i; ++k) {
          double iv = dp[k][j-1] + calculate_iv(k + 1, i);
          if (iv > dp[i][j]) {
            dp[i][j] = iv;
            split_points[i][j] = k;
          }
        }
      }
    }

    // Backtrack to find the optimal split points
    std::vector<int> result;
    int i = unique_values.size() - 1;
    for (int j = num_bins - 1; j > 0; --j) {
      result.push_back(split_points[i][j]);
      i = split_points[i][j];
    }
    std::reverse(result.begin(), result.end());
    return result;
  }

  double calculate_iv(int start, int end) {
    double pos_rate = 0, neg_rate = 0;
    int total_pos = 0, total_neg = 0;

    for (int i = start; i <= end; ++i) {
      pos_rate += pos_counts[i];
      neg_rate += value_counts[i] - pos_counts[i];
    }

    for (size_t i = 0; i < pos_counts.size(); ++i) {
      total_pos += pos_counts[i];
      total_neg += value_counts[i] - pos_counts[i];
    }

    pos_rate /= total_pos;
    neg_rate /= total_neg;

    double woe = std::log((pos_rate + 1e-10) / (neg_rate + 1e-10));
    return (pos_rate - neg_rate) * woe;
  }

  double calculate_total_iv(const std::vector<int>& split_points) {
    double total_iv = 0.0;
    int start = 0;
    for (int split : split_points) {
      total_iv += calculate_iv(start, split);
      start = split + 1;
    }
    total_iv += calculate_iv(start, unique_values.size() - 1);
    return total_iv;
  }

  void apply_split_points(const std::vector<int>& split_points) {
    std::vector<double> new_unique_values;
    std::vector<int> new_value_counts;
    std::vector<int> new_pos_counts;
    bin_intervals.clear();

    int start = 0;
    for (int split : split_points) {
      new_unique_values.push_back(unique_values[split]);
      int bin_count = 0;
      int bin_pos_count = 0;
      for (int i = start; i <= split; ++i) {
        bin_count += value_counts[i];
        bin_pos_count += pos_counts[i];
      }
      new_value_counts.push_back(bin_count);
      new_pos_counts.push_back(bin_pos_count);

      std::ostringstream oss;
      if (start == 0) {
        oss << "(-Inf;" << unique_values[split] << "]";
      } else {
        oss << "(" << unique_values[start - 1] << ";" << unique_values[split] << "]";
      }
      bin_intervals.push_back(oss.str());

      start = split + 1;
    }

    // Last bin
    new_unique_values.push_back(unique_values.back());
    int bin_count = 0;
    int bin_pos_count = 0;
    for (int i = start; i < static_cast<int>(unique_values.size()); ++i) {
      bin_count += value_counts[i];
      bin_pos_count += pos_counts[i];
    }
    new_value_counts.push_back(bin_count);
    new_pos_counts.push_back(bin_pos_count);

    std::ostringstream oss;
    oss << "(" << unique_values[start - 1] << ";+Inf]";
    bin_intervals.push_back(oss.str());

    unique_values = new_unique_values;
    value_counts = new_value_counts;
    pos_counts = new_pos_counts;
  }

public:
  OptimalBinningNumericalIVB(const std::vector<double>& feature_, const std::vector<int>& target_,
                             int min_bins_ = 2, int max_bins_ = 5, int max_n_prebins_ = 20)
    : feature(feature_), target(target_),
      min_bins(std::max(2, min_bins_)),
      max_bins(std::max(min_bins, max_bins_)),
      max_n_prebins(max_n_prebins_) {
    validate_inputs();
  }

  void fit() {
    preprocess();
    merge_rare_bins();
    optimize_bins();
    calculate_woe_iv();
  }

  std::vector<double> transform(const std::vector<double>& new_feature) {
    std::vector<double> woe_feature(new_feature.size());
#pragma omp parallel for
    for (size_t i = 0; i < new_feature.size(); ++i) {
      auto it = std::upper_bound(unique_values.begin(), unique_values.end(), new_feature[i]);
      int bin_index = std::distance(unique_values.begin(), it) - 1;
      if (bin_index < 0) bin_index = 0;
      if (bin_index >= static_cast<int>(woe_values.size())) bin_index = woe_values.size() - 1;
      woe_feature[i] = woe_values[bin_index];
    }
    return woe_feature;
  }

  Rcpp::NumericVector parallel_transform(const Rcpp::NumericVector& new_feature) {
    Rcpp::NumericVector woe_feature(new_feature.size());
#pragma omp parallel for
    for (size_t i = 0; i < new_feature.size(); ++i) {
      auto it = std::upper_bound(unique_values.begin(), unique_values.end(), new_feature[i]);
      int bin_index = std::distance(unique_values.begin(), it) - 1;
      if (bin_index < 0) bin_index = 0;
      if (bin_index >= static_cast<int>(woe_values.size())) bin_index = woe_values.size() - 1;
      woe_feature[i] = woe_values[bin_index];
    }
    return woe_feature;
  }

  Rcpp::DataFrame get_bin_info() {
    return Rcpp::DataFrame::create(
      Rcpp::Named("bin") = bin_intervals,
      Rcpp::Named("woe") = woe_values,
      Rcpp::Named("iv") = iv_values,
      Rcpp::Named("count") = value_counts,
      Rcpp::Named("count_pos") = pos_counts,
      Rcpp::Named("count_neg") = Rcpp::wrap(Rcpp::as<Rcpp::IntegerVector>(Rcpp::wrap(value_counts)) - Rcpp::as<Rcpp::IntegerVector>(Rcpp::wrap(pos_counts)))
    );
  }
};

// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_ivb(Rcpp::IntegerVector target,
                                         Rcpp::NumericVector feature,
                                         int min_bins = 3, int max_bins = 5, int max_n_prebins = 20) {
  OptimalBinningNumericalIVB binner(
      Rcpp::as<std::vector<double>>(feature),
      Rcpp::as<std::vector<int>>(target),
      min_bins, max_bins, max_n_prebins
  );

  binner.fit();

  Rcpp::List bin_info = binner.get_bin_info();
  Rcpp::NumericVector woe_feature = binner.parallel_transform(feature);

  return Rcpp::List::create(
    Rcpp::Named("woefeature") = woe_feature,
    Rcpp::Named("woebin") = bin_info
  );
}

