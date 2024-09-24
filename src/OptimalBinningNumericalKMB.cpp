#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#include <string>
#include <sstream>
#ifdef _OPENMP
#include <omp.h>
#endif

class OptimalBinningNumericalKMB {
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

  double calculateWOE(int pos, int neg, int total_pos, int total_neg) {
    double pos_rate = (pos + 0.5) / (total_pos + 1.0);
    double neg_rate = (neg + 0.5) / (total_neg + 1.0);
    return std::log(pos_rate / neg_rate);
  }

  double calculateIV(double woe, int pos, int neg, int total_pos, int total_neg) {
    double pos_rate = static_cast<double>(pos) / total_pos;
    double neg_rate = static_cast<double>(neg) / total_neg;
    return (pos_rate - neg_rate) * woe;
  }

  void initialBinning() {
    std::vector<double> unique_values = feature;
    std::sort(unique_values.begin(), unique_values.end());
    unique_values.erase(std::unique(unique_values.begin(), unique_values.end()), unique_values.end());

    int n_bins = std::min(static_cast<int>(unique_values.size()), max_n_prebins);
    n_bins = std::max(n_bins, min_bins);
    n_bins = std::min(n_bins, max_bins);

    std::vector<double> boundaries;
    for (int i = 0; i <= n_bins; ++i) {
      int index = i * (unique_values.size() - 1) / n_bins;
      boundaries.push_back(unique_values[index]);
    }

    bins.clear();
    for (size_t i = 0; i < boundaries.size() - 1; ++i) {
      bins.push_back({boundaries[i], boundaries[i + 1], 0, 0, 0, 0.0, 0.0});
    }
    bins.front().lower_bound = std::numeric_limits<double>::lowest();
    bins.back().upper_bound = std::numeric_limits<double>::max();
  }

  void assignDataToBins() {
    for (size_t i = 0; i < feature.size(); ++i) {
      auto it = std::lower_bound(bins.begin(), bins.end(), feature[i],
                                 [](const Bin& bin, double value) { return bin.upper_bound <= value; });

      if (it != bins.end()) {
        it->count++;
        if (target[i] == 1) {
          it->count_pos++;
        } else {
          it->count_neg++;
        }
      }
    }
  }

  void mergeLowFrequencyBins() {
    int total_count = feature.size();
    std::vector<Bin> merged_bins;

    for (const auto& bin : bins) {
      if (static_cast<double>(bin.count) / total_count >= bin_cutoff) {
        merged_bins.push_back(bin);
      } else if (!merged_bins.empty()) {
        auto& last_bin = merged_bins.back();
        last_bin.upper_bound = bin.upper_bound;
        last_bin.count += bin.count;
        last_bin.count_pos += bin.count_pos;
        last_bin.count_neg += bin.count_neg;
      } else {
        merged_bins.push_back(bin);
      }
    }

    bins = merged_bins;
  }

  void adjustBinCount() {
    while (static_cast<int>(bins.size()) > max_bins) {
      auto min_iv_diff_it = std::min_element(bins.begin(), bins.end() - 1,
                                             [this](const Bin& a, const Bin& b) {
                                               size_t index_a = &a - &bins[0];
                                               size_t index_b = &b - &bins[0];
                                               return std::abs(a.iv - bins[index_a + 1].iv) < std::abs(b.iv - bins[index_b + 1].iv);
                                             });

      auto next_bin = std::next(min_iv_diff_it);
      min_iv_diff_it->upper_bound = next_bin->upper_bound;
      min_iv_diff_it->count += next_bin->count;
      min_iv_diff_it->count_pos += next_bin->count_pos;
      min_iv_diff_it->count_neg += next_bin->count_neg;
      bins.erase(next_bin);
    }

    while (static_cast<int>(bins.size()) < min_bins) {
      auto max_range_it = std::max_element(bins.begin(), bins.end(),
                                           [](const Bin& a, const Bin& b) {
                                             return (a.upper_bound - a.lower_bound) < (b.upper_bound - b.lower_bound);
                                           });

      double mid = (max_range_it->lower_bound + max_range_it->upper_bound) / 2;
      Bin new_bin = {mid, max_range_it->upper_bound, 0, 0, 0, 0.0, 0.0};
      max_range_it->upper_bound = mid;

      // Redistribute counts
      for (size_t i = 0; i < feature.size(); ++i) {
        if (feature[i] > mid && feature[i] <= new_bin.upper_bound) {
          new_bin.count++;
          max_range_it->count--;
          if (target[i] == 1) {
            new_bin.count_pos++;
            max_range_it->count_pos--;
          } else {
            new_bin.count_neg++;
            max_range_it->count_neg--;
          }
        }
      }

      bins.insert(max_range_it + 1, new_bin);
    }
  }

  void calculateBinStatistics() {
    int total_pos = 0, total_neg = 0;

    for (const auto& bin : bins) {
      total_pos += bin.count_pos;
      total_neg += bin.count_neg;
    }

    for (auto& bin : bins) {
      bin.woe = calculateWOE(bin.count_pos, bin.count_neg, total_pos, total_neg);
      bin.iv = calculateIV(bin.woe, bin.count_pos, bin.count_neg, total_pos, total_neg);
    }
  }

  std::string formatBinInterval(double lower, double upper) {
    std::ostringstream oss;
    if (lower == std::numeric_limits<double>::lowest()) {
      oss << "(-Inf;";
    } else {
      oss << "(" << lower << ";";
    }

    if (upper == std::numeric_limits<double>::max()) {
      oss << "+Inf]";
    } else {
      oss << upper << "]";
    }
    return oss.str();
  }

public:
  OptimalBinningNumericalKMB(const std::vector<double>& feature, const std::vector<int>& target,
                             int min_bins, int max_bins, double bin_cutoff, int max_n_prebins)
    : feature(feature), target(target), min_bins(min_bins), max_bins(max_bins),
      bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins) {}

  Rcpp::List fit() {
    if (feature.size() != target.size()) {
      Rcpp::stop("Feature and target vectors must have the same length");
    }

    if (min_bins < 2 || max_bins < min_bins) {
      Rcpp::stop("Invalid bin constraints");
    }

    initialBinning();
    assignDataToBins();
    mergeLowFrequencyBins();
    adjustBinCount();
    calculateBinStatistics();

    // Prepare output
    std::vector<double> woefeature(feature.size());
    Rcpp::NumericVector bin_vec(bins.size());
    Rcpp::NumericVector woe_vec(bins.size());
    Rcpp::NumericVector iv_vec(bins.size());
    Rcpp::IntegerVector count_vec(bins.size());
    Rcpp::IntegerVector count_pos_vec(bins.size());
    Rcpp::IntegerVector count_neg_vec(bins.size());
    Rcpp::StringVector bin_labels(bins.size());

    for (size_t i = 0; i < bins.size(); ++i) {
      const auto& bin = bins[i];
      bin_vec[i] = i + 1;
      woe_vec[i] = bin.woe;
      iv_vec[i] = bin.iv;
      count_vec[i] = bin.count;
      count_pos_vec[i] = bin.count_pos;
      count_neg_vec[i] = bin.count_neg;
      bin_labels[i] = formatBinInterval(bin.lower_bound, bin.upper_bound);

      for (size_t j = 0; j < feature.size(); ++j) {
        if (feature[j] > bin.lower_bound && feature[j] <= bin.upper_bound) {
          woefeature[j] = bin.woe;
        }
      }
    }

    Rcpp::DataFrame woebin = Rcpp::DataFrame::create(
      Rcpp::Named("bin") = bin_labels,
      Rcpp::Named("woe") = woe_vec,
      Rcpp::Named("iv") = iv_vec,
      Rcpp::Named("count") = count_vec,
      Rcpp::Named("count_pos") = count_pos_vec,
      Rcpp::Named("count_neg") = count_neg_vec
    );

    return Rcpp::List::create(
      Rcpp::Named("woefeature") = woefeature,
      Rcpp::Named("woebin") = woebin
    );
  }
};

// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_kmb(Rcpp::IntegerVector target,
                                         Rcpp::NumericVector feature,
                                         int min_bins = 3, int max_bins = 5,
                                         double bin_cutoff = 0.05, int max_n_prebins = 20) {
  std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
  std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);

  OptimalBinningNumericalKMB binner(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins);
  return binner.fit();
}
