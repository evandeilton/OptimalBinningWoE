#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#include <omp.h>

class OptimalBinningNumericalEFB {
private:
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  std::vector<double> feature;
  std::vector<int> target;
  std::vector<double> bin_edges;
  std::vector<double> woe_values;
  std::vector<double> iv_values;

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

  void validate_inputs() {
    if (min_bins < 2) {
      Rcpp::stop("min_bins must be at least 2");
    }
    if (max_bins < min_bins) {
      Rcpp::stop("max_bins must be greater than or equal to min_bins");
    }
    if (bin_cutoff <= 0 || bin_cutoff >= 1) {
      Rcpp::stop("bin_cutoff must be between 0 and 1");
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

  std::string double_to_string(double value) {
    if (std::isinf(value)) {
      return value > 0 ? "+Inf" : "-Inf";
    }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6) << value;
    return ss.str();
  }

  void create_prebins() {
    std::vector<std::pair<double, int>> feature_target;
    for (size_t i = 0; i < feature.size(); ++i) {
      feature_target.push_back({feature[i], target[i]});
    }
    std::sort(feature_target.begin(), feature_target.end());

    int n = feature_target.size();
    int observations_per_bin = std::max(1, n / max_n_prebins);

    bin_edges.push_back(-std::numeric_limits<double>::infinity());

    for (int i = 1; i < max_n_prebins; ++i) {
      int index = i * observations_per_bin;
      if (index < n) {
        bin_edges.push_back(feature_target[index].first);
      }
    }

    bin_edges.push_back(std::numeric_limits<double>::infinity());
  }

  void calculate_bin_statistics() {
    bins.clear();

    for (size_t i = 0; i < bin_edges.size() - 1; ++i) {
      Bin bin;
      bin.lower_bound = bin_edges[i];
      bin.upper_bound = bin_edges[i + 1];
      bin.count = 0;
      bin.count_pos = 0;
      bin.count_neg = 0;

      bins.push_back(bin);
    }

#pragma omp parallel for
    for (size_t i = 0; i < feature.size(); ++i) {
      double value = feature[i];
      int target_value = target[i];

      auto it = std::upper_bound(bin_edges.begin(), bin_edges.end(), value) - 1;
      int bin_index = it - bin_edges.begin();

#pragma omp atomic
      bins[bin_index].count++;

      if (target_value == 1) {
#pragma omp atomic
        bins[bin_index].count_pos++;
      } else {
#pragma omp atomic
        bins[bin_index].count_neg++;
      }
    }
  }

  void merge_rare_bins() {
    int total_count = 0;
    for (const auto& bin : bins) {
      total_count += bin.count;
    }

    double cutoff_count = bin_cutoff * total_count;

    std::vector<Bin> merged_bins;
    Bin current_bin = bins[0];

    for (size_t i = 1; i < bins.size(); ++i) {
      if (current_bin.count < cutoff_count) {
        current_bin.upper_bound = bins[i].upper_bound;
        current_bin.count += bins[i].count;
        current_bin.count_pos += bins[i].count_pos;
        current_bin.count_neg += bins[i].count_neg;
      } else {
        merged_bins.push_back(current_bin);
        current_bin = bins[i];
      }
    }

    if (current_bin.count > 0) {
      merged_bins.push_back(current_bin);
    }

    bins = merged_bins;
  }

  void calculate_woe_and_iv() {
    int total_pos = 0;
    int total_neg = 0;

    for (const auto& bin : bins) {
      total_pos += bin.count_pos;
      total_neg += bin.count_neg;
    }

    double total_iv = 0.0;

    for (auto& bin : bins) {
      double pos_rate = static_cast<double>(bin.count_pos) / total_pos;
      double neg_rate = static_cast<double>(bin.count_neg) / total_neg;

      if (pos_rate > 0 && neg_rate > 0) {
        bin.woe = std::log(pos_rate / neg_rate);
        bin.iv = (pos_rate - neg_rate) * bin.woe;
      } else {
        bin.woe = 0.0;
        bin.iv = 0.0;
      }

      total_iv += bin.iv;
    }

    woe_values.clear();
    iv_values.clear();

    for (const auto& bin : bins) {
      woe_values.push_back(bin.woe);
      iv_values.push_back(bin.iv);
    }
  }

  void optimize_bins() {
    while (bins.size() > max_bins) {
      int merge_index = -1;
      double min_iv_loss = std::numeric_limits<double>::max();

      for (size_t i = 0; i < bins.size() - 1; ++i) {
        double iv_before = bins[i].iv + bins[i + 1].iv;

        Bin merged_bin;
        merged_bin.lower_bound = bins[i].lower_bound;
        merged_bin.upper_bound = bins[i + 1].upper_bound;
        merged_bin.count = bins[i].count + bins[i + 1].count;
        merged_bin.count_pos = bins[i].count_pos + bins[i + 1].count_pos;
        merged_bin.count_neg = bins[i].count_neg + bins[i + 1].count_neg;

        int total_pos = 0;
        int total_neg = 0;
        for (const auto& bin : bins) {
          total_pos += bin.count_pos;
          total_neg += bin.count_neg;
        }

        double pos_rate = static_cast<double>(merged_bin.count_pos) / total_pos;
        double neg_rate = static_cast<double>(merged_bin.count_neg) / total_neg;

        if (pos_rate > 0 && neg_rate > 0) {
          merged_bin.woe = std::log(pos_rate / neg_rate);
          merged_bin.iv = (pos_rate - neg_rate) * merged_bin.woe;
        } else {
          merged_bin.woe = 0.0;
          merged_bin.iv = 0.0;
        }

        double iv_after = merged_bin.iv;
        double iv_loss = iv_before - iv_after;

        if (iv_loss < min_iv_loss) {
          min_iv_loss = iv_loss;
          merge_index = i;
        }
      }

      if (merge_index != -1) {
        Bin merged_bin;
        merged_bin.lower_bound = bins[merge_index].lower_bound;
        merged_bin.upper_bound = bins[merge_index + 1].upper_bound;
        merged_bin.count = bins[merge_index].count + bins[merge_index + 1].count;
        merged_bin.count_pos = bins[merge_index].count_pos + bins[merge_index + 1].count_pos;
        merged_bin.count_neg = bins[merge_index].count_neg + bins[merge_index + 1].count_neg;

        int total_pos = 0;
        int total_neg = 0;
        for (const auto& bin : bins) {
          total_pos += bin.count_pos;
          total_neg += bin.count_neg;
        }

        double pos_rate = static_cast<double>(merged_bin.count_pos) / total_pos;
        double neg_rate = static_cast<double>(merged_bin.count_neg) / total_neg;

        if (pos_rate > 0 && neg_rate > 0) {
          merged_bin.woe = std::log(pos_rate / neg_rate);
          merged_bin.iv = (pos_rate - neg_rate) * merged_bin.woe;
        } else {
          merged_bin.woe = 0.0;
          merged_bin.iv = 0.0;
        }

        bins.erase(bins.begin() + merge_index, bins.begin() + merge_index + 2);
        bins.insert(bins.begin() + merge_index, merged_bin);
      } else {
        break;
      }
    }
  }

public:
  OptimalBinningNumericalEFB(int min_bins_, int max_bins_, double bin_cutoff_, int max_n_prebins_)
    : min_bins(min_bins_), max_bins(max_bins_), bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_) {}

  void fit(const std::vector<double>& feature_, const std::vector<int>& target_) {
    feature = feature_;
    target = target_;

    validate_inputs();
    create_prebins();
    calculate_bin_statistics();
    merge_rare_bins();
    optimize_bins();
    calculate_woe_and_iv();
  }

  Rcpp::List get_results() {
    std::vector<std::string> bin_labels;
    std::vector<int> counts;
    std::vector<int> counts_pos;
    std::vector<int> counts_neg;

    for (size_t i = 0; i < bins.size(); ++i) {
      const auto& bin = bins[i];
      std::string label = "(" + double_to_string(bin.lower_bound) + ";" + double_to_string(bin.upper_bound) + "]";
      bin_labels.push_back(label);
      counts.push_back(bin.count);
      counts_pos.push_back(bin.count_pos);
      counts_neg.push_back(bin.count_neg);
    }

    return Rcpp::List::create(
      Rcpp::Named("woefeature") = woe_values,
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

};

// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_efb(Rcpp::IntegerVector target, Rcpp::NumericVector feature,
                                         int min_bins = 3, int max_bins = 5, double bin_cutoff = 0.05,
                                         int max_n_prebins = 20) {
  OptimalBinningNumericalEFB binner(min_bins, max_bins, bin_cutoff, max_n_prebins);

  std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
  std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);

  binner.fit(feature_vec, target_vec);
  return binner.get_results();
}
