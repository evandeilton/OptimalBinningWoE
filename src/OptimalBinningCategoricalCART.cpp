// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::plugins(openmp)]]
#include <Rcpp.h>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>
#include <omp.h>
#include <cmath>
#include <limits>

using namespace Rcpp;

class OptimalBinningCategoricalCART {
private:
  std::vector<std::string> feature;
  std::vector<int> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;

  std::unordered_map<std::string, double> category_counts;
  std::unordered_map<std::string, double> category_pos_counts;
  std::unordered_map<std::string, double> category_neg_counts;

  std::vector<std::vector<std::string>> bins;
  std::vector<double> woe_values;
  std::vector<double> iv_values;

public:
  OptimalBinningCategoricalCART(const std::vector<std::string>& feature,
                                const std::vector<int>& target,
                                int min_bins = 3,
                                int max_bins = 5,
                                double bin_cutoff = 0.05,
                                int max_n_prebins = 20)
    : feature(feature),
      target(target),
      min_bins(min_bins),
      max_bins(max_bins),
      bin_cutoff(bin_cutoff),
      max_n_prebins(max_n_prebins) {}

  void validate_inputs() {
    // Validate feature and target lengths
    if (feature.size() != target.size()) {
      stop("Feature and target vectors must have the same length.");
    }
    // Validate binary target
    for (int t : target) {
      if (t != 0 && t != 1) {
        stop("Target must be binary (0 or 1).");
      }
    }
    // Validate min_bins and max_bins
    if (min_bins < 2) {
      stop("min_bins must be at least 2.");
    }
    if (max_bins < min_bins) {
      stop("max_bins must be greater than or equal to min_bins.");
    }
  }

  void calculate_category_counts() {
    // Initialize counts
    for (size_t i = 0; i < feature.size(); ++i) {
      std::string cat = feature[i];
      int tar = target[i];
      category_counts[cat]++;
      if (tar == 1) {
        category_pos_counts[cat]++;
      } else {
        category_neg_counts[cat]++;
      }
    }
  }

  void handle_rare_categories() {
    // Merge rare categories based on bin_cutoff
    double total_count = feature.size();
    std::unordered_map<std::string, std::string> category_mapping;
    std::string rare_label = "Other";
    for (auto& kv : category_counts) {
      double freq = kv.second / total_count;
      if (freq < bin_cutoff) {
        category_mapping[kv.first] = rare_label;
      } else {
        category_mapping[kv.first] = kv.first;
      }
    }

    // Update feature vector
    for (size_t i = 0; i < feature.size(); ++i) {
      feature[i] = category_mapping[feature[i]];
    }

    // Recalculate counts
    category_counts.clear();
    category_pos_counts.clear();
    category_neg_counts.clear();
    calculate_category_counts();

    // Check if number of unique categories is less than min_bins
    if (category_counts.size() < static_cast<size_t>(min_bins)) {
      stop("Number of categories after merging rare categories is less than min_bins. Consider reducing bin_cutoff.");
    }

    // If pre-bins exceed max_n_prebins, further merge least frequent categories
    if (category_counts.size() > static_cast<size_t>(max_n_prebins)) {
      // Sort categories by frequency
      std::vector<std::pair<std::string, double>> sorted_categories(category_counts.begin(), category_counts.end());
      std::sort(sorted_categories.begin(), sorted_categories.end(),
                [](const std::pair<std::string, double>& a, const std::pair<std::string, double>& b) {
                  return a.second < b.second;
                });

      // Merge least frequent categories into "Other" until size <= max_n_prebins
      size_t current_size = sorted_categories.size();
      size_t target_size = max_n_prebins;
      for (size_t i = 0; i < current_size - target_size; ++i) {
        std::string cat_to_merge = sorted_categories[i].first;
        category_mapping[cat_to_merge] = "Other";
      }

      // Update feature vector again
      for (size_t i = 0; i < feature.size(); ++i) {
        feature[i] = category_mapping[feature[i]];
      }

      // Recalculate counts
      category_counts.clear();
      category_pos_counts.clear();
      category_neg_counts.clear();
      calculate_category_counts();
    }
  }

  void initial_binning() {
    // Create initial bins (each category is its own bin)
    bins.clear();
    for (auto& kv : category_counts) {
      bins.push_back({kv.first});
    }
  }

  double calculate_bin_woe(const std::vector<std::string>& bin_categories) {
    double total_pos = 0.0, total_neg = 0.0;
    double bin_pos = 0.0, bin_neg = 0.0;
    for (auto& kv : category_pos_counts) {
      total_pos += kv.second;
    }
    for (auto& kv : category_neg_counts) {
      total_neg += kv.second;
    }
    for (const std::string& cat : bin_categories) {
      bin_pos += category_pos_counts[cat];
      bin_neg += category_neg_counts[cat];
    }
    double dist_pos = bin_pos / total_pos;
    double dist_neg = bin_neg / total_neg;
    if (dist_pos == 0) dist_pos = 1e-6; // Prevent division by zero
    if (dist_neg == 0) dist_neg = 1e-6;
    return std::log(dist_pos / dist_neg);
  }

  double calculate_bin_iv(const std::vector<std::string>& bin_categories) {
    double total_pos = 0.0, total_neg = 0.0;
    double bin_pos = 0.0, bin_neg = 0.0;
    for (auto& kv : category_pos_counts) {
      total_pos += kv.second;
    }
    for (auto& kv : category_neg_counts) {
      total_neg += kv.second;
    }
    for (const std::string& cat : bin_categories) {
      bin_pos += category_pos_counts[cat];
      bin_neg += category_neg_counts[cat];
    }
    double dist_pos = bin_pos / total_pos;
    double dist_neg = bin_neg / total_neg;
    double woe = std::log(dist_pos / dist_neg);
    return (dist_pos - dist_neg) * woe;
  }

  void merge_bins(int max_allowed_bins) {
    // Merge bins until the number of bins is <= max_allowed_bins
    while (bins.size() > static_cast<size_t>(max_allowed_bins)) {
      // Find the pair of adjacent bins with the least difference in WoE
      double min_diff = std::numeric_limits<double>::max();
      size_t merge_idx = 0;

      // Calculate WoE for each bin
      std::vector<double> current_woe(bins.size(), 0.0);
      for (size_t i = 0; i < bins.size(); ++i) {
        current_woe[i] = calculate_bin_woe(bins[i]);
      }

      for (size_t i = 0; i < bins.size() - 1; ++i) {
        double diff = std::abs(current_woe[i] - current_woe[i + 1]);
        if (diff < min_diff) {
          min_diff = diff;
          merge_idx = i;
        }
      }

      // Merge bins at merge_idx and merge_idx + 1
      bins[merge_idx].insert(bins[merge_idx].end(),
                             bins[merge_idx + 1].begin(),
                             bins[merge_idx + 1].end());
      bins.erase(bins.begin() + merge_idx + 1);
    }
  }

  void calculate_woe_iv() {
    woe_values.clear();
    iv_values.clear();
    for (const auto& bin : bins) {
      double woe = calculate_bin_woe(bin);
      double iv = calculate_bin_iv(bin);
      woe_values.push_back(woe);
      iv_values.push_back(iv);
    }
  }

  void enforce_monotonicity() {
    bool is_monotonic = false;
    while (!is_monotonic) {
      is_monotonic = true;
      calculate_woe_iv();

      // Check for monotonicity (either non-decreasing or non-increasing)
      bool increasing = true, decreasing = true;
      for (size_t i = 1; i < woe_values.size(); ++i) {
        if (woe_values[i] < woe_values[i - 1]) {
          increasing = false;
        }
        if (woe_values[i] > woe_values[i - 1]) {
          decreasing = false;
        }
      }

      if (increasing || decreasing) {
        break; // Monotonicity satisfied
      }

      // If not monotonic, merge the pair with the smallest IV contribution
      double min_iv = std::numeric_limits<double>::max();
      size_t merge_idx = 0;

      for (size_t i = 0; i < bins.size() - 1; ++i) {
        double combined_iv = calculate_bin_iv(bins[i]) + calculate_bin_iv(bins[i + 1]);
        if (combined_iv < min_iv) {
          min_iv = combined_iv;
          merge_idx = i;
        }
      }

      // Merge bins at merge_idx and merge_idx + 1
      bins[merge_idx].insert(bins[merge_idx].end(),
                             bins[merge_idx + 1].begin(),
                             bins[merge_idx + 1].end());
      bins.erase(bins.begin() + merge_idx + 1);

      // Ensure that we do not go below min_bins
      if (bins.size() < static_cast<size_t>(min_bins)) {
        // Cannot enforce monotonicity without violating min_bins
        stop("Cannot enforce monotonicity without violating min_bins constraint.");
      }
      is_monotonic = false; // Continue checking
    }
  }

  List get_result() {
    // Create woefeature vector
    std::vector<double> woefeature(feature.size());
    // Precompute a map from category to WoE
    std::unordered_map<std::string, double> category_to_woe;
    for (size_t j = 0; j < bins.size(); ++j) {
      for (const std::string& cat : bins[j]) {
        category_to_woe[cat] = woe_values[j];
      }
    }
    for (size_t i = 0; i < feature.size(); ++i) {
      woefeature[i] = category_to_woe[feature[i]];
    }

    // Create woebin DataFrame
    std::vector<std::string> bin_names;
    std::vector<double> bin_counts;
    std::vector<double> bin_pos_counts;
    std::vector<double> bin_neg_counts;
    for (size_t j = 0; j < bins.size(); ++j) {
      std::string bin_name = "";
      for (size_t k = 0; k < bins[j].size(); ++k) {
        bin_name += bins[j][k];
        if (k != bins[j].size() - 1) {
          bin_name += "+";
        }
      }
      bin_names.push_back(bin_name);

      double count = 0.0, pos = 0.0, neg = 0.0;
      for (const auto& cat : bins[j]) {
        count += category_counts[cat];
        pos += category_pos_counts[cat];
        neg += category_neg_counts[cat];
      }
      bin_counts.push_back(count);
      bin_pos_counts.push_back(pos);
      bin_neg_counts.push_back(neg);
    }

    // Recalculate WoE and IV after final binning
    calculate_woe_iv();

    DataFrame woebin = DataFrame::create(
      Named("bin") = bin_names,
      Named("woe") = woe_values,
      Named("iv") = iv_values,
      Named("count") = bin_counts,
      Named("count_pos") = bin_pos_counts,
      Named("count_neg") = bin_neg_counts
    );

    // Calculate total IV
    double total_iv = std::accumulate(iv_values.begin(), iv_values.end(), 0.0);

    return List::create(
      Named("woefeature") = woefeature,
      Named("woebin") = woebin,
      Named("total_iv") = total_iv
    );
  }

  List fit() {
    validate_inputs();
    calculate_category_counts();
    handle_rare_categories();
    initial_binning();

    // Merge bins to satisfy max_bins constraint, ensuring at least min_bins
    merge_bins(max_bins);

    calculate_woe_iv();
    enforce_monotonicity();

    return get_result();
  }
};

// [[Rcpp::export]]
List optimal_binning_categorical_cart(IntegerVector target,
                                      CharacterVector feature,
                                      int min_bins = 3,
                                      int max_bins = 5,
                                      double bin_cutoff = 0.05,
                                      int max_n_prebins = 20) {
  std::vector<std::string> feature_std = as<std::vector<std::string>>(feature);
  std::vector<int> target_std = as<std::vector<int>>(target);
  OptimalBinningCategoricalCART obc(feature_std, target_std, min_bins, max_bins, bin_cutoff, max_n_prebins);
  return obc.fit();
}

