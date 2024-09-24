#include <Rcpp.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <omp.h>

// [[Rcpp::plugins(openmp)]]

using namespace Rcpp;

// Helper function to compute WOE and IV
inline void compute_woe_iv(double count_pos, double count_neg, double total_pos, double total_neg,
                           double &woe, double &iv) {
  double dist_pos = count_pos / total_pos;
  double dist_neg = count_neg / total_neg;
  // Avoid division by zero and log of zero
  if (dist_pos < 1e-10) dist_pos = 1e-10;
  if (dist_neg < 1e-10) dist_neg = 1e-10;
  woe = log(dist_pos / dist_neg);
  iv = (dist_pos - dist_neg) * woe;
}

// Main class for Optimal Binning with Dynamic Programming and Linear Constraints
class OptimalBinningCategoricalDPLC {
public:
  OptimalBinningCategoricalDPLC(const std::vector<std::string> &feature,
                                const std::vector<int> &target,
                                int min_bins,
                                int max_bins,
                                double bin_cutoff,
                                int max_n_prebins) :
  feature(feature),
  target(target),
  min_bins(min_bins),
  max_bins(max_bins),
  bin_cutoff(bin_cutoff),
  max_n_prebins(max_n_prebins) {}

  // Function to perform the optimal binning
  List perform_binning() {
    // Step 1: Input validation
    if (min_bins < 2) {
      stop("min_bins must be at least 2.");
    }
    if (max_bins < min_bins) {
      stop("max_bins must be greater than or equal to min_bins.");
    }
    if (feature.size() != target.size()) {
      stop("feature and target must have the same length.");
    }

    // Additional Validation: Ensure Target is Binary
    std::unordered_set<int> unique_targets(target.begin(), target.end());
    if (unique_targets.size() != 2 || unique_targets.find(0) == unique_targets.end() || unique_targets.find(1) == unique_targets.end()) {
      stop("Target must be binary, containing only 0s and 1s.");
    }

    // Step 2: Preprocess data - count occurrences
    std::unordered_map<std::string, int> category_counts;
    std::unordered_map<std::string, int> category_pos_counts;
    for (size_t i = 0; i < feature.size(); ++i) {
      category_counts[feature[i]] += 1;
      if (target[i] == 1) {
        category_pos_counts[feature[i]] += 1;
      }
    }

    // Total counts
    double total_count = feature.size();
    double total_pos = std::accumulate(target.begin(), target.end(), 0.0);
    double total_neg = total_count - total_pos;

    // Step 3: Merge rare categories based on bin_cutoff
    double cutoff_count = bin_cutoff * total_count;
    std::vector<std::pair<std::string, int>> sorted_categories_count(category_counts.begin(), category_counts.end());
    std::sort(sorted_categories_count.begin(), sorted_categories_count.end(),
              [](const std::pair<std::string, int> &a, const std::pair<std::string, int> &b) {
                return a.second < b.second;
              });

    std::vector<std::string> merged_categories;
    std::unordered_map<std::string, std::string> category_mapping;
    std::string current_merged = "";
    int current_count = 0;

    for (const auto &cat : sorted_categories_count) {
      if (cat.second < cutoff_count) {
        if (current_merged.empty()) {
          current_merged = cat.first;
        } else {
          current_merged += "+" + cat.first;
        }
        current_count += cat.second;
      } else {
        if (!current_merged.empty()) {
          merged_categories.push_back(current_merged);
          // Map all merged categories to the merged bin name
          std::vector<std::string> split_cats = split_string(current_merged, '+');
          for (const auto &merged_cat : split_cats) {
            category_mapping[merged_cat] = current_merged;
          }
          current_merged = "";
          current_count = 0;
        }
        merged_categories.push_back(cat.first);
        category_mapping[cat.first] = cat.first;
      }
    }

    if (!current_merged.empty()) {
      merged_categories.push_back(current_merged);
      std::vector<std::string> split_cats = split_string(current_merged, '+');
      for (const auto &merged_cat : split_cats) {
        category_mapping[merged_cat] = current_merged;
      }
    }

    // Step 4: Ensure max_n_prebins is not exceeded
    while (merged_categories.size() > static_cast<size_t>(max_n_prebins)) {
      // Find the two smallest bins
      auto min_it1 = std::min_element(merged_categories.begin(), merged_categories.end(),
                                      [&](const std::string &a, const std::string &b) {
                                        return get_bin_count(a, category_counts) < get_bin_count(b, category_counts);
                                      });

      std::string smallest_bin = *min_it1;
      merged_categories.erase(min_it1);

      auto min_it2 = std::min_element(merged_categories.begin(), merged_categories.end(),
                                      [&](const std::string &a, const std::string &b) {
                                        return get_bin_count(a, category_counts) < get_bin_count(b, category_counts);
                                      });

      // Merge the two smallest bins
      std::string merged_bin = smallest_bin + "+" + *min_it2;
      *min_it2 = merged_bin;

      // Update category_mapping
      std::vector<std::string> split_cats = split_string(merged_bin, '+');
      for (const auto &cat : split_cats) {
        category_mapping[cat] = merged_bin;
      }
    }

    // Step 5: Compute event rates and sort categories
    std::vector<std::string> final_categories = merged_categories;
    std::vector<double> event_rates;
    for (const auto &bin : final_categories) {
      double pos = 0.0;
      double total = 0.0;
      std::vector<std::string> split_cats = split_string(bin, '+');
      for (const auto &cat : split_cats) {
        pos += category_pos_counts[cat];
        total += category_counts[cat];
      }
      event_rates.push_back(pos / total);
    }

    // Sort final_categories based on event rate
    std::vector<size_t> indices(final_categories.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&event_rates](size_t i1, size_t i2) { return event_rates[i1] < event_rates[i2]; });

    std::vector<std::string> sorted_categories;
    std::vector<double> sorted_event_rates;
    for (size_t i : indices) {
      sorted_categories.push_back(final_categories[i]);
      sorted_event_rates.push_back(event_rates[i]);
    }

    // Step 6: Initialize dynamic programming structures
    size_t n = sorted_categories.size();
    std::vector<std::vector<double>> dp(n + 1, std::vector<double>(max_bins + 1, -INFINITY));
    std::vector<std::vector<int>> prev_bin(n + 1, std::vector<int>(max_bins + 1, -1));

    dp[0][0] = 0.0; // Base case

    // Precompute cumulative counts
    std::vector<double> cum_count_pos(n + 1, 0.0);
    std::vector<double> cum_count_neg(n + 1, 0.0);

    for (size_t i = 0; i < n; ++i) {
      double pos = 0.0;
      double neg = 0.0;
      std::vector<std::string> split_cats = split_string(sorted_categories[i], '+');
      for (const auto &cat : split_cats) {
        pos += category_pos_counts[cat];
        neg += category_counts[cat] - category_pos_counts[cat];
      }
      cum_count_pos[i + 1] = cum_count_pos[i] + pos;
      cum_count_neg[i + 1] = cum_count_neg[i] + neg;
    }

    // Step 7: Dynamic Programming to find optimal binning
    for (size_t i = 1; i <= n; ++i) {
      for (int k = 1; k <= max_bins && k <= static_cast<int>(i); ++k) {
        for (size_t j = (k - 1 > 0 ? k - 1 : 0); j < i; ++j) {
          double count_pos_bin = cum_count_pos[i] - cum_count_pos[j];
          double count_neg_bin = cum_count_neg[i] - cum_count_neg[j];

          double woe_bin, iv_bin;
          compute_woe_iv(count_pos_bin, count_neg_bin, total_pos, total_neg, woe_bin, iv_bin);

          double total_iv = dp[j][k - 1] + iv_bin;
          if (total_iv > dp[i][k]) {
            dp[i][k] = total_iv;
            prev_bin[i][k] = j;
          }
        }
      }
    }

    // Step 8: Backtrack to find the optimal bins
    double max_total_iv = -INFINITY;
    int best_k = -1;

    for (int k = min_bins; k <= max_bins; ++k) {
      if (dp[n][k] > max_total_iv) {
        max_total_iv = dp[n][k];
        best_k = k;
      }
    }

    if (best_k == -1) {
      stop("Failed to find optimal binning with given constraints.");
    }

    std::vector<size_t> bin_edges;
    size_t idx = n;
    int k = best_k;
    while (k > 0) {
      int prev_j = prev_bin[idx][k];
      bin_edges.push_back(prev_j);
      idx = prev_j;
      k -= 1;
    }
    std::reverse(bin_edges.begin(), bin_edges.end());

    // Step 9: Prepare output
    std::vector<std::string> bin_names;
    std::vector<double> bin_woe;
    std::vector<double> bin_iv;
    std::vector<int> bin_count;
    std::vector<int> bin_count_pos;
    std::vector<int> bin_count_neg;

    size_t start = 0;
    for (size_t edge_idx = 0; edge_idx <= bin_edges.size(); ++edge_idx) {
      size_t end = (edge_idx < bin_edges.size()) ? bin_edges[edge_idx] : n;

      std::vector<std::string> bin_categories;
      double count_bin = 0.0;
      double count_pos_bin = 0.0;
      std::string bin_name = "";

      for (size_t i = start; i < end; ++i) {
        if (i > start) bin_name += "+";
        bin_name += sorted_categories[i];

        std::vector<std::string> split_cats = split_string(sorted_categories[i], '+');
        for (const auto &cat : split_cats) {
          bin_categories.push_back(cat);
          count_bin += category_counts[cat];
          count_pos_bin += category_pos_counts[cat];
        }
      }

      double count_neg_bin = count_bin - count_pos_bin;

      // Only add the bin if it's not empty
      if (count_bin > 0) {
        double woe_bin, iv_bin_value;
        compute_woe_iv(count_pos_bin, count_neg_bin, total_pos, total_neg, woe_bin, iv_bin_value);

        bin_names.push_back(bin_name);
        bin_woe.push_back(woe_bin);
        bin_iv.push_back(iv_bin_value);
        bin_count.push_back(static_cast<int>(count_bin));
        bin_count_pos.push_back(static_cast<int>(count_pos_bin));
        bin_count_neg.push_back(static_cast<int>(count_neg_bin));

        // Update category_mapping for all categories in this bin
        for (const auto &cat : bin_categories) {
          category_mapping[cat] = bin_name;
        }
      }

      start = end;
    }

    // Step 10: Apply WOE mapping to feature
    std::vector<double> woefeature(feature.size());
#pragma omp parallel for
    for (size_t i = 0; i < feature.size(); ++i) {
      std::string mapped_category = category_mapping[feature[i]];
      auto it = std::find(bin_names.begin(), bin_names.end(), mapped_category);
      if (it != bin_names.end()) {
        size_t bin_index = std::distance(bin_names.begin(), it);
        woefeature[i] = bin_woe[bin_index];
      } else {
        woefeature[i] = 0.0; // Default value if category is not found
      }
    }

    // Prepare woebin DataFrame
    DataFrame woebin = DataFrame::create(
      Named("bin") = bin_names,
      Named("woe") = bin_woe,
      Named("iv") = bin_iv,
      Named("count") = bin_count,
      Named("count_pos") = bin_count_pos,
      Named("count_neg") = bin_count_neg
    );
    // Return results
    return List::create(
      Named("woefeature") = woefeature,
      Named("woebin") = woebin
    );
  }

private:
  const std::vector<std::string> &feature;
  const std::vector<int> &target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;

  // Helper function to split strings by a delimiter
  std::vector<std::string> split_string(const std::string &s, char delimiter) const {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
      tokens.push_back(token);
    }
    return tokens;
  }

  // Helper function to get the total count for a bin
  int get_bin_count(const std::string &bin, const std::unordered_map<std::string, int> &category_counts) const {
    int count = 0;
    std::vector<std::string> categories = split_string(bin, '+');
    for (const auto &cat : categories) {
      auto it = category_counts.find(cat);
      if (it != category_counts.end()) {
        count += it->second;
      }
    }
    return count;
  }
};

// [[Rcpp::export]]
List optimal_binning_categorical_dplc(IntegerVector target,
                                      CharacterVector feature,
                                      int min_bins = 3,
                                      int max_bins = 5,
                                      double bin_cutoff = 0.05,
                                      int max_n_prebins = 20) {
  std::vector<std::string> feature_vec = as<std::vector<std::string>>(feature);
  std::vector<int> target_vec = as<std::vector<int>>(target);

  OptimalBinningCategoricalDPLC binning(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins);
  return binning.perform_binning();
}
