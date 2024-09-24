#include <Rcpp.h>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <cmath>
#include <omp.h>

class OptimalBinningCategoricalOBNP {
private:
  std::vector<std::string> feature;
  std::vector<int> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;

  struct BinInfo {
    std::vector<std::string> categories;
    int count;
    int count_pos;
    int count_neg;
    double woe;
    double iv;
  };

  std::vector<BinInfo> bins;

  void validate_inputs() {
    if (feature.size() != target.size()) {
      Rcpp::stop("Feature and target must have the same length");
    }
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
  }

  void merge_rare_categories() {
    std::unordered_map<std::string, int> category_counts;
    int total_count = feature.size();

    // Count occurrences of each category
    for (const auto& cat : feature) {
      category_counts[cat]++;
    }

    // Identify rare categories
    std::vector<std::string> rare_categories;
    for (const auto& pair : category_counts) {
      if (static_cast<double>(pair.second) / total_count < bin_cutoff) {
        rare_categories.push_back(pair.first);
      }
    }

    // Merge rare categories
    std::string merged_category = "Other";
    for (auto& cat : feature) {
      if (std::find(rare_categories.begin(), rare_categories.end(), cat) != rare_categories.end()) {
        cat = merged_category;
      }
    }
  }

  void create_initial_bins() {
    std::unordered_map<std::string, BinInfo> bin_map;

#pragma omp parallel for
    for (size_t i = 0; i < feature.size(); ++i) {
      const std::string& cat = feature[i];
      int t = target[i];

#pragma omp critical
{
  if (bin_map.find(cat) == bin_map.end()) {
    bin_map[cat] = BinInfo{{cat}, 1, t, 1 - t, 0.0, 0.0};
  } else {
    bin_map[cat].count++;
    bin_map[cat].count_pos += t;
    bin_map[cat].count_neg += 1 - t;
  }
}
    }

    bins.clear();
    for (const auto& pair : bin_map) {
      bins.push_back(pair.second);
    }

    // Sort bins by count_pos / count ratio (descending)
    std::sort(bins.begin(), bins.end(), [](const BinInfo& a, const BinInfo& b) {
      return static_cast<double>(a.count_pos) / a.count > static_cast<double>(b.count_pos) / b.count;
    });

    // Limit to max_n_prebins
    if (bins.size() > static_cast<size_t>(max_n_prebins)) {
      bins.resize(max_n_prebins);
    }
  }

  void optimize_bins() {
    while (bins.size() > static_cast<size_t>(min_bins) && bins.size() > static_cast<size_t>(max_bins)) {
      merge_least_significant_bins();
    }

    calculate_woe_and_iv();
  }

  void merge_least_significant_bins() {
    auto min_iv_it = std::min_element(bins.begin(), bins.end(),
                                      [](const BinInfo& a, const BinInfo& b) { return a.iv < b.iv; });

    if (min_iv_it != bins.end() && std::next(min_iv_it) != bins.end()) {
      auto next_bin = std::next(min_iv_it);
      min_iv_it->categories.insert(min_iv_it->categories.end(),
                                   next_bin->categories.begin(), next_bin->categories.end());
      min_iv_it->count += next_bin->count;
      min_iv_it->count_pos += next_bin->count_pos;
      min_iv_it->count_neg += next_bin->count_neg;
      bins.erase(next_bin);
    }
  }

  void calculate_woe_and_iv() {
    int total_pos = 0, total_neg = 0;
    for (const auto& bin : bins) {
      total_pos += bin.count_pos;
      total_neg += bin.count_neg;
    }

    double total_iv = 0.0;
    for (auto& bin : bins) {
      double pos_rate = static_cast<double>(bin.count_pos) / total_pos;
      double neg_rate = static_cast<double>(bin.count_neg) / total_neg;
      bin.woe = std::log(pos_rate / neg_rate);
      bin.iv = (pos_rate - neg_rate) * bin.woe;
      total_iv += bin.iv;
    }
  }

public:
  OptimalBinningCategoricalOBNP(const std::vector<std::string>& feature,
                                const std::vector<int>& target,
                                int min_bins = 3,
                                int max_bins = 5,
                                double bin_cutoff = 0.05,
                                int max_n_prebins = 20)
    : feature(feature), target(target), min_bins(min_bins), max_bins(max_bins),
      bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins) {
    validate_inputs();
  }

  void fit() {
    merge_rare_categories();
    create_initial_bins();
    optimize_bins();
  }

  Rcpp::List get_results() {
    std::vector<std::string> bin_names;
    std::vector<double> woe_values;
    std::vector<double> iv_values;
    std::vector<int> count_values;
    std::vector<int> count_pos_values;
    std::vector<int> count_neg_values;

    for (const auto& bin : bins) {
      std::string bin_name = bin.categories[0];
      for (size_t i = 1; i < bin.categories.size(); ++i) {
        bin_name += "+" + bin.categories[i];
      }
      bin_names.push_back(bin_name);
      woe_values.push_back(bin.woe);
      iv_values.push_back(bin.iv);
      count_values.push_back(bin.count);
      count_pos_values.push_back(bin.count_pos);
      count_neg_values.push_back(bin.count_neg);
    }

    return Rcpp::DataFrame::create(
      Rcpp::Named("bin") = bin_names,
      Rcpp::Named("woe") = woe_values,
      Rcpp::Named("iv") = iv_values,
      Rcpp::Named("count") = count_values,
      Rcpp::Named("count_pos") = count_pos_values,
      Rcpp::Named("count_neg") = count_neg_values
    );
  }

  std::vector<double> transform(const std::vector<std::string>& new_feature) {
    std::vector<double> woe_feature(new_feature.size());

#pragma omp parallel for
    for (size_t i = 0; i < new_feature.size(); ++i) {
      const std::string& cat = new_feature[i];
      auto it = std::find_if(bins.begin(), bins.end(), [&cat](const BinInfo& bin) {
        return std::find(bin.categories.begin(), bin.categories.end(), cat) != bin.categories.end();
      });

      if (it != bins.end()) {
        woe_feature[i] = it->woe;
      } else {
        // Assign the WoE of the last bin (typically for unseen categories)
        woe_feature[i] = bins.back().woe;
      }
    }

    return woe_feature;
  }
};

// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_obnp(Rcpp::IntegerVector target,
                                            Rcpp::CharacterVector feature,
                                            int min_bins = 3,
                                            int max_bins = 5,
                                            double bin_cutoff = 0.05,
                                            int max_n_prebins = 20) {
  std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
  std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);

  OptimalBinningCategoricalOBNP binner(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins);
  binner.fit();
  Rcpp::List woebin = binner.get_results();
  std::vector<double> woefeature = binner.transform(feature_vec);

  return Rcpp::List::create(
    Rcpp::Named("woefeature") = woefeature,
    Rcpp::Named("woebin") = woebin
  );
}
