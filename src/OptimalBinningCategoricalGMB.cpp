#include <Rcpp.h>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <cmath>
#include <limits>

class OptimalBinningCategoricalGMB {
private:
  std::vector<std::string> feature;
  std::vector<int> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;

  struct BinInfo {
    std::vector<std::string> categories;
    int count = 0;
    int count_pos = 0;
    int count_neg = 0;
    double woe = 0.0;
    double iv = 0.0;
  };

  std::vector<BinInfo> bins;

  double calculateWOE(int pos, int neg, int total_pos, int total_neg) {
    const double epsilon = 0.5;
    double adjusted_pos = (pos == 0) ? epsilon : static_cast<double>(pos);
    double adjusted_neg = (neg == 0) ? epsilon : static_cast<double>(neg);
    double pos_rate = adjusted_pos / total_pos;
    double neg_rate = adjusted_neg / total_neg;
    return std::log(pos_rate / neg_rate);
  }

  double calculateIV(const std::vector<BinInfo>& bins, int total_pos, int total_neg) {
    double iv = 0.0;
    const double epsilon = 0.5;
    for (const auto& bin : bins) {
      double adjusted_pos = (bin.count_pos == 0) ? epsilon : static_cast<double>(bin.count_pos);
      double adjusted_neg = (bin.count_neg == 0) ? epsilon : static_cast<double>(bin.count_neg);
      double pos_rate = adjusted_pos / total_pos;
      double neg_rate = adjusted_neg / total_neg;
      double woe = std::log(pos_rate / neg_rate);
      iv += (pos_rate - neg_rate) * woe;
    }
    return iv;
  }

  void initializeBins() {
    std::unordered_map<std::string, BinInfo> category_map;

    for (size_t i = 0; i < feature.size(); ++i) {
      auto& bin = category_map[feature[i]];
      if (bin.categories.empty()) {
        bin.categories.push_back(feature[i]);
      }
      bin.count++;
      if (target[i] == 1) {
        bin.count_pos++;
      } else {
        bin.count_neg++;
      }
    }

    bins.clear();
    bins.reserve(category_map.size());
    for (auto& pair : category_map) {
      bins.push_back(std::move(pair.second));
    }

    // Sort bins by positive rate
    std::sort(bins.begin(), bins.end(), [](const BinInfo& a, const BinInfo& b) {
      return (static_cast<double>(a.count_pos) / a.count) < (static_cast<double>(b.count_pos) / b.count);
    });

    // Merge rare categories
    int total_count = 0;
    for (const auto& bin : bins) {
      total_count += bin.count;
    }

    std::vector<BinInfo> merged_bins;
    BinInfo current_bin;
    for (const auto& bin : bins) {
      if (static_cast<double>(bin.count) / total_count < bin_cutoff) {
        current_bin.categories.insert(current_bin.categories.end(), bin.categories.begin(), bin.categories.end());
        current_bin.count += bin.count;
        current_bin.count_pos += bin.count_pos;
        current_bin.count_neg += bin.count_neg;
      } else {
        if (!current_bin.categories.empty()) {
          merged_bins.push_back(std::move(current_bin));
          current_bin = BinInfo();
        }
        merged_bins.push_back(bin);
      }
    }

    if (!current_bin.categories.empty()) {
      merged_bins.push_back(std::move(current_bin));
    }

    bins = std::move(merged_bins);

    // Limit to max_n_prebins
    if (static_cast<int>(bins.size()) > max_n_prebins) {
      bins.resize(max_n_prebins);
    }
  }

  void greedyMerge() {
    int total_pos = 0, total_neg = 0;
    for (const auto& bin : bins) {
      total_pos += bin.count_pos;
      total_neg += bin.count_neg;
    }

    while (static_cast<int>(bins.size()) > min_bins) {
      double best_merge_score = -std::numeric_limits<double>::infinity();
      size_t best_merge_index = 0;

      for (size_t i = 0; i < bins.size() - 1; ++i) {
        BinInfo merged_bin;
        merged_bin.categories = bins[i].categories;
        merged_bin.categories.insert(merged_bin.categories.end(), bins[i + 1].categories.begin(), bins[i + 1].categories.end());
        merged_bin.count = bins[i].count + bins[i + 1].count;
        merged_bin.count_pos = bins[i].count_pos + bins[i + 1].count_pos;
        merged_bin.count_neg = bins[i].count_neg + bins[i + 1].count_neg;

        std::vector<BinInfo> temp_bins = bins;
        temp_bins[i] = merged_bin;
        temp_bins.erase(temp_bins.begin() + i + 1);

        for (auto& bin : temp_bins) {
          bin.woe = calculateWOE(bin.count_pos, bin.count_neg, total_pos, total_neg);
        }
        double merge_score = calculateIV(temp_bins, total_pos, total_neg);

        if (merge_score > best_merge_score) {
          best_merge_score = merge_score;
          best_merge_index = i;
        }
      }

      // Perform the best merge
      bins[best_merge_index].categories.insert(bins[best_merge_index].categories.end(),
                                               bins[best_merge_index + 1].categories.begin(),
                                               bins[best_merge_index + 1].categories.end());
      bins[best_merge_index].count = bins[best_merge_index].count + bins[best_merge_index + 1].count;
      bins[best_merge_index].count_pos = bins[best_merge_index].count_pos + bins[best_merge_index + 1].count_pos;
      bins[best_merge_index].count_neg = bins[best_merge_index].count_neg + bins[best_merge_index + 1].count_neg;
      bins.erase(bins.begin() + best_merge_index + 1);

      if (static_cast<int>(bins.size()) <= max_bins) {
        break;
      }
    }

    // Calculate WOE and IV for final bins
    for (auto& bin : bins) {
      bin.woe = calculateWOE(bin.count_pos, bin.count_neg, total_pos, total_neg);
      double adjusted_pos = (bin.count_pos == 0) ? 0.5 : static_cast<double>(bin.count_pos);
      double adjusted_neg = (bin.count_neg == 0) ? 0.5 : static_cast<double>(bin.count_neg);
      double pos_rate = adjusted_pos / total_pos;
      double neg_rate = adjusted_neg / total_neg;
      bin.iv = (pos_rate - neg_rate) * bin.woe;
    }
  }

public:
  OptimalBinningCategoricalGMB(const std::vector<std::string>& feature,
                               const std::vector<int>& target,
                               int min_bins = 3,
                               int max_bins = 5,
                               double bin_cutoff = 0.05,
                               int max_n_prebins = 20)
    : feature(feature), target(target), min_bins(min_bins), max_bins(max_bins),
      bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins) {

    if (min_bins < 2) {
      Rcpp::stop("min_bins must be at least 2");
    }
    if (max_bins < min_bins) {
      Rcpp::stop("max_bins must be greater than or equal to min_bins");
    }
  }

  Rcpp::List fit() {
    initializeBins();
    greedyMerge();

    // Prepare output
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

    // Create WOE feature
    std::vector<double> woefeature(feature.size());
    for (size_t i = 0; i < feature.size(); ++i) {
      for (const auto& bin : bins) {
        if (std::find(bin.categories.begin(), bin.categories.end(), feature[i]) != bin.categories.end()) {
          woefeature[i] = bin.woe;
          break;
        }
      }
    }

    return Rcpp::List::create(
      Rcpp::Named("woefeature") = woefeature,
      Rcpp::Named("woebin") = Rcpp::DataFrame::create(
        Rcpp::Named("bin") = bin_names,
        Rcpp::Named("woe") = woe_values,
        Rcpp::Named("iv") = iv_values,
        Rcpp::Named("count") = count_values,
        Rcpp::Named("count_pos") = count_pos_values,
        Rcpp::Named("count_neg") = count_neg_values
      )
    );
  }
};

// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_gmb(Rcpp::IntegerVector target,
                                           Rcpp::StringVector feature,
                                           int min_bins = 3,
                                           int max_bins = 5,
                                           double bin_cutoff = 0.05,
                                           int max_n_prebins = 20) {
  std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
  std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);

  OptimalBinningCategoricalGMB binner(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins);
  return binner.fit();
}
