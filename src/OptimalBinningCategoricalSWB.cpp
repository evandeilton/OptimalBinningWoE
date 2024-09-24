#include <Rcpp.h>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <sstream>

class OptimalBinningCategoricalSWB {
private:
  std::vector<std::string> feature;
  std::vector<int> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;

  struct BinStats {
    std::vector<std::string> categories;
    int count;
    int count_pos;
    int count_neg;
    double woe;
    double iv;
  };

  std::vector<BinStats> bins;
  int total_pos;
  int total_neg;

  double calculate_woe(int pos, int neg) {
    double pos_rate = static_cast<double>(pos) / total_pos;
    double neg_rate = static_cast<double>(neg) / total_neg;
    if (pos_rate == 0 || neg_rate == 0) {
      return 0.0;  // Avoid log(0)
    }
    return std::log(pos_rate / neg_rate);
  }

  double calculate_iv(const std::vector<BinStats>& current_bins) {
    double iv = 0.0;
    for (const auto& bin : current_bins) {
      double pos_rate = static_cast<double>(bin.count_pos) / total_pos;
      double neg_rate = static_cast<double>(bin.count_neg) / total_neg;
      if (pos_rate > 0 && neg_rate > 0) {
        iv += (pos_rate - neg_rate) * std::log(pos_rate / neg_rate);
      }
    }
    return iv;
  }

  bool is_monotonic(const std::vector<BinStats>& current_bins) {
    bool increasing = true;
    bool decreasing = true;

    for (size_t i = 1; i < current_bins.size(); ++i) {
      if (current_bins[i].woe < current_bins[i-1].woe) {
        increasing = false;
      }
      if (current_bins[i].woe > current_bins[i-1].woe) {
        decreasing = false;
      }
    }

    return increasing || decreasing;
  }

  void initialize_bins() {
    std::unordered_map<std::string, BinStats> initial_bins;
    total_pos = 0;
    total_neg = 0;

#pragma omp parallel for reduction(+:total_pos,total_neg)
    for (size_t i = 0; i < feature.size(); ++i) {
      const std::string& cat = feature[i];
      int target_val = target[i];

#pragma omp critical
{
  auto& bin = initial_bins[cat];
  if (std::find(bin.categories.begin(), bin.categories.end(), cat) == bin.categories.end()) {
    bin.categories.push_back(cat);
  }
  bin.count++;
  bin.count_pos += target_val;
  bin.count_neg += 1 - target_val;
}
total_pos += target_val;
total_neg += 1 - target_val;
    }

    bins.reserve(initial_bins.size());
    for (auto& pair : initial_bins) {
      pair.second.woe = calculate_woe(pair.second.count_pos, pair.second.count_neg);
      bins.push_back(std::move(pair.second));
    }

    std::sort(bins.begin(), bins.end(), [](const BinStats& a, const BinStats& b) {
      return a.woe < b.woe;
    });

    while (bins.size() > max_n_prebins) {
      merge_adjacent_bins();
    }
  }

  void merge_adjacent_bins() {
    double min_iv_loss = std::numeric_limits<double>::max();
    size_t merge_index = 0;

    for (size_t i = 0; i < bins.size() - 1; ++i) {
      BinStats merged_bin = bins[i];
      merged_bin.count += bins[i+1].count;
      merged_bin.count_pos += bins[i+1].count_pos;
      merged_bin.count_neg += bins[i+1].count_neg;
      merged_bin.categories.insert(merged_bin.categories.end(),
                                   bins[i+1].categories.begin(),
                                   bins[i+1].categories.end());
      merged_bin.woe = calculate_woe(merged_bin.count_pos, merged_bin.count_neg);

      std::vector<BinStats> temp_bins = bins;
      temp_bins[i] = merged_bin;
      temp_bins.erase(temp_bins.begin() + i + 1);

      double new_iv = calculate_iv(temp_bins);
      double iv_loss = calculate_iv(bins) - new_iv;

      if (iv_loss < min_iv_loss) {
        min_iv_loss = iv_loss;
        merge_index = i;
      }
    }

    bins[merge_index].count += bins[merge_index+1].count;
    bins[merge_index].count_pos += bins[merge_index+1].count_pos;
    bins[merge_index].count_neg += bins[merge_index+1].count_neg;
    bins[merge_index].categories.insert(bins[merge_index].categories.end(),
                                        bins[merge_index+1].categories.begin(),
                                        bins[merge_index+1].categories.end());
    bins[merge_index].woe = calculate_woe(bins[merge_index].count_pos, bins[merge_index].count_neg);
    bins.erase(bins.begin() + merge_index + 1);
  }

  void optimize_bins() {
    while (bins.size() > min_bins) {
      if (is_monotonic(bins) && bins.size() <= max_bins) {
        break;
      }

      merge_adjacent_bins();
    }

    // Calculate final IV for each bin
    double total_iv = calculate_iv(bins);

    // Assign IV to each bin
    for (auto& bin : bins) {
      double pos_rate = static_cast<double>(bin.count_pos) / total_pos;
      double neg_rate = static_cast<double>(bin.count_neg) / total_neg;
      bin.iv = (pos_rate - neg_rate) * bin.woe / total_iv;
    }
  }

  std::string join_categories(const std::vector<std::string>& categories, const std::string& delimiter) {
    std::ostringstream result;
    for (size_t i = 0; i < categories.size(); ++i) {
      if (i > 0) result << delimiter;
      result << categories[i];
    }
    return result.str();
  }

public:
  OptimalBinningCategoricalSWB(const std::vector<std::string>& feature,
                               const std::vector<int>& target,
                               int min_bins = 3,
                               int max_bins = 5,
                               double bin_cutoff = 0.05,
                               int max_n_prebins = 20)
    : feature(feature), target(target), min_bins(min_bins), max_bins(max_bins),
      bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins) {
    if (feature.size() != target.size()) {
      Rcpp::stop("Feature and target vectors must have the same length");
    }
    if (min_bins < 2 || max_bins < min_bins) {
      Rcpp::stop("Invalid bin constraints");
    }
  }

  void fit() {
    initialize_bins();
    optimize_bins();
  }

  Rcpp::List get_results() {
    std::vector<std::string> bin_categories;
    std::vector<double> woes;
    std::vector<double> ivs;
    std::vector<int> counts;
    std::vector<int> counts_pos;
    std::vector<int> counts_neg;
    std::vector<double> woe_feature(feature.size());

    std::unordered_map<std::string, double> category_to_woe;
    for (const auto& bin : bins) {
      std::string bin_name = join_categories(bin.categories, ",");
      bin_categories.push_back(bin_name);
      woes.push_back(bin.woe);
      ivs.push_back(bin.iv);
      counts.push_back(bin.count);
      counts_pos.push_back(bin.count_pos);
      counts_neg.push_back(bin.count_neg);

      for (const auto& category : bin.categories) {
        category_to_woe[category] = bin.woe;
      }
    }

#pragma omp parallel for
    for (size_t i = 0; i < feature.size(); ++i) {
      woe_feature[i] = category_to_woe[feature[i]];
    }

    return Rcpp::List::create(
      Rcpp::Named("woefeature") = woe_feature,
      Rcpp::Named("woebin") = Rcpp::DataFrame::create(
        Rcpp::Named("bin") = bin_categories,
        Rcpp::Named("woe") = woes,
        Rcpp::Named("iv") = ivs,
        Rcpp::Named("count") = counts,
        Rcpp::Named("count_pos") = counts_pos,
        Rcpp::Named("count_neg") = counts_neg
      )
    );
  }
};

// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_swb(Rcpp::IntegerVector target,
                                           Rcpp::StringVector feature,
                                           int min_bins = 3,
                                           int max_bins = 5,
                                           double bin_cutoff = 0.05,
                                           int max_n_prebins = 20) {
  std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
  std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);

  OptimalBinningCategoricalSWB binner(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins);
  binner.fit();
  return binner.get_results();
}
