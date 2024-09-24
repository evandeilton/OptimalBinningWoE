// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::plugins(openmp)]]
#include <Rcpp.h>
#include <vector>
#include <map>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <sstream>
#include <cfloat>

using namespace Rcpp;
using namespace std;

class OptimalBinningCategoricalLDB {
private:
  IntegerVector target;
  CharacterVector feature;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;

  struct CategoryStats {
    string category;
    int count;
    int count_pos;
    int count_neg;
    double event_rate;
    double woe;
    double iv;
  };

  vector<CategoryStats> category_stats;

  void validate_inputs() {
    if (min_bins < 2) {
      stop("min_bins must be at least 2.");
    }
    if (max_bins < min_bins) {
      stop("max_bins must be greater than or equal to min_bins.");
    }
    IntegerVector unique_targets = sort_unique(target);
    if (unique_targets.size() != 2 || !(unique_targets[0] == 0 && unique_targets[1] == 1)) {
      stop("Target must be binary (0 and 1).");
    }
    if (feature.size() != target.size()) {
      stop("Feature and target must have the same length.");
    }
  }

  void compute_category_stats() {
    unordered_map<string, CategoryStats> stats_map;
    int total_pos = 0;
    int total_neg = 0;

    int n = target.size();
    for (int i = 0; i < n; ++i) {
      string cat = as<string>(feature[i]);
      int tar = target[i];
      if (stats_map.find(cat) == stats_map.end()) {
        stats_map[cat] = {cat, 0, 0, 0, 0.0, 0.0, 0.0};
      }
      stats_map[cat].count += 1;
      if (tar == 1) {
        stats_map[cat].count_pos += 1;
        total_pos += 1;
      } else {
        stats_map[cat].count_neg += 1;
        total_neg += 1;
      }
    }

    for (auto& kv : stats_map) {
      double rate = (double)kv.second.count_pos / kv.second.count;
      kv.second.event_rate = rate;

      double dist_pos = (double)kv.second.count_pos / total_pos;
      double dist_neg = (double)kv.second.count_neg / total_neg;

      if (dist_pos == 0) dist_pos = 1e-10;
      if (dist_neg == 0) dist_neg = 1e-10;

      kv.second.woe = log(dist_pos / dist_neg);
      kv.second.iv = (dist_pos - dist_neg) * kv.second.woe;

      category_stats.push_back(kv.second);
    }
  }

  void handle_rare_categories() {
    int total_count = target.size();
    double cutoff_count = bin_cutoff * total_count;

    vector<string> rare_categories;
    for (auto& cat_stat : category_stats) {
      if (cat_stat.count < cutoff_count) {
        rare_categories.push_back(cat_stat.category);
      }
    }

    if (!rare_categories.empty()) {
      for (auto& rare_cat : rare_categories) {
        auto it = find_if(category_stats.begin(), category_stats.end(),
                          [&](const CategoryStats& cs) { return cs.category == rare_cat; });
        if (it != category_stats.end()) {
          double min_diff = DBL_MAX;
          int min_idx = -1;
          for (size_t i = 0; i < category_stats.size(); ++i) {
            if (category_stats[i].category != rare_cat &&
                find(rare_categories.begin(), rare_categories.end(), category_stats[i].category) == rare_categories.end()) {
              double diff = fabs(category_stats[i].woe - it->woe);
              if (diff < min_diff) {
                min_diff = diff;
                min_idx = i;
              }
            }
          }
          if (min_idx != -1) {
            category_stats[min_idx].category += "+" + it->category;
            category_stats[min_idx].count += it->count;
            category_stats[min_idx].count_pos += it->count_pos;
            category_stats[min_idx].count_neg += it->count_neg;
            category_stats.erase(it);
          }
        }
      }

      int total_pos = 0;
      int total_neg = 0;
      for (auto& cs : category_stats) {
        total_pos += cs.count_pos;
        total_neg += cs.count_neg;
      }
      for (auto& cs : category_stats) {
        double dist_pos = (double)cs.count_pos / total_pos;
        double dist_neg = (double)cs.count_neg / total_neg;

        if (dist_pos == 0) dist_pos = 1e-10;
        if (dist_neg == 0) dist_neg = 1e-10;

        cs.woe = log(dist_pos / dist_neg);
        cs.iv = (dist_pos - dist_neg) * cs.woe;
      }
    }
  }

  void limit_prebins() {
    if ((int)category_stats.size() > max_n_prebins) {
      sort(category_stats.begin(), category_stats.end(),
           [](const CategoryStats& a, const CategoryStats& b) {
             return a.woe < b.woe;
           });
      int n_bins = max_n_prebins;
      int n_categories = category_stats.size();
      int bin_size = ceil((double)n_categories / n_bins);
      vector<CategoryStats> new_category_stats;

      int total_pos = 0;
      int total_neg = 0;
      for (auto& cs : category_stats) {
        total_pos += cs.count_pos;
        total_neg += cs.count_neg;
      }

      for (int i = 0; i < n_bins; ++i) {
        int start_idx = i * bin_size;
        int end_idx = min(start_idx + bin_size, n_categories);
        if (start_idx >= end_idx) break;

        CategoryStats cs = category_stats[start_idx];
        for (int j = start_idx + 1; j < end_idx; ++j) {
          cs.category += "+" + category_stats[j].category;
          cs.count += category_stats[j].count;
          cs.count_pos += category_stats[j].count_pos;
          cs.count_neg += category_stats[j].count_neg;
        }
        double dist_pos = (double)cs.count_pos / total_pos;
        double dist_neg = (double)cs.count_neg / total_neg;
        if (dist_pos == 0) dist_pos = 1e-10;
        if (dist_neg == 0) dist_neg = 1e-10;
        cs.woe = log(dist_pos / dist_neg);
        cs.iv = (dist_pos - dist_neg) * cs.woe;

        new_category_stats.push_back(cs);
      }
      category_stats = new_category_stats;
    }
  }

  void merge_bins() {
    sort(category_stats.begin(), category_stats.end(),
         [](const CategoryStats& a, const CategoryStats& b) {
           return a.woe < b.woe;
         });

    while ((int)category_stats.size() > max_bins || !is_monotonic()) {
      double min_iv_loss = DBL_MAX;
      int merge_idx = -1;

      int total_pos = 0;
      int total_neg = 0;
      for (auto& cs : category_stats) {
        total_pos += cs.count_pos;
        total_neg += cs.count_neg;
      }

      for (size_t i = 0; i < category_stats.size() - 1; ++i) {
        CategoryStats merged_bin = category_stats[i];
        merged_bin.category += "+" + category_stats[i + 1].category;
        merged_bin.count += category_stats[i + 1].count;
        merged_bin.count_pos += category_stats[i + 1].count_pos;
        merged_bin.count_neg += category_stats[i + 1].count_neg;

        double dist_pos = (double)merged_bin.count_pos / total_pos;
        double dist_neg = (double)merged_bin.count_neg / total_neg;
        if (dist_pos == 0) dist_pos = 1e-10;
        if (dist_neg == 0) dist_neg = 1e-10;

        merged_bin.woe = log(dist_pos / dist_neg);
        merged_bin.iv = (dist_pos - dist_neg) * merged_bin.woe;

        double iv_loss = category_stats[i].iv + category_stats[i + 1].iv - merged_bin.iv;

        if (iv_loss < min_iv_loss) {
          min_iv_loss = iv_loss;
          merge_idx = i;
        }
      }

      if (merge_idx != -1) {
        CategoryStats merged_bin = category_stats[merge_idx];
        merged_bin.category += "+" + category_stats[merge_idx + 1].category;
        merged_bin.count += category_stats[merge_idx + 1].count;
        merged_bin.count_pos += category_stats[merge_idx + 1].count_pos;
        merged_bin.count_neg += category_stats[merge_idx + 1].count_neg;

        double dist_pos = (double)merged_bin.count_pos / total_pos;
        double dist_neg = (double)merged_bin.count_neg / total_neg;
        if (dist_pos == 0) dist_pos = 1e-10;
        if (dist_neg == 0) dist_neg = 1e-10;
        merged_bin.woe = log(dist_pos / dist_neg);
        merged_bin.iv = (dist_pos - dist_neg) * merged_bin.woe;

        category_stats[merge_idx] = merged_bin;
        category_stats.erase(category_stats.begin() + merge_idx + 1);

        sort(category_stats.begin(), category_stats.end(),
             [](const CategoryStats& a, const CategoryStats& b) {
               return a.woe < b.woe;
             });
      } else {
        break;
      }

      if (is_monotonic() && (int)category_stats.size() <= max_bins) {
        break;
      }

      if ((int)category_stats.size() <= min_bins) {
        break;
      }
    }
  }

  bool is_monotonic() {
    if (category_stats.empty()) return true;

    bool increasing = true;
    bool decreasing = true;
    for (size_t i = 1; i < category_stats.size(); ++i) {
      if (category_stats[i].woe < category_stats[i - 1].woe) {
        increasing = false;
      }
      if (category_stats[i].woe > category_stats[i - 1].woe) {
        decreasing = false;
      }
    }
    return increasing || decreasing;
  }

public:
  OptimalBinningCategoricalLDB(IntegerVector target, CharacterVector feature,
                               int min_bins = 3, int max_bins = 5,
                               double bin_cutoff = 0.05, int max_n_prebins = 20)
    : target(target), feature(feature), min_bins(min_bins), max_bins(max_bins),
      bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins) {}

  List fit() {
    validate_inputs();
    compute_category_stats();
    handle_rare_categories();
    limit_prebins();
    merge_bins();

    unordered_map<string, double> category_woe_map;
    for (auto& cs : category_stats) {
      vector<string> categories = split(cs.category, '+');
      for (auto& cat : categories) {
        category_woe_map[cat] = cs.woe;
      }
    }

    NumericVector woefeature(target.size());
    for (int i = 0; i < target.size(); ++i) {
      string cat = as<string>(feature[i]);
      auto it = category_woe_map.find(cat);
      if (it != category_woe_map.end()) {
        woefeature[i] = it->second;
      } else {
        woefeature[i] = NA_REAL;
      }
    }

    vector<string> bins;
    NumericVector woe_values;
    NumericVector iv_values;
    IntegerVector counts;
    IntegerVector counts_pos;
    IntegerVector counts_neg;

    for (auto& cs : category_stats) {
      bins.push_back(cs.category);
      woe_values.push_back(cs.woe);
      iv_values.push_back(cs.iv);
      counts.push_back(cs.count);
      counts_pos.push_back(cs.count_pos);
      counts_neg.push_back(cs.count_neg);
    }

    DataFrame woebin = DataFrame::create(
      Named("bin") = bins,
      Named("woe") = woe_values,
      Named("iv") = iv_values,
      Named("count") = counts,
      Named("count_pos") = counts_pos,
      Named("count_neg") = counts_neg
    );

    return List::create(
      Named("woefeature") = woefeature,
      Named("woebin") = woebin
    );
  }

private:
  vector<string> split(const string& s, char delimiter) {
    vector<string> tokens;
    string token;
    stringstream tokenStream(s);
    while (getline(tokenStream, token, delimiter)) {
      tokens.push_back(token);
    }
    return tokens;
  }
};
// [[Rcpp::export]]
List optimal_binning_categorical_ldb(IntegerVector target, CharacterVector feature,
                                     int min_bins = 3, int max_bins = 5,
                                     double bin_cutoff = 0.05, int max_n_prebins = 20) {
  OptimalBinningCategoricalLDB obc(target, feature, min_bins, max_bins, bin_cutoff, max_n_prebins);
  return obc.fit();
}
