// [[Rcpp::plugins(cpp11)]]
#include <Rcpp.h>
#include <unordered_map>
#include <map>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <omp.h>
using namespace Rcpp;

class OptimalBinningCategoricalIVB {
private:
  std::vector<std::string> feature;
  std::vector<int> target;
  double bin_cutoff;
  int min_bins;
  int max_bins;
  int max_n_prebins;
  
  std::map<std::string, int> category_counts;
  std::map<std::string, int> category_pos_counts;
  std::map<std::string, int> category_neg_counts;
  
  std::vector<std::string> merged_bins;
  std::vector<double> woe_values;
  std::vector<double> iv_values;
  
  void validate_inputs();
  void compute_category_stats();
  void optimize_binning();
  void compute_woe_iv();
  
public:
  OptimalBinningCategoricalIVB(std::vector<std::string> feature,
                               std::vector<int> target,
                               double bin_cutoff,
                               int min_bins,
                               int max_bins,
                               int max_n_prebins);
  
  Rcpp::List fit();
};
OptimalBinningCategoricalIVB::OptimalBinningCategoricalIVB(std::vector<std::string> feature,
                                                           std::vector<int> target,
                                                           double bin_cutoff,
                                                           int min_bins,
                                                           int max_bins,
                                                           int max_n_prebins) {
  this->feature = feature;
  this->target = target;
  this->bin_cutoff = bin_cutoff;
  this->min_bins = min_bins;
  this->max_bins = max_bins;
  this->max_n_prebins = max_n_prebins;
  validate_inputs();
}

void OptimalBinningCategoricalIVB::validate_inputs() {
  if (min_bins < 2) {
    Rcpp::stop("min_bins must be at least 2.");
  }
  if (max_bins < min_bins) {
    Rcpp::stop("max_bins must be greater than or equal to min_bins.");
  }
  if (bin_cutoff < 0 || bin_cutoff > 1) {
    Rcpp::stop("bin_cutoff must be between 0 and 1.");
  }
  if (target.size() != feature.size()) {
    Rcpp::stop("target and feature must have the same length.");
  }
  for (auto& t : target) {
    if (t != 0 && t != 1) {
      Rcpp::stop("target must be binary (0 or 1).");
    }
  }
}

void OptimalBinningCategoricalIVB::compute_category_stats() {
  size_t n = target.size();
  for (size_t i = 0; i < n; ++i) {
    category_counts[feature[i]]++;
    if (target[i] == 1) {
      category_pos_counts[feature[i]]++;
    } else {
      category_neg_counts[feature[i]]++;
    }
  }
}

void OptimalBinningCategoricalIVB::optimize_binning() {
  struct CategoryStats {
    std::string category;
    double event_rate;
    int count;
  };

  std::vector<CategoryStats> stats;
  for (auto& pair : category_counts) {
    double event_rate = static_cast<double>(category_pos_counts[pair.first]) / pair.second;
    stats.push_back({pair.first, event_rate, pair.second});
  }

  // Sort categories by event rate to ensure monotonicity
  std::sort(stats.begin(), stats.end(), [](const CategoryStats& a, const CategoryStats& b) {
    return a.event_rate < b.event_rate;
  });

  // Respect max_n_prebins
  if (static_cast<int>(stats.size()) > max_n_prebins) {
    stats.resize(max_n_prebins);
  }

  // Initialize bins
  size_t n_bins = std::max(static_cast<size_t>(min_bins),
                           std::min(static_cast<size_t>(max_bins), stats.size()));
  size_t bin_size = stats.size() / n_bins;
  size_t remainder = stats.size() % n_bins;

  merged_bins.clear();
  size_t idx = 0;
  for (size_t i = 0; i < n_bins; ++i) {
    size_t current_bin_size = bin_size + (i < remainder ? 1 : 0);
    std::vector<std::string> bin_categories;
    for (size_t j = 0; j < current_bin_size; ++j) {
      bin_categories.push_back(stats[idx++].category);
    }
    std::string bin_name = bin_categories[0];
    for (size_t k = 1; k < bin_categories.size(); ++k) {
      bin_name += "+" + bin_categories[k];
    }
    merged_bins.push_back(bin_name);
  }
}

void OptimalBinningCategoricalIVB::compute_woe_iv() {
  int total_pos = std::accumulate(target.begin(), target.end(), 0);
  int total_neg = target.size() - total_pos;

  woe_values.clear();
  iv_values.clear();

  for (auto& bin : merged_bins) {
    std::vector<std::string> categories;
    std::stringstream ss(bin);
    std::string item;
    while (std::getline(ss, item, '+')) {
      categories.push_back(item);
    }

    int bin_pos = 0, bin_neg = 0;
    for (auto& cat : categories) {
      bin_pos += category_pos_counts[cat];
      bin_neg += category_neg_counts[cat];
    }

    double dist_pos = static_cast<double>(bin_pos) / total_pos;
    double dist_neg = static_cast<double>(bin_neg) / total_neg;

    // Avoid division by zero
    if (dist_pos == 0) dist_pos = 1e-6;
    if (dist_neg == 0) dist_neg = 1e-6;

    double woe = std::log(dist_pos / dist_neg);
    double iv = (dist_pos - dist_neg) * woe;

    woe_values.push_back(woe);
    iv_values.push_back(iv);
  }
}

Rcpp::List OptimalBinningCategoricalIVB::fit() {
  compute_category_stats();
  optimize_binning();
  compute_woe_iv();

  // Create woefeature vector
  std::vector<double> woefeature(target.size());
  std::unordered_map<std::string, double> category_to_woe;

  for (size_t i = 0; i < merged_bins.size(); ++i) {
    std::vector<std::string> categories;
    std::stringstream ss(merged_bins[i]);
    std::string item;
    while (std::getline(ss, item, '+')) {
      category_to_woe[item] = woe_values[i];
    }
  }

#pragma omp parallel for
  for (size_t i = 0; i < feature.size(); ++i) {
    if (category_to_woe.find(feature[i]) != category_to_woe.end()) {
      woefeature[i] = category_to_woe[feature[i]];
    } else {
      // For unseen categories, use the WOE of the bin with the closest event rate
      double event_rate = static_cast<double>(category_pos_counts[feature[i]]) / category_counts[feature[i]];
      auto it = std::lower_bound(merged_bins.begin(), merged_bins.end(), event_rate,
                                 [this](const std::string& bin, double rate) {
                                   std::vector<std::string> categories;
                                   std::stringstream ss(bin);
                                   std::string item;
                                   while (std::getline(ss, item, '+')) {
                                     categories.push_back(item);
                                   }
                                   double bin_pos = 0, bin_total = 0;
                                   for (auto& cat : categories) {
                                     bin_pos += category_pos_counts[cat];
                                     bin_total += category_counts[cat];
                                   }
                                   return (static_cast<double>(bin_pos) / bin_total) < rate;
                                 });
      size_t index = std::distance(merged_bins.begin(), it);
      if (index == merged_bins.size()) index--;
      woefeature[i] = woe_values[index];
    }
  }

  // Create woebin DataFrame
  Rcpp::DataFrame woebin = Rcpp::DataFrame::create(
    Rcpp::Named("bin") = merged_bins,
    Rcpp::Named("woe") = woe_values,
    Rcpp::Named("iv") = iv_values
  );

  return Rcpp::List::create(
    Rcpp::Named("woefeature") = woefeature,
    Rcpp::Named("woebin") = woebin
  );
}

// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_ivb(Rcpp::IntegerVector target,
                                           Rcpp::CharacterVector feature,
                                           int min_bins = 3,
                                           int max_bins = 5,
                                           double bin_cutoff = 0.05,
                                           int max_n_prebins = 20) {
  std::vector<std::string> feature_str;
  if (Rf_isFactor(feature)) {
    Rcpp::IntegerVector levels = Rcpp::as<Rcpp::IntegerVector>(feature);
    Rcpp::CharacterVector level_names = levels.attr("levels");
    feature_str.reserve(levels.size());
    for (int i = 0; i < levels.size(); ++i) {
      feature_str.push_back(Rcpp::as<std::string>(level_names[levels[i] - 1]));
    }
  } else if (TYPEOF(feature) == STRSXP) {
    feature_str = Rcpp::as<std::vector<std::string>>(feature);
  } else {
    Rcpp::stop("feature must be a factor or character vector");
  }

  // Convertendo Rcpp::IntegerVector para std::vector<int>
  std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);

  OptimalBinningCategoricalIVB obcivb(feature_str, target_vec, bin_cutoff, min_bins, max_bins, max_n_prebins);
  return obcivb.fit();
}