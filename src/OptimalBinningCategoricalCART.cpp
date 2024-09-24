// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::plugins(openmp)]]
#include <Rcpp.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <numeric>

using namespace Rcpp;
using namespace std;

// Helper function to calculate WOE and IV
void calculate_woe_iv(const std::vector<double>& count_pos, const std::vector<double>& count_neg,
                      std::vector<double>& woe, std::vector<double>& iv) {
  double total_pos = std::accumulate(count_pos.begin(), count_pos.end(), 0.0);
  double total_neg = std::accumulate(count_neg.begin(), count_neg.end(), 0.0);
  
  for (size_t i = 0; i < count_pos.size(); ++i) {
    double dist_pos = count_pos[i] / total_pos;
    double dist_neg = count_neg[i] / total_neg;
    if (dist_pos == 0) dist_pos = 1e-10;
    if (dist_neg == 0) dist_neg = 1e-10;
    woe[i] = std::log(dist_pos / dist_neg);
    iv[i] = (dist_pos - dist_neg) * woe[i];
  }
}

// Main class for Optimal Binning
class OptimalBinningCategoricalOSLP {
private:
  std::vector<int> target;
  std::vector<std::string> feature;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  bool monotonic;
  
  std::map<std::string, int> category_counts;
  std::map<std::string, double> category_pos_counts;
  std::map<std::string, double> category_neg_counts;
  
  std::vector<std::vector<std::string>> bins;
  std::vector<double> woe_values;
  std::vector<double> iv_values;
  
public:
  OptimalBinningCategoricalOSLP(const Rcpp::IntegerVector& target,
                                const Rcpp::CharacterVector& feature,
                                int min_bins = 3,
                                int max_bins = 5,
                                double bin_cutoff = 0.05,
                                int max_n_prebins = 20,
                                bool monotonic = true)
    : target(Rcpp::as<std::vector<int>>(target)), 
      feature(Rcpp::as<std::vector<std::string>>(feature)), 
      min_bins(min_bins), max_bins(max_bins), bin_cutoff(bin_cutoff),
      max_n_prebins(max_n_prebins), monotonic(monotonic) {
    // Validate target
    for (int t : this->target) {
      if (t != 0 && t != 1) {
        Rcpp::stop("Target must contain only 0 and 1");
      }
    }
  }
  
  void fit() {
    // Input validation
    int unique_categories = std::set<std::string>(feature.begin(), feature.end()).size();
    min_bins = std::min(min_bins, unique_categories);
    if (min_bins < 2) min_bins = 2;
    max_bins = std::max(max_bins, min_bins);
    
    // Step 1: Calculate category counts and target distribution
    calculate_category_stats();
    
    // Step 2: Create initial bins
    create_initial_bins();
    
    // Step 3: Optimize bins
    optimize_bins();
    
    // Step 4: Calculate WOE and IV for the bins
    calculate_bin_woe_iv();
  }
  
  Rcpp::List transform() {
    // Map each category to its WOE value
    std::unordered_map<std::string, double> category_to_woe;
    for (size_t i = 0; i < bins.size(); ++i) {
      for (const auto& cat : bins[i]) {
        category_to_woe[cat] = woe_values[i];
      }
    }
    
    // Apply WOE values to the feature
    Rcpp::NumericVector woefeature(feature.size());
#pragma omp parallel for
    for (int i = 0; i < feature.size(); ++i) {
      std::string cat = feature[i];
      if (category_to_woe.find(cat) != category_to_woe.end()) {
        woefeature[i] = category_to_woe[cat];
      } else {
        woefeature[i] = 0.0; // Handle unseen categories
      }
    }
    
    // Prepare the output DataFrame
    Rcpp::CharacterVector bin_names(bins.size());
    Rcpp::NumericVector woe_output(bins.size());
    Rcpp::NumericVector iv_output(bins.size());
    Rcpp::NumericVector count_output(bins.size());
    Rcpp::NumericVector count_pos_output(bins.size());
    Rcpp::NumericVector count_neg_output(bins.size());
    
    for (size_t i = 0; i < bins.size(); ++i) {
      bin_names[i] = join_bins(bins[i]);
      woe_output[i] = woe_values[i];
      iv_output[i] = iv_values[i];
      double count = 0.0, count_pos = 0.0, count_neg = 0.0;
      for (const auto& cat : bins[i]) {
        count += category_counts[cat];
        count_pos += category_pos_counts[cat];
        count_neg += category_neg_counts[cat];
      }
      count_output[i] = count;
      count_pos_output[i] = count_pos;
      count_neg_output[i] = count_neg;
    }
    
    Rcpp::DataFrame woebin = Rcpp::DataFrame::create(
      Rcpp::Named("bin") = bin_names,
      Rcpp::Named("woe") = woe_output,
      Rcpp::Named("iv") = iv_output,
      Rcpp::Named("count") = count_output,
      Rcpp::Named("count_pos") = count_pos_output,
      Rcpp::Named("count_neg") = count_neg_output
    );
    
    return Rcpp::List::create(Rcpp::Named("woefeature") = woefeature,
                              Rcpp::Named("woebin") = woebin);
  }
  
private:
  void calculate_category_stats() {
    for (size_t i = 0; i < feature.size(); ++i) {
      std::string cat = feature[i];
      int tgt = target[i];
      category_counts[cat]++;
      if (tgt == 1) {
        category_pos_counts[cat]++;
      } else {
        category_neg_counts[cat]++;
      }
    }
  }
  
  void create_initial_bins() {
    // Sort categories based on WOE
    std::vector<std::pair<std::string, double>> cat_woe_list;
    for (const auto& pair : category_counts) {
      double pos_count = category_pos_counts[pair.first];
      double neg_count = category_neg_counts[pair.first];
      if (pos_count == 0) pos_count = 1e-10;
      if (neg_count == 0) neg_count = 1e-10;
      double total_pos = std::accumulate(target.begin(), target.end(), 0.0);
      double woe = std::log((pos_count / total_pos) / (neg_count / (target.size() - total_pos)));
      cat_woe_list.push_back(std::make_pair(pair.first, woe));
    }
    std::sort(cat_woe_list.begin(), cat_woe_list.end(),
              [](const std::pair<std::string, double>& a, const std::pair<std::string, double>& b) {
                return a.second < b.second;
              });
    
    // Create initial bins
    int n_categories = cat_woe_list.size();
    int n_bins = std::min(std::max(min_bins, std::min(max_bins, n_categories)), max_n_prebins);
    int cats_per_bin = std::max(1, n_categories / n_bins);
    
    for (int i = 0; i < n_categories; i += cats_per_bin) {
      std::vector<std::string> bin;
      for (int j = i; j < std::min(i + cats_per_bin, n_categories); ++j) {
        bin.push_back(cat_woe_list[j].first);
      }
      bins.push_back(bin);
    }
    
    // Merge small bins if necessary
    while (bins.size() > max_bins) {
      merge_bins();
    }
  }
  
  void optimize_bins() {
    bool changed;
    do {
      changed = false;
      for (size_t i = 0; i < bins.size() - 1; ++i) {
        double woe1 = calculate_bin_woe(bins[i]);
        double woe2 = calculate_bin_woe(bins[i + 1]);
        if (std::abs(woe1 - woe2) < bin_cutoff) {
          bins[i].insert(bins[i].end(), bins[i + 1].begin(), bins[i + 1].end());
          bins.erase(bins.begin() + i + 1);
          changed = true;
          break;
        }
      }
    } while (changed && bins.size() > min_bins);
    
    // Ensure min_bins constraint is met
    while (bins.size() < min_bins && bins.size() > 1) {
      split_largest_bin();
    }
    
    // Enforce monotonicity if required
    if (monotonic) {
      enforce_monotonicity();
    }
  }
  
  void merge_bins() {
    // Merge bins with the smallest difference in WOE
    double min_diff = std::numeric_limits<double>::max();
    size_t idx_to_merge = 0;
    for (size_t i = 0; i < bins.size() - 1; ++i) {
      double woe1 = calculate_bin_woe(bins[i]);
      double woe2 = calculate_bin_woe(bins[i + 1]);
      double diff = std::abs(woe1 - woe2);
      if (diff < min_diff) {
        min_diff = diff;
        idx_to_merge = i;
      }
    }
    // Merge bins[idx_to_merge] and bins[idx_to_merge + 1]
    bins[idx_to_merge].insert(bins[idx_to_merge].end(),
                              bins[idx_to_merge + 1].begin(),
                              bins[idx_to_merge + 1].end());
    bins.erase(bins.begin() + idx_to_merge + 1);
  }
  
  void split_largest_bin() {
    // Find the largest bin
    size_t largest_bin_idx = 0;
    size_t largest_bin_size = 0;
    for (size_t i = 0; i < bins.size(); ++i) {
      if (bins[i].size() > largest_bin_size) {
        largest_bin_size = bins[i].size();
        largest_bin_idx = i;
      }
    }
    
    // Split the largest bin
    if (largest_bin_size > 1) {
      size_t split_point = largest_bin_size / 2;
      std::vector<std::string> new_bin(bins[largest_bin_idx].begin() + split_point, bins[largest_bin_idx].end());
      bins[largest_bin_idx].erase(bins[largest_bin_idx].begin() + split_point, bins[largest_bin_idx].end());
      bins.insert(bins.begin() + largest_bin_idx + 1, new_bin);
    }
  }
  
  void enforce_monotonicity() {
    // Sort bins based on their WOE values
    std::vector<std::pair<size_t, double>> bin_woes;
    for (size_t i = 0; i < bins.size(); ++i) {
      bin_woes.push_back(std::make_pair(i, calculate_bin_woe(bins[i])));
    }
    std::sort(bin_woes.begin(), bin_woes.end(),
              [](const std::pair<size_t, double>& a, const std::pair<size_t, double>& b) {
                return a.second < b.second;
              });
    
    // Reorder bins to ensure monotonicity
    std::vector<std::vector<std::string>> new_bins;
    for (const auto& pair : bin_woes) {
      new_bins.push_back(bins[pair.first]);
    }
    bins = new_bins;
  }
  
  double calculate_bin_woe(const std::vector<std::string>& bin) {
    double total_pos = 0.0, total_neg = 0.0;
    for (const auto& cat : bin) {
      total_pos += category_pos_counts[cat];
      total_neg += category_neg_counts[cat];
    }
    if (total_pos == 0) total_pos = 1e-10;
    if (total_neg == 0) total_neg = 1e-10;
    double sum_target = std::accumulate(target.begin(), target.end(), 0.0);
    return std::log((total_pos / sum_target) / (total_neg / (target.size() - sum_target)));
  }
  
  void calculate_bin_woe_iv() {
    size_t n_bins = bins.size();
    woe_values.resize(n_bins);
    iv_values.resize(n_bins);
    std::vector<double> count_pos(n_bins, 0.0);
    std::vector<double> count_neg(n_bins, 0.0);
    
    double total_pos = std::accumulate(target.begin(), target.end(), 0.0);
    double total_neg = target.size() - total_pos;
    
    for (size_t i = 0; i < n_bins; ++i) {
      double bin_pos = 0.0, bin_neg = 0.0;
      for (const auto& cat : bins[i]) {
        bin_pos += category_pos_counts[cat];
        bin_neg += category_neg_counts[cat];
      }
      count_pos[i] = bin_pos;
      count_neg[i] = bin_neg;
    }
    
    calculate_woe_iv(count_pos, count_neg, woe_values, iv_values);
  }
  
  std::string join_bins(const std::vector<std::string>& bin) {
    std::string bin_name = "";
    for (size_t i = 0; i < bin.size(); ++i) {
      bin_name += bin[i];
      if (i != bin.size() - 1) bin_name += "+";
    }
    return bin_name;
  }
};

// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_oslp(Rcpp::IntegerVector target,
                                            Rcpp::CharacterVector feature,
                                            int min_bins = 3,
                                            int max_bins = 5,
                                            double bin_cutoff = 0.05,
                                            int max_n_prebins = 20,
                                            bool monotonic = true) {
  // Validate target
  for (int i = 0; i < target.size(); ++i) {
    if (target[i] != 0 && target[i] != 1) {
      Rcpp::stop("Target must contain only 0 and 1");
    }
  }
  
  OptimalBinningCategoricalOSLP obc(target, feature, min_bins, max_bins,
                                    bin_cutoff, max_n_prebins, monotonic);
  obc.fit();
  return obc.transform();
}
