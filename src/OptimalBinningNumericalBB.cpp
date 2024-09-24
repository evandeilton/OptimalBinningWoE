// [[Rcpp::plugins(openmp)]]
#include <Rcpp.h>
#include <omp.h>
#include <algorithm>
#include <vector>
#include <numeric>
#include <limits>
#include <cmath>
#include <string>
#include <stdexcept>

using namespace Rcpp;

class OptimalBinningNumericalBB {
private:
  std::vector<double> feature;
  std::vector<int> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  bool is_monotonic;
  
  struct Bin {
    double lower;
    double upper;
    int count_pos;
    int count_neg;
    double woe;
    double iv;
  };
  
  std::vector<Bin> bins;
  
  void validate_inputs() {
    if (feature.size() != target.size()) {
      throw std::invalid_argument("Feature and target vectors must have the same length.");
    }
    if (min_bins < 2) {
      throw std::invalid_argument("min_bins must be at least 2.");
    }
    if (max_bins < min_bins) {
      throw std::invalid_argument("max_bins must be greater than or equal to min_bins.");
    }
    if (bin_cutoff < 0 || bin_cutoff > 1) {
      throw std::invalid_argument("bin_cutoff must be between 0 and 1.");
    }
    if (max_n_prebins < min_bins) {
      throw std::invalid_argument("max_n_prebins must be greater than or equal to min_bins.");
    }
  }
  
  void prebinning() {
    // Divide data into quantile-based pre-bins
    std::vector<double> quantiles;
    int n = feature.size();
    for (int i = 1; i < max_n_prebins; ++i) {
      quantiles.push_back(quantile(feature, i / double(max_n_prebins)));
    }
    quantiles.erase(std::unique(quantiles.begin(), quantiles.end()), quantiles.end());
    
    // Initialize bins
    bins.clear();
    bins.resize(quantiles.size() + 1);
    for (size_t i = 0; i < bins.size(); ++i) {
      if (i == 0) {
        bins[i].lower = -std::numeric_limits<double>::infinity();
        bins[i].upper = quantiles[i];
      } else if (i == bins.size() - 1) {
        bins[i].lower = quantiles[i - 1];
        bins[i].upper = std::numeric_limits<double>::infinity();
      } else {
        bins[i].lower = quantiles[i - 1];
        bins[i].upper = quantiles[i];
      }
      bins[i].count_pos = 0;
      bins[i].count_neg = 0;
    }
    
    // Assign observations to bins
    for (size_t i = 0; i < n; ++i) {
      double val = feature[i];
      int tgt = target[i];
      for (auto& bin : bins) {
        if (val > bin.lower && val <= bin.upper) {
          if (tgt == 1) {
            bin.count_pos++;
          } else {
            bin.count_neg++;
          }
          break;
        }
      }
    }
  }
  
  double quantile(const std::vector<double>& data, double q) {
    std::vector<double> temp = data;
    std::sort(temp.begin(), temp.end());
    size_t idx = std::floor(q * (temp.size() - 1));
    return temp[idx];
  }
  
  void merge_rare_bins() {
    int total_count = std::accumulate(bins.begin(), bins.end(), 0,
                                      [](int sum, const Bin& bin) { return sum + bin.count_pos + bin.count_neg; });
    double cutoff_count = bin_cutoff * total_count;
    
    for (auto it = bins.begin(); it != bins.end(); ) {
      if (it->count_pos + it->count_neg < cutoff_count && bins.size() > static_cast<size_t>(min_bins)) {
        if (it != bins.begin()) {
          auto prev = std::prev(it);
          prev->upper = it->upper;
          prev->count_pos += it->count_pos;
          prev->count_neg += it->count_neg;
          it = bins.erase(it);
        } else if (std::next(it) != bins.end()) {
          auto next = std::next(it);
          next->lower = it->lower;
          next->count_pos += it->count_pos;
          next->count_neg += it->count_neg;
          it = bins.erase(it);
        } else {
          ++it;
        }
      } else {
        ++it;
      }
    }
  }
  
  void compute_woe_iv() {
    int total_pos = std::accumulate(bins.begin(), bins.end(), 0,
                                    [](int sum, const Bin& bin) { return sum + bin.count_pos; });
    int total_neg = std::accumulate(bins.begin(), bins.end(), 0,
                                    [](int sum, const Bin& bin) { return sum + bin.count_neg; });
    
    for (auto& bin : bins) {
      double dist_pos = (bin.count_pos + 0.5) / (total_pos + 1);
      double dist_neg = (bin.count_neg + 0.5) / (total_neg + 1);
      bin.woe = std::log(dist_pos / dist_neg);
      bin.iv = (dist_pos - dist_neg) * bin.woe;
    }
  }
  
  void enforce_monotonicity() {
    bool increasing = std::is_sorted(bins.begin(), bins.end(),
                                     [](const Bin& a, const Bin& b) { return a.woe < b.woe; });
    bool decreasing = std::is_sorted(bins.begin(), bins.end(),
                                     [](const Bin& a, const Bin& b) { return a.woe > b.woe; });
    
    if (!increasing && !decreasing) {
      for (auto it = std::next(bins.begin()); it != bins.end(); ) {
        if ((it->woe < std::prev(it)->woe && decreasing) ||
            (it->woe > std::prev(it)->woe && increasing)) {
          std::prev(it)->upper = it->upper;
          std::prev(it)->count_pos += it->count_pos;
          std::prev(it)->count_neg += it->count_neg;
          it = bins.erase(it);
        } else {
          ++it;
        }
      }
      compute_woe_iv();
    }
  }
  
public:
  OptimalBinningNumericalBB(const std::vector<double>& feature_,
                            const std::vector<int>& target_,
                            int min_bins_ = 2,
                            int max_bins_ = 5,
                            double bin_cutoff_ = 0.05,
                            int max_n_prebins_ = 20,
                            bool is_monotonic_ = true)
    : feature(feature_), target(target_), min_bins(min_bins_), max_bins(max_bins_),
      bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_), is_monotonic(is_monotonic_) {
    validate_inputs();
  }
  
  List fit() {
    prebinning();
    merge_rare_bins();
    compute_woe_iv();
    if (is_monotonic) {
      enforce_monotonicity();
    }
    
    while (bins.size() > static_cast<size_t>(max_bins)) {
      auto min_iv_it = std::min_element(bins.begin(), bins.end(),
                                        [](const Bin& a, const Bin& b) { return a.iv < b.iv; });
      
      if (min_iv_it != bins.begin()) {
        auto prev = std::prev(min_iv_it);
        prev->upper = min_iv_it->upper;
        prev->count_pos += min_iv_it->count_pos;
        prev->count_neg += min_iv_it->count_neg;
        bins.erase(min_iv_it);
      } else {
        auto next = std::next(min_iv_it);
        next->lower = min_iv_it->lower;
        next->count_pos += min_iv_it->count_pos;
        next->count_neg += min_iv_it->count_neg;
        bins.erase(min_iv_it);
      }
      compute_woe_iv();
    }
    
    // Prepare outputs
    NumericVector woefeature(feature.size());
    std::vector<std::string> bin_labels;
    NumericVector woe_values;
    NumericVector iv_values;
    IntegerVector counts;
    IntegerVector counts_pos;
    IntegerVector counts_neg;
    
    // Assign WoE to feature
#pragma omp parallel for
    for (size_t i = 0; i < feature.size(); ++i) {
      double val = feature[i];
      for (const auto& bin : bins) {
        if (val > bin.lower && val <= bin.upper) {
          woefeature[i] = bin.woe;
          break;
        }
      }
    }
    
    // Prepare binning table
    for (const auto& bin : bins) {
      std::string bin_label = "(" + (std::isinf(bin.lower) ? "-Inf" : std::to_string(bin.lower)) +
        ";" + (std::isinf(bin.upper) ? "+Inf" : std::to_string(bin.upper)) + "]";
      bin_labels.push_back(bin_label);
      woe_values.push_back(bin.woe);
      iv_values.push_back(bin.iv);
      counts.push_back(bin.count_pos + bin.count_neg);
      counts_pos.push_back(bin.count_pos);
      counts_neg.push_back(bin.count_neg);
    }
    
    DataFrame woebin = DataFrame::create(
      Named("bin") = bin_labels,
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
};

// [[Rcpp::export]]
List optimal_binning_numerical_bb(IntegerVector target,
                                  NumericVector feature,
                                  int min_bins = 3,
                                  int max_bins = 5,
                                  double bin_cutoff = 0.05,
                                  int max_n_prebins = 20,
                                  bool is_monotonic = true) {
  try {
    OptimalBinningNumericalBB obb(as<std::vector<double>>(feature),
                                  as<std::vector<int>>(target),
                                  min_bins,
                                  max_bins,
                                  bin_cutoff,
                                  max_n_prebins,
                                  is_monotonic);
    return obb.fit();
  } catch (const std::exception& e) {
    stop(e.what());
  }
}