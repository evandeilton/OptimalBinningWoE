// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::plugins(openmp)]]

#include <Rcpp.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>

using namespace Rcpp;

class OptimalBinningNumericalDPLC {
private:
  std::vector<double> feature;
  std::vector<unsigned int> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  
  std::vector<double> bin_edges;
  std::vector<double> woe_values;
  std::vector<double> iv_values;
  std::vector<std::string> bin_labels;
  std::vector<double> count_pos;
  std::vector<double> count_neg;
  std::vector<double> counts;
  
  double total_pos;
  double total_neg;
  
public:
  OptimalBinningNumericalDPLC(const std::vector<double>& feature,
                              const std::vector<unsigned int>& target,
                              int min_bins,
                              int max_bins,
                              double bin_cutoff,
                              int max_n_prebins)
    : feature(feature),
      target(target),
      min_bins(min_bins),
      max_bins(max_bins),
      bin_cutoff(bin_cutoff),
      max_n_prebins(max_n_prebins) {
    total_pos = std::accumulate(target.begin(), target.end(), 0.0);
    total_neg = target.size() - total_pos;
  }
  
  void fit() {
    prebinning();
    calculate_counts_woe();
    enforce_monotonicity();
    ensure_bin_constraints();
    calculate_iv();
  }
  
  List get_results() {
    std::vector<double> woefeature(feature.size());
    
#pragma omp parallel for
    for (size_t i = 0; i < feature.size(); ++i) {
      int bin_idx = find_bin(feature[i]);
      woefeature[i] = woe_values[bin_idx];
    }
    
    DataFrame woebin = DataFrame::create(
      Named("bin") = bin_labels,
      Named("woe") = woe_values,
      Named("iv") = iv_values,
      Named("count") = counts,
      Named("count_pos") = count_pos,
      Named("count_neg") = count_neg
    );
    
    return List::create(
      Named("woefeature") = woefeature,
      Named("woebin") = woebin
    );
  }
  
private:
  void prebinning() {
    std::vector<size_t> sorted_indices(feature.size());
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [this](size_t i1, size_t i2) { return feature[i1] < feature[i2]; });
    
    std::vector<double> sorted_feature(feature.size());
    for (size_t i = 0; i < feature.size(); ++i) {
      sorted_feature[i] = feature[sorted_indices[i]];
    }
    
    int n = feature.size();
    int bin_size = std::max(1, n / max_n_prebins);
    std::vector<double> edges;
    
    for (int i = 1; i < max_n_prebins; ++i) {
      int idx = i * bin_size;
      if (idx < n) {
        edges.push_back(sorted_feature[idx]);
      }
    }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    
    bin_edges = std::vector<double>(edges.size() + 2);
    bin_edges[0] = -std::numeric_limits<double>::infinity();
    std::copy(edges.begin(), edges.end(), bin_edges.begin() + 1);
    bin_edges[bin_edges.size() - 1] = std::numeric_limits<double>::infinity();
  }
  
  void calculate_counts_woe() {
    int num_bins = bin_edges.size() - 1;
    count_pos = std::vector<double>(num_bins, 0);
    count_neg = std::vector<double>(num_bins, 0);
    counts = std::vector<double>(num_bins, 0);
    
#pragma omp parallel for
    for (size_t i = 0; i < feature.size(); ++i) {
      int bin_idx = find_bin(feature[i]);
#pragma omp atomic
      counts[bin_idx] += 1;
      if (target[i] == 1) {
#pragma omp atomic
        count_pos[bin_idx] += 1;
      } else {
#pragma omp atomic
        count_neg[bin_idx] += 1;
      }
    }
    
    calculate_woe();
    update_bin_labels();
  }
  
  void calculate_woe() {
    int num_bins = counts.size();
    woe_values = std::vector<double>(num_bins);
    for (int i = 0; i < num_bins; ++i) {
      double rate_pos = (count_pos[i] + 0.5) / (total_pos + 0.5 * num_bins);
      double rate_neg = (count_neg[i] + 0.5) / (total_neg + 0.5 * num_bins);
      woe_values[i] = std::log(rate_pos / rate_neg);
    }
  }
  
  void update_bin_labels() {
    bin_labels.clear();
    for (size_t i = 0; i < bin_edges.size() - 1; ++i) {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(6);
      if (i == 0) {
        oss << "(-Inf;" << bin_edges[i + 1] << "]";
      } else if (i == bin_edges.size() - 2) {
        oss << "(" << bin_edges[i] << ";+Inf]";
      } else {
        oss << "(" << bin_edges[i] << ";" << bin_edges[i + 1] << "]";
      }
      bin_labels.push_back(oss.str());
    }
  }
  
  int find_bin(double value) {
    auto it = std::upper_bound(bin_edges.begin(), bin_edges.end(), value);
    return std::distance(bin_edges.begin(), it) - 1;
  }
  
  void enforce_monotonicity() {
    bool is_monotonic = false;
    while (!is_monotonic && counts.size() > min_bins) {
      is_monotonic = true;
      for (size_t i = 1; i < woe_values.size(); ++i) {
        if (woe_values[i] < woe_values[i - 1]) {
          merge_bins(i - 1);
          is_monotonic = false;
          break;
        }
      }
    }
  }
  
  void merge_bins(int idx) {
    bin_edges.erase(bin_edges.begin() + idx + 1);
    counts[idx] += counts[idx + 1];
    counts.erase(counts.begin() + idx + 1);
    count_pos[idx] += count_pos[idx + 1];
    count_pos.erase(count_pos.begin() + idx + 1);
    count_neg[idx] += count_neg[idx + 1];
    count_neg.erase(count_neg.begin() + idx + 1);
    woe_values.erase(woe_values.begin() + idx + 1);
    bin_labels.erase(bin_labels.begin() + idx + 1);
    update_bin_labels();
    calculate_woe();
  }
  
  void ensure_bin_constraints() {
    while (counts.size() > max_bins) {
      int idx = find_smallest_iv_diff();
      if (idx == -1) break;
      merge_bins(idx);
    }
    
    handle_rare_bins();
  }
  
  int find_smallest_iv_diff() {
    if (woe_values.size() <= 2) return -1;
    std::vector<double> iv_diffs(woe_values.size() - 1);
    for (size_t i = 0; i < iv_diffs.size(); ++i) {
      iv_diffs[i] = std::abs(woe_values[i + 1] - woe_values[i]);
    }
    return std::distance(iv_diffs.begin(), std::min_element(iv_diffs.begin(), iv_diffs.end()));
  }
  
  void handle_rare_bins() {
    double total_count = std::accumulate(counts.begin(), counts.end(), 0.0);
    bool merged;
    do {
      merged = false;
      for (size_t i = 0; i < counts.size(); ++i) {
        if (counts[i] / total_count < bin_cutoff && counts.size() > min_bins) {
          int merge_idx = (i == 0) ? 0 : i - 1;
          merge_bins(merge_idx);
          merged = true;
          break;
        }
      }
    } while (merged && counts.size() > min_bins);
  }
  
  void calculate_iv() {
    iv_values = std::vector<double>(woe_values.size());
    for (size_t i = 0; i < woe_values.size(); ++i) {
      double rate_pos = count_pos[i] / total_pos;
      double rate_neg = count_neg[i] / total_neg;
      iv_values[i] = (rate_pos - rate_neg) * woe_values[i];
    }
  }
};

// [[Rcpp::export]]
List optimal_binning_numerical_dplc(IntegerVector target,
                                    NumericVector feature,
                                    int min_bins = 3,
                                    int max_bins = 5,
                                    double bin_cutoff = 0.05,
                                    int max_n_prebins = 20) {
  if (min_bins < 2) {
    stop("min_bins must be at least 2");
  }
  if (max_bins < min_bins) {
    stop("max_bins must be greater than or equal to min_bins");
  }
  
  std::vector<double> feature_vec = as<std::vector<double>>(feature);
  std::vector<unsigned int> target_vec = as<std::vector<unsigned int>>(target);
  
  OptimalBinningNumericalDPLC ob(feature_vec, target_vec,
                                 min_bins, max_bins, bin_cutoff, max_n_prebins);
  ob.fit();
  
  return ob.get_results();
}
