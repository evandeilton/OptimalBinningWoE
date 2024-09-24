// [[Rcpp::plugins(cpp11)]]

#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <iomanip>

using namespace Rcpp;

class OptimalBinningNumericalOSLP {
private:
  std::vector<double> feature;
  std::vector<double> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  std::vector<double> bin_edges;
  std::vector<std::string> bin_labels;
  std::vector<double> woe_values;
  std::vector<double> iv_values;
  std::vector<int> count_values;
  std::vector<int> count_pos_values;
  std::vector<int> count_neg_values;
  double total_iv;
  bool is_monotonic_increasing;
  
public:
  OptimalBinningNumericalOSLP(const std::vector<double>& feature,
                              const std::vector<double>& target,
                              int min_bins = 3,
                              int max_bins = 5,
                              double bin_cutoff = 0.05,
                              int max_n_prebins = 20) {
    this->feature = feature;
    this->target = target;
    this->min_bins = std::max(min_bins, 2);
    this->max_bins = std::max(max_bins, this->min_bins);
    this->bin_cutoff = bin_cutoff;
    this->max_n_prebins = max_n_prebins;
    this->total_iv = 0.0;
    this->is_monotonic_increasing = true;
  }
  
  void fit() {
    prebin_data();
    merge_bins();
    calculate_woe_iv();
    enforce_monotonicity();
  }
  
  Rcpp::List transform() {
    std::vector<double> woefeature = apply_woe();
    
    DataFrame woebin = DataFrame::create(
      Named("bin") = bin_labels,
      Named("woe") = woe_values,
      Named("iv") = iv_values,
      Named("count") = count_values,
      Named("count_pos") = count_pos_values,
      Named("count_neg") = count_neg_values
    );
    
    return List::create(
      Named("woefeature") = woefeature,
      Named("woebin") = woebin
    );
  }
  
private:
  void prebin_data() {
    std::vector<double> unique_values = feature;
    std::sort(unique_values.begin(), unique_values.end());
    unique_values.erase(std::unique(unique_values.begin(), unique_values.end()), unique_values.end());
    int n_unique = unique_values.size();
    int n_prebins = std::min(max_n_prebins, n_unique);
    
    std::vector<double> quantiles(n_prebins + 1);
    for (int i = 0; i <= n_prebins; ++i) {
      quantiles[i] = i * 1.0 / n_prebins;
    }
    
    std::vector<double> cuts(n_prebins + 1);
    for (int i = 0; i <= n_prebins; ++i) {
      int index = std::min(static_cast<int>(quantiles[i] * n_unique), n_unique - 1);
      cuts[i] = unique_values[index];
    }
    
    bin_edges.clear();
    bin_edges.push_back(-std::numeric_limits<double>::infinity());
    for (size_t i = 1; i < cuts.size() - 1; ++i) {
      if (cuts[i] != cuts[i-1]) {
        bin_edges.push_back(cuts[i]);
      }
    }
    bin_edges.push_back(std::numeric_limits<double>::infinity());
  }
  
  void merge_bins() {
    while (bin_edges.size() - 1 > max_bins || !check_bin_counts()) {
      if (bin_edges.size() - 1 <= min_bins) break;
      merge_adjacent_bins();
    }
  }
  
  void merge_adjacent_bins() {
    int n_bins = bin_edges.size() - 1;
    if (n_bins <= min_bins) return;
    
    std::vector<double> temp_woe_values;
    std::vector<double> temp_iv_values;
    std::vector<int> temp_count_values;
    std::vector<int> temp_count_pos_values;
    std::vector<int> temp_count_neg_values;
    
    calculate_bins(bin_edges, temp_woe_values, temp_iv_values,
                   temp_count_values, temp_count_pos_values, temp_count_neg_values);
    
    int merge_idx = -1;
    double min_iv_loss = std::numeric_limits<double>::max();
    
    for (int i = 0; i < n_bins - 1; ++i) {
      double iv_loss = temp_iv_values[i] + temp_iv_values[i + 1];
      if (iv_loss < min_iv_loss) {
        min_iv_loss = iv_loss;
        merge_idx = i;
      }
    }
    
    if (merge_idx != -1) {
      bin_edges.erase(bin_edges.begin() + merge_idx + 1);
    }
  }
  
  bool check_bin_counts() {
    int total_count = feature.size();
    for (size_t i = 0; i < count_values.size(); ++i) {
      double proportion = (double)count_values[i] / total_count;
      if (proportion < bin_cutoff) {
        return false;
      }
    }
    return true;
  }
  
  void calculate_woe_iv() {
    calculate_bins(bin_edges, woe_values, iv_values,
                   count_values, count_pos_values, count_neg_values);
    total_iv = std::accumulate(iv_values.begin(), iv_values.end(), 0.0);
    
    update_bin_labels();
  }
  
  void calculate_bins(const std::vector<double>& edges,
                      std::vector<double>& woe_vals,
                      std::vector<double>& iv_vals,
                      std::vector<int>& counts,
                      std::vector<int>& counts_pos,
                      std::vector<int>& counts_neg) {
    int n_bins = edges.size() - 1;
    woe_vals.resize(n_bins);
    iv_vals.resize(n_bins);
    counts.resize(n_bins);
    counts_pos.resize(n_bins);
    counts_neg.resize(n_bins);
    
    std::fill(counts.begin(), counts.end(), 0);
    std::fill(counts_pos.begin(), counts_pos.end(), 0);
    std::fill(counts_neg.begin(), counts_neg.end(), 0);
    
    for (size_t i = 0; i < feature.size(); ++i) {
      double val = feature[i];
      int bin_idx = find_bin(edges, val);
      if (bin_idx >= 0 && bin_idx < n_bins) {
        counts[bin_idx] += 1;
        if (target[i] == 1) {
          counts_pos[bin_idx] += 1;
        } else {
          counts_neg[bin_idx] += 1;
        }
      }
    }
    
    double total_pos = std::accumulate(counts_pos.begin(), counts_pos.end(), 0);
    double total_neg = std::accumulate(counts_neg.begin(), counts_neg.end(), 0);
    
    for (int i = 0; i < n_bins; ++i) {
      double pos_rate = (counts_pos[i] + 0.5) / (total_pos + 1.0);
      double neg_rate = (counts_neg[i] + 0.5) / (total_neg + 1.0);
      double woe = std::log(pos_rate / neg_rate);
      double iv = (pos_rate - neg_rate) * woe;
      
      woe_vals[i] = woe;
      iv_vals[i] = iv;
    }
  }
  
  void enforce_monotonicity() {
    bool is_increasing = woe_values[0] < woe_values.back();
    std::vector<double> new_woe_values = woe_values;
    
    for (size_t i = 1; i < new_woe_values.size(); ++i) {
      if (is_increasing && new_woe_values[i] < new_woe_values[i-1]) {
        new_woe_values[i] = new_woe_values[i-1];
      } else if (!is_increasing && new_woe_values[i] > new_woe_values[i-1]) {
        new_woe_values[i] = new_woe_values[i-1];
      }
    }
    
    woe_values = new_woe_values;
    recalculate_iv();
  }
  
  void recalculate_iv() {
    double total_pos = std::accumulate(count_pos_values.begin(), count_pos_values.end(), 0);
    double total_neg = std::accumulate(count_neg_values.begin(), count_neg_values.end(), 0);
    
    for (size_t i = 0; i < woe_values.size(); ++i) {
      double pos_rate = count_pos_values[i] / total_pos;
      double neg_rate = count_neg_values[i] / total_neg;
      iv_values[i] = (pos_rate - neg_rate) * woe_values[i];
    }
    
    total_iv = std::accumulate(iv_values.begin(), iv_values.end(), 0.0);
  }
  
  std::vector<double> apply_woe() {
    int n = feature.size();
    std::vector<double> woefeature(n);
    
    for (int i = 0; i < n; ++i) {
      double val = feature[i];
      int bin_idx = find_bin(bin_edges, val);
      if (bin_idx >= 0 && bin_idx < (int)woe_values.size()) {
        woefeature[i] = woe_values[bin_idx];
      } else {
        woefeature[i] = 0.0;
      }
    }
    
    return woefeature;
  }
  
  int find_bin(const std::vector<double>& edges, double val) {
    int n_bins = edges.size() - 1;
    for (int i = 0; i < n_bins; ++i) {
      if (val > edges[i] && val <= edges[i + 1]) {
        return i;
      }
    }
    if (val <= edges[1]) return 0;
    if (val > edges[n_bins - 1]) return n_bins - 1;
    return -1;
  }
  
  void update_bin_labels() {
    bin_labels.clear();
    for (size_t i = 0; i < bin_edges.size() - 1; ++i) {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(2);
      oss << "(" << (i == 0 ? "-Inf" : std::to_string(bin_edges[i])) << ";"
          << (i == bin_edges.size() - 2 ? "+Inf" : std::to_string(bin_edges[i + 1])) << "]";
      bin_labels.push_back(oss.str());
    }
  }
  
  std::string edge_to_string(double edge) {
    if (std::isinf(edge)) {
      if (edge < 0) return "-Inf";
      else return "+Inf";
    } else {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(2) << edge;
      return oss.str();
    }
  }
};

// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_oslp(Rcpp::NumericVector target,
                                          Rcpp::NumericVector feature,
                                          int min_bins = 3,
                                          int max_bins = 5,
                                          double bin_cutoff = 0.05,
                                          int max_n_prebins = 20) {
  std::vector<double> feature_vec(feature.begin(), feature.end());
  std::vector<double> target_vec(target.begin(), target.end());
  
  OptimalBinningNumericalOSLP binning(feature_vec, target_vec,
                                      min_bins, max_bins,
                                      bin_cutoff, max_n_prebins);
  binning.fit();
  
  Rcpp::List result = binning.transform();
  
  return result;
}
