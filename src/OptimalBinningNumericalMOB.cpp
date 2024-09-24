#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>
#include <cmath>
#include <limits>
#include <omp.h>

using namespace Rcpp;

struct BinMetrics {
  double lower;
  double upper;
  double woe;
  double iv;
  int count;
  int count_pos;
  int count_neg;
};

class OptimalBinningNumericalMOB {
public:
  OptimalBinningNumericalMOB(int min_bins_ = 2, int max_bins_ = 5, double bin_cutoff_ = 0.05, int max_n_prebins_ = 20)
    : min_bins(min_bins_), max_bins(std::max(min_bins_, max_bins_)), bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_) {}

  void fit(const std::vector<double>& feature_, const std::vector<int>& target_) {
    feature = feature_;
    target = target_;
    validate_input();
    initial_binning();
    optimize_bins();
    calculate_woe_iv();
    enforce_min_bins();
  }

  std::vector<double> get_woefeature() const {
    std::vector<double> woe_feature(feature.size(), 0.0);
#pragma omp parallel for
    for (size_t i = 0; i < feature.size(); ++i) {
      int bin_idx = find_bin(feature[i]);
      if (bin_idx >= 0 && bin_idx < static_cast<int>(woe_values.size())) {
        woe_feature[i] = woe_values[bin_idx];
      }
    }
    return woe_feature;
  }

  std::vector<BinMetrics> get_bin_metrics() const {
    std::vector<BinMetrics> bins;
    for (size_t i = 0; i < bin_edges.size() - 1; ++i) {
      BinMetrics bm;
      bm.lower = bin_edges[i];
      bm.upper = bin_edges[i + 1];
      bm.woe = woe_values[i];
      bm.iv = iv_values[i];
      bm.count = count[i];
      bm.count_pos = count_pos[i];
      bm.count_neg = count_neg[i];
      bins.push_back(bm);
    }
    return bins;
  }

private:
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  std::vector<double> feature;
  std::vector<int> target;
  std::vector<double> bin_edges;
  std::vector<double> woe_values;
  std::vector<double> iv_values;
  std::vector<int> count;
  std::vector<int> count_pos;
  std::vector<int> count_neg;
  double total_pos;
  double total_neg;

  void validate_input() {
    if (feature.empty() || target.empty()) {
      stop("Feature and target vectors cannot be empty.");
    }
    if (feature.size() != target.size()) {
      stop("Feature and target vectors must be of the same length.");
    }
    for (int t : target) {
      if (t != 0 && t != 1) {
        stop("Target vector must be binary (0 and 1).");
      }
    }
    total_pos = std::accumulate(target.begin(), target.end(), 0.0);
    total_neg = feature.size() - total_pos;
    if (total_pos == 0 || total_neg == 0) {
      stop("Target vector must contain both classes (0 and 1).");
    }
  }

  void initial_binning() {
    std::vector<std::pair<double, int>> feature_target(feature.size());
    for (size_t i = 0; i < feature.size(); ++i) {
      feature_target[i] = std::make_pair(feature[i], target[i]);
    }
    std::sort(feature_target.begin(), feature_target.end());

    size_t n = feature_target.size();
    size_t prebin_size = std::max(size_t(1), n / max_n_prebins);
    bin_edges.clear();
    bin_edges.push_back(feature_target.front().first);

    for (size_t i = prebin_size - 1; i < n; i += prebin_size) {
      if (feature_target[i].first > bin_edges.back()) {
        bin_edges.push_back(feature_target[i].first);
      }
    }
    if (bin_edges.back() < feature_target.back().first) {
      bin_edges.push_back(feature_target.back().first);
    }

    // Ensure we don't exceed max_n_prebins
    while (bin_edges.size() > max_n_prebins + 1) {
      auto it = std::min_element(bin_edges.begin() + 1, bin_edges.end() - 1,
                                 [&](double a, double b) { return std::abs(a - b) < std::abs(a - b); });
      bin_edges.erase(it);
    }

    // Ensure we have at least min_bins
    while (bin_edges.size() < min_bins + 1) {
      auto max_gap = std::max_element(bin_edges.begin(), bin_edges.end() - 1,
                                      [&](double a, double b) { return (*(std::next(&a)) - a) < (*(std::next(&b)) - b); });
      bin_edges.insert(std::next(max_gap), (*max_gap + *(std::next(max_gap))) / 2);
    }
  }

  int find_bin(double value) const {
    auto it = std::upper_bound(bin_edges.begin(), bin_edges.end(), value);
    return static_cast<int>(std::distance(bin_edges.begin(), it)) - 1;
  }

  bool is_monotonic(const std::vector<double>& woe) const {
    if (woe.size() < 2) return true;
    bool increasing = true, decreasing = true;
    for (size_t i = 1; i < woe.size(); ++i) {
      if (woe[i] < woe[i-1]) increasing = false;
      if (woe[i] > woe[i-1]) decreasing = false;
      if (!increasing && !decreasing) return false;
    }
    return true;
  }

  void merge_bins() {
    size_t iterations = 0;
    const size_t max_iterations = bin_edges.size() * 2; // Safeguard against infinite loop

    while (iterations++ < max_iterations && bin_edges.size() > min_bins + 1) {
      bool merged = false;

      // Check for bins below bin_cutoff
      for (size_t i = 0; i < count.size(); ++i) {
        double freq = static_cast<double>(count[i]) / feature.size();
        if (freq < bin_cutoff && bin_edges.size() > min_bins + 1) {
          if (i == 0 && count.size() > 1) {
            merge_with_next(0);
          } else if (i > 0) {
            merge_with_prev(i);
          }
          merged = true;
          break;
        }
      }
      if (merged) continue;

      // Check for monotonicity
      if (!is_monotonic(woe_values) && bin_edges.size() > min_bins + 1) {
        size_t break_point = 1;
        bool increasing = woe_values[1] > woe_values[0];
        for (; break_point < woe_values.size(); ++break_point) {
          if ((increasing && woe_values[break_point] < woe_values[break_point-1]) ||
              (!increasing && woe_values[break_point] > woe_values[break_point-1])) {
            break;
          }
        }
        if (break_point < woe_values.size()) {
          merge_with_prev(break_point);
          merged = true;
        }
      }

      if (!merged) break; // If no merges were performed, we're done
    }

    if (iterations >= max_iterations) {
      Rcpp::warning("Maximum iterations reached in merge_bins. Results may be suboptimal.");
    }
  }

  void merge_with_prev(size_t i) {
    if (i <= 0 || i >= bin_edges.size() - 1) return;
    bin_edges.erase(bin_edges.begin() + i);
    count[i-1] += count[i];
    count_pos[i-1] += count_pos[i];
    count_neg[i-1] += count_neg[i];
    count.erase(count.begin() + i);
    count_pos.erase(count_pos.begin() + i);
    count_neg.erase(count_neg.begin() + i);
    woe_values.erase(woe_values.begin() + i);
    iv_values.erase(iv_values.begin() + i);
  }

  void merge_with_next(size_t i) {
    if (i >= bin_edges.size() - 2) return;
    bin_edges.erase(bin_edges.begin() + i + 1);
    count[i] += count[i+1];
    count_pos[i] += count_pos[i+1];
    count_neg[i] += count_neg[i+1];
    count.erase(count.begin() + i + 1);
    count_pos.erase(count_pos.begin() + i + 1);
    count_neg.erase(count_neg.begin() + i + 1);
    woe_values.erase(woe_values.begin() + i + 1);
    iv_values.erase(iv_values.begin() + i + 1);
  }

  void optimize_bins() {
    count.assign(bin_edges.size() - 1, 0);
    count_pos.assign(bin_edges.size() - 1, 0);
    count_neg.assign(bin_edges.size() - 1, 0);

#pragma omp parallel for
    for (size_t i = 0; i < feature.size(); ++i) {
      int bin_idx = find_bin(feature[i]);
      if (bin_idx >= 0 && bin_idx < static_cast<int>(count.size())) {
#pragma omp atomic
        ++count[bin_idx];
        if (target[i] == 1) {
#pragma omp atomic
          ++count_pos[bin_idx];
        } else {
#pragma omp atomic
          ++count_neg[bin_idx];
        }
      }
    }

    calculate_initial_woe();
    merge_bins();

    while (count.size() > max_bins) {
      double min_iv = std::numeric_limits<double>::max();
      size_t merge_idx = 0;
      for (size_t i = 0; i < iv_values.size() - 1; ++i) {
        if (iv_values[i] + iv_values[i+1] < min_iv) {
          min_iv = iv_values[i] + iv_values[i+1];
          merge_idx = i;
        }
      }
      merge_with_next(merge_idx);
      calculate_initial_woe();
    }
  }

  void calculate_initial_woe() {
    woe_values.resize(count.size());
    iv_values.resize(count.size());
    for (size_t i = 0; i < count.size(); ++i) {
      double pct_pos = static_cast<double>(count_pos[i]) / total_pos;
      double pct_neg = static_cast<double>(count_neg[i]) / total_neg;
      pct_pos = std::max(pct_pos, 0.0001);
      pct_neg = std::max(pct_neg, 0.0001);
      woe_values[i] = std::log(pct_pos / pct_neg);
      iv_values[i] = (pct_pos - pct_neg) * woe_values[i];
    }
  }

  void calculate_woe_iv() {
    total_pos = std::accumulate(count_pos.begin(), count_pos.end(), 0.0);
    total_neg = std::accumulate(count_neg.begin(), count_neg.end(), 0.0);
    calculate_initial_woe();
    merge_bins();
  }

  void enforce_min_bins() {
    while (bin_edges.size() < min_bins + 1) {
      // Find the bin with the highest count
      auto max_count_it = std::max_element(count.begin(), count.end());
      size_t max_count_idx = std::distance(count.begin(), max_count_it);

      // Split this bin
      double split_point = (bin_edges[max_count_idx] + bin_edges[max_count_idx + 1]) / 2;
      bin_edges.insert(bin_edges.begin() + max_count_idx + 1, split_point);

      // Recalculate counts for the split bin
      int left_count = 0, left_count_pos = 0, left_count_neg = 0;
      int right_count = 0, right_count_pos = 0, right_count_neg = 0;

      for (size_t i = 0; i < feature.size(); ++i) {
        if (feature[i] > bin_edges[max_count_idx] && feature[i] <= split_point) {
          left_count++;
          if (target[i] == 1) left_count_pos++;
          else left_count_neg++;
        } else if (feature[i] > split_point && feature[i] <= bin_edges[max_count_idx + 2]) {
          right_count++;
          if (target[i] == 1) right_count_pos++;
          else right_count_neg++;
        }
      }

      // Update counts
      count[max_count_idx] = left_count;
      count.insert(count.begin() + max_count_idx + 1, right_count);
      count_pos[max_count_idx] = left_count_pos;
      count_pos.insert(count_pos.begin() + max_count_idx + 1, right_count_pos);
      count_neg[max_count_idx] = left_count_neg;
      count_neg.insert(count_neg.begin() + max_count_idx + 1, right_count_neg);

      // Recalculate WoE and IV
      double pct_pos_left = static_cast<double>(left_count_pos) / total_pos;
      double pct_neg_left = static_cast<double>(left_count_neg) / total_neg;
      double pct_pos_right = static_cast<double>(right_count_pos) / total_pos;
      double pct_neg_right = static_cast<double>(right_count_neg) / total_neg;

      pct_pos_left = std::max(pct_pos_left, 0.0001);
      pct_neg_left = std::max(pct_neg_left, 0.0001);
      pct_pos_right = std::max(pct_pos_right, 0.0001);
      pct_neg_right = std::max(pct_neg_right, 0.0001);

      double woe_left = std::log(pct_pos_left / pct_neg_left);
      double woe_right = std::log(pct_pos_right / pct_neg_right);
      double iv_left = (pct_pos_left - pct_neg_left) * woe_left;
      double iv_right = (pct_pos_right - pct_neg_right) * woe_right;

      woe_values[max_count_idx] = woe_left;
      woe_values.insert(woe_values.begin() + max_count_idx + 1, woe_right);
      iv_values[max_count_idx] = iv_left;
      iv_values.insert(iv_values.begin() + max_count_idx + 1, iv_right);
    }
  }
};

// [[Rcpp::export]]
List optimal_binning_numerical_mob(IntegerVector target, NumericVector feature,
                                   int min_bins = 3, int max_bins = 5,
                                   double bin_cutoff = 0.05, int max_n_prebins = 20) {
  if (feature.size() != target.size()) {
    stop("Feature and target vectors must be of the same length.");
  }

  // Ensure max_bins is at least equal to min_bins
  max_bins = std::max(min_bins, max_bins);

  std::vector<double> feature_vec = as<std::vector<double>>(feature);
  std::vector<int> target_vec = as<std::vector<int>>(target);

  OptimalBinningNumericalMOB binning(min_bins, max_bins, bin_cutoff, max_n_prebins);
  binning.fit(feature_vec, target_vec);

  std::vector<double> woe_feature = binning.get_woefeature();
  std::vector<BinMetrics> bins = binning.get_bin_metrics();

  std::vector<std::string> bin_labels;
  std::vector<double> woe, iv;
  std::vector<int> counts, counts_pos, counts_neg;

  for (const auto& b : bins) {
    bin_labels.push_back("(" + std::to_string(b.lower) + ";" + std::to_string(b.upper) + "]");
    woe.push_back(b.woe);
    iv.push_back(b.iv);
    counts.push_back(b.count);
    counts_pos.push_back(b.count_pos);
    counts_neg.push_back(b.count_neg);
  }

  NumericVector woefeature = wrap(woe_feature);

  DataFrame bin_metrics = DataFrame::create(
    Named("bin") = bin_labels,
    Named("woe") = woe,
    Named("iv") = iv,
    Named("count") = counts,
    Named("count_pos") = counts_pos,
    Named("count_neg") = counts_neg
  );

  return List::create(
    Named("woefeature") = woefeature,
    Named("woebin") = bin_metrics
  );
}
