// [[Rcpp::plugins(openmp)]]
#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

// Class for Monotonic Binning via Linear Programming for Numerical Features
class OptimalBinningNumericalMBLP {
public:
  OptimalBinningNumericalMBLP(NumericVector feature, IntegerVector target,
                              int min_bins, int max_bins, double bin_cutoff, int max_n_prebins)
    : feature(feature), target(target), min_bins(min_bins), max_bins(max_bins),
      bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins) {
    N = feature.size();
  }

  List fit() {
    validate_input();
    prebin();
    enforce_min_bin();
    enforce_monotonicity();
    calculate_woe_iv();
    NumericVector woefeature = apply_woe();

    List result = List::create(
      Named("woefeature") = woefeature,
      Named("woebin") = prepare_woebin()
    );

    return result;
  }

private:
  NumericVector feature;
  IntegerVector target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;

  int N;

  std::vector<double> bin_edges;
  std::vector<int> bin_assignments;
  std::vector<double> bin_woe;
  std::vector<double> bin_iv;
  std::vector<int> bin_count;
  std::vector<int> bin_count_pos;
  std::vector<int> bin_count_neg;
  std::vector<std::string> bin_labels;

  void validate_input() {
    if (N != target.size()) {
      stop("feature and target must have the same length.");
    }

    std::set<int> unique_targets(target.begin(), target.end());
    if (unique_targets.size() != 2 ||
        unique_targets.find(0) == unique_targets.end() ||
        unique_targets.find(1) == unique_targets.end()) {
      stop("target must be a binary vector with values 0 and 1.");
    }

    if (min_bins < 2) {
      stop("min_bins must be at least 2.");
    }
    if (max_bins < min_bins) {
      stop("max_bins must be greater than or equal to min_bins.");
    }

    if (bin_cutoff <= 0 || bin_cutoff >= 1) {
      stop("bin_cutoff must be between 0 and 1.");
    }

    if (max_n_prebins < min_bins) {
      stop("max_n_prebins must be greater than or equal to min_bins.");
    }
  }

  void prebin() {
    // Remove missing values
    std::vector<double> feature_clean;
    std::vector<int> target_clean;
    feature_clean.reserve(N);
    target_clean.reserve(N);
    for (int i = 0; i < N; ++i) {
      if (!ISNA(feature[i])) {
        feature_clean.push_back(feature[i]);
        target_clean.push_back(target[i]);
      }
    }
    int N_clean = feature_clean.size();

    if (N_clean == 0) {
      stop("All feature values are NA.");
    }

    // Sort feature and target together
    std::vector<std::pair<double, int>> paired;
    paired.reserve(N_clean);
    for (int i = 0; i < N_clean; ++i) {
      paired.emplace_back(std::make_pair(feature_clean[i], target_clean[i]));
    }
    std::sort(paired.begin(), paired.end());

    // Reconstruct sorted feature and target
    std::vector<double> feature_sorted(N_clean);
    std::vector<int> target_sorted(N_clean);
    for (int i = 0; i < N_clean; ++i) {
      feature_sorted[i] = paired[i].first;
      target_sorted[i] = paired[i].second;
    }

    // Determine unique values
    std::vector<double> unique_feature = feature_sorted;
    std::sort(unique_feature.begin(), unique_feature.end());
    unique_feature.erase(std::unique(unique_feature.begin(), unique_feature.end()), unique_feature.end());
    int n_unique = unique_feature.size();

    // Determine number of pre-bins
    int n_prebins = std::min(max_n_prebins, n_unique);
    n_prebins = std::max(n_prebins, min_bins);

    // Calculate quantiles for pre-binning
    bin_edges = calculate_quantiles(unique_feature, n_prebins);

    // Assign bins
    bin_assignments.assign(N_clean, -1);
#pragma omp parallel for if(N_clean > 1000)
    for (int i = 0; i < N_clean; ++i) {
      double val = feature_sorted[i];
      // Find bin using upper_bound
      int bin_idx = std::upper_bound(bin_edges.begin(), bin_edges.end(), val) - bin_edges.begin() - 1;
      if (bin_idx < 0) bin_idx = 0;
      if (bin_idx >= (int)(bin_edges.size() - 1)) bin_idx = bin_edges.size() - 2;
      bin_assignments[i] = bin_idx;
    }

    // Initialize counts
    int n_bins = bin_edges.size() - 1;
    bin_count.assign(n_bins, 0);
    bin_count_pos.assign(n_bins, 0);
    bin_count_neg.assign(n_bins, 0);

#pragma omp parallel for if(N_clean > 1000)
    for (int i = 0; i < N_clean; ++i) {
      int bin_idx = bin_assignments[i];
#pragma omp atomic
      bin_count[bin_idx]++;
      if (target_sorted[i] == 1) {
#pragma omp atomic
        bin_count_pos[bin_idx]++;
      } else {
#pragma omp atomic
        bin_count_neg[bin_idx]++;
      }
    }

    // Merge rare bins based on bin_cutoff
    double total = std::accumulate(bin_count.begin(), bin_count.end(), 0.0);
    bool merged = true;
    while (merged) {
      merged = false;
      for (int i = 0; i < bin_count.size(); ++i) {
        double freq = bin_count[i] / total;
        if (freq < bin_cutoff && bin_count.size() > min_bins) {
          // Merge with adjacent bin
          if (i == 0) {
            merge_bins(i, i + 1);
          } else {
            merge_bins(i, i - 1);
          }
          merged = true;
          break;
        }
      }
    }

    // Ensure max_n_prebins is respected
    while (bin_count.size() > max_n_prebins && bin_count.size() > min_bins) {
      int merge_idx = find_min_iv_loss_merge();
      merge_bins(merge_idx, merge_idx + 1);
    }

    // Reassign bins after merging
    n_bins = bin_edges.size() - 1;
    bin_assignments.assign(N_clean, -1);
#pragma omp parallel for if(N_clean > 1000)
    for (int i = 0; i < N_clean; ++i) {
      double val = feature_sorted[i];
      int bin_idx = std::upper_bound(bin_edges.begin(), bin_edges.end(), val) - bin_edges.begin() - 1;
      if (bin_idx < 0) bin_idx = 0;
      if (bin_idx >= (int)(bin_edges.size() - 1)) bin_idx = bin_edges.size() - 2;
      bin_assignments[i] = bin_idx;
    }
  }

  std::vector<double> calculate_quantiles(const std::vector<double>& data, int n_quantiles) {
    std::vector<double> quantiles;
    quantiles.reserve(n_quantiles + 1);

    quantiles.push_back(std::numeric_limits<double>::lowest());

    for (int i = 1; i < n_quantiles; ++i) {
      double p = static_cast<double>(i) / n_quantiles;
      size_t idx = static_cast<size_t>(std::ceil(p * (data.size() - 1)));
      quantiles.push_back(data[idx]);
    }

    quantiles.push_back(std::numeric_limits<double>::max());

    return quantiles;
  }

  void enforce_min_bin() {
    // Ensure the number of bins is not less than min_bins
    while (bin_count.size() < min_bins) {
      // Merge the first two bins
      if (bin_count.size() >= 2) {
        merge_bins(0, 1);
      } else {
        break;
      }
    }
  }

  void enforce_monotonicity() {
    calculate_bin_woe();
    bool monotonic = check_monotonicity(bin_woe);

    while (!monotonic && bin_count.size() > min_bins) {
      int merge_idx = find_min_iv_loss_merge();
      if (merge_idx == -1) {
        Rcpp::Rcout << "No suitable bins to merge. Current number of bins: " << bin_count.size() << std::endl;
        break; // No suitable bins to merge, exit the loop
      }
      merge_bins(merge_idx, merge_idx + 1);
      calculate_bin_woe();
      monotonic = check_monotonicity(bin_woe);
    }

    // Ensure the number of bins does not exceed max_bins
    while (bin_count.size() > max_bins) {
      int merge_idx = find_min_iv_loss_merge();
      if (merge_idx == -1) {
        Rcpp::Rcout << "No suitable bins to merge. Current number of bins: " << bin_count.size() << std::endl;
        break; // No suitable bins to merge, exit the loop
      }
      merge_bins(merge_idx, merge_idx + 1);
      calculate_bin_woe();
    }
  }

  void calculate_bin_woe() {
    int n_bins = bin_count.size();
    double total_pos = std::accumulate(bin_count_pos.begin(), bin_count_pos.end(), 0.0);
    double total_neg = std::accumulate(bin_count_neg.begin(), bin_count_neg.end(), 0.0);

    bin_woe.assign(n_bins, 0.0);
    bin_iv.assign(n_bins, 0.0);

    for (int i = 0; i < n_bins; ++i) {
      double dist_pos = bin_count_pos[i] / total_pos;
      double dist_neg = bin_count_neg[i] / total_neg;

      // Avoid division by zero
      if (dist_pos == 0) {
        dist_pos = 1e-10;
      }
      if (dist_neg == 0) {
        dist_neg = 1e-10;
      }

      bin_woe[i] = std::log(dist_pos / dist_neg);
      bin_iv[i] = (dist_pos - dist_neg) * bin_woe[i];
    }
  }

  bool check_monotonicity(const std::vector<double>& vec) {
    if (vec.size() < 2) {
      return true;
    }

    bool increasing = true;
    bool decreasing = true;

    for (int i = 1; i < vec.size(); ++i) {
      if (vec[i] < vec[i-1]) {
        increasing = false;
      }
      if (vec[i] > vec[i-1]) {
        decreasing = false;
      }
    }

    return increasing || decreasing;
  }


  int find_min_iv_loss_merge() {
    if (bin_iv.size() < 2) {
      Rcpp::Rcout << "Not enough bins to merge. Current number of bins: " << bin_iv.size() << std::endl;
      return -1; // Not enough bins to merge
    }

    double min_iv_loss = std::numeric_limits<double>::max();
    int merge_idx = -1;

    for (int i = 0; i < bin_iv.size() - 1; ++i) {
      double iv_before = bin_iv[i] + bin_iv[i+1];

      // Calculate merged WOE and IV
      double merged_pos = bin_count_pos[i] + bin_count_pos[i+1];
      double merged_neg = bin_count_neg[i] + bin_count_neg[i+1];
      double total_pos = std::accumulate(bin_count_pos.begin(), bin_count_pos.end(), 0.0);
      double total_neg = std::accumulate(bin_count_neg.begin(), bin_count_neg.end(), 0.0);

      double dist_pos = merged_pos / total_pos;
      double dist_neg = merged_neg / total_neg;

      if (dist_pos == 0) {
        dist_pos = 1e-10;
      }
      if (dist_neg == 0) {
        dist_neg = 1e-10;
      }

      double woe_merged = std::log(dist_pos / dist_neg);
      double iv_merged = (dist_pos - dist_neg) * woe_merged;

      double iv_after = iv_merged;

      double iv_loss = iv_before - iv_after;

      if (iv_loss < min_iv_loss) {
        min_iv_loss = iv_loss;
        merge_idx = i;
      }
    }

    return merge_idx;
  }


  void merge_bins(int idx1, int idx2) {
    if (idx1 < 0 || idx2 < 0 || idx1 >= bin_count.size() || idx2 >= bin_count.size()) {
      Rcpp::Rcout << "Invalid merge indices: " << idx1 << ", " << idx2 << std::endl;
      Rcpp::Rcout << "Current number of bins: " << bin_count.size() << std::endl;
      stop("Invalid merge indices.");
    }

    if (idx1 == idx2) {
      Rcpp::Rcout << "Attempting to merge a bin with itself: " << idx1 << std::endl;
      return; // No need to merge, just return
    }

    int lower_idx = std::min(idx1, idx2);
    int higher_idx = std::max(idx1, idx2);

    // Merge bin higher_idx into lower_idx
    bin_edges.erase(bin_edges.begin() + higher_idx);
    bin_count[lower_idx] += bin_count[higher_idx];
    bin_count_pos[lower_idx] += bin_count_pos[higher_idx];
    bin_count_neg[lower_idx] += bin_count_neg[higher_idx];

    bin_count.erase(bin_count.begin() + higher_idx);
    bin_count_pos.erase(bin_count_pos.begin() + higher_idx);
    bin_count_neg.erase(bin_count_neg.begin() + higher_idx);
  }

  void calculate_woe_iv() {
    calculate_bin_woe();
  }

  NumericVector apply_woe() {
    int n_bins = bin_count.size();
    NumericVector woefeature(N, NA_REAL);

    // Create a vector of pairs (feature value, original index)
    std::vector<std::pair<double, int>> feature_with_index(N);
    for (int i = 0; i < N; ++i) {
      feature_with_index[i] = std::make_pair(feature[i], i);
    }

    // Sort the vector based on feature values
    std::sort(feature_with_index.begin(), feature_with_index.end());

    // Apply WoE values
    int bin_idx = 0;
    for (int i = 0; i < N; ++i) {
      double val = feature_with_index[i].first;
      int original_idx = feature_with_index[i].second;

      // Find the correct bin for this value
      while (bin_idx < n_bins - 1 && val > bin_edges[bin_idx + 1]) {
        bin_idx++;
      }

      // Assign WoE value
      if (!ISNA(val)) {
        woefeature[original_idx] = bin_woe[bin_idx];
      }
    }

    return woefeature;
  }

  DataFrame prepare_woebin() {
    int n_bins = bin_count.size();
    bin_labels.assign(n_bins, "");

    for (int i = 0; i < n_bins; ++i) {
      std::string left, right;

      if (bin_edges[i] == std::numeric_limits<double>::lowest()) {
        left = "(-Inf";
      } else {
        left = "(" + std::to_string(bin_edges[i]);
      }

      if (bin_edges[i + 1] == std::numeric_limits<double>::max()) {
        right = "+Inf]";
      } else {
        right = std::to_string(bin_edges[i + 1]) + "]";
      }

      bin_labels[i] = left + ";" + right;
    }

    DataFrame woebin = DataFrame::create(
      Named("bin") = wrap(bin_labels),
      Named("woe") = wrap(bin_woe),
      Named("iv") = wrap(bin_iv),
      Named("count") = wrap(bin_count),
      Named("count_pos") = wrap(bin_count_pos),
      Named("count_neg") = wrap(bin_count_neg)
    );
    return woebin;
  }
};

// [[Rcpp::export]]
List optimal_binning_numerical_mblp(IntegerVector target, NumericVector feature,
                                    int min_bins = 3, int max_bins = 5, double bin_cutoff = 0.05, int max_n_prebins = 20) {
  OptimalBinningNumericalMBLP ob(feature, target, min_bins, max_bins, bin_cutoff, max_n_prebins);
  return ob.fit();
}
