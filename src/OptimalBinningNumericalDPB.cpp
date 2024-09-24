// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::plugins(openmp)]]

#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <omp.h>

using namespace Rcpp;

class OptimalBinningNumericalDPB {
private:
  std::vector<double> feature;
  std::vector<int> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  int n_threads;

  struct Bin {
    double lower_bound;
    double upper_bound;
    int count;
    int count_pos;
    int count_neg;
    double woe;
    double iv;
  };

  std::vector<Bin> bins;
  std::vector<double> woefeature;

public:
  OptimalBinningNumericalDPB(const std::vector<double>& feature,
                             const std::vector<int>& target,
                             int min_bins = 3,
                             int max_bins = 5,
                             double bin_cutoff = 0.05,
                             int max_n_prebins = 20,
                             int n_threads = 1)
    : feature(feature),
      target(target),
      min_bins(min_bins),
      max_bins(max_bins),
      bin_cutoff(bin_cutoff),
      max_n_prebins(max_n_prebins),
      n_threads(n_threads) {}

  void fit() {
    // Input validation
    if (min_bins < 2) {
      stop("min_bins must be at least 2.");
    }
    if (max_bins < min_bins) {
      stop("max_bins must be greater than or equal to min_bins.");
    }
    if (bin_cutoff < 0 || bin_cutoff > 0.5) {
      stop("bin_cutoff must be between 0 and 0.5.");
    }
    if (max_n_prebins < min_bins) {
      stop("max_n_prebins must be greater than or equal to min_bins.");
    }

    // Pre-binning
    std::vector<Bin> prebins = create_prebins();

    // Dynamic programming to find optimal bins
    bins = optimal_binning(prebins);

    // Compute WOE and IV
    compute_woe_iv();

    // Apply WOE to feature
    woefeature = apply_woe();
  }

  List get_result() {
    // Prepare output DataFrame
    int n_bins = bins.size();
    CharacterVector bin_intervals(n_bins);
    NumericVector woe_values(n_bins);
    NumericVector iv_values(n_bins);
    IntegerVector counts(n_bins);
    IntegerVector count_pos(n_bins);
    IntegerVector count_neg(n_bins);

    for (int i = 0; i < n_bins; ++i) {
      std::string interval = "(" +
        (bins[i].lower_bound == R_NegInf ? "-Inf" : std::to_string(bins[i].lower_bound)) + ";" +
        (bins[i].upper_bound == R_PosInf ? "+Inf" : std::to_string(bins[i].upper_bound)) + "]";
      bin_intervals[i] = interval;
      woe_values[i] = bins[i].woe;
      iv_values[i] = bins[i].iv;
      counts[i] = bins[i].count;
      count_pos[i] = bins[i].count_pos;
      count_neg[i] = bins[i].count_neg;
    }

    DataFrame woebin = DataFrame::create(
      Named("bin") = bin_intervals,
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
  std::vector<Bin> create_prebins() {
    // Sort feature and target
    int n = feature.size();
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int i, int j) { return feature[i] < feature[j]; });

    // Determine pre-bin edges
    int n_prebins = std::min(max_n_prebins, n);
    std::vector<double> edges;
    edges.push_back(R_NegInf);
    for (int i = 1; i < n_prebins; ++i) {
      int idx = i * n / n_prebins - 1;
      double edge = feature[indices[idx]];
      edges.push_back(edge);
    }
    edges.push_back(R_PosInf);

    // Create prebins
    std::vector<Bin> prebins(edges.size() - 1);
    for (int i = 0; i < edges.size() - 1; ++i) {
      prebins[i].lower_bound = edges[i];
      prebins[i].upper_bound = edges[i + 1];
      prebins[i].count = 0;
      prebins[i].count_pos = 0;
      prebins[i].count_neg = 0;
    }

    // Assign data to prebins
    for (int idx = 0; idx < n; ++idx) {
      double x = feature[idx];
      int y = target[idx];
      for (int i = 0; i < prebins.size(); ++i) {
        if (x > prebins[i].lower_bound && x <= prebins[i].upper_bound) {
          prebins[i].count++;
          if (y == 1) {
            prebins[i].count_pos++;
          } else {
            prebins[i].count_neg++;
          }
          break;
        }
      }
    }

    // Remove empty bins
    prebins.erase(
      std::remove_if(prebins.begin(), prebins.end(), [](const Bin& b) { return b.count == 0; }),
      prebins.end()
    );

    return prebins;
  }

  std::vector<Bin> optimal_binning(const std::vector<Bin>& prebins) {
    int n_prebins = prebins.size();
    int total_pos = std::accumulate(target.begin(), target.end(), 0);
    int total_neg = target.size() - total_pos;

    // Initialize DP table
    std::vector<std::vector<double>> dp(n_prebins + 1, std::vector<double>(max_bins + 1, -INFINITY));
    std::vector<std::vector<int>> last_split(n_prebins + 1, std::vector<int>(max_bins + 1, -1));
    dp[0][0] = 0;

    // Dynamic programming
    for (int i = 1; i <= n_prebins; ++i) {
      for (int k = 1; k <= std::min(i, max_bins); ++k) {
        for (int j = k - 1; j < i; ++j) {
          // Merge bins from j to i-1
          int count = 0;
          int count_pos = 0;
          int count_neg = 0;
          for (int m = j; m < i; ++m) {
            count += prebins[m].count;
            count_pos += prebins[m].count_pos;
            count_neg += prebins[m].count_neg;
          }

          // Check bin_cutoff
          if ((double)count / target.size() < bin_cutoff) {
            continue;
          }

          // Calculate WOE and IV
          double dist_pos = (double)count_pos / total_pos;
          double dist_neg = (double)count_neg / total_neg;
          if (dist_pos == 0 || dist_neg == 0) continue; // Avoid division by zero
          double woe = std::log(dist_pos / dist_neg);
          double iv = (dist_pos - dist_neg) * woe;

          double total_iv = dp[j][k - 1] + iv;

          if (total_iv > dp[i][k]) {
            dp[i][k] = total_iv;
            last_split[i][k] = j;
          }
        }
      }
    }

    // Backtracking to find optimal bins
    int bins_used = -1;
    double max_iv = -INFINITY;
    for (int k = min_bins; k <= max_bins; ++k) {
      if (dp[n_prebins][k] > max_iv) {
        max_iv = dp[n_prebins][k];
        bins_used = k;
      }
    }
    if (bins_used == -1) {
      stop("Failed to find optimal binning solution.");
    }

    std::vector<Bin> optimal_bins;
    int i = n_prebins;
    int k = bins_used;
    while (k > 0) {
      int j = last_split[i][k];
      // Merge bins from j to i-1
      Bin merged_bin;
      merged_bin.lower_bound = prebins[j].lower_bound;
      merged_bin.upper_bound = prebins[i - 1].upper_bound;
      merged_bin.count = 0;
      merged_bin.count_pos = 0;
      merged_bin.count_neg = 0;
      for (int m = j; m < i; ++m) {
        merged_bin.count += prebins[m].count;
        merged_bin.count_pos += prebins[m].count_pos;
        merged_bin.count_neg += prebins[m].count_neg;
      }
      optimal_bins.push_back(merged_bin);
      i = j;
      k--;
    }

    // Reverse bins to correct order
    std::reverse(optimal_bins.begin(), optimal_bins.end());

    // Enforce monotonicity
    enforce_monotonicity(optimal_bins, total_pos, total_neg);

    return optimal_bins;
  }

  void enforce_monotonicity(std::vector<Bin>& bins, int total_pos, int total_neg) {
    // Calculate initial WOE for bins
    for (auto& bin : bins) {
      double dist_pos = (double)bin.count_pos / total_pos;
      double dist_neg = (double)bin.count_neg / total_neg;
      if (dist_pos == 0 || dist_neg == 0) {
        bin.woe = 0;
      } else {
        bin.woe = std::log(dist_pos / dist_neg);
      }
    }

    // Determine monotonicity direction
    bool increasing = true;
    bool decreasing = true;
    for (size_t i = 1; i < bins.size(); ++i) {
      if (bins[i].woe < bins[i - 1].woe) {
        increasing = false;
      }
      if (bins[i].woe > bins[i - 1].woe) {
        decreasing = false;
      }
    }

    // If not monotonic, merge bins to enforce monotonicity
    if (!increasing && !decreasing) {
      // Simple approach: merge bins to enforce monotonicity
      // This can be improved with more sophisticated methods
      for (size_t i = 1; i < bins.size(); ++i) {
        if ((bins[i].woe < bins[i - 1].woe && increasing) ||
            (bins[i].woe > bins[i - 1].woe && decreasing)) {
          // Merge bins[i - 1] and bins[i]
          bins[i - 1].upper_bound = bins[i].upper_bound;
          bins[i - 1].count += bins[i].count;
          bins[i - 1].count_pos += bins[i].count_pos;
          bins[i - 1].count_neg += bins[i].count_neg;
          bins.erase(bins.begin() + i);
          i--;
        }
      }
    }
  }

  void compute_woe_iv() {
    int total_pos = std::accumulate(target.begin(), target.end(), 0);
    int total_neg = target.size() - total_pos;

    // Compute WOE and IV for bins
    for (auto& bin : bins) {
      double dist_pos = (double)bin.count_pos / total_pos;
      double dist_neg = (double)bin.count_neg / total_neg;
      if (dist_pos == 0 || dist_neg == 0) {
        bin.woe = 0;
        bin.iv = 0;
      } else {
        bin.woe = std::log(dist_pos / dist_neg);
        bin.iv = (dist_pos - dist_neg) * bin.woe;
      }
    }
  }

  std::vector<double> apply_woe() {
    std::vector<double> woefeature(feature.size());
    int n = feature.size();

#pragma omp parallel for num_threads(n_threads)
    for (int idx = 0; idx < n; ++idx) {
      double x = feature[idx];
      for (const auto& bin : bins) {
        if (x > bin.lower_bound && x <= bin.upper_bound) {
          woefeature[idx] = bin.woe;
          break;
        }
      }
    }

    return woefeature;
  }
};

// [[Rcpp::export]]
List optimal_binning_numerical_dpb(IntegerVector target,
                                   NumericVector feature,
                                   int min_bins = 3,
                                   int max_bins = 5,
                                   double bin_cutoff = 0.05,
                                   int max_n_prebins = 20,
                                   int n_threads = 1) {
  OptimalBinningNumericalDPB binning(as<std::vector<double>>(feature),
                                     as<std::vector<int>>(target),
                                     min_bins,
                                     max_bins,
                                     bin_cutoff,
                                     max_n_prebins,
                                     n_threads);
  binning.fit();
  return binning.get_result();
}

