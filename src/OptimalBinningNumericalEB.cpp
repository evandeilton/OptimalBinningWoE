// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::plugins(openmp)]]
#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <omp.h>

using namespace Rcpp;

class OptimalBinningNumericalEB {
private:
  NumericVector feature;
  IntegerVector target;
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

  void validate_inputs() {
    if (feature.size() != target.size()) {
      stop("Feature and target vectors must be of the same length.");
    }

    if (min_bins < 2) {
      stop("min_bins must be at least 2.");
    }

    if (max_bins < min_bins) {
      stop("max_bins must be greater than or equal to min_bins.");
    }

    if (bin_cutoff <= 0 || bin_cutoff >= 0.5) {
      stop("bin_cutoff must be between 0 and 0.5.");
    }

    if (max_n_prebins < max_bins) {
      stop("max_n_prebins must be greater than or equal to max_bins.");
    }

    if (is_true(any(is_na(feature)))) {
      stop("Feature vector contains NA values.");
    }

    if (is_true(any(is_na(target)))) {
      stop("Target vector contains NA values.");
    }

    IntegerVector unique_targets = sort_unique(target);
    if (unique_targets.size() != 2 || unique_targets[0] != 0 || unique_targets[1] != 1) {
      stop("Target vector must be binary (0 and 1).");
    }
  }

  NumericVector compute_quantiles(NumericVector x, NumericVector probs) {
    int n = x.size();
    NumericVector x_sorted = clone(x).sort();
    NumericVector quantiles(probs.size());

    for (int i = 0; i < probs.size(); ++i) {
      double p = probs[i];
      double h = (n - 1) * p;
      int h_int = (int)std::floor(h);
      double h_frac = h - h_int;

      if (h_int + 1 < n) {
        quantiles[i] = x_sorted[h_int] + h_frac * (x_sorted[h_int + 1] - x_sorted[h_int]);
      } else {
        quantiles[i] = x_sorted[n - 1];
      }
    }
    return quantiles;
  }

  void initialize_prebins(std::vector<double>& cut_points) {
    // Create max_n_prebins equally spaced quantiles
    NumericVector probs(max_n_prebins - 1);
    for (int i = 1; i < max_n_prebins; ++i) {
      probs[i - 1] = (double)i / max_n_prebins;
    }

    // Compute quantiles
    NumericVector cuts = compute_quantiles(feature, probs);

    // Remove duplicate cut points
    std::set<double> unique_cuts(cuts.begin(), cuts.end());
    cut_points.assign(unique_cuts.begin(), unique_cuts.end());
    std::sort(cut_points.begin(), cut_points.end());
  }

  void create_initial_bins(const std::vector<double>& cut_points) {
    // Initialize bins based on cut points
    bins.clear();
    std::vector<double> boundaries = cut_points;
    boundaries.insert(boundaries.begin(), R_NegInf);
    boundaries.push_back(R_PosInf);

    int n_bins = boundaries.size() - 1;
    bins.resize(n_bins);

    // Initialize bin counts
#pragma omp parallel for num_threads(n_threads)
    for (int i = 0; i < n_bins; ++i) {
      bins[i].lower_bound = boundaries[i];
      bins[i].upper_bound = boundaries[i + 1];
      bins[i].count = 0;
      bins[i].count_pos = 0;
      bins[i].count_neg = 0;
      bins[i].woe = 0.0;
      bins[i].iv = 0.0;
    }

    // Assign observations to bins
    int n = feature.size();
#pragma omp parallel for num_threads(n_threads)
    for (int i = 0; i < n; ++i) {
      double val = feature[i];
      int bin_index = -1;
      // Find the bin for the current value
      for (int j = 0; j < n_bins; ++j) {
        if (val > bins[j].lower_bound && val <= bins[j].upper_bound) {
          bin_index = j;
          break;
        }
      }
      if (bin_index != -1) {
#pragma omp atomic
        bins[bin_index].count += 1;

        if (target[i] == 1) {
#pragma omp atomic
          bins[bin_index].count_pos += 1;
        } else {
#pragma omp atomic
          bins[bin_index].count_neg += 1;
        }
      }
    }
  }

  void merge_small_bins() {
    // Merge bins with frequency below bin_cutoff
    int total_count = feature.size();
    double min_bin_count = bin_cutoff * total_count;

    bool bins_merged = true;
    while (bins_merged) {
      bins_merged = false;
      for (size_t i = 0; i < bins.size(); ++i) {
        if (bins[i].count < min_bin_count) {
          if (i == 0) {
            // Merge with next bin
            bins[i + 1].lower_bound = bins[i].lower_bound;
            bins[i + 1].count += bins[i].count;
            bins[i + 1].count_pos += bins[i].count_pos;
            bins[i + 1].count_neg += bins[i].count_neg;
            bins.erase(bins.begin() + i);
          } else {
            // Merge with previous bin
            bins[i - 1].upper_bound = bins[i].upper_bound;
            bins[i - 1].count += bins[i].count;
            bins[i - 1].count_pos += bins[i].count_pos;
            bins[i - 1].count_neg += bins[i].count_neg;
            bins.erase(bins.begin() + i);
          }
          bins_merged = true;
          break;
        }
      }
    }
  }

  void calculate_woe_iv() {
    int total_pos = std::accumulate(target.begin(), target.end(), 0);
    int total_neg = target.size() - total_pos;

#pragma omp parallel for num_threads(n_threads)
    for (size_t i = 0; i < bins.size(); ++i) {
      double dist_pos = (double)bins[i].count_pos / total_pos;
      double dist_neg = (double)bins[i].count_neg / total_neg;

      // Avoid division by zero
      if (dist_pos == 0) dist_pos = 1e-8;
      if (dist_neg == 0) dist_neg = 1e-8;

      bins[i].woe = std::log(dist_pos / dist_neg);
      bins[i].iv = (dist_pos - dist_neg) * bins[i].woe;
    }
  }

  void enforce_monotonicity() {
    // Ensure that WoE is monotonic across bins
    bool monotonic = false;
    while (!monotonic) {
      monotonic = true;
      for (size_t i = 1; i < bins.size(); ++i) {
        if (bins[i - 1].woe > bins[i].woe) {
          // Merge bins i-1 and i
          bins[i - 1].upper_bound = bins[i].upper_bound;
          bins[i - 1].count += bins[i].count;
          bins[i - 1].count_pos += bins[i].count_pos;
          bins[i - 1].count_neg += bins[i].count_neg;
          bins.erase(bins.begin() + i);
          calculate_woe_iv();
          monotonic = false;
          break;
        }
      }
    }
  }

  void adjust_bins() {
    // Ensure number of bins is within min_bins and max_bins
    while ((int)bins.size() > max_bins) {
      // Merge bins with smallest total count
      size_t merge_index = 0;
      int min_count = bins[0].count + bins[1].count;
      for (size_t i = 1; i < bins.size() - 1; ++i) {
        int combined_count = bins[i].count + bins[i + 1].count;
        if (combined_count < min_count) {
          min_count = combined_count;
          merge_index = i;
        }
      }
      // Merge bins at merge_index and merge_index + 1
      bins[merge_index].upper_bound = bins[merge_index + 1].upper_bound;
      bins[merge_index].count += bins[merge_index + 1].count;
      bins[merge_index].count_pos += bins[merge_index + 1].count_pos;
      bins[merge_index].count_neg += bins[merge_index + 1].count_neg;
      bins.erase(bins.begin() + merge_index + 1);
      calculate_woe_iv();
      enforce_monotonicity();
    }

    while ((int)bins.size() < min_bins) {
      // Cannot split bins further; break the loop
      break;
    }
  }

public:
  OptimalBinningNumericalEB(NumericVector feature, IntegerVector target,
                            int min_bins = 3, int max_bins = 5,
                            double bin_cutoff = 0.05, int max_n_prebins = 20,
                            int n_threads = 1)
    : feature(feature), target(target), min_bins(min_bins),
      max_bins(max_bins), bin_cutoff(bin_cutoff),
      max_n_prebins(max_n_prebins), n_threads(n_threads) {
    validate_inputs();
  }

  List fit() {
    std::vector<double> cut_points;
    initialize_prebins(cut_points);
    create_initial_bins(cut_points);
    merge_small_bins();
    calculate_woe_iv();
    enforce_monotonicity();
    adjust_bins();

    // Prepare output
    NumericVector woefeature(feature.size(), NA_REAL);
    NumericVector woevalues(bins.size());
    NumericVector ivvalues(bins.size());
    CharacterVector bin_names(bins.size());
    IntegerVector counts(bins.size());
    IntegerVector count_pos(bins.size());
    IntegerVector count_neg(bins.size());

    // Assign WoE to feature values
    int n = feature.size();
#pragma omp parallel for num_threads(n_threads)
    for (int i = 0; i < n; ++i) {
      double val = feature[i];
      for (size_t j = 0; j < bins.size(); ++j) {
        if (val > bins[j].lower_bound && val <= bins[j].upper_bound) {
          woefeature[i] = bins[j].woe;
          break;
        }
      }
    }

    // Prepare binning details
    for (size_t i = 0; i < bins.size(); ++i) {
      std::string bin_name = "(";
      if (std::isinf(bins[i].lower_bound)) {
        bin_name += "-Inf";
      } else {
        bin_name += std::to_string(bins[i].lower_bound);
      }
      bin_name += ";";
      if (std::isinf(bins[i].upper_bound)) {
        bin_name += "+Inf";
      } else {
        bin_name += std::to_string(bins[i].upper_bound);
      }
      bin_name += "]";
      bin_names[i] = bin_name;
      woevalues[i] = bins[i].woe;
      ivvalues[i] = bins[i].iv;
      counts[i] = bins[i].count;
      count_pos[i] = bins[i].count_pos;
      count_neg[i] = bins[i].count_neg;
    }

    DataFrame woebin = DataFrame::create(
      Named("bin") = bin_names,
      Named("woe") = woevalues,
      Named("iv") = ivvalues,
      Named("count") = counts,
      Named("count_pos") = count_pos,
      Named("count_neg") = count_neg
    );

    return List::create(
      Named("woefeature") = woefeature,
      Named("woebin") = woebin
    );
  }
};

// [[Rcpp::export]]
List optimal_binning_numerical_eb(IntegerVector target, NumericVector feature,
                                  int min_bins = 3, int max_bins = 5,
                                  double bin_cutoff = 0.05, int max_n_prebins = 20,
                                  int n_threads = 1) {
  OptimalBinningNumericalEB ob(feature, target, min_bins, max_bins, bin_cutoff, max_n_prebins, n_threads);
  return ob.fit();
}
