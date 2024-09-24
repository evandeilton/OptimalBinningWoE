#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <omp.h>

using namespace Rcpp;

class OptimalBinningNumericalSBB {
public:
  // Constructor
  OptimalBinningNumericalSBB(int min_bins, int max_bins, double bin_cutoff, int max_n_prebins)
    : min_bins(min_bins), max_bins(max_bins), bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins) {}

  // Fit method
  void fit(const std::vector<int>& target, const std::vector<double>& feature) {
    // Validate inputs
    if (target.size() != feature.size()) {
      stop("Length of target and feature must be the same.");
    }
    size_t n = target.size();
    if (n == 0) {
      stop("Input vectors are empty.");
    }
    // Check that target is binary
    for (size_t i = 0; i < n; ++i) {
      if (target[i] != 0 && target[i] != 1) {
        stop("Target must be binary (0 or 1).");
      }
    }
    // Prebinning
    prebinning(target, feature);
    // Merge bins with low counts
    merge_bins_by_bin_cutoff();
    // Enforce monotonicity
    enforce_monotonicity();
    // Ensure number of bins is between min_bins and max_bins
    enforce_bin_limits();
    // Compute final WOE and IV
    compute_woe_iv();
    // Assign WOE values to feature
    assign_woe_feature(feature);
    // Prepare woebin DataFrame
    prepare_woebin_dataframe();
  }

  // Get methods
  std::vector<double> get_woefeature() {
    return woefeature;
  }

  DataFrame get_woebin() {
    return woebin;
  }

private:
  // Member variables
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;

  std::vector<double> woefeature;
  DataFrame woebin;

  // Helper methods
  void prebinning(const std::vector<int>& target, const std::vector<double>& feature);
  void merge_bins_by_bin_cutoff();
  void compute_woe_iv();
  void enforce_monotonicity();
  void enforce_bin_limits();
  void assign_woe_feature(const std::vector<double>& feature);
  void prepare_woebin_dataframe();

  // Data structures for internal computation
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
};

void OptimalBinningNumericalSBB::prebinning(const std::vector<int>& target, const std::vector<double>& feature) {
  size_t n = target.size();
  // Create a vector of indices to sort feature
  std::vector<size_t> idx(n);
  for (size_t i = 0; i < n; ++i) idx[i] = i;
  std::sort(idx.begin(), idx.end(), [&](size_t i1, size_t i2) {
    return feature[i1] < feature[i2];
  });
  // Determine bin edges for pre-binning
  size_t bin_size = n / max_n_prebins;
  if (bin_size == 0) bin_size = 1; // Ensure at least one observation per bin
  std::vector<double> bin_edges;
  for (size_t i = bin_size; i < n; i += bin_size) {
    bin_edges.push_back(feature[idx[i]]);
  }
  // Ensure unique bin edges
  std::sort(bin_edges.begin(), bin_edges.end());
  bin_edges.erase(std::unique(bin_edges.begin(), bin_edges.end()), bin_edges.end());
  // Create bins
  bins.clear();
  double lower = -std::numeric_limits<double>::infinity();
  for (size_t i = 0; i <= bin_edges.size(); ++i) {
    double upper;
    if (i < bin_edges.size()) {
      upper = bin_edges[i];
    } else {
      upper = std::numeric_limits<double>::infinity();
    }
    bins.push_back({lower, upper, 0, 0, 0, 0.0, 0.0});
    lower = upper;
  }
  // Assign observations to bins and compute counts
  size_t num_bins = bins.size();
  std::vector<int> bin_indices(n);
#pragma omp parallel for
  for (size_t i = 0; i < n; ++i) {
    double x = feature[i];
    int bin_index = -1;
    for (size_t j = 0; j < num_bins; ++j) {
      if (x > bins[j].lower_bound && x <= bins[j].upper_bound) {
        bin_index = j;
        break;
      }
    }
    if (bin_index == -1) {
      if (x == bins[0].lower_bound) {
        bin_index = 0;
      } else if (x > bins.back().upper_bound) {
        bin_index = num_bins - 1;
      } else {
        // Should not reach here
        stop("Error in assigning bins.");
      }
    }
    bin_indices[i] = bin_index;
  }
  // Compute counts in each bin
#pragma omp parallel for
  for (size_t j = 0; j < num_bins; ++j) {
    bins[j].count = 0;
    bins[j].count_pos = 0;
    bins[j].count_neg = 0;
  }
#pragma omp parallel
{
  std::vector<int> local_count(num_bins, 0);
  std::vector<int> local_count_pos(num_bins, 0);
  std::vector<int> local_count_neg(num_bins, 0);
#pragma omp for nowait
  for (size_t i = 0; i < n; ++i) {
    int bin_index = bin_indices[i];
    local_count[bin_index]++;
    if (target[i] == 1) {
      local_count_pos[bin_index]++;
    } else {
      local_count_neg[bin_index]++;
    }
  }
#pragma omp critical
{
  for (size_t j = 0; j < num_bins; ++j) {
    bins[j].count += local_count[j];
    bins[j].count_pos += local_count_pos[j];
    bins[j].count_neg += local_count_neg[j];
  }
}
}
}

void OptimalBinningNumericalSBB::merge_bins_by_bin_cutoff() {
  // Merge bins with low counts based on bin_cutoff
  size_t num_bins = bins.size();
  int total_count = 0;
  for (size_t i = 0; i < num_bins; ++i) {
    total_count += bins[i].count;
  }
  // Compute minimum count threshold
  int min_count = static_cast<int>(total_count * bin_cutoff);
  if (min_count < 1) min_count = 1;
  // Merge bins with counts less than min_count
  std::vector<Bin> new_bins;
  Bin current_bin = bins[0];
  for (size_t i = 1; i < num_bins; ++i) {
    if (current_bin.count < min_count || bins[i].count < min_count) {
      // Merge current_bin and bins[i]
      current_bin.upper_bound = bins[i].upper_bound;
      current_bin.count += bins[i].count;
      current_bin.count_pos += bins[i].count_pos;
      current_bin.count_neg += bins[i].count_neg;
    } else {
      new_bins.push_back(current_bin);
      current_bin = bins[i];
    }
  }
  new_bins.push_back(current_bin);
  bins = new_bins;
}

void OptimalBinningNumericalSBB::compute_woe_iv() {
  // Compute WOE and IV for each bin
  int total_pos = 0;
  int total_neg = 0;
  for (size_t i = 0; i < bins.size(); ++i) {
    total_pos += bins[i].count_pos;
    total_neg += bins[i].count_neg;
  }
  for (size_t i = 0; i < bins.size(); ++i) {
    double dist_pos = (bins[i].count_pos + 1e-6) / (total_pos + 1e-6 * bins.size());
    double dist_neg = (bins[i].count_neg + 1e-6) / (total_neg + 1e-6 * bins.size());
    bins[i].woe = std::log(dist_pos / dist_neg);
    bins[i].iv = (dist_pos - dist_neg) * bins[i].woe;
  }
}

void OptimalBinningNumericalSBB::enforce_monotonicity() {
  // Enforce monotonicity of WOE
  bool monotonic = false;
  while (!monotonic) {
    monotonic = true;
    for (size_t i = 1; i < bins.size(); ++i) {
      if (bins[i-1].woe > bins[i].woe) {
        // Merge bins[i-1] and bins[i]
        bins[i-1].upper_bound = bins[i].upper_bound;
        bins[i-1].count += bins[i].count;
        bins[i-1].count_pos += bins[i].count_pos;
        bins[i-1].count_neg += bins[i].count_neg;
        bins.erase(bins.begin() + i);
        compute_woe_iv();
        monotonic = false;
        break;
      }
    }
  }
}

void OptimalBinningNumericalSBB::enforce_bin_limits() {
  // Ensure that number of bins is between min_bins and max_bins
  while (bins.size() > static_cast<size_t>(max_bins)) {
    // Merge bins with smallest IV
    size_t min_iv_index = 0;
    double min_iv = bins[0].iv;
    for (size_t i = 1; i < bins.size(); ++i) {
      if (bins[i].iv < min_iv) {
        min_iv = bins[i].iv;
        min_iv_index = i;
      }
    }
    if (min_iv_index == 0) {
      bins[0].upper_bound = bins[1].upper_bound;
      bins[0].count += bins[1].count;
      bins[0].count_pos += bins[1].count_pos;
      bins[0].count_neg += bins[1].count_neg;
      bins.erase(bins.begin() + 1);
    } else {
      bins[min_iv_index-1].upper_bound = bins[min_iv_index].upper_bound;
      bins[min_iv_index-1].count += bins[min_iv_index].count;
      bins[min_iv_index-1].count_pos += bins[min_iv_index].count_pos;
      bins[min_iv_index-1].count_neg += bins[min_iv_index].count_neg;
      bins.erase(bins.begin() + min_iv_index);
    }
    compute_woe_iv();
  }
  while (bins.size() < static_cast<size_t>(min_bins)) {
    // Merge bins with smallest counts
    size_t min_count_index = 0;
    int min_count = bins[0].count;
    for (size_t i = 1; i < bins.size(); ++i) {
      if (bins[i].count < min_count) {
        min_count = bins[i].count;
        min_count_index = i;
      }
    }
    if (min_count_index == 0) {
      bins[0].upper_bound = bins[1].upper_bound;
      bins[0].count += bins[1].count;
      bins[0].count_pos += bins[1].count_pos;
      bins[0].count_neg += bins[1].count_neg;
      bins.erase(bins.begin() + 1);
    } else {
      bins[min_count_index-1].upper_bound = bins[min_count_index].upper_bound;
      bins[min_count_index-1].count += bins[min_count_index].count;
      bins[min_count_index-1].count_pos += bins[min_count_index].count_pos;
      bins[min_count_index-1].count_neg += bins[min_count_index].count_neg;
      bins.erase(bins.begin() + min_count_index);
    }
    compute_woe_iv();
  }
}

void OptimalBinningNumericalSBB::assign_woe_feature(const std::vector<double>& feature) {
  size_t n = feature.size();
  woefeature.resize(n);
  size_t num_bins = bins.size();
#pragma omp parallel for
  for (size_t i = 0; i < n; ++i) {
    double x = feature[i];
    double woe_value = 0.0;
    for (size_t j = 0; j < num_bins; ++j) {
      if (x > bins[j].lower_bound && x <= bins[j].upper_bound) {
        woe_value = bins[j].woe;
        break;
      }
    }
    woefeature[i] = woe_value;
  }
}

void OptimalBinningNumericalSBB::prepare_woebin_dataframe() {
  size_t num_bins = bins.size();
  CharacterVector bin_strings(num_bins);
  NumericVector woe_values(num_bins);
  NumericVector iv_values(num_bins);
  IntegerVector counts(num_bins);
  IntegerVector counts_pos(num_bins);
  IntegerVector counts_neg(num_bins);
  for (size_t i = 0; i < num_bins; ++i) {
    std::string bin_str = "(";
    if (bins[i].lower_bound == -std::numeric_limits<double>::infinity()) {
      bin_str += "-Inf";
    } else {
      bin_str += std::to_string(bins[i].lower_bound);
    }
    bin_str += ";";
    if (bins[i].upper_bound == std::numeric_limits<double>::infinity()) {
      bin_str += "+Inf";
    } else {
      bin_str += std::to_string(bins[i].upper_bound);
    }
    bin_str += "]";
    bin_strings[i] = bin_str;
    woe_values[i] = bins[i].woe;
    iv_values[i] = bins[i].iv;
    counts[i] = bins[i].count;
    counts_pos[i] = bins[i].count_pos;
    counts_neg[i] = bins[i].count_neg;
  }
  woebin = DataFrame::create(
    _["bin"] = bin_strings,
    _["woe"] = woe_values,
    _["iv"] = iv_values,
    _["count"] = counts,
    _["count_pos"] = counts_pos,
    _["count_neg"] = counts_neg
  );
}

// [[Rcpp::export]]
List optimal_binning_numerical_sbb(IntegerVector target, NumericVector feature,
                                   int min_bins = 3, int max_bins = 5, double bin_cutoff = 0.05, int max_n_prebins = 20) {
  // Convert target and feature to std::vector
  std::vector<int> target_vec = as<std::vector<int>>(target);
  std::vector<double> feature_vec = as<std::vector<double>>(feature);

  // Create instance of OptimalBinningNumericalSBB
  OptimalBinningNumericalSBB ob(min_bins, max_bins, bin_cutoff, max_n_prebins);

  // Fit the model
  ob.fit(target_vec, feature_vec);

  // Get outputs
  std::vector<double> woefeature = ob.get_woefeature();
  DataFrame woebin = ob.get_woebin();

  return List::create(
    _["woefeature"] = woefeature,
    _["woebin"] = woebin
  );
}
