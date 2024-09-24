#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <sstream>
#include <numeric>
#include <limits>

// [[Rcpp::plugins(openmp)]]

using namespace Rcpp;

// Correlation function to compute Pearson correlation between two vectors
double compute_correlation(const std::vector<double> &x, const std::vector<double> &y) {
  if (x.size() != y.size() || x.empty()) {
    Rcpp::stop("Vectors must be of the same non-zero length for correlation.");
  }
  
  double mean_x = std::accumulate(x.begin(), x.end(), 0.0) / x.size();
  double mean_y = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
  
  double numerator = 0.0;
  double denom_x = 0.0;
  double denom_y = 0.0;
  
  for (size_t i = 0; i < x.size(); ++i) {
    double dx = x[i] - mean_x;
    double dy = y[i] - mean_y;
    numerator += dx * dy;
    denom_x += dx * dx;
    denom_y += dy * dy;
  }
  
  if (denom_x == 0 || denom_y == 0) {
    Rcpp::warning("Standard deviation is zero. Returning correlation as 0.");
    return 0.0;
  }
  
  return numerator / std::sqrt(denom_x * denom_y);
}

class OptimalBinningNumericalLPDB {
public:
  OptimalBinningNumericalLPDB(int min_bins = 3, int max_bins = 5, double bin_cutoff = 0.05, int max_n_prebins = 20)
    : min_bins(min_bins), max_bins(max_bins), bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins) {
    // Validate constraints
    if (min_bins < 2) {
      Rcpp::stop("min_bins must be at least 2.");
    }
    if (max_bins < min_bins) {
      Rcpp::stop("max_bins must be greater than or equal to min_bins.");
    }
    if (bin_cutoff < 0 || bin_cutoff > 1) {
      Rcpp::stop("bin_cutoff must be between 0 and 1.");
    }
    if (max_n_prebins < min_bins) {
      Rcpp::stop("max_n_prebins must be greater than or equal to min_bins.");
    }
  }
  
  Rcpp::List fit(Rcpp::NumericVector feature, Rcpp::IntegerVector target);
  
private:
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  
  struct Bin {
    double lower_bound;
    double upper_bound;
    int count;
    int count_pos;
    int count_neg;
    double woe;
    double iv;
  };
  
  std::vector<Bin> prebinning(Rcpp::NumericVector &feature, Rcpp::IntegerVector &target);
  void calculate_woe_iv(std::vector<Bin> &bins, int total_pos, int total_neg);
  std::vector<Bin> merge_bins(std::vector<Bin> &bins, int total_pos, int total_neg);
  bool check_monotonicity(const std::vector<Bin> &bins);
  std::string format_bin_interval(double lower, double upper, bool first = false, bool last = false);
};

Rcpp::List OptimalBinningNumericalLPDB::fit(Rcpp::NumericVector feature, Rcpp::IntegerVector target) {
  int n = feature.size();
  if (n != target.size()) {
    Rcpp::stop("feature and target must have the same length.");
  }
  
  // Ensure target is binary
  int unique_targets = Rcpp::unique(target).size();
  if (unique_targets != 2) {
    Rcpp::stop("Target must be binary (0 and 1).");
  }
  
  // Remove missing values
  Rcpp::LogicalVector not_na = !Rcpp::is_na(feature) & !Rcpp::is_na(target);
  Rcpp::NumericVector clean_feature = feature[not_na];
  Rcpp::IntegerVector clean_target = target[not_na];
  
  if (clean_feature.size() == 0) {
    Rcpp::stop("No valid observations after removing missing values.");
  }
  
  // Total positives and negatives
  int total_pos = std::accumulate(clean_target.begin(), clean_target.end(), 0);
  int total_neg = clean_target.size() - total_pos;
  
  if (total_pos == 0 || total_neg == 0) {
    Rcpp::stop("Target must have both positive and negative classes.");
  }
  
  // Pre-binning
  std::vector<Bin> bins = prebinning(clean_feature, clean_target);
  
  // Calculate WoE and IV
  calculate_woe_iv(bins, total_pos, total_neg);
  
  // Merge bins to ensure monotonicity and satisfy bin constraints
  bins = merge_bins(bins, total_pos, total_neg);
  
  // Apply WoE to the feature
  Rcpp::NumericVector woefeature(clean_feature.size());
  
#pragma omp parallel for
  for (int i = 0; i < clean_feature.size(); ++i) {
    double val = clean_feature[i];
    for (size_t j = 0; j < bins.size(); ++j) {
      // Handle the edge cases for the first and last bins
      bool lower_bound_condition = (j == 0) ? (val >= bins[j].lower_bound) : (val > bins[j].lower_bound);
      bool upper_bound_condition = (j == bins.size() - 1) ? (val <= bins[j].upper_bound) : (val <= bins[j].upper_bound);
      if (lower_bound_condition && upper_bound_condition) {
        woefeature[i] = bins[j].woe;
        break;
      }
    }
  }
  
  // Prepare output DataFrame
  Rcpp::CharacterVector bin_intervals(bins.size());
  Rcpp::NumericVector woe_values(bins.size());
  Rcpp::NumericVector iv_values(bins.size());
  Rcpp::IntegerVector counts(bins.size());
  Rcpp::IntegerVector counts_pos(bins.size());
  Rcpp::IntegerVector counts_neg(bins.size());
  
  for (size_t i = 0; i < bins.size(); ++i) {
    bin_intervals[i] = format_bin_interval(bins[i].lower_bound, bins[i].upper_bound, i == 0, i == bins.size() - 1);
    woe_values[i] = bins[i].woe;
    iv_values[i] = bins[i].iv;
    counts[i] = bins[i].count;
    counts_pos[i] = bins[i].count_pos;
    counts_neg[i] = bins[i].count_neg;
  }
  
  Rcpp::DataFrame woebin = Rcpp::DataFrame::create(
    Rcpp::Named("bin") = bin_intervals,
    Rcpp::Named("woe") = woe_values,
    Rcpp::Named("iv") = iv_values,
    Rcpp::Named("count") = counts,
    Rcpp::Named("count_pos") = counts_pos,
    Rcpp::Named("count_neg") = counts_neg
  );
  
  // Return results
  return Rcpp::List::create(
    Rcpp::Named("woefeature") = woefeature,
    Rcpp::Named("woebin") = woebin
  );
}

std::vector<OptimalBinningNumericalLPDB::Bin> OptimalBinningNumericalLPDB::prebinning(Rcpp::NumericVector &feature, Rcpp::IntegerVector &target) {
  int n = feature.size();
  
  // Sort the feature and target together
  std::vector<std::pair<double, int>> data(n);
  for (int i = 0; i < n; ++i) {
    data[i] = std::make_pair(feature[i], target[i]);
  }
  std::sort(data.begin(), data.end());
  
  // Determine initial cut points for pre-bins
  std::vector<double> cut_points;
  int bin_size = n / max_n_prebins;
  if (bin_size == 0) bin_size = 1;
  
  for (int i = bin_size; i < n; i += bin_size) {
    double val = data[i].first;
    if (cut_points.empty() || val != cut_points.back()) {
      cut_points.push_back(val);
    }
  }
  
  // Create bins
  std::vector<Bin> bins;
  double lower = data[0].first;
  size_t idx = 0;
  
  for (size_t cp = 0; cp < cut_points.size(); ++cp) {
    double upper = cut_points[cp];
    Bin bin;
    bin.lower_bound = lower;
    bin.upper_bound = upper;
    bin.count = 0;
    bin.count_pos = 0;
    bin.count_neg = 0;
    
    while (idx < data.size() && data[idx].first <= upper) {
      bin.count++;
      if (data[idx].second == 1) {
        bin.count_pos++;
      } else {
        bin.count_neg++;
      }
      idx++;
    }
    bins.push_back(bin);
    lower = upper;
  }
  
  // Last bin
  if (idx < data.size()) {
    Bin bin;
    bin.lower_bound = lower;
    bin.upper_bound = data.back().first;
    bin.count = 0;
    bin.count_pos = 0;
    bin.count_neg = 0;
    while (idx < data.size()) {
      bin.count++;
      if (data[idx].second == 1) {
        bin.count_pos++;
      } else {
        bin.count_neg++;
      }
      idx++;
    }
    bins.push_back(bin);
  }
  
  return bins;
}

void OptimalBinningNumericalLPDB::calculate_woe_iv(std::vector<Bin> &bins, int total_pos, int total_neg) {
#pragma omp parallel for
  for (size_t i = 0; i < bins.size(); ++i) {
    double dist_pos = static_cast<double>(bins[i].count_pos) / total_pos;
    double dist_neg = static_cast<double>(bins[i].count_neg) / total_neg;
    
    // Handle cases where dist_pos or dist_neg is zero to avoid log(0)
    if (dist_pos == 0) {
      bins[i].woe = std::log((dist_pos + 1e-10) / dist_neg);
    } else if (dist_neg == 0) {
      bins[i].woe = std::log(dist_pos / (dist_neg + 1e-10));
    } else {
      bins[i].woe = std::log(dist_pos / dist_neg);
    }
    bins[i].iv = (dist_pos - dist_neg) * bins[i].woe;
  }
}

std::vector<OptimalBinningNumericalLPDB::Bin> OptimalBinningNumericalLPDB::merge_bins(std::vector<Bin> &bins, int total_pos, int total_neg) {
  // Merge bins with low counts according to bin_cutoff
  int total_count = total_pos + total_neg;
  double min_count = bin_cutoff * total_count;
  
  // First, merge bins with counts less than min_count
  bool merged_any = true;
  while (merged_any) {
    merged_any = false;
    for (size_t i = 0; i < bins.size(); ++i) {
      if (bins[i].count < min_count) {
        if (bins.size() == 1) {
          break; // Cannot merge further
        }
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
        merged_any = true;
        break; // Restart after a merge
      }
    }
  }
  
  // Recalculate WoE and IV after merging
  calculate_woe_iv(bins, total_pos, total_neg);
  
  // Enforce monotonicity
  bool is_monotonic = check_monotonicity(bins);
  if (!is_monotonic) {
    // Determine the direction of monotonicity based on correlation
    std::vector<double> bin_means(bins.size());
    std::vector<double> woe_values(bins.size());
    for (size_t i = 0; i < bins.size(); ++i) {
      bin_means[i] = (bins[i].lower_bound + bins[i].upper_bound) / 2.0;
      woe_values[i] = bins[i].woe;
    }
    double corr = compute_correlation(bin_means, woe_values);
    bool monotonic_increasing = (corr >= 0);
    
    // Iteratively merge bins to enforce monotonicity
    bool monotonic = false;
    while (!monotonic && bins.size() > static_cast<size_t>(min_bins)) {
      monotonic = check_monotonicity(bins);
      if (monotonic) break;
      
      // Find the pair of adjacent bins with the smallest WoE difference
      double min_diff = std::numeric_limits<double>::max();
      size_t idx_to_merge = 0;
      for (size_t i = 1; i < bins.size(); ++i) {
        double diff = std::abs(bins[i].woe - bins[i - 1].woe);
        if (diff < min_diff) {
          min_diff = diff;
          idx_to_merge = i;
        }
      }
      // Merge bins[idx_to_merge - 1] and bins[idx_to_merge]
      bins[idx_to_merge - 1].upper_bound = bins[idx_to_merge].upper_bound;
      bins[idx_to_merge - 1].count += bins[idx_to_merge].count;
      bins[idx_to_merge - 1].count_pos += bins[idx_to_merge].count_pos;
      bins[idx_to_merge - 1].count_neg += bins[idx_to_merge].count_neg;
      bins.erase(bins.begin() + idx_to_merge);
      
      // Recalculate WoE and IV after merging
      calculate_woe_iv(bins, total_pos, total_neg);
    }
  }
  
  // Ensure number of bins is within min_bins and max_bins
  while (bins.size() > static_cast<size_t>(max_bins)) {
    // Merge the pair of bins with the smallest WoE difference
    double min_diff = std::numeric_limits<double>::max();
    size_t idx_to_merge = 0;
    for (size_t i = 1; i < bins.size(); ++i) {
      double diff = std::abs(bins[i].woe - bins[i - 1].woe);
      if (diff < min_diff) {
        min_diff = diff;
        idx_to_merge = i;
      }
    }
    // Merge bins[idx_to_merge - 1] and bins[idx_to_merge]
    bins[idx_to_merge - 1].upper_bound = bins[idx_to_merge].upper_bound;
    bins[idx_to_merge - 1].count += bins[idx_to_merge].count;
    bins[idx_to_merge - 1].count_pos += bins[idx_to_merge].count_pos;
    bins[idx_to_merge - 1].count_neg += bins[idx_to_merge].count_neg;
    bins.erase(bins.begin() + idx_to_merge);
    
    // Recalculate WoE and IV after merging
    calculate_woe_iv(bins, total_pos, total_neg);
  }
  
  // Final check to ensure minimum number of bins
  while (bins.size() < static_cast<size_t>(min_bins)) {
    // Since we cannot split bins, this should not happen if max_n_prebins >= min_bins
    Rcpp::stop("Unable to achieve the minimum number of bins with the given data and constraints.");
  }
  
  return bins;
}

bool OptimalBinningNumericalLPDB::check_monotonicity(const std::vector<Bin> &bins) {
  if (bins.empty()) return true;
  
  // Check if WoE is monotonically increasing or decreasing
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
  
  return increasing || decreasing;
}

std::string OptimalBinningNumericalLPDB::format_bin_interval(double lower, double upper, bool first, bool last) {
  std::ostringstream oss;
  oss << "(";
  if (first) {
    oss << "-Inf";
  } else {
    oss << lower;
  }
  oss << ";";
  if (last) {
    oss << "+Inf";
  } else {
    oss << upper;
  }
  oss << "]";
  return oss.str();
}

// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_lpdb(Rcpp::IntegerVector target,
                                          Rcpp::NumericVector feature,
                                          int min_bins = 3,
                                          int max_bins = 5,
                                          double bin_cutoff = 0.05,
                                          int max_n_prebins = 20) {
  OptimalBinningNumericalLPDB binning(min_bins, max_bins, bin_cutoff, max_n_prebins);
  return binning.fit(feature, target);
}
