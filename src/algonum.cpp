// ---------------------------------------------------------------------------------------------- //
// NUMERIC VARIABLES
// ---------------------------------------------------------------------------------------------- //

#include <Rcpp.h>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <algorithm>
#include <limits>
#include <cmath>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <queue>

// [[Rcpp::plugins(openmp)]]

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;


// Performs optimal binning of a numeric variable for Weight of Evidence (WoE) and Information Value (IV) using a Mixed-Integer Programming (MIP) approach
//
// This function processes a numeric variable, removes missing values, and creates pre-bins based on unique values. It iteratively splits bins to maximize Information Value (IV) while ensuring monotonicity. The function also calculates WoE and IV for the generated bins.
//
// @param target Integer vector representing the binary target variable, where 1 indicates a positive event (e.g., default) and 0 indicates a negative event (e.g., non-default).
// @param feature Numeric vector representing the numeric variable to be binned.
// @param min_bins (Optional) Minimum number of bins to generate. Default is 2.
// @param max_bins (Optional) Maximum number of bins to generate. Default is 7.
// @param bin_cutoff (Optional) Cutoff value that determines the frequency of values to define pre-bins. Default is 0.05.
// @param max_n_prebins (Optional) Maximum number of pre-bins to consider before final binning. Default is 20.
//
// @return A list with the following elements:
// \itemize{
//   \item \code{feature_woe}: Numeric vector with the WoE assigned to each instance of the processed numeric variable.
//   \item \code{bin}: DataFrame containing the generated bins, with the following fields:
//     \itemize{
//       \item \code{bin}: String representing the range of values for each bin.
//       \item \code{woe}: Weight of Evidence (WoE) for each bin.
//       \item \code{iv}: Information Value (IV) for each bin.
//       \item \code{count}: Total number of observations in each bin.
//       \item \code{count_pos}: Count of positive events in each bin.
//       \item \code{count_neg}: Count of negative events in each bin.
//     }
//   \item \code{woe}: Numeric vector with the WoE for each bin.
//   \item \code{iv}: Total Information Value (IV) calculated for the variable.
//   \item \code{pos}: Vector with the count of positive events in each bin.
//   \item \code{neg}: Vector with the count of negative events in each bin.
// }
//
// [[Rcpp::export]]
Rcpp::List OptimalBinningNumericMIP(Rcpp::IntegerVector target, Rcpp::NumericVector feature, 
                                    int min_bins = 2, int max_bins = 7, 
                                    double bin_cutoff = 0.05, int max_n_prebins = 20) {
  int N = target.size();
  if (feature.size() != N) {
    Rcpp::stop("Length of target and feature must be the same.");
  }
  if (min_bins > max_bins) {
    Rcpp::stop("min_bins must be less than or equal to max_bins.");
  }
  if (bin_cutoff < 0 || bin_cutoff > 1) {
    Rcpp::stop("bin_cutoff must be between 0 and 1.");
  }
  if (max_n_prebins <= 0) {
    Rcpp::stop("max_n_prebins must be greater than 0.");
  }
  
  // Remove missing values and create index mapping
  std::vector<double> feature_clean;
  std::vector<int> target_clean;
  std::vector<int> original_indices;
  for (int i = 0; i < N; ++i) {
    if (!Rcpp::NumericVector::is_na(feature[i]) && !Rcpp::IntegerVector::is_na(target[i])) {
      feature_clean.push_back(feature[i]);
      target_clean.push_back(target[i]);
      original_indices.push_back(i);
    }
  }
  
  int N_clean = feature_clean.size();
  if (N_clean == 0) {
    Rcpp::stop("No valid data after removing missing values.");
  }
  
  // Create pre-bins
  std::vector<double> sorted_values = feature_clean;
  std::sort(sorted_values.begin(), sorted_values.end());
  auto last = std::unique(sorted_values.begin(), sorted_values.end());
  sorted_values.erase(last, sorted_values.end());
  
  if (sorted_values.size() == 1) {
    Rcpp::stop("All feature values are the same. Cannot create bins.");
  }
  
  int n_prebins = std::min((int)sorted_values.size() - 1, max_n_prebins);
  std::vector<double> candidate_cutpoints;
  
  if (n_prebins > 0) {
    int step = std::max(1, (int)sorted_values.size() / (n_prebins + 1));
    for (int i = step; i < (int)sorted_values.size() - 1; i += step) {
      candidate_cutpoints.push_back(sorted_values[i]);
    }
  } else {
    candidate_cutpoints = std::vector<double>(sorted_values.begin() + 1, sorted_values.end() - 1);
  }
  
  int n_cutpoints = candidate_cutpoints.size();
  
  // Prepare data for binning
  std::vector<int> bin_indices(N_clean);
  std::vector<int> pos_counts(n_cutpoints + 1, 0);
  std::vector<int> neg_counts(n_cutpoints + 1, 0);
  
  for (int i = 0; i < N_clean; ++i) {
    double value = feature_clean[i];
    int tgt = target_clean[i];
    
    // Determine bin index
    int bin_idx = std::lower_bound(candidate_cutpoints.begin(), candidate_cutpoints.end(), value) - candidate_cutpoints.begin();
    bin_indices[i] = bin_idx;
    
    if (tgt == 1) {
      pos_counts[bin_idx]++;
    } else {
      neg_counts[bin_idx]++;
    }
  }
  
  int total_pos = std::accumulate(pos_counts.begin(), pos_counts.end(), 0);
  int total_neg = std::accumulate(neg_counts.begin(), neg_counts.end(), 0);
  
  // Define a structure for bins
  struct Bin {
    double lower_bound;
    double upper_bound;
    int pos_count;
    int neg_count;
    double event_rate;
  };
  
  // Initialize bins
  std::vector<Bin> bins(n_cutpoints + 1);
  bins[0].lower_bound = -std::numeric_limits<double>::infinity();
  for (int i = 0; i < n_cutpoints; ++i) {
    bins[i].upper_bound = candidate_cutpoints[i];
    bins[i+1].lower_bound = candidate_cutpoints[i];
  }
  bins.back().upper_bound = std::numeric_limits<double>::infinity();
  
  for (size_t i = 0; i < bins.size(); ++i) {
    bins[i].pos_count = pos_counts[i];
    bins[i].neg_count = neg_counts[i];
    bins[i].event_rate = bins[i].pos_count + bins[i].neg_count > 0 ? 
    (double)bins[i].pos_count / (bins[i].pos_count + bins[i].neg_count) : 0;
  }
  
  // Function to compute IV
  auto compute_iv = [](const std::vector<Bin>& bins, int total_pos, int total_neg) {
    double iv = 0.0;
    for (const auto& bin : bins) {
      if (bin.pos_count + bin.neg_count == 0) continue;
      double dist_pos = (double)bin.pos_count / total_pos;
      double dist_neg = (double)bin.neg_count / total_neg;
      if (dist_pos > 0 && dist_neg > 0) {
        double woe = std::log(dist_pos / dist_neg);
        iv += (dist_pos - dist_neg) * woe;
      }
    }
    return iv;
  };
  
  // Merge bins
  while (bins.size() > (size_t)min_bins) {
    double best_iv_decrease = std::numeric_limits<double>::max();
    size_t best_merge_idx = 0;
    
    double current_iv = compute_iv(bins, total_pos, total_neg);
    
    for (size_t i = 0; i < bins.size() - 1; ++i) {
      Bin merged_bin = bins[i];
      merged_bin.upper_bound = bins[i+1].upper_bound;
      merged_bin.pos_count += bins[i+1].pos_count;
      merged_bin.neg_count += bins[i+1].neg_count;
      merged_bin.event_rate = (double)merged_bin.pos_count / (merged_bin.pos_count + merged_bin.neg_count);
      
      std::vector<Bin> temp_bins = bins;
      temp_bins[i] = merged_bin;
      temp_bins.erase(temp_bins.begin() + i + 1);
      
      double new_iv = compute_iv(temp_bins, total_pos, total_neg);
      double iv_decrease = current_iv - new_iv;
      
      if (iv_decrease < best_iv_decrease) {
        best_iv_decrease = iv_decrease;
        best_merge_idx = i;
      }
    }
    
    // Merge the best pair of bins
    bins[best_merge_idx].upper_bound = bins[best_merge_idx + 1].upper_bound;
    bins[best_merge_idx].pos_count += bins[best_merge_idx + 1].pos_count;
    bins[best_merge_idx].neg_count += bins[best_merge_idx + 1].neg_count;
    bins[best_merge_idx].event_rate = (double)bins[best_merge_idx].pos_count / 
      (bins[best_merge_idx].pos_count + bins[best_merge_idx].neg_count);
    bins.erase(bins.begin() + best_merge_idx + 1);
    
    // Stop if merging would violate the bin_cutoff
    if ((double)(bins[best_merge_idx].pos_count + bins[best_merge_idx].neg_count) / N_clean < bin_cutoff) {
      break;
    }
  }
  
  // Compute WoE and IV for final bins
  std::vector<double> woe(bins.size());
  std::vector<double> iv_bin(bins.size());
  double total_iv = 0.0;
  for (size_t i = 0; i < bins.size(); ++i) {
    double dist_pos = (double)bins[i].pos_count / total_pos;
    double dist_neg = (double)bins[i].neg_count / total_neg;
    if (dist_pos > 0 && dist_neg > 0) {
      woe[i] = std::log(dist_pos / dist_neg);
      iv_bin[i] = (dist_pos - dist_neg) * woe[i];
      total_iv += iv_bin[i];
    } else {
      woe[i] = 0;
      iv_bin[i] = 0;
    }
  }
  
  // Map feature values to WoE
  Rcpp::NumericVector feature_woe(N, NA_REAL);
  for (int i = 0; i < N_clean; ++i) {
    int original_idx = original_indices[i];
    int bin_idx = bin_indices[i];
    while (bin_idx > 0 && bins[bin_idx].lower_bound > feature_clean[i]) {
      bin_idx--;
    }
    feature_woe[original_idx] = woe[bin_idx];
  }
  
  // Prepare bin output
  Rcpp::StringVector bin_names(bins.size());
  Rcpp::IntegerVector count(bins.size());
  Rcpp::IntegerVector pos(bins.size());
  Rcpp::IntegerVector neg(bins.size());
  
  for (size_t i = 0; i < bins.size(); ++i) {
    std::string lower = (bins[i].lower_bound == -std::numeric_limits<double>::infinity()) ? "[-Inf" : "[" + std::to_string(bins[i].lower_bound);
    std::string upper = (bins[i].upper_bound == std::numeric_limits<double>::infinity()) ? "+Inf]" : std::to_string(bins[i].upper_bound) + ")";
    bin_names[i] = lower + ";" + upper;
    count[i] = bins[i].pos_count + bins[i].neg_count;
    pos[i] = bins[i].pos_count;
    neg[i] = bins[i].neg_count;
  }
  
  // Create List for bins
  Rcpp::List bin_lst = Rcpp::List::create(
    Rcpp::Named("bin") = bin_names,
    Rcpp::Named("woe") = woe,
    Rcpp::Named("iv") = iv_bin,
    Rcpp::Named("count") = count,
    Rcpp::Named("count_pos") = pos,
    Rcpp::Named("count_neg") = neg
  );
  
  // Create List for woe vector feature
  Rcpp::List woe_lst = Rcpp::List::create(
    Rcpp::Named("woefeature") = feature_woe
  );
  
  // Attrib class for compatibility with data.table in memory superfast tables
  bin_lst.attr("class") = Rcpp::CharacterVector::create("data.table", "data.frame");
  woe_lst.attr("class") = Rcpp::CharacterVector::create("data.table", "data.frame");
  
  // Return output
  Rcpp::List output_list = Rcpp::List::create(
    Rcpp::Named("woefeature") = woe_lst,
    Rcpp::Named("woebin") = bin_lst
  );
  return output_list;
}

// Helper function to calculate quantiles
std::vector<double> calculate_quantiles(const std::vector<double>& data, const std::vector<double>& probs) {
  std::vector<double> sorted_data = data;
  std::sort(sorted_data.begin(), sorted_data.end());
  
  std::vector<double> result;
  for (double p : probs) {
    double h = (sorted_data.size() - 1) * p;
    int i = static_cast<int>(h);
    double v = sorted_data[i];
    if (h > i) {
      v += (h - i) * (sorted_data[i + 1] - v);
    }
    result.push_back(v);
  }
  return result;
}

// Performs optimal binning of a numeric variable for Weight of Evidence (WoE) and Information Value (IV) using the Monotonic Optimal Binning (MOB) approach
//
// This function processes a numeric variable by removing missing values and creating pre-bins based on unique values. It iteratively merges or splits bins to ensure monotonicity of event rates, with constraints on the minimum number of bad events (\code{min_bads}) and the number of pre-bins. The function also calculates WoE and IV for the generated bins.
//
// @param target Integer vector representing the binary target variable, where 1 indicates a positive event (e.g., default) and 0 indicates a negative event (e.g., non-default).
// @param feature Numeric vector representing the numeric variable to be binned.
// @param min_bins (Optional) Minimum number of bins to generate. Default is 2.
// @param max_bins (Optional) Maximum number of bins to generate. Default is 7.
// @param bin_cutoff (Optional) Cutoff value that determines the frequency of values to define pre-bins. Default is 0.05.
// @param min_bads (Optional) Minimum proportion of bad events (positive target events) that a bin must contain. Default is 0.05.
// @param max_n_prebins (Optional) Maximum number of pre-bins to consider before final binning. Default is 20.
//
// @return A list with the following elements:
// \itemize{
//   \item \code{feature_woe}: Numeric vector with the WoE assigned to each instance of the processed numeric variable.
//   \item \code{bin}: DataFrame with the generated bins, containing the following fields:
//     \itemize{
//       \item \code{bin}: String representing the range of values for each bin.
//       \item \code{woe}: Weight of Evidence (WoE) for each bin.
//       \item \code{iv}: Information Value (IV) for each bin.
//       \item \code{count}: Total number of observations in each bin.
//       \item \code{count_pos}: Count of positive events in each bin.
//       \item \code{count_neg}: Count of negative events in each bin.
//     }
//   \item \code{woe}: Numeric vector with the WoE for each bin.
//   \item \code{iv}: Total Information Value (IV) calculated for the variable.
//   \item \code{pos}: Vector with the count of positive events in each bin.
//   \item \code{neg}: Vector with the count of negative events in each bin.
// }
//
//
// [[Rcpp::export]]
Rcpp::List OptimalBinningNumericMOB(Rcpp::IntegerVector target, Rcpp::NumericVector feature,
                                    int min_bins = 2, int max_bins = 7, double bin_cutoff = 0.05,
                                    double min_bads = 0.05, int max_n_prebins = 20) {
  // Input validation
  if (target.size() != feature.size()) {
    Rcpp::stop("Length of target and feature must be the same.");
  }
  if (min_bins < 2 || max_bins < min_bins) {
    Rcpp::stop("Invalid min_bins or max_bins values.");
  }
  if (bin_cutoff <= 0 || bin_cutoff >= 1) {
    Rcpp::stop("bin_cutoff must be between 0 and 1.");
  }
  if (min_bads <= 0 || min_bads >= 1) {
    Rcpp::stop("min_bads must be between 0 and 1.");
  }
  if (max_n_prebins < min_bins) {
    Rcpp::stop("max_n_prebins must be at least equal to min_bins.");
  }
  
  int N = target.size();
  
  // Remove missing values
  std::vector<double> feature_clean;
  std::vector<int> target_clean;
  feature_clean.reserve(N);
  target_clean.reserve(N);
  for (int i = 0; i < N; ++i) {
    if (!NumericVector::is_na(feature[i]) && !IntegerVector::is_na(target[i])) {
      feature_clean.push_back(feature[i]);
      target_clean.push_back(target[i]);
    }
  }
  
  int N_clean = feature_clean.size();
  if (N_clean == 0) {
    Rcpp::stop("No valid data after removing missing values.");
  }
  
  // Create pre-bins based on quantiles
  int num_prebins = std::min(max_n_prebins, N_clean);
  std::vector<double> cutpoints;
  
  if (num_prebins > 1) {
    std::vector<double> probs;
    for (int i = 1; i < num_prebins; ++i) {
      probs.push_back(static_cast<double>(i) / num_prebins);
    }
    
    std::vector<double> prebin_edges = calculate_quantiles(feature_clean, probs);
    
    // Ensure unique edges
    std::set<double> unique_edges(prebin_edges.begin(), prebin_edges.end());
    cutpoints.assign(unique_edges.begin(), unique_edges.end());
    std::sort(cutpoints.begin(), cutpoints.end());
  }
  
  // Initialize bins
  struct Bin {
    double lower_bound;
    double upper_bound;
    int pos_count;
    int neg_count;
    double event_rate;
  };
  
  std::vector<Bin> bins;
  
  // Initial bin edges
  std::vector<double> bin_edges;
  bin_edges.push_back(-std::numeric_limits<double>::infinity());
  bin_edges.insert(bin_edges.end(), cutpoints.begin(), cutpoints.end());
  bin_edges.push_back(std::numeric_limits<double>::infinity());
  
  // Initialize bins with counts
  int num_bins = bin_edges.size() - 1;
  bins.resize(num_bins);
  for (int i = 0; i < num_bins; ++i) {
    bins[i].lower_bound = bin_edges[i];
    bins[i].upper_bound = bin_edges[i + 1];
    bins[i].pos_count = 0;
    bins[i].neg_count = 0;
  }
  
  // Assign data to bins
  for (int i = 0; i < N_clean; ++i) {
    double value = feature_clean[i];
    int tgt = target_clean[i];
    int bin_idx = std::lower_bound(bin_edges.begin(), bin_edges.end(), value) - bin_edges.begin() - 1;
    if (bin_idx >= num_bins) bin_idx = num_bins - 1;
    
    if (tgt == 1) {
      bins[bin_idx].pos_count++;
    } else {
      bins[bin_idx].neg_count++;
    }
  }
  
  // Compute event rates
  int total_pos = 0;
  int total_neg = 0;
  for (int i = 0; i < num_bins; ++i) {
    int bin_total = bins[i].pos_count + bins[i].neg_count;
    bins[i].event_rate = (bin_total > 0) ? static_cast<double>(bins[i].pos_count) / bin_total : 0.0;
    total_pos += bins[i].pos_count;
    total_neg += bins[i].neg_count;
  }
  
  // Merge bins based on size constraints
  auto merge_bins = [&](int idx1, int idx2) {
    bins[idx1].upper_bound = bins[idx2].upper_bound;
    bins[idx1].pos_count += bins[idx2].pos_count;
    bins[idx1].neg_count += bins[idx2].neg_count;
    int bin_total = bins[idx1].pos_count + bins[idx1].neg_count;
    bins[idx1].event_rate = (bin_total > 0) ? static_cast<double>(bins[idx1].pos_count) / bin_total : 0.0;
    bins.erase(bins.begin() + idx2);
  };
  
  // Ensure minimum bin size and min_bads
  bool bins_merged = true;
  while (bins_merged && bins.size() > static_cast<size_t>(min_bins)) {
    bins_merged = false;
    for (size_t i = 0; i < bins.size(); ++i) {
      int bin_total = bins[i].pos_count + bins[i].neg_count;
      double bin_size = static_cast<double>(bin_total) / N_clean;
      double bad_rate = static_cast<double>(bins[i].pos_count) / total_pos;
      if (bin_size < bin_cutoff || bad_rate < min_bads) {
        // Merge with adjacent bin
        size_t merge_idx = (i == 0) ? i + 1 : i - 1;
        if (merge_idx >= bins.size()) continue;
        merge_bins(std::min(i, merge_idx), std::max(i, merge_idx));
        bins_merged = true;
        break;
      }
    }
  }
  
  // Ensure monotonicity
  auto is_monotonic = [](const std::vector<Bin>& bins) {
    bool increasing = true, decreasing = true;
    for (size_t i = 1; i < bins.size(); ++i) {
      if (bins[i].event_rate < bins[i - 1].event_rate) increasing = false;
      if (bins[i].event_rate > bins[i - 1].event_rate) decreasing = false;
    }
    return increasing || decreasing;
  };
  
  while (!is_monotonic(bins) && bins.size() > static_cast<size_t>(min_bins)) {
    // Find the bin to merge to enforce monotonicity
    size_t merge_idx = 0;
    double min_diff = std::numeric_limits<double>::max();
    for (size_t i = 0; i < bins.size() - 1; ++i) {
      double diff = std::abs(bins[i].event_rate - bins[i + 1].event_rate);
      if (diff < min_diff) {
        min_diff = diff;
        merge_idx = i;
      }
    }
    // Merge bins[merge_idx] and bins[merge_idx + 1]
    merge_bins(merge_idx, merge_idx + 1);
  }
  
  // Ensure the number of bins is within the specified range
  while (bins.size() > static_cast<size_t>(max_bins)) {
    // Find the pair of adjacent bins with the smallest difference in event rates
    size_t merge_idx = 0;
    double min_diff = std::numeric_limits<double>::max();
    for (size_t i = 0; i < bins.size() - 1; ++i) {
      double diff = std::abs(bins[i].event_rate - bins[i + 1].event_rate);
      if (diff < min_diff) {
        min_diff = diff;
        merge_idx = i;
      }
    }
    // Merge the selected bins
    merge_bins(merge_idx, merge_idx + 1);
  }
  
  // Compute WoE and IV
  std::vector<double> woe(bins.size());
  std::vector<double> iv_bin(bins.size());
  double total_iv = 0.0;
  for (size_t i = 0; i < bins.size(); ++i) {
    double dist_pos = static_cast<double>(bins[i].pos_count) / total_pos;
    double dist_neg = static_cast<double>(bins[i].neg_count) / total_neg;
    dist_pos = std::max(dist_pos, 1e-10);
    dist_neg = std::max(dist_neg, 1e-10);
    woe[i] = std::log(dist_pos / dist_neg);
    iv_bin[i] = (dist_pos - dist_neg) * woe[i];
    total_iv += iv_bin[i];
  }
  
  // Map feature values to WoE
  NumericVector feature_woe(N);
  for (int i = 0; i < N; ++i) {
    double value = feature[i];
    if (NumericVector::is_na(value)) {
      feature_woe[i] = NA_REAL;
      continue;
    }
    int bin_idx = std::lower_bound(bin_edges.begin(), bin_edges.end(), value) - bin_edges.begin() - 1;
    if (bin_idx >= static_cast<int>(bins.size())) bin_idx = bins.size() - 1;
    feature_woe[i] = woe[bin_idx];
  }
  
  // Prepare bin output
  std::vector<std::string> bin_names(bins.size());
  std::vector<int> count(bins.size());
  std::vector<int> pos(bins.size());
  std::vector<int> neg(bins.size());
  
  for (size_t i = 0; i < bins.size(); ++i) {
    std::string lower = (bins[i].lower_bound == -std::numeric_limits<double>::infinity()) ? "[-Inf" : "[" + std::to_string(bins[i].lower_bound);
    std::string upper = (bins[i].upper_bound == std::numeric_limits<double>::infinity()) ? "+Inf]" : std::to_string(bins[i].upper_bound) + ")";
    bin_names[i] = lower + ";" + upper;
    count[i] = bins[i].pos_count + bins[i].neg_count;
    pos[i] = bins[i].pos_count;
    neg[i] = bins[i].neg_count;
  }
  
  // Create List for bins
  List bin_lst = List::create(
    Named("bin") = bin_names,
    Named("woe") = woe,
    Named("iv") = iv_bin,
    Named("count") = count,
    Named("count_pos") = pos,
    Named("count_neg") = neg
  );
  
  // Create List for woe vector feature
  List woe_lst = List::create(
    Named("woefeature") = feature_woe
  );
  
  // Attrib class for compatibility with data.table in memory superfast tables
  bin_lst.attr("class") = CharacterVector::create("data.table", "data.frame");
  woe_lst.attr("class") = CharacterVector::create("data.table", "data.frame");
  
  // Return output
  List output_list = List::create(
    Named("woefeature") = woe_lst,
    Named("woebin") = bin_lst,
    Named("total_iv") = total_iv
  );
  return output_list;
}

// Structure to represent each bin
struct Bin {
  double lower_bound;
  double upper_bound;
  int count;
  int count_pos;
  int count_neg;
};

double compute_p_value(const Bin& bin1, const Bin& bin2) {
  int total = bin1.count + bin2.count;
  int total_pos = bin1.count_pos + bin2.count_pos;
  int total_neg = bin1.count_neg + bin2.count_neg;
  
  double expected_pos1 = bin1.count * (double)total_pos / total;
  double expected_neg1 = bin1.count * (double)total_neg / total;
  double expected_pos2 = bin2.count * (double)total_pos / total;
  double expected_neg2 = bin2.count * (double)total_neg / total;
  
  double chi_square = 
    std::pow(bin1.count_pos - expected_pos1, 2) / expected_pos1 +
    std::pow(bin1.count_neg - expected_neg1, 2) / expected_neg1 +
    std::pow(bin2.count_pos - expected_pos2, 2) / expected_pos2 +
    std::pow(bin2.count_neg - expected_neg2, 2) / expected_neg2;
  
  // Approximate p-value using chi-square distribution with 1 degree of freedom
  return 1 - R::pchisq(chi_square, 1, 1, 0);
}

// Performs optimal binning of a numeric variable for Weight of Evidence (WoE) and Information Value (IV) using the ChiMerge algorithm
//
// This function processes a numeric variable by removing missing values and creating pre-bins based on unique values. It iteratively merges bins using the Chi-square test of independence, with a p-value threshold to determine whether bins should be merged. It ensures the minimum number of bad events (\code{min_bads}) is respected, while also calculating WoE and IV for the generated bins.
//
// @param target Integer vector representing the binary target variable, where 1 indicates a positive event (e.g., default) and 0 indicates a negative event (e.g., non-default).
// @param feature Numeric vector representing the numeric variable to be binned.
// @param min_bins (Optional) Minimum number of bins to generate. Default is 2.
// @param max_bins (Optional) Maximum number of bins to generate. Default is 7.
// @param pvalue_threshold (Optional) P-value threshold for the chi-square test used to determine whether to merge bins. Default is 0.05.
// @param bin_cutoff (Optional) Cutoff value that determines the frequency of values to define pre-bins. Default is 0.05.
// @param min_bads (Optional) Minimum proportion of bad events (positive target events) that a bin must contain. Default is 0.05.
// @param max_n_prebins (Optional) Maximum number of pre-bins to consider before final binning. Default is 20.
//
// @return A list with the following elements:
// \itemize{
//   \item \code{feature_woe}: Numeric vector with the WoE assigned to each instance of the processed numeric variable.
//   \item \code{bin}: DataFrame with the generated bins, containing the following fields:
//     \itemize{
//       \item \code{bin}: String representing the range of values for each bin.
//       \item \code{woe}: Weight of Evidence (WoE) for each bin.
//       \item \code{iv}: Information Value (IV) for each bin.
//       \item \code{count}: Total number of observations in each bin.
//       \item \code{count_pos}: Count of positive events in each bin.
//       \item \code{count_neg}: Count of negative events in each bin.
//     }
//   \item \code{woe}: Numeric vector with the WoE for each bin.
//   \item \code{iv}: Total Information Value (IV) calculated for the variable.
//   \item \code{pos}: Vector with the count of positive events in each bin.
//   \item \code{neg}: Vector with the count of negative events in each bin.
// }
//
//
// [[Rcpp::export]]
List OptimalBinningNumericChiMerge(IntegerVector target, NumericVector feature,
                                   int min_bins = 2, int max_bins = 7,
                                   double pvalue_threshold = 0.05, double bin_cutoff = 0.05,
                                   double min_bads = 0.05, int max_n_prebins = 20) {
  if (target.size() != feature.size()) {
    throw std::invalid_argument("Target and feature must have the same length");
  }
  
  if (min_bins < 2 || max_bins < min_bins) {
    throw std::invalid_argument("Invalid min_bins or max_bins");
  }
  
  int n = feature.size();
  int total_bads = std::accumulate(target.begin(), target.end(), 0);
  int total_goods = n - total_bads;
  
  if (total_bads == 0 || total_goods == 0) {
    throw std::invalid_argument("Target must contain both 0s and 1s");
  }
  
  double min_bads_count = min_bads * total_bads;
  double bin_cutoff_count = bin_cutoff * n;
  
  // Create indices and sort them based on feature values
  std::vector<int> indices(n);
  std::iota(indices.begin(), indices.end(), 0);
  
  std::sort(indices.begin(), indices.end(), [&](int i1, int i2) {
    return feature[i1] < feature[i2];
  });
  
  // Determine initial bins
  std::vector<double> unique_values = as<std::vector<double>>(unique(feature));
  int num_unique_values = unique_values.size();
  
  std::vector<Bin> bins;
  if (num_unique_values <= max_n_prebins) {
    // Each unique value is its own bin
    for (int i = 0; i < n; ++i) {
      int idx = indices[i];
      if (bins.empty() || feature[idx] != bins.back().upper_bound) {
        Bin bin;
        bin.lower_bound = feature[idx];
        bin.upper_bound = feature[idx];
        bin.count = 1;
        bin.count_pos = (target[idx] == 1) ? 1 : 0;
        bin.count_neg = (target[idx] == 0) ? 1 : 0;
        bins.push_back(bin);
      } else {
        bins.back().count++;
        if (target[idx] == 1) {
          bins.back().count_pos++;
        } else {
          bins.back().count_neg++;
        }
      }
    }
  } else {
    // Pre-bin into max_n_prebins bins
    int bin_size = n / max_n_prebins;
    int remainder = n % max_n_prebins;
    
    for (int i = 0; i < max_n_prebins; ++i) {
      int start = i * bin_size + std::min(i, remainder);
      int end = (i + 1) * bin_size + std::min(i + 1, remainder) - 1;
      
      Bin bin;
      bin.lower_bound = feature[indices[start]];
      bin.upper_bound = feature[indices[end]];
      bin.count = end - start + 1;
      bin.count_pos = 0;
      bin.count_neg = 0;
      
      for (int j = start; j <= end; ++j) {
        if (target[indices[j]] == 1)
          bin.count_pos++;
        else
          bin.count_neg++;
      }
      bins.push_back(bin);
    }
  }
  
  // Compute initial p-values between bins
  std::vector<double> p_values(bins.size() - 1);
#pragma omp parallel for
  for (size_t i = 0; i < bins.size() - 1; ++i) {
    p_values[i] = compute_p_value(bins[i], bins[i + 1]);
  }
  
  // Initial merging to enforce constraints
  bool merged;
  do {
    merged = false;
    for (size_t i = 0; i < bins.size(); ++i) {
      if (bins[i].count_pos < min_bads_count || bins[i].count < bin_cutoff_count) {
        if (i == 0 && bins.size() > 1) {
          // Merge with next bin
          bins[0].upper_bound = bins[1].upper_bound;
          bins[0].count += bins[1].count;
          bins[0].count_pos += bins[1].count_pos;
          bins[0].count_neg += bins[1].count_neg;
          bins.erase(bins.begin() + 1);
          p_values.erase(p_values.begin());
        } else if (i == bins.size() - 1 && bins.size() > 1) {
          // Merge with previous bin
          bins[i-1].upper_bound = bins[i].upper_bound;
          bins[i-1].count += bins[i].count;
          bins[i-1].count_pos += bins[i].count_pos;
          bins[i-1].count_neg += bins[i].count_neg;
          bins.erase(bins.begin() + i);
          p_values.erase(p_values.end() - 1);
        } else if (i > 0 && i < bins.size() - 1) {
          // Merge with the bin that has higher p-value
          if (p_values[i-1] > p_values[i]) {
            bins[i-1].upper_bound = bins[i].upper_bound;
            bins[i-1].count += bins[i].count;
            bins[i-1].count_pos += bins[i].count_pos;
            bins[i-1].count_neg += bins[i].count_neg;
            bins.erase(bins.begin() + i);
            p_values.erase(p_values.begin() + i - 1);
          } else {
            bins[i].upper_bound = bins[i+1].upper_bound;
            bins[i].count += bins[i+1].count;
            bins[i].count_pos += bins[i+1].count_pos;
            bins[i].count_neg += bins[i+1].count_neg;
            bins.erase(bins.begin() + i + 1);
            p_values.erase(p_values.begin() + i);
          }
        }
        merged = true;
        break;
      }
    }
  } while (merged);
  
  // Main merging process based on p-values
  while (bins.size() > (size_t)min_bins) {
    auto max_p_iter = std::max_element(p_values.begin(), p_values.end());
    double max_p_value = *max_p_iter;
    size_t idx = std::distance(p_values.begin(), max_p_iter);
    
    if (bins.size() <= (size_t)max_bins && max_p_value <= pvalue_threshold) {
      break;
    }
    
    // Merge bins[idx] and bins[idx+1]
    bins[idx].upper_bound = bins[idx + 1].upper_bound;
    bins[idx].count += bins[idx + 1].count;
    bins[idx].count_pos += bins[idx + 1].count_pos;
    bins[idx].count_neg += bins[idx + 1].count_neg;
    
    bins.erase(bins.begin() + idx + 1);
    p_values.erase(p_values.begin() + idx);
    
    // Update p-values
    if (idx > 0) {
      p_values[idx - 1] = compute_p_value(bins[idx - 1], bins[idx]);
    }
    if (idx < p_values.size()) {
      p_values[idx] = compute_p_value(bins[idx], bins[idx + 1]);
    }
  }
  
  // Enforce monotonicity while respecting min_bins
  std::vector<double> event_rates(bins.size());
  for (size_t i = 0; i < bins.size(); ++i) {
    event_rates[i] = (double)bins[i].count_pos / bins[i].count;
  }
  
  bool increasing = event_rates.front() < event_rates.back();
  bool monotonic = false;
  
  while (!monotonic && bins.size() > (size_t)min_bins) {
    monotonic = true;
    for (size_t i = 0; i < bins.size() - 1; ++i) {
      if ((increasing && event_rates[i] > event_rates[i + 1]) ||
          (!increasing && event_rates[i] < event_rates[i + 1])) {
        // Merge bins[i] and bins[i+1]
        bins[i].upper_bound = bins[i + 1].upper_bound;
        bins[i].count += bins[i + 1].count;
        bins[i].count_pos += bins[i + 1].count_pos;
        bins[i].count_neg += bins[i + 1].count_neg;
        
        bins.erase(bins.begin() + i + 1);
        event_rates[i] = (double)bins[i].count_pos / bins[i].count;
        event_rates.erase(event_rates.begin() + i + 1);
        
        monotonic = false;
        break;
      }
    }
  }
  
  // Compute WoE and IV
  std::vector<double> woe(bins.size());
  double iv = 0.0;
  
#pragma omp parallel for reduction(+:iv)
  for (size_t i = 0; i < bins.size(); ++i) {
    double dist_pos = std::max(1e-10, (double)bins[i].count_pos / total_bads);
    double dist_neg = std::max(1e-10, (double)bins[i].count_neg / total_goods);
    
    woe[i] = std::log(dist_pos / dist_neg);
    iv += (dist_pos - dist_neg) * woe[i];
  }
  
  // Map feature values to WoE
  std::vector<double> bin_upper_bounds(bins.size());
  for (size_t i = 0; i < bins.size(); ++i) {
    bin_upper_bounds[i] = bins[i].upper_bound;
  }
  
  NumericVector feature_woe(n);
#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    double val = feature[i];
    auto it = std::upper_bound(bin_upper_bounds.begin(), bin_upper_bounds.end(), val);
    int idx = it - bin_upper_bounds.begin();
    if (idx >= (int)bins.size()) idx = bins.size() - 1;
    feature_woe[i] = woe[idx];
  }
  
  // Prepare output
  std::vector<std::string> bin_names(bins.size());
  NumericVector bin_lower_bounds(bins.size());
  NumericVector bin_upper_bounds_output(bins.size());
  IntegerVector bin_count(bins.size());
  IntegerVector bin_count_pos(bins.size());
  IntegerVector bin_count_neg(bins.size());
  
  for (size_t i = 0; i < bins.size(); ++i) {
    bin_lower_bounds[i] = bins[i].lower_bound;
    bin_upper_bounds_output[i] = bins[i].upper_bound;
    bin_count[i] = bins[i].count;
    bin_count_pos[i] = bins[i].count_pos;
    bin_count_neg[i] = bins[i].count_neg;
    std::string lower_str = (i == 0) ? "[-Inf" : "[" + std::to_string(bin_lower_bounds[i]);
    std::string upper_str = (i == bins.size() - 1) ? "+Inf]" : std::to_string(bin_upper_bounds_output[i]) + ")";
    bin_names[i] = lower_str + ";" + upper_str;
  }
  
  DataFrame bin_df = DataFrame::create(
    Named("bin") = bin_names,
    Named("woe") = woe,
    Named("iv") = iv / bins.size(),  // IV per bin
    Named("count") = bin_count,
    Named("count_pos") = bin_count_pos,
    Named("count_neg") = bin_count_neg
  );
  
  // Create DataFrame for WoE vector feature
  DataFrame woe_df = DataFrame::create(
    Named("woefeature") = feature_woe
  );
  
  // Set class attributes for compatibility with data.table
  bin_df.attr("class") = CharacterVector::create("data.table", "data.frame");
  woe_df.attr("class") = CharacterVector::create("data.table", "data.frame");
  
  List output_list = List::create(
    Named("woefeature") = woe_df,
    Named("woebin") = bin_df,
    Named("total_iv") = iv
  );
  
  return output_list;
}

// Structure to represent an interval
struct Interval {
  int start;
  int end;
  double entropy;
  double cut_point = std::numeric_limits<double>::infinity();
  int cut_index = -1;
  std::vector<Interval> children;
};

// Function to compute entropy
double compute_entropy(const IntegerVector& target, int start, int end) {
  int count = end - start + 1;
  if (count == 0) return 0.0;
  
  int count_pos = 0;
  for (int i = start; i <= end; ++i) {
    if (target[i] == 1) ++count_pos;
  }
  int count_neg = count - count_pos;
  
  double p_pos = static_cast<double>(count_pos) / count;
  double p_neg = static_cast<double>(count_neg) / count;
  
  double entropy = 0.0;
  if (p_pos > 0.0) entropy -= p_pos * std::log2(p_pos);
  if (p_neg > 0.0) entropy -= p_neg * std::log2(p_neg);
  return entropy;
}

// Function to compute information gain
double compute_info_gain(const IntegerVector& target, int start, int end, int split) {
  double entropy_parent = compute_entropy(target, start, end);
  int count_parent = end - start + 1;
  
  int count_left = split - start + 1;
  int count_right = end - split;
  
  double entropy_left = compute_entropy(target, start, split);
  double entropy_right = compute_entropy(target, split + 1, end);
  
  double info_gain = entropy_parent - 
    ((static_cast<double>(count_left) / count_parent) * entropy_left +
    (static_cast<double>(count_right) / count_parent) * entropy_right);
  
  return info_gain;
}

// Function to find optimal split
bool find_optimal_split(const NumericVector& feature, const IntegerVector& target, int start, int end, Interval& interval) {
  if (start >= end) return false;
  
  std::vector<int> split_candidates;
  for (int i = start; i < end; ++i) {
    if (feature[i] != feature[i + 1]) {
      split_candidates.push_back(i);
    }
  }
  if (split_candidates.empty()) return false;
  
  double best_gain = -std::numeric_limits<double>::infinity();
  int best_split = -1;
  
#pragma omp parallel for reduction(max:best_gain)
  for (size_t i = 0; i < split_candidates.size(); ++i) {
    int split = split_candidates[i];
    double gain = compute_info_gain(target, start, end, split);
    if (gain > best_gain) {
#pragma omp critical
{
  if (gain > best_gain) {
    best_gain = gain;
    best_split = split;
  }
}
    }
  }
  
  if (best_split == -1) return false;
  
  // Apply MDL stopping criterion
  int count_parent = end - start + 1;
  double entropy_parent = compute_entropy(target, start, end);
  double entropy_left = compute_entropy(target, start, best_split);
  double entropy_right = compute_entropy(target, best_split + 1, end);
  
  int k = 2; // Number of classes (binary classification)
  double delta = std::log2(std::pow(3.0, k) - 2.0) - (k * entropy_parent - k * entropy_left - k * entropy_right);
  double threshold = (std::log2(count_parent - 1) + delta) / count_parent;
  
  if (best_gain > threshold) {
    interval.cut_point = (feature[best_split] + feature[best_split + 1]) / 2.0;
    interval.cut_index = best_split;
    return true;
  } else {
    return false;
  }
}

// Recursive partitioning function
void mdlp_partition(const NumericVector& feature, const IntegerVector& target, int start, int end, Interval& interval, double bin_cutoff_count, double min_bads_count) {
  int count = end - start + 1;
  if (count <= 1) return;
  
  int count_pos = 0;
  for (int i = start; i <= end; ++i) {
    if (target[i] == 1) ++count_pos;
  }
  int count_neg = count - count_pos;
  
  if (count < bin_cutoff_count || count_pos < min_bads_count) {
    return;
  }
  
  if (find_optimal_split(feature, target, start, end, interval)) {
    Interval left_child;
    left_child.start = start;
    left_child.end = interval.cut_index;
    left_child.entropy = compute_entropy(target, start, interval.cut_index);
    
    Interval right_child;
    right_child.start = interval.cut_index + 1;
    right_child.end = end;
    right_child.entropy = compute_entropy(target, interval.cut_index + 1, end);
    
    mdlp_partition(feature, target, left_child.start, left_child.end, left_child, bin_cutoff_count, min_bads_count);
    mdlp_partition(feature, target, right_child.start, right_child.end, right_child, bin_cutoff_count, min_bads_count);
    
    interval.children.push_back(left_child);
    interval.children.push_back(right_child);
  }
}

void collect_intervals(const Interval& node, std::vector<Interval>& intervals) {
  if (node.children.empty()) {
    intervals.push_back(node);
  } else {
    for (const auto& child : node.children) {
      collect_intervals(child, intervals);
    }
  }
}

// Function to compute WoE and IV
void compute_woe_iv(const IntegerVector& target, const std::vector<Interval>& intervals, NumericVector& woe_values, double& iv, IntegerVector& pos_counts, IntegerVector& neg_counts) {
  int total_bads = std::accumulate(target.begin(), target.end(), 0);
  int total_goods = target.size() - total_bads;
  
  woe_values = NumericVector(intervals.size());
  pos_counts = IntegerVector(intervals.size());
  neg_counts = IntegerVector(intervals.size());
  
  iv = 0.0;
  
  for (size_t idx = 0; idx < intervals.size(); ++idx) {
    const Interval& interval = intervals[idx];
    int count_pos = 0;
    for (int i = interval.start; i <= interval.end; ++i) {
      if (target[i] == 1) ++count_pos;
    }
    int count_neg = interval.end - interval.start + 1 - count_pos;
    
    pos_counts[idx] = count_pos;
    neg_counts[idx] = count_neg;
    
    double dist_pos = static_cast<double>(count_pos) / total_bads;
    double dist_neg = static_cast<double>(count_neg) / total_goods;
    
    if (dist_pos == 0.0) dist_pos = 1e-10;
    if (dist_neg == 0.0) dist_neg = 1e-10;
    
    woe_values[idx] = std::log(dist_pos / dist_neg);
    iv += (dist_pos - dist_neg) * woe_values[idx];
  }
}


// [[Rcpp::export]]
List OptimalBinningNumericMDLP(IntegerVector target, NumericVector feature, int min_bins = 2, int max_bins = 7, double bin_cutoff = 0.05, double min_bads = 0.05, int max_n_prebins = 20) {
  // Input validation
  // Validate target values
  int N = target.size();
  for (int i = 0; i < N; ++i) {
    if (target[i] != 0 && target[i] != 1) {
      Rcpp::stop("Target must contain only 0s and 1s.");
    }
  }
  
  if (target.size() != feature.size()) {
    stop("Target and feature must have the same length");
  }
  if (min_bins > max_bins) {
    stop("min_bins must be less than or equal to max_bins");
  }
  
  int n = feature.size();
  int total_bads = sum(target);
  
  double min_bads_count = min_bads * total_bads;
  double bin_cutoff_count = bin_cutoff * n;
  
  // Create indices and sort them based on feature values
  IntegerVector indices = seq(0, n - 1);
  std::sort(indices.begin(), indices.end(), [&](int i1, int i2) {
    return feature[i1] < feature[i2];
  });
  
  // Sort feature and target accordingly
  NumericVector sorted_feature = feature[indices];
  IntegerVector sorted_target = target[indices];
  
  // Create root interval
  Interval root;
  root.start = 0;
  root.end = n - 1;
  root.entropy = compute_entropy(sorted_target, 0, n - 1);
  
  // Perform MDLP partitioning
  mdlp_partition(sorted_feature, sorted_target, 0, n - 1, root, bin_cutoff_count, min_bads_count);
  
  // Collect the final intervals
  std::vector<Interval> intervals;
  collect_intervals(root, intervals);
  
  // Sort intervals by start index
  std::sort(intervals.begin(), intervals.end(), [](const Interval& a, const Interval& b) {
    return a.start < b.start;
  });
  
  // Adjust number of bins if necessary
  while (intervals.size() > max_bins && intervals.size() > 2) {
    // Find the pair of adjacent intervals with the smallest difference in WoE
    double min_woe_diff = std::numeric_limits<double>::infinity();
    size_t merge_index = 0;
    for (size_t i = 0; i < intervals.size() - 1; ++i) {
      double woe1 = std::log(static_cast<double>(intervals[i].entropy) / (1 - intervals[i].entropy));
      double woe2 = std::log(static_cast<double>(intervals[i+1].entropy) / (1 - intervals[i+1].entropy));
      double woe_diff = std::abs(woe1 - woe2);
      if (woe_diff < min_woe_diff) {
        min_woe_diff = woe_diff;
        merge_index = i;
      }
    }
    
    // Merge the two intervals
    intervals[merge_index].end = intervals[merge_index + 1].end;
    intervals[merge_index].entropy = compute_entropy(sorted_target, intervals[merge_index].start, intervals[merge_index].end);
    intervals.erase(intervals.begin() + merge_index + 1);
  }
  
  // Ensure minimum number of bins
  while (intervals.size() < min_bins && intervals.size() < n) {
    // Find the interval with the highest entropy to split
    size_t split_index = 0;
    double max_entropy = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < intervals.size(); ++i) {
      if (intervals[i].entropy > max_entropy) {
        max_entropy = intervals[i].entropy;
        split_index = i;
      }
    }
    
    // Split the interval
    int mid = (intervals[split_index].start + intervals[split_index].end) / 2;
    Interval new_interval;
    new_interval.start = mid + 1;
    new_interval.end = intervals[split_index].end;
    new_interval.entropy = compute_entropy(sorted_target, new_interval.start, new_interval.end);
    intervals[split_index].end = mid;
    intervals[split_index].entropy = compute_entropy(sorted_target, intervals[split_index].start, intervals[split_index].end);
    intervals.insert(intervals.begin() + split_index + 1, new_interval);
  }
  
  // Compute WoE and IV
  NumericVector woe_values;
  double iv = 0.0;
  IntegerVector pos_counts;
  IntegerVector neg_counts;
  
  compute_woe_iv(sorted_target, intervals, woe_values, iv, pos_counts, neg_counts);
  
  // Map WoE back to feature vector
  NumericVector bin_cut_points;
  for (const auto& interval : intervals) {
    if (interval.cut_point != std::numeric_limits<double>::infinity()) {
      bin_cut_points.push_back(interval.cut_point);
    }
  }
  std::sort(bin_cut_points.begin(), bin_cut_points.end());
  
  NumericVector feature_woe(n);
#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    double val = sorted_feature[i];
    auto it = std::upper_bound(bin_cut_points.begin(), bin_cut_points.end(), val);
    int idx = it - bin_cut_points.begin();
    feature_woe[i] = woe_values[idx];
  }
  
  // Map feature_woe back to original order
  NumericVector feature_woe_original(n);
  for (int i = 0; i < n; ++i) {
    feature_woe_original[indices[i]] = feature_woe[i];
  }
  
  // Prepare bin information
  CharacterVector bin_names(intervals.size());
  NumericVector bin_lower_bounds(intervals.size());
  NumericVector bin_upper_bounds(intervals.size());
  IntegerVector bin_count(intervals.size());
  IntegerVector bin_count_pos(intervals.size());
  IntegerVector bin_count_neg(intervals.size());
  
  for (size_t i = 0; i < intervals.size(); ++i) {
    const Interval& interval = intervals[i];
    bin_lower_bounds[i] = sorted_feature[interval.start];
    bin_upper_bounds[i] = sorted_feature[interval.end];
    bin_count[i] = interval.end - interval.start + 1;
    
    int count_pos = 0;
    for (int j = interval.start; j <= interval.end; ++j) {
      if (sorted_target[j] == 1)
        ++count_pos;
    }
    int count_neg = bin_count[i] - count_pos;
    bin_count_pos[i] = count_pos;
    bin_count_neg[i] = count_neg;
    
    std::stringstream ss;
    if (i == 0) {
      ss << "[-Inf;";
    } else {
      ss << "[" << bin_lower_bounds[i] << ";";
    }
    if (i == intervals.size() - 1) {
      ss << "+Inf]";
    } else {
      ss << bin_upper_bounds[i] << ")";
    }
    bin_names[i] = ss.str();
  }
  
  // Create DataFrame for bin information
  DataFrame bin_df = DataFrame::create(
    Named("bin") = bin_names,
    Named("woe") = woe_values,
    Named("iv") = rep(iv, intervals.size()),
    Named("count") = bin_count,
    Named("count_pos") = bin_count_pos,
    Named("count_neg") = bin_count_neg
  );
  
  // Create DataFrame for WoE vector feature
  DataFrame woe_df = DataFrame::create(
    Named("woefeature") = feature_woe_original
  );
  
  // Set class attributes for compatibility with data.table
  bin_df.attr("class") = CharacterVector::create("data.table", "data.frame");
  woe_df.attr("class") = CharacterVector::create("data.table", "data.frame");
  
  // Create output list
  List output_list = List::create(
    Named("woefeature") = woe_df,
    Named("woebin") = bin_df
  // Named("woe") = woe_values,
  // Named("iv") = iv,
  // Named("pos") = pos_counts,
  // Named("neg") = neg_counts
  );
  
  return output_list;
}


// Function to compute the CAIM criterion
double compute_caim(const std::vector<int>& bin_starts,
                    const std::vector<int>& bin_ends,
                    const std::vector<int>& sorted_target) {
  int r = bin_starts.size();
  double caim = 0.0;
  
  for (int i = 0; i < r; ++i) {
    int start = bin_starts[i];
    int end = bin_ends[i];
    
    int count_pos = 0;
    int count_neg = 0;
    for (int j = start; j <= end; ++j) {
      if (sorted_target[j] == 1)
        ++count_pos;
      else
        ++count_neg;
    }
    
    int M_i = count_pos + count_neg;
    int max_i = std::max(count_pos, count_neg);
    
    if (M_i > 0) {
      caim += (max_i * max_i) / static_cast<double>(M_i);
    }
  }
  
  caim /= r;
  return caim;
}

// Performs optimal binning of a numeric variable for Weight of Evidence (WoE) and Information Value (IV) using the Class-Attribute Interdependence Maximization (CAIM) criterion
//
// This function processes a numeric variable by creating pre-bins based on unique values. It iteratively merges or splits bins to maximize the CAIM criterion, ensuring monotonicity of event rates while respecting the minimum number of bad events (\code{min_bads}) per bin. It also calculates WoE and IV for the generated bins.
//
// @param target Integer vector representing the binary target variable, where 1 indicates a positive event (e.g., default) and 0 indicates a negative event (e.g., non-default).
// @param feature Numeric vector representing the numeric variable to be binned.
// @param min_bins (Optional) Minimum number of bins to generate. Default is 2.
// @param max_bins (Optional) Maximum number of bins to generate. Default is 7.
// @param bin_cutoff (Optional) Cutoff value that determines the frequency of values to define pre-bins. Default is 0.05.
// @param min_bads (Optional) Minimum proportion of bad events (positive target events) that a bin must contain. Default is 0.05.
// @param max_n_prebins (Optional) Maximum number of pre-bins to consider before final binning. Default is 20.
//
// @return A list with the following elements:
// \itemize{
//   \item \code{feature_woe}: Numeric vector with the WoE assigned to each instance of the processed numeric variable.
//   \item \code{bin}: DataFrame with the generated bins, containing the following fields:
//     \itemize{
//       \item \code{bin}: String representing the range of values for each bin.
//       \item \code{woe}: Weight of Evidence (WoE) for each bin.
//       \item \code{iv}: Information Value (IV) for each bin.
//       \item \code{count}: Total number of observations in each bin.
//       \item \code{count_pos}: Count of positive events in each bin.
//       \item \code{count_neg}: Count of negative events in each bin.
//     }
//   \item \code{woe}: Numeric vector with the WoE for each bin.
//   \item \code{iv}: Total Information Value (IV) calculated for the variable.
//   \item \code{pos}: Vector with the count of positive events in each bin.
//   \item \code{neg}: Vector with the count of negative events in each bin.
// }
//
// [[Rcpp::export]]
Rcpp::List OptimalBinningNumericCAIM(Rcpp::IntegerVector target, Rcpp::NumericVector feature,
                                     int min_bins = 2, int max_bins = 7, double bin_cutoff = 0.05,
                                     double min_bads = 0.05, int max_n_prebins = 20) {
  
  // Input validation
  if (target.size() != feature.size()) {
    stop("'target' and 'feature' must have the same length");
  }
  
  int n = feature.size();
  std::vector<int> target_vec = as<std::vector<int>>(target);
  std::vector<double> feature_vec = as<std::vector<double>>(feature);
  
  // Check for invalid values in target and feature
  int total_bads = 0;
  int total_goods = 0;
  std::vector<bool> valid_indices(n, true);
  for (int i = 0; i < n; ++i) {
    if (target_vec[i] != 0 && target_vec[i] != 1) {
      stop("'target' must contain only 0s and 1s");
    }
    if (std::isnan(feature_vec[i]) || std::isinf(feature_vec[i])) {
      valid_indices[i] = false;
    } else {
      total_bads += target_vec[i];
      total_goods += (1 - target_vec[i]);
    }
  }
  
  // Remove invalid indices
  feature_vec.erase(
    std::remove_if(feature_vec.begin(), feature_vec.end(), 
                   [&](double) { return !valid_indices[&feature_vec[0] - &*feature_vec.begin()]; }),
                   feature_vec.end()
  );
  target_vec.erase(
    std::remove_if(target_vec.begin(), target_vec.end(), 
                   [&](int) { return !valid_indices[&target_vec[0] - &*target_vec.begin()]; }),
                   target_vec.end()
  );
  n = feature_vec.size();
  
  double min_bads_count = min_bads * total_bads;
  double bin_cutoff_count = bin_cutoff * n;
  
  // Create indices and sort them based on feature values
  std::vector<int> indices(n);
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [&](int i1, int i2) {
    return feature_vec[i1] < feature_vec[i2];
  });
  
  // Sort feature and target accordingly
  std::vector<double> sorted_feature(n);
  std::vector<int> sorted_target(n);
  for (int i = 0; i < n; ++i) {
    sorted_feature[i] = feature_vec[indices[i]];
    sorted_target[i] = target_vec[indices[i]];
  }
  
  // Initialize bins
  std::vector<int> bin_starts = {0};
  std::vector<int> bin_ends = {n - 1};
  
  // Compute initial CAIM
  double current_caim = compute_caim(bin_starts, bin_ends, sorted_target);
  
  bool improvement = true;
  
  while (improvement && bin_starts.size() < static_cast<size_t>(max_bins)) {
    improvement = false;
    double best_caim = current_caim;
    int best_bin = -1;
    int best_split = -1;
    
    // For each bin, consider all possible splits
    for (size_t b = 0; b < bin_starts.size(); ++b) {
      int start = bin_starts[b];
      int end = bin_ends[b];
      
      // Potential split points are where the feature value changes
      std::vector<int> split_candidates;
      split_candidates.reserve(end - start);
      for (int i = start; i < end; ++i) {
        if (sorted_feature[i] != sorted_feature[i + 1]) {
          split_candidates.push_back(i);
        }
      }
      
      // Consider each split candidate
      for (int split : split_candidates) {
        // Check bin constraints
        int left_count = split - start + 1;
        int right_count = end - split;
        
        if (left_count < bin_cutoff_count || right_count < bin_cutoff_count)
          continue;
        
        // Count bads in each bin
        int left_bads = std::accumulate(sorted_target.begin() + start, sorted_target.begin() + split + 1, 0);
        int right_bads = std::accumulate(sorted_target.begin() + split + 1, sorted_target.begin() + end + 1, 0);
        
        if (left_bads < min_bads_count || right_bads < min_bads_count)
          continue;
        
        // Temporarily add split
        std::vector<int> temp_bin_starts = bin_starts;
        std::vector<int> temp_bin_ends = bin_ends;
        
        temp_bin_starts.erase(temp_bin_starts.begin() + b);
        temp_bin_ends.erase(temp_bin_ends.begin() + b);
        
        temp_bin_starts.insert(temp_bin_starts.begin() + b, start);
        temp_bin_starts.insert(temp_bin_starts.begin() + b + 1, split + 1);
        
        temp_bin_ends.insert(temp_bin_ends.begin() + b, split);
        temp_bin_ends.insert(temp_bin_ends.begin() + b + 1, end);
        
        // Compute CAIM
        double new_caim = compute_caim(temp_bin_starts, temp_bin_ends, sorted_target);
        
        if (new_caim > best_caim) {
          best_caim = new_caim;
          best_bin = b;
          best_split = split;
          improvement = true;
        }
      }
    }
    
    if (improvement) {
      // Accept the best split
      int start = bin_starts[best_bin];
      int end = bin_ends[best_bin];
      
      bin_starts.erase(bin_starts.begin() + best_bin);
      bin_ends.erase(bin_ends.begin() + best_bin);
      
      bin_starts.insert(bin_starts.begin() + best_bin, start);
      bin_starts.insert(bin_starts.begin() + best_bin + 1, best_split + 1);
      
      bin_ends.insert(bin_ends.begin() + best_bin, best_split);
      bin_ends.insert(bin_ends.begin() + best_bin + 1, end);
      
      current_caim = best_caim;
    }
  }
  
  // Enforce minimum number of bins by splitting bins
  while (bin_starts.size() < static_cast<size_t>(min_bins)) {
    // Find the bin with the largest count to split
    size_t max_bin_idx = 0;
    int max_bin_count = bin_ends[0] - bin_starts[0] + 1;
    for (size_t i = 1; i < bin_starts.size(); ++i) {
      int bin_count = bin_ends[i] - bin_starts[i] + 1;
      if (bin_count > max_bin_count) {
        max_bin_idx = i;
        max_bin_count = bin_count;
      }
    }
    
    int start = bin_starts[max_bin_idx];
    int end = bin_ends[max_bin_idx];
    
    // Potential split points are where the feature value changes
    std::vector<int> split_candidates;
    for (int i = start; i < end; ++i) {
      if (sorted_feature[i] != sorted_feature[i + 1]) {
        split_candidates.push_back(i);
      }
    }
    
    if (split_candidates.empty()) {
      // Cannot split further
      break;
    }
    
    // Choose the middle split candidate
    int split = split_candidates[split_candidates.size() / 2];
    
    // Check bin constraints
    int left_count = split - start + 1;
    int right_count = end - split;
    
    if (left_count < bin_cutoff_count || right_count < bin_cutoff_count) {
      // Cannot split due to bin size constraint
      break;
    }
    
    // Split the bin
    bin_starts.insert(bin_starts.begin() + max_bin_idx + 1, split + 1);
    bin_ends.insert(bin_ends.begin() + max_bin_idx, split);
  }
  
  // Compute WoE and IV
  int num_bins = bin_starts.size();
  std::vector<double> woe_values(num_bins);
  double iv = 0.0;
  std::vector<int> pos_counts(num_bins);
  std::vector<int> neg_counts(num_bins);
  std::vector<double> event_rates(num_bins);
  
  for (int i = 0; i < num_bins; ++i) {
    int start = bin_starts[i];
    int end = bin_ends[i];
    
    int count_pos = 0;
    int count_neg = 0;
    for (int j = start; j <= end; ++j) {
      if (sorted_target[j] == 1)
        ++count_pos;
      else
        ++count_neg;
    }
    
    pos_counts[i] = count_pos;
    neg_counts[i] = count_neg;
    
    double dist_pos = count_pos / static_cast<double>(total_bads);
    double dist_neg = count_neg / static_cast<double>(total_goods);
    
    if (dist_pos == 0.0) dist_pos = 1e-10;
    if (dist_neg == 0.0) dist_neg = 1e-10;
    
    woe_values[i] = std::log(dist_pos / dist_neg);
    iv += (dist_pos - dist_neg) * woe_values[i];
    
    event_rates[i] = count_pos / static_cast<double>(count_pos + count_neg);
  }
  
  // Enforce monotonicity
  bool increasing = event_rates.front() < event_rates.back();
  bool monotonic = false;
  
  while (!monotonic && num_bins > min_bins) {
    monotonic = true;
    for (int i = 0; i < num_bins - 1; ++i) {
      if ((increasing && event_rates[i] > event_rates[i + 1]) ||
          (!increasing && event_rates[i] < event_rates[i + 1])) {
        // Merge bins i and i+1
        bin_ends[i] = bin_ends[i + 1];
        bin_starts.erase(bin_starts.begin() + i + 1);
        bin_ends.erase(bin_ends.begin() + i + 1);
        
        pos_counts[i] += pos_counts[i + 1];
        neg_counts[i] += neg_counts[i + 1];
        pos_counts.erase(pos_counts.begin() + i + 1);
        neg_counts.erase(neg_counts.begin() + i + 1);
        
        event_rates[i] = pos_counts[i] / static_cast<double>(pos_counts[i] + neg_counts[i]);
        event_rates.erase(event_rates.begin() + i + 1);
        
        woe_values.erase(woe_values.begin() + i + 1);
        
        num_bins -= 1;
        monotonic = false;
        break;
      }
    }
  }
  
  // Map WoE back to feature vector
  std::vector<double> feature_woe(n);
  std::vector<double> bin_upper_bounds(num_bins);
  for (int i = 0; i < num_bins; ++i) {
    bin_upper_bounds[i] = sorted_feature[bin_ends[i]];
  }
  
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < n; ++i) {
    double val = sorted_feature[i];
    auto it = std::upper_bound(bin_upper_bounds.begin(), bin_upper_bounds.end(), val);
    int idx = it - bin_upper_bounds.begin();
    if (idx >= num_bins) idx = num_bins - 1;
    feature_woe[i] = woe_values[idx];
  }
  
  // Map feature_woe back to original order
  std::vector<double> feature_woe_original(feature.size(), std::numeric_limits<double>::quiet_NaN());
  for (int i = 0; i < n; ++i) {
    feature_woe_original[indices[i]] = feature_woe[i];
  }
  
  // Prepare bin information
  std::vector<std::string> bin_names(num_bins);
  std::vector<double> bin_lower_bounds(num_bins);
  std::vector<double> bin_upper_bounds_output(num_bins);
  std::vector<int> bin_counts(num_bins);
  
  for (int i = 0; i < num_bins; ++i) {
    bin_lower_bounds[i] = sorted_feature[bin_starts[i]];
    bin_upper_bounds_output[i] = sorted_feature[bin_ends[i]];
    bin_counts[i] = bin_ends[i] - bin_starts[i] + 1;
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << (i == 0 ? "[-Inf;" : "[") << bin_lower_bounds[i] << ";";
    oss << (i == num_bins - 1 ? "+Inf]" : std::to_string(bin_upper_bounds_output[i]) + ")");
    bin_names[i] = oss.str();
  }
  
  DataFrame bin_df = DataFrame::create(
    Named("bin") = bin_names,
    Named("woe") = woe_values,
    Named("iv") = iv,
    Named("count") = bin_counts,
    Named("count_pos") = pos_counts,
    Named("count_neg") = neg_counts
  );
  
  // Create DataFrame for woe vector feature
  DataFrame woe_df = DataFrame::create(
    Named("woefeature") = feature_woe_original
  );
  
  // Set class attributes for compatibility with data.table
  bin_df.attr("class") = CharacterVector::create("data.table", "data.frame");
  woe_df.attr("class") = CharacterVector::create("data.table", "data.frame");
  
  List output_list = List::create(
    Named("woefeature") = woe_df,
    Named("woebin") = bin_df,
    Named("total_iv") = iv
  );
  
  return output_list;
}

// Function to format doubles to six decimal places
std::string format_double(double val) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6) << val;
  return oss.str();
}

// Performs optimal binning of a numeric variable for Weight of Evidence (WoE) and Information Value (IV) using the Pool Adjacent Violators Algorithm (PAVA) to enforce monotonicity
//
// This function processes a numeric variable by removing missing values and creating pre-bins based on unique values. It then applies the Pool Adjacent Violators Algorithm (PAVA) to ensure the monotonicity of event rates, either increasing or decreasing based on the specified direction. The function respects the minimum number of bad events (\code{min_bads}) per bin and calculates WoE and IV for the generated bins.
//
// @param target Integer vector representing the binary target variable, where 1 indicates a positive event (e.g., default) and 0 indicates a negative event (e.g., non-default).
// @param feature Numeric vector representing the numeric variable to be binned.
// @param max_bins (Optional) Maximum number of bins to generate. Default is 7.
// @param bin_cutoff (Optional) Cutoff value that determines the frequency of values to define pre-bins. Default is 0.05.
// @param min_bads (Optional) Minimum proportion of bad events (positive target events) that a bin must contain. Default is 0.05.
// @param max_n_prebins (Optional) Maximum number of pre-bins to consider before final binning. Default is 20.
// @param monotonicity_direction (Optional) String that defines the monotonicity direction of event rates, either "increase" for increasing monotonicity or "decrease" for decreasing monotonicity. Default is "increase".
//
// @return A list with the following elements:
// \itemize{
//   \item \code{feature_woe}: Numeric vector with the WoE assigned to each instance of the processed numeric variable.
//   \item \code{bin}: DataFrame with the generated bins, containing the following fields:
//     \itemize{
//       \item \code{bin}: String representing the range of values for each bin.
//       \item \code{woe}: Weight of Evidence (WoE) for each bin.
//       \item \code{iv}: Information Value (IV) for each bin.
//       \item \code{count}: Total number of observations in each bin.
//       \item \code{count_pos}: Count of positive events in each bin.
//       \item \code{count_neg}: Count of negative events in each bin.
//     }
//   \item \code{woe}: Numeric vector with the WoE for each bin.
//   \item \code{iv}: Total Information Value (IV) calculated for the variable.
//   \item \code{pos}: Vector with the count of positive events in each bin.
//   \item \code{neg}: Vector with the count of negative events in each bin.
// }
//
//
// [[Rcpp::export]]
List OptimalBinningNumericPAVA(Rcpp::IntegerVector target, Rcpp::NumericVector feature, int max_bins = 7, double bin_cutoff = 0.05, double min_bads = 0.05, int max_n_prebins = 20, std::string monotonicity_direction = "increase") {
  // Ensure input vectors are of the same length
  int N = feature.size();
  if (N != target.size()) {
    stop("feature and target must be the same length.");
  }
  
  // Total counts of positives and negatives
  int total_pos = std::accumulate(target.begin(), target.end(), 0);
  int total_neg = N - total_pos;
  
  // Sort feature and get sorted indices
  IntegerVector indices = seq(0, N - 1);
  NumericVector feature_copy = clone(feature);
  
  std::sort(indices.begin(), indices.end(), [&](int i, int j) {
    return feature_copy[i] < feature_copy[j];
  });
  
  // Initial binning
  int bin_size = std::max(1, N / max_n_prebins);
  IntegerVector bin_indices(N);
  int current_bin = 0;
  
  for (int i = 0; i < N; ++i) {
    if (i >= (current_bin + 1) * bin_size && current_bin < max_n_prebins - 1) {
      current_bin++;
    }
    bin_indices[indices[i]] = current_bin;
  }
  
  int num_bins = current_bin + 1;
  
  // Initialize bins
  struct Bin {
    int count_pos = 0;
    int count_neg = 0;
    double event_rate = 0.0;
    int total_count = 0;
    double lower_bound = R_PosInf;
    double upper_bound = R_NegInf;
    double woe = 0.0;
  };
  
  std::vector<Bin> bins(num_bins);
  
  for (int i = 0; i < N; ++i) {
    int b = bin_indices[i];
    bins[b].count_pos += target[i];
    bins[b].count_neg += 1 - target[i];
    bins[b].total_count += 1;
    if (feature[i] < bins[b].lower_bound) bins[b].lower_bound = feature[i];
    if (feature[i] > bins[b].upper_bound) bins[b].upper_bound = feature[i];
  }
  
  for (int i = 0; i < num_bins; ++i) {
    bins[i].event_rate = (double)bins[i].count_pos / bins[i].total_count;
  }
  
  // Apply PAVA
  bool merged = true;
  while (merged) {
    merged = false;
    for (size_t i = 0; i < bins.size() - 1; ++i) {
      bool violation = false;
      if (monotonicity_direction == "increase") {
        if (bins[i].event_rate > bins[i + 1].event_rate) {
          violation = true;
        }
      } else if (monotonicity_direction == "decrease") {
        if (bins[i].event_rate < bins[i + 1].event_rate) {
          violation = true;
        }
      }
      if (violation) {
        // Merge bins i and i+1
        bins[i].count_pos += bins[i + 1].count_pos;
        bins[i].count_neg += bins[i + 1].count_neg;
        bins[i].total_count += bins[i + 1].total_count;
        bins[i].event_rate = (double)bins[i].count_pos / bins[i].total_count;
        bins[i].upper_bound = bins[i + 1].upper_bound;
        bins.erase(bins.begin() + i + 1);
        merged = true;
        break;
      }
    }
  }
  
  // Enforce bin size constraints
  for (size_t i = 0; i < bins.size(); ++i) {
    while (bins[i].total_count < N * bin_cutoff && bins.size() > 1) {
      if (i == bins.size() - 1) {
        bins[i - 1].count_pos += bins[i].count_pos;
        bins[i - 1].count_neg += bins[i].count_neg;
        bins[i - 1].total_count += bins[i].total_count;
        bins[i - 1].event_rate = (double)bins[i - 1].count_pos / bins[i - 1].total_count;
        bins[i - 1].upper_bound = bins[i].upper_bound;
        bins.erase(bins.begin() + i);
        i--;
      } else {
        bins[i].count_pos += bins[i + 1].count_pos;
        bins[i].count_neg += bins[i + 1].count_neg;
        bins[i].total_count += bins[i + 1].total_count;
        bins[i].event_rate = (double)bins[i].count_pos / bins[i].total_count;
        bins[i].upper_bound = bins[i + 1].upper_bound;
        bins.erase(bins.begin() + i + 1);
      }
    }
  }
  
  // Limit the number of bins to max_bins
  while ((int)bins.size() > max_bins) {
    // Merge the two bins with the smallest total_count
    size_t min_idx = 0;
    int min_total = bins[0].total_count;
    for (size_t i = 1; i < bins.size(); ++i) {
      if (bins[i].total_count < min_total) {
        min_total = bins[i].total_count;
        min_idx = i;
      }
    }
    if (min_idx == bins.size() - 1) {
      bins[min_idx - 1].count_pos += bins[min_idx].count_pos;
      bins[min_idx - 1].count_neg += bins[min_idx].count_neg;
      bins[min_idx - 1].total_count += bins[min_idx].total_count;
      bins[min_idx - 1].event_rate = (double)bins[min_idx - 1].count_pos / bins[min_idx - 1].total_count;
      bins[min_idx - 1].upper_bound = bins[min_idx].upper_bound;
      bins.erase(bins.begin() + min_idx);
    } else {
      bins[min_idx].count_pos += bins[min_idx + 1].count_pos;
      bins[min_idx].count_neg += bins[min_idx + 1].count_neg;
      bins[min_idx].total_count += bins[min_idx + 1].total_count;
      bins[min_idx].event_rate = (double)bins[min_idx].count_pos / bins[min_idx].total_count;
      bins[min_idx].upper_bound = bins[min_idx + 1].upper_bound;
      bins.erase(bins.begin() + min_idx + 1);
    }
  }
  
  // Compute WoE and IV
  NumericVector woe(bins.size());
  double iv = 0.0;
  
  for (size_t i = 0; i < bins.size(); ++i) {
    double dist_pos = bins[i].count_pos / (double)total_pos;
    double dist_neg = bins[i].count_neg / (double)total_neg;
    
    // Avoid division by zero
    if (dist_pos == 0) dist_pos = 0.0001;
    if (dist_neg == 0) dist_neg = 0.0001;
    
    bins[i].woe = log(dist_pos / dist_neg);
    woe[i] = bins[i].woe;
    iv += (dist_pos - dist_neg) * bins[i].woe;
  }
  
  // Map WoE back to feature values
  NumericVector feature_woe(N);
  
  // Parallel processing
#pragma omp parallel for if(N > 1000)
  for (int i = 0; i < N; ++i) {
    double x = feature[i];
    for (size_t j = 0; j < bins.size(); ++j) {
      if (x >= bins[j].lower_bound && x <= bins[j].upper_bound) {
        feature_woe[i] = bins[j].woe;
        break;
      }
    }
  }
  
  // Prepare binning output
  CharacterVector bin_names(bins.size());
  IntegerVector bin_count(bins.size());
  IntegerVector bin_count_pos(bins.size());
  IntegerVector bin_count_neg(bins.size());
  
  for (size_t i = 0; i < bins.size(); ++i) {
    double lower = bins[i].lower_bound;
    double upper = bins[i].upper_bound;
    
    std::string lower_bracket, upper_bracket;
    
    if (i == 0) {
      // First bin
      lower_bracket = "[";
    } else {
      lower_bracket = "(";
    }
    
    if (i == bins.size() - 1) {
      // Last bin
      upper_bracket = "]";
    } else {
      upper_bracket = ")";
    }
    
    std::string lower_str = (lower == R_NegInf) ? "-Inf" : format_double(lower);
    std::string upper_str = (upper == R_PosInf) ? "+Inf" : format_double(upper);
    
    bin_names[i] = lower_bracket + lower_str + ";" + upper_str + upper_bracket;
    
    bin_count[i] = bins[i].total_count;
    bin_count_pos[i] = bins[i].count_pos;
    bin_count_neg[i] = bins[i].count_neg;
  }
  
  List bin_lst = List::create(
    Named("bin") = bin_names,
    Named("woe") = woe,
    Named("iv") = iv,
    Named("count") = bin_count,
    Named("count_pos") = bin_count_pos,
    Named("count_neg") = bin_count_neg
  );
  
  // Create List for woe vector feature
  List woe_lst = List::create(
    Named("woefeature") = feature_woe
  );
  
  // Attrib class for compatibility with data.table in memory superfast tables
  bin_lst.attr("class") = CharacterVector::create("data.table", "data.frame");
  woe_lst.attr("class") = CharacterVector::create("data.table", "data.frame");
  
  List output_list = List::create(
    Named("woefeature") = woe_lst,
    Named("woebin") = bin_lst
  // Named("woe") = woe,
  // Named("iv") = total_iv,
  // Named("pos") = pos,
  // Named("neg") = neg
  );
  
  // DataFrame bin_df = DataFrame::create(
  //   Named("bin") = bin_names,
  //   Named("woe") = woe,
  //   Named("count") = bin_count,
  //   Named("count_pos") = bin_count_pos,
  //   Named("count_neg") = bin_count_neg
  // );
  // 
  // List output_list = List::create(
  //   Named("woefeature") = feature_woe,
  //   Named("woebin") = bin_df
  // // Named("woe") = woe,
  // // Named("iv") = total_iv,
  // // Named("pos") = pos,
  // // Named("neg") = neg
  // );
  // 
  // output_list.attr("class") = CharacterVector::create("data.table", "data.frame");
  // 
  return output_list;
}

// Structure to hold bin information, renamed to BinTree
struct BinTree {
  double start;
  double end;
  int count;
  int count_pos;
  int count_neg;
  double woe;
  double iv;
};

// Structure for tree nodes
struct TreeNode {
  double split_point;
  TreeNode* left;
  TreeNode* right;
  BinTree bin; // Only for leaf nodes
  
  TreeNode() : split_point(0.0), left(nullptr), right(nullptr) {}
};

// Function to calculate WoE
double calculateWoE(double perc_good, double perc_bad) {
  // To handle division by zero, add a small epsilon
  const double epsilon = 1e-10;
  perc_good = perc_good < epsilon ? epsilon : perc_good;
  perc_bad = perc_bad < epsilon ? epsilon : perc_bad;
  return std::log(perc_good / perc_bad);
}

// Function to calculate IV
double calculateIV(double perc_good, double perc_bad, double woe) {
  return (perc_good - perc_bad) * woe;
}

// Function to build the decision tree
TreeNode* buildTree(const std::vector<double>& feature,
                    const std::vector<int>& target,
                    int start,
                    int end,
                    int total_pos,
                    int total_neg,
                    double lambda,
                    double min_iv_gain,
                    double min_bin_size,
                    int max_depth,
                    int current_depth,
                    std::string monotonicity_direction) {
  
  // Initialize a new tree node
  TreeNode* node = new TreeNode();
  
  // Calculate total positives and negatives in this segment
  int count_pos = 0;
  int count_neg = 0;
  for(int i = start; i < end; ++i){
    if(target[i] == 1) count_pos++;
    else count_neg++;
  }
  
  // Calculate WoE and IV for this node
  double perc_good = (double)count_pos / total_pos;
  double perc_bad = (double)count_neg / total_neg;
  double woe = calculateWoE(perc_good, perc_bad);
  double iv = calculateIV(perc_good, perc_bad, woe);
  
  // Check stopping criteria (removed iv < min_iv_gain)
  double bin_size = (double)(end - start) / feature.size();
  if(bin_size < min_bin_size || current_depth >= max_depth){
    // Assign bin information to this leaf node
    node->bin.start = feature[start];
    node->bin.end = feature[end - 1];
    node->bin.count = end - start;
    node->bin.count_pos = count_pos;
    node->bin.count_neg = count_neg;
    node->bin.woe = woe;
    node->bin.iv = iv;
    return node;
  }
  
  // Try to find the best split
  double best_split = feature[start];
  double best_gain = -std::numeric_limits<double>::infinity();
  int best_split_idx = -1;
  
  // Precompute cumulative positives and negatives for efficiency
  std::vector<int> cum_pos(end - start, 0);
  std::vector<int> cum_neg(end - start, 0);
  cum_pos[0] = (target[start] == 1) ? 1 : 0;
  cum_neg[0] = (target[start] == 0) ? 1 : 0;
  for(int i = start +1; i < end; ++i){
    cum_pos[i - start] = cum_pos[i - start -1] + ((target[i] ==1) ? 1 : 0);
    cum_neg[i - start] = cum_neg[i - start -1] + ((target[i] ==0) ? 1 : 0);
  }
  
  for(int i = start + 1; i < end; ++i){
    if(feature[i] == feature[i-1]){
      continue; // Skip identical values
    }
    
    // Calculate left counts using cumulative sums
    int left_pos = cum_pos[i - start -1];
    int left_neg = cum_neg[i - start -1];
    int right_pos = count_pos - left_pos;
    int right_neg = count_neg - left_neg;
    
    // Ensure minimum bin size
    double left_bin_size = (double)(i - start) / feature.size();
    double right_bin_size = (double)(end - i) / feature.size();
    if(left_bin_size < min_bin_size || right_bin_size < min_bin_size){
      continue;
    }
    
    // Calculate WoE and IV for left and right
    double perc_good_left = (double)left_pos / total_pos;
    double perc_bad_left = (double)left_neg / total_neg;
    double woe_left = calculateWoE(perc_good_left, perc_bad_left);
    double iv_left = calculateIV(perc_good_left, perc_bad_left, woe_left);
    
    double perc_good_right = (double)right_pos / total_pos;
    double perc_bad_right = (double)right_neg / total_neg;
    double woe_right = calculateWoE(perc_good_right, perc_bad_right);
    double iv_right = calculateIV(perc_good_right, perc_bad_right, woe_right);
    
    // Use Information Value gain as the splitting criterion
    double iv_gain = iv_left + iv_right - iv;
    
    if(iv_gain > best_gain){
      best_gain = iv_gain;
      best_split = (feature[i-1] + feature[i]) / 2.0;
      best_split_idx = i;
    }
  }
  
  // If no valid split is found or gain is insufficient, make this a leaf node
  if(best_split_idx == -1 || best_gain < min_iv_gain){
    node->bin.start = feature[start];
    node->bin.end = feature[end - 1];
    node->bin.count = end - start;
    node->bin.count_pos = count_pos;
    node->bin.count_neg = count_neg;
    node->bin.woe = woe;
    node->bin.iv = iv;
    return node;
  }
  
  // Assign the best split point
  node->split_point = best_split;
  
  // Recursively build left and right subtrees
  node->left = buildTree(feature, target, start, best_split_idx, total_pos, total_neg,
                         lambda, min_iv_gain, min_bin_size, max_depth, current_depth + 1, monotonicity_direction);
  node->right = buildTree(feature, target, best_split_idx, end, total_pos, total_neg,
                          lambda, min_iv_gain, min_bin_size, max_depth, current_depth + 1, monotonicity_direction);
  
  return node;
}

// Function to collect bins from the tree
void collectBins(TreeNode* node, std::vector<BinTree>& bins){
  if(node->left == nullptr && node->right == nullptr){
    bins.push_back(node->bin);
  }
  else{
    if(node->left != nullptr) collectBins(node->left, bins);
    if(node->right != nullptr) collectBins(node->right, bins);
  }
}

// Function to enforce monotonicity
void enforceMonotonicity(std::vector<BinTree>& bins, std::string direction){
  bool is_monotonic = false;
  while(!is_monotonic){
    is_monotonic = true;
    for(int i = 0; i < bins.size() - 1; ++i){
      if(direction == "increase" && bins[i].woe > bins[i+1].woe){
        // Merge bins i and i+1
        bins[i].end = bins[i+1].end;
        bins[i].count += bins[i+1].count;
        bins[i].count_pos += bins[i+1].count_pos;
        bins[i].count_neg += bins[i+1].count_neg;
        bins[i].woe = calculateWoE((double)bins[i].count_pos / bins[i].count,
                                   (double)bins[i].count_neg / bins[i].count);
        bins[i].iv = calculateIV((double)bins[i].count_pos / bins[i].count,
                                 (double)bins[i].count_neg / bins[i].count,
                                 bins[i].woe);
        bins.erase(bins.begin() + i + 1);
        is_monotonic = false;
        break;
      }
      else if(direction == "decrease" && bins[i].woe < bins[i+1].woe){
        // Merge bins i and i+1
        bins[i].end = bins[i+1].end;
        bins[i].count += bins[i+1].count;
        bins[i].count_pos += bins[i+1].count_pos;
        bins[i].count_neg += bins[i+1].count_neg;
        bins[i].woe = calculateWoE((double)bins[i].count_pos / bins[i].count,
                                   (double)bins[i].count_neg / bins[i].count);
        bins[i].iv = calculateIV((double)bins[i].count_pos / bins[i].count,
                                 (double)bins[i].count_neg / bins[i].count,
                                 bins[i].woe);
        bins.erase(bins.begin() + i + 1);
        is_monotonic = false;
        break;
      }
    }
  }
}

// Function to prune bins based on IV
void pruneBins(std::vector<BinTree>& bins, double min_iv_gain){
  bool pruned = false;
  do{
    pruned = false;
    double max_gain = -std::numeric_limits<double>::infinity();
    int merge_idx = -1;
    
    for(int i = 0; i < bins.size() - 1; ++i){
      // Calculate combined IV if bins i and i+1 are merged
      int combined_count = bins[i].count + bins[i+1].count;
      int combined_pos = bins[i].count_pos + bins[i+1].count_pos;
      int combined_neg = bins[i].count_neg + bins[i+1].count_neg;
      double perc_good = (double)combined_pos / combined_count;
      double perc_bad = (double)combined_neg / combined_count;
      double woe = calculateWoE(perc_good, perc_bad);
      double iv = calculateIV(perc_good, perc_bad, woe);
      double gain = iv - (bins[i].iv + bins[i+1].iv);
      
      if(gain > max_gain && gain > min_iv_gain){
        max_gain = gain;
        merge_idx = i;
      }
    }
    
    if(merge_idx != -1){
      // Merge bins at merge_idx and merge_idx +1
      bins[merge_idx].end = bins[merge_idx +1].end;
      bins[merge_idx].count += bins[merge_idx +1].count;
      bins[merge_idx].count_pos += bins[merge_idx +1].count_pos;
      bins[merge_idx].count_neg += bins[merge_idx +1].count_neg;
      bins[merge_idx].woe = calculateWoE((double)bins[merge_idx].count_pos / bins[merge_idx].count,
                                         (double)bins[merge_idx].count_neg / bins[merge_idx].count);
      bins[merge_idx].iv = calculateIV((double)bins[merge_idx].count_pos / bins[merge_idx].count,
                                       (double)bins[merge_idx].count_neg / bins[merge_idx].count,
                                       bins[merge_idx].woe);
      bins.erase(bins.begin() + merge_idx +1);
      pruned = true;
    }
  } while(pruned);
}

// Performs optimal binning of a numeric variable for Weight of Evidence (WoE) and Information Value (IV) using a decision tree-based approach
//
// This function processes a numeric variable and applies a decision tree algorithm to iteratively split the data into bins. The tree splits are based on Information Value (IV) gain, with constraints on minimum bin size and maximum depth. The function ensures monotonicity in event rates, merging bins if necessary to reduce the number of bins to the specified maximum. It also calculates WoE and IV for the generated bins.
//
// @param target Integer vector representing the binary target variable, where 1 indicates a positive event (e.g., default) and 0 indicates a negative event (e.g., non-default).
// @param feature Numeric vector representing the numeric variable to be binned.
// @param max_bins (Optional) Maximum number of bins to generate. Default is 7.
// @param lambda (Optional) Regularization parameter to penalize tree splits. Default is 0.1.
// @param min_bin_size (Optional) Minimum size a bin must have as a proportion of the total data. Default is 0.05.
// @param min_iv_gain (Optional) Minimum Information Value (IV) gain required to perform a split. Default is 0.01.
// @param max_depth (Optional) Maximum depth of the decision tree. Default is 10.
// @param monotonicity_direction (Optional) String that defines the monotonicity direction of event rates, either "increase" for increasing monotonicity or "decrease" for decreasing monotonicity. Default is "increase".
//
// @return A list with the following elements:
// \itemize{
//   \item \code{feature_woe}: Numeric vector with the WoE assigned to each instance of the processed numeric variable.
//   \item \code{bin}: DataFrame with the generated bins, containing the following fields:
//     \itemize{
//       \item \code{bin}: String representing the range of values for each bin.
//       \item \code{woe}: Weight of Evidence (WoE) for each bin.
//       \item \code{count}: Total number of observations in each bin.
//       \item \code{count_pos}: Count of positive events in each bin.
//       \item \code{count_neg}: Count of negative events in each bin.
//     }
//   \item \code{woe}: Numeric vector with the WoE for each bin.
//   \item \code{iv}: Total Information Value (IV) calculated for the variable.
//   \item \code{pos}: Vector with the count of positive events in each bin.
//   \item \code{neg}: Vector with the count of negative events in each bin.
// }
//
//
// [[Rcpp::export]]
List OptimalBinningNumericTree(IntegerVector target, NumericVector feature, int max_bins = 7, double lambda = 0.1, double min_bin_size = 0.05, double min_iv_gain = 0.01, int max_depth = 10, std::string monotonicity_direction = "increase"){
  
  // Check input lengths
  int n = feature.size();
  if(n != target.size()){
    stop("Feature and target vectors must be of the same length.");
  }
  
  // Check target is binary
  for(int i = 0; i < n; ++i){
    if(target[i] != 0 && target[i] != 1){
      stop("Target vector must be binary (0 and 1).");
    }
  }
  
  // Create a vector of original indices
  std::vector<int> original_indices(n);
  for(int i = 0; i < n; ++i){
    original_indices[i] = i;
  }
  
  // Sort the indices based on feature values
  std::sort(original_indices.begin(), original_indices.end(),
            [&](int a, int b) -> bool{
              return feature[a] < feature[b];
            });
  
  // Create sorted_feature and sorted_target based on sorted indices
  std::vector<double> sorted_feature(n);
  std::vector<int> sorted_target(n);
  for(int i = 0; i < n; ++i){
    sorted_feature[i] = feature[original_indices[i]];
    sorted_target[i] = target[original_indices[i]];
  }
  
  // Calculate total positives and negatives
  int total_pos = 0;
  int total_neg = 0;
  for(int i = 0; i < n; ++i){
    if(sorted_target[i] == 1) total_pos++;
    else total_neg++;
  }
  
  // Handle case where total_pos or total_neg is zero
  if(total_pos == 0 || total_neg == 0){
    stop("The target variable must have both positive and negative classes.");
  }
  
  // Build the initial decision tree
  TreeNode* root = buildTree(sorted_feature, sorted_target, 0, n, total_pos, total_neg,
                             lambda, min_iv_gain, min_bin_size, max_depth, 0, monotonicity_direction);
  
  // Collect bins from the tree
  std::vector<BinTree> bins;
  collectBins(root, bins);
  
  // Ensure maximum number of bins
  while(bins.size() > max_bins){
    // Find the pair of adjacent bins with the smallest IV and merge them
    double min_iv = std::numeric_limits<double>::infinity();
    int merge_idx = -1;
    for(int i = 0; i < bins.size() -1; ++i){
      if(bins[i].iv < min_iv){
        min_iv = bins[i].iv;
        merge_idx = i;
      }
    }
    if(merge_idx == -1){
      break;
    }
    // Merge bins at merge_idx and merge_idx +1
    bins[merge_idx].end = bins[merge_idx +1].end;
    bins[merge_idx].count += bins[merge_idx +1].count;
    bins[merge_idx].count_pos += bins[merge_idx +1].count_pos;
    bins[merge_idx].count_neg += bins[merge_idx +1].count_neg;
    bins[merge_idx].woe = calculateWoE((double)bins[merge_idx].count_pos / bins[merge_idx].count,
                                       (double)bins[merge_idx].count_neg / bins[merge_idx].count);
    bins[merge_idx].iv = calculateIV((double)bins[merge_idx].count_pos / bins[merge_idx].count,
                                     (double)bins[merge_idx].count_neg / bins[merge_idx].count,
                                     bins[merge_idx].woe);
    bins.erase(bins.begin() + merge_idx +1);
  }
  
  // Enforce monotonicity
  enforceMonotonicity(bins, monotonicity_direction);
  
  // Final pruning based on IV gain
  pruneBins(bins, min_iv_gain);
  
  // If after pruning, bins exceed max_bins, merge the least IV gain bins
  while(bins.size() > max_bins){
    // Find the pair of adjacent bins with the smallest IV and merge them
    double min_iv = std::numeric_limits<double>::infinity();
    int merge_idx = -1;
    for(int i = 0; i < bins.size() -1; ++i){
      if(bins[i].iv < min_iv){
        min_iv = bins[i].iv;
        merge_idx = i;
      }
    }
    if(merge_idx == -1){
      break;
    }
    // Merge bins at merge_idx and merge_idx +1
    bins[merge_idx].end = bins[merge_idx +1].end;
    bins[merge_idx].count += bins[merge_idx +1].count;
    bins[merge_idx].count_pos += bins[merge_idx +1].count_pos;
    bins[merge_idx].count_neg += bins[merge_idx +1].count_neg;
    bins[merge_idx].woe = calculateWoE((double)bins[merge_idx].count_pos / bins[merge_idx].count,
                                       (double)bins[merge_idx].count_neg / bins[merge_idx].count);
    bins[merge_idx].iv = calculateIV((double)bins[merge_idx].count_pos / bins[merge_idx].count,
                                     (double)bins[merge_idx].count_neg / bins[merge_idx].count,
                                     bins[merge_idx].woe);
    bins.erase(bins.begin() + merge_idx +1);
  }
  
  // Create WoE mapping for each feature value based on sorted order
  std::vector<double> feature_woe_sorted(n, 0.0);
  int bin_idx = 0;
  for(int i = 0; i < n; ++i){
    if(bin_idx < bins.size() -1 && sorted_feature[i] > bins[bin_idx].end){
      bin_idx++;
    }
    feature_woe_sorted[i] = bins[bin_idx].woe;
  }
  
  // Map the sorted WOE back to the original feature order
  std::vector<double> feature_woe_original(n, 0.0);
  for(int i = 0; i < n; ++i){
    feature_woe_original[original_indices[i]] = feature_woe_sorted[i];
  }
  
  // Prepare bin names with the desired interval format
  std::vector<std::string> bin_names;
  for(int i = 0; i < bins.size(); ++i){
    std::string bin_name;
    if(i == 0){
      // First bin: [-Inf; end)
      bin_name = "[-Inf;" + std::to_string(bins[i].end) + ")";
    }
    else if(i == bins.size() -1){
      // Last bin: (start;+Inf]
      bin_name = "(" + std::to_string(bins[i].start) + "; +Inf]";
    }
    else{
      // Middle bins: [start; end)
      bin_name = "[" + std::to_string(bins[i].start) + "; " + std::to_string(bins[i].end) + ")";
    }
    bin_names.push_back(bin_name);
  }
  
  // Prepare vectors for bin DataFrame
  std::vector<double> woe_values;
  std::vector<double> iv_values;
  std::vector<int> bin_counts;
  std::vector<int> pos_counts;
  std::vector<int> neg_counts;
  for(int i = 0; i < bins.size(); ++i){
    woe_values.push_back(bins[i].woe);
    iv_values.push_back(bins[i].iv);
    bin_counts.push_back(bins[i].count);
    pos_counts.push_back(bins[i].count_pos);
    neg_counts.push_back(bins[i].count_neg);
  }
  
  // Calculate total IV
  double total_iv = 0.0;
  for(int i = 0; i < bins.size(); ++i){
    total_iv += bins[i].iv;
  }
  
  // total_iv
  
  List bin_lst = List::create(
    Named("bin") = bin_names,
    Named("woe") = woe_values,
    Named("iv") = iv_values,
    Named("count") = bin_counts,
    Named("count_pos") = pos_counts,
    Named("count_neg") = neg_counts
  );
  
  // Create List for woe vector feature
  List woe_lst = List::create(
    Named("woefeature") = feature_woe_original
  );
  
  // Attrib class for compatibility with data.table in memory superfast tables
  bin_lst.attr("class") = CharacterVector::create("data.table", "data.frame");
  woe_lst.attr("class") = CharacterVector::create("data.table", "data.frame");
  
  List output_list = List::create(
    Named("woefeature") = woe_lst,
    Named("woebin") = bin_lst
  // Named("woe") = woe,
  // Named("iv") = total_iv,
  // Named("pos") = pos,
  // Named("neg") = neg
  );
  
  // // Create bin DataFrame with updated bin names
  // DataFrame bin_df = DataFrame::create(
  //   Named("bin") = bin_names,
  //   Named("woe") = woe_values,
  //   Named("count") = bin_counts,
  //   Named("count_pos") = pos_counts,
  //   Named("count_neg") = neg_counts
  // );
  // 
  // // Return as Rcpp List with the desired structure
  // List output_list = List::create(
  //   Named("woefeature") = feature_woe_original,
  //   Named("woebin") = bin_df
  //   // Named("woe") = woe_values,
  //   // Named("iv") = total_iv,
  //   // Named("pos") = pos_counts,
  //   // Named("neg") = neg_counts
  // );
  // 
  // output_list.attr("class") = CharacterVector::create("data.table", "data.frame");
  
  return output_list;
  
}

// [[Rcpp::export]]
Rcpp::List OptimalBinningCategoricalBreakList(Rcpp::IntegerVector target, 
                                              Rcpp::CharacterVector feature,
                                              Rcpp::List predefined_bins) {
  int N = target.size();
  if (feature.size() != N) {
    Rcpp::stop("Length of target and feature must be the same.");
  }
  
  // Validate target values
  for (int i = 0; i < N; ++i) {
    if (target[i] != 0 && target[i] != 1) {
      Rcpp::stop("Target must contain only 0s and 1s.");
    }
  }
  
  // Create a map to store the bin assignment for each category
  std::map<std::string, int> category_to_bin;
  int num_bins = predefined_bins.size();
  
  for (int i = 0; i < num_bins; ++i) {
    Rcpp::CharacterVector bin_categories = predefined_bins[i];
    for (int j = 0; j < bin_categories.size(); ++j) {
      std::string category = Rcpp::as<std::string>(bin_categories[j]);
      category_to_bin[category] = i;
    }
  }
  
  // Initialize bins
  struct Bin {
    Rcpp::CharacterVector categories;
    int pos_count;
    int neg_count;
    double event_rate;
  };
  std::vector<Bin> bins(num_bins);
  
  for (int i = 0; i < num_bins; ++i) {
    bins[i].categories = predefined_bins[i];
    bins[i].pos_count = 0;
    bins[i].neg_count = 0;
  }
  
  // Process data and assign to bins
  Rcpp::CharacterVector feature_processed(N);
  int unassigned_count = 0;
  
  for (int i = 0; i < N; ++i) {
    if (feature[i] == NA_STRING) {
      Rcpp::stop("NA values are not allowed in the feature vector.");
    }
    
    std::string cat = Rcpp::as<std::string>(feature[i]);
    auto it = category_to_bin.find(cat);
    
    if (it != category_to_bin.end()) {
      int bin_index = it->second;
      feature_processed[i] = Rcpp::as<Rcpp::CharacterVector>(predefined_bins[bin_index])[0];  // Use the first category as the bin name
      
      if (target[i] == 1) {
        bins[bin_index].pos_count++;
      } else {
        bins[bin_index].neg_count++;
      }
    } else {
      // Category not found in predefined bins
      feature_processed[i] = "Unassigned";
      unassigned_count++;
    }
  }
  
  if (unassigned_count > 0) {
    Rcpp::warning(std::to_string(unassigned_count) + " observations were unassigned to any predefined bin.");
  }
  
  // Calculate event rates and sort bins
  for (auto& bin : bins) {
    int total = bin.pos_count + bin.neg_count;
    bin.event_rate = total > 0 ? (double)bin.pos_count / total : 0.0;
  }
  
  std::sort(bins.begin(), bins.end(), 
            [](const Bin& a, const Bin& b) { return a.event_rate < b.event_rate; });
  
  // Compute WoE and IV
  int total_pos = 0;
  int total_neg = 0;
  for (const auto& bin : bins) {
    total_pos += bin.pos_count;
    total_neg += bin.neg_count;
  }
  
  std::vector<double> woe(num_bins);
  std::vector<double> iv_bin(num_bins);
  double total_iv = 0.0;
  
  for (int i = 0; i < num_bins; ++i) {
    double dist_pos = (double)bins[i].pos_count / total_pos;
    double dist_neg = (double)bins[i].neg_count / total_neg;
    if (dist_pos == 0) dist_pos = 1e-10;
    if (dist_neg == 0) dist_neg = 1e-10;
    woe[i] = std::log(dist_pos / dist_neg);
    iv_bin[i] = (dist_pos - dist_neg) * woe[i];
    total_iv += iv_bin[i];
  }
  
  // Map categories to WoE
  std::map<std::string, double> category_woe_map;
  for (int i = 0; i < num_bins; ++i) {
    for (int j = 0; j < bins[i].categories.size(); ++j) {
      std::string cat = Rcpp::as<std::string>(bins[i].categories[j]);
      category_woe_map[cat] = woe[i];
    }
  }
  
  Rcpp::NumericVector feature_woe(N);
  for (int i = 0; i < N; ++i) {
    std::string cat = Rcpp::as<std::string>(feature_processed[i]);
    auto it = category_woe_map.find(cat);
    if (it != category_woe_map.end()) {
      feature_woe[i] = it->second;
    } else {
      feature_woe[i] = NA_REAL;
    }
  }
  
  // Prepare bin output
  Rcpp::CharacterVector bin_names(num_bins);
  Rcpp::IntegerVector count(num_bins);
  Rcpp::IntegerVector pos(num_bins);
  Rcpp::IntegerVector neg(num_bins);
  
  for (int i = 0; i < num_bins; ++i) {
    std::string bin_name = "";
    for (size_t j = 0; j < bins[i].categories.size(); ++j) {
      if (j > 0) bin_name += "+";
      bin_name += bins[i].categories[j];
    }
    bin_names[i] = bin_name;
    count[i] = bins[i].pos_count + bins[i].neg_count;
    pos[i] = bins[i].pos_count;
    neg[i] = bins[i].neg_count;
  }
  
  // Prepare bin output
  // Rcpp::CharacterVector bin_names(num_bins);
  // Rcpp::IntegerVector count(num_bins);
  // Rcpp::IntegerVector pos(num_bins);
  // Rcpp::IntegerVector neg(num_bins);
  // 
  // for (int i = 0; i < num_bins; ++i) {
  //   bin_names[i] = Rcpp::collapse(bins[i].categories, "+");
  //   count[i] = bins[i].pos_count + bins[i].neg_count;
  //   pos[i] = bins[i].pos_count;
  //   neg[i] = bins[i].neg_count;
  // }
  
  // Create List for bins
  Rcpp::List bin_lst = Rcpp::List::create(
    Rcpp::Named("bin") = bin_names,
    Rcpp::Named("woe") = woe,
    Rcpp::Named("iv") = iv_bin,
    Rcpp::Named("count") = count,
    Rcpp::Named("count_pos") = pos,
    Rcpp::Named("count_neg") = neg);
  
  // Create List for woe vector feature
  Rcpp::List woe_lst = Rcpp::List::create(
    Rcpp::Named("woefeature") = feature_woe
  );
  
  // Attrib class for compatibility with data.table in memory superfast tables
  bin_lst.attr("class") = Rcpp::CharacterVector::create("data.table", "data.frame");
  woe_lst.attr("class") = Rcpp::CharacterVector::create("data.table", "data.frame");
  
  // Return output
  Rcpp::List output_list = Rcpp::List::create(
    Rcpp::Named("woefeature") = woe_lst,
    Rcpp::Named("woebin") = bin_lst
  );
  
  return output_list;
}

// [[Rcpp::export]]
Rcpp::List OptimalBinningNumericalBreakList(Rcpp::IntegerVector target, 
                                            Rcpp::NumericVector feature,
                                            Rcpp::NumericVector break_points) {
  int N = target.size();
  if (feature.size() != N) {
    Rcpp::stop("Length of target and feature must be the same.");
  }
  
  // Validate target values
  for (int i = 0; i < N; ++i) {
    if (target[i] != 0 && target[i] != 1) {
      Rcpp::stop("Target must contain only 0s and 1s.");
    }
  }
  
  // Ensure break_points are sorted
  std::vector<double> sorted_breaks = Rcpp::as<std::vector<double>>(break_points);
  std::sort(sorted_breaks.begin(), sorted_breaks.end());
  
  int num_bins = sorted_breaks.size() + 1;
  
  // Initialize bins
  struct Bin {
    double lower_bound;
    double upper_bound;
    int pos_count;
    int neg_count;
    double event_rate;
  };
  std::vector<Bin> bins(num_bins);
  
  bins[0].lower_bound = -std::numeric_limits<double>::infinity();
  bins[0].upper_bound = sorted_breaks[0];
  for (int i = 1; i < num_bins - 1; ++i) {
    bins[i].lower_bound = sorted_breaks[i-1];
    bins[i].upper_bound = sorted_breaks[i];
  }
  bins[num_bins-1].lower_bound = sorted_breaks.back();
  bins[num_bins-1].upper_bound = std::numeric_limits<double>::infinity();
  
  // Process data and assign to bins
  Rcpp::NumericVector feature_processed(N);
  int unassigned_count = 0;
  
  for (int i = 0; i < N; ++i) {
    if (Rcpp::NumericVector::is_na(feature[i])) {
      Rcpp::stop("NA values are not allowed in the feature vector.");
    }
    
    double value = feature[i];
    int bin_index = std::lower_bound(sorted_breaks.begin(), sorted_breaks.end(), value) - sorted_breaks.begin();
    
    feature_processed[i] = bin_index;
    
    if (target[i] == 1) {
      bins[bin_index].pos_count++;
    } else {
      bins[bin_index].neg_count++;
    }
  }
  
  // Calculate event rates
  for (auto& bin : bins) {
    int total = bin.pos_count + bin.neg_count;
    bin.event_rate = total > 0 ? (double)bin.pos_count / total : 0.0;
  }
  
  // Compute WoE and IV
  int total_pos = 0;
  int total_neg = 0;
  for (const auto& bin : bins) {
    total_pos += bin.pos_count;
    total_neg += bin.neg_count;
  }
  
  std::vector<double> woe(num_bins);
  std::vector<double> iv_bin(num_bins);
  double total_iv = 0.0;
  
  for (int i = 0; i < num_bins; ++i) {
    double dist_pos = (double)bins[i].pos_count / total_pos;
    double dist_neg = (double)bins[i].neg_count / total_neg;
    if (dist_pos == 0) dist_pos = 1e-10;
    if (dist_neg == 0) dist_neg = 1e-10;
    woe[i] = std::log(dist_pos / dist_neg);
    iv_bin[i] = (dist_pos - dist_neg) * woe[i];
    total_iv += iv_bin[i];
  }
  
  // Map feature values to WoE
  Rcpp::NumericVector feature_woe(N);
  for (int i = 0; i < N; ++i) {
    int bin_index = feature_processed[i];
    feature_woe[i] = woe[bin_index];
  }
  
  // Prepare bin output
  Rcpp::CharacterVector bin_names(num_bins);
  Rcpp::IntegerVector count(num_bins);
  Rcpp::IntegerVector pos(num_bins);
  Rcpp::IntegerVector neg(num_bins);
  
  for (int i = 0; i < num_bins; ++i) {
    std::ostringstream oss;
    if (i == 0) {
      oss << "(-Inf," << bins[i].upper_bound << "]";
    } else if (i == num_bins - 1) {
      oss << "(" << bins[i].lower_bound << ",Inf)";
    } else {
      oss << "(" << bins[i].lower_bound << "," << bins[i].upper_bound << "]";
    }
    bin_names[i] = oss.str();
    count[i] = bins[i].pos_count + bins[i].neg_count;
    pos[i] = bins[i].pos_count;
    neg[i] = bins[i].neg_count;
  }
  
  // Create List for bins
  Rcpp::List bin_lst = Rcpp::List::create(
    Rcpp::Named("bin") = bin_names,
    Rcpp::Named("woe") = woe,
    Rcpp::Named("iv") = iv_bin,
    Rcpp::Named("count") = count,
    Rcpp::Named("count_pos") = pos,
    Rcpp::Named("count_neg") = neg);
  
  // Create List for woe vector feature
  Rcpp::List woe_lst = Rcpp::List::create(
    Rcpp::Named("woefeature") = feature_woe
  );
  
  // Attrib class for compatibility with data.table in memory superfast tables
  bin_lst.attr("class") = Rcpp::CharacterVector::create("data.table", "data.frame");
  woe_lst.attr("class") = Rcpp::CharacterVector::create("data.table", "data.frame");
  
  // Return output
  Rcpp::List output_list = Rcpp::List::create(
    Rcpp::Named("woefeature") = woe_lst,
    Rcpp::Named("woebin") = bin_lst
  );
  
  return output_list;
}
