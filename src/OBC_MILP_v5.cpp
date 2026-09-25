// [[Rcpp::depends(Rcpp)]]
// [[Rcpp::plugins(cpp11)]]

#include <Rcpp.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <numeric>
#include <utility>


// Include shared headers
#include "common/optimal_binning_common.h"
#include "common/bin_structures.h"

using namespace Rcpp;
using namespace OptimalBinning;


// Constants for better readability and consistency
// Constant removed (uses shared definition)
// Bayesian smoothing parameter (adjustable prior strength)
// Constant removed (uses shared definition)

/**
 * @brief Optimal Binning for Categorical Variables using Greedy Merging
 *
 * IMPORTANT: Despite the "MILP" name, this algorithm uses greedy heuristics,
 * not true Mixed Integer Linear Programming with branch-and-bound.
 *
 * Algorithm Overview:
 * 1. Pre-binning: Create initial bins (one per category)
 * 2. Greedy merging: Iteratively merge similar bins to optimize IV
 * 3. Monotonicity enforcement: Ensure WoE increases or decreases
 * 4. Constraint satisfaction: Respect min_bins, max_bins, bin_cutoff
 *
 * Complexity: O(k² log k) where k = number of categories
 *
 * Note: For true MILP implementation, see optimization literature on
 * binning as integer programming (requires external solvers like CPLEX/Gurobi).
 */
class OBC_MILP {
private:
  // Local CategoricalBin definition removed

  
  std::vector<int> target;
  std::vector<std::string> feature;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  std::string bin_separator;
  double convergence_threshold;
  int max_iterations;
  
  std::vector<CategoricalBin> bins;
  int total_pos;
  int total_neg;
  bool converged;
  int iterations_run;
  
public:
  OBC_MILP(
    std::vector<int> target_,
    std::vector<std::string> feature_,
    int min_bins_,
    int max_bins_,
    double bin_cutoff_,
    int max_n_prebins_,
    const std::string& bin_separator_,
    double convergence_threshold_,
    int max_iterations_
  );
  
  Rcpp::List fit();
  
private:
  void validate_input();
  void initialize_bins();
  void merge_bins();
  void calculate_woe_iv(CategoricalBin& bin);
  void handle_zero_counts();
  std::string join_categories(const std::vector<std::string>& categories) const;
  double calculate_total_iv() const;
  size_t find_best_merge_candidate(size_t bin_idx) const;
  void merge_rare_categories();
  void sort_bins_by_woe();
  void reduce_prebins_by_similarity(size_t target_bins);
  
  inline double safe_log(double value) const {
    // Safe log: avoids log(0) by adding a small epsilon
    return std::log(std::max(value, EPSILON));
  }
};

OBC_MILP::OBC_MILP(
  std::vector<int> target_,
  std::vector<std::string> feature_,
  int min_bins_,
  int max_bins_,
  double bin_cutoff_,
  int max_n_prebins_,
  const std::string& bin_separator_,
  double convergence_threshold_,
  int max_iterations_
) : target(std::move(target_)), feature(std::move(feature_)),
min_bins(min_bins_), max_bins(max_bins_), bin_cutoff(bin_cutoff_),
max_n_prebins(max_n_prebins_), bin_separator(bin_separator_),
convergence_threshold(convergence_threshold_), max_iterations(max_iterations_),
total_pos(0), total_neg(0), converged(false), iterations_run(0) {}

void OBC_MILP::validate_input() {
  if (target.size() != feature.size()) {
    throw std::invalid_argument("Length of target and feature vectors must be the same.");
  }
  if (target.empty() || feature.empty()) {
    throw std::invalid_argument("Target and feature vectors must not be empty.");
  }
  if (min_bins < 2) {
    throw std::invalid_argument("min_bins must be at least 2.");
  }
  if (max_bins < min_bins) {
    throw std::invalid_argument("max_bins must be greater or equal to min_bins.");
  }
  if (bin_cutoff < 0.0 || bin_cutoff > 1.0) {
    throw std::invalid_argument("bin_cutoff must be between 0 and 1.");
  }
  if (convergence_threshold <= 0.0) {
    throw std::invalid_argument("convergence_threshold must be positive.");
  }
  if (max_iterations <= 0) {
    throw std::invalid_argument("max_iterations must be positive.");
  }
  
  // Check for empty strings in feature
  if (std::any_of(feature.begin(), feature.end(), [](const std::string& s) { 
    return s.empty(); 
  })) {
    throw std::invalid_argument("Feature cannot contain empty strings. Consider preprocessing your data.");
  }
  
  // Check for binary target
  bool has_zero = false;
  bool has_one = false;
  
  for (int val : target) {
    if (val == 0) has_zero = true;
    else if (val == 1) has_one = true;
    else throw std::invalid_argument("Target must contain only 0 and 1.");
    // No early exit: a 2 after the first 0 and 1 used to be counted as
    // count_pos += 2, count_neg += -1.
  }
  
  if (!has_zero || !has_one) {
    throw std::invalid_argument("Target must contain both 0 and 1 values.");
  }
}

void OBC_MILP::handle_zero_counts() {
  if (total_pos == 0 || total_neg == 0) {
    throw std::runtime_error("Target variable must have at least one positive and one negative case.");
  }
  
  // Check for extremely imbalanced datasets
  if (total_pos < 5 || total_neg < 5) {
    Rcpp::warning("Dataset has fewer than 5 samples in one class. Results may be unstable.");
  }
}

void OBC_MILP::initialize_bins() {
  std::unordered_map<std::string, CategoricalBin> bin_map;
  bin_map.reserve(std::min(feature.size() / 4, static_cast<size_t>(1024)));
  
  total_pos = 0;
  total_neg = 0;
  
  for (size_t i = 0; i < target.size(); ++i) {
    const std::string& cat = feature[i];
    int tar = target[i];
    
    // One lookup per row, and the category name is stored once per bin: it
    // used to be appended for every observation (n strings in total, all
    // copied again on every merge and only de-duplicated when printing).
    auto it = bin_map.find(cat);
    if (it == bin_map.end()) {
      it = bin_map.emplace(cat, CategoricalBin()).first;
      it->second.categories.push_back(cat);
    }
    CategoricalBin& b = it->second;
    b.count++;
    b.count_pos += tar;
    b.count_neg += (1 - tar);
    
    if (tar == 1) {
      total_pos++;
    } else {
      total_neg++;
    }
  }
  
  handle_zero_counts();
  
  bins.reserve(bin_map.size());
  for (auto& kv : bin_map) {
    bins.push_back(std::move(kv.second));
  }
  
  // Calculate initial WoE and IV for each bin
  for (auto& bin : bins) {
    calculate_woe_iv(bin);
  }
}

void OBC_MILP::calculate_woe_iv(CategoricalBin& bin) {
  // Calculate Bayesian prior based on overall prevalence
  double prior_weight = BAYESIAN_PRIOR_STRENGTH;
  double overall_event_rate = static_cast<double>(total_pos) / 
    (total_pos + total_neg);
  
  double prior_pos = prior_weight * overall_event_rate;
  double prior_neg = prior_weight * (1.0 - overall_event_rate);
  
  // Apply Bayesian smoothing to proportions
  double dist_pos = static_cast<double>(bin.count_pos + prior_pos) / 
    static_cast<double>(total_pos + prior_weight);
  double dist_neg = static_cast<double>(bin.count_neg + prior_neg) / 
    static_cast<double>(total_neg + prior_weight);
  
  // Calculate WoE and IV with numerical stability
  if (dist_pos < EPSILON && dist_neg < EPSILON) {
    // Both are effectively zero
    bin.woe = 0.0;
    bin.iv = 0.0;
  } else {
    bin.woe = safe_log(dist_pos / dist_neg);
    bin.iv = (dist_pos - dist_neg) * bin.woe;
  }
  
  // Handle non-finite values
  if (!std::isfinite(bin.woe)) bin.woe = 0.0;
  if (!std::isfinite(bin.iv)) bin.iv = 0.0;
}

double OBC_MILP::calculate_total_iv() const {
  double total_iv = 0.0;
  for (const auto& bin : bins) {
    total_iv += std::fabs(bin.iv);
  }
  return total_iv;
}

size_t OBC_MILP::find_best_merge_candidate(size_t bin_idx) const {
  if (bin_idx >= bins.size()) return bin_idx;
  
  // Find best merge candidate based on event rate similarity
  double best_similarity = -1.0;
  size_t best_candidate = bin_idx;
  
  for (size_t j = 0; j < bins.size(); ++j) {
    if (j == bin_idx) continue;
    
    // Calculate event rate similarity
    double rate_diff = std::fabs(bins[bin_idx].event_rate() - bins[j].event_rate());
    double similarity = 1.0 / (rate_diff + EPSILON);
    
    if (similarity > best_similarity) {
      best_similarity = similarity;
      best_candidate = j;
    }
  }
  
  return best_candidate;
}

void OBC_MILP::merge_rare_categories() {
  double total_count = static_cast<double>(total_pos + total_neg);
  double min_count = bin_cutoff * total_count;

  // Passes are repeated until no bin is below the cutoff (or min_bins is
  // reached). A single pass could leave one: two rare bins merged into a bin
  // that was still rare, at a position whose entry had already been handled.
  bool merged_any = true;
  while (merged_any) {
    merged_any = false;

    // Identify all low-frequency bins at once
    std::vector<size_t> low_freq_bins;
    for (size_t i = 0; i < bins.size(); ++i) {
      if (bins[i].total() < min_count) {
        low_freq_bins.push_back(i);
      }
    }
    if (low_freq_bins.empty()) break;

    // Sort bins by frequency (ascending) for better merging strategy
    std::sort(low_freq_bins.begin(), low_freq_bins.end(),
              [this](size_t a, size_t b) {
                return bins[a].total() < bins[b].total();
              });

    for (size_t idx : low_freq_bins) {
      if (static_cast<int>(bins.size()) <= min_bins) {
        return; // Never go below min_bins
      }

      // Check if bin still exists and is still below cutoff
      if (idx >= bins.size() || bins[idx].total() >= min_count) {
        continue;
      }

      // Find best merge candidate based on event rate similarity
      size_t best_candidate = find_best_merge_candidate(idx);

      if (best_candidate != idx && best_candidate < bins.size()) {
        CategoricalBin merged_bin = bins[idx];
        merged_bin.merge_with(bins[best_candidate]);
        calculate_woe_iv(merged_bin);

        // Replace the bin with lower IV with the merged bin
        size_t replace_idx = (std::fabs(bins[idx].iv) <= std::fabs(bins[best_candidate].iv))
          ? idx : best_candidate;
        size_t remove_idx = (replace_idx == idx) ? best_candidate : idx;

        bins[replace_idx] = std::move(merged_bin);
        bins.erase(bins.begin() + static_cast<std::ptrdiff_t>(remove_idx));
        merged_any = true;

        // Adjust indices for remaining bins. The merged bin itself moves
        // down by one when it sat after the erased position (the previous
        // remapping pointed such entries one past it).
        const size_t merged_pos = replace_idx > remove_idx ? replace_idx - 1 : replace_idx;
        for (auto& remaining_idx : low_freq_bins) {
          if (remaining_idx == remove_idx || remaining_idx == replace_idx) {
            remaining_idx = merged_pos;
          } else if (remaining_idx > remove_idx) {
            remaining_idx--;
          }
        }
      }
    }
  }
}

void OBC_MILP::merge_bins() {
  const size_t min_bins_size = static_cast<size_t>(min_bins);
  const size_t max_bins_size = static_cast<size_t>(std::min(max_bins, static_cast<int>(bins.size())));
  const size_t max_n_prebins_size = static_cast<size_t>(std::max(max_n_prebins, min_bins));
  
  // Reduce pre-bins if necessary
  if (bins.size() > max_n_prebins_size) {
    // Sort by count (ascending) for better merging strategy
    std::sort(bins.begin(), bins.end(), [](const CategoricalBin& a, const CategoricalBin& b) {
      return (a.count_pos + a.count_neg) < (b.count_pos + b.count_neg);
    });
    
    reduce_prebins_by_similarity(std::max(max_n_prebins_size, min_bins_size));
  }
  
  // Handle rare categories
  merge_rare_categories();

  // A categorical feature has no natural order, so the monotone arrangement
  // of its bins is simply their WoE order. The previous code checked
  // monotonicity -- and merged "violating" neighbours -- in whatever order
  // the bins happened to be in: hash-map order after pre-binning, and
  // |IV| order inside the main loop, where a strongly negative and a
  // strongly positive WoE bin are neighbours. It therefore merged the two
  // extremes of the feature into one bin, collapsed almost everything down
  // to min_bins (IV 0.007 instead of 0.99 on eight well-separated
  // categories), and its IV-convergence exit could stop with more than
  // max_bins bins.
  sort_bins_by_woe();

  // Greedy reduction to max_bins: merge the WoE-adjacent pair that loses the
  // least IV, then restore the WoE order (with Bayesian smoothing a merged
  // bin's WoE is not guaranteed to lie between its parents').
  iterations_run = 0;
  while (bins.size() > max_bins_size && bins.size() > min_bins_size) {
    double best_loss = std::numeric_limits<double>::infinity();
    size_t best_idx = 0;
    for (size_t i = 0; i + 1 < bins.size(); ++i) {
      CategoricalBin merged;
      merged.count = bins[i].count + bins[i + 1].count;
      merged.count_pos = bins[i].count_pos + bins[i + 1].count_pos;
      merged.count_neg = bins[i].count_neg + bins[i + 1].count_neg;
      calculate_woe_iv(merged);
      const double loss = std::fabs(bins[i].iv) + std::fabs(bins[i + 1].iv) -
        std::fabs(merged.iv);
      if (loss < best_loss) {
        best_loss = loss;
        best_idx = i;
      }
    }
    bins[best_idx].merge_with(bins[best_idx + 1]);
    calculate_woe_iv(bins[best_idx]);
    bins.erase(bins.begin() + static_cast<std::ptrdiff_t>(best_idx) + 1);
    sort_bins_by_woe();
    ++iterations_run;
  }

  // Reaching max_bins is the stopping state; only a reduction that needed
  // more merges than max_iterations reports converged = FALSE.
  converged = (iterations_run <= max_iterations);
}

// Repeatedly merge the pair (i < j) with the most similar event rates -- the
// first such pair in index order -- until `target` bins remain. Rescanning all
// pairs after every merge was O(B^2) per merge, O(B^3) overall (7 s for 2000
// categories). The similarity of a pair depends only on its two bins, so each
// bin keeps its best partner among the bins after it and, after a merge, only
// the merged bin's row and the rows that pointed at the merged or removed bin
// are rescanned. Bins keep their relative order (a merge keeps the lower
// slot), so the pair chosen is the one the full scan chose.
void OBC_MILP::reduce_prebins_by_similarity(size_t target_bins) {
  const size_t nb = bins.size();
  if (nb <= target_bins || nb < 2) return;
  const size_t NONE = std::numeric_limits<size_t>::max();

  std::vector<double> rate(nb);
  for (size_t i = 0; i < nb; ++i) rate[i] = bins[i].event_rate();
  auto similarity = [&rate](size_t i, size_t j) {
    double rate_diff = std::fabs(rate[i] - rate[j]);
    return 1.0 / (rate_diff + EPSILON);
  };

  std::vector<size_t> nxt(nb);
  for (size_t i = 0; i < nb; ++i) nxt[i] = (i + 1 < nb) ? i + 1 : NONE;
  std::vector<size_t> prv(nb);
  for (size_t i = 0; i < nb; ++i) prv[i] = (i > 0) ? i - 1 : NONE;
  std::vector<char> alive(nb, 1);
  std::vector<double> best_s(nb, -1.0);
  std::vector<size_t> best_j(nb, NONE);

  auto recompute_row = [&](size_t i) {
    best_s[i] = -1.0;
    best_j[i] = NONE;
    for (size_t j = nxt[i]; j != NONE; j = nxt[j]) {
      const double sim = similarity(i, j);
      if (sim > best_s[i]) {
        best_s[i] = sim;
        best_j[i] = j;
      }
    }
  };
  for (size_t i = 0; i < nb; ++i) recompute_row(i);

  size_t head = 0;
  size_t count = nb;
  while (count > target_bins && count >= 2) {
    double bs = -1.0;
    size_t a = NONE;
    for (size_t i = head; i != NONE; i = nxt[i]) {
      if (best_j[i] != NONE && best_s[i] > bs) {
        bs = best_s[i];
        a = i;
      }
    }
    if (a == NONE) break;
    const size_t b = best_j[a];

    bins[a].merge_with(bins[b]);
    calculate_woe_iv(bins[a]);
    rate[a] = bins[a].event_rate();
    alive[b] = 0;
    const size_t p = prv[b];
    const size_t q = nxt[b];
    if (p != NONE) nxt[p] = q;
    if (q != NONE) prv[q] = p;
    if (b == head) head = q;
    --count;

    recompute_row(a);
    for (size_t i = head; i != NONE; i = nxt[i]) {
      if (i == a) continue;
      if (best_j[i] == a || best_j[i] == b) {
        recompute_row(i);
      } else if (i < a) {
        const double sim = similarity(i, a);
        if (sim > best_s[i] || (sim == best_s[i] && a < best_j[i])) {
          best_s[i] = sim;
          best_j[i] = a;
        }
      }
    }
    Rcpp::checkUserInterrupt();
  }

  std::vector<CategoricalBin> kept;
  kept.reserve(count);
  for (size_t i = 0; i < nb; ++i) {
    if (alive[i]) kept.push_back(std::move(bins[i]));
  }
  bins = std::move(kept);
}

void OBC_MILP::sort_bins_by_woe() {
  std::stable_sort(bins.begin(), bins.end(),
                   [](const CategoricalBin& a, const CategoricalBin& b) {
                     return a.woe < b.woe;
                   });
}

std::string OBC_MILP::join_categories(const std::vector<std::string>& categories) const {
  // Efficient concatenation with uniqueness check
  if (categories.empty()) return "";
  
  std::unordered_set<std::string> unique_categories;
  std::vector<std::string> unique_vec;
  unique_vec.reserve(categories.size());
  
  for (const auto& cat : categories) {
    if (unique_categories.insert(cat).second) {
      unique_vec.push_back(cat);
    }
  }
  
  size_t total_length = 0;
  for (const auto& c : unique_vec) total_length += c.size() + bin_separator.size();
  total_length = (total_length > bin_separator.size()) ? total_length - bin_separator.size() : total_length;
  
  std::string result;
  result.reserve(total_length);
  for (size_t i = 0; i < unique_vec.size(); ++i) {
    if (i > 0) result += bin_separator;
    result += unique_vec[i];
  }
  return result;
}

Rcpp::List OBC_MILP::fit() {
  try {
    validate_input();
    initialize_bins();
    
    // If number of unique categories <= max_bins, no need for optimization
    if (bins.size() <= static_cast<size_t>(max_bins)) {
      // Already within max_bins: report the bins in WoE order (they were
      // returned in hash-map order, i.e. not monotone and platform-dependent).
      sort_bins_by_woe();
      converged = true;
      iterations_run = 0;
    } else {
      merge_bins();
    }
    
    // Prepare output
    size_t num_bins = bins.size();
    Rcpp::CharacterVector bin_names(num_bins);
    Rcpp::NumericVector bin_woe(num_bins);
    Rcpp::NumericVector bin_iv(num_bins);
    Rcpp::IntegerVector bin_count(num_bins);
    Rcpp::IntegerVector bin_count_pos(num_bins);
    Rcpp::IntegerVector bin_count_neg(num_bins);
    Rcpp::NumericVector ids(num_bins);
    
    double total_iv = 0.0;
    
    for (size_t i = 0; i < num_bins; ++i) {
      const CategoricalBin& bin = bins[i];
      bin_names[i] = join_categories(bin.categories);
      bin_woe[i] = bin.woe;
      bin_iv[i] = bin.iv;
      bin_count[i] = bin.count_pos + bin.count_neg;
      bin_count_pos[i] = bin.count_pos;
      bin_count_neg[i] = bin.count_neg;
      ids[i] = static_cast<double>(i + 1);
      
      total_iv += std::fabs(bin.iv);
    }
    
    return Rcpp::List::create(
      Rcpp::Named("id") = ids,
      Rcpp::Named("bin") = bin_names,
      Rcpp::Named("woe") = bin_woe,
      Rcpp::Named("iv") = bin_iv,
      Rcpp::Named("count") = bin_count,
      Rcpp::Named("count_pos") = bin_count_pos,
      Rcpp::Named("count_neg") = bin_count_neg,
      Rcpp::Named("total_iv") = total_iv,
      Rcpp::Named("converged") = converged,
      Rcpp::Named("iterations") = iterations_run
    );
  } catch (const std::exception& e) {
    Rcpp::stop("Error in optimal binning: %s", e.what());
  }
}


// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_milp(
   Rcpp::IntegerVector target,
   Rcpp::CharacterVector feature,
   int min_bins = 3,
   int max_bins = 5,
   double bin_cutoff = 0.05,
   int max_n_prebins = 20,
   std::string bin_separator = "%;%",
   double convergence_threshold = 1e-6,
   int max_iterations = 1000
) {
 // Preliminary validation
 if (feature.size() == 0 || target.size() == 0) {
   Rcpp::stop("Feature and target cannot be empty.");
 }
 
 if (feature.size() != target.size()) {
   Rcpp::stop("Feature and target must have the same length.");
 }
 
 // Handle NA values
 std::vector<int> target_vec;
 std::vector<std::string> feature_vec;
 
 target_vec.reserve(target.size());
 feature_vec.reserve(feature.size());
 
 int na_feature_count = 0;
 
 for (R_xlen_t i = 0; i < feature.size(); ++i) {
   // Handle NA in feature
   SEXP s = STRING_ELT(feature, i);
   if (s == NA_STRING) {
     feature_vec.emplace_back("NA");
     na_feature_count++;
   } else {
     feature_vec.emplace_back(CHAR(s));
   }
   
   // Check for NA in target
   if (IntegerVector::is_na(target[i])) {
     Rcpp::stop("Target cannot contain missing values at position %d.", i+1);
   } else {
     target_vec.push_back(target[i]);
   }
 }
 
 // Warn about NA values in feature
 if (na_feature_count > 0) {
   Rcpp::warning("%d missing values found in feature and converted to \"NA\" category.", 
                 na_feature_count);
 }
 
 OBC_MILP obcm(
     std::move(target_vec),
     std::move(feature_vec),
     min_bins,
     max_bins,
     bin_cutoff,
     max_n_prebins,
     bin_separator,
     convergence_threshold,
     max_iterations
 );
 
 return obcm.fit();
}
