// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>
#include <stdexcept>
#include <sstream>

using namespace Rcpp;

// Include shared headers
#include "common/optimal_binning_common.h"
#include "common/bin_structures.h"

using namespace Rcpp;
using namespace OptimalBinning;


// Classe para binning ótimo SBLP
class OBC_SBLP {
public:
  OBC_SBLP(const IntegerVector& target,
                                const CharacterVector& feature,
                                int min_bins,
                                int max_bins,
                                double bin_cutoff,
                                int max_n_prebins,
                                double convergence_threshold,
                                int max_iterations,
                                std::string bin_separator,
                                double alpha);
  
  List fit();
  
private:
  // Dados de entrada e parâmetros
  const IntegerVector& target;
  const CharacterVector& feature;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  double convergence_threshold;
  int max_iterations;
  std::string bin_separator;
  double alpha; // Laplace smoothing parameter
  
  // Estruturas internas
  std::vector<std::string> unique_categories;
  std::vector<int> count_total;
  std::vector<int> count_pos;
  std::vector<int> count_neg;
  std::vector<double> category_target_rate;
  std::vector<size_t> sorted_indices;
  
  // Cache values
  mutable int total_pos_all;
  mutable int total_neg_all;
  mutable bool cache_initialized;
  
  // Funções auxiliares
  void validate_input();
  void compute_initial_counts();
  void handle_rare_categories();
  void ensure_max_prebins();
  void sort_categories();
  std::vector<std::vector<size_t>> perform_binning();
  double calculate_bin_iv(const std::vector<size_t>& bin) const;
  bool is_monotonic(const std::vector<std::vector<size_t>>& bins) const;
  void initialize_cache() const;
  List prepare_output(const std::vector<std::vector<size_t>>& bins, bool converged, int iterations) const;
  static std::string merge_category_names(const std::vector<std::string>& categories, const std::string& separator);
};

OBC_SBLP::OBC_SBLP(
  const IntegerVector& target,
  const CharacterVector& feature,
  int min_bins,
  int max_bins,
  double bin_cutoff,
  int max_n_prebins,
  double convergence_threshold,
  int max_iterations,
  std::string bin_separator,
  double alpha)
  : target(target), feature(feature),
    min_bins(min_bins), max_bins(max_bins),
    bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins),
    convergence_threshold(convergence_threshold),
    max_iterations(max_iterations),
    bin_separator(bin_separator),
    alpha(alpha),
    total_pos_all(-1), total_neg_all(-1), cache_initialized(false) {}

// Initialize cache for frequently used values
void OBC_SBLP::initialize_cache() const {
  if (!cache_initialized) {
    total_pos_all = std::accumulate(count_pos.begin(), count_pos.end(), 0);
    total_neg_all = std::accumulate(count_neg.begin(), count_neg.end(), 0);
    cache_initialized = true;
  }
}

// Validações iniciais com mensagens de erro detalhadas
void OBC_SBLP::validate_input() {
  if (target.size() == 0 || feature.size() == 0) {
    throw std::invalid_argument("Target and feature vectors cannot be empty");
  }
  
  if (target.size() != feature.size()) {
    throw std::invalid_argument("Target and feature must have the same length (got " + 
                                std::to_string(target.size()) + " and " + 
                                std::to_string(feature.size()) + ")");
  }
  
  // Check if target has values other than 0 and 1
  for (int i = 0; i < target.size(); ++i) {
    if (target[i] != 0 && target[i] != 1 && !ISNA(target[i])) {
      throw std::invalid_argument("Target must be binary (0 or 1), found value " + 
                                  std::to_string(target[i]) + " at position " + 
                                  std::to_string(i+1));
    }
  }
  
  if (min_bins < 2) {
    throw std::invalid_argument("min_bins must be at least 2 (got " + std::to_string(min_bins) + ")");
  }
  
  if (max_bins < min_bins) {
    throw std::invalid_argument("max_bins must be greater than or equal to min_bins (got max_bins=" + 
                                std::to_string(max_bins) + ", min_bins=" + std::to_string(min_bins) + ")");
  }
  
  if (bin_cutoff <= 0 || bin_cutoff >= 1) {
    throw std::invalid_argument("bin_cutoff must be between 0 and 1 (got " + 
                                std::to_string(bin_cutoff) + ")");
  }
  
  if (max_n_prebins < min_bins) {
    throw std::invalid_argument("max_n_prebins must be at least equal to min_bins (got max_n_prebins=" + 
                                std::to_string(max_n_prebins) + ", min_bins=" + std::to_string(min_bins) + ")");
  }
  
  if (convergence_threshold <= 0) {
    throw std::invalid_argument("convergence_threshold must be positive (got " + 
                                std::to_string(convergence_threshold) + ")");
  }
  
  if (max_iterations <= 0) {
    throw std::invalid_argument("max_iterations must be positive (got " + 
                                std::to_string(max_iterations) + ")");
  }
  
  if (alpha < 0) {
    throw std::invalid_argument("alpha (smoothing parameter) must be non-negative (got " + 
                                std::to_string(alpha) + ")");
  }
}

// Cálculo inicial das contagens por categoria com tratamento para valores ausentes
void OBC_SBLP::compute_initial_counts() {
  std::unordered_map<std::string, size_t> category_indices;
  bool has_missing = false;
  
  // First check for missing values
  for (int i = 0; i < feature.size(); ++i) {
    if (CharacterVector::is_na(feature[i])) {
      has_missing = true;
      break;
    }
  }
  
  // Add special "MISSING" category if needed
  size_t missing_idx = 0;
  if (has_missing) {
    unique_categories.push_back("MISSING");
    count_total.push_back(0);
    count_pos.push_back(0);
    count_neg.push_back(0);
    missing_idx = 0;
    category_indices["MISSING"] = 0;
  }
  
  // Process all values
  for (int i = 0; i < feature.size(); ++i) {
    // Skip if target is NA
    if (ISNA(target[i])) {
      continue;
    }
    
    size_t idx;
    if (CharacterVector::is_na(feature[i])) {
      idx = missing_idx;
    } else {
      std::string cat = as<std::string>(feature[i]);
      auto it = category_indices.find(cat);
      if (it == category_indices.end()) {
        idx = unique_categories.size();
        category_indices[cat] = idx;
        unique_categories.push_back(cat);
        count_total.push_back(0);
        count_pos.push_back(0);
        count_neg.push_back(0);
      } else {
        idx = it->second;
      }
    }
    
    count_total[idx]++;
    if (target[i] == 1) {
      count_pos[idx]++;
    } else if (target[i] == 0) {
      count_neg[idx]++;
    } else {
      // This should never happen due to the validation above
      throw std::invalid_argument("Target must be binary (0 or 1)");
    }
  }
  
  // Handle edge case where all observations are NA or no categories
  if (unique_categories.empty()) {
    throw std::invalid_argument("No valid observations found after processing missing values");
  }
  
  // Compute target rates with Laplace smoothing for stability
  category_target_rate.resize(unique_categories.size());
  for (size_t i = 0; i < unique_categories.size(); ++i) {
    // Apply mild smoothing for rate calculation to avoid extreme values
    category_target_rate[i] = static_cast<double>(count_pos[i] + 0.5) / (count_total[i] + 1.0);
  }
}

// Tratamento de categorias raras unindo-as com categorias similares
void OBC_SBLP::handle_rare_categories() {
  int total_count = std::accumulate(count_total.begin(), count_total.end(), 0);
  
  // Edge case: if total count is 0, return
  if (total_count == 0) return;
  
  std::vector<size_t> rare_indices;
  
  // Identify rare categories
  for (size_t i = 0; i < unique_categories.size(); ++i) {
    double proportion = static_cast<double>(count_total[i]) / total_count;
    if (proportion < bin_cutoff) {
      rare_indices.push_back(i);
    }
  }
  
  if (rare_indices.empty()) {
    return;
  }
  
  // Edge case: if all categories are rare
  if (rare_indices.size() == unique_categories.size()) {
    // Keep the most frequent ones
    std::vector<size_t> sorted_by_freq(unique_categories.size());
    std::iota(sorted_by_freq.begin(), sorted_by_freq.end(), 0);
    std::sort(sorted_by_freq.begin(), sorted_by_freq.end(),
              [this](size_t i, size_t j) { return count_total[i] > count_total[j]; });
    
    // Keep at least min_bins most frequent categories
    size_t keep_count = std::min(static_cast<size_t>(min_bins), unique_categories.size());
    rare_indices.clear();
    for (size_t i = keep_count; i < unique_categories.size(); ++i) {
      rare_indices.push_back(sorted_by_freq[i]);
    }
  }
  
  // Group similar rare categories based on target rate
  double similarity_threshold = 0.1; // Threshold for considering rates similar
  
  // Sort rare categories by target rate
  std::sort(rare_indices.begin(), rare_indices.end(),
            [this](size_t i, size_t j) { return category_target_rate[i] < category_target_rate[j]; });
  
  std::vector<std::vector<size_t>> similar_groups;
  
  // Create initial group
  if (!rare_indices.empty()) {
    similar_groups.push_back({rare_indices[0]});
  }
  
  // Assign each rare category to a group with similar target rate
  for (size_t i = 1; i < rare_indices.size(); ++i) {
    size_t idx = rare_indices[i];
    double rate = category_target_rate[idx];
    bool assigned = false;
    
    for (auto& group : similar_groups) {
      // Calculate average rate for group
      double group_rate = 0.0;
      int group_total = 0;
      for (size_t g_idx : group) {
        group_rate += category_target_rate[g_idx] * count_total[g_idx];
        group_total += count_total[g_idx];
      }
      
      if (group_total > 0) {
        group_rate /= group_total;
        
        // If similar enough, add to this group
        if (std::abs(rate - group_rate) <= similarity_threshold) {
          group.push_back(idx);
          assigned = true;
          break;
        }
      }
    }
    
    // If not assigned to any group, create new group
    if (!assigned) {
      similar_groups.push_back({idx});
    }
  }
  
  // Now merge categories within each group
  for (const auto& group : similar_groups) {
    if (group.size() <= 1) continue;
    
    // Merge all to first index in group
    size_t first_idx = group[0];
    std::vector<std::string> merged_categories = {unique_categories[first_idx]};
    
    for (size_t i = 1; i < group.size(); ++i) {
      size_t curr_idx = group[i];
      count_total[first_idx] += count_total[curr_idx];
      count_pos[first_idx] += count_pos[curr_idx];
      count_neg[first_idx] += count_neg[curr_idx];
      merged_categories.push_back(unique_categories[curr_idx]);
      unique_categories[curr_idx].clear(); // Mark for removal
    }
    
    unique_categories[first_idx] = merge_category_names(merged_categories, bin_separator);
  }
  
  // Remove empty categories
  std::vector<std::string> new_unique_categories;
  std::vector<int> new_count_total;
  std::vector<int> new_count_pos;
  std::vector<int> new_count_neg;
  
  for (size_t i = 0; i < unique_categories.size(); ++i) {
    if (!unique_categories[i].empty()) {
      new_unique_categories.push_back(unique_categories[i]);
      new_count_total.push_back(count_total[i]);
      new_count_pos.push_back(count_pos[i]);
      new_count_neg.push_back(count_neg[i]);
    }
  }
  
  unique_categories = std::move(new_unique_categories);
  count_total = std::move(new_count_total);
  count_pos = std::move(new_count_pos);
  count_neg = std::move(new_count_neg);
  
  // Recompute target rates
  category_target_rate.resize(unique_categories.size());
  for (size_t i = 0; i < unique_categories.size(); ++i) {
    // Apply Laplace smoothing
    category_target_rate[i] = static_cast<double>(count_pos[i] + 0.5) / (count_total[i] + 1.0);
  }
  
  // Invalidate cache
  cache_initialized = false;
}

// Garante que o número de pré-bins não exceda max_n_prebins
void OBC_SBLP::ensure_max_prebins() {
  if (unique_categories.size() <= static_cast<size_t>(max_n_prebins)) {
    return;
  }
  
  // Create sorted indices by target rate for merging
  std::vector<size_t> indices(unique_categories.size());
  std::iota(indices.begin(), indices.end(), 0);
  
  // Sort by target rate to merge adjacent categories
  std::sort(indices.begin(), indices.end(),
            [this](size_t i, size_t j) { return category_target_rate[i] < category_target_rate[j]; });
  
  size_t bins_to_merge = unique_categories.size() - max_n_prebins;
  
  // Identify which adjacent pairs to merge (those with smallest rate differences)
  std::vector<std::pair<double, std::pair<size_t, size_t>>> merge_candidates;
  for (size_t i = 0; i < indices.size() - 1; ++i) {
    double rate_diff = std::abs(category_target_rate[indices[i+1]] - category_target_rate[indices[i]]);
    merge_candidates.push_back({rate_diff, {indices[i], indices[i+1]}});
  }
  
  // Sort by similarity (smaller difference = more similar)
  std::sort(merge_candidates.begin(), merge_candidates.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
  
  // Track which bins have already been merged
  std::vector<bool> merged(unique_categories.size(), false);
  size_t merges_performed = 0;
  
  // Merge bins starting with most similar pairs
  for (const auto& candidate : merge_candidates) {
    if (merges_performed >= bins_to_merge) break;
    
    size_t idx1 = candidate.second.first;
    size_t idx2 = candidate.second.second;
    
    // Skip if either bin has already been merged
    if (merged[idx1] || merged[idx2]) continue;
    
    // Merge bins
    count_total[idx1] += count_total[idx2];
    count_pos[idx1] += count_pos[idx2];
    count_neg[idx1] += count_neg[idx2];
    unique_categories[idx1] = merge_category_names({unique_categories[idx1], unique_categories[idx2]}, bin_separator);
    unique_categories[idx2].clear();
    
    merged[idx2] = true;
    merges_performed++;
  }
  
  // Remove empty categories
  std::vector<std::string> new_unique_categories;
  std::vector<int> new_count_total;
  std::vector<int> new_count_pos;
  std::vector<int> new_count_neg;
  
  for (size_t i = 0; i < unique_categories.size(); ++i) {
    if (!unique_categories[i].empty()) {
      new_unique_categories.push_back(unique_categories[i]);
      new_count_total.push_back(count_total[i]);
      new_count_pos.push_back(count_pos[i]);
      new_count_neg.push_back(count_neg[i]);
    }
  }
  
  unique_categories = std::move(new_unique_categories);
  count_total = std::move(new_count_total);
  count_pos = std::move(new_count_pos);
  count_neg = std::move(new_count_neg);
  
  // Recompute target rates
  category_target_rate.resize(unique_categories.size());
  for (size_t i = 0; i < unique_categories.size(); ++i) {
    category_target_rate[i] = static_cast<double>(count_pos[i] + 0.5) / (count_total[i] + 1.0);
  }
  
  // Invalidate cache
  cache_initialized = false;
}

// Ordena categorias pela taxa alvo
void OBC_SBLP::sort_categories() {
  sorted_indices.resize(unique_categories.size());
  std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
  std::sort(sorted_indices.begin(), sorted_indices.end(),
            [this](size_t i, size_t j) { return category_target_rate[i] < category_target_rate[j]; });
}

// Executa o binning via programação dinâmica otimizada
std::vector<std::vector<size_t>> OBC_SBLP::perform_binning() {
  size_t n = sorted_indices.size();
  
  // Edge case: if n is less than or equal to max_bins, each category is its own bin
  if (n <= static_cast<size_t>(max_bins)) {
    std::vector<std::vector<size_t>> bins(n);
    for (size_t i = 0; i < n; ++i) {
      bins[i] = {sorted_indices[i]};
    }
    return bins;
  }
  
  size_t k = std::min(static_cast<size_t>(max_bins), n);
  
  // Use two vectors instead of a full matrix for DP (space optimization)
  std::vector<double> prev_dp(n + 1, -std::numeric_limits<double>::infinity());
  std::vector<double> curr_dp(n + 1, -std::numeric_limits<double>::infinity());
  
  // We still need to track split points for reconstruction
  std::vector<std::vector<size_t>> split(n + 1, std::vector<size_t>(k + 1, 0));
  
  // Initialize for j=1 (one bin)
  for (size_t i = 1; i <= n; ++i) {
    std::vector<size_t> bin(sorted_indices.begin(), sorted_indices.begin() + i);
    prev_dp[i] = calculate_bin_iv(bin);
  }
  
  // Fill DP table
  for (size_t j = 2; j <= k; ++j) {
    for (size_t i = j; i <= n; ++i) {
      curr_dp[i] = -std::numeric_limits<double>::infinity();
      for (size_t s = j - 1; s < i; ++s) {
        std::vector<size_t> bin(sorted_indices.begin() + s, sorted_indices.begin() + i);
        double current_iv = prev_dp[s] + calculate_bin_iv(bin);
        if (current_iv > curr_dp[i]) {
          curr_dp[i] = current_iv;
          split[i][j] = s;
        }
      }
    }
    // Swap for next iteration
    prev_dp.swap(curr_dp);
  }
  
  // Recover solution from split points
  std::vector<std::vector<size_t>> bins;
  
  // Traceback to find bin boundaries
  size_t i = n, j = k;
  while (j > 0) {
    size_t s = split[i][j];
    std::vector<size_t> bin(sorted_indices.begin() + s, sorted_indices.begin() + i);
    bins.push_back(bin);
    i = s;
    --j;
  }
  
  // First bin if any elements remain
  if (i > 0) {
    std::vector<size_t> bin(sorted_indices.begin(), sorted_indices.begin() + i);
    bins.push_back(bin);
  }
  
  std::reverse(bins.begin(), bins.end());
  
  // Ensure min_bins by splitting bins with highest IV
  while (bins.size() < static_cast<size_t>(min_bins) && bins.size() < n) {
    // Find bin with highest IV
    size_t max_iv_bin = 0;
    double max_iv = -std::numeric_limits<double>::infinity();
    
    for (size_t i = 0; i < bins.size(); ++i) {
      if (bins[i].size() > 1) {
        double bin_iv = calculate_bin_iv(bins[i]);
        if (bin_iv > max_iv) {
          max_iv = bin_iv;
          max_iv_bin = i;
        }
      }
    }
    
    // If no bin can be split further, break
    if (bins[max_iv_bin].size() <= 1) {
      break;
    }
    
    // Split the bin with highest IV
    size_t split_point = bins[max_iv_bin].size() / 2;
    std::vector<size_t> new_bin(bins[max_iv_bin].begin() + split_point, bins[max_iv_bin].end());
    bins[max_iv_bin].resize(split_point);
    bins.insert(bins.begin() + max_iv_bin + 1, new_bin);
  }
  
  return bins;
}

// Cálculo de IV para um único bin com Laplace smoothing
double OBC_SBLP::calculate_bin_iv(const std::vector<size_t>& bin) const {
  // Initialize cache if needed
  initialize_cache();
  
  // Edge case: empty bin
  if (bin.empty()) {
    return 0.0;
  }
  
  int bin_pos = 0, bin_neg = 0;
  for (size_t idx : bin) {
    bin_pos += count_pos[idx];
    bin_neg += count_neg[idx];
  }
  
  // Apply Laplace smoothing
  double smoothed_bin_pos = bin_pos + alpha;
  double smoothed_bin_neg = bin_neg + alpha;
  double smoothed_total_pos = total_pos_all + alpha * bin.size();
  double smoothed_total_neg = total_neg_all + alpha * bin.size();
  
  double pos_rate = smoothed_bin_pos / smoothed_total_pos;
  double neg_rate = smoothed_bin_neg / smoothed_total_neg;
  
  // Handle potential numerical issues
  const double min_rate = 1e-10;
  pos_rate = std::max(pos_rate, min_rate);
  neg_rate = std::max(neg_rate, min_rate);
  
  double woe = std::log(pos_rate / neg_rate);
  double iv = (pos_rate - neg_rate) * woe;
  
  return iv;
}

// Verifica monotonicidade em relação à taxa alvo
bool OBC_SBLP::is_monotonic(const std::vector<std::vector<size_t>>& bins) const {
  // Edge case: less than 2 bins is always monotonic
  if (bins.size() < 2) {
    return true;
  }
  
  std::vector<double> bin_rates;
  for (const auto& bin : bins) {
    int bin_total = 0, bin_pos_count = 0;
    for (size_t idx : bin) {
      bin_total += count_total[idx];
      bin_pos_count += count_pos[idx];
    }
    
    // Apply Laplace smoothing for stability
    double rate = (bin_total > 0) ? 
    static_cast<double>(bin_pos_count + alpha) / (bin_total + 2 * alpha) : 0.0;
    bin_rates.push_back(rate);
  }
  
  return std::is_sorted(bin_rates.begin(), bin_rates.end());
}

// Prepara a saída com estatísticas completas e WoE/IV
List OBC_SBLP::prepare_output(const std::vector<std::vector<size_t>>& bins, bool converged, int iterations) const {
  // Initialize cache if needed
  initialize_cache();
  
  std::vector<std::string> bin_names;
  std::vector<double> bin_woe;
  std::vector<double> bin_iv_vals;
  std::vector<int> bin_count_vals;
  std::vector<int> bin_count_pos_vals;
  std::vector<int> bin_count_neg_vals;
  std::vector<double> bin_rate_vals;
  
  // Compute bin statistics
  for (const auto& bin : bins) {
    std::vector<std::string> bin_categories;
    int bin_total = 0, bin_pos_count = 0, bin_neg_count = 0;
    
    for (size_t idx : bin) {
      bin_categories.push_back(unique_categories[idx]);
      bin_total += count_total[idx];
      bin_pos_count += count_pos[idx];
      bin_neg_count += count_neg[idx];
    }
    
    // Apply Laplace smoothing
    double smoothed_bin_pos = bin_pos_count + alpha;
    double smoothed_bin_neg = bin_neg_count + alpha;
    double smoothed_total_pos = total_pos_all + alpha * bins.size();
    double smoothed_total_neg = total_neg_all + alpha * bins.size();
    
    double pos_rate = smoothed_bin_pos / smoothed_total_pos;
    double neg_rate = smoothed_bin_neg / smoothed_total_neg;
    
    // Handle potential numerical issues
    const double min_rate = 1e-10;
    pos_rate = std::max(pos_rate, min_rate);
    neg_rate = std::max(neg_rate, min_rate);
    
    double woe = std::log(pos_rate / neg_rate);
    double iv = (pos_rate - neg_rate) * woe;
    double target_rate = (bin_total > 0) ? static_cast<double>(bin_pos_count) / bin_total : 0.0;
    
    bin_names.push_back(merge_category_names(bin_categories, bin_separator));
    bin_woe.push_back(woe);
    bin_iv_vals.push_back(iv);
    bin_count_vals.push_back(bin_total);
    bin_count_pos_vals.push_back(bin_pos_count);
    bin_count_neg_vals.push_back(bin_neg_count);
    bin_rate_vals.push_back(target_rate);
  }
  
  // Calculate total IV
  double total_iv = std::accumulate(bin_iv_vals.begin(), bin_iv_vals.end(), 0.0);
  
  // Create bin IDs (1-based indexing for R)
  Rcpp::NumericVector ids(bin_names.size());
  for (size_t i = 0; i < bin_names.size(); i++) {
    ids[i] = static_cast<double>(i + 1);
  }
  
  return Rcpp::List::create(
    Named("id") = ids,
    Named("bin") = bin_names,
    Named("woe") = bin_woe,
    Named("iv") = bin_iv_vals,
    Named("count") = bin_count_vals,
    Named("count_pos") = bin_count_pos_vals,
    Named("count_neg") = bin_count_neg_vals,
    Named("rate") = bin_rate_vals,
    Named("total_iv") = total_iv,
    Named("converged") = converged,
    Named("iterations") = iterations
  );
}

// Use separator for merging categories

std::string OBC_SBLP::merge_category_names(const std::vector<std::string>& categories, const std::string& separator) {
  std::vector<std::string> unique_cats;
  
  // Process each category
  for (const auto& cat : categories) {
    // Skip empty categories
    if (cat.empty()) continue;
    
    size_t start_pos = 0;
    size_t found_pos = cat.find(separator);
    
    // If category doesn't contain separator, add it as is
    if (found_pos == std::string::npos) {
      if (std::find(unique_cats.begin(), unique_cats.end(), cat) == unique_cats.end()) {
        unique_cats.push_back(cat);
      }
      continue;
    }
    
    // Process each part separated by the separator
    while (found_pos != std::string::npos) {
      std::string part = cat.substr(start_pos, found_pos - start_pos);
      if (!part.empty() && std::find(unique_cats.begin(), unique_cats.end(), part) == unique_cats.end()) {
        unique_cats.push_back(part);
      }
      start_pos = found_pos + separator.length();
      found_pos = cat.find(separator, start_pos);
    }
    
    // Add the last part
    std::string last_part = cat.substr(start_pos);
    if (!last_part.empty() && std::find(unique_cats.begin(), unique_cats.end(), last_part) == unique_cats.end()) {
      unique_cats.push_back(last_part);
    }
  }
  
  // Sort categories for consistency
  std::sort(unique_cats.begin(), unique_cats.end());
  
  // Join with separator
  if (unique_cats.empty()) {
    return ""; // Guard against empty categories
  }
  
  std::string result = unique_cats[0];
  for (size_t i = 1; i < unique_cats.size(); ++i) {
    result += separator + unique_cats[i];
  }
  
  return result;
}

// Função principal de ajuste
List OBC_SBLP::fit() {
  try {
    validate_input();
    compute_initial_counts();
    
    // Handle edge case: no valid categories
    if (unique_categories.empty()) {
      throw std::invalid_argument("No valid categories found after processing");
    }
    
    handle_rare_categories();
    ensure_max_prebins();
    sort_categories();
    
    // Handle edge case: not enough data for binning
    if (unique_categories.size() <= 1) {
      std::vector<std::vector<size_t>> bins(unique_categories.size());
      for (size_t i = 0; i < unique_categories.size(); ++i) {
        bins[i] = {i};
      }
      return prepare_output(bins, true, 0);
    }
    
    // Fast path if unique categories <= max_bins
    if (unique_categories.size() <= static_cast<size_t>(max_bins)) {
      std::vector<std::vector<size_t>> bins(unique_categories.size());
      for (size_t i = 0; i < unique_categories.size(); ++i) {
        bins[i] = {sorted_indices[i]};
      }
      return prepare_output(bins, true, 0);
    }
    
    std::vector<std::vector<size_t>> best_bins;
    double best_iv = -std::numeric_limits<double>::infinity();
    bool converged = false;
    int iterations = 0;
    
    // Iterative refinement
    while (iterations < max_iterations) {
      std::vector<std::vector<size_t>> current_bins = perform_binning();
      
      // Calculate total IV for current solution
      double current_iv = 0.0;
      for (const auto& bin : current_bins) {
        current_iv += calculate_bin_iv(bin);
      }
      
      // Check convergence
      if (std::abs(current_iv - best_iv) < convergence_threshold) {
        converged = true;
        break;
      }
      
      // Update best solution if improved
      if (current_iv > best_iv) {
        best_iv = current_iv;
        best_bins = current_bins;
      }
      
      ++iterations;
    }
    
    // Adjust for monotonicity if needed
    if (!is_monotonic(best_bins) && best_bins.size() > static_cast<size_t>(min_bins)) {
      // First attempt: try to merge adjacent non-monotonic bins
      std::vector<double> bin_rates;
      for (const auto& bin : best_bins) {
        int bin_total = 0, bin_pos_count = 0;
        for (size_t idx : bin) {
          bin_total += count_total[idx];
          bin_pos_count += count_pos[idx];
        }
        bin_rates.push_back(static_cast<double>(bin_pos_count) / std::max(bin_total, 1));
      }
      
      std::vector<std::vector<size_t>> monotonic_bins;
      std::vector<size_t> current_bin;
      
      for (size_t i = 0; i < best_bins.size(); ++i) {
        if (i == 0 || bin_rates[i] >= bin_rates[i-1] || monotonic_bins.size() < static_cast<size_t>(min_bins)) {
          // Start new bin
          if (!current_bin.empty()) {
            monotonic_bins.push_back(current_bin);
            current_bin.clear();
          }
          current_bin = best_bins[i];
        } else {
          // Merge with current bin
          current_bin.insert(current_bin.end(), best_bins[i].begin(), best_bins[i].end());
        }
      }
      
      // Add last bin
      if (!current_bin.empty()) {
        monotonic_bins.push_back(current_bin);
      }
      
      // Second attempt if still non-monotonic: force sort by rate
      if (!is_monotonic(monotonic_bins)) {
        // Collect all indices from all bins
        std::vector<size_t> all_indices;
        for (const auto& bin : best_bins) {
          all_indices.insert(all_indices.end(), bin.begin(), bin.end());
        }
        
        // Sort by target rate
        std::sort(all_indices.begin(), all_indices.end(),
                  [this](size_t i, size_t j) { return category_target_rate[i] < category_target_rate[j]; });
        
        // Create approximately equal-sized bins
        monotonic_bins.clear();
        size_t bin_size = all_indices.size() / std::max(static_cast<size_t>(min_bins), static_cast<size_t>(1));
        bin_size = std::max(bin_size, static_cast<size_t>(1));
        
        for (size_t i = 0; i < all_indices.size(); i += bin_size) {
          size_t end_idx = std::min(i + bin_size, all_indices.size());
          monotonic_bins.push_back(std::vector<size_t>(all_indices.begin() + i, all_indices.begin() + end_idx));
        }
        
        // Ensure we have at least min_bins
        while (monotonic_bins.size() < static_cast<size_t>(min_bins) && monotonic_bins.size() > 1) {
          // Find largest bin
          size_t largest_bin = 0;
          size_t largest_size = monotonic_bins[0].size();
          
          for (size_t i = 1; i < monotonic_bins.size(); ++i) {
            if (monotonic_bins[i].size() > largest_size) {
              largest_size = monotonic_bins[i].size();
              largest_bin = i;
            }
          }
          
          // Split largest bin
          if (monotonic_bins[largest_bin].size() > 1) {
            size_t split_point = monotonic_bins[largest_bin].size() / 2;
            std::vector<size_t> new_bin(monotonic_bins[largest_bin].begin() + split_point, 
                                        monotonic_bins[largest_bin].end());
            monotonic_bins[largest_bin].resize(split_point);
            monotonic_bins.push_back(new_bin);
          } else {
            break; // Can't split further
          }
        }
      }
      
      best_bins = monotonic_bins;
    }
    
    return prepare_output(best_bins, converged, iterations);
  } catch (const std::exception& e) {
    Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
  }
}


// [[Rcpp::export]]
List optimal_binning_categorical_sblp(const IntegerVector& target,
                                     const CharacterVector& feature,
                                     int min_bins = 3,
                                     int max_bins = 5,
                                     double bin_cutoff = 0.05,
                                     int max_n_prebins = 20,
                                     double convergence_threshold = 1e-6,
                                     int max_iterations = 1000,
                                     std::string bin_separator = "%;%",
                                     double alpha = 0.5) {
 OBC_SBLP optbin(target, feature, min_bins, max_bins, bin_cutoff, max_n_prebins,
                                      convergence_threshold, max_iterations, bin_separator, alpha);
 return optbin.fit();
}