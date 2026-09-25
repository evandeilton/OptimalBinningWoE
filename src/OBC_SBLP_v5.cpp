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
  // Original category names held by each (possibly merged) group. Labels are
  // built from these names directly: the old code re-split the joined label
  // on bin_separator, which broke categories whose own name contains the
  // separator, and used the empty string as its "removed" marker, which
  // silently dropped a genuine "" category together with its counts.
  std::vector<std::vector<std::string>> unique_categories;
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
  double iv_from_counts(int bin_pos, int bin_neg, size_t n_members) const;
  double smoothed_rate(const std::vector<size_t>& bin) const;
  void compact(const std::vector<bool>& alive);
  bool is_monotonic(const std::vector<std::vector<size_t>>& bins) const;
  void initialize_cache() const;
  List prepare_output(const std::vector<std::vector<size_t>>& bins, bool converged, int iterations) const;
  static std::string merge_category_names(std::vector<std::string> categories, const std::string& separator);
};

OBC_SBLP::OBC_SBLP(
  const IntegerVector& target_,
  const CharacterVector& feature_,
  int min_bins_,
  int max_bins_,
  double bin_cutoff_,
  int max_n_prebins_,
  double convergence_threshold_,
  int max_iterations_,
  std::string bin_separator_,
  double alpha_)
  : target(target_), feature(feature_),
    min_bins(min_bins_), max_bins(max_bins_),
    bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_),
    convergence_threshold(convergence_threshold_),
    max_iterations(max_iterations_),
    bin_separator(std::move(bin_separator_)),
    alpha(alpha_),
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
  
  // Target must be 0/1 without NA (ISNA() is for doubles and never matched an
  // integer NA, which was reported as "found value -2147483648") and must
  // contain both classes: with a single class and alpha = 0 every IV is NaN
  // and the result used to come back with zero bins.
  bool has_zero = false, has_one = false;
  for (R_xlen_t i = 0; i < target.size(); ++i) {
    if (target[i] == NA_INTEGER) {
      throw std::invalid_argument("Target cannot contain missing values");
    }
    if (target[i] == 0) {
      has_zero = true;
    } else if (target[i] == 1) {
      has_one = true;
    } else {
      throw std::invalid_argument("Target must be binary (0 or 1), found value " + 
                                  std::to_string(target[i]) + " at position " + 
                                  std::to_string(i+1));
    }
  }
  if (!has_zero || !has_one) {
    throw std::invalid_argument("Target must contain both 0 and 1 values");
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
  // Categories are indexed in order of first appearance. Every distinct
  // CHARSXP is resolved once (pointer lookup); the string map only merges
  // CHARSXPs that hold the same bytes under different encodings.
  std::unordered_map<std::string, size_t> category_indices;
  std::unordered_map<SEXP, size_t> ptr_indices;
  bool has_missing = false;

  for (R_xlen_t i = 0; i < feature.size(); ++i) {
    if (STRING_ELT(feature, i) == NA_STRING) {
      has_missing = true;
      break;
    }
  }

  // Add special "MISSING" category if needed (only reachable through a direct
  // .Call(): the R wrapper maps NA to "NA" beforehand)
  size_t missing_idx = 0;
  if (has_missing) {
    unique_categories.push_back({"MISSING"});
    count_total.push_back(0);
    count_pos.push_back(0);
    count_neg.push_back(0);
    missing_idx = 0;
    category_indices["MISSING"] = 0;
  }

  for (R_xlen_t i = 0; i < feature.size(); ++i) {
    SEXP cs = STRING_ELT(feature, i);
    size_t idx;
    if (cs == NA_STRING) {
      idx = missing_idx;
    } else {
      auto pit = ptr_indices.find(cs);
      if (pit != ptr_indices.end()) {
        idx = pit->second;
      } else {
        std::string cat(CHAR(cs));
        auto it = category_indices.find(cat);
        if (it == category_indices.end()) {
          idx = unique_categories.size();
          category_indices.emplace(cat, idx);
          unique_categories.push_back({cat});
          count_total.push_back(0);
          count_pos.push_back(0);
          count_neg.push_back(0);
        } else {
          idx = it->second;
        }
        ptr_indices.emplace(cs, idx);
      }
    }

    count_total[idx]++;
    if (target[i] == 1) {
      count_pos[idx]++;
    } else {
      count_neg[idx]++;
    }
  }

  if (unique_categories.empty()) {
    throw std::invalid_argument("No valid observations found after processing missing values");
  }

  // Target rates with mild (Laplace) smoothing, used as the sort key
  category_target_rate.resize(unique_categories.size());
  for (size_t i = 0; i < unique_categories.size(); ++i) {
    category_target_rate[i] = static_cast<double>(count_pos[i] + 0.5) / (count_total[i] + 1.0);
  }
}

// Drops the groups flagged as merged away and recomputes the target rates.
void OBC_SBLP::compact(const std::vector<bool>& alive) {
  std::vector<std::vector<std::string>> new_unique_categories;
  std::vector<int> new_count_total;
  std::vector<int> new_count_pos;
  std::vector<int> new_count_neg;

  for (size_t i = 0; i < unique_categories.size(); ++i) {
    if (alive[i]) {
      new_unique_categories.push_back(std::move(unique_categories[i]));
      new_count_total.push_back(count_total[i]);
      new_count_pos.push_back(count_pos[i]);
      new_count_neg.push_back(count_neg[i]);
    }
  }

  unique_categories = std::move(new_unique_categories);
  count_total = std::move(new_count_total);
  count_pos = std::move(new_count_pos);
  count_neg = std::move(new_count_neg);

  category_target_rate.resize(unique_categories.size());
  for (size_t i = 0; i < unique_categories.size(); ++i) {
    category_target_rate[i] = static_cast<double>(count_pos[i] + 0.5) / (count_total[i] + 1.0);
  }

  cache_initialized = false;
}

// Tratamento de categorias raras unindo-as com categorias similares
void OBC_SBLP::handle_rare_categories() {
  int total_count = std::accumulate(count_total.begin(), count_total.end(), 0);
  if (total_count == 0) return;

  std::vector<size_t> rare_indices;
  for (size_t i = 0; i < unique_categories.size(); ++i) {
    double proportion = static_cast<double>(count_total[i]) / total_count;
    if (proportion < bin_cutoff) {
      rare_indices.push_back(i);
    }
  }

  if (rare_indices.empty()) {
    return;
  }

  // Edge case: if all categories are rare, keep the min_bins most frequent
  if (rare_indices.size() == unique_categories.size()) {
    std::vector<size_t> sorted_by_freq(unique_categories.size());
    std::iota(sorted_by_freq.begin(), sorted_by_freq.end(), 0);
    std::sort(sorted_by_freq.begin(), sorted_by_freq.end(),
              [this](size_t i, size_t j) { return count_total[i] > count_total[j]; });

    size_t keep_count = std::min(static_cast<size_t>(min_bins), unique_categories.size());
    rare_indices.clear();
    for (size_t i = keep_count; i < unique_categories.size(); ++i) {
      rare_indices.push_back(sorted_by_freq[i]);
    }
  }

  const double similarity_threshold = 0.1;

  std::sort(rare_indices.begin(), rare_indices.end(),
            [this](size_t i, size_t j) { return category_target_rate[i] < category_target_rate[j]; });

  std::vector<std::vector<size_t>> similar_groups;
  if (!rare_indices.empty()) {
    similar_groups.push_back({rare_indices[0]});
  }

  // Assign each rare category to the first group with a similar target rate
  for (size_t i = 1; i < rare_indices.size(); ++i) {
    size_t idx = rare_indices[i];
    double rate = category_target_rate[idx];
    bool assigned = false;

    for (auto& group : similar_groups) {
      double group_rate = 0.0;
      int group_total = 0;
      for (size_t g_idx : group) {
        group_rate += category_target_rate[g_idx] * count_total[g_idx];
        group_total += count_total[g_idx];
      }

      if (group_total > 0) {
        group_rate /= group_total;
        if (std::abs(rate - group_rate) <= similarity_threshold) {
          group.push_back(idx);
          assigned = true;
          break;
        }
      }
    }

    if (!assigned) {
      similar_groups.push_back({idx});
    }
  }

  // Merge categories within each group into its first member
  std::vector<bool> alive(unique_categories.size(), true);
  for (const auto& group : similar_groups) {
    if (group.size() <= 1) continue;
    size_t first_idx = group[0];
    for (size_t i = 1; i < group.size(); ++i) {
      size_t curr_idx = group[i];
      count_total[first_idx] += count_total[curr_idx];
      count_pos[first_idx] += count_pos[curr_idx];
      count_neg[first_idx] += count_neg[curr_idx];
      unique_categories[first_idx].insert(unique_categories[first_idx].end(),
                                          unique_categories[curr_idx].begin(),
                                          unique_categories[curr_idx].end());
      alive[curr_idx] = false;
    }
  }

  compact(alive);
}

// Garante que o número de pré-bins não exceda max_n_prebins
void OBC_SBLP::ensure_max_prebins() {
  if (unique_categories.size() <= static_cast<size_t>(max_n_prebins)) {
    return;
  }

  std::vector<size_t> indices(unique_categories.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(),
            [this](size_t i, size_t j) { return category_target_rate[i] < category_target_rate[j]; });

  size_t bins_to_merge = unique_categories.size() - static_cast<size_t>(max_n_prebins);

  // Adjacent pairs (in rate order) with the smallest rate differences first
  std::vector<std::pair<double, std::pair<size_t, size_t>>> merge_candidates;
  merge_candidates.reserve(indices.size());
  for (size_t i = 0; i + 1 < indices.size(); ++i) {
    double rate_diff = std::abs(category_target_rate[indices[i+1]] - category_target_rate[indices[i]]);
    merge_candidates.push_back({rate_diff, {indices[i], indices[i+1]}});
  }
  std::sort(merge_candidates.begin(), merge_candidates.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

  std::vector<bool> merged(unique_categories.size(), false);
  size_t merges_performed = 0;

  for (const auto& candidate : merge_candidates) {
    if (merges_performed >= bins_to_merge) break;

    size_t idx1 = candidate.second.first;
    size_t idx2 = candidate.second.second;
    if (merged[idx1] || merged[idx2]) continue;

    count_total[idx1] += count_total[idx2];
    count_pos[idx1] += count_pos[idx2];
    count_neg[idx1] += count_neg[idx2];
    unique_categories[idx1].insert(unique_categories[idx1].end(),
                                   unique_categories[idx2].begin(),
                                   unique_categories[idx2].end());
    merged[idx2] = true;
    merges_performed++;
  }

  std::vector<bool> alive(merged.size());
  for (size_t i = 0; i < merged.size(); ++i) alive[i] = !merged[i];
  compact(alive);
}

// Ordena categorias pela taxa alvo
void OBC_SBLP::sort_categories() {
  sorted_indices.resize(unique_categories.size());
  std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
  std::sort(sorted_indices.begin(), sorted_indices.end(),
            [this](size_t i, size_t j) { return category_target_rate[i] < category_target_rate[j]; });
}

// Executa o binning via programação dinâmica
//
// dp[j][i] = max over s of dp[j-1][s] + IV(sorted[s..i)). The IV of a
// contiguous segment only needs its positive/negative totals, which come from
// prefix sums in O(1): the recurrence costs O(k n^2) instead of the O(k n^3)
// of materialising every candidate segment as a vector. Integer sums are
// exact, so the values (and the chosen partition) are unchanged.
std::vector<std::vector<size_t>> OBC_SBLP::perform_binning() {
  const size_t n = sorted_indices.size();

  if (n <= static_cast<size_t>(max_bins)) {
    std::vector<std::vector<size_t>> bins(n);
    for (size_t i = 0; i < n; ++i) {
      bins[i] = {sorted_indices[i]};
    }
    return bins;
  }

  initialize_cache();
  const size_t k = std::min(static_cast<size_t>(max_bins), n);

  std::vector<int> pre_pos(n + 1, 0), pre_neg(n + 1, 0);
  for (size_t i = 0; i < n; ++i) {
    pre_pos[i + 1] = pre_pos[i] + count_pos[sorted_indices[i]];
    pre_neg[i + 1] = pre_neg[i] + count_neg[sorted_indices[i]];
  }
  auto segment_iv = [&](size_t s, size_t e) {
    return iv_from_counts(pre_pos[e] - pre_pos[s], pre_neg[e] - pre_neg[s], e - s);
  };

  const double NEG_INF = -std::numeric_limits<double>::infinity();
  std::vector<double> prev_dp(n + 1, NEG_INF);
  std::vector<double> curr_dp(n + 1, NEG_INF);
  std::vector<std::vector<size_t>> split(n + 1, std::vector<size_t>(k + 1, 0));

  for (size_t i = 1; i <= n; ++i) {
    prev_dp[i] = segment_iv(0, i);
  }

  for (size_t j = 2; j <= k; ++j) {
    for (size_t i = j; i <= n; ++i) {
      curr_dp[i] = NEG_INF;
      for (size_t s = j - 1; s < i; ++s) {
        double current_iv = prev_dp[s] + segment_iv(s, i);
        if (current_iv > curr_dp[i]) {
          curr_dp[i] = current_iv;
          split[i][j] = s;
        }
      }
    }
    prev_dp.swap(curr_dp);
  }

  // Traceback: exactly k non-empty contiguous bins
  std::vector<std::vector<size_t>> bins;
  size_t i = n, j = k;
  while (j > 0) {
    size_t s = split[i][j];
    bins.emplace_back(sorted_indices.begin() + static_cast<std::ptrdiff_t>(s),
                      sorted_indices.begin() + static_cast<std::ptrdiff_t>(i));
    i = s;
    --j;
  }
  if (i > 0) {
    bins.emplace_back(sorted_indices.begin(), sorted_indices.begin() + static_cast<std::ptrdiff_t>(i));
  }
  std::reverse(bins.begin(), bins.end());
  return bins;
}

// IV contribution of a bin with Laplace smoothing. The smoothed totals add
// alpha once per member group, as the original implementation did.
double OBC_SBLP::iv_from_counts(int bin_pos, int bin_neg, size_t n_members) const {
  double smoothed_bin_pos = bin_pos + alpha;
  double smoothed_bin_neg = bin_neg + alpha;
  double smoothed_total_pos = total_pos_all + alpha * static_cast<double>(n_members);
  double smoothed_total_neg = total_neg_all + alpha * static_cast<double>(n_members);

  double pos_rate = smoothed_bin_pos / smoothed_total_pos;
  double neg_rate = smoothed_bin_neg / smoothed_total_neg;

  const double min_rate = 1e-10;
  pos_rate = std::max(pos_rate, min_rate);
  neg_rate = std::max(neg_rate, min_rate);

  double woe = std::log(pos_rate / neg_rate);
  return (pos_rate - neg_rate) * woe;
}

double OBC_SBLP::calculate_bin_iv(const std::vector<size_t>& bin) const {
  initialize_cache();
  if (bin.empty()) {
    return 0.0;
  }
  int bin_pos = 0, bin_neg = 0;
  for (size_t idx : bin) {
    bin_pos += count_pos[idx];
    bin_neg += count_neg[idx];
  }
  return iv_from_counts(bin_pos, bin_neg, bin.size());
}

// Smoothed event rate of a bin; the WoE reported by prepare_output() is
// increasing in this quantity, so ordering bins by it orders them by WoE.
double OBC_SBLP::smoothed_rate(const std::vector<size_t>& bin) const {
  int bin_total = 0, bin_pos_count = 0;
  for (size_t idx : bin) {
    bin_total += count_total[idx];
    bin_pos_count += count_pos[idx];
  }
  return (bin_total > 0) ?
    static_cast<double>(bin_pos_count + alpha) / (bin_total + 2 * alpha) : 0.0;
}

// Verifica monotonicidade em relação à taxa alvo
bool OBC_SBLP::is_monotonic(const std::vector<std::vector<size_t>>& bins) const {
  if (bins.size() < 2) {
    return true;
  }
  std::vector<double> bin_rates;
  bin_rates.reserve(bins.size());
  for (const auto& bin : bins) {
    bin_rates.push_back(smoothed_rate(bin));
  }
  return std::is_sorted(bin_rates.begin(), bin_rates.end());
}

// Prepara a saída com estatísticas completas e WoE/IV
List OBC_SBLP::prepare_output(const std::vector<std::vector<size_t>>& bins, bool converged, int iterations) const {
  initialize_cache();

  const size_t nb = bins.size();
  std::vector<std::string> bin_names;
  std::vector<double> bin_woe;
  std::vector<double> bin_iv_vals;
  std::vector<int> bin_count_vals;
  std::vector<int> bin_count_pos_vals;
  std::vector<int> bin_count_neg_vals;
  std::vector<double> bin_rate_vals;
  bin_names.reserve(nb);
  bin_woe.reserve(nb);
  bin_iv_vals.reserve(nb);
  bin_count_vals.reserve(nb);
  bin_count_pos_vals.reserve(nb);
  bin_count_neg_vals.reserve(nb);
  bin_rate_vals.reserve(nb);

  for (const auto& bin : bins) {
    std::vector<std::string> bin_categories;
    int bin_total = 0, bin_pos_count = 0, bin_neg_count = 0;

    for (size_t idx : bin) {
      bin_categories.insert(bin_categories.end(), unique_categories[idx].begin(), unique_categories[idx].end());
      bin_total += count_total[idx];
      bin_pos_count += count_pos[idx];
      bin_neg_count += count_neg[idx];
    }

    double smoothed_bin_pos = bin_pos_count + alpha;
    double smoothed_bin_neg = bin_neg_count + alpha;
    double smoothed_total_pos = total_pos_all + alpha * static_cast<double>(nb);
    double smoothed_total_neg = total_neg_all + alpha * static_cast<double>(nb);

    double pos_rate = smoothed_bin_pos / smoothed_total_pos;
    double neg_rate = smoothed_bin_neg / smoothed_total_neg;

    const double min_rate = 1e-10;
    pos_rate = std::max(pos_rate, min_rate);
    neg_rate = std::max(neg_rate, min_rate);

    double woe = std::log(pos_rate / neg_rate);
    double iv = (pos_rate - neg_rate) * woe;
    double target_rate = (bin_total > 0) ? static_cast<double>(bin_pos_count) / bin_total : 0.0;

    bin_names.push_back(merge_category_names(std::move(bin_categories), bin_separator));
    bin_woe.push_back(woe);
    bin_iv_vals.push_back(iv);
    bin_count_vals.push_back(bin_total);
    bin_count_pos_vals.push_back(bin_pos_count);
    bin_count_neg_vals.push_back(bin_neg_count);
    bin_rate_vals.push_back(target_rate);
  }

  double total_iv = std::accumulate(bin_iv_vals.begin(), bin_iv_vals.end(), 0.0);

  Rcpp::NumericVector ids(static_cast<R_xlen_t>(nb));
  for (size_t i = 0; i < nb; i++) {
    ids[static_cast<R_xlen_t>(i)] = static_cast<double>(i + 1);
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

// Bin label: the original category names, de-duplicated, sorted and joined
// with the separator (the same label the old split-and-rejoin produced for
// ordinary names, without mangling names that contain the separator).
std::string OBC_SBLP::merge_category_names(std::vector<std::string> categories, const std::string& separator) {
  std::sort(categories.begin(), categories.end());
  categories.erase(std::unique(categories.begin(), categories.end()), categories.end());

  std::string result;
  for (size_t i = 0; i < categories.size(); ++i) {
    if (i > 0) result += separator;
    result += categories[i];
  }
  return result;
}

// Função principal de ajuste
List OBC_SBLP::fit() {
  try {
    validate_input();
    compute_initial_counts();

    if (unique_categories.empty()) {
      throw std::invalid_argument("No valid categories found after processing");
    }

    handle_rare_categories();
    ensure_max_prebins();
    sort_categories();

    std::vector<std::vector<size_t>> best_bins;
    bool converged = true;
    int iterations = 0;

    if (unique_categories.size() <= static_cast<size_t>(max_bins)) {
      // One bin per (merged) category, in rate order
      best_bins.resize(unique_categories.size());
      for (size_t i = 0; i < unique_categories.size(); ++i) {
        best_bins[i] = {sorted_indices[i]};
      }
    } else {
      // The partition is a deterministic function of the counts, so the old
      // refinement loop recomputed the identical DP solution on its second
      // pass and stopped with |IV change| = 0: one pass gives the same bins,
      // iterations = 1, and converged unless max_iterations == 1.
      best_bins = perform_binning();
      iterations = 1;
      converged = max_iterations > 1;

      // Adjust for monotonicity if needed: merge adjacent bins whose raw
      // event rate drops.
      if (!is_monotonic(best_bins) && best_bins.size() > static_cast<size_t>(min_bins)) {
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
            if (!current_bin.empty()) {
              monotonic_bins.push_back(current_bin);
              current_bin.clear();
            }
            current_bin = best_bins[i];
          } else {
            current_bin.insert(current_bin.end(), best_bins[i].begin(), best_bins[i].end());
          }
        }
        if (!current_bin.empty()) {
          monotonic_bins.push_back(current_bin);
        }

        if (is_monotonic(monotonic_bins)) {
          best_bins = std::move(monotonic_bins);
        } else {
          // Second attempt: re-chunk the categories, in rate order, into
          // groups of size floor(N / min_bins). This can produce more than
          // max_bins groups (e.g. N = 7, min_bins = 4 gives 7 groups) and
          // is not guaranteed to be monotonic either, so it is only accepted
          // when it satisfies both; otherwise the optimal DP partition is
          // kept and its bins are put in WoE order below.
          std::vector<size_t> all_indices;
          for (const auto& bin : best_bins) {
            all_indices.insert(all_indices.end(), bin.begin(), bin.end());
          }
          std::sort(all_indices.begin(), all_indices.end(),
                    [this](size_t a, size_t b) { return category_target_rate[a] < category_target_rate[b]; });

          std::vector<std::vector<size_t>> chunked;
          size_t bin_size = all_indices.size() / std::max(static_cast<size_t>(min_bins), static_cast<size_t>(1));
          bin_size = std::max(bin_size, static_cast<size_t>(1));
          for (size_t i = 0; i < all_indices.size(); i += bin_size) {
            size_t end_idx = std::min(i + bin_size, all_indices.size());
            chunked.emplace_back(all_indices.begin() + static_cast<std::ptrdiff_t>(i),
                                 all_indices.begin() + static_cast<std::ptrdiff_t>(end_idx));
          }

          if (chunked.size() <= static_cast<size_t>(max_bins) && is_monotonic(chunked)) {
            best_bins = std::move(chunked);
          }
        }
      }
    }

    // Categorical bins have no natural order: listing them by smoothed event
    // rate (equivalently, by WoE) makes the reported WoE monotonic. This only
    // reorders bins, and only when they are not already in that order (e.g.
    // alpha != 0.5, where the sort key of the categories and the smoothing of
    // the bins differ).
    if (!is_monotonic(best_bins)) {
      std::vector<double> key(best_bins.size());
      std::vector<size_t> order(best_bins.size());
      for (size_t i = 0; i < best_bins.size(); ++i) {
        key[i] = smoothed_rate(best_bins[i]);
        order[i] = i;
      }
      std::stable_sort(order.begin(), order.end(),
                       [&key](size_t a, size_t b) { return key[a] < key[b]; });
      std::vector<std::vector<size_t>> reordered;
      reordered.reserve(best_bins.size());
      for (size_t i : order) reordered.push_back(std::move(best_bins[i]));
      best_bins = std::move(reordered);
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
