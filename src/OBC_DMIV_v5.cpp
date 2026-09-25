// [[Rcpp::depends(Rcpp)]]
#include <Rcpp.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <numeric> // Required for std::accumulate


// Include shared headers
#include "common/optimal_binning_common.h"
#include "common/bin_structures.h"

using namespace Rcpp;
using namespace OptimalBinning;

namespace {

// Category names that contain bin_separator. A bin label is its categories
// joined with the separator, and obwoe_apply() recovers the categories by
// splitting the label on it, so such a name is cut into pieces that are not
// categories (and a piece equal to another category claims it too). The
// binning itself is unaffected; the caller is warned that its labels are
// ambiguous. Checked over the distinct categories only, so the cost is O(k).
template <typename T>
inline const std::string& separator_key(const std::pair<const std::string, T>& kv) {
  return kv.first;
}

template <typename Container>
std::size_t count_separator_hits(const Container& categories, const std::string& sep,
                                 std::string& example) {
  std::size_t hits = 0;
  if (sep.empty()) return 0;
  for (const auto& item : categories) {
    const std::string& cat = separator_key(item);
    if (cat.find(sep) != std::string::npos) {
      if (hits == 0) example = cat;
      ++hits;
    }
  }
  return hits;
}

inline void warn_separator_hits(const std::string& sep, std::size_t hits,
                                const std::string& example) {
  if (hits == 0) return;
  Rcpp::warning("bin_separator \"%s\" occurs inside %d categor%s of the feature "
                "(e.g. \"%s\"): the bin labels are ambiguous and cannot be split "
                "back into categories. Choose a bin_separator that does not occur "
                "in the category names.",
                sep, hits, (hits == 1 ? "y" : "ies"), example);
}

} // namespace


/**
 * Core class implementing Optimal Binning for categorical variables using various Divergence Measures. (Version 2)
 * Based on the theoretical framework from Zeng (2013) "Metric Divergence Measures and Information Value in Credit Scoring".
 * V2 Corrections:
 *  - Fixed crash potential in similarity matrix update after merging.
 *  - Nearest-neighbour search replaces the dense divergence matrix (merge
 *    phase O(k^2) time and O(k) memory instead of O(k^3) and O(k^2)).
 *  - Corrected calculation and reporting of L2/L-infinity divergence.
 *  - Implemented max_n_prebins logic for handling very high cardinality features.
 *  - Added const correctness and improved comments.
 */
class OBC_DMIV {
private:
  // Input parameters (made const where appropriate)
  const std::vector<std::string>& feature;
  const std::vector<int>& target;
  int min_bins;
  int max_bins;
  const double bin_cutoff; // Frequency threshold
  const int max_n_prebins; // Max initial bins before merging rare ones
  const std::string bin_separator;
  const double convergence_threshold;
  const int max_iterations;
  const std::string bin_method;      // 'woe' or 'woe1'
  const std::string divergence_method; // 'he', 'kl', 'tr', 'klj', 'sc', 'js', 'l1', 'l2', 'ln'
  const int min_prebin_count = 5; // Min count threshold for pre-binning rare categories if max_n_prebins is exceeded
  
  // Small constant to handle numerical issues
  // Constant removed (uses shared definition)
  
  /**
   * Structure representing a bin of categorical values
   */
  // Local CategoricalBin definition removed

  
  // Internal state
  std::vector<CategoricalBin> bins;
  int total_pos = 0; // Initialize here
  int total_neg = 0; // Initialize here
  std::unordered_map<std::string, int> count_pos_map;
  std::unordered_map<std::string, int> count_neg_map;
  std::unordered_map<std::string, int> total_count_map;
  int initial_unique_categories = 0; // Before pre-binning
  bool converged = false;            // Initialize here
  int iterations_run = 0;            // Initialize here
  
  // Nearest-neighbour bookkeeping for the agglomerative merge (see
  // initialize_neighbours()): for each bin i, the smallest divergence to a bin
  // j > i and that j (-1 when there is none).
  std::vector<double> nn_div;
  std::vector<std::ptrdiff_t> nn_idx;

  // divergence_method / bin_method parsed once, so the hot divergence
  // function does not compare strings on every call.
  enum class Divergence { HE, KL, KLJ, TR, SC, JS, L1, L2, LN };
  Divergence div_kind = Divergence::L2;
  bool use_woe1 = true;
  
  
  // --- Private Methods ---
  
  /**
   * Validate input arguments for correctness and consistency
   * Throws std::invalid_argument if validation fails
   */
  void validate_inputs() {
    // Basic validation
    if (feature.empty() || target.empty()) {
      throw std::invalid_argument("Feature and target cannot be empty.");
    }
    if (feature.size() != target.size()) {
      throw std::invalid_argument("Feature and target must have the same length.");
    }
    if (min_bins <= 0 || max_bins <= 0 || min_bins > max_bins) {
      throw std::invalid_argument("Invalid values for min_bins or max_bins (must be > 0 and min_bins <= max_bins).");
    }
    if (bin_cutoff <= 0 || bin_cutoff >= 1) {
      throw std::invalid_argument("bin_cutoff must be between 0 and 1 (exclusive).");
    }
    if (max_n_prebins < 2) {
      // Need at least 2 bins potentially to merge
      throw std::invalid_argument("max_n_prebins must be at least 2.");
    }
    if (convergence_threshold <= 0) {
      throw std::invalid_argument("convergence_threshold must be positive.");
    }
    if (max_iterations <= 0) {
      throw std::invalid_argument("max_iterations must be positive.");
    }
    
    // Validate bin_method and divergence_method
    if (bin_method != "woe" && bin_method != "woe1") {
      throw std::invalid_argument("bin_method must be either 'woe' or 'woe1'.");
    }
    
    const std::unordered_set<std::string> valid_divergence_methods = {
      "he", "kl", "tr", "klj", "sc", "js", "l1", "l2", "ln"
    };
    if (valid_divergence_methods.find(divergence_method) == valid_divergence_methods.end()) {
      throw std::invalid_argument("Invalid divergence_method. Must be one of: 'he', 'kl', 'tr', 'klj', 'sc', 'js', 'l1', 'l2', 'ln'.");
    }
    if (divergence_method == "he") div_kind = Divergence::HE;
    else if (divergence_method == "kl") div_kind = Divergence::KL;
    else if (divergence_method == "klj") div_kind = Divergence::KLJ;
    else if (divergence_method == "tr") div_kind = Divergence::TR;
    else if (divergence_method == "sc") div_kind = Divergence::SC;
    else if (divergence_method == "js") div_kind = Divergence::JS;
    else if (divergence_method == "l1") div_kind = Divergence::L1;
    else if (divergence_method == "l2") div_kind = Divergence::L2;
    else div_kind = Divergence::LN;
    use_woe1 = (bin_method == "woe1");
    
    // Efficiently process data in a single pass to get counts
    const size_t total_count = target.size();
    std::unordered_map<std::string, std::pair<int, int>> counts;
    counts.reserve(std::min(static_cast<size_t>(total_count), static_cast<size_t>(max_n_prebins) + 100)); // Heuristic reservation
    
    for (size_t i = 0; i < total_count; ++i) {
      const int t = target[i];
      if (t != 0 && t != 1) {
        throw std::invalid_argument("Target must be binary (0 or 1).");
      }
      
      const std::string& cat = feature[i]; // Assume "NA" is already handled by wrapper
      auto& count_pair = counts[cat]; // Creates if not exists
      
      if (t == 1) {
        count_pair.first++;
        total_pos++;
      } else {
        count_pair.second++;
        total_neg++;
      }
    }
    
    if (total_pos == 0 || total_neg == 0) {
      throw std::invalid_argument("Target must contain both 0 and 1 values.");
    }
    
    // Transfer counts to final maps
    count_pos_map.reserve(counts.size());
    count_neg_map.reserve(counts.size());
    total_count_map.reserve(counts.size());
    for (const auto& item : counts) {
      const std::string& cat = item.first;
      const auto& count_pair = item.second;
      
      count_pos_map[cat] = count_pair.first;
      count_neg_map[cat] = count_pair.second;
      total_count_map[cat] = count_pair.first + count_pair.second;
    }
    
    initial_unique_categories = static_cast<int>(counts.size());
    
    // Adjust bin constraints based on available unique categories *after* potential pre-binning
    // This adjustment will happen after initialize_bins if pre-binning occurs.
    if (initial_unique_categories < 2) {
      throw std::invalid_argument("Feature must have at least 2 unique categories.");
    }
  }
  
  /**
   * Initialize bins, potentially pre-binning rare categories if cardinality is high.
   */
  void initialize_bins() {
    bins.clear();
    
    // Pre-binning logic if unique categories exceed max_n_prebins
    if (initial_unique_categories > max_n_prebins) {
      bins.reserve(static_cast<size_t>(max_n_prebins)); // Approximate final size
      // Pooled bin for the rare categories. It records the categories it
      // holds, like every other bin. It used to be labelled with the
      // placeholder "PREBIN_OTHER" instead, so the pooled categories appeared
      // in no bin label and could not be mapped to a WoE when the binning was
      // applied (obwoe_apply() matches categories against the labels).
      CategoricalBin other_bin;

      for (const auto& item : total_count_map) {
        const std::string& cat = item.first;
        int cat_total = item.second;

        // Keep categories if count is >= min_prebin_count, otherwise add to 'other' bin
        if (cat_total >= min_prebin_count) {
          CategoricalBin bin;
          bin.categories.push_back(cat);
          bin.count_pos = count_pos_map[cat];
          bin.count_neg = count_neg_map[cat];
          bins.push_back(std::move(bin));
        } else {
          other_bin.categories.push_back(cat);
          other_bin.count_pos += count_pos_map[cat];
          other_bin.count_neg += count_neg_map[cat];
        }
      }
      // Add the 'other' bin if it collected any categories
      if (other_bin.total() > 0) {
        bins.push_back(std::move(other_bin));
      }

      // Fallback: ignore pre-binning if it leaves fewer than min_bins bins,
      // and proceed with the normal initialization below; the merge phase
      // then groups the rare categories by divergence. The threshold used to
      // be 2, so a feature with one frequent level and many rare ones was
      // pre-binned into 2 bins, min_bins was clamped down to 2 to match, and
      // the fit silently returned fewer bins than min_bins although more
      // categories were available. (The console messages this block used to
      // print unconditionally have been removed.)
      if (bins.size() < static_cast<size_t>(std::max(2, min_bins))) {
        bins.clear();
      }
    }
    
    // Normal initialization if no pre-binning occurred or if fallback triggered
    if (bins.empty()) {
      bins.reserve(initial_unique_categories);
      for (const auto& item : total_count_map) {
        CategoricalBin bin;
        bin.categories.push_back(item.first);
        bin.count_pos = count_pos_map[item.first];
        bin.count_neg = count_neg_map[item.first];
        bins.push_back(std::move(bin));
      }
    }
    
    // Adjust bin constraints based on the actual number of initial bins
    const int current_bins = static_cast<int>(bins.size());
    min_bins = std::max(2, std::min(min_bins, current_bins));
    max_bins = std::min(max_bins, current_bins);
    if (min_bins > max_bins) {
      min_bins = max_bins; // Ensure min <= max
    }
    
    // Compute initial metrics and sort (optional but can help)
    compute_bin_metrics();
    sort_bins_by_woe();
  }
  
  
  /**
   * Sort bins by Weight of Evidence (ascending).
   */
  void sort_bins_by_woe() {
    std::sort(bins.begin(), bins.end(),
              [](const CategoricalBin& a, const CategoricalBin& b) { return a.woe < b.woe; });
    // NOTE: Sorting invalidates the distance matrix if done mid-process.
    // It's mainly for final output or potentially better initial state.
    // We recompute/update the distance matrix after merges/splits anyway.
  }
  
  
  /**
   * Compute divergence between two bins based on their pos/neg distributions.
   * Lower values indicate more similar bins (less divergence).
   * @param bin1 First bin.
   * @param bin2 Second bin.
   * @return Divergence score.
   */
  double compute_bin_divergence(const CategoricalBin& bin1, const CategoricalBin& bin2) const {
    double total1 = static_cast<double>(bin1.total());
    double total2 = static_cast<double>(bin2.total());
    
    // Handle empty bins (shouldn't happen in main loop but maybe in rare category handling)
    if (total1 < EPSILON || total2 < EPSILON) return std::numeric_limits<double>::max(); // Max divergence
    
    // Use overall totals for distribution calculation, consistent with compute_bin_metrics
    double dist1_pos = static_cast<double>(bin1.count_pos) / static_cast<double>(total_pos);
    double dist1_neg = static_cast<double>(bin1.count_neg) / static_cast<double>(total_neg);
    double dist2_pos = static_cast<double>(bin2.count_pos) / static_cast<double>(total_pos);
    double dist2_neg = static_cast<double>(bin2.count_neg) / static_cast<double>(total_neg);
    
    // Apply epsilon smoothing *only when needed* (for log or division)
    double divergence = 0.0;
    
    if (div_kind == Divergence::HE) {
      // Hellinger Distance (already a metric, >= 0)
      divergence = std::pow(std::sqrt(std::max(dist1_pos, 0.0)) - std::sqrt(std::max(dist2_pos, 0.0)), 2) +
        std::pow(std::sqrt(std::max(dist1_neg, 0.0)) - std::sqrt(std::max(dist2_neg, 0.0)), 2);
    } else if (div_kind == Divergence::KL) {
      // Symmetrized KL divergence (>= 0)
      double kl12 = (dist1_pos > EPSILON ? dist1_pos * std::log(dist1_pos / std::max(dist2_pos, EPSILON)) : 0.0) +
        (dist1_neg > EPSILON ? dist1_neg * std::log(dist1_neg / std::max(dist2_neg, EPSILON)) : 0.0);
      double kl21 = (dist2_pos > EPSILON ? dist2_pos * std::log(dist2_pos / std::max(dist1_pos, EPSILON)) : 0.0) +
        (dist2_neg > EPSILON ? dist2_neg * std::log(dist2_neg / std::max(dist1_neg, EPSILON)) : 0.0);
      divergence = kl12 + kl21;
    } else if (div_kind == Divergence::KLJ) {
      // J-Divergence (same as symmetrized KL)
      double kl12 = (dist1_pos > EPSILON ? dist1_pos * std::log(dist1_pos / std::max(dist2_pos, EPSILON)) : 0.0) +
        (dist1_neg > EPSILON ? dist1_neg * std::log(dist1_neg / std::max(dist2_neg, EPSILON)) : 0.0);
      double kl21 = (dist2_pos > EPSILON ? dist2_pos * std::log(dist2_pos / std::max(dist1_pos, EPSILON)) : 0.0) +
        (dist2_neg > EPSILON ? dist2_neg * std::log(dist2_neg / std::max(dist1_neg, EPSILON)) : 0.0);
      divergence = kl12 + kl21;
      // Original Zeng formula: (p1-p2)log(p1/p2) + (n1-n2)log(n1/n2) - seems different? Let's stick to symmetric KL definition based on context.
      // If using Zeng's literal formula:
      // divergence = (dist1_pos - dist2_pos) * (dist1_pos > EPSILON && dist2_pos > EPSILON ? std::log(dist1_pos / dist2_pos) : 0.0) +
      //             (dist1_neg - dist2_neg) * (dist1_neg > EPSILON && dist2_neg > EPSILON ? std::log(dist1_neg / dist2_neg) : 0.0);
    } else if (div_kind == Divergence::TR) {
      // Triangular Discrimination (>= 0)
      divergence = (total1 > EPSILON && total2 > EPSILON) ?
      (std::pow(dist1_pos - dist2_pos, 2) / std::max(dist1_pos + dist2_pos, EPSILON)) +
      (std::pow(dist1_neg - dist2_neg, 2) / std::max(dist1_neg + dist2_neg, EPSILON))
        : std::numeric_limits<double>::max(); // Avoid division by zero if sums are zero
    } else if (div_kind == Divergence::SC) {
      // Chi-Square Symmetric (>= 0)
      divergence = (dist1_pos > EPSILON && dist2_pos > EPSILON ?
                      std::pow(dist1_pos - dist2_pos, 2) * (dist1_pos + dist2_pos) / (dist1_pos * dist2_pos) : 0.0) +
                      (dist1_neg > EPSILON && dist2_neg > EPSILON ?
                      std::pow(dist1_neg - dist2_neg, 2) * (dist1_neg + dist2_neg) / (dist1_neg * dist2_neg) : 0.0);
      // Add large penalty if any denominator is zero but numerator isn't
      if (((dist1_pos < EPSILON || dist2_pos < EPSILON) && std::abs(dist1_pos - dist2_pos) > EPSILON) ||
          ((dist1_neg < EPSILON || dist2_neg < EPSILON) && std::abs(dist1_neg - dist2_neg) > EPSILON)) {
        divergence = std::numeric_limits<double>::max();
      }
      
    } else if (div_kind == Divergence::JS) {
      // Jensen-Shannon Divergence (>= 0)
      double m_pos = (dist1_pos + dist2_pos) / 2.0;
      double m_neg = (dist1_neg + dist2_neg) / 2.0;
      double js1 = (dist1_pos > EPSILON ? dist1_pos * std::log(dist1_pos / std::max(m_pos, EPSILON)) : 0.0) +
        (dist1_neg > EPSILON ? dist1_neg * std::log(dist1_neg / std::max(m_neg, EPSILON)) : 0.0);
      double js2 = (dist2_pos > EPSILON ? dist2_pos * std::log(dist2_pos / std::max(m_pos, EPSILON)) : 0.0) +
        (dist2_neg > EPSILON ? dist2_neg * std::log(dist2_neg / std::max(m_neg, EPSILON)) : 0.0);
      divergence = 0.5 * (js1 + js2);
    } else if (div_kind == Divergence::L1) {
      // L1 metric (Manhattan) (>= 0) - uses local proportions
      double local_dist1_pos = (total1 > EPSILON) ? static_cast<double>(bin1.count_pos) / total1 : 0.0;
      double local_dist1_neg = (total1 > EPSILON) ? static_cast<double>(bin1.count_neg) / total1 : 0.0;
      double local_dist2_pos = (total2 > EPSILON) ? static_cast<double>(bin2.count_pos) / total2 : 0.0;
      double local_dist2_neg = (total2 > EPSILON) ? static_cast<double>(bin2.count_neg) / total2 : 0.0;
      divergence = std::abs(local_dist1_pos - local_dist2_pos) + std::abs(local_dist1_neg - local_dist2_neg);
    } else if (div_kind == Divergence::L2) {
      // L2 metric (Euclidean) (>= 0) - uses local proportions
      double local_dist1_pos = (total1 > EPSILON) ? static_cast<double>(bin1.count_pos) / total1 : 0.0;
      double local_dist1_neg = (total1 > EPSILON) ? static_cast<double>(bin1.count_neg) / total1 : 0.0;
      double local_dist2_pos = (total2 > EPSILON) ? static_cast<double>(bin2.count_pos) / total2 : 0.0;
      double local_dist2_neg = (total2 > EPSILON) ? static_cast<double>(bin2.count_neg) / total2 : 0.0;
      divergence = std::sqrt(std::pow(local_dist1_pos - local_dist2_pos, 2) + std::pow(local_dist1_neg - local_dist2_neg, 2));
    } else if (div_kind == Divergence::LN) {
      // L∞ metric (Maximum) (>= 0) - uses local proportions
      double local_dist1_pos = (total1 > EPSILON) ? static_cast<double>(bin1.count_pos) / total1 : 0.0;
      double local_dist1_neg = (total1 > EPSILON) ? static_cast<double>(bin1.count_neg) / total1 : 0.0;
      double local_dist2_pos = (total2 > EPSILON) ? static_cast<double>(bin2.count_pos) / total2 : 0.0;
      double local_dist2_neg = (total2 > EPSILON) ? static_cast<double>(bin2.count_neg) / total2 : 0.0;
      divergence = std::max(std::abs(local_dist1_pos - local_dist2_pos), std::abs(local_dist1_neg - local_dist2_neg));
    }
    
    // Ensure divergence is not negative due to potential floating point issues
    return std::max(0.0, divergence);
  }
  
  
  /**
   * Recompute the nearest-neighbour entry of row i: the smallest divergence
   * between bins[i] and any bins[j] with j > i, and the smallest such j.
   *
   * The scan uses a strict "<" from an initial value of double::max, exactly
   * like the full-matrix search it replaces, so a pair whose divergence is
   * double::max (the "sc" penalty) or NaN is never selected, and among equal
   * minima the smallest j wins.
   */
  void recompute_neighbour(size_t i) {
    const size_t n = bins.size();
    double best = std::numeric_limits<double>::max();
    std::ptrdiff_t arg = -1;
    for (size_t j = i + 1; j < n; ++j) {
      const double d = compute_bin_divergence(bins[i], bins[j]);
      if (d < best) {
        best = d;
        arg = static_cast<std::ptrdiff_t>(j);
      }
    }
    nn_div[i] = best;
    nn_idx[i] = arg;
  }

  /**
   * Build the nearest-neighbour arrays for the current bins.
   *
   * This replaces the dense k x k divergence matrix. The matrix cost O(k^2)
   * memory (a 3,000-level feature allocated 72 MB) and every merge erased a
   * row and a column from it and rescanned all k^2/2 entries, so the merge
   * phase was O(k^3): 66 s for 3,000 levels. Per-row minima give the same
   * global minimum -- the first row holding the smallest row minimum, and in
   * it the first column, i.e. the same lexicographic tie-break as the
   * row-major scan of the matrix -- in O(k) per merge plus the rows whose
   * neighbour was one of the two merged bins.
   *
   * Divergences are recomputed from the bin counts rather than cached. They
   * are a pure function of the two bins' counts and are exactly symmetric in
   * their arguments (every measure is built from commutative sums, products
   * and squared differences), so the values, and hence the merge sequence,
   * are bit-identical to the matrix version.
   */
  void initialize_neighbours() {
    const size_t n = bins.size();
    nn_div.assign(n, std::numeric_limits<double>::max());
    nn_idx.assign(n, -1);
    for (size_t i = 0; i < n; ++i) {
      recompute_neighbour(i);
    }
  }

  /**
   * Pair of bins with the minimum divergence (best merge candidate).
   * Falls back to {0, 1} when no pair has a divergence below double::max,
   * which is what the matrix scan returned in that case.
   */
  std::pair<double, std::pair<size_t, size_t>> find_most_similar_bins() const {
    double min_divergence = std::numeric_limits<double>::max();
    std::pair<size_t, size_t> best_pair = {0, 1};
    for (size_t i = 0; i < nn_div.size(); ++i) {
      if (nn_idx[i] >= 0 && nn_div[i] < min_divergence) {
        min_divergence = nn_div[i];
        best_pair = {i, static_cast<size_t>(nn_idx[i])};
      }
    }
    return {min_divergence, best_pair};
  }

  /**
   * Update the nearest-neighbour arrays after bins[b] was merged into
   * bins[a] (a < b) and erased. Called after the erase, with the original
   * indices.
   */
  void update_neighbours_after_merge(size_t a, size_t b) {
    nn_div.erase(nn_div.begin() + static_cast<std::ptrdiff_t>(b));
    nn_idx.erase(nn_idx.begin() + static_cast<std::ptrdiff_t>(b));
    const std::ptrdiff_t pa = static_cast<std::ptrdiff_t>(a);
    const std::ptrdiff_t pb = static_cast<std::ptrdiff_t>(b);
    const size_t n = bins.size();

    std::vector<size_t> stale;
    for (size_t i = 0; i < n; ++i) {
      if (i == a) continue;
      std::ptrdiff_t& j = nn_idx[i];
      if (j == pb || (i < a && j == pa)) {
        // Its nearest neighbour disappeared or changed: rescan the row.
        stale.push_back(i);
      } else if (j > pb) {
        --j; // index shift caused by the erase
      }
    }

    // Rows before `a` whose neighbour is untouched: the merged bin is a new
    // candidate for them. Same strict-less / smallest-index rule as the scan.
    std::vector<char> is_stale(n, 0);
    for (size_t i : stale) is_stale[i] = 1;
    for (size_t i = 0; i < a; ++i) {
      if (is_stale[i]) continue;
      const double d = compute_bin_divergence(bins[i], bins[a]);
      if (d < nn_div[i] || (nn_idx[i] >= 0 && d == nn_div[i] && pa < nn_idx[i])) {
        nn_div[i] = d;
        nn_idx[i] = pa;
      }
    }

    recompute_neighbour(a);
    for (size_t i : stale) recompute_neighbour(i);
  }


  /**
   * Perform optimal binning using divergence measures via hierarchical merging.
   */
  void perform_binning() {
    iterations_run = 0;
    converged = false;

    if (bins.size() <= static_cast<size_t>(max_bins)) {
      // Already at a valid stopping state: the bin-count target is met without
      // any merging. (This used to print an unconditional "Info: ... Skipping
      // merging phase." line to the console for every such feature.)
      converged = true;
      return;
    }

    initialize_neighbours();

    double previous_min_divergence = -1.0;
    // Set when max_iterations merges have been made while the bin count was
    // still above max_bins and the divergence tolerance had not been met.
    bool exhausted = false;

    // Merge until max_bins is reached.
    //
    // max_bins is a hard constraint. Merging used to stop at max_iterations,
    // which returned 2,000 bins for max_bins = 5 on a 3,000-level feature
    // (every level with >= 5 rows survives pre-binning, so k - max_bins
    // merges are needed). Each merge removes one bin, so the loop terminates
    // after at most k - max_bins merges regardless; max_iterations now only
    // decides the `converged` flag: it is FALSE when the cap was reached
    // before the tolerance was met, exactly as before.
    while (bins.size() > static_cast<size_t>(max_bins) && bins.size() > 1) {
      if (iterations_run == max_iterations && !converged) {
        exhausted = true;
      }

      std::pair<double, std::pair<size_t, size_t>> best_merge = find_most_similar_bins();
      double current_min_divergence = best_merge.first;
      size_t bin1_idx = best_merge.second.first;  // bin to keep / merge into
      size_t bin2_idx = best_merge.second.second; // bin to remove
      if (bin1_idx > bin2_idx) std::swap(bin1_idx, bin2_idx);

      // Record convergence on a small absolute change in the min divergence.
      // This used to break out of the loop, which abandoned the descent to
      // max_bins (299 bins for max_bins = 5 on 300 levels); the merge order
      // is unchanged, the tolerance only sets the flag.
      if (previous_min_divergence >= 0 &&
          std::fabs(current_min_divergence - previous_min_divergence) < convergence_threshold) {
        converged = true;
      }

      merge_two_bins(bins[bin1_idx], bins[bin2_idx]);
      bins.erase(bins.begin() + static_cast<std::ptrdiff_t>(bin2_idx));
      compute_single_bin_metrics(bins[bin1_idx]);
      update_neighbours_after_merge(bin1_idx, bin2_idx);

      previous_min_divergence = current_min_divergence;
      iterations_run++;
    }

    // Reaching max_bins is a valid stopping state; only running out of
    // max_iterations first leaves converged == FALSE.
    converged = !exhausted;
  }

  /*
   * min_bins needs no splitting phase. initialize_bins() clamps
   * min_bins <= max_bins <= number of initial bins, and perform_binning()
   * never merges below max_bins, so the final bin count is always within
   * [min_bins, max_bins]. The former ensure_min_bins() / split_bin_into_two()
   * / calculate_bin_heterogeneity() / distance-matrix split update were
   * unreachable for that reason and have been removed.
   */


  /**
   * Merge source bin (`src_bin`) into destination bin (`dest_bin`).
   * Modifies `dest_bin` in place.
   * @param dest_bin Destination bin.
   * @param src_bin Source bin.
   */
  void merge_two_bins(CategoricalBin& dest_bin, const CategoricalBin& src_bin) {
    // Merge categories
    dest_bin.categories.reserve(dest_bin.categories.size() + src_bin.categories.size());
    dest_bin.categories.insert(dest_bin.categories.end(),
                               src_bin.categories.begin(),
                               src_bin.categories.end());
    // Could sort categories here if needed for consistent naming, but affects performance.
    // std::sort(dest_bin.categories.begin(), dest_bin.categories.end());
    
    // Merge counts
    dest_bin.count_pos += src_bin.count_pos;
    dest_bin.count_neg += src_bin.count_neg;
    
    // WoE and divergence will be recomputed later for the merged bin
    dest_bin.woe = 0.0;
    dest_bin.divergence = 0.0;
  }
  
  /**
   * Compute bin metrics (WoE and divergence) for a single bin.
   * Used for efficiency after merges/splits.
   * @param bin The bin to compute metrics for (modified in place).
   */
  void compute_single_bin_metrics(CategoricalBin& bin) {
    double total_bin = static_cast<double>(bin.total());
    if (total_bin < EPSILON) { // # nocov start (every bin holds at least one observation)
      bin.woe = 0.0;
      bin.divergence = 0.0;
      return;
    } // # nocov end
    
    // Distributions relative to overall totals
    double dist_pos = static_cast<double>(bin.count_pos) / static_cast<double>(total_pos);
    double dist_neg = static_cast<double>(bin.count_neg) / static_cast<double>(total_neg);
    
    // --- Calculate WoE ---
    if (!use_woe1) {
      // Traditional WoE: ln((p_i/P)/(n_i/N))
      bin.woe = std::log(std::max(dist_pos, EPSILON) / std::max(dist_neg, EPSILON));
    } else { // bin_method == "woe1"
      // Zeng's WOE1: ln(g_i/b_i) using smoothed counts
      double smoothed_pos = std::max(static_cast<double>(bin.count_pos) + 0.5, EPSILON);
      double smoothed_neg = std::max(static_cast<double>(bin.count_neg) + 0.5, EPSILON);
      bin.woe = std::log(smoothed_pos / smoothed_neg);
    }
    
    // --- Calculate Divergence Contribution ---
    // Apply epsilon only where needed (log or division)
    dist_pos = std::max(dist_pos, 0.0); // Use 0 for divergences not involving log/div
    dist_neg = std::max(dist_neg, 0.0);
    
    if (div_kind == Divergence::HE) {
      // Hellinger Discrimination Term: 0.5 * (sqrt(p) - sqrt(n))^2
      bin.divergence = 0.5 * std::pow(std::sqrt(dist_pos) - std::sqrt(dist_neg), 2);
    } else if (div_kind == Divergence::KL) {
      // Kullback-Leibler Term: p * log(p/n)
      bin.divergence = (dist_pos > EPSILON) ? dist_pos * std::log(dist_pos / std::max(dist_neg, EPSILON)) : 0.0;
    } else if (div_kind == Divergence::TR) {
      // Triangular Discrimination Term: (p-n)^2 / (p+n)
      bin.divergence = (dist_pos + dist_neg > EPSILON) ? std::pow(dist_pos - dist_neg, 2) / (dist_pos + dist_neg) : 0.0;
    } else if (div_kind == Divergence::KLJ) {
      // J-Divergence Term: (p-n) * log(p/n)
      bin.divergence = (dist_pos - dist_neg) * ((dist_pos > EPSILON && dist_neg > EPSILON) ? std::log(dist_pos / dist_neg) : 0.0);
    } else if (div_kind == Divergence::SC) {
      // Symmetric Chi-Square Term: (p-n)^2 * (p+n) / (p*n)
      bin.divergence = (dist_pos > EPSILON && dist_neg > EPSILON) ?
      std::pow(dist_pos - dist_neg, 2) * (dist_pos + dist_neg) / (dist_pos * dist_neg) :
      ((dist_pos + dist_neg > EPSILON) ? std::numeric_limits<double>::infinity() : 0.0); // Handle div by zero
      
    } else if (div_kind == Divergence::JS) {
      // Jensen-Shannon Term: 0.5 * [ p*log(2p/(p+n)) + n*log(2n/(p+n)) ]
      double m = (dist_pos + dist_neg) / 2.0;
      double js_p = (dist_pos > EPSILON && m > EPSILON) ? dist_pos * std::log(dist_pos / m) : 0.0;
      double js_n = (dist_neg > EPSILON && m > EPSILON) ? dist_neg * std::log(dist_neg / m) : 0.0;
      bin.divergence = 0.5 * (js_p + js_n);
    } else if (div_kind == Divergence::L1) {
      // L1 Term: |p-n|
      bin.divergence = std::abs(dist_pos - dist_neg);
    } else if (div_kind == Divergence::L2) {
      // L2 Intermediate Term: (p-n)^2
      bin.divergence = std::pow(dist_pos - dist_neg, 2);
    } else if (div_kind == Divergence::LN) {
      // L-infinity Intermediate Term: |p-n|
      bin.divergence = std::abs(dist_pos - dist_neg);
    }
    // Ensure non-negative divergence value
    if (!std::isinf(bin.divergence)) {
      bin.divergence = std::max(0.0, bin.divergence);
    }
  }
  
  /**
   * Compute bin metrics (WoE and divergence) for ALL bins.
   * Less efficient than single bin update but needed initially or after major changes.
   */
  void compute_bin_metrics() {
    for (auto& bin : bins) {
      compute_single_bin_metrics(bin);
    }
  }
  
  /**
   * Join category names with separator for display.
   * @param categories Vector of category names.
   * @return String of joined category names.
   */
  std::string join_categories(const std::vector<std::string>& categories) const {
    if (categories.empty()) return "EMPTY_BIN"; // Should not happen
    
    // Sort categories for consistent bin naming (optional, adds overhead)
    // std::vector<std::string> sorted_cats = categories;
    // std::sort(sorted_cats.begin(), sorted_cats.end());
    
    // Pre-allocate string with estimated size
    size_t estimated_size = 0;
    for (const auto& cat : categories) { // Use original order if not sorting
      estimated_size += cat.size();
    }
    estimated_size += bin_separator.size() * (categories.size() > 0 ? categories.size() - 1 : 0);
    
    std::string result;
    result.reserve(estimated_size);
    
    // Join categories
    bool first = true;
    for (const auto& cat : categories) { // Use original order if not sorting
      if (!first) {
        result += bin_separator;
      }
      result += cat;
      first = false;
    }
    return result;
  }
  
  /**
   * Prepare output List for R.
   * @return List containing binning results.
   */
  Rcpp::List prepare_output() const {
    const size_t n_bins = bins.size();
    Rcpp::StringVector bin_names(n_bins);
    Rcpp::NumericVector woe_values(n_bins);
    Rcpp::NumericVector divergence_values(n_bins); // Per-bin contribution/value
    Rcpp::IntegerVector bin_counts(n_bins);
    Rcpp::IntegerVector counts_pos(n_bins);
    Rcpp::IntegerVector counts_neg(n_bins);
    Rcpp::IntegerVector ids(n_bins);
    Rcpp::NumericVector iv_values(n_bins);

    // Standard Information Value, reported alongside the divergence measure.
    // It is computed directly from the smoothed class distributions rather than
    // from bins[i].woe, because the default bin_method "woe1" is Zeng's log-odds
    // ln((pos+0.5)/(neg+0.5)), which differs from standard WoE by the constant
    // ln(TP/TN); deriving IV from it would give a wrong value.
    const double iv_pos_denom = static_cast<double>(total_pos) + static_cast<double>(n_bins) * 0.5;
    const double iv_neg_denom = static_cast<double>(total_neg) + static_cast<double>(n_bins) * 0.5;

    for (size_t i = 0; i < n_bins; ++i) {
      ids[i] = static_cast<int>(i) + 1; // 1-based index for R
      bin_names[i] = join_categories(bins[i].categories);
      woe_values[i] = bins[i].woe;
      divergence_values[i] = bins[i].divergence; // Store per-bin value (or intermediate for L2/Ln)
      bin_counts[i] = bins[i].total();
      counts_pos[i] = bins[i].count_pos;
      counts_neg[i] = bins[i].count_neg;

      // iv_bin = (dist_pos - dist_neg) * ln(dist_pos / dist_neg)
      double iv_dist_pos = (static_cast<double>(bins[i].count_pos) + 0.5) / iv_pos_denom;
      double iv_dist_neg = (static_cast<double>(bins[i].count_neg) + 0.5) / iv_neg_denom;
      iv_values[i] = (iv_dist_pos - iv_dist_neg) * std::log(iv_dist_pos / iv_dist_neg);
    }

    double total_iv = 0.0;
    for (size_t i = 0; i < n_bins; ++i) {
      total_iv += iv_values[i];
    }

    // Calculate total divergence correctly based on the method
    double total_divergence = 0.0;
    if (div_kind == Divergence::L2) {
      double sum_sq_diff = 0.0;
      for (const auto& bin : bins) {
        sum_sq_diff += bin.divergence; // bin.divergence stores (p-n)^2
      }
      total_divergence = std::sqrt(sum_sq_diff);
    } else if (div_kind == Divergence::LN) {
      double max_abs_diff = 0.0;
      for (const auto& bin : bins) {
        max_abs_diff = std::max(max_abs_diff, bin.divergence); // bin.divergence stores |p-n|
      }
      total_divergence = max_abs_diff;
    } else {
      // For other methods, divergence is additive
      for (const auto& bin : bins) {
        if (!std::isinf(bin.divergence)) { // Avoid adding Inf
          total_divergence += bin.divergence;
        } else {
          total_divergence = std::numeric_limits<double>::infinity(); // If any part is Inf, total is Inf
          break;
        }
      }
    }
    
    
    return Rcpp::List::create(
      Rcpp::Named("id") = ids,
      Rcpp::Named("bin") = bin_names,
      Rcpp::Named("woe") = woe_values,
      Rcpp::Named("iv") = iv_values,
      Rcpp::Named("divergence") = divergence_values, // Per-bin value/contribution
      Rcpp::Named("count") = bin_counts,
      Rcpp::Named("count_pos") = counts_pos,
      Rcpp::Named("count_neg") = counts_neg,
      Rcpp::Named("converged") = converged,
      Rcpp::Named("iterations") = iterations_run,
      Rcpp::Named("total_divergence") = total_divergence, // Correct total divergence
      Rcpp::Named("total_iv") = total_iv,
      Rcpp::Named("bin_method") = bin_method,
      Rcpp::Named("divergence_method") = divergence_method
    );
  }
  
public:
  /**
   * Constructor for OBC_DMIV
   * (Parameters descriptions omitted for brevity, see Rcpp wrapper doc)
   */
  OBC_DMIV(
    const std::vector<std::string>& feature_,
    const std::vector<int>& target_,
    int min_bins_ = 3,
    int max_bins_ = 5,
    double bin_cutoff_ = 0.05,
    int max_n_prebins_ = 20,
    const std::string& bin_separator_ = "%;%",
    double convergence_threshold_ = 1e-6,
    int max_iterations_ = 1000,
    std::string bin_method_ = "woe1",
    std::string divergence_method_ = "l2"
  ) : feature(feature_),
  target(target_),
  min_bins(min_bins_),
  max_bins(max_bins_),
  bin_cutoff(bin_cutoff_),
  max_n_prebins(max_n_prebins_),
  bin_separator(bin_separator_),
  convergence_threshold(convergence_threshold_),
  max_iterations(max_iterations_),
  bin_method(bin_method_),
  divergence_method(divergence_method_)
  // Other members initialized inline or in validate_inputs/initialize_bins
  {
    // Constructor body can be empty if all initialization is done via initializer list
    // or subsequent method calls within fit().
  }
  
  
  /// Number of categories whose name contains bin_separator (and one of them)
  std::size_t separator_hits(std::string& example) const {
    return count_separator_hits(total_count_map, bin_separator, example);
  }

  /**
   * Execute the optimal binning algorithm (v2)
   * @return List with binning results
   */
  Rcpp::List fit() {
    try {
      // Step 1: Validate inputs and calculate initial counts
      validate_inputs();
      
      // Step 2: Initialize bins (potentially pre-binning rare categories)
      initialize_bins();
      
      // Step 3: Perform optimal merging based on divergence measure
      perform_binning(); // Merges down to max_bins or convergence
      
      // (min_bins is guaranteed by construction; see perform_binning().)
      
      // Step 4: Compute final metrics for all bins
      compute_bin_metrics();
      
      // Step 5: Sort final bins by WoE for better interpretability
      sort_bins_by_woe();
      
      // Step 6: Finalize and return results
      return prepare_output();
      
    } catch (const std::exception& e) {
      // Catch standard exceptions and report via Rcpp::stop
      Rcpp::stop("Error in optimal binning v2: " + std::string(e.what()));
    } catch (...) {
      // Catch any other unknown exceptions
      Rcpp::stop("Unknown error occurred during optimal binning v2.");
    }
  }
}; // End class OBC_DMIV


// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_dmiv(
   Rcpp::IntegerVector target,
   Rcpp::CharacterVector feature,
   int min_bins = 3,
   int max_bins = 5,
   double bin_cutoff = 0.05,
   int max_n_prebins = 20,
   std::string bin_separator = "%;%",
   double convergence_threshold = 1e-6,
   int max_iterations = 1000,
   std::string bin_method = "woe1",
   std::string divergence_method = "l2"
) {
 try {
   // Convert R vectors to C++ vectors and handle NAs
   R_xlen_t n = feature.size();
   if (n != target.size()) {
     Rcpp::stop("Feature and target must have the same length.");
   }
   std::vector<std::string> feature_vec(n);
   std::vector<int> target_vec(n);
   
   for (R_xlen_t i = 0; i < n; ++i) {
     // Handle NAs in feature vector -> Treat as category "NA"
     if (CharacterVector::is_na(feature[i])) {
       feature_vec[i] = "NA";
     } else {
       feature_vec[i] = Rcpp::as<std::string>(feature[i]);
     }
     
     // Check NAs in target vector -> Error
     if (IntegerVector::is_na(target[i])) {
       Rcpp::stop("Target variable cannot contain missing values (NA).");
     }
     target_vec[i] = target[i];
   }
   
   // Create algorithm object (V2)
   OBC_DMIV obcat_v2(
       feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins,
       bin_separator, convergence_threshold, max_iterations, bin_method,
       divergence_method
   );
   
   // Execute algorithm and return results
   Rcpp::List res = obcat_v2.fit();
   std::string example;
   warn_separator_hits(bin_separator, obcat_v2.separator_hits(example), example);
   return res;
   
 } catch (const std::exception& e) {
   Rcpp::stop("Error in optimal binning v2: " + std::string(e.what()));
 } catch (...) {
   Rcpp::stop("Unknown error occurred during optimal binning v2.");
 }
}
