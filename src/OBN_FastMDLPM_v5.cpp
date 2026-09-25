#include <Rcpp.h>

// Include shared headers
#include "common/optimal_binning_common.h"
#include "common/bin_structures.h"

using namespace Rcpp;
using namespace OptimalBinning;

#include <algorithm>
#include <vector>
#include <cmath>
#include <string>
#include <limits>
#include <stdexcept>

/**
 * @title Optimal Binning using Minimum Description Length Principle with Monotonicity
 * @description Implementation of numerical variable binning using the Minimum Description Length
 * Principle (MDLP) with optional monotonicity constraints on the Weight of Evidence (WoE).
 *
 * The algorithm recursively partitions the feature space by finding cut points that
 * maximize information gain, subject to the MDLP criterion that determines whether a cut is
 * justified based on information theory principles. The monotonicity constraint ensures that
 * the WoE values across bins follow a monotonic (strictly increasing or decreasing) pattern,
 * which is often desirable in credit risk modeling applications.
 *
 * Data layout. The sorted sample is compressed once into "blocks": one block per
 * distinct feature value, carrying the number of positive and negative cases that
 * share that value, plus prefix sums over the blocks. A cut can only fall between
 * two distinct values, so every quantity the algorithm needs -- class counts on
 * either side of a cut, bin counts, cutpoints -- is an O(1) prefix-sum lookup on
 * the blocks. Nothing downstream of the sort touches the raw observations again.
 *
 * @references
 * - Fayyad, U., & Irani, K. (1993). Multi-interval discretization of continuous-valued
 *   attributes for classification learning. Proceedings of the 13th International
 *   Joint Conference on Artificial Intelligence, 1022-1027.
 * - Fayyad, U., & Irani, K. (1992). On the handling of continuous-valued attributes
 *   in decision tree generation. Machine Learning, 8, 87-102.
 */

namespace {

// ----------------------------------------------------
// Sorted, tie-compressed sample
// ----------------------------------------------------

/**
 * One block per distinct feature value, in ascending order.
 *
 * cpos / cneg / cend are prefix sums of size m + 1: cpos[j] is the number of
 * positive cases in blocks [0, j), cend[j] the number of observations in them.
 * A cut "after block j" therefore separates observations [0, cend[j+1]) from
 * [cend[j+1], N), exactly the raw-index split cend[j+1] - 1 of the sorted data.
 */
struct Blocks {
  std::vector<double> value;
  std::vector<int> cpos;
  std::vector<int> cneg;
  std::vector<int> cend;

  int m() const { return static_cast<int>(value.size()); }
  int n() const { return cend.empty() ? 0 : cend.back(); }
  int pos(int j) const { return cpos[static_cast<size_t>(j) + 1] - cpos[static_cast<size_t>(j)]; }
  int neg(int j) const { return cneg[static_cast<size_t>(j) + 1] - cneg[static_cast<size_t>(j)]; }
  // Raw sorted index of the last observation of block j.
  int last_raw(int j) const { return cend[static_cast<size_t>(j) + 1] - 1; }
  // Block holding raw sorted index r (0 <= r < n()).
  int block_of_raw(int r) const {
    auto it = std::upper_bound(cend.begin(), cend.end(), r);
    return static_cast<int>(it - cend.begin()) - 1;
  }
};

/**
 * Build the blocks from the positive and negative feature values.
 *
 * Sorting the two classes separately and merging them is both cheaper than an
 * indirect sort of (value, class) pairs and exactly equivalent: the class of an
 * observation only matters through the counts per distinct value.
 */
Blocks build_blocks(std::vector<double>& pos_vals, std::vector<double>& neg_vals) {
  std::sort(pos_vals.begin(), pos_vals.end());
  std::sort(neg_vals.begin(), neg_vals.end());

  Blocks b;
  const size_t np = pos_vals.size();
  const size_t nn = neg_vals.size();
  b.cpos.push_back(0);
  b.cneg.push_back(0);
  b.cend.push_back(0);

  size_t i = 0, k = 0;
  while (i < np || k < nn) {
    double v;
    if (k >= nn || (i < np && pos_vals[i] < neg_vals[k])) {
      v = pos_vals[i];
    } else {
      v = neg_vals[k];
    }
    int cp = 0, cn = 0;
    while (i < np && pos_vals[i] == v) { ++i; ++cp; }
    while (k < nn && neg_vals[k] == v) { ++k; ++cn; }
    // -0.0 and +0.0 are the same value; report it without a sign.
    b.value.push_back(v + 0.0);
    b.cpos.push_back(b.cpos.back() + cp);
    b.cneg.push_back(b.cneg.back() + cn);
    b.cend.push_back(b.cend.back() + cp + cn);
  }
  return b;
}

// ----------------------------------------------------
// Helper functions
// ----------------------------------------------------

bool is_strictly_increasing(const std::vector<double> &x) {
  for (size_t i = 1; i < x.size(); i++) {
    if (x[i] <= x[i-1]) return false;
  }
  return true;
}

bool is_strictly_decreasing(const std::vector<double> &x) {
  for (size_t i = 1; i < x.size(); i++) {
    if (x[i] >= x[i-1]) return false;
  }
  return true;
}

/**
 * Binary entropy (bits): E = -p log2 p - q log2 q, with 0 log 0 = 0.
 */
double entropy(int count_pos, int count_neg) {
  int total = count_pos + count_neg;
  if (total == 0) return 0.0;

  double p_pos = static_cast<double>(count_pos) / static_cast<double>(total);
  double p_neg = static_cast<double>(count_neg) / static_cast<double>(total);

  double E = 0.0;
  if (p_pos > 0) E -= p_pos * std::log2(p_pos);
  if (p_neg > 0) E -= p_neg * std::log2(p_neg);
  return E;
}

/**
 * Class-weighted entropy of a binary split:
 * E(T; S) = |S1|/|S| E(S1) + |S2|/|S| E(S2).
 */
double conditional_entropy(int pos_left, int neg_left, int pos_right, int neg_right) {
  int total_left = pos_left + neg_left;
  int total_right = pos_right + neg_right;
  int total = total_left + total_right;
  if (total == 0) return 0.0;

  double E_left = entropy(pos_left, neg_left);
  double E_right = entropy(pos_right, neg_right);

  double weight_left = static_cast<double>(total_left) / static_cast<double>(total);
  double weight_right = static_cast<double>(total_right) / static_cast<double>(total);
  return weight_left * E_left + weight_right * E_right;
}

/**
 * MDLP stopping criterion. Returns true when the split must NOT be made.
 */
bool mdlp_stop_criterion(int pos_left, int neg_left, int pos_right, int neg_right,
                         int total_pos, int total_neg) {
  double E_parent = entropy(total_pos, total_neg);
  double E_child = conditional_entropy(pos_left, neg_left, pos_right, neg_right);
  double IG = E_parent - E_child;
  double Delta = std::log2(7.0) - 2.0 * E_parent;
  int N = total_pos + total_neg;
  double threshold = (std::log2(static_cast<double>(N - 1)) / N) + (Delta / N);
  return (IG <= threshold);
}

/**
 * A cut after block j is a boundary point in the sense of Fayyad & Irani
 * (1992) unless blocks j and j+1 are both pure and of the same class. The
 * entropy-minimising cut of any interval is always a boundary point, so the
 * others never need to be evaluated.
 */
inline bool is_boundary(const Blocks& b, int j) {
  const int p0 = b.pos(j), n0 = b.neg(j);
  const int p1 = b.pos(j + 1), n1 = b.neg(j + 1);
  const bool pure_pos = (n0 == 0 && n1 == 0);
  const bool pure_neg = (p0 == 0 && p1 == 0);
  return !(pure_pos || pure_neg);
}

/**
 * Recursive MDLP on blocks [bstart, bend). Accepted cuts are appended to
 * `splits` as block indices (cut after that block).
 */
void mdlp_recursion(const Blocks& b, int bstart, int bend, std::vector<int>& splits) {
  const int start = b.cend[static_cast<size_t>(bstart)];
  const int end = b.cend[static_cast<size_t>(bend)];
  if ((end - start) <= 1) return;

  const int pos_total = b.cpos[static_cast<size_t>(bend)] - b.cpos[static_cast<size_t>(bstart)];
  const int neg_total = b.cneg[static_cast<size_t>(bend)] - b.cneg[static_cast<size_t>(bstart)];
  if (pos_total == 0 || neg_total == 0) return;

  double best_IG = -std::numeric_limits<double>::infinity();
  int best_split = -1;
  const double E_parent = entropy(pos_total, neg_total);

  for (int j = bstart; j < bend - 1; j++) {
    if (!is_boundary(b, j)) continue;
    const int pos_left = b.cpos[static_cast<size_t>(j) + 1] - b.cpos[static_cast<size_t>(bstart)];
    const int neg_left = b.cneg[static_cast<size_t>(j) + 1] - b.cneg[static_cast<size_t>(bstart)];
    const double E_child = conditional_entropy(pos_left, neg_left,
                                               pos_total - pos_left, neg_total - neg_left);
    const double IG = E_parent - E_child;
    if (IG > best_IG) {
      best_IG = IG;
      best_split = j;
    }
  }
  if (best_split == -1) return;

  const int pos_left = b.cpos[static_cast<size_t>(best_split) + 1] - b.cpos[static_cast<size_t>(bstart)];
  const int neg_left = b.cneg[static_cast<size_t>(best_split) + 1] - b.cneg[static_cast<size_t>(bstart)];
  if (mdlp_stop_criterion(pos_left, neg_left, pos_total - pos_left, neg_total - neg_left,
                          pos_total, neg_total)) {
    return;
  }

  splits.push_back(best_split);
  mdlp_recursion(b, bstart, best_split + 1, splits);
  mdlp_recursion(b, best_split + 1, bend, splits);
}

/**
 * Counts, smoothed WoE / IV and cutpoints of the bins delimited by `splits`
 * (sorted block indices).
 */
void calc_bins_metrics(const Blocks& b,
                       const std::vector<int>& splits,
                       std::vector<int>& counts,
                       std::vector<int>& pos_counts,
                       std::vector<int>& neg_counts,
                       std::vector<double>& woe,
                       std::vector<double>& iv,
                       std::vector<double>& cutpoints) {
  const double ALPHA = 0.5;

  counts.clear();
  pos_counts.clear();
  neg_counts.clear();
  woe.clear();
  iv.clear();
  cutpoints.clear();

  const int total_pos = b.cpos.back();
  const int total_neg = b.cneg.back();

  // Bin boundaries as block indices: bin i covers blocks [bnd[i], bnd[i+1]).
  std::vector<int> bnd;
  bnd.reserve(splits.size() + 2);
  bnd.push_back(0);
  for (size_t i = 0; i < splits.size(); i++) bnd.push_back(splits[i] + 1);
  bnd.push_back(b.m());

  const double n_bnd = static_cast<double>(bnd.size());
  for (size_t i = 0; i + 1 < bnd.size(); i++) {
    const size_t s = static_cast<size_t>(bnd[i]);
    const size_t e = static_cast<size_t>(bnd[i + 1]);
    const int c_pos = b.cpos[e] - b.cpos[s];
    const int c_neg = b.cneg[e] - b.cneg[s];

    counts.push_back(c_pos + c_neg);
    pos_counts.push_back(c_pos);
    neg_counts.push_back(c_neg);

    const double pct_pos = (c_pos + ALPHA) / (total_pos + ALPHA * n_bnd);
    const double pct_neg = (c_neg + ALPHA) / (total_neg + ALPHA * n_bnd);
    const double w = std::log(pct_pos / pct_neg);
    woe.push_back(w);
    iv.push_back((pct_pos - pct_neg) * w);

    if (i + 2 < bnd.size()) cutpoints.push_back(b.value[e - 1]);
  }
}

/**
 * Raise the number of bins to min_bins when MDLP found fewer.
 */
void force_min_bins(const Blocks& b, int min_bins, std::vector<int>& splits) {
  if (static_cast<int>(splits.size()) + 1 >= min_bins) return;
  splits.clear();

  const int N = b.n();
  const int n_unique = b.m();

  if (n_unique < min_bins) {
    // Fewer distinct values than requested bins: one bin per distinct value is
    // the finest partition whose counts can be reproduced from its cutpoints.
    for (int j = 0; j + 1 < n_unique; j++) splits.push_back(j);
    return;
  }

  // Splits at approximately equal intervals in the distinct values.
  const double step = static_cast<double>(n_unique - 1) / static_cast<double>(min_bins - 1);
  for (int i = 1; i < min_bins; i++) {
    const int unique_idx = static_cast<int>(std::floor(i * step));
    if (unique_idx < n_unique - 1) {
      // Raw index of the last observation holding this value.
      const int idx = b.last_raw(unique_idx);
      if (idx > 0 && idx < N - 1) splits.push_back(unique_idx);
    }
  }
  std::sort(splits.begin(), splits.end());
  splits.erase(std::unique(splits.begin(), splits.end()), splits.end());

  // Still short: bisect the widest gap (in observations) between existing
  // splits, moving the cut to the nearest boundary between distinct values.
  while (splits.size() + 1 < static_cast<size_t>(min_bins)) {
    // Implicit boundaries at raw indices 0 and N-1.
    std::vector<int> all_boundaries;
    all_boundaries.reserve(splits.size() + 2);
    all_boundaries.push_back(0);
    for (int s : splits) all_boundaries.push_back(b.last_raw(s));
    all_boundaries.push_back(N - 1);

    int largest_gap = 0;
    int gap_start = 0;
    for (size_t i = 0; i + 1 < all_boundaries.size(); i++) {
      const int gap = all_boundaries[i + 1] - all_boundaries[i];
      if (gap > largest_gap) {
        largest_gap = gap;
        gap_start = all_boundaries[i];
      }
    }
    if (largest_gap <= 1) break;

    const int new_split = gap_start + largest_gap / 2;
    const int gap_end = gap_start + largest_gap;
    // A valid cut ends a block. Prefer the end of the block holding the
    // midpoint; if that block reaches the end of the gap, fall back to the end
    // of the block before it.
    const int blk = b.block_of_raw(new_split);
    int adj = b.last_raw(blk);
    int chosen = -1;
    if (adj < gap_end && adj < N - 1) {
      chosen = blk;
    } else if (blk > 0) {
      adj = b.last_raw(blk - 1);
      if (adj > gap_start && adj > 0 && adj < N - 1) chosen = blk - 1;
    }
    if (chosen < 0) break;
    splits.push_back(chosen);
    std::sort(splits.begin(), splits.end());
  }
}

/**
 * Reduce to max_bins by dropping the rightmost splits.
 */
void enforce_max_bins(int max_bins, std::vector<int>& splits) {
  if (static_cast<int>(splits.size()) + 1 <= max_bins) return;
  std::sort(splits.begin(), splits.end());
  while (splits.size() + 1 > static_cast<size_t>(max_bins) && !splits.empty()) {
    splits.pop_back();
  }
}

bool enforce_monotonicity(
    std::vector<int> &counts,
    std::vector<int> &pos_counts,
    std::vector<int> &neg_counts,
    std::vector<double> &woe,
    std::vector<double> &iv,
    std::vector<double> &cutpoints,
    bool force_monotonicity,
    int min_bins
) {
  if (woe.size() <= 1) return true;
  if (is_strictly_increasing(woe) || is_strictly_decreasing(woe)) return true;
  if (!force_monotonicity) return true;

  auto recompute_metrics = [&](const std::vector<int> &boundaries) {
    std::vector<int> new_counts;
    std::vector<int> new_pos_counts;
    std::vector<int> new_neg_counts;
    std::vector<double> new_woe;
    std::vector<double> new_iv;
    std::vector<double> new_cutpoints;
    const double ALPHA = 0.5;
    int total_pos = 0;
    int total_neg = 0;
    for (size_t i = 0; i < pos_counts.size(); i++) {
      total_pos += pos_counts[i];
      total_neg += neg_counts[i];
    }
    for (size_t i = 0; i < boundaries.size() - 1; i++) {
      int c_pos = 0;
      int c_neg = 0;
      for (int bb = boundaries[i]; bb < boundaries[i + 1]; bb++) {
        c_pos += pos_counts[bb];
        c_neg += neg_counts[bb];
      }
      new_counts.push_back(c_pos + c_neg);
      new_pos_counts.push_back(c_pos);
      new_neg_counts.push_back(c_neg);
      double pct_pos = (c_pos + ALPHA) / (total_pos + ALPHA * boundaries.size());
      double pct_neg = (c_neg + ALPHA) / (total_neg + ALPHA * boundaries.size());
      double w = std::log(pct_pos / pct_neg);
      new_woe.push_back(w);
      new_iv.push_back((pct_pos - pct_neg) * w);
      if (i < boundaries.size() - 2) {
        int last_bin_idx = boundaries[i + 1] - 1;
        if (last_bin_idx >= 0 && last_bin_idx < static_cast<int>(cutpoints.size())) {
          new_cutpoints.push_back(cutpoints[last_bin_idx]);
        } else if (!cutpoints.empty()) {
          new_cutpoints.push_back(cutpoints.back());
        }
      }
    }
    counts = new_counts;
    pos_counts = new_pos_counts;
    neg_counts = new_neg_counts;
    woe = new_woe;
    iv = new_iv;
    cutpoints = new_cutpoints;
  };

  int n_bins = static_cast<int>(woe.size());
  std::vector<int> boundaries;
  for (int i = 0; i <= n_bins; i++) boundaries.push_back(i);

  const int MAX_ITERATIONS = 1000;
  for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
    if (woe.size() <= 1) return true;
    if (is_strictly_increasing(woe) || is_strictly_decreasing(woe)) return true;
    int current_bins = static_cast<int>(woe.size());
    if (current_bins <= min_bins) return true;
    double min_diff = std::numeric_limits<double>::infinity();
    int merge_pos = -1;
    for (int i = 0; i < current_bins - 1; i++) {
      double diff = std::fabs(woe[i + 1] - woe[i]);
      if (diff < min_diff) {
        min_diff = diff;
        merge_pos = i;
      }
    }
    if (merge_pos >= 0) {
      boundaries.erase(boundaries.begin() + merge_pos + 1);
      recompute_metrics(boundaries);
    } else {
      return false;
    }
    if (static_cast<int>(woe.size()) < min_bins) return true;
  }
  return false;
}

Rcpp::CharacterVector make_bin_names(const std::vector<double>& cutpoints, size_t nb) {
  Rcpp::CharacterVector bin_names(nb);
  double lower = -std::numeric_limits<double>::infinity();
  for (size_t b = 0; b < nb; b++) {
    double upper = (b < cutpoints.size()) ? cutpoints[b] : std::numeric_limits<double>::infinity();
    std::string interval = "("
      + (std::isinf(lower) ? std::string("-Inf") : std::to_string(lower))
      + ";"
      + (std::isinf(upper) ? std::string("+Inf") : std::to_string(upper))
      + "]";
    bin_names[static_cast<R_xlen_t>(b)] = interval;
    lower = upper;
  }
  return bin_names;
}

} // namespace

// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_fast_mdlpm(
   Rcpp::IntegerVector target,
   Rcpp::NumericVector feature,
   int min_bins = 2,
   int max_bins = 5,
   double bin_cutoff = 0.05,
   int max_n_prebins = 100,
   double convergence_threshold = 1e-6,
   int max_iterations = 1000,
   bool force_monotonicity = true
) {
 (void) bin_cutoff;      // documented as reserved / unused
 (void) max_n_prebins;   // documented as reserved / unused

 if (target.size() != feature.size()) {
   Rcpp::stop("Target and feature must have the same length.");
 }
 if (min_bins < 2) {
   Rcpp::stop("min_bins must be >= 2.");
 }
 if (max_bins < min_bins) {
   Rcpp::stop("max_bins must be >= min_bins.");
 }

 const R_xlen_t n_in = target.size();
 bool is_binary = true;
 for (R_xlen_t i = 0; i < n_in; i++) {
   if (!Rcpp::IntegerVector::is_na(target[i]) && target[i] != 0 && target[i] != 1) {
     is_binary = false;
     break;
   }
 }
 if (!is_binary) {
   Rcpp::warning("Target variable should be binary (0/1). Non-binary values detected.");
 }

 // Drop NA / NaN, split the remaining feature values by class.
 std::vector<double> pos_vals, neg_vals;
 for (R_xlen_t i = 0; i < n_in; i++) {
   if (Rcpp::NumericVector::is_na(feature[i]) || Rcpp::IntegerVector::is_na(target[i])) continue;
   if (target[i] == 1) pos_vals.push_back(feature[i]);
   else if (target[i] == 0) neg_vals.push_back(feature[i]);
 }

 const Blocks b = build_blocks(pos_vals, neg_vals);
 const int N = b.n();
 if (N == 0) {
   Rcpp::warning("No valid data after removing NA values.");
   return Rcpp::List::create(
     Rcpp::Named("id") = Rcpp::NumericVector(),
     Rcpp::Named("bin") = Rcpp::CharacterVector(),
     Rcpp::Named("woe") = Rcpp::NumericVector(),
     Rcpp::Named("iv") = Rcpp::NumericVector(),
     Rcpp::Named("count") = Rcpp::IntegerVector(),
     Rcpp::Named("count_pos") = Rcpp::IntegerVector(),
     Rcpp::Named("count_neg") = Rcpp::IntegerVector(),
     Rcpp::Named("cutpoints") = Rcpp::NumericVector(),
     Rcpp::Named("converged") = false,
     Rcpp::Named("iterations") = 0
   );
 }

 std::vector<int> splits;
 std::vector<int> counts, pos_counts, neg_counts;
 std::vector<double> woe, iv, cutpoints;
 bool converged = false;
 int iterations = 0;

 if (b.m() == 1) {
   // A constant feature has exactly one bin: there is no cut that separates
   // two different values, so any "artificial" split would report counts that
   // cannot be reproduced from its own cutpoints.
   Rcpp::warning("All feature values are identical. Returning a single bin.");
   calc_bins_metrics(b, splits, counts, pos_counts, neg_counts, woe, iv, cutpoints);
   converged = true;
 } else {
   mdlp_recursion(b, 0, b.m(), splits);
   std::sort(splits.begin(), splits.end());

   if (static_cast<int>(splits.size()) + 1 < min_bins) force_min_bins(b, min_bins, splits);
   if (static_cast<int>(splits.size()) + 1 > max_bins) enforce_max_bins(max_bins, splits);

   calc_bins_metrics(b, splits, counts, pos_counts, neg_counts, woe, iv, cutpoints);

   std::vector<double> old_woe = woe;
   for (iterations = 0; iterations < max_iterations; iterations++) {
     bool mono_res = enforce_monotonicity(counts, pos_counts, neg_counts, woe, iv, cutpoints,
                                          force_monotonicity, min_bins);
     double diff = 0.0;
     {
       size_t len = std::min(old_woe.size(), woe.size());
       for (size_t i = 0; i < len; i++) diff += std::fabs(old_woe[i] - woe[i]);
       if (len > 0) diff /= static_cast<double>(len);
     }
     if (mono_res && diff < convergence_threshold) {
       converged = true;
       break;
     }
     old_woe = woe;
   }

   if (static_cast<int>(woe.size()) < min_bins) {
     splits.clear();
     for (int j = 0; j + 1 < b.m() && static_cast<int>(splits.size()) < min_bins - 1; j++) {
       splits.push_back(j);
     }
     if (splits.size() + 1 < static_cast<size_t>(min_bins)) {
       force_min_bins(b, min_bins, splits);
     }
     calc_bins_metrics(b, splits, counts, pos_counts, neg_counts, woe, iv, cutpoints);
   }
 }

 Rcpp::CharacterVector bin_names = make_bin_names(cutpoints, woe.size());
 Rcpp::NumericVector ids(bin_names.size());
 for (R_xlen_t i = 0; i < bin_names.size(); i++) ids[i] = static_cast<double>(i + 1);

 const int final_bins = static_cast<int>(woe.size());
 if (b.m() > 1 && (final_bins < min_bins || final_bins > max_bins)) {
   Rcpp::warning("The algorithm failed to respect min_bins/max_bins constraints. Resulted in %d bins instead of [%d,%d].",
                 final_bins, min_bins, max_bins);
 }

 return Rcpp::List::create(
   Rcpp::Named("id") = ids,
   Rcpp::Named("bin") = bin_names,
   Rcpp::Named("woe") = Rcpp::wrap(woe),
   Rcpp::Named("iv") = Rcpp::wrap(iv),
   Rcpp::Named("count") = Rcpp::wrap(counts),
   Rcpp::Named("count_pos") = Rcpp::wrap(pos_counts),
   Rcpp::Named("count_neg") = Rcpp::wrap(neg_counts),
   Rcpp::Named("cutpoints") = Rcpp::wrap(cutpoints),
   Rcpp::Named("converged") = converged,
   Rcpp::Named("iterations") = iterations
 );
}
