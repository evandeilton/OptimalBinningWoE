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
#include <queue>
#include <cstddef>

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

  // -Inf and +Inf are ordinary extreme values: they belong to the first and
  // last bin, and a cut must never isolate them (its cutpoint would be -Inf,
  // or the interval above the last finite value would be (x, +Inf] anyway).
  // Fold them into the adjacent finite block.
  if (b.value.size() >= 2 && std::isinf(b.value.front()) && b.value.front() < 0) {
    b.value.erase(b.value.begin());
    b.cpos.erase(b.cpos.begin() + 1);
    b.cneg.erase(b.cneg.begin() + 1);
    b.cend.erase(b.cend.begin() + 1);
  }
  if (b.value.size() >= 2 && std::isinf(b.value.back()) && b.value.back() > 0) {
    b.value.pop_back();
    b.cpos.erase(b.cpos.end() - 2);
    b.cneg.erase(b.cneg.end() - 2);
    b.cend.erase(b.cend.end() - 2);
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
 * Table of c * log2(c) for c = 0..n (0 log 0 = 0).
 *
 * With it, t * E(S) for a set with p positives and n negatives, t = p + n, is
 *   A(p, n) = t log2 t - p log2 p - n log2 n
 * -- three lookups and no logarithm. The candidate scan of every MDLP
 * partition needs this quantity on both sides of every cut, so the table
 * removes all transcendental calls from the O(candidates) inner loop.
 */
struct XLogX {
  std::vector<double> tab;
  explicit XLogX(int n) : tab(static_cast<size_t>(n) + 1, 0.0) {
    for (int c = 2; c <= n; ++c) {
      const double x = static_cast<double>(c);
      tab[static_cast<size_t>(c)] = x * std::log2(x);
    }
  }
  // t * E(S): the entropy of the set times its size, in bits.
  double A(int p, int n) const {
    return tab[static_cast<size_t>(p + n)] - tab[static_cast<size_t>(p)] - tab[static_cast<size_t>(n)];
  }
};

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

/** An MDL-accepted cut of the block range [bstart, bend). */
struct Cut {
  double gain_bits;  // N * Gain(A, T; S): entropy reduction in bits
  int bstart;
  int bend;
  int cut;           // cut after this block
};

struct CutOrder {
  // Max-heap on the entropy reduction; ties go to the leftmost cut.
  bool operator()(const Cut& a, const Cut& b) const {
    if (a.gain_bits != b.gain_bits) return a.gain_bits < b.gain_bits;
    return a.cut > b.cut;
  }
};

/**
 * Best cut of the block range [bstart, bend) and the MDLP acceptance test of
 * Fayyad & Irani (1993). Returns false when the range has no candidate cut or
 * the best one is rejected.
 *
 * The best cut maximises Gain(A, T; S) = E(S) - E(A, T; S), i.e. minimises
 * t1 E(S1) + t2 E(S2); only boundary points are evaluated. It is accepted iff
 *
 *   Gain(A, T; S) > log2(N - 1) / N + Delta(A, T; S) / N,
 *   Delta(A, T; S) = log2(3^k - 2) - [k E(S) - k1 E(S1) - k2 E(S2)],
 *
 * with k, k1, k2 the number of classes present in S, S1 and S2.
 */
bool best_mdlp_cut(const Blocks& b, const XLogX& xl, int bstart, int bend, Cut& out) {
  const size_t s = static_cast<size_t>(bstart);
  const size_t e = static_cast<size_t>(bend);
  const int N = b.cend[e] - b.cend[s];
  if (N <= 1) return false;

  const int pos_total = b.cpos[e] - b.cpos[s];
  const int neg_total = b.cneg[e] - b.cneg[s];
  if (pos_total == 0 || neg_total == 0) return false;  // pure: nothing to gain

  double best = std::numeric_limits<double>::infinity();
  int best_cut = -1;
  for (int j = bstart; j < bend - 1; ++j) {
    if (!is_boundary(b, j)) continue;
    const int pl = b.cpos[static_cast<size_t>(j) + 1] - b.cpos[s];
    const int nl = b.cneg[static_cast<size_t>(j) + 1] - b.cneg[s];
    const double child = xl.A(pl, nl) + xl.A(pos_total - pl, neg_total - nl);
    if (child < best) {
      best = child;
      best_cut = j;
    }
  }
  if (best_cut < 0) return false;

  const int pl = b.cpos[static_cast<size_t>(best_cut) + 1] - b.cpos[s];
  const int nl = b.cneg[static_cast<size_t>(best_cut) + 1] - b.cneg[s];
  const int pr = pos_total - pl;
  const int nr = neg_total - nl;
  const int n1 = pl + nl;
  const int n2 = pr + nr;

  const double A_S = xl.A(pos_total, neg_total);
  const double A_1 = xl.A(pl, nl);
  const double A_2 = xl.A(pr, nr);
  const double Nd = static_cast<double>(N);

  const double E_S = A_S / Nd;
  const double E_1 = A_1 / static_cast<double>(n1);
  const double E_2 = A_2 / static_cast<double>(n2);
  const double gain = (A_S - A_1 - A_2) / Nd;

  const int k = 2;  // S holds both classes (pure sets returned above)
  const int k1 = (pl > 0 ? 1 : 0) + (nl > 0 ? 1 : 0);
  const int k2 = (pr > 0 ? 1 : 0) + (nr > 0 ? 1 : 0);
  const double delta = std::log2(std::pow(3.0, k) - 2.0) -
    (k * E_S - k1 * E_1 - k2 * E_2);
  const double threshold = (std::log2(Nd - 1.0) + delta) / Nd;

  if (!(gain > threshold)) return false;

  out.gain_bits = A_S - A_1 - A_2;
  out.bstart = bstart;
  out.bend = bend;
  out.cut = best_cut;
  return true;
}

/**
 * Multi-interval MDLP (Fayyad & Irani, 1993), best-first.
 *
 * Every accepted cut splits its interval in two, and each half is examined in
 * turn, exactly as in the recursive formulation; the set of cuts is the same
 * whenever it fits into max_bins. Expanding the intervals in order of the
 * entropy reduction their cut achieves (a priority queue instead of depth-first
 * recursion) makes the truncation to max_bins keep the most informative cuts
 * instead of whichever ones happened to be found first, and needs no call
 * stack proportional to the number of cuts.
 */
std::vector<int> mdlp_cuts(const Blocks& b, int max_splits) {
  std::vector<int> splits;
  if (max_splits <= 0 || b.m() < 2) return splits;

  const XLogX xl(b.n());
  std::priority_queue<Cut, std::vector<Cut>, CutOrder> heap;
  Cut c;
  if (best_mdlp_cut(b, xl, 0, b.m(), c)) heap.push(c);

  while (!heap.empty() && static_cast<int>(splits.size()) < max_splits) {
    const Cut top = heap.top();
    heap.pop();
    splits.push_back(top.cut);
    Cut child;
    if (best_mdlp_cut(b, xl, top.bstart, top.cut + 1, child)) heap.push(child);
    if (best_mdlp_cut(b, xl, top.cut + 1, top.bend, child)) heap.push(child);
  }
  std::sort(splits.begin(), splits.end());
  return splits;
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
  // When the widest gap holds no such boundary, the next widest is tried; the
  // loop only gives up when no gap can be split, i.e. every distinct-value
  // boundary is already a split.
  struct Gap { int size; int start; bool first; };
  while (splits.size() + 1 < static_cast<size_t>(min_bins)) {
    // Implicit boundaries at raw indices 0 and N-1.
    std::vector<int> all_boundaries;
    all_boundaries.reserve(splits.size() + 2);
    all_boundaries.push_back(0);
    for (int s : splits) all_boundaries.push_back(b.last_raw(s));
    all_boundaries.push_back(N - 1);

    std::vector<Gap> gaps;
    gaps.reserve(all_boundaries.size());
    for (size_t i = 0; i + 1 < all_boundaries.size(); i++) {
      gaps.push_back(Gap{all_boundaries[i + 1] - all_boundaries[i], all_boundaries[i], i == 0});
    }
    // Widest first; among equal widths, leftmost first.
    std::stable_sort(gaps.begin(), gaps.end(),
                     [](const Gap& x, const Gap& y) { return x.size > y.size; });

    int chosen = -1;
    for (const Gap& g : gaps) {
      if (g.size < 1) continue;
      const int gap_start = g.start;
      const int gap_end = g.start + g.size;
      // The implicit boundary 0 is not a cut: a cut right after the first
      // observation is admissible in the first gap.
      const int lo = g.first ? -1 : gap_start;
      const int new_split = gap_start + g.size / 2;
      // A valid cut ends a block. Prefer the end of the block holding the
      // midpoint; if that block reaches the end of the gap, fall back to the
      // end of the block before it.
      const int blk = b.block_of_raw(new_split);
      int adj = b.last_raw(blk);
      if (adj > lo && adj < gap_end && adj < N - 1) {
        chosen = blk;
      } else if (blk > 0) {
        adj = b.last_raw(blk - 1);
        if (adj > lo && adj < N - 1) chosen = blk - 1;
      }
      if (chosen >= 0) break;
    }
    if (chosen < 0) break;
    splits.push_back(chosen);
    std::sort(splits.begin(), splits.end());
  }
}

/**
 * Smoothed WoE / IV of every bin from its class counts. The smoothing adds
 * ALPHA to each bin count and ALPHA * (n_bins + 1) to each class total, the
 * convention calc_bins_metrics() uses.
 */
void recompute_woe_iv(const std::vector<int>& pos_counts,
                      const std::vector<int>& neg_counts,
                      std::vector<double>& woe,
                      std::vector<double>& iv) {
  const double ALPHA = 0.5;
  int total_pos = 0, total_neg = 0;
  for (size_t i = 0; i < pos_counts.size(); ++i) {
    total_pos += pos_counts[i];
    total_neg += neg_counts[i];
  }
  const double n_bnd = static_cast<double>(pos_counts.size() + 1);
  woe.resize(pos_counts.size());
  iv.resize(pos_counts.size());
  for (size_t i = 0; i < pos_counts.size(); ++i) {
    const double pct_pos = (pos_counts[i] + ALPHA) / (total_pos + ALPHA * n_bnd);
    const double pct_neg = (neg_counts[i] + ALPHA) / (total_neg + ALPHA * n_bnd);
    woe[i] = std::log(pct_pos / pct_neg);
    iv[i] = (pct_pos - pct_neg) * woe[i];
  }
}

/**
 * Merge adjacent bins until the WoE is strictly monotone (or min_bins bins
 * remain), always merging the adjacent pair with the closest WoE.
 *
 * The bins are merged in place: counts are added, the cutpoint between the two
 * bins is removed and every WoE / IV is recomputed from the merged counts.
 */
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
  if (!force_monotonicity) return true;

  while (woe.size() > 1) {
    if (is_strictly_increasing(woe) || is_strictly_decreasing(woe)) return true;
    if (static_cast<int>(woe.size()) <= min_bins) return true;

    size_t merge_pos = 0;
    double min_diff = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i + 1 < woe.size(); ++i) {
      const double diff = std::fabs(woe[i + 1] - woe[i]);
      if (diff < min_diff) {
        min_diff = diff;
        merge_pos = i;
      }
    }

    const auto nxt = static_cast<std::ptrdiff_t>(merge_pos) + 1;
    counts[merge_pos] += counts[merge_pos + 1];
    pos_counts[merge_pos] += pos_counts[merge_pos + 1];
    neg_counts[merge_pos] += neg_counts[merge_pos + 1];
    counts.erase(counts.begin() + nxt);
    pos_counts.erase(pos_counts.begin() + nxt);
    neg_counts.erase(neg_counts.begin() + nxt);
    cutpoints.erase(cutpoints.begin() + static_cast<std::ptrdiff_t>(merge_pos));
    recompute_woe_iv(pos_counts, neg_counts, woe, iv);
  }
  return true;
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

 // Numerical NA contract: a missing target is an error; rows whose feature is
 // NA / NaN are excluded; -Inf / +Inf are extreme values of the first / last
 // bin (see build_blocks()).
 const R_xlen_t n_in = target.size();
 for (R_xlen_t i = 0; i < n_in; i++) {
   if (Rcpp::IntegerVector::is_na(target[i])) {
     Rcpp::stop("Target contains missing values (NA).");
   }
   if (target[i] != 0 && target[i] != 1) {
     Rcpp::stop("Target must be binary (0/1).");
   }
 }

 std::vector<double> pos_vals, neg_vals;
 for (R_xlen_t i = 0; i < n_in; i++) {
   if (Rcpp::NumericVector::is_na(feature[i])) continue;
   if (target[i] == 1) pos_vals.push_back(feature[i]);
   else if (target[i] == 0) neg_vals.push_back(feature[i]);
 }

 const Blocks b = build_blocks(pos_vals, neg_vals);
 const int N = b.n();
 if (N == 0) {
   Rcpp::stop("Feature has no non-missing values.");
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
   splits = mdlp_cuts(b, max_bins - 1);
   if (static_cast<int>(splits.size()) + 1 < min_bins) force_min_bins(b, min_bins, splits);

   calc_bins_metrics(b, splits, counts, pos_counts, neg_counts, woe, iv, cutpoints);

   // enforce_monotonicity() finishes in one call; the loop and the WoE-change
   // test are kept so that `iterations` / `converged` keep their meaning.
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
