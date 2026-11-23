// [[Rcpp::plugins(cpp11)]]

#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <utility>


// Include shared headers
#include "common/optimal_binning_common.h"
#include "common/bin_structures.h"

using namespace Rcpp;
using namespace OptimalBinning;

namespace {

/**
 * @brief A finite cut c with a <= c < b between two consecutive distinct
 * values a < b, for the right-closed (lower, upper] convention.
 *
 * The midpoint (a + b) / 2 is used whenever it is valid, so ordinary data is
 * unaffected; it overflows to +/-Inf near the largest finite double and, for
 * two adjacent doubles, can round up to b. Then a/2 + b/2, and finally a, are
 * used instead.
 */
inline double oslp_safe_cut(double a, double b) {
  double m = (a + b) / 2.0;
  if (!std::isfinite(m)) m = a / 2.0 + b / 2.0;
  if (std::isfinite(m) && m >= a && m < b) return m;
  if (std::isfinite(a)) return a;
  // a == -Inf: the only finite value that keeps -Inf alone is the lowest double
  return std::numeric_limits<double>::lowest();
}

/**
 * Significance level of the pure-bin evidence rule (audit B-03 / B-04).
 * Chosen on the paired 20-seed differential gate among the pre-registered
 * levels 0.10 / 0.05 / 0.01 (audit/diffs/54-*): 0.05 and 0.10 still isolated
 * chance-pure bins of 4-6 rows in samples of 28-42 rows.
 */
constexpr double kPureBinAlpha = 0.01;

/**
 * @brief Exact binomial test of zero occurrences for a pure bin.
 *
 * Under the global event rate r0 = tp / (tp + tn), the probability that a
 * bin of n rows shows no event by chance is (1 - r0)^n, and no non-event
 * r0^n. A bin with both classes returns 0 (nothing to test); so does a sample
 * with one class only (no evidence either way).
 */
inline double pure_bin_pvalue(double pos, double neg, double tp, double tn) {
  const double n = pos + neg;
  if (n <= 0.0 || tp <= 0.0 || tn <= 0.0) return 0.0;
  const double r0 = tp / (tp + tn);
  if (pos == 0.0) return std::pow(1.0 - r0, n);
  if (neg == 0.0) return std::pow(r0, n);
  return 0.0;
}

} // namespace


/**
 * @brief Optimal Binning for Numerical Variables using Optimal Supervised Learning Partitioning (OSLP)
 *
 * IMPORTANT: Despite "LP" (Linear Programming) in the name, this algorithm uses greedy
 * heuristics with information-theoretic optimization, not formal LP.
 *
 * Algorithm Overview:
 * 1. Pre-binning on quantiles of the distinct values
 * 2. Greedy merging of bins below bin_cutoff (IV-loss based direction)
 * 3. Monotonicity enforcement in WoE values (direction by majority vote)
 * 4. Reduction to max_bins (smallest combined IV), then monotonicity again
 *
 * Bins are right-closed, (lower, upper].
 *
 * Bin statistics are computed from prefix sums over the sorted distinct
 * values, so each re-evaluation costs O(k log u) instead of a pass over the
 * n observations.
 */
class OBN_OSLP {
private:
  // Input data
  std::vector<double> feature;
  std::vector<double> target;

  // Algorithm parameters
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  int max_iterations;
  double laplace_smoothing;

  // Sorted distinct values and cumulative class counts:
  // cum_pos[j] / cum_neg[j] = positives / negatives among values < uvals[j].
  std::vector<double> uvals;
  std::vector<int> cum_pos;
  std::vector<int> cum_neg;

  // Bin structure
  std::vector<double> bin_edges;
  std::vector<std::string> bin_labels;
  std::vector<double> woe_values;
  std::vector<double> iv_values;
  std::vector<int> count_values;
  std::vector<int> count_pos_values;
  std::vector<int> count_neg_values;
  std::vector<double> event_rate_values;

  // Algorithm state
  double n_obs;  // number of non-missing observations
  double total_iv;
  bool converged;
  int iterations_run;

public:
  OBN_OSLP(
    const std::vector<double>& feature_,
    const std::vector<double>& target_,
    int min_bins_ = 3,
    int max_bins_ = 5,
    double bin_cutoff_ = 0.05,
    int max_n_prebins_ = 20,
    int max_iterations_ = 1000,
    double laplace_smoothing_ = 0.5
  ) : feature(feature_), target(target_),
  min_bins(std::max(min_bins_, 2)),
  max_bins(std::max(max_bins_, min_bins_)),
  bin_cutoff(bin_cutoff_),
  max_n_prebins(std::max(max_n_prebins_, min_bins_)),
  max_iterations(max_iterations_),
  laplace_smoothing(laplace_smoothing_),
  n_obs(0.0),
  total_iv(0.0),
  converged(false),
  iterations_run(0) {
    validateInputs();
  }

  Rcpp::List fit() {
    buildValueTable();

    if (uvals.size() <= 2) {
      handleLowUniqueValues();
      converged = true;
      iterations_run = 0;
      return createOutput();
    }

    createInitialBins();
    mergeSmallBins();
    mergeUnsupportedPureBins();
    enforceMonotonicity();
    calculateBins();

    return createOutput();
  }

private:
  // Sizes and numeric parameters are validated by the exported wrapper.
  void validateInputs() const {
    bool has_zero = false, has_one = false;
    for (double t : target) {
      if (t == 0) has_zero = true;
      else if (t == 1) has_one = true;
      else throw std::invalid_argument("Target must contain only 0 and 1.");
      if (has_zero && has_one) break;
    }
    if (!has_zero || !has_one) {
      throw std::invalid_argument("Target must contain both classes (0 and 1).");
    }
  }

  /**
   * @brief Sorted distinct values with cumulative class counts (one sort).
   *
   * Missing values (NaN / NA) are dropped silently; +/-Inf are kept as extreme
   * values. The quantile edges never select -Inf (an edge must exceed the
   * previous one) nor the maximum, so neither becomes a cutpoint.
   */
  void buildValueTable() {
    std::vector<std::pair<double, int>> data;
    data.reserve(feature.size());
    for (size_t i = 0; i < feature.size(); ++i) {
      if (!std::isnan(feature[i])) data.emplace_back(feature[i], target[i] == 1 ? 1 : 0);
    }
    if (data.empty()) {
      throw std::invalid_argument("All feature values are missing (NA/NaN); nothing to bin.");
    }
    n_obs = static_cast<double>(data.size());
    std::sort(data.begin(), data.end(),
              [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                return a.first < b.first;
              });

    uvals.clear();
    cum_pos.assign(1, 0);
    cum_neg.assign(1, 0);
    for (size_t i = 0; i < data.size(); ++i) {
      if (i == 0 || data[i].first != data[i - 1].first) {
        uvals.push_back(data[i].first + 0.0);  // -0.0 -> +0.0 (label "0.000000")
        cum_pos.push_back(cum_pos.back());
        cum_neg.push_back(cum_neg.back());
      }
      if (data[i].second == 1) cum_pos.back()++; else cum_neg.back()++;
    }
  }

  void handleLowUniqueValues() {
    bin_edges.clear();
    bin_edges.push_back(-std::numeric_limits<double>::infinity());
    if (uvals.size() == 2) {
      bin_edges.push_back(oslp_safe_cut(uvals[0], uvals[1]));
    }
    bin_edges.push_back(std::numeric_limits<double>::infinity());
    calculateBins();
  }

  /**
   * @brief Pre-bin edges on quantiles of the distinct values. Each edge is a
   * distinct observed value, so every pre-bin (e_k, e_{k+1}] is non-empty.
   */
  void createInitialBins() {
    const int n_unique = static_cast<int>(uvals.size());
    int n_pre = std::min(max_n_prebins, n_unique);
    n_pre = std::max(n_pre, min_bins);

    bin_edges.clear();
    bin_edges.push_back(-std::numeric_limits<double>::infinity());
    for (int i = 1; i < n_pre; ++i) {
      const int idx = static_cast<int>((i / static_cast<double>(n_pre)) * (n_unique - 1));
      const double edge = uvals[static_cast<size_t>(idx)];
      if (edge > bin_edges.back()) {
        bin_edges.push_back(edge);
      }
    }
    bin_edges.push_back(std::numeric_limits<double>::infinity());

    calculateBins();
  }

  /**
   * @brief Counts, event rates, WoE and IV of all bins (a, b].
   */
  void calculateBins() {
    const int n_bins = static_cast<int>(bin_edges.size() - 1);
    const size_t nb = static_cast<size_t>(n_bins);

    woe_values.assign(nb, 0.0);
    iv_values.assign(nb, 0.0);
    count_values.assign(nb, 0);
    count_pos_values.assign(nb, 0);
    count_neg_values.assign(nb, 0);
    event_rate_values.assign(nb, 0.0);

    // values v with e_b < v <= e_{b+1} occupy uvals[lo, hi)
    size_t lo = 0;  // the first bin also holds -Inf
    for (size_t b = 0; b < nb; ++b) {
      const size_t hi = static_cast<size_t>(
        std::upper_bound(uvals.begin(), uvals.end(), bin_edges[b + 1]) - uvals.begin());
      count_pos_values[b] = cum_pos[hi] - cum_pos[lo];
      count_neg_values[b] = cum_neg[hi] - cum_neg[lo];
      count_values[b] = count_pos_values[b] + count_neg_values[b];
      lo = hi;
    }

    const double total_pos = std::accumulate(count_pos_values.begin(), count_pos_values.end(), 0.0);
    const double total_neg = std::accumulate(count_neg_values.begin(), count_neg_values.end(), 0.0);
    const double total_smoothed_pos = total_pos + n_bins * laplace_smoothing;
    const double total_smoothed_neg = total_neg + n_bins * laplace_smoothing;
    // Only one class left after dropping missing features: no evidence.
    const bool one_class = total_pos == 0.0 || total_neg == 0.0;

    total_iv = 0.0;
    for (size_t i = 0; i < nb; ++i) {
      event_rate_values[i] = static_cast<double>(count_pos_values[i]) / count_values[i];
      if (one_class) continue;  // WoE = IV = 0

      const double dist_pos = (count_pos_values[i] + laplace_smoothing) / total_smoothed_pos;
      const double dist_neg = (count_neg_values[i] + laplace_smoothing) / total_smoothed_neg;

      if (dist_pos <= 0.0) {
        woe_values[i] = -20.0;  // Cap for numerical stability
      } else if (dist_neg <= 0.0) {
        woe_values[i] = 20.0;   // Cap for numerical stability
      } else {
        woe_values[i] = std::log(dist_pos / dist_neg);
      }
      iv_values[i] = (dist_pos - dist_neg) * woe_values[i];
      total_iv += iv_values[i];
    }

    updateBinLabels();
  }

  bool hasSmallBin() const {
    const double total = n_obs;
    for (int c : count_values) {
      if (static_cast<double>(c) / total < bin_cutoff) return true;
    }
    return false;
  }

  /**
   * @brief Merge bins whose share is below bin_cutoff.
   */
  void mergeSmallBins() {
    bool merged = true;
    const double total_count = n_obs;
    iterations_run = 0;

    while (merged && static_cast<int>(bin_edges.size()) - 1 > min_bins && iterations_run < max_iterations) {
      merged = false;

      size_t smallest_idx = 0;
      double smallest_prop = std::numeric_limits<double>::max();
      for (size_t i = 0; i < count_values.size(); i++) {
        const double prop = static_cast<double>(count_values[i]) / total_count;
        if (prop < smallest_prop) {
          smallest_prop = prop;
          smallest_idx = i;
        }
      }

      if (smallest_prop < bin_cutoff) {
        if (smallest_idx == 0) {
          mergeBins(0);
        } else if (smallest_idx == count_values.size() - 1) {
          mergeBins(count_values.size() - 2);
        } else {
          const double iv_loss_left = computeIVLoss(smallest_idx - 1, smallest_idx);
          const double iv_loss_right = computeIVLoss(smallest_idx, smallest_idx + 1);
          mergeBins(iv_loss_left <= iv_loss_right ? smallest_idx - 1 : smallest_idx);
        }
        calculateBins();
        merged = true;
      }

      iterations_run++;
    }
  }

  /**
   * @brief Pure bins must be supported by evidence (audit B-03 / X-08).
   *
   * The smoothed IV of a pure bin is large, so the IV-loss criterion never
   * merges a block without events into a bin with events. With rare events
   * (4-7 training events) that kept blocks that are empty of events only by
   * chance, which predict worse out of sample than the merged bin. Two steps:
   *  1. adjacent pure bins of the same class are merged: the binomial
   *     likelihood is unchanged (G^2 = 0), only the bin count drops;
   *  2. a pure bin may stay alone only if the exact binomial test of zero
   *     occurrences under the global rate rejects chance,
   *     P0 = (1 - r0)^n <= alpha (r0^n for a bin without non-events),
   *     alpha = kPureBinAlpha, i.e. n * r0 >~ 4.6. Otherwise the least
   *     supported pure bin is merged with the neighbour whose merge keeps the
   *     largest reported IV, and the test is repeated.
   * Perfect separation survives (n * r0 is large); min_bins is respected.
   */
  void mergeUnsupportedPureBins() {
    const double tp = std::accumulate(count_pos_values.begin(), count_pos_values.end(), 0.0);
    const double tn = std::accumulate(count_neg_values.begin(), count_neg_values.end(), 0.0);
    if (tp == 0.0 || tn == 0.0) return;  // one class: no evidence to weigh

    // 1. Same-class pure neighbours.
    size_t i = 0;
    while (i + 1 < count_values.size() &&
           static_cast<int>(bin_edges.size()) - 1 > min_bins && iterations_run < max_iterations) {
      const bool no_pos = count_pos_values[i] == 0 && count_pos_values[i + 1] == 0;
      const bool no_neg = count_neg_values[i] == 0 && count_neg_values[i + 1] == 0;
      if (no_pos || no_neg) {
        mergeBins(i);
        calculateBins();
        iterations_run++;
      } else {
        ++i;
      }
    }

    // 2. Pure bins without significant evidence.
    while (static_cast<int>(bin_edges.size()) - 1 > min_bins && iterations_run < max_iterations) {
      size_t idx = 0;
      double worst = kPureBinAlpha;
      bool found = false;
      for (size_t b = 0; b < count_values.size(); ++b) {
        const double p = pure_bin_pvalue(count_pos_values[b], count_neg_values[b], tp, tn);
        if (p > worst) {
          worst = p;
          idx = b;
          found = true;
        }
      }
      if (!found) break;
      if (idx == 0) {
        mergeBins(0);
      } else if (idx == count_values.size() - 1) {
        mergeBins(idx - 1);
      } else {
        mergeBins(computeIVLoss(idx - 1, idx) <= computeIVLoss(idx, idx + 1) ? idx - 1 : idx);
      }
      calculateBins();
      iterations_run++;
    }
  }

  /**
   * @brief Merge monotonicity violations until the WoE is monotonic in the
   * given direction, min_bins is reached, or max_iterations is exhausted.
   *
   * The earlier version also stopped as soon as one merge changed the total IV
   * by less than convergence_threshold, returning WoE that was still not
   * monotonic; the documented algorithm merges violations until none is left.
   */
  void mergeViolations(bool increasing) {
    while (!isMonotonic(increasing) &&
           static_cast<int>(bin_edges.size()) - 1 > min_bins &&
           iterations_run < max_iterations) {
      // First violation (one exists because the WoE is not monotonic).
      size_t i = 1;
      while ((increasing && !(woe_values[i] < woe_values[i - 1])) ||
             (!increasing && !(woe_values[i] > woe_values[i - 1]))) {
        ++i;
      }

      if (i > 1) {
        // If merging i-1 and i would cause a new violation with i-2, merge
        // i and i+1 instead when possible.
        const double test_woe = estimateMergedWoE(i - 1, i);
        const bool new_violation = (increasing && test_woe < woe_values[i - 2]) ||
                                   (!increasing && test_woe > woe_values[i - 2]);
        if (new_violation && i < woe_values.size() - 1) {
          mergeBins(i);
        } else {
          mergeBins(i - 1);
        }
      } else {
        mergeBins(0);
      }

      calculateBins();
      iterations_run++;
    }
  }

  void enforceMonotonicity() {
    const bool increasing = guessTrend();

    mergeViolations(increasing);

    // If too many bins, merge the adjacent pair whose fusion loses the least IV.
    while (static_cast<int>(bin_edges.size()) - 1 > max_bins && iterations_run < max_iterations) {
      mergeBins(findMinIVMerge());
      calculateBins();
      iterations_run++;
    }

    // The max_bins reduction can re-create a violation: enforce again.
    mergeViolations(increasing);

    // converged == false only when max_iterations stopped work that remained.
    const int k = static_cast<int>(bin_edges.size()) - 1;
    const bool pending = k > max_bins || (k > min_bins && (!isMonotonic(increasing) || hasSmallBin()));
    converged = !(iterations_run >= max_iterations && pending);
  }

  bool guessTrend() const {
    int inc = 0, dec = 0;
    for (size_t i = 1; i < woe_values.size(); i++) {
      if (woe_values[i] > woe_values[i - 1]) inc++;
      else if (woe_values[i] < woe_values[i - 1]) dec++;
    }
    return inc >= dec;
  }

  bool isMonotonic(bool increasing) const {
    for (size_t i = 1; i < woe_values.size(); i++) {
      if (increasing && woe_values[i] < woe_values[i - 1]) return false;
      if (!increasing && woe_values[i] > woe_values[i - 1]) return false;
    }
    return true;
  }

  /**
   * @brief Left index of the adjacent pair whose merge loses the least IV.
   *
   * Audit B-03 / X-08: the pair used to be the one with the smallest SUM
   * IV_i + IV_{i+1}, which is small for two bins with WoE near 0 even when
   * their signs differ and large for two neighbours with the same extreme WoE
   * (whose merge loses nothing). The loss is the change of the reported
   * total IV (computeIVLoss()). Ties keep the leftmost pair.
   */
  size_t findMinIVMerge() const {
    double best = std::numeric_limits<double>::infinity();
    size_t idx = 0;
    for (size_t i = 0; i + 1 < count_values.size(); i++) {
      const double loss = computeIVLoss(i, i + 1);
      if (loss < best) {
        best = loss;
        idx = i;
      }
    }
    return idx;
  }

  /**
   * @brief Total IV, with calculateBins()'s estimator, of the partition in
   * which bins i and i+1 are merged (K - 1 bins, all renormalized).
   */
  double ivAfterMerge(size_t i) const {
    const size_t nb = count_values.size();
    const double total_pos = std::accumulate(count_pos_values.begin(), count_pos_values.end(), 0.0);
    const double total_neg = std::accumulate(count_neg_values.begin(), count_neg_values.end(), 0.0);
    if (total_pos == 0.0 || total_neg == 0.0) return 0.0;  // one class: WoE = IV = 0
    const double k = static_cast<double>(nb - 1);
    const double sp = total_pos + k * laplace_smoothing;
    const double sn = total_neg + k * laplace_smoothing;
    auto term = [&](double pos, double neg) {
      const double dp = (pos + laplace_smoothing) / sp;
      const double dn = (neg + laplace_smoothing) / sn;
      const double w = dp <= 0.0 ? -20.0 : (dn <= 0.0 ? 20.0 : std::log(dp / dn));
      return (dp - dn) * w;
    };
    double s = 0.0;
    for (size_t j = 0; j < nb; ++j) {
      if (j == i) {
        s += term(count_pos_values[j] + count_pos_values[j + 1],
                  count_neg_values[j] + count_neg_values[j + 1]);
        ++j;
      } else {
        s += term(count_pos_values[j], count_neg_values[j]);
      }
    }
    return s;
  }

  /**
   * @brief IV lost by merging bins i and j = i + 1: the change of the
   * REPORTED total IV (K bins before, K - 1 after, same estimator). It can be
   * negative (merging pure bins raises the smoothed IV).
   *
   * Audit B-03 / X-08. The former version took IV_i + IV_j from the K-bin
   * partition but the merged IV with K - 1 bins and ignored the other bins'
   * renormalization, mixing two scales. The exact one-step change of the
   * reported objective was preferred over a fixed-K per-pair loss after a
   * comparison on 1800 differential runs (audit/diffs/54-*).
   */
  double computeIVLoss(size_t i, size_t /*j*/) const {
    return total_iv - ivAfterMerge(i);
  }

  /**
   * @brief WoE of the bin that merging bins i and j would produce.
   */
  double estimateMergedWoE(size_t i, size_t j) const {
    const double merged_pos = count_pos_values[i] + count_pos_values[j];
    const double merged_neg = count_neg_values[i] + count_neg_values[j];
    const double total_pos = std::accumulate(count_pos_values.begin(), count_pos_values.end(), 0.0);
    const double total_neg = std::accumulate(count_neg_values.begin(), count_neg_values.end(), 0.0);

    const double k_after = static_cast<double>(bin_edges.size() - 2);
    const double dist_pos = (merged_pos + laplace_smoothing) / (total_pos + k_after * laplace_smoothing);
    const double dist_neg = (merged_neg + laplace_smoothing) / (total_neg + k_after * laplace_smoothing);

    if (dist_pos <= 0.0 || dist_neg <= 0.0) {
      return 0.0;
    }
    return std::log(dist_pos / dist_neg);
  }

  /**
   * @brief Merge bins i and i+1 by removing the edge between them.
   */
  void mergeBins(size_t i) {
    bin_edges.erase(bin_edges.begin() + static_cast<std::ptrdiff_t>(i + 1));
  }

  /**
   * @brief Right-closed labels "(lower;upper]". They used to read
   * "[lower;upper)", the opposite of how boundary values are assigned.
   */
  void updateBinLabels() {
    bin_labels.clear();
    for (size_t i = 0; i + 1 < bin_edges.size(); i++) {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(6);
      if (std::isinf(bin_edges[i])) {
        oss << "(-Inf;";
      } else {
        oss << "(" << bin_edges[i] << ";";
      }
      if (std::isinf(bin_edges[i + 1])) {
        oss << "+Inf]";
      } else {
        oss << bin_edges[i + 1] << "]";
      }
      bin_labels.push_back(oss.str());
    }
  }

  Rcpp::List createOutput() const {
    // Interior edges (finite by construction)
    std::vector<double> cutpoints(bin_edges.begin() + 1, bin_edges.end() - 1);

    Rcpp::NumericVector ids(static_cast<R_xlen_t>(bin_labels.size()));
    for (R_xlen_t i = 0; i < ids.size(); i++) {
      ids[i] = static_cast<double>(i + 1);
    }

    return Rcpp::List::create(
      Named("id") = ids,
      Named("bin") = bin_labels,
      Named("woe") = woe_values,
      Named("iv") = iv_values,
      Named("count") = count_values,
      Named("count_pos") = count_pos_values,
      Named("count_neg") = count_neg_values,
      Named("event_rate") = event_rate_values,
      Named("cutpoints") = cutpoints,
      Named("total_iv") = total_iv,
      Named("converged") = converged,
      Named("iterations") = iterations_run
    );
  }
};

// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_oslp(
   Rcpp::NumericVector target,
   Rcpp::NumericVector feature,
   int min_bins = 3,
   int max_bins = 5,
   double bin_cutoff = 0.05,
   int max_n_prebins = 20,
   double convergence_threshold = 1e-6,
   int max_iterations = 1000,
   double laplace_smoothing = 0.5
) {
 if (feature.size() != target.size()) {
   Rcpp::stop("Feature and target must have the same length.");
 }
 if (feature.size() == 0) {
   Rcpp::stop("Feature and target cannot be empty.");
 }
 if (min_bins < 2) {
   Rcpp::stop("min_bins must be at least 2.");
 }
 if (max_bins < min_bins) {
   Rcpp::stop("max_bins must be greater or equal to min_bins.");
 }
 if (bin_cutoff <= 0 || bin_cutoff >= 1) {
   Rcpp::stop("bin_cutoff must be between 0 and 1.");
 }
 if (max_n_prebins < min_bins) {
   Rcpp::stop("max_n_prebins must be at least min_bins.");
 }
 // convergence_threshold is validated for API compatibility; monotonicity
 // enforcement runs until no violation is left, without an IV-change stop.
 if (convergence_threshold <= 0) {
   Rcpp::stop("convergence_threshold must be positive.");
 }
 if (max_iterations <= 0) {
   Rcpp::stop("max_iterations must be positive.");
 }
 if (laplace_smoothing < 0) {
   Rcpp::stop("laplace_smoothing must be non-negative.");
 }

 std::vector<double> feature_vec(feature.begin(), feature.end());
 std::vector<double> target_vec(target.begin(), target.end());

 try {
   OBN_OSLP binning(
       feature_vec, target_vec,
       min_bins, max_bins,
       bin_cutoff, max_n_prebins,
       max_iterations,
       laplace_smoothing
   );
   return binning.fit();
 } catch (const std::exception &e) {
   Rcpp::stop("Error in optimal_binning_numerical_oslp: " + std::string(e.what()));
 }
}
