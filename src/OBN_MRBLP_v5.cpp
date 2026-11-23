// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <string>
#include <utility>


// Include shared headers
#include "common/optimal_binning_common.h"
#include "common/bin_structures.h"

using namespace Rcpp;
using namespace OptimalBinning;

namespace {

/**
 * @brief A finite cut c with a <= c < b between two consecutive distinct values
 * a < b, for the right-closed (lower, upper] convention.
 *
 * The midpoint (a + b) / 2 is used whenever it is valid, so ordinary data is
 * unaffected; it overflows to +/-Inf near the largest finite double and, for
 * two adjacent doubles, can round up to b. Then a/2 + b/2, and finally a, are
 * used instead.
 */
inline double mrblp_safe_cut(double a, double b) {
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
 * levels 0.10 / 0.05 / 0.01 (audit/diffs/54-*, 55-*): 0.05 and 0.10 still
 * isolated chance-pure bins of 4-6 rows in samples of 28-42 rows.
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
 * @brief Optimal Binning using Monotonic Risk Binning with Likelihood Ratio Pre-binning (MRBLP)
 *
 * IMPORTANT: Despite "LP" in the name, uses greedy heuristics, not formal
 * Linear Programming.
 *
 * Algorithm Overview:
 * 1. Equal-frequency, tie-aware pre-binning
 * 2. Greedy merging of bins below bin_cutoff
 * 3. WoE monotonicity enforcement (direction by majority vote)
 * 4. Reduction to max_bins (smallest absolute IV difference), after which
 *    monotonicity is enforced again
 *
 * Complexity: O(n log n + k^2 * iterations)
 * Space: O(n + k)
 */
class OBN_MRBLP {
private:
  // Feature and target vectors
  std::vector<double> feature;
  std::vector<int> target;

  // Binning parameters
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  double convergence_threshold;
  int max_iterations;
  double laplace_smoothing;

  // Algorithm state
  bool converged;
  int iterations_run;
  double n_obs;  // number of non-missing observations

  std::vector<NumericalBin> bins;

public:
  OBN_MRBLP(const NumericVector& feature_,
            const IntegerVector& target_,
            int min_bins_ = 3,
            int max_bins_ = 5,
            double bin_cutoff_ = 0.05,
            int max_n_prebins_ = 20,
            double convergence_threshold_ = 1e-6,
            int max_iterations_ = 1000,
            double laplace_smoothing_ = 0.5)
    : feature(feature_.begin(), feature_.end()),
      target(target_.begin(), target_.end()),
      min_bins(min_bins_),
      max_bins(std::max(min_bins_, max_bins_)),
      bin_cutoff(bin_cutoff_),
      max_n_prebins(std::max(max_n_prebins_, min_bins_)),
      convergence_threshold(convergence_threshold_),
      max_iterations(max_iterations_),
      laplace_smoothing(laplace_smoothing_),
      converged(false),
      iterations_run(0),
      n_obs(0.0) {
    validateInputs();
  }

  /**
   * @brief Fit the binning model.
   */
  void fit() {
    // Missing values (NaN / NA) are dropped silently; +/-Inf are kept as
    // extreme values (first / last bin). Sort once; the distinct values are
    // read off the sorted pairs.
    std::vector<std::pair<double, int>> data;
    data.reserve(feature.size());
    for (size_t i = 0; i < feature.size(); ++i) {
      if (!std::isnan(feature[i])) data.emplace_back(feature[i], target[i]);
    }
    if (data.empty()) {
      throw std::invalid_argument("All feature values are missing (NA/NaN); nothing to bin.");
    }
    n_obs = static_cast<double>(data.size());
    std::sort(data.begin(), data.end(),
              [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                return a.first < b.first;
              });

    size_t n_unique = 1;
    for (size_t i = 1; i < data.size(); i++) {
      if (data[i].first != data[i - 1].first) n_unique++;
    }

    adjustBinParameters(n_unique);

    if (n_unique <= 2) {
      handleLowUniqueValues(data, n_unique);
      converged = true;
      iterations_run = 0;
      return;
    }

    performPreBinning(data, static_cast<int>(n_unique));
    mergeSmallBins();
    mergeUnsupportedPureBins();
    enforceMonotonicity();
    computeWoEIV();
  }

  List getResults() const {
    const size_t n_bins = bins.size();
    const R_xlen_t nb = static_cast<R_xlen_t>(n_bins);
    CharacterVector bin_names(nb);
    NumericVector bin_woe(nb);
    NumericVector bin_iv(nb);
    IntegerVector bin_count(nb);
    IntegerVector bin_count_pos(nb);
    IntegerVector bin_count_neg(nb);
    NumericVector bin_event_rates(nb);
    NumericVector bin_cutpoints(n_bins > 1 ? nb - 1 : 0);
    Rcpp::NumericVector ids(nb);
    double total_iv = 0.0;

    for (R_xlen_t i = 0; i < nb; ++i) {
      const NumericalBin& b = bins[static_cast<size_t>(i)];
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(6);

      // Right-closed (lower; upper]
      if (std::isinf(b.lower_bound)) {
        oss << "(-Inf;";
      } else {
        oss << "(" << b.lower_bound << ";";
      }
      if (std::isinf(b.upper_bound)) {
        oss << "+Inf]";
      } else {
        oss << b.upper_bound << "]";
      }

      bin_names[i] = oss.str();
      bin_woe[i] = b.woe;
      bin_iv[i] = b.iv;
      bin_count[i] = b.count;
      bin_count_pos[i] = b.count_pos;
      bin_count_neg[i] = b.count_neg;
      bin_event_rates[i] = b.event_rate();
      ids[i] = static_cast<double>(i + 1);
      total_iv += b.iv;

      if (i < nb - 1) {
        bin_cutpoints[i] = b.upper_bound;
      }
    }

    return Rcpp::List::create(
      Named("id") = ids,
      Named("bin") = bin_names,
      Named("woe") = bin_woe,
      Named("iv") = bin_iv,
      Named("count") = bin_count,
      Named("count_pos") = bin_count_pos,
      Named("count_neg") = bin_count_neg,
      Named("event_rate") = bin_event_rates,
      Named("cutpoints") = bin_cutpoints,
      Named("total_iv") = total_iv,
      Named("converged") = converged,
      Named("iterations") = iterations_run
    );
  }

private:
  void validateInputs() const {
    if (feature.size() != target.size()) {
      throw std::invalid_argument("Feature and target vectors must be of the same length.");
    }
    if (feature.empty()) {
      throw std::invalid_argument("Feature and target vectors cannot be empty.");
    }
    if (min_bins < 1) {
      throw std::invalid_argument("min_bins must be at least 1.");
    }
    if (bin_cutoff <= 0 || bin_cutoff >= 1) {
      throw std::invalid_argument("bin_cutoff must be between 0 and 1.");
    }
    if (convergence_threshold <= 0) {
      throw std::invalid_argument("convergence_threshold must be greater than 0.");
    }
    if (max_iterations <= 0) {
      throw std::invalid_argument("max_iterations must be greater than 0.");
    }
    if (laplace_smoothing < 0) {
      throw std::invalid_argument("laplace_smoothing must be non-negative.");
    }

    bool has_zero = false, has_one = false;
    for (int t : target) {
      if (t == 0) has_zero = true;
      else if (t == 1) has_one = true;
      else throw std::invalid_argument("Target must contain only 0 and 1.");
      if (has_zero && has_one) break;
    }
    if (!has_zero || !has_one) {
      throw std::invalid_argument("Target must contain both classes (0 and 1).");
    }
  }

  void adjustBinParameters(size_t n_unique) {
    if (n_unique < static_cast<size_t>(min_bins)) {
      min_bins = std::max(1, static_cast<int>(n_unique));  // only lowers min_bins
    }
    if (n_unique < static_cast<size_t>(max_bins)) {
      max_bins = static_cast<int>(n_unique);
    }
  }

  /**
   * @brief One bin (one distinct value) or two bins (two distinct values).
   */
  void handleLowUniqueValues(const std::vector<std::pair<double, int>>& data, size_t n_unique) {
    bins.clear();
    int total_pos = 0;
    for (const auto& p : data) total_pos += p.second;
    const int total_n = static_cast<int>(data.size());

    if (n_unique == 1) {
      NumericalBin b;
      b.lower_bound = -std::numeric_limits<double>::infinity();
      b.upper_bound = std::numeric_limits<double>::infinity();
      b.count = total_n;
      b.count_pos = total_pos;
      b.count_neg = total_n - total_pos;
      bins.push_back(b);
    } else {
      const double cut = mrblp_safe_cut(data.front().first, data.back().first);

      NumericalBin b1, b2;
      b1.lower_bound = -std::numeric_limits<double>::infinity();
      b1.upper_bound = cut;
      b2.lower_bound = cut;
      b2.upper_bound = std::numeric_limits<double>::infinity();

      for (const auto& p : data) {
        NumericalBin& b = (p.first <= cut) ? b1 : b2;
        b.count++;
        if (p.second == 1) b.count_pos++; else b.count_neg++;
      }
      bins.push_back(b1);
      bins.push_back(b2);
    }
    computeWoEIV();
  }

  /**
   * @brief Equal-frequency pre-bins on the sorted data, never splitting a run
   * of tied values (so every pre-bin is non-empty).
   */
  void performPreBinning(const std::vector<std::pair<double, int>>& data, int distinct_count) {
    int n_pre = std::min(max_n_prebins, distinct_count);
    n_pre = std::max(n_pre, min_bins);
    const size_t bin_size = std::max(static_cast<size_t>(1), data.size() / static_cast<size_t>(n_pre));

    bins.clear();
    for (size_t i = 0; i < data.size(); ) {
      size_t end = std::min(i + bin_size, data.size());
      // A pre-bin never ends on -Inf (that would make -Inf a cutpoint): the
      // -Inf observations are pooled with the next value run.
      while (end < data.size() && (data[end].first == data[end - 1].first ||
                                   data[end - 1].first == -std::numeric_limits<double>::infinity())) {
        ++end;
      }

      NumericalBin b;
      b.lower_bound = bins.empty()
                          ? -std::numeric_limits<double>::infinity()
                          : bins.back().upper_bound;
      b.upper_bound = (end == data.size())
                          ? std::numeric_limits<double>::infinity()
                          : data[end - 1].first;
      for (size_t j = i; j < end; j++) {
        b.count++;
        if (data[j].second == 1) b.count_pos++; else b.count_neg++;
      }
      bins.push_back(b);
      i = end;
    }

    computeWoEIV();
  }

  /**
   * @brief WoE / IV of all bins with Laplace smoothing. Every bin is
   * non-empty, so all values are finite.
   */
  void computeWoEIV() {
    int sum_pos = 0;
    int sum_neg = 0;
    for (const auto& b : bins) {
      sum_pos += b.count_pos;
      sum_neg += b.count_neg;
    }

    // Only one class left after dropping missing features: no evidence.
    if (sum_pos == 0 || sum_neg == 0) {
      for (auto& b : bins) {
        b.woe = 0.0;
        b.iv = 0.0;
      }
      return;
    }

    const double total_smoothed_pos = sum_pos + static_cast<double>(bins.size()) * laplace_smoothing;
    const double total_smoothed_neg = sum_neg + static_cast<double>(bins.size()) * laplace_smoothing;

    for (auto& b : bins) {
      const double dist_pos = (b.count_pos + laplace_smoothing) / total_smoothed_pos;
      const double dist_neg = (b.count_neg + laplace_smoothing) / total_smoothed_neg;

      if (dist_pos <= 0.0) {
        b.woe = -20.0;  // Cap for stability
      } else if (dist_neg <= 0.0) {
        b.woe = 20.0;   // Cap for stability
      } else {
        b.woe = std::log(dist_pos / dist_neg);
      }
      b.iv = (dist_pos - dist_neg) * b.woe;
    }
  }

  bool hasSmallBin() const {
    const double total = n_obs;
    for (const auto& b : bins) {
      if (static_cast<double>(b.count) / total < bin_cutoff) return true;
    }
    return false;
  }

  /**
   * @brief Merge bins whose share is below bin_cutoff.
   */
  void mergeSmallBins() {
    const double total = n_obs;
    bool merged = true;
    while (merged && static_cast<int>(bins.size()) > min_bins && iterations_run < max_iterations) {
      merged = false;

      size_t smallest_idx = 0;
      double smallest_prop = std::numeric_limits<double>::max();
      for (size_t i = 0; i < bins.size(); i++) {
        const double prop = static_cast<double>(bins[i].count) / total;
        if (prop < smallest_prop) {
          smallest_prop = prop;
          smallest_idx = i;
        }
      }

      if (smallest_prop < bin_cutoff) {
        if (smallest_idx == 0) {
          mergeBins(0);
        } else if (smallest_idx == bins.size() - 1) {
          mergeBins(bins.size() - 2);
        } else {
          // IV actually lost by each merge (audit X-08 / D-12): the former
          // "loss" IV_a + IV_b never subtracted the IV of the merged bin.
          const double iv_loss_left = mergeLoss(smallest_idx - 1);
          const double iv_loss_right = mergeLoss(smallest_idx);
          mergeBins(iv_loss_left <= iv_loss_right ? smallest_idx - 1 : smallest_idx);
        }
        computeWoEIV();
        merged = true;
      }

      iterations_run++;
    }
  }

  /**
   * @brief Pure bins must be supported by evidence (audit B-04 / X-08).
   *
   * The smoothed IV of a pure bin is large, so the IV-loss criterion never
   * merges a block without events into a bin with events, and with a high
   * max_bins it keeps runs of pure bins of one class apart. Two steps:
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
    double tp = 0.0, tn = 0.0;
    for (const auto& b : bins) {
      tp += b.count_pos;
      tn += b.count_neg;
    }
    if (tp == 0.0 || tn == 0.0) return;  // one class: no evidence to weigh

    // 1. Same-class pure neighbours.
    size_t i = 0;
    while (i + 1 < bins.size() && static_cast<int>(bins.size()) > min_bins &&
           iterations_run < max_iterations) {
      const bool no_pos = bins[i].count_pos == 0 && bins[i + 1].count_pos == 0;
      const bool no_neg = bins[i].count_neg == 0 && bins[i + 1].count_neg == 0;
      if (no_pos || no_neg) {
        mergeBins(i);
        computeWoEIV();
        iterations_run++;
      } else {
        ++i;
      }
    }

    // 2. Pure bins without significant evidence.
    while (static_cast<int>(bins.size()) > min_bins && iterations_run < max_iterations) {
      size_t idx = 0;
      double worst = kPureBinAlpha;
      bool found = false;
      for (size_t b = 0; b < bins.size(); ++b) {
        const double p = pure_bin_pvalue(bins[b].count_pos, bins[b].count_neg, tp, tn);
        if (p > worst) {
          worst = p;
          idx = b;
          found = true;
        }
      }
      if (!found) break;
      if (idx == 0) {
        mergeBins(0);
      } else if (idx == bins.size() - 1) {
        mergeBins(idx - 1);
      } else {
        mergeBins(mergeLoss(idx - 1) <= mergeLoss(idx) ? idx - 1 : idx);
      }
      computeWoEIV();
      iterations_run++;
    }
  }

  bool isMonotonic(bool increasing) const {
    for (size_t i = 1; i < bins.size(); i++) {
      if (increasing && bins[i].woe < bins[i - 1].woe) return false;
      if (!increasing && bins[i].woe > bins[i - 1].woe) return false;
    }
    return true;
  }

  /**
   * @brief Monotonic direction by majority vote over adjacent pairs.
   */
  bool guessIncreasing() const {
    int inc = 0, dec = 0;
    for (size_t i = 1; i < bins.size(); i++) {
      if (bins[i].woe > bins[i - 1].woe) inc++;
      else if (bins[i].woe < bins[i - 1].woe) dec++;
    }
    return inc >= dec;
  }

  /**
   * @brief Merge monotonicity violations until the WoE is monotonic in the
   * given direction, min_bins is reached, or max_iterations is exhausted.
   */
  void mergeViolations(bool increasing) {
    while (!isMonotonic(increasing) && static_cast<int>(bins.size()) > min_bins &&
           iterations_run < max_iterations) {
      // First violation (one exists because the WoE is not monotonic).
      size_t i = 1;
      while ((increasing && !(bins[i].woe < bins[i - 1].woe)) ||
             (!increasing && !(bins[i].woe > bins[i - 1].woe))) {
        ++i;
      }

      // Would merging i-1 and i create a violation with bin i-2?
      bool merge_fixes = true;
      if (i > 1) {
        double sum_pos = 0, sum_neg = 0;
        for (const auto& b : bins) {
          sum_pos += b.count_pos;
          sum_neg += b.count_neg;
        }
        const double smoothed_pos = bins[i - 1].count_pos + bins[i].count_pos + laplace_smoothing;
        const double smoothed_neg = bins[i - 1].count_neg + bins[i].count_neg + laplace_smoothing;
        const double k_after = static_cast<double>(bins.size() - 1);
        const double dist_pos = smoothed_pos / (sum_pos + k_after * laplace_smoothing);
        const double dist_neg = smoothed_neg / (sum_neg + k_after * laplace_smoothing);
        const double merged_woe = std::log(dist_pos / dist_neg);

        if ((increasing && merged_woe < bins[i - 2].woe) ||
            (!increasing && merged_woe > bins[i - 2].woe)) {
          merge_fixes = false;
        }
      }

      if (!merge_fixes && i < bins.size() - 1) {
        mergeBins(i);       // merge i and i+1 instead
      } else {
        mergeBins(i - 1);   // merge i-1 and i
      }

      computeWoEIV();
      iterations_run++;
    }
  }

  /**
   * @brief Monotonicity enforcement followed by reduction to max_bins.
   *
   * The earlier version also stopped enforcing as soon as the first and last
   * bins had (nearly) the same WoE, returning a non-monotonic binning that the
   * documentation promises cannot occur. The reduction to max_bins can itself
   * re-create a violation (the smoothed WoE of a merged bin need not lie
   * between its parts), so monotonicity is enforced again afterwards.
   */
  void enforceMonotonicity() {
    const bool increasing = guessIncreasing();

    mergeViolations(increasing);

    while (static_cast<int>(bins.size()) > max_bins && iterations_run < max_iterations) {
      mergeBins(findMinIVLossMerge());
      computeWoEIV();
      iterations_run++;
    }

    mergeViolations(increasing);

    // converged == false only when max_iterations stopped work that remained.
    const bool pending = static_cast<int>(bins.size()) > max_bins ||
      (static_cast<int>(bins.size()) > min_bins && (!isMonotonic(increasing) || hasSmallBin()));
    converged = !(iterations_run >= max_iterations && pending);
  }

  /**
   * @brief IV lost by merging bins i and i+1: the change of the REPORTED total
   * IV (K bins before, K - 1 after, computeWoEIV()'s estimator with every bin
   * renormalized). It can be negative (merging pure bins raises the smoothed
   * IV). The exact one-step change of the reported objective was preferred
   * over a fixed-K per-pair loss after a comparison on 1800 differential runs
   * (audit/diffs/55-*).
   */
  double mergeLoss(size_t i) const {
    double tp = 0.0, tn = 0.0, iv_now = 0.0;
    for (const auto& b : bins) {
      tp += b.count_pos;
      tn += b.count_neg;
      iv_now += b.iv;
    }
    if (tp == 0.0 || tn == 0.0) return 0.0;  // one class: WoE = IV = 0
    const double k = static_cast<double>(bins.size() - 1);
    const double sp = tp + k * laplace_smoothing;
    const double sn = tn + k * laplace_smoothing;
    auto term = [&](double pos, double neg) {
      const double dp = (pos + laplace_smoothing) / sp;
      const double dn = (neg + laplace_smoothing) / sn;
      const double w = dp <= 0.0 ? -20.0 : (dn <= 0.0 ? 20.0 : std::log(dp / dn));
      return (dp - dn) * w;
    };
    double after = 0.0;
    for (size_t j = 0; j < bins.size(); ++j) {
      if (j == i) {
        after += term(bins[j].count_pos + bins[j + 1].count_pos,
                      bins[j].count_neg + bins[j + 1].count_neg);
        ++j;
      } else {
        after += term(bins[j].count_pos, bins[j].count_neg);
      }
    }
    return iv_now - after;
  }

  /**
   * @brief Left index of the adjacent pair whose merge loses the least IV.
   *
   * Audit B-04 / X-08: the pair used to be the one with the smallest
   * |IV_i - IV_{i+1}|, which is 0 for two bins with opposite WoE and equal IV
   * -- the most distinct pair. Ties keep the leftmost pair.
   */
  size_t findMinIVLossMerge() const {
    double best = std::numeric_limits<double>::infinity();
    size_t idx = 0;
    for (size_t i = 0; i + 1 < bins.size(); i++) {
      const double loss = mergeLoss(i);
      if (loss < best) {
        best = loss;
        idx = i;
      }
    }
    return idx;
  }

  /**
   * @brief Merge bins idx and idx+1 into bin idx.
   */
  void mergeBins(size_t idx) {
    const size_t j = idx + 1;
    bins[idx].upper_bound = bins[j].upper_bound;
    bins[idx].count += bins[j].count;
    bins[idx].count_pos += bins[j].count_pos;
    bins[idx].count_neg += bins[j].count_neg;
    bins.erase(bins.begin() + static_cast<std::ptrdiff_t>(j));
  }
};


// [[Rcpp::export]]
List optimal_binning_numerical_mrblp(const IntegerVector& target,
                                    const NumericVector& feature,
                                    int min_bins = 3,
                                    int max_bins = 5,
                                    double bin_cutoff = 0.05,
                                    int max_n_prebins = 20,
                                    double convergence_threshold = 1e-6,
                                    int max_iterations = 1000,
                                    double laplace_smoothing = 0.5) {
 try {
   OBN_MRBLP binning(feature, target, min_bins, max_bins, bin_cutoff,
                     max_n_prebins, convergence_threshold, max_iterations,
                     laplace_smoothing);
   binning.fit();
   return binning.getResults();
 } catch (const std::exception& e) {
   Rcpp::stop(std::string("Error in optimal_binning_numerical_mrblp: ") + e.what());
 }
}
