// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <memory>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Include shared headers
#include "common/bin_structures.h"
#include "common/optimal_binning_common.h"
#include "common/ob_core.h"

using namespace Rcpp;
using namespace OptimalBinning;

// ============================================================================
// Utilities
// ============================================================================
namespace utils {
// A finite cut c with a <= c < b between two consecutive distinct finite values
// a < b (right-closed bins). The midpoint is used whenever it is valid; it
// overflows near the largest double and can round up to b for adjacent doubles.
inline double safe_cut(double a, double b) {
  double m = (a + b) / 2.0;
  if (!std::isfinite(m)) m = a / 2.0 + b / 2.0;
  if (m >= a && m < b) return m;
  return a;
}

// Exact maximum-IV partition of m ordered slots into exactly K contiguous
// bins (audit X-09). Same recursion as ob::partition_dp (common/ob_core.h,
// validated against brute force), objective ob::woe_iv (normalized Jeffreys,
// alpha = 0.5, fixed K), each bin with at least `min_count` observations.
// The only difference is the optional monotone constraint: it compares the
// SMOOTHED odds (pos + alpha) / (neg + alpha) of adjacent bins, i.e. the
// order of the WoE that the IV objective is built on, instead of the raw
// event rate. With few events the raw rates of several event-free bins tie at
// 0 while their smoothed WoE differ, so a partition monotone in raw rate can
// report a non-monotone WoE. With alpha = 0.5 the test is exact in integers:
// (2 p1 + 1)(2 n2 + 1) <= (2 p2 + 1)(2 n1 + 1).
// The WoE the engine reports (NumericalBin::calculate_metrics) is smoothed
// differently -- a prior of BAYESIAN_PRIOR_STRENGTH = s pseudo-observations
// centred on the overall rate r0, odds (p + s r0) / (n + s (1 - r0)) -- and the
// `monotonic` flag checks both orders, so adjacent bins must be ordered under
// both. With very rare events (0.2%) the two orders disagree on an event-free
// bin next to a bin with 1-2 events, and the alpha = 0.5 test alone let the
// DP return a reported WoE such as -8.1 / 1.0 / -5.7.
ob::DpResult sketch_partition_dp(const std::vector<int64_t> &pos,
                                 const std::vector<int64_t> &neg, int K,
                                 bool monotone, bool asc, int64_t min_count) {
  const int m = static_cast<int>(pos.size());
  const double NEG = -std::numeric_limits<double>::infinity();
  ob::DpResult r{NEG, {}};
  if (K < 1 || m < K) return r;

  std::vector<int64_t> P(m + 1, 0), N(m + 1, 0);
  for (int i = 0; i < m; ++i) {
    P[i + 1] = P[i] + pos[i];
    N[i + 1] = N[i] + neg[i];
  }
  const double Tp = static_cast<double>(P[m]);
  const double Tn = static_cast<double>(N[m]);

  auto cnt = [&](int j, int i) { return (P[i] - P[j]) + (N[i] - N[j]); };
  auto val = [&](int j, int i) {
    return ob::woe_iv(static_cast<double>(P[i] - P[j]),
                      static_cast<double>(N[i] - N[j]), Tp, Tn, K).iv;
  };
  auto odds_le = [&](int a0, int a1, int b0, int b1) {  // odds[a0,a1) <= odds[b0,b1)
    const int64_t pa = 2 * (P[a1] - P[a0]) + 1, na = 2 * (N[a1] - N[a0]) + 1;
    const int64_t pb = 2 * (P[b1] - P[b0]) + 1, nb = 2 * (N[b1] - N[b0]) + 1;
    return pa * nb <= pb * na;
  };
  const double s_pr = BAYESIAN_PRIOR_STRENGTH;
  const double r0 = (Tp + Tn) > 0.0 ? Tp / (Tp + Tn) : 0.5;
  auto log_odds_rep = [&](int a0, int a1) {  // order of the reported WoE
    return std::log((static_cast<double>(P[a1] - P[a0]) + s_pr * r0) /
                    (static_cast<double>(N[a1] - N[a0]) + s_pr * (1.0 - r0)));
  };
  auto rep_le = [&](int a0, int a1, int b0, int b1) {
    return log_odds_rep(a0, a1) <= log_odds_rep(b0, b1) + 1e-10;
  };
  auto ordered = [&](int j, int i, int e) {  // [j,i) followed by [i,e)
    if (!monotone) return true;
    return asc ? (odds_le(j, i, i, e) && rep_le(j, i, i, e))
               : (odds_le(i, e, j, i) && rep_le(i, e, j, i));
  };

  using M2 = std::vector<std::vector<double>>;
  using B2 = std::vector<std::vector<int>>;
  std::vector<M2> f(K + 1, M2(m + 1, std::vector<double>(m + 1, NEG)));
  std::vector<B2> bp(K + 1, B2(m + 1, std::vector<int>(m + 1, -1)));

  for (int i = 1; i <= m; ++i)
    if (cnt(0, i) >= min_count) f[1][i][0] = val(0, i);
  for (int k = 2; k <= K; ++k)
    for (int e = 1; e <= m; ++e)
      for (int i = 1; i < e; ++i) {
        if (cnt(i, e) < min_count) continue;
        const double v = val(i, e);
        for (int j = 0; j < i; ++j) {
          if (f[k - 1][i][j] == NEG || !ordered(j, i, e)) continue;
          const double c = f[k - 1][i][j] + v;
          if (c > f[k][e][i]) {
            f[k][e][i] = c;
            bp[k][e][i] = j;
          }
        }
      }

  int bj = -1;
  for (int j = 0; j < m; ++j)
    if (f[K][m][j] > r.iv) {
      r.iv = f[K][m][j];
      bj = j;
    }
  if (bj < 0) return r;
  for (int k = K, e = m, i = bj; k > 1; --k) {
    r.cuts.push_back(i);
    const int j = bp[k][e][i];
    e = i;
    i = j;
  }
  std::reverse(r.cuts.begin(), r.cuts.end());
  return r;
}
} // namespace utils

// ============================================================================
// KLL-style sketch for approximate quantiles in a stream
// ============================================================================
//
// Every level holds at most k items; an overflowing level is sorted and its
// adjacent pairs are collapsed into one item carrying the pair's weight, kept
// at the lower element on even levels and at the upper one on odd levels (a
// deterministic alternative to KLL's random coin, so results do not depend on
// the RNG state). The weights are tracked explicitly, so the total weight
// always equals the number of items seen.
//
// Accuracy: a compaction at a level whose items weigh w moves any rank by at
// most w, and a level is compacted at most n / (w k / 2) times, so the rank
// error is at most about 2 n H / k with H = log2(n / k) levels (worst case);
// the alternating choice makes the errors largely cancel in practice. The
// quantiles are only used to propose candidate cutpoints: every bin statistic
// is computed exactly from the data afterwards.
class KLLSketch {
private:
  struct Item {
    double value;
    int weight;

    Item(double v, int w = 1) : value(v), weight(w) {}

    bool operator<(const Item &other) const { return value < other.value; }
  };

  using Compactor = std::vector<Item>;
  std::vector<Compactor> compactors;
  int k;          // Capacity of every level
  int n;          // Number of items processed
  int max_level;  // Levels at or above this one are never compacted

  // Non-recursive compaction of a level (and of the levels it overflows)
  void compact_level(size_t level) {
    std::queue<size_t> levels_to_compact;
    levels_to_compact.push(level);

    while (!levels_to_compact.empty()) {
      const size_t current_level = levels_to_compact.front();
      levels_to_compact.pop();

      if (current_level >= static_cast<size_t>(max_level)) {
        continue;
      }

      std::sort(compactors[current_level].begin(), compactors[current_level].end());

      // The next level must exist BEFORE a reference into `compactors` is taken
      if (current_level + 1 >= compactors.size()) {
        compactors.push_back(Compactor());
      }

      Compactor &compactor = compactors[current_level];
      Compactor &next = compactors[current_level + 1];
      const bool keep_lower = (current_level % 2 == 0);

      for (size_t i = 0; i + 1 < compactor.size(); i += 2) {
        const int w = compactor[i].weight + compactor[i + 1].weight;
        next.push_back(Item(keep_lower ? compactor[i].value : compactor[i + 1].value, w));
      }
      // Odd size: the remaining (largest) item is promoted unchanged
      if (compactor.size() % 2 == 1) {
        next.push_back(compactor.back());
      }
      compactor.clear();

      if (next.size() > static_cast<size_t>(k)) {
        levels_to_compact.push(current_level + 1);
      }
    }
  }

public:
  explicit KLLSketch(int k_param = 200)
    : k(k_param),
      n(0),
      max_level(20) {
    compactors.push_back(Compactor());
    compactors[0].reserve(static_cast<size_t>(k) + 1);
  }

  // Add a (finite) value to the sketch
  void update(double value) {
    compactors[0].push_back(Item(value));
    n++;

    if (compactors[0].size() > static_cast<size_t>(k)) {
      compact_level(0);
    }
  }

  // Estimate the q-th quantile, 0 < q < 1 (after at least one update)
  double get_quantile(double q) const {
    std::vector<Item> flattened;
    for (const auto &compactor : compactors) {
      flattened.insert(flattened.end(), compactor.begin(), compactor.end());
    }
    std::sort(flattened.begin(), flattened.end());

    int total_weight = 0;
    for (const auto &item : flattened) {
      total_weight += item.weight;
    }

    const int target_weight = static_cast<int>(q * total_weight);
    int cumulative_weight = 0;
    for (const auto &item : flattened) {
      cumulative_weight += item.weight;
      if (cumulative_weight >= target_weight) {
        return item.value;
      }
    }
    return flattened.back().value;
  }

  int count() const { return n; }
};

// ============================================================================
// Optimal numerical binning with a quantile sketch
// ============================================================================
class OBN_Sketch {
private:
  std::vector<double> feature;
  std::vector<int> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  bool monotonic;
  int max_iterations;
  int sketch_k;
  int dp_size_limit;

  int total_pos;  // Total events (target=1)
  int total_neg;  // Total non-events (target=0)
  double obs_min;  // smallest observation (may be -Inf)
  double obs_max;  // largest observation (may be +Inf)

  std::unique_ptr<KLLSketch> sketch;
  std::vector<NumericalBin> bins;
  std::vector<double> cutpoints;

  // Sizes, missing values and 0/1 targets are handled by the exported
  // wrapper before the class is built. One class may be absent once rows
  // with a missing feature are dropped; WoE and IV are then 0.
  void validate_inputs() const {
    if (min_bins < 2) {
      throw std::invalid_argument("min_bins must be >= 2.");
    }
    if (max_bins < min_bins) {
      throw std::invalid_argument("max_bins must be >= min_bins.");
    }
    if (bin_cutoff <= 0.0 || bin_cutoff >= 1.0) {
      throw std::invalid_argument("bin_cutoff must be between 0 and 1.");
    }
    if (sketch_k < 10) {
      throw std::invalid_argument("sketch_k must be >= 10.");
    }
    if (max_iterations < 1) {
      throw std::invalid_argument("max_iterations must be >= 1.");
    }
  }

  // The sketch summarises the finite values only, so candidate cutpoints are
  // always finite; +/-Inf observations still count in the first / last bin.
  void build_sketch() {
    sketch.reset(new KLLSketch(sketch_k));
    total_pos = 0;
    total_neg = 0;
    obs_min = feature[0];
    obs_max = feature[0];
    for (size_t i = 0; i < feature.size(); ++i) {
      obs_min = std::min(obs_min, feature[i]);
      obs_max = std::max(obs_max, feature[i]);
      if (std::isfinite(feature[i])) sketch->update(feature[i]);
      if (target[i] == 1) {
        total_pos++;
      } else {
        total_neg++;
      }
    }
  }

  // Candidate cutpoints: sketch quantiles, finer in the tails
  std::vector<double> extract_candidates() const {
    std::vector<double> candidates;
    candidates.reserve(40);

    for (double q = 0.01; q <= 0.1; q += 0.01) {
      candidates.push_back(sketch->get_quantile(q));
      candidates.push_back(sketch->get_quantile(1.0 - q));
    }
    for (double q = 0.1; q <= 0.9; q += 0.05) {
      candidates.push_back(sketch->get_quantile(q));
    }

    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    return candidates;
  }

  // Candidate cutpoints for small samples (n <= dp_size_limit): every
  // boundary between two consecutive distinct finite values.
  std::vector<double> extract_exact_candidates() const {
    std::vector<double> v;
    v.reserve(feature.size());
    for (double x : feature) if (std::isfinite(x)) v.push_back(x);
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    std::vector<double> candidates;
    for (size_t i = 0; i + 1 < v.size(); ++i) candidates.push_back(utils::safe_cut(v[i], v[i + 1]));
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    return candidates;
  }

  // Audit X-09 / E-01 (+E-05): the cutpoints used to be the max_bins - 1
  // candidates with the largest single-cut IV. Single-cut IVs of adjacent
  // candidates are almost equal, so the chosen cuts clustered next to each
  // other (-29% IV against the optimum over the same candidates), and the
  // exact DP only ran for n <= 50 with an unsmoothed sum of |IV| and no
  // minimum support (bins of 1-2 observations). Now both branches run the
  // exact contiguous partition DP (utils::sketch_partition_dp, the
  // ob::partition_dp recursion) over the slots between
  // consecutive candidates: the objective is the normalized Jeffreys IV
  // (alpha = 0.5, K = number of bins; the package reference estimator) and
  // every bin must hold at least ceil(bin_cutoff * n) observations inside the
  // DP.
  //
  // Number of bins (audit Section 15.5, minimum evidence): the DP solution for
  // every K in [min_bins, max_bins] is scored with the Schwarz (1978) BIC of
  // the binned event-rate model, ob::bic_score() = binomial log-likelihood -
  // (K - 1) / 2 * log(n), and the highest score wins (ties: fewer bins). A bin
  // is kept only when it raises the log-likelihood by more than log(n) / 2,
  // so max_bins is a ceiling, not a target. Picking the K with the largest
  // training IV (IV never falls when K grows) over-split small samples and
  // rare events (n = 40-60, 0.2%-1% events) in the 20-seed hold-out gate.
  //
  // With monotonic = TRUE the DP is run with the monotone (smoothed-odds)
  // constraint in both directions and the BIC decides between them, as it
  // does between values of K (audit Section 7.2, T4): optimizing the unconstrained IV and then pooling violators (the
  // later enforce_monotonicity step) would bring back a greedy, worse
  // partition on non-monotone signals.
  //
  // Fallbacks, in order, when no K in [min_bins, max_bins] meets the support
  // (e.g. 90% of the rows on one value): K = min_bins bins with the largest
  // feasible minimum size; then the monotone constraint is dropped; and only
  // then fewer than min_bins bins (features with too few distinct values,
  // e.g. a binary flag).
  void select_optimal_cutpoints(const std::vector<double> &candidates) {
    cutpoints.clear();
    const size_t C = candidates.size();
    std::vector<int64_t> slot_pos(C + 1, 0), slot_neg(C + 1, 0);
    for (size_t i = 0; i < feature.size(); ++i) {
      // first candidate >= value: slot s holds (candidates[s-1], candidates[s]]
      const size_t s = static_cast<size_t>(
        std::lower_bound(candidates.begin(), candidates.end(), feature[i]) - candidates.begin());
      if (target[i]) slot_pos[s]++; else slot_neg[s]++;
    }

    const int64_t support = std::max<int64_t>(1, static_cast<int64_t>(
      std::ceil(bin_cutoff * static_cast<double>(feature.size()))));
    const int m = static_cast<int>(C + 1);
    const int k_hi = std::min(max_bins, m);
    const int k_lo = std::min(min_bins, k_hi);

    // BIC of the partition of the slots given by `cuts` (start slot of bins
    // 2..K), to be maximized.
    auto bic_of = [&](const std::vector<int> &cuts) {
      std::vector<double> bp, bn;
      size_t s0 = 0;
      for (size_t q = 0; q <= cuts.size(); ++q) {
        const size_t s1 = q < cuts.size() ? static_cast<size_t>(cuts[q]) : slot_pos.size();
        double a = 0.0, b = 0.0;
        for (size_t t = s0; t < s1; ++t) {
          a += static_cast<double>(slot_pos[t]);
          b += static_cast<double>(slot_neg[t]);
        }
        bp.push_back(a);
        bn.push_back(b);
        s0 = s1;
      }
      return ob::bic_score(bp, bn);
    };
    ob::DpResult best{-std::numeric_limits<double>::infinity(), {}};
    double best_score = -std::numeric_limits<double>::infinity();
    auto stage = [&](int k_from, int k_to, int64_t min_count, bool mono) {
      bool found = false;
      for (int k = k_from; k <= k_to; ++k) {
        for (int dir = 0; dir < (mono ? 2 : 1); ++dir) {
          const ob::DpResult r = utils::sketch_partition_dp(slot_pos, slot_neg, k, mono,
                                                  /*asc=*/dir == 0, min_count);
          if (!std::isfinite(r.iv)) continue;
          const double sc = bic_of(r.cuts);
          if (!found || sc > best_score + 1e-12) {
            best = r;
            best_score = sc;
            found = true;
          }
        }
      }
      return found;
    };

    // Largest minimum bin size (<= support) for which k bins are feasible,
    // 0 if none: feasibility is monotone in min_count, so bisect.
    auto max_support = [&](int k, bool mono) {
      int64_t lo = 0, hi = support;
      while (lo < hi) {
        const int64_t mid = lo + (hi - lo + 1) / 2;
        bool ok = false;
        for (int dir = 0; dir < (mono ? 2 : 1) && !ok; ++dir)
          ok = std::isfinite(utils::sketch_partition_dp(slot_pos, slot_neg, k, mono,
                                                        dir == 0, mid).iv);
        if (ok) lo = mid; else hi = mid - 1;
      }
      return lo;
    };

    bool found = false;
    for (int few = 0; few < 2 && !found; ++few) {
      const int a = few ? 2 : k_lo, b = few ? k_lo - 1 : k_hi;
      for (int pass = 0; pass < (monotonic ? 2 : 1) && !found; ++pass) {
        const bool mono = monotonic && pass == 0;
        found = stage(a, b, support, mono);
        if (!found && a <= b) {
          // No k in [a, b] meets bin_cutoff: use k = a bins (fewest allowed;
          // enforce_bin_cutoff() would merge down to it anyway) that are as
          // large as possible.
          const int64_t mc = max_support(a, mono);
          if (mc > 0) found = stage(a, a, mc, mono);
        }
      }
    }
    if (found) {
      for (int c : best.cuts) cutpoints.push_back(candidates[static_cast<size_t>(c) - 1]);
    }

    create_initial_bins();
  }

  // Bins [min, c1], (c1, c2], ..., (c_m, max]; each observation is located by
  // binary search (first cutpoint >= value), which is exactly the bin the
  // former linear scan picked.
  void create_initial_bins() {
    bins.clear();

    const double min_val = obs_min;
    const double max_val = obs_max;

    if (cutpoints.empty()) {
      bins.push_back(NumericalBin(min_val, max_val));
    } else {
      bins.push_back(NumericalBin(min_val, cutpoints[0]));
      for (size_t i = 0; i + 1 < cutpoints.size(); ++i) {
        bins.push_back(NumericalBin(cutpoints[i], cutpoints[i + 1]));
      }
      bins.push_back(NumericalBin(cutpoints.back(), max_val));
    }

    for (size_t i = 0; i < feature.size(); ++i) {
      const size_t b = static_cast<size_t>(
        std::lower_bound(cutpoints.begin(), cutpoints.end(), feature[i]) - cutpoints.begin());
      bins[b].add_value(target[i]);
    }
  }

  // Merge bins below bin_cutoff into the neighbour with the closest event
  // rate, down to min_bins. Empty bins -- a candidate equal to the maximum,
  // or fewer distinct values than min_bins -- are merged even below
  // min_bins: they carry no observation and used to be returned as bins
  // with count 0 and duplicate cutpoints.
  void enforce_bin_cutoff() {
    const int min_count = static_cast<int>(std::ceil(bin_cutoff * static_cast<double>(feature.size())));

    bool any_change = true;
    while (any_change) {
      any_change = false;

      for (size_t i = 0; i < bins.size() && bins.size() > 1; ++i) {
        if (bins[i].count >= min_count) continue;
        if (bins.size() <= static_cast<size_t>(min_bins) && bins[i].count > 0) continue;

        size_t best_neighbor = i + 1;
        double min_diff = std::numeric_limits<double>::max();
        if (i > 0) {
          min_diff = std::fabs(bins[i].event_rate() - bins[i - 1].event_rate());
          best_neighbor = i - 1;
        }
        if (i + 1 < bins.size()) {
          const double diff = std::fabs(bins[i].event_rate() - bins[i + 1].event_rate());
          if (diff < min_diff) {
            best_neighbor = i + 1;
          }
        }

        // Always merge into the lower index
        if (best_neighbor < i) {
          bins[best_neighbor].merge_with(bins[i]);
          bins.erase(bins.begin() + static_cast<std::ptrdiff_t>(i));
        } else {
          bins[i].merge_with(bins[best_neighbor]);
          bins.erase(bins.begin() + static_cast<std::ptrdiff_t>(best_neighbor));
        }
        any_change = true;
        break;
      }
    }

    update_cutpoints_from_bins();
  }

  void calculate_initial_woe() {
    for (auto &bin : bins) {
      bin.calculate_metrics(total_pos, total_neg);
    }
  }

  // Pool adjacent violators until the WoE is monotonic or min_bins is reached
  void enforce_monotonicity() {
    if (!monotonic || bins.size() <= 1) {
      return;
    }

    const bool increasing = bins.back().woe >= bins.front().woe;

    bool any_change = true;
    while (any_change && bins.size() > static_cast<size_t>(min_bins)) {
      any_change = false;
      for (size_t i = 0; i + 1 < bins.size(); ++i) {
        const bool violation = (increasing && bins[i].woe > bins[i + 1].woe + EPSILON) ||
          (!increasing && bins[i].woe < bins[i + 1].woe - EPSILON);
        if (violation) {
          bins[i].merge_with(bins[i + 1]);
          bins.erase(bins.begin() + static_cast<std::ptrdiff_t>(i) + 1);
          bins[i].calculate_metrics(total_pos, total_neg);
          any_change = true;
          break;
        }
      }
    }

    update_cutpoints_from_bins();
  }

  void update_cutpoints_from_bins() {
    cutpoints.clear();
    for (size_t i = 1; i < bins.size(); ++i) {
      cutpoints.push_back(bins[i].lower_bound);
    }
  }

public:
  OBN_Sketch(const std::vector<double> &feature_,
             const std::vector<int> &target_,
             int min_bins_ = 3,
             int max_bins_ = 5,
             double bin_cutoff_ = 0.05,
             bool monotonic_ = true,
             int max_iterations_ = 1000,
             int sketch_k_ = 200)
    : feature(feature_),
      target(target_),
      min_bins(min_bins_),
      max_bins(max_bins_),
      bin_cutoff(bin_cutoff_),
      monotonic(monotonic_),
      max_iterations(max_iterations_),
      sketch_k(sketch_k_),
      dp_size_limit(50),
      total_pos(0),
      total_neg(0),
      obs_min(0.0),
      obs_max(0.0) {}

  Rcpp::List fit() {
    try {
      validate_inputs();
      build_sketch();
      select_optimal_cutpoints(feature.size() <= static_cast<size_t>(dp_size_limit)
                                 ? extract_exact_candidates()
                                 : extract_candidates());
      enforce_bin_cutoff();
      calculate_initial_woe();
      enforce_monotonicity();

      // The DP returns at most max_bins - 1 cutpoints
      // and every later step only merges, so bins.size() <= max_bins holds
      // here and no further reduction is needed: the binning has converged.
      const bool converged_flag = true;
      const int iterations_done = 0;

      const size_t n_bins = bins.size();
      const R_xlen_t nb = static_cast<R_xlen_t>(n_bins);

      NumericVector bin_lower(nb);
      NumericVector bin_upper(nb);
      NumericVector bin_woe(nb);
      NumericVector bin_iv(nb);
      IntegerVector bin_count(nb);
      IntegerVector bin_count_pos(nb);
      IntegerVector bin_count_neg(nb);
      NumericVector ids(nb);
      CharacterVector bin_labels(nb);
      double total_iv_value = 0.0;

      for (R_xlen_t i = 0; i < nb; ++i) {
        const NumericalBin &b = bins[static_cast<size_t>(i)];
        bin_lower[i] = b.lower_bound;
        bin_upper[i] = b.upper_bound;
        bin_woe[i] = b.woe;
        bin_iv[i] = b.iv;
        bin_count[i] = b.count;
        bin_count_pos[i] = b.count_pos;
        bin_count_neg[i] = b.count_neg;
        ids[i] = static_cast<double>(i + 1);
        total_iv_value += b.iv;

        // "(lower;upper]" like every sibling algorithm. The first bin is
        // labelled from -Inf and the last up to +Inf: they hold everything
        // below the first / above the last cutpoint (the observed minimum is
        // inside the first bin, which "(min;" wrongly excluded).
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6);
        if (i == 0) {
          oss << "(-Inf;";
        } else {
          oss << "(" << b.lower_bound << ";";
        }
        if (i == nb - 1) {
          oss << "+Inf]";
        } else {
          oss << b.upper_bound << "]";
        }
        bin_labels[i] = oss.str();
      }

      return Rcpp::List::create(
        Named("id") = ids,
        Named("bin") = bin_labels,
        Named("bin_lower") = bin_lower,
        Named("bin_upper") = bin_upper,
        Named("woe") = bin_woe,
        Named("iv") = bin_iv,
        Named("count") = bin_count,
        Named("count_pos") = bin_count_pos,
        Named("count_neg") = bin_count_neg,
        Named("total_iv") = total_iv_value,
        Named("cutpoints") = cutpoints,
        Named("converged") = converged_flag,
        Named("iterations") = iterations_done);

    } catch (const std::exception &e) {
      Rcpp::stop("Binning error: " + std::string(e.what()));
    }
  }
};

// ============================================================================
// Exported function for R
// ============================================================================

// Audit X-11 (B-05, C-09, E-14): honest monotonicity flag. The merge loops
// that enforce monotonicity stop at min_bins and used to return a WoE that is
// not monotonic with no signal at all (converged = TRUE, a direction label
// that only states the intended direction). The final WoE of the regular bins
// (a missing-value bin labelled "NA" is excluded) is now verified and
// reported as `monotonic`: non-decreasing for dir > 0, non-increasing for
// dir < 0, either for dir == 0 (tolerance 1e-9). The same order is required
// of the reference WoE, log((pos + 0.5) / (neg + 0.5)) up to a constant (the
// package's normalized Jeffreys estimator), so that the flag does not depend
// on how ties of the raw event rate (e.g. several event-free bins) are
// smoothed. A warning is raised when monotonicity was requested and is not
// met. The partition is unchanged.
static void flag_monotonic_result(Rcpp::List &res, bool requested, int dir,
                                  const char *engine) {
  const Rcpp::NumericVector woe = res["woe"];
  const Rcpp::NumericVector pos = res["count_pos"];
  const Rcpp::NumericVector neg = res["count_neg"];
  const Rcpp::CharacterVector lab = res["bin"];
  bool inc = true, dec = true, rinc = true, rdec = true, have = false;
  double prev = 0.0, rprev = 0.0;
  for (R_xlen_t i = 0; i < woe.size(); ++i) {
    if (i < lab.size() && std::string(lab[i]) == "NA") continue;
    const double w = woe[i];
    const double rw = std::log((pos[i] + 0.5) / (neg[i] + 0.5));
    if (have) {
      if (w < prev - 1e-9) inc = false;
      if (w > prev + 1e-9) dec = false;
      if (rw < rprev - 1e-9) rinc = false;
      if (rw > rprev + 1e-9) rdec = false;
    }
    prev = w;
    rprev = rw;
    have = true;
  }
  inc = inc && rinc;
  dec = dec && rdec;
  const bool mono = dir > 0 ? inc : (dir < 0 ? dec : (inc || dec));
  res.push_back(Rcpp::wrap(mono), "monotonic");
  if (requested && !mono) {
    Rcpp::warning("%s: a monotonic WoE was requested but the returned WoE "
                  "(or its reference estimate, 0.5 per bin and class) is not "
                  "monotonic; see $monotonic.", engine);
  }
}

// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_sketch(
    Rcpp::IntegerVector target,
    Rcpp::NumericVector feature,
    int min_bins = 3,
    int max_bins = 5,
    double bin_cutoff = 0.05,
    int max_n_prebins = 20,
    bool monotonic = true,
    double convergence_threshold = 1e-6,
    int max_iterations = 1000,
    int sketch_k = 200) {
  // max_n_prebins and convergence_threshold are accepted for API
  // compatibility: the candidate grid is a fixed set of ~37 sketch quantiles
  // and the selected cutpoints never need an iterative reduction.
  (void)max_n_prebins;
  (void)convergence_threshold;

  if (feature.size() == 0 || target.size() == 0) {
    Rcpp::stop("Feature and target cannot be empty.");
  }
  if (feature.size() != target.size()) {
    Rcpp::stop("Feature and target must have the same size.");
  }

  std::vector<double> feature_vec;
  std::vector<int> target_vec;
  feature_vec.reserve(static_cast<size_t>(feature.size()));
  target_vec.reserve(static_cast<size_t>(target.size()));

  bool has_zero = false;
  bool has_one = false;
  for (R_xlen_t i = 0; i < target.size(); ++i) {
    if (IntegerVector::is_na(target[i])) {
      Rcpp::stop("Target cannot contain missing values (NA).");
    }
    if (target[i] != 0 && target[i] != 1) {
      Rcpp::stop("Target must contain only 0 and 1.");
    }
    if (target[i] == 0) has_zero = true; else has_one = true;
  }
  if (!has_zero || !has_one) {
    Rcpp::stop("Target must contain both 0 and 1.");
  }

  // Rows with a missing feature (NA / NaN) are dropped silently; +/-Inf are
  // kept as extreme values.
  double fin_min = std::numeric_limits<double>::infinity();
  double fin_max = -std::numeric_limits<double>::infinity();
  for (R_xlen_t i = 0; i < feature.size(); ++i) {
    const double v = feature[i];
    if (std::isnan(v)) continue;
    if (std::isfinite(v)) {
      fin_min = std::min(fin_min, v);
      fin_max = std::max(fin_max, v);
    }
    feature_vec.push_back(v);
    target_vec.push_back(target[i]);
  }
  if (feature_vec.empty()) {
    Rcpp::stop("All feature values are missing (NA/NaN); nothing to bin.");
  }

  const double min_val = *std::min_element(feature_vec.begin(), feature_vec.end());
  const double max_val = *std::max_element(feature_vec.begin(), feature_vec.end());

  // Fewer than two distinct finite values (constant feature, or only +/-Inf
  // besides one value): a single bin. (A tolerance of 1e-10 used to collapse
  // genuinely distinct small-scale features as well.)
  if (!(fin_min < fin_max)) {
    int count_pos = 0;
    int count_neg = 0;
    for (int t : target_vec) {
      if (t == 1) count_pos++;
      else count_neg++;
    }
    const int total_count = static_cast<int>(feature_vec.size());

    // A single bin holds every observation: WoE = log(1 / 1) = 0, IV = 0.
    const double woe = 0.0;
    const double iv = 0.0;

    return Rcpp::List::create(
      Named("id") = NumericVector::create(1),
      Named("bin") = CharacterVector::create("(-Inf;+Inf]"),
      Named("bin_lower") = NumericVector::create(min_val),
      Named("bin_upper") = NumericVector::create(max_val),
      Named("woe") = NumericVector::create(woe),
      Named("iv") = NumericVector::create(iv),
      Named("count") = IntegerVector::create(total_count),
      Named("count_pos") = IntegerVector::create(count_pos),
      Named("count_neg") = IntegerVector::create(count_neg),
      Named("total_iv") = iv,
      Named("cutpoints") = NumericVector::create(),
      Named("converged") = true,
      Named("iterations") = 0,
      Named("monotonic") = true);  // one bin: trivially monotonic (audit X-11)
  }

  OBN_Sketch sketch_binner(feature_vec, target_vec, min_bins, max_bins,
                           bin_cutoff, monotonic, max_iterations, sketch_k);
  Rcpp::List res = sketch_binner.fit();
  flag_monotonic_result(res, monotonic, 0,
                        "ob_numerical_sketch");
  return res;
}
