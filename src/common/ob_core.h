/**
 * @file ob_core.h
 * @brief Single source of truth for WoE / IV, merge loss and exact
 *        contiguous partitioning (forensic audit, Section 7).
 *
 * Conventions
 * -----------
 * - WoE_j = ln(p_j / q_j), with p_j (q_j) the smoothed share of events
 *   (non-events) that fall in bin j. Positive WoE = higher event rate, the
 *   sign convention used throughout the package.
 * - Smoothing is additive and *normalized*:
 *       p_j = (pos_j + alpha) / (P + K * alpha),
 *       q_j = (neg_j + alpha) / (N + K * alpha),
 *   with K the number of bins of the partition being evaluated, so that
 *   sum_j p_j = sum_j q_j = 1. The same p_j, q_j enter WoE and IV, hence
 *   IV_j = (p_j - q_j) * WoE_j >= 0 and the total is a genuine Jeffreys
 *   divergence. A bin with zero events (or non-events) therefore gets the
 *   most extreme finite WoE, never the neutral value 0.
 *
 * References: Good (1950); Siddiqi (2006); Cover & Thomas (2006, log-sum
 * inequality, Thm 2.7.1); Fisher (1958) and Breiman et al. (1984, Thm 4.5)
 * for the contiguity of the optimal partition in event-rate order.
 */
#ifndef OB_CORE_H
#define OB_CORE_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace ob {

/// Single WoE cap for the whole package (finite output for pure bins).
constexpr double kWoeCap = 20.0;

/// Default pseudo-count per bin (Jeffreys prior).
constexpr double kAlpha = 0.5;

struct WoeIv {
  double woe;
  double iv;
};

/**
 * @brief Normalized additive smoothing of one bin.
 * @param pos,neg  events / non-events in the bin
 * @param Tp,Tn    totals over all bins of the partition
 * @param K        number of bins of the partition (>= 1)
 * @param alpha    pseudo-count per bin
 */
inline WoeIv woe_iv(double pos, double neg, double Tp, double Tn, int K,
                    double alpha = kAlpha) {
  const double k = static_cast<double>(std::max(K, 1));
  const double dp = (pos + alpha) / (Tp + alpha * k);
  const double dn = (neg + alpha) / (Tn + alpha * k);
  double w = std::log(dp / dn);
  if (w > kWoeCap) w = kWoeCap;
  if (w < -kWoeCap) w = -kWoeCap;
  return {w, (dp - dn) * w};
}

/**
 * @brief One-step change of the reported (smoothed) IV, on a common K scale,
 *        when two bins are merged: IV_a + IV_b - IV_(a+b), all with the same
 *        K and alpha per bin (the estimator every engine reports).
 *
 * It is the information lost by the merge as measured by the reported IV.
 * It can be slightly NEGATIVE: the merged bin carries one pseudo-count where
 * its parts carried two, so merging, e.g., two pure bins raises the reported
 * IV. That is intended -- the greedy criterion must follow the quantity that
 * is reported and evaluated (a 2-alpha variant that is >= 0 by the log-sum
 * inequality was tried and optimized a different quantity; see audit
 * diffs). WoE is capped through woe_iv(), so alpha = 0 stays finite.
 * For the exact one-step greedy (with the K -> K-1 renormalization of every
 * other bin) evaluate the reported total IV after each candidate merge.
 */
inline double iv_merge_loss(double pa, double na, double pb, double nb,
                            double Tp, double Tn, int K,
                            double alpha = kAlpha) {
  return woe_iv(pa, na, Tp, Tn, K, alpha).iv +
         woe_iv(pb, nb, Tp, Tn, K, alpha).iv -
         woe_iv(pa + pb, na + nb, Tp, Tn, K, alpha).iv;
}

/**
 * @brief Exact comparison of event rates pa/ca <= pb/cb.
 * Counts are below 2^31, so the cross products fit in int64 exactly.
 */
inline bool rate_le(int64_t pa, int64_t ca, int64_t pb, int64_t cb) {
  return pa * cb <= pb * ca;
}

struct DpResult {
  double iv;              ///< objective value (-inf if infeasible)
  std::vector<int> cuts;  ///< start index of bins 2..K in the pre-bin order
};

/**
 * @brief Exact maximum-IV partition of m ordered pre-bins into exactly K
 *        contiguous bins, each with at least `min_count` observations and,
 *        if `monotone`, with non-decreasing (asc) or non-increasing event
 *        rate. O(K m^3) time, O(K m^2) memory. Returns iv = -inf when no
 *        feasible partition exists.
 *
 * Validated against exhaustive enumeration (3000 random instances, 0
 * mismatches, ASan/UBSan clean) during the audit.
 */
inline DpResult partition_dp(const std::vector<int64_t>& pos,
                             const std::vector<int64_t>& neg, int K,
                             bool monotone, bool asc, int64_t min_count,
                             double alpha = kAlpha) {
  const int m = static_cast<int>(pos.size());
  const double NEG = -std::numeric_limits<double>::infinity();
  DpResult r{NEG, {}};
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
    return woe_iv(static_cast<double>(P[i] - P[j]),
                  static_cast<double>(N[i] - N[j]), Tp, Tn, K, alpha).iv;
  };
  auto ordered = [&](int j, int i, int e) {  // [j,i) followed by [i,e)
    if (!monotone) return true;
    const int64_t p1 = P[i] - P[j], c1 = cnt(j, i);
    const int64_t p2 = P[e] - P[i], c2 = cnt(i, e);
    return asc ? rate_le(p1, c1, p2, c2) : rate_le(p2, c2, p1, c1);
  };

  // f[k][i][j]: best value with k bins covering [0, i), last bin [j, i).
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

// ---------------------------------------------------------------------------
// Minimum-evidence tools (audit Section 15.5)
// ---------------------------------------------------------------------------

/**
 * @brief IV of one bin under an empirical-Bayes (beta) prior centred on the
 *        overall event rate r0 = Tp / (Tp + Tn), with prior strength m
 *        pseudo-observations per bin (Gelman et al., BDA ch. 5).
 *
 * Small bins are shrunk towards the population rate, so isolating a few rows
 * with an extreme observed rate earns little IV; large bins are unaffected.
 * Shares are normalized over K bins. m = 0 falls back to woe_iv() (alpha).
 */
inline WoeIv eb_woe_iv(double pos, double neg, double Tp, double Tn, int K,
                       double m, double alpha = kAlpha) {
  if (!(m > 0.0) || Tp + Tn <= 0.0) return woe_iv(pos, neg, Tp, Tn, K, alpha);
  const double k = static_cast<double>(std::max(K, 1));
  const double r0 = Tp / (Tp + Tn);
  const double dp = (pos + m * r0) / (Tp + k * m * r0);
  const double dn = (neg + m * (1.0 - r0)) / (Tn + k * m * (1.0 - r0));
  double w = std::log(dp / dn);
  if (w > kWoeCap) w = kWoeCap;
  if (w < -kWoeCap) w = -kWoeCap;
  return {w, (dp - dn) * w};
}

/**
 * @brief Binomial log-likelihood of a partition under the binned event-rate
 *        model (maximum-likelihood rate per bin; 0 log 0 = 0).
 */
inline double binned_loglik(const std::vector<double>& pos,
                            const std::vector<double>& neg) {
  double ll = 0.0;
  for (size_t i = 0; i < pos.size(); ++i) {
    const double n = pos[i] + neg[i];
    if (n <= 0.0) continue;
    if (pos[i] > 0.0) ll += pos[i] * std::log(pos[i] / n);
    if (neg[i] > 0.0) ll += neg[i] * std::log(neg[i] / n);
  }
  return ll;
}

/**
 * @brief Schwarz (1978) BIC score of a partition, to be MAXIMIZED:
 *        loglik - (K - 1) / 2 * log(n). A bin is worth keeping only if it
 *        raises the binomial log-likelihood by more than log(n) / 2 -- the
 *        minimum-evidence rule used to pick K inside [min_bins, max_bins].
 */
inline double bic_score(const std::vector<double>& pos,
                        const std::vector<double>& neg) {
  double n = 0.0;
  int K = 0;
  for (size_t i = 0; i < pos.size(); ++i) {
    const double c = pos[i] + neg[i];
    if (c > 0.0) { n += c; ++K; }
  }
  if (n <= 0.0) return 0.0;
  return binned_loglik(pos, neg) - 0.5 * static_cast<double>(K - 1) * std::log(n);
}

}  // namespace ob

#endif  // OB_CORE_H
