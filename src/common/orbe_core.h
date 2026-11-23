#ifndef OPTIMAL_BINNING_ORBE_CORE_H
#define OPTIMAL_BINNING_ORBE_CORE_H

#include <Rcpp.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

/**
 * @file orbe_core.h
 * @brief Order-Restricted Bayesian Evidence (ORBE) binning: shared engine
 *
 * The feature is reduced to an ordered sequence of atoms (distinct values or
 * categories). A partition of the atoms into contiguous bins is scored by the
 * sum of the Beta-Binomial log marginal likelihoods of its bins plus a log
 * prior on the partition. The engine maximizes that score exactly by dynamic
 * programming, with the minimum bin size, the number of bins and the shape of
 * the posterior WoE (monotone or unimodal) as constraints of the recursion.
 */

namespace OptimalBinning {
namespace orbe {

const double NEG_INF = -std::numeric_limits<double>::infinity();

// Shape imposed on the posterior WoE across bins.
enum class Shape { None, Ascending, Descending, Peak, Valley };

inline std::string shape_to_string(Shape s) {
  switch (s) {
    case Shape::Ascending: return "ascending";
    case Shape::Descending: return "descending";
    case Shape::Peak: return "peak";
    case Shape::Valley: return "valley";
    default: return "none";
  }
}

// Beta(a, b) prior with mean pi0 and `strength` pseudo-events of the minority
// class, so min(a, b) = strength and the prior is uniform when pi0 = 0.5.
inline void prior_parameters(double pi0, double strength, double& a, double& b) {
  const double minority = std::min(pi0, 1.0 - pi0);
  a = strength * pi0 / minority;
  b = strength * (1.0 - pi0) / minority;
}

// Log marginal likelihood of the 0/1 sequence of a bin with n rows and e
// events; c0 = lgamma(a + b) - lgamma(a) - lgamma(b).
inline double bin_log_evidence(double n, double e, double a, double b, double c0) {
  return R::lgammafn(a + e) + R::lgammafn(b + n - e) - R::lgammafn(a + b + n) + c0;
}

// Posterior mean of logit(theta) for a bin: digamma(A) - digamma(B).
inline double posterior_logit(double n, double e, double a, double b) {
  return R::digamma(a + e) - R::digamma(b + n - e);
}

// ---------------------------------------------------------------------------
// Posterior probability of an order among independent Beta(A_k, B_k) rates
// ---------------------------------------------------------------------------

// One step of the chain G_k(t) = integral_0^t G_{k-1}(u) dF_k(u), evaluated on
// the grid by the trapezoid rule in F_k (a Stieltjes sum).
inline void chain_step(const std::vector<double>& g_prev, const std::vector<double>& cdf,
                       std::vector<double>& g_out) {
  const size_t t = cdf.size();
  g_out.assign(t, 0.0);
  for (size_t l = 1; l < t; ++l) {
    g_out[l] = g_out[l - 1] + 0.5 * (g_prev[l - 1] + g_prev[l]) * (cdf[l] - cdf[l - 1]);
  }
}

// Probability that the rates follow the shape. For Peak and Valley the result
// is summed over the position of the mode, so it includes the monotone orders.
inline double order_probability(std::vector<double> A, std::vector<double> B, Shape shape) {
  const size_t k = A.size();
  if (k <= 1 || shape == Shape::None) return 1.0;
  if (shape == Shape::Descending) {
    std::reverse(A.begin(), A.end());
    std::reverse(B.begin(), B.end());
  }
  // theta -> 1 - theta turns a valley into a peak
  if (shape == Shape::Valley) std::swap(A, B);

  // Common grid: 0, 1 and equally spaced quantiles of every posterior. The
  // quadrature error grows with k, so the grid is refined with it.
  const int per_bin = std::max(64, 16 * static_cast<int>(k));
  std::vector<double> grid;
  grid.reserve(k * static_cast<size_t>(per_bin) + 2);
  grid.push_back(0.0);
  grid.push_back(1.0);
  for (size_t i = 0; i < k; ++i) {
    for (int l = 0; l < per_bin; ++l) {
      const double q = R::qbeta((l + 0.5) / per_bin, A[i], B[i], 1, 0);
      if (q > 0.0 && q < 1.0) grid.push_back(q);
    }
  }
  std::sort(grid.begin(), grid.end());
  grid.erase(std::unique(grid.begin(), grid.end()), grid.end());
  const size_t t = grid.size();

  // Posterior CDF of every bin on the grid
  std::vector<std::vector<double>> cdf(k, std::vector<double>(t));
  for (size_t i = 0; i < k; ++i) {
    for (size_t l = 0; l < t; ++l) cdf[i][l] = R::pbeta(grid[l], A[i], B[i], 1, 0);
    cdf[i][0] = 0.0;
    cdf[i][t - 1] = 1.0;
  }

  const std::vector<double> ones(t, 1.0);
  double prob = 0.0;

  if (shape == Shape::Ascending || shape == Shape::Descending) {
    std::vector<double> g = ones, g_next;
    for (size_t i = 0; i < k; ++i) {
      chain_step(g, cdf[i], g_next);
      g.swap(g_next);
    }
    prob = g[t - 1];
  } else {
    // left[p]: P(theta_1 < ... < theta_p < t); right[p]: P(theta_K < ... < theta_p < t)
    std::vector<std::vector<double>> left(k + 1), right(k + 2);
    left[0] = ones;
    for (size_t p = 1; p <= k; ++p) chain_step(left[p - 1], cdf[p - 1], left[p]);
    right[k + 1] = ones;
    for (size_t p = k; p >= 1; --p) chain_step(right[p + 1], cdf[p - 1], right[p]);
    // Mode at bin p: both sides stay below theta_p
    for (size_t p = 1; p <= k; ++p) {
      const std::vector<double>& lo = left[p - 1];
      const std::vector<double>& hi = right[p + 1];
      const std::vector<double>& f = cdf[p - 1];
      for (size_t l = 1; l < t; ++l) {
        prob += 0.5 * (lo[l - 1] * hi[l - 1] + lo[l] * hi[l]) * (f[l] - f[l - 1]);
      }
    }
  }
  return std::min(1.0, std::max(0.0, prob));
}

// ---------------------------------------------------------------------------
// Exact dynamic program
// ---------------------------------------------------------------------------

// Best additive score and bin ends (atom prefix indices) for every bin count.
struct DpResult {
  std::vector<double> best;            // best[k], NEG_INF when infeasible
  std::vector<std::vector<int>> ends;  // ends[k]: k increasing indices, last = u
};

struct Entry {
  double rate;  // sign-adjusted posterior logit of the previous bin
  double value; // best score of a partition ending with that bin
  int code;     // predecessor boundary (and phase, for unimodal shapes)
};

/**
 * State (i, j): the last bin holds atoms i+1..j. A bin may follow (h, i] only
 * when its posterior logit respects the shape, so the predecessors of every
 * boundary i are sorted by rate and queried by binary search on a running
 * maximum. Time O(kmax * u^2 * log u).
 *
 * @param mode 0 = no shape, 1 = increasing in sign * rate, 2 = peak in sign * rate
 */
inline DpResult run_dp(int u, int kmax, const std::vector<double>& score,
                       const std::vector<double>& rate, int mode, double sign) {
  const size_t w = static_cast<size_t>(u) + 1;
  auto ix = [w](int i, int j) { return static_cast<size_t>(i) * w + static_cast<size_t>(j); };

  DpResult out;
  out.best.assign(static_cast<size_t>(kmax) + 1, NEG_INF);
  out.ends.resize(static_cast<size_t>(kmax) + 1);

  // Phase U: still increasing. Phase D (unimodal only): already decreasing.
  std::vector<double> prev_u(w * w, NEG_INF), cur_u(w * w, NEG_INF);
  std::vector<double> prev_d, cur_d;
  if (mode == 2) {
    prev_d.assign(w * w, NEG_INF);
    cur_d.assign(w * w, NEG_INF);
  }
  std::vector<std::vector<int>> back_u(static_cast<size_t>(kmax) + 1);
  std::vector<std::vector<int>> back_d(static_cast<size_t>(kmax) + 1);
  std::vector<int> end_i(static_cast<size_t>(kmax) + 1, -1);
  std::vector<int> end_phase(static_cast<size_t>(kmax) + 1, 0);

  // One bin: atoms 1..j
  for (int j = 1; j <= u; ++j) prev_u[ix(0, j)] = score[ix(0, j)];
  out.best[1] = score[ix(0, u)];
  end_i[1] = 0;

  std::vector<Entry> ent_u, ent_d;
  std::vector<double> run_val;
  std::vector<int> run_code;
  auto by_rate = [](const Entry& x, const Entry& y) {
    if (x.rate != y.rate) return x.rate < y.rate;
    return x.code < y.code;
  };

  int last_layer = 1;
  for (int k = 2; k <= kmax; ++k) {
    const size_t ks = static_cast<size_t>(k);
    std::fill(cur_u.begin(), cur_u.end(), NEG_INF);
    back_u[ks].assign(w * w, -1);
    if (mode == 2) {
      std::fill(cur_d.begin(), cur_d.end(), NEG_INF);
      back_d[ks].assign(w * w, INT_MIN);
    }
    bool any = false;

    for (int i = k - 1; i < u; ++i) {
      // Partitions of atoms 1..i into k-1 bins, by their last bin (h, i]
      ent_u.clear();
      ent_d.clear();
      for (int h = 0; h < i; ++h) {
        const double vu = prev_u[ix(h, i)];
        const double r = sign * rate[ix(h, i)];
        if (vu > NEG_INF) {
          ent_u.push_back({r, vu, h});
          if (mode == 2) ent_d.push_back({r, vu, -(h + 1)});
        }
        if (mode == 2) {
          const double vd = prev_d[ix(h, i)];
          if (vd > NEG_INF) ent_d.push_back({r, vd, h});
        }
      }
      if (ent_u.empty() && ent_d.empty()) continue;

      if (mode == 0) {
        // No shape: every predecessor is admissible
        double bv = NEG_INF;
        int bc = -1;
        for (const Entry& e : ent_u) {
          if (e.value > bv) { bv = e.value; bc = e.code; }
        }
        for (int j = i + 1; j <= u; ++j) {
          const double f = score[ix(i, j)];
          if (f == NEG_INF) continue;
          cur_u[ix(i, j)] = f + bv;
          back_u[ks][ix(i, j)] = bc;
          any = true;
        }
        continue;
      }

      if (!ent_u.empty()) {
        // Running maximum over predecessors with a smaller rate
        std::sort(ent_u.begin(), ent_u.end(), by_rate);
        const size_t m = ent_u.size();
        run_val.assign(m, NEG_INF);
        run_code.assign(m, -1);
        for (size_t t = 0; t < m; ++t) {
          if (t == 0 || ent_u[t].value > run_val[t - 1]) {
            run_val[t] = ent_u[t].value;
            run_code[t] = ent_u[t].code;
          } else {
            run_val[t] = run_val[t - 1];
            run_code[t] = run_code[t - 1];
          }
        }
        for (int j = i + 1; j <= u; ++j) {
          const double f = score[ix(i, j)];
          if (f == NEG_INF) continue;
          const double q = sign * rate[ix(i, j)];
          // Entries strictly below q
          size_t lo = 0, hi = m;
          while (lo < hi) {
            const size_t mid = (lo + hi) / 2;
            if (ent_u[mid].rate < q) lo = mid + 1; else hi = mid;
          }
          if (lo == 0) continue;
          cur_u[ix(i, j)] = f + run_val[lo - 1];
          back_u[ks][ix(i, j)] = run_code[lo - 1];
          any = true;
        }
      }

      if (mode == 2 && !ent_d.empty()) {
        // Running maximum, from the right, over predecessors with a larger rate
        std::sort(ent_d.begin(), ent_d.end(), by_rate);
        const size_t m = ent_d.size();
        run_val.assign(m, NEG_INF);
        run_code.assign(m, INT_MIN);
        for (size_t t = m; t-- > 0;) {
          if (t == m - 1 || ent_d[t].value > run_val[t + 1]) {
            run_val[t] = ent_d[t].value;
            run_code[t] = ent_d[t].code;
          } else {
            run_val[t] = run_val[t + 1];
            run_code[t] = run_code[t + 1];
          }
        }
        for (int j = i + 1; j <= u; ++j) {
          const double f = score[ix(i, j)];
          if (f == NEG_INF) continue;
          const double q = sign * rate[ix(i, j)];
          // First entry strictly above q
          size_t lo = 0, hi = m;
          while (lo < hi) {
            const size_t mid = (lo + hi) / 2;
            if (ent_d[mid].rate <= q) lo = mid + 1; else hi = mid;
          }
          if (lo == m) continue;
          cur_d[ix(i, j)] = f + run_val[lo];
          back_d[ks][ix(i, j)] = run_code[lo];
          any = true;
        }
      }
    }

    if (!any) break;  // no partition with k bins, hence none with more
    last_layer = k;

    // Best partition of all u atoms into k bins
    for (int i = k - 1; i < u; ++i) {
      if (cur_u[ix(i, u)] > out.best[ks]) {
        out.best[ks] = cur_u[ix(i, u)];
        end_i[ks] = i;
        end_phase[ks] = 0;
      }
      if (mode == 2 && cur_d[ix(i, u)] > out.best[ks]) {
        out.best[ks] = cur_d[ix(i, u)];
        end_i[ks] = i;
        end_phase[ks] = 1;
      }
    }
    prev_u.swap(cur_u);
    if (mode == 2) prev_d.swap(cur_d);
    Rcpp::checkUserInterrupt();
  }

  // Recover the bin ends of every feasible bin count
  for (int k = 1; k <= last_layer; ++k) {
    const size_t ks = static_cast<size_t>(k);
    if (out.best[ks] == NEG_INF) continue;
    std::vector<int> ends(ks);
    int j = u, i = end_i[ks], phase = end_phase[ks];
    for (int kk = k; kk >= 1; --kk) {
      ends[static_cast<size_t>(kk) - 1] = j;
      if (kk == 1) break;
      int h;
      if (phase == 0) {
        h = back_u[static_cast<size_t>(kk)][ix(i, j)];
      } else {
        const int code = back_d[static_cast<size_t>(kk)][ix(i, j)];
        if (code < 0) { h = -(code + 1); phase = 0; } else { h = code; }
      }
      j = i;
      i = h;
    }
    out.ends[ks] = ends;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Model selection across bin counts and shapes
// ---------------------------------------------------------------------------

struct Fit {
  std::vector<int> ends;           // bin ends as atom prefix indices
  Shape shape = Shape::None;       // shape of the selected model
  double log_evidence = NEG_INF;   // score of the selected partition
  double order_probability = 1.0;  // posterior probability of the shape
  std::vector<double> score_by_k;  // score of the best partition of each size
};

/**
 * @param cum_n,cum_e prefix sums of rows and events over the atoms (size u + 1)
 * @param n_min minimum number of rows in a bin
 * @param log_prior log prior of a partition with k bins, indexed by k
 * @param shapes candidate shapes; with two (auto) the direction is unknown
 * @param order_factor add the Bayes factor of the order restriction
 */
inline Fit fit(const std::vector<double>& cum_n, const std::vector<double>& cum_e,
               int min_bins, int max_bins, double n_min, double a, double b,
               const std::vector<double>& log_prior, const std::vector<Shape>& shapes,
               bool order_factor) {
  const int u = static_cast<int>(cum_n.size()) - 1;
  const int kmax = std::min(max_bins, u);
  const size_t w = static_cast<size_t>(u) + 1;
  const double c0 = R::lgammafn(a + b) - R::lgammafn(a) - R::lgammafn(b);

  // Score and posterior logit of every candidate bin (i, j]
  std::vector<double> score(w * w, NEG_INF), rate(w * w, 0.0);
  for (int i = 0; i < u; ++i) {
    for (int j = i + 1; j <= u; ++j) {
      const double n = cum_n[static_cast<size_t>(j)] - cum_n[static_cast<size_t>(i)];
      if (n < n_min) continue;
      const double e = cum_e[static_cast<size_t>(j)] - cum_e[static_cast<size_t>(i)];
      const size_t id = static_cast<size_t>(i) * w + static_cast<size_t>(j);
      score[id] = bin_log_evidence(n, e, a, b, c0);
      rate[id] = posterior_logit(n, e, a, b);
    }
  }

  const bool unknown_direction = shapes.size() > 1;
  Fit best;
  bool have = false, best_in_range = false;
  int best_k = 0;

  for (Shape shape : shapes) {
    int mode = 1;
    double sign = 1.0;
    if (shape == Shape::None) mode = 0;
    if (shape == Shape::Peak || shape == Shape::Valley) mode = 2;
    if (shape == Shape::Descending || shape == Shape::Valley) sign = -1.0;

    const DpResult dp = run_dp(u, kmax, score, rate, mode, sign);
    std::vector<double> by_k(static_cast<size_t>(kmax), NA_REAL);

    for (int k = 1; k <= kmax; ++k) {
      const size_t ks = static_cast<size_t>(k);
      if (dp.best[ks] == NEG_INF) continue;
      double total = dp.best[ks] + log_prior[ks];
      double prob = 1.0;

      if (k > 1 && shape != Shape::None) {
        // Posterior Beta parameters of the bins
        std::vector<double> A(ks), B(ks);
        int lo = 0;
        for (size_t t = 0; t < ks; ++t) {
          const int hi = dp.ends[ks][t];
          const double n = cum_n[static_cast<size_t>(hi)] - cum_n[static_cast<size_t>(lo)];
          const double e = cum_e[static_cast<size_t>(hi)] - cum_e[static_cast<size_t>(lo)];
          A[t] = a + e;
          B[t] = b + n - e;
          lo = hi;
        }
        prob = order_probability(A, B, shape);
        if (order_factor) {
          // Posterior over prior mass of the order cone: 1/k! for one
          // direction, 2/k! for either, 2^(k-1)/k! for a unimodal order.
          double log_prior_mass = -R::lgammafn(k + 1.0);
          if (mode == 2) log_prior_mass += (k - 1.0) * M_LN2;
          else if (unknown_direction) log_prior_mass += M_LN2;
          total += std::log(std::max(prob, 1e-300)) - log_prior_mass;
        }
      }
      by_k[ks - 1] = total;

      // In-range bin counts win; below min_bins the largest count wins.
      const bool in_range = k >= min_bins;
      bool better = false;
      if (!have) better = true;
      else if (in_range != best_in_range) better = in_range;
      else if (in_range) better = total > best.log_evidence;
      else better = (k > best_k) || (k == best_k && total > best.log_evidence);

      if (better) {
        have = true;
        best_in_range = in_range;
        best_k = k;
        best.ends = dp.ends[ks];
        best.shape = shape;
        best.log_evidence = total;
        best.order_probability = prob;
      }
    }
    if (have && best.shape == shape) best.score_by_k = by_k;
  }
  return best;
}

// Largest number of atoms whose back-pointer tables stay near 80 MB. A fit
// with u atoms allocates min(max_bins, u) layers of (u + 1)^2 entries.
inline int atom_cap(int max_n_prebins, int max_bins, bool unimodal) {
  const double tables = unimodal ? 2.0 : 1.0;
  int cap = std::max(2, std::min(max_n_prebins, 1000));
  while (cap > 2) {
    const double layers = static_cast<double>(std::min(std::max(max_bins, 1), cap));
    if (tables * layers * (cap + 1.0) * (cap + 1.0) <= 2e7) break;
    --cap;
  }
  return cap;
}

// Minimum rows in a bin; the tolerance absorbs the rounding of cutoff * n.
inline double min_bin_rows(double bin_cutoff, double tot_n) {
  return std::min(tot_n, std::max(1.0, std::ceil(bin_cutoff * tot_n - 1e-9)));
}

// ---------------------------------------------------------------------------
// Posterior summaries of the fitted bins
// ---------------------------------------------------------------------------

struct BinSummary {
  std::vector<double> woe, woe_se, woe_lower, woe_upper, iv, event_rate;
  double total_iv = 0.0;
  double total_iv_corrected = 0.0;
};

/**
 * Posterior-mean WoE digamma(A) - digamma(B) - logit(pi0), its posterior
 * standard deviation, an equal-tailed credible interval, and the IV with its
 * first-order plug-in bias removed.
 */
inline BinSummary summarize(const std::vector<double>& n, const std::vector<double>& e,
                            double a, double b, double level) {
  const size_t k = n.size();
  double tot_n = 0.0, tot_e = 0.0;
  for (size_t i = 0; i < k; ++i) { tot_n += n[i]; tot_e += e[i]; }
  const double tot_ne = tot_n - tot_e;
  const double logit0 = std::log(tot_e) - std::log(tot_ne);
  const double tail = 0.5 * (1.0 - level);

  // Keeps the logit of an extreme quantile finite
  auto logit = [](double p) {
    p = std::min(std::max(p, 1e-300), 1.0 - 1e-16);
    return std::log(p) - std::log1p(-p);
  };

  BinSummary s;
  s.woe.resize(k); s.woe_se.resize(k); s.woe_lower.resize(k);
  s.woe_upper.resize(k); s.iv.resize(k); s.event_rate.resize(k);
  double bias = 0.0;
  for (size_t i = 0; i < k; ++i) {
    const double A = a + e[i], B = b + n[i] - e[i];
    s.woe[i] = R::digamma(A) - R::digamma(B) - logit0;
    s.woe_se[i] = std::sqrt(R::trigamma(A) + R::trigamma(B));
    s.woe_lower[i] = logit(R::qbeta(tail, A, B, 1, 0)) - logit0;
    s.woe_upper[i] = logit(R::qbeta(1.0 - tail, A, B, 1, 0)) - logit0;
    s.event_rate[i] = n[i] > 0.0 ? e[i] / n[i] : 0.0;
    const double p = e[i] / tot_e, q = (n[i] - e[i]) / tot_ne;
    s.iv[i] = (p - q) * s.woe[i];
    s.total_iv += s.iv[i];
    // Smoothed shares keep the bias terms finite in pure bins
    const double ps = A / (tot_e + k * a), qs = B / (tot_ne + k * b);
    bias += (1.0 - ps) * (1.0 + qs / ps) / (2.0 * tot_e) +
            (1.0 - qs) * (1.0 + ps / qs) / (2.0 * tot_ne);
  }
  s.total_iv_corrected = std::max(0.0, s.total_iv - bias);
  return s;
}

} // namespace orbe
} // namespace OptimalBinning

#endif // OPTIMAL_BINNING_ORBE_CORE_H
