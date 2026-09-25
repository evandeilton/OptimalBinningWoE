// [[Rcpp::plugins(openmp)]]
// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
#include <limits>
#include <numeric>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

// -----------------------------------------------------------------------------
// Pairwise correlation engine behind obcorr().
//
// Every coefficient is computed on the pairwise-complete observations of the
// two columns, with EXACT tie detection. Earlier versions declared two values
// tied whenever they differed by less than an absolute 1e-10, which made every
// rank-based coefficient depend on the unit of measurement: multiplying a
// column by 1e-12 turned all its values into one tie and Spearman, Kendall and
// percentage bend came back NA, Hoeffding's D changed sign and the biweight
// silently fell back to Pearson. Correlations must be invariant to a positive
// rescaling of either variable, and with exact ties they are.
//
// All helpers live in an anonymous namespace (internal linkage), so they can
// never collide with a same-named function in another translation unit.
// -----------------------------------------------------------------------------
namespace {

constexpr double MAD_NORMAL_SCALE   = 1.4826; // robust scaling
constexpr double HOEFFDING_SCALE    = 30.0;   // Hmisc: 30x of the original D
constexpr double BIWEIGHT_C         = 9.0;    // WGCNA/Mosteller-Tukey
constexpr double PBEND_BETA_DEFAULT = 0.2;    // default Wilcox value
constexpr int    MIN_PAIRS_COR      = 3;      // Pearson/Spearman/Kendall/bicor/pbend/dcor
constexpr int    MIN_HOEFFDING_N    = 5;      // Hoeffding requires n >= 5
constexpr int    MIN_BIWEIGHT_N     = 8;

// Complete (x, y) pairs of two columns.
struct ValidPairs {
  std::vector<double> x, y;
  int size() const { return static_cast<int>(x.size()); }
};

void complete_pairs(const std::vector<double>& x, const std::vector<double>& y,
                    ValidPairs& P) {
  P.x.clear(); P.y.clear();
  P.x.reserve(x.size()); P.y.reserve(y.size());
  for (size_t i = 0; i < x.size(); ++i) {
    if (!std::isnan(x[i]) && !std::isnan(y[i])) {
      P.x.push_back(x[i]);
      P.y.push_back(y[i]);
    }
  }
}

double median_copy(std::vector<double> v) {
  if (v.empty()) return NA_REAL;
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  if (n % 2 == 0) return 0.5 * (v[n / 2 - 1] + v[n / 2]);
  return v[n / 2];
}

double mad_copy_scaled(const std::vector<double>& v, double med) {
  if (v.empty() || !R_finite(med)) return NA_REAL;
  std::vector<double> dev;
  dev.reserve(v.size());
  for (double x : v) if (!std::isnan(x)) dev.push_back(std::abs(x - med));
  if (dev.empty()) return NA_REAL;
  const double mad = median_copy(dev);
  if (!R_finite(mad)) return NA_REAL;
  return MAD_NORMAL_SCALE * mad;
}

// At most two distinct values (the input holds no NaN).
bool is_binary_or_two_values(const std::vector<double>& v) {
  if (v.empty()) return true;
  const double a = v[0];
  bool have_b = false;
  double b = 0.0;
  for (double z : v) {
    if (z == a) continue;
    if (!have_b) { b = z; have_b = true; continue; }
    if (z != b) return false;
  }
  return true;
}

// Mid-ranks (1-based, ties get the average rank) of a NaN-free vector.
void midranks(const std::vector<double>& v, std::vector<double>& r,
              std::vector<std::pair<double, int>>& vp) {
  const size_t n = v.size();
  r.assign(n, 0.0);
  vp.resize(n);
  for (size_t i = 0; i < n; ++i) vp[i] = {v[i], static_cast<int>(i)};
  std::sort(vp.begin(), vp.end());
  size_t i = 0;
  while (i < n) {
    size_t j = i;
    const double val = vp[i].first;
    while (j < n && vp[j].first == val) ++j;
    const double avg = (static_cast<double>(i) + static_cast<double>(j) + 1.0) / 2.0;
    for (size_t k = i; k < j; ++k) r[static_cast<size_t>(vp[k].second)] = avg;
    i = j;
  }
}

// -----------------------------------------------------------------------------
// Pearson: two-pass (centred) sums.
//
// The one-pass form sxx - m*mean^2 cancels catastrophically once the mean is
// large relative to the spread: for a column like 1e9 + rnorm(n) it returned
// NA (or garbage) instead of the correlation. Centring first is the textbook
// fix and matches stats::cor to rounding. The result is clamped to [-1, 1] as
// stats::cor does.
// -----------------------------------------------------------------------------
struct Moments {
  double mean = 0.0;
  double ss = 0.0;   // sum of squared deviations from the mean
};

Moments centred_moments(const std::vector<double>& v) {
  Moments m;
  const size_t n = v.size();
  if (n == 0) return m;
  const double dn = static_cast<double>(n);
  double mu = 0.0;
  for (double z : v) mu += z;
  mu /= dn;
  // One correction step for the rounding error of the mean (as stats::cor).
  double c = 0.0;
  for (double z : v) c += z - mu;
  if (std::isfinite(c)) mu += c / dn;
  double ss = 0.0;
  for (double z : v) { const double d = z - mu; ss += d * d; }
  m.mean = mu;
  m.ss = ss;
  return m;
}

double pearson_from(const std::vector<double>& x, const Moments& mx,
                    const std::vector<double>& y, const Moments& my) {
  const size_t n = x.size();
  if (n < static_cast<size_t>(MIN_PAIRS_COR)) return NA_REAL;
  const double sxx = mx.ss, syy = my.ss;
  if (!(sxx > 0.0) || !(syy > 0.0) || !std::isfinite(sxx) || !std::isfinite(syy)) return NA_REAL;
  double sxy = 0.0;
  for (size_t i = 0; i < n; ++i) sxy += (x[i] - mx.mean) * (y[i] - my.mean);
  if (!std::isfinite(sxy)) return NA_REAL;
  const double vv = sxx * syy;          // may overflow for huge values
  double r = (std::isfinite(vv) && vv > 0.0) ? sxy / std::sqrt(vv)
                                             : sxy / (std::sqrt(sxx) * std::sqrt(syy));
  if (r > 1.0) r = 1.0;
  if (r < -1.0) r = -1.0;
  return r;
}

double pearson_pairs(const std::vector<double>& x, const std::vector<double>& y) {
  if (x.size() < static_cast<size_t>(MIN_PAIRS_COR)) return NA_REAL;
  return pearson_from(x, centred_moments(x), y, centred_moments(y));
}

// Spearman: Pearson on the mid-ranks of the complete pairs (stats::cor with
// use = "pairwise.complete.obs" ranks the complete pairs only; the old code
// ranked every non-missing x, including those whose y was missing).
double spearman_pairs(const std::vector<double>& px, const std::vector<double>& py,
                      std::vector<double>& Rx, std::vector<double>& Ry,
                      std::vector<std::pair<double, int>>& vp) {
  if (px.size() < static_cast<size_t>(MIN_PAIRS_COR)) return NA_REAL;
  midranks(px, Rx, vp);
  midranks(py, Ry, vp);
  return pearson_pairs(Rx, Ry);
}

// -----------------------------------------------------------------------------
// Kendall tau-b in O(n log n) (Knight, 1966, JASA 61:436-439).
//
// Replaces the O(n^2) double loop over all pairs (5e9 comparisons for one pair
// of columns at n = 1e5). The counts of concordant/discordant/tied pairs are
// integers, and the final expression is the one used before, so the value is
// bit-identical whenever the tie structure is the same.
// -----------------------------------------------------------------------------
long long merge_count_inversions(std::vector<double>& a, std::vector<double>& buf) {
  const size_t n = a.size();
  long long inv = 0;
  for (size_t width = 1; width < n; width *= 2) {
    for (size_t lo = 0; lo < n; lo += 2 * width) {
      const size_t mid = std::min(lo + width, n);
      const size_t hi = std::min(lo + 2 * width, n);
      size_t i = lo, j = mid, k = lo;
      while (i < mid && j < hi) {
        if (a[i] <= a[j]) buf[k++] = a[i++];
        else { inv += static_cast<long long>(mid - i); buf[k++] = a[j++]; }
      }
      while (i < mid) buf[k++] = a[i++];
      while (j < hi) buf[k++] = a[j++];
    }
    a.swap(buf);
  }
  return inv;
}

double kendall_tau_b_pairs(const std::vector<double>& px, const std::vector<double>& py, std::vector<int>& idx) {
  const int n = static_cast<int>(px.size());
  if (n < MIN_PAIRS_COR) return NA_REAL;
  const std::vector<double>& x = px;
  const std::vector<double>& y = py;
  idx.resize(static_cast<size_t>(n));
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&x, &y](int a, int b) {
    const double xa = x[static_cast<size_t>(a)], xb = x[static_cast<size_t>(b)];
    if (xa < xb) return true;
    if (xb < xa) return false;
    return y[static_cast<size_t>(a)] < y[static_cast<size_t>(b)];
  });

  long long n1 = 0, n3 = 0;   // pairs tied in x; pairs tied in both
  std::vector<double> ys(static_cast<size_t>(n)), buf(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) ys[static_cast<size_t>(i)] = y[static_cast<size_t>(idx[static_cast<size_t>(i)])];
  {
    int i = 0;
    while (i < n) {
      int j = i;
      const double xv = x[static_cast<size_t>(idx[static_cast<size_t>(i)])];
      while (j < n && x[static_cast<size_t>(idx[static_cast<size_t>(j)])] == xv) ++j;
      const long long t = j - i;
      n1 += t * (t - 1) / 2;
      int k = i;
      while (k < j) {
        int l = k;
        while (l < j && ys[static_cast<size_t>(l)] == ys[static_cast<size_t>(k)]) ++l;
        const long long u = l - k;
        n3 += u * (u - 1) / 2;
        k = l;
      }
      i = j;
    }
  }
  // Pairs ordered by x (ties by y) that are out of order in y: discordant.
  const long long disc = merge_count_inversions(ys, buf);
  long long n2 = 0;             // pairs tied in y (ys is now sorted)
  {
    int i = 0;
    while (i < n) {
      int j = i;
      while (j < n && ys[static_cast<size_t>(j)] == ys[static_cast<size_t>(i)]) ++j;
      const long long t = j - i;
      n2 += t * (t - 1) / 2;
      i = j;
    }
  }
  const long long n0 = 1LL * n * (n - 1) / 2;
  const long long denom_x = n0 - n1;
  const long long denom_y = n0 - n2;
  if (denom_x == 0 || denom_y == 0) return NA_REAL;
  const double num = static_cast<double>(n0 - n1 - n2 + n3 - 2 * disc); // conc - disc
  return num / std::sqrt(static_cast<double>(denom_x) * static_cast<double>(denom_y));
}

// Fenwick tree over 1..K.
template <typename T>
struct Fenwick {
  std::vector<T> t;
  explicit Fenwick(size_t k) : t(k + 1, T(0)) {}
  void add(size_t i, T v) { for (; i < t.size(); i += i & (~i + 1)) t[i] += v; }
  T prefix(size_t i) const { T s(0); for (; i > 0; i -= i & (~i + 1)) s += t[i]; return s; }
};

// Dense 1-based ranks of a NaN-free vector (equal values share a rank).
size_t dense_ranks(const std::vector<double>& v, std::vector<size_t>& out,
                   std::vector<int>& idx) {
  const size_t n = v.size();
  out.assign(n, 0);
  idx.resize(n);
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&v](int a, int b) {
    return v[static_cast<size_t>(a)] < v[static_cast<size_t>(b)];
  });
  size_t r = 0;
  for (size_t i = 0; i < n; ++i) {
    if (i == 0 || v[static_cast<size_t>(idx[i])] != v[static_cast<size_t>(idx[i - 1])]) ++r;
    out[static_cast<size_t>(idx[i])] = r;
  }
  return r;
}

// -----------------------------------------------------------------------------
// Hoeffding's D (SAS PROC CORR / Hmisc formula; scale 30x; n >= 5).
//
// The bivariate ranks
//   Q_i = 1 + #{x_j < x_i, y_j < y_i} + 1/2 #{x_j = x_i, y_j < y_i}
//           + 1/2 #{x_j < x_i, y_j = y_i} + 1/4 #{j != i: x_j = x_i, y_j = y_i}
// used to be counted with an O(n^2) double loop; a sweep over x with a Fenwick
// tree on the y ranks gives the same (exact, quarter-integer) values in
// O(n log n).
// -----------------------------------------------------------------------------
void bivariate_Q(const std::vector<double>& px, const std::vector<double>& py, std::vector<double>& Q, std::vector<int>& idx) {
  const size_t n = px.size();
  Q.assign(n, 1.0);
  std::vector<size_t> ry;
  const size_t K = dense_ranks(py, ry, idx);
  idx.resize(n);
  std::iota(idx.begin(), idx.end(), 0);
  const std::vector<double>& x = px;
  std::sort(idx.begin(), idx.end(), [&x, &ry](int a, int b) {
    const double xa = x[static_cast<size_t>(a)], xb = x[static_cast<size_t>(b)];
    if (xa < xb) return true;
    if (xb < xa) return false;
    return ry[static_cast<size_t>(a)] < ry[static_cast<size_t>(b)];
  });
  Fenwick<long long> bit(K);
  size_t i = 0;
  while (i < n) {
    size_t j = i;
    const double xv = x[static_cast<size_t>(idx[i])];
    while (j < n && x[static_cast<size_t>(idx[j])] == xv) ++j;
    // Group [i, j) shares x and is sorted by y rank.
    size_t k = i;
    while (k < j) {
      size_t l = k;
      const size_t r = ry[static_cast<size_t>(idx[k])];
      while (l < j && ry[static_cast<size_t>(idx[l])] == r) ++l;
      const long long below = bit.prefix(r - 1);            // x_j < x_i, y_j < y_i
      const long long eq_y  = bit.prefix(r) - below;        // x_j < x_i, y_j = y_i
      const long long same_x_lower_y = static_cast<long long>(k - i);
      const long long same_both = static_cast<long long>(l - k - 1);
      const double q = 1.0 + static_cast<double>(below)
                     + 0.5 * static_cast<double>(same_x_lower_y)
                     + 0.5 * static_cast<double>(eq_y)
                     + 0.25 * static_cast<double>(same_both);
      for (size_t m = k; m < l; ++m) Q[static_cast<size_t>(idx[m])] = q;
      k = l;
    }
    for (size_t m = i; m < j; ++m) bit.add(ry[static_cast<size_t>(idx[m])], 1LL);
    i = j;
  }
}

double hoeffding_D_pairs(const std::vector<double>& px, const std::vector<double>& py,
                         const std::vector<double>* Rc, const std::vector<double>* Sc,
                         std::vector<double>& Rb, std::vector<double>& Sb,
                         std::vector<double>& Q, std::vector<int>& idx,
                         std::vector<std::pair<double, int>>& vp) {
  const int n = static_cast<int>(px.size());
  if (n < MIN_HOEFFDING_N) return NA_REAL;

  // Mid-ranks: cached per column when the column has no missing value.
  if (Rc == nullptr) { midranks(px, Rb, vp); Rc = &Rb; }
  if (Sc == nullptr) { midranks(py, Sb, vp); Sc = &Sb; }
  const std::vector<double>& R = *Rc;
  const std::vector<double>& S = *Sc;
  bivariate_Q(px, py, Q, idx); // includes tie weights

  long double D1 = 0.0L, D2 = 0.0L, D3 = 0.0L;
  for (int i = 0; i < n; ++i) {
    const long double Ri = R[static_cast<size_t>(i)], Si = S[static_cast<size_t>(i)],
                      Qi = Q[static_cast<size_t>(i)];
    D1 += (Qi - 1.0L) * (Qi - 2.0L);
    D2 += (Ri - 1.0L) * (Ri - 2.0L) * (Si - 1.0L) * (Si - 2.0L);
    D3 += (Ri - 2.0L) * (Si - 2.0L) * (Qi - 1.0L);
  }

  const long double num = (static_cast<long double>(n - 2) * (n - 3)) * D1 + D2 -
                          2.0L * (n - 2) * D3;
  const long double den = static_cast<long double>(n) * (n - 1) * (n - 2) * (n - 3) * (n - 4);
  if (!(den > 0.0L)) return NA_REAL;

  const long double D = static_cast<long double>(HOEFFDING_SCALE) * (num / den);
  return static_cast<double>(D);
}

// -----------------------------------------------------------------------------
// Distance correlation (Szekely, Rizzo & Bakirov 2007), V-statistic.
//
// Two defects are fixed here:
//  * the value returned was sqrt(dCov^2 / (dVar^2_X * dVar^2_Y)) instead of
//    sqrt(dCov^2 / sqrt(dVar^2_X * dVar^2_Y)): it was not scale-invariant and
//    not even bounded by 1 (a column against itself gave 1.69 in one example,
//    rescaling one column by 10 moved the value from 1 to 0.53);
//  * it cost O(n^2) per pair of columns. With
//        dCov^2 = S_ab/n^2 - 2 sum_i a_i. b_i. / n^3 + a.. b.. / n^4
//    (S_ab = sum_ij |x_i - x_j| |y_i - y_j|, a_i. = sum_j |x_i - x_j|), the row
//    sums come from one sort and prefix sums, and S_ab from a sweep over x with
//    a Fenwick tree on the y ranks (Huo & Szekely 2016, Technometrics 58:435),
//    i.e. O(n log n).
// Each column is centred and scaled first (dCor is invariant to both) to keep
// the expanded sums well conditioned.
// -----------------------------------------------------------------------------
void standardise(const std::vector<double>& v, std::vector<double>& out, bool& ok) {
  const size_t n = v.size();
  long double m = 0.0L;
  for (double z : v) m += z;
  m /= static_cast<long double>(n);
  long double ss = 0.0L;
  for (double z : v) { const long double d = z - m; ss += d * d; }
  const long double sd = std::sqrt(ss / static_cast<long double>(n));
  out.resize(n);
  ok = std::isfinite(static_cast<double>(sd)) && sd > 0.0L;
  if (!ok) return;
  for (size_t i = 0; i < n; ++i) out[i] = static_cast<double>((v[i] - m) / sd);
}

// a_i. = sum_j |v_i - v_j| for every i, plus sum_ij (v_i - v_j)^2.
void abs_row_sums(const std::vector<double>& v, std::vector<long double>& rows,
                  long double& sum_sq, std::vector<int>& idx) {
  const size_t n = v.size();
  idx.resize(n);
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&v](int a, int b) {
    return v[static_cast<size_t>(a)] < v[static_cast<size_t>(b)];
  });
  long double total = 0.0L, total_sq = 0.0L;
  for (double z : v) { total += z; total_sq += static_cast<long double>(z) * z; }
  rows.assign(n, 0.0L);
  long double before = 0.0L;
  for (size_t k = 0; k < n; ++k) {
    const long double z = v[static_cast<size_t>(idx[k])];
    const long double after = total - before - z;
    const long double nb = static_cast<long double>(k);
    const long double na = static_cast<long double>(n - k - 1);
    rows[static_cast<size_t>(idx[k])] = (z * nb - before) + (after - z * na);
    before += z;
  }
  const long double dn = static_cast<long double>(n);
  sum_sq = 2.0L * dn * total_sq - 2.0L * total * total;
}

double distance_correlation_pairs(const std::vector<double>& px, const std::vector<double>& py, std::vector<int>& idx) {
  const int n = static_cast<int>(px.size());
  if (n < MIN_PAIRS_COR) return NA_REAL;
  std::vector<double> u, v;
  bool okx = false, oky = false;
  standardise(px, u, okx);
  standardise(py, v, oky);
  if (!okx || !oky) {
    // A constant column has zero distance variance (as before, dCor = 0);
    // a non-finite one gives no usable distances.
    const bool finite_x = std::all_of(px.begin(), px.end(), [](double z) { return std::isfinite(z); });
    const bool finite_y = std::all_of(py.begin(), py.end(), [](double z) { return std::isfinite(z); });
    return (finite_x && finite_y) ? 0.0 : NA_REAL;
  }
  const size_t N = static_cast<size_t>(n);

  std::vector<long double> ra, rb;
  long double saa = 0.0L, sbb = 0.0L;
  abs_row_sums(u, ra, saa, idx);
  abs_row_sums(v, rb, sbb, idx);

  // S_ab by sweeping in x order; for the points already seen, a Fenwick tree
  // keyed by y rank holds count, sum x, sum y and sum x*y.
  std::vector<size_t> ry;
  const size_t K = dense_ranks(v, ry, idx);
  idx.resize(N);
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&u](int a, int b) {
    return u[static_cast<size_t>(a)] < u[static_cast<size_t>(b)];
  });
  Fenwick<long double> fc(K), fx(K), fy(K), fxy(K);
  long double tc = 0.0L, tx = 0.0L, ty = 0.0L, txy = 0.0L;
  long double sab = 0.0L;
  for (size_t k = 0; k < N; ++k) {
    const size_t j = static_cast<size_t>(idx[k]);
    const long double xj = u[j], yj = v[j];
    const size_t r = ry[j];
    // points with y <= y_j: (x_j - x_i)(y_j - y_i)
    const long double c1 = fc.prefix(r), sx1 = fx.prefix(r), sy1 = fy.prefix(r), sxy1 = fxy.prefix(r);
    const long double le = xj * yj * c1 - xj * sy1 - yj * sx1 + sxy1;
    // points with y > y_j: (x_j - x_i)(y_i - y_j)
    const long double c2 = tc - c1, sx2 = tx - sx1, sy2 = ty - sy1, sxy2 = txy - sxy1;
    const long double gt = xj * sy2 - xj * yj * c2 - sxy2 + yj * sx2;
    sab += le + gt;
    fc.add(r, 1.0L); fx.add(r, xj); fy.add(r, yj); fxy.add(r, xj * yj);
    tc += 1.0L; tx += xj; ty += yj; txy += xj * yj;
  }
  sab *= 2.0L;

  const long double dn = static_cast<long double>(n);
  long double a_dd = 0.0L, b_dd = 0.0L, ab_rows = 0.0L, aa_rows = 0.0L, bb_rows = 0.0L;
  for (size_t i = 0; i < N; ++i) {
    a_dd += ra[i]; b_dd += rb[i];
    ab_rows += ra[i] * rb[i];
    aa_rows += ra[i] * ra[i];
    bb_rows += rb[i] * rb[i];
  }
  const long double n2 = dn * dn, n3 = n2 * dn, n4 = n2 * n2;
  const long double dcov2 = sab / n2 - 2.0L * ab_rows / n3 + a_dd * b_dd / n4;
  const long double dvarx = saa / n2 - 2.0L * aa_rows / n3 + a_dd * a_dd / n4;
  const long double dvary = sbb / n2 - 2.0L * bb_rows / n3 + b_dd * b_dd / n4;
  if (!(dvarx > 0.0L) || !(dvary > 0.0L)) return 0.0;

  long double dcor2 = dcov2 / std::sqrt(dvarx * dvary);
  if (dcor2 < 0.0L) dcor2 = 0.0L; // numerical protection
  if (dcor2 > 1.0L) dcor2 = 1.0L;
  return std::sqrt(static_cast<double>(dcor2));
}

// -----------------------------------------------------------------------------
// Biweight midcorrelation (WGCNA-style) with scaled MAD and fallbacks.
// A zero MAD triggers the Pearson fallback; the old absolute threshold
// (MAD <= 1e-10) also triggered it for any column measured on a small scale.
// -----------------------------------------------------------------------------
double bicor_pairs(const std::vector<double>& px, const std::vector<double>& py) {
  const int n = static_cast<int>(px.size());
  if (n < MIN_PAIRS_COR) return NA_REAL;

  if (is_binary_or_two_values(px) || is_binary_or_two_values(py))
    return pearson_pairs(px, py);

  if (n < MIN_BIWEIGHT_N)
    return pearson_pairs(px, py);

  const double medx = median_copy(px);
  const double medy = median_copy(py);
  const double madx = mad_copy_scaled(px, medx);
  const double mady = mad_copy_scaled(py, medy);
  if (!(madx > 0.0) || !(mady > 0.0) || !std::isfinite(madx) || !std::isfinite(mady))
    return pearson_pairs(px, py);

  const double c = BIWEIGHT_C;
  long double num = 0.0L, dx2w = 0.0L, dy2w = 0.0L;
  int eff = 0;

  for (int i = 0; i < n; ++i) {
    const double xi = px[static_cast<size_t>(i)], yi = py[static_cast<size_t>(i)];
    const double ux = (xi - medx) / (c * madx);
    const double uy = (yi - medy) / (c * mady);
    if (std::abs(ux) < 1.0 && std::abs(uy) < 1.0) {
      const double wx = (1.0 - ux * ux); const double wy = (1.0 - uy * uy);
      const double wx2 = wx * wx, wy2 = wy * wy;
      const double dx = xi - medx, dy = yi - medy;
      num  += static_cast<long double>(wx2 * wy2) * static_cast<long double>(dx) * static_cast<long double>(dy);
      dx2w += static_cast<long double>(wx2 * wx2) * static_cast<long double>(dx) * static_cast<long double>(dx);
      dy2w += static_cast<long double>(wy2 * wy2) * static_cast<long double>(dy) * static_cast<long double>(dy);
      ++eff;
    }
  }
  if (eff < MIN_PAIRS_COR) return pearson_pairs(px, py);
  if (!(dx2w > 0.0L) || !(dy2w > 0.0L)) return pearson_pairs(px, py);
  return static_cast<double>(num / std::sqrt(dx2w * dy2w));
}

// -----------------------------------------------------------------------------
// Percentage Bend correlation (Wilcox 1994) - canonical implementation.
// -----------------------------------------------------------------------------
inline double psi_clip(double z) {
  if (z < -1.0) return -1.0;
  if (z >  1.0) return  1.0;
  return z;
}

double pbend_pairs(const std::vector<double>& px, const std::vector<double>& py, double beta = PBEND_BETA_DEFAULT) {
  const int n = static_cast<int>(px.size());
  if (n < MIN_PAIRS_COR) return NA_REAL;
  if (!(beta >= 0.0) || !(beta <= 0.5)) beta = PBEND_BETA_DEFAULT;
  const size_t N = static_cast<size_t>(n);

  // 1-2: median and absolute deviations
  const double mx = median_copy(px), my = median_copy(py);
  std::vector<double> Wx, Wy; Wx.reserve(N); Wy.reserve(N);
  for (size_t i = 0; i < N; ++i) {
    Wx.push_back(std::abs(px[i] - mx));
    Wy.push_back(std::abs(py[i] - my));
  }
  std::sort(Wx.begin(), Wx.end());
  std::sort(Wy.begin(), Wy.end());

  // 3-4: m = floor((1-beta)*n + 0.5), W_hat = W_(m)
  const int m = static_cast<int>(std::floor((1.0 - beta) * n + 0.5));
  if (m <= 0 || m > n) return NA_REAL;
  const double What_x = Wx[static_cast<size_t>(std::min(m - 1, n - 1))];
  const double What_y = Wy[static_cast<size_t>(std::min(m - 1, n - 1))];
  // Undefined only when W_hat is exactly zero (the old absolute 1e-10
  // threshold rejected every small-scale column).
  if (!(What_x > 0.0) || !(What_y > 0.0) || !std::isfinite(What_x) || !std::isfinite(What_y))
    return NA_REAL;

  // 5: count i1 (# z < -1) and i2 (# z > +1), sum the middle order statistics
  std::vector<double> Xs = px, Ys = py;
  std::sort(Xs.begin(), Xs.end());
  std::sort(Ys.begin(), Ys.end());

  int i1x = 0, i2x = 0, i1y = 0, i2y = 0;
  for (size_t i = 0; i < N; ++i) {
    const double zx = (px[i] - mx) / What_x;
    const double zy = (py[i] - my) / What_y;
    if (zx < -1.0) ++i1x; else if (zx > 1.0) ++i2x;
    if (zy < -1.0) ++i1y; else if (zy > 1.0) ++i2y;
  }
  if (i1x + i2x >= n || i1y + i2y >= n) return NA_REAL;

  double Sx = 0.0, Sy = 0.0;
  for (int i = i1x; i <= n - 1 - i2x; ++i) Sx += Xs[static_cast<size_t>(i)];
  for (int i = i1y; i <= n - 1 - i2y; ++i) Sy += Ys[static_cast<size_t>(i)];

  const int nx_mid = n - i1x - i2x;
  const int ny_mid = n - i1y - i2y;
  if (nx_mid <= 0 || ny_mid <= 0) return NA_REAL;

  const double phix = (What_x * (i2x - i1x) + Sx) / static_cast<double>(nx_mid);
  const double phiy = (What_y * (i2y - i1y) + Sy) / static_cast<double>(ny_mid);

  // 6-8: U = (x - phi_hat)/W_hat, A = psi(U); idem B
  long double num = 0.0L, nax = 0.0L, nay = 0.0L;
  for (size_t i = 0; i < N; ++i) {
    const double Ai = psi_clip((px[i] - phix) / What_x);
    const double Bi = psi_clip((py[i] - phiy) / What_y);
    num += static_cast<long double>(Ai) * static_cast<long double>(Bi);
    nax += static_cast<long double>(Ai) * static_cast<long double>(Ai);
    nay += static_cast<long double>(Bi) * static_cast<long double>(Bi);
  }
  if (!(nax > 0.0L) || !(nay > 0.0L)) return NA_REAL;
  return static_cast<double>(num / std::sqrt(nax * nay));
}

// -----------------------------------------------------------------------------
// Data extraction (structure of arrays).
// -----------------------------------------------------------------------------
bool is_numeric_compatible(SEXP col) {
  // A factor is an INTSXP whose codes carry no numeric meaning; the documented
  // contract is that non-numeric columns are excluded.
  if (Rf_isFactor(col)) return false;
  return TYPEOF(col) == INTSXP || TYPEOF(col) == REALSXP || TYPEOF(col) == LGLSXP;
}

struct SoA {
  std::vector<std::vector<double>> data;
  std::vector<std::string> names;
  int p = 0, n = 0;
};

SoA extract_soa(const DataFrame& df) {
  SEXP nm = Rf_getAttrib(df, R_NamesSymbol);
  std::vector<int> idx;
  const int ncol = static_cast<int>(df.size());
  for (int i = 0; i < ncol; ++i) if (is_numeric_compatible(VECTOR_ELT(df, i))) idx.push_back(i);
  if (static_cast<int>(idx.size()) < 2) stop("At least two numeric variables are needed");

  const int p = static_cast<int>(idx.size());
  const int n = static_cast<int>(df.nrows());
  SoA S; S.p = p; S.n = n; S.data.resize(static_cast<size_t>(p)); S.names.resize(static_cast<size_t>(p));
  for (int k = 0; k < p; ++k) {
    const int j = idx[static_cast<size_t>(k)];
    const size_t K = static_cast<size_t>(k);
    S.names[K] = (Rf_isNull(nm) || STRING_ELT(nm, j) == NA_STRING)
      ? std::string("NA") : std::string(CHAR(STRING_ELT(nm, j)));
    std::vector<double>& out = S.data[K];
    out.resize(static_cast<size_t>(n));
    SEXP col = VECTOR_ELT(df, j);
    if (Rf_xlength(col) != static_cast<R_xlen_t>(n)) stop("Malformed data frame: column lengths differ.");
    // Plain pointer reads: no Rcpp proxy (and hence no R API call) per element.
    switch (TYPEOF(col)) {
    case INTSXP: {
      const int* iv = INTEGER(col);
      for (int i = 0; i < n; ++i)
        out[static_cast<size_t>(i)] = (iv[i] == NA_INTEGER) ? NA_REAL : static_cast<double>(iv[i]);
      break;
    }
    case REALSXP: {
      const double* nv = REAL(col);
      std::copy(nv, nv + n, out.begin());
      break;
    }
    case LGLSXP: {
      const int* lv = LOGICAL(col);
      for (int i = 0; i < n; ++i)
        out[static_cast<size_t>(i)] = (lv[i] == NA_LOGICAL) ? NA_REAL : static_cast<double>(lv[i]);
      break;
    }
    default:
      break;
    }
  }
  return S;
}

} // namespace


// [[Rcpp::export]]
DataFrame obcorr(DataFrame df, std::string method = "all", int threads = 0) {

  const bool do_pearson   = (method == "all" || method == "pearson");
  const bool do_spearman  = (method == "all" || method == "spearman");
  const bool do_kendall   = (method == "all" || method == "kendall");
  const bool do_hoeffding = (method == "all" || method == "hoeffding" || method == "alternative");
  const bool do_distance  = (method == "all" || method == "distance"  || method == "alternative");
  const bool do_biweight  = (method == "all" || method == "biweight"  || method == "robust");
  const bool do_pbend     = (method == "all" || method == "pbend"     || method == "robust");
  // An unknown method used to fall through every branch and build a data
  // frame out of zero-length columns.
  if (!(do_pearson || do_spearman || do_kendall || do_hoeffding || do_distance ||
        do_biweight || do_pbend)) {
    stop("method must be one of: 'all', 'pearson', 'spearman', 'kendall', "
         "'hoeffding', 'distance', 'biweight', 'pbend', 'robust', 'alternative'.");
  }

#ifdef _OPENMP
  // CRAN Repository Policy: a package must never use more than two cores
  // simultaneously by default, and the user must remain in control. An
  // explicit positive `threads` wins; otherwise use at most 2, and never more
  // than the environment already allows (omp_get_max_threads() reflects
  // OMP_NUM_THREADS). The count is passed to this parallel region only
  // (num_threads clause): omp_set_num_threads() used to change the thread
  // count of the whole R session as a side effect.
  int n_threads;
  if (threads > 0) {
    n_threads = threads;
  } else {
    n_threads = std::min(2, omp_get_max_threads());
  }
  n_threads = std::max(1, std::min(n_threads, omp_get_num_procs()));
  const int max_threads = n_threads;
#else
  const int max_threads = 1;
  if (threads > 1) Rcpp::warning("OpenMP not available; running on 1 thread.");
#endif

  if (df.nrows() == 0) stop("Empty data frame provided");

  SoA S = extract_soa(df);
  const int p = S.p;
  const int total_pairs = p * (p - 1) / 2;
  const size_t TP = static_cast<size_t>(total_pairs);

  std::vector<std::string> vx, vy;
  std::vector<double> v_pear, v_spear, v_kend, v_hoef, v_dcor, v_bic, v_pb;

  // Sized (not merely reserved) so each iteration writes to its own slot:
  // the row order never depends on thread scheduling.
  vx.resize(TP); vy.resize(TP);
  if (do_pearson)   v_pear.resize(TP);
  if (do_spearman)  v_spear.resize(TP);
  if (do_kendall)   v_kend.resize(TP);
  if (do_hoeffding) v_hoef.resize(TP);
  if (do_distance)  v_dcor.resize(TP);
  if (do_biweight)  v_bic.resize(TP);
  if (do_pbend)     v_pb.resize(TP);

  const int chunk = std::max(1, total_pairs / (max_threads * 4));
  (void)chunk;

  // Per-column work shared by every pair the column takes part in, done once
  // per column instead of once per pair: for a column without missing values
  // its centred moments (Pearson) and mid-ranks with their moments
  // (Spearman, Hoeffding). Columns with missing values fall back to
  // pairwise-complete computations.
  struct ColCache {
    bool has_na = false;
    Moments m;
    std::vector<double> ranks;
    Moments rm;
  };
  std::vector<ColCache> C(static_cast<size_t>(p));
  const bool need_ranks = do_spearman || do_hoeffding;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(n_threads)
#endif
  for (int c = 0; c < p; ++c) {
    const std::vector<double>& v = S.data[static_cast<size_t>(c)];
    ColCache& cc = C[static_cast<size_t>(c)];
    cc.has_na = std::any_of(v.begin(), v.end(), [](double z) { return std::isnan(z); });
    if (cc.has_na) continue;
    if (do_pearson) cc.m = centred_moments(v);
    if (need_ranks) {
      std::vector<std::pair<double, int>> vp;
      midranks(v, cc.ranks, vp);
      cc.rm = centred_moments(cc.ranks);
    }
  }

#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
#endif
{
  // Scratch buffers reused across iterations by this thread only. Nothing in
  // the loop touches the R API.
  ValidPairs P;
  std::vector<double> R1, R2, Q;
  std::vector<int> idx;
  std::vector<std::pair<double, int>> vp;

#ifdef _OPENMP
#pragma omp for schedule(dynamic, chunk)
#endif
  for (int k = 0; k < total_pairs; ++k) {
    // map linear -> (i,j)
    int i = 0, j = 1, r = k;
    while (r >= p - 1 - i) { r -= (p - 1 - i); ++i; j = i + 1; }
    j += r;
    const size_t K = static_cast<size_t>(k);
    const ColCache& ci = C[static_cast<size_t>(i)];
    const ColCache& cj = C[static_cast<size_t>(j)];

    vx[K] = S.names[static_cast<size_t>(i)];
    vy[K] = S.names[static_cast<size_t>(j)];

    // Complete pairs: the columns themselves when neither has a missing value.
    const bool clean = !ci.has_na && !cj.has_na;
    if (!clean) complete_pairs(S.data[static_cast<size_t>(i)], S.data[static_cast<size_t>(j)], P);
    const std::vector<double>& X = clean ? S.data[static_cast<size_t>(i)] : P.x;
    const std::vector<double>& Y = clean ? S.data[static_cast<size_t>(j)] : P.y;

    if (do_pearson)
      v_pear[K] = clean ? pearson_from(X, ci.m, Y, cj.m) : pearson_pairs(X, Y);
    if (do_spearman)
      v_spear[K] = clean ? pearson_from(ci.ranks, ci.rm, cj.ranks, cj.rm)
                         : spearman_pairs(X, Y, R1, R2, vp);
    if (do_kendall)   v_kend[K]  = kendall_tau_b_pairs(X, Y, idx);
    if (do_hoeffding)
      v_hoef[K] = hoeffding_D_pairs(X, Y, clean ? &ci.ranks : nullptr,
                                    clean ? &cj.ranks : nullptr, R1, R2, Q, idx, vp);
    if (do_distance)  v_dcor[K]  = distance_correlation_pairs(X, Y, idx);
    if (do_biweight)  v_bic[K]   = bicor_pairs(X, Y);
    if (do_pbend)     v_pb[K]    = pbend_pairs(X, Y);
  }
}
  // construct DataFrame
  CharacterVector X(vx.begin(), vx.end()), Y(vy.begin(), vy.end());
  if (method == "pearson")
    return DataFrame::create(_["x"] = X, _["y"] = Y, _["pearson"] = NumericVector(v_pear.begin(), v_pear.end()));
  if (method == "spearman")
    return DataFrame::create(_["x"] = X, _["y"] = Y, _["spearman"] = NumericVector(v_spear.begin(), v_spear.end()));
  if (method == "kendall")
    return DataFrame::create(_["x"] = X, _["y"] = Y, _["kendall"] = NumericVector(v_kend.begin(), v_kend.end()));
  if (method == "hoeffding")
    return DataFrame::create(_["x"] = X, _["y"] = Y, _["hoeffding"] = NumericVector(v_hoef.begin(), v_hoef.end()));
  if (method == "distance")
    return DataFrame::create(_["x"] = X, _["y"] = Y, _["distance"] = NumericVector(v_dcor.begin(), v_dcor.end()));
  if (method == "biweight")
    return DataFrame::create(_["x"] = X, _["y"] = Y, _["biweight"] = NumericVector(v_bic.begin(), v_bic.end()));
  if (method == "pbend")
    return DataFrame::create(_["x"] = X, _["y"] = Y, _["pbend"] = NumericVector(v_pb.begin(), v_pb.end()));
  if (method == "robust")
    return DataFrame::create(_["x"] = X, _["y"] = Y,
                             _["biweight"] = NumericVector(v_bic.begin(), v_bic.end()),
                             _["pbend"]    = NumericVector(v_pb.begin(),  v_pb.end()));
  if (method == "alternative")
    return DataFrame::create(_["x"] = X, _["y"] = Y,
                             _["hoeffding"] = NumericVector(v_hoef.begin(), v_hoef.end()),
                             _["distance"]  = NumericVector(v_dcor.begin(), v_dcor.end()));
  // all
  return DataFrame::create(_["x"] = X, _["y"] = Y,
                           _["pearson"]   = NumericVector(v_pear.begin(),  v_pear.end()),
                           _["spearman"]  = NumericVector(v_spear.begin(), v_spear.end()),
                           _["kendall"]   = NumericVector(v_kend.begin(),  v_kend.end()),
                           _["hoeffding"] = NumericVector(v_hoef.begin(),  v_hoef.end()),
                           _["distance"]  = NumericVector(v_dcor.begin(),  v_dcor.end()),
                           _["biweight"]  = NumericVector(v_bic.begin(),   v_bic.end()),
                           _["pbend"]     = NumericVector(v_pb.begin(),    v_pb.end()));
}
