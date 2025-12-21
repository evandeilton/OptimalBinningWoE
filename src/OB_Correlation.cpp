// [[Rcpp::plugins(openmp)]]
// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
#include <thread>
#include <unordered_set> // (needed if using set; in V6 we use sort for unique)
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace Rcpp;

// Include shared headers
#include "common/optimal_binning_common.h"
#include "common/bin_structures.h"

using namespace Rcpp;
using namespace OptimalBinning;


namespace StatisticalConstants {
// Local constant removed (uses shared definition)
constexpr double MAD_NORMAL_SCALE = 1.4826; // robust scaling
constexpr double HOEFFDING_SCALE = 30.0;    // Hmisc: 30x of the original D
constexpr double BIWEIGHT_C       = 9.0;    // WGCNA/Mosteller-Tukey
constexpr double PBEND_BETA_DEFAULT = 0.2;  // default Wilcox value
constexpr int    MIN_PAIRS_COR     = 3;     // for Pearson/Spearman/Kendall/bicor/pbend
constexpr int    MIN_HOEFFDING_N   = 5;     // Hoeffding requires n>=5
constexpr int    PARALLEL_THRESHOLD = 200;
constexpr int    MIN_BIWEIGHT_N    = 8;
}

// ---------------------------------------------------------------------
// Numerical utilities
// ---------------------------------------------------------------------
inline bool nearly_equal(double a, double b, double eps = OptimalBinning::EPSILON) {
  return std::abs(a - b) < eps;
}
double median_copy(std::vector<double> v) {
  if (v.empty()) return R_NaReal;
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  if (n % 2 == 0) return 0.5 * (v[n/2 - 1] + v[n/2]);
  return v[n/2];
}
double mad_copy_scaled(const std::vector<double>& v, double med) {
  if (v.empty() || !R_finite(med)) return R_NaReal;
  std::vector<double> dev;
  dev.reserve(v.size());
  for (double x : v) if (!std::isnan(x)) dev.push_back(std::abs(x - med));
  if (dev.empty()) return R_NaReal;
  double mad = median_copy(dev);
  if (!R_finite(mad)) return R_NaReal;
  return StatisticalConstants::MAD_NORMAL_SCALE * mad;
}
int count_unique_non_na(std::vector<double> v) {
  v.erase(std::remove_if(v.begin(), v.end(), [](double z){ return std::isnan(z); }), v.end());
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  int cnt = 1;
  for (size_t i = 1; i < v.size(); ++i) {
    if (!nearly_equal(v[i], v[i-1])) ++cnt;
  }
  return cnt;
}
struct ValidPairs {
  std::vector<double> x, y;
  inline void add(double xi, double yi){ x.push_back(xi); y.push_back(yi); }
  inline int size() const { return (int)x.size(); }
};
// ---------------------------------------------------------------------
// Ranking (midranks) and bivariate rank Q with tie weights
// ---------------------------------------------------------------------
class FastRanker {
  std::vector<std::pair<double,int>> idx_;
public:
  void ranks(const std::vector<double>& v, std::vector<double>& r) {
    const int n = (int)v.size();
    r.assign(n, R_NaReal);
    idx_.clear(); idx_.reserve(n);
    for (int i=0;i<n;++i) if (!std::isnan(v[i])) idx_.emplace_back(v[i], i);
    if (idx_.empty()) return;
    std::stable_sort(idx_.begin(), idx_.end(),
                     [](const auto& a, const auto& b){ return a.first < b.first; });
    size_t i=0;
    while (i<idx_.size()) {
      size_t j=i;
      const double val = idx_[i].first;
      while (j<idx_.size() && nearly_equal(idx_[j].first, val)) ++j;
      // midrank 1-based: average of [i+1, j]
      const double avg = ( (double)i + (double)j + 1.0 ) / 2.0;
      for (size_t k=i;k<j;++k) r[idx_[k].second] = avg;
      i=j;
    }
  }
  
  // Bivariate rank Q_i: 1 + (# points with x<xi & y<yi) + 1/2*(tie in one axis) + 1/4*(tie in both)
  void bivariate_Q(const ValidPairs& P, std::vector<double>& Q) {
    const int n = P.size();
    Q.assign(n, 1.0); // starts at 1
    for (int i=0;i<n;++i) {
      double q = 1.0;
      const double xi = P.x[i], yi = P.y[i];
      for (int j=0;j<n;++j) if (j!=i) {
        const double dx = P.x[j] - xi;
        const double dy = P.y[j] - yi;
        const bool ex = nearly_equal(dx, 0.0), ey = nearly_equal(dy, 0.0);
        if (!ex && !ey) {
          if (P.x[j] < xi && P.y[j] < yi) q += 1.0;
        } else if (ex && !ey) {
          if (P.y[j] < yi) q += 0.5;
        } else if (!ex && ey) {
          if (P.x[j] < xi) q += 0.5;
        } else { // ex && ey
          q += 0.25;
        }
      }
      Q[i] = q;
    }
  }
};
// ---------------------------------------------------------------------
// Classic correlations
// ---------------------------------------------------------------------
double pearson_fast(const std::vector<double>& x, const std::vector<double>& y) {
  const int n = (int)x.size();
  double sx=0, sy=0, sxx=0, syy=0, sxy=0; int m=0;
#ifdef _OPENMP
#pragma omp simd reduction(+:sx,sy,sxx,syy,sxy,m)
#endif
  for (int i=0;i<n;++i) {
    const double xi=x[i], yi=y[i];
    if (!std::isnan(xi) && !std::isnan(yi)) {
      sx += xi; sy += yi; sxx += xi*xi; syy += yi*yi; sxy += xi*yi; ++m;
    }
  }
  if (m < StatisticalConstants::MIN_PAIRS_COR) return R_NaReal;
  const double mx = sx/m, my = sy/m;
  const double vx = sxx - m*mx*mx, vy = syy - m*my*my;
  if (!(vx>0.0) || !(vy>0.0)) return R_NaReal;
  const double cov = sxy - m*mx*my;
  return cov / std::sqrt(vx*vy);
}
double spearman_fast(const std::vector<double>& x, const std::vector<double>& y,
                     FastRanker& rx, FastRanker& ry,
                     std::vector<double>& Rx, std::vector<double>& Ry) {
  rx.ranks(x, Rx); ry.ranks(y, Ry);
  return pearson_fast(Rx, Ry);
}
double kendall_tau_b_fast(const std::vector<double>& x, const std::vector<double>& y) {
  ValidPairs P; P.x.reserve(x.size()); P.y.reserve(y.size());
  for (size_t i=0;i<x.size();++i) if (!std::isnan(x[i]) && !std::isnan(y[i])) P.add(x[i],y[i]);
  const int n = P.size();
  if (n < StatisticalConstants::MIN_PAIRS_COR) return R_NaReal;
  
  long long conc=0, disc=0, tx_only=0, ty_only=0, txy=0;
  for (int i=0;i<n-1;++i) {
    const double xi=P.x[i], yi=P.y[i];
    for (int j=i+1;j<n;++j) {
      const double dx = xi - P.x[j], dy = yi - P.y[j];
      const bool ex = nearly_equal(dx,0.0), ey = nearly_equal(dy,0.0);
      if (ex && ey) ++txy;
      else if (ex) ++tx_only;
      else if (ey) ++ty_only;
      else if ((dx>0 && dy>0) || (dx<0 && dy<0)) ++conc;
      else ++disc;
    }
  }
  const long long n0 = 1LL * n * (n-1) / 2;
  const long long t1 = tx_only + txy;
  const long long t2 = ty_only + txy;
  const long long denom_x = n0 - t1;
  const long long denom_y = n0 - t2;
  if (denom_x==0 || denom_y==0) return R_NaReal;
  const double num = (double)(conc - disc);
  return num / std::sqrt((double)denom_x * (double)denom_y);
}
// ---------------------------------------------------------------------
// Hoeffding's D (SAS/PROC CORR/Hmisc formula; scale 30x; n>=5)
// ---------------------------------------------------------------------
double hoeffding_D_v6(const std::vector<double>& x, const std::vector<double>& y) {
  ValidPairs P; P.x.reserve(x.size()); P.y.reserve(y.size());
  for (size_t i=0;i<x.size();++i) if (!std::isnan(x[i]) && !std::isnan(y[i])) P.add(x[i],y[i]);
  const int n = P.size();
  if (n < StatisticalConstants::MIN_HOEFFDING_N) return R_NaReal;
  
  FastRanker rnk;
  std::vector<double> R, S, Q;
  rnk.ranks(P.x, R);
  rnk.ranks(P.y, S);
  rnk.bivariate_Q(P, Q); // includes tie weights
  
  long double D1=0.0L, D2=0.0L, D3=0.0L;
  for (int i=0;i<n;++i) {
    const long double Ri=R[i], Si=S[i], Qi=Q[i];
    D1 += (Qi-1.0L)*(Qi-2.0L);
    D2 += (Ri-1.0L)*(Ri-2.0L)*(Si-1.0L)*(Si-2.0L);
    D3 += (Ri-2.0L)*(Si-2.0L)*(Qi-1.0L);
  }
  
  const long double num = ( (long double)(n-2)*(n-3) )*D1 + D2 - 2.0L*(n-2)*D3;
  const long double den = (long double)n*(n-1)*(n-2)*(n-3)*(n-4);
  if (!(den>0.0L)) return R_NaReal;
  
  long double D = (long double)StatisticalConstants::HOEFFDING_SCALE * (num / den);
  // convert to double
  return (double)D;
}
// ---------------------------------------------------------------------
// Distance Correlation (2007) - without NxN matrices
// ---------------------------------------------------------------------
double distance_correlation_v6(const std::vector<double>& x, const std::vector<double>& y) {
  ValidPairs P; P.x.reserve(x.size()); P.y.reserve(y.size());
  for (size_t i=0;i<x.size();++i) if (!std::isnan(x[i]) && !std::isnan(y[i])) P.add(x[i],y[i]);
  const int n = P.size();
  if (n < StatisticalConstants::MIN_PAIRS_COR) return R_NaReal;
  
  std::vector<double> rowA(n,0.0), rowB(n,0.0);
  // 1st pass: row sums
  for (int i=0;i<n;++i) {
    const double xi=P.x[i], yi=P.y[i];
    double sa=0.0, sb=0.0;
    for (int j=0;j<n;++j) {
      sa += std::abs(xi - P.x[j]);
      sb += std::abs(yi - P.y[j]);
    }
    rowA[i]=sa/n; rowB[i]=sb/n;
  }
  double gA=0.0, gB=0.0;
  for (int i=0;i<n;++i) { gA += rowA[i]; gB += rowB[i]; }
  gA/=n; gB/=n;
  
  // 2nd pass: accumulate dCov^2 and dVar^2
  long double dCov2=0.0L, dVarX=0.0L, dVarY=0.0L;
  for (int i=0;i<n;++i) {
    const double xi=P.x[i], yi=P.y[i];
    for (int j=0;j<n;++j) {
      const double Aij = std::abs(xi - P.x[j]) - rowA[i] - rowA[j] + gA;
      const double Bij = std::abs(yi - P.y[j]) - rowB[i] - rowB[j] + gB;
      dCov2 += (long double)Aij * (long double)Bij;
      dVarX += (long double)Aij * (long double)Aij;
      dVarY += (long double)Bij * (long double)Bij;
    }
  }
  const long double nn = (long double)n * (long double)n;
  dCov2 /= nn; dVarX /= nn; dVarY /= nn;
  if (!(dVarX>0.0L) || !(dVarY>0.0L)) return 0.0;
  
  long double dcor2 = dCov2 / (dVarX * dVarY);
  if (dcor2 < 0.0L) dcor2 = 0.0L; // numerical protection
  return std::sqrt( (double)dcor2 );
}
// ---------------------------------------------------------------------
// Biweight midcorrelation (WGCNA-style) with scaled MAD and fallbacks
// ---------------------------------------------------------------------
bool is_binary_or_two_values(const std::vector<double>& v) {
  return count_unique_non_na(v) <= 2;
}
double bicor_v6(const std::vector<double>& x, const std::vector<double>& y) {
  ValidPairs P; P.x.reserve(x.size()); P.y.reserve(y.size());
  for (size_t i=0;i<x.size();++i) if (!std::isnan(x[i]) && !std::isnan(y[i])) P.add(x[i],y[i]);
  const int n = P.size();
  if (n < StatisticalConstants::MIN_PAIRS_COR) return R_NaReal;
  
  if (is_binary_or_two_values(P.x) || is_binary_or_two_values(P.y))
    return pearson_fast(P.x, P.y);
  
  if (n < StatisticalConstants::MIN_BIWEIGHT_N)
    return pearson_fast(P.x, P.y);
  
  const double medx = median_copy(P.x);
  const double medy = median_copy(P.y);
  const double madx = mad_copy_scaled(P.x, medx);
  const double mady = mad_copy_scaled(P.y, medy);
  if (!(madx>OptimalBinning::EPSILON) || !(mady>OptimalBinning::EPSILON))
    return pearson_fast(P.x, P.y);
  
  const double c = StatisticalConstants::BIWEIGHT_C;
  long double num=0.0L, dx2w=0.0L, dy2w=0.0L;
  int eff=0;
  
  for (int i=0;i<n;++i) {
    const double ux = (P.x[i]-medx)/(c*madx);
    const double uy = (P.y[i]-medy)/(c*mady);
    if (std::abs(ux) < 1.0 && std::abs(uy) < 1.0) {
      const double wx = (1.0 - ux*ux); const double wy = (1.0 - uy*uy);
      const double wx2 = wx*wx, wy2 = wy*wy;
      const double dx = P.x[i]-medx, dy = P.y[i]-medy;
      num  += (long double)(wx2*wy2) * (long double)dx * (long double)dy;
      dx2w += (long double)(wx2*wx2) * (long double)dx * (long double)dx;
      dy2w += (long double)(wy2*wy2) * (long double)dy * (long double)dy;
      ++eff;
    }
  }
  if (eff < StatisticalConstants::MIN_PAIRS_COR) return pearson_fast(P.x, P.y);
  if (!(dx2w>0.0L) || !(dy2w>0.0L)) return pearson_fast(P.x, P.y);
  return (double)( num / std::sqrt(dx2w*dy2w) );
}
// ---------------------------------------------------------------------
// Percentage Bend correlation (Wilcox/NIST) - canonical implementation
// ---------------------------------------------------------------------
static inline double psi_clip(double z) {
  if (z < -1.0) return -1.0;
  if (z >  1.0) return  1.0;
  return z;
}
double pbend_v6(const std::vector<double>& x, const std::vector<double>& y,
                double beta = StatisticalConstants::PBEND_BETA_DEFAULT) {
  ValidPairs P; P.x.reserve(x.size()); P.y.reserve(y.size());
  for (size_t i=0;i<x.size();++i) if (!std::isnan(x[i]) && !std::isnan(y[i])) P.add(x[i],y[i]);
  const int n = P.size();
  if (n < StatisticalConstants::MIN_PAIRS_COR) return R_NaReal;
  if (!(beta >= 0.0) || !(beta <= 0.5)) beta = StatisticalConstants::PBEND_BETA_DEFAULT;
  
  // 1-2: median and absolute deviations
  const double mx = median_copy(P.x), my = median_copy(P.y);
  std::vector<double> Wx, Wy; Wx.reserve(n); Wy.reserve(n);
  for (int i=0;i<n;++i){ Wx.push_back(std::abs(P.x[i]-mx)); Wy.push_back(std::abs(P.y[i]-my)); }
  std::sort(Wx.begin(), Wx.end());
  std::sort(Wy.begin(), Wy.end());
  
  // 3-4: m = floor((1-β)*n + 0.5), W_hat = W_(m)
  const int m = (int)std::floor((1.0 - beta) * n + 0.5);
  if (m <= 0 || m > n) return R_NaReal;
  const double What_x = Wx[std::min(m-1, n-1)];
  const double What_y = Wy[std::min(m-1, n-1)];
  if (!(What_x>OptimalBinning::EPSILON) || !(What_y>OptimalBinning::EPSILON))
    return R_NaReal;
  
  // 5: count i1 (# z < -1) and i2 (# z > +1), sum x order of the middle to estimate phi_hat
  std::vector<double> Xs = P.x, Ys = P.y;
  std::sort(Xs.begin(), Xs.end());
  std::sort(Ys.begin(), Ys.end());
  
  int i1x=0, i2x=0, i1y=0, i2y=0;
  for (int i=0;i<n;++i) {
    const double zx = (P.x[i]-mx)/What_x;
    const double zy = (P.y[i]-my)/What_y;
    if (zx < -1.0) ++i1x; else if (zx > 1.0) ++i2x;
    if (zy < -1.0) ++i1y; else if (zy > 1.0) ++i2y;
  }
  if (i1x + i2x >= n || i1y + i2y >= n) return R_NaReal;
  
  double Sx = 0.0, Sy = 0.0;
  for (int i=i1x; i<= n-1-i2x; ++i) Sx += Xs[i];
  for (int i=i1y; i<= n-1-i2y; ++i) Sy += Ys[i];
  
  const int nx_mid = n - i1x - i2x;
  const int ny_mid = n - i1y - i2y;
  if (nx_mid<=0 || ny_mid<=0) return R_NaReal;
  
  const double phix = (What_x * (i2x - i1x) + Sx) / (double)nx_mid;
  const double phiy = (What_y * (i2y - i1y) + Sy) / (double)ny_mid;
  
  // 6-8: U=(x - phi_hat)/W_hat, A=psi(U); idem B
  long double num=0.0L, nax=0.0L, nay=0.0L;
  for (int i=0;i<n;++i) {
    const double Ai = psi_clip( (P.x[i]-phix)/What_x );
    const double Bi = psi_clip( (P.y[i]-phiy)/What_y );
    num += (long double)Ai * (long double)Bi;
    nax += (long double)Ai * (long double)Ai;
    nay += (long double)Bi * (long double)Bi;
  }
  if (!(nax>0.0L) || !(nay>0.0L)) return R_NaReal;
  return (double)( num / std::sqrt(nax*nay) );
}
// ---------------------------------------------------------------------
// Data extraction (SoA) and main function
// ---------------------------------------------------------------------
bool is_numeric_compatible(SEXP col) {
  return TYPEOF(col)==INTSXP || TYPEOF(col)==REALSXP || TYPEOF(col)==LGLSXP;
}
struct SoA {
  std::vector<std::vector<double>> data;
  std::vector<std::string> names;
  int p=0, n=0;
};
SoA extract_soa(const DataFrame& df) {
  CharacterVector nm = df.names();
  std::vector<int> idx;
  for (int i=0;i<df.size();++i) if (is_numeric_compatible(df[i])) idx.push_back(i);
  if ((int)idx.size() < 2) stop("At least two numeric variables are needed");
  
  const int p = (int)idx.size();
  const int n = df.nrows();
  SoA S; S.p=p; S.n=n; S.data.resize(p); S.names.resize(p);
  for (int k=0;k<p;++k) {
    const int j = idx[k];
    S.names[k] = as<std::string>(nm[j]);
    S.data[k].resize(n);
    SEXP col = df[j];
    switch(TYPEOF(col)) {
    case INTSXP: {
      IntegerVector iv = as<IntegerVector>(col);
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (int i=0;i<n;++i)
        S.data[k][i] = (iv[i]==NA_INTEGER) ? R_NaReal : (double)iv[i];
      break;
    }
    case REALSXP: {
      NumericVector nv = as<NumericVector>(col);
      std::copy(nv.begin(), nv.end(), S.data[k].begin());
      break;
    }
    case LGLSXP: {
      LogicalVector lv = as<LogicalVector>(col);
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (int i=0;i<n;++i)
        S.data[k][i] = (lv[i]==NA_LOGICAL) ? R_NaReal : (double)lv[i];
      break;
    }
    }
  }
  return S;
}



// [[Rcpp::export]]
DataFrame obcorr(DataFrame df, std::string method="all", int threads=0) {
   
#ifdef _OPENMP
   if (threads>0) omp_set_num_threads(threads);
   else {
     int h = std::max(1, (int)std::thread::hardware_concurrency());
     omp_set_num_threads(h);
   }
   int max_threads = omp_get_max_threads();
#else
   int max_threads = 1;
   if (threads>1) Rcpp::warning("OpenMP not available; running on 1 thread.");
#endif
   
   if (df.nrows()==0) stop("Empty data frame provided");
   
   SoA S = extract_soa(df);
   const int p = S.p;
   const int total_pairs = p*(p-1)/2;
   
   const bool do_pearson    = (method=="all" || method=="pearson");
   const bool do_spearman   = (method=="all" || method=="spearman");
   const bool do_kendall    = (method=="all" || method=="kendall");
   const bool do_hoeffding = (method=="all" || method=="hoeffding" || method=="alternative");
   const bool do_distance   = (method=="all" || method=="distance"   || method=="alternative");
   const bool do_biweight   = (method=="all" || method=="biweight"   || method=="robust");
   const bool do_pbend      = (method=="all" || method=="pbend"      || method=="robust");
   
   std::vector<std::string> vx, vy;
   std::vector<double> v_pear, v_spear, v_kend, v_hoef, v_dcor, v_bic, v_pb;
   
   vx.reserve(total_pairs); vy.reserve(total_pairs);
   if (do_pearson)   v_pear.reserve(total_pairs);
   if (do_spearman) v_spear.reserve(total_pairs);
   if (do_kendall)   v_kend.reserve(total_pairs);
   if (do_hoeffding)v_hoef.reserve(total_pairs);
   if (do_distance) v_dcor.reserve(total_pairs);
   if (do_biweight) v_bic.reserve(total_pairs);
   if (do_pbend)     v_pb .reserve(total_pairs);
   
   const int chunk = std::max(1, total_pairs / (max_threads * 4));
   
#ifdef _OPENMP
#pragma omp parallel
#endif
{
  FastRanker rx, ry;
  std::vector<double> Rx, Ry;
  
  std::vector<std::string> lx, ly;
  std::vector<double> lpear, lspear, lkend, lhoef, ldcor, lbic, lpb;
  
#ifdef _OPENMP
#pragma omp for schedule(dynamic, chunk) nowait
#endif
  for (int idx=0; idx<total_pairs; ++idx) {
    // map linear -> (i,j)
    int i=0, j=1, r=idx;
    while (r >= p-1-i) { r -= (p-1-i); ++i; j=i+1; }
    j += r;
    
    const std::vector<double>& Xi = S.data[i];
    const std::vector<double>& Yj = S.data[j];
    
    lx.push_back(S.names[i]);
    ly.push_back(S.names[j]);
    
    if (do_pearson)   lpear.push_back( pearson_fast(Xi, Yj) );
    if (do_spearman) lspear.push_back( spearman_fast(Xi, Yj, rx, ry, Rx, Ry) );
    if (do_kendall)   lkend.push_back( kendall_tau_b_fast(Xi, Yj) );
    if (do_hoeffding)lhoef.push_back( hoeffding_D_v6(Xi, Yj) );
    if (do_distance) ldcor.push_back( distance_correlation_v6(Xi, Yj) );
    if (do_biweight) lbic .push_back( bicor_v6(Xi, Yj) );
    if (do_pbend)     lpb  .push_back( pbend_v6(Xi, Yj) );
  }
  
#ifdef _OPENMP
#pragma omp critical
#endif
{
  vx.insert(vx.end(), lx.begin(), lx.end());
  vy.insert(vy.end(), ly.begin(), ly.end());
  if (do_pearson)   v_pear.insert(v_pear.end(), lpear.begin(), lpear.end());
  if (do_spearman) v_spear.insert(v_spear.end(), lspear.begin(), lspear.end());
  if (do_kendall)   v_kend.insert(v_kend.end(), lkend.begin(), lkend.end());
  if (do_hoeffding)v_hoef.insert(v_hoef.end(), lhoef.begin(), lhoef.end());
  if (do_distance) v_dcor.insert(v_dcor.end(), ldcor.begin(), ldcor.end());
  if (do_biweight) v_bic .insert(v_bic .end(), lbic .begin(), lbic .end());
  if (do_pbend)     v_pb  .insert(v_pb  .end(), lpb  .begin(), lpb  .end());
}
}
// construct DataFrame
CharacterVector X(vx.begin(), vx.end()), Y(vy.begin(), vy.end());
if (method=="pearson")
  return DataFrame::create(_["x"]=X, _["y"]=Y, _["pearson"]=NumericVector(v_pear.begin(), v_pear.end()));
if (method=="spearman")
  return DataFrame::create(_["x"]=X, _["y"]=Y, _["spearman"]=NumericVector(v_spear.begin(), v_spear.end()));
if (method=="kendall")
  return DataFrame::create(_["x"]=X, _["y"]=Y, _["kendall"]=NumericVector(v_kend.begin(), v_kend.end()));
if (method=="hoeffding")
  return DataFrame::create(_["x"]=X, _["y"]=Y, _["hoeffding"]=NumericVector(v_hoef.begin(), v_hoef.end()));
if (method=="distance")
  return DataFrame::create(_["x"]=X, _["y"]=Y, _["distance"]=NumericVector(v_dcor.begin(), v_dcor.end()));
if (method=="biweight")
  return DataFrame::create(_["x"]=X, _["y"]=Y, _["biweight"]=NumericVector(v_bic.begin(), v_bic.end()));
if (method=="pbend")
  return DataFrame::create(_["x"]=X, _["y"]=Y, _["pbend"]=NumericVector(v_pb.begin(), v_pb.end()));
if (method=="robust")
  return DataFrame::create(_["x"]=X, _["y"]=Y,
                           _["biweight"]=NumericVector(v_bic.begin(), v_bic.end()),
                           _["pbend"]   =NumericVector(v_pb.begin(),   v_pb.end()));
if (method=="alternative")
  return DataFrame::create(_["x"]=X, _["y"]=Y,
                           _["hoeffding"]=NumericVector(v_hoef.begin(), v_hoef.end()),
                           _["distance"] =NumericVector(v_dcor.begin(), v_dcor.end()));
// all
return DataFrame::create(_["x"]=X, _["y"]=Y,
                         _["pearson"] =NumericVector(v_pear.begin(), v_pear.end()),
                         _["spearman"]=NumericVector(v_spear.begin(), v_spear.end()),
                         _["kendall"] =NumericVector(v_kend.begin(), v_kend.end()),
                         _["hoeffding"]=NumericVector(v_hoef.begin(), v_hoef.end()),
                         _["distance"] =NumericVector(v_dcor.begin(), v_dcor.end()),
                         _["biweight"] =NumericVector(v_bic.begin(), v_bic.end()),
                         _["pbend"]    =NumericVector(v_pb.begin(),   v_pb.end()));
 }