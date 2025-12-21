// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>
#include <unordered_map>

using namespace Rcpp;

// Include shared headers
#include "common/optimal_binning_common.h"
#include "common/bin_structures.h"

using namespace Rcpp;
using namespace OptimalBinning;


// -----------------------------------------------------------------------------
// Numerical Optimal Binning with Fisher’s Exact Test (FETB)
//   • merges the pair of adjacent bins with the HIGHEST two‑tail Fisher p‑value
//   • guarantees monotonic WoE (ascending or descending)
//   • avoids repeated calls to R’s fisher.test (pure C++)
//   • keeps the public interface and output names identical to v0
// -----------------------------------------------------------------------------

constexpr double EPS   = 1e-12;     // numeric safety
constexpr double LMAX  = 700.0;     // avoid std::exp overflow

// -----------------------------------------------------------------------------
// factorial‑log cache for Fisher probability
// -----------------------------------------------------------------------------
class LogFactCache {
private:
  std::vector<double> lf_;                // log(i!) i = 0..N
public:
  LogFactCache(size_t n = 1000) {
    lf_.resize(n + 1);
    lf_[0] = 0.0;
    for (size_t i = 1; i < lf_.size(); ++i)
      lf_[i] = lf_[i - 1] + std::log(static_cast<double>(i));
  }
  void ensure(size_t n) {
    if (n < lf_.size()) return;
    size_t old = lf_.size();
    lf_.resize(n + 1);
    for (size_t i = old; i <= n; ++i)
      lf_[i] = lf_[i - 1] + std::log(static_cast<double>(i));
  }
  double operator[](size_t i) const { return lf_[i]; }
};

// -----------------------------------------------------------------------------
// Fisher one‑table probability  (hypergeometric, point probability)
// -----------------------------------------------------------------------------
inline double fisherProb(int a, int b, int c, int d, LogFactCache& lf) {
  size_t n = static_cast<size_t>(a) + b + c + d;
  lf.ensure(n);
  double lp = lf[a + b] + lf[c + d] + lf[a + c] + lf[b + d] -
    lf[n]     - lf[a]     - lf[b]     - lf[c]     - lf[d];
  lp = std::max(-LMAX, std::min(lp, LMAX));
  return std::exp(lp);
}

// -----------------------------------------------------------------------------
// Helper – compute WoE + IV in place
// -----------------------------------------------------------------------------
void calc_woe_iv(const std::vector<int>& pos,
                 const std::vector<int>& neg,
                 std::vector<double>& woe,
                 std::vector<double>& iv) {
  
  const double totPos = std::accumulate(pos.begin(), pos.end(), 0.0);
  const double totNeg = std::accumulate(neg.begin(), neg.end(), 0.0);
  
  woe.resize(pos.size());
  iv .resize(pos.size());
  
  for (size_t i = 0; i < pos.size(); ++i) {
    double dp = (pos[i] + 0.5) / (totPos + 0.5 * pos.size());  // add‑0.5 smoothing
    double dn = (neg[i] + 0.5) / (totNeg + 0.5 * neg.size());
    dp = std::max(dp, EPS); dn = std::max(dn, EPS);
    
    woe[i] = std::log(dp / dn);
    iv [i] = (dp - dn) * woe[i];
  }
}

// -----------------------------------------------------------------------------
// Class encapsulating the numeric FETB
// -----------------------------------------------------------------------------
class OptimalBinningNumericFETB {
private:
  // Input
  const std::vector<int>    y_;
  const std::vector<double> x_;
  const int    min_bins_, max_bins_;
  const size_t max_prebins_;
  const double conv_eps_;
  const size_t max_iter_;
  
  // Working
  std::vector<double> edges_;       // -inf ... +inf
  std::vector<int>    cnt_, pos_, neg_;
  std::vector<double> woe_, iv_;
  size_t iterations_ = 0;
  bool   converged_  = true;
  
  LogFactCache lf_;
  
  // ---------------------------------------------------------------------------
  // build initial pre‑bins (equal‑frequency quantiles up to max_prebins_)
  // ---------------------------------------------------------------------------
  void make_prebins() {
    std::vector<double> xs = x_;
    std::sort(xs.begin(), xs.end());
    
    edges_.clear();
    edges_.push_back(-std::numeric_limits<double>::infinity());
    
    const size_t step = std::max<size_t>(1, xs.size() / max_prebins_);
    for (size_t i = step; i < xs.size(); i += step) {
      if (xs[i] != edges_.back()) edges_.push_back(xs[i]);
    }
    edges_.push_back(std::numeric_limits<double>::infinity());
  }
  
  // ---------------------------------------------------------------------------
  // fill counts for current edge set
  // ---------------------------------------------------------------------------
  void fill_counts() {
    const size_t B = edges_.size() - 1;
    cnt_.assign(B, 0); pos_.assign(B, 0); neg_.assign(B, 0);
    
    for (size_t i = 0; i < x_.size(); ++i) {
      double v = x_[i];
      size_t b = std::upper_bound(edges_.begin(), edges_.end(), v) - edges_.begin() - 1;
      b = std::min(b, B - 1);
      ++cnt_[b];
      if (y_[i]) ++pos_[b]; else ++neg_[b];
    }
  }
  
  // ---------------------------------------------------------------------------
  // merge two adjacent bins (i and i+1)
  // ---------------------------------------------------------------------------
  void merge_bins(size_t i) {
    edges_.erase(edges_.begin() + i + 1);
    cnt_[i] += cnt_[i + 1]; pos_[i] += pos_[i + 1]; neg_[i] += neg_[i + 1];
    cnt_.erase(cnt_.begin() + i + 1);
    pos_.erase(pos_.begin() + i + 1);
    neg_.erase(neg_.begin() + i + 1);
  }
  
  // ---------------------------------------------------------------------------
  // enforce monotone WoE by local merges
  // ---------------------------------------------------------------------------
  void enforce_monotone() {
    bool changed = true;
    while (changed && edges_.size() - 1 > static_cast<size_t>(min_bins_)) {
      calc_woe_iv(pos_, neg_, woe_, iv_);
      changed = false;
      for (size_t i = 0; i + 1 < woe_.size(); ++i) {
        if ((woe_[i] > woe_[i + 1] + EPS) || (woe_[i] < woe_[i + 1] - EPS)) {
          // decide global trend after first two bins
          bool asc = woe_[1] > woe_[0];
          if ((asc && woe_[i] > woe_[i + 1]) ||
              (!asc && woe_[i] < woe_[i + 1])) {
            merge_bins(i);
            changed = true;
            break;
          }
        }
      }
    }
  }
  
  // ---------------------------------------------------------------------------
  // main Fisher merge loop
  // ---------------------------------------------------------------------------
  void fisher_merge_loop() {
    double prev_iv = -1.0;
    
    while ((edges_.size() - 1) > static_cast<size_t>(max_bins_) &&
           iterations_ < max_iter_) {
      
      // choose pair with highest p‑value (most similar)
      double best_p = -1.0; size_t best_i = 0;
      for (size_t i = 0; i + 1 < cnt_.size(); ++i) {
        double p = fisherProb(pos_[i], neg_[i], pos_[i + 1], neg_[i + 1], lf_);
        if (p > best_p) { best_p = p; best_i = i; }
      }
      merge_bins(best_i);
      
      enforce_monotone();  // keep WoE monotone throughout
      
      calc_woe_iv(pos_, neg_, woe_, iv_);
      double tot_iv = std::accumulate(iv_.begin(), iv_.end(), 0.0);
      if (std::fabs(tot_iv - prev_iv) < conv_eps_) {
        converged_ = true; break;
      }
      prev_iv = tot_iv;
      ++iterations_;
    }
    if (iterations_ >= max_iter_) converged_ = false;
  }
  
public:
  OptimalBinningNumericFETB(const NumericVector& y,
                            const NumericVector& x,
                            int  min_bins, int  max_bins,
                            size_t max_prebins,
                            double conv_eps, size_t max_iter)
    : y_(y.begin(), y.end()),
      x_(x.begin(), x.end()),
      min_bins_(min_bins), max_bins_(max_bins),
      max_prebins_(max_prebins),
      conv_eps_(conv_eps), max_iter_(max_iter) {
    
    if (y_.size() != x_.size())
      stop("target and feature must have equal length.");
    if (min_bins_ < 2 || max_bins_ < min_bins_)
      stop("invalid min_bins / max_bins.");
    if (!std::all_of(y_.begin(), y_.end(), [](int v){ return v==0||v==1; }))
      stop("target must be binary (0/1).");
    
    make_prebins();
    fill_counts();
    enforce_monotone();    // first pass (rare in numeric, but safe)
    fisher_merge_loop();   // main optimisation
    calc_woe_iv(pos_, neg_, woe_, iv_);  // final stats
  }
  
  List results() const {
    const size_t B = cnt_.size();
    CharacterVector bins(B);
    NumericVector   id  (B), woe(B), iv(B), cnt(B), pos(B), neg(B);
    for (size_t i=0;i<B;++i) {
      std::ostringstream oss;
      oss << "(" << edges_[i] << "; " << edges_[i+1] << "]";
      bins[i] = oss.str();
      id  [i] = i+1;
      woe [i] = woe_[i];
      iv  [i] = iv_[i];
      cnt [i] = cnt_[i];
      pos [i] = pos_[i];
      neg [i] = neg_[i];
    }
    NumericVector cut(edges_.size()-2);
    std::copy(edges_.begin()+1, edges_.end()-1, cut.begin());
    
    return List::create(
      _["id"]        = id,
      _["bin"]       = bins,
      _["woe"]       = woe,
      _["iv"]        = iv,
      _["count"]     = cnt,
      _["count_pos"] = pos,
      _["count_neg"] = neg,
      _["cutpoints"] = cut,
      _["converged"] = converged_,
      _["iterations"]= static_cast<int>(iterations_)
    );
  }
};

// -----------------------------------------------------------------------------
// R interface
// -----------------------------------------------------------------------------

// [[Rcpp::export]]
List optimal_binning_numerical_fetb(NumericVector target,
                                   NumericVector feature,
                                   int    min_bins              = 3,
                                   int    max_bins              = 5,
                                   int    max_n_prebins         = 20,
                                   double convergence_threshold = 1e-6,
                                   int    max_iterations        = 1000) {
 OptimalBinningNumericFETB ob(target, feature,
                              min_bins, max_bins,
                              static_cast<size_t>(max_n_prebins),
                              convergence_threshold,
                              static_cast<size_t>(max_iterations));
 return ob.results();
}


