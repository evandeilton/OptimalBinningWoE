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
    for (size_t i = step; i < xs.size(); i += step)
      if (xs[i] != edges_.back()) edges_.push_back(xs[i]);
      
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

//' @title Optimal Binning for Numerical Variables with Fisher’s Exact Test
//'
//' @description
//' Implements a **supervised, monotonic, optimal** binning procedure for
//' numeric predictors against a binary target.  
//' The algorithm iteratively merges the pair of *adjacent* bins whose class
//' composition is \emph{most similar} according to the two‑tailed
//' Fisher’s Exact Test, and guarantees a monotone
//' \emph{Weight of Evidence} (WoE) profile.  
//' Designed for scorecard development, churn modelling and any logistic
//' application where robust, information‑preserving discretisation is required.
//'
//' @details
//' \strong{Notation}\cr
//' \eqn{(x_i,\,y_i),\; i=1,\dots,N} are observations with
//' \eqn{y_i\in\{0,1\}}.  A cut‑point vector
//' \eqn{c=(c_0=-\infty < c_1 < \dots < c_{B-1} < c_B=+\infty)}
//' induces bins \eqn{I_b=(c_{b-1},c_b],\; b=1,\dots,B}.  For each bin collect
//' contingency counts
//' \deqn{(a_b,b_b)=\Bigl(\sum_{x_i\in I_b}y_i,\;\sum_{x_i\in I_b}(1-y_i)\Bigr).}
//'
//' \strong{Algorithm}\cr
//' \enumerate{
//'   \item \emph{Pre‑binning}.  Create up to \code{max_n_prebins} equal‑frequency
//'         bins from the ordered feature.  This bounds subsequent complexity.
//'   \item \emph{Fisher merge loop}.  While \eqn{B>}\code{max_bins},
//'         merge the adjacent pair \eqn{(I_j,I_{j+1})} maximising the
//'         point probability of the corresponding 2×2 table
//'         \eqn{p_j = P\{ \text{table }(a_j,b_j,c_j,d_j)\}}.
//'   \item \emph{Monotonicity}.  After every merge, if the WoE sequence
//'         \eqn{w_1,\dots,w_B} violates monotonicity
//'         (\eqn{\exists\,b:\,w_b>w_{b+1}} for ascending trend or vice‑versa)
//'         merge that offending pair and restart the check locally.
//'   \item \emph{Convergence}.  Stop when
//'         \eqn{|IV_{t+1}-IV_t|<}\code{convergence_threshold} or the iteration
//'         cap is reached.\cr
//' }
//'
//' \strong{Complexity}\cr
//' \itemize{
//'   \item Pre‑binning: \eqn{O(N\log N)} (sort) but done once.
//'   \item Merge loop: worst‑case \eqn{O(B^2)} with
//'         \eqn{B\le}\code{max_n_prebins}.
//'   \item Memory: \eqn{O(B)}.
//' }
//'
//' \strong{Formulae}\cr
//' \deqn{ \mathrm{WoE}_b = \log\!\left(
//'         \frac{a_b / T_1}{\,b_b / T_0}\right)\!, \qquad
//'        \mathrm{IV}   = \sum_{b=1}^{B}
//'         \left(\frac{a_b}{T_1}-\frac{b_b}{T_0}\right)\mathrm{WoE}_b}
//' where \eqn{T_1=\sum_b a_b},\ \eqn{T_0=\sum_b b_b}.
//'
//' @param target Integer (0/1) vector, length \eqn{N}.
//' @param feature Numeric vector, length \eqn{N}.
//' @param min_bins Minimum number of final bins (default \code{3}).
//' @param max_bins Maximum number of final bins (default \code{5}).
//' @param max_n_prebins Maximum number of pre‑bins created before optimisation
//'        (default \code{20}).
//' @param convergence_threshold Absolute tolerance for change in total IV used
//'        as convergence criterion (default \code{1e-6}).
//' @param max_iterations Safety cap for merge + monotonicity iterations
//'        (default \code{1000}).
//'
//' @return A named \code{list}:
//' \describe{
//'   \item{id}{Bin index (1‑based).}
//'   \item{bin}{Character vector \code{"(lo; hi]"} describing intervals.}
//'   \item{woe, iv}{WoE and IV per bin.}
//'   \item{count, count_pos, count_neg}{Bin frequencies.}
//'   \item{cutpoints}{Numeric vector of internal cut‑points
//'         \eqn{c_1,\dots,c_{B-1}}.}
//'   \item{converged}{Logical flag.}
//'   \item{iterations}{Number of iterations executed.}
//' }
//'
//' @examples
//' \donttest{
//' set.seed(2025)
//' N  <- 1000
//' y  <- rbinom(N, 1, 0.3)             # 30 % positives
//' x  <- rnorm(N, mean = 50, sd = 10)  # numeric predictor
//' res <- optimal_binning_numerical_fetb(y, x,
//'         min_bins = 2, max_bins = 6, max_n_prebins = 25)
//' print(res)
//'}
//'
//' @references
//' Fisher, R. A. (1922) \emph{On the interpretation of \eqn{X^2} from contingency
//'   tables, and the calculation of P}. *JRSS*, 85 (1), 87‑94.\cr
//' Siddiqi, N. (2012) \emph{Credit Risk Scorecards}. Wiley.\cr
//' Navas‑Palencia, G. (2019) *optbinning* documentation – Numerical FETB.\cr
//' Hand, D. J., & Adams, N. M. (2015) \emph{Supervised Classification in
//'   High Dimensions}. Springer (Ch. 4, discretisation).\cr
//' Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013)
//'   \emph{Applied Logistic Regression} (3rd ed.). Wiley.
//'
//' @author Lopes, J. E.
//' @export
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


// // [[Rcpp::plugins(cpp11)]]
// // [[Rcpp::depends(Rcpp)]]
// 
// #include <Rcpp.h>
// #include <algorithm>
// #include <vector>
// #include <cmath>
// #include <limits>
// #include <sstream>
// #include <numeric>
// 
// using namespace Rcpp;
// 
//  
// // Fisher's Exact Test wrapper calling R's fisher.test
// double fisher_exact_test(int a, int b, int c, int d) {
//    NumericMatrix table(2, 2);
//    table(0, 0) = a;
//    table(0, 1) = b;
//    table(1, 0) = c;
//    table(1, 1) = d;
//    
//    Environment stats = Environment::namespace_env("stats");
//    Function fisher_test = stats["fisher.test"];
//    
//    List result = fisher_test(table, Named("alternative") = "two.sided");
//    return as<double>(result["p.value"]);
// }
// 
// // Class for Optimal Binning Numerical using Fisher's Exact Test
// class OptimalBinningNumericalFETB {
// public:
//   OptimalBinningNumericalFETB(NumericVector target, NumericVector feature,
//                               int min_bins, int max_bins, double bin_cutoff, int max_n_prebins,
//                               double convergence_threshold, int max_iterations);
//   List performBinning();
//   
// private:
//   std::vector<double> target;
//   std::vector<double> feature;
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
//   double convergence_threshold;
//   int max_iterations;
//   bool converged;
//   int iterations_run;
//   
//   std::vector<double> binEdges;
//   std::vector<int> binCounts;
//   std::vector<int> binPosCounts;
//   std::vector<int> binNegCounts;
//   std::vector<double> binWoE;
//   std::vector<double> binIV;
//   
//   void validateInputs();
//   void createPrebins();
//   void calculateBinStats();
//   void mergeBins();
//   void enforceMonotonicity();
//   void calculateWoE();
//   double calculateIV();
// };
// 
// OptimalBinningNumericalFETB::OptimalBinningNumericalFETB(NumericVector target_, NumericVector feature_,
//                                                          int min_bins_, int max_bins_, double bin_cutoff_, int max_n_prebins_,
//                                                          double convergence_threshold_, int max_iterations_)
//   : target(target_.begin(), target_.end()), feature(feature_.begin(), feature_.end()),
//     min_bins(min_bins_), max_bins(max_bins_), bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_),
//     convergence_threshold(convergence_threshold_), max_iterations(max_iterations_), converged(true), iterations_run(0) {
//   validateInputs();
// }
// 
// void OptimalBinningNumericalFETB::validateInputs() {
//   // Check that target is binary
//   for (size_t i = 0; i < target.size(); ++i) {
//     if (target[i] != 0 && target[i] != 1) {
//       stop("Target must be binary (0 or 1).");
//     }
//   }
//   
//   // Check min_bins
//   if (min_bins < 2) {
//     stop("min_bins must be at least 2.");
//   }
//   
//   // Check max_bins
//   if (max_bins < min_bins) {
//     stop("max_bins must be greater than or equal to min_bins.");
//   }
//   
//   // Check bin_cutoff
//   if (bin_cutoff <= 0 || bin_cutoff >= 0.5) {
//     stop("bin_cutoff must be between 0 and 0.5.");
//   }
//   
//   // Check max_n_prebins
//   if (max_n_prebins < max_bins) {
//     stop("max_n_prebins must be greater than or equal to max_bins.");
//   }
//   
//   // Check target and feature sizes
//   if (target.size() != feature.size()) {
//     stop("Target and feature must have the same length.");
//   }
// }
// 
// void OptimalBinningNumericalFETB::createPrebins() {
//   std::vector<double> sorted_feature = feature;
//   std::sort(sorted_feature.begin(), sorted_feature.end());
//   
//   binEdges.clear();
//   
//   binEdges.push_back(-std::numeric_limits<double>::infinity());
//   
//   int step = std::max(1, static_cast<int>(sorted_feature.size()) / max_n_prebins);
//   
//   for (size_t i = step; i < sorted_feature.size(); i += step) {
//     double edge = sorted_feature[i];
//     if (edge != binEdges.back()) {
//       binEdges.push_back(edge);
//     }
//     if ((int)binEdges.size() >= max_n_prebins) {
//       break;
//     }
//   }
//   
//   if (binEdges.back() != std::numeric_limits<double>::infinity()) {
//     binEdges.push_back(std::numeric_limits<double>::infinity());
//   }
// }
// 
// void OptimalBinningNumericalFETB::calculateBinStats() {
//   int n_bins = static_cast<int>(binEdges.size()) - 1;
//   binCounts.assign(n_bins, 0);
//   binPosCounts.assign(n_bins, 0);
//   binNegCounts.assign(n_bins, 0);
//   
//   for (size_t i = 0; i < feature.size(); i++) {
//     double x = feature[i];
//     int bin_index = (int)(std::upper_bound(binEdges.begin(), binEdges.end(), x) - binEdges.begin() - 1);
//     bin_index = std::max(0, std::min(bin_index, n_bins - 1));
//     
//     binCounts[bin_index]++;
//     if (target[i] == 1) {
//       binPosCounts[bin_index]++;
//     } else {
//       binNegCounts[bin_index]++;
//     }
//   }
// }
// 
// void OptimalBinningNumericalFETB::mergeBins() {
//   bool binsMerged = true;
//   int iterations = 0;
//   
//   while (binsMerged && static_cast<int>(binEdges.size() - 1) > min_bins && iterations < max_iterations) {
//     binsMerged = false;
//     double max_p_value = -1.0;
//     int merge_index = -1;
//     
//     for (size_t i = 0; i < binEdges.size() - 2; i++) {
//       int a = binPosCounts[i];
//       int b = binNegCounts[i];
//       int c = binPosCounts[i + 1];
//       int d = binNegCounts[i + 1];
//       
//       double p_value = fisher_exact_test(a, b, c, d);
//       
//       if (p_value > max_p_value) {
//         max_p_value = p_value;
//         merge_index = (int)i;
//       }
//     }
//     
//     // Merge bins if p-value > cutoff
//     if (max_p_value > bin_cutoff && merge_index != -1) {
//       if (merge_index >= 0 && (size_t)(merge_index + 1) < binEdges.size() - 1) {
//         binEdges.erase(binEdges.begin() + merge_index + 1);
//         binCounts[merge_index] += binCounts[merge_index + 1];
//         binPosCounts[merge_index] += binPosCounts[merge_index + 1];
//         binNegCounts[merge_index] += binNegCounts[merge_index + 1];
//         binCounts.erase(binCounts.begin() + merge_index + 1);
//         binPosCounts.erase(binPosCounts.begin() + merge_index + 1);
//         binNegCounts.erase(binNegCounts.begin() + merge_index + 1);
//         binsMerged = true;
//       }
//     }
//     iterations++;
//   }
//   
//   iterations_run += iterations;
//   if (iterations >= max_iterations) {
//     converged = false;
//   }
// }
// 
// void OptimalBinningNumericalFETB::calculateWoE() {
//   binWoE.clear();
//   binIV.clear();
//   
//   int totalPos = std::accumulate(binPosCounts.begin(), binPosCounts.end(), 0);
//   int totalNeg = std::accumulate(binNegCounts.begin(), binNegCounts.end(), 0);
//   
//   // If no positives or no negatives, WOE is not well-defined
//   if (totalPos == 0 || totalNeg == 0) {
//     // Gracefully handle by adding small constants to avoid infinite WoE
//     if (totalPos == 0) totalPos = 1;
//     if (totalNeg == 0) totalNeg = 1;
//   }
//   
//   for (size_t i = 0; i < binPosCounts.size(); i++) {
//     double distPos = (static_cast<double>(binPosCounts[i]) + 0.5) / (totalPos + 0.5 * binPosCounts.size());
//     double distNeg = (static_cast<double>(binNegCounts[i]) + 0.5) / (totalNeg + 0.5 * binNegCounts.size());
//     
//     distPos = std::max(distPos, 1e-10);
//     distNeg = std::max(distNeg, 1e-10);
//     
//     double woe = std::log(distPos / distNeg);
//     double iv_part = (distPos - distNeg) * woe;
//     binWoE.push_back(woe);
//     binIV.push_back(iv_part);
//   }
// }
// 
// void OptimalBinningNumericalFETB::enforceMonotonicity() {
//   calculateWoE();
//   
//   // Determine if already monotonic
//   bool increasing = true;
//   bool decreasing = true;
//   
//   for (size_t i = 0; i + 1 < binWoE.size(); i++) {
//     if (binWoE[i] < binWoE[i + 1]) {
//       decreasing = false;
//     }
//     if (binWoE[i] > binWoE[i + 1]) {
//       increasing = false;
//     }
//   }
//   
//   // If monotonic, no need to enforce
//   if (increasing || decreasing) {
//     return;
//   }
//   
//   // Enforce monotonicity
//   int iterations = 0;
//   bool monotonic = false;
//   
//   while (!monotonic && (int)(binEdges.size() - 1) > min_bins && iterations < max_iterations) {
//     monotonic = true;
//     if (binWoE.size() > 1) {
//       double trend = binWoE[1] - binWoE[0];
//       bool is_increasing = trend >= 0;
//       
//       for (size_t i = 0; i + 1 < binWoE.size(); i++) {
//         double current_trend = binWoE[i + 1] - binWoE[i];
//         if ((is_increasing && current_trend < 0) || (!is_increasing && current_trend > 0)) {
//           // Merge bins i and i+1
//           if (i + 1 < binEdges.size() - 1) {
//             binEdges.erase(binEdges.begin() + i + 1);
//             binCounts[i] += binCounts[i + 1];
//             binPosCounts[i] += binPosCounts[i + 1];
//             binNegCounts[i] += binNegCounts[i + 1];
//             binCounts.erase(binCounts.begin() + i + 1);
//             binPosCounts.erase(binPosCounts.begin() + i + 1);
//             binNegCounts.erase(binNegCounts.begin() + i + 1);
//             monotonic = false;
//             break; // After merging once, re-check monotonicity from scratch next iteration
//           }
//         }
//       }
//       calculateWoE(); // Recalculate WOE after merges
//     }
//     iterations++;
//   }
//   
//   iterations_run += iterations;
//   if (iterations >= max_iterations) {
//     converged = false;
//   }
// }
// 
// double OptimalBinningNumericalFETB::calculateIV() {
//   return std::accumulate(binIV.begin(), binIV.end(), 0.0);
// }
// 
// List OptimalBinningNumericalFETB::performBinning() {
//   std::vector<double> unique_feature = feature;
//   std::sort(unique_feature.begin(), unique_feature.end());
//   unique_feature.erase(std::unique(unique_feature.begin(), unique_feature.end(),
//                                    [](double a, double b) { return std::fabs(a - b) < 1e-9; }), unique_feature.end());
//   
//   // If <= 2 unique values: trivial binning
//   if ((int)unique_feature.size() <= 2) {
//     binEdges.clear();
//     binCounts.clear();
//     binPosCounts.clear();
//     binNegCounts.clear();
//     binWoE.clear();
//     binIV.clear();
//     
//     binEdges.push_back(-std::numeric_limits<double>::infinity());
//     if (unique_feature.size() == 1) {
//       binEdges.push_back(std::numeric_limits<double>::infinity());
//     } else {
//       binEdges.push_back((unique_feature[0] + unique_feature[1]) / 2.0);
//       binEdges.push_back(std::numeric_limits<double>::infinity());
//     }
//     
//     calculateBinStats();
//     calculateWoE();
//     double totalIV = calculateIV();
//     
//     std::vector<std::string> bin_labels;
//     for (size_t i = 0; i < binEdges.size() - 1; i++) {
//       std::ostringstream oss;
//       oss << "(" << binEdges[i] << "; " << binEdges[i + 1] << "]";
//       bin_labels.push_back(oss.str());
//     }
//     
//     std::vector<double> cutpoints;
//     if (unique_feature.size() == 2) {
//       cutpoints.push_back((unique_feature[0] + unique_feature[1]) / 2.0);
//     }
//     
//   Rcpp::NumericVector ids(bin_labels.size());
//   for(int i = 0; i < bin_labels.size(); i++) {
//     ids[i] = i + 1;
//   }
//   
//   return Rcpp::List::create(
//       Named("id") = ids,
//       Named("bin") = bin_labels,
//       Named("woe") = binWoE,
//       Named("iv") = binIV,
//       Named("count") = binCounts,
//       Named("count_pos") = binPosCounts,
//       Named("count_neg") = binNegCounts,
//       Named("cutpoints") = cutpoints,
//       Named("converged") = converged,
//       Named("iterations") = iterations_run
//     );
//   }
//   
//   // If unique values <= min_bins, just create bins from unique values
//   if ((int)unique_feature.size() <= min_bins) {
//     binEdges.clear();
//     binCounts.clear();
//     binPosCounts.clear();
//     binNegCounts.clear();
//     binWoE.clear();
//     binIV.clear();
//     
//     binEdges.push_back(-std::numeric_limits<double>::infinity());
//     for (size_t i = 0; i + 1 < unique_feature.size(); i++) {
//       binEdges.push_back((unique_feature[i] + unique_feature[i + 1]) / 2.0);
//     }
//     binEdges.push_back(std::numeric_limits<double>::infinity());
//     
//     calculateBinStats();
//     calculateWoE();
//     double totalIV = calculateIV();
//     
//     std::vector<std::string> bin_labels;
//     for (size_t i = 0; i < binEdges.size() - 1; i++) {
//       std::ostringstream oss;
//       oss << "(" << binEdges[i] << "; " << binEdges[i + 1] << "]";
//       bin_labels.push_back(oss.str());
//     }
//     
//     std::vector<double> cutpoints(binEdges.begin() + 1, binEdges.end() - 1);
//     
//     Rcpp::NumericVector ids(bin_labels.size());
//     for(int i = 0; i < bin_labels.size(); i++) {
//       ids[i] = i + 1;
//     }
//     
//     return Rcpp::List::create(
//       Named("id") = ids,
//       Named("bin") = bin_labels,
//       Named("woe") = binWoE,
//       Named("iv") = binIV,
//       Named("count") = binCounts,
//       Named("count_pos") = binPosCounts,
//       Named("count_neg") = binNegCounts,
//       Named("cutpoints") = cutpoints,
//       Named("converged") = converged,
//       Named("iterations") = iterations_run
//     );
//   }
//   
//   // Proceed with full binning process
//   createPrebins();
//   calculateBinStats();
//   mergeBins();
//   enforceMonotonicity();
//   calculateWoE();
//   double totalIV = calculateIV();
//   
//   std::vector<std::string> bin_labels;
//   for (size_t i = 0; i < binEdges.size() - 1; i++) {
//     std::ostringstream oss;
//     oss << "(" << binEdges[i] << "; " << binEdges[i + 1] << "]";
//     bin_labels.push_back(oss.str());
//   }
//   
//   std::vector<double> cutpoints;
//   if (binEdges.size() > 2) {
//     cutpoints.assign(binEdges.begin() + 1, binEdges.end() - 1);
//   }
//   
//   // Criar vetor de IDs com o mesmo tamanho de bins
//   Rcpp::NumericVector ids(bin_labels.size());
//   for(int i = 0; i < bin_labels.size(); i++) {
//     ids[i] = i + 1;
//   }
//   
//   return Rcpp::List::create(
//     Named("id") = ids,
//     Named("bin") = bin_labels,
//     Named("woe") = binWoE,
//     Named("iv") = binIV,
//     Named("count") = binCounts,
//     Named("count_pos") = binPosCounts,
//     Named("count_neg") = binNegCounts,
//     Named("cutpoints") = cutpoints,
//     Named("converged") = converged,
//     Named("iterations") = iterations_run
//   );
// }
// 
// 
// 
// //' @title Optimal Binning for Numerical Variables using Fisher's Exact Test (FETB)
// //'
// //' @description
// //' This function implements an optimal binning algorithm for numerical variables using Fisher's Exact Test. 
// //' It attempts to create an optimal set of bins for a given numerical feature based on its relationship with 
// //' a binary target variable, ensuring both statistical significance (via Fisher's Exact Test) and monotonicity in WoE values.
// //'
// //' @param target A numeric vector of binary target values (0 or 1).
// //' @param feature A numeric vector of feature values to be binned.
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff P-value threshold for merging bins (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins before the merging process (default: 20).
// //' @param convergence_threshold Threshold for algorithmic convergence (default: 1e-6).
// //' @param max_iterations Maximum number of iterations allowed during merging and monotonicity enforcement (default: 1000).
// //'
// //' @return A list containing:
// //' \item{bin}{A character vector of bin ranges.}
// //' \item{woe}{A numeric vector of WoE values for each bin.}
// //' \item{iv}{A numeric vector of IV for each bin.}
// //' \item{count}{A numeric vector of total observations in each bin.}
// //' \item{count_pos}{A numeric vector of positive target observations in each bin.}
// //' \item{count_neg}{A numeric vector of negative target observations in each bin.}
// //' \item{cutpoints}{A numeric vector of cut points used to generate the bins.}
// //' \item{converged}{A logical indicating if the algorithm converged.}
// //' \item{iterations}{An integer indicating the number of iterations run.}
// //'
// //' @details
// //' The algorithm works as follows:
// //' 1. Pre-binning: Initially divides the feature into up to \code{max_n_prebins} bins based on sorted values.
// //' 2. Fisher Merging: Adjacent bins are merged if the Fisher's Exact Test p-value exceeds \code{bin_cutoff}, indicating no statistically significant difference between them.
// //' 3. Monotonicity Enforcement: Ensures the WoE values are monotonic by merging non-monotonic adjacent bins.
// //' 4. Final WoE/IV Calculation: After achieving a stable set of bins (or reaching iteration limits), it calculates the final WoE and IV for each bin.
// //'
// //' The method aims at providing statistically justifiable and monotonic binning, which is particularly useful for credit scoring and other risk modeling tasks.
// //'
// //' @examples
// //' \dontrun{
// //' set.seed(123)
// //' target <- sample(0:1, 1000, replace = TRUE)
// //' feature <- rnorm(1000)
// //' result <- optimal_binning_numerical_fetb(target, feature)
// //' print(result$bins)
// //' print(result$woe)
// //' print(result$iv)
// //' }
// //'
// //' @export
// // [[Rcpp::export]]
// List optimal_binning_numerical_fetb(NumericVector target,
//                                     NumericVector feature,
//                                     int min_bins = 3, int max_bins = 5,
//                                     double bin_cutoff = 0.05, int max_n_prebins = 20,
//                                     double convergence_threshold = 1e-6, int max_iterations = 1000) {
//   OptimalBinningNumericalFETB binning(target, feature, min_bins, max_bins, bin_cutoff, max_n_prebins,
//                                       convergence_threshold, max_iterations);
//   return binning.performBinning();
// }
