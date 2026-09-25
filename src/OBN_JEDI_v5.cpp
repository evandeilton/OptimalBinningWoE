// [[Rcpp::plugins(cpp11)]]

#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sstream>
#include <iomanip>
#include <cstdint>
#include <cstring>
#include <cstddef>


// Include shared headers
#include "common/optimal_binning_common.h"
#include "common/bin_structures.h"

using namespace Rcpp;
using namespace OptimalBinning;


static constexpr double EPS = 1e-10;

// Data layout
// -----------
// The engine never needs the observations in their original order: every step
// works on the sorted distinct values and on class counts per interval. The
// feature values are therefore split by class once, each class is sorted, and
// every count is a binary search in those two sorted arrays:
//   #positives in (lo, hi] = upper_bound(pos, hi) - upper_bound(pos, lo).
// That replaces a per-observation bin search with O(bins * log n) work and lets
// the distinct values be produced by merging the two sorted arrays in O(n).

// Sort finite doubles ascending.
//
// This is the dominant cost of the whole algorithm (everything after it is
// O(n) or O(bins * log n)), so large inputs use an LSD radix sort on the
// order-preserving integer image of each double: 6 passes of 11 bits, with
// the histograms of all passes gathered in one read and passes whose digit is
// constant skipped. It orders exactly like operator< on finite values except
// that -0.0 precedes +0.0, which compare equal and are merged downstream.
static void sort_doubles(std::vector<double>& v) {
  const size_t n = v.size();
  if (n < 4096) {
    std::sort(v.begin(), v.end());
    return;
  }
  const int BITS = 11;
  const int PASSES = 6;
  const size_t RADIX = static_cast<size_t>(1) << BITS;
  const std::uint64_t MASK = RADIX - 1;

  std::vector<std::uint64_t> key(n), tmp(n);
  std::vector<size_t> hist(RADIX * PASSES, 0);
  for (size_t i = 0; i < n; ++i) {
    std::uint64_t u;
    std::memcpy(&u, &v[i], sizeof u);
    u = (u >> 63) ? ~u : (u | (static_cast<std::uint64_t>(1) << 63));
    key[i] = u;
    for (int p = 0; p < PASSES; ++p) {
      ++hist[static_cast<size_t>(p) * RADIX + static_cast<size_t>((u >> (p * BITS)) & MASK)];
    }
  }
  for (int p = 0; p < PASSES; ++p) {
    size_t* h = &hist[static_cast<size_t>(p) * RADIX];
    const size_t first = static_cast<size_t>((key[0] >> (p * BITS)) & MASK);
    if (h[first] == n) continue;  // every key has the same digit: nothing to do
    size_t sum = 0;
    for (size_t d = 0; d < RADIX; ++d) {
      const size_t c = h[d];
      h[d] = sum;
      sum += c;
    }
    for (size_t i = 0; i < n; ++i) {
      tmp[h[static_cast<size_t>((key[i] >> (p * BITS)) & MASK)]++] = key[i];
    }
    key.swap(tmp);
  }
  for (size_t i = 0; i < n; ++i) {
    std::uint64_t u = key[i];
    u = (u >> 63) ? (u & ~(static_cast<std::uint64_t>(1) << 63)) : ~u;
    std::memcpy(&v[i], &u, sizeof u);
  }
}

class OBN_Jedi {
private:
  std::vector<double> pos_vals;   // feature values with target == 1, ascending
  std::vector<double> neg_vals;   // feature values with target == 0, ascending
  size_t n_obs;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  double convergence_threshold;
  int max_iterations;

  std::vector<NumericalBin> bins;
  bool converged;
  int iterations_run;

public:
  // pos / neg: the feature values of each class (unsorted; moved in).
  OBN_Jedi(std::vector<double>&& pos, std::vector<double>&& neg,
           int min_b, int max_b, double cutoff,
           int max_pre, double conv_thr, int max_iter)
    : pos_vals(std::move(pos)), neg_vals(std::move(neg)),
      n_obs(0),
      min_bins(std::max(min_b,2)),
      max_bins(std::max(max_b,min_b)),
      bin_cutoff(cutoff),
      max_n_prebins(std::max(max_pre,min_b)),
      convergence_threshold(conv_thr),
      max_iterations(max_iter),
      converged(false),
      iterations_run(0) {

    if(bin_cutoff<=0||bin_cutoff>=1)
      throw std::invalid_argument("bin_cutoff must be in (0,1).");
    if(convergence_threshold<=0)
      throw std::invalid_argument("convergence_threshold must be positive.");
    if(max_iterations<=0)
      throw std::invalid_argument("max_iterations must be positive.");
    if(pos_vals.empty() || neg_vals.empty())
      throw std::invalid_argument("Target must contain both classes (0 and 1).");

    n_obs = pos_vals.size() + neg_vals.size();
    sort_doubles(pos_vals);
    sort_doubles(neg_vals);
  }

  void fit() {
    const std::vector<double> unique_vals = distinct_values();
    size_t n_unique = unique_vals.size();

    if(n_unique<=2) {
      handle_low_unique(unique_vals);
      converged=true;
      iterations_run=0;
      return;
    }

    // Pre-binning: use quantiles
    initial_prebin(unique_vals);
    assign_bins();
    merge_small_bins();
    calculate_woe_iv();

    double prev_iv = total_iv();
    for(int iter=0;iter<max_iterations;iter++) {
      enforce_monotonicity();
      merge_to_max_bins();
      calculate_woe_iv();
      double current_iv = total_iv();
      if(std::fabs(current_iv - prev_iv)<convergence_threshold) {
        converged=true;
        iterations_run=iter+1;
        break;
      }
      prev_iv=current_iv;
      iterations_run=iter+1;
    }

    if(!converged) iterations_run=max_iterations;
  }

  List create_output() const {
    const R_xlen_t nb = static_cast<R_xlen_t>(bins.size());
    CharacterVector bin_names(nb);
    NumericVector woe_vals(nb);
    NumericVector iv_vals(nb);
    IntegerVector count_vals(nb);
    IntegerVector cpos_vals(nb);
    IntegerVector cneg_vals(nb);
    NumericVector cutpoints;
    if(nb>1) cutpoints = NumericVector(nb-1);

    for (R_xlen_t i=0;i<nb;i++){
      const NumericalBin& b = bins[static_cast<size_t>(i)];
      std::ostringstream oss;
      oss<<"("<<edge_to_str(b.lower_bound)<<";"<<edge_to_str(b.upper_bound)<<"]";
      bin_names[i]=oss.str();
      woe_vals[i]=b.woe;
      iv_vals[i]=b.iv;
      count_vals[i]=b.count;
      cpos_vals[i]=b.count_pos;
      cneg_vals[i]=b.count_neg;
      if(i<nb-1) {
        cutpoints[i]=b.upper_bound;
      }
    }

    Rcpp::NumericVector ids(nb);
    for(R_xlen_t i = 0; i < nb; i++) {
      ids[i] = static_cast<double>(i + 1);
    }

    return Rcpp::List::create(
      Named("id") = ids,
      Named("bin")=bin_names,
      Named("woe")=woe_vals,
      Named("iv")=iv_vals,
      Named("count")=count_vals,
      Named("count_pos")=cpos_vals,
      Named("count_neg")=cneg_vals,
      Named("cutpoints")=cutpoints,
      Named("converged")=converged,
      Named("iterations")=iterations_run
    );
  }

private:

  //-------------------------------------------------------------------------
  // Detailed Steps and Mathematical Formulation
  //-------------------------------------------------------------------------
  // The Weight of Evidence for bin i is:
  // WOE_i = ln((Pos_i / Total_Pos) / (Neg_i / Total_Neg))
  //        = ln( (Pos_i / Neg_i) * (Total_Neg / Total_Pos) )
  // IV_i = (Pos_i/Total_Pos - Neg_i/Total_Neg)*WOE_i
  //
  // Let total_pos = sum Pos_i and total_neg = sum Neg_i over all bins.
  // If total_pos=0 or total_neg=0, WOE and IV can't be computed meaningfully, fallback to zero.
  //
  // Monotonicity:
  // We define monotonic order as either strictly increasing or decreasing WOE across bin edges.
  // Guess direction by comparing number of WOE increments vs decrements.
  // If not monotonic, merge adjacent bins that cause violations until monotonic order is restored or min_bins reached.
  //
  // Minimizing IV loss merges:
  // When the number of bins exceeds max_bins, merge the adjacent pair with the
  // smallest combined IV.
  //
  // Convergence:
  // Convergence is reached when |IV_current - IV_previous| < convergence_threshold or max_iterations is hit.

  // Sorted distinct feature values: a merge of the two sorted class arrays.
  std::vector<double> distinct_values() const {
    std::vector<double> u;
    u.reserve(std::min<size_t>(n_obs, 1024));
    size_t i = 0, k = 0;
    const size_t np = pos_vals.size(), nn = neg_vals.size();
    while (i < np || k < nn) {
      double v;
      if (k >= nn || (i < np && pos_vals[i] < neg_vals[k])) v = pos_vals[i];
      else v = neg_vals[k];
      while (i < np && pos_vals[i] == v) ++i;
      while (k < nn && neg_vals[k] == v) ++k;
      u.push_back(v + 0.0);  // -0.0 and +0.0 are one value; report it unsigned
    }
    return u;
  }

  // Number of elements of the sorted array v that are <= x.
  static int count_le(const std::vector<double>& v, double x) {
    return static_cast<int>(std::upper_bound(v.begin(), v.end(), x) - v.begin());
  }

  // Fill a bin's counts from its interval (lower, upper] (the first bin also
  // holds -Inf..upper; every value is finite).
  void count_interval(NumericalBin& b) const {
    const int p_hi = std::isinf(b.upper_bound) && b.upper_bound > 0
      ? static_cast<int>(pos_vals.size()) : count_le(pos_vals, b.upper_bound);
    const int n_hi = std::isinf(b.upper_bound) && b.upper_bound > 0
      ? static_cast<int>(neg_vals.size()) : count_le(neg_vals, b.upper_bound);
    const int p_lo = std::isinf(b.lower_bound) && b.lower_bound < 0
      ? 0 : count_le(pos_vals, b.lower_bound);
    const int n_lo = std::isinf(b.lower_bound) && b.lower_bound < 0
      ? 0 : count_le(neg_vals, b.lower_bound);
    b.count_pos = p_hi - p_lo;
    b.count_neg = n_hi - n_lo;
    b.count = b.count_pos + b.count_neg;
  }

  void handle_low_unique(const std::vector<double>& unique_vals) {
    bins.clear();
    if(unique_vals.size()==1) {
      // All identical
      bins.emplace_back(-std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity());
    } else {
      // Two unique values
      double cut = unique_vals[0];
      bins.emplace_back(-std::numeric_limits<double>::infinity(), cut);
      bins.emplace_back(cut, std::numeric_limits<double>::infinity());
    }
    for (auto& b : bins) count_interval(b);
    calculate_woe_iv();
  }

  void initial_prebin(const std::vector<double>& unique_vals) {
    int n_unique=(int)unique_vals.size();
    int n_pre = std::min(max_n_prebins,n_unique);
    n_pre=std::max(n_pre,min_bins);

    // Edges at equally spaced positions among the distinct values.
    //
    // When the feature has at least min_bins distinct values the positions are
    // all different and this yields exactly n_pre bins. When it has fewer, the
    // positions collide and the edges reduce to every distinct value but the
    // largest: one bin per distinct value, the finest binning the data allow.
    // (Splitting further, as this routine used to do by halving intervals,
    // could only create bins containing no observation at all.)
    bins.clear();
    bins.reserve(static_cast<size_t>(n_pre));

    std::vector<double> edges;
    edges.push_back(-std::numeric_limits<double>::infinity());
    for(int i=1;i<n_pre;i++){
      double p=(double)i/n_pre;
      int idx=(int)std::floor(p*(n_unique-1));
      double edge=unique_vals[static_cast<size_t>(idx)];
      if(edge>edges.back()) {
        edges.push_back(edge);
      }
    }
    edges.push_back(std::numeric_limits<double>::infinity());

    for (size_t i=0;i+1<edges.size();i++){
      bins.emplace_back(edges[i],edges[i+1]);
    }
  }

  void assign_bins() {
    for (auto& b : bins) count_interval(b);
  }

  void merge_small_bins() {
    bool merged=true;
    double total=(double)n_obs;
    while(merged && (int)bins.size()>min_bins && iterations_run<max_iterations) {
      merged=false;
      for (size_t i=0;i<bins.size();i++){
        double prop=(double)bins[i].count/total;
        if(prop<bin_cutoff && (int)bins.size()>min_bins) {
          if(i==0) {
            if(bins.size()<2)break;
            merge_bins(0,1);
          } else if(i==bins.size()-1) {
            merge_bins(bins.size()-2,bins.size()-1);
          } else {
            // Merge with neighbor minimal count
            if(bins[i-1].count<=bins[i+1].count) {
              merge_bins(i-1,i);
            } else {
              merge_bins(i,i+1);
            }
          }
          merged=true;
          break;
        }
      }
      iterations_run++;
    }
  }

  void calculate_woe_iv() {
    int total_pos=0;
    int total_neg=0;
    for (auto &b:bins) {
      total_pos+=b.count_pos;
      total_neg+=b.count_neg;
    }
    if(total_pos==0||total_neg==0) {
      for (auto &b:bins) {b.woe=0.0;b.iv=0.0;}
      return;
    }
    for(auto &b:bins) {
      double p=(b.count_pos>0)?(double)b.count_pos/total_pos:EPS;
      double q=(b.count_neg>0)?(double)b.count_neg/total_neg:EPS;
      double w=std::log(p/q);
      double iv=(p-q)*w;
      b.woe=w;
      b.iv=iv;
    }
  }

  double total_iv() const {
    double sum=0.0;
    for (auto &b:bins) sum+=b.iv;
    return sum;
  }

  bool guess_trend() {
    if(bins.size()<2)return true;
    int inc=0;int dec=0;
    for(size_t i=1;i<bins.size();i++) {
      if(bins[i].woe>bins[i-1].woe)inc++;
      else if(bins[i].woe<bins[i-1].woe)dec++;
    }
    return inc>=dec;
  }

  void enforce_monotonicity() {
    bool increasing=guess_trend();
    bool merged=true;
    while(merged && (int)bins.size()>min_bins && iterations_run<max_iterations) {
      merged=false;
      for (size_t i=1;i<bins.size();i++){
        if((increasing && bins[i].woe<bins[i-1].woe)||
           (!increasing && bins[i].woe>bins[i-1].woe)) {
          merge_bins(i-1,i);
          calculate_woe_iv();
          merged=true;
          break;
        }
      }
      iterations_run++;
    }
  }

  size_t find_min_iv_merge() const {
    if(bins.size()<2)return bins.size();
    double min_iv_sum=std::numeric_limits<double>::max();
    size_t idx=bins.size();
    for(size_t i=0;i<bins.size()-1;i++){
      double iv_sum=bins[i].iv+bins[i+1].iv;
      if(iv_sum<min_iv_sum) {
        min_iv_sum=iv_sum;
        idx=i;
      }
    }
    return idx;
  }

  void merge_to_max_bins() {
    while((int)bins.size()>max_bins && iterations_run<max_iterations){
      size_t idx=find_min_iv_merge();
      if(idx>=bins.size()-1)break;
      merge_bins(idx,idx+1);
      calculate_woe_iv();
      iterations_run++;
    }
  }

  void merge_bins(size_t i, size_t j) {
    if(i>j)std::swap(i,j);
    if(j>=bins.size())return;
    bins[i].upper_bound=bins[j].upper_bound;
    bins[i].count+=bins[j].count;
    bins[i].count_pos+=bins[j].count_pos;
    bins[i].count_neg+=bins[j].count_neg;
    bins.erase(bins.begin()+static_cast<std::ptrdiff_t>(j));
  }

  std::string edge_to_str(double val) const {
    if(std::isinf(val)) {
      return val<0?"-Inf":"+Inf";
    } else {
      std::ostringstream oss;
      oss<<std::fixed<<std::setprecision(6)<<val;
      return oss.str();
    }
  }
};


// [[Rcpp::export]]
List optimal_binning_numerical_jedi(NumericVector target,
                                   NumericVector feature,
                                   int min_bins=3,
                                   int max_bins=5,
                                   double bin_cutoff=0.05,
                                   int max_n_prebins=20,
                                   double convergence_threshold=1e-6,
                                   int max_iterations=1000) {
 if(feature.size()!=target.size()) {
   stop("Feature and target must have the same length.");
 }

 // Missing feature values (NA / NaN) are excluded, as the R wrapper documents;
 // the remaining values are split by class.
 const R_xlen_t n = feature.size();
 const double* fp = feature.begin();
 const double* tp = target.begin();
 size_t n_pos = 0, n_neg = 0;
 bool has_inf = false, bad_target = false;
 for (R_xlen_t i = 0; i < n; ++i) {
   const double f = fp[i];
   if (std::isnan(f)) continue;
   const double t = tp[i];
   if (t == 1.0) ++n_pos;
   else if (t == 0.0) ++n_neg;
   else { bad_target = true; break; }
   if (std::isinf(f)) has_inf = true;
 }
 std::vector<double> pos, neg;
 if (!bad_target && !has_inf) {
   pos.reserve(n_pos);
   neg.reserve(n_neg);
   for (R_xlen_t i = 0; i < n; ++i) {
     const double f = fp[i];
     if (std::isnan(f)) continue;
     if (tp[i] == 1.0) pos.push_back(f);
     else neg.push_back(f);
   }
 }

 try {
   if (bad_target)
     throw std::invalid_argument("Target must contain only 0 and 1.");
   if (has_inf)
     throw std::invalid_argument("Feature contains Inf.");
   if (pos.empty() && neg.empty())
     throw std::invalid_argument("Feature has no non-missing values.");
   OBN_Jedi model(std::move(pos), std::move(neg), min_bins, max_bins,
                  bin_cutoff, max_n_prebins,
                  convergence_threshold, max_iterations);
   model.fit();
   return model.create_output();
 } catch(const std::exception &e) {
   stop("Error in optimal_binning_numerical_jedi: "+std::string(e.what()));
 }
}
