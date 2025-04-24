// [[Rcpp::depends(Rcpp)]]
#include <Rcpp.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <memory>

using namespace Rcpp;

/********************************************************************
 *  Categorical Optimal Binning with Fisher’s Exact Test (v2)
 *  Author : OB – 2025‑04‑17
 *  Licence: MIT
 *
 *  Key changes relative to v1
 *    • Merge‑selection criterion now uses the HIGHEST p‑value
 *    • WoE‑monotonicity checked after *every* merge
 *    • Safer factorial cache & NA handling
 ********************************************************************/

// -----------------------------------------------------------------
// Constants
// -----------------------------------------------------------------
constexpr double EPSILON        = 1e-10;
constexpr double MAX_LOG_VALUE  = 700.0;   // Prevent exp overflow

// -----------------------------------------------------------------
// Bin information container
// -----------------------------------------------------------------
struct BinInfo {
  std::vector<std::string> categories;
  size_t count      = 0;
  size_t count_pos  = 0;
  size_t count_neg  = 0;
  double woe        = 0.0;
  double iv         = 0.0;
  
  BinInfo() { categories.reserve(8); }
  
  inline void add_category(const std::string& cat,
                           size_t cat_count,
                           size_t cat_pos) {
    categories.push_back(cat);
    count      += cat_count;
    count_pos  += cat_pos;
    count_neg  += (cat_count - cat_pos);
  }
  
  inline void merge_with(const BinInfo& other) {
    categories.reserve(categories.size() + other.categories.size());
    categories.insert(categories.end(),
                      other.categories.begin(), other.categories.end());
    count     += other.count;
    count_pos += other.count_pos;
    count_neg += other.count_neg;
  }
};

// -----------------------------------------------------------------
// Fisher’s Exact Test cache (unordered_map keyed by 4‑tuple)
// -----------------------------------------------------------------
class FisherTestCache {
private:
  struct Key {
    size_t a, b, c, d;
    bool operator==(const Key& o) const {
      return a == o.a && b == o.b && c == o.c && d == o.d;
    }
  };
  struct KeyHash {
    size_t operator()(const Key& k) const {
      size_t h1 = k.a ^ (k.b << 16) ^ (k.b >> 16);
      size_t h2 = k.c ^ (k.d << 16) ^ (k.d >> 16);
      return h1 ^ (h2 << 16) ^ (h2 >> 16);
    }
  };
  std::unordered_map<Key, double, KeyHash> cache;
  
public:
  FisherTestCache() { cache.reserve(1024); }
  
  inline double get(size_t a, size_t b, size_t c, size_t d) {
    auto it = cache.find({a, b, c, d});
    return (it == cache.end()) ? -1.0 : it->second;
  }
  inline void set(size_t a, size_t b, size_t c, size_t d, double v) {
    cache[{a, b, c, d}] = v;
  }
};

// -----------------------------------------------------------------
// Main class
// -----------------------------------------------------------------
class OptimalBinningCategoricalFETB {
private:
  // ---------------- Input & hyper‑parameters --------------------
  const std::vector<std::string>& feature;
  const std::vector<int>&         target;
  const size_t  min_bins;
  size_t        max_bins;
  const double  bin_cutoff;
  const size_t  max_n_prebins;
  const double  convergence_threshold;
  const size_t  max_iterations;
  const std::string bin_separator;
  
  // ---------------- Internals -----------------------------------
  std::unordered_map<std::string, size_t> category_counts;
  std::unordered_map<std::string, size_t> category_pos_counts;
  std::vector<BinInfo> bins;
  size_t total_pos = 0, total_neg = 0;
  
  std::vector<double> log_factorials;
  FisherTestCache     fisher_cache;
  
  bool   converged  = false;
  size_t iterations = 0;
  
  // ------------ Utility: extend factorial cache -----------------
  inline void extendLogFactorials(size_t n) {
    size_t old = log_factorials.size();
    log_factorials.resize(n + 1);
    for (size_t i = old; i <= n; ++i)
      log_factorials[i] = log_factorials[i - 1] + std::log(static_cast<double>(i));
  }
  
  // ------------ Fisher one‑step probability ---------------------
  inline double fisherExactProb(size_t a, size_t b, size_t c, size_t d) {
    // Check cache
    double cached = fisher_cache.get(a, b, c, d);
    if (cached >= 0.0) return cached;
    
    size_t n = a + b + c + d;
    if (n >= log_factorials.size()) extendLogFactorials(n);
    
    double lp = log_factorials[a + b] + log_factorials[c + d] +
      log_factorials[a + c] + log_factorials[b + d] -
      log_factorials[n]     - log_factorials[a]     -
      log_factorials[b]     - log_factorials[c]     -
      log_factorials[d];
    
    lp = std::min(std::max(lp, -MAX_LOG_VALUE), MAX_LOG_VALUE);
    double p = std::exp(lp);
    fisher_cache.set(a, b, c, d, p);
    return p;
  }
  
  // ------------ WoE & IV ----------------------------------------
  inline void computeWoeIv(BinInfo& bin) const {
    double pos = static_cast<double>(bin.count_pos);
    double neg = static_cast<double>(bin.count_neg);
    if (pos <= EPSILON || neg <= EPSILON) { bin.woe = bin.iv = 0.0; return; }
    
    double dist_pos = pos / static_cast<double>(total_pos);
    double dist_neg = neg / static_cast<double>(total_neg);
    bin.woe = std::log(dist_pos / dist_neg);
    bin.iv  = (dist_pos - dist_neg) * bin.woe;
  }
  
  // ------------ Pre‑processing ----------------------------------
  void preprocess() {
    category_counts.reserve(std::min(feature.size() / 4, size_t(1024)));
    category_pos_counts.reserve(category_counts.bucket_count());
    
    for (size_t i = 0; i < feature.size(); ++i) {
      const std::string& cat = feature[i];
      category_counts[cat]++;
      if (target[i] == 1) { category_pos_counts[cat]++; total_pos++; }
      else                 { total_neg++; }
    }
  }
  
  // ------------ Initialise bins (rare categories together) ------
  void initialiseBins() {
    bins.clear();
    const double cutoff_cnt = bin_cutoff * static_cast<double>(feature.size());
    
    std::vector<std::pair<std::string,size_t>> sorted;
    sorted.reserve(category_counts.size());
    for (auto& kv : category_counts) sorted.emplace_back(kv.first, kv.second);
    std::sort(sorted.begin(), sorted.end(),
              [](auto& a, auto& b){ return a.second > b.second; });
    
    BinInfo rare;
    bool has_rare=false;
    
    for (auto& kv : sorted) {
      const std::string& cat = kv.first;
      size_t cnt  = kv.second;
      size_t cntp = category_pos_counts[cat];
      
      if (cnt < cutoff_cnt) { rare.add_category(cat,cnt,cntp); has_rare=true; }
      else {
        BinInfo bin; bin.add_category(cat,cnt,cntp); computeWoeIv(bin);
        bins.push_back(std::move(bin));
      }
    }
    if (has_rare) { computeWoeIv(rare); bins.push_back(std::move(rare)); }
    
    std::sort(bins.begin(), bins.end(),
              [](const BinInfo& a, const BinInfo& b){ return a.woe < b.woe; });
  }
  
  // ------------ Merge two bins ----------------------------------
  inline void mergeBins(size_t i) {
    // merge bins[i] with bins[i+1]
    if (i+1 >= bins.size()) return;
    bins[i].merge_with(bins[i+1]);
    computeWoeIv(bins[i]);
    bins.erase(bins.begin()+i+1);
  }
  
  // ------------ Local monotonicity fix --------------------------
  void enforceLocalMonotonicity(size_t start_idx=0) {
    if (bins.size() <= 1) return;
    for (size_t i = start_idx; i+1 < bins.size(); ++i) {
      if (bins[i].woe > bins[i+1].woe + EPSILON) {
        mergeBins(i); i = (i==0) ? size_t(-1) : i-1; // re‑check backward
      }
    }
  }
  
  // ------------ Core merge loop ---------------------------------
  void mergeLoop() {
    iterations = 0;
    double prev_iv = -1.0;
    
    while (bins.size() > max_bins && iterations < max_iterations) {
      // choose pair with HIGHEST p-value (= most similar)
      double best_p  = -1.0;
      size_t best_i  = 0;
      
      for (size_t i = 0; i+1 < bins.size(); ++i) {
        double p = fisherExactProb(bins[i].count_pos, bins[i].count_neg,
                                   bins[i+1].count_pos, bins[i+1].count_neg);
        if (p > best_p) { best_p = p; best_i = i; }
      }
      mergeBins(best_i);
      enforceLocalMonotonicity(best_i == 0 ? 0 : best_i-1);
      
      // convergence (IV change)
      double iv_total = 0.0;
      for (const auto& b : bins) iv_total += b.iv;
      if (std::fabs(iv_total - prev_iv) < convergence_threshold) {
        converged = true; break;
      }
      prev_iv = iv_total;
      ++iterations;
    }
  }
  
  // ------------ Join categories for R output --------------------
  std::string joinCats(const std::vector<std::string>& cats) const {
    if (cats.empty()) return "";
    if (cats.size() == 1) return cats[0];
    size_t len=0; for (auto& c:cats) len+=c.size();
    len += bin_separator.size() * (cats.size()-1);
    std::string out; out.reserve(len);
    out = cats[0];
    for (size_t i=1;i<cats.size();++i){ out+=bin_separator; out+=cats[i]; }
    return out;
  }
  
public:
  OptimalBinningCategoricalFETB(const std::vector<std::string>& feature_,
                                const std::vector<int>&         target_,
                                size_t  min_bins_=3,
                                size_t  max_bins_=5,
                                double  bin_cutoff_=0.05,
                                size_t  max_n_prebins_=20,
                                double  convergence_threshold_=1e-6,
                                size_t  max_iterations_=1000,
                                const std::string& bin_sep="%;%")
    : feature(feature_), target(target_),
      min_bins(min_bins_), max_bins(max_bins_),
      bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_),
      convergence_threshold(convergence_threshold_),
      max_iterations(max_iterations_),
      bin_separator(bin_sep)
  {
    // quick input sanity
    if (feature.empty() || target.empty())
      throw std::invalid_argument("Feature/target cannot be empty.");
    if (feature.size()!=target.size())
      throw std::invalid_argument("Feature and target must match in length.");
    if (min_bins<2 || max_bins<min_bins)
      throw std::invalid_argument("Invalid min_bins / max_bins.");
    if (!(bin_cutoff>0.0 && bin_cutoff<1.0))
      throw std::invalid_argument("bin_cutoff must be in (0,1).");
    
    // factorial cache init (0! ... 1000!)
    log_factorials.resize(1001);
    log_factorials[0]=0.0;
    for (size_t i=1;i<log_factorials.size();++i)
      log_factorials[i]=log_factorials[i-1]+std::log(static_cast<double>(i));
  }
  
  // ---------------- Public API ----------------------------------
  void fit() {
    preprocess();
    initialiseBins();
    max_bins = std::min(max_bins, bins.size());
    if (bins.size() > max_bins) mergeLoop();
    enforceLocalMonotonicity();
    converged = converged || (bins.size() <= max_bins);
  }
  
  List results() const {
    size_t n = bins.size();
    NumericVector id(n), w(n), iv(n);
    CharacterVector lbl(n);
    IntegerVector cnt(n), cntp(n), cntn(n);
    
    for (size_t i=0;i<n;++i) {
      id[i]   = i+1;
      lbl[i]  = joinCats(bins[i].categories);
      w[i]    = bins[i].woe;
      iv[i]   = bins[i].iv;
      cnt[i]  = bins[i].count;
      cntp[i] = bins[i].count_pos;
      cntn[i] = bins[i].count_neg;
    }
    return List::create(
      _["id"]=id, _["bin"]=lbl, _["woe"]=w, _["iv"]=iv,
        _["count"]=cnt, _["count_pos"]=cntp, _["count_neg"]=cntn,
          _["converged"]=converged, _["iterations"]=static_cast<int>(iterations)
    );
  }
};

// -----------------------------------------------------------------
// R interface
// -----------------------------------------------------------------

//' @title Categorical Optimal Binning with Fisher’s Exact Test
//' 
//' @description
//' Performs supervised optimal binning of a **categorical** predictor versus a
//' **binary** target by iteratively merging the *most similar* adjacent bins
//' according to Fisher’s Exact Test.  The routine returns monotonic
//' \emph{Weight of Evidence} (WoE) values and the associated
//' \emph{Information Value} (IV), both key metrics in credit‑scoring,
//' churn prediction and other binary‑response models.
//' 
//' @details
//' \strong{Algorithm outline}\cr
//' Let \eqn{X \in \{\mathcal{C}_1,\dots,\mathcal{C}_K\}} be a categorical
//' feature and \eqn{Y\in\{0,1\}} the target.  For each category
//' \eqn{\mathcal{C}_k} compute the contingency table
//' \deqn{
//'   \begin{array}{c|cc}
//'      & Y=1 & Y=0 \\ \hline
//'   X=\mathcal{C}_k & a_k & b_k
//'   \end{array}}{}
//' with \eqn{a_k+b_k=n_k}.  Rare categories where
//' \eqn{n_k < \textrm{cutoff}\times N} are grouped into a ``rare'' bin.
//' The remaining categories start as singleton bins ordered by their WoE:
//' \deqn{\mathrm{WoE}_k = \log\left(\frac{a_k/T_1}{b_k/T_0}\right)}{}
//' where \eqn{T_1=\sum_k a_k,\; T_0=\sum_k b_k}.  \cr\cr
//' At every iteration the two adjacent bins \eqn{i,i+1} that maximise the
//' two‑tail Fisher \emph{p‑value}
//' \deqn{p_{i,i+1} = P\!\left(
//'    \begin{array}{c|cc} & Y=1 & Y=0\\\hline
//'    \text{bin }i & a_i & b_i\\
//'    \text{bin }i+1 & a_{i+1}& b_{i+1}
//'    \end{array}
//' \right)}
//' are merged.  The process stops when either
//' \eqn{\#\text{bins}\le\texttt{max\_bins}} or the
//' change in global IV,
//' \deqn{\mathrm{IV}= \sum_{\text{bins}} (\tfrac{a}{T_1}-\tfrac{b}{T_0})
//'                       \log\!\left(\tfrac{a\,T_0}{b\,T_1}\right)}{}
//' is below \code{convergence_threshold}.  After each merge a local
//' \emph{monotonicity enforcement} step guarantees
//' \eqn{\mathrm{WoE}_1\le\cdots\le\mathrm{WoE}_m} (or the reverse).
//' 
//' \strong{Complexity}\cr
//' \itemize{
//'   \item Counting pass: \eqn{O(N)} time and \eqn{O(K)} memory.
//'   \item Merging loop: worst‑case \eqn{O(B^2)} time where
//'         \eqn{B\le K} is the initial number of bins;
//'         in practice \eqn{B\ll N} and the loop is very fast.
//' }
//' Overall complexity is \eqn{O(N + B^2)} time and \eqn{O(K)} memory.
//' 
//' \strong{Statistical background}\cr
//' The use of Fisher’s Exact Test provides an exact
//' significance measure for 2×2 tables, ensuring the merged bins are those
//' whose class proportions do not differ significantly.  Monotone WoE
//' facilitates downstream monotonic logistic regression or scorecard
//' scaling.
//' 
//' @param target \code{integer} vector of 0/1 values (length \eqn{N}).
//' @param feature \code{character} vector of categories (length \eqn{N}).
//' @param min_bins Minimum number of final bins.  Default is \code{3}.
//' @param max_bins Maximum number of final bins.  Default is \code{5}.
//' @param bin_cutoff Relative frequency threshold below which categories
//'        are folded into the rare‑bin (default \code{0.05}).
//' @param max_n_prebins Reserved for future use (ignored internally).
//' @param convergence_threshold Absolute tolerance for the change in total IV
//'        required to declare convergence (default \code{0.0000001}).
//' @param max_iterations Safety cap for merge iterations (default \code{1000}).
//' @param bin_separator String used to concatenate category labels in the output.
//' 
//' @return A \code{list} with components
//' \itemize{
//'   \item \code{id}           – numeric id of each resulting bin
//'   \item \code{bin}          – concatenated category labels
//'   \item \code{woe}, \code{iv} – WoE and IV per bin
//'   \item \code{count}, \code{count_pos}, \code{count_neg} – bin counts
//'   \item \code{converged}    – logical flag
//'   \item \code{iterations}   – number of merge iterations
//' }
//' 
//' @examples
//' \donttest{
//' ## simulated example -------------------------------------------------
//' set.seed(42)
//' n        <- 1000
//' target   <- rbinom(n, 1, 0.3)                 # 30 % positives
//' cats     <- LETTERS[1:6]
//' probs    <- c(0.25, 0.20, 0.18, 0.15, 0.12, 0.10)
//' feature  <- sample(cats, n, TRUE, probs)      # imbalanced categories
//' 
//' res <- optimal_binning_categorical_fetb(
//'   target, feature,
//'   min_bins = 2, max_bins = 4,
//'   bin_cutoff = 0.02, bin_separator = "|"
//' )
//' 
//' str(res)
//' 
//' ## inspect WoE curve
//' plot(res$woe, type = "b", pch = 19,
//'      xlab = "Bin index", ylab = "Weight of Evidence")
//' }
//' 
//' @references
//' Fisher, R. A. (1922). *On the interpretation of \eqn{X^2} from contingency
//'   tables, and the calculation of P*. *Journal of the Royal Statistical
//'   Society*, **85**, 87‑94.\cr
//' Hosmer, D. W., & Lemeshow, S. (2000).
//'   *Applied Logistic Regression* (2nd ed.). Wiley.\cr
//' Navas‑Palencia, G. (2019).
//'   *optbinning: Optimal Binning in Python* – documentation v0.19.\cr
//' Freeman, J. V., & Campbell, M. J. (2007).
//'   *The analysis of categorical data: Fisher’s exact test*. *Significance*.\cr
//' Siddiqi, N. (2012).
//'   *Credit Risk Scorecards: Developing and Implementing Intelligent Credit
//'   Scoring*. Wiley.
//' 
//' @author Lopes, J. E.
//' 
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_fetb(
   Rcpp::IntegerVector   target,
   Rcpp::CharacterVector feature,
   int    min_bins              = 3,
   int    max_bins              = 5,
   double bin_cutoff            = 0.05,
   int    max_n_prebins         = 20,
   double convergence_threshold = 1e-6,
   int    max_iterations        = 1000,
   std::string bin_separator    = "%;%")
{
 // ----------- Fast early checks --------------------------------
 if (feature.size() == 0 || target.size() == 0)
   Rcpp::stop("Feature and target cannot be empty.");
 if (feature.size() != target.size())
   Rcpp::stop("Feature and target must have the same length.");
 
 // ----------- Convert to STL containers ------------------------
 std::vector<std::string> feat; feat.reserve(feature.size());
 std::vector<int>         tar;  tar.reserve(target.size());
 
 for (R_xlen_t i = 0; i < feature.size(); ++i) {
   feat.push_back( (feature[i]==NA_STRING) ? "__NA__"
                     : Rcpp::as<std::string>(feature[i]) );
   
   if (IntegerVector::is_na(target[i]))
     Rcpp::stop("Target cannot contain NA.");
   tar.push_back(target[i]);
 }
 
 // ----------- Run algorithm ------------------------------------
 try {
   OptimalBinningCategoricalFETB ob(
       feat, tar,
       static_cast<size_t>(min_bins),
       static_cast<size_t>(max_bins),
       bin_cutoff,
       static_cast<size_t>(max_n_prebins),
       convergence_threshold,
       static_cast<size_t>(max_iterations),
       bin_separator
   );
   ob.fit();
   return ob.results();
 } catch (const std::exception& e) {
   Rcpp::stop("Optimal binning failed: %s", e.what());
 }
}



// // [[Rcpp::depends(Rcpp)]]
// 
// #include <Rcpp.h>
// #include <vector>
// #include <string>
// #include <algorithm>
// #include <cmath>
// #include <unordered_map>
// #include <unordered_set>
// #include <limits>
// #include <sstream>
// #include <stdexcept>
// #include <memory>
// 
// using namespace Rcpp;
// 
// // Constantes
// constexpr double EPSILON = 1e-10;
// constexpr double MAX_LOG_VALUE = 700.0; // Limite para evitar overflow em exp()
// 
// // Estrutura para armazenar informações do bin - otimizada
// struct BinInfo {
//   std::vector<std::string> categories; // Mudança de unordered_set para vector (mais eficiente)
//   size_t count = 0;
//   size_t count_pos = 0;
//   size_t count_neg = 0;
//   double woe = 0.0;
//   double iv = 0.0;
//   
//   // Constructor com pré-alocação
//   BinInfo() {
//     categories.reserve(8); // Estimativa inicial
//   }
//   
//   // Método para adicionar categoria e suas estatísticas
//   inline void add_category(const std::string& category, size_t cat_count, size_t cat_pos) {
//     categories.push_back(category);
//     count += cat_count;
//     count_pos += cat_pos;
//     count_neg += (cat_count - cat_pos);
//   }
//   
//   // Método eficiente para mesclar com outro bin
//   inline void merge_with(const BinInfo& other) {
//     // Reserva espaço para evitar múltiplas realocações
//     categories.reserve(categories.size() + other.categories.size());
//     categories.insert(categories.end(), other.categories.begin(), other.categories.end());
//     count += other.count;
//     count_pos += other.count_pos;
//     count_neg += other.count_neg;
//   }
// };
// 
// // Cache otimizado para resultados do Teste Exato de Fisher
// class FisherTestCache {
// private:
//   // Chave otimizada para 4 valores
//   struct FisherCacheKey {
//     size_t a, b, c, d;
//     
//     FisherCacheKey(size_t a_, size_t b_, size_t c_, size_t d_) 
//       : a(a_), b(b_), c(c_), d(d_) {}
//     
//     bool operator==(const FisherCacheKey& other) const {
//       return a == other.a && b == other.b && c == other.c && d == other.d;
//     }
//   };
//   
//   // Hash personalizado para FisherCacheKey
//   struct FisherCacheKeyHash {
//     size_t operator()(const FisherCacheKey& key) const {
//       // Combina 4 valores usando XOR e shifts
//       size_t h1 = key.a ^ (key.b << 16) ^ (key.b >> 16);
//       size_t h2 = key.c ^ (key.d << 16) ^ (key.d >> 16);
//       return h1 ^ (h2 << 16) ^ (h2 >> 16);
//     }
//   };
//   
//   std::unordered_map<FisherCacheKey, double, FisherCacheKeyHash> cache;
//   
// public:
//   FisherTestCache() {
//     cache.reserve(1024); // Reservar espaço para reduzir realocações
//   }
//   
//   inline double get(size_t a, size_t b, size_t c, size_t d) {
//     FisherCacheKey key(a, b, c, d);
//     auto it = cache.find(key);
//     return (it != cache.end()) ? it->second : -1.0;
//   }
//   
//   inline void set(size_t a, size_t b, size_t c, size_t d, double value) {
//     cache[FisherCacheKey(a, b, c, d)] = value;
//   }
//   
//   inline void clear() {
//     cache.clear();
//   }
//   
//   inline size_t size() const {
//     return cache.size();
//   }
// };
// 
// // Classe principal otimizada
// class OptimalBinningCategoricalFETB {
// private:
//   const std::vector<std::string>& feature;
//   const std::vector<int>& target;
//   const size_t min_bins;
//   size_t max_bins;
//   const double bin_cutoff;
//   const size_t max_n_prebins;
//   const double convergence_threshold;
//   const size_t max_iterations;
//   std::string bin_separator;
//   
//   // Estruturas de dados otimizadas
//   std::unordered_map<std::string, size_t> category_counts;
//   std::unordered_map<std::string, size_t> category_pos_counts;
//   std::vector<BinInfo> bins;
//   size_t total_pos = 0;
//   size_t total_neg = 0;
//   
//   std::vector<double> log_factorials;
//   FisherTestCache fisher_cache;
//   
//   bool converged = false;
//   size_t iterations = 0;
//   
//   // Cálculo otimizado de WoE e IV
//   inline void calculateWoeIv(BinInfo& bin) const {
//     // Evita log(0) e divisão por zero
//     double pos = static_cast<double>(bin.count_pos);
//     double neg = static_cast<double>(bin.count_neg);
//     double tpos = static_cast<double>(total_pos);
//     double tneg = static_cast<double>(total_neg);
//     
//     if (pos <= EPSILON || neg <= EPSILON) {
//       bin.woe = 0.0;
//       bin.iv = 0.0;
//       return;
//     }
//     
//     double dist_pos = pos / tpos;
//     double dist_neg = neg / tneg;
//     
//     bin.woe = std::log(dist_pos / dist_neg);
//     bin.iv = (dist_pos - dist_neg) * bin.woe;
//   }
//   
//   // Teste exato de Fisher otimizado com cache e pré-cálculos
//   inline double fisherExactTest(size_t a, size_t b, size_t c, size_t d) {
//     // Verificar cache primeiro
//     double cached_result = fisher_cache.get(a, b, c, d);
//     if (cached_result >= 0.0) {
//       return cached_result;
//     }
//     
//     // Caso especial: se algum valor for zero, p-valor será alto
//     if (a == 0 || b == 0 || c == 0 || d == 0) {
//       fisher_cache.set(a, b, c, d, 1.0);
//       return 1.0;
//     }
//     
//     size_t n = a + b + c + d;
//     size_t row1 = a + b;
//     size_t row2 = c + d;
//     size_t col1 = a + c;
//     size_t col2 = b + d;
//     
//     // Expandir log_factorials se necessário
//     if (n >= log_factorials.size()) {
//       extendLogFactorials(n);
//     }
//     
//     // Cálculo eficiente usando log-fatoriais pré-calculados
//     double log_p = log_factorials[row1] + log_factorials[row2] + 
//       log_factorials[col1] + log_factorials[col2] -
//       log_factorials[n] - log_factorials[a] - 
//       log_factorials[b] - log_factorials[c] - log_factorials[d];
//     
//     // Limita log_p para evitar overflow/underflow
//     log_p = std::min(std::max(log_p, -MAX_LOG_VALUE), MAX_LOG_VALUE);
//     
//     double result = std::exp(log_p);
//     fisher_cache.set(a, b, c, d, result);
//     
//     return result;
//   }
//   
//   // Expandir log_factorials conforme necessário
//   inline void extendLogFactorials(size_t n) {
//     size_t old_size = log_factorials.size();
//     log_factorials.resize(n + 1);
//     
//     for (size_t i = old_size; i <= n; ++i) {
//       log_factorials[i] = log_factorials[i - 1] + std::log(static_cast<double>(i));
//     }
//   }
//   
//   // Pré-processamento otimizado
//   void preprocessData() {
//     // Contagens em uma única passagem
//     category_counts.clear();
//     category_pos_counts.clear();
//     total_pos = 0;
//     total_neg = 0;
//     
//     // Estima tamanho para os mapas
//     size_t estimated_categories = std::min(feature.size() / 4, static_cast<size_t>(1024));
//     category_counts.reserve(estimated_categories);
//     category_pos_counts.reserve(estimated_categories);
//     
//     for (size_t i = 0; i < feature.size(); ++i) {
//       const std::string& cat = feature[i];
//       category_counts[cat]++;
//       
//       if (target[i] == 1) {
//         category_pos_counts[cat]++;
//         total_pos++;
//       } else {
//         total_neg++;
//       }
//     }
//   }
//   
//   // Inicialização de bins otimizada
//   void initializeBins() {
//     bins.clear();
//     
//     // Verifica categorias raras e agrupa conforme bin_cutoff
//     double cutoff_count = bin_cutoff * static_cast<double>(feature.size());
//     
//     // Organiza categorias por contagem
//     std::vector<std::pair<std::string, size_t>> sorted_categories;
//     sorted_categories.reserve(category_counts.size());
//     
//     for (const auto& [cat, count] : category_counts) {
//       sorted_categories.emplace_back(cat, count);
//     }
//     
//     std::sort(sorted_categories.begin(), sorted_categories.end(),
//               [](const auto& a, const auto& b) {
//                 return a.second > b.second; // Ordem decrescente
//               });
//     
//     // Cria bins iniciais
//     BinInfo rare_bin;
//     bool has_rare = false;
//     
//     for (const auto& [cat, count] : sorted_categories) {
//       if (count < cutoff_count) {
//         // Categoria rara
//         size_t pos_count = category_pos_counts[cat];
//         rare_bin.add_category(cat, count, pos_count);
//         has_rare = true;
//       } else {
//         // Categoria normal
//         BinInfo bin;
//         size_t pos_count = category_pos_counts[cat];
//         bin.add_category(cat, count, pos_count);
//         calculateWoeIv(bin);
//         bins.push_back(std::move(bin));
//       }
//     }
//     
//     // Adiciona bin de categorias raras se existir
//     if (has_rare) {
//       calculateWoeIv(rare_bin);
//       bins.push_back(std::move(rare_bin));
//     }
//     
//     // Ordena bins por WoE para facilitar monotonicidade
//     std::sort(bins.begin(), bins.end(), [](const BinInfo& a, const BinInfo& b) {
//       return a.woe < b.woe;
//     });
//   }
//   
//   // Mesclagem eficiente de dois bins
//   inline void mergeTwoBins(size_t index1, size_t index2) {
//     if (index1 >= bins.size() || index2 >= bins.size() || index1 == index2) return;
//     if (index2 < index1) std::swap(index1, index2);
//     
//     BinInfo& bin1 = bins[index1];
//     const BinInfo& bin2 = bins[index2];
//     
//     bin1.merge_with(bin2);
//     calculateWoeIv(bin1);
//     
//     bins.erase(bins.begin() + index2);
//   }
//   
//   // Método de mesclagem de bins para enforçar monotonicidade
//   void mergeBinsForMonotonicity() {
//     if (bins.size() <= min_bins) return;
//     
//     bool any_merge;
//     size_t safety_counter = 0;
//     const size_t max_passes = bins.size() * 3;
//     
//     do {
//       any_merge = false;
//       
//       for (size_t i = 0; i + 1 < bins.size(); ++i) {
//         if (bins[i].woe > bins[i + 1].woe + EPSILON) {
//           mergeTwoBins(i, i + 1);
//           any_merge = true;
//           break;
//         }
//       }
//       
//       safety_counter++;
//     } while (any_merge && bins.size() > min_bins && safety_counter < max_passes);
//   }
//   
//   // Construção eficiente de string de categorias
//   std::string joinCategories(const std::vector<std::string>& categories) const {
//     if (categories.empty()) return "";
//     if (categories.size() == 1) return categories[0];
//     
//     // Estima tamanho total
//     size_t total_length = 0;
//     for (const auto& cat : categories) {
//       total_length += cat.length();
//     }
//     total_length += bin_separator.length() * (categories.size() - 1);
//     
//     // Pré-aloca e constrói
//     std::string result;
//     result.reserve(total_length);
//     
//     result = categories[0];
//     for (size_t i = 1; i < categories.size(); ++i) {
//       result += bin_separator;
//       result += categories[i];
//     }
//     
//     return result;
//   }
//   
// public:
//   OptimalBinningCategoricalFETB(const std::vector<std::string>& feature,
//                                 const std::vector<int>& target,
//                                 size_t min_bins = 3,
//                                 size_t max_bins = 5,
//                                 double bin_cutoff = 0.05,
//                                 size_t max_n_prebins = 20,
//                                 double convergence_threshold = 1e-6,
//                                 size_t max_iterations = 1000,
//                                 const std::string& bin_separator_input = "%;%")
//     : feature(feature), target(target),
//       min_bins(min_bins), max_bins(max_bins),
//       bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins),
//       convergence_threshold(convergence_threshold), max_iterations(max_iterations),
//       bin_separator(bin_separator_input) {
//     
//     validateInput();
//     
//     // Inicializa log-factoriais com tamanho inicial
//     size_t initial_size = std::min(feature.size(), static_cast<size_t>(1000));
//     log_factorials.resize(initial_size + 1);
//     log_factorials[0] = 0.0;
//     
//     for (size_t i = 1; i <= initial_size; ++i) {
//       log_factorials[i] = log_factorials[i - 1] + std::log(static_cast<double>(i));
//     }
//   }
//   
//   // Validação otimizada de entradas
//   void validateInput() const {
//     if (feature.empty()) {
//       throw std::invalid_argument("Feature não pode ser vazio.");
//     }
//     
//     if (target.empty()) {
//       throw std::invalid_argument("Target não pode ser vazio.");
//     }
//     
//     if (feature.size() != target.size()) {
//       throw std::invalid_argument("Feature e target devem ter o mesmo tamanho.");
//     }
//     
//     if (min_bins < 2) {
//       throw std::invalid_argument("min_bins deve ser >= 2.");
//     }
//     
//     if (max_bins < min_bins) {
//       throw std::invalid_argument("max_bins deve ser >= min_bins.");
//     }
//     
//     if (bin_cutoff <= 0.0 || bin_cutoff >= 1.0) {
//       throw std::invalid_argument("bin_cutoff deve estar entre 0 e 1 (exclusivo).");
//     }
//     
//     if (max_n_prebins < min_bins) {
//       throw std::invalid_argument("max_n_prebins deve ser >= min_bins.");
//     }
//     
//     // Verificação eficiente de target binário
//     bool has_zero = false;
//     bool has_one = false;
//     
//     for (int t : target) {
//       if (t == 0) has_zero = true;
//       else if (t == 1) has_one = true;
//       else throw std::invalid_argument("Target deve conter apenas 0 e 1.");
//       
//       if (has_zero && has_one) break;
//     }
//     
//     if (!has_zero || !has_one) {
//       throw std::invalid_argument("Target deve conter tanto 0 quanto 1.");
//     }
//   }
//   
//   // Método principal de treinamento - otimizado
//   void fit() {
//     // Pré-processamento de dados
//     preprocessData();
//     
//     // Inicializa estruturas e bins
//     initializeBins();
//     
//     // Ajusta max_bins baseado em categorias disponíveis
//     max_bins = std::min(max_bins, bins.size());
//     
//     // Se já estamos dentro do limite de bins, apenas impõe monotonicidade
//     if (bins.size() <= max_bins) {
//       mergeBinsForMonotonicity();
//       converged = true;
//       return;
//     }
//     
//     double prev_total_iv = 0.0;
//     iterations = 0;
//     
//     // Loop principal de mesclagem com teste exato de Fisher
//     while (bins.size() > max_bins && iterations < max_iterations) {
//       double min_p_value = std::numeric_limits<double>::max();
//       size_t min_index = 0;
//       
//       // Encontra o par de bins com menor p-valor (mais similar)
//       for (size_t i = 0; i < bins.size() - 1; ++i) {
//         size_t a = bins[i].count_pos;
//         size_t b = bins[i].count_neg;
//         size_t c = bins[i + 1].count_pos;
//         size_t d = bins[i + 1].count_neg;
//         
//         double p_value = fisherExactTest(a, b, c, d);
//         
//         if (p_value < min_p_value) {
//           min_p_value = p_value;
//           min_index = i;
//         }
//       }
//       
//       // Mescla os bins mais similares
//       mergeTwoBins(min_index, min_index + 1);
//       
//       // Verifica convergência
//       double total_iv = 0.0;
//       for (const auto& bin : bins) {
//         total_iv += bin.iv;
//       }
//       
//       if (std::fabs(total_iv - prev_total_iv) < convergence_threshold) {
//         converged = true;
//         break;
//       }
//       
//       prev_total_iv = total_iv;
//       iterations++;
//     }
//     
//     // Impõe monotonicidade após mesclagem
//     mergeBinsForMonotonicity();
//   }
//   
//   // Resultados otimizados
//   Rcpp::List getResults() const {
//     const size_t n_bins = bins.size();
//     
//     // Pré-alocação de vetores de resultado
//     Rcpp::NumericVector ids(n_bins);
//     Rcpp::CharacterVector bin_labels(n_bins);
//     Rcpp::NumericVector woe_values(n_bins);
//     Rcpp::NumericVector iv_values(n_bins);
//     Rcpp::IntegerVector counts(n_bins);
//     Rcpp::IntegerVector counts_pos(n_bins);
//     Rcpp::IntegerVector counts_neg(n_bins);
//     
//     for (size_t i = 0; i < n_bins; ++i) {
//       ids[i] = i + 1;
//       bin_labels[i] = joinCategories(bins[i].categories);
//       woe_values[i] = bins[i].woe;
//       iv_values[i] = bins[i].iv;
//       counts[i] = static_cast<int>(bins[i].count);
//       counts_pos[i] = static_cast<int>(bins[i].count_pos);
//       counts_neg[i] = static_cast<int>(bins[i].count_neg);
//     }
//     
//     return Rcpp::List::create(
//       Rcpp::Named("id") = ids,
//       Rcpp::Named("bin") = bin_labels,
//       Rcpp::Named("woe") = woe_values,
//       Rcpp::Named("iv") = iv_values,
//       Rcpp::Named("count") = counts,
//       Rcpp::Named("count_pos") = counts_pos,
//       Rcpp::Named("count_neg") = counts_neg,
//       Rcpp::Named("converged") = converged,
//       Rcpp::Named("iterations") = static_cast<int>(iterations)
//     );
//   }
// };
// 
// 
// //' @title Categorical Optimal Binning with Fisher's Exact Test
// //'
// //' @description
// //' Implements optimal binning for categorical variables using Fisher's Exact Test,
// //' calculating Weight of Evidence (WoE) and Information Value (IV).
// //'
// //' @param target Integer vector of binary target values (0 or 1).
// //' @param feature Character vector of categorical feature values.
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff Minimum frequency for a separate bin (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins before merging (default: 20).
// //' @param convergence_threshold Threshold for convergence (default: 1e-6).
// //' @param max_iterations Maximum number of iterations (default: 1000).
// //' @param bin_separator Separator for bin labels (default: "%;%").
// //'
// //' @return A list containing:
// //' \itemize{
// //'   \item bin: Character vector of bin labels (merged categories).
// //'   \item woe: Numeric vector of Weight of Evidence values for each bin.
// //'   \item iv: Numeric vector of Information Value for each bin.
// //'   \item count: Integer vector of total count in each bin.
// //'   \item count_pos: Integer vector of positive class count in each bin.
// //'   \item count_neg: Integer vector of negative class count in each bin.
// //'   \item converged: Logical indicating whether the algorithm converged.
// //'   \item iterations: Integer indicating the number of iterations performed.
// //' }
// //'
// //' @details
// //' The algorithm uses Fisher's Exact Test to iteratively merge bins, maximizing
// //' the statistical significance of the difference between adjacent bins. It ensures
// //' monotonicity in the resulting bins and respects the minimum number of bins specified.
// //'
// //' @examples
// //' \dontrun{
// //' target <- c(1, 0, 1, 1, 0, 1, 0, 0, 1, 1)
// //' feature <- c("A", "B", "A", "C", "B", "D", "C", "A", "D", "B")
// //' result <- optimal_binning_categorical_fetb(target, feature, min_bins = 2,
// //' max_bins = 4, bin_separator = "|")
// //' print(result)
// //' }
// //'
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_categorical_fetb(
//    Rcpp::IntegerVector target,
//    Rcpp::CharacterVector feature,
//    int min_bins = 3,
//    int max_bins = 5,
//    double bin_cutoff = 0.05,
//    int max_n_prebins = 20,
//    double convergence_threshold = 1e-6,
//    int max_iterations = 1000,
//    std::string bin_separator = "%;%"
// ) {
//  // Verificações rápidas antes de converter vetores
//  if (feature.size() == 0 || target.size() == 0) {
//    Rcpp::stop("Feature e target não podem ser vazios.");
//  }
//  
//  if (feature.size() != target.size()) {
//    Rcpp::stop("Feature e target devem ter o mesmo tamanho.");
//  }
//  
//  // Convertendo vetores R para C++
//  std::vector<std::string> feature_vec;
//  std::vector<int> target_vec;
//  
//  feature_vec.reserve(feature.size());
//  target_vec.reserve(target.size());
//  
//  for (R_xlen_t i = 0; i < feature.size(); ++i) {
//    // Tratamento de NA
//    if (feature[i] == NA_STRING) {
//      feature_vec.push_back("NA");
//    } else {
//      feature_vec.push_back(Rcpp::as<std::string>(feature[i]));
//    }
//    
//    // Verificação de NA em target
//    if (IntegerVector::is_na(target[i])) {
//      Rcpp::stop("Target não pode conter valores ausentes.");
//    } else {
//      target_vec.push_back(target[i]);
//    }
//  }
//  
//  try {
//    OptimalBinningCategoricalFETB binner(
//        feature_vec, target_vec,
//        static_cast<size_t>(min_bins),
//        static_cast<size_t>(max_bins),
//        bin_cutoff,
//        static_cast<size_t>(max_n_prebins),
//        convergence_threshold,
//        static_cast<size_t>(max_iterations),
//        bin_separator
//    );
//    
//    binner.fit();
//    return binner.getResults();
//  } catch (const std::exception& e) {
//    Rcpp::stop("Erro no binning ótimo: " + std::string(e.what()));
//  }
// }