// [[Rcpp::plugins(cpp11)]]
#include <Rcpp.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
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

// ----------------------------------------------------------------------
// Sort finite doubles ascending (LSD radix sort for large inputs).
//
// Sorting is the dominant cost of the engine: everything after it is O(n) or
// O(bins * classes * log n). Large inputs are sorted on the order-preserving
// integer image of each double (6 passes of 11 bits, histograms gathered in
// one read, constant-digit passes skipped). The order is that of operator< on
// finite values, except that -0.0 precedes +0.0; the two compare equal and are
// merged into one distinct value downstream.
// ----------------------------------------------------------------------
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
    if (h[first] == n) continue;
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

// ----------------------------------------------------------------------
// Estrutura para armazenar informações de um bin (intervalo) em M classes
// ----------------------------------------------------------------------
struct NumBinMulti {
  double lower_bound;                // Limite inferior do bin
  double upper_bound;                // Limite superior do bin
  int total_count;             // Contagem total de observações no bin

  std::vector<int> class_counts;   // Contagem de observações por classe
  std::vector<double> woes;        // M-WOE por classe
  std::vector<double> ivs;         // Contribuição de IV por classe

  NumBinMulti(double l, double u, size_t n_classes)
    : lower_bound(l), upper_bound(u),
      total_count(0),
      class_counts(n_classes, 0),
      woes(n_classes, 0.0),
      ivs(n_classes, 0.0) {}
};

// ----------------------------------------------------------------------
// Classe principal para Binning Numérico com M-WOE (multinomial)
//
// Layout: the feature values are split by class and each class is sorted once.
// The number of observations of class k in a bin (lower, upper] is then
//   upper_bound(vals[k], upper) - upper_bound(vals[k], lower),
// so no step after the sort touches individual observations again.
// ----------------------------------------------------------------------
class OBN_JEDIMWoE {
private:
  // Valores da feature por classe (ordenados)
  std::vector<std::vector<double>> class_vals_;
  size_t n_obs_;

  // Número de classes (0,1,2,...,n_classes_-1)
  size_t n_classes_;

  // Parâmetros
  int min_bins_;
  int max_bins_;
  double bin_cutoff_;
  int max_n_prebins_;
  double convergence_threshold_;
  int max_iterations_;

  // Bins resultantes
  std::vector<NumBinMulti> bins_;

  // Convergência e iterações
  bool converged_;
  int iterations_run_;

  // Contagem total por classe (para cálculo de M-WOE)
  std::vector<int> total_class_counts_;

public:
  // --------------------------------------------------------------------
  // Construtor
  //   class_vals: the feature values of each class 0..K-1 (unsorted; moved in).
  // --------------------------------------------------------------------
  OBN_JEDIMWoE(std::vector<std::vector<double>>&& class_vals,
               int min_b, int max_b,
               double cutoff,
               int max_pre,
               double conv_thr,
               int max_iter)
    : class_vals_(std::move(class_vals)),
      n_obs_(0),
      n_classes_(0),
      min_bins_(std::max(min_b, 2)),
      max_bins_(std::max(max_b, min_b)),
      bin_cutoff_(cutoff),
      max_n_prebins_(std::max(max_pre, min_b)),
      convergence_threshold_(conv_thr),
      max_iterations_(max_iter),
      converged_(false),
      iterations_run_(0)
  {
    n_classes_ = class_vals_.size();
    if(n_classes_ < 2) {
      throw std::invalid_argument("You must have at least 2 distinct classes in the target.");
    }

    if(cutoff <= 0.0 || cutoff >= 1.0)
      throw std::invalid_argument("bin_cutoff must be in (0,1).");
    if(convergence_threshold_ <= 0.0)
      throw std::invalid_argument("convergence_threshold must be positive.");
    if(max_iterations_ <= 0)
      throw std::invalid_argument("max_iterations must be positive.");

    total_class_counts_.resize(n_classes_, 0);
    for(size_t k = 0; k < n_classes_; k++) {
      total_class_counts_[k] = static_cast<int>(class_vals_[k].size());
      n_obs_ += class_vals_[k].size();
      sort_doubles(class_vals_[k]);
    }
  }

  // --------------------------------------------------------------------
  // Método principal de ajuste
  // --------------------------------------------------------------------
  void fit() {
    // 1) Preparar vetor único de valores
    const std::vector<double> unique_vals = distinct_values();

    // 2) Se poucas ou nenhuma variação, trata caso trivial
    if(unique_vals.size() <= 1) {
      handle_single_bin();
      converged_ = true;
      iterations_run_ = 0;
      return;
    }
    if(unique_vals.size() == 2) {
      handle_two_bins(unique_vals);
      converged_ = true;
      iterations_run_ = 0;
      return;
    }

    // 3) Caso geral: criar pre-bins (quantis) + merges
    initial_prebin(unique_vals);
    assign_bins();
    merge_small_bins();
    compute_mwoe_iv();

    // 4) Loop de otimização
    double prev_iv = total_iv();
    for(int iter = 0; iter < max_iterations_; iter++) {
      enforce_monotonicity();
      merge_until_max_bins();
      compute_mwoe_iv();

      double current_iv = total_iv();
      if(std::fabs(current_iv - prev_iv) < convergence_threshold_) {
        converged_ = true;
        iterations_run_ = iter + 1;
        break;
      }
      prev_iv = current_iv;
      iterations_run_ = iter + 1;
    }

    if(!converged_) {
      iterations_run_ = max_iterations_;
    }
  }

  // --------------------------------------------------------------------
  // Cria saída no formato Rcpp::List
  // --------------------------------------------------------------------
  Rcpp::List create_output() const {
    const int n_bins = static_cast<int>(bins_.size());
    const int n_cls = static_cast<int>(n_classes_);
    CharacterVector bin_names(n_bins);
    // woes e ivs em formato (n_bins x n_classes)
    NumericMatrix woes(n_bins, n_cls);
    NumericMatrix ivs(n_bins, n_cls);

    IntegerVector counts(n_bins);
    IntegerMatrix class_counts(n_bins, n_cls);

    NumericVector cutpoints;  // se houver mais de 1 bin, armazenar cutpoints

    if(n_bins > 1) {
      cutpoints = NumericVector(n_bins - 1);
    }

    for(int i = 0; i < n_bins; i++) {
      const NumBinMulti& b = bins_[static_cast<size_t>(i)];
      bin_names[i] = interval_to_string(b.lower_bound, b.upper_bound);
      counts[i]    = b.total_count;

      // Guardar WOE e IV por classe
      for(int k = 0; k < n_cls; k++) {
        const size_t kk = static_cast<size_t>(k);
        woes(i, k) = b.woes[kk];
        ivs(i, k)  = b.ivs[kk];
        class_counts(i, k) = b.class_counts[kk];
      }
      // Gerar cutpoints (exclui último bin, pois é +Inf)
      if(i < n_bins - 1) {
        cutpoints[i] = b.upper_bound;
      }
    }

    // IDs sequenciais
    NumericVector ids(n_bins);
    for(int i = 0; i < n_bins; i++) {
      ids[i] = static_cast<double>(i + 1);
    }

    return Rcpp::List::create(
      Named("id")           = ids,
      Named("bin")          = bin_names,
      Named("woe")          = woes,
      Named("iv")           = ivs,
      Named("count")        = counts,
      Named("class_counts") = class_counts,
      Named("cutpoints")    = cutpoints,
      Named("converged")    = converged_,
      Named("iterations")   = iterations_run_,
      Named("n_classes")    = n_cls
    );
  }

private:
  // Valores distintos ordenados: fusão (merge) dos vetores ordenados por classe.
  std::vector<double> distinct_values() const {
    std::vector<double> u;
    std::vector<size_t> pos(n_classes_, 0);
    for (;;) {
      bool any = false;
      double v = 0.0;
      for (size_t k = 0; k < n_classes_; k++) {
        if (pos[k] < class_vals_[k].size()) {
          const double c = class_vals_[k][pos[k]];
          if (!any || c < v) v = c;
          any = true;
        }
      }
      if (!any) break;
      for (size_t k = 0; k < n_classes_; k++) {
        while (pos[k] < class_vals_[k].size() && class_vals_[k][pos[k]] == v) ++pos[k];
      }
      // -Inf / +Inf stay in the class arrays (first / last bin) but are never
      // a quantile edge; -0.0 == +0.0 is reported unsigned.
      if (std::isfinite(v)) u.push_back(v + 0.0);
    }
    return u;
  }

  // Contagens de um bin a partir do intervalo (lower, upper].
  void count_interval(NumBinMulti& b) const {
    const bool lo_inf = std::isinf(b.lower_bound) && b.lower_bound < 0;
    const bool hi_inf = std::isinf(b.upper_bound) && b.upper_bound > 0;
    b.total_count = 0;
    for (size_t k = 0; k < n_classes_; k++) {
      const std::vector<double>& v = class_vals_[k];
      const auto hi = hi_inf ? v.end() : std::upper_bound(v.begin(), v.end(), b.upper_bound);
      const auto lo = lo_inf ? v.begin() : std::upper_bound(v.begin(), v.end(), b.lower_bound);
      b.class_counts[k] = static_cast<int>(hi - lo);
      b.total_count += b.class_counts[k];
    }
  }

  // --------------------------------------------------------------------
  // Cria um único bin se todos os valores são idênticos
  // --------------------------------------------------------------------
  void handle_single_bin() {
    bins_.clear();
    bins_.emplace_back(-std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::infinity(),
                       n_classes_);
    count_interval(bins_[0]);
    compute_mwoe_iv();
  }

  // --------------------------------------------------------------------
  // Cria dois bins se só existem 2 valores distintos
  // --------------------------------------------------------------------
  void handle_two_bins(const std::vector<double>& unique_vals) {
    bins_.clear();
    double cut = unique_vals[0];

    bins_.emplace_back(-std::numeric_limits<double>::infinity(), cut, n_classes_);
    bins_.emplace_back(cut, std::numeric_limits<double>::infinity(), n_classes_);
    for (auto& b : bins_) count_interval(b);
    compute_mwoe_iv();
  }

  // --------------------------------------------------------------------
  // Pré-binning usando quantis dos valores distintos.
  //
  // With at least min_bins_ distinct values the positions below are all
  // different and give exactly n_pre bins. With fewer, they collide and the
  // edges reduce to every distinct value but the largest: one bin per distinct
  // value, the finest binning the data allow. (Halving intervals further, as
  // this routine used to do, could only create bins with no observation.)
  // --------------------------------------------------------------------
  void initial_prebin(const std::vector<double>& unique_vals) {
    bins_.clear();
    int n_unique = (int)unique_vals.size();
    int n_pre = std::min(max_n_prebins_, n_unique);
    n_pre = std::max(n_pre, min_bins_);

    // Edges iniciais
    std::vector<double> edges;
    edges.push_back(-std::numeric_limits<double>::infinity());

    // Gera pontos de corte baseado em quantil
    for(int i = 1; i < n_pre; i++) {
      double p = (double)i / n_pre;
      int idx  = (int)std::floor(p * (n_unique - 1));
      double edge = unique_vals[static_cast<size_t>(idx)];
      // Evitar duplicação de edges
      if(edge > edges.back()) {
        edges.push_back(edge);
      }
    }
    edges.push_back(std::numeric_limits<double>::infinity());

    // Constrói bins
    for(size_t i = 0; i + 1 < edges.size(); i++) {
      bins_.emplace_back(edges[i], edges[i+1], n_classes_);
    }
  }

  // --------------------------------------------------------------------
  // Atribuir contagens a cada bin
  // --------------------------------------------------------------------
  void assign_bins() {
    for (auto& b : bins_) count_interval(b);
  }

  // --------------------------------------------------------------------
  // Mesclar bins de baixa frequência (bin_cutoff)
  // --------------------------------------------------------------------
  void merge_small_bins() {
    bool merged = true;
    double total = (double)n_obs_;
    while(merged && (int)bins_.size() > min_bins_ && iterations_run_ < max_iterations_) {
      merged = false;
      for(size_t i = 0; i < bins_.size(); i++) {
        double prop = (double)bins_[i].total_count / total;
        if(prop < bin_cutoff_ && (int)bins_.size() > min_bins_) {
          if(i == 0) {
            if(bins_.size() < 2) break;
            merge_two_bins(0, 1);
          } else if(i == bins_.size() - 1) {
            merge_two_bins(bins_.size() - 2, bins_.size() - 1);
          } else {
            // Merge com o vizinho de menor contagem
            if(bins_[i-1].total_count <= bins_[i+1].total_count) {
              merge_two_bins(i-1, i);
            } else {
              merge_two_bins(i, i+1);
            }
          }
          merged = true;
          break;
        }
      }
      iterations_run_++;
    }
  }

  // --------------------------------------------------------------------
  // Cálculo de M-WOE e IV
  //
  //   class_rate_k  = bin.class_counts[k] / total_class_counts[k]
  //   others_rate_k = (bin.total - bin.class_counts[k]) / (N - total_class_counts[k])
  //   mwoe_k = ln( class_rate_k / others_rate_k )   (rates floored at EPS)
  //   iv_k   = (class_rate_k - others_rate_k) * mwoe_k
  //
  // Every class 0..K-1 is present (the interface guarantees it), so both
  // denominators are positive.
  // --------------------------------------------------------------------
  void compute_mwoe_iv() {
    const int n_all = static_cast<int>(n_obs_);
    for(auto &b : bins_) {
      for(size_t k = 0; k < n_classes_; k++) {
        b.woes[k] = 0.0;
        b.ivs[k]  = 0.0;
      }
      if(b.total_count == 0) {
        continue;
      }
      for(size_t k = 0; k < n_classes_; k++) {
        const int sum_others_bin = b.total_count - b.class_counts[k];
        const int sum_others_all = n_all - total_class_counts_[k];

        double class_rate  = (double)b.class_counts[k] / (double)total_class_counts_[k];
        double others_rate = (double)sum_others_bin / (double)sum_others_all;
        double safe_p  = std::max(class_rate, EPS);
        double safe_q  = std::max(others_rate, EPS);

        double woe_k = std::log(safe_p / safe_q);
        double iv_k  = (class_rate - others_rate) * woe_k;

        b.woes[k] = woe_k;
        b.ivs[k]  = iv_k;
      }
    }
  }

  // --------------------------------------------------------------------
  // Soma total do IV (across all classes e bins)
  // --------------------------------------------------------------------
  double total_iv() const {
    double sum_iv = 0.0;
    for(const auto &b : bins_) {
      for(size_t k = 0; k < n_classes_; k++) {
        sum_iv += b.ivs[k];
      }
    }
    return sum_iv;
  }

  // --------------------------------------------------------------------
  // Força monotonicidade para cada classe
  // --------------------------------------------------------------------
  void enforce_monotonicity() {
    bool changed = true;
    int local_iterations = 0;

    while(changed && (int)bins_.size() > min_bins_ && local_iterations < max_iterations_) {
      changed = false;
      for(size_t k = 0; k < n_classes_; k++) {
        bool increasing = guess_trend_for_class(k);
        for(size_t i = 1; i < bins_.size(); i++) {
          if((increasing && (bins_[i].woes[k] < bins_[i-1].woes[k])) ||
             (!increasing && (bins_[i].woes[k] > bins_[i-1].woes[k]))) {
            // viola monotonicidade => mescla
            merge_two_bins(i-1, i);
            compute_mwoe_iv();
            changed = true;
            break;
          }
        }
        if(changed) break; // recomeçar do k=0 após a mescla
      }
      local_iterations++;
    }
  }

  // --------------------------------------------------------------------
  // Descobre se a classe k parece ter WOE crescente ou decrescente
  // --------------------------------------------------------------------
  bool guess_trend_for_class(size_t k) const {
    int inc = 0, dec = 0;
    for(size_t i = 1; i < bins_.size(); i++) {
      if(bins_[i].woes[k] > bins_[i-1].woes[k]) {
        inc++;
      } else if(bins_[i].woes[k] < bins_[i-1].woes[k]) {
        dec++;
      }
    }
    return (inc >= dec);
  }

  // --------------------------------------------------------------------
  // Respeitar max_bins: se houver bins demais, mesclar gradualmente
  // --------------------------------------------------------------------
  void merge_until_max_bins() {
    while((int)bins_.size() > max_bins_ && iterations_run_ < max_iterations_) {
      size_t idx = find_min_iv_merge();
      if(idx >= bins_.size() - 1) break;
      merge_two_bins(idx, idx+1);
      compute_mwoe_iv();
      iterations_run_++;
    }
  }

  // --------------------------------------------------------------------
  // Identifica o par de bins adjacentes cuja soma de IV seja menor
  // --------------------------------------------------------------------
  size_t find_min_iv_merge() const {
    if(bins_.size() < 2) return bins_.size();

    double min_iv_sum = std::numeric_limits<double>::max();
    size_t best_idx = bins_.size();

    for(size_t i = 0; i < bins_.size() - 1; i++) {
      double local_sum = 0.0;
      for(size_t k = 0; k < n_classes_; k++) {
        local_sum += bins_[i].ivs[k];
        local_sum += bins_[i+1].ivs[k];
      }
      if(local_sum < min_iv_sum) {
        min_iv_sum = local_sum;
        best_idx = i;
      }
    }
    return best_idx;
  }

  // --------------------------------------------------------------------
  // Mescla efetivamente dois bins i e j
  // --------------------------------------------------------------------
  void merge_two_bins(size_t i, size_t j) {
    if(i > j) std::swap(i, j);
    if(j >= bins_.size()) return;

    bins_[i].upper_bound = bins_[j].upper_bound;
    bins_[i].total_count += bins_[j].total_count;
    for(size_t k = 0; k < n_classes_; k++) {
      bins_[i].class_counts[k] += bins_[j].class_counts[k];
    }
    bins_.erase(bins_.begin() + static_cast<std::ptrdiff_t>(j));
  }

  // --------------------------------------------------------------------
  // Gera string do tipo (lower; upper]
  // --------------------------------------------------------------------
  std::string interval_to_string(double l, double u) const {
    std::ostringstream oss;
    oss << "(" << edge_to_str(l) << "; " << edge_to_str(u) << "]";
    return oss.str();
  }

  std::string edge_to_str(double val) const {
    if(std::isinf(val)) {
      return (val < 0) ? "-Inf" : "+Inf";
    } else {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(6) << val;
      return oss.str();
    }
  }
};

// ----------------------------------------------------------------------
// Função de interface Rcpp
// ----------------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_jedi_mwoe(
   Rcpp::IntegerVector target,
   Rcpp::NumericVector feature,
   int min_bins = 3,
   int max_bins = 5,
   double bin_cutoff = 0.05,
   int max_n_prebins = 20,
   double convergence_threshold = 1e-6,
   int max_iterations = 1000
) {
 try {
   if(feature.size() != target.size()) {
     throw std::invalid_argument("feature and target must have the same length.");
   }
   const R_xlen_t n = feature.size();

   // Numerical NA contract: rows whose feature is NA / NaN are excluded;
   // -Inf / +Inf are ordinary extreme values (first / last bin, never a
   // cutpoint); a missing target is an error. The classes are the distinct
   // target values of the remaining rows and must be exactly 0..K-1, K >= 2.
   const double* fp = feature.begin();
   const int* tp = target.begin();
   for (R_xlen_t i = 0; i < n; ++i) {
     if (tp[i] == NA_INTEGER)
       throw std::invalid_argument("Target contains missing values (NA).");
   }
   int max_t = -1;
   size_t n_used = 0;
   for (R_xlen_t i = 0; i < n; ++i) {
     if (std::isnan(fp[i])) continue;
     const int t = tp[i];
     if (t < 0)
       throw std::invalid_argument("Target values must be in [0..(n_classes-1)].");
     if (t > max_t) max_t = t;
     ++n_used;
   }
   if (n_used == 0)
     throw std::invalid_argument("Feature has no non-missing values.");
   if (static_cast<size_t>(max_t) >= n_used + 1)
     throw std::invalid_argument("Target values must be in [0..(n_classes-1)].");

   const size_t n_cls = static_cast<size_t>(max_t) + 1;
   std::vector<size_t> cls_n(n_cls, 0);
   for (R_xlen_t i = 0; i < n; ++i) {
     if (!std::isnan(fp[i])) ++cls_n[static_cast<size_t>(tp[i])];
   }
   std::vector<std::vector<double>> class_vals(n_cls);
   for (size_t k = 0; k < n_cls; ++k) class_vals[k].reserve(cls_n[k]);
   for (R_xlen_t i = 0; i < n; ++i) {
     if (std::isnan(fp[i])) continue;
     class_vals[static_cast<size_t>(tp[i])].push_back(fp[i]);
   }
   size_t n_present = 0;
   for (const auto& cv : class_vals) if (!cv.empty()) ++n_present;
   if (n_present < 2)
     throw std::invalid_argument("You must have at least 2 distinct classes in the target.");
   if (n_present != class_vals.size())
     throw std::invalid_argument("Target values must be in [0..(n_classes-1)].");

   OBN_JEDIMWoE model(std::move(class_vals),
                      min_bins, max_bins,
                      bin_cutoff, max_n_prebins,
                      convergence_threshold,
                      max_iterations);
   model.fit();
   return model.create_output();
 } catch(const std::exception &ex) {
   Rcpp::stop("Error in optimal_binning_numerical_jedi_mwoe: " + std::string(ex.what()));
 }
}
