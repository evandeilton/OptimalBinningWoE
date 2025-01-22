// [[Rcpp::plugins(cpp11)]]
#include <Rcpp.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <unordered_set>

using namespace Rcpp;

static constexpr double EPS = 1e-10;

// ----------------------------------------------------------------------
// Estrutura para armazenar informações de um bin (intervalo) em M classes
// ----------------------------------------------------------------------
struct NumBinMulti {
  double lower;                // Limite inferior do bin
  double upper;                // Limite superior do bin
  int total_count;             // Contagem total de observações no bin
  
  std::vector<int> class_counts;   // Contagem de observações por classe
  std::vector<double> woes;        // M-WOE por classe
  std::vector<double> ivs;         // Contribuição de IV por classe
  
  NumBinMulti(double l, double u, size_t n_classes)
    : lower(l), upper(u),
      total_count(0),
      class_counts(n_classes, 0),
      woes(n_classes, 0.0),
      ivs(n_classes, 0.0) {}
};

// ----------------------------------------------------------------------
// Classe principal para Binning Numérico com M-WOE (multinomial)
// ----------------------------------------------------------------------
class OptimalBinningNumericalJEDIMWoE {
private:
  // Dados de entrada
  std::vector<double> feature_;
  std::vector<int> target_;
  
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
  // --------------------------------------------------------------------
  OptimalBinningNumericalJEDIMWoE(const std::vector<double>& feature,
                                  const std::vector<int>& target,
                                  int min_b, int max_b,
                                  double cutoff,
                                  int max_pre,
                                  double conv_thr,
                                  int max_iter)
    : feature_(feature),
      target_(target),
      min_bins_(std::max(min_b, 2)),
      max_bins_(std::max(max_b, min_b)),
      bin_cutoff_(cutoff),
      max_n_prebins_(std::max(max_pre, min_b)),
      convergence_threshold_(conv_thr),
      max_iterations_(max_iter),
      converged_(false),
      iterations_run_(0)
  {
    // 1) Validar tamanho
    if(feature_.size() != target_.size()) {
      throw std::invalid_argument("feature and target must have the same length.");
    }
    
    // 2) Identificar classes existentes
    std::unordered_set<int> unique_targets(target_.begin(), target_.end());
    n_classes_ = unique_targets.size();
    if(n_classes_ < 2) {
      throw std::invalid_argument("You must have at least 2 distinct classes in the target.");
    }
    
    // 3) Validar range
    if(cutoff <= 0.0 || cutoff >= 1.0)
      throw std::invalid_argument("bin_cutoff must be in (0,1).");
    if(convergence_threshold_ <= 0.0)
      throw std::invalid_argument("convergence_threshold must be positive.");
    if(max_iterations_ <= 0)
      throw std::invalid_argument("max_iterations must be positive.");
    
    // 4) Checar se target tem valores válidos (0..n_classes_-1)
    //    Obs.: Se estiverem fora desse range, não funciona.
    for(int t : target_) {
      if(t < 0 || t >= (int)n_classes_)
        throw std::invalid_argument("Target values must be in [0..(n_classes-1)].");
    }
    
    // 5) Checar se feature possui algum NaN/Inf
    for(double val : feature_) {
      if(std::isnan(val) || std::isinf(val)) {
        throw std::invalid_argument("Feature contains NaN/Inf.");
      }
    }
    
    // 6) Calcular total por classe
    total_class_counts_.resize(n_classes_, 0);
    for(size_t i = 0; i < target_.size(); i++) {
      total_class_counts_[ target_[i] ]++;
    }
  }
  
  // --------------------------------------------------------------------
  // Método principal de ajuste
  // --------------------------------------------------------------------
  void fit() {
    // 1) Preparar vetor único de valores
    std::vector<double> unique_vals = feature_;
    std::sort(unique_vals.begin(), unique_vals.end());
    unique_vals.erase(std::unique(unique_vals.begin(), unique_vals.end()), unique_vals.end());
    
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
      // - garantir monotonicidade simultânea nas classes
      enforce_monotonicity();
      // - garantir limites de bins
      merge_until_max_bins();
      // - recalcular M-WOE/IV
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
    size_t n_bins = bins_.size();
    CharacterVector bin_names(n_bins);
    // woes e ivs em formato (n_bins x n_classes)
    NumericMatrix woes(n_bins, n_classes_);
    NumericMatrix ivs(n_bins, n_classes_);
    
    IntegerVector counts(n_bins);
    IntegerMatrix class_counts(n_bins, n_classes_);
    
    NumericVector cutpoints;  // se houver mais de 1 bin, armazenar cutpoints
    
    if(n_bins > 1) {
      cutpoints = NumericVector((int)n_bins - 1);
    }
    
    for(size_t i = 0; i < n_bins; i++) {
      bin_names[i] = interval_to_string(bins_[i].lower, bins_[i].upper);
      counts[i]    = bins_[i].total_count;
      
      // Guardar WOE e IV por classe
      for(size_t k = 0; k < n_classes_; k++) {
        woes(i, k) = bins_[i].woes[k];
        ivs(i, k)  = bins_[i].ivs[k];
        class_counts(i, k) = bins_[i].class_counts[k];
      }
      // Gerar cutpoints (exclui último bin, pois é +Inf)
      if(i < n_bins - 1) {
        cutpoints[i] = bins_[i].upper;
      }
    }
    
    // IDs sequenciais
    NumericVector ids(n_bins);
    for(size_t i = 0; i < n_bins; i++) {
      ids[i] = i + 1;
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
      Named("n_classes")    = (int)n_classes_
    );
  }
  
private:
  // --------------------------------------------------------------------
  // Cria um único bin se todos os valores são idênticos
  // --------------------------------------------------------------------
  void handle_single_bin() {
    bins_.clear();
    bins_.emplace_back(-std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::infinity(),
                       n_classes_);
    
    // Preenche contagens
    for(size_t i = 0; i < feature_.size(); i++) {
      bins_[0].total_count++;
      bins_[0].class_counts[target_[i]]++;
    }
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
    
    for(size_t i = 0; i < feature_.size(); i++) {
      if(feature_[i] <= cut) {
        bins_[0].total_count++;
        bins_[0].class_counts[target_[i]]++;
      } else {
        bins_[1].total_count++;
        bins_[1].class_counts[target_[i]]++;
      }
    }
    compute_mwoe_iv();
  }
  
  // --------------------------------------------------------------------
  // Pré-binning usando quantis
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
      double edge = unique_vals[idx];
      // Evitar duplicação de edges
      if(edge > edges.back()) {
        edges.push_back(edge);
      }
    }
    edges.push_back(std::numeric_limits<double>::infinity());
    
    // Constrói bins
    for(size_t i = 0; i < edges.size() - 1; i++) {
      bins_.emplace_back(edges[i], edges[i+1], n_classes_);
    }
    
    // Se (int)bins_.size() < min_bins_, tentar split extra
    while((int)bins_.size() < min_bins_) {
      size_t idx = find_largest_bin(); 
      split_bin(idx);
    }
  }
  
  // --------------------------------------------------------------------
  // Atribuir cada valor ao seu bin
  // --------------------------------------------------------------------
  void assign_bins() {
    // Zerar contagens
    for(auto &b : bins_) {
      b.total_count = 0;
      std::fill(b.class_counts.begin(), b.class_counts.end(), 0);
    }
    
    // Fazer atribuição
    for(size_t i = 0; i < feature_.size(); i++) {
      double val = feature_[i];
      int idx = find_bin_index(val);
      if(idx < 0) {
        idx = (int)bins_.size() - 1;
      }
      bins_[idx].total_count++;
      bins_[idx].class_counts[target_[i]]++;
    }
  }
  
  // --------------------------------------------------------------------
  // Localizar bin via busca binária
  // --------------------------------------------------------------------
  int find_bin_index(double val) const {
    int left = 0;
    int right = (int)bins_.size() - 1;
    while(left <= right) {
      int mid = left + (right - left)/2;
      if(val > bins_[mid].lower && val <= bins_[mid].upper) {
        return mid;
      } else if(val <= bins_[mid].lower) {
        right = mid - 1;
      } else {
        left = mid + 1;
      }
    }
    return (int)bins_.size() - 1;
  }
  
  // --------------------------------------------------------------------
  // Mesclar bins de baixa frequência (bin_cutoff)
  // --------------------------------------------------------------------
  void merge_small_bins() {
    bool merged = true;
    double total = (double)feature_.size();
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
  // --------------------------------------------------------------------
  void compute_mwoe_iv() {
    // Precisamos das contagens totais para cada classe
    // (já calculadas no construtor e armazenadas em total_class_counts_)
    
    // Para cada bin e classe, calcular:
    //   class_rate_k = bin.class_counts[k] / total_class_counts[k]
    //   others_rate_k = (sum(bin.class_counts[j], j!=k)) / (sum(total_class_counts[j], j!=k))
    //   mwoe_k = ln( class_rate_k / others_rate_k )
    //   iv_k   = (class_rate_k - others_rate_k) * mwoe_k
    //
    // O IV total do bin é a soma (ou o bin carrega esse vetor).
    
    // Evitar zeros e divisões
    for(auto &b : bins_) {
      for(size_t k = 0; k < n_classes_; k++) {
        b.woes[k] = 0.0;
        b.ivs[k]  = 0.0;
      }
      if(b.total_count == 0) {
        continue;
      }
      for(size_t k = 0; k < n_classes_; k++) {
        double numerator   = (double)b.class_counts[k];
        double denominator = (double)total_class_counts_[k];
        
        // Evitar classes inexistentes
        if(denominator < 1) {
          // Ex: se target_ = {1,2} mas k=0 => classe 0 não existe
          b.woes[k] = 0.0;
          b.ivs[k]  = 0.0;
          continue;
        }
        
        double class_rate  = numerator / denominator; // p_k
        // soma dos outros no bin
        int sum_others_bin = 0;
        int sum_others_all = 0;
        for(size_t j = 0; j < n_classes_; j++) {
          if(j != k) {
            sum_others_bin += b.class_counts[j];
            sum_others_all += total_class_counts_[j];
          }
        }
        
        if(sum_others_all < 1) {
          // Significa que não há "outras classes" no dataset?
          // Então M-WOE fica indefinido. Vamos forçar 0:
          b.woes[k] = 0.0;
          b.ivs[k]  = 0.0;
          continue;
        }
        
        double others_rate = (double)sum_others_bin / (double)sum_others_all; // q_k
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
    
    // Precisamos decidir se, para cada classe, é monotonicamente crescente ou decrescente
    // Aqui, faremos algo simples: detectamos a "tendência" da classe e, se violar, merge.
    // Repetimos até não haver mais merges ou chegar em min_bins_
    while(changed && (int)bins_.size() > min_bins_ && local_iterations < max_iterations_) {
      changed = false;
      // Checar cada classe
      for(size_t k = 0; k < n_classes_; k++) {
        bool increasing = guess_trend_for_class(k);
        // Detectar violações
        for(size_t i = 1; i < bins_.size(); i++) {
          if(increasing && (bins_[i].woes[k] < bins_[i-1].woes[k])) {
            // viola monotonicidade => mescla
            merge_two_bins(i-1, i);
            compute_mwoe_iv();
            changed = true;
            break; 
          } else if(!increasing && (bins_[i].woes[k] > bins_[i-1].woes[k])) {
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
  // para minimizar a perda de IV
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
    
    bins_[i].upper = bins_[j].upper;
    bins_[i].total_count += bins_[j].total_count;
    for(size_t k = 0; k < n_classes_; k++) {
      bins_[i].class_counts[k] += bins_[j].class_counts[k];
    }
    bins_.erase(bins_.begin() + j);
  }
  
  // --------------------------------------------------------------------
  // Localiza bin com maior contagem (para split)
  // --------------------------------------------------------------------
  size_t find_largest_bin() const {
    size_t idx = 0;
    int max_count = bins_[0].total_count;
    for(size_t i = 1; i < bins_.size(); i++) {
      if(bins_[i].total_count > max_count) {
        idx = i;
        max_count = bins_[i].total_count;
      }
    }
    return idx;
  }
  
  // --------------------------------------------------------------------
  // Split de bin (usado apenas se bins_ < min_bins_)
  // --------------------------------------------------------------------
  void split_bin(size_t idx) {
    if(idx >= bins_.size()) return;
    
    NumBinMulti &b = bins_[idx];
    double mid = (b.lower + b.upper) / 2.0;
    if(!std::isfinite(mid)) return; 
    
    // Cria um bin novo e realoca ~ metade das contagens
    NumBinMulti new_bin(mid, b.upper, n_classes_);
    for(size_t k = 0; k < n_classes_; k++) {
      new_bin.class_counts[k] = b.class_counts[k]/2;
      b.class_counts[k]       = b.class_counts[k] - new_bin.class_counts[k];
    }
    
    new_bin.total_count = 0;
    for(size_t k = 0; k < n_classes_; k++) {
      new_bin.total_count += new_bin.class_counts[k];
    }
    
    b.upper = mid;
    b.total_count = 0;
    for(size_t k = 0; k < n_classes_; k++) {
      b.total_count += b.class_counts[k];
    }
    
    bins_.insert(bins_.begin() + idx + 1, new_bin);
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

//' @title Optimal Numerical Binning JEDI M-WOE (Multinomial Weight of Evidence)
//'
//' @description
//' Versão multinomial do binning numérico JEDI, que estende o WOE/IV tradicional (binário)
//' para uma abordagem M-WOE, considerando várias classes simultaneamente.
//'
//' @param target IntegerVector de tamanho n, com valores de 0..(K-1) indicando a classe.
//' @param feature NumericVector de tamanho n, com os valores contínuos da feature.
//' @param min_bins Número mínimo de bins no resultado (>=2).
//' @param max_bins Número máximo de bins no resultado (>= min_bins).
//' @param bin_cutoff Frequência mínima relativa de um bin para não ser mesclado (0<bin_cutoff<1).
//' @param max_n_prebins Número máximo de pré-bins (fase inicial via quantis).
//' @param convergence_threshold Tolerância para parar iterações com base na variação do IV total.
//' @param max_iterations Número máximo de iterações permitidas.
//'
//' @details
//' Implementa a discretização de variáveis numéricas em múltiplas classes (K>2), calculando
//' o M-WOE e o M-IV (Information Value Multinomial), e forçando monotonicidade para cada classe
//' por mesclagem iterativa de bins adjacentes que violem a ordem (crescente ou decrescente) de WOE.
//'
//' Fórmulas de M-WOE e M-IV para cada classe k em um bin i:
//' \deqn{M-WOE_{i,k} = \ln\left(\frac{ \frac{\text{count}_{i,k}}{ \text{Total}_k }}{ \frac{\sum_{j \neq k} \text{count}_{i,j}}{\sum_{j \neq k} \text{Total}_j}} \right)}
//'
//' \deqn{IV_{i,k} = \Bigl(\frac{\text{count}_{i,k}}{\text{Total}_k} - \frac{\sum_{j \neq k}\text{count}_{i,j}}{\sum_{j \neq k}\text{Total}_j}\Bigr) \times M-WOE_{i,k}}
//'
//' O IV total do bin i é \eqn{\sum_k IV_{i,k}} e o IV global é \eqn{\sum_i \sum_k IV_{i,k}}.
//'
//' @return Uma lista com:
//' \itemize{
//'   \item `bin`: vetor de rótulos dos bins (intervalos).
//'   \item `woe`: matriz (n_bins x n_classes) de M-WOE para cada bin e classe.
//'   \item `iv`: matriz (n_bins x n_classes) de IV por bin e classe.
//'   \item `count`: vetor com contagem total por bin.
//'   \item `class_counts`: matriz (n_bins x n_classes) com contagem por classe em cada bin.
//'   \item `cutpoints`: pontos de corte (excluindo ±Inf).
//'   \item `converged`: indica se houve convergência via `convergence_threshold`.
//'   \item `iterations`: número de iterações realizadas.
//'   \item `n_classes`: número de classes detectadas.
//' }
//'
//' @examples
//' \dontrun{
//' # Exemplo com 3 classes: 0, 1 e 2
//' target <- c(0,1,2,1,0,2,2,1,0,0,2)
//' feature <- c(1.1,2.2,3.5,2.7,1.0,4.2,3.9,2.8,1.2,1.0,3.6)
//' result <- optimal_binning_numerical_jedi_mwoe(target, feature,
//'                min_bins = 3, max_bins = 6, bin_cutoff = 0.05,
//'                max_n_prebins = 10, convergence_threshold = 1e-6,
//'                max_iterations = 100)
//' print(result)
//' }
//'
//' @export
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
 // Conversão de R para std::vector
 std::vector<double> feat(feature.begin(), feature.end());
 std::vector<int>    targ(target.begin(), target.end());
 
 try {
   // Instancia a classe e faz o fit
   OptimalBinningNumericalJEDIMWoE model(feat, targ,
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

