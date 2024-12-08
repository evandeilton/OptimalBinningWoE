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

using namespace Rcpp;

static constexpr double EPS = 1e-10;

struct NumBin {
  double lower;
  double upper;
  int count;
  int count_pos;
  int count_neg;
  double woe;
  double iv;
  
  NumBin(double l, double u): lower(l), upper(u),
  count(0), count_pos(0), count_neg(0),
  woe(0.0), iv(0.0) {}
};

class OptimalBinningNumericalJedi {
private:
  std::vector<double> feature;
  std::vector<double> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  double convergence_threshold;
  int max_iterations;
  
  std::vector<NumBin> bins;
  bool converged;
  int iterations_run;
  
public:
  OptimalBinningNumericalJedi(const std::vector<double>& feat,
                              const std::vector<double>& targ,
                              int min_b, int max_b, double cutoff,
                              int max_pre, double conv_thr, int max_iter)
    : feature(feat), target(targ),
      min_bins(std::max(min_b,2)), 
      max_bins(std::max(max_b,min_b)),
      bin_cutoff(cutoff),
      max_n_prebins(std::max(max_pre,min_b)),
      convergence_threshold(conv_thr),
      max_iterations(max_iter),
      converged(false),
      iterations_run(0) {
    
    if(feature.size()!=target.size()) 
      throw std::invalid_argument("Feature and target must have same length.");
    if(bin_cutoff<=0||bin_cutoff>=1)
      throw std::invalid_argument("bin_cutoff must be in (0,1).");
    if(convergence_threshold<=0)
      throw std::invalid_argument("convergence_threshold must be positive.");
    if(max_iterations<=0)
      throw std::invalid_argument("max_iterations must be positive.");
    
    // Check target binary
    bool has_zero=false,has_one=false;
    for(double t:target) {
      if(t==0) has_zero=true; else if(t==1) has_one=true; 
      else throw std::invalid_argument("Target must contain only 0 and 1.");
      if(has_zero&&has_one) break;
    }
    if(!has_zero || !has_one)
      throw std::invalid_argument("Target must contain both classes (0 and 1).");
    
    for (double f:feature) {
      if(std::isnan(f)||std::isinf(f))
        throw std::invalid_argument("Feature contains NaN/Inf.");
    }
  }
  
  void fit() {
    // Check unique values
    std::vector<double> unique_vals = feature;
    std::sort(unique_vals.begin(),unique_vals.end());
    unique_vals.erase(std::unique(unique_vals.begin(),unique_vals.end()), unique_vals.end());
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
    CharacterVector bin_names(bins.size());
    NumericVector woe_vals(bins.size());
    NumericVector iv_vals(bins.size());
    IntegerVector count_vals(bins.size());
    IntegerVector cpos_vals(bins.size());
    IntegerVector cneg_vals(bins.size());
    NumericVector cutpoints;
    if(bins.size()>1) cutpoints = NumericVector((int)bins.size()-1);
    
    for (size_t i=0;i<bins.size();i++){
      std::ostringstream oss;
      oss<<"("<<edge_to_str(bins[i].lower)<<";"<<edge_to_str(bins[i].upper)<<"]";
      bin_names[i]=oss.str();
      woe_vals[i]=bins[i].woe;
      iv_vals[i]=bins[i].iv;
      count_vals[i]=bins[i].count;
      cpos_vals[i]=bins[i].count_pos;
      cneg_vals[i]=bins[i].count_neg;
      if(i<bins.size()-1) {
        cutpoints[i]=bins[i].upper;
      }
    }
    
    return List::create(
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
  // Let total_pos = Σ Pos_i and total_neg = Σ Neg_i over all bins.
  // If total_pos=0 or total_neg=0, WOE and IV can't be computed meaningfully, fallback to zero.
  //
  // Monotonicity:
  // We define monotonic order as either strictly increasing or decreasing WOE across bin edges.
  // Guess direction by comparing number of WOE increments vs decrements.
  // If not monotonic, merge adjacent bins that cause violations until monotonic order is restored or min_bins reached.
  //
  // Minimizing IV loss merges:
  // When adjusting bin counts, choose pairs of bins to merge that yield minimal increase in IV loss (or minimal IV sum),
  // ensuring stable and minimal-information-loss merges.
  //
  // Convergence:
  // Convergence is reached when |IV_current - IV_previous| < convergence_threshold or max_iterations is hit.
  // This ensures the algorithm stops when no significant improvements in the binning structure can be made.
  //
  // Heuristic blending:
  // - From IR/KMB: Use changes in IV for convergence and WOE ordering to guess monotonic direction.
  // - From EWB/FETB: Merge small bins first, ensuring stable counts per bin.
  // - From MBLP/MDLP: Minimal IV-loss merges to handle max_bins constraints.
  // - From DPLC/MOB: EPSILON in WOE calculation, careful infinite checks, stable computations.
  // - From all: a unified approach that first pre-bins (for complexity control), then iteratively refines.
  
  void handle_low_unique(const std::vector<double>& unique_vals) {
    bins.clear();
    if(unique_vals.size()==1) {
      // All identical
      bins.emplace_back(-std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity());
      for (size_t i=0;i<feature.size();i++){
        bins[0].count++;
        if(target[i]==1) bins[0].count_pos++; else bins[0].count_neg++;
      }
      calculate_woe_iv();
    } else {
      // Two unique values
      double cut = unique_vals[0];
      bins.emplace_back(-std::numeric_limits<double>::infinity(), cut);
      bins.emplace_back(cut, std::numeric_limits<double>::infinity());
      for (size_t i=0;i<feature.size();i++) {
        if(feature[i]<=cut) {
          bins[0].count++;
          if(target[i]==1) bins[0].count_pos++; else bins[0].count_neg++;
        } else {
          bins[1].count++;
          if(target[i]==1) bins[1].count_pos++; else bins[1].count_neg++;
        }
      }
      calculate_woe_iv();
    }
  }
  
  void initial_prebin(const std::vector<double>& unique_vals) {
    int n_unique=(int)unique_vals.size();
    int n_pre = std::min(max_n_prebins,n_unique);
    n_pre=std::max(n_pre,min_bins);
    
    // quantile approach:
    // Let's create n_pre equally spaced quantiles
    bins.clear();
    bins.reserve(n_pre);
    
    std::vector<double> edges;
    edges.push_back(-std::numeric_limits<double>::infinity());
    for(int i=1;i<n_pre;i++){
      double p=(double)i/n_pre;
      int idx=(int)std::floor(p*(n_unique-1));
      double edge=unique_vals[idx];
      if(edge>edges.back()) {
        edges.push_back(edge);
      }
    }
    edges.push_back(std::numeric_limits<double>::infinity());
    
    // Create bins from edges
    bins.clear();
    for (size_t i=0;i<edges.size()-1;i++){
      bins.emplace_back(edges[i],edges[i+1]);
    }
    // Ensure at least min_bins
    while((int)bins.size()<min_bins) {
      // If less, split the largest bin
      size_t large_idx=find_largest_bin();
      split_bin(large_idx);
    }
  }
  
  void assign_bins() {
    // reset counts
    for (auto &b:bins) {
      b.count=0;b.count_pos=0;b.count_neg=0;
    }
    
    for (size_t i=0;i<feature.size();i++){
      double val=feature[i];
      int idx=find_bin(val);
      if(idx<0 || idx>=(int)bins.size()) idx=(int)bins.size()-1;
      bins[idx].count++;
      if(target[i]==1) bins[idx].count_pos++; else bins[idx].count_neg++;
    }
  }
  
  int find_bin(double val) const {
    // binary search or linear since bins are sorted by edges
    int left=0;
    int right=(int)bins.size()-1;
    while(left<=right) {
      int mid=left+(right-left)/2;
      if(val>bins[mid].lower && val<=bins[mid].upper)
        return mid;
      else if(val<=bins[mid].lower)
        right=mid-1;
      else
        left=mid+1;
    }
    return (int)bins.size()-1;
  }
  
  void merge_small_bins() {
    bool merged=true;
    double total=(double)feature.size();
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
  
  bool is_monotonic(bool increasing) {
    for (size_t i=1;i<bins.size();i++) {
      if(increasing && bins[i].woe<bins[i-1].woe)return false;
      if(!increasing && bins[i].woe>bins[i-1].woe)return false;
    }
    return true;
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
    bins[i].upper=bins[j].upper;
    bins[i].count+=bins[j].count;
    bins[i].count_pos+=bins[j].count_pos;
    bins[i].count_neg+=bins[j].count_neg;
    bins.erase(bins.begin()+j);
  }
  
  size_t find_largest_bin() const {
    size_t idx=0;
    int max_count=bins[0].count;
    for(size_t i=1;i<bins.size();i++){
      if(bins[i].count>max_count) {
        max_count=bins[i].count;
        idx=i;
      }
    }
    return idx;
  }
  
  void split_bin(size_t idx) {
    if(idx>=bins.size())return;
    NumBin &b=bins[idx];
    double mid=(b.lower+b.upper)/2.0;
    if(!std::isfinite(mid)) return; // can't split infinite intervals meaningfully
    // approximate half counts
    NumBin new_bin(mid,b.upper);
    new_bin.count=b.count/2;
    new_bin.count_pos=b.count_pos/2;
    new_bin.count_neg=b.count_neg/2;
    
    b.upper=mid;
    b.count-=new_bin.count;
    b.count_pos-=new_bin.count_pos;
    b.count_neg-=new_bin.count_neg;
    
    bins.insert(bins.begin()+idx+1,new_bin);
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
  
  // no separate update_cutpoints needed, done in create_output
};


//' @title Binning Ótimo Numérico JEDI (Discretização por Intervalos Guiada por Entropia Conjunta)
//'
//' @description
//' Um algoritmo avançado de binning numérico que otimiza o valor de informação (IV) mantendo 
//' relações monotônicas de weight of evidence (WoE). O algoritmo emprega pré-binning baseado em 
//' quantis com estratégias adaptativas de fusão, garantindo tanto estabilidade estatística quanto 
//' preservação ótima de informação.
//'
//' @details
//' Framework Matemático:
//' Para uma variável numérica \eqn{X} e alvo binário \eqn{Y \in \{0,1\}}, o algoritmo cria \eqn{K} bins 
//' definidos por \eqn{K-1} pontos de corte onde cada bin \eqn{B_i = (c_{i-1}, c_i]} maximiza o conteúdo 
//' de informação satisfazendo as restrições:
//'
//' \enumerate{
//'   \item Monotonicidade WoE: \eqn{WoE_i \le WoE_{i+1}} (ou \eqn{\ge} para tendência decrescente)
//'   \item Tamanho mínimo do bin: \eqn{contagem(B_i)/N \ge bin\_cutoff}
//'   \item Limites de quantidade de bins: \eqn{min\_bins \le K \le max\_bins}
//' }
//'
//' O Weight of Evidence para bin \eqn{i} é definido como:
//' \deqn{WoE_i = ln(\frac{Pos_i / \sum Pos_i}{Neg_i / \sum Neg_i})}
//'
//' E o Information Value por bin como:
//' \deqn{IV_i = (Pos_i/\sum Pos_i - Neg_i/\sum Neg_i) \times WoE_i}
//'
//' O IV total é dado por:
//' \deqn{IV_{total} = \sum_{i=1}^K IV_i}
//'
//' Fases do Algoritmo:
//' 1. Pré-binning baseado em quantis com validação de frequência mínima
//' 2. Tratamento de bins pequenos através de fusões minimizando perda de IV
//' 3. Imposição de monotonicidade via fusão adaptativa de bins
//' 4. Otimização da quantidade de bins através de divisões/fusões controladas
//' 5. Monitoramento de convergência usando estabilidade do IV
//'
//' Características Principais:
//' - Cálculo de WoE protegido por epsilon para estabilidade numérica
//' - Estratégia adaptativa de fusão minimizando perda de informação
//' - Tratamento robusto de casos extremos e distribuições extremas
//' - Busca binária eficiente para atribuição de bins
//' - Detecção antecipada de convergência
//'
//' @param target Vetor numérico (0,1) representando variável alvo binária
//' @param feature Vetor numérico do preditor contínuo
//' @param min_bins Número mínimo de bins de saída (padrão: 3, deve ser ≥2)
//' @param max_bins Número máximo de bins de saída (padrão: 5, deve ser ≥ min_bins)
//' @param bin_cutoff Frequência relativa mínima por bin (padrão: 0.05)
//' @param max_n_prebins Número máximo de pré-bins antes da otimização (padrão: 20)
//' @param convergence_threshold Limite de diferença de IV para convergência (padrão: 1e-6)
//' @param max_iterations Máximo de iterações de otimização (padrão: 1000)
//'
//' @return Uma lista contendo:
//' \itemize{
//'   \item bin: Vetor de caracteres dos intervalos dos bins
//'   \item woe: Vetor numérico dos valores de Weight of Evidence
//'   \item iv: Vetor numérico dos valores de Information Value por bin
//'   \item count: Vetor inteiro de contagens de observações por bin
//'   \item count_pos: Vetor inteiro de contagens da classe positiva por bin
//'   \item count_neg: Vetor inteiro de contagens da classe negativa por bin
//'   \item cutpoints: Vetor numérico dos pontos de corte (excluindo ±Inf)
//'   \item converged: Lógico indicando se o algoritmo convergiu
//'   \item iterations: Contagem inteira de iterações de otimização realizadas
//' }
//'
//' @references
//' \itemize{
//'   \item Teoria da Informação e Aprendizado Estatístico (Cover & Thomas, 2006)
//'   \item Binning Ótimo para Modelos de Scoring (Mironchyk & Tchistiakov, 2017)
//'   \item Binning e Scoring Monotônico (Beltrami & Bassani, 2021)
//' }
//'
//' @examples
//' \dontrun{
//' # Uso básico com parâmetros padrão
//' resultado <- optimal_binning_numerical_jedi(
//'   target = c(1,0,1,0,1),
//'   feature = c(1.2,3.4,2.1,4.5,2.8)
//' )
//'
//' # Configuração personalizada para binning mais granular
//' resultado <- optimal_binning_numerical_jedi(
//'   target = vetor_target,
//'   feature = vetor_feature,
//'   min_bins = 5,
//'   max_bins = 10,
//'   bin_cutoff = 0.03
//' )
//' }
//'
//' @export
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
 
 std::vector<double> f(feature.begin(), feature.end());
 std::vector<double> t(target.begin(), target.end());
 
 try {
   OptimalBinningNumericalJedi model(f,t,min_bins,max_bins,
                                     bin_cutoff,max_n_prebins,
                                     convergence_threshold,max_iterations);
   model.fit();
   return model.create_output();
 } catch(const std::exception &e) {
   stop("Error in optimal_binning_numerical_jedi: "+std::string(e.what()));
 }
}

/*
 Breves Comentários das Melhorias Implementadas:
 - Combina lógica de quantilização (pré-bins) + fusão de bins pequenos (EWB/FETB).
 - Garantia de monotonicidade via merges inspiradas em IR/KMB/MOB.
 - Uso de minimal IV loss merges da família MBLP/MDLP para ajustar a contagem final de bins.
 - Convergência baseada em variação do IV (IR/KMB), abortando cedo se convergência alcançada.
 - Tratamento cuidadoso de casos triviais (<=2 unique values).
 - Manuseio robusto de EPS, evitando log(0).
 - Estrutura do código e comentários em inglês, mantendo nomes e I/O.
 - Estabilidade computacional e sem loops infinitos devido a checks e limites.
*/

