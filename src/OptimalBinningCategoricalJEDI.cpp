// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

using namespace Rcpp;

//' @title Binning Ótimo Categórico JEDI (Discretização Conjunta Guiada por Entropia)
//'
//' @description
//' Um algoritmo robusto de binning categórico que otimiza o valor de informação (IV) mantendo
//' relações monotônicas de weight of evidence (WoE). Implementa uma estratégia adaptativa de 
//' fusão com proteções de estabilidade numérica e controle sofisticado do número de bins.
//'
//' @details
//' O algoritmo emprega uma abordagem de otimização em múltiplas fases:
//' 
//' Framework Matemático:
//' Para um bin i, o WoE é calculado como:
//' \deqn{WoE_i = ln(\frac{p_i + \epsilon}{n_i + \epsilon})}
//' onde:
//' \itemize{
//'   \item \eqn{p_i} é a proporção de casos positivos no bin i relativo ao total de positivos
//'   \item \eqn{n_i} é a proporção de casos negativos no bin i relativo ao total de negativos
//'   \item \eqn{\epsilon} é uma pequena constante (1e-10) para prevenir logaritmos indefinidos
//' }
//'
//' O IV para cada bin é calculado como:
//' \deqn{IV_i = (p_i - n_i) \times WoE_i}
//'
//' E o IV total é:
//' \deqn{IV_{total} = \sum_{i=1}^{k} IV_i}
//'
//' Fases:
//' 1. Binning Inicial: Cria bins individuais para categorias únicas com validação de frequência
//' 2. Tratamento de Baixa Frequência: Combina categorias raras (< bin_cutoff) para garantir estabilidade estatística
//' 3. Otimização: Combina bins iterativamente usando minimização de perda de IV mantendo monotonicidade de WoE
//' 4. Ajuste Final: Garante restrições de contagem de bins (min_bins <= bins <= max_bins) quando possível
//'
//' Características Principais:
//' - Cálculos de WoE protegidos por epsilon para estabilidade numérica
//' - Estratégia adaptativa de fusão que minimiza perda de informação
//' - Tratamento robusto de casos extremos e violações de restrições
//' - Sem criação artificial de categorias, garantindo resultados interpretáveis
//'
//' Controle de Quantidade de Bins:
//' - Se bins > max_bins: Continua fusões usando minimização de perda de IV
//' - Se bins < min_bins: Retorna melhor solução disponível em vez de criar divisões artificiais
//'
//' @param target Vetor inteiro binário (0 ou 1) representando a variável resposta
//' @param feature Vetor de caracteres dos valores categóricos preditores
//' @param min_bins Número mínimo de bins de saída (padrão: 3). Ajustado se categorias únicas < min_bins
//' @param max_bins Número máximo de bins de saída (padrão: 5). Deve ser >= min_bins
//' @param bin_cutoff Limite mínimo de frequência relativa para bins individuais (padrão: 0.05)
//' @param max_n_prebins Número máximo de pré-bins antes da otimização (padrão: 20)
//' @param bin_separator Delimitador para nomes de categorias combinadas (padrão: "%;%")
//' @param convergence_threshold Limite de diferença de IV para convergência (padrão: 1e-6)
//' @param max_iterations Máximo de iterações de otimização (padrão: 1000)
//'
//' @return Uma lista contendo:
//' \itemize{
//'   \item bin: Vetor de caracteres com nomes dos bins (categorias concatenadas)
//'   \item woe: Vetor numérico com valores de Weight of Evidence
//'   \item iv: Vetor numérico com valores de Information Value por bin
//'   \item count: Vetor inteiro com contagens de observações por bin
//'   \item count_pos: Vetor inteiro com contagens da classe positiva por bin
//'   \item count_neg: Vetor inteiro com contagens da classe negativa por bin
//'   \item converged: Lógico indicando se o algoritmo convergiu
//'   \item iterations: Contagem inteira de iterações de otimização realizadas
//' }
//'
//' @references
//' \itemize{
//'   \item Framework de Binning Ótimo (Beltrami et al., 2021)
//'   \item Teoria do Valor da Informação em Gestão de Risco (Thomas et al., 2002)
//'   \item Algoritmos de Binning Monotônico em Credit Scoring (Mironchyk & Tchistiakov, 2017)
//' }
//'
//' @examples
//' \dontrun{
//' # Uso básico
//' resultado <- optimal_binning_categorical_jedi(
//'   target = c(1,0,1,1,0),
//'   feature = c("A","B","A","C","B"),
//'   min_bins = 2,
//'   max_bins = 3
//' )
//'
//' # Tratamento de categorias raras
//' resultado <- optimal_binning_categorical_jedi(
//'   target = vetor_target,
//'   feature = vetor_feature,
//'   bin_cutoff = 0.03,  # Tratamento mais agressivo de categorias raras
//'   max_n_prebins = 15  # Limite de bins iniciais
//' )
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_jedi(
   Rcpp::IntegerVector target,
   Rcpp::StringVector feature,
   int min_bins = 3,
   int max_bins = 5,
   double bin_cutoff = 0.05,
   int max_n_prebins = 20,
   std::string bin_separator = "%;%",
   double convergence_threshold = 1e-6,
   int max_iterations = 1000
) {
 // Classe interna corrigida
 class OptimalBinningCategoricalJEDI {
 private:
   struct BinInfo {
     std::vector<std::string> categories;
     int count;
     int count_pos;
     int count_neg;
     double woe;
     double iv;
     
     BinInfo() : count(0), count_pos(0), count_neg(0), woe(0.0), iv(0.0) {}
   };
   
   std::vector<std::string> feature_;
   std::vector<int> target_;
   int min_bins_;
   int max_bins_;
   double bin_cutoff_;
   int max_n_prebins_;
   std::string bin_separator_;
   double convergence_threshold_;
   int max_iterations_;
   
   std::vector<BinInfo> bins_;
   int total_pos_;
   int total_neg_;
   bool converged_;
   int iterations_run_;
   double epsilon_ = 1e-10;
   
   void validate_inputs() {
     if (feature_.size() != target_.size()) {
       throw std::invalid_argument("Feature and target vectors must have the same length");
     }
     if (feature_.empty()) {
       throw std::invalid_argument("Feature and target cannot be empty");
     }
     for (auto t : target_) {
       if (t != 0 && t != 1) {
         throw std::invalid_argument("Target must be binary (0/1)");
       }
     }
   }
   
   double calculate_woe(int pos, int neg) const {
     double pos_rate = (double)pos / total_pos_;
     double neg_rate = (double)neg / total_neg_;
     double safe_pos = std::max(pos_rate, epsilon_);
     double safe_neg = std::max(neg_rate, epsilon_);
     return std::log(safe_pos / safe_neg);
   }
   
   double calculate_iv(const std::vector<BinInfo>& current_bins) const {
     double iv = 0.0;
     for (const auto& bin : current_bins) {
       double pos_rate = (double)bin.count_pos / total_pos_;
       double neg_rate = (double)bin.count_neg / total_neg_;
       if (pos_rate > 0 && neg_rate > 0) {
         iv += (pos_rate - neg_rate)*std::log(std::max(pos_rate,epsilon_)/std::max(neg_rate,epsilon_));
       }
     }
     return iv;
   }
   
   void compute_woe_iv(std::vector<BinInfo>& current_bins) {
     for (auto& bin : current_bins) {
       double pos_rate = (double)bin.count_pos / total_pos_;
       double neg_rate = (double)bin.count_neg / total_neg_;
       double safe_pos = std::max(pos_rate, epsilon_);
       double safe_neg = std::max(neg_rate, epsilon_);
       bin.woe = std::log(safe_pos / safe_neg);
       bin.iv = (pos_rate - neg_rate)*bin.woe;
     }
   }
   
   bool is_monotonic(const std::vector<BinInfo>& current_bins) const {
     if (current_bins.size() <= 2) return true;
     bool increasing = true;
     bool decreasing = true;
     
     for (size_t i = 1; i < current_bins.size(); ++i) {
       if (current_bins[i].woe < current_bins[i-1].woe) {
         increasing = false;
       }
       if (current_bins[i].woe > current_bins[i-1].woe) {
         decreasing = false;
       }
       if (!increasing && !decreasing) break;
     }
     return increasing || decreasing;
   }
   
   static std::string join_categories(const std::vector<std::string>& cats, const std::string& sep) {
     std::ostringstream oss;
     for (size_t i = 0; i < cats.size(); ++i) {
       if (i > 0) oss << sep;
       oss << cats[i];
     }
     return oss.str();
   }
   
   void initial_binning() {
     std::unordered_map<std::string, BinInfo> bin_map;
     total_pos_ = 0;
     total_neg_ = 0;
     
     for (size_t i = 0; i < feature_.size(); ++i) {
       const std::string& cat = feature_[i];
       int val = target_[i];
       auto &b = bin_map[cat];
       if (b.categories.empty()) {
         b.categories.push_back(cat);
       }
       b.count++;
       b.count_pos += val;
       b.count_neg += (1 - val);
       total_pos_ += val;
       total_neg_ += (1 - val);
     }
     bins_.clear();
     for (auto &kv : bin_map) {
       bins_.push_back(std::move(kv.second));
     }
   }
   
   void merge_low_freq() {
     int total_count = 0;
     for (auto &b : bins_) {
       total_count += b.count;
     }
     double cutoff_count = total_count * bin_cutoff_;
     
     std::sort(bins_.begin(), bins_.end(), [](const BinInfo&a, const BinInfo& b){
       return a.count < b.count;
     });
     
     std::vector<BinInfo> new_bins;
     BinInfo others;
     for (auto &b : bins_) {
       if (b.count >= cutoff_count || (int)new_bins.size() < min_bins_) {
         new_bins.push_back(b);
       } else {
         others.categories.insert(others.categories.end(), b.categories.begin(), b.categories.end());
         others.count += b.count;
         others.count_pos += b.count_pos;
         others.count_neg += b.count_neg;
       }
     }
     if (others.count > 0) {
       others.categories.push_back("Others");
       new_bins.push_back(others);
     }
     bins_ = new_bins;
   }
   
   void ensure_monotonic_order() {
     std::sort(bins_.begin(), bins_.end(), [this](const BinInfo&a, const BinInfo&b){
       return a.woe < b.woe;
     });
   }
   
   void merge_bins_in_temp(std::vector<BinInfo>& temp, size_t idx1, size_t idx2) {
     temp[idx1].categories.insert(temp[idx1].categories.end(),
                                  temp[idx2].categories.begin(), temp[idx2].categories.end());
     temp[idx1].count += temp[idx2].count;
     temp[idx1].count_pos += temp[idx2].count_pos;
     temp[idx1].count_neg += temp[idx2].count_neg;
     temp.erase(temp.begin() + idx2);
     compute_woe_iv(temp);
   }
   
   void merge_bins(size_t idx1, size_t idx2) {
     bins_[idx1].categories.insert(bins_[idx1].categories.end(),
                                   bins_[idx2].categories.begin(),
                                   bins_[idx2].categories.end());
     bins_[idx1].count += bins_[idx2].count;
     bins_[idx1].count_pos += bins_[idx2].count_pos;
     bins_[idx1].count_neg += bins_[idx2].count_neg;
     bins_.erase(bins_.begin() + idx2);
     compute_woe_iv(bins_);
   }
   
   void merge_adjacent_bins_for_prebins() {
     while (bins_.size() > (size_t)max_n_prebins_ && bins_.size() > (size_t)min_bins_) {
       double original_iv = calculate_iv(bins_);
       double min_iv_loss = std::numeric_limits<double>::max();
       size_t merge_index = 0;
       
       for (size_t i = 0; i < bins_.size()-1; ++i) {
         std::vector<BinInfo> temp = bins_;
         merge_bins_in_temp(temp, i, i+1);
         double new_iv = calculate_iv(temp);
         double iv_loss = original_iv - new_iv;
         if (iv_loss < min_iv_loss) {
           min_iv_loss = iv_loss;
           merge_index = i;
         }
       }
       merge_bins(merge_index, merge_index+1);
     }
   }
   
   void improve_monotonicity_step() {
     for (size_t i = 1; i + 1 < bins_.size(); ++i) {
       if ((bins_[i].woe < bins_[i-1].woe && bins_[i].woe < bins_[i+1].woe) ||
           (bins_[i].woe > bins_[i-1].woe && bins_[i].woe > bins_[i+1].woe)) {
         
         double orig_iv = calculate_iv(bins_);
         
         // Testa merge i-1,i
         std::vector<BinInfo> temp1 = bins_;
         merge_bins_in_temp(temp1, i-1, i);
         double iv_merge_1 = calculate_iv(temp1);
         double loss1 = orig_iv - iv_merge_1;
         
         // Testa merge i,i+1
         std::vector<BinInfo> temp2 = bins_;
         merge_bins_in_temp(temp2, i, i+1);
         double iv_merge_2 = calculate_iv(temp2);
         double loss2 = orig_iv - iv_merge_2;
         
         if (loss1 < loss2) {
           merge_bins(i-1, i);
         } else {
           merge_bins(i, i+1);
         }
         break;
       }
     }
   }
   
   void optimize() {
     double prev_iv = calculate_iv(bins_);
     converged_ = false;
     iterations_run_ = 0;
     
     while (iterations_run_ < max_iterations_) {
       if (is_monotonic(bins_) && bins_.size() <= (size_t)max_bins_ && bins_.size() >= (size_t)min_bins_) {
         converged_ = true;
         break;
       }
       
       if (bins_.size() > (size_t)min_bins_) {
         // Se temos mais bins que min_bins, podemos tentar merges ou monotonicidade
         if (bins_.size() > (size_t)max_bins_) {
           merge_adjacent_bins_for_prebins();
         } else {
           // Tentar melhorar monotonicidade
           improve_monotonicity_step();
         }
       } else {
         // bins_.size() < min_bins_, não há splitting implementado
         // Portanto paramos aqui, pois não conseguimos atingir min_bins
         break;
       }
       
       double current_iv = calculate_iv(bins_);
       if (std::abs(current_iv - prev_iv) < convergence_threshold_) {
         converged_ = true;
         break;
       }
       prev_iv = current_iv;
       iterations_run_++;
     }
     
     // Ao final, tentar forçar [min_bins_, max_bins_]
     // Se temos mais bins que max_bins_, mesclar até max_bins_
     while ((int)bins_.size() > max_bins_) {
       double orig_iv = calculate_iv(bins_);
       double min_iv_loss = std::numeric_limits<double>::max();
       size_t merge_index = 0;
       
       if (bins_.size() == 1) break;
       
       for (size_t i = 0; i < bins_.size()-1; ++i) {
         std::vector<BinInfo> temp = bins_;
         merge_bins_in_temp(temp, i, i+1);
         double new_iv = calculate_iv(temp);
         double iv_loss = orig_iv - new_iv;
         if (iv_loss < min_iv_loss) {
           min_iv_loss = iv_loss;
           merge_index = i;
         }
       }
       merge_bins(merge_index, merge_index+1);
     }
     
     // Se temos menos bins que min_bins_ e não há splitting, 
     // apenas aceitar como está, pois não podemos criar bins artificiais.
     // Ao menos teremos convergido ou chegado ao fim das iterações.
     
     ensure_monotonic_order();
     compute_woe_iv(bins_);
   }
   
 public:
   OptimalBinningCategoricalJEDI(
     const std::vector<std::string>& feature,
     const std::vector<int>& target,
     int min_bins,
     int max_bins,
     double bin_cutoff,
     int max_n_prebins,
     std::string bin_separator,
     double convergence_threshold,
     int max_iterations
   ) : feature_(feature), target_(target),
   min_bins_(min_bins), max_bins_(max_bins), bin_cutoff_(bin_cutoff),
   max_n_prebins_(max_n_prebins), bin_separator_(bin_separator),
   convergence_threshold_(convergence_threshold), max_iterations_(max_iterations),
   converged_(false), iterations_run_(0), total_pos_(0), total_neg_(0)
   {
     validate_inputs();
     std::unordered_set<std::string> unique_cats(feature_.begin(), feature_.end());
     int ncat = (int)unique_cats.size();
     if (ncat < min_bins_) {
       min_bins_ = std::max(1, ncat);
     }
     if (max_bins_ < min_bins_) {
       max_bins_ = min_bins_;
     }
     if (max_n_prebins_ < min_bins_) {
       max_n_prebins_ = min_bins_;
     }
   }
   
   void fit() {
     std::unordered_set<std::string> unique_cats(feature_.begin(), feature_.end());
     int ncat = (int)unique_cats.size();
     
     if (ncat <= 2) {
       // Apenas calcular WoE/IV
       int total_pos = 0, total_neg = 0;
       std::unordered_map<std::string, BinInfo> bin_map;
       for (size_t i = 0; i < feature_.size(); ++i) {
         auto& b = bin_map[feature_[i]];
         if (b.categories.empty()) {
           b.categories.push_back(feature_[i]);
         }
         b.count++;
         b.count_pos += target_[i];
         b.count_neg += (1 - target_[i]);
         total_pos += target_[i];
         total_neg += (1 - target_[i]);
       }
       bins_.clear();
       for (auto &kv : bin_map) {
         bins_.push_back(kv.second);
       }
       total_pos_ = total_pos;
       total_neg_ = total_neg;
       compute_woe_iv(bins_);
       converged_ = true;
       iterations_run_ = 0;
       return;
     }
     
     initial_binning();
     merge_low_freq();
     compute_woe_iv(bins_);
     ensure_monotonic_order();
     
     if ((int)bins_.size() > max_n_prebins_) {
       merge_adjacent_bins_for_prebins();
     }
     
     optimize();
   }
   
   Rcpp::List get_results() const {
     CharacterVector bin_names;
     NumericVector woes, ivs;
     IntegerVector counts, counts_pos, counts_neg;
     
     for (auto &b : bins_) {
       bin_names.push_back(join_categories(b.categories, bin_separator_));
       woes.push_back(b.woe);
       ivs.push_back(b.iv);
       counts.push_back(b.count);
       counts_pos.push_back(b.count_pos);
       counts_neg.push_back(b.count_neg);
     }
     
     return List::create(
       Named("bin") = bin_names,
       Named("woe") = woes,
       Named("iv") = ivs,
       Named("count") = counts,
       Named("count_pos") = counts_pos,
       Named("count_neg") = counts_neg,
       Named("converged") = converged_,
       Named("iterations") = iterations_run_
     );
   }
 };
 
 std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
 std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
 
 try {
   OptimalBinningCategoricalJEDI jedi(feature_vec, target_vec, min_bins, max_bins,
                                      bin_cutoff, max_n_prebins,
                                      bin_separator, convergence_threshold, max_iterations);
   jedi.fit();
   return jedi.get_results();
 } catch (const std::exception& e) {
   Rcpp::stop("Error in optimal binning JEDI: " + std::string(e.what()));
 }
}
