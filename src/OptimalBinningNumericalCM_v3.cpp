// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <stdexcept>
#include <numeric>
#include <iterator>

using namespace Rcpp;

// Classe para binning ótimo numérico via ChiMerge
class OptimalBinningNumericalCM {
private:
 // Dados de entrada
 const std::vector<double> feature;
 const std::vector<int> target;
 int min_bins;
 int max_bins;
 double bin_cutoff;
 int max_n_prebins;
 double convergence_threshold;
 int max_iterations;
 
 // Estrutura do bin
 struct Bin {
   double lower_bound;
   double upper_bound;
   int count;
   int count_pos;
   int count_neg;
   double woe;
   double iv;
   
   Bin()
     : lower_bound(0.0), upper_bound(0.0), count(0), count_pos(0), count_neg(0), woe(0.0), iv(0.0) {}
 };
 
 // Resultado
 std::vector<Bin> bins;
 bool converged;
 int iterations_run;
 
 static constexpr double EPSILON = 1e-10;
 
 // Calcula Qui-quadrado entre dois bins
 inline double calculate_chi_square(const Bin& bin1, const Bin& bin2) const {
   int total_pos = bin1.count_pos + bin2.count_pos;
   int total_neg = bin1.count_neg + bin2.count_neg;
   int total = total_pos + total_neg;
   
   if (total == 0) {
     // Evita divisão por zero
     return std::numeric_limits<double>::max();
   }
   
   double expected_pos1 = std::max((double)bin1.count * (double)total_pos / (double)total, EPSILON);
   double expected_neg1 = std::max((double)bin1.count * (double)total_neg / (double)total, EPSILON);
   double expected_pos2 = std::max((double)bin2.count * (double)total_pos / (double)total, EPSILON);
   double expected_neg2 = std::max((double)bin2.count * (double)total_neg / (double)total, EPSILON);
   
   double chi_square =
     std::pow(bin1.count_pos - expected_pos1, 2.0) / expected_pos1 +
     std::pow(bin1.count_neg - expected_neg1, 2.0) / expected_neg1 +
     std::pow(bin2.count_pos - expected_pos2, 2.0) / expected_pos2 +
     std::pow(bin2.count_neg - expected_neg2, 2.0) / expected_neg2;
   
   // Penaliza bins com contagens zero para encorajar mesclas
   if (bin1.count_pos == 0 || bin1.count_neg == 0 || bin2.count_pos == 0 || bin2.count_neg == 0) {
     chi_square = 0.0;
   }
   
   return chi_square;
 }
 
 // Mescla bins adjacentes
 inline void merge_bins(size_t index) {
   if (index >= bins.size() - 1) return;
   Bin& left = bins[index];
   const Bin& right = bins[index + 1];
   
   left.upper_bound = right.upper_bound;
   left.count += right.count;
   left.count_pos += right.count_pos;
   left.count_neg += right.count_neg;
   
   bins.erase(bins.begin() + index + 1);
 }
 
 // Mescla bins com contagem zero em alguma classe
 void merge_zero_bins() {
   bool merged = true;
   while (merged && bins.size() > 1) {
     merged = false;
     for (size_t i = 0; i < bins.size(); ++i) {
       if (bins[i].count_pos == 0 || bins[i].count_neg == 0) {
         if (i == 0) {
           merge_bins(i);
         } else if (i == bins.size() - 1) {
           merge_bins(i - 1);
         } else {
           double chi_left = calculate_chi_square(bins[i - 1], bins[i]);
           double chi_right = calculate_chi_square(bins[i], bins[i + 1]);
           if (chi_left < chi_right) {
             merge_bins(i - 1);
           } else {
             merge_bins(i);
           }
         }
         merged = true;
         break;
       }
     }
   }
 }
 
 // Calcula WoE e IV
 void calculate_woe_iv() {
   double total_pos = 0.0;
   double total_neg = 0.0;
   for (auto& b : bins) {
     total_pos += b.count_pos;
     total_neg += b.count_neg;
   }
   
   if (total_pos == 0 || total_neg == 0) {
     for (auto& b : bins) {
       b.woe = 0.0;
       b.iv = 0.0;
     }
     return;
   }
   
   for (auto& b : bins) {
     double pos_rate = std::max((double)b.count_pos / total_pos, EPSILON);
     double neg_rate = std::max((double)b.count_neg / total_neg, EPSILON);
     b.woe = std::log(pos_rate / neg_rate);
     b.iv = (pos_rate - neg_rate) * b.woe;
   }
 }
 
 // Verifica monotonicidade pelo WoE
 bool is_monotonic() const {
   if (bins.size() <= 2) return true;
   bool increasing = (bins[1].woe >= bins[0].woe);
   for (size_t i = 2; i < bins.size(); ++i) {
     if ((increasing && bins[i].woe < bins[i - 1].woe - EPSILON) ||
         (!increasing && bins[i].woe > bins[i - 1].woe + EPSILON)) {
       return false;
     }
   }
   return true;
 }
 
 // Binagem inicial via quantis
 void initial_binning() {
   std::vector<std::pair<double,int>> sorted_data(feature.size());
   for (size_t i = 0; i < feature.size(); ++i) {
     sorted_data[i] = {feature[i], target[i]};
   }
   std::sort(sorted_data.begin(), sorted_data.end(),
             [](const std::pair<double,int>& a, const std::pair<double,int>& b) {
               return a.first < b.first;
             });
   
   bins.clear();
   size_t total_records = sorted_data.size();
   size_t records_per_bin = std::max((size_t)1, total_records / (size_t)max_n_prebins);
   
   size_t start = 0;
   while (start < total_records) {
     size_t end = std::min(start + records_per_bin, total_records);
     Bin bin;
     bin.lower_bound = (start == 0) ? -std::numeric_limits<double>::infinity() : sorted_data[start].first;
     bin.upper_bound = (end == total_records) ? std::numeric_limits<double>::infinity() : sorted_data[end - 1].first;
     bin.count = (int)(end - start);
     bin.count_pos = 0;
     bin.count_neg = 0;
     
     for (size_t i = start; i < end; ++i) {
       if (sorted_data[i].second == 1) {
         bin.count_pos++;
       } else {
         bin.count_neg++;
       }
     }
     
     bins.push_back(std::move(bin));
     start = end;
   }
 }
 
 // Algoritmo ChiMerge principal
 void chi_merge() {
   double prev_total_iv = 0.0;
   for (iterations_run = 0; iterations_run < max_iterations; ++iterations_run) {
     if (bins.size() <= (size_t)min_bins) {
       converged = true;
       break;
     }
     
     double min_chi_square = std::numeric_limits<double>::max();
     size_t merge_index = 0;
     
     for (size_t i = 0; i < bins.size() - 1; ++i) {
       double chi_square = calculate_chi_square(bins[i], bins[i + 1]);
       if (chi_square < min_chi_square) {
         min_chi_square = chi_square;
         merge_index = i;
       }
     }
     
     merge_bins(merge_index);
     merge_zero_bins();
     calculate_woe_iv();
     
     double total_iv = 0.0;
     for (auto& b : bins) {
       total_iv += b.iv;
     }
     
     if (std::fabs(total_iv - prev_total_iv) < convergence_threshold) {
       converged = true;
       break;
     }
     
     prev_total_iv = total_iv;
     
     if (bins.size() <= (size_t)max_bins && is_monotonic()) {
       converged = true;
       break;
     }
   }
 }
 
 // Mescla bins raros
 void merge_rare_bins() {
   double total_count = (double)feature.size();
   bool merged_bins = true;
   while (merged_bins && bins.size() > (size_t)min_bins) {
     merged_bins = false;
     for (size_t i = 0; i < bins.size(); ) {
       double freq = (double)bins[i].count / total_count;
       if (freq < bin_cutoff) {
         if (i == 0) {
           merge_bins(0);
         } else {
           merge_bins(i - 1);
         }
         merged_bins = true;
         merge_zero_bins();
         calculate_woe_iv();
         i = 0; // reinicia a checagem após merge
       } else {
         ++i;
       }
     }
   }
 }
 
 // Impõe monotonicidade do WoE
 void enforce_monotonicity() {
   while (!is_monotonic() && bins.size() > (size_t)min_bins) {
     double min_chi_square = std::numeric_limits<double>::max();
     size_t merge_index = 0;
     
     for (size_t i = 0; i < bins.size() - 1; ++i) {
       double chi_square = calculate_chi_square(bins[i], bins[i + 1]);
       if (chi_square < min_chi_square) {
         min_chi_square = chi_square;
         merge_index = i;
       }
     }
     
     merge_bins(merge_index);
     merge_zero_bins();
     calculate_woe_iv();
   }
 }
 
public:
 // Construtor com validações
 OptimalBinningNumericalCM(
   const std::vector<double>& feature_,
   const std::vector<int>& target_,
   int min_bins_,
   int max_bins_,
   double bin_cutoff_,
   int max_n_prebins_,
   double convergence_threshold_,
   int max_iterations_
 ) : feature(feature_),
 target(target_),
 min_bins(min_bins_),
 max_bins(max_bins_),
 bin_cutoff(bin_cutoff_),
 max_n_prebins(max_n_prebins_),
 convergence_threshold(convergence_threshold_),
 max_iterations(max_iterations_),
 converged(false),
 iterations_run(0) {
   
   if (feature.size() != target.size()) {
     throw std::invalid_argument("Feature e target devem ter o mesmo tamanho.");
   }
   if (min_bins < 2 || max_bins < min_bins) {
     throw std::invalid_argument("min_bins >= 2 e max_bins >= min_bins.");
   }
   if (bin_cutoff <= 0 || bin_cutoff >= 1) {
     throw std::invalid_argument("bin_cutoff deve estar entre 0 e 1.");
   }
   if (max_n_prebins < max_bins) {
     throw std::invalid_argument("max_n_prebins deve ser >= max_bins.");
   }
   for (size_t i = 0; i < target.size(); ++i) {
     if (target[i] != 0 && target[i] != 1) {
       throw std::invalid_argument("Target deve conter apenas 0 ou 1.");
     }
   }
   for (double val : feature) {
     if (std::isnan(val) || std::isinf(val)) {
       throw std::invalid_argument("Feature contém valores NaN ou Inf.");
     }
   }
 }
 
 // Execução do binning
 void fit() {
   double total_pos = 0.0;
   for (int val : target) total_pos += val;
   double total_neg = (double)target.size() - total_pos;
   
   if (total_pos == 0 || total_neg == 0) {
     throw std::runtime_error("Target é constante (só 0s ou só 1s), impossibilitando binning.");
   }
   
   // Contagem de valores únicos
   std::vector<double> unique_values(feature);
   std::sort(unique_values.begin(), unique_values.end());
   unique_values.erase(std::unique(unique_values.begin(), unique_values.end()), unique_values.end());
   
   int num_unique_values = (int)unique_values.size();
   
   if (num_unique_values <= 2) {
     // Caso trivial: criar bins pelos valores únicos
     bins.clear();
     for (int i = 0; i < num_unique_values; ++i) {
       Bin bin;
       bin.lower_bound = (i == 0) ? -std::numeric_limits<double>::infinity() : unique_values[i - 1];
       bin.upper_bound = unique_values[i];
       bin.count = 0;
       bin.count_pos = 0;
       bin.count_neg = 0;
       bins.push_back(std::move(bin));
     }
     for (size_t i = 0; i < feature.size(); ++i) {
       double val = feature[i];
       int tgt = target[i];
       auto it = std::upper_bound(unique_values.begin(), unique_values.end(), val);
       size_t bin_index = (size_t)std::distance(unique_values.begin(), it);
       bin_index = (bin_index == 0) ? 0 : bin_index - 1;
       bins[bin_index].count++;
       if (tgt == 1) {
         bins[bin_index].count_pos++;
       } else {
         bins[bin_index].count_neg++;
       }
     }
     merge_zero_bins();
     calculate_woe_iv();
     converged = true;
     iterations_run = 0;
     return;
   }
   
   // Binagem inicial e merges
   initial_binning();
   merge_zero_bins();
   calculate_woe_iv();
   
   chi_merge();
   merge_rare_bins();
   enforce_monotonicity();
 }
 
 // Resultados do binning
 Rcpp::List get_results() const {
   // Inicializa os vetores com o tamanho apropriado
   Rcpp::StringVector bin_names(bins.size());
   Rcpp::NumericVector bin_woe(bins.size());
   Rcpp::NumericVector bin_iv(bins.size());
   Rcpp::IntegerVector bin_count(bins.size());
   Rcpp::IntegerVector bin_count_pos(bins.size());
   Rcpp::IntegerVector bin_count_neg(bins.size());
   Rcpp::NumericVector bin_cutpoints;  // Tamanho dinâmico
   
   for (size_t i = 0; i < bins.size(); ++i) {
     const auto& bin = bins[i];
     std::string bin_name = 
       (bin.lower_bound == -std::numeric_limits<double>::infinity() ? "(-Inf" : "(" + std::to_string(bin.lower_bound)) +
       ";" + (bin.upper_bound == std::numeric_limits<double>::infinity() ? "+Inf]" : std::to_string(bin.upper_bound) + "]");
     
     bin_names[i] = bin_name;
     bin_woe[i] = bin.woe;
     bin_iv[i] = bin.iv;
     bin_count[i] = bin.count;
     bin_count_pos[i] = bin.count_pos;
     bin_count_neg[i] = bin.count_neg;
     
     if (bin.upper_bound != std::numeric_limits<double>::infinity() && i != bins.size() - 1) {
       bin_cutpoints.push_back(bin.upper_bound);
     }
   }
   
   return Rcpp::List::create(
     Rcpp::Named("bins") = bin_names,
     Rcpp::Named("woe") = bin_woe,
     Rcpp::Named("iv") = bin_iv,
     Rcpp::Named("count") = bin_count,
     Rcpp::Named("count_pos") = bin_count_pos,
     Rcpp::Named("count_neg") = bin_count_neg,
     Rcpp::Named("cutpoints") = bin_cutpoints,
     Rcpp::Named("converged") = converged,
     Rcpp::Named("iterations") = iterations_run
   );
 }
};

 
 
 
//' @title Binning Ótimo para Variáveis Numéricas usando ChiMerge (Versão Aprimorada)
//'
//' @description
//' Implementa um algoritmo de binning ótimo para variáveis numéricas utilizando o método ChiMerge,
//' calculando WoE (Weight of Evidence) e IV (Information Value). Este código foi otimizado
//' em legibilidade, eficiência e robustez, mantendo compatibilidade de tipos e nomes.
//'
//' @param target Vetor inteiro binário (0/1) do target.
//' @param feature Vetor numérico de valores da feature a ser binada.
//' @param min_bins Número mínimo de bins (default: 3).
//' @param max_bins Número máximo de bins (default: 5).
//' @param bin_cutoff Frequência mínima (proporção) de observações em cada bin (default: 0.05).
//' @param max_n_prebins Número máximo de pré-bins para discretização inicial (default: 20).
//' @param convergence_threshold Limite de convergência do algoritmo (default: 1e-6).
//' @param max_iterations Número máximo de iterações (default: 1000).
//'
//' @return Uma lista com:
//' \itemize{
//'   \item bins: Vetor de nomes dos bins.
//'   \item woe: Vetor de WoE por bin.
//'   \item iv: Vetor de IV por bin.
//'   \item count: Contagem total por bin.
//'   \item count_pos: Contagem de casos positivos (target=1) por bin.
//'   \item count_neg: Contagem de casos negativos (target=0) por bin.
//'   \item cutpoints: Pontos de corte utilizados para criar os bins.
//'   \item converged: Booleano indicando se o algoritmo convergiu.
//'   \item iterations: Número de iterações executadas.
//' }
//'
//' @details
//' O algoritmo segue estes passos:
//' 1. Discretização inicial em max_n_prebins via quantis.
//' 2. Mesclagem iterativa de bins adjacentes com base na estatística Qui-quadrado.
//' 3. Mesclagem de bins com contagens zero em alguma classe.
//' 4. Mesclagem de bins raros (baseado em bin_cutoff).
//' 5. Cálculo de WoE e IV para cada bin final.
//' 6. Aplicação de monotonicidade (se possível).
//'
//' Referências:
//' \itemize{
//'   \item Kerber, R. (1992). ChiMerge: Discretization of Numeric Attributes. AAAI Press.
//'   \item Zeng, G. (2014). A necessary condition for a good binning algorithm in credit scoring. Applied Mathematical Sciences, 8(65), 3229-3242.
//' }
//'
//' @examples
//' \dontrun{
//' set.seed(123)
//' n <- 10000
//' feature <- rnorm(n)
//' target <- rbinom(n, 1, plogis(0.5 * feature))
//' result <- optimal_binning_numerical_cm(target, feature, min_bins = 3, max_bins = 5)
//' print(result)
//' }
//'
//' @export 
// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_cm(
   Rcpp::IntegerVector target,
   Rcpp::NumericVector feature,
   int min_bins = 3,
   int max_bins = 5,
   double bin_cutoff = 0.05,
   int max_n_prebins = 20,
   double convergence_threshold = 1e-6,
   int max_iterations = 1000
) {
 if (target.size() != feature.size()) {
   Rcpp::stop("Target e feature devem ter o mesmo tamanho.");
 }
 for (int i = 0; i < target.size(); ++i) {
   if (IntegerVector::is_na(target[i])) {
     Rcpp::stop("Target contém valores NA.");
   }
   if (target[i] != 0 && target[i] != 1) {
     Rcpp::stop("Target deve conter apenas 0 ou 1.");
   }
 }
 if (min_bins < 2 || max_bins < min_bins) {
   Rcpp::stop("min_bins >= 2 e max_bins >= min_bins.");
 }
 if (bin_cutoff <= 0 || bin_cutoff >= 1) {
   Rcpp::stop("bin_cutoff deve estar entre 0 e 1.");
 }
 if (max_n_prebins < max_bins) {
   Rcpp::stop("max_n_prebins deve ser >= max_bins.");
 }
 if (Rcpp::is_true(any(is_na(feature)))) {
   Rcpp::stop("Feature contém valores NA.");
 }
 if (Rcpp::is_true(any(!is_finite(feature)))) {
   Rcpp::stop("Feature contém valores NaN ou Inf.");
 }
 
 std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
 std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
 
 try {
   OptimalBinningNumericalCM binner(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins, convergence_threshold, max_iterations);
   binner.fit();
   return binner.get_results();
 } catch (const std::exception& e) {
   Rcpp::stop("Erro no binning ótimo: " + std::string(e.what()));
 }
}
 
       











// // [[Rcpp::depends(Rcpp)]]
// //' @title Binning Ótimo para Variáveis Numéricas usando ChiMerge (Versão Aprimorada)
// //'
// //' @description
// //' Implementa um algoritmo de binning ótimo para variáveis numéricas utilizando o método ChiMerge,
// //' calculando WoE (Weight of Evidence) e IV (Information Value). Este código foi otimizado
// //' em legibilidade, eficiência e robustez, mantendo compatibilidade de tipos e nomes.
// //'
// //' @param target Vetor inteiro binário (0/1) do target.
// //' @param feature Vetor numérico de valores da feature a ser binada.
// //' @param min_bins Número mínimo de bins (default: 3).
// //' @param max_bins Número máximo de bins (default: 5).
// //' @param bin_cutoff Frequência mínima (proporção) de observações em cada bin (default: 0.05).
// //' @param max_n_prebins Número máximo de pré-bins para discretização inicial (default: 20).
// //' @param convergence_threshold Limite de convergência do algoritmo (default: 1e-6).
// //' @param max_iterations Número máximo de iterações (default: 1000).
// //'
// //' @return Uma lista com:
// //' \itemize{
// //'   \item bins: Vetor de nomes dos bins.
// //'   \item woe: Vetor de WoE por bin.
// //'   \item iv: Vetor de IV por bin.
// //'   \item count: Contagem total por bin.
// //'   \item count_pos: Contagem de casos positivos (target=1) por bin.
// //'   \item count_neg: Contagem de casos negativos (target=0) por bin.
// //'   \item cutpoints: Pontos de corte utilizados para criar os bins.
// //'   \item converged: Booleano indicando se o algoritmo convergiu.
// //'   \item iterations: Número de iterações executadas.
// //' }
// //'
// //' @details
// //' O algoritmo segue estes passos:
// //' 1. Discretização inicial em max_n_prebins via quantis.
// //' 2. Mesclagem iterativa de bins adjacentes com base na estatística Qui-quadrado.
// //' 3. Mesclagem de bins com contagens zero em alguma classe.
// //' 4. Mesclagem de bins raros (baseado em bin_cutoff).
// //' 5. Cálculo de WoE e IV para cada bin final.
// //' 6. Aplicação de monotonicidade (se possível).
// //'
// //' Referências:
// //' \itemize{
// //'   \item Kerber, R. (1992). ChiMerge: Discretization of Numeric Attributes. AAAI Press.
// //'   \item Zeng, G. (2014). A necessary condition for a good binning algorithm in credit scoring. Applied Mathematical Sciences, 8(65), 3229-3242.
// //' }
// //'
// //' @examples
// //' \dontrun{
// //' set.seed(123)
// //' n <- 10000
// //' feature <- rnorm(n)
// //' target <- rbinom(n, 1, plogis(0.5 * feature))
// //' result <- optimal_binning_numerical_cm(target, feature, min_bins = 3, max_bins = 5)
// //' print(result)
// //' }
// //'
// //' @export
// 
// #include <Rcpp.h>
// #include <vector>
// #include <algorithm>
// #include <cmath>
// #include <limits>
// #include <string>
// #include <stdexcept>
// #include <numeric>
// #include <iterator>
// 
// using namespace Rcpp;
// 
// // Classe para binning ótimo numérico via ChiMerge
// class OptimalBinningNumericalCM {
// private:
//  // Dados de entrada
//  const std::vector<double> feature;
//  const std::vector<int> target;
//  int min_bins;
//  int max_bins;
//  double bin_cutoff;
//  int max_n_prebins;
//  double convergence_threshold;
//  int max_iterations;
//  
//  // Estrutura do bin
//  struct Bin {
//    double lower_bound;
//    double upper_bound;
//    int count;
//    int count_pos;
//    int count_neg;
//    double woe;
//    double iv;
//    
//    Bin()
//      : lower_bound(0.0), upper_bound(0.0), count(0), count_pos(0), count_neg(0), woe(0.0), iv(0.0) {}
//  };
//  
//  // Resultado
//  std::vector<Bin> bins;
//  bool converged;
//  int iterations_run;
//  
//  static constexpr double EPSILON = 1e-10;
//  
//  // Calcula Qui-quadrado entre dois bins
//  inline double calculate_chi_square(const Bin& bin1, const Bin& bin2) const {
//    int total_pos = bin1.count_pos + bin2.count_pos;
//    int total_neg = bin1.count_neg + bin2.count_neg;
//    int total = total_pos + total_neg;
//    
//    if (total == 0) {
//      // Evita divisão por zero
//      return std::numeric_limits<double>::max();
//    }
//    
//    double expected_pos1 = std::max((double)bin1.count * (double)total_pos / (double)total, EPSILON);
//    double expected_neg1 = std::max((double)bin1.count * (double)total_neg / (double)total, EPSILON);
//    double expected_pos2 = std::max((double)bin2.count * (double)total_pos / (double)total, EPSILON);
//    double expected_neg2 = std::max((double)bin2.count * (double)total_neg / (double)total, EPSILON);
//    
//    double chi_square =
//      std::pow(bin1.count_pos - expected_pos1, 2.0) / expected_pos1 +
//      std::pow(bin1.count_neg - expected_neg1, 2.0) / expected_neg1 +
//      std::pow(bin2.count_pos - expected_pos2, 2.0) / expected_pos2 +
//      std::pow(bin2.count_neg - expected_neg2, 2.0) / expected_neg2;
//    
//    // Penaliza bins com contagens zero para encorajar mesclas
//    if (bin1.count_pos == 0 || bin1.count_neg == 0 || bin2.count_pos == 0 || bin2.count_neg == 0) {
//      chi_square = 0.0;
//    }
//    
//    return chi_square;
//  }
//  
//  // Mescla bins adjacentes
//  inline void merge_bins(size_t index) {
//    if (index >= bins.size() - 1) return;
//    Bin& left = bins[index];
//    const Bin& right = bins[index + 1];
//    
//    left.upper_bound = right.upper_bound;
//    left.count += right.count;
//    left.count_pos += right.count_pos;
//    left.count_neg += right.count_neg;
//    
//    bins.erase(bins.begin() + index + 1);
//  }
//  
//  // Mescla bins com contagem zero em alguma classe
//  void merge_zero_bins() {
//    bool merged = true;
//    while (merged && bins.size() > 1) {
//      merged = false;
//      for (size_t i = 0; i < bins.size(); ++i) {
//        if (bins[i].count_pos == 0 || bins[i].count_neg == 0) {
//          if (i == 0) {
//            merge_bins(i);
//          } else if (i == bins.size() - 1) {
//            merge_bins(i - 1);
//          } else {
//            double chi_left = calculate_chi_square(bins[i - 1], bins[i]);
//            double chi_right = calculate_chi_square(bins[i], bins[i + 1]);
//            if (chi_left < chi_right) {
//              merge_bins(i - 1);
//            } else {
//              merge_bins(i);
//            }
//          }
//          merged = true;
//          break;
//        }
//      }
//    }
//  }
//  
//  // Calcula WoE e IV
//  void calculate_woe_iv() {
//    double total_pos = 0.0;
//    double total_neg = 0.0;
//    for (auto& b : bins) {
//      total_pos += b.count_pos;
//      total_neg += b.count_neg;
//    }
//    
//    if (total_pos == 0 || total_neg == 0) {
//      for (auto& b : bins) {
//        b.woe = 0.0;
//        b.iv = 0.0;
//      }
//      return;
//    }
//    
//    for (auto& b : bins) {
//      double pos_rate = std::max((double)b.count_pos / total_pos, EPSILON);
//      double neg_rate = std::max((double)b.count_neg / total_neg, EPSILON);
//      b.woe = std::log(pos_rate / neg_rate);
//      b.iv = (pos_rate - neg_rate) * b.woe;
//    }
//  }
//  
//  // Verifica monotonicidade pelo WoE
//  bool is_monotonic() const {
//    if (bins.size() <= 2) return true;
//    bool increasing = (bins[1].woe >= bins[0].woe);
//    for (size_t i = 2; i < bins.size(); ++i) {
//      if ((increasing && bins[i].woe < bins[i - 1].woe - EPSILON) ||
//          (!increasing && bins[i].woe > bins[i - 1].woe + EPSILON)) {
//        return false;
//      }
//    }
//    return true;
//  }
//  
//  // Binagem inicial via quantis
//  void initial_binning() {
//    std::vector<std::pair<double,int>> sorted_data(feature.size());
//    for (size_t i = 0; i < feature.size(); ++i) {
//      sorted_data[i] = {feature[i], target[i]};
//    }
//    std::sort(sorted_data.begin(), sorted_data.end(),
//              [](const std::pair<double,int>& a, const std::pair<double,int>& b) {
//                return a.first < b.first;
//              });
//    
//    bins.clear();
//    bins.reserve((size_t)max_n_prebins);
//    size_t total_records = sorted_data.size();
//    size_t records_per_bin = std::max((size_t)1, total_records / (size_t)max_n_prebins);
//    
//    size_t start = 0;
//    while (start < total_records) {
//      size_t end = std::min(start + records_per_bin, total_records);
//      Bin bin;
//      bin.lower_bound = (start == 0) ? -std::numeric_limits<double>::infinity() : sorted_data[start].first;
//      bin.upper_bound = (end == total_records) ? std::numeric_limits<double>::infinity() : sorted_data[end - 1].first;
//      bin.count = (int)(end - start);
//      bin.count_pos = 0;
//      bin.count_neg = 0;
//      
//      for (size_t i = start; i < end; ++i) {
//        if (sorted_data[i].second == 1) {
//          bin.count_pos++;
//        } else {
//          bin.count_neg++;
//        }
//      }
//      
//      bins.push_back(std::move(bin));
//      start = end;
//    }
//  }
//  
//  // Algoritmo ChiMerge principal
//  void chi_merge() {
//    double prev_total_iv = 0.0;
//    for (iterations_run = 0; iterations_run < max_iterations; ++iterations_run) {
//      if (bins.size() <= (size_t)min_bins) {
//        converged = true;
//        break;
//      }
//      
//      double min_chi_square = std::numeric_limits<double>::max();
//      size_t merge_index = 0;
//      
//      for (size_t i = 0; i < bins.size() - 1; ++i) {
//        double chi_square = calculate_chi_square(bins[i], bins[i + 1]);
//        if (chi_square < min_chi_square) {
//          min_chi_square = chi_square;
//          merge_index = i;
//        }
//      }
//      
//      merge_bins(merge_index);
//      merge_zero_bins();
//      calculate_woe_iv();
//      
//      double total_iv = 0.0;
//      for (auto& b : bins) {
//        total_iv += b.iv;
//      }
//      
//      if (std::fabs(total_iv - prev_total_iv) < convergence_threshold) {
//        converged = true;
//        break;
//      }
//      
//      prev_total_iv = total_iv;
//      
//      if (bins.size() <= (size_t)max_bins && is_monotonic()) {
//        converged = true;
//        break;
//      }
//    }
//  }
//  
//  // Mescla bins raros
//  void merge_rare_bins() {
//    double total_count = (double)feature.size();
//    bool merged_bins = true;
//    while (merged_bins && bins.size() > (size_t)min_bins) {
//      merged_bins = false;
//      for (size_t i = 0; i < bins.size(); ) {
//        double freq = (double)bins[i].count / total_count;
//        if (freq < bin_cutoff) {
//          if (i == 0) {
//            merge_bins(0);
//          } else {
//            merge_bins(i - 1);
//          }
//          merged_bins = true;
//          merge_zero_bins();
//          calculate_woe_iv();
//          i = 0; // reinicia a checagem após merge
//        } else {
//          ++i;
//        }
//      }
//    }
//  }
//  
//  // Impõe monotonicidade do WoE
//  void enforce_monotonicity() {
//    while (!is_monotonic() && bins.size() > (size_t)min_bins) {
//      double min_chi_square = std::numeric_limits<double>::max();
//      size_t merge_index = 0;
//      
//      for (size_t i = 0; i < bins.size() - 1; ++i) {
//        double chi_square = calculate_chi_square(bins[i], bins[i + 1]);
//        if (chi_square < min_chi_square) {
//          min_chi_square = chi_square;
//          merge_index = i;
//        }
//      }
//      
//      merge_bins(merge_index);
//      merge_zero_bins();
//      calculate_woe_iv();
//    }
//  }
//  
// public:
//  // Construtor com validações
//  OptimalBinningNumericalCM(
//    const std::vector<double>& feature_,
//    const std::vector<int>& target_,
//    int min_bins_,
//    int max_bins_,
//    double bin_cutoff_,
//    int max_n_prebins_,
//    double convergence_threshold_,
//    int max_iterations_
//  ) : feature(feature_),
//  target(target_),
//  min_bins(min_bins_),
//  max_bins(max_bins_),
//  bin_cutoff(bin_cutoff_),
//  max_n_prebins(max_n_prebins_),
//  convergence_threshold(convergence_threshold_),
//  max_iterations(max_iterations_),
//  converged(false),
//  iterations_run(0) {
//    
//    if (feature.size() != target.size()) {
//      throw std::invalid_argument("Feature e target devem ter o mesmo tamanho.");
//    }
//    if (min_bins < 2 || max_bins < min_bins) {
//      throw std::invalid_argument("min_bins >= 2 e max_bins >= min_bins.");
//    }
//    if (bin_cutoff <= 0 || bin_cutoff >= 1) {
//      throw std::invalid_argument("bin_cutoff deve estar entre 0 e 1.");
//    }
//    if (max_n_prebins < max_bins) {
//      throw std::invalid_argument("max_n_prebins deve ser >= max_bins.");
//    }
//    for (size_t i = 0; i < target.size(); ++i) {
//      if (target[i] != 0 && target[i] != 1) {
//        throw std::invalid_argument("Target deve conter apenas 0 ou 1.");
//      }
//    }
//    for (double val : feature) {
//      if (std::isnan(val) || std::isinf(val)) {
//        throw std::invalid_argument("Feature contém valores NaN ou Inf.");
//      }
//    }
//  }
//  
//  // Execução do binning
//  void fit() {
//    double total_pos = 0.0;
//    for (int val : target) total_pos += val;
//    double total_neg = (double)target.size() - total_pos;
//    
//    if (total_pos == 0 || total_neg == 0) {
//      throw std::runtime_error("Target é constante (só 0s ou só 1s), impossibilitando binning.");
//    }
//    
//    // Contagem de valores únicos
//    std::vector<double> unique_values(feature);
//    std::sort(unique_values.begin(), unique_values.end());
//    unique_values.erase(std::unique(unique_values.begin(), unique_values.end()), unique_values.end());
//    
//    int num_unique_values = (int)unique_values.size();
//    
//    if (num_unique_values <= 2) {
//      // Caso trivial: criar bins pelos valores únicos
//      bins.clear();
//      bins.reserve((size_t)num_unique_values);
//      for (int i = 0; i < num_unique_values; ++i) {
//        Bin bin;
//        bin.lower_bound = (i == 0) ? -std::numeric_limits<double>::infinity() : unique_values[i - 1];
//        bin.upper_bound = unique_values[i];
//        bin.count = 0;
//        bin.count_pos = 0;
//        bin.count_neg = 0;
//        bins.push_back(std::move(bin));
//      }
//      for (size_t i = 0; i < feature.size(); ++i) {
//        double val = feature[i];
//        int tgt = target[i];
//        auto it = std::upper_bound(unique_values.begin(), unique_values.end(), val);
//        size_t bin_index = (size_t)std::distance(unique_values.begin(), it);
//        bin_index = (bin_index == 0) ? 0 : bin_index - 1;
//        bins[bin_index].count++;
//        if (tgt == 1) {
//          bins[bin_index].count_pos++;
//        } else {
//          bins[bin_index].count_neg++;
//        }
//      }
//      merge_zero_bins();
//      calculate_woe_iv();
//      converged = true;
//      iterations_run = 0;
//      return;
//    }
//    
//    // Binagem inicial e merges
//    initial_binning();
//    merge_zero_bins();
//    calculate_woe_iv();
//    
//    chi_merge();
//    merge_rare_bins();
//    enforce_monotonicity();
//  }
//  
//  // Resultados do binning
//  Rcpp::List get_results() const {
//    Rcpp::StringVector bin_names;
//    bin_names.reserve(bins.size());
//    Rcpp::NumericVector bin_woe(bins.size());
//    Rcpp::NumericVector bin_iv(bins.size());
//    Rcpp::IntegerVector bin_count(bins.size());
//    Rcpp::IntegerVector bin_count_pos(bins.size());
//    Rcpp::IntegerVector bin_count_neg(bins.size());
//    Rcpp::NumericVector bin_cutpoints;
//    bin_cutpoints.reserve(bins.size());
//    
//    for (size_t i = 0; i < bins.size(); ++i) {
//      const auto& bin = bins[i];
//      std::string bin_name = 
//        (bin.lower_bound == -std::numeric_limits<double>::infinity() ? "(-Inf" : "(" + std::to_string(bin.lower_bound)) +
//        ";" + (bin.upper_bound == std::numeric_limits<double>::infinity() ? "+Inf]" : std::to_string(bin.upper_bound) + "]");
//      
//      bin_names.push_back(bin_name);
//      bin_woe[i] = bin.woe;
//      bin_iv[i] = bin.iv;
//      bin_count[i] = bin.count;
//      bin_count_pos[i] = bin.count_pos;
//      bin_count_neg[i] = bin.count_neg;
//      
//      if (bin.upper_bound != std::numeric_limits<double>::infinity() && i != bins.size() - 1) {
//        bin_cutpoints.push_back(bin.upper_bound);
//      }
//    }
//    
//    return Rcpp::List::create(
//      Rcpp::Named("bins") = bin_names,
//      Rcpp::Named("woe") = bin_woe,
//      Rcpp::Named("iv") = bin_iv,
//      Rcpp::Named("count") = bin_count,
//      Rcpp::Named("count_pos") = bin_count_pos,
//      Rcpp::Named("count_neg") = bin_count_neg,
//      Rcpp::Named("cutpoints") = bin_cutpoints,
//      Rcpp::Named("converged") = converged,
//      Rcpp::Named("iterations") = iterations_run
//    );
//  }
// };
// 
// //' @rdname optimal_binning_numerical_cm
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_numerical_cm(
//    Rcpp::IntegerVector target,
//    Rcpp::NumericVector feature,
//    int min_bins = 3,
//    int max_bins = 5,
//    double bin_cutoff = 0.05,
//    int max_n_prebins = 20,
//    double convergence_threshold = 1e-6,
//    int max_iterations = 1000
// ) {
//  if (target.size() != feature.size()) {
//    Rcpp::stop("Target e feature devem ter o mesmo tamanho.");
//  }
//  for (int i = 0; i < target.size(); ++i) {
//    if (IntegerVector::is_na(target[i])) {
//      Rcpp::stop("Target contém valores NA.");
//    }
//    if (target[i] != 0 && target[i] != 1) {
//      Rcpp::stop("Target deve conter apenas 0 ou 1.");
//    }
//  }
//  if (min_bins < 2 || max_bins < min_bins) {
//    Rcpp::stop("min_bins >= 2 e max_bins >= min_bins.");
//  }
//  if (bin_cutoff <= 0 || bin_cutoff >= 1) {
//    Rcpp::stop("bin_cutoff deve estar entre 0 e 1.");
//  }
//  if (max_n_prebins < max_bins) {
//    Rcpp::stop("max_n_prebins deve ser >= max_bins.");
//  }
//  if (Rcpp::is_true(any(is_na(feature)))) {
//    Rcpp::stop("Feature contém valores NA.");
//  }
//  if (Rcpp::is_true(any(!is_finite(feature)))) {
//    Rcpp::stop("Feature contém valores NaN ou Inf.");
//  }
//  
//  std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
//  std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
//  
//  try {
//    OptimalBinningNumericalCM binner(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins, convergence_threshold, max_iterations);
//    binner.fit();
//    return binner.get_results();
//  } catch (const std::exception& e) {
//    Rcpp::stop("Erro no binning ótimo: " + std::string(e.what()));
//  }
// }
// 
// /*
// Resumo das melhorias implementadas:
// - Adição de reservas de capacidade (reserve) em vetores para evitar realocações desnecessárias.
// - Uso de inline e const referências para otimizar acessos.
// - Melhor tratamento de erros e validação de entradas.
// - Comentários mais claros e organização do código.
// - Otimizações nos cálculos do Qui-quadrado, WoE e IV.
// - Melhoria no fluxo lógico, minimizando o retrabalho e a complexidade.
// - Garantia de robustez e legibilidade do código.
// */



// // Optimized Implementation
// #include <Rcpp.h>
// #include <vector>
// #include <algorithm>
// #include <cmath>
// #include <limits>
// #include <string>
// #include <stdexcept>
// #include <numeric> // Added for std::accumulate
// #include <iterator> // Added for iterators
// 
// using namespace Rcpp;
// 
// // Class implementing the Optimal Binning using ChiMerge method
// class OptimalBinningNumericalCM {
// private:
//   // Input data
//   std::vector<double> feature;
//   std::vector<int> target;
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
//   double convergence_threshold;
//   int max_iterations;
//   
//   // Structure representing a bin
//   struct Bin {
//     double lower_bound;
//     double upper_bound;
//     int count;
//     int count_pos;
//     int count_neg;
//     double woe;
//     double iv;
//     
//     Bin() : lower_bound(0), upper_bound(0), count(0), count_pos(0), count_neg(0), woe(0.0), iv(0.0) {}
//   };
//   
//   // Binning results
//   std::vector<Bin> bins;
//   bool converged;
//   int iterations_run;
//   
//   static constexpr double EPSILON = 1e-10;
//   
//   // Calculate chi-square statistic between two bins
//   double calculate_chi_square(const Bin& bin1, const Bin& bin2) const {
//     int total_pos = bin1.count_pos + bin2.count_pos;
//     int total_neg = bin1.count_neg + bin2.count_neg;
//     int total = total_pos + total_neg;
//     
//     if (total == 0) {
//       // Avoid division by zero; return large chi_square to avoid merging
//       return std::numeric_limits<double>::max();
//     }
//     
//     double expected_pos1 = std::max(static_cast<double>(bin1.count) * total_pos / total, EPSILON);
//     double expected_neg1 = std::max(static_cast<double>(bin1.count) * total_neg / total, EPSILON);
//     double expected_pos2 = std::max(static_cast<double>(bin2.count) * total_pos / total, EPSILON);
//     double expected_neg2 = std::max(static_cast<double>(bin2.count) * total_neg / total, EPSILON);
//     
//     double chi_square =
//       std::pow(bin1.count_pos - expected_pos1, 2) / expected_pos1 +
//       std::pow(bin1.count_neg - expected_neg1, 2) / expected_neg1 +
//       std::pow(bin2.count_pos - expected_pos2, 2) / expected_pos2 +
//       std::pow(bin2.count_neg - expected_neg2, 2) / expected_neg2;
//     
//     // Penalize bins with zero counts to encourage merging them
//     if (bin1.count_pos == 0 || bin1.count_neg == 0 || bin2.count_pos == 0 || bin2.count_neg == 0) {
//       chi_square = 0;
//     }
//     
//     return chi_square;
//   }
//   
//   // Merge two adjacent bins at the given index
//   void merge_bins(size_t index) {
//     if (index >= bins.size() - 1) return;
//     
//     Bin& left = bins[index];
//     Bin& right = bins[index + 1];
//     
//     left.upper_bound = right.upper_bound;
//     left.count += right.count;
//     left.count_pos += right.count_pos;
//     left.count_neg += right.count_neg;
//     
//     bins.erase(bins.begin() + index + 1);
//   }
//   
//   // Merge bins with zero counts in either class
//   void merge_zero_bins() {
//     bool merged = true;
//     while (merged && bins.size() > 1) {
//       merged = false;
//       for (size_t i = 0; i < bins.size(); ++i) {
//         if (bins[i].count_pos == 0 || bins[i].count_neg == 0) {
//           if (i == 0) {
//             merge_bins(i);
//           } else if (i == bins.size() - 1) {
//             merge_bins(i - 1);
//           } else {
//             // Merge with the bin that results in better chi-square
//             double chi_square_left = calculate_chi_square(bins[i - 1], bins[i]);
//             double chi_square_right = calculate_chi_square(bins[i], bins[i + 1]);
//             if (chi_square_left < chi_square_right) {
//               merge_bins(i - 1);
//             } else {
//               merge_bins(i);
//             }
//           }
//           merged = true;
//           break; // Restart loop after merging
//         }
//       }
//     }
//   }
//   
//   // Calculate Weight of Evidence (WoE) and Information Value (IV) for each bin
//   void calculate_woe_iv() {
//     double total_pos = std::accumulate(bins.begin(), bins.end(), 0.0, 
//                                        [](double sum, const Bin& bin) { return sum + bin.count_pos; });
//     double total_neg = std::accumulate(bins.begin(), bins.end(), 0.0, 
//                                        [](double sum, const Bin& bin) { return sum + bin.count_neg; });
//     
//     if (total_pos == 0 || total_neg == 0) {
//       // Cannot compute WoE and IV properly
//       // Set woe and iv to zero
//       for (auto& bin : bins) {
//         bin.woe = 0.0;
//         bin.iv = 0.0;
//       }
//       return;
//     }
//     
//     for (auto& bin : bins) {
//       double pos_rate = std::max(static_cast<double>(bin.count_pos) / total_pos, EPSILON);
//       double neg_rate = std::max(static_cast<double>(bin.count_neg) / total_neg, EPSILON);
//       
//       bin.woe = std::log(pos_rate / neg_rate);
//       bin.iv = (pos_rate - neg_rate) * bin.woe;
//     }
//   }
//   
//   // Check if the bins are monotonic in terms of WoE
//   bool is_monotonic() const {
//     if (bins.size() <= 2) return true;
//     
//     bool increasing = bins[1].woe >= bins[0].woe;
//     for (size_t i = 2; i < bins.size(); ++i) {
//       if ((increasing && bins[i].woe < bins[i - 1].woe - EPSILON) ||
//           (!increasing && bins[i].woe > bins[i - 1].woe + EPSILON)) {
//         return false;
//       }
//     }
//     return true;
//   }
//   
//   // Initial binning based on quantiles
//   void initial_binning() {
//     // Sort data
//     std::vector<std::pair<double, int>> sorted_data(feature.size());
//     for (size_t i = 0; i < feature.size(); ++i) {
//       sorted_data[i] = {feature[i], target[i]};
//     }
//     std::sort(sorted_data.begin(), sorted_data.end(),
//               [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
//                 return a.first < b.first;
//               });
//     
//     // Create initial bins based on quantiles
//     bins.clear();
//     size_t total_records = sorted_data.size();
//     size_t records_per_bin = std::max(static_cast<size_t>(1), total_records / max_n_prebins);
//     
//     size_t start = 0;
//     while (start < total_records) {
//       size_t end = std::min(start + records_per_bin, total_records);
//       Bin bin;
//       bin.lower_bound = (start == 0) ? -std::numeric_limits<double>::infinity() : sorted_data[start].first;
//       bin.upper_bound = (end == total_records) ? std::numeric_limits<double>::infinity() : sorted_data[end - 1].first;
//       bin.count = static_cast<int>(end - start);
//       bin.count_pos = 0;
//       bin.count_neg = 0;
//       
//       for (size_t i = start; i < end; ++i) {
//         if (sorted_data[i].second == 1) {
//           bin.count_pos++;
//         } else {
//           bin.count_neg++;
//         }
//       }
//       
//       bins.push_back(bin);
//       start = end;
//     }
//   }
//   
//   // Merge bins until desired number of bins is reached
//   void chi_merge() {
//     double prev_total_iv = 0;
//     for (iterations_run = 0; iterations_run < max_iterations; ++iterations_run) {
//       if (bins.size() <= static_cast<size_t>(min_bins)) {
//         converged = true;
//         break;
//       }
//       
//       double min_chi_square = std::numeric_limits<double>::max();
//       size_t merge_index = 0;
//       
//       for (size_t i = 0; i < bins.size() - 1; ++i) {
//         double chi_square = calculate_chi_square(bins[i], bins[i + 1]);
//         if (chi_square < min_chi_square) {
//           min_chi_square = chi_square;
//           merge_index = i;
//         }
//       }
//       
//       merge_bins(merge_index);
//       merge_zero_bins();
//       calculate_woe_iv();
//       
//       double total_iv = std::accumulate(bins.begin(), bins.end(), 0.0, 
//                                         [](double sum, const Bin& bin) { return sum + bin.iv; });
//       
//       if (std::abs(total_iv - prev_total_iv) < convergence_threshold) {
//         converged = true;
//         break;
//       }
//       
//       prev_total_iv = total_iv;
//       
//       if (bins.size() <= static_cast<size_t>(max_bins) && is_monotonic()) {
//         converged = true;
//         break;
//       }
//     }
//   }
//   
//   // Merge rare bins based on bin_cutoff
//   void merge_rare_bins() {
//     double total_count = static_cast<double>(feature.size());
//     bool merged_bins = true;
//     while (merged_bins && bins.size() > static_cast<size_t>(min_bins)) {
//       merged_bins = false;
//       for (size_t i = 0; i < bins.size(); ) {
//         if (static_cast<double>(bins[i].count) / total_count < bin_cutoff) {
//           if (i == 0) {
//             merge_bins(0);
//           } else {
//             merge_bins(i - 1);
//           }
//           merged_bins = true;
//           merge_zero_bins();
//           calculate_woe_iv();
//           i = 0; // Restart after merging
//         } else {
//           ++i;
//         }
//       }
//     }
//   }
//   
//   // Ensure monotonicity of WoE
//   void enforce_monotonicity() {
//     while (!is_monotonic() && bins.size() > static_cast<size_t>(min_bins)) {
//       double min_chi_square = std::numeric_limits<double>::max();
//       size_t merge_index = 0;
//       
//       for (size_t i = 0; i < bins.size() - 1; ++i) {
//         double chi_square = calculate_chi_square(bins[i], bins[i + 1]);
//         if (chi_square < min_chi_square) {
//           min_chi_square = chi_square;
//           merge_index = i;
//         }
//       }
//       
//       merge_bins(merge_index);
//       merge_zero_bins();
//       calculate_woe_iv();
//     }
//   }
//   
// public:
//   // Constructor with input validation
//   OptimalBinningNumericalCM(
//     const std::vector<double>& feature,
//     const std::vector<int>& target,
//     int min_bins = 3,
//     int max_bins = 5,
//     double bin_cutoff = 0.05,
//     int max_n_prebins = 20,
//     double convergence_threshold = 1e-6,
//     int max_iterations = 1000
//   ) : feature(feature), target(target), min_bins(min_bins), max_bins(max_bins),
//   bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins),
//   convergence_threshold(convergence_threshold), max_iterations(max_iterations),
//   converged(false), iterations_run(0) {
//     
//     if (feature.size() != target.size()) {
//       throw std::invalid_argument("Feature and target vectors must have the same length.");
//     }
//     
//     if (min_bins < 2 || max_bins < min_bins) {
//       throw std::invalid_argument("Invalid bin constraints: min_bins must be at least 2 and max_bins must be greater than or equal to min_bins.");
//     }
//     
//     if (bin_cutoff <= 0 || bin_cutoff >= 1) {
//       throw std::invalid_argument("bin_cutoff must be between 0 and 1.");
//     }
//     
//     if (max_n_prebins < max_bins) {
//       throw std::invalid_argument("max_n_prebins must be greater than or equal to max_bins.");
//     }
//     
//     // Check if target contains only 0 or 1
//     for (size_t i = 0; i < target.size(); ++i) {
//       if (target[i] != 0 && target[i] != 1) {
//         throw std::invalid_argument("Target values must be 0 or 1.");
//       }
//     }
//     
//     // Check for NaN or infinite values in feature
//     for (size_t i = 0; i < feature.size(); ++i) {
//       if (std::isnan(feature[i]) || std::isinf(feature[i])) {
//         throw std::invalid_argument("Feature vector contains NaN or infinite values.");
//       }
//     }
//   }
//   
//   // Fit the optimal binning model
//   void fit() {
//     // Compute total_pos and total_neg
//     double total_pos = std::accumulate(target.begin(), target.end(), 0.0);
//     double total_neg = static_cast<double>(target.size()) - total_pos;
//     
//     if (total_pos == 0 || total_neg == 0) {
//       throw std::runtime_error("Cannot perform binning: target variable is constant (all zeros or all ones).");
//     }
//     
//     // Count number of unique feature values
//     std::vector<double> unique_values = feature;
//     std::sort(unique_values.begin(), unique_values.end());
//     unique_values.erase(std::unique(unique_values.begin(), unique_values.end()), unique_values.end());
//     
//     int num_unique_values = static_cast<int>(unique_values.size());
//     
//     if (num_unique_values <= 2) {
//       // No need to optimize; create bins based on unique values
//       bins.clear();
//       bins.reserve(num_unique_values);
//       for (size_t i = 0; i < unique_values.size(); ++i) {
//         Bin bin;
//         bin.lower_bound = (i == 0) ? -std::numeric_limits<double>::infinity() : unique_values[i - 1];
//         bin.upper_bound = unique_values[i];
//         bin.count = 0;
//         bin.count_pos = 0;
//         bin.count_neg = 0;
//         bins.push_back(bin);
//       }
//       
//       // Assign counts to bins
//       for (size_t i = 0; i < feature.size(); ++i) {
//         double val = feature[i];
//         int tgt = target[i];
//         auto it = std::upper_bound(unique_values.begin(), unique_values.end(), val);
//         size_t bin_index = std::distance(unique_values.begin(), it);
//         bin_index = (bin_index == 0) ? 0 : bin_index - 1;
//         bins[bin_index].count++;
//         if (tgt == 1) {
//           bins[bin_index].count_pos++;
//         } else {
//           bins[bin_index].count_neg++;
//         }
//       }
//       merge_zero_bins();
//       calculate_woe_iv();
//       converged = true;
//       iterations_run = 0;
//       return;
//     }
//     
//     // Initial binning
//     initial_binning();
//     merge_zero_bins();
//     calculate_woe_iv();
//     
//     // ChiMerge algorithm
//     chi_merge();
//     
//     // Merge rare bins
//     merge_rare_bins();
//     
//     // Ensure monotonicity
//     enforce_monotonicity();
//   }
//   
//   // Retrieve the binning results
//   Rcpp::List get_results() const {
//     Rcpp::StringVector bin_names;
//     Rcpp::NumericVector bin_woe;
//     Rcpp::NumericVector bin_iv;
//     Rcpp::IntegerVector bin_count;
//     Rcpp::IntegerVector bin_count_pos;
//     Rcpp::IntegerVector bin_count_neg;
//     Rcpp::NumericVector bin_cutpoints;
//     
//     for (size_t i = 0; i < bins.size(); ++i) {
//       const auto& bin = bins[i];
//       std::string bin_name = (bin.lower_bound == -std::numeric_limits<double>::infinity() ? "(-Inf" : "(" + std::to_string(bin.lower_bound)) +
//         ";" + (bin.upper_bound == std::numeric_limits<double>::infinity() ? "+Inf]" : std::to_string(bin.upper_bound) + "]");
//       bin_names.push_back(bin_name);
//       bin_woe.push_back(bin.woe);
//       bin_iv.push_back(bin.iv);
//       bin_count.push_back(bin.count);
//       bin_count_pos.push_back(bin.count_pos);
//       bin_count_neg.push_back(bin.count_neg);
//       
//       if (bin.upper_bound != std::numeric_limits<double>::infinity() && i != bins.size() - 1) {
//         bin_cutpoints.push_back(bin.upper_bound);
//       }
//     }
//     
//     return Rcpp::List::create(
//       Rcpp::Named("bins") = bin_names,
//       Rcpp::Named("woe") = bin_woe,
//       Rcpp::Named("iv") = bin_iv,
//       Rcpp::Named("count") = bin_count,
//       Rcpp::Named("count_pos") = bin_count_pos,
//       Rcpp::Named("count_neg") = bin_count_neg,
//       Rcpp::Named("cutpoints") = bin_cutpoints,
//       Rcpp::Named("converged") = converged,
//       Rcpp::Named("iterations") = iterations_run
//     );
//   }
// };
// 
// //' @title Optimal Binning for Numerical Variables using ChiMerge
// //'
// //' @description
// //' This function implements an optimal binning algorithm for numerical variables using the ChiMerge approach with Weight of Evidence (WoE) and Information Value (IV) criteria.
// //'
// //' @param target An integer vector of binary target values (0 or 1).
// //' @param feature A numeric vector of feature values to be binned.
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff Minimum frequency of observations in each bin (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins for initial discretization (default: 20).
// //' @param convergence_threshold Threshold for convergence of the algorithm (default: 1e-6).
// //' @param max_iterations Maximum number of iterations for the algorithm (default: 1000).
// //'
// //' @return A list containing the following elements:
// //' \item{bins}{A character vector of bin names.}
// //' \item{woe}{A numeric vector of Weight of Evidence values for each bin.}
// //' \item{iv}{A numeric vector of Information Value for each bin.}
// //' \item{count}{An integer vector of total counts for each bin.}
// //' \item{count_pos}{An integer vector of positive target counts for each bin.}
// //' \item{count_neg}{An integer vector of negative target counts for each bin.}
// //' \item{cutpoints}{A numeric vector of cutpoints used to create the bins.}
// //' \item{converged}{A logical value indicating whether the algorithm converged.}
// //' \item{iterations}{An integer value indicating the number of iterations run.}
// //'
// //' @details
// //' The optimal binning algorithm for numerical variables uses the ChiMerge approach with Weight of Evidence (WoE) and Information Value (IV) to create bins that maximize the predictive power of the feature while maintaining interpretability.
// //'
// //' The algorithm follows these steps:
// //' 1. Initial discretization into max_n_prebins based on quantiles.
// //' 2. Iterative merging of adjacent bins based on chi-square statistic.
// //' 3. Merging of bins with zero counts in either class.
// //' 4. Merging of rare bins based on the bin_cutoff parameter.
// //' 5. Calculation of WoE and IV for each final bin.
// //' 6. Enforcement of monotonicity in WoE if possible.
// //'
// //' @references
// //' \itemize{
// //'   \item Kerber, R. (1992). ChiMerge: Discretization of Numeric Attributes. In Proceedings of the tenth national conference on Artificial intelligence (pp. 123-128). AAAI Press.
// //'   \item Zeng, G. (2014). A necessary condition for a good binning algorithm in credit scoring. Applied Mathematical Sciences, 8(65), 3229-3242.
// //' }
// //'
// //' @examples
// //' \dontrun{
// //' # Generate sample data
// //' set.seed(123)
// //' n <- 10000
// //' feature <- rnorm(n)
// //' target <- rbinom(n, 1, plogis(0.5 * feature))
// //'
// //' # Apply optimal binning
// //' result <- optimal_binning_numerical_cm(target, feature, min_bins = 3, max_bins = 5)
// //'
// //' # View binning results
// //' print(result)
// //' }
// //'
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_numerical_cm(
//    Rcpp::IntegerVector target,
//    Rcpp::NumericVector feature,
//    int min_bins = 3,
//    int max_bins = 5,
//    double bin_cutoff = 0.05,
//    int max_n_prebins = 20,
//    double convergence_threshold = 1e-6,
//    int max_iterations = 1000
// ) {
//  // Input validation
//  if (target.size() != feature.size()) {
//    Rcpp::stop("Target and feature vectors must have the same length.");
//  }
//  
//  // Check that target values are 0 or 1 and not NA
//  for (int i = 0; i < target.size(); ++i) {
//    if (IntegerVector::is_na(target[i])) {
//      Rcpp::stop("Target vector contains NA values.");
//    }
//    if (target[i] != 0 && target[i] != 1) {
//      Rcpp::stop("Target values must be 0 or 1.");
//    }
//  }
//  
//  if (min_bins < 2 || max_bins < min_bins) {
//    Rcpp::stop("Invalid bin constraints: min_bins must be at least 2 and max_bins must be greater than or equal to min_bins.");
//  }
//  
//  if (bin_cutoff <= 0 || bin_cutoff >= 1) {
//    Rcpp::stop("bin_cutoff must be between 0 and 1.");
//  }
//  
//  if (max_n_prebins < max_bins) {
//    Rcpp::stop("max_n_prebins must be greater than or equal to max_bins.");
//  }
//  
//  // Check for NA, NaN, or infinite values in feature
//  if (Rcpp::is_true(Rcpp::any(Rcpp::is_na(feature)))) {
//    Rcpp::stop("Feature vector contains NA values.");
//  }
//  
//  if (Rcpp::is_true(Rcpp::any(!Rcpp::is_finite(feature)))) {
//    Rcpp::stop("Feature vector contains NaN or infinite values.");
//  }
//  
//  // Convert Rcpp vectors to std::vector
//  std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
//  std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
//  
//  // Create and run the binner
//  try {
//    OptimalBinningNumericalCM binner(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins, convergence_threshold, max_iterations);
//    binner.fit();
//    
//    // Get and return the results
//    return binner.get_results();
//  } catch (const std::exception& e) {
//    Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
//  }
// }
// 
