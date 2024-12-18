// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

using namespace Rcpp;

struct BinInfo {
 std::vector<std::string> categories;
 int count = 0;
 int count_pos = 0;
 int count_neg = 0;
 double woe = 0.0;
 double iv = 0.0;
};

class OptimalBinningCategoricalGMB {
private:
 const std::vector<std::string>& feature;
 const std::vector<int>& target;
 int min_bins;
 int max_bins;
 double bin_cutoff;
 int max_n_prebins;
 std::string bin_separator;
 double convergence_threshold;
 int max_iterations;
 
 std::vector<BinInfo> bins;
 bool converged = false;
 int iterations_run = 0;
 
 static constexpr double EPSILON = 1e-10;
 
 // Calcula WoE para um bin com proteção contra log(0)
 inline double calculateWOE(int pos, int neg, int total_pos, int total_neg) const {
   double pos_rate = (pos + EPSILON) / (total_pos + EPSILON);
   double neg_rate = (neg + EPSILON) / (total_neg + EPSILON);
   return std::log(pos_rate / neg_rate);
 }
 
 // Calcula IV para um conjunto de bins
 double calculateIV(const std::vector<BinInfo>& bins, int total_pos, int total_neg) const {
   double iv = 0.0;
   for (const auto& bin : bins) {
     double pos_rate = (double)bin.count_pos / (double)total_pos;
     double neg_rate = (double)bin.count_neg / (double)total_neg;
     double local_iv = (pos_rate - neg_rate) * bin.woe;
     // Evita NaN
     if (std::isfinite(local_iv)) {
       iv += local_iv;
     }
   }
   return iv;
 }
 
 void validateInput() const {
   if (feature.size() != target.size()) {
     throw std::invalid_argument("Feature e target devem ter o mesmo tamanho.");
   }
   if (feature.empty()) {
     throw std::invalid_argument("Feature não pode ser vazia.");
   }
   if (min_bins < 2) {
     throw std::invalid_argument("min_bins deve ser >= 2.");
   }
   if (max_bins < min_bins) {
     throw std::invalid_argument("max_bins >= min_bins.");
   }
   if (bin_cutoff <= 0 || bin_cutoff >= 1) {
     throw std::invalid_argument("bin_cutoff deve estar entre 0 e 1.");
   }
   if (max_n_prebins < min_bins) {
     throw std::invalid_argument("max_n_prebins >= min_bins.");
   }
   for (int t : target) {
     if (t != 0 && t != 1) {
       throw std::invalid_argument("Target deve ser binário (0 ou 1).");
     }
   }
   // Checa valores ausentes
   // Aqui consideramos "NA_STRING" para feature e IntegerVector::is_na para target
   // Se for requerido explicitamente, podemos verificar antes da conversão.
   // Caso o R já garanta integridade, podemos omitir.
 }
 
 void initializeBins() {
   std::unordered_map<std::string, BinInfo> category_map;
   category_map.reserve(feature.size());
   
   // Conta ocorrências e inicializa bins
   for (size_t i = 0; i < feature.size(); ++i) {
     const std::string& cat = feature[i];
     BinInfo &bin = category_map[cat];
     if (bin.categories.empty()) {
       bin.categories.push_back(cat);
     }
     bin.count++;
     bin.count_pos += target[i];
     bin.count_neg += (1 - target[i]);
   }
   
   bins.clear();
   bins.reserve(category_map.size());
   for (auto& pair : category_map) {
     bins.push_back(std::move(pair.second));
   }
   
   // Ordena por taxa de positivos
   std::sort(bins.begin(), bins.end(), [](const BinInfo& a, const BinInfo& b) {
     double a_rate = (double)a.count_pos / (double)std::max(a.count,1);
     double b_rate = (double)b.count_pos / (double)std::max(b.count,1);
     return a_rate < b_rate;
   });
   
   // Mescla categorias raras
   int total_count = std::accumulate(bins.begin(), bins.end(), 0,
                                     [](int sum, const BinInfo& bin) { return sum + bin.count; });
   
   std::vector<BinInfo> merged_bins;
   merged_bins.reserve(bins.size());
   BinInfo current_bin;
   
   for (const auto& bin : bins) {
     double freq = (double)bin.count / (double)std::max(total_count,1);
     if (freq < bin_cutoff) {
       // Mescla bin raro em current_bin
       current_bin.categories.insert(current_bin.categories.end(), bin.categories.begin(), bin.categories.end());
       current_bin.count += bin.count;
       current_bin.count_pos += bin.count_pos;
       current_bin.count_neg += bin.count_neg;
     } else {
       // Fecha current_bin se existir
       if (!current_bin.categories.empty()) {
         merged_bins.push_back(std::move(current_bin));
         current_bin = BinInfo();
       }
       merged_bins.push_back(bin);
     }
   }
   if (!current_bin.categories.empty()) {
     merged_bins.push_back(std::move(current_bin));
   }
   
   bins = std::move(merged_bins);
   
   // Limita a max_n_prebins se necessário
   if ((int)bins.size() > max_n_prebins) {
     bins.resize(max_n_prebins);
   }
 }
 
 void greedyMerge() {
   int total_pos = 0, total_neg = 0;
   for (const auto& bin : bins) {
     total_pos += bin.count_pos;
     total_neg += bin.count_neg;
   }
   
   // Calcula IV inicial
   for (auto& bin : bins) {
     bin.woe = calculateWOE(bin.count_pos, bin.count_neg, total_pos, total_neg);
     double pos_rate = (double)bin.count_pos / (double)total_pos;
     double neg_rate = (double)bin.count_neg / (double)total_neg;
     bin.iv = (pos_rate - neg_rate) * bin.woe;
   }
   
   double prev_iv = calculateIV(bins, total_pos, total_neg);
   
   while ((int)bins.size() > min_bins && iterations_run < max_iterations) {
     double best_merge_score = -std::numeric_limits<double>::infinity();
     size_t best_merge_index = 0;
     
     // Tenta mesclar cada par de bins adjacentes
     for (size_t i = 0; i < bins.size() - 1; ++i) {
       BinInfo merged_bin;
       merged_bin.categories.reserve(bins[i].categories.size() + bins[i+1].categories.size());
       merged_bin.categories.insert(merged_bin.categories.end(), bins[i].categories.begin(), bins[i].categories.end());
       merged_bin.categories.insert(merged_bin.categories.end(), bins[i+1].categories.begin(), bins[i+1].categories.end());
       merged_bin.count = bins[i].count + bins[i+1].count;
       merged_bin.count_pos = bins[i].count_pos + bins[i+1].count_pos;
       merged_bin.count_neg = bins[i].count_neg + bins[i+1].count_neg;
       
       std::vector<BinInfo> temp_bins = bins;
       temp_bins[i] = merged_bin;
       temp_bins.erase(temp_bins.begin() + i + 1);
       
       for (auto& tb : temp_bins) {
         tb.woe = calculateWOE(tb.count_pos, tb.count_neg, total_pos, total_neg);
         double pos_rate = (double)tb.count_pos / (double)total_pos;
         double neg_rate = (double)tb.count_neg / (double)total_neg;
         tb.iv = (pos_rate - neg_rate) * tb.woe;
       }
       
       double merge_score = calculateIV(temp_bins, total_pos, total_neg);
       
       if (merge_score > best_merge_score && std::isfinite(merge_score)) {
         best_merge_score = merge_score;
         best_merge_index = i;
       }
     }
     
     // Executa a melhor fusão encontrada
     bins[best_merge_index].categories.insert(bins[best_merge_index].categories.end(),
                                              bins[best_merge_index + 1].categories.begin(),
                                              bins[best_merge_index + 1].categories.end());
     bins[best_merge_index].count += bins[best_merge_index + 1].count;
     bins[best_merge_index].count_pos += bins[best_merge_index + 1].count_pos;
     bins[best_merge_index].count_neg += bins[best_merge_index + 1].count_neg;
     bins.erase(bins.begin() + best_merge_index + 1);
     
     // Recalcula WOE e IV após fusão
     for (auto& bin : bins) {
       bin.woe = calculateWOE(bin.count_pos, bin.count_neg, total_pos, total_neg);
       double pos_rate = (double)bin.count_pos / (double)total_pos;
       double neg_rate = (double)bin.count_neg / (double)total_neg;
       bin.iv = (pos_rate - neg_rate) * bin.woe;
     }
     
     double current_iv = calculateIV(bins, total_pos, total_neg);
     
     // Checa convergência
     if (std::fabs(current_iv - prev_iv) < convergence_threshold) {
       converged = true;
       break;
     }
     
     prev_iv = current_iv;
     iterations_run++;
     
     if ((int)bins.size() <= max_bins) {
       break;
     }
   }
 }
 
 void ensureMonotonicity() {
   // Aplica mesclas para garantir monotonicidade do WoE
   // Caso haja violação da monotonicidade, mescla bins adjacentes até corrigir
   bool monotonic = false;
   int safety_counter = 0;
   while (!monotonic && (int)bins.size() > min_bins && safety_counter < 10 * (int)bins.size()) {
     monotonic = true;
     for (size_t i = 1; i < bins.size(); ++i) {
       if (bins[i].woe < bins[i-1].woe) {
         // Mescla bins i-1 e i
         bins[i-1].categories.insert(bins[i-1].categories.end(), bins[i].categories.begin(), bins[i].categories.end());
         bins[i-1].count += bins[i].count;
         bins[i-1].count_pos += bins[i].count_pos;
         bins[i-1].count_neg += bins[i].count_neg;
         bins.erase(bins.begin() + i);
         monotonic = false;
         break;
       }
     }
     safety_counter++;
   }
   if (safety_counter >= 10 * (int)bins.size()) {
     Rcpp::warning("Não foi possível garantir monotonicidade dentro de tentativas razoáveis.");
   }
 }
 
public:
 OptimalBinningCategoricalGMB(const std::vector<std::string>& feature,
                              const std::vector<int>& target,
                              int min_bins = 3,
                              int max_bins = 5,
                              double bin_cutoff = 0.05,
                              int max_n_prebins = 20,
                              std::string bin_separator = "%;%",
                              double convergence_threshold = 1e-6,
                              int max_iterations = 1000)
   : feature(feature), target(target), min_bins(min_bins), max_bins(max_bins),
     bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins), bin_separator(bin_separator),
     convergence_threshold(convergence_threshold), max_iterations(max_iterations) {
   
   validateInput();
   // Ajusta max_bins se necessário
   std::unordered_set<std::string> unique_cats(feature.begin(), feature.end());
   int ncat = (int)unique_cats.size();
   if (max_bins > ncat) {
     max_bins = ncat;
   }
 }
 
 Rcpp::List fit() {
   initializeBins();
   
   // Se já temos poucos bins, não precisa otimizar
   if ((int)bins.size() <= max_bins) {
     converged = true;
   } else {
     greedyMerge();
     ensureMonotonicity();
   }
   
   // Prepara saída
   std::vector<std::string> bin_names;
   bin_names.reserve(bins.size());
   std::vector<double> woe_values; woe_values.reserve(bins.size());
   std::vector<double> iv_values; iv_values.reserve(bins.size());
   std::vector<int> count_values; count_values.reserve(bins.size());
   std::vector<int> count_pos_values; count_pos_values.reserve(bins.size());
   std::vector<int> count_neg_values; count_neg_values.reserve(bins.size());
   
   for (const auto& bin : bins) {
     // Cria nome do bin
     std::string bin_name;
     bin_name.reserve(64);
     if (!bin.categories.empty()) {
       bin_name = bin.categories[0];
       for (size_t i = 1; i < bin.categories.size(); ++i) {
         bin_name += bin_separator + bin.categories[i];
       }
     }
     
     bin_names.push_back(bin_name);
     woe_values.push_back(bin.woe);
     iv_values.push_back(bin.iv);
     count_values.push_back(bin.count);
     count_pos_values.push_back(bin.count_pos);
     count_neg_values.push_back(bin.count_neg);
   }
   
   Rcpp::NumericVector ids(bin_names.size());
   for(int i = 0; i < bin_names.size(); i++) {
      ids[i] = i + 1;
   }
   
   return Rcpp::List::create(
     Rcpp::Named("id") = ids,
     Rcpp::Named("bin") = bin_names,
     Rcpp::Named("woe") = woe_values,
     Rcpp::Named("iv") = iv_values,
     Rcpp::Named("count") = count_values,
     Rcpp::Named("count_pos") = count_pos_values,
     Rcpp::Named("count_neg") = count_neg_values,
     Rcpp::Named("converged") = converged,
     Rcpp::Named("iterations") = iterations_run
   );
 }
};

//' @title Categorical Optimal Binning with Greedy Merge Binning
//'
//' @description
//' Implements optimal binning for categorical variables using a Greedy Merge approach,
//' calculating Weight of Evidence (WoE) and Information Value (IV).
//'
//' @param target Integer vector of binary target values (0 ou 1).
//' @param feature Character vector of categorical feature values.
//' @param min_bins Número mínimo de bins (padrão: 3).
//' @param max_bins Número máximo de bins (padrão: 5).
//' @param bin_cutoff Frequência mínima para um bin separado (padrão: 0.05).
//' @param max_n_prebins Número máximo de pré-bins antes da fusão (padrão: 20).
//' @param bin_separator Separador usado para mesclar nomes de categorias (padrão: "%;%").
//' @param convergence_threshold Limite para convergência (padrão: 1e-6).
//' @param max_iterations Número máximo de iterações (padrão: 1000).
//'
//' @return Uma lista com os seguintes elementos:
//' \itemize{
//'   \item bins: Vetor de caracteres com os nomes dos bins (categorias mescladas).
//'   \item woe: Vetor numérico dos valores de Weight of Evidence para cada bin.
//'   \item iv: Vetor numérico do Information Value para cada bin.
//'   \item count: Vetor inteiro da contagem total para cada bin.
//'   \item count_pos: Vetor inteiro da contagem da classe positiva para cada bin.
//'   \item count_neg: Vetor inteiro da contagem da classe negativa para cada bin.
//'   \item converged: Lógico indicando se o algoritmo convergiu.
//'   \item iterations: Inteiro indicando o número de iterações realizadas.
//' }
//'
//' @details
//' O algoritmo utiliza uma abordagem de fusão gulosa para encontrar uma solução de binning ótima.
//' Ele começa com cada categoria única como um bin separado e itera fusões de
//' bins para maximizar o Information Value (IV) geral, respeitando as
//' restrições no número de bins.
//'
//' O Weight of Evidence (WoE) para cada bin é calculado como:
//'
//' \deqn{WoE = \ln\left(\frac{P(X|Y=1)}{P(X|Y=0)}\right)}
//'
//' O Information Value (IV) para cada bin é calculado como:
//'
//' \deqn{IV = (P(X|Y=1) - P(X|Y=0)) \times WoE}
//'
//' O algoritmo inclui os seguintes passos principais:
//' \enumerate{
//'   \item Inicializar bins com cada categoria única.
//'   \item Mesclar categorias raras com base no bin_cutoff.
//'   \item Iterativamente mesclar bins adjacentes que resultem no maior IV.
//'   \item Parar de mesclar quando o número de bins atingir min_bins ou max_bins.
//'   \item Garantir a monotonicidade dos valores de WoE através dos bins.
//'   \item Calcular o WoE e IV final para cada bin.
//' }
//'
//' O algoritmo lida com contagens zero usando uma constante pequena (epsilon) para evitar
//' logaritmos indefinidos e divisão por zero.
//'
//' @examples
//' \dontrun{
//' # Dados de exemplo
//' target <- c(1, 0, 1, 1, 0, 1, 0, 0, 1, 1)
//' feature <- c("A", "B", "A", "C", "B", "D", "C", "A", "D", "B")
//'
//' # Executar binning ótimo
//' result <- optimal_binning_categorical_gmb(target, feature, min_bins = 2, max_bins = 4)
//'
//' # Ver resultados
//' print(result)
//' }
//'
//' @author
//' Lopes, J. E.
//'
//' @references
//' \itemize{
//'   \item Beltrami, M., Mach, M., & Dall'Aglio, M. (2021). Monotonic Optimal Binning Algorithm for Credit Risk Modeling. Risks, 9(3), 58.
//'   \item Siddiqi, N. (2006). Credit risk scorecards: developing and implementing intelligent credit scoring (Vol. 3). John Wiley & Sons.
//' }
//' @export
//'
// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_gmb(Rcpp::IntegerVector target,
                                          Rcpp::StringVector feature,
                                          int min_bins = 3,
                                          int max_bins = 5,
                                          double bin_cutoff = 0.05,
                                          int max_n_prebins = 20,
                                          std::string bin_separator = "%;%",
                                          double convergence_threshold = 1e-6,
                                          int max_iterations = 1000) {
 std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
 std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
 
 try {
   OptimalBinningCategoricalGMB binner(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, 
                                       max_n_prebins, bin_separator, convergence_threshold, 
                                       max_iterations);
   return binner.fit();
 } catch (const std::exception& e) {
   Rcpp::stop("Erro no binning ótimo: " + std::string(e.what()));
 }
}



// #include <Rcpp.h>
// #include <vector>
// #include <string>
// #include <algorithm>
// #include <unordered_map>
// #include <cmath>
// #include <limits>
// #include <numeric>
// 
// using namespace Rcpp;
// 
// // Structure to hold bin information
// struct BinInfo {
//   std::vector<std::string> categories;
//   int count = 0;
//   int count_pos = 0;
//   int count_neg = 0;
//   double woe = 0.0;
//   double iv = 0.0;
// };
// 
// class OptimalBinningCategoricalGMB {
// private:
//   std::vector<std::string> feature;
//   std::vector<int> target;
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
//   std::string bin_separator;
//   double convergence_threshold;
//   int max_iterations;
//   
//   std::vector<BinInfo> bins;
//   bool converged = false;
//   int iterations_run = 0;
//   
//   // Improved WOE calculation with better handling of edge cases
//   double calculateWOE(int pos, int neg, int total_pos, int total_neg) {
//     const double epsilon = 1e-10;
//     double pos_rate = (pos + epsilon) / (total_pos + epsilon);
//     double neg_rate = (neg + epsilon) / (total_neg + epsilon);
//     return std::log(pos_rate / neg_rate);
//   }
//   
//   // Improved IV calculation
//   double calculateIV(const std::vector<BinInfo>& bins, int total_pos, int total_neg) {
//     double iv = 0.0;
//     for (const auto& bin : bins) {
//       double pos_rate = static_cast<double>(bin.count_pos) / total_pos;
//       double neg_rate = static_cast<double>(bin.count_neg) / total_neg;
//       iv += (pos_rate - neg_rate) * bin.woe;
//     }
//     return iv;
//   }
//   
//   void initializeBins() {
//     std::unordered_map<std::string, BinInfo> category_map;
//     
//     // Count occurrences and initialize bins
//     for (size_t i = 0; i < feature.size(); ++i) {
//       auto& bin = category_map[feature[i]];
//       if (bin.categories.empty()) {
//         bin.categories.push_back(feature[i]);
//       }
//       bin.count++;
//       bin.count_pos += target[i];
//       bin.count_neg += 1 - target[i];
//     }
//     
//     // Convert map to vector and sort by positive rate
//     bins.clear();
//     bins.reserve(category_map.size());
//     for (auto& pair : category_map) {
//       bins.push_back(std::move(pair.second));
//     }
//     
//     std::sort(bins.begin(), bins.end(), [](const BinInfo& a, const BinInfo& b) {
//       return (static_cast<double>(a.count_pos) / a.count) < (static_cast<double>(b.count_pos) / b.count);
//     });
//     
//     // Merge rare categories
//     int total_count = std::accumulate(bins.begin(), bins.end(), 0,
//                                       [](int sum, const BinInfo& bin) { return sum + bin.count; });
//     
//     std::vector<BinInfo> merged_bins;
//     BinInfo current_bin;
//     for (const auto& bin : bins) {
//       if (static_cast<double>(bin.count) / total_count < bin_cutoff) {
//         current_bin.categories.insert(current_bin.categories.end(), bin.categories.begin(), bin.categories.end());
//         current_bin.count += bin.count;
//         current_bin.count_pos += bin.count_pos;
//         current_bin.count_neg += bin.count_neg;
//       } else {
//         if (!current_bin.categories.empty()) {
//           merged_bins.push_back(std::move(current_bin));
//           current_bin = BinInfo();
//         }
//         merged_bins.push_back(bin);
//       }
//     }
//     
//     if (!current_bin.categories.empty()) {
//       merged_bins.push_back(std::move(current_bin));
//     }
//     
//     bins = std::move(merged_bins);
//     
//     // Limit to max_n_prebins
//     if (static_cast<int>(bins.size()) > max_n_prebins) {
//       bins.resize(max_n_prebins);
//     }
//   }
//   
//   void greedyMerge() {
//     int total_pos = 0, total_neg = 0;
//     for (const auto& bin : bins) {
//       total_pos += bin.count_pos;
//       total_neg += bin.count_neg;
//     }
//     
//     double prev_iv = -std::numeric_limits<double>::infinity();
//     
//     while (static_cast<int>(bins.size()) > min_bins && iterations_run < max_iterations) {
//       double best_merge_score = -std::numeric_limits<double>::infinity();
//       size_t best_merge_index = 0;
//       
//       for (size_t i = 0; i < bins.size() - 1; ++i) {
//         BinInfo merged_bin;
//         merged_bin.categories.insert(merged_bin.categories.end(), bins[i].categories.begin(), bins[i].categories.end());
//         merged_bin.categories.insert(merged_bin.categories.end(), bins[i + 1].categories.begin(), bins[i + 1].categories.end());
//         merged_bin.count = bins[i].count + bins[i + 1].count;
//         merged_bin.count_pos = bins[i].count_pos + bins[i + 1].count_pos;
//         merged_bin.count_neg = bins[i].count_neg + bins[i + 1].count_neg;
//         
//         std::vector<BinInfo> temp_bins = bins;
//         temp_bins[i] = merged_bin;
//         temp_bins.erase(temp_bins.begin() + i + 1);
//         
//         for (auto& bin : temp_bins) {
//           bin.woe = calculateWOE(bin.count_pos, bin.count_neg, total_pos, total_neg);
//         }
//         double merge_score = calculateIV(temp_bins, total_pos, total_neg);
//         
//         if (merge_score > best_merge_score) {
//           best_merge_score = merge_score;
//           best_merge_index = i;
//         }
//       }
//       
//       // Perform the best merge
//       bins[best_merge_index].categories.insert(bins[best_merge_index].categories.end(),
//                                                bins[best_merge_index + 1].categories.begin(),
//                                                bins[best_merge_index + 1].categories.end());
//       bins[best_merge_index].count += bins[best_merge_index + 1].count;
//       bins[best_merge_index].count_pos += bins[best_merge_index + 1].count_pos;
//       bins[best_merge_index].count_neg += bins[best_merge_index + 1].count_neg;
//       bins.erase(bins.begin() + best_merge_index + 1);
//       
//       // Calculate WOE and IV for current bins
//       for (auto& bin : bins) {
//         bin.woe = calculateWOE(bin.count_pos, bin.count_neg, total_pos, total_neg);
//         double pos_rate = static_cast<double>(bin.count_pos) / total_pos;
//         double neg_rate = static_cast<double>(bin.count_neg) / total_neg;
//         bin.iv = (pos_rate - neg_rate) * bin.woe;
//       }
//       
//       double current_iv = std::accumulate(bins.begin(), bins.end(), 0.0,
//                                           [](double sum, const BinInfo& bin) { return sum + bin.iv; });
//       
//       // Check for convergence
//       if (std::abs(current_iv - prev_iv) < convergence_threshold) {
//         converged = true;
//         break;
//       }
//       
//       prev_iv = current_iv;
//       iterations_run++;
//       
//       if (static_cast<int>(bins.size()) <= max_bins) {
//         break;
//       }
//     }
//   }
//   
//   // Function to ensure monotonicity
//   void ensureMonotonicity() {
//     bool monotonic = false;
//     while (!monotonic && static_cast<int>(bins.size()) > min_bins) {
//       monotonic = true;
//       for (size_t i = 1; i < bins.size(); ++i) {
//         if (bins[i].woe < bins[i-1].woe) {
//           monotonic = false;
//           // Merge the non-monotonic bins
//           bins[i-1].categories.insert(bins[i-1].categories.end(), bins[i].categories.begin(), bins[i].categories.end());
//           bins[i-1].count += bins[i].count;
//           bins[i-1].count_pos += bins[i].count_pos;
//           bins[i-1].count_neg += bins[i].count_neg;
//           bins.erase(bins.begin() + i);
//           break;
//         }
//       }
//     }
//   }
//   
// public:
//   OptimalBinningCategoricalGMB(const std::vector<std::string>& feature,
//                                const std::vector<int>& target,
//                                int min_bins = 3,
//                                int max_bins = 5,
//                                double bin_cutoff = 0.05,
//                                int max_n_prebins = 20,
//                                std::string bin_separator = "%;%",
//                                double convergence_threshold = 1e-6,
//                                int max_iterations = 1000)
//     : feature(feature), target(target), min_bins(min_bins), max_bins(max_bins),
//       bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins), bin_separator(bin_separator),
//       convergence_threshold(convergence_threshold), max_iterations(max_iterations) {
//     
//     // Input validation
//     if (feature.size() != target.size()) {
//       Rcpp::stop("Feature and target vectors must have the same length");
//     }
//     if (min_bins < 2) {
//       Rcpp::stop("min_bins must be at least 2");
//     }
//     if (max_bins < min_bins) {
//       Rcpp::stop("max_bins must be greater than or equal to min_bins");
//     }
//     if (bin_cutoff <= 0 || bin_cutoff >= 1) {
//       Rcpp::stop("bin_cutoff must be between 0 and 1");
//     }
//     if (max_n_prebins < min_bins) {
//       Rcpp::stop("max_n_prebins must be greater than or equal to min_bins");
//     }
//     
//     // Check if target is binary
//     for (int t : target) {
//       if (t != 0 && t != 1) {
//         Rcpp::stop("Target must be binary (0 or 1)");
//       }
//     }
//     
//     // Adjust max_bins if necessary
//     int ncat = std::unordered_set<std::string>(feature.begin(), feature.end()).size();
//     max_bins = std::min(max_bins, ncat);
//   }
//   
//   Rcpp::List fit() {
//     initializeBins();
//     
//     // If ncat <= max_bins, skip optimization
//     if (static_cast<int>(bins.size()) <= max_bins) {
//       converged = true;
//     } else {
//       greedyMerge();
//       ensureMonotonicity();
//     }
//     
//     // Prepare output
//     std::vector<std::string> bin_names;
//     std::vector<double> woe_values;
//     std::vector<double> iv_values;
//     std::vector<int> count_values;
//     std::vector<int> count_pos_values;
//     std::vector<int> count_neg_values;
//     
//     for (const auto& bin : bins) {
//       std::string bin_name = bin.categories[0];
//       for (size_t i = 1; i < bin.categories.size(); ++i) {
//         bin_name += bin_separator + bin.categories[i];
//       }
//       
//       bin_names.push_back(bin_name);
//       woe_values.push_back(bin.woe);
//       iv_values.push_back(bin.iv);
//       count_values.push_back(bin.count);
//       count_pos_values.push_back(bin.count_pos);
//       count_neg_values.push_back(bin.count_neg);
//     }
//     
//     // Calculate total IV
//     double total_iv = std::accumulate(iv_values.begin(), iv_values.end(), 0.0);
//     
//     return Rcpp::List::create(
//       Rcpp::Named("bin") = bin_names,
//       Rcpp::Named("woe") = woe_values,
//       Rcpp::Named("iv") = iv_values,
//       Rcpp::Named("count") = count_values,
//       Rcpp::Named("count_pos") = count_pos_values,
//       Rcpp::Named("count_neg") = count_neg_values,
//       Rcpp::Named("converged") = converged,
//       Rcpp::Named("iterations") = iterations_run
//     );
//   }
// };
// 
// //' @title Categorical Optimal Binning with Greedy Merge Binning
// //'
// //' @description
// //' Implements optimal binning for categorical variables using a Greedy Merge approach,
// //' calculating Weight of Evidence (WoE) and Information Value (IV).
// //'
// //' @param target Integer vector of binary target values (0 ou 1).
// //' @param feature Character vector of categorical feature values.
// //' @param min_bins Número mínimo de bins (padrão: 3).
// //' @param max_bins Número máximo de bins (padrão: 5).
// //' @param bin_cutoff Frequência mínima para um bin separado (padrão: 0.05).
// //' @param max_n_prebins Número máximo de pré-bins antes da fusão (padrão: 20).
// //' @param bin_separator Separador usado para mesclar nomes de categorias (padrão: "%;%").
// //' @param convergence_threshold Limite para convergência (padrão: 1e-6).
// //' @param max_iterations Número máximo de iterações (padrão: 1000).
// //'
// //' @return Uma lista com os seguintes elementos:
// //' \itemize{
// //'   \item bins: Vetor de caracteres com os nomes dos bins (categorias mescladas).
// //'   \item woe: Vetor numérico dos valores de Weight of Evidence para cada bin.
// //'   \item iv: Vetor numérico do Information Value para cada bin.
// //'   \item count: Vetor inteiro da contagem total para cada bin.
// //'   \item count_pos: Vetor inteiro da contagem da classe positiva para cada bin.
// //'   \item count_neg: Vetor inteiro da contagem da classe negativa para cada bin.
// //'   \item converged: Lógico indicando se o algoritmo convergiu.
// //'   \item iterations: Inteiro indicando o número de iterações realizadas.
// //' }
// //'
// //' @details
// //' O algoritmo utiliza uma abordagem de fusão gulosa para encontrar uma solução de binning ótima.
// //' Ele começa com cada categoria única como um bin separado e itera fusões de
// //' bins para maximizar o Information Value (IV) geral, respeitando as
// //' restrições no número de bins.
// //'
// //' O Weight of Evidence (WoE) para cada bin é calculado como:
// //'
// //' \deqn{WoE = \ln\left(\frac{P(X|Y=1)}{P(X|Y=0)}\right)}
// //'
// //' O Information Value (IV) para cada bin é calculado como:
// //'
// //' \deqn{IV = (P(X|Y=1) - P(X|Y=0)) \times WoE}
// //'
// //' O algoritmo inclui os seguintes passos principais:
// //' \enumerate{
// //'   \item Inicializar bins com cada categoria única.
// //'   \item Mesclar categorias raras com base no bin_cutoff.
// //'   \item Iterativamente mesclar bins adjacentes que resultem no maior IV.
// //'   \item Parar de mesclar quando o número de bins atingir min_bins ou max_bins.
// //'   \item Garantir a monotonicidade dos valores de WoE através dos bins.
// //'   \item Calcular o WoE e IV final para cada bin.
// //' }
// //'
// //' O algoritmo lida com contagens zero usando uma constante pequena (epsilon) para evitar
// //' logaritmos indefinidos e divisão por zero.
// //'
// //' @examples
// //' \dontrun{
// //' # Dados de exemplo
// //' target <- c(1, 0, 1, 1, 0, 1, 0, 0, 1, 1)
// //' feature <- c("A", "B", "A", "C", "B", "D", "C", "A", "D", "B")
// //'
// //' # Executar binning ótimo
// //' result <- optimal_binning_categorical_gmb(target, feature, min_bins = 2, max_bins = 4)
// //'
// //' # Ver resultados
// //' print(result)
// //' }
// //'
// //' @author
// //' Lopes, J. E.
// //'
// //' @references
// //' \itemize{
// //'   \item Beltrami, M., Mach, M., & Dall'Aglio, M. (2021). Monotonic Optimal Binning Algorithm for Credit Risk Modeling. Risks, 9(3), 58.
// //'   \item Siddiqi, N. (2006). Credit risk scorecards: developing and implementing intelligent credit scoring (Vol. 3). John Wiley & Sons.
// //' }
// //' @export
// //'
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_categorical_gmb(Rcpp::IntegerVector target,
//                                            Rcpp::StringVector feature,
//                                            int min_bins = 3,
//                                            int max_bins = 5,
//                                            double bin_cutoff = 0.05,
//                                            int max_n_prebins = 20,
//                                            std::string bin_separator = "%;%",
//                                            double convergence_threshold = 1e-6,
//                                            int max_iterations = 1000) {
//   std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
//   std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
//   
//   OptimalBinningCategoricalGMB binner(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, 
//                                       max_n_prebins, bin_separator, convergence_threshold, 
//                                       max_iterations);
//   return binner.fit();
// }
