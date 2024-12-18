// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>

using namespace Rcpp;

// Função auxiliar para calcular WOE e IV
inline void compute_woe_iv(double count_pos, double count_neg, double total_pos, double total_neg,
                          double &woe, double &iv) {
 if (count_pos == 0 || count_neg == 0) {
   woe = 0.0;
   iv = 0.0;
   return;
 }
 double dist_pos = count_pos / total_pos;
 double dist_neg = count_neg / total_neg;
 woe = std::log(dist_pos / dist_neg);
 iv = (dist_pos - dist_neg) * woe;
}

// Classe principal para binning ótimo categórico via programação dinâmica
class OptimalBinningCategoricalDPLC {
public:
 OptimalBinningCategoricalDPLC(const std::vector<std::string> &feature,
                               const std::vector<int> &target,
                               int min_bins,
                               int max_bins,
                               double bin_cutoff,
                               int max_n_prebins,
                               double convergence_threshold,
                               int max_iterations,
                               const std::string &bin_separator) :
 feature(feature),
 target(target),
 min_bins(min_bins),
 max_bins(max_bins),
 bin_cutoff(bin_cutoff),
 max_n_prebins(max_n_prebins),
 convergence_threshold(convergence_threshold),
 max_iterations(max_iterations),
 bin_separator(bin_separator),
 converged(false),
 iterations_run(0) {}
 
 // Função principal para executar o binning
 Rcpp::List perform_binning() {
   try {
     // 1: Validação de entradas
     validate_input();
     
     // 2: Pré-processamento (contagens)
     preprocess_data();
     
     // Verifica se já estamos abaixo ou igual a max_bins
     size_t ncat = category_counts.size();
     if (ncat <= static_cast<size_t>(max_bins)) {
       compute_event_rates_no_optimization();
       return prepare_output_no_optimization();
     }
     
     // 4: Mescla categorias raras
     merge_rare_categories();
     
     // 5: Limita o número de pré-bins
     ensure_max_prebins();
     
     // 6: Calcula taxas de evento, ordena categorias
     compute_and_sort_event_rates();
     
     // 7: Inicializa estruturas de DP
     initialize_dp_structures();
     
     // 8: Executa a programação dinâmica
     perform_dynamic_programming();
     
     // 9: Faz backtrack para encontrar os bins ótimos
     backtrack_optimal_bins();
     
     // 10: Retorna resultado final
     return prepare_output();
   } catch (const std::exception &e) {
     Rcpp::stop("Erro no binning ótimo: " + std::string(e.what()));
   }
 }
 
private:
 const std::vector<std::string> &feature;
 const std::vector<int> &target;
 int min_bins;
 int max_bins;
 double bin_cutoff;
 int max_n_prebins;
 double convergence_threshold;
 int max_iterations;
 std::string bin_separator;
 
 // Variáveis auxiliares
 std::unordered_map<std::string, int> category_counts;
 std::unordered_map<std::string, int> category_pos_counts;
 double total_count;
 double total_pos;
 double total_neg;
 
 // Vetores intermediários
 std::vector<std::string> merged_categories;
 std::unordered_map<std::string, std::string> category_mapping;
 std::vector<std::string> sorted_categories;
 std::vector<double> sorted_event_rates;
 
 // Estruturas de DP
 std::vector<std::vector<double>> dp;
 std::vector<std::vector<int>> prev_bin;
 std::vector<double> cum_count_pos;
 std::vector<double> cum_count_neg;
 
 // Resultados finais
 std::vector<std::string> bin_names;
 std::vector<double> bin_woe;
 std::vector<double> bin_iv;
 std::vector<int> bin_count;
 std::vector<int> bin_count_pos;
 std::vector<int> bin_count_neg;
 
 bool converged;
 int iterations_run;
 
 // Função para dividir string por delimitador
 std::vector<std::string> split_string(const std::string &s, const std::string &delimiter) const {
   std::vector<std::string> tokens;
   tokens.reserve(8);
   size_t start = 0, end = 0;
   while ((end = s.find(delimiter, start)) != std::string::npos) {
     tokens.push_back(s.substr(start, end - start));
     start = end + delimiter.length();
   }
   tokens.push_back(s.substr(start));
   return tokens;
 }
 
 // Obter contagem total de um bin (composto ou não)
 int get_bin_count(const std::string &bin) const {
   int count = 0;
   std::vector<std::string> categories = split_string(bin, bin_separator);
   for (const auto &cat : categories) {
     auto it = category_counts.find(cat);
     if (it != category_counts.end()) {
       count += it->second;
     }
   }
   return count;
 }
 
 void validate_input() {
   if (min_bins < 2) {
     throw std::invalid_argument("min_bins deve ser >= 2.");
   }
   if (max_bins < min_bins) {
     throw std::invalid_argument("max_bins deve ser >= min_bins.");
   }
   if (feature.size() != target.size()) {
     throw std::invalid_argument("feature e target devem ter o mesmo tamanho.");
   }
   if (feature.empty()) {
     throw std::invalid_argument("Vetores de entrada não podem ser vazios.");
   }
   if (bin_cutoff <= 0 || bin_cutoff >= 1) {
     throw std::invalid_argument("bin_cutoff deve estar entre 0 e 1 (exclusivo).");
   }
   if (convergence_threshold <= 0) {
     throw std::invalid_argument("convergence_threshold deve ser positivo.");
   }
   if (max_iterations <= 0) {
     throw std::invalid_argument("max_iterations deve ser positivo.");
   }
   
   // Verifica se target é binário
   std::unordered_set<int> unique_targets(target.begin(), target.end());
   if (unique_targets.size() != 2 || unique_targets.find(0) == unique_targets.end() || unique_targets.find(1) == unique_targets.end()) {
     throw std::invalid_argument("Target deve ser binário (0 e 1).");
   }
 }
 
 void preprocess_data() {
   category_counts.reserve(feature.size());
   category_pos_counts.reserve(feature.size());
   for (size_t i = 0; i < feature.size(); ++i) {
     category_counts[feature[i]] += 1;
     if (target[i] == 1) {
       category_pos_counts[feature[i]] += 1;
     }
   }
   
   total_count = (double)feature.size();
   total_pos = std::accumulate(target.begin(), target.end(), 0.0);
   total_neg = total_count - total_pos;
 }
 
 void compute_event_rates_no_optimization() {
   for (const auto &pair : category_counts) {
     std::string bin_name = pair.first;
     int count = pair.second;
     int count_pos = category_pos_counts[bin_name];
     int count_neg = count - count_pos;
     double woe_bin, iv_bin;
     compute_woe_iv(count_pos, count_neg, total_pos, total_neg, woe_bin, iv_bin);
     
     bin_names.push_back(bin_name);
     bin_woe.push_back(woe_bin);
     bin_iv.push_back(iv_bin);
     bin_count.push_back(count);
     bin_count_pos.push_back(count_pos);
     bin_count_neg.push_back(count_neg);
   }
 }
 
 Rcpp::List prepare_output_no_optimization() {
    Rcpp::NumericVector ids(bin_names.size());
    for(int i = 0; i < bin_names.size(); i++) {
       ids[i] = i + 1;
    }
    
    return Rcpp::List::create(
     Rcpp::Named("id") = ids,
     Rcpp::Named("bin") = bin_names,
     Rcpp::Named("woe") = bin_woe,
     Rcpp::Named("iv") = bin_iv,
     Rcpp::Named("count") = bin_count,
     Rcpp::Named("count_pos") = bin_count_pos,
     Rcpp::Named("count_neg") = bin_count_neg,
     Rcpp::Named("converged") = true,
     Rcpp::Named("iterations") = 0
   );
 }
 
 void merge_rare_categories() {
   double cutoff_count = bin_cutoff * total_count;
   std::vector<std::pair<std::string, int>> sorted_categories_count(category_counts.begin(), category_counts.end());
   std::sort(sorted_categories_count.begin(), sorted_categories_count.end(),
             [](const std::pair<std::string, int> &a, const std::pair<std::string, int> &b) {
               return a.second < b.second;
             });
   
   std::string current_merged;
   current_merged.reserve(256);
   int current_count = 0;
   
   for (const auto &cat : sorted_categories_count) {
     if (cat.second < cutoff_count) {
       if (current_merged.empty()) {
         current_merged = cat.first;
       } else {
         current_merged += bin_separator + cat.first;
       }
       current_count += cat.second;
     } else {
       if (!current_merged.empty()) {
         merged_categories.push_back(current_merged);
         auto split_cats = split_string(current_merged, bin_separator);
         for (const auto &merged_cat : split_cats) {
           category_mapping[merged_cat] = current_merged;
         }
         current_merged.clear();
         current_count = 0;
       }
       merged_categories.push_back(cat.first);
       category_mapping[cat.first] = cat.first;
     }
   }
   
   if (!current_merged.empty()) {
     merged_categories.push_back(current_merged);
     auto split_cats = split_string(current_merged, bin_separator);
     for (const auto &merged_cat : split_cats) {
       category_mapping[merged_cat] = current_merged;
     }
   }
 }
 
 void ensure_max_prebins() {
   while (merged_categories.size() > (size_t)max_n_prebins) {
     // Encontra o bin com menor contagem
     auto min_it1 = std::min_element(merged_categories.begin(), merged_categories.end(),
                                     [this](const std::string &a, const std::string &b) {
                                       return get_bin_count(a) < get_bin_count(b);
                                     });
     std::string smallest_bin = *min_it1;
     merged_categories.erase(min_it1);
     
     // Encontra outro bin com menor contagem
     auto min_it2 = std::min_element(merged_categories.begin(), merged_categories.end(),
                                     [this](const std::string &a, const std::string &b) {
                                       return get_bin_count(a) < get_bin_count(b);
                                     });
     
     std::string merged_bin = smallest_bin + bin_separator + *min_it2;
     *min_it2 = merged_bin;
     
     auto split_cats = split_string(merged_bin, bin_separator);
     for (const auto &cat : split_cats) {
       category_mapping[cat] = merged_bin;
     }
   }
 }
 
 void compute_and_sort_event_rates() {
   std::vector<double> event_rates;
   event_rates.reserve(merged_categories.size());
   for (const auto &bin : merged_categories) {
     double pos = 0.0;
     double total = 0.0;
     auto split_cats = split_string(bin, bin_separator);
     for (const auto &cat : split_cats) {
       pos += category_pos_counts[cat];
       total += category_counts[cat];
     }
     event_rates.push_back(pos / total);
   }
   
   std::vector<size_t> indices(merged_categories.size());
   std::iota(indices.begin(), indices.end(), 0);
   std::sort(indices.begin(), indices.end(),
             [&event_rates](size_t i1, size_t i2) { return event_rates[i1] < event_rates[i2]; });
   
   sorted_categories.reserve(indices.size());
   sorted_event_rates.reserve(indices.size());
   for (size_t i : indices) {
     sorted_categories.push_back(merged_categories[i]);
     sorted_event_rates.push_back(event_rates[i]);
   }
 }
 
 void initialize_dp_structures() {
   size_t n = sorted_categories.size();
   dp.assign(n + 1, std::vector<double>(max_bins + 1, -INFINITY));
   prev_bin.assign(n + 1, std::vector<int>(max_bins + 1, -1));
   
   dp[0][0] = 0.0; // Caso base
   
   cum_count_pos.assign(n + 1, 0.0);
   cum_count_neg.assign(n + 1, 0.0);
   
   for (size_t i = 0; i < n; ++i) {
     double pos = 0.0;
     double neg = 0.0;
     auto split_cats = split_string(sorted_categories[i], bin_separator);
     for (const auto &cat : split_cats) {
       pos += category_pos_counts[cat];
       neg += category_counts[cat] - category_pos_counts[cat];
     }
     cum_count_pos[i + 1] = cum_count_pos[i] + pos;
     cum_count_neg[i + 1] = cum_count_neg[i] + neg;
   }
 }
 
 void perform_dynamic_programming() {
   size_t n = sorted_categories.size();
   std::vector<double> last_dp_row(max_bins + 1, -INFINITY);
   last_dp_row[0] = 0.0;
   converged = false;
   iterations_run = 0;
   
   for (iterations_run = 1; iterations_run <= max_iterations; ++iterations_run) {
     bool any_update = false;
     for (size_t i = 1; i <= n; ++i) {
       for (int k = 1; k <= max_bins && k <= (int)i; ++k) {
         for (size_t j = (k - 1 > 0 ? k - 1 : 0); j < i; ++j) {
           // Aplica monotonicidade somente se k > min_bins
           if (k > min_bins && j > 0 && sorted_event_rates[j] < sorted_event_rates[j - 1]) {
             continue;
           }
           double count_pos_bin = cum_count_pos[i] - cum_count_pos[j];
           double count_neg_bin = cum_count_neg[i] - cum_count_neg[j];
           
           double woe_bin, iv_bin;
           compute_woe_iv(count_pos_bin, count_neg_bin, total_pos, total_neg, woe_bin, iv_bin);
           
           double total_iv = dp[j][k - 1] + iv_bin;
           if (total_iv > dp[i][k]) {
             dp[i][k] = total_iv;
             prev_bin[i][k] = (int)j;
             any_update = true;
           }
         }
       }
     }
     // Checa convergência
     double max_diff = 0.0;
     for (int k = 0; k <= max_bins; ++k) {
       double diff = std::fabs(dp[n][k] - last_dp_row[k]);
       if (diff > max_diff) {
         max_diff = diff;
       }
       last_dp_row[k] = dp[n][k];
     }
     if (max_diff < convergence_threshold) {
       converged = true;
       break;
     }
     if (!any_update) {
       break;
     }
   }
   
   if (!converged) {
     Rcpp::warning("Convergência não alcançada em max_iterations. Usando melhor solução encontrada.");
   }
 }
 
 void backtrack_optimal_bins() {
   size_t n = sorted_categories.size();
   double max_total_iv = -INFINITY;
   int best_k = -1;
   
   for (int k = min_bins; k <= max_bins; ++k) {
     if (dp[n][k] > max_total_iv) {
       max_total_iv = dp[n][k];
       best_k = k;
     }
   }
   
   if (best_k == -1) {
     throw std::runtime_error("Falha ao encontrar binagem ótima com as restrições fornecidas.");
   }
   
   std::vector<size_t> bin_edges;
   size_t idx = n;
   int k = best_k;
   while (k > 0) {
     int prev_j = prev_bin[idx][k];
     bin_edges.push_back((size_t)prev_j);
     idx = (size_t)prev_j;
     k -= 1;
   }
   std::reverse(bin_edges.begin(), bin_edges.end());
   
   size_t start = 0;
   for (size_t edge_idx = 0; edge_idx <= bin_edges.size(); ++edge_idx) {
     size_t end = (edge_idx < bin_edges.size()) ? bin_edges[edge_idx] : n;
     
     double count_bin = 0.0;
     double count_pos_bin = 0.0;
     std::string bin_name;
     bin_name.reserve(256);
     
     for (size_t i = start; i < end; ++i) {
       if (i > start) bin_name += bin_separator;
       bin_name += sorted_categories[i];
       
       auto split_cats = split_string(sorted_categories[i], bin_separator);
       for (const auto &cat : split_cats) {
         count_bin += category_counts[cat];
         count_pos_bin += category_pos_counts[cat];
       }
     }
     
     double count_neg_bin = count_bin - count_pos_bin;
     
     if (count_bin > 0) {
       double woe_bin, iv_bin_value;
       compute_woe_iv(count_pos_bin, count_neg_bin, total_pos, total_neg, woe_bin, iv_bin_value);
       
       bin_names.push_back(bin_name);
       bin_woe.push_back(woe_bin);
       bin_iv.push_back(iv_bin_value);
       bin_count.push_back((int)count_bin);
       bin_count_pos.push_back((int)count_pos_bin);
       bin_count_neg.push_back((int)count_neg_bin);
     }
     
     start = end;
   }
 }
 
 Rcpp::List prepare_output() const {
    Rcpp::NumericVector ids(bin_names.size());
    for(int i = 0; i < bin_names.size(); i++) {
       ids[i] = i + 1;
    }
    
    return Rcpp::List::create(
     Rcpp::Named("id") = ids,
     Rcpp::Named("bin") = bin_names,
     Rcpp::Named("woe") = bin_woe,
     Rcpp::Named("iv") = bin_iv,
     Rcpp::Named("count") = bin_count,
     Rcpp::Named("count_pos") = bin_count_pos,
     Rcpp::Named("count_neg") = bin_count_neg,
     Rcpp::Named("converged") = converged,
     Rcpp::Named("iterations") = iterations_run
   );
 }
};

//' @title
//' Optimal Binning for Categorical Variables using Dynamic Programming with Linear Constraints
//'
//' @description
//' This function performs optimal binning for categorical variables using a dynamic programming approach with linear constraints. It aims to find the optimal grouping of categories that maximizes the Information Value (IV) while respecting user-defined constraints on the number of bins.
//'
//' @param target An integer vector of binary target values (0 or 1).
//' @param feature A character vector of categorical feature values.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum proportion of total observations for a bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins before merging (default: 20).
//' @param convergence_threshold Convergence threshold for the dynamic programming algorithm (default: 1e-6).
//' @param max_iterations Maximum number of iterations for the dynamic programming algorithm (default: 1000).
//' @param bin_separator Separator for concatenating category names in bins (default: "%;%").
//'
//' @return A data frame containing binning information, including bin names, WOE, IV, and counts.
//'
//' @details
//' The algorithm uses dynamic programming to find the optimal binning solution that maximizes the total Information Value (IV) while respecting the constraints on the number of bins. It follows these main steps:
//'
//' \enumerate{
//'   \item Preprocess the data by counting occurrences and merging rare categories.
//'   \item Sort categories based on their event rates.
//'   \item Use dynamic programming to find the optimal binning solution.
//'   \item Backtrack to determine the final bin edges.
//'   \item Calculate WOE and IV for each bin.
//' }
//'
//' The dynamic programming approach uses a recurrence relation to find the maximum total IV achievable for a given number of categories and bins.
//'
//' The Weight of Evidence (WOE) for each bin is calculated as:
//'
//' \deqn{WOE = \ln\left(\frac{\text{Distribution of Good}}{\text{Distribution of Bad}}\right)}
//'
//' And the Information Value (IV) for each bin is:
//'
//' \deqn{IV = (\text{Distribution of Good} - \text{Distribution of Bad}) \times WOE}
//'
//' The algorithm aims to find the binning solution that maximizes the total IV while respecting the constraints on the number of bins and ensuring monotonicity when possible.
//'
//' @references
//' \itemize{
//'   \item Belotti, P., Bonami, P., Fischetti, M., Lodi, A., Monaci, M., Nogales-Gómez, A., & Salvagnin, D. (2016). On handling indicator constraints in mixed integer programming. Computational Optimization and Applications, 65(3), 545-566.
//'   \item Mironchyk, P., & Tchistiakov, V. (2017). Monotone optimal binning algorithm for credit risk modeling. SSRN Electronic Journal.
//' }
//'
//' @examples
//' \dontrun{
//' # Create sample data
//' set.seed(123)
//' n <- 1000
//' target <- sample(0:1, n, replace = TRUE)
//' feature <- sample(c("A", "B", "C", "D", "E"), n, replace = TRUE)
//'
//' # Perform optimal binning
//' result <- optimal_binning_categorical_dplc(target, feature, min_bins = 2, max_bins = 4)
//'
//' # View results
//' print(result)
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_dplc(
   Rcpp::IntegerVector target,
   Rcpp::CharacterVector feature,
   int min_bins = 3,
   int max_bins = 5,
   double bin_cutoff = 0.05,
   int max_n_prebins = 20,
   double convergence_threshold = 1e-6,
   int max_iterations = 1000,
   std::string bin_separator = "%;%"
) {
 std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
 std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
 
 OptimalBinningCategoricalDPLC binning(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins,
                                       convergence_threshold, max_iterations, bin_separator);
 return binning.perform_binning();
}

/*
Resumo das melhorias:
- Código reorganizado e comentado.
- Uso de reservas de capacidade, minimizando realocações.
- Tratamento de exceções e validações reforçado.
- Cálculos e laços otimizados, inline funções auxiliares.
- Fluxo lógico mais claro, seguindo passos numerados.
*/
 






// // [[Rcpp::plugins(cpp11)]]
// #include <Rcpp.h>
// #include <unordered_map>
// #include <unordered_set>
// #include <vector>
// #include <string>
// #include <algorithm>
// #include <cmath>
// #include <numeric>
// #include <sstream>
// #include <stdexcept>
// 
// using namespace Rcpp;
// 
// // Helper function to compute WOE and IV
// inline void compute_woe_iv(double count_pos, double count_neg, double total_pos, double total_neg,
//                            double &woe, double &iv) {
//   if (count_pos == 0 || count_neg == 0) {
//     woe = 0.0;
//     iv = 0.0;
//     return;
//   }
//   double dist_pos = count_pos / total_pos;
//   double dist_neg = count_neg / total_neg;
//   woe = std::log(dist_pos / dist_neg);
//   iv = (dist_pos - dist_neg) * woe;
// }
// 
// // Main class for Optimal Binning with Dynamic Programming and Linear Constraints
// class OptimalBinningCategoricalDPLC {
// public:
//   OptimalBinningCategoricalDPLC(const std::vector<std::string> &feature,
//                                 const std::vector<int> &target,
//                                 int min_bins,
//                                 int max_bins,
//                                 double bin_cutoff,
//                                 int max_n_prebins,
//                                 double convergence_threshold,
//                                 int max_iterations,
//                                 const std::string &bin_separator) :
//   feature(feature),
//   target(target),
//   min_bins(min_bins),
//   max_bins(max_bins),
//   bin_cutoff(bin_cutoff),
//   max_n_prebins(max_n_prebins),
//   convergence_threshold(convergence_threshold),
//   max_iterations(max_iterations),
//   bin_separator(bin_separator) {}
//   
//   // Function to perform the optimal binning
//   Rcpp::List perform_binning() {
//     try {
//       // Step 1: Input validation
//       validate_input();
//       
//       // Step 2: Preprocess data - count occurrences
//       preprocess_data();
//       
//       // Step 3: Check number of categories
//       size_t ncat = category_counts.size();
//       if (ncat <= static_cast<size_t>(max_bins)) {
//         // No need to optimize, calculate statistics and return
//         compute_event_rates();
//         return prepare_output_no_optimization();
//       }
//       
//       // Step 4: Merge rare categories based on bin_cutoff
//       merge_rare_categories();
//       
//       // Step 5: Ensure max_n_prebins is not exceeded
//       ensure_max_prebins();
//       
//       // Step 6: Compute event rates and sort categories
//       compute_and_sort_event_rates();
//       
//       // Step 7: Initialize dynamic programming structures
//       initialize_dp_structures();
//       
//       // Step 8: Dynamic Programming to find optimal binning with monotonicity constraint
//       perform_dynamic_programming();
//       
//       // Step 9: Backtrack to find the optimal bins
//       backtrack_optimal_bins();
//       
//       // Step 10: Prepare output
//       return prepare_output();
//     } catch (const std::exception &e) {
//       Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
//     }
//   }
//   
// private:
//   const std::vector<std::string> &feature;
//   const std::vector<int> &target;
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
//   double convergence_threshold;
//   int max_iterations;
//   std::string bin_separator;
//   
//   // Additional member variables
//   std::unordered_map<std::string, int> category_counts;
//   std::unordered_map<std::string, int> category_pos_counts;
//   double total_count;
//   double total_pos;
//   double total_neg;
//   std::vector<std::string> merged_categories;
//   std::unordered_map<std::string, std::string> category_mapping;
//   std::vector<std::string> sorted_categories;
//   std::vector<double> sorted_event_rates;
//   std::vector<std::vector<double>> dp;
//   std::vector<std::vector<int>> prev_bin;
//   std::vector<double> cum_count_pos;
//   std::vector<double> cum_count_neg;
//   std::vector<std::string> bin_names;
//   std::vector<double> bin_woe;
//   std::vector<double> bin_iv;
//   std::vector<int> bin_count;
//   std::vector<int> bin_count_pos;
//   std::vector<int> bin_count_neg;
//   bool converged;
//   int iterations_run;
//   
//   // Helper function to split strings by a delimiter
//   std::vector<std::string> split_string(const std::string &s, const std::string &delimiter) const {
//     std::vector<std::string> tokens;
//     size_t start = 0, end = 0;
//     while ((end = s.find(delimiter, start)) != std::string::npos) {
//       tokens.push_back(s.substr(start, end - start));
//       start = end + delimiter.length();
//     }
//     tokens.push_back(s.substr(start));
//     return tokens;
//   }
//   
//   // Helper function to get the total count for a bin
//   int get_bin_count(const std::string &bin) const {
//     int count = 0;
//     std::vector<std::string> categories = split_string(bin, bin_separator);
//     for (const auto &cat : categories) {
//       auto it = category_counts.find(cat);
//       if (it != category_counts.end()) {
//         count += it->second;
//       }
//     }
//     return count;
//   }
//   
//   void validate_input() {
//     if (min_bins < 2) {
//       throw std::invalid_argument("min_bins must be at least 2.");
//     }
//     if (max_bins < min_bins) {
//       throw std::invalid_argument("max_bins must be greater than or equal to min_bins.");
//     }
//     if (feature.size() != target.size()) {
//       throw std::invalid_argument("feature and target must have the same length.");
//     }
//     if (feature.empty()) {
//       throw std::invalid_argument("Input vectors cannot be empty.");
//     }
//     if (bin_cutoff <= 0 || bin_cutoff >=1) {
//       throw std::invalid_argument("bin_cutoff must be between 0 and 1 (exclusive).");
//     }
//     if (convergence_threshold <= 0) {
//       throw std::invalid_argument("convergence_threshold must be positive.");
//     }
//     if (max_iterations <= 0) {
//       throw std::invalid_argument("max_iterations must be positive.");
//     }
//     
//     // Ensure Target is Binary
//     std::unordered_set<int> unique_targets(target.begin(), target.end());
//     if (unique_targets.size() != 2 || unique_targets.find(0) == unique_targets.end() || unique_targets.find(1) == unique_targets.end()) {
//       throw std::invalid_argument("Target must be binary, containing only 0s and 1s.");
//     }
//   }
//   
//   void preprocess_data() {
//     for (size_t i = 0; i < feature.size(); ++i) {
//       category_counts[feature[i]] += 1;
//       if (target[i] == 1) {
//         category_pos_counts[feature[i]] += 1;
//       }
//     }
//     
//     total_count = feature.size();
//     total_pos = std::accumulate(target.begin(), target.end(), 0.0);
//     total_neg = total_count - total_pos;
//   }
//   
//   void compute_event_rates() {
//     // Compute event rates for categories without optimization
//     for (const auto &pair : category_counts) {
//       std::string bin_name = pair.first;
//       int count = pair.second;
//       int count_pos = category_pos_counts[bin_name];
//       int count_neg = count - count_pos;
//       double woe_bin, iv_bin;
//       compute_woe_iv(count_pos, count_neg, total_pos, total_neg, woe_bin, iv_bin);
//       
//       bin_names.push_back(bin_name);
//       bin_woe.push_back(woe_bin);
//       bin_iv.push_back(iv_bin);
//       bin_count.push_back(count);
//       bin_count_pos.push_back(count_pos);
//       bin_count_neg.push_back(count_neg);
//     }
//   }
//   
//   Rcpp::List prepare_output_no_optimization() {
//     // Prepare woebin List without optimization
//     return Rcpp::List::create(
//       Rcpp::Named("bin") = bin_names,
//       Rcpp::Named("woe") = bin_woe,
//       Rcpp::Named("iv") = bin_iv,
//       Rcpp::Named("count") = bin_count,
//       Rcpp::Named("count_pos") = bin_count_pos,
//       Rcpp::Named("count_neg") = bin_count_neg,
//       Rcpp::Named("converged") = true,
//       Rcpp::Named("iterations") = 0
//     );
//   }
//   
//   void merge_rare_categories() {
//     double cutoff_count = bin_cutoff * total_count;
//     std::vector<std::pair<std::string, int>> sorted_categories_count(category_counts.begin(), category_counts.end());
//     std::sort(sorted_categories_count.begin(), sorted_categories_count.end(),
//               [](const std::pair<std::string, int> &a, const std::pair<std::string, int> &b) {
//                 return a.second < b.second;
//               });
//     
//     std::string current_merged = "";
//     int current_count = 0;
//     
//     for (const auto &cat : sorted_categories_count) {
//       if (cat.second < cutoff_count) {
//         if (current_merged.empty()) {
//           current_merged = cat.first;
//         } else {
//           current_merged += bin_separator + cat.first;
//         }
//         current_count += cat.second;
//       } else {
//         if (!current_merged.empty()) {
//           merged_categories.push_back(current_merged);
//           std::vector<std::string> split_cats = split_string(current_merged, bin_separator);
//           for (const auto &merged_cat : split_cats) {
//             category_mapping[merged_cat] = current_merged;
//           }
//           current_merged = "";
//           current_count = 0;
//         }
//         merged_categories.push_back(cat.first);
//         category_mapping[cat.first] = cat.first;
//       }
//     }
//     
//     if (!current_merged.empty()) {
//       merged_categories.push_back(current_merged);
//       std::vector<std::string> split_cats = split_string(current_merged, bin_separator);
//       for (const auto &merged_cat : split_cats) {
//         category_mapping[merged_cat] = current_merged;
//       }
//     }
//   }
//   
//   void ensure_max_prebins() {
//     while (merged_categories.size() > static_cast<size_t>(max_n_prebins)) {
//       auto min_it1 = std::min_element(merged_categories.begin(), merged_categories.end(),
//                                       [this](const std::string &a, const std::string &b) {
//                                         return get_bin_count(a) < get_bin_count(b);
//                                       });
//       
//       std::string smallest_bin = *min_it1;
//       merged_categories.erase(min_it1);
//       
//       auto min_it2 = std::min_element(merged_categories.begin(), merged_categories.end(),
//                                       [this](const std::string &a, const std::string &b) {
//                                         return get_bin_count(a) < get_bin_count(b);
//                                       });
//       
//       std::string merged_bin = smallest_bin + bin_separator + *min_it2;
//       *min_it2 = merged_bin;
//       
//       std::vector<std::string> split_cats = split_string(merged_bin, bin_separator);
//       for (const auto &cat : split_cats) {
//         category_mapping[cat] = merged_bin;
//       }
//     }
//   }
//   
//   void compute_and_sort_event_rates() {
//     std::vector<double> event_rates;
//     for (const auto &bin : merged_categories) {
//       double pos = 0.0;
//       double total = 0.0;
//       std::vector<std::string> split_cats = split_string(bin, bin_separator);
//       for (const auto &cat : split_cats) {
//         pos += category_pos_counts[cat];
//         total += category_counts[cat];
//       }
//       event_rates.push_back(pos / total);
//     }
//     
//     std::vector<size_t> indices(merged_categories.size());
//     std::iota(indices.begin(), indices.end(), 0);
//     std::sort(indices.begin(), indices.end(),
//               [&event_rates](size_t i1, size_t i2) { return event_rates[i1] < event_rates[i2]; });
//     
//     for (size_t i : indices) {
//       sorted_categories.push_back(merged_categories[i]);
//       sorted_event_rates.push_back(event_rates[i]);
//     }
//   }
//   
//   void initialize_dp_structures() {
//     size_t n = sorted_categories.size();
//     dp.resize(n + 1, std::vector<double>(max_bins + 1, -INFINITY));
//     prev_bin.resize(n + 1, std::vector<int>(max_bins + 1, -1));
//     
//     dp[0][0] = 0.0; // Base case
//     
//     cum_count_pos.resize(n + 1, 0.0);
//     cum_count_neg.resize(n + 1, 0.0);
//     
//     for (size_t i = 0; i < n; ++i) {
//       double pos = 0.0;
//       double neg = 0.0;
//       std::vector<std::string> split_cats = split_string(sorted_categories[i], bin_separator);
//       for (const auto &cat : split_cats) {
//         pos += category_pos_counts[cat];
//         neg += category_counts[cat] - category_pos_counts[cat];
//       }
//       cum_count_pos[i + 1] = cum_count_pos[i] + pos;
//       cum_count_neg[i + 1] = cum_count_neg[i] + neg;
//     }
//   }
//   
//   void perform_dynamic_programming() {
//     size_t n = sorted_categories.size();
//     std::vector<double> last_dp_row(max_bins + 1, -INFINITY);
//     last_dp_row[0] = 0.0;
//     converged = false;
//     iterations_run = 0;
//     
//     for (iterations_run = 1; iterations_run <= max_iterations; ++iterations_run) {
//       bool any_update = false;
//       for (size_t i = 1; i <= n; ++i) {
//         for (int k = 1; k <= max_bins && k <= static_cast<int>(i); ++k) {
//           for (size_t j = (k - 1 > 0 ? k - 1 : 0); j < i; ++j) {
//             // Enforce monotonicity only if we have more than min_bins
//             if (k > min_bins && j > 0 && sorted_event_rates[j] < sorted_event_rates[j - 1]) {
//               continue;
//             }
//             double count_pos_bin = cum_count_pos[i] - cum_count_pos[j];
//             double count_neg_bin = cum_count_neg[i] - cum_count_neg[j];
//             
//             double woe_bin, iv_bin;
//             compute_woe_iv(count_pos_bin, count_neg_bin, total_pos, total_neg, woe_bin, iv_bin);
//             
//             double total_iv = dp[j][k - 1] + iv_bin;
//             if (total_iv > dp[i][k]) {
//               dp[i][k] = total_iv;
//               prev_bin[i][k] = j;
//               any_update = true;
//             }
//           }
//         }
//       }
//       // Check convergence
//       double max_diff = 0.0;
//       for (size_t i = 0; i <= n; ++i) {
//         for (int k = 0; k <= max_bins; ++k) {
//           double diff = std::abs(dp[i][k] - last_dp_row[k]);
//           if (diff > max_diff) {
//             max_diff = diff;
//           }
//           last_dp_row[k] = dp[i][k];
//         }
//       }
//       if (max_diff < convergence_threshold) {
//         converged = true;
//         break;
//       }
//       if (!any_update) {
//         break;
//       }
//     }
//     
//     if (!converged) {
//       Rcpp::warning("Convergence not achieved within max_iterations. Using best solution found.");
//     }
//   }
//   
//   void backtrack_optimal_bins() {
//     size_t n = sorted_categories.size();
//     double max_total_iv = -INFINITY;
//     int best_k = -1;
//     
//     for (int k = min_bins; k <= max_bins; ++k) {
//       if (dp[n][k] > max_total_iv) {
//         max_total_iv = dp[n][k];
//         best_k = k;
//       }
//     }
//     
//     if (best_k == -1) {
//       throw std::runtime_error("Failed to find optimal binning with given constraints.");
//     }
//     
//     std::vector<size_t> bin_edges;
//     size_t idx = n;
//     int k = best_k;
//     while (k > 0) {
//       int prev_j = prev_bin[idx][k];
//       bin_edges.push_back(prev_j);
//       idx = prev_j;
//       k -= 1;
//     }
//     std::reverse(bin_edges.begin(), bin_edges.end());
//     
//     size_t start = 0;
//     for (size_t edge_idx = 0; edge_idx <= bin_edges.size(); ++edge_idx) {
//       size_t end = (edge_idx < bin_edges.size()) ? bin_edges[edge_idx] : n;
//       
//       std::vector<std::string> bin_categories;
//       double count_bin = 0.0;
//       double count_pos_bin = 0.0;
//       std::string bin_name = "";
//       
//       for (size_t i = start; i < end; ++i) {
//         if (i > start) bin_name += bin_separator;
//         bin_name += sorted_categories[i];
//         
//         std::vector<std::string> split_cats = split_string(sorted_categories[i], bin_separator);
//         for (const auto &cat : split_cats) {
//           bin_categories.push_back(cat);
//           count_bin += category_counts[cat];
//           count_pos_bin += category_pos_counts[cat];
//         }
//       }
//       
//       double count_neg_bin = count_bin - count_pos_bin;
//       
//       // Only add the bin if it's not empty
//       if (count_bin > 0) {
//         double woe_bin, iv_bin_value;
//         compute_woe_iv(count_pos_bin, count_neg_bin, total_pos, total_neg, woe_bin, iv_bin_value);
//         
//         bin_names.push_back(bin_name);
//         bin_woe.push_back(woe_bin);
//         bin_iv.push_back(iv_bin_value);
//         bin_count.push_back(static_cast<int>(count_bin));
//         bin_count_pos.push_back(static_cast<int>(count_pos_bin));
//         bin_count_neg.push_back(static_cast<int>(count_neg_bin));
//         
//         // Update category_mapping for all categories in this bin
//         for (const auto &cat : bin_categories) {
//           category_mapping[cat] = bin_name;
//         }
//       }
//       
//       start = end;
//     }
//   }
//   
//   Rcpp::List prepare_output() {
//     // Prepare woebin List
//     return Rcpp::List::create(
//       Rcpp::Named("bin") = bin_names,
//       Rcpp::Named("woe") = bin_woe,
//       Rcpp::Named("iv") = bin_iv,
//       Rcpp::Named("count") = bin_count,
//       Rcpp::Named("count_pos") = bin_count_pos,
//       Rcpp::Named("count_neg") = bin_count_neg,
//       Rcpp::Named("converged") = converged,
//       Rcpp::Named("iterations") = iterations_run
//     );
//   }
//   
// };
// 
// //' @title
// //' Optimal Binning for Categorical Variables using Dynamic Programming with Linear Constraints
// //'
// //' @description
// //' This function performs optimal binning for categorical variables using a dynamic programming approach with linear constraints. It aims to find the optimal grouping of categories that maximizes the Information Value (IV) while respecting user-defined constraints on the number of bins.
// //'
// //' @param target An integer vector of binary target values (0 or 1).
// //' @param feature A character vector of categorical feature values.
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff Minimum proportion of total observations for a bin (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins before merging (default: 20).
// //' @param convergence_threshold Convergence threshold for the dynamic programming algorithm (default: 1e-6).
// //' @param max_iterations Maximum number of iterations for the dynamic programming algorithm (default: 1000).
// //' @param bin_separator Separator for concatenating category names in bins (default: "%;%").
// //'
// //' @return A data frame containing binning information, including bin names, WOE, IV, and counts.
// //'
// //' @details
// //' The algorithm uses dynamic programming to find the optimal binning solution that maximizes the total Information Value (IV) while respecting the constraints on the number of bins. It follows these main steps:
// //'
// //' \enumerate{
// //'   \item Preprocess the data by counting occurrences and merging rare categories.
// //'   \item Sort categories based on their event rates.
// //'   \item Use dynamic programming to find the optimal binning solution.
// //'   \item Backtrack to determine the final bin edges.
// //'   \item Calculate WOE and IV for each bin.
// //' }
// //'
// //' The dynamic programming approach uses a recurrence relation to find the maximum total IV achievable for a given number of categories and bins.
// //'
// //' The Weight of Evidence (WOE) for each bin is calculated as:
// //'
// //' \deqn{WOE = \ln\left(\frac{\text{Distribution of Good}}{\text{Distribution of Bad}}\right)}
// //'
// //' And the Information Value (IV) for each bin is:
// //'
// //' \deqn{IV = (\text{Distribution of Good} - \text{Distribution of Bad}) \times WOE}
// //'
// //' The algorithm aims to find the binning solution that maximizes the total IV while respecting the constraints on the number of bins and ensuring monotonicity when possible.
// //'
// //' @references
// //' \itemize{
// //'   \item Belotti, P., Bonami, P., Fischetti, M., Lodi, A., Monaci, M., Nogales-Gómez, A., & Salvagnin, D. (2016). On handling indicator constraints in mixed integer programming. Computational Optimization and Applications, 65(3), 545-566.
// //'   \item Mironchyk, P., & Tchistiakov, V. (2017). Monotone optimal binning algorithm for credit risk modeling. SSRN Electronic Journal.
// //' }
// //'
// //' @examples
// //' \dontrun{
// //' # Create sample data
// //' set.seed(123)
// //' n <- 1000
// //' target <- sample(0:1, n, replace = TRUE)
// //' feature <- sample(c("A", "B", "C", "D", "E"), n, replace = TRUE)
// //'
// //' # Perform optimal binning
// //' result <- optimal_binning_categorical_dplc(target, feature, min_bins = 2, max_bins = 4)
// //'
// //' # View results
// //' print(result)
// //' }
// //'
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_categorical_dplc(Rcpp::IntegerVector target,
//                                             Rcpp::CharacterVector feature,
//                                             int min_bins = 3,
//                                             int max_bins = 5,
//                                             double bin_cutoff = 0.05,
//                                             int max_n_prebins = 20,
//                                             double convergence_threshold = 1e-6,
//                                             int max_iterations = 1000,
//                                             std::string bin_separator = "%;%") {
//   std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
//   std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
//   
//   OptimalBinningCategoricalDPLC binning(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins,
//                                         convergence_threshold, max_iterations, bin_separator);
//   return binning.perform_binning();
// }
