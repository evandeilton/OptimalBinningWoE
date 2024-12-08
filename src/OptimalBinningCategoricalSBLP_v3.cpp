// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>
#include <stdexcept>
#include <sstream>

using namespace Rcpp;

// Classe para binning ótimo SBLP
class OptimalBinningCategoricalSBLP {
public:
 OptimalBinningCategoricalSBLP(const IntegerVector& target,
                               const CharacterVector& feature,
                               int min_bins,
                               int max_bins,
                               double bin_cutoff,
                               int max_n_prebins,
                               double convergence_threshold,
                               int max_iterations,
                               std::string bin_separator);
 
 List fit();
 
private:
 // Dados de entrada e parâmetros
 const IntegerVector& target;
 const CharacterVector& feature;
 int min_bins;
 int max_bins;
 double bin_cutoff;
 int max_n_prebins;
 double convergence_threshold;
 int max_iterations;
 std::string bin_separator;
 
 // Estruturas internas
 std::vector<std::string> unique_categories;
 std::vector<int> count_total;
 std::vector<int> count_pos;
 std::vector<int> count_neg;
 std::vector<double> category_target_rate;
 std::vector<size_t> sorted_indices;
 
 // Funções auxiliares
 void validate_input();
 void compute_initial_counts();
 void handle_rare_categories();
 void ensure_max_prebins();
 void sort_categories();
 std::vector<std::vector<size_t>> perform_binning();
 double calculate_bin_iv(const std::vector<size_t>& bin) const;
 bool is_monotonic(const std::vector<std::vector<size_t>>& bins) const;
 List prepare_output(const std::vector<std::vector<size_t>>& bins, bool converged, int iterations) const;
 static std::string merge_category_names(const std::vector<std::string>& categories, const std::string& separator);
};

OptimalBinningCategoricalSBLP::OptimalBinningCategoricalSBLP(
 const IntegerVector& target,
 const CharacterVector& feature,
 int min_bins,
 int max_bins,
 double bin_cutoff,
 int max_n_prebins,
 double convergence_threshold,
 int max_iterations,
 std::string bin_separator)
 : target(target), feature(feature),
   min_bins(min_bins), max_bins(max_bins),
   bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins),
   convergence_threshold(convergence_threshold),
   max_iterations(max_iterations),
   bin_separator(bin_separator) {}

// Validações iniciais
void OptimalBinningCategoricalSBLP::validate_input() {
 if (target.size() != feature.size()) {
   throw std::invalid_argument("Target and feature must have the same length");
 }
 if (min_bins < 2) {
   throw std::invalid_argument("min_bins must be at least 2");
 }
 if (max_bins < min_bins) {
   throw std::invalid_argument("max_bins must be greater than or equal to min_bins");
 }
 if (bin_cutoff <= 0 || bin_cutoff >= 1) {
   throw std::invalid_argument("bin_cutoff must be between 0 and 1");
 }
 if (max_n_prebins < min_bins) {
   throw std::invalid_argument("max_n_prebins must be at least equal to min_bins");
 }
 if (convergence_threshold <= 0) {
   throw std::invalid_argument("convergence_threshold must be positive");
 }
 if (max_iterations <= 0) {
   throw std::invalid_argument("max_iterations must be positive");
 }
}

// Cálculo inicial das contagens por categoria
void OptimalBinningCategoricalSBLP::compute_initial_counts() {
 std::unordered_map<std::string, size_t> category_indices;
 
 for (int i = 0; i < feature.size(); ++i) {
   std::string cat = as<std::string>(feature[i]);
   auto [it, inserted] = category_indices.emplace(cat, unique_categories.size());
   if (inserted) {
     unique_categories.push_back(cat);
     count_total.push_back(0);
     count_pos.push_back(0);
     count_neg.push_back(0);
   }
   size_t idx = it->second;
   count_total[idx]++;
   if (target[i] == 1) {
     count_pos[idx]++;
   } else if (target[i] == 0) {
     count_neg[idx]++;
   } else {
     throw std::invalid_argument("Target must be binary (0 or 1)");
   }
 }
 
 category_target_rate.resize(unique_categories.size());
 for (size_t i = 0; i < unique_categories.size(); ++i) {
   category_target_rate[i] = static_cast<double>(count_pos[i]) / count_total[i];
 }
}

// Tratamento de categorias raras unindo-as com categorias similares
void OptimalBinningCategoricalSBLP::handle_rare_categories() {
 int total_count = std::accumulate(count_total.begin(), count_total.end(), 0);
 std::vector<bool> is_rare(unique_categories.size(), false);
 
 for (size_t i = 0; i < unique_categories.size(); ++i) {
   double proportion = static_cast<double>(count_total[i]) / total_count;
   if (proportion < bin_cutoff) {
     is_rare[i] = true;
   }
 }
 
 if (std::none_of(is_rare.begin(), is_rare.end(), [](bool b) { return b; })) {
   return;
 }
 
 std::vector<size_t> rare_indices;
 for (size_t i = 0; i < unique_categories.size(); ++i) {
   if (is_rare[i]) {
     rare_indices.push_back(i);
   }
 }
 
 std::sort(rare_indices.begin(), rare_indices.end(),
           [this](size_t i, size_t j) { return category_target_rate[i] < category_target_rate[j]; });
 
 for (size_t i = 1; i < rare_indices.size(); ++i) {
   size_t prev_idx = rare_indices[i - 1];
   size_t curr_idx = rare_indices[i];
   count_total[prev_idx] += count_total[curr_idx];
   count_pos[prev_idx] += count_pos[curr_idx];
   count_neg[prev_idx] += count_neg[curr_idx];
   unique_categories[prev_idx] = merge_category_names({unique_categories[prev_idx], unique_categories[curr_idx]}, bin_separator);
   unique_categories[curr_idx].clear();
 }
 
 auto it = std::remove_if(unique_categories.begin(), unique_categories.end(),
                          [](const std::string& s) { return s.empty(); });
 size_t new_size = std::distance(unique_categories.begin(), it);
 
 unique_categories.resize(new_size);
 count_total.resize(new_size);
 count_pos.resize(new_size);
 count_neg.resize(new_size);
 
 category_target_rate.resize(new_size);
 for (size_t i = 0; i < new_size; ++i) {
   category_target_rate[i] = static_cast<double>(count_pos[i]) / count_total[i];
 }
}

// Garante que o número de pré-bins não exceda max_n_prebins
void OptimalBinningCategoricalSBLP::ensure_max_prebins() {
 if (unique_categories.size() <= static_cast<size_t>(max_n_prebins)) {
   return;
 }
 
 std::vector<size_t> indices(unique_categories.size());
 std::iota(indices.begin(), indices.end(), 0);
 std::sort(indices.begin(), indices.end(),
           [this](size_t i, size_t j) { return category_target_rate[i] < category_target_rate[j]; });
 
 size_t bins_to_merge = unique_categories.size() - max_n_prebins;
 for (size_t i = 0; i < bins_to_merge; ++i) {
   size_t idx1 = indices[i];
   size_t idx2 = indices[i + 1];
   count_total[idx1] += count_total[idx2];
   count_pos[idx1] += count_pos[idx2];
   count_neg[idx1] += count_neg[idx2];
   unique_categories[idx1] = merge_category_names({unique_categories[idx1], unique_categories[idx2]}, bin_separator);
   unique_categories[idx2].clear();
 }
 
 auto it = std::remove_if(unique_categories.begin(), unique_categories.end(),
                          [](const std::string& s) { return s.empty(); });
 size_t new_size = std::distance(unique_categories.begin(), it);
 
 unique_categories.resize(new_size);
 count_total.resize(new_size);
 count_pos.resize(new_size);
 count_neg.resize(new_size);
 
 category_target_rate.resize(new_size);
 for (size_t i = 0; i < new_size; ++i) {
   category_target_rate[i] = static_cast<double>(count_pos[i]) / count_total[i];
 }
}

// Ordena categorias pela taxa alvo
void OptimalBinningCategoricalSBLP::sort_categories() {
 sorted_indices.resize(unique_categories.size());
 std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
 std::sort(sorted_indices.begin(), sorted_indices.end(),
           [this](size_t i, size_t j) { return category_target_rate[i] < category_target_rate[j]; });
}

// Executa o binning via programação dinâmica
std::vector<std::vector<size_t>> OptimalBinningCategoricalSBLP::perform_binning() {
 size_t n = sorted_indices.size();
 size_t k = std::min(static_cast<size_t>(max_bins), n);
 
 std::vector<std::vector<double>> dp(n + 1, std::vector<double>(k + 1, -std::numeric_limits<double>::infinity()));
 std::vector<std::vector<size_t>> split(n + 1, std::vector<size_t>(k + 1, 0));
 
 for (size_t i = 1; i <= n; ++i) {
   dp[i][1] = calculate_bin_iv(std::vector<size_t>(sorted_indices.begin(), sorted_indices.begin() + i));
 }
 
 for (size_t j = 2; j <= k; ++j) {
   for (size_t i = j; i <= n; ++i) {
     for (size_t s = j - 1; s < i; ++s) {
       double current_iv = dp[s][j-1] + calculate_bin_iv(std::vector<size_t>(sorted_indices.begin() + s, sorted_indices.begin() + i));
       if (current_iv > dp[i][j]) {
         dp[i][j] = current_iv;
         split[i][j] = s;
       }
     }
   }
 }
 
 std::vector<std::vector<size_t>> bins;
 size_t i = n, j = k;
 while (j > 0) {
   std::vector<size_t> bin(sorted_indices.begin() + split[i][j], sorted_indices.begin() + i);
   bins.push_back(bin);
   i = split[i][j];
   --j;
 }
 std::reverse(bins.begin(), bins.end());
 
 // Garante min_bins
 while (bins.size() < static_cast<size_t>(min_bins) && bins.size() < n) {
   size_t max_iv_bin = std::max_element(bins.begin(), bins.end(),
                                        [this](const std::vector<size_t>& a, const std::vector<size_t>& b) {
                                          return calculate_bin_iv(a) < calculate_bin_iv(b);
                                        }) - bins.begin();
   
   if (bins[max_iv_bin].size() > 1) {
     size_t split_point = bins[max_iv_bin].size() / 2;
     std::vector<size_t> new_bin(bins[max_iv_bin].begin() + split_point, bins[max_iv_bin].end());
     bins[max_iv_bin].resize(split_point);
     bins.insert(bins.begin() + max_iv_bin + 1, new_bin);
   } else {
     break;
   }
 }
 
 return bins;
}

// Cálculo de IV para um único bin
double OptimalBinningCategoricalSBLP::calculate_bin_iv(const std::vector<size_t>& bin) const {
 int total_pos_all = std::accumulate(count_pos.begin(), count_pos.end(), 0);
 int total_neg_all = std::accumulate(count_neg.begin(), count_neg.end(), 0);
 
 int bin_pos = 0, bin_neg = 0;
 for (size_t idx : bin) {
   bin_pos += count_pos[idx];
   bin_neg += count_neg[idx];
 }
 
 double pos_rate = static_cast<double>(bin_pos) / total_pos_all;
 double neg_rate = static_cast<double>(bin_neg) / total_neg_all;
 
 const double epsilon = 1e-10;
 pos_rate = std::max(pos_rate, epsilon);
 neg_rate = std::max(neg_rate, epsilon);
 
 double woe = std::log(pos_rate / neg_rate);
 double iv = (pos_rate - neg_rate) * woe;
 
 return iv;
}

// Verifica monotonicidade em relação à taxa alvo
bool OptimalBinningCategoricalSBLP::is_monotonic(const std::vector<std::vector<size_t>>& bins) const {
 std::vector<double> bin_rates;
 for (const auto& bin : bins) {
   int bin_total = 0, bin_pos_count = 0;
   for (size_t idx : bin) {
     bin_total += count_total[idx];
     bin_pos_count += count_pos[idx];
   }
   bin_rates.push_back(static_cast<double>(bin_pos_count) / bin_total);
 }
 
 return std::is_sorted(bin_rates.begin(), bin_rates.end());
}

// Prepara a saída
List OptimalBinningCategoricalSBLP::prepare_output(const std::vector<std::vector<size_t>>& bins, bool converged, int iterations) const {
 std::vector<std::string> bin_names;
 std::vector<double> bin_woe;
 std::vector<double> bin_iv_vals;
 std::vector<int> bin_count_vals;
 std::vector<int> bin_count_pos_vals;
 std::vector<int> bin_count_neg_vals;
 
 int total_pos_all = std::accumulate(count_pos.begin(), count_pos.end(), 0);
 int total_neg_all = std::accumulate(count_neg.begin(), count_neg.end(), 0);
 
 const double epsilon = 1e-10;
 
 for (const auto& bin : bins) {
   std::vector<std::string> bin_categories;
   int bin_total = 0, bin_pos_count = 0, bin_neg_count = 0;
   for (size_t idx : bin) {
     bin_categories.push_back(unique_categories[idx]);
     bin_total += count_total[idx];
     bin_pos_count += count_pos[idx];
     bin_neg_count += count_neg[idx];
   }
   
   double pos_rate = static_cast<double>(bin_pos_count) / total_pos_all;
   double neg_rate = static_cast<double>(bin_neg_count) / total_neg_all;
   pos_rate = std::max(pos_rate, epsilon);
   neg_rate = std::max(neg_rate, epsilon);
   
   double woe = std::log(pos_rate / neg_rate);
   double iv = (pos_rate - neg_rate) * woe;
   
   bin_names.push_back(merge_category_names(bin_categories, bin_separator));
   bin_woe.push_back(woe);
   bin_iv_vals.push_back(iv);
   bin_count_vals.push_back(bin_total);
   bin_count_pos_vals.push_back(bin_pos_count);
   bin_count_neg_vals.push_back(bin_neg_count);
 }
 
 return List::create(
   Named("bin") = bin_names,
   Named("woe") = bin_woe,
   Named("iv") = bin_iv_vals,
   Named("count") = bin_count_vals,
   Named("count_pos") = bin_count_pos_vals,
   Named("count_neg") = bin_count_neg_vals,
   Named("converged") = converged,
   Named("iterations") = iterations
 );
}

// Une nomes de categorias com o separador
std::string OptimalBinningCategoricalSBLP::merge_category_names(const std::vector<std::string>& categories, const std::string& separator) {
 std::vector<std::string> unique_cats;
 for (const auto& cat : categories) {
   std::stringstream ss(cat);
   std::string part;
   while (std::getline(ss, part, ';')) {
     if (std::find(unique_cats.begin(), unique_cats.end(), part) == unique_cats.end()) {
       unique_cats.push_back(part);
     }
   }
 }
 std::sort(unique_cats.begin(), unique_cats.end());
 return std::accumulate(unique_cats.begin(), unique_cats.end(), std::string(),
                        [&separator](const std::string& a, const std::string& b) {
                          return a.empty() ? b : a + separator + b;
                        });
}

// Função principal de ajuste
List OptimalBinningCategoricalSBLP::fit() {
 try {
   validate_input();
   compute_initial_counts();
   handle_rare_categories();
   ensure_max_prebins();
   sort_categories();
   
   if (unique_categories.size() <= static_cast<size_t>(max_bins)) {
     std::vector<std::vector<size_t>> bins(unique_categories.size());
     for (size_t i = 0; i < unique_categories.size(); ++i) {
       bins[i] = {sorted_indices[i]};
     }
     return prepare_output(bins, true, 0);
   }
   
   std::vector<std::vector<size_t>> best_bins;
   double best_iv = -std::numeric_limits<double>::infinity();
   bool converged = false;
   int iterations = 0;
   
   while (iterations < max_iterations) {
     std::vector<std::vector<size_t>> current_bins = perform_binning();
     double current_iv = std::accumulate(current_bins.begin(), current_bins.end(), 0.0,
                                         [this](double sum, const std::vector<size_t>& bin) {
                                           return sum + calculate_bin_iv(bin);
                                         });
     
     if (std::abs(current_iv - best_iv) < convergence_threshold) {
       converged = true;
       break;
     }
     
     if (current_iv > best_iv) {
       best_iv = current_iv;
       best_bins = current_bins;
     }
     
     ++iterations;
   }
   
   // Ajuste de monotonicidade, se necessário
   if (!is_monotonic(best_bins) && best_bins.size() > static_cast<size_t>(min_bins)) {
     std::vector<std::vector<size_t>> monotonic_bins;
     std::vector<size_t> current_bin;
     double prev_rate = -1.0;
     
     for (const auto& bin : best_bins) {
       double bin_rate = std::accumulate(bin.begin(), bin.end(), 0.0,
                                         [this](double sum, size_t idx) {
                                           return sum + category_target_rate[idx] * count_total[idx];
                                         }) / std::accumulate(bin.begin(), bin.end(), 0,
                                         [this](int sum, size_t idx) {
                                           return sum + count_total[idx];
                                         });
       
       if (bin_rate >= prev_rate || monotonic_bins.size() < static_cast<size_t>(min_bins)) {
         if (!current_bin.empty()) {
           monotonic_bins.push_back(current_bin);
           current_bin.clear();
         }
         monotonic_bins.push_back(bin);
         prev_rate = bin_rate;
       } else {
         current_bin.insert(current_bin.end(), bin.begin(), bin.end());
       }
     }
     
     if (!current_bin.empty()) {
       monotonic_bins.push_back(current_bin);
     }
     
     best_bins = monotonic_bins;
   }
   
   return prepare_output(best_bins, converged, iterations);
 } catch (const std::exception& e) {
   Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
 }
}


//' @title Optimal Binning for Categorical Variables using Similarity-Based Logistic Partitioning (SBLP)
//'
//' @description
//' This function performs optimal binning for categorical variables using a Similarity-Based Logistic Partitioning (SBLP) approach.
//' The goal is to produce bins that maximize the Information Value (IV) and provide consistent Weight of Evidence (WoE), considering target rates
//' and ensuring quality through similarity-based merges.
//' The implementation has been revised to improve readability, efficiency, robustness, and to maintain compatibility
//' with the names and types of input/output parameters.
//'
//' @param target Integer binary vector (0 or 1) representing the response variable.
//' @param feature Character vector with the categories of the explanatory variable.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum frequency proportion for a category to be considered as a separate bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins before the partitioning process (default: 20).
//' @param convergence_threshold Threshold for algorithm convergence (default: 1e-6).
//' @param max_iterations Maximum number of iterations of the algorithm (default: 1000).
//' @param bin_separator Separator used to concatenate category names within bins (default: ";").
//'
//' @return A list containing:
//' \itemize{
//'   \item bin: String vector with the names of the bins (concatenated categories).
//'   \item woe: Numeric vector with the Weight of Evidence (WoE) values for each bin.
//'   \item iv: Numeric vector with the Information Value (IV) values for each bin.
//'   \item count: Integer vector with the total count of observations in each bin.
//'   \item count_pos: Integer vector with the count of positive cases (target=1) in each bin.
//'   \item count_neg: Integer vector with the count of negative cases (target=0) in each bin.
//'   \item converged: Logical value indicating whether the algorithm converged.
//'   \item iterations: Integer value indicating the number of iterations executed.
//' }
//'
//' @details
//' Steps of the SBLP algorithm:
//' 1. Validate input and calculate initial counts by category.
//' 2. Handle rare categories by merging them with other similar ones in terms of target rate.
//' 3. Ensure the maximum number of pre-bins by merging uninformative bins.
//' 4. Sort categories by target rate.
//' 5. Apply dynamic programming to determine the optimal partition, considering min_bins and max_bins.
//' 6. Adjust WoE monotonicity, if necessary, provided the number of bins is greater than min_bins.
//' 7. Perform final calculation of WoE and IV for each bin and return the result.
//'
//' Key formulas:
//' \deqn{WoE = \ln\left(\frac{P(X|Y=1)}{P(X|Y=0)}\right)}
//' \deqn{IV = \sum_{bins} (P(X|Y=1) - P(X|Y=0)) \times WoE}
//'
//' @examples
//' \dontrun{
//' set.seed(123)
//' target <- sample(0:1, 1000, replace = TRUE)
//' feature <- sample(LETTERS[1:5], 1000, replace = TRUE)
//' result <- optimal_binning_categorical_sblp(target, feature)
//' print(result)
//' }
//'
//' @export
// [[Rcpp::export]]
List optimal_binning_categorical_sblp(const IntegerVector& target,
                                     const CharacterVector& feature,
                                     int min_bins = 3,
                                     int max_bins = 5,
                                     double bin_cutoff = 0.05,
                                     int max_n_prebins = 20,
                                     double convergence_threshold = 1e-6,
                                     int max_iterations = 1000,
                                     std::string bin_separator = ";") {
 OptimalBinningCategoricalSBLP optbin(target, feature, min_bins, max_bins, bin_cutoff, max_n_prebins,
                                      convergence_threshold, max_iterations, bin_separator);
 return optbin.fit();
}


// #include <Rcpp.h>
// #include <vector>
// #include <string>
// #include <unordered_map>
// #include <algorithm>
// #include <cmath>
// #include <numeric>
// #include <limits>
// #include <stdexcept>
// #include <sstream>
// 
// using namespace Rcpp;
// 
// // Class definition for OptimalBinningCategoricalSBLP
// class OptimalBinningCategoricalSBLP {
// public:
//   OptimalBinningCategoricalSBLP(const IntegerVector& target,
//                                 const CharacterVector& feature,
//                                 int min_bins,
//                                 int max_bins,
//                                 double bin_cutoff,
//                                 int max_n_prebins,
//                                 double convergence_threshold,
//                                 int max_iterations,
//                                 std::string bin_separator);
//   
//   List fit();
//   
// private:
//   // Input data and parameters
//   const IntegerVector& target;
//   const CharacterVector& feature;
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
//   double convergence_threshold;
//   int max_iterations;
//   std::string bin_separator;
//   
//   // Internal data structures
//   std::vector<std::string> unique_categories;
//   std::vector<int> count_total;
//   std::vector<int> count_pos;
//   std::vector<int> count_neg;
//   std::vector<double> category_target_rate;
//   std::vector<size_t> sorted_indices;
//   
//   // Helper functions
//   void validate_input();
//   void compute_initial_counts();
//   void handle_rare_categories();
//   void ensure_max_prebins();
//   void sort_categories();
//   std::vector<std::vector<size_t>> perform_binning();
//   double calculate_bin_iv(const std::vector<size_t>& bin) const;
//   bool is_monotonic(const std::vector<std::vector<size_t>>& bins) const;
//   List prepare_output(const std::vector<std::vector<size_t>>& bins, bool converged, int iterations) const;
//   static std::string merge_category_names(const std::vector<std::string>& categories, const std::string& separator);
// };
// 
// // Constructor implementation
// OptimalBinningCategoricalSBLP::OptimalBinningCategoricalSBLP(
//   const IntegerVector& target,
//   const CharacterVector& feature,
//   int min_bins,
//   int max_bins,
//   double bin_cutoff,
//   int max_n_prebins,
//   double convergence_threshold,
//   int max_iterations,
//   std::string bin_separator)
//   : target(target), feature(feature),
//     min_bins(min_bins), max_bins(max_bins),
//     bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins),
//     convergence_threshold(convergence_threshold),
//     max_iterations(max_iterations),
//     bin_separator(bin_separator) {
//   // Constructor body is empty as all initializations are done in the initializer list
// }
// 
// // Input validation
// void OptimalBinningCategoricalSBLP::validate_input() {
//   if (target.size() != feature.size()) {
//     throw std::invalid_argument("Target and feature must have the same length");
//   }
//   if (min_bins < 2) {
//     throw std::invalid_argument("min_bins must be at least 2");
//   }
//   if (max_bins < min_bins) {
//     throw std::invalid_argument("max_bins must be greater than or equal to min_bins");
//   }
//   if (bin_cutoff <= 0 || bin_cutoff >= 1) {
//     throw std::invalid_argument("bin_cutoff must be between 0 and 1");
//   }
//   if (max_n_prebins < min_bins) {
//     throw std::invalid_argument("max_n_prebins must be at least equal to min_bins");
//   }
//   if (convergence_threshold <= 0) {
//     throw std::invalid_argument("convergence_threshold must be positive");
//   }
//   if (max_iterations <= 0) {
//     throw std::invalid_argument("max_iterations must be positive");
//   }
// }
// 
// // Compute initial counts for each category
// void OptimalBinningCategoricalSBLP::compute_initial_counts() {
//   std::unordered_map<std::string, size_t> category_indices;
//   
//   for (int i = 0; i < feature.size(); ++i) {
//     std::string cat = as<std::string>(feature[i]);
//     auto [it, inserted] = category_indices.emplace(cat, unique_categories.size());
//     if (inserted) {
//       unique_categories.push_back(cat);
//       count_total.push_back(0);
//       count_pos.push_back(0);
//       count_neg.push_back(0);
//     }
//     size_t idx = it->second;
//     count_total[idx]++;
//     if (target[i] == 1) {
//       count_pos[idx]++;
//     } else if (target[i] == 0) {
//       count_neg[idx]++;
//     } else {
//       throw std::invalid_argument("Target must be binary (0 or 1)");
//     }
//   }
//   
//   category_target_rate.resize(unique_categories.size());
//   for (size_t i = 0; i < unique_categories.size(); ++i) {
//     category_target_rate[i] = static_cast<double>(count_pos[i]) / count_total[i];
//   }
// }
// 
// // Handle rare categories by merging them based on similarity in target rates
// void OptimalBinningCategoricalSBLP::handle_rare_categories() {
//   int total_count = std::accumulate(count_total.begin(), count_total.end(), 0);
//   std::vector<bool> is_rare(unique_categories.size(), false);
//   
//   for (size_t i = 0; i < unique_categories.size(); ++i) {
//     double proportion = static_cast<double>(count_total[i]) / total_count;
//     if (proportion < bin_cutoff) {
//       is_rare[i] = true;
//     }
//   }
//   
//   if (std::none_of(is_rare.begin(), is_rare.end(), [](bool b) { return b; })) {
//     return;
//   }
//   
//   std::vector<size_t> rare_indices;
//   for (size_t i = 0; i < unique_categories.size(); ++i) {
//     if (is_rare[i]) {
//       rare_indices.push_back(i);
//     }
//   }
//   
//   std::sort(rare_indices.begin(), rare_indices.end(),
//             [this](size_t i, size_t j) { return category_target_rate[i] < category_target_rate[j]; });
//   
//   for (size_t i = 1; i < rare_indices.size(); ++i) {
//     size_t prev_idx = rare_indices[i - 1];
//     size_t curr_idx = rare_indices[i];
//     count_total[prev_idx] += count_total[curr_idx];
//     count_pos[prev_idx] += count_pos[curr_idx];
//     count_neg[prev_idx] += count_neg[curr_idx];
//     unique_categories[prev_idx] = merge_category_names({unique_categories[prev_idx], unique_categories[curr_idx]}, bin_separator);
//     unique_categories[curr_idx].clear();
//   }
//   
//   auto it = std::remove_if(unique_categories.begin(), unique_categories.end(),
//                            [](const std::string& s) { return s.empty(); });
//   size_t new_size = std::distance(unique_categories.begin(), it);
//   
//   unique_categories.resize(new_size);
//   count_total.resize(new_size);
//   count_pos.resize(new_size);
//   count_neg.resize(new_size);
//   
//   category_target_rate.resize(new_size);
//   for (size_t i = 0; i < new_size; ++i) {
//     category_target_rate[i] = static_cast<double>(count_pos[i]) / count_total[i];
//   }
// }
// 
// // Ensure the number of pre-bins does not exceed max_n_prebins
// void OptimalBinningCategoricalSBLP::ensure_max_prebins() {
//   if (unique_categories.size() <= static_cast<size_t>(max_n_prebins)) {
//     return;
//   }
//   
//   std::vector<size_t> indices(unique_categories.size());
//   std::iota(indices.begin(), indices.end(), 0);
//   std::sort(indices.begin(), indices.end(),
//             [this](size_t i, size_t j) { return category_target_rate[i] < category_target_rate[j]; });
//   
//   size_t bins_to_merge = unique_categories.size() - max_n_prebins;
//   for (size_t i = 0; i < bins_to_merge; ++i) {
//     size_t idx1 = indices[i];
//     size_t idx2 = indices[i + 1];
//     count_total[idx1] += count_total[idx2];
//     count_pos[idx1] += count_pos[idx2];
//     count_neg[idx1] += count_neg[idx2];
//     unique_categories[idx1] = merge_category_names({unique_categories[idx1], unique_categories[idx2]}, bin_separator);
//     unique_categories[idx2].clear();
//   }
//   
//   auto it = std::remove_if(unique_categories.begin(), unique_categories.end(),
//                            [](const std::string& s) { return s.empty(); });
//   size_t new_size = std::distance(unique_categories.begin(), it);
//   
//   unique_categories.resize(new_size);
//   count_total.resize(new_size);
//   count_pos.resize(new_size);
//   count_neg.resize(new_size);
//   
//   category_target_rate.resize(new_size);
//   for (size_t i = 0; i < new_size; ++i) {
//     category_target_rate[i] = static_cast<double>(count_pos[i]) / count_total[i];
//   }
// }
// 
// // Sort categories based on their target rates
// void OptimalBinningCategoricalSBLP::sort_categories() {
//   sorted_indices.resize(unique_categories.size());
//   std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
//   std::sort(sorted_indices.begin(), sorted_indices.end(),
//             [this](size_t i, size_t j) { return category_target_rate[i] < category_target_rate[j]; });
// }
// 
// // Perform binning using dynamic programming
// std::vector<std::vector<size_t>> OptimalBinningCategoricalSBLP::perform_binning() {
//   size_t n = sorted_indices.size();
//   size_t k = std::min(static_cast<size_t>(max_bins), n);
//   
//   // Initialize dynamic programming table
//   std::vector<std::vector<double>> dp(n + 1, std::vector<double>(k + 1, -std::numeric_limits<double>::infinity()));
//   std::vector<std::vector<size_t>> split(n + 1, std::vector<size_t>(k + 1, 0));
//   
//   // Base case: one bin
//   for (size_t i = 1; i <= n; ++i) {
//     dp[i][1] = calculate_bin_iv(std::vector<size_t>(sorted_indices.begin(), sorted_indices.begin() + i));
//   }
//   
//   // Fill the dp table
//   for (size_t j = 2; j <= k; ++j) {
//     for (size_t i = j; i <= n; ++i) {
//       for (size_t s = j - 1; s < i; ++s) {
//         double current_iv = dp[s][j-1] + calculate_bin_iv(std::vector<size_t>(sorted_indices.begin() + s, sorted_indices.begin() + i));
//         if (current_iv > dp[i][j]) {
//           dp[i][j] = current_iv;
//           split[i][j] = s;
//         }
//       }
//     }
//   }
//   
//   // Backtrack to find the optimal bins
//   std::vector<std::vector<size_t>> bins;
//   size_t i = n, j = k;
//   while (j > 0) {
//     std::vector<size_t> bin(sorted_indices.begin() + split[i][j], sorted_indices.begin() + i);
//     bins.push_back(bin);
//     i = split[i][j];
//     --j;
//   }
//   std::reverse(bins.begin(), bins.end());
//   
//   // Ensure min_bins is respected
//   while (bins.size() < static_cast<size_t>(min_bins) && bins.size() < n) {
//     // Find the bin with the highest IV to split
//     size_t max_iv_bin = std::max_element(bins.begin(), bins.end(),
//                                          [this](const std::vector<size_t>& a, const std::vector<size_t>& b) {
//                                            return calculate_bin_iv(a) < calculate_bin_iv(b);
//                                          }) - bins.begin();
//     
//     if (bins[max_iv_bin].size() > 1) {
//       size_t split_point = bins[max_iv_bin].size() / 2;
//       std::vector<size_t> new_bin(bins[max_iv_bin].begin() + split_point, bins[max_iv_bin].end());
//       bins[max_iv_bin].resize(split_point);
//       bins.insert(bins.begin() + max_iv_bin + 1, new_bin);
//     } else {
//       // If we can't split any further, break to avoid infinite loop
//       break;
//     }
//   }
//   
//   return bins;
// }
// 
// // Calculate the Information Value (IV) for a single bin
// double OptimalBinningCategoricalSBLP::calculate_bin_iv(const std::vector<size_t>& bin) const {
//   int total_pos = std::accumulate(count_pos.begin(), count_pos.end(), 0);
//   int total_neg = std::accumulate(count_neg.begin(), count_neg.end(), 0);
//   
//   int bin_pos = 0, bin_neg = 0;
//   for (size_t idx : bin) {
//     bin_pos += count_pos[idx];
//     bin_neg += count_neg[idx];
//   }
//   
//   double pos_rate = static_cast<double>(bin_pos) / total_pos;
//   double neg_rate = static_cast<double>(bin_neg) / total_neg;
//   
//   // Avoid division by zero and log(0)
//   const double epsilon = 1e-10;
//   pos_rate = std::max(pos_rate, epsilon);
//   neg_rate = std::max(neg_rate, epsilon);
//   
//   double woe = std::log(pos_rate / neg_rate);
//   double iv = (pos_rate - neg_rate) * woe;
//   
//   return iv;
// }
// 
// // Check if the bins are monotonic with respect to the target rate
// bool OptimalBinningCategoricalSBLP::is_monotonic(const std::vector<std::vector<size_t>>& bins) const {
//   std::vector<double> bin_rates;
//   for (const auto& bin : bins) {
//     int bin_total = 0, bin_pos = 0;
//     for (size_t idx : bin) {
//       bin_total += count_total[idx];
//       bin_pos += count_pos[idx];
//     }
//     bin_rates.push_back(static_cast<double>(bin_pos) / bin_total);
//   }
//   
//   return std::is_sorted(bin_rates.begin(), bin_rates.end());
// }
// 
// // Prepare the output List
// List OptimalBinningCategoricalSBLP::prepare_output(const std::vector<std::vector<size_t>>& bins, bool converged, int iterations) const {
//   std::vector<std::string> bin_names;
//   std::vector<double> bin_woe;
//   std::vector<double> bin_iv;
//   std::vector<int> bin_count;
//   std::vector<int> bin_count_pos;
//   std::vector<int> bin_count_neg;
//   
//   int total_pos = std::accumulate(count_pos.begin(), count_pos.end(), 0);
//   int total_neg = std::accumulate(count_neg.begin(), count_neg.end(), 0);
//   
//   const double epsilon = 1e-10;
//   
//   for (const auto& bin : bins) {
//     std::vector<std::string> bin_categories;
//     int bin_total = 0, bin_pos = 0, bin_neg = 0;
//     for (size_t idx : bin) {
//       bin_categories.push_back(unique_categories[idx]);
//       bin_total += count_total[idx];
//       bin_pos += count_pos[idx];
//       bin_neg += count_neg[idx];
//     }
//     
//     double pos_rate = static_cast<double>(bin_pos) / total_pos;
//     double neg_rate = static_cast<double>(bin_neg) / total_neg;
//     pos_rate = std::max(pos_rate, epsilon);
//     neg_rate = std::max(neg_rate, epsilon);
//     
//     double woe = std::log(pos_rate / neg_rate);
//     double iv = (pos_rate - neg_rate) * woe;
//     
//     bin_names.push_back(merge_category_names(bin_categories, bin_separator));
//     bin_woe.push_back(woe);
//     bin_iv.push_back(iv);
//     bin_count.push_back(bin_total);
//     bin_count_pos.push_back(bin_pos);
//     bin_count_neg.push_back(bin_neg);
//   }
//   
//   return List::create(
//     Named("bin") = bin_names,
//     Named("woe") = bin_woe,
//     Named("iv") = bin_iv,
//     Named("count") = bin_count,
//     Named("count_pos") = bin_count_pos,
//     Named("count_neg") = bin_count_neg,
//     Named("converged") = converged,
//     Named("iterations") = iterations
//   );
// }
// 
// // Merge category names
// std::string OptimalBinningCategoricalSBLP::merge_category_names(const std::vector<std::string>& categories, const std::string& separator) {
//   std::vector<std::string> unique_cats;
//   for (const auto& cat : categories) {
//     std::stringstream ss(cat);
//     std::string part;
//     while (std::getline(ss, part, ';')) {
//       if (std::find(unique_cats.begin(), unique_cats.end(), part) == unique_cats.end()) {
//         unique_cats.push_back(part);
//       }
//     }
//   }
//   std::sort(unique_cats.begin(), unique_cats.end());
//   return std::accumulate(unique_cats.begin(), unique_cats.end(), std::string(),
//                          [&separator](const std::string& a, const std::string& b) {
//                            return a.empty() ? b : a + separator + b;
//                          });
// }
// 
// // Main fitting function
// List OptimalBinningCategoricalSBLP::fit() {
//   try {
//     validate_input();
//     compute_initial_counts();
//     handle_rare_categories();
//     ensure_max_prebins();
//     sort_categories();
//     
//     // If the number of categories is less than or equal to max_bins, no optimization is needed
//     if (unique_categories.size() <= static_cast<size_t>(max_bins)) {
//       std::vector<std::vector<size_t>> bins(unique_categories.size());
//       for (size_t i = 0; i < unique_categories.size(); ++i) {
//         bins[i] = {sorted_indices[i]};
//       }
//       return prepare_output(bins, true, 0);
//     }
//     
//     std::vector<std::vector<size_t>> best_bins;
//     double best_iv = -std::numeric_limits<double>::infinity();
//     bool converged = false;
//     int iterations = 0;
//     
//     while (iterations < max_iterations) {
//       std::vector<std::vector<size_t>> current_bins = perform_binning();
//       double current_iv = std::accumulate(current_bins.begin(), current_bins.end(), 0.0,
//                                           [this](double sum, const std::vector<size_t>& bin) {
//                                             return sum + calculate_bin_iv(bin);
//                                           });
//       
//       if (std::abs(current_iv - best_iv) < convergence_threshold) {
//         converged = true;
//         break;
//       }
//       
//       if (current_iv > best_iv) {
//         best_iv = current_iv;
//         best_bins = current_bins;
//       }
//       
//       ++iterations;
//     }
//     
//     // Force monotonicity if possible
//     if (!is_monotonic(best_bins) && best_bins.size() > static_cast<size_t>(min_bins)) {
//       std::vector<std::vector<size_t>> monotonic_bins;
//       std::vector<size_t> current_bin;
//       double prev_rate = -1.0;
//       
//       for (const auto& bin : best_bins) {
//         double bin_rate = std::accumulate(bin.begin(), bin.end(), 0.0,
//                                           [this](double sum, size_t idx) {
//                                             return sum + category_target_rate[idx] * count_total[idx];
//                                           }) / std::accumulate(bin.begin(), bin.end(), 0,
//                                           [this](int sum, size_t idx) {
//                                             return sum + count_total[idx];
//                                           });
//         
//         if (bin_rate >= prev_rate || monotonic_bins.size() < static_cast<size_t>(min_bins)) {
//           if (!current_bin.empty()) {
//             monotonic_bins.push_back(current_bin);
//             current_bin.clear();
//           }
//           monotonic_bins.push_back(bin);
//           prev_rate = bin_rate;
//         } else {
//           current_bin.insert(current_bin.end(), bin.begin(), bin.end());
//         }
//       }
//       
//       if (!current_bin.empty()) {
//         monotonic_bins.push_back(current_bin);
//       }
//       
//       best_bins = monotonic_bins;
//     }
//     
//     return prepare_output(best_bins, converged, iterations);
//   } catch (const std::exception& e) {
//     Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
//   }
// }
// 
// //' Optimal Binning for Categorical Variables using Similarity-Based Logistic Partitioning (SBLP)
// //'
// //' @description
// //' This function performs optimal binning for categorical variables using a Similarity-Based Logistic Partitioning (SBLP) approach,
// //' which combines Weight of Evidence (WOE) and Information Value (IV) methods with a similarity-based merging strategy.
// //'
// //' @param target An integer vector of binary target values (0 or 1).
// //' @param feature A character vector of categorical feature values.
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff Minimum frequency for a category to be considered as a separate bin (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins before merging (default: 20).
// //' @param convergence_threshold Threshold for convergence of the algorithm (default: 1e-6).
// //' @param max_iterations Maximum number of iterations for the algorithm (default: 1000).
// //' @param bin_separator Separator used for merging category names (default: ";").
// //'
// //' @return A list containing the following elements:
// //' \itemize{
// //'   \item bins: A character vector of bin labels
// //'   \item woe: A numeric vector of Weight of Evidence values for each bin
// //'   \item iv: A numeric vector of Information Values for each bin
// //'   \item count: An integer vector of total counts for each bin
// //'   \item count_pos: An integer vector of positive class counts for each bin
// //'   \item count_neg: An integer vector of negative class counts for each bin
// //'   \item converged: A logical value indicating whether the algorithm converged
// //'   \item iterations: An integer value indicating the number of iterations performed
// //' }
// //'
// //' @details
// //' The algorithm performs the following steps:
// //' \enumerate{
// //'   \item Validate input parameters
// //'   \item Compute initial counts and target rates for each category
// //'   \item Handle rare categories by merging them based on similarity in target rates
// //'   \item Ensure the number of pre-bins does not exceed max_n_prebins
// //'   \item Sort categories based on their target rates
// //'   \item Perform iterative binning using dynamic programming
// //'   \item Enforce monotonicity in the final binning if possible
// //'   \item Calculate final statistics for each bin
// //' }
// //'
// //' The Weight of Evidence (WOE) is calculated as:
// //' \deqn{WOE = \ln(\frac{\text{Proportion of Events}}{\text{Proportion of Non-Events}})}
// //'
// //' The Information Value (IV) for each bin is calculated as:
// //' \deqn{IV = (\text{Proportion of Events} - \text{Proportion of Non-Events}) \times WOE}
// //'
// //' @examples
// //' \dontrun{
// //' # Create sample data
// //' set.seed(123)
// //' target <- sample(0:1, 1000, replace = TRUE)
// //' feature <- sample(LETTERS[1:5], 1000, replace = TRUE)
// //'
// //' # Run optimal binning
// //' result <- optimal_binning_categorical_sblp(target, feature)
// //'
// //' # View results
// //' print(result)
// //' }
// //'
// // [[Rcpp::export]]
// List optimal_binning_categorical_sblp(const IntegerVector& target,
//                                      const CharacterVector& feature,
//                                      int min_bins = 3,
//                                      int max_bins = 5,
//                                      double bin_cutoff = 0.05,
//                                      int max_n_prebins = 20,
//                                      double convergence_threshold = 1e-6,
//                                      int max_iterations = 1000,
//                                      std::string bin_separator = ";") {
//  OptimalBinningCategoricalSBLP optbin(target, feature, min_bins, max_bins, bin_cutoff, max_n_prebins,
//                                       convergence_threshold, max_iterations, bin_separator);
//  return optbin.fit();
// }
