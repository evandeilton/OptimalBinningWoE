// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <string>
#include <cmath>
#include <numeric>
#include <sstream>
#include <limits>
#include <unordered_set>

using namespace Rcpp;

// Classe para Binning Ótimo Numérico usando Local Density Binning (LDB)
class OptimalBinningNumericalLDB {
private:
  // Parâmetros
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  double convergence_threshold;
  int max_iterations;
  
  // Status de convergência e iterações
  bool converged;
  int iterations_run;
  
  // Dados
  std::vector<double> feature;
  std::vector<int> target;
  
  // Estruturas de binning
  std::vector<double> bin_edges;
  std::vector<double> woe_values;
  std::vector<double> iv_values;
  std::vector<int> counts;
  std::vector<int> count_pos;
  std::vector<int> count_neg;
  std::vector<std::string> bin_labels;
  
  // IV total
  double total_iv;
  
  // Métodos privados
  void compute_prebins();
  void compute_woe_iv();
  void enforce_monotonicity();
  void merge_bins();
  void create_bin_labels();
  
  // Métodos utilitários
  double calculateWOE(int pos, int neg, double total_pos, double total_neg) const;
  double calculateIV(double woe, int pos, int neg, double total_pos, double total_neg) const;
  
public:
  // Construtor
  OptimalBinningNumericalLDB(int min_bins = 3, int max_bins = 5, double bin_cutoff = 0.05,
                             int max_n_prebins = 20, double convergence_threshold = 1e-6,
                             int max_iterations = 1000);
  
  // Ajuste do modelo
  void fit(const std::vector<double>& feature_input, const std::vector<int>& target_input);
  
  // Transform para obter resultados
  Rcpp::List transform();
};

// Construtor
OptimalBinningNumericalLDB::OptimalBinningNumericalLDB(int min_bins, int max_bins, double bin_cutoff,
                                                       int max_n_prebins, double convergence_threshold,
                                                       int max_iterations) {
  this->min_bins = min_bins;
  this->max_bins = max_bins;
  this->bin_cutoff = bin_cutoff;
  this->max_n_prebins = max_n_prebins;
  this->convergence_threshold = convergence_threshold;
  this->max_iterations = max_iterations;
  this->total_iv = 0.0;
  this->converged = true;
  this->iterations_run = 0;
}

// Cálculo de WOE com suavização Laplace
double OptimalBinningNumericalLDB::calculateWOE(int pos, int neg, double total_pos, double total_neg) const {
  double good = static_cast<double>(pos);
  double bad = static_cast<double>(neg);
  
  // Suavização Laplace
  good = (good + 0.5) / (total_pos + 1.0);
  bad = (bad + 0.5) / (total_neg + 1.0);
  
  // Evita log(0)
  double epsilon = 1e-14;
  good = std::max(good, epsilon);
  bad = std::max(bad, epsilon);
  
  return std::log(good / bad);
}

// Cálculo de IV
double OptimalBinningNumericalLDB::calculateIV(double woe, int pos, int neg, double total_pos, double total_neg) const {
  double dist_good = (pos > 0) ? static_cast<double>(pos) / total_pos : 0.0;
  double dist_bad = (neg > 0) ? static_cast<double>(neg) / total_neg : 0.0;
  return (dist_good - dist_bad) * woe;
}

// Método fit
void OptimalBinningNumericalLDB::fit(const std::vector<double>& feature_input, const std::vector<int>& target_input) {
  // Validações de entrada
  if (feature_input.empty() || target_input.empty()) {
    Rcpp::stop("Feature and target vectors must not be empty.");
  }
  
  if (feature_input.size() != target_input.size()) {
    Rcpp::stop("Feature and target must have the same length.");
  }
  
  if (min_bins < 2) {
    Rcpp::stop("min_bins must be at least 2.");
  }
  
  if (max_bins < min_bins) {
    Rcpp::stop("max_bins must be greater than or equal to min_bins.");
  }
  
  if (bin_cutoff < 0.0 || bin_cutoff > 1.0) {
    Rcpp::stop("bin_cutoff must be between 0 and 1.");
  }
  
  if (max_n_prebins < min_bins) {
    Rcpp::stop("max_n_prebins must be greater than or equal to min_bins.");
  }
  
  // Valida se target é binário e contém 0 e 1
  std::unordered_set<int> target_set(target_input.begin(), target_input.end());
  if (target_set.find(0) == target_set.end() || target_set.find(1) == target_set.end()) {
    Rcpp::stop("Target must contain at least one 0 and one 1.");
  }
  
  this->feature = feature_input;
  this->target = target_input;
  
  // Número de valores únicos
  std::vector<double> unique_feature = feature_input;
  std::sort(unique_feature.begin(), unique_feature.end());
  unique_feature.erase(std::unique(unique_feature.begin(), unique_feature.end()), unique_feature.end());
  
  int unique_count = static_cast<int>(unique_feature.size());
  
  // Caso <= 2 valores únicos
  if (unique_count <= 2) {
    bin_edges.clear();
    bin_edges.push_back(-std::numeric_limits<double>::infinity());
    
    if (unique_count == 1) {
      bin_edges.push_back(std::numeric_limits<double>::infinity());
    } else if (unique_count == 2) {
      double midpoint = (unique_feature[0] + unique_feature[1]) / 2.0;
      bin_edges.push_back(midpoint);
      bin_edges.push_back(std::numeric_limits<double>::infinity());
    }
    
    compute_woe_iv();
    create_bin_labels();
    converged = true;
    iterations_run = 0;
    return;
  }
  
  // Caso <= min_bins valores únicos
  if (unique_count <= min_bins) {
    bin_edges.clear();
    bin_edges.push_back(-std::numeric_limits<double>::infinity());
    for (int i = 1; i < unique_count; ++i) {
      double mid = (unique_feature[i - 1] + unique_feature[i]) / 2.0;
      bin_edges.push_back(mid);
    }
    bin_edges.push_back(std::numeric_limits<double>::infinity());
    
    compute_woe_iv();
    create_bin_labels();
    converged = true;
    iterations_run = 0;
    return;
  }
  
  // Caso geral: mais do que min_bins valores
  compute_prebins();
  compute_woe_iv();
  enforce_monotonicity();
  merge_bins();
  create_bin_labels();
  
  // Verifica se convergência foi atingida
  converged = (iterations_run < max_iterations);
}

// Computa pré-bins
void OptimalBinningNumericalLDB::compute_prebins() {
  size_t n = feature.size();
  std::vector<double> sorted_feature = feature;
  std::sort(sorted_feature.begin(), sorted_feature.end());
  
  bin_edges.clear();
  bin_edges.push_back(-std::numeric_limits<double>::infinity());
  
  // Cria cortes baseados em quantis
  for (int i = 1; i < max_n_prebins; ++i) {
    size_t idx = static_cast<size_t>(std::floor(n * i / static_cast<double>(max_n_prebins)));
    if (idx >= n) idx = n - 1;
    double edge = sorted_feature[idx];
    bin_edges.push_back(edge);
  }
  
  bin_edges.push_back(std::numeric_limits<double>::infinity());
  
  // Remove duplicatas
  bin_edges.erase(std::unique(bin_edges.begin(), bin_edges.end()), bin_edges.end());
}

// Computa WoE e IV
void OptimalBinningNumericalLDB::compute_woe_iv() {
  size_t n = feature.size();
  size_t num_bins = bin_edges.size() - 1;
  
  counts.assign(num_bins, 0);
  count_pos.assign(num_bins, 0);
  count_neg.assign(num_bins, 0);
  woe_values.assign(num_bins, 0.0);
  iv_values.assign(num_bins, 0.0);
  
  double total_pos = std::accumulate(target.begin(), target.end(), 0.0);
  double total_neg = static_cast<double>(n) - total_pos;
  
  if (total_pos <= 0.0 || total_neg <= 0.0) {
    Rcpp::stop("Target vector must contain both positive and negative cases.");
  }
  
  // Atribui cada ponto ao bin
  for (size_t i = 0; i < n; ++i) {
    double x = feature[i];
    int t = target[i];
    int bin = -1;
    
    // Busca bin
    // Otimização: poderia usar busca binária, mas mantendo por simplicidade
    for (size_t b = 0; b < num_bins; ++b) {
      if (x > bin_edges[b] && x <= bin_edges[b + 1]) {
        bin = static_cast<int>(b);
        break;
      }
    }
    if (bin == -1) {
      // Fallback: se não encontrou, pode ser caso extremo
      if (x <= bin_edges.front()) {
        bin = 0;
      } else if (x > bin_edges.back()) {
        bin = static_cast<int>(num_bins - 1);
      } else {
        Rcpp::stop("Error assigning data point to a bin.");
      }
    }
    
    counts[static_cast<size_t>(bin)]++;
    if (t == 1) {
      count_pos[static_cast<size_t>(bin)]++;
    } else {
      count_neg[static_cast<size_t>(bin)]++;
    }
  }
  
  for (size_t b = 0; b < num_bins; ++b) {
    woe_values[b] = calculateWOE(count_pos[b], count_neg[b], total_pos, total_neg);
    iv_values[b] = calculateIV(woe_values[b], count_pos[b], count_neg[b], total_pos, total_neg);
  }
}

// Impõe monotonicidade
void OptimalBinningNumericalLDB::enforce_monotonicity() {
  if (counts.size() <= static_cast<size_t>(min_bins)) {
    return;
  }
  
  // Determina direção da monotonicidade
  int direction = 0;
  // Tenta inferir a direção a partir dos primeiros bins
  for (size_t b = 1; b < woe_values.size(); ++b) {
    double diff = woe_values[b] - woe_values[b - 1];
    if (diff > 0) {
      direction++;
    } else if (diff < 0) {
      direction--;
    }
  }
  
  // direction > 0 sugere tendência crescente, < 0 decrescente
  bool monotonic = false;
  int iter = 0;
  
  while (!monotonic && counts.size() > static_cast<size_t>(min_bins) && (iterations_run + iter) < max_iterations) {
    monotonic = true;
    for (size_t b = 1; b < woe_values.size(); ++b) {
      double diff = woe_values[b] - woe_values[b - 1];
      if ((direction >= 0 && diff < 0) || (direction < 0 && diff > 0)) {
        // Merge bins b-1 e b
        int b1 = static_cast<int>(b - 1);
        int b2 = static_cast<int>(b);
        
        counts[b1] += counts[b2];
        count_pos[b1] += count_pos[b2];
        count_neg[b1] += count_neg[b2];
        
        counts.erase(counts.begin() + b2);
        count_pos.erase(count_pos.begin() + b2);
        count_neg.erase(count_neg.begin() + b2);
        bin_edges.erase(bin_edges.begin() + b2);
        woe_values.erase(woe_values.begin() + b2);
        iv_values.erase(iv_values.begin() + b2);
        
        double total_pos = std::accumulate(count_pos.begin(), count_pos.end(), 0.0);
        double total_neg = std::accumulate(count_neg.begin(), count_neg.end(), 0.0);
        
        woe_values[b1] = calculateWOE(count_pos[b1], count_neg[b1], total_pos, total_neg);
        iv_values[b1] = calculateIV(woe_values[b1], count_pos[b1], count_neg[b1], total_pos, total_neg);
        
        monotonic = false;
        break;
      }
    }
    iter++;
  }
  
  iterations_run += iter;
}

// Merge bins
void OptimalBinningNumericalLDB::merge_bins() {
  size_t n = feature.size();
  double min_bin_count = bin_cutoff * static_cast<double>(n);
  
  bool bins_merged = true;
  
  // Merge bins com frequência baixa
  while (bins_merged && counts.size() > static_cast<size_t>(min_bins) && iterations_run < max_iterations) {
    bins_merged = false;
    
    for (size_t b = 0; b < counts.size(); ++b) {
      if (counts[b] < min_bin_count && counts.size() > static_cast<size_t>(min_bins)) {
        bins_merged = true;
        size_t merge_with;
        if (b == 0) {
          merge_with = b + 1;
        } else if (b == counts.size() - 1) {
          merge_with = b - 1;
        } else {
          // Merge com vizinho com menor contagem
          merge_with = (counts[b - 1] <= counts[b + 1]) ? (b - 1) : (b + 1);
        }
        
        counts[merge_with] += counts[b];
        count_pos[merge_with] += count_pos[b];
        count_neg[merge_with] += count_neg[b];
        
        counts.erase(counts.begin() + b);
        count_pos.erase(count_pos.begin() + b);
        count_neg.erase(count_neg.begin() + b);
        bin_edges.erase(bin_edges.begin() + b);
        woe_values.erase(woe_values.begin() + b);
        iv_values.erase(iv_values.begin() + b);
        
        double total_pos = std::accumulate(count_pos.begin(), count_pos.end(), 0.0);
        double total_neg = std::accumulate(count_neg.begin(), count_neg.end(), 0.0);
        
        woe_values[merge_with] = calculateWOE(count_pos[merge_with], count_neg[merge_with], total_pos, total_neg);
        iv_values[merge_with] = calculateIV(woe_values[merge_with], count_pos[merge_with], count_neg[merge_with], total_pos, total_neg);
        
        break;
      }
    }
    iterations_run++;
  }
  
  // Garante que número de bins não exceda max_bins
  while (counts.size() > static_cast<size_t>(max_bins) && iterations_run < max_iterations) {
    // Achar bin com menor IV para mesclar
    size_t min_iv_idx = 0;
    double min_iv = iv_values[0];
    for (size_t b = 1; b < iv_values.size(); ++b) {
      if (iv_values[b] < min_iv) {
        min_iv = iv_values[b];
        min_iv_idx = b;
      }
    }
    
    size_t merge_with = (min_iv_idx == 0) ? min_iv_idx + 1 : min_iv_idx - 1;
    
    counts[merge_with] += counts[min_iv_idx];
    count_pos[merge_with] += count_pos[min_iv_idx];
    count_neg[merge_with] += count_neg[min_iv_idx];
    
    counts.erase(counts.begin() + min_iv_idx);
    count_pos.erase(count_pos.begin() + min_iv_idx);
    count_neg.erase(count_neg.begin() + min_iv_idx);
    bin_edges.erase(bin_edges.begin() + min_iv_idx);
    woe_values.erase(woe_values.begin() + min_iv_idx);
    iv_values.erase(iv_values.begin() + min_iv_idx);
    
    double total_pos = std::accumulate(count_pos.begin(), count_pos.end(), 0.0);
    double total_neg = std::accumulate(count_neg.begin(), count_neg.end(), 0.0);
    woe_values[merge_with] = calculateWOE(count_pos[merge_with], count_neg[merge_with], total_pos, total_neg);
    iv_values[merge_with] = calculateIV(woe_values[merge_with], count_pos[merge_with], count_neg[merge_with], total_pos, total_neg);
    
    iterations_run++;
  }
  
  // Calcula IV total
  total_iv = std::accumulate(iv_values.begin(), iv_values.end(), 0.0);
}

// Cria labels dos bins
void OptimalBinningNumericalLDB::create_bin_labels() {
  bin_labels.clear();
  size_t num_bins = bin_edges.size() - 1;
  bin_labels.reserve(num_bins);
  
  for (size_t b = 0; b < num_bins; ++b) {
    std::ostringstream oss;
    oss.precision(6);
    oss << std::fixed;
    oss << "(";
    if (bin_edges[b] == -std::numeric_limits<double>::infinity()) {
      oss << "-Inf";
    } else {
      oss << bin_edges[b];
    }
    oss << ";";
    if (bin_edges[b + 1] == std::numeric_limits<double>::infinity()) {
      oss << "+Inf";
    } else {
      oss << bin_edges[b + 1];
    }
    oss << "]";
    bin_labels.emplace_back(oss.str());
  }
}

// Retorna resultados
Rcpp::List OptimalBinningNumericalLDB::transform() {
  std::vector<double> cutpoints;
  if (bin_edges.size() > 2) {
    cutpoints.assign(bin_edges.begin() + 1, bin_edges.end() - 1);
  }
  
  return Rcpp::List::create(
    Rcpp::Named("bins") = bin_labels,
    Rcpp::Named("woe") = woe_values,
    Rcpp::Named("iv") = iv_values,
    Rcpp::Named("count") = counts,
    Rcpp::Named("count_pos") = count_pos,
    Rcpp::Named("count_neg") = count_neg,
    Rcpp::Named("cutpoints") = cutpoints,
    Rcpp::Named("converged") = converged,
    Rcpp::Named("iterations") = iterations_run
  );
}

//' @title Optimal Binning for Numerical Variables using Local Density Binning (LDB)
//'
//' @description Implementa o algoritmo Local Density Binning (LDB) para binning ótimo de variáveis numéricas.
//'
//' @param target Vetor inteiro binário (0 ou 1).
//' @param feature Vetor numérico a ser binned.
//' @param min_bins Número mínimo de bins (default: 3).
//' @param max_bins Número máximo de bins (default: 5).
//' @param bin_cutoff Frequência mínima para um bin (default: 0.05).
//' @param max_n_prebins Número máximo de pré-bins (default: 20).
//' @param convergence_threshold Limite de convergência (default: 1e-6).
//' @param max_iterations Máximo de iterações (default: 1000).
//'
//' @return Uma lista com bins, woe, iv, contagens, cutpoints, converged e iterations.
//'
//' @examples
//' \dontrun{
//' set.seed(123)
//' target <- sample(0:1, 1000, replace = TRUE)
//' feature <- rnorm(1000)
//' result <- optimal_binning_numerical_ldb(target, feature)
//' print(result$bins)
//' print(result$woe)
//' print(result$iv)
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_ldb(Rcpp::IntegerVector target,
                                        Rcpp::NumericVector feature,
                                        int min_bins = 3,
                                        int max_bins = 5,
                                        double bin_cutoff = 0.05,
                                        int max_n_prebins = 20,
                                        double convergence_threshold = 1e-6,
                                        int max_iterations = 1000) {
 std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
 std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
 
 OptimalBinningNumericalLDB ob(min_bins, max_bins, bin_cutoff, max_n_prebins,
                               convergence_threshold, max_iterations);
 
 ob.fit(feature_vec, target_vec);
 
 return ob.transform();
}

/*
 Melhorias Realizadas:
 - Adicionada verificação robusta e fallback caso algum ponto não seja atribuído a um bin esperado.
 - Evitado log(0) ao calcular WOE com epsilon.
 - Garantia de que max_bins, min_bins e outras condições de entrada sejam validadas.
 - Processo de enforce_monotonicity e merge_bins revisado para evitar loops infinitos, controlado por max_iterations.
 - Comentários e mensagens de erro mais claros.
 - Cálculo de WOE/IV com suavização Laplace e checagens para evitar situações indefinidas.
 - Mantida a estrutura de entrada/saída conforme solicitado.
*/



// #include <Rcpp.h>
// #include <algorithm>
// #include <vector>
// #include <string>
// #include <cmath>
// #include <numeric>
// #include <sstream>
// #include <limits>
// #include <unordered_set>
// 
// using namespace Rcpp;
// 
// // Class for Optimal Binning Numerical using Local Density Binning (LDB)
// class OptimalBinningNumericalLDB {
// private:
//   // Parameters
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
//   double convergence_threshold;
//   int max_iterations;
//   bool converged;
//   int iterations_run;
//   
//   // Data vectors
//   std::vector<double> feature;
//   std::vector<int> target;
//   
//   // Binning structures
//   std::vector<double> bin_edges;
//   std::vector<double> woe_values;
//   std::vector<double> iv_values;
//   std::vector<int> counts;
//   std::vector<int> count_pos;
//   std::vector<int> count_neg;
//   std::vector<std::string> bin_labels;
//   
//   // Total Information Value
//   double total_iv;
//   
//   // Private methods
//   void compute_prebins();
//   void compute_woe_iv();
//   void enforce_monotonicity();
//   void merge_bins();
//   void create_bin_labels();
//   
//   // Utility methods
//   double calculateWOE(int pos, int neg, double total_pos, double total_neg) const;
//   double calculateIV(double woe, int pos, int neg, double total_pos, double total_neg) const;
//   
// public:
//   // Constructor
//   OptimalBinningNumericalLDB(int min_bins = 3, int max_bins = 5, double bin_cutoff = 0.05,
//                              int max_n_prebins = 20, double convergence_threshold = 1e-6,
//                              int max_iterations = 1000);
//   
//   // Fit method
//   void fit(const std::vector<double>& feature_input, const std::vector<int>& target_input);
//   
//   // Transform method to get results
//   Rcpp::List transform();
// };
// 
// // Constructor Implementation
// OptimalBinningNumericalLDB::OptimalBinningNumericalLDB(int min_bins, int max_bins, double bin_cutoff,
//                                                        int max_n_prebins, double convergence_threshold,
//                                                        int max_iterations) {
//   this->min_bins = min_bins;
//   this->max_bins = max_bins;
//   this->bin_cutoff = bin_cutoff;
//   this->max_n_prebins = max_n_prebins;
//   this->convergence_threshold = convergence_threshold;
//   this->max_iterations = max_iterations;
//   this->total_iv = 0.0;
//   this->converged = true;
//   this->iterations_run = 0;
// }
// 
// // Calculate WoE with Laplace smoothing
// double OptimalBinningNumericalLDB::calculateWOE(int pos, int neg, double total_pos, double total_neg) const {
//   double good = static_cast<double>(pos);
//   double bad = static_cast<double>(neg);
//   
//   // Apply Laplace smoothing
//   good = (good + 0.5) / (total_pos + 1.0);
//   bad = (bad + 0.5) / (total_neg + 1.0);
//   
//   return std::log(good / bad);
// }
// 
// // Calculate IV
// double OptimalBinningNumericalLDB::calculateIV(double woe, int pos, int neg, double total_pos, double total_neg) const {
//   double dist_good = static_cast<double>(pos) / total_pos;
//   double dist_bad = static_cast<double>(neg) / total_neg;
//   return (dist_good - dist_bad) * woe;
// }
// 
// // Fit method Implementation
// void OptimalBinningNumericalLDB::fit(const std::vector<double>& feature_input, const std::vector<int>& target_input) {
//   // Input validation
//   if (feature_input.empty() || target_input.empty()) {
//     Rcpp::stop("Feature and target vectors must not be empty.");
//   }
//   
//   if (feature_input.size() != target_input.size()) {
//     Rcpp::stop("Feature and target must have the same length.");
//   }
//   
//   if (min_bins < 2) {
//     Rcpp::stop("min_bins must be at least 2.");
//   }
//   
//   if (max_bins < min_bins) {
//     Rcpp::stop("max_bins must be greater than or equal to min_bins.");
//   }
//   
//   if (bin_cutoff < 0.0 || bin_cutoff > 1.0) {
//     Rcpp::stop("bin_cutoff must be between 0 and 1.");
//   }
//   
//   if (max_n_prebins < min_bins) {
//     Rcpp::stop("max_n_prebins must be greater than or equal to min_bins.");
//   }
//   
//   // Validate target values (must contain only 0 and 1)
//   std::unordered_set<int> target_set(target_input.begin(), target_input.end());
//   if (target_set.find(0) == target_set.end() || target_set.find(1) == target_set.end()) {
//     Rcpp::stop("Target must contain at least one 0 and one 1.");
//   }
//   
//   this->feature = feature_input;
//   this->target = target_input;
//   
//   // Check the number of unique values in the feature
//   std::vector<double> unique_feature = feature_input;
//   std::sort(unique_feature.begin(), unique_feature.end());
//   unique_feature.erase(std::unique(unique_feature.begin(), unique_feature.end()), unique_feature.end());
//   
//   // Handle cases where the number of unique values is less than or equal to 2
//   if (static_cast<int>(unique_feature.size()) <= 2) {
//     // Create bin_edges based on unique_feature
//     bin_edges.clear();
//     bin_edges.push_back(-std::numeric_limits<double>::infinity());
//     
//     if (unique_feature.size() == 1) {
//       // Only one unique value, single bin
//       bin_edges.push_back(std::numeric_limits<double>::infinity());
//     } else if (unique_feature.size() == 2) {
//       // Two unique values, create a midpoint for binning
//       double midpoint = (unique_feature[0] + unique_feature[1]) / 2.0;
//       bin_edges.push_back(midpoint);
//       bin_edges.push_back(std::numeric_limits<double>::infinity());
//     }
//     
//     // Compute WoE and IV for these bins
//     compute_woe_iv();
//     create_bin_labels();
//     
//     // Since no optimization was performed, set converged to true and iterations to 0
//     converged = true;
//     iterations_run = 0;
//     return;
//   }
//   
//   // Handle cases where the number of unique values is less than or equal to min_bins
//   if (static_cast<int>(unique_feature.size()) <= min_bins) {
//     // No need to optimize; create bins based on unique_feature
//     bin_edges.clear();
//     bin_edges.push_back(-std::numeric_limits<double>::infinity());
//     for (size_t i = 1; i < unique_feature.size(); ++i) {
//       bin_edges.push_back((unique_feature[i - 1] + unique_feature[i]) / 2.0);
//     }
//     bin_edges.push_back(std::numeric_limits<double>::infinity());
//     
//     compute_woe_iv();
//     create_bin_labels();
//     
//     // Since we did not iterate, set converged to true
//     converged = true;
//     iterations_run = 0;
//     return;
//   }
//   
//   // Execute binning steps for cases with more than min_bins unique values
//   compute_prebins();
//   compute_woe_iv();
//   enforce_monotonicity();
//   merge_bins();
//   create_bin_labels();
//   
//   // Set convergence status
//   converged = iterations_run < max_iterations;
// }
// 
// // Compute Pre-bins Implementation
// void OptimalBinningNumericalLDB::compute_prebins() {
//   size_t n = feature.size();
//   std::vector<double> sorted_feature = feature;
//   std::sort(sorted_feature.begin(), sorted_feature.end());
//   
//   // Generate initial bin edges based on quantiles
//   bin_edges.clear();
//   bin_edges.emplace_back(-std::numeric_limits<double>::infinity());
//   
//   for (int i = 1; i < max_n_prebins; ++i) {
//     size_t idx = static_cast<size_t>(n * i / static_cast<double>(max_n_prebins));
//     if (idx >= n) idx = n - 1;
//     double edge = sorted_feature[idx];
//     bin_edges.emplace_back(edge);
//   }
//   
//   bin_edges.emplace_back(std::numeric_limits<double>::infinity());
//   
//   // Remove duplicate edges to ensure unique bin boundaries
//   bin_edges.erase(std::unique(bin_edges.begin(), bin_edges.end()), bin_edges.end());
// }
// 
// // Compute WoE and IV Implementation
// void OptimalBinningNumericalLDB::compute_woe_iv() {
//   size_t n = feature.size();
//   size_t num_bins = bin_edges.size() - 1;
//   
//   // Initialize vectors
//   counts.assign(num_bins, 0);
//   count_pos.assign(num_bins, 0);
//   count_neg.assign(num_bins, 0);
//   woe_values.assign(num_bins, 0.0);
//   iv_values.assign(num_bins, 0.0);
//   
//   // Calculate total positives and negatives
//   double total_pos = std::accumulate(target.begin(), target.end(), 0.0);
//   double total_neg = static_cast<double>(n) - total_pos;
//   
//   if (total_pos == 0.0 || total_neg == 0.0) {
//     Rcpp::stop("Target vector must contain both positive and negative cases.");
//   }
//   
//   // Assign feature values to bins
//   for (size_t i = 0; i < n; ++i) {
//     double x = feature[i];
//     int bin = -1;
//     for (size_t b = 0; b < num_bins; ++b) {
//       if (x > bin_edges[b] && x <= bin_edges[b + 1]) {
//         bin = static_cast<int>(b);
//         break;
//       }
//     }
//     if (bin == -1) {
//       Rcpp::stop("Error assigning data point to pre-bin.");
//     }
//     counts[bin]++;
//     if (target[i] == 1) {
//       count_pos[bin]++;
//     } else {
//       count_neg[bin]++;
//     }
//   }
//   
//   // Compute WoE and IV for each bin
//   for (size_t b = 0; b < num_bins; ++b) {
//     woe_values[b] = calculateWOE(count_pos[b], count_neg[b], total_pos, total_neg);
//     iv_values[b] = calculateIV(woe_values[b], count_pos[b], count_neg[b], total_pos, total_neg);
//   }
// }
// 
// // Enforce Monotonicity Implementation
// void OptimalBinningNumericalLDB::enforce_monotonicity() {
//   // Determine the direction of monotonicity based on WoE trends
//   std::vector<int> woe_trends;
//   for (size_t b = 1; b < woe_values.size(); ++b) {
//     double diff = woe_values[b] - woe_values[b - 1];
//     if (diff > 0) {
//       woe_trends.push_back(1); // Increasing
//     } else if (diff < 0) {
//       woe_trends.push_back(-1); // Decreasing
//     } else {
//       woe_trends.push_back(0); // No change
//     }
//   }
//   
//   // Determine predominant trend
//   int sum_trends = std::accumulate(woe_trends.begin(), woe_trends.end(), 0);
//   int direction = (sum_trends >= 0) ? 1 : -1; // 1 for increasing, -1 for decreasing
//   
//   bool monotonicity_violated = true;
//   
//   int iterations = 0;
//   
//   // Iteratively merge bins until monotonicity is enforced or min_bins is reached
//   while (monotonicity_violated && counts.size() > static_cast<size_t>(min_bins) &&
//          iterations_run + iterations < max_iterations) {
//     monotonicity_violated = false;
//     size_t merge_idx = 0;
//     double min_iv_loss = std::numeric_limits<double>::max();
//     
//     // Identify the first bin pair that violates monotonicity
//     for (size_t b = 1; b < woe_values.size(); ++b) {
//       double diff = woe_values[b] - woe_values[b - 1];
//       if ((direction == 1 && diff < 0) || (direction == -1 && diff > 0)) {
//         // Calculate IV loss if these bins are merged
//         double merged_good = static_cast<double>(count_pos[b - 1] + count_pos[b]);
//         double merged_bad = static_cast<double>(count_neg[b - 1] + count_neg[b]);
//         double total_pos = std::accumulate(count_pos.begin(), count_pos.end(), 0.0);
//         double total_neg = std::accumulate(count_neg.begin(), count_neg.end(), 0.0);
//         double merged_woe = calculateWOE(static_cast<int>(merged_good), static_cast<int>(merged_bad), total_pos, total_neg);
//         double merged_iv = calculateIV(merged_woe, static_cast<int>(merged_good), static_cast<int>(merged_bad), total_pos, total_neg);
//         double iv_loss = iv_values[b - 1] + iv_values[b] - merged_iv;
//         
//         if (iv_loss < min_iv_loss) {
//           min_iv_loss = iv_loss;
//           merge_idx = b;
//         }
//         
//         monotonicity_violated = true;
//       }
//     }
//     
//     // Merge the identified bin pair with the least IV loss
//     if (monotonicity_violated) {
//       size_t b = merge_idx;
//       size_t merge_with = b - 1;
//       
//       // Update bin counts
//       counts[merge_with] += counts[b];
//       count_pos[merge_with] += count_pos[b];
//       count_neg[merge_with] += count_neg[b];
//       
//       // Remove the merged bin
//       counts.erase(counts.begin() + b);
//       count_pos.erase(count_pos.begin() + b);
//       count_neg.erase(count_neg.begin() + b);
//       bin_edges.erase(bin_edges.begin() + b);
//       woe_values.erase(woe_values.begin() + b);
//       iv_values.erase(iv_values.begin() + b);
//       
//       // Recalculate WoE and IV for the merged bin
//       double good = static_cast<double>(count_pos[merge_with]);
//       double bad = static_cast<double>(count_neg[merge_with]);
//       double total_pos = std::accumulate(count_pos.begin(), count_pos.end(), 0.0);
//       double total_neg = std::accumulate(count_neg.begin(), count_neg.end(), 0.0);
//       woe_values[merge_with] = calculateWOE(static_cast<int>(good), static_cast<int>(bad), total_pos, total_neg);
//       iv_values[merge_with] = calculateIV(woe_values[merge_with], static_cast<int>(good), static_cast<int>(bad), total_pos, total_neg);
//     }
//     
//     iterations++;
//   }
//   
//   iterations_run += iterations;
// }
// 
// // Merge Bins Implementation
// void OptimalBinningNumericalLDB::merge_bins() {
//   size_t n = feature.size();
//   double min_bin_count = bin_cutoff * static_cast<double>(n);
//   
//   bool bins_merged = true;
//   
//   // Merge bins with counts below bin_cutoff
//   while (bins_merged && counts.size() > static_cast<size_t>(min_bins) && iterations_run < max_iterations) {
//     bins_merged = false;
//     
//     for (size_t b = 0; b < counts.size(); ++b) {
//       if (counts[b] < min_bin_count) {
//         bins_merged = true;
//         // Determine merge direction
//         size_t merge_with;
//         if (b == 0) {
//           merge_with = b + 1;
//         } else if (b == counts.size() - 1) {
//           merge_with = b - 1;
//         } else {
//           // Merge with the neighbor with the smaller count
//           merge_with = (counts[b - 1] <= counts[b + 1]) ? (b - 1) : (b + 1);
//         }
//         
//         // Merge bins
//         counts[merge_with] += counts[b];
//         count_pos[merge_with] += count_pos[b];
//         count_neg[merge_with] += count_neg[b];
//         
//         // Remove the merged bin
//         counts.erase(counts.begin() + b);
//         count_pos.erase(count_pos.begin() + b);
//         count_neg.erase(count_neg.begin() + b);
//         bin_edges.erase(bin_edges.begin() + b);
//         woe_values.erase(woe_values.begin() + b);
//         iv_values.erase(iv_values.begin() + b);
//         
//         // Recalculate WoE and IV for the merged bin
//         double good = static_cast<double>(count_pos[merge_with]);
//         double bad = static_cast<double>(count_neg[merge_with]);
//         double total_pos = std::accumulate(count_pos.begin(), count_pos.end(), 0.0);
//         double total_neg = std::accumulate(count_neg.begin(), count_neg.end(), 0.0);
//         woe_values[merge_with] = calculateWOE(static_cast<int>(good), static_cast<int>(bad), total_pos, total_neg);
//         iv_values[merge_with] = calculateIV(woe_values[merge_with], static_cast<int>(good), static_cast<int>(bad), total_pos, total_neg);
//         
//         break; // Restart the loop after a merge
//       }
//     }
//     iterations_run++;
//   }
//   
//   // Ensure number of bins does not exceed max_bins
//   while (counts.size() > static_cast<size_t>(max_bins) && iterations_run < max_iterations) {
//     // Find the bin with the smallest IV
//     size_t min_iv_idx = 0;
//     double min_iv = iv_values[0];
//     for (size_t b = 1; b < iv_values.size(); ++b) {
//       if (iv_values[b] < min_iv) {
//         min_iv = iv_values[b];
//         min_iv_idx = b;
//       }
//     }
//     
//     // Merge with adjacent bin (prefer left)
//     size_t merge_with = (min_iv_idx == 0) ? min_iv_idx + 1 : min_iv_idx - 1;
//     
//     // Merge bins
//     counts[merge_with] += counts[min_iv_idx];
//     count_pos[merge_with] += count_pos[min_iv_idx];
//     count_neg[merge_with] += count_neg[min_iv_idx];
//     
//     // Remove the merged bin
//     counts.erase(counts.begin() + min_iv_idx);
//     count_pos.erase(count_pos.begin() + min_iv_idx);
//     count_neg.erase(count_neg.begin() + min_iv_idx);
//     bin_edges.erase(bin_edges.begin() + min_iv_idx);
//     woe_values.erase(woe_values.begin() + min_iv_idx);
//     iv_values.erase(iv_values.begin() + min_iv_idx);
//     
//     iterations_run++;
//   }
//   
//   // Calculate total IV
//   total_iv = std::accumulate(iv_values.begin(), iv_values.end(), 0.0);
// }
// 
// // Create Bin Labels Implementation
// void OptimalBinningNumericalLDB::create_bin_labels() {
//   bin_labels.clear();
//   size_t num_bins = bin_edges.size() - 1;
//   bin_labels.reserve(num_bins);
//   
//   for (size_t b = 0; b < num_bins; ++b) {
//     std::ostringstream oss;
//     oss << "(";
//     if (bin_edges[b] == -std::numeric_limits<double>::infinity()) {
//       oss << "-Inf";
//     } else {
//       oss << bin_edges[b];
//     }
//     oss << "; ";
//     if (bin_edges[b + 1] == std::numeric_limits<double>::infinity()) {
//       oss << "+Inf";
//     } else {
//       oss << bin_edges[b + 1];
//     }
//     oss << "]";
//     bin_labels.emplace_back(oss.str());
//   }
// }
// 
// // Transform Method Implementation
// Rcpp::List OptimalBinningNumericalLDB::transform() {
//   // Create vector of cutpoints (excluding -Inf and +Inf)
//   std::vector<double> cutpoints(bin_edges.begin() + 1, bin_edges.end() - 1);
//   
//   // Create result List
//   return Rcpp::List::create(
//     Rcpp::Named("bins") = bin_labels,
//     Rcpp::Named("woe") = woe_values,
//     Rcpp::Named("iv") = iv_values,
//     Rcpp::Named("count") = counts,
//     Rcpp::Named("count_pos") = count_pos,
//     Rcpp::Named("count_neg") = count_neg,
//     Rcpp::Named("cutpoints") = cutpoints,
//     Rcpp::Named("converged") = converged,
//     Rcpp::Named("iterations") = iterations_run
//   );
// }
// 
// //' @title Optimal Binning for Numerical Variables using Local Density Binning (LDB)
// //'
// //' @description This function implements the Local Density Binning (LDB) algorithm for optimal binning of numerical variables.
// //'
// //' @param target An integer vector of binary target values (0 or 1).
// //' @param feature A numeric vector of feature values to be binned.
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff Minimum frequency for a bin (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins (default: 20).
// //' @param convergence_threshold Threshold for convergence (default: 1e-6).
// //' @param max_iterations Maximum number of iterations (default: 1000).
// //'
// //' @return A list containing:
// //' \item{bins}{A vector of bin labels}
// //' \item{woe}{A numeric vector of Weight of Evidence (WoE) values for each bin}
// //' \item{iv}{A numeric vector of Information Value (IV) for each bin}
// //' \item{count}{Total count of observations in each bin}
// //' \item{count_pos}{Count of positive target observations in each bin}
// //' \item{count_neg}{Count of negative target observations in each bin}
// //' \item{cutpoints}{Numeric vector of cutpoints used to generate the bins}
// //' \item{converged}{Logical value indicating if the algorithm converged}
// //' \item{iterations}{Number of iterations run by the algorithm}
// //'
// //' @examples
// //' \dontrun{
// //' # Create sample data
// //' set.seed(123)
// //' target <- sample(0:1, 1000, replace = TRUE)
// //' feature <- rnorm(1000)
// //'
// //' # Run optimal binning
// //' result <- optimal_binning_numerical_ldb(target, feature)
// //'
// //' # View results
// //' print(result$bins)
// //' print(result$woe)
// //' print(result$iv)
// //' }
// //'
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_numerical_ldb(Rcpp::IntegerVector target,
//                                          Rcpp::NumericVector feature,
//                                          int min_bins = 3,
//                                          int max_bins = 5,
//                                          double bin_cutoff = 0.05,
//                                          int max_n_prebins = 20,
//                                          double convergence_threshold = 1e-6,
//                                          int max_iterations = 1000) {
//   // Convert R vectors to C++ vectors
//   std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
//   std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
//   
//   // Initialize binning class
//   OptimalBinningNumericalLDB ob(min_bins, max_bins, bin_cutoff, max_n_prebins,
//                                 convergence_threshold, max_iterations);
//   
//   // Perform binning
//   ob.fit(feature_vec, target_vec);
//   
//   // Retrieve results
//   return ob.transform();
// }
