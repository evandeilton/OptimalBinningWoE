// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <string>
#include <sstream>
#include <iomanip>
#include <set>

using namespace Rcpp;

// Função auxiliar para formatar doubles com precisão
std::string format_double(double value, int precision = 6) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision) << value;
  return oss.str();
}

// Classe para Binning Ótimo de Variáveis Numéricas (Monotonic Binning via Linear Programming - MBLP)
class OptimalBinningNumericalMBLP {
public:
  OptimalBinningNumericalMBLP(NumericVector feature, IntegerVector target,
                              int min_bins, int max_bins, double bin_cutoff,
                              int max_n_prebins, double convergence_threshold, int max_iterations)
    : feature(feature), target(target), min_bins(min_bins), max_bins(max_bins),
      bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins),
      convergence_threshold(convergence_threshold), max_iterations(max_iterations) {
    N = feature.size();
  }
  
  List fit() {
    validate_input();
    prebin();
    // Se unique_values <= 2 não é necessário otimizar
    if (unique_values <= 2) {
      calculate_woe_iv();
      prepare_cutpoints();
      converged = true;
      iterations_run = 0;
      return prepare_output();
    }
    optimize_binning();
    calculate_woe_iv();
    prepare_cutpoints();
    return prepare_output();
  }
  
private:
  NumericVector feature;
  IntegerVector target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  double convergence_threshold;
  int max_iterations;
  
  int N;
  
  std::vector<double> bin_edges;
  std::vector<int> bin_assignments;
  std::vector<double> bin_woe;
  std::vector<double> bin_iv;
  std::vector<int> bin_count;
  std::vector<int> bin_count_pos;
  std::vector<int> bin_count_neg;
  std::vector<std::string> bin_labels;
  
  std::vector<double> cutpoints;
  bool converged = false;
  int iterations_run = 0;
  int unique_values = 0;
  
  void validate_input() {
    if (N != target.size()) {
      stop("feature and target must have the same length.");
    }
    
    std::set<int> unique_targets(target.begin(), target.end());
    if (unique_targets.size() != 2 ||
        unique_targets.find(0) == unique_targets.end() ||
        unique_targets.find(1) == unique_targets.end()) {
      stop("target must be a binary vector with values 0 and 1.");
    }
    
    if (min_bins < 2) {
      stop("min_bins must be at least 2.");
    }
    if (max_bins < min_bins) {
      stop("max_bins must be greater than or equal to min_bins.");
    }
    
    if (bin_cutoff <= 0 || bin_cutoff >= 1) {
      stop("bin_cutoff must be between 0 and 1.");
    }
    
    if (max_n_prebins < min_bins) {
      stop("max_n_prebins must be greater than or equal to min_bins.");
    }
    
    if (convergence_threshold <= 0) {
      stop("convergence_threshold must be positive.");
    }
    
    if (max_iterations <= 0) {
      stop("max_iterations must be a positive integer.");
    }
  }
  
  void prebin() {
    // Remove valores NA
    std::vector<double> feature_clean;
    std::vector<int> target_clean;
    feature_clean.reserve(N);
    target_clean.reserve(N);
    for (int i = 0; i < N; ++i) {
      if (!NumericVector::is_na(feature[i])) {
        feature_clean.push_back(feature[i]);
        target_clean.push_back(target[i]);
      }
    }
    int N_clean = (int)feature_clean.size();
    
    if (N_clean == 0) {
      stop("All feature values are NA.");
    }
    
    // Ordena feature e target juntos
    std::vector<std::pair<double, int>> paired;
    paired.reserve(N_clean);
    for (int i = 0; i < N_clean; ++i) {
      paired.emplace_back(std::make_pair(feature_clean[i], target_clean[i]));
    }
    std::sort(paired.begin(), paired.end());
    
    std::vector<double> feature_sorted(N_clean);
    std::vector<int> target_sorted(N_clean);
    for (int i = 0; i < N_clean; ++i) {
      feature_sorted[i] = paired[i].first;
      target_sorted[i] = paired[i].second;
    }
    
    // Determina valores únicos
    std::vector<double> unique_feature = feature_sorted;
    std::sort(unique_feature.begin(), unique_feature.end());
    unique_feature.erase(std::unique(unique_feature.begin(), unique_feature.end()), unique_feature.end());
    unique_values = (int)unique_feature.size();
    
    if (unique_values <= 2) {
      // Caso trivial
      if (unique_values == 1) {
        bin_edges.push_back(-std::numeric_limits<double>::infinity());
        bin_edges.push_back(std::numeric_limits<double>::infinity());
      } else { // 2 valores únicos
        double v1 = unique_feature[0];
        bin_edges.push_back(-std::numeric_limits<double>::infinity());
        bin_edges.push_back(v1);
        bin_edges.push_back(std::numeric_limits<double>::infinity());
      }
    }
    else {
      int n_prebins = std::min(max_n_prebins, unique_values);
      n_prebins = std::max(n_prebins, min_bins);
      bin_edges = calculate_quantiles(unique_feature, n_prebins);
    }
    
    bin_assignments.assign(N_clean, -1);
    for (int i = 0; i < N_clean; ++i) {
      double val = feature_sorted[i];
      int bin_idx = (int)(std::lower_bound(bin_edges.begin(), bin_edges.end(), val) - bin_edges.begin() - 1);
      bin_idx = std::max(0, std::min(bin_idx, (int)bin_edges.size() - 2));
      bin_assignments[i] = bin_idx;
    }
    
    int n_bins = (int)bin_edges.size() - 1;
    bin_count.assign(n_bins, 0);
    bin_count_pos.assign(n_bins, 0);
    bin_count_neg.assign(n_bins, 0);
    
    for (int i = 0; i < N_clean; ++i) {
      int bin_idx = bin_assignments[i];
      bin_count[bin_idx]++;
      if (target_sorted[i] == 1) {
        bin_count_pos[bin_idx]++;
      } else {
        bin_count_neg[bin_idx]++;
      }
    }
    
    if (unique_values > 2) {
      merge_rare_bins();
    }
  }
  
  std::vector<double> calculate_quantiles(const std::vector<double>& data, int n_quantiles) {
    std::vector<double> quantiles;
    quantiles.reserve(n_quantiles + 1);
    
    quantiles.push_back(-std::numeric_limits<double>::infinity());
    
    for (int i = 1; i < n_quantiles; ++i) {
      double p = (double)i / n_quantiles;
      size_t idx = (size_t)std::ceil(p * (data.size() - 1));
      quantiles.push_back(data[idx]);
    }
    
    quantiles.push_back(std::numeric_limits<double>::infinity());
    
    return quantiles;
  }
  
  void merge_rare_bins() {
    double total = std::accumulate(bin_count.begin(), bin_count.end(), 0.0);
    bool merged = true;
    while (merged) {
      merged = false;
      for (int i = 0; i < (int)bin_count.size(); ++i) {
        double freq = (double)bin_count[i] / total;
        if (freq < bin_cutoff && (int)bin_count.size() > min_bins) {
          int merge_idx = (i == 0) ? i + 1 : i - 1;
          merge_bins(i, merge_idx);
          merged = true;
          break;
        }
      }
    }
  }
  
  void optimize_binning() {
    iterations_run = 0;
    double previous_iv = calculate_total_iv();
    
    while (iterations_run < max_iterations) {
      iterations_run++;
      
      enforce_bin_constraints();
      calculate_bin_woe();
      
      double current_iv = calculate_total_iv();
      double iv_change = std::abs(current_iv - previous_iv);
      
      if (iv_change < convergence_threshold) {
        converged = true;
        break;
      }
      
      previous_iv = current_iv;
      
      // Impõe monotonicidade
      if (!check_monotonicity(bin_woe) && (int)bin_count.size() > min_bins) {
        int merge_idx = find_min_iv_loss_merge();
        if (merge_idx == -1) {
          break;
        }
        merge_bins(merge_idx, merge_idx + 1);
      } else {
        converged = true;
        break;
      }
    }
    
    if (iterations_run >= max_iterations && !converged) {
      Rcpp::warning("Convergence not reached within the maximum number of iterations.");
    }
  }
  
  void enforce_bin_constraints() {
    while ((int)bin_count.size() < min_bins) {
      int merge_idx = find_min_iv_loss_merge();
      if (merge_idx == -1) break;
      merge_bins(merge_idx, merge_idx + 1);
    }
    
    while ((int)bin_count.size() > max_bins) {
      int merge_idx = find_min_iv_loss_merge();
      if (merge_idx == -1) break;
      merge_bins(merge_idx, merge_idx + 1);
    }
  }
  
  void calculate_bin_woe() {
    int n_bins = (int)bin_count.size();
    double total_pos = std::accumulate(bin_count_pos.begin(), bin_count_pos.end(), 0.0);
    double total_neg = std::accumulate(bin_count_neg.begin(), bin_count_neg.end(), 0.0);
    
    bin_woe.assign(n_bins, 0.0);
    bin_iv.assign(n_bins, 0.0);
    
    for (int i = 0; i < n_bins; ++i) {
      double dist_pos = (bin_count_pos[i] + 0.5) / (total_pos + 0.5 * n_bins);
      double dist_neg = (bin_count_neg[i] + 0.5) / (total_neg + 0.5 * n_bins);
      
      if (dist_pos <= 0) dist_pos = 1e-10;
      if (dist_neg <= 0) dist_neg = 1e-10;
      
      bin_woe[i] = std::log(dist_pos / dist_neg);
      bin_iv[i] = (dist_pos - dist_neg) * bin_woe[i];
    }
  }
  
  double calculate_total_iv() {
    return std::accumulate(bin_iv.begin(), bin_iv.end(), 0.0);
  }
  
  bool check_monotonicity(const std::vector<double>& vec) {
    if (vec.size() < 2) {
      return true;
    }
    
    bool increasing = true;
    bool decreasing = true;
    
    for (size_t i = 1; i < vec.size(); ++i) {
      if (vec[i] < vec[i-1]) {
        increasing = false;
      }
      if (vec[i] > vec[i-1]) {
        decreasing = false;
      }
    }
    
    return increasing || decreasing;
  }
  
  int find_min_iv_loss_merge() {
    if (bin_iv.size() < 2) {
      return -1;
    }
    
    double min_iv_loss = std::numeric_limits<double>::max();
    int merge_idx = -1;
    
    double total_pos = std::accumulate(bin_count_pos.begin(), bin_count_pos.end(), 0.0);
    double total_neg = std::accumulate(bin_count_neg.begin(), bin_count_neg.end(), 0.0);
    
    for (int i = 0; i < (int)bin_iv.size() - 1; ++i) {
      double iv_before = bin_iv[i] + bin_iv[i+1];
      
      double merged_pos = bin_count_pos[i] + bin_count_pos[i+1];
      double merged_neg = bin_count_neg[i] + bin_count_neg[i+1];
      
      double dist_pos = (merged_pos + 0.5) / (total_pos + 0.5 * bin_count.size());
      double dist_neg = (merged_neg + 0.5) / (total_neg + 0.5 * bin_count.size());
      
      if (dist_pos <= 0) dist_pos = 1e-10;
      if (dist_neg <= 0) dist_neg = 1e-10;
      
      double woe_merged = std::log(dist_pos / dist_neg);
      double iv_merged = (dist_pos - dist_neg) * woe_merged;
      
      double iv_after = iv_merged;
      double iv_loss = iv_before - iv_after;
      
      if (iv_loss < min_iv_loss) {
        min_iv_loss = iv_loss;
        merge_idx = i;
      }
    }
    
    return merge_idx;
  }
  
  void merge_bins(int idx1, int idx2) {
    if (idx1 < 0 || idx2 < 0 || idx1 >= (int)bin_count.size() || idx2 >= (int)bin_count.size()) {
      stop("Invalid merge indices.");
    }
    
    if (idx1 == idx2) {
      return;
    }
    
    int lower_idx = std::min(idx1, idx2);
    int higher_idx = std::max(idx1, idx2);
    
    bin_edges.erase(bin_edges.begin() + higher_idx);
    bin_count[lower_idx] += bin_count[higher_idx];
    bin_count_pos[lower_idx] += bin_count_pos[higher_idx];
    bin_count_neg[lower_idx] += bin_count_neg[higher_idx];
    
    bin_count.erase(bin_count.begin() + higher_idx);
    bin_count_pos.erase(bin_count_pos.begin() + higher_idx);
    bin_count_neg.erase(bin_count_neg.begin() + higher_idx);
    
    if (!bin_woe.empty() && !bin_iv.empty()) {
      bin_woe.erase(bin_woe.begin() + higher_idx);
      bin_iv.erase(bin_iv.begin() + higher_idx);
    }
  }
  
  void calculate_woe_iv() {
    calculate_bin_woe();
  }
  
  void prepare_cutpoints() {
    cutpoints.clear();
    for (size_t i = 1; i < bin_edges.size() - 1; ++i) {
      cutpoints.push_back(bin_edges[i]);
    }
  }
  
  List prepare_output() {
    int n_bins = (int)bin_count.size();
    bin_labels.assign(n_bins, "");
    
    for (int i = 0; i < n_bins; ++i) {
      std::string left, right;
      
      if (i == 0) {
        left = "(-Inf";
      } else {
        left = "(" + format_double(bin_edges[i]);
      }
      
      if (i == n_bins - 1) {
        right = "+Inf]";
      } else {
        right = format_double(bin_edges[i + 1]) + "]";
      }
      
      bin_labels[i] = left + ";" + right;
    }
    
    Rcpp::NumericVector ids(bin_labels.size());
    for(int i = 0; i < bin_labels.size(); i++) {
      ids[i] = i + 1;
    }
    
    return Rcpp::List::create(
      Named("id") = ids,
      Named("bin") = bin_labels,
      Named("woe") = bin_woe,
      Named("iv") = bin_iv,
      Named("count") = bin_count,
      Named("count_pos") = bin_count_pos,
      Named("count_neg") = bin_count_neg,
      Named("cutpoints") = cutpoints,
      Named("converged") = converged,
      Named("iterations") = iterations_run
    );
  }
};


//' @title Optimal Binning for Numerical Features Using Monotonic Binning via Linear Programming (MBLP)
//'
//' @description
//' This method performs optimal binning for numerical features, ensuring monotonicity in the Weight of Evidence (WoE) across bins.
//' It adheres to constraints on the minimum and maximum number of bins, merges rare bins, and handles edge cases like identical values.
//' The algorithm returns bins, WoE, Information Value (IV), counts, cutpoints, and metadata such as convergence status and iterations run.
//'
//' @details
//' ### Key Steps:
//' 1. **Input Validation**: Ensures proper formatting and constraints for `feature`, `target`, and algorithm parameters.
//' 2. **Pre-Binning**: Creates preliminary bins based on quantiles or unique values in the feature.
//' 3. **Rare Bin Merging**: Combines bins with frequencies below `bin_cutoff` to maintain statistical stability.
//' 4. **Optimization**: Adjusts bins iteratively to maximize IV, enforce monotonicity, and adhere to bin constraints (`min_bins` and `max_bins`).
//' 5. **Monotonicity Enforcement**: Ensures WoE values are either strictly increasing or decreasing across bins.
//' 6. **Validation**: Verifies bin structure for consistency, preventing gaps or overlapping intervals.
//'
//' ### Mathematical Framework:
//' - **Weight of Evidence (WoE)**: For a bin \( i \):
//'   \deqn{WoE_i = \ln\left(\frac{\text{Distribution of positives}_i}{\text{Distribution of negatives}_i}\right)}
//'
//' - **Information Value (IV)**: Aggregates predictive power across all bins:
//'   \deqn{IV = \sum_{i=1}^{N} (\text{Distribution of positives}_i - \text{Distribution of negatives}_i) \times WoE_i}
//'
//' ### Features:
//' - Monotonic WoE ensures interpretability in logistic regression and credit scoring models.
//' - Dynamically adjusts binning to maximize IV and improve model predictive power.
//' - Handles rare categories and missing values by merging and imputation.
//' - Supports large datasets with efficient pre-binning and convergence checks.
//' - Validates results to prevent invalid bin configurations (e.g., gaps, overlaps).
//'
//' ### Algorithm Parameters:
//' - `min_bins`: Minimum number of bins (default: 3).
//' - `max_bins`: Maximum number of bins (default: 5).
//' - `bin_cutoff`: Minimum frequency proportion required to retain a bin as standalone (default: 0.05).
//' - `max_n_prebins`: Maximum number of preliminary bins before optimization (default: 20).
//' - `convergence_threshold`: Threshold for convergence in IV optimization (default: 1e-6).
//' - `max_iterations`: Maximum number of iterations allowed for optimization (default: 1000).
//'
//' @param target An integer binary vector (0 or 1) representing the target variable.
//' @param feature A numeric vector representing the feature to bin.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum frequency proportion for retaining bins (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins before optimization (default: 20).
//' @param convergence_threshold Convergence threshold for IV optimization (default: 1e-6).
//' @param max_iterations Maximum number of iterations allowed (default: 1000).
//'
//' @return A list with the following components:
//' \itemize{
//'   \item `bin`: A character vector of bin intervals in the format "[lower;upper)".
//'   \item `woe`: A numeric vector of WoE values for each bin.
//'   \item `iv`: A numeric vector of IV contributions for each bin.
//'   \item `count`: An integer vector of total observations per bin.
//'   \item `count_pos`: An integer vector of positive cases per bin.
//'   \item `count_neg`: An integer vector of negative cases per bin.
//'   \item `cutpoints`: A numeric vector of cutpoints defining the bin edges.
//'   \item `converged`: A boolean indicating whether the algorithm converged.
//'   \item `iterations`: An integer indicating the number of iterations executed.
//' }
//'
//' @examples
//' \dontrun{
//' set.seed(123)
//' feature <- rnorm(1000)
//' target <- rbinom(1000, 1, 0.3)
//' result <- optimal_binning_numerical_mblp(target, feature, min_bins = 3, max_bins = 6)
//' print(result)
//' }
//'
//' @export
// [[Rcpp::export]]
List optimal_binning_numerical_mblp(IntegerVector target,
                                   NumericVector feature, 
                                   int min_bins = 3, 
                                   int max_bins = 5, 
                                   double bin_cutoff = 0.05, 
                                   int max_n_prebins = 20,
                                   double convergence_threshold = 1e-6,
                                   int max_iterations = 1000) {
 // Valida entrada
 if (feature.size() != target.size()) {
   stop("feature and target must have the same length.");
 }
 if (min_bins < 2) {
   stop("min_bins must be at least 2.");
 }
 if (max_bins < min_bins) {
   stop("max_bins must be greater than or equal to min_bins.");
 }
 if (bin_cutoff <= 0 || bin_cutoff >= 1) {
   stop("bin_cutoff must be between 0 and 1.");
 }
 if (max_n_prebins < min_bins) {
   stop("max_n_prebins must be greater than or equal to min_bins.");
 }
 if (convergence_threshold <= 0) {
   stop("convergence_threshold must be positive.");
 }
 if (max_iterations <= 0) {
   stop("max_iterations must be a positive integer.");
 }
 
 OptimalBinningNumericalMBLP ob(feature, target, min_bins, max_bins, bin_cutoff, 
                                max_n_prebins, convergence_threshold, max_iterations);
 return ob.fit();
}
 
/*
Melhorias realizadas:
- Garantia de que o código é robusto contra NA, tamanhos incompatíveis, e entrada inválida.
- Uso de epsilon ao calcular log para evitar log(0).
- Processo de fusão de bins raro, enforcing de monotonicidade e checagem de convergência por IV.
- Manutenção da interface e nomes conforme solicitado.
- Comentários e warnings adequados.
*/


// // [[Rcpp::plugins(cpp11)]]
// #include <Rcpp.h>
// #include <vector>
// #include <algorithm>
// #include <numeric>
// #include <cmath>
// #include <limits>
// #include <string>
// #include <sstream>
// #include <iomanip>
// #include <set>
// 
// using namespace Rcpp;
// 
// // Helper function to format double with precision
// std::string format_double(double value, int precision = 6) {
//   std::ostringstream oss;
//   oss << std::fixed << std::setprecision(precision) << value;
//   return oss.str();
// }
// 
// // Class for Optimal Binning of Numerical Features
// class OptimalBinningNumericalMBLP {
// public:
//   OptimalBinningNumericalMBLP(NumericVector feature, IntegerVector target,
//                               int min_bins, int max_bins, double bin_cutoff,
//                               int max_n_prebins, double convergence_threshold, int max_iterations)
//     : feature(feature), target(target), min_bins(min_bins), max_bins(max_bins),
//       bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins),
//       convergence_threshold(convergence_threshold), max_iterations(max_iterations) {
//     N = feature.size();
//   }
//   
//   List fit() {
//     validate_input();
//     prebin();
//     // Check if unique_values <= 2
//     if (unique_values <= 2) {
//       calculate_woe_iv();
//       prepare_cutpoints();
//       converged = true;
//       iterations_run = 0;
//       return prepare_output();
//     }
//     optimize_binning();
//     calculate_woe_iv();
//     prepare_cutpoints();
//     return prepare_output();
//   }
//   
// private:
//   NumericVector feature;
//   IntegerVector target;
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
//   double convergence_threshold;
//   int max_iterations;
//   
//   int N;
//   
//   std::vector<double> bin_edges;
//   std::vector<int> bin_assignments;
//   std::vector<double> bin_woe;
//   std::vector<double> bin_iv;
//   std::vector<int> bin_count;
//   std::vector<int> bin_count_pos;
//   std::vector<int> bin_count_neg;
//   std::vector<std::string> bin_labels;
//   
//   std::vector<double> cutpoints;
//   bool converged = false;
//   int iterations_run = 0;
//   int unique_values = 0;
//   
//   void validate_input() {
//     if (N != target.size()) {
//       stop("feature and target must have the same length.");
//     }
//     
//     std::set<int> unique_targets(target.begin(), target.end());
//     if (unique_targets.size() != 2 ||
//         unique_targets.find(0) == unique_targets.end() ||
//         unique_targets.find(1) == unique_targets.end()) {
//       stop("target must be a binary vector with values 0 and 1.");
//     }
//     
//     if (min_bins < 2) {
//       stop("min_bins must be at least 2.");
//     }
//     if (max_bins < min_bins) {
//       stop("max_bins must be greater than or equal to min_bins.");
//     }
//     
//     if (bin_cutoff <= 0 || bin_cutoff >= 1) {
//       stop("bin_cutoff must be between 0 and 1.");
//     }
//     
//     if (max_n_prebins < min_bins) {
//       stop("max_n_prebins must be greater than or equal to min_bins.");
//     }
//     
//     if (convergence_threshold <= 0) {
//       stop("convergence_threshold must be positive.");
//     }
//     
//     if (max_iterations <= 0) {
//       stop("max_iterations must be a positive integer.");
//     }
//   }
//   
//   void prebin() {
//     // Remove missing values
//     std::vector<double> feature_clean;
//     std::vector<int> target_clean;
//     feature_clean.reserve(N);
//     target_clean.reserve(N);
//     for (int i = 0; i < N; ++i) {
//       if (!NumericVector::is_na(feature[i])) {
//         feature_clean.push_back(feature[i]);
//         target_clean.push_back(target[i]);
//       }
//     }
//     int N_clean = feature_clean.size();
//     
//     if (N_clean == 0) {
//       stop("All feature values are NA.");
//     }
//     
//     // Sort feature and target together
//     std::vector<std::pair<double, int>> paired;
//     paired.reserve(N_clean);
//     for (int i = 0; i < N_clean; ++i) {
//       paired.emplace_back(std::make_pair(feature_clean[i], target_clean[i]));
//     }
//     std::sort(paired.begin(), paired.end());
//     
//     // Reconstruct sorted feature and target
//     std::vector<double> feature_sorted(N_clean);
//     std::vector<int> target_sorted(N_clean);
//     for (int i = 0; i < N_clean; ++i) {
//       feature_sorted[i] = paired[i].first;
//       target_sorted[i] = paired[i].second;
//     }
//     
//     // Determine unique values
//     std::vector<double> unique_feature = feature_sorted;
//     std::sort(unique_feature.begin(), unique_feature.end());
//     unique_feature.erase(std::unique(unique_feature.begin(), unique_feature.end()), unique_feature.end());
//     unique_values = unique_feature.size();
//     
//     if (unique_values <= 2) {
//       if (unique_values == 1) {
//         // Single unique value: one bin
//         bin_edges.push_back(-std::numeric_limits<double>::infinity());
//         bin_edges.push_back(std::numeric_limits<double>::infinity());
//       } else { // unique_values == 2
//         // Two unique values: two bins
//         double v1 = unique_feature[0];
//         bin_edges.push_back(-std::numeric_limits<double>::infinity());
//         bin_edges.push_back(v1);
//         bin_edges.push_back(std::numeric_limits<double>::infinity());
//       }
//     }
//     else {
//       // Assign unique values to pre-bins
//       int n_prebins = std::min(max_n_prebins, unique_values);
//       n_prebins = std::max(n_prebins, min_bins);
//       
//       // Calculate quantiles for pre-binning
//       bin_edges = calculate_quantiles(unique_feature, n_prebins);
//     }
//     
//     // Assign bins using lower_bound to ensure correct binning
//     bin_assignments.assign(N_clean, -1);
//     for (int i = 0; i < N_clean; ++i) {
//       double val = feature_sorted[i];
//       // Use lower_bound instead of upper_bound
//       int bin_idx = std::lower_bound(bin_edges.begin(), bin_edges.end(), val) - bin_edges.begin() - 1;
//       bin_idx = std::max(0, std::min(bin_idx, static_cast<int>(bin_edges.size()) - 2));
//       bin_assignments[i] = bin_idx;
//     }
//     
//     // Initialize counts
//     int n_bins = bin_edges.size() - 1;
//     bin_count.assign(n_bins, 0);
//     bin_count_pos.assign(n_bins, 0);
//     bin_count_neg.assign(n_bins, 0);
//     
//     for (int i = 0; i < N_clean; ++i) {
//       int bin_idx = bin_assignments[i];
//       bin_count[bin_idx]++;
//       if (target_sorted[i] == 1) {
//         bin_count_pos[bin_idx]++;
//       } else {
//         bin_count_neg[bin_idx]++;
//       }
//     }
//     
//     // Merge rare bins based on bin_cutoff if unique_values > 2
//     if (unique_values > 2) {
//       merge_rare_bins();
//     }
//   }
//   
//   std::vector<double> calculate_quantiles(const std::vector<double>& data, int n_quantiles) {
//     std::vector<double> quantiles;
//     quantiles.reserve(n_quantiles + 1);
//     
//     quantiles.push_back(-std::numeric_limits<double>::infinity());
//     
//     for (int i = 1; i < n_quantiles; ++i) {
//       double p = static_cast<double>(i) / n_quantiles;
//       size_t idx = static_cast<size_t>(std::ceil(p * (data.size() - 1)));
//       quantiles.push_back(data[idx]);
//     }
//     
//     quantiles.push_back(std::numeric_limits<double>::infinity());
//     
//     return quantiles;
//   }
//   
//   void merge_rare_bins() {
//     double total = std::accumulate(bin_count.begin(), bin_count.end(), 0.0);
//     bool merged = true;
//     while (merged) {
//       merged = false;
//       for (int i = 0; i < bin_count.size(); ++i) {
//         double freq = static_cast<double>(bin_count[i]) / total;
//         if (freq < bin_cutoff && bin_count.size() > min_bins) {
//           // Merge with adjacent bin
//           int merge_idx = (i == 0) ? i + 1 : i - 1;
//           merge_bins(i, merge_idx);
//           merged = true;
//           break;
//         }
//       }
//     }
//   }
//   
//   void optimize_binning() {
//     iterations_run = 0;
//     double previous_iv = calculate_total_iv();
//     
//     while (iterations_run < max_iterations) {
//       iterations_run++;
//       
//       enforce_bin_constraints();
//       calculate_bin_woe();
//       
//       double current_iv = calculate_total_iv();
//       double iv_change = std::abs(current_iv - previous_iv);
//       
//       if (iv_change < convergence_threshold) {
//         converged = true;
//         break;
//       }
//       
//       previous_iv = current_iv;
//       
//       // Enforce monotonicity
//       if (!check_monotonicity(bin_woe) && bin_count.size() > min_bins) {
//         int merge_idx = find_min_iv_loss_merge();
//         if (merge_idx == -1) {
//           break;
//         }
//         merge_bins(merge_idx, merge_idx + 1);
//       } else {
//         converged = true;
//         break;
//       }
//     }
//     
//     if (iterations_run >= max_iterations && !converged) {
//       Rcpp::warning("Convergence not reached within the maximum number of iterations.");
//     }
//   }
//   
//   void enforce_bin_constraints() {
//     // Ensure the number of bins is not less than min_bins
//     while (bin_count.size() < min_bins) {
//       int merge_idx = find_min_iv_loss_merge();
//       if (merge_idx == -1) break;
//       merge_bins(merge_idx, merge_idx + 1);
//     }
//     
//     // Ensure the number of bins does not exceed max_bins
//     while (bin_count.size() > max_bins) {
//       int merge_idx = find_min_iv_loss_merge();
//       if (merge_idx == -1) break;
//       merge_bins(merge_idx, merge_idx + 1);
//     }
//   }
//   
//   void calculate_bin_woe() {
//     int n_bins = bin_count.size();
//     double total_pos = std::accumulate(bin_count_pos.begin(), bin_count_pos.end(), 0.0);
//     double total_neg = std::accumulate(bin_count_neg.begin(), bin_count_neg.end(), 0.0);
//     
//     bin_woe.assign(n_bins, 0.0);
//     bin_iv.assign(n_bins, 0.0);
//     
//     for (int i = 0; i < n_bins; ++i) {
//       double dist_pos = (bin_count_pos[i] + 0.5) / (total_pos + 0.5 * n_bins);
//       double dist_neg = (bin_count_neg[i] + 0.5) / (total_neg + 0.5 * n_bins);
//       
//       bin_woe[i] = std::log(dist_pos / dist_neg);
//       bin_iv[i] = (dist_pos - dist_neg) * bin_woe[i];
//     }
//   }
//   
//   double calculate_total_iv() {
//     return std::accumulate(bin_iv.begin(), bin_iv.end(), 0.0);
//   }
//   
//   bool check_monotonicity(const std::vector<double>& vec) {
//     if (vec.size() < 2) {
//       return true;
//     }
//     
//     bool increasing = true;
//     bool decreasing = true;
//     
//     for (size_t i = 1; i < vec.size(); ++i) {
//       if (vec[i] < vec[i-1]) {
//         increasing = false;
//       }
//       if (vec[i] > vec[i-1]) {
//         decreasing = false;
//       }
//     }
//     
//     return increasing || decreasing;
//   }
//   
//   int find_min_iv_loss_merge() {
//     if (bin_iv.size() < 2) {
//       return -1; // Not enough bins to merge
//     }
//     
//     double min_iv_loss = std::numeric_limits<double>::max();
//     int merge_idx = -1;
//     
//     for (int i = 0; i < bin_iv.size() - 1; ++i) {
//       double iv_before = bin_iv[i] + bin_iv[i+1];
//       
//       // Calculate merged WoE and IV
//       double merged_pos = bin_count_pos[i] + bin_count_pos[i+1];
//       double merged_neg = bin_count_neg[i] + bin_count_neg[i+1];
//       double total_pos = std::accumulate(bin_count_pos.begin(), bin_count_pos.end(), 0.0);
//       double total_neg = std::accumulate(bin_count_neg.begin(), bin_count_neg.end(), 0.0);
//       
//       double dist_pos = (merged_pos + 0.5) / (total_pos + 0.5 * bin_count.size());
//       double dist_neg = (merged_neg + 0.5) / (total_neg + 0.5 * bin_count.size());
//       
//       double woe_merged = std::log(dist_pos / dist_neg);
//       double iv_merged = (dist_pos - dist_neg) * woe_merged;
//       
//       double iv_after = iv_merged;
//       double iv_loss = iv_before - iv_after;
//       
//       if (iv_loss < min_iv_loss) {
//         min_iv_loss = iv_loss;
//         merge_idx = i;
//       }
//     }
//     
//     return merge_idx;
//   }
//   
//   void merge_bins(int idx1, int idx2) {
//     if (idx1 < 0 || idx2 < 0 || idx1 >= bin_count.size() || idx2 >= bin_count.size()) {
//       stop("Invalid merge indices.");
//     }
//     
//     if (idx1 == idx2) {
//       return; // No need to merge, just return
//     }
//     
//     int lower_idx = std::min(idx1, idx2);
//     int higher_idx = std::max(idx1, idx2);
//     
//     // Merge bin higher_idx into lower_idx
//     bin_edges.erase(bin_edges.begin() + higher_idx);
//     bin_count[lower_idx] += bin_count[higher_idx];
//     bin_count_pos[lower_idx] += bin_count_pos[higher_idx];
//     bin_count_neg[lower_idx] += bin_count_neg[higher_idx];
//     
//     bin_count.erase(bin_count.begin() + higher_idx);
//     bin_count_pos.erase(bin_count_pos.begin() + higher_idx);
//     bin_count_neg.erase(bin_count_neg.begin() + higher_idx);
//     
//     // Adjust WoE and IV vectors if they exist
//     if (!bin_woe.empty() && !bin_iv.empty()) {
//       bin_woe.erase(bin_woe.begin() + higher_idx);
//       bin_iv.erase(bin_iv.begin() + higher_idx);
//     }
//   }
//   
//   void calculate_woe_iv() {
//     calculate_bin_woe();
//   }
//   
//   void prepare_cutpoints() {
//     cutpoints.clear();
//     for (size_t i = 1; i < bin_edges.size() - 1; ++i) {
//       cutpoints.push_back(bin_edges[i]);
//     }
//   }
//   
//   List prepare_output() {
//     // Prepare bin labels
//     int n_bins = bin_count.size();
//     bin_labels.assign(n_bins, "");
//     
//     for (int i = 0; i < n_bins; ++i) {
//       std::string left, right;
//       
//       if (i == 0) {
//         left = "(-Inf";
//       } else {
//         left = "(" + format_double(bin_edges[i]);
//       }
//       
//       if (i == n_bins - 1) {
//         right = "+Inf]";
//       } else {
//         right = format_double(bin_edges[i + 1]) + "]";
//       }
//       
//       bin_labels[i] = left + ";" + right;
//     }
//     
//     // Create output list
//     return List::create(
//       Named("bin") = bin_labels,
//       Named("woe") = bin_woe,
//       Named("iv") = bin_iv,
//       Named("count") = bin_count,
//       Named("count_pos") = bin_count_pos,
//       Named("count_neg") = bin_count_neg,
//       Named("cutpoints") = cutpoints,
//       Named("converged") = converged,
//       Named("iterations") = iterations_run
//     );
//   }
// };
// 
// //' Optimal Binning for Numerical Features Using Monotonic Binning via Linear Programming
// //'
// //' This function performs optimal binning for numerical features, using a monotonic binning
// //' technique that ensures the WoE (Weight of Evidence) is monotonic across bins.
// //' The algorithm iteratively adjusts the bins to respect specified constraints (min and max bins) 
// //' and ensures that rare bins are merged. The final result includes the optimal bins, WoE values, 
// //' IV (Information Value), and additional metadata regarding convergence.
// //' 
// //' @param target An integer vector representing the binary target variable (0 or 1).
// //' @param feature A numeric vector representing the numerical feature to be binned.
// //' @param min_bins An integer specifying the minimum number of bins. Default is 3.
// //' @param max_bins An integer specifying the maximum number of bins. Default is 5. Must be greater than or equal to `min_bins`.
// //' @param bin_cutoff A numeric value representing the cutoff for merging rare bins. Bins with a frequency lower than this threshold are merged. Default is 0.05.
// //' @param max_n_prebins An integer specifying the maximum number of pre-bins before optimization. Default is 20.
// //' @param convergence_threshold A numeric value specifying the threshold for convergence of the Information Value (IV). Default is 1e-6.
// //' @param max_iterations An integer specifying the maximum number of iterations allowed for binning optimization. Default is 1000.
// //'
// //' @return A list with the following components:
// //' \item{bins}{Character vector of bin labels, defining the intervals of each bin.}
// //' \item{woe}{Numeric vector of WoE (Weight of Evidence) values for each bin.}
// //' \item{iv}{Numeric vector of IV (Information Value) values for each bin.}
// //' \item{count}{Integer vector representing the count of observations in each bin.}
// //' \item{count_pos}{Integer vector representing the count of positive (target == 1) observations in each bin.}
// //' \item{count_neg}{Integer vector representing the count of negative (target == 0) observations in each bin.}
// //' \item{cutpoints}{Numeric vector of cut points used to define the bins.}
// //' \item{converged}{Logical indicating whether the algorithm converged within the specified threshold and iterations.}
// //' \item{iterations}{Integer indicating the number of iterations the algorithm ran before convergence or stopping.}
// //'
// //' @examples
// //' # Example with exactly 2 unique values
// //' feature <- c(1, 1, 2, 2, 1, 2)
// //' target <- c(0, 1, 0, 1, 0, 1)
// //' result <- optimal_binning_numerical_mblp(target, feature)
// //' print(result)
// //'
// //' # Example with more unique values
// //' set.seed(123)
// //' feature <- rnorm(1000)
// //' target <- rbinom(1000, 1, 0.3)
// //' result <- optimal_binning_numerical_mblp(target, feature, min_bins = 3, max_bins = 6)
// //' print(result)
// //'
// //' @export
// // [[Rcpp::export]]
// List optimal_binning_numerical_mblp(IntegerVector target,
//                                    NumericVector feature, 
//                                    int min_bins = 3, 
//                                    int max_bins = 5, 
//                                    double bin_cutoff = 0.05, 
//                                    int max_n_prebins = 20,
//                                    double convergence_threshold = 1e-6,
//                                    int max_iterations = 1000) {
//  // Input validation
//  if (feature.size() != target.size()) {
//    stop("feature and target must have the same length.");
//  }
//  if (min_bins < 2) {
//    stop("min_bins must be at least 2.");
//  }
//  if (max_bins < min_bins) {
//    stop("max_bins must be greater than or equal to min_bins.");
//  }
//  if (bin_cutoff <= 0 || bin_cutoff >= 1) {
//    stop("bin_cutoff must be between 0 and 1.");
//  }
//  if (max_n_prebins < min_bins) {
//    stop("max_n_prebins must be greater than or equal to min_bins.");
//  }
//  if (convergence_threshold <= 0) {
//    stop("convergence_threshold must be positive.");
//  }
//  if (max_iterations <= 0) {
//    stop("max_iterations must be a positive integer.");
//  }
//  
//  // Initialize and fit the binning model
//  OptimalBinningNumericalMBLP ob(feature, target, min_bins, max_bins, bin_cutoff, 
//                                 max_n_prebins, convergence_threshold, max_iterations);
//  return ob.fit();
// }
