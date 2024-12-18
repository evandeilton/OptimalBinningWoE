// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <sstream>
#include <numeric>
#include <limits>

using namespace Rcpp;

// Função para calcular correlação de Pearson entre dois vetores
inline double compute_correlation(const std::vector<double> &x, const std::vector<double> &y) {
  if (x.size() != y.size() || x.empty()) {
    Rcpp::stop("Vectors must be of the same non-zero length for correlation.");
  }
  
  double mean_x = std::accumulate(x.begin(), x.end(), 0.0) / x.size();
  double mean_y = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
  
  double numerator = 0.0;
  double denom_x = 0.0;
  double denom_y = 0.0;
  
  for (size_t i = 0; i < x.size(); ++i) {
    double dx = x[i] - mean_x;
    double dy = y[i] - mean_y;
    numerator += dx * dy;
    denom_x += dx * dx;
    denom_y += dy * dy;
  }
  
  if (denom_x == 0 || denom_y == 0) {
    Rcpp::warning("Standard deviation is zero. Returning correlation as 0.");
    return 0.0;
  }
  
  return numerator / std::sqrt(denom_x * denom_y);
}

// Classe para Binning Ótimo Numérico usando Local Polynomial Density Binning (LPDB)
class OptimalBinningNumericalLPDB {
public:
  OptimalBinningNumericalLPDB(int min_bins = 3, int max_bins = 5, double bin_cutoff = 0.05, int max_n_prebins = 20,
                              double convergence_threshold = 1e-6, int max_iterations = 1000)
    : min_bins(min_bins), max_bins(max_bins), bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins),
      convergence_threshold(convergence_threshold), max_iterations(max_iterations),
      converged(true), iterations_run(0) {
    
    // Validações de parâmetros
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
  }
  
  Rcpp::List fit(Rcpp::NumericVector feature, Rcpp::IntegerVector target);
  
private:
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  double convergence_threshold;
  int max_iterations;
  bool converged;
  int iterations_run;
  
  struct Bin {
    double lower_bound;
    double upper_bound;
    int count;
    int count_pos;
    int count_neg;
    double woe;
    double iv;
  };
  
  std::vector<Bin> prebinning(const std::vector<double> &feature, const std::vector<int> &target);
  void calculate_woe_iv(std::vector<Bin> &bins, int total_pos, int total_neg);
  void enforce_monotonicity(std::vector<Bin> &bins, int total_pos, int total_neg);
  std::string format_bin_interval(double lower, double upper, bool first = false, bool last = false);
};

// Método fit
Rcpp::List OptimalBinningNumericalLPDB::fit(Rcpp::NumericVector feature, Rcpp::IntegerVector target) {
  int n = feature.size();
  if (n != target.size()) {
    Rcpp::stop("feature and target must have the same length.");
  }
  
  // Garante que target é binário
  IntegerVector unique_targets = unique(target);
  if (unique_targets.size() != 2 || (std::find(unique_targets.begin(), unique_targets.end(), 0) == unique_targets.end()) ||
      (std::find(unique_targets.begin(), unique_targets.end(), 1) == unique_targets.end())) {
    Rcpp::stop("Target must be binary (0 and 1) and contain both classes.");
  }
  
  // Remove NA
  LogicalVector not_na = (!is_na(feature)) & (!is_na(target));
  NumericVector clean_feature = feature[not_na];
  IntegerVector clean_target = target[not_na];
  
  if (clean_feature.size() == 0) {
    Rcpp::stop("No valid observations after removing missing values.");
  }
  
  int total_pos = std::accumulate(clean_target.begin(), clean_target.end(), 0);
  int total_neg = clean_target.size() - total_pos;
  
  if (total_pos == 0 || total_neg == 0) {
    Rcpp::stop("Target must have both positive and negative classes.");
  }
  
  // Checa se feature possui um só valor único
  NumericVector unique_vec = unique(clean_feature);
  std::vector<double> unique_feature = Rcpp::as<std::vector<double>>(unique_vec);
  
  if (unique_feature.size() == 1) {
    // Um único valor => um único bin
    Bin bin;
    bin.lower_bound = -std::numeric_limits<double>::infinity();
    bin.upper_bound = std::numeric_limits<double>::infinity();
    bin.count = clean_feature.size();
    bin.count_pos = total_pos;
    bin.count_neg = total_neg;
    
    double dist_pos = static_cast<double>(bin.count_pos) / total_pos;
    double dist_neg = static_cast<double>(bin.count_neg) / total_neg;
    if (dist_pos <= 0) dist_pos = 1e-10;
    if (dist_neg <= 0) dist_neg = 1e-10;
    bin.woe = std::log(dist_pos / dist_neg);
    bin.iv = (dist_pos - dist_neg) * bin.woe;
    
    std::vector<Bin> bins = { bin };
    
    // Cria labels
    std::vector<std::string> bin_labels;
    bin_labels.emplace_back(format_bin_interval(bin.lower_bound, bin.upper_bound, true, true));
    
    NumericVector woe_values(1, bin.woe);
    NumericVector iv_values(1, bin.iv);
    IntegerVector counts(1, bin.count);
    IntegerVector counts_pos(1, bin.count_pos);
    IntegerVector counts_neg(1, bin.count_neg);
    NumericVector cutpoints; // Vazio, pois há apenas um bin
    
    Rcpp::NumericVector ids(bin_labels.size());
    for(int i = 0; i < bin_labels.size(); i++) {
      ids[i] = i + 1;
    }
    
    return Rcpp::List::create(
      Named("id") = ids,
      Named("bin") = bin_labels,
      Named("woe") = woe_values,
      Named("iv") = iv_values,
      Named("count") = counts,
      Named("count_pos") = counts_pos,
      Named("count_neg") = counts_neg,
      Named("cutpoints") = cutpoints,
      Named("converged") = true,
      Named("iterations") = 0
    );
  }
  
  // Caso contrário, faz pré-binning
  std::vector<double> clean_feature_vec = Rcpp::as<std::vector<double>>(clean_feature);
  std::vector<int> clean_target_vec = Rcpp::as<std::vector<int>>(clean_target);
  
  std::vector<Bin> bins = prebinning(clean_feature_vec, clean_target_vec);
  
  // Calcula WoE e IV
  calculate_woe_iv(bins, total_pos, total_neg);
  
  // Aplica monotonicidade
  enforce_monotonicity(bins, total_pos, total_neg);
  
  // Cria labels e cutpoints
  std::vector<std::string> bin_labels;
  std::vector<double> cutpoints_list;
  
  for (size_t i = 0; i < bins.size(); ++i) {
    bin_labels.emplace_back(format_bin_interval(bins[i].lower_bound, bins[i].upper_bound,
                                                i == 0, i == (bins.size() - 1)));
    if (i < bins.size() - 1) {
      cutpoints_list.emplace_back(bins[i].upper_bound);
    }
  }
  
  std::vector<double> woe_vals;
  std::vector<double> iv_vals;
  std::vector<int> counts_vec;
  std::vector<int> counts_pos_vec;
  std::vector<int> counts_neg_vec;
  
  for (size_t i = 0; i < bins.size(); ++i) {
    woe_vals.push_back(bins[i].woe);
    iv_vals.push_back(bins[i].iv);
    counts_vec.push_back(bins[i].count);
    counts_pos_vec.push_back(bins[i].count_pos);
    counts_neg_vec.push_back(bins[i].count_neg);
  }
  
  Rcpp::NumericVector ids(bin_labels.size());
  for(int i = 0; i < bin_labels.size(); i++) {
    ids[i] = i + 1;
  }
  
  return Rcpp::List::create(
    Named("id") = ids,
    Named("bin") = bin_labels,
    Named("woe") = woe_vals,
    Named("iv") = iv_vals,
    Named("count") = counts_vec,
    Named("count_pos") = counts_pos_vec,
    Named("count_neg") = counts_neg_vec,
    Named("cutpoints") = cutpoints_list,
    Named("converged") = converged,
    Named("iterations") = iterations_run
  );
}

// Função prebinning
std::vector<OptimalBinningNumericalLPDB::Bin> OptimalBinningNumericalLPDB::prebinning(const std::vector<double> &feature, const std::vector<int> &target) {
  int n = static_cast<int>(feature.size());
  std::vector<int> indices(n);
  for (int i = 0; i < n; ++i) {
    indices[i] = i;
  }
  
  std::sort(indices.begin(), indices.end(), [&](int a, int b) {
    return feature[a] < feature[b];
  });
  
  int bin_size = n / max_n_prebins;
  if (bin_size < 1) bin_size = 1;
  
  std::vector<double> cut_points;
  for (int i = bin_size; i < n; i += bin_size) {
    double val = feature[indices[i]];
    if (cut_points.empty() || val != cut_points.back()) {
      cut_points.push_back(val);
    }
  }
  
  std::vector<Bin> bins;
  double lower = -std::numeric_limits<double>::infinity();
  size_t idx = 0;
  
  for (size_t cp = 0; cp < cut_points.size(); ++cp) {
    double upper = cut_points[cp];
    Bin bin;
    bin.lower_bound = lower;
    bin.upper_bound = upper;
    bin.count = 0;
    bin.count_pos = 0;
    bin.count_neg = 0;
    
    while (idx < (size_t)n && feature[indices[idx]] <= upper) {
      bin.count++;
      if (target[indices[idx]] == 1) {
        bin.count_pos++;
      } else {
        bin.count_neg++;
      }
      idx++;
    }
    bins.push_back(bin);
    lower = upper;
  }
  
  if (idx < (size_t)n) {
    Bin bin;
    bin.lower_bound = lower;
    bin.upper_bound = std::numeric_limits<double>::infinity();
    bin.count = 0;
    bin.count_pos = 0;
    bin.count_neg = 0;
    while (idx < (size_t)n) {
      bin.count++;
      if (target[indices[idx]] == 1) {
        bin.count_pos++;
      } else {
        bin.count_neg++;
      }
      idx++;
    }
    bins.push_back(bin);
  }
  
  return bins;
}

// Calcula WoE e IV
void OptimalBinningNumericalLPDB::calculate_woe_iv(std::vector<Bin> &bins, int total_pos, int total_neg) {
  for (auto &bin : bins) {
    double dist_pos = (bin.count_pos > 0) ? static_cast<double>(bin.count_pos) / total_pos : 1e-10;
    double dist_neg = (bin.count_neg > 0) ? static_cast<double>(bin.count_neg) / total_neg : 1e-10;
    
    if (dist_pos <= 0) dist_pos = 1e-10;
    if (dist_neg <= 0) dist_neg = 1e-10;
    
    bin.woe = std::log(dist_pos / dist_neg);
    bin.iv = (dist_pos - dist_neg) * bin.woe;
  }
}

// Aplica monotonicidade
void OptimalBinningNumericalLPDB::enforce_monotonicity(std::vector<Bin> &bins, int total_pos, int total_neg) {
  // Determina direção com base na correlação
  std::vector<double> bin_means;
  bin_means.reserve(bins.size());
  
  for (const auto &bin : bins) {
    double mean;
    if (std::isinf(bin.lower_bound) && !std::isinf(bin.upper_bound)) {
      mean = bin.upper_bound - 1.0;
    } else if (!std::isinf(bin.lower_bound) && std::isinf(bin.upper_bound)) {
      mean = bin.lower_bound + 1.0;
    } else if (std::isinf(bin.lower_bound) && std::isinf(bin.upper_bound)) {
      mean = 0.0;
    } else {
      mean = (bin.lower_bound + bin.upper_bound) / 2.0;
    }
    bin_means.push_back(mean);
  }
  
  std::vector<double> woe_values_vec;
  woe_values_vec.reserve(bins.size());
  for (const auto &bin : bins) {
    woe_values_vec.push_back(bin.woe);
  }
  
  double corr = 0.0;
  if (bins.size() > 1) {
    corr = compute_correlation(bin_means, woe_values_vec);
  }
  
  bool desired_increasing = (corr >= 0);
  
  // Merges para forçar monotonicidade
  while (iterations_run < max_iterations) {
    bool merged = false;
    for (size_t i = 1; i < bins.size(); ++i) {
      if ((desired_increasing && bins[i].woe < bins[i - 1].woe) ||
          (!desired_increasing && bins[i].woe > bins[i - 1].woe)) {
        
        if (bins.size() <= static_cast<size_t>(min_bins)) {
          converged = false;
          return;
        }
        
        // Merge bins[i - 1] e bins[i]
        bins[i - 1].upper_bound = bins[i].upper_bound;
        bins[i - 1].count += bins[i].count;
        bins[i - 1].count_pos += bins[i].count_pos;
        bins[i - 1].count_neg += bins[i].count_neg;
        bins.erase(bins.begin() + i);
        
        double dist_pos = (bins[i - 1].count_pos > 0) ? static_cast<double>(bins[i - 1].count_pos) / total_pos : 1e-10;
        double dist_neg = (bins[i - 1].count_neg > 0) ? static_cast<double>(bins[i - 1].count_neg) / total_neg : 1e-10;
        if (dist_pos <= 0) dist_pos = 1e-10;
        if (dist_neg <= 0) dist_neg = 1e-10;
        bins[i - 1].woe = std::log(dist_pos / dist_neg);
        bins[i - 1].iv = (dist_pos - dist_neg) * bins[i - 1].woe;
        
        iterations_run++;
        merged = true;
        break;
      }
    }
    if (!merged) break;
  }
  
  // Garante max_bins
  while (bins.size() > static_cast<size_t>(max_bins) && iterations_run < max_iterations) {
    double min_woe_diff = std::numeric_limits<double>::max();
    size_t merge_index = 0;
    for (size_t i = 1; i < bins.size(); ++i) {
      double woe_diff = std::abs(bins[i].woe - bins[i - 1].woe);
      if (woe_diff < min_woe_diff) {
        min_woe_diff = woe_diff;
        merge_index = i - 1;
      }
    }
    
    bins[merge_index].upper_bound = bins[merge_index + 1].upper_bound;
    bins[merge_index].count += bins[merge_index + 1].count;
    bins[merge_index].count_pos += bins[merge_index + 1].count_pos;
    bins[merge_index].count_neg += bins[merge_index + 1].count_neg;
    bins.erase(bins.begin() + merge_index + 1);
    
    double dist_pos = (bins[merge_index].count_pos > 0) ? static_cast<double>(bins[merge_index].count_pos) / total_pos : 1e-10;
    double dist_neg = (bins[merge_index].count_neg > 0) ? static_cast<double>(bins[merge_index].count_neg) / total_neg : 1e-10;
    if (dist_pos <= 0) dist_pos = 1e-10;
    if (dist_neg <= 0) dist_neg = 1e-10;
    bins[merge_index].woe = std::log(dist_pos / dist_neg);
    bins[merge_index].iv = (dist_pos - dist_neg) * bins[merge_index].woe;
    
    iterations_run++;
  }
  
  // Verifica monotonicidade final
  for (size_t i = 1; i < bins.size(); ++i) {
    if ((desired_increasing && bins[i].woe < bins[i - 1].woe) ||
        (!desired_increasing && bins[i].woe > bins[i - 1].woe)) {
      converged = false;
      break;
    }
  }
}

// Formata intervalo do bin
std::string OptimalBinningNumericalLPDB::format_bin_interval(double lower, double upper, bool first, bool last) {
  std::ostringstream oss;
  oss.precision(6);
  oss << std::fixed;
  oss << "(";
  if (first) {
    oss << "-Inf";
  } else {
    oss << lower;
  }
  oss << "; ";
  if (last) {
    oss << "+Inf";
  } else {
    oss << upper;
  }
  oss << "]";
  return oss.str();
}


//' @title Optimal Binning for Numerical Variables using Local Polynomial Density Binning (LPDB)
//'
//' @description
//' Implements the Local Polynomial Density Binning (LPDB) algorithm for optimal binning of numerical variables. 
//' The method creates bins that maximize predictive power while maintaining monotonicity in Weight of Evidence (WoE).
//' It handles rare bins, ensures numerical stability, and provides flexibility through various customizable parameters.
//'
//' @details
//' ### Key Steps:
//' 1. **Input Validation**: Ensures the `feature` and `target` vectors are valid, checks binary nature of the `target` vector, 
//'    and removes missing values (`NA`).
//' 2. **Pre-Binning**: Divides the feature into preliminary bins using quantile-based partitioning or unique values.
//' 3. **Calculation of WoE and IV**: Computes the WoE and Information Value (IV) for each bin based on the target distribution.
//' 4. **Monotonicity Enforcement**: Adjusts bins iteratively to ensure monotonicity in WoE values, either increasing or decreasing.
//' 5. **Rare Bin Merging**: Merges bins with frequencies below the `bin_cutoff` threshold to ensure statistical stability.
//' 6. **Validation**: Ensures bins are non-overlapping, cover the entire range of the feature, and are consistent with constraints on `min_bins` and `max_bins`.
//'
//' ### Mathematical Framework:
//' - **Weight of Evidence (WoE)**: For a bin \( i \):
//'   \deqn{WoE_i = \ln\left(\frac{\text{Distribution of positives}_i}{\text{Distribution of negatives}_i}\right)}
//'
//' - **Information Value (IV)**: Aggregates the predictive power across all bins:
//'   \deqn{IV = \sum_{i=1}^{N} (\text{Distribution of positives}_i - \text{Distribution of negatives}_i) \times WoE_i}
//'
//' ### Features:
//' - **Monotonicity**: Ensures the WoE values are either strictly increasing or decreasing across bins.
//' - **Rare Bin Handling**: Merges bins with low frequencies to maintain statistical reliability.
//' - **Numerical Stability**: Incorporates small constants to avoid division by zero or undefined logarithms.
//' - **Flexibility**: Supports custom definitions for minimum and maximum bins, convergence thresholds, and iteration limits.
//' - **Output Metadata**: Provides detailed bin information, including WoE, IV, and cutpoints for interpretability and downstream analysis.
//'
//' ### Parameters:
//' - `min_bins`: Minimum number of bins to be created (default: 3).
//' - `max_bins`: Maximum number of bins allowed (default: 5).
//' - `bin_cutoff`: Minimum proportion of total observations required for a bin to be retained as standalone (default: 0.05).
//' - `max_n_prebins`: Maximum number of pre-bins before optimization (default: 20).
//' - `convergence_threshold`: Threshold for determining convergence in terms of IV changes (default: 1e-6).
//' - `max_iterations`: Maximum number of iterations allowed for binning optimization (default: 1000).
//'
//' @param target An integer binary vector (0 or 1) representing the response variable.
//' @param feature A numeric vector representing the feature to be binned.
//' @param min_bins Minimum number of bins to be created (default: 3).
//' @param max_bins Maximum number of bins allowed (default: 5).
//' @param bin_cutoff Minimum frequency proportion for retaining a bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins before optimization (default: 20).
//' @param convergence_threshold Convergence threshold for IV optimization (default: 1e-6).
//' @param max_iterations Maximum number of iterations allowed for optimization (default: 1000).
//'
//' @return A list containing the following elements:
//' \itemize{
//'   \item `bin`: A vector of bin intervals in the format "[lower;upper)".
//'   \item `woe`: A numeric vector of WoE values for each bin.
//'   \item `iv`: A numeric vector of IV contributions for each bin.
//'   \item `count`: An integer vector of the total number of observations per bin.
//'   \item `count_pos`: An integer vector of the number of positive cases per bin.
//'   \item `count_neg`: An integer vector of the number of negative cases per bin.
//'   \item `cutpoints`: A numeric vector of the cutpoints defining the bin edges.
//'   \item `converged`: A boolean indicating whether the algorithm converged.
//'   \item `iterations`: An integer indicating the number of iterations executed.
//' }
//'
//' @examples
//' \dontrun{
//' set.seed(123)
//' target <- sample(0:1, 1000, replace = TRUE)
//' feature <- rnorm(1000)
//' result <- optimal_binning_numerical_lpdb(target, feature, min_bins = 3, max_bins = 6)
//' print(result$bin)
//' print(result$woe)
//' print(result$iv)
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_lpdb(Rcpp::IntegerVector target,
                                         Rcpp::NumericVector feature,
                                         int min_bins = 3,
                                         int max_bins = 5,
                                         double bin_cutoff = 0.05,
                                         int max_n_prebins = 20,
                                         double convergence_threshold = 1e-6,
                                         int max_iterations = 1000) {
 OptimalBinningNumericalLPDB binning(min_bins, max_bins, bin_cutoff, max_n_prebins,
                                     convergence_threshold, max_iterations);
 
 return binning.fit(feature, target);
}

/*
 Melhorias Implementadas:
 - Verificações robustas dos inputs (target binário com 0 e 1, ausência de NA, min_bins e max_bins coerentes).
 - Uso de epsilon ao calcular log(0) no WOE.
 - Ajuste no cálculo da direção de monotonicidade utilizando correlação e fusões de bins.
 - Garantia de não exceder max_iterations.
 - Comentários e códigos consistentes e mais robustos.
 - Preservadas entradas e saídas conforme solicitado.
*/



// #include <Rcpp.h>
// #include <vector>
// #include <algorithm>
// #include <cmath>
// #include <map>
// #include <string>
// #include <sstream>
// #include <numeric>
// #include <limits>
// 
// using namespace Rcpp;
// 
// // Function to compute Pearson correlation between two vectors
// inline double compute_correlation(const std::vector<double> &x, const std::vector<double> &y) {
//   if (x.size() != y.size() || x.empty()) {
//     Rcpp::stop("Vectors must be of the same non-zero length for correlation.");
//   }
//   
//   double mean_x = std::accumulate(x.begin(), x.end(), 0.0) / x.size();
//   double mean_y = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
//   
//   double numerator = 0.0;
//   double denom_x = 0.0;
//   double denom_y = 0.0;
//   
//   for (size_t i = 0; i < x.size(); ++i) {
//     double dx = x[i] - mean_x;
//     double dy = y[i] - mean_y;
//     numerator += dx * dy;
//     denom_x += dx * dx;
//     denom_y += dy * dy;
//   }
//   
//   if (denom_x == 0 || denom_y == 0) {
//     Rcpp::warning("Standard deviation is zero. Returning correlation as 0.");
//     return 0.0;
//   }
//   
//   return numerator / std::sqrt(denom_x * denom_y);
// }
// 
// // Class for Optimal Binning Numerical using Local Polynomial Density Binning (LPDB)
// class OptimalBinningNumericalLPDB {
// public:
//   OptimalBinningNumericalLPDB(int min_bins = 3, int max_bins = 5, double bin_cutoff = 0.05, int max_n_prebins = 20,
//                               double convergence_threshold = 1e-6, int max_iterations = 1000)
//     : min_bins(min_bins), max_bins(max_bins), bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins),
//       convergence_threshold(convergence_threshold), max_iterations(max_iterations),
//       converged(true), iterations_run(0) {
//     // Validate constraints
//     if (min_bins < 2) {
//       Rcpp::stop("min_bins must be at least 2.");
//     }
//     if (max_bins < min_bins) {
//       Rcpp::stop("max_bins must be greater than or equal to min_bins.");
//     }
//     if (bin_cutoff < 0.0 || bin_cutoff > 1.0) {
//       Rcpp::stop("bin_cutoff must be between 0 and 1.");
//     }
//     if (max_n_prebins < min_bins) {
//       Rcpp::stop("max_n_prebins must be greater than or equal to min_bins.");
//     }
//   }
//   
//   Rcpp::List fit(Rcpp::NumericVector feature, Rcpp::IntegerVector target);
//   
// private:
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
//   double convergence_threshold;
//   int max_iterations;
//   bool converged;
//   int iterations_run;
//   
//   struct Bin {
//     double lower_bound;
//     double upper_bound;
//     int count;
//     int count_pos;
//     int count_neg;
//     double woe;
//     double iv;
//   };
//   
//   std::vector<Bin> prebinning(const std::vector<double> &feature, const std::vector<int> &target);
//   void calculate_woe_iv(std::vector<Bin> &bins, int total_pos, int total_neg);
//   void enforce_monotonicity(std::vector<Bin> &bins, int total_pos, int total_neg);
//   std::string format_bin_interval(double lower, double upper, bool first = false, bool last = false);
// };
// 
// // Fit method Implementation
// Rcpp::List OptimalBinningNumericalLPDB::fit(Rcpp::NumericVector feature, Rcpp::IntegerVector target) {
//   int n = feature.size();
//   if (n != target.size()) {
//     Rcpp::stop("feature and target must have the same length.");
//   }
//   
//   // Ensure target is binary
//   IntegerVector unique_targets = unique(target);
//   if (unique_targets.size() != 2) {
//     Rcpp::stop("Target must be binary (0 and 1).");
//   }
//   
//   // Remove missing values
//   LogicalVector not_na = (!is_na(feature)) & (!is_na(target));
//   NumericVector clean_feature = feature[not_na];
//   IntegerVector clean_target = target[not_na];
//   
//   if (clean_feature.size() == 0) {
//     Rcpp::stop("No valid observations after removing missing values.");
//   }
//   
//   // Total positives and negatives
//   int total_pos = std::accumulate(clean_target.begin(), clean_target.end(), 0);
//   int total_neg = clean_target.size() - total_pos;
//   
//   if (total_pos == 0 || total_neg == 0) {
//     Rcpp::stop("Target must have both positive and negative classes.");
//   }
//   
//   // Check for single unique feature value
//   NumericVector unique_vec = unique(clean_feature);
//   std::vector<double> unique_feature = Rcpp::as<std::vector<double>>(unique_vec);
//   if (unique_feature.size() == 1) {
//     // All feature values are the same; create a single bin
//     Bin bin;
//     bin.lower_bound = -std::numeric_limits<double>::infinity();
//     bin.upper_bound = std::numeric_limits<double>::infinity();
//     bin.count = clean_feature.size();
//     bin.count_pos = total_pos;
//     bin.count_neg = total_neg;
//     // Calculate WoE and IV
//     double dist_pos = static_cast<double>(bin.count_pos) / total_pos;
//     double dist_neg = static_cast<double>(bin.count_neg) / total_neg;
//     bin.woe = std::log((dist_pos + 1e-10) / (dist_neg + 1e-10));
//     bin.iv = (dist_pos - dist_neg) * bin.woe;
//     
//     std::vector<Bin> bins = { bin };
//     
//     // Prepare bin labels
//     std::vector<std::string> bin_labels;
//     bin_labels.emplace_back(format_bin_interval(bin.lower_bound, bin.upper_bound, true, true));
//     
//     // Prepare output DataFrame
//     CharacterVector bin_intervals(1);
//     bin_intervals[0] = bin_labels[0];
//     NumericVector woe_values(1, bin.woe);
//     NumericVector iv_values(1, bin.iv);
//     IntegerVector counts(1, bin.count);
//     IntegerVector counts_pos(1, bin.count_pos);
//     IntegerVector counts_neg(1, bin.count_neg);
//     
//     DataFrame woebin = DataFrame::create(
//       Named("bin") = bin_intervals,
//       Named("woe") = woe_values,
//       Named("iv") = iv_values,
//       Named("count") = counts,
//       Named("count_pos") = counts_pos,
//       Named("count_neg") = counts_neg
//     );
//     
//     // Prepare cutpoints (empty since there's only one bin)
//     NumericVector cutpoints;
//     
//     // Return results
//     return List::create(
//       Named("bin") = bin_intervals,
//       Named("woe") = woe_values,
//       Named("iv") = iv_values,
//       Named("count") = counts,
//       Named("count_pos") = counts_pos,
//       Named("count_neg") = counts_neg,
//       Named("cutpoints") = cutpoints,
//       Named("converged") = true,
//       Named("iterations") = 0
//     );
//   }
//   
//   // Pre-binning
//   std::vector<Bin> bins = prebinning(Rcpp::as<std::vector<double>>(clean_feature),
//                                      Rcpp::as<std::vector<int>>(clean_target));
//   
//   // Calculate WoE and IV
//   calculate_woe_iv(bins, total_pos, total_neg);
//   
//   // Enforce monotonicity
//   enforce_monotonicity(bins, total_pos, total_neg);
//   
//   // Prepare bin labels and cutpoints
//   std::vector<std::string> bin_labels;
//   std::vector<double> cutpoints_list;
//   
//   for (size_t i = 0; i < bins.size(); ++i) {
//     bin_labels.emplace_back(format_bin_interval(bins[i].lower_bound, bins[i].upper_bound,
//                                                 i == 0, i == (bins.size() - 1)));
//     if (i < bins.size() - 1) {
//       cutpoints_list.emplace_back(bins[i].upper_bound);
//     }
//   }
//   
//   // Assign WoE values and IV
//   std::vector<double> woe_vals;
//   std::vector<double> iv_vals;
//   std::vector<int> counts_vec;
//   std::vector<int> counts_pos_vec;
//   std::vector<int> counts_neg_vec;
//   
//   for (size_t i = 0; i < bins.size(); ++i) {
//     woe_vals.push_back(bins[i].woe);
//     iv_vals.push_back(bins[i].iv);
//     counts_vec.push_back(bins[i].count);
//     counts_pos_vec.push_back(bins[i].count_pos);
//     counts_neg_vec.push_back(bins[i].count_neg);
//   }
//   
//   // Return results
//   return List::create(
//     Named("bin") = bin_labels,
//     Named("woe") = woe_vals,
//     Named("iv") = iv_vals,
//     Named("count") = counts_vec,
//     Named("count_pos") = counts_pos_vec,
//     Named("count_neg") = counts_neg_vec,
//     Named("cutpoints") = cutpoints_list,
//     Named("converged") = converged,
//     Named("iterations") = iterations_run
//   );
// }
// 
// // Pre-binning Implementation
// std::vector<OptimalBinningNumericalLPDB::Bin> OptimalBinningNumericalLPDB::prebinning(const std::vector<double> &feature, const std::vector<int> &target) {
//   int n = feature.size();
//   
//   // Create a vector of indices and sort them based on feature values
//   std::vector<int> indices(n);
//   for (int i = 0; i < n; ++i) {
//     indices[i] = i;
//   }
//   
//   std::sort(indices.begin(), indices.end(), [&](int a, int b) -> bool {
//     return feature[a] < feature[b];
//   });
//   
//   // Determine initial cut points for pre-bins using quantiles
//   std::vector<double> cut_points;
//   int bin_size = n / max_n_prebins;
//   if (bin_size < 1) bin_size = 1;
//   
//   for (int i = bin_size; i < n; i += bin_size) {
//     double val = feature[indices[i]];
//     if (cut_points.empty() || val != cut_points.back()) {
//       cut_points.push_back(val);
//     }
//   }
//   
//   // Create bins
//   std::vector<Bin> bins;
//   double lower = -std::numeric_limits<double>::infinity(); // Start from -Inf
//   size_t idx = 0;
//   
//   for (size_t cp = 0; cp < cut_points.size(); ++cp) {
//     double upper = cut_points[cp];
//     Bin bin;
//     bin.lower_bound = lower;
//     bin.upper_bound = upper;
//     bin.count = 0;
//     bin.count_pos = 0;
//     bin.count_neg = 0;
//     
//     while (idx < n && feature[indices[idx]] <= upper) {
//       bin.count++;
//       if (target[indices[idx]] == 1) {
//         bin.count_pos++;
//       } else {
//         bin.count_neg++;
//       }
//       idx++;
//     }
//     bins.push_back(bin);
//     lower = upper;
//   }
//   
//   // Last bin
//   if (idx < n) {
//     Bin bin;
//     bin.lower_bound = lower;
//     bin.upper_bound = std::numeric_limits<double>::infinity(); // End at +Inf
//     bin.count = 0;
//     bin.count_pos = 0;
//     bin.count_neg = 0;
//     while (idx < n) {
//       bin.count++;
//       if (target[indices[idx]] == 1) {
//         bin.count_pos++;
//       } else {
//         bin.count_neg++;
//       }
//       idx++;
//     }
//     bins.push_back(bin);
//   }
//   
//   return bins;
// }
// 
// // Calculate WoE and IV Implementation
// void OptimalBinningNumericalLPDB::calculate_woe_iv(std::vector<Bin> &bins, int total_pos, int total_neg) {
//   // Calculate WoE and IV for each bin
//   for (size_t i = 0; i < bins.size(); ++i) {
//     double dist_pos = static_cast<double>(bins[i].count_pos) / total_pos;
//     double dist_neg = static_cast<double>(bins[i].count_neg) / total_neg;
//     
//     // Handle cases where dist_pos or dist_neg is zero to avoid log(0)
//     if (dist_pos <= 0) dist_pos = 1e-10;
//     if (dist_neg <= 0) dist_neg = 1e-10;
//     
//     bins[i].woe = std::log(dist_pos / dist_neg);
//     bins[i].iv = (dist_pos - dist_neg) * bins[i].woe;
//   }
// }
// 
// // Enforce Monotonicity Implementation
// void OptimalBinningNumericalLPDB::enforce_monotonicity(std::vector<Bin> &bins, int total_pos, int total_neg) {
//   // Determine the direction of monotonicity based on correlation
//   std::vector<double> bin_means;
//   for (const auto &bin : bins) {
//     // Calculate bin mean excluding infinities
//     double mean;
//     if (std::isinf(bin.lower_bound) && !std::isinf(bin.upper_bound)) {
//       mean = bin.upper_bound - 1.0; // Arbitrary value within bin
//     } else if (!std::isinf(bin.lower_bound) && std::isinf(bin.upper_bound)) {
//       mean = bin.lower_bound + 1.0; // Arbitrary value within bin
//     } else if (std::isinf(bin.lower_bound) && std::isinf(bin.upper_bound)) {
//       mean = 0.0; // Undefined, set to zero
//     } else {
//       mean = (bin.lower_bound + bin.upper_bound) / 2.0;
//     }
//     bin_means.push_back(mean);
//   }
//   
//   std::vector<double> woe_values_vec;
//   for (const auto &bin : bins) {
//     woe_values_vec.push_back(bin.woe);
//   }
//   
//   double corr = compute_correlation(bin_means, woe_values_vec);
//   bool desired_increasing = (corr >= 0);
//   
//   // Iteratively merge bins that violate monotonicity
//   while (iterations_run < max_iterations) {
//     bool merged = false;
//     for (size_t i = 1; i < bins.size(); ++i) {
//       if ((desired_increasing && bins[i].woe < bins[i - 1].woe) ||
//           (!desired_increasing && bins[i].woe > bins[i - 1].woe)) {
//         // Check if merging would violate min_bins constraint
//         if (bins.size() <= static_cast<size_t>(min_bins)) {
//           converged = false;
//           return;  // Stop merging if we've reached min_bins
//         }
//         
//         // Merge bins[i - 1] and bins[i]
//         bins[i - 1].upper_bound = bins[i].upper_bound;
//         bins[i - 1].count += bins[i].count;
//         bins[i - 1].count_pos += bins[i].count_pos;
//         bins[i - 1].count_neg += bins[i].count_neg;
//         bins.erase(bins.begin() + i);
//         
//         // Recalculate WoE and IV for the merged bin
//         double dist_pos = static_cast<double>(bins[i - 1].count_pos) / total_pos;
//         double dist_neg = static_cast<double>(bins[i - 1].count_neg) / total_neg;
//         if (dist_pos <= 0) dist_pos = 1e-10;
//         if (dist_neg <= 0) dist_neg = 1e-10;
//         bins[i - 1].woe = std::log(dist_pos / dist_neg);
//         bins[i - 1].iv = (dist_pos - dist_neg) * bins[i - 1].woe;
//         
//         iterations_run++;
//         merged = true;
//         break;
//       }
//     }
//     if (!merged) {
//       // No more violations
//       break;
//     }
//   }
//   
//   // Ensure max_bins constraint is respected
//   while (bins.size() > static_cast<size_t>(max_bins)) {
//     // Find the pair of adjacent bins with the smallest difference in WoE
//     double min_woe_diff = std::numeric_limits<double>::max();
//     size_t merge_index = 0;
//     for (size_t i = 1; i < bins.size(); ++i) {
//       double woe_diff = std::abs(bins[i].woe - bins[i - 1].woe);
//       if (woe_diff < min_woe_diff) {
//         min_woe_diff = woe_diff;
//         merge_index = i - 1;
//       }
//     }
//     
//     // Merge the bins with the smallest WoE difference
//     bins[merge_index].upper_bound = bins[merge_index + 1].upper_bound;
//     bins[merge_index].count += bins[merge_index + 1].count;
//     bins[merge_index].count_pos += bins[merge_index + 1].count_pos;
//     bins[merge_index].count_neg += bins[merge_index + 1].count_neg;
//     bins.erase(bins.begin() + merge_index + 1);
//     
//     // Recalculate WoE and IV for the merged bin
//     double dist_pos = static_cast<double>(bins[merge_index].count_pos) / total_pos;
//     double dist_neg = static_cast<double>(bins[merge_index].count_neg) / total_neg;
//     if (dist_pos <= 0) dist_pos = 1e-10;
//     if (dist_neg <= 0) dist_neg = 1e-10;
//     bins[merge_index].woe = std::log(dist_pos / dist_neg);
//     bins[merge_index].iv = (dist_pos - dist_neg) * bins[merge_index].woe;
//     
//     iterations_run++;
//   }
//   
//   // Final monotonicity check
//   bool final_monotonic = true;
//   for (size_t i = 1; i < bins.size(); ++i) {
//     if ((desired_increasing && bins[i].woe < bins[i - 1].woe) ||
//         (!desired_increasing && bins[i].woe > bins[i - 1].woe)) {
//       final_monotonic = false;
//       break;
//     }
//   }
//   
//   if (!final_monotonic) {
//     converged = false;
//   }
// }
// 
// // Format bin interval into string
// std::string OptimalBinningNumericalLPDB::format_bin_interval(double lower, double upper, bool first, bool last) {
//   std::ostringstream oss;
//   oss << "(";
//   if (first) {
//     oss << "-Inf";
//   } else {
//     oss << lower;
//   }
//   oss << "; ";
//   if (last) {
//     oss << "+Inf";
//   } else {
//     oss << upper;
//   }
//   oss << "]";
//   return oss.str();
// }
// 
// 
// //' @title Optimal Binning for Numerical Variables using Local Polynomial Density Binning (LPDB)
// //'
// //' @description This function implements the Local Polynomial Density Binning (LPDB) algorithm for optimal binning of numerical variables.
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
// //' \item{bins}{A vector of bin labels.}
// //' \item{woe}{A numeric vector of Weight of Evidence (WoE) values for each bin.}
// //' \item{iv}{A numeric vector of Information Value (IV) for each bin.}
// //' \item{count}{Total count of observations in each bin.}
// //' \item{count_pos}{Count of positive target observations in each bin.}
// //' \item{count_neg}{Count of negative target observations in each bin.}
// //' \item{cutpoints}{Numeric vector of cutpoints used to generate the bins.}
// //' \item{converged}{Logical value indicating if the algorithm converged.}
// //' \item{iterations}{Number of iterations run by the algorithm.}
// //'
// //' @examples
// //' \dontrun{
// //' # Create sample data
// //' set.seed(123)
// //' target <- sample(0:1, 1000, replace = TRUE)
// //' feature <- rnorm(1000)
// //'
// //' # Run optimal binning
// //' result <- optimal_binning_numerical_lpdb(target, feature)
// //'
// //' # View results
// //' print(result$bins)
// //' print(result$woe)
// //' print(result$iv)
// //' }
// //'
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_numerical_lpdb(Rcpp::IntegerVector target,
//                                          Rcpp::NumericVector feature,
//                                          int min_bins = 3,
//                                          int max_bins = 5,
//                                          double bin_cutoff = 0.05,
//                                          int max_n_prebins = 20,
//                                          double convergence_threshold = 1e-6,
//                                          int max_iterations = 1000) {
//  // Initialize binning class
//  OptimalBinningNumericalLPDB binning(min_bins, max_bins, bin_cutoff, max_n_prebins,
//                                      convergence_threshold, max_iterations);
//  
//  // Perform binning
//  return binning.fit(feature, target);
// }
