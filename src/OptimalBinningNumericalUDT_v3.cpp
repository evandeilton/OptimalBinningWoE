#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_set>
#include <sstream>

using namespace Rcpp;

// Helper structure for bins
struct Bin {
  double lower;
  double upper;
  double woe;
  double iv;
  int count;
  int count_pos;
  int count_neg;
};

// Comparator for sorting
bool compareBins(const Bin &a, const Bin &b) {
  return a.lower < b.lower;
}

// Forward declaration of handle_few_unique_values
Rcpp::List handle_few_unique_values(
    const std::vector<double>& feature_vec,
    const std::vector<int>& target_vec
);

// Class for Optimal Binning
class OptimalBinningNumericalUDT {
private:
  std::vector<double> feature;
  std::vector<int> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  std::vector<Bin> bins;
  double convergence_threshold;
  int max_iterations;
  bool converged;
  int iterations_run;
  
public:
  OptimalBinningNumericalUDT(std::vector<double> feature,
                             std::vector<int> target,
                             int min_bins = 3,
                             int max_bins = 5,
                             double bin_cutoff = 0.05,
                             int max_n_prebins = 20,
                             double convergence_threshold = 1e-6,
                             int max_iterations = 1000) 
    : feature(std::move(feature)), 
      target(std::move(target)), 
      min_bins(std::max(2, min_bins)),
      max_bins(std::max(this->min_bins, max_bins)),
      bin_cutoff(bin_cutoff),
      max_n_prebins(max_n_prebins),
      convergence_threshold(convergence_threshold),
      max_iterations(max_iterations),
      converged(false),
      iterations_run(0) {}
  
  void fit() {
    // Pre-binning using quantiles
    std::vector<double> cut_points = get_quantile_cutpoints(feature, max_n_prebins);
    
    // Initial binning
    bins = initial_binning(feature, target, cut_points);
    
    // Rare bin merging
    merge_rare_bins();
    
    // Bin optimization
    optimize_bins();
  }
  
  Rcpp::List get_result() const {
    size_t n_bins = bins.size();
    CharacterVector bin_intervals(n_bins);
    NumericVector woe_values(n_bins), iv_values(n_bins);
    IntegerVector counts(n_bins), counts_pos(n_bins), counts_neg(n_bins);
    NumericVector cutpoints(n_bins - 1);
    
    for (size_t i = 0; i < n_bins; ++i) {
      bin_intervals[i] = create_interval(bins[i].lower, bins[i].upper);
      woe_values[i] = bins[i].woe;
      iv_values[i] = bins[i].iv;
      counts[i] = bins[i].count;
      counts_pos[i] = bins[i].count_pos;
      counts_neg[i] = bins[i].count_neg;
      if (i < n_bins - 1) {
        cutpoints[i] = bins[i].upper;
      }
    }
    
    return Rcpp::List::create(
      Rcpp::Named("bin") = bin_intervals,
      Rcpp::Named("woe") = woe_values,
      Rcpp::Named("iv") = iv_values,
      Rcpp::Named("count") = counts,
      Rcpp::Named("count_pos") = counts_pos,
      Rcpp::Named("count_neg") = counts_neg,
      Rcpp::Named("cutpoints") = cutpoints,
      Rcpp::Named("converged") = converged,
      Rcpp::Named("iterations") = iterations_run
    );
  }
  
  void calculate_woe_iv(std::vector<Bin> &bins) const {
    int total_pos = 0, total_neg = 0;
    for (const auto &bin : bins) {
      total_pos += bin.count_pos;
      total_neg += bin.count_neg;
    }
    
    double total_pos_d = total_pos > 0 ? static_cast<double>(total_pos) : 1.0;
    double total_neg_d = total_neg > 0 ? static_cast<double>(total_neg) : 1.0;
    
    for (auto &bin : bins) {
      double dist_pos = static_cast<double>(bin.count_pos) / total_pos_d;
      double dist_neg = static_cast<double>(bin.count_neg) / total_neg_d;
      if (dist_pos == 0.0) dist_pos = 0.0001;
      if (dist_neg == 0.0) dist_neg = 0.0001;
      bin.woe = std::log(dist_pos / dist_neg);
      bin.iv = (dist_pos - dist_neg) * bin.woe;
    }
  }
  
  std::string create_interval(double lower, double upper) const {
    std::stringstream ss;
    ss << "(";
    if (std::isinf(lower))
      ss << "-Inf";
    else
      ss << lower;
    ss << ";";
    if (std::isinf(upper))
      ss << "+Inf";
    else
      ss << upper;
    ss << "]";
    return ss.str();
  }
  
private:
  std::vector<double> get_quantile_cutpoints(const std::vector<double> &data, int n_bins) const {
    std::vector<double> quantiles;
    if (n_bins <= 1) {
      return quantiles;
    }
    
    std::vector<double> sorted_data = data;
    sorted_data.erase(std::remove_if(sorted_data.begin(), sorted_data.end(),
                                     [](double x) { return std::isnan(x); }), sorted_data.end());
    
    if (sorted_data.empty()) {
      return quantiles;
    }
    
    std::sort(sorted_data.begin(), sorted_data.end());
    
    for (int i = 1; i < n_bins; ++i) {
      double p = static_cast<double>(i) / n_bins;
      double pos = p * (sorted_data.size() - 1);
      size_t idx = static_cast<size_t>(std::floor(pos));
      double frac = pos - static_cast<double>(idx);
      double quantile;
      if (idx + 1 < sorted_data.size()) {
        quantile = sorted_data[idx] + frac * (sorted_data[idx + 1] - sorted_data[idx]);
      } else {
        quantile = sorted_data[idx];
      }
      quantiles.push_back(quantile);
    }
    
    std::vector<double> unique_quantiles;
    std::unordered_set<double> seen;
    for (double q : quantiles) {
      if (seen.find(q) == seen.end()) {
        unique_quantiles.push_back(q);
        seen.insert(q);
      }
    }
    
    return unique_quantiles;
  }
  
  std::vector<Bin> initial_binning(const std::vector<double> &feature,
                                   const std::vector<int> &target,
                                   const std::vector<double> &cut_points) const {
    std::vector<Bin> initial_bins;
    std::vector<double> boundaries = cut_points;
    boundaries.insert(boundaries.begin(), -std::numeric_limits<double>::infinity());
    boundaries.push_back(std::numeric_limits<double>::infinity());
    size_t n_bins = boundaries.size() - 1;
    
    initial_bins.resize(n_bins);
    for (size_t i = 0; i < n_bins; ++i) {
      initial_bins[i].lower = boundaries[i];
      initial_bins[i].upper = boundaries[i + 1];
      initial_bins[i].count = 0;
      initial_bins[i].count_pos = 0;
      initial_bins[i].count_neg = 0;
      initial_bins[i].woe = 0.0;
      initial_bins[i].iv = 0.0;
    }
    
    for (size_t i = 0; i < feature.size(); ++i) {
      double val = feature[i];
      int bin_idx = find_bin_index(val, boundaries);
      if (bin_idx >= 0 && static_cast<size_t>(bin_idx) < n_bins) {
        initial_bins[bin_idx].count++;
        if (target[i] == 1) {
          initial_bins[bin_idx].count_pos++;
        } else {
          initial_bins[bin_idx].count_neg++;
        }
      }
    }
    
    calculate_woe_iv(initial_bins);
    
    return initial_bins;
  }
  
  int find_bin_index(double value, const std::vector<double> &boundaries) const {
    int left = 0;
    int right = boundaries.size() - 1;
    while (left < right - 1) {
      int mid = left + (right - left) / 2;
      if (value <= boundaries[mid]) {
        right = mid;
      } else {
        left = mid;
      }
    }
    return left;
  }
  
  void merge_rare_bins() {
    size_t total_count = feature.size();
    double cutoff_count = bin_cutoff * static_cast<double>(total_count);
    bool merged = false;
    
    do {
      merged = false;
      for (size_t i = 0; i < bins.size(); ++i) {
        if (static_cast<double>(bins[i].count) < cutoff_count && bins.size() > static_cast<size_t>(min_bins)) {
          if (i > 0) {
            bins[i - 1] = merge_bins(bins[i - 1], bins[i]);
            bins.erase(bins.begin() + i);
          } else if (i < bins.size() - 1) {
            bins[i + 1] = merge_bins(bins[i], bins[i + 1]);
            bins.erase(bins.begin() + i);
          }
          merged = true;
          break;
        }
      }
    } while (merged && bins.size() > static_cast<size_t>(min_bins));
    
    calculate_woe_iv(bins);
  }
  
  Bin merge_bins(const Bin &bin1, const Bin &bin2) const {
    Bin merged;
    merged.lower = bin1.lower;
    merged.upper = bin2.upper;
    merged.count = bin1.count + bin2.count;
    merged.count_pos = bin1.count_pos + bin2.count_pos;
    merged.count_neg = bin1.count_neg + bin2.count_neg;
    merged.woe = 0.0;
    merged.iv = 0.0;
    return merged;
  }
  
  void optimize_bins() {
    double prev_total_iv = 0.0;
    iterations_run = 0;
    
    while (iterations_run < max_iterations) {
      // Ensure number of bins within min_bins and max_bins
      while (static_cast<int>(bins.size()) > max_bins) {
        merge_adjacent_bins();
      }
      
      // Enforce monotonicity
      enforce_monotonicity();
      
      // Ensure minimum number of bins
      while (static_cast<int>(bins.size()) < min_bins && bins.size() > 1) {
        split_largest_bin();
      }
      
      double total_iv = calculate_total_iv();
      
      if (std::abs(total_iv - prev_total_iv) < convergence_threshold) {
        converged = true;
        break;
      }
      
      prev_total_iv = total_iv;
      iterations_run++;
    }
  }
  
  void merge_adjacent_bins() {
    double min_iv_gain = std::numeric_limits<double>::max();
    size_t merge_idx = 0;
    for (size_t i = 0; i < bins.size() - 1; ++i) {
      double iv_gain = bins[i].iv + bins[i + 1].iv;
      if (iv_gain < min_iv_gain) {
        min_iv_gain = iv_gain;
        merge_idx = i;
      }
    }
    bins[merge_idx] = merge_bins(bins[merge_idx], bins[merge_idx + 1]);
    bins.erase(bins.begin() + merge_idx + 1);
    calculate_woe_iv(bins);
  }
  
  void enforce_monotonicity() {
    bool is_monotonic = false;
    while (!is_monotonic && bins.size() > static_cast<size_t>(min_bins)) {
      is_monotonic = true;
      for (size_t i = 1; i < bins.size(); ++i) {
        if (bins[i].woe < bins[i - 1].woe) {
          bins[i - 1] = merge_bins(bins[i - 1], bins[i]);
          bins.erase(bins.begin() + i);
          is_monotonic = false;
          break;
        }
      }
      calculate_woe_iv(bins);
    }
  }
  
  void split_largest_bin() {
    size_t largest_bin_idx = 0;
    int max_count = 0;
    for (size_t i = 0; i < bins.size(); ++i) {
      if (bins[i].count > max_count) {
        max_count = bins[i].count;
        largest_bin_idx = i;
      }
    }
    
    Bin &largest_bin = bins[largest_bin_idx];
    double mid_point = (largest_bin.lower + largest_bin.upper) / 2.0;
    
    Bin new_bin = largest_bin;
    new_bin.upper = mid_point;
    largest_bin.lower = mid_point;
    
    bins.insert(bins.begin() + largest_bin_idx, new_bin);
    
    redistribute_points(largest_bin_idx);
    calculate_woe_iv(bins);
  }
  
  void redistribute_points(size_t split_bin_idx) {
    Bin &left_bin = bins[split_bin_idx];
    Bin &right_bin = bins[split_bin_idx + 1];
    
    left_bin.count = 0;
    left_bin.count_pos = 0;
    left_bin.count_neg = 0;
    right_bin.count = 0;
    right_bin.count_pos = 0;
    right_bin.count_neg = 0;
    
    for (size_t i = 0; i < feature.size(); ++i) {
      if (feature[i] >= left_bin.lower && feature[i] < left_bin.upper) {
        left_bin.count++;
        if (target[i] == 1) left_bin.count_pos++;
        else left_bin.count_neg++;
      } else if (feature[i] >= right_bin.lower && feature[i] <= right_bin.upper) {
        right_bin.count++;
        if (target[i] == 1) right_bin.count_pos++;
        else right_bin.count_neg++;
      }
    }
  }
  
  double calculate_total_iv() const {
    double total_iv = 0.0;
    for (const auto &bin : bins) {
      total_iv += bin.iv;
    }
    return total_iv;
  }
};

// Implementation of handle_few_unique_values
Rcpp::List handle_few_unique_values(
    const std::vector<double>& feature_vec,
    const std::vector<int>& target_vec
) {
  std::vector<Bin> unique_bins;
  std::unordered_set<double> unique_values_set(feature_vec.begin(), feature_vec.end());
  size_t unique_size = unique_values_set.size();
  
  if (unique_size == 1) {
    // Only one unique value, single bin
    Bin b;
    b.lower = -std::numeric_limits<double>::infinity();
    b.upper = std::numeric_limits<double>::infinity();
    b.count = feature_vec.size();
    b.count_pos = 0;
    b.count_neg = 0;
    for (size_t i = 0; i < feature_vec.size(); ++i) {
      if (target_vec[i] == 1) b.count_pos++;
      else b.count_neg++;
    }
    unique_bins.push_back(b);
  } else if (unique_size == 2) {
    // Two unique values, create two bins with min value as cutpoint
    double min_val = *std::min_element(feature_vec.begin(), feature_vec.end());
    Bin bin1, bin2;
    
    bin1.lower = -std::numeric_limits<double>::infinity();
    bin1.upper = min_val;
    bin1.count = 0;
    bin1.count_pos = 0;
    bin1.count_neg = 0;
    
    bin2.lower = min_val;
    bin2.upper = std::numeric_limits<double>::infinity();
    bin2.count = 0;
    bin2.count_pos = 0;
    bin2.count_neg = 0;
    
    for (size_t i = 0; i < feature_vec.size(); ++i) {
      double val = feature_vec[i];
      if (val <= min_val) {
        bin1.count++;
        if (target_vec[i] == 1) bin1.count_pos++;
        else bin1.count_neg++;
      } else {
        bin2.count++;
        if (target_vec[i] == 1) bin2.count_pos++;
        else bin2.count_neg++;
      }
    }
    
    unique_bins.push_back(bin1);
    unique_bins.push_back(bin2);
  }
  
  // Calculate WOE and IV
  OptimalBinningNumericalUDT temp_obj(feature_vec, target_vec);
  temp_obj.calculate_woe_iv(unique_bins);
  
  // Prepare output
  size_t n_bins = unique_bins.size();
  CharacterVector bin_intervals(n_bins);
  NumericVector woe_values(n_bins), iv_values(n_bins);
  IntegerVector counts(n_bins), counts_pos(n_bins), counts_neg(n_bins);
  NumericVector cutpoints(n_bins - 1);
  
  for (size_t i = 0; i < n_bins; ++i) {
    bin_intervals[i] = temp_obj.create_interval(unique_bins[i].lower, unique_bins[i].upper);
    woe_values[i] = unique_bins[i].woe;
    iv_values[i] = unique_bins[i].iv;
    counts[i] = unique_bins[i].count;
    counts_pos[i] = unique_bins[i].count_pos;
    counts_neg[i] = unique_bins[i].count_neg;
    if (i < n_bins - 1) {
      cutpoints[i] = unique_bins[i].upper;
    }
  }
  
  // Criar vetor de IDs com o mesmo tamanho de bins
  Rcpp::NumericVector ids(bin_intervals.size());
  for(int i = 0; i < bin_intervals.size(); i++) {
    ids[i] = i + 1;
  }
  
  return Rcpp::List::create(
    Rcpp::Named("id") = ids,
    Rcpp::Named("bin") = bin_intervals,
    Rcpp::Named("woe") = woe_values,
    Rcpp::Named("iv") = iv_values,
    Rcpp::Named("count") = counts,
    Rcpp::Named("count_pos") = counts_pos,
    Rcpp::Named("count_neg") = counts_neg,
    Rcpp::Named("cutpoints") = cutpoints,
    Rcpp::Named("converged") = true,
    Rcpp::Named("iterations") = 0
  );
}

//' @title Optimal Binning for Numerical Variables using Unsupervised Decision Trees
//'
//' @description 
//' This function implements an optimal binning algorithm for numerical variables
//' using an Unsupervised Decision Tree (UDT) approach with Weight of Evidence (WoE)
//' and Information Value (IV) criteria.
//'
//' @param target An integer vector of binary target values (0 or 1).
//' @param feature A numeric vector of feature values to be binned.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum frequency of observations in each bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins for initial quantile-based discretization (default: 20).
//' @param convergence_threshold Threshold for convergence of the optimization process (default: 1e-6).
//' @param max_iterations Maximum number of iterations for the optimization process (default: 1000).
//'
//' @return A list containing binning details:
//' \item{bins}{A character vector of bin intervals.}
//' \item{woe}{A numeric vector of Weight of Evidence values for each bin.}
//' \item{iv}{A numeric vector of Information Value for each bin.}
//' \item{count}{An integer vector of total observations in each bin.}
//' \item{count_pos}{An integer vector of positive observations in each bin.}
//' \item{count_neg}{An integer vector of negative observations in each bin.}
//' \item{cutpoints}{A numeric vector of cut points between bins.}
//' \item{converged}{A logical value indicating whether the algorithm converged.}
//' \item{iterations}{An integer value of the number of iterations run.}
//'
//' @details
//' The optimal binning algorithm for numerical variables uses an Unsupervised Decision Tree
//' approach with Weight of Evidence (WoE) and Information Value (IV) to create bins that
//' maximize the predictive power of the feature while maintaining interpretability.
//'
//' The algorithm follows these steps:
//' 1. Initial discretization using quantile-based binning
//' 2. Merging of rare bins based on the bin_cutoff parameter
//' 3. Bin optimization using IV and WoE criteria
//' 4. Enforcement of monotonicity in WoE across bins
//' 5. Adjustment of the number of bins to be within the specified range
//'
//' @examples
//' \dontrun{
//' # Generate sample data
//' set.seed(123)
//' n <- 10000
//' feature <- rnorm(n)
//' target <- rbinom(n, 1, plogis(0.5 * feature))
//'
//' # Apply optimal binning
//' result <- optimal_binning_numerical_udt(target, feature, min_bins = 3, max_bins = 5)
//'
//' # View binning results
//' print(result)
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_udt(
   Rcpp::IntegerVector target,
   Rcpp::NumericVector feature,
   int min_bins = 3,
   int max_bins = 5,
   double bin_cutoff = 0.05,
   int max_n_prebins = 20,
   double convergence_threshold = 1e-6,
   int max_iterations = 1000
) {
 // Input validation
 if (feature.size() == 0) {
   Rcpp::stop("Feature vector is empty.");
 }
 if (target.size() != feature.size()) {
   Rcpp::stop("Target and feature vectors must be of the same length.");
 }
 for (int i = 0; i < target.size(); ++i) {
   if (target[i] != 0 && target[i] != 1) {
     Rcpp::stop("Target vector must contain only binary values 0 and 1.");
   }
 }
 if (min_bins < 2) {
   Rcpp::stop("min_bins must be at least 2.");
 }
 if (max_bins < min_bins) {
   Rcpp::stop("max_bins must be greater than or equal to min_bins.");
 }
 if (bin_cutoff <= 0 || bin_cutoff >= 1) {
   Rcpp::stop("bin_cutoff must be between 0 and 1.");
 }
 if (max_n_prebins < min_bins) {
   Rcpp::stop("max_n_prebins must be greater than or equal to min_bins.");
 }
 
 // Convert inputs to std::vector
 std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
 std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
 
 // Handle features with <=2 unique values
 std::unordered_set<double> unique_values(feature_vec.begin(), feature_vec.end());
 if (unique_values.size() <= 2) {
   return handle_few_unique_values(feature_vec, target_vec);
 }
 
 // Perform optimal binning
 OptimalBinningNumericalUDT obj(feature_vec, target_vec, min_bins, max_bins, 
                                bin_cutoff, max_n_prebins, convergence_threshold, max_iterations);
 obj.fit();
 return obj.get_result();
}

/*  
Comentários das melhorias (pt-BR):

- Utilização de limites infinitos (-Inf e +Inf) para definir intervalos abertos nas extremidades.
- Implementação de pesquisa binária (find_bin_index) para alocação de pontos em bins, reduzindo complexidade.
- Tratamento de casos com <=2 valores únicos de forma especial, garantindo a consistência do resultado.
- Mecanismo de mesclagem de bins raros (menos frequentes que um corte mínimo), reduzindo bins com pouca informação.
- Otimização iterativa dos bins para ajustar tamanho, monotonicidade (WoE crescente) e convergência do IV total.
- Suavização na contagem de positivos e negativos para evitar log(0) ao calcular WOE.
- Verificações de parâmetros de entrada (min_bins, max_bins, bin_cutoff, etc.) assegurando integridade e robustez.
- Convergência baseada em limiar do IV total, impedindo loops infinitos e garantindo término estável.
- Separação em funções distintas (get_quantile_cutpoints, initial_binning, merge_rare_bins, etc.) para maior clareza e manutenção.
- Documentação do código e comentários nas principais etapas, facilitando entendimento e futuras manutenções.
*/  


// #include <Rcpp.h>
// #include <algorithm>
// #include <vector>
// #include <cmath>
// #include <limits>
// #include <string>
// #include <unordered_set>
// #include <sstream>
// 
// using namespace Rcpp;
// 
// // Helper structure for bins
// struct Bin {
//   double lower;
//   double upper;
//   double woe;
//   double iv;
//   int count;
//   int count_pos;
//   int count_neg;
// };
// 
// // Comparator for sorting
// bool compareBins(const Bin &a, const Bin &b) {
//   return a.lower < b.lower;
// }
// 
// // Forward declaration of handle_few_unique_values
// Rcpp::List handle_few_unique_values(
//     const std::vector<double>& feature_vec,
//     const std::vector<int>& target_vec
// );
// 
// // Class for Optimal Binning
// class OptimalBinningNumericalUDT {
// private:
//   std::vector<double> feature;
//   std::vector<int> target;
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
//   std::vector<Bin> bins;
//   double convergence_threshold;
//   int max_iterations;
//   bool converged;
//   int iterations_run;
//   
// public:
//   OptimalBinningNumericalUDT(std::vector<double> feature,
//                              std::vector<int> target,
//                              int min_bins = 3,
//                              int max_bins = 5,
//                              double bin_cutoff = 0.05,
//                              int max_n_prebins = 20,
//                              double convergence_threshold = 1e-6,
//                              int max_iterations = 1000) 
//     : feature(std::move(feature)), 
//       target(std::move(target)), 
//       min_bins(std::max(2, min_bins)),
//       max_bins(std::max(this->min_bins, max_bins)),
//       bin_cutoff(bin_cutoff),
//       max_n_prebins(max_n_prebins),
//       convergence_threshold(convergence_threshold),
//       max_iterations(max_iterations),
//       converged(false),
//       iterations_run(0) {}
//   
//   void fit() {
//     // Pre-binning using quantiles
//     std::vector<double> cut_points = get_quantile_cutpoints(feature, max_n_prebins);
//     
//     // Initial binning
//     bins = initial_binning(feature, target, cut_points);
//     
//     // Rare bin merging
//     merge_rare_bins();
//     
//     // Bin optimization
//     optimize_bins();
//   }
//   
//   Rcpp::List get_result() const {
//     size_t n_bins = bins.size();
//     CharacterVector bin_intervals(n_bins);
//     NumericVector woe_values(n_bins), iv_values(n_bins);
//     IntegerVector counts(n_bins), counts_pos(n_bins), counts_neg(n_bins);
//     NumericVector cutpoints(n_bins - 1);
//     
//     for (size_t i = 0; i < n_bins; ++i) {
//       bin_intervals[i] = create_interval(bins[i].lower, bins[i].upper);
//       woe_values[i] = bins[i].woe;
//       iv_values[i] = bins[i].iv;
//       counts[i] = bins[i].count;
//       counts_pos[i] = bins[i].count_pos;
//       counts_neg[i] = bins[i].count_neg;
//       if (i < n_bins - 1) {
//         cutpoints[i] = bins[i].upper;
//       }
//     }
//     
//     return Rcpp::List::create(
//       Rcpp::Named("bin") = bin_intervals,
//       Rcpp::Named("woe") = woe_values,
//       Rcpp::Named("iv") = iv_values,
//       Rcpp::Named("count") = counts,
//       Rcpp::Named("count_pos") = counts_pos,
//       Rcpp::Named("count_neg") = counts_neg,
//       Rcpp::Named("cutpoints") = cutpoints,
//       Rcpp::Named("converged") = converged,
//       Rcpp::Named("iterations") = iterations_run
//     );
//   }
//   
//   void calculate_woe_iv(std::vector<Bin> &bins) const {
//     int total_pos = 0, total_neg = 0;
//     for (const auto &bin : bins) {
//       total_pos += bin.count_pos;
//       total_neg += bin.count_neg;
//     }
//     
//     double total_pos_d = total_pos > 0 ? static_cast<double>(total_pos) : 1.0;
//     double total_neg_d = total_neg > 0 ? static_cast<double>(total_neg) : 1.0;
//     
//     for (auto &bin : bins) {
//       double dist_pos = static_cast<double>(bin.count_pos) / total_pos_d;
//       double dist_neg = static_cast<double>(bin.count_neg) / total_neg_d;
//       if (dist_pos == 0.0) dist_pos = 0.0001;
//       if (dist_neg == 0.0) dist_neg = 0.0001;
//       bin.woe = std::log(dist_pos / dist_neg);
//       bin.iv = (dist_pos - dist_neg) * bin.woe;
//     }
//   }
//   
//   std::string create_interval(double lower, double upper) const {
//     std::stringstream ss;
//     ss << "(";
//     if (std::isinf(lower))
//       ss << "-Inf";
//     else
//       ss << lower;
//     ss << ";";
//     if (std::isinf(upper))
//       ss << "+Inf";
//     else
//       ss << upper;
//     ss << "]";
//     return ss.str();
//   }
//   
// private:
//   std::vector<double> get_quantile_cutpoints(const std::vector<double> &data, int n_bins) const {
//     std::vector<double> quantiles;
//     if (n_bins <= 1) {
//       return quantiles;
//     }
//     
//     std::vector<double> sorted_data = data;
//     sorted_data.erase(std::remove_if(sorted_data.begin(), sorted_data.end(),
//                                      [](double x) { return std::isnan(x); }), sorted_data.end());
//     
//     if (sorted_data.empty()) {
//       return quantiles;
//     }
//     
//     std::sort(sorted_data.begin(), sorted_data.end());
//     
//     for (int i = 1; i < n_bins; ++i) {
//       double p = static_cast<double>(i) / n_bins;
//       double pos = p * (sorted_data.size() - 1);
//       size_t idx = static_cast<size_t>(std::floor(pos));
//       double frac = pos - static_cast<double>(idx);
//       double quantile;
//       if (idx + 1 < sorted_data.size()) {
//         quantile = sorted_data[idx] + frac * (sorted_data[idx + 1] - sorted_data[idx]);
//       } else {
//         quantile = sorted_data[idx];
//       }
//       quantiles.push_back(quantile);
//     }
//     
//     std::vector<double> unique_quantiles;
//     std::unordered_set<double> seen;
//     for (double q : quantiles) {
//       if (seen.find(q) == seen.end()) {
//         unique_quantiles.push_back(q);
//         seen.insert(q);
//       }
//     }
//     
//     return unique_quantiles;
//   }
//   
//   std::vector<Bin> initial_binning(const std::vector<double> &feature,
//                                    const std::vector<int> &target,
//                                    const std::vector<double> &cut_points) const {
//     std::vector<Bin> initial_bins;
//     std::vector<double> boundaries = cut_points;
//     boundaries.insert(boundaries.begin(), -std::numeric_limits<double>::infinity());
//     boundaries.push_back(std::numeric_limits<double>::infinity());
//     size_t n_bins = boundaries.size() - 1;
//     
//     initial_bins.resize(n_bins);
//     for (size_t i = 0; i < n_bins; ++i) {
//       initial_bins[i].lower = boundaries[i];
//       initial_bins[i].upper = boundaries[i + 1];
//       initial_bins[i].count = 0;
//       initial_bins[i].count_pos = 0;
//       initial_bins[i].count_neg = 0;
//       initial_bins[i].woe = 0.0;
//       initial_bins[i].iv = 0.0;
//     }
//     
//     for (size_t i = 0; i < feature.size(); ++i) {
//       double val = feature[i];
//       int bin_idx = find_bin_index(val, boundaries);
//       if (bin_idx >= 0 && static_cast<size_t>(bin_idx) < n_bins) {
//         initial_bins[bin_idx].count++;
//         if (target[i] == 1) {
//           initial_bins[bin_idx].count_pos++;
//         } else {
//           initial_bins[bin_idx].count_neg++;
//         }
//       }
//     }
//     
//     calculate_woe_iv(initial_bins);
//     
//     return initial_bins;
//   }
//   
//   int find_bin_index(double value, const std::vector<double> &boundaries) const {
//     int left = 0;
//     int right = boundaries.size() - 1;
//     while (left < right - 1) {
//       int mid = left + (right - left) / 2;
//       if (value <= boundaries[mid]) {
//         right = mid;
//       } else {
//         left = mid;
//       }
//     }
//     return left;
//   }
//   
//   void merge_rare_bins() {
//     size_t total_count = feature.size();
//     double cutoff_count = bin_cutoff * static_cast<double>(total_count);
//     bool merged = false;
//     
//     do {
//       merged = false;
//       for (size_t i = 0; i < bins.size(); ++i) {
//         if (static_cast<double>(bins[i].count) < cutoff_count && bins.size() > static_cast<size_t>(min_bins)) {
//           if (i > 0) {
//             bins[i - 1] = merge_bins(bins[i - 1], bins[i]);
//             bins.erase(bins.begin() + i);
//           } else if (i < bins.size() - 1) {
//             bins[i + 1] = merge_bins(bins[i], bins[i + 1]);
//             bins.erase(bins.begin() + i);
//           }
//           merged = true;
//           break;
//         }
//       }
//     } while (merged && bins.size() > static_cast<size_t>(min_bins));
//     
//     calculate_woe_iv(bins);
//   }
//   
//   Bin merge_bins(const Bin &bin1, const Bin &bin2) const {
//     Bin merged;
//     merged.lower = bin1.lower;
//     merged.upper = bin2.upper;
//     merged.count = bin1.count + bin2.count;
//     merged.count_pos = bin1.count_pos + bin2.count_pos;
//     merged.count_neg = bin1.count_neg + bin2.count_neg;
//     merged.woe = 0.0;
//     merged.iv = 0.0;
//     return merged;
//   }
//   
//   void optimize_bins() {
//     double prev_total_iv = 0.0;
//     iterations_run = 0;
//     
//     while (iterations_run < max_iterations) {
//       // Ensure number of bins within min_bins and max_bins
//       while (static_cast<int>(bins.size()) > max_bins) {
//         merge_adjacent_bins();
//       }
//       
//       // Enforce monotonicity
//       enforce_monotonicity();
//       
//       // Ensure minimum number of bins
//       while (static_cast<int>(bins.size()) < min_bins && bins.size() > 1) {
//         split_largest_bin();
//       }
//       
//       double total_iv = calculate_total_iv();
//       
//       if (std::abs(total_iv - prev_total_iv) < convergence_threshold) {
//         converged = true;
//         break;
//       }
//       
//       prev_total_iv = total_iv;
//       iterations_run++;
//     }
//   }
//   
//   void merge_adjacent_bins() {
//     double min_iv_gain = std::numeric_limits<double>::max();
//     size_t merge_idx = 0;
//     for (size_t i = 0; i < bins.size() - 1; ++i) {
//       double iv_gain = bins[i].iv + bins[i + 1].iv;
//       if (iv_gain < min_iv_gain) {
//         min_iv_gain = iv_gain;
//         merge_idx = i;
//       }
//     }
//     bins[merge_idx] = merge_bins(bins[merge_idx], bins[merge_idx + 1]);
//     bins.erase(bins.begin() + merge_idx + 1);
//     calculate_woe_iv(bins);
//   }
//   
//   void enforce_monotonicity() {
//     bool is_monotonic = false;
//     while (!is_monotonic && bins.size() > static_cast<size_t>(min_bins)) {
//       is_monotonic = true;
//       for (size_t i = 1; i < bins.size(); ++i) {
//         if (bins[i].woe < bins[i - 1].woe) {
//           bins[i - 1] = merge_bins(bins[i - 1], bins[i]);
//           bins.erase(bins.begin() + i);
//           is_monotonic = false;
//           break;
//         }
//       }
//       calculate_woe_iv(bins);
//     }
//   }
//   
//   void split_largest_bin() {
//     size_t largest_bin_idx = 0;
//     int max_count = 0;
//     for (size_t i = 0; i < bins.size(); ++i) {
//       if (bins[i].count > max_count) {
//         max_count = bins[i].count;
//         largest_bin_idx = i;
//       }
//     }
//     
//     Bin &largest_bin = bins[largest_bin_idx];
//     double mid_point = (largest_bin.lower + largest_bin.upper) / 2.0;
//     
//     Bin new_bin = largest_bin;
//     new_bin.upper = mid_point;
//     largest_bin.lower = mid_point;
//     
//     bins.insert(bins.begin() + largest_bin_idx, new_bin);
//     
//     redistribute_points(largest_bin_idx);
//     calculate_woe_iv(bins);
//   }
//   
//   void redistribute_points(size_t split_bin_idx) {
//     Bin &left_bin = bins[split_bin_idx];
//     Bin &right_bin = bins[split_bin_idx + 1];
//     
//     left_bin.count = 0;
//     left_bin.count_pos = 0;
//     left_bin.count_neg = 0;
//     right_bin.count = 0;
//     right_bin.count_pos = 0;
//     right_bin.count_neg = 0;
//     
//     for (size_t i = 0; i < feature.size(); ++i) {
//       if (feature[i] >= left_bin.lower && feature[i] < left_bin.upper) {
//         left_bin.count++;
//         if (target[i] == 1) left_bin.count_pos++;
//         else left_bin.count_neg++;
//       } else if (feature[i] >= right_bin.lower && feature[i] <= right_bin.upper) {
//         right_bin.count++;
//         if (target[i] == 1) right_bin.count_pos++;
//         else right_bin.count_neg++;
//       }
//     }
//   }
//   
//   double calculate_total_iv() const {
//     double total_iv = 0.0;
//     for (const auto &bin : bins) {
//       total_iv += bin.iv;
//     }
//     return total_iv;
//   }
// };
// 
// // Implementation of handle_few_unique_values
// Rcpp::List handle_few_unique_values(
//     const std::vector<double>& feature_vec,
//     const std::vector<int>& target_vec
// ) {
//   std::vector<Bin> unique_bins;
//   std::unordered_set<double> unique_values_set(feature_vec.begin(), feature_vec.end());
//   size_t unique_size = unique_values_set.size();
//   
//   if (unique_size == 1) {
//     // Only one unique value, single bin
//     Bin b;
//     b.lower = -std::numeric_limits<double>::infinity();
//     b.upper = std::numeric_limits<double>::infinity();
//     b.count = feature_vec.size();
//     b.count_pos = 0;
//     b.count_neg = 0;
//     for (size_t i = 0; i < feature_vec.size(); ++i) {
//       if (target_vec[i] == 1) b.count_pos++;
//       else b.count_neg++;
//     }
//     unique_bins.push_back(b);
//   } else if (unique_size == 2) {
//     // Two unique values, create two bins with min value as cutpoint
//     double min_val = *std::min_element(feature_vec.begin(), feature_vec.end());
//     Bin bin1, bin2;
//     
//     bin1.lower = -std::numeric_limits<double>::infinity();
//     bin1.upper = min_val;
//     bin1.count = 0;
//     bin1.count_pos = 0;
//     bin1.count_neg = 0;
//     
//     bin2.lower = min_val;
//     bin2.upper = std::numeric_limits<double>::infinity();
//     bin2.count = 0;
//     bin2.count_pos = 0;
//     bin2.count_neg = 0;
//     
//     for (size_t i = 0; i < feature_vec.size(); ++i) {
//       double val = feature_vec[i];
//       if (val <= min_val) {
//         bin1.count++;
//         if (target_vec[i] == 1) bin1.count_pos++;
//         else bin1.count_neg++;
//       } else {
//         bin2.count++;
//         if (target_vec[i] == 1) bin2.count_pos++;
//         else bin2.count_neg++;
//       }
//     }
//     
//     unique_bins.push_back(bin1);
//     unique_bins.push_back(bin2);
//   }
//   
//   // Calculate WOE and IV
//   OptimalBinningNumericalUDT temp_obj(feature_vec, target_vec);
//   temp_obj.calculate_woe_iv(unique_bins);
//   
//   // Prepare output
//   size_t n_bins = unique_bins.size();
//   CharacterVector bin_intervals(n_bins);
//   NumericVector woe_values(n_bins), iv_values(n_bins);
//   IntegerVector counts(n_bins), counts_pos(n_bins), counts_neg(n_bins);
//   NumericVector cutpoints(n_bins - 1);
//   
//   for (size_t i = 0; i < n_bins; ++i) {
//     bin_intervals[i] = temp_obj.create_interval(unique_bins[i].lower, unique_bins[i].upper);
//     woe_values[i] = unique_bins[i].woe;
//     iv_values[i] = unique_bins[i].iv;
//     counts[i] = unique_bins[i].count;
//     counts_pos[i] = unique_bins[i].count_pos;
//     counts_neg[i] = unique_bins[i].count_neg;
//     if (i < n_bins - 1) {
//       cutpoints[i] = unique_bins[i].upper;
//     }
//   }
//   
//   return Rcpp::List::create(
//     Rcpp::Named("bin") = bin_intervals,
//     Rcpp::Named("woe") = woe_values,
//     Rcpp::Named("iv") = iv_values,
//     Rcpp::Named("count") = counts,
//     Rcpp::Named("count_pos") = counts_pos,
//     Rcpp::Named("count_neg") = counts_neg,
//     Rcpp::Named("cutpoints") = cutpoints,
//     Rcpp::Named("converged") = true,
//     Rcpp::Named("iterations") = 0
//   );
// }
// 
// //' Optimal Binning for Numerical Variables using Unsupervised Decision Trees
// //'
// //' This function implements an optimal binning algorithm for numerical variables
// //' using an Unsupervised Decision Tree (UDT) approach with Weight of Evidence (WoE)
// //' and Information Value (IV) criteria.
// //'
// //' @param target An integer vector of binary target values (0 or 1).
// //' @param feature A numeric vector of feature values to be binned.
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff Minimum frequency of observations in each bin (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins for initial quantile-based discretization (default: 20).
// //' @param convergence_threshold Threshold for convergence of the optimization process (default: 1e-6).
// //' @param max_iterations Maximum number of iterations for the optimization process (default: 1000).
// //'
// //' @return A list containing binning details:
// //' \item{bins}{A character vector of bin intervals.}
// //' \item{woe}{A numeric vector of Weight of Evidence values for each bin.}
// //' \item{iv}{A numeric vector of Information Value for each bin.}
// //' \item{count}{An integer vector of total observations in each bin.}
// //' \item{count_pos}{An integer vector of positive observations in each bin.}
// //' \item{count_neg}{An integer vector of negative observations in each bin.}
// //' \item{cutpoints}{A numeric vector of cut points between bins.}
// //' \item{converged}{A logical value indicating whether the algorithm converged.}
// //' \item{iterations}{An integer value of the number of iterations run.}
// //'
// //' @details
// //' The optimal binning algorithm for numerical variables uses an Unsupervised Decision Tree
// //' approach with Weight of Evidence (WoE) and Information Value (IV) to create bins that
// //' maximize the predictive power of the feature while maintaining interpretability.
// //'
// //' The algorithm follows these steps:
// //' 1. Initial discretization using quantile-based binning
// //' 2. Merging of rare bins based on the bin_cutoff parameter
// //' 3. Bin optimization using IV and WoE criteria
// //' 4. Enforcement of monotonicity in WoE across bins
// //' 5. Adjustment of the number of bins to be within the specified range
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
// //' result <- optimal_binning_numerical_udt(target, feature, min_bins = 3, max_bins = 5)
// //'
// //' # View binning results
// //' print(result)
// //' }
// //'
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_numerical_udt(
//     Rcpp::IntegerVector target,
//     Rcpp::NumericVector feature,
//     int min_bins = 3,
//     int max_bins = 5,
//     double bin_cutoff = 0.05,
//     int max_n_prebins = 20,
//     double convergence_threshold = 1e-6,
//     int max_iterations = 1000
// ) {
//   // Input validation
//   if (feature.size() == 0) {
//     Rcpp::stop("Feature vector is empty.");
//   }
//   if (target.size() != feature.size()) {
//     Rcpp::stop("Target and feature vectors must be of the same length.");
//   }
//   for (int i = 0; i < target.size(); ++i) {
//     if (target[i] != 0 && target[i] != 1) {
//       Rcpp::stop("Target vector must contain only binary values 0 and 1.");
//     }
//   }
//   if (min_bins < 2) {
//     Rcpp::stop("min_bins must be at least 2.");
//   }
//   if (max_bins < min_bins) {
//     Rcpp::stop("max_bins must be greater than or equal to min_bins.");
//   }
//   if (bin_cutoff <= 0 || bin_cutoff >= 1) {
//     Rcpp::stop("bin_cutoff must be between 0 and 1.");
//   }
//   if (max_n_prebins < min_bins) {
//     Rcpp::stop("max_n_prebins must be greater than or equal to min_bins.");
//   }
//   
//   // Convert inputs to std::vector
//   std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
//   std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
//   
//   // Handle features with <=2 unique values
//   std::unordered_set<double> unique_values(feature_vec.begin(), feature_vec.end());
//   if (unique_values.size() <= 2) {
//     return handle_few_unique_values(feature_vec, target_vec);
//   }
//   
//   // Perform optimal binning
//   OptimalBinningNumericalUDT obj(feature_vec, target_vec, min_bins, max_bins, 
//                                  bin_cutoff, max_n_prebins, convergence_threshold, max_iterations);
//   obj.fit();
//   return obj.get_result();
// }
