// [[Rcpp::plugins(cpp11)]]

#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <stdexcept>

using namespace Rcpp;

class OptimalBinningNumericalOSLP {
private:
  std::vector<double> feature;
  std::vector<double> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  double convergence_threshold;
  int max_iterations;
  
  std::vector<double> bin_edges;
  std::vector<std::string> bin_labels;
  std::vector<double> woe_values;
  std::vector<double> iv_values;
  std::vector<int> count_values;
  std::vector<int> count_pos_values;
  std::vector<int> count_neg_values;
  
  double total_iv;
  bool converged;
  int iterations_run;
  
public:
  OptimalBinningNumericalOSLP(const std::vector<double>& feature,
                              const std::vector<double>& target,
                              int min_bins = 3,
                              int max_bins = 5,
                              double bin_cutoff = 0.05,
                              int max_n_prebins = 20,
                              double convergence_threshold = 1e-6,
                              int max_iterations = 1000)
    : feature(feature), target(target),
      min_bins(std::max(min_bins, 2)),
      max_bins(std::max(max_bins, min_bins)),
      bin_cutoff(bin_cutoff),
      max_n_prebins(std::max(max_n_prebins, min_bins)),
      convergence_threshold(convergence_threshold),
      max_iterations(max_iterations),
      total_iv(0.0),
      converged(false),
      iterations_run(0) {
    
    if (feature.size() != target.size()) {
      throw std::invalid_argument("Feature and target must have the same length.");
    }
    if (bin_cutoff <= 0 || bin_cutoff >= 1) {
      throw std::invalid_argument("bin_cutoff must be between 0 and 1.");
    }
    if (convergence_threshold <= 0) {
      throw std::invalid_argument("convergence_threshold must be positive.");
    }
    if (max_iterations <= 0) {
      throw std::invalid_argument("max_iterations must be positive.");
    }
    
    // Check target is binary
    bool has_zero = false, has_one = false;
    for (double t : target) {
      if (t == 0) has_zero = true;
      else if (t == 1) has_one = true;
      else throw std::invalid_argument("Target must contain only 0 and 1.");
      if (has_zero && has_one) break;
    }
    if (!has_zero || !has_one) {
      throw std::invalid_argument("Target must contain both classes.");
    }
    
    for (double f : feature) {
      if (std::isnan(f) || std::isinf(f)) {
        throw std::invalid_argument("Feature contains NaN or Inf.");
      }
    }
  }
  
  Rcpp::List fit() {
    // Determine unique values
    std::vector<double> unique_vals = feature;
    std::sort(unique_vals.begin(), unique_vals.end());
    unique_vals.erase(std::unique(unique_vals.begin(), unique_vals.end()), unique_vals.end());
    
    if (unique_vals.size() <= 2) {
      handle_low_unique_values(unique_vals);
      converged = true;
      iterations_run = 0;
      return create_output();
    }
    
    prebin_data(unique_vals);
    merge_small_bins();
    enforce_monotonicity();
    
    // Final calculations
    calculate_bins();
    return create_output();
  }
  
private:
  void handle_low_unique_values(const std::vector<double>& unique_vals) {
    bin_edges.clear();
    if (unique_vals.size() == 1) {
      bin_edges.push_back(-std::numeric_limits<double>::infinity());
      bin_edges.push_back(std::numeric_limits<double>::infinity());
    } else { 
      double val = unique_vals[0];
      bin_edges.push_back(-std::numeric_limits<double>::infinity());
      bin_edges.push_back(val);
      bin_edges.push_back(std::numeric_limits<double>::infinity());
    }
    calculate_bins();
  }
  
  void prebin_data(const std::vector<double>& unique_vals) {
    int n_unique = (int)unique_vals.size();
    int n_pre = std::min(max_n_prebins, n_unique);
    n_pre = std::max(n_pre, min_bins);
    
    bin_edges.clear();
    bin_edges.push_back(-std::numeric_limits<double>::infinity());
    
    // Use approximate quantiles
    for (int i = 1; i < n_pre; ++i) {
      int idx = (int)((i / (double)n_pre) * (n_unique - 1));
      double edge = unique_vals[idx];
      if (edge > bin_edges.back()) {
        bin_edges.push_back(edge);
      }
    }
    bin_edges.push_back(std::numeric_limits<double>::infinity());
    
    calculate_bins();
  }
  
  void calculate_bins() {
    int n_bins = (int)bin_edges.size() - 1;
    woe_values.assign(n_bins, 0.0);
    iv_values.assign(n_bins, 0.0);
    count_values.assign(n_bins, 0);
    count_pos_values.assign(n_bins, 0);
    count_neg_values.assign(n_bins, 0);
    
    double total_pos = 0.0;
    double total_neg = 0.0;
    
    for (size_t i = 0; i < feature.size(); ++i) {
      double fval = feature[i];
      double tval = target[i];
      int idx = find_bin(fval);
      if (idx >= 0 && idx < n_bins) {
        count_values[idx]++;
        if (tval == 1) count_pos_values[idx]++; else count_neg_values[idx]++;
      }
    }
    
    total_pos = std::accumulate(count_pos_values.begin(), count_pos_values.end(), 0.0);
    total_neg = std::accumulate(count_neg_values.begin(), count_neg_values.end(), 0.0);
    
    // Compute WoE and IV
    for (int i = 0; i < n_bins; ++i) {
      double p = (count_pos_values[i] > 0) ? (double)count_pos_values[i]/total_pos : 1e-10;
      double q = (count_neg_values[i] > 0) ? (double)count_neg_values[i]/total_neg : 1e-10;
      p = std::max(p, 1e-10);
      q = std::max(q, 1e-10);
      double woe = std::log(p/q);
      double iv = (p - q)*woe;
      woe_values[i] = woe;
      iv_values[i] = iv;
    }
    total_iv = std::accumulate(iv_values.begin(), iv_values.end(), 0.0);
    update_bin_labels();
  }
  
  void merge_small_bins() {
    bool merged = true;
    int total_count = (int)feature.size();
    iterations_run = 0;
    while (merged && (int)bin_edges.size()-1 > min_bins && iterations_run < max_iterations) {
      merged = false;
      for (size_t i = 0; i < count_values.size(); i++) {
        double prop = (double)count_values[i]/total_count;
        if (prop < bin_cutoff && count_values.size() > (size_t)min_bins) {
          // Merge with neighbor
          if (i == 0) {
            merge_bins(0, 1);
          } else if (i == count_values.size() - 1) {
            merge_bins(count_values.size()-2, count_values.size()-1);
          } else {
            // Merge with the smaller neighbor
            if (count_values[i-1] <= count_values[i+1]) {
              merge_bins(i-1, i);
            } else {
              merge_bins(i, i+1);
            }
          }
          calculate_bins();
          merged = true;
          break;
        }
      }
      iterations_run++;
    }
  }
  
  void enforce_monotonicity() {
    bool increasing = guess_trend();
    
    double prev_iv = total_iv;
    while (!is_monotonic(increasing) && (int)bin_edges.size()-1 > min_bins && iterations_run < max_iterations) {
      for (size_t i = 1; i < woe_values.size(); i++) {
        if ((increasing && woe_values[i] < woe_values[i-1]) ||
            (!increasing && woe_values[i] > woe_values[i-1])) {
          // Merge bins i-1 and i
          merge_bins(i-1, i);
          break;
        }
      }
      calculate_bins();
      double diff = std::fabs(total_iv - prev_iv);
      if (diff < convergence_threshold) {
        converged = true;
        break;
      }
      prev_iv = total_iv;
      iterations_run++;
    }
    
    // If too many bins, merge by minimal IV difference
    while ((int)bin_edges.size()-1 > max_bins && iterations_run < max_iterations) {
      size_t idx = find_min_iv_merge();
      merge_bins(idx, idx+1);
      calculate_bins();
      iterations_run++;
    }
  }
  
  bool guess_trend() {
    int inc = 0, dec = 0;
    for (size_t i = 1; i < woe_values.size(); i++) {
      if (woe_values[i] > woe_values[i-1]) inc++;
      else if (woe_values[i] < woe_values[i-1]) dec++;
    }
    return inc >= dec;
  }
  
  bool is_monotonic(bool increasing) {
    for (size_t i = 1; i < woe_values.size(); i++) {
      if (increasing && woe_values[i] < woe_values[i-1]) return false;
      if (!increasing && woe_values[i] > woe_values[i-1]) return false;
    }
    return true;
  }
  
  size_t find_min_iv_merge() {
    double min_iv_sum = std::numeric_limits<double>::max();
    size_t idx = 0;
    for (size_t i = 0; i < iv_values.size() - 1; i++) {
      double iv_sum = iv_values[i] + iv_values[i+1];
      if (iv_sum < min_iv_sum) {
        min_iv_sum = iv_sum;
        idx = i;
      }
    }
    return idx;
  }
  
  void merge_bins(size_t i, size_t j) {
    if (i > j) std::swap(i,j);
    if (j >= bin_edges.size()-1) return;
    
    bin_edges.erase(bin_edges.begin() + j);
  }
  
  int find_bin(double val) const {
    auto it = std::upper_bound(bin_edges.begin(), bin_edges.end(), val);
    int idx = (int)std::distance(bin_edges.begin(), it) - 1;
    if (idx < 0) idx = 0;
    if (idx >= (int)bin_edges.size()-1) idx = (int)bin_edges.size()-2;
    return idx;
  }
  
  void update_bin_labels() {
    bin_labels.clear();
    for (size_t i = 0; i < bin_edges.size() - 1; i++) {
      std::ostringstream oss;
      oss << "(" << edge_to_str(bin_edges[i]) << ";" << edge_to_str(bin_edges[i+1]) << "]";
      bin_labels.push_back(oss.str());
    }
  }
  
  std::string edge_to_str(double val) {
    if (std::isinf(val)) {
      return val < 0 ? "-Inf" : "+Inf";
    } else {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(6) << val;
      return oss.str();
    }
  }
  
  Rcpp::List create_output() {
    std::vector<double> cutpoints(bin_edges.begin()+1, bin_edges.end()-1);
    return Rcpp::List::create(
      Named("bin") = bin_labels,
      Named("woe") = woe_values,
      Named("iv") = iv_values,
      Named("count") = count_values,
      Named("count_pos") = count_pos_values,
      Named("count_neg") = count_neg_values,
      Named("cutpoints") = cutpoints,
      Named("converged") = converged,
      Named("iterations") = iterations_run
    );
  }
};

//' @title Optimal Binning for Numerical Variables using OSLP
//'
//' @description
//' Performs optimal binning for numerical variables using the Optimal
//' Supervised Learning Partitioning (OSLP) approach.
//'
//' @param target A numeric vector of binary target values (0 or 1).
//' @param feature A numeric vector of feature values.
//' @param min_bins Minimum number of bins (default: 3, must be >= 2).
//' @param max_bins Maximum number of bins (default: 5, must be > min_bins).
//' @param bin_cutoff Minimum proportion of total observations for a bin
//'   to avoid being merged (default: 0.05, must be in (0, 1)).
//' @param max_n_prebins Maximum number of pre-bins before optimization
//'   (default: 20).
//' @param convergence_threshold Threshold for convergence (default: 1e-6).
//' @param max_iterations Maximum number of iterations (default: 1000).
//'
//' @return A list containing:
//' \item{bins}{Character vector of bin labels.}
//' \item{woe}{Numeric vector of Weight of Evidence (WoE) values for each bin.}
//' \item{iv}{Numeric vector of Information Value (IV) for each bin.}
//' \item{count}{Integer vector of total count of observations in each bin.}
//' \item{count_pos}{Integer vector of positive class count in each bin.}
//' \item{count_neg}{Integer vector of negative class count in each bin.}
//' \item{cutpoints}{Numeric vector of cutpoints used to create the bins.}
//' \item{converged}{Logical value indicating whether the algorithm converged.}
//' \item{iterations}{Integer value indicating the number of iterations run.}
//'
//' @examples
//' \dontrun{
//' # Sample data
//' set.seed(123)
//' n <- 1000
//' target <- sample(0:1, n, replace = TRUE)
//' feature <- rnorm(n)
//'
//' # Optimal binning
//' result <- optimal_binning_numerical_oslp(target, feature,
//'                                          min_bins = 2, max_bins = 4)
//'
//' # Print results
//' print(result)
//' }
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_oslp(Rcpp::NumericVector target,
                                          Rcpp::NumericVector feature,
                                          int min_bins = 3,
                                          int max_bins = 5,
                                          double bin_cutoff = 0.05,
                                          int max_n_prebins = 20,
                                          double convergence_threshold = 1e-6,
                                          int max_iterations = 1000) {
  // Validate inputs
  if (feature.size() != target.size()) {
    Rcpp::stop("Feature and target must have the same length.");
  }
  if (min_bins < 2) {
    Rcpp::stop("min_bins must be at least 2.");
  }
  if (max_bins < min_bins) {
    Rcpp::stop("max_bins must be greater or equal to min_bins.");
  }
  if (bin_cutoff <= 0 || bin_cutoff >= 1) {
    Rcpp::stop("bin_cutoff must be between 0 and 1.");
  }
  if (max_n_prebins < min_bins) {
    Rcpp::stop("max_n_prebins must be at least min_bins.");
  }
  if (convergence_threshold <= 0) {
    Rcpp::stop("convergence_threshold must be positive.");
  }
  if (max_iterations <= 0) {
    Rcpp::stop("max_iterations must be positive.");
  }
  
  std::vector<double> feature_vec(feature.begin(), feature.end());
  std::vector<double> target_vec(target.begin(), target.end());
  
  try {
    OptimalBinningNumericalOSLP binning(feature_vec, target_vec,
                                        min_bins, max_bins,
                                        bin_cutoff, max_n_prebins,
                                        convergence_threshold, max_iterations);
    return binning.fit();
  } catch (const std::exception &e) {
    Rcpp::stop("Error in optimal_binning_numerical_oslp: " + std::string(e.what()));
  }
}

/*
Resumo das melhorias:
- Verificações de entrada mais robustas.
- Tratamento cuidadoso de casos com poucos valores únicos, evitando erro.
- Uso de EPSILON para evitar log(0).
- Garantia de não exceder min_bins ou max_bins.
- Detecção e mesclagem de bins não-monotônicos.
- Detecção de convergência baseada em variação de IV.
- Comentários em inglês e código estruturado.
- Preservação de nomes e tipos de input/output finais.
*/


// // [[Rcpp::plugins(cpp11)]]
// 
// #include <Rcpp.h>
// #include <algorithm>
// #include <vector>
// #include <cmath>
// #include <limits>
// #include <numeric>
// #include <sstream>
// #include <iomanip>
// #include <stdexcept>
// 
// using namespace Rcpp;
// 
// class OptimalBinningNumericalOSLP {
// private:
//   std::vector<double> feature;
//   std::vector<double> target;
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
//   std::vector<double> bin_edges;
//   std::vector<std::string> bin_labels;
//   std::vector<double> woe_values;
//   std::vector<double> iv_values;
//   std::vector<int> count_values;
//   std::vector<int> count_pos_values;
//   std::vector<int> count_neg_values;
//   double total_iv;
//   bool is_monotonic_increasing;
//   double convergence_threshold;
//   int max_iterations;
//   bool converged;
//   int iterations_run;
//   
// public:
//   OptimalBinningNumericalOSLP(const std::vector<double>& feature,
//                               const std::vector<double>& target,
//                               int min_bins = 3,
//                               int max_bins = 5,
//                               double bin_cutoff = 0.05,
//                               int max_n_prebins = 20,
//                               double convergence_threshold = 1e-6,
//                               int max_iterations = 1000) {
//     this->feature = feature;
//     this->target = target;
//     this->min_bins = std::max(min_bins, 2);
//     this->max_bins = std::max(max_bins, this->min_bins);
//     this->bin_cutoff = bin_cutoff;
//     this->max_n_prebins = max_n_prebins;
//     this->total_iv = 0.0;
//     this->is_monotonic_increasing = true;
//     this->convergence_threshold = convergence_threshold;
//     this->max_iterations = max_iterations;
//     this->converged = false;
//     this->iterations_run = 0;
//   }
//   
//   Rcpp::List fit() {
//     try {
//       if (feature.size() != target.size()) {
//         throw std::runtime_error("Feature and target vectors must have the same length.");
//       }
//       
//       // Get unique sorted feature values
//       std::vector<double> unique_values = feature;
//       std::sort(unique_values.begin(), unique_values.end());
//       unique_values.erase(std::unique(unique_values.begin(), unique_values.end()), unique_values.end());
//       
//       // If number of unique values is less than or equal to 2, do not optimize or create extra bins
//       if (unique_values.size() <= 2) {
//         bin_edges.clear();
//         if (unique_values.size() == 1) {
//           // All values are the same, single bin
//           bin_edges.push_back(-std::numeric_limits<double>::infinity());
//           bin_edges.push_back(std::numeric_limits<double>::infinity());
//         } else {
//           // Two unique values, use the minimum value as the cutpoint
//           double min_val = unique_values[0];
//           bin_edges.push_back(-std::numeric_limits<double>::infinity());
//           bin_edges.push_back(min_val);
//           bin_edges.push_back(std::numeric_limits<double>::infinity());
//         }
//       } else {
//         prebin_data();
//         merge_bins();
//       }
//       
//       calculate_woe_iv();
//       enforce_monotonicity();
//       
//       return create_output();
//     } catch (const std::exception& e) {
//       Rcpp::stop("Error in OptimalBinningNumericalOSLP: " + std::string(e.what()));
//     }
//   }
//   
// private:
//   void prebin_data() {
//     // Get unique sorted feature values
//     std::vector<double> unique_values = feature;
//     std::sort(unique_values.begin(), unique_values.end());
//     unique_values.erase(std::unique(unique_values.begin(), unique_values.end()), unique_values.end());
//     int n_unique = unique_values.size();
//     int n_prebins = std::min(max_n_prebins, n_unique);
//     
//     // Calculate quantiles for pre-binning
//     std::vector<double> quantiles(n_prebins + 1);
//     for (int i = 0; i <= n_prebins; ++i) {
//       quantiles[i] = i * 1.0 / n_prebins;
//     }
//     
//     // Determine cut points based on quantiles
//     std::vector<double> cuts(n_prebins + 1);
//     for (int i = 0; i <= n_prebins; ++i) {
//       int index = std::min(static_cast<int>(quantiles[i] * (n_unique - 1)), n_unique - 1);
//       cuts[i] = unique_values[index];
//     }
//     
//     // Create initial bin edges
//     bin_edges.clear();
//     bin_edges.push_back(-std::numeric_limits<double>::infinity());
//     for (size_t i = 1; i < cuts.size() - 1; ++i) {
//       if (cuts[i] != cuts[i - 1]) {
//         bin_edges.push_back(cuts[i]);
//       }
//     }
//     bin_edges.push_back(std::numeric_limits<double>::infinity());
//   }
//   
//   void merge_bins() {
//     bool bins_acceptable = false;
//     double prev_total_iv = 0.0;
//     iterations_run = 0;
//     
//     while (!bins_acceptable && 
//            (bin_edges.size() - 1 > min_bins) && 
//            (iterations_run < max_iterations)) {
//       calculate_woe_iv();
//       bins_acceptable = check_bin_counts();
//       
//       if (!bins_acceptable) {
//         merge_adjacent_bins();
//       }
//       
//       // Check for convergence
//       if (std::abs(total_iv - prev_total_iv) < convergence_threshold) {
//         converged = true;
//         break;
//       }
//       
//       prev_total_iv = total_iv;
//       iterations_run++;
//     }
//     
//     // Ensure we do not exceed max_bins
//     while (bin_edges.size() - 1 > max_bins) {
//       merge_adjacent_bins();
//       calculate_woe_iv();
//     }
//   }
//   
//   void merge_adjacent_bins() {
//     int n_bins = bin_edges.size() - 1;
//     if (n_bins <= min_bins) return;
//     
//     std::vector<double> temp_woe_values;
//     std::vector<double> temp_iv_values;
//     std::vector<int> temp_count_values;
//     std::vector<int> temp_count_pos_values;
//     std::vector<int> temp_count_neg_values;
//     
//     calculate_bins(bin_edges, temp_woe_values, temp_iv_values,
//                    temp_count_values, temp_count_pos_values,
//                    temp_count_neg_values);
//     
//     int merge_idx = -1;
//     double min_iv_loss = std::numeric_limits<double>::max();
//     
//     // Merge bins with zero counts first
//     for (int i = 0; i < n_bins - 1; ++i) {
//       if (temp_count_pos_values[i] == 0 ||
//           temp_count_neg_values[i] == 0 ||
//           temp_count_pos_values[i + 1] == 0 ||
//           temp_count_neg_values[i + 1] == 0) {
//         merge_idx = i;
//         break;
//       }
//     }
//     
//     // If no zero counts, merge bins with minimal IV loss
//     if (merge_idx == -1) {
//       for (int i = 0; i < n_bins - 1; ++i) {
//         double iv_loss = temp_iv_values[i] + temp_iv_values[i + 1];
//         if (iv_loss < min_iv_loss) {
//           min_iv_loss = iv_loss;
//           merge_idx = i;
//         }
//       }
//     }
//     
//     if (merge_idx != -1) {
//       bin_edges.erase(bin_edges.begin() + merge_idx + 1);
//     }
//   }
//   
//   bool check_bin_counts() {
//     int total_count = feature.size();
//     calculate_bins(bin_edges, woe_values, iv_values,
//                    count_values, count_pos_values, count_neg_values);
//     for (size_t i = 0; i < count_values.size(); ++i) {
//       double proportion = static_cast<double>(count_values[i]) / total_count;
//       if (proportion < bin_cutoff ||
//           count_pos_values[i] == 0 ||
//           count_neg_values[i] == 0) {
//         return false;
//       }
//     }
//     return true;
//   }
//   
//   void calculate_woe_iv() {
//     calculate_bins(bin_edges, woe_values, iv_values,
//                    count_values, count_pos_values, count_neg_values);
//     total_iv = std::accumulate(iv_values.begin(), iv_values.end(), 0.0);
//     update_bin_labels();
//   }
//   
//   void calculate_bins(const std::vector<double>& edges,
//                       std::vector<double>& woe_vals,
//                       std::vector<double>& iv_vals,
//                       std::vector<int>& counts,
//                       std::vector<int>& counts_pos,
//                       std::vector<int>& counts_neg) {
//     int n_bins = edges.size() - 1;
//     woe_vals.resize(n_bins);
//     iv_vals.resize(n_bins);
//     counts.resize(n_bins);
//     counts_pos.resize(n_bins);
//     counts_neg.resize(n_bins);
//     
//     std::fill(counts.begin(), counts.end(), 0);
//     std::fill(counts_pos.begin(), counts_pos.end(), 0);
//     std::fill(counts_neg.begin(), counts_neg.end(), 0);
//     
//     for (size_t i = 0; i < feature.size(); ++i) {
//       double val = feature[i];
//       int bin_idx = find_bin(edges, val);
//       if (bin_idx >= 0 && bin_idx < n_bins) {
//         counts[bin_idx] += 1;
//         if (target[i] == 1) {
//           counts_pos[bin_idx] += 1;
//         } else {
//           counts_neg[bin_idx] += 1;
//         }
//       }
//     }
//     
//     double total_pos = std::accumulate(counts_pos.begin(), counts_pos.end(), 0.0);
//     double total_neg = std::accumulate(counts_neg.begin(), counts_neg.end(), 0.0);
//     
//     for (int i = 0; i < n_bins; ++i) {
//       double pos_count = counts_pos[i];
//       double neg_count = counts_neg[i];
//       
//       // Avoid division by zero
//       if (pos_count == 0) pos_count = 0.5;
//       if (neg_count == 0) neg_count = 0.5;
//       double pos_rate = pos_count / total_pos;
//       double neg_rate = neg_count / total_neg;
//       
//       double woe = std::log(pos_rate / neg_rate);
//       double iv = (pos_rate - neg_rate) * woe;
//       
//       woe_vals[i] = woe;
//       iv_vals[i] = iv;
//     }
//   }
//   
//   void enforce_monotonicity() {
//     bool monotonic = false;
//     while (!monotonic && bin_edges.size() - 1 > min_bins) {
//       calculate_woe_iv();
//       bool is_increasing = woe_values.front() <= woe_values.back();
//       for (size_t i = 1; i < woe_values.size(); ++i) {
//         if ((is_increasing && woe_values[i] < woe_values[i - 1]) ||
//             (!is_increasing && woe_values[i] > woe_values[i - 1])) {
//           bin_edges.erase(bin_edges.begin() + i);
//           monotonic = false;
//           break;
//         }
//       }
//       if (monotonic) {
//         break;
//       }
//     }
//     calculate_woe_iv();
//   }
//   
//   int find_bin(const std::vector<double>& edges, double val) {
//     auto it = std::lower_bound(edges.begin(), edges.end(), val);
//     int idx = std::distance(edges.begin(), it) - 1;
//     if (idx < 0) idx = 0;
//     if (idx >= static_cast<int>(edges.size()) - 1)
//       idx = edges.size() - 2;
//     return idx;
//   }
//   
//   void update_bin_labels() {
//     bin_labels.clear();
//     for (size_t i = 0; i < bin_edges.size() - 1; ++i) {
//       std::ostringstream oss;
//       oss << "(" << edge_to_string(bin_edges[i]) << ";"
//           << edge_to_string(bin_edges[i + 1]) << "]";
//       bin_labels.push_back(oss.str());
//     }
//   }
//   
//   std::string edge_to_string(double edge) {
//     if (std::isinf(edge)) {
//       if (edge < 0) return "-Inf";
//       else return "+Inf";
//     } else {
//       std::ostringstream oss;
//       oss << std::fixed << std::setprecision(2) << edge;
//       return oss.str();
//     }
//   }
//   
//   Rcpp::List create_output() {
//     std::vector<double> cutpoints(bin_edges.begin() + 1, bin_edges.end() - 1);
//     
//     return Rcpp::List::create(
//       Rcpp::Named("bin") = bin_labels,
//       Rcpp::Named("woe") = woe_values,
//       Rcpp::Named("iv") = iv_values,
//       Rcpp::Named("count") = count_values,
//       Rcpp::Named("count_pos") = count_pos_values,
//       Rcpp::Named("count_neg") = count_neg_values,
//       Rcpp::Named("cutpoints") = cutpoints,
//       Rcpp::Named("converged") = converged,
//       Rcpp::Named("iterations") = iterations_run
//     );
//   }
// };
// 
// //' @title Optimal Binning for Numerical Variables using OSLP
// //'
// //' @description
// //' Performs optimal binning for numerical variables using the Optimal
// //' Supervised Learning Partitioning (OSLP) approach.
// //'
// //' @param target A numeric vector of binary target values (0 or 1).
// //' @param feature A numeric vector of feature values.
// //' @param min_bins Minimum number of bins (default: 3, must be >= 2).
// //' @param max_bins Maximum number of bins (default: 5, must be > min_bins).
// //' @param bin_cutoff Minimum proportion of total observations for a bin
// //'   to avoid being merged (default: 0.05, must be in (0, 1)).
// //' @param max_n_prebins Maximum number of pre-bins before optimization
// //'   (default: 20).
// //' @param convergence_threshold Threshold for convergence (default: 1e-6).
// //' @param max_iterations Maximum number of iterations (default: 1000).
// //'
// //' @return A list containing:
// //' \item{bins}{Character vector of bin labels.}
// //' \item{woe}{Numeric vector of Weight of Evidence (WoE) values for each bin.}
// //' \item{iv}{Numeric vector of Information Value (IV) for each bin.}
// //' \item{count}{Integer vector of total count of observations in each bin.}
// //' \item{count_pos}{Integer vector of positive class count in each bin.}
// //' \item{count_neg}{Integer vector of negative class count in each bin.}
// //' \item{cutpoints}{Numeric vector of cutpoints used to create the bins.}
// //' \item{converged}{Logical value indicating whether the algorithm converged.}
// //' \item{iterations}{Integer value indicating the number of iterations run.}
// //'
// //' @examples
// //' \dontrun{
// //' # Sample data
// //' set.seed(123)
// //' n <- 1000
// //' target <- sample(0:1, n, replace = TRUE)
// //' feature <- rnorm(n)
// //'
// //' # Optimal binning
// //' result <- optimal_binning_numerical_oslp(target, feature,
// //'                                          min_bins = 2, max_bins = 4)
// //'
// //' # Print results
// //' print(result)
// //' }
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_numerical_oslp(Rcpp::NumericVector target,
//                                          Rcpp::NumericVector feature,
//                                          int min_bins = 3,
//                                          int max_bins = 5,
//                                          double bin_cutoff = 0.05,
//                                          int max_n_prebins = 20,
//                                          double convergence_threshold = 1e-6,
//                                          int max_iterations = 1000) {
//  // Input validation
//  if (target.size() != feature.size()) {
//    Rcpp::stop("Target and feature vectors must have the same length.");
//  }
//  if (min_bins < 2) {
//    Rcpp::stop("min_bins must be at least 2.");
//  }
//  if (max_bins < min_bins) {
//    Rcpp::stop("max_bins must be greater than or equal to min_bins.");
//  }
//  if (bin_cutoff <= 0 || bin_cutoff >= 1) {
//    Rcpp::stop("bin_cutoff must be between 0 and 1.");
//  }
//  if (max_n_prebins < min_bins) {
//    Rcpp::stop("max_n_prebins must be at least equal to min_bins.");
//  }
//  if (convergence_threshold <= 0) {
//    Rcpp::stop("convergence_threshold must be positive.");
//  }
//  if (max_iterations <= 0) {
//    Rcpp::stop("max_iterations must be positive.");
//  }
//  
//  try {
//    std::vector<double> feature_vec(feature.begin(), feature.end());
//    std::vector<double> target_vec(target.begin(), target.end());
//    
//    OptimalBinningNumericalOSLP binning(feature_vec, target_vec,
//                                        min_bins, max_bins,
//                                        bin_cutoff, max_n_prebins,
//                                        convergence_threshold, max_iterations);
//    
//    Rcpp::List result = binning.fit();
//    
//    return result;
//  } catch (const std::exception& e) {
//    Rcpp::stop("Error in optimal_binning_numerical_oslp: " + std::string(e.what()));
//  }
// }
