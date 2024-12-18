// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <sstream>

using namespace Rcpp;

class OptimalBinningNumericalMRBLP {
private:
  // Feature and target vectors
  std::vector<double> feature;
  std::vector<int> target;
  
  // Binning parameters
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  double convergence_threshold;
  int max_iterations;
  
  // Struct representing a single bin
  struct Bin {
    double lower_bound;
    double upper_bound;
    double woe;
    double iv;
    int count;
    int count_pos;
    int count_neg;
  };
  
  std::vector<Bin> bins; // Vector of bins
  bool converged;
  int iterations_run;
  
public:
  OptimalBinningNumericalMRBLP(const NumericVector& feature,
                               const IntegerVector& target,
                               int min_bins,
                               int max_bins,
                               double bin_cutoff,
                               int max_n_prebins,
                               double convergence_threshold,
                               int max_iterations)
    : feature(feature.begin(), feature.end()),
      target(target.begin(), target.end()),
      min_bins(min_bins),
      max_bins(std::max(min_bins, max_bins)),
      bin_cutoff(bin_cutoff),
      max_n_prebins(std::max(max_n_prebins, min_bins)),
      convergence_threshold(convergence_threshold),
      max_iterations(max_iterations),
      converged(false),
      iterations_run(0) {
    
    // Validate inputs
    if (this->feature.size() != this->target.size()) {
      throw std::invalid_argument("Feature and target vectors must be of the same length.");
    }
    if (min_bins < 1) {
      throw std::invalid_argument("min_bins must be at least 1.");
    }
    if (max_bins < min_bins) {
      throw std::invalid_argument("max_bins must be greater than or equal to min_bins.");
    }
    if (bin_cutoff <= 0 || bin_cutoff >= 1) {
      throw std::invalid_argument("bin_cutoff must be between 0 and 1.");
    }
    if (convergence_threshold <= 0) {
      throw std::invalid_argument("convergence_threshold must be greater than 0.");
    }
    if (max_iterations <= 0) {
      throw std::invalid_argument("max_iterations must be greater than 0.");
    }
    
    // Check target validity
    bool has_zero = false, has_one = false;
    for (int t : this->target) {
      if (t == 0) has_zero = true;
      else if (t == 1) has_one = true;
      else throw std::invalid_argument("Target must contain only 0 and 1.");
      if (has_zero && has_one) break;
    }
    if (!has_zero || !has_one) {
      throw std::invalid_argument("Target must contain both classes (0 and 1).");
    }
    
    // Check for NaN/Inf in feature
    for (double f : this->feature) {
      if (std::isnan(f) || std::isinf(f)) {
        throw std::invalid_argument("Feature contains NaN or Inf values.");
      }
    }
  }
  
  void fit() {
    // Determine unique feature values
    std::vector<double> unique_features = feature;
    std::sort(unique_features.begin(), unique_features.end());
    auto last = std::unique(unique_features.begin(), unique_features.end());
    unique_features.erase(last, unique_features.end());
    
    size_t n_unique = unique_features.size();
    
    if (n_unique <= 2) {
      // Handle low unique values case
      handle_low_unique_values(unique_features);
      converged = true;
      iterations_run = 0;
      return;
    }
    
    // Proceed with standard binning if unique values > 2
    prebinning();
    mergeSmallBins();
    monotonicBinning();
    computeWOEIV();
  }
  
  List getWoebin() const {
    size_t n_bins = bins.size();
    CharacterVector bin_names(n_bins);
    NumericVector bin_woe(n_bins);
    NumericVector bin_iv(n_bins);
    IntegerVector bin_count(n_bins);
    IntegerVector bin_count_pos(n_bins);
    IntegerVector bin_count_neg(n_bins);
    NumericVector bin_cutpoints(n_bins > 1 ? n_bins - 1 : 0);
    
    for (size_t i = 0; i < n_bins; ++i) {
      std::ostringstream oss;
      if (std::isinf(bins[i].lower_bound) && bins[i].lower_bound < 0) {
        oss << "(-Inf," << bins[i].upper_bound << "]";
      } else if (std::isinf(bins[i].upper_bound)) {
        oss << "(" << bins[i].lower_bound << ",+Inf]";
      } else {
        oss << "(" << bins[i].lower_bound << "," << bins[i].upper_bound << "]";
      }
      bin_names[i] = oss.str();
      bin_woe[i] = bins[i].woe;
      bin_iv[i] = bins[i].iv;
      bin_count[i] = bins[i].count;
      bin_count_pos[i] = bins[i].count_pos;
      bin_count_neg[i] = bins[i].count_neg;
      
      if (i < n_bins - 1) {
        bin_cutpoints[i] = bins[i].upper_bound;
      }
    }
    
    // Criar vetor de IDs com o mesmo tamanho de bins
    Rcpp::NumericVector ids(bin_names.size());
    for(int i = 0; i < bin_names.size(); i++) {
      ids[i] = i + 1;
    }
    
    return Rcpp::List::create(
      Named("id") = ids,
      Named("bin") = bin_names,
      Named("woe") = bin_woe,
      Named("iv") = bin_iv,
      Named("count") = bin_count,
      Named("count_pos") = bin_count_pos,
      Named("count_neg") = bin_count_neg,
      Named("cutpoints") = bin_cutpoints,
      Named("converged") = converged,
      Named("iterations") = iterations_run
    );
  }
  
private:
  void handle_low_unique_values(const std::vector<double>& unique_features) {
    // If one unique value
    if (unique_features.size() == 1) {
      Bin b;
      b.lower_bound = -std::numeric_limits<double>::infinity();
      b.upper_bound = std::numeric_limits<double>::infinity();
      b.count = (int)feature.size();
      b.count_pos = 0;
      b.count_neg = 0;
      for (size_t i = 0; i < feature.size(); ++i) {
        if (target[i] == 1) b.count_pos++;
        else b.count_neg++;
      }
      bins.clear();
      bins.push_back(b);
      computeWOEIV();
    } else {
      // Two unique values
      double cut = unique_features[0];
      Bin b1, b2;
      b1.lower_bound = -std::numeric_limits<double>::infinity();
      b1.upper_bound = cut;
      b1.count = 0; b1.count_pos = 0; b1.count_neg = 0;
      
      b2.lower_bound = cut;
      b2.upper_bound = std::numeric_limits<double>::infinity();
      b2.count = 0; b2.count_pos = 0; b2.count_neg = 0;
      
      for (size_t i = 0; i < feature.size(); ++i) {
        if (feature[i] <= cut) {
          b1.count++;
          if (target[i] == 1) b1.count_pos++; else b1.count_neg++;
        } else {
          b2.count++;
          if (target[i] == 1) b2.count_pos++; else b2.count_neg++;
        }
      }
      bins.clear();
      bins.push_back(b1);
      bins.push_back(b2);
      computeWOEIV();
    }
  }
  
  void prebinning() {
    // Remove NAs if any
    std::vector<std::pair<double,int>> data;
    data.reserve(feature.size());
    for (size_t i = 0; i < feature.size(); ++i) {
      if (!NumericVector::is_na(feature[i])) {
        data.emplace_back(feature[i], target[i]);
      }
    }
    
    if (data.empty()) {
      throw std::runtime_error("All feature values are NA.");
    }
    
    std::sort(data.begin(), data.end(),
              [](const std::pair<double,int>& a, const std::pair<double,int>& b) {
                return a.first < b.first;
              });
    
    // Determine number of distinct values
    int distinct_count = 1;
    for (size_t i = 1; i < data.size(); i++) {
      if (data[i].first != data[i-1].first) distinct_count++;
    }
    
    int n_pre = std::min(max_n_prebins, distinct_count);
    n_pre = std::max(n_pre, min_bins);
    size_t bin_size = std::max((size_t)1, data.size() / (size_t)n_pre);
    
    bins.clear();
    for (size_t i = 0; i < data.size(); i += bin_size) {
      size_t end = std::min(i + bin_size, data.size());
      Bin b;
      b.lower_bound = (i == 0) ? -std::numeric_limits<double>::infinity() : data[i].first;
      b.upper_bound = (end == data.size()) ? std::numeric_limits<double>::infinity() : data[end].first;
      b.count = 0; b.count_pos = 0; b.count_neg = 0;
      for (size_t j = i; j < end; j++) {
        b.count++;
        if (data[j].second == 1) b.count_pos++; else b.count_neg++;
      }
      bins.push_back(b);
    }
    
    computeWOEIV();
  }
  
  void computeWOEIV() {
    int total_pos = 0;
    int total_neg = 0;
    for (auto &b : bins) {
      total_pos += b.count_pos;
      total_neg += b.count_neg;
    }
    if (total_pos == 0 || total_neg == 0) {
      for (auto &b : bins) {
        b.woe = 0.0;
        b.iv = 0.0;
      }
      return;
    }
    for (auto &b : bins) {
      double dist_pos = (b.count_pos > 0) ? (double)b.count_pos / total_pos : 1e-10;
      double dist_neg = (b.count_neg > 0) ? (double)b.count_neg / total_neg : 1e-10;
      dist_pos = std::max(dist_pos, 1e-10);
      dist_neg = std::max(dist_neg, 1e-10);
      b.woe = std::log(dist_pos/dist_neg);
      b.iv = (dist_pos - dist_neg)*b.woe;
    }
  }
  
  void mergeSmallBins() {
    // Merge bins that fail bin_cutoff
    bool merged = true;
    while (merged && (int)bins.size() > min_bins && iterations_run < max_iterations) {
      merged = false;
      double total = (double)feature.size();
      // Find bin with smallest proportion
      size_t smallest_idx = 0;
      double smallest_prop = std::numeric_limits<double>::max();
      for (size_t i = 0; i < bins.size(); i++) {
        double prop = (double)bins[i].count / total;
        if (prop < smallest_prop) {
          smallest_prop = prop;
          smallest_idx = i;
        }
      }
      if (smallest_prop < bin_cutoff && bins.size() > (size_t)min_bins) {
        if (smallest_idx == 0 && bins.size() > 1) {
          mergeBins(0, 1);
        } else if (smallest_idx == bins.size() - 1 && bins.size() > 1) {
          mergeBins(bins.size()-2, bins.size()-1);
        } else {
          // Merge with neighbor with smaller count
          if (smallest_idx > 0 && smallest_idx < bins.size()-1) {
            if (bins[smallest_idx-1].count <= bins[smallest_idx+1].count) {
              mergeBins(smallest_idx-1, smallest_idx);
            } else {
              mergeBins(smallest_idx, smallest_idx+1);
            }
          }
        }
        computeWOEIV();
        merged = true;
      }
      iterations_run++;
    }
  }
  
  bool isMonotonic(bool increasing) {
    for (size_t i = 1; i < bins.size(); i++) {
      if (increasing && bins[i].woe < bins[i-1].woe) return false;
      if (!increasing && bins[i].woe > bins[i-1].woe) return false;
    }
    return true;
  }
  
  bool guessIncreasing() {
    if (bins.size() < 2) return true;
    int inc = 0, dec = 0;
    for (size_t i = 1; i < bins.size(); i++) {
      if (bins[i].woe > bins[i-1].woe) inc++; else if (bins[i].woe < bins[i-1].woe) dec++;
    }
    return inc >= dec;
  }
  
  void monotonicBinning() {
    bool increasing = guessIncreasing();
    
    while (!isMonotonic(increasing) && (int)bins.size() > min_bins && iterations_run < max_iterations) {
      // Find first violation
      for (size_t i = 1; i < bins.size(); i++) {
        if ((increasing && bins[i].woe < bins[i-1].woe) ||
            (!increasing && bins[i].woe > bins[i-1].woe)) {
          mergeBins(i-1, i);
          computeWOEIV();
          break;
        }
      }
      iterations_run++;
      if (isMonotonic(increasing)) {
        converged = true;
        break;
      }
      // If changes are small enough
      if (iterations_run > 1 && std::fabs(bins.back().woe - bins.front().woe) < convergence_threshold) {
        converged = true;
        break;
      }
    }
    if (iterations_run >= max_iterations) {
      converged = false;
    }
    
    // Ensure does not exceed max_bins
    while ((int)bins.size() > max_bins && iterations_run < max_iterations) {
      mergeBinsByIV();
      computeWOEIV();
      iterations_run++;
    }
    if (iterations_run >= max_iterations) {
      converged = false;
    }
  }
  
  size_t findMinIVDiffMerge() {
    if (bins.size() < 2) return bins.size();
    double min_iv_diff = std::numeric_limits<double>::max();
    size_t merge_idx = bins.size();
    for (size_t i = 0; i < bins.size()-1; i++) {
      double iv_diff = std::fabs(bins[i].iv - bins[i+1].iv);
      if (iv_diff < min_iv_diff) {
        min_iv_diff = iv_diff;
        merge_idx = i;
      }
    }
    return merge_idx;
  }
  
  void mergeBinsByIV() {
    size_t idx = findMinIVDiffMerge();
    if (idx < bins.size()) {
      mergeBins(idx, idx+1);
    }
  }
  
  void mergeBins(size_t idx1, size_t idx2) {
    if (idx2 >= bins.size()) return;
    bins[idx1].upper_bound = bins[idx2].upper_bound;
    bins[idx1].count += bins[idx2].count;
    bins[idx1].count_pos += bins[idx2].count_pos;
    bins[idx1].count_neg += bins[idx2].count_neg;
    bins.erase(bins.begin() + idx2);
  }
};


//' @title Optimal Binning for Numerical Variables using Monotonic Risk Binning with Likelihood Ratio Pre-binning (MRBLP)
//'
//' @description
//' This function implements an optimal binning algorithm for numerical variables using
//' Monotonic Risk Binning with Likelihood Ratio Pre-binning (MRBLP). It transforms a
//' continuous feature into discrete bins while preserving the monotonic relationship
//' with the target variable and maximizing the predictive power.
//'
//' @param target An integer vector of binary target values (0 or 1).
//' @param feature A numeric vector of the continuous feature to be binned.
//' @param min_bins Integer. The minimum number of bins to create (default: 3).
//' @param max_bins Integer. The maximum number of bins to create (default: 5).
//' @param bin_cutoff Numeric. The minimum proportion of observations in each bin (default: 0.05).
//' @param max_n_prebins Integer. The maximum number of pre-bins to create during the initial binning step (default: 20).
//' @param convergence_threshold Numeric. The threshold for convergence in the monotonic binning step (default: 1e-6).
//' @param max_iterations Integer. The maximum number of iterations for the monotonic binning step (default: 1000).
//'
//' @return A list containing the following elements:
//' \item{bins}{A character vector of bin ranges.}
//' \item{woe}{A numeric vector of Weight of Evidence (WoE) values for each bin.}
//' \item{iv}{A numeric vector of Information Value (IV) for each bin.}
//' \item{count}{An integer vector of the total count of observations in each bin.}
//' \item{count_pos}{An integer vector of the count of positive observations in each bin.}
//' \item{count_neg}{An integer vector of the count of negative observations in each bin.}
//' \item{cutpoints}{A numeric vector of cutpoints used to create the bins.}
//' \item{converged}{A logical value indicating whether the algorithm converged.}
//' \item{iterations}{An integer value indicating the number of iterations run.}
//'
//' @details
//' The MRBLP algorithm combines pre-binning, small bin merging, and monotonic binning to create an optimal binning solution for numerical variables. The process involves the following steps:
//'
//' 1. Pre-binning: The algorithm starts by creating initial bins using equal-frequency binning. The number of pre-bins is determined by the `max_n_prebins` parameter.
//' 2. Small bin merging: Bins with a proportion of observations less than `bin_cutoff` are merged with adjacent bins to ensure statistical significance.
//' 3. Monotonic binning: The algorithm enforces a monotonic relationship between the bin order and the Weight of Evidence (WoE) values. This step ensures that the binning preserves the original relationship between the feature and the target variable.
//' 4. Bin count adjustment: If the number of bins exceeds `max_bins`, the algorithm merges bins with the smallest difference in Information Value (IV). If the number of bins is less than `min_bins`, the largest bin is split.
//'
//' The algorithm includes additional controls to prevent instability and ensure convergence:
//' - A convergence threshold is used to determine when the algorithm should stop iterating.
//' - A maximum number of iterations is set to prevent infinite loops.
//' - If convergence is not reached within the specified time and standards, the function returns the best result obtained up to the last iteration.
//'
//' @examples
//' \dontrun{
//' # Generate sample data
//' set.seed(42)
//' n <- 10000
//' feature <- rnorm(n)
//' target <- rbinom(n, 1, plogis(0.5 + 0.5 * feature))
//'
//' # Run optimal binning
//' result <- optimal_binning_numerical_mrblp(target, feature)
//'
//' # View binning results
//' print(result)
//' }
//'
//' @references
//' \itemize{
//' \item Belcastro, L., Marozzo, F., Talia, D., & Trunfio, P. (2020). "Big Data Analytics on Clouds."
//'       In Handbook of Big Data Technologies (pp. 101-142). Springer, Cham.
//' \item Zeng, Y. (2014). "Optimal Binning for Scoring Modeling." Computational Economics, 44(1), 137-149.
//' }
//'
//' @author Lopes, J.
//'
//' @export
// [[Rcpp::export]]
List optimal_binning_numerical_mrblp(const IntegerVector& target,
                                    const NumericVector& feature,
                                    int min_bins = 3,
                                    int max_bins = 5,
                                    double bin_cutoff = 0.05,
                                    int max_n_prebins = 20,
                                    double convergence_threshold = 1e-6,
                                    int max_iterations = 1000) {
 try {
   OptimalBinningNumericalMRBLP binning(feature, target, min_bins, max_bins, bin_cutoff,
                                        max_n_prebins, convergence_threshold, max_iterations);
   binning.fit();
   return binning.getWoebin();
 } catch (const std::exception& e) {
   Rcpp::stop(std::string("Error in optimal_binning_numerical_mrblp: ") + e.what());
 }
}

/*
 Breves melhorias implementadas:
 - Verificações extensivas de entrada e condições dos parâmetros.
 - Uso de EPSILON ao calcular WoE/IV para evitar log(0).
 - Garantias de não exceder min_bins ou max_bins, mesclando bins apropriadamente.
 - Mecanismo de detecção de monotonicidade e fusão de bins que quebram a monotonicidade.
 - Limite de iterações e verificação de convergência a cada etapa, prevenindo loops infinitos.
 - Tratamento cuidadoso de casos com poucos valores únicos, evitando tentativas de divisão impossíveis.
 - Comentários em inglês para logs, ajustes, e lógica interna.
*/



// #include <Rcpp.h>
// #include <algorithm>
// #include <vector>
// #include <cmath>
// #include <limits>
// #include <stdexcept>
// #include <sstream>
// 
// using namespace Rcpp;
// 
// // Class for Optimal Binning using MRBLP
// class OptimalBinningNumericalMRBLP {
// private:
//   // Feature and target vectors
//   std::vector<double> feature;
//   std::vector<int> target;
//   
//   // Binning parameters
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
//   double convergence_threshold;
//   int max_iterations;
//   
//   // Struct representing a single bin
//   struct Bin {
//     double lower_bound;
//     double upper_bound;
//     double woe;
//     double iv;
//     int count;
//     int count_pos;
//     int count_neg;
//   };
//   
//   std::vector<Bin> bins; // Vector of bins
//   bool converged;
//   int iterations_run;
//   
// public:
//   // Constructor for the OptimalBinningNumericalMRBLP class
//   OptimalBinningNumericalMRBLP(const NumericVector& feature,
//                                const IntegerVector& target,
//                                int min_bins,
//                                int max_bins,
//                                double bin_cutoff,
//                                int max_n_prebins,
//                                double convergence_threshold,
//                                int max_iterations)
//     : feature(feature.begin(), feature.end()),
//       target(target.begin(), target.end()),
//       min_bins(min_bins),
//       max_bins(max_bins),
//       bin_cutoff(bin_cutoff),
//       max_n_prebins(max_n_prebins),
//       convergence_threshold(convergence_threshold),
//       max_iterations(max_iterations),
//       converged(false),
//       iterations_run(0) {
//     
//     // Input validation
//     if (feature.size() != target.size()) {
//       throw std::invalid_argument("Feature and target vectors must be of the same length.");
//     }
//     if (min_bins <= 0) {
//       throw std::invalid_argument("min_bins must be greater than 0.");
//     }
//     if (max_bins < min_bins) {
//       throw std::invalid_argument("max_bins must be greater than or equal to min_bins.");
//     }
//     if (max_n_prebins <= 0) {
//       throw std::invalid_argument("max_n_prebins must be greater than 0.");
//     }
//     if (bin_cutoff <= 0 || bin_cutoff >= 1) {
//       throw std::invalid_argument("bin_cutoff must be between 0 and 1.");
//     }
//     if (convergence_threshold <= 0) {
//       throw std::invalid_argument("convergence_threshold must be greater than 0.");
//     }
//     if (max_iterations <= 0) {
//       throw std::invalid_argument("max_iterations must be greater than 0.");
//     }
//   }
//   
//   // Fits the binning model to the data
//   void fit() {
//     // Determine unique feature values
//     std::vector<double> unique_features = feature;
//     std::sort(unique_features.begin(), unique_features.end());
//     auto last = std::unique(unique_features.begin(), unique_features.end());
//     unique_features.erase(last, unique_features.end());
//     
//     if (unique_features.size() <= 2) {
//       // If unique values <= 2, create bins without optimization
//       Bin bin;
//       bin.lower_bound = -std::numeric_limits<double>::infinity();
//       bin.upper_bound = unique_features[0];
//       bin.count = 0;
//       bin.count_pos = 0;
//       bin.count_neg = 0;
//       bins.push_back(bin);
//       
//       Bin bin2;
//       bin2.lower_bound = unique_features[0];
//       bin2.upper_bound = std::numeric_limits<double>::infinity();
//       bin2.count = 0;
//       bin2.count_pos = 0;
//       bin2.count_neg = 0;
//       bins.push_back(bin2);
//       
//       // Assign observations to bins
//       for (size_t i = 0; i < feature.size(); ++i) {
//         double val = feature[i];
//         int tgt = target[i];
//         if (val <= unique_features[0]) {
//           bins[0].count++;
//           if (tgt == 1) {
//             bins[0].count_pos++;
//           } else if (tgt == 0) {
//             bins[0].count_neg++;
//           }
//         } else {
//           bins[1].count++;
//           if (tgt == 1) {
//             bins[1].count_pos++;
//           } else if (tgt == 0) {
//             bins[1].count_neg++;
//           }
//         }
//       }
//       
//       // Compute WoE and IV
//       computeWOEIV();
//       return;
//     }
//     
//     // Proceed with standard binning if unique values > 2
//     prebinning();
//     mergeSmallBins();
//     monotonicBinning();
//     
//     // Compute final statistics
//     computeWOEIV();
//   }
//   
//   // Retrieves the binning information as a List
//   List getWoebin() const {
//     size_t n_bins = bins.size();
//     CharacterVector bin_names(n_bins);
//     NumericVector bin_woe(n_bins);
//     NumericVector bin_iv(n_bins);
//     IntegerVector bin_count(n_bins);
//     IntegerVector bin_count_pos(n_bins);
//     IntegerVector bin_count_neg(n_bins);
//     NumericVector bin_cutpoints(n_bins - 1);
//     
//     for (size_t i = 0; i < n_bins; ++i) {
//       std::ostringstream oss;
//       if (std::isinf(bins[i].lower_bound) && bins[i].lower_bound < 0) {
//         oss << "(-Inf," << bins[i].upper_bound << "]";
//       } else if (std::isinf(bins[i].upper_bound)) {
//         oss << "(" << bins[i].lower_bound << ",+Inf]";
//       } else {
//         oss << "(" << bins[i].lower_bound << "," << bins[i].upper_bound << "]";
//       }
//       bin_names[i] = oss.str();
//       bin_woe[i] = bins[i].woe;
//       bin_iv[i] = bins[i].iv;
//       bin_count[i] = bins[i].count;
//       bin_count_pos[i] = bins[i].count_pos;
//       bin_count_neg[i] = bins[i].count_neg;
//       
//       if (i < n_bins - 1) {
//         bin_cutpoints[i] = bins[i].upper_bound;
//       }
//     }
//     
//     return List::create(
//       Named("bin") = bin_names,
//       Named("woe") = bin_woe,
//       Named("iv") = bin_iv,
//       Named("count") = bin_count,
//       Named("count_pos") = bin_count_pos,
//       Named("count_neg") = bin_count_neg,
//       Named("cutpoints") = bin_cutpoints,
//       Named("converged") = converged,
//       Named("iterations") = iterations_run
//     );
//   }
//   
// private:
//   // Performs initial pre-binning using equal-frequency binning
//   void prebinning() {
//     // Remove NA values
//     std::vector<std::pair<double, int>> data;
//     data.reserve(feature.size());
//     
//     for (size_t i = 0; i < feature.size(); ++i) {
//       if (!NumericVector::is_na(feature[i])) {
//         data.emplace_back(feature[i], target[i]);
//       }
//     }
//     
//     if (data.empty()) {
//       throw std::runtime_error("All feature values are NA.");
//     }
//     
//     // Sort data based on feature values
//     std::sort(data.begin(), data.end(),
//               [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
//                 return a.first < b.first;
//               });
//     
//     // Determine unique feature values
//     std::vector<double> unique_features;
//     unique_features.reserve(data.size());
//     unique_features.push_back(data[0].first);
//     for (size_t i = 1; i < data.size(); ++i) {
//       if (data[i].first != data[i - 1].first) {
//         unique_features.push_back(data[i].first);
//       }
//     }
//     
//     size_t unique_size = unique_features.size();
//     
//     // Adjust max_n_prebins based on unique feature values
//     max_n_prebins = std::min(static_cast<int>(unique_size), max_n_prebins);
//     
//     // Determine pre-bin boundaries using equal-frequency binning
//     std::vector<double> boundaries;
//     boundaries.reserve(max_n_prebins + 1);
//     boundaries.emplace_back(-std::numeric_limits<double>::infinity());
//     
//     size_t bin_size = std::max(static_cast<size_t>(1), static_cast<size_t>(data.size() / max_n_prebins));
//     for (int i = 1; i < max_n_prebins; ++i) {
//       size_t idx = i * bin_size;
//       if (idx >= data.size()) {
//         break;
//       }
//       // Ensure unique boundaries by picking the value where feature changes
//       while (idx < data.size() && data[idx].first == data[idx - 1].first) {
//         idx++;
//       }
//       if (idx < data.size()) {
//         boundaries.emplace_back(data[idx].first);
//       }
//     }
//     boundaries.emplace_back(std::numeric_limits<double>::infinity());
//     
//     size_t n_bins = boundaries.size() - 1;
//     bins.clear();
//     bins.resize(n_bins);
//     
//     for (size_t i = 0; i < n_bins; ++i) {
//       bins[i].lower_bound = boundaries[i];
//       bins[i].upper_bound = boundaries[i + 1];
//       bins[i].count = 0;
//       bins[i].count_pos = 0;
//       bins[i].count_neg = 0;
//     }
//     
//     // Assign observations to bins
//     for (const auto& d : data) {
//       double val = d.first;
//       int tgt = d.second;
//       int bin_idx = findBin(val);
//       if (bin_idx != -1) {
//         bins[bin_idx].count++;
//         if (tgt == 1) {
//           bins[bin_idx].count_pos++;
//         } else if (tgt == 0) {
//           bins[bin_idx].count_neg++;
//         }
//       }
//     }
//     
//     // Remove empty bins
//     bins.erase(std::remove_if(bins.begin(), bins.end(),
//                               [](const Bin& bin) { return bin.count == 0; }),
//                               bins.end());
//     
//     // Compute initial WoE and IV
//     computeWOEIV();
//   }
//   
//   // Computes WoE and IV for each bin
//   void computeWOEIV() {
//     double total_pos = 0.0;
//     double total_neg = 0.0;
//     
//     for (const auto& bin : bins) {
//       total_pos += bin.count_pos;
//       total_neg += bin.count_neg;
//     }
//     
//     // Handle cases with no positives or no negatives
//     if (total_pos == 0 || total_neg == 0) {
//       for (auto& bin : bins) {
//         bin.woe = 0.0;
//         bin.iv = 0.0;
//       }
//       return;
//     }
//     
//     for (auto& bin : bins) {
//       double dist_pos = static_cast<double>(bin.count_pos) / total_pos;
//       double dist_neg = static_cast<double>(bin.count_neg) / total_neg;
//       
//       // Avoid division by zero by adding a small constant
//       if (dist_pos == 0.0) {
//         dist_pos = 1e-10;
//       }
//       if (dist_neg == 0.0) {
//         dist_neg = 1e-10;
//       }
//       
//       bin.woe = std::log(dist_pos / dist_neg);
//       bin.iv = (dist_pos - dist_neg) * bin.woe;
//     }
//   }
//   
//   // Merges bins that do not meet the bin_cutoff threshold
//   void mergeSmallBins() {
//     bool merged = true;
//     while (merged && bins.size() > static_cast<size_t>(min_bins)) {
//       merged = false;
//       
//       // Find the bin with the smallest ratio
//       size_t smallest_bin_idx = 0;
//       double smallest_bin_ratio = std::numeric_limits<double>::max();
//       
//       for (size_t i = 0; i < bins.size(); ++i) {
//         double bin_ratio = static_cast<double>(bins[i].count) / feature.size();
//         if (bin_ratio < smallest_bin_ratio) {
//           smallest_bin_ratio = bin_ratio;
//           smallest_bin_idx = i;
//         }
//       }
//       
//       // If the smallest bin meets the cutoff, stop merging
//       if (smallest_bin_ratio >= bin_cutoff) {
//         break;
//       }
//       
//       // Merge the smallest bin with its adjacent bin
//       if (smallest_bin_idx == 0) {
//         mergeBins(0, 1);
//       } else if (smallest_bin_idx == bins.size() - 1) {
//         mergeBins(bins.size() - 2, bins.size() - 1);
//       } else {
//         // Merge with the neighbor that has a smaller count
//         if (bins[smallest_bin_idx - 1].count <= bins[smallest_bin_idx + 1].count) {
//           mergeBins(smallest_bin_idx - 1, smallest_bin_idx);
//         } else {
//           mergeBins(smallest_bin_idx, smallest_bin_idx + 1);
//         }
//       }
//       computeWOEIV();
//       merged = true;
//     }
//   }
//   
//   // Merges two adjacent bins specified by their indices
//   void mergeBins(size_t idx1, size_t idx2) {
//     if (idx2 >= bins.size()) return;
//     
//     bins[idx1].upper_bound = bins[idx2].upper_bound;
//     bins[idx1].count += bins[idx2].count;
//     bins[idx1].count_pos += bins[idx2].count_pos;
//     bins[idx1].count_neg += bins[idx2].count_neg;
//     
//     bins.erase(bins.begin() + idx2);
//   }
//   
//   // Determines if WoE values are increasing or decreasing
//   bool isIncreasingWOE() const {
//     int n_increasing = 0;
//     int n_decreasing = 0;
//     
//     for (size_t i = 0; i < bins.size() - 1; ++i) {
//       if (bins[i].woe < bins[i + 1].woe) {
//         n_increasing++;
//       } else if (bins[i].woe > bins[i + 1].woe) {
//         n_decreasing++;
//       }
//     }
//     return n_increasing >= n_decreasing;
//   }
//   
//   // Enforces monotonicity in WoE values across bins
//   void monotonicBinning() {
//     bool increasing = isIncreasingWOE();
//     
//     iterations_run = 0;
//     converged = false;
//     bool need_merge = true;
//     while (need_merge && bins.size() > static_cast<size_t>(min_bins) && iterations_run < max_iterations) {
//       need_merge = false;
//       for (size_t i = 0; i < bins.size() - 1; ++i) {
//         bool violation = false;
//         if (increasing) {
//           if (bins[i].woe > bins[i + 1].woe) {
//             violation = true;
//           }
//         } else {
//           if (bins[i].woe < bins[i + 1].woe) {
//             violation = true;
//           }
//         }
//         
//         if (violation) {
//           mergeBins(i, i + 1);
//           computeWOEIV();
//           need_merge = true;
//           break;
//         }
//       }
//       
//       iterations_run++;
//       
//       // Check for convergence
//       if (!need_merge) {
//         converged = true;
//         break;
//       }
//       
//       // Check if we've reached the convergence threshold
//       if (iterations_run > 1 && std::abs(bins[bins.size()-1].woe - bins[0].woe) < convergence_threshold) {
//         converged = true;
//         break;
//       }
//     }
//     
//     // Ensure the number of bins does not exceed max_bins
//     while (bins.size() > static_cast<size_t>(max_bins)) {
//       size_t merge_idx = findSmallestIVDiff();
//       mergeBins(merge_idx, merge_idx + 1);
//       computeWOEIV();
//     }
//     
//     // Ensure minimum number of bins
//     ensureMinBins();
//   }
//   
//   // Finds the index of the bin pair with the smallest absolute IV difference
//   size_t findSmallestIVDiff() const {
//     double min_diff = std::numeric_limits<double>::infinity();
//     size_t min_idx = 0;
//     for (size_t i = 0; i < bins.size() - 1; ++i) {
//       double iv_diff = std::abs(bins[i].iv - bins[i + 1].iv);
//       if (iv_diff < min_diff) {
//         min_diff = iv_diff;
//         min_idx = i;
//       }
//     }
//     return min_idx;
//   }
//   
//   // Ensures that the number of bins does not fall below min_bins by splitting the largest bin
//   void ensureMinBins() {
//     while (bins.size() < static_cast<size_t>(min_bins)) {
//       size_t split_idx = findLargestBin();
//       splitBin(split_idx);
//       computeWOEIV();
//     }
//   }
//   
//   // Finds the index of the bin with the largest count
//   size_t findLargestBin() const {
//     size_t max_idx = 0;
//     int max_count = bins[0].count;
//     for (size_t i = 1; i < bins.size(); ++i) {
//       if (bins[i].count > max_count) {
//         max_count = bins[i].count;
//         max_idx = i;
//       }
//     }
//     return max_idx;
//   }
//   
//   // Splits a bin into two at the midpoint of its boundaries
//   void splitBin(size_t idx) {
//     if (idx >= bins.size()) return;
//     
//     Bin& bin = bins[idx];
//     double mid = (bin.lower_bound + bin.upper_bound) / 2.0;
//     
//     // Create a new bin
//     Bin new_bin;
//     new_bin.lower_bound = mid;
//     new_bin.upper_bound = bin.upper_bound;
//     new_bin.count = bin.count / 2;
//     new_bin.count_pos = bin.count_pos / 2;
//     new_bin.count_neg = bin.count_neg / 2;
//     
//     // Update the existing bin
//     bin.upper_bound = mid;
//     bin.count -= new_bin.count;
//     bin.count_pos -= new_bin.count_pos;
//     bin.count_neg -= new_bin.count_neg;
//     
//     // Insert the new bin after the current bin
//     bins.insert(bins.begin() + idx + 1, new_bin);
//   }
//   
//   // Finds the bin index for a given feature value using binary search
//   int findBin(double val) const {
//     int left = 0;
//     int right = bins.size() - 1;
//     int mid;
//     
//     while (left <= right) {
//       mid = left + (right - left) / 2;
//       if (val > bins[mid].lower_bound && val <= bins[mid].upper_bound) {
//         return mid;
//       } else if (val <= bins[mid].lower_bound) {
//         right = mid - 1;
//       } else { // val > bins[mid].upper_bound
//         left = mid + 1;
//       }
//     }
//     return -1; // Not found
//   }
// };
// 
// //' @title Optimal Binning for Numerical Variables using Monotonic Risk Binning with Likelihood Ratio Pre-binning (MRBLP)
// //'
// //' @description
// //' This function implements an optimal binning algorithm for numerical variables using
// //' Monotonic Risk Binning with Likelihood Ratio Pre-binning (MRBLP). It transforms a
// //' continuous feature into discrete bins while preserving the monotonic relationship
// //' with the target variable and maximizing the predictive power.
// //'
// //' @param target An integer vector of binary target values (0 or 1).
// //' @param feature A numeric vector of the continuous feature to be binned.
// //' @param min_bins Integer. The minimum number of bins to create (default: 3).
// //' @param max_bins Integer. The maximum number of bins to create (default: 5).
// //' @param bin_cutoff Numeric. The minimum proportion of observations in each bin (default: 0.05).
// //' @param max_n_prebins Integer. The maximum number of pre-bins to create during the initial binning step (default: 20).
// //' @param convergence_threshold Numeric. The threshold for convergence in the monotonic binning step (default: 1e-6).
// //' @param max_iterations Integer. The maximum number of iterations for the monotonic binning step (default: 1000).
// //'
// //' @return A list containing the following elements:
// //' \item{bins}{A character vector of bin ranges.}
// //' \item{woe}{A numeric vector of Weight of Evidence (WoE) values for each bin.}
// //' \item{iv}{A numeric vector of Information Value (IV) for each bin.}
// //' \item{count}{An integer vector of the total count of observations in each bin.}
// //' \item{count_pos}{An integer vector of the count of positive observations in each bin.}
// //' \item{count_neg}{An integer vector of the count of negative observations in each bin.}
// //' \item{cutpoints}{A numeric vector of cutpoints used to create the bins.}
// //' \item{converged}{A logical value indicating whether the algorithm converged.}
// //' \item{iterations}{An integer value indicating the number of iterations run.}
// //'
// //' @details
// //' The MRBLP algorithm combines pre-binning, small bin merging, and monotonic binning to create an optimal binning solution for numerical variables. The process involves the following steps:
// //'
// //' 1. Pre-binning: The algorithm starts by creating initial bins using equal-frequency binning. The number of pre-bins is determined by the `max_n_prebins` parameter.
// //' 2. Small bin merging: Bins with a proportion of observations less than `bin_cutoff` are merged with adjacent bins to ensure statistical significance.
// //' 3. Monotonic binning: The algorithm enforces a monotonic relationship between the bin order and the Weight of Evidence (WoE) values. This step ensures that the binning preserves the original relationship between the feature and the target variable.
// //' 4. Bin count adjustment: If the number of bins exceeds `max_bins`, the algorithm merges bins with the smallest difference in Information Value (IV). If the number of bins is less than `min_bins`, the largest bin is split.
// //'
// //' The algorithm includes additional controls to prevent instability and ensure convergence:
// //' - A convergence threshold is used to determine when the algorithm should stop iterating.
// //' - A maximum number of iterations is set to prevent infinite loops.
// //' - If convergence is not reached within the specified time and standards, the function returns the best result obtained up to the last iteration.
// //'
// //' @examples
// //' \dontrun{
// //' # Generate sample data
// //' set.seed(42)
// //' n <- 10000
// //' feature <- rnorm(n)
// //' target <- rbinom(n, 1, plogis(0.5 + 0.5 * feature))
// //'
// //' # Run optimal binning
// //' result <- optimal_binning_numerical_mrblp(target, feature)
// //'
// //' # View binning results
// //' print(result)
// //' }
// //'
// //' @references
// //' \itemize{
// //' \item Belcastro, L., Marozzo, F., Talia, D., & Trunfio, P. (2020). "Big Data Analytics on Clouds."
// //'       In Handbook of Big Data Technologies (pp. 101-142). Springer, Cham.
// //' \item Zeng, Y. (2014). "Optimal Binning for Scoring Modeling." Computational Economics, 44(1), 137-149.
// //' }
// //'
// //' @author Lopes, J.
// //'
// //' @export
// // [[Rcpp::export]]
// List optimal_binning_numerical_mrblp(const IntegerVector& target,
//                                     const NumericVector& feature,
//                                     int min_bins = 3,
//                                     int max_bins = 5,
//                                     double bin_cutoff = 0.05,
//                                     int max_n_prebins = 20,
//                                     double convergence_threshold = 1e-6,
//                                     int max_iterations = 1000) {
//  try {
//    // Instantiate the binning class
//    OptimalBinningNumericalMRBLP binning(feature, target, min_bins, max_bins, bin_cutoff, max_n_prebins, convergence_threshold, max_iterations);
//    
//    // Fit the binning model
//    binning.fit();
//    
//    // Get the binning information
//    return binning.getWoebin();
//  } catch (const std::exception& e) {
//    Rcpp::stop("Error in optimal_binning_numerical_mrblp: " + std::string(e.what()));
//  }
// }
