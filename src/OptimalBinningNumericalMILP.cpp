#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <limits>
#include <cmath>
#include <numeric>
#ifdef _OPENMP
#include <omp.h>
#endif

// [[Rcpp::plugins(openmp)]]

using namespace Rcpp;

// Define a struct to hold bin information
struct Bin {
  double lower_bound;
  double upper_bound;
  int count;
  int count_pos;
  int count_neg;
  double woe;
  double iv;
  
  Bin(double lb, double ub) : lower_bound(lb), upper_bound(ub),
  count(0), count_pos(0), count_neg(0),
  woe(0.0), iv(0.0) {}
};

// Comparator for sorting indices based on feature values
struct CompareFeature {
  const NumericVector& feature;
  CompareFeature(const NumericVector& feat) : feature(feat) {}
  bool operator()(const int& a, const int& b) const {
    return feature[a] < feature[b];
  }
};

// Function to calculate WoE and IV
void calculate_woe_iv(std::vector<Bin>& bins, double total_pos, double total_neg) {
  const double epsilon = 1e-10; // Small constant to avoid division by zero
  for(auto &bin : bins){
    double dist_pos = (bin.count_pos + epsilon) / (total_pos + epsilon);
    double dist_neg = (bin.count_neg + epsilon) / (total_neg + epsilon);
    
    // Handle cases where dist_pos or dist_neg are effectively zero
    if(dist_pos <= 0.0){
      bin.woe = std::log(epsilon / dist_neg);
    }
    else if(dist_neg <= 0.0){
      bin.woe = std::log(dist_pos / epsilon);
    }
    else{
      bin.woe = std::log(dist_pos / dist_neg);
    }
    bin.iv = (dist_pos - dist_neg) * bin.woe;
  }
}

// Modified function to enforce monotonicity with min_bins constraint
bool enforce_monotonicity(std::vector<Bin>& bins, size_t min_bins) {
  bool merged = false;
  for(size_t i = 1; i < bins.size(); ++i){
    if(bins[i].woe < bins[i-1].woe){
      if(bins.size() <= min_bins) {
        // Do not merge if it would violate min_bins constraint
        continue;
      }
      // Merge bin i with bin i-1
      bins[i-1].upper_bound = bins[i].upper_bound;
      bins[i-1].count += bins[i].count;
      bins[i-1].count_pos += bins[i].count_pos;
      bins[i-1].count_neg += bins[i].count_neg;
      bins.erase(bins.begin() + i);
      merged = true;
      break;
    }
  }
  return merged;
}

// Function to merge bins with count below cutoff
void merge_bins_by_count(std::vector<Bin>& bins, double cutoff_count) {
  bool merged = true;
  while(merged && bins.size() > 1){
    merged = false;
    for(size_t i = 0; i < bins.size(); ++i){
      if(bins[i].count < cutoff_count){
        if(i == 0){
          // Merge with the next bin
          bins[1].lower_bound = bins[0].lower_bound;
          bins[1].count += bins[0].count;
          bins[1].count_pos += bins[0].count_pos;
          bins[1].count_neg += bins[0].count_neg;
          bins.erase(bins.begin());
        } else {
          // Merge with the previous bin
          bins[i-1].upper_bound = bins[i].upper_bound;
          bins[i-1].count += bins[i].count;
          bins[i-1].count_pos += bins[i].count_pos;
          bins[i-1].count_neg += bins[i].count_neg;
          bins.erase(bins.begin() + i);
        }
        merged = true;
        break;
      }
    }
  }
}

// Function to merge bins to meet max_bins constraint
void merge_to_max_bins(std::vector<Bin>& bins, int max_bins) {
  while(bins.size() > (size_t)max_bins){
    // Find the pair of adjacent bins with the smallest IV difference
    double min_diff = std::numeric_limits<double>::max();
    size_t merge_idx = 0;
    for(size_t i = 1; i < bins.size(); ++i){
      double diff = std::abs(bins[i].iv - bins[i-1].iv);
      if(diff < min_diff){
        min_diff = diff;
        merge_idx = i - 1;
      }
    }
    // Merge bins[merge_idx] and bins[merge_idx + 1]
    bins[merge_idx].upper_bound = bins[merge_idx + 1].upper_bound;
    bins[merge_idx].count += bins[merge_idx + 1].count;
    bins[merge_idx].count_pos += bins[merge_idx + 1].count_pos;
    bins[merge_idx].count_neg += bins[merge_idx + 1].count_neg;
    bins.erase(bins.begin() + merge_idx + 1);
  }
}

// Function to generate bin labels
std::string generate_bin_label(const Bin& bin){
  std::ostringstream oss;
  oss.precision(10); // Set precision to avoid scientific notation
  oss << "(";
  if(bin.lower_bound == -std::numeric_limits<double>::infinity()){
    oss << "-Inf";
  } else {
    oss << bin.lower_bound;
  }
  oss << ";";
  if(bin.upper_bound == std::numeric_limits<double>::infinity()){
    oss << "+Inf";
  } else {
    oss << bin.upper_bound;
  }
  oss << "]";
  return oss.str();
}

class OptimalBinningNumericalMILP {
private:
  NumericVector feature;
  IntegerVector target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  int n_threads;
  
  NumericVector woefeature;
  DataFrame woebin;
  
public:
  OptimalBinningNumericalMILP(NumericVector feature, IntegerVector target,
                              int min_bins, int max_bins,
                              double bin_cutoff, int max_n_prebins,
                              int n_threads=1) :
  feature(feature), target(target), min_bins(min_bins),
  max_bins(max_bins), bin_cutoff(bin_cutoff),
  max_n_prebins(max_n_prebins), n_threads(n_threads) {}
  
  void fit(){
    int n = feature.size();
    
    // Create sorted indices based on feature
    std::vector<int> indices(n);
    for(int i = 0; i < n; ++i) indices[i] = i;
    std::sort(indices.begin(), indices.end(), CompareFeature(feature));
    
    // Sort feature and target
    NumericVector sorted_feature(n);
    IntegerVector sorted_target(n);
    for(int i = 0; i < n; ++i){
      sorted_feature[i] = feature[indices[i]];
      sorted_target[i] = target[indices[i]];
    }
    
    // Check if all feature values are identical
    bool all_same = std::all_of(sorted_feature.begin(), sorted_feature.end(),
                                [&](double x){ return x == sorted_feature[0]; });
    
    if(all_same){
      // Create a single bin covering all values
      std::vector<Bin> bins;
      bins.emplace_back(-std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity());
      bins[0].count = n;
      for(int i = 0; i < n; ++i){
        if(sorted_target[i] == 1){
          bins[0].count_pos++;
        } else {
          bins[0].count_neg++;
        }
      }
      
      // Calculate total pos and neg
      double total_pos = bins[0].count_pos;
      double total_neg = bins[0].count_neg;
      
      // Calculate WoE and IV
      calculate_woe_iv(bins, total_pos, total_neg);
      
      // Assign WoE values to observations
      woefeature = NumericVector(n, bins[0].woe);
      
      // Prepare woebin DataFrame
      int n_bins = bins.size();
      CharacterVector bin_char(n_bins);
      NumericVector woe_vec(n_bins);
      NumericVector iv_vec(n_bins);
      IntegerVector count_vec(n_bins);
      IntegerVector count_pos_vec(n_bins);
      IntegerVector count_neg_vec(n_bins);
      
      bin_char[0] = generate_bin_label(bins[0]);
      woe_vec[0] = bins[0].woe;
      iv_vec[0] = bins[0].iv;
      count_vec[0] = bins[0].count;
      count_pos_vec[0] = bins[0].count_pos;
      count_neg_vec[0] = bins[0].count_neg;
      
      woebin = DataFrame::create(
        Named("bin") = bin_char,
        Named("woe") = woe_vec,
        Named("iv") = iv_vec,
        Named("count") = count_vec,
        Named("count_pos") = count_pos_vec,
        Named("count_neg") = count_neg_vec
      );
      return;
    }
    
    // Initial prebinning
    int actual_prebins = std::min(max_n_prebins, n);
    int bin_size = n / actual_prebins;
    if(bin_size == 0) bin_size = 1;
    actual_prebins = std::min(max_n_prebins, (int)std::ceil((double)n / bin_size));
    
    std::vector<Bin> bins;
    double lower = -std::numeric_limits<double>::infinity();
    for(int i = 0; i < actual_prebins; ++i){
      double upper;
      if(i == actual_prebins - 1){
        upper = std::numeric_limits<double>::infinity();
      } else {
        upper = sorted_feature[(i+1)*bin_size - 1];
      }
      bins.emplace_back(lower, upper);
      lower = upper;
    }
    
    // Assign counts to prebins using cumulative counts for efficiency
    for(int i = 0; i < n; ++i){
      if(sorted_target[i] == 1){
        // Increment positive count
        bins.back().count_pos++;
      }
      else{
        // Increment negative count
        bins.back().count_neg++;
      }
      bins.back().count++;
      
      // Check if next feature value should start a new bin
      if(i < n -1 && sorted_feature[i] != sorted_feature[i+1] && (i+1) % bin_size == 0 && bins.size() < (size_t)actual_prebins){
        bins.emplace_back(sorted_feature[i], sorted_feature[i+1]);
      }
    }
    
    // Merge bins with counts below cutoff
    double cutoff_count = bin_cutoff * n;
    merge_bins_by_count(bins, cutoff_count);
    
    // Calculate total pos and neg
    double total_pos = 0.0, total_neg = 0.0;
    for(auto &bin : bins){
      total_pos += bin.count_pos;
      total_neg += bin.count_neg;
    }
    
    // Handle case where total_pos or total_neg is zero
    if(total_pos == 0 || total_neg == 0){
      // Assign zero information
      woefeature = NumericVector(n, 0.0);
      
      // Prepare woebin DataFrame
      int n_bins = bins.size();
      CharacterVector bin_char(n_bins);
      NumericVector woe_vec(n_bins, 0.0);
      NumericVector iv_vec(n_bins, 0.0);
      IntegerVector count_vec(n_bins);
      IntegerVector count_pos_vec(n_bins);
      IntegerVector count_neg_vec(n_bins);
      
      for(int i = 0; i < n_bins; ++i){
        bin_char[i] = generate_bin_label(bins[i]);
        count_vec[i] = bins[i].count;
        count_pos_vec[i] = bins[i].count_pos;
        count_neg_vec[i] = bins[i].count_neg;
      }
      
      woebin = DataFrame::create(
        Named("bin") = bin_char,
        Named("woe") = woe_vec,
        Named("iv") = iv_vec,
        Named("count") = count_vec,
        Named("count_pos") = count_pos_vec,
        Named("count_neg") = count_neg_vec
      );
      return;
    }
    
    // Calculate WoE and IV
    calculate_woe_iv(bins, total_pos, total_neg);
    
    // Enforce monotonicity
    bool merged_monotonic = true;
    while(merged_monotonic){
      merged_monotonic = enforce_monotonicity(bins, min_bins);
      if(merged_monotonic){
        calculate_woe_iv(bins, total_pos, total_neg);
      }
    }
    
    // Merge bins to meet max_bins
    if(bins.size() > (size_t)max_bins){
      merge_to_max_bins(bins, max_bins);
      calculate_woe_iv(bins, total_pos, total_neg);
    }
    
    // Ensure min_bins
    int max_iterations = 100; // Prevent infinite loops
    int iteration = 0;
    while(bins.size() < (size_t)min_bins && iteration < max_iterations){
      iteration++;
      // Split the largest bin
      size_t split_idx = 0;
      int max_count = 0;
      for(size_t i = 0; i < bins.size(); ++i){
        if(bins[i].count > max_count){
          max_count = bins[i].count;
          split_idx = i;
        }
      }
      
      // Find the split point using cumulative counts
      int split_pos = bins[split_idx].count / 2;
      int cumulative_count = 0;
      double split_value = bins[split_idx].lower_bound;
      for(int i = 0; i < n; ++i){
        double x = sorted_feature[i];
        if(x > bins[split_idx].lower_bound && x <= bins[split_idx].upper_bound){
          cumulative_count++;
          if(cumulative_count == split_pos){
            split_value = x;
            break;
          }
        }
      }
      
      // Check if split_value is valid
      if(split_value <= bins[split_idx].lower_bound || split_value >= bins[split_idx].upper_bound){
        // Cannot split further
        break;
      }
      
      // Create new bins
      Bin new_bin1(bins[split_idx].lower_bound, split_value);
      Bin new_bin2(split_value, bins[split_idx].upper_bound);
      
      // Reassign counts efficiently
      for(int i = 0; i < n; ++i){
        double x = sorted_feature[i];
        if(x > new_bin1.lower_bound && x <= new_bin1.upper_bound){
          new_bin1.count++;
          if(sorted_target[i] == 1){
            new_bin1.count_pos++;
          } else {
            new_bin1.count_neg++;
          }
        }
        else if(x > new_bin2.lower_bound && x <= new_bin2.upper_bound){
          new_bin2.count++;
          if(sorted_target[i] == 1){
            new_bin2.count_pos++;
          } else {
            new_bin2.count_neg++;
          }
        }
      }
      
      // Replace the old bin with new bins
      bins.erase(bins.begin() + split_idx);
      bins.insert(bins.begin() + split_idx, new_bin2);
      bins.insert(bins.begin() + split_idx, new_bin1);
      
      // Recalculate WoE and IV
      calculate_woe_iv(bins, total_pos, total_neg);
      
      // Enforce monotonicity after split
      merged_monotonic = true;
      while(merged_monotonic){
        merged_monotonic = enforce_monotonicity(bins, min_bins);
        if(merged_monotonic){
          calculate_woe_iv(bins, total_pos, total_neg);
        }
      }
    }
    if(iteration == max_iterations){
      warning("Maximum iterations reached while ensuring min_bins. Possible conflicting constraints.");
    }
    
    // Assign WoE values to observations using binary search for efficiency
    woefeature = NumericVector(n);
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads)
#endif
    for(int i = 0; i < n; ++i){
      double x = feature[i];
      // Binary search to find the correct bin
      size_t left = 0;
      size_t right = bins.size() - 1;
      size_t mid;
      size_t found = bins.size() -1;
      while(left <= right){
        mid = left + (right - left) / 2;
        if(x > bins[mid].lower_bound && x <= bins[mid].upper_bound){
          found = mid;
          break;
        }
        else if(x <= bins[mid].lower_bound){
          if(mid == 0) break;
          right = mid -1;
        }
        else{
          left = mid +1;
        }
      }
      woefeature[i] = bins[found].woe;
    }
    
    // Prepare woebin DataFrame
    int n_bins = bins.size();
    CharacterVector bin_char(n_bins);
    NumericVector woe_vec(n_bins);
    NumericVector iv_vec(n_bins);
    IntegerVector count_vec(n_bins);
    IntegerVector count_pos_vec(n_bins);
    IntegerVector count_neg_vec(n_bins);
    
    for(int i = 0; i < n_bins; ++i){
      bin_char[i] = generate_bin_label(bins[i]);
      woe_vec[i] = bins[i].woe;
      iv_vec[i] = bins[i].iv;
      count_vec[i] = bins[i].count;
      count_pos_vec[i] = bins[i].count_pos;
      count_neg_vec[i] = bins[i].count_neg;
    }
    
    woebin = DataFrame::create(
      Named("bin") = bin_char,
      Named("woe") = woe_vec,
      Named("iv") = iv_vec,
      Named("count") = count_vec,
      Named("count_pos") = count_pos_vec,
      Named("count_neg") = count_neg_vec
    );
  }
  
  NumericVector get_woefeature(){
    return woefeature;
  }
  
  DataFrame get_woebin(){
    return woebin;
  }
};

//' @title Optimal Binning for Numerical Variables using MILP
//'
//' @description
//' This function performs optimal binning for numerical variables using a Mixed Integer Linear Programming (MILP) inspired approach. It creates optimal bins for a numerical feature based on its relationship with a binary target variable, maximizing the predictive power while respecting user-defined constraints.
//'
//' @param target An integer vector of binary target values (0 or 1).
//' @param feature A numeric vector of feature values.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum proportion of total observations for a bin to avoid being merged (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins before the optimization process (default: 20).
//' @param n_threads Number of threads to use for parallel processing (default: 1).
//'
//' @return A list containing two elements:
//' \item{woefeature}{A numeric vector of Weight of Evidence (WoE) values for each observation.}
//' \item{woebin}{A data frame with the following columns:
//'   \itemize{
//'     \item bin: Character vector of bin ranges.
//'     \item woe: Numeric vector of WoE values for each bin.
//'     \item iv: Numeric vector of Information Value (IV) for each bin.
//'     \item count: Integer vector of total observations in each bin.
//'     \item count_pos: Integer vector of positive target observations in each bin.
//'     \item count_neg: Integer vector of negative target observations in each bin.
//'   }
//' }
//'
//' @details
//' The Optimal Binning algorithm for numerical variables using a MILP-inspired approach works as follows:
//' 1. Create initial pre-bins using equal-frequency binning.
//' 2. Merge bins with counts below the cutoff.
//' 3. Calculate initial Weight of Evidence (WoE) and Information Value (IV) for each bin.
//' 4. Enforce monotonicity of WoE values across bins.
//' 5. Merge bins to meet the max_bins constraint.
//' 6. Split bins if necessary to meet the min_bins constraint.
//' 7. Recalculate WoE and IV for the final bins.
//'
//' The algorithm aims to create bins that maximize the predictive power of the numerical variable while adhering to the specified constraints. It enforces monotonicity of WoE values, which is particularly useful for credit scoring and risk modeling applications.
//'
//' Weight of Evidence (WoE) is calculated as:
//' \deqn{WoE = \ln(\frac{\text{Positive Rate}}{\text{Negative Rate}})}
//'
//' Information Value (IV) is calculated as:
//' \deqn{IV = (\text{Positive Rate} - \text{Negative Rate}) \times WoE}
//'
//' This implementation uses OpenMP for parallel processing when available, which can significantly speed up the computation for large datasets.
//'
//' @references
//' \itemize{
//'   \item Belotti, P., Kirches, C., Leyffer, S., Linderoth, J., Luedtke, J., & Mahajan, A. (2013). Mixed-integer nonlinear optimization. Acta Numerica, 22, 1-131.
//'   \item Mironchyk, P., & Tchistiakov, V. (2017). Monotone optimal binning algorithm for credit risk modeling. SSRN Electronic Journal. doi:10.2139/ssrn.2978774
//' }
//'
//' @examples
//' \dontrun{
//' # Create sample data
//' set.seed(123)
//' n <- 1000
//' target <- sample(0:1, n, replace = TRUE)
//' feature <- rnorm(n)
//'
//' # Run optimal binning
//' result <- optimal_binning_numerical_milp(target, feature, min_bins = 2, max_bins = 4)
//'
//' # Print results
//' print(result$woebin)
//'
//' # Plot WoE values
//' plot(result$woebin$woe, type = "s", xaxt = "n", xlab = "Bins", ylab = "WoE",
//'      main = "Weight of Evidence by Bin")
//' axis(1, at = 1:nrow(result$woebin), labels = result$woebin$bin, las = 2)
//' }
//'
//' @usage
//' optimal_binning_numerical_milp(target, feature, min_bins = 3, max_bins = 5,
//'                                bin_cutoff = 0.05, max_n_prebins = 20, n_threads = 1)
//'
//' @author Lopes, J. E.
//'
//' @export
// [[Rcpp::export]]
List optimal_binning_numerical_milp(IntegerVector target,
                                   NumericVector feature,
                                   int min_bins = 3, int max_bins = 5,
                                   double bin_cutoff = 0.05,
                                   int max_n_prebins = 20,
                                   int n_threads = 1) {
 // Ensure that feature and target have the same length
 if(feature.size() != target.size()){
   stop("Feature and target vectors must have the same length.");
 }
 
 // Ensure that target is binary
 IntegerVector unique_targets = unique(target);
 if(unique_targets.size() != 2){
   stop("Target variable must be binary.");
 }
 
 // Ensure min_bins and max_bins are valid
 if(min_bins < 2){
   warning("min_bins must be at least 2. Setting min_bins to 2.");
   min_bins = 2;
 }
 if(max_bins < min_bins){
   warning("max_bins must be greater than or equal to min_bins. Setting max_bins equal to min_bins.");
   max_bins = min_bins;
 }
 
 // Ensure bin_cutoff is between 0 and 1
 if(bin_cutoff <= 0 || bin_cutoff >= 1){
   warning("bin_cutoff must be between 0 and 1. Setting bin_cutoff to 0.05.");
   bin_cutoff = 0.05;
 }
 
 // Ensure max_n_prebins is at least max_bins
 if(max_n_prebins < max_bins){
   warning("max_n_prebins must be at least max_bins. Setting max_n_prebins equal to max_bins.");
   max_n_prebins = max_bins;
 }
 
 // Ensure n_threads is at least 1
 if(n_threads < 1){
   warning("n_threads must be at least 1. Setting n_threads to 1.");
   n_threads = 1;
 }
 
 // Initialize binning object
 OptimalBinningNumericalMILP binning(feature, target, min_bins, max_bins, bin_cutoff, max_n_prebins, n_threads);
 
 // Fit the binning
 binning.fit();
 
 // Return the results
 return List::create(
   Named("woefeature") = binning.get_woefeature(),
   Named("woebin") = binning.get_woebin()
 );
}



// #include <Rcpp.h>
// #include <vector>
// #include <algorithm>
// #include <string>
// #include <sstream>
// #include <limits>
// #include <cmath>
// 
// // [[Rcpp::plugins(openmp)]]
// 
// using namespace Rcpp;
// 
// // Define a struct to hold bin information
// struct Bin {
//   double lower_bound;
//   double upper_bound;
//   int count;
//   int count_pos;
//   int count_neg;
//   double woe;
//   double iv;
//   
//   Bin(double lb, double ub) : lower_bound(lb), upper_bound(ub),
//   count(0), count_pos(0), count_neg(0),
//   woe(0.0), iv(0.0) {}
// };
// 
// // Comparator for sorting indices based on feature values
// struct CompareFeature {
//   const NumericVector& feature;
//   CompareFeature(const NumericVector& feat) : feature(feat) {}
//   bool operator()(const int& a, const int& b) const {
//     return feature[a] < feature[b];
//   }
// };
// 
// // Function to calculate WoE and IV
// void calculate_woe_iv(std::vector<Bin>& bins, double total_pos, double total_neg) {
//   const double epsilon = 1e-10; // Small constant to avoid division by zero
//   for(auto &bin : bins){
//     double dist_pos = (bin.count_pos + epsilon) / (total_pos + epsilon);
//     double dist_neg = (bin.count_neg + epsilon) / (total_neg + epsilon);
//     
//     bin.woe = std::log(dist_pos / dist_neg);
//     bin.iv = (dist_pos - dist_neg) * bin.woe;
//   }
// }
// 
// // Modified function to enforce monotonicity with min_bins constraint
// bool enforce_monotonicity(std::vector<Bin>& bins, size_t min_bins) {
//   bool merged = false;
//   for(size_t i = 1; i < bins.size(); ++i){
//     if(bins[i].woe < bins[i-1].woe){
//       if(bins.size() <= min_bins) {
//         // Do not merge if it would violate min_bins constraint
//         continue;
//       }
//       // Merge bin i with bin i-1
//       bins[i-1].upper_bound = bins[i].upper_bound;
//       bins[i-1].count += bins[i].count;
//       bins[i-1].count_pos += bins[i].count_pos;
//       bins[i-1].count_neg += bins[i].count_neg;
//       bins.erase(bins.begin() + i);
//       merged = true;
//       break;
//     }
//   }
//   return merged;
// }
// 
// // Function to merge bins with count below cutoff
// void merge_bins_by_count(std::vector<Bin>& bins, double cutoff_count) {
//   bool merged = true;
//   while(merged && bins.size() > 1){
//     merged = false;
//     for(size_t i = 0; i < bins.size(); ++i){
//       if(bins[i].count < cutoff_count){
//         if(i == 0){
//           // Merge with the next bin
//           bins[1].lower_bound = bins[0].lower_bound;
//           bins[1].count += bins[0].count;
//           bins[1].count_pos += bins[0].count_pos;
//           bins[1].count_neg += bins[0].count_neg;
//           bins.erase(bins.begin());
//         } else {
//           // Merge with the previous bin
//           bins[i-1].upper_bound = bins[i].upper_bound;
//           bins[i-1].count += bins[i].count;
//           bins[i-1].count_pos += bins[i].count_pos;
//           bins[i-1].count_neg += bins[i].count_neg;
//           bins.erase(bins.begin() + i);
//         }
//         merged = true;
//         break;
//       }
//     }
//   }
// }
// 
// // Function to merge bins to meet max_bins constraint
// void merge_to_max_bins(std::vector<Bin>& bins, int max_bins) {
//   while(bins.size() > (size_t)max_bins){
//     // Find the pair of adjacent bins with the smallest IV difference
//     double min_diff = std::numeric_limits<double>::max();
//     size_t merge_idx = 0;
//     for(size_t i = 1; i < bins.size(); ++i){
//       double diff = std::abs(bins[i].iv - bins[i-1].iv);
//       if(diff < min_diff){
//         min_diff = diff;
//         merge_idx = i - 1;
//       }
//     }
//     // Merge bins[merge_idx] and bins[merge_idx + 1]
//     bins[merge_idx].upper_bound = bins[merge_idx + 1].upper_bound;
//     bins[merge_idx].count += bins[merge_idx + 1].count;
//     bins[merge_idx].count_pos += bins[merge_idx + 1].count_pos;
//     bins[merge_idx].count_neg += bins[merge_idx + 1].count_neg;
//     bins.erase(bins.begin() + merge_idx + 1);
//   }
// }
// 
// // Function to generate bin labels
// std::string generate_bin_label(const Bin& bin){
//   std::ostringstream oss;
//   oss.precision(10); // Set precision to avoid scientific notation
//   oss << "(";
//   if(bin.lower_bound == -std::numeric_limits<double>::infinity()){
//     oss << "-Inf";
//   } else {
//     oss << bin.lower_bound;
//   }
//   oss << ";";
//   if(bin.upper_bound == std::numeric_limits<double>::infinity()){
//     oss << "+Inf";
//   } else {
//     oss << bin.upper_bound;
//   }
//   oss << "]";
//   return oss.str();
// }
// 
// class OptimalBinningNumericalMILP {
// private:
//   NumericVector feature;
//   IntegerVector target;
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
//   int n_threads;
//   
//   NumericVector woefeature;
//   DataFrame woebin;
//   
// public:
//   OptimalBinningNumericalMILP(NumericVector feature, IntegerVector target,
//                               int min_bins, int max_bins,
//                               double bin_cutoff, int max_n_prebins,
//                               int n_threads=1) :
//   feature(feature), target(target), min_bins(min_bins),
//   max_bins(max_bins), bin_cutoff(bin_cutoff),
//   max_n_prebins(max_n_prebins), n_threads(n_threads) {}
//   
//   void fit(){
//     int n = feature.size();
//     
//     // Create sorted indices based on feature
//     std::vector<int> indices(n);
//     for(int i = 0; i < n; ++i) indices[i] = i;
//     std::sort(indices.begin(), indices.end(), CompareFeature(feature));
//     
//     // Sort feature and target
//     NumericVector sorted_feature(n);
//     IntegerVector sorted_target(n);
//     for(int i = 0; i < n; ++i){
//       sorted_feature[i] = feature[indices[i]];
//       sorted_target[i] = target[indices[i]];
//     }
//     
//     // Initial prebinning
//     int actual_prebins = std::min(max_n_prebins, n);
//     int bin_size = n / actual_prebins;
//     if(bin_size == 0) bin_size = 1;
//     actual_prebins = std::min(max_n_prebins, (int)std::ceil((double)n / bin_size));
//     
//     std::vector<Bin> bins;
//     double lower = -std::numeric_limits<double>::infinity();
//     for(int i = 0; i < actual_prebins; ++i){
//       double upper;
//       if(i == actual_prebins - 1){
//         upper = std::numeric_limits<double>::infinity();
//       } else {
//         upper = sorted_feature[(i+1)*bin_size - 1];
//       }
//       bins.emplace_back(lower, upper);
//       lower = upper;
//     }
//     
//     // Assign counts to prebins
//     for(int i = 0; i < n; ++i){
//       double x = sorted_feature[i];
//       for(auto &bin : bins){
//         if(x > bin.lower_bound && x <= bin.upper_bound){
//           bin.count++;
//           if(sorted_target[i] == 1){
//             bin.count_pos++;
//           } else {
//             bin.count_neg++;
//           }
//           break;
//         }
//       }
//     }
//     
//     // Merge bins with counts below cutoff
//     double cutoff_count = bin_cutoff * n;
//     merge_bins_by_count(bins, cutoff_count);
//     
//     // Calculate total pos and neg
//     double total_pos = 0.0, total_neg = 0.0;
//     for(auto &bin : bins){
//       total_pos += bin.count_pos;
//       total_neg += bin.count_neg;
//     }
//     
//     // Calculate WoE and IV
//     calculate_woe_iv(bins, total_pos, total_neg);
//     
//     // Enforce monotonicity
//     bool merged_monotonic = true;
//     while(merged_monotonic){
//       merged_monotonic = enforce_monotonicity(bins, min_bins);
//       if(merged_monotonic){
//         calculate_woe_iv(bins, total_pos, total_neg);
//       }
//     }
//     
//     // Merge bins to meet max_bins
//     if(bins.size() > (size_t)max_bins){
//       merge_to_max_bins(bins, max_bins);
//       calculate_woe_iv(bins, total_pos, total_neg);
//     }
//     
//     // Ensure min_bins
//     int max_iterations = 100; // Prevent infinite loops
//     int iteration = 0;
//     while(bins.size() < (size_t)min_bins && bins.size() < (size_t)n && iteration < max_iterations){
//       iteration++;
//       // Split the largest bin
//       size_t split_idx = 0;
//       int max_count = 0;
//       for(size_t i = 0; i < bins.size(); ++i){
//         if(bins[i].count > max_count){
//           max_count = bins[i].count;
//           split_idx = i;
//         }
//       }
//       
//       // Find the split point
//       int split_pos = bins[split_idx].count / 2;
//       int cumulative_count = 0;
//       double split_value = 0.0;
//       for(int i = 0; i < n; ++i){
//         if(sorted_feature[i] > bins[split_idx].lower_bound &&
//            sorted_feature[i] <= bins[split_idx].upper_bound){
//           cumulative_count++;
//           if(cumulative_count == split_pos){
//             split_value = sorted_feature[i];
//             break;
//           }
//         }
//       }
//       
//       // Check if split_value is valid
//       if(split_value == bins[split_idx].lower_bound || split_value == bins[split_idx].upper_bound){
//         // Cannot split further
//         break;
//       }
//       
//       // Create new bins
//       Bin new_bin1(bins[split_idx].lower_bound, split_value);
//       Bin new_bin2(split_value, bins[split_idx].upper_bound);
//       
//       // Reassign counts
//       for(int i = 0; i < n; ++i){
//         double x = sorted_feature[i];
//         if(x > new_bin1.lower_bound && x <= new_bin1.upper_bound){
//           new_bin1.count++;
//           if(sorted_target[i] == 1){
//             new_bin1.count_pos++;
//           } else {
//             new_bin1.count_neg++;
//           }
//         } else if(x > new_bin2.lower_bound && x <= new_bin2.upper_bound){
//           new_bin2.count++;
//           if(sorted_target[i] == 1){
//             new_bin2.count_pos++;
//           } else {
//             new_bin2.count_neg++;
//           }
//         }
//       }
//       
//       // Replace the old bin with new bins
//       bins.erase(bins.begin() + split_idx);
//       bins.insert(bins.begin() + split_idx, new_bin2);
//       bins.insert(bins.begin() + split_idx, new_bin1);
//       
//       // Recalculate WoE and IV
//       calculate_woe_iv(bins, total_pos, total_neg);
//       
//       // Enforce monotonicity after split
//       merged_monotonic = true;
//       while(merged_monotonic){
//         merged_monotonic = enforce_monotonicity(bins, min_bins);
//         if(merged_monotonic){
//           calculate_woe_iv(bins, total_pos, total_neg);
//         }
//       }
//     }
//     if(iteration == max_iterations){
//       warning("Maximum iterations reached while ensuring min_bins. Possible conflicting constraints.");
//     }
//     
//     // Assign WoE values to observations
//     woefeature = NumericVector(n);
// #pragma omp parallel for num_threads(n_threads)
//     for(int i = 0; i < n; ++i){
//       double x = feature[i];
//       for(const auto &bin : bins){
//         if(x > bin.lower_bound && x <= bin.upper_bound){
//           woefeature[i] = bin.woe;
//           break;
//         }
//       }
//     }
//     
//     // Prepare woebin DataFrame
//     int n_bins = bins.size();
//     CharacterVector bin_char(n_bins);
//     NumericVector woe_vec(n_bins);
//     NumericVector iv_vec(n_bins);
//     IntegerVector count_vec(n_bins);
//     IntegerVector count_pos_vec(n_bins);
//     IntegerVector count_neg_vec(n_bins);
//     
//     for(int i = 0; i < n_bins; ++i){
//       bin_char[i] = generate_bin_label(bins[i]);
//       woe_vec[i] = bins[i].woe;
//       iv_vec[i] = bins[i].iv;
//       count_vec[i] = bins[i].count;
//       count_pos_vec[i] = bins[i].count_pos;
//       count_neg_vec[i] = bins[i].count_neg;
//     }
//     
//     woebin = DataFrame::create(
//       Named("bin") = bin_char,
//       Named("woe") = woe_vec,
//       Named("iv") = iv_vec,
//       Named("count") = count_vec,
//       Named("count_pos") = count_pos_vec,
//       Named("count_neg") = count_neg_vec
//     );
//   }
//   
//   NumericVector get_woefeature(){
//     return woefeature;
//   }
//   
//   DataFrame get_woebin(){
//     return woebin;
//   }
// };
// 
// //' @title Optimal Binning for Numerical Variables using MILP
// //'
// //' @description
// //' This function performs optimal binning for numerical variables using a Mixed Integer Linear Programming (MILP) inspired approach. It creates optimal bins for a numerical feature based on its relationship with a binary target variable, maximizing the predictive power while respecting user-defined constraints.
// //'
// //' @param target An integer vector of binary target values (0 or 1).
// //' @param feature A numeric vector of feature values.
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff Minimum proportion of total observations for a bin to avoid being merged (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins before the optimization process (default: 20).
// //' @param n_threads Number of threads to use for parallel processing (default: 1).
// //'
// //' @return A list containing two elements:
// //' \item{woefeature}{A numeric vector of Weight of Evidence (WoE) values for each observation.}
// //' \item{woebin}{A data frame with the following columns:
// //'   \itemize{
// //'     \item bin: Character vector of bin ranges.
// //'     \item woe: Numeric vector of WoE values for each bin.
// //'     \item iv: Numeric vector of Information Value (IV) for each bin.
// //'     \item count: Integer vector of total observations in each bin.
// //'     \item count_pos: Integer vector of positive target observations in each bin.
// //'     \item count_neg: Integer vector of negative target observations in each bin.
// //'   }
// //' }
// //'
// //' @details
// //' The Optimal Binning algorithm for numerical variables using a MILP-inspired approach works as follows:
// //' 1. Create initial pre-bins using equal-frequency binning.
// //' 2. Merge bins with counts below the cutoff.
// //' 3. Calculate initial Weight of Evidence (WoE) and Information Value (IV) for each bin.
// //' 4. Enforce monotonicity of WoE values across bins.
// //' 5. Merge bins to meet the max_bins constraint.
// //' 6. Split bins if necessary to meet the min_bins constraint.
// //' 7. Recalculate WoE and IV for the final bins.
// //'
// //' The algorithm aims to create bins that maximize the predictive power of the numerical variable while adhering to the specified constraints. It enforces monotonicity of WoE values, which is particularly useful for credit scoring and risk modeling applications.
// //'
// //' Weight of Evidence (WoE) is calculated as:
// //' \deqn{WoE = \ln(\frac{\text{Positive Rate}}{\text{Negative Rate}})}
// //'
// //' Information Value (IV) is calculated as:
// //' \deqn{IV = (\text{Positive Rate} - \text{Negative Rate}) \times WoE}
// //'
// //' This implementation uses OpenMP for parallel processing when available, which can significantly speed up the computation for large datasets.
// //'
// //' @references
// //' \itemize{
// //'   \item Belotti, P., Kirches, C., Leyffer, S., Linderoth, J., Luedtke, J., & Mahajan, A. (2013). Mixed-integer nonlinear optimization. Acta Numerica, 22, 1-131.
// //'   \item Mironchyk, P., & Tchistiakov, V. (2017). Monotone optimal binning algorithm for credit risk modeling. SSRN Electronic Journal. doi:10.2139/ssrn.2978774
// //' }
// //'
// //' @examples
// //' \dontrun{
// //' # Create sample data
// //' set.seed(123)
// //' n <- 1000
// //' target <- sample(0:1, n, replace = TRUE)
// //' feature <- rnorm(n)
// //'
// //' # Run optimal binning
// //' result <- optimal_binning_numerical_milp(target, feature, min_bins = 2, max_bins = 4)
// //'
// //' # Print results
// //' print(result$woebin)
// //'
// //' # Plot WoE values
// //' plot(result$woebin$woe, type = "s", xaxt = "n", xlab = "Bins", ylab = "WoE",
// //'      main = "Weight of Evidence by Bin")
// //' axis(1, at = 1:nrow(result$woebin), labels = result$woebin$bin, las = 2)
// //' }
// //'
// //' @usage
// //' optimal_binning_numerical_milp(target, feature, min_bins = 3, max_bins = 5,
// //'                                bin_cutoff = 0.05, max_n_prebins = 20, n_threads = 1)
// //'
// //' @author Lopes, J. E.
// //'
// //' @export
// // [[Rcpp::export]]
// List optimal_binning_numerical_milp(IntegerVector target,
//                                     NumericVector feature,
//                                     int min_bins = 3, int max_bins = 5,
//                                     double bin_cutoff = 0.05,
//                                     int max_n_prebins = 20,
//                                     int n_threads = 1) {
//   // Ensure that feature and target have the same length
//   if(feature.size() != target.size()){
//     stop("Feature and target vectors must have the same length.");
//   }
//   
//   // Ensure that target is binary
//   IntegerVector unique_targets = unique(target);
//   if(unique_targets.size() != 2){
//     stop("Target variable must be binary.");
//   }
//   
//   // Ensure min_bins and max_bins are valid
//   if(min_bins < 2){
//     warning("min_bins must be at least 2. Setting min_bins to 2.");
//     min_bins = 2;
//   }
//   if(max_bins < min_bins){
//     warning("max_bins must be greater than or equal to min_bins. Setting max_bins equal to min_bins.");
//     max_bins = min_bins;
//   }
//   
//   // Ensure bin_cutoff is between 0 and 1
//   if(bin_cutoff <= 0 || bin_cutoff >= 1){
//     warning("bin_cutoff must be between 0 and 1. Setting bin_cutoff to 0.05.");
//     bin_cutoff = 0.05;
//   }
//   
//   // Ensure max_n_prebins is at least max_bins
//   if(max_n_prebins < max_bins){
//     warning("max_n_prebins must be at least max_bins. Setting max_n_prebins equal to max_bins.");
//     max_n_prebins = max_bins;
//   }
//   
//   // Ensure n_threads is at least 1
//   if(n_threads < 1){
//     warning("n_threads must be at least 1. Setting n_threads to 1.");
//     n_threads = 1;
//   }
//   
//   // Initialize binning object
//   OptimalBinningNumericalMILP binning(feature, target, min_bins, max_bins, bin_cutoff, max_n_prebins, n_threads);
//   
//   // Fit the binning
//   binning.fit();
//   
//   // Return the results
//   return List::create(
//     Named("woefeature") = binning.get_woefeature(),
//     Named("woebin") = binning.get_woebin()
//   );
// }
