#include <Rcpp.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <random>
#include <cmath>
#include <limits>
#include <mutex>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

// Mutex for critical section in OpenMP
std::mutex mtx;

// Constants
static constexpr double EPSILON = 1e-10;

// Class for Optimal Binning using Simulated Annealing
class OptimalBinningCategoricalSAB {
private:
  // Input data
  std::vector<std::string> feature;
  std::vector<int> target;
  
  // Binning parameters
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  
  // Category statistics
  std::vector<std::string> unique_categories;
  std::unordered_map<std::string, int> category_counts;
  std::unordered_map<std::string, int> positive_counts;
  int total_count;
  int total_positive;
  int total_negative;
  
  // Binning results
  int actual_bins;
  
  // Simulated Annealing parameters
  double initial_temperature;
  double cooling_rate;
  int max_iterations;
  
  std::vector<int> current_solution;
  std::vector<int> best_solution;
  double current_iv;
  double best_iv;
  
  // Random number generators for each thread
  std::vector<std::mt19937> generators;
  
  // Helper function to initialize category statistics and initial solution
  void initialize() {
    // Extract unique categories
    std::unordered_set<std::string> unique_set(feature.begin(), feature.end());
    unique_categories = std::vector<std::string>(unique_set.begin(), unique_set.end());
    
    // Count total observations and positives/negatives
    total_count = feature.size();
    total_positive = std::count(target.begin(), target.end(), 1);
    total_negative = total_count - total_positive;
    
    // Count occurrences and positives per category
    for (size_t i = 0; i < feature.size(); ++i) {
      category_counts[feature[i]]++;
      if (target[i] == 1) {
        positive_counts[feature[i]]++;
      }
    }
    
    int n_categories = unique_categories.size();
    
    // Adjust actual_bins based on the number of unique categories
    if (n_categories <= 2) {
      actual_bins = n_categories;
    } else {
      actual_bins = std::min(std::max(min_bins, 3), max_bins);
      actual_bins = std::min(actual_bins, n_categories); // Cannot have more bins than categories
    }
    
    // Initialize current solution: assign categories to bins in a round-robin fashion
    current_solution.resize(n_categories);
    for (size_t i = 0; i < current_solution.size(); ++i) {
      current_solution[i] = i % actual_bins;
    }
    
    // Shuffle the initial solution to start with a random assignment
    std::random_device rd;
    std::mt19937 temp_gen(rd());
    std::shuffle(current_solution.begin(), current_solution.end(), temp_gen);
    
    // Set the best solution initially
    best_solution = current_solution;
    current_iv = calculate_iv(current_solution);
    best_iv = current_iv;
  }
  
  // Calculate Information Value (IV) for a given binning solution
  double calculate_iv(const std::vector<int>& solution) const {
    std::vector<int> bin_counts(actual_bins, 0);
    std::vector<int> bin_positives(actual_bins, 0);
    
    // Aggregate counts per bin
    for (size_t i = 0; i < unique_categories.size(); ++i) {
      const std::string& category = unique_categories[i];
      int bin = solution[i];
      bin_counts[bin] += category_counts.at(category);
      bin_positives[bin] += positive_counts.at(category);
    }
    
    double iv = 0.0;
    
    // Calculate IV based on WoE for each bin
    for (int i = 0; i < actual_bins; ++i) {
      if (bin_counts[i] > 0) {
        int bin_negatives = bin_counts[i] - bin_positives[i];
        double pos_dist = static_cast<double>(bin_positives[i]) / total_positive;
        double neg_dist = static_cast<double>(bin_negatives) / total_negative;
        
        // Enforce bin_cutoff
        double bin_proportion = static_cast<double>(bin_counts[i]) / total_count;
        if (bin_proportion < bin_cutoff) {
          // Penalize bins that do not meet the bin_cutoff
          iv -= 1000.0; // Large negative penalty
          continue;
        }
        
        if (pos_dist > EPSILON && neg_dist > EPSILON) {
          double woe = std::log(pos_dist / neg_dist);
          iv += (pos_dist - neg_dist) * woe;
        }
      }
    }
    
    return iv;
  }
  
  // Check if the binning solution is monotonic based on positive rates
  bool is_monotonic(const std::vector<int>& solution) const {
    std::vector<double> bin_rates(actual_bins, 0.0);
    std::vector<int> bin_counts(actual_bins, 0);
    
    // Aggregate counts and positive rates per bin
    for (size_t i = 0; i < unique_categories.size(); ++i) {
      const std::string& category = unique_categories[i];
      int bin = solution[i];
      bin_counts[bin] += category_counts.at(category);
      bin_rates[bin] += positive_counts.at(category);
    }
    
    // Calculate positive rate for each bin
    for (int i = 0; i < actual_bins; ++i) {
      if (bin_counts[i] > 0) {
        bin_rates[i] /= bin_counts[i];
      }
    }
    
    // Allow non-monotonicity for features with two or fewer categories
    if (unique_categories.size() <= 2) {
      return true;
    }
    
    // Check for monotonic increasing order
    bool increasing = true;
    bool decreasing = true;
    for (int i = 1; i < actual_bins; ++i) {
      if (bin_rates[i] < bin_rates[i - 1] - EPSILON) {
        increasing = false;
      }
      if (bin_rates[i] > bin_rates[i - 1] + EPSILON) {
        decreasing = false;
      }
    }
    
    return increasing || decreasing;
  }
  
  // Generate a neighboring solution by swapping two category assignments
  std::vector<int> generate_neighbor(const std::vector<int>& solution, std::mt19937& gen_local) const {
    std::vector<int> neighbor = solution;
    if (neighbor.empty()) return neighbor;
    
    std::uniform_int_distribution<> dis(0, neighbor.size() - 1);
    int idx1 = dis(gen_local);
    int idx2 = dis(gen_local);
    
    // Ensure two different indices
    while (idx2 == idx1 && neighbor.size() > 1) {
      idx2 = dis(gen_local);
    }
    
    std::swap(neighbor[idx1], neighbor[idx2]);
    return neighbor;
  }
  
public:
  // Constructor
  OptimalBinningCategoricalSAB(const std::vector<std::string>& feature_,
                               const std::vector<int>& target_,
                               int min_bins_ = 3,
                               int max_bins_ = 5,
                               double bin_cutoff_ = 0.05,
                               int max_n_prebins_ = 20,
                               double initial_temperature_ = 1.0,
                               double cooling_rate_ = 0.995,
                               int max_iterations_ = 1000)
    : feature(feature_), target(target_), min_bins(min_bins_), max_bins(max_bins_),
      bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_),
      initial_temperature(initial_temperature_), cooling_rate(cooling_rate_),
      max_iterations(max_iterations_) {
    
    // Input validation
    if (feature.size() != target.size()) {
      Rcpp::stop("Feature and target vectors must have the same length");
    }
    if (min_bins < 2) {
      Rcpp::stop("min_bins must be at least 2");
    }
    if (max_bins < min_bins) {
      Rcpp::stop("max_bins must be greater than or equal to min_bins");
    }
    if (bin_cutoff <= 0 || bin_cutoff >= 1) {
      Rcpp::stop("bin_cutoff must be between 0 and 1");
    }
    if (max_n_prebins < max_bins) {
      Rcpp::stop("max_n_prebins must be greater than or equal to max_bins");
    }
    
    // Initialize category statistics and initial solution
    initialize();
  }
  
  // Fit the binning solution using Simulated Annealing
  void fit() {
    double temperature = initial_temperature;
    int n_categories = unique_categories.size();
    
    // Initialize thread-local random number generators
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif
    generators.resize(n_threads);
    std::random_device rd;
    for (int i = 0; i < n_threads; ++i) {
      generators[i].seed(rd() + i);
    }
    
    // OpenMP parallel region
#pragma omp parallel
{
  int thread_id = 0;
#ifdef _OPENMP
  thread_id = omp_get_thread_num();
#endif
  std::mt19937& gen_local = generators[thread_id];
  
  std::vector<int> local_best_solution = best_solution;
  double local_best_iv = best_iv;
  
  for (int iter = 0; iter < max_iterations; ++iter) {
    // Generate a neighboring solution
    std::vector<int> neighbor = generate_neighbor(local_best_solution, gen_local);
    double neighbor_iv = calculate_iv(neighbor);
    
    // Check for monotonicity
    if (!is_monotonic(neighbor)) {
      continue; // Skip non-monotonic solutions
    }
    
    // Acceptance criteria
    if (neighbor_iv > local_best_iv) {
      local_best_solution = neighbor;
      local_best_iv = neighbor_iv;
    } else {
      double acceptance_probability = std::exp((neighbor_iv - local_best_iv) / temperature);
      std::uniform_real_distribution<> dis(0.0, 1.0);
      if (dis(gen_local) < acceptance_probability) {
        local_best_solution = neighbor;
        local_best_iv = neighbor_iv;
      }
    }
    
    // Cool down the temperature
    temperature *= cooling_rate;
  }
  
  // Update the global best solution in a thread-safe manner
#pragma omp critical
{
  if (local_best_iv > best_iv) {
    best_solution = local_best_solution;
    best_iv = local_best_iv;
  }
}
}

// Ensure the final solution is monotonic with a limited number of attempts
int max_attempts = 100;
int attempts = 0;
while (!is_monotonic(best_solution) && attempts < max_attempts) {
  best_solution = generate_neighbor(best_solution, generators[0]);
  best_iv = calculate_iv(best_solution);
  attempts++;
}

if (attempts == max_attempts) {
  Rcpp::warning("Failed to achieve monotonicity after maximum attempts.");
}
  }
  
  // Retrieve the binning results
  List get_results() const {
    std::vector<std::string> bins;
    std::vector<double> woe;
    std::vector<double> iv;
    std::vector<int> count;
    std::vector<int> count_pos;
    std::vector<int> count_neg;
    
    // Aggregate categories into bins
    std::vector<std::vector<std::string>> bin_categories(actual_bins, std::vector<std::string>());
    std::vector<int> bin_counts(actual_bins, 0);
    std::vector<int> bin_positives(actual_bins, 0);
    
    for (size_t i = 0; i < unique_categories.size(); ++i) {
      const std::string& category = unique_categories[i];
      int bin = best_solution[i];
      bin_categories[bin].push_back(category);
      bin_counts[bin] += category_counts.at(category);
      bin_positives[bin] += positive_counts.at(category);
    }
    
    double total_iv = 0.0;
    
    // Calculate WoE and IV for each bin
    for (int i = 0; i < actual_bins; ++i) {
      if (bin_counts[i] > 0) {
        // Create bin name by concatenating categories with "+"
        std::string bin_name = "";
        for (const auto& category : bin_categories[i]) {
          if (!bin_name.empty()) bin_name += "+";
          bin_name += category;
        }
        bins.push_back(bin_name);
        count.push_back(bin_counts[i]);
        count_pos.push_back(bin_positives[i]);
        int bin_negatives = bin_counts[i] - bin_positives[i];
        count_neg.push_back(bin_negatives);
        
        double pos_dist = static_cast<double>(bin_positives[i]) / total_positive;
        double neg_dist = static_cast<double>(bin_negatives) / total_negative;
        
        double woe_value = 0.0;
        double iv_value = 0.0;
        
        if (pos_dist > EPSILON && neg_dist > EPSILON) {
          woe_value = std::log(pos_dist / neg_dist);
          iv_value = (pos_dist - neg_dist) * woe_value;
          total_iv += iv_value;
        }
        
        woe.push_back(woe_value);
        iv.push_back(iv_value);
      }
    }
    
    // Handle case with no valid bins
    size_t n = bins.size();
    if (n == 0) {
      return List::create(
        Named("woefeature") = NumericVector(feature.size()),
        Named("woebin") = DataFrame::create(),
        Named("total_iv") = 0.0
      );
    }
    
    // Create woebin DataFrame
    DataFrame woebin = DataFrame::create(
      Named("bin") = bins,
      Named("woe") = woe,
      Named("iv") = iv,
      Named("count") = count,
      Named("count_pos") = count_pos,
      Named("count_neg") = count_neg
    );
    
    // Create a map from category to bin index for efficient lookup
    std::unordered_map<std::string, int> category_to_bin;
    for (size_t i = 0; i < unique_categories.size(); ++i) {
      category_to_bin[unique_categories[i]] = best_solution[i];
    }
    
    // Assign WoE to each feature value based on its bin
    std::vector<double> woefeature(feature.size());
    for (size_t i = 0; i < feature.size(); ++i) {
      auto it = category_to_bin.find(feature[i]);
      if (it != category_to_bin.end()) {
        int bin = it->second;
        if (bin < static_cast<int>(woe.size())) {
          woefeature[i] = woe[bin];
        } else {
          woefeature[i] = 0.0; // Default WoE for out-of-range bins
        }
      } else {
        woefeature[i] = 0.0; // Default WoE for unknown categories
      }
    }
    
    return List::create(
      Named("woefeature") = woefeature,
      Named("woebin") = woebin,
      Named("total_iv") = total_iv
    );
  }
};

//' @title
//' Optimal Binning for Categorical Variables using Simulated Annealing
//'
//' @description
//' This function performs optimal binning for categorical variables using a Simulated Annealing approach.
//' It maximizes the Information Value (IV) while maintaining monotonicity in the bins.
//'
//' @param target An integer vector of binary target values (0 or 1).
//' @param feature A character vector of categorical feature values.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum proportion of observations in a bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins (default: 20).
//' @param initial_temperature Initial temperature for Simulated Annealing (default: 1.0).
//' @param cooling_rate Cooling rate for Simulated Annealing (default: 0.995).
//' @param max_iterations Maximum number of iterations for Simulated Annealing (default: 1000).
//'
//' @return A list containing three elements:
//' \itemize{
//'   \item woefeature: A numeric vector of Weight of Evidence (WoE) values for each observation
//'   \item woebin: A data frame containing binning information, including bin names, WoE, Information Value (IV), and counts
//'   \item total_iv: The total Information Value for the binning solution
//' }
//'
//' @examples
//' \dontrun{
//' set.seed(123)
//' target <- sample(0:1, 1000, replace = TRUE)
//' feature <- sample(LETTERS[1:5], 1000, replace = TRUE)
//' result <- optimal_binning_categorical_sab(target, feature)
//' print(result$woebin)
//' print(result$total_iv)
//' }
//'
//' @details
//' The algorithm uses Simulated Annealing to find an optimal binning solution that maximizes
//' the Information Value while maintaining monotonicity. It respects the specified constraints
//' on the number of bins and bin sizes.
//'
//' The Weight of Evidence (WoE) is calculated as:
//' \deqn{WoE_i = \ln(\frac{\text{Distribution of positives}_i}{\text{Distribution of negatives}_i})}
//'
//' Where:
//' \deqn{\text{Distribution of positives}_i = \frac{\text{Number of positives in bin } i}{\text{Total Number of positives}}}
//' \deqn{\text{Distribution of negatives}_i = \frac{\text{Number of negatives in bin } i}{\text{Total Number of negatives}}}
//'
//' The Information Value (IV) is calculated as:
//' \deqn{IV = \sum_{i=1}^{N} (\text{Distribution of positives}_i - \text{Distribution of negatives}_i) \times WoE_i}
//'
//' The algorithm uses OpenMP for parallel processing to improve performance on multi-core systems.
//'
//' @references
//' \itemize{
//'   \item Bertsimas, D., & Dunn, J. (2017). Optimal classification trees. Machine Learning, 106(7), 1039-1082.
//'   \item Mironchyk, P., & Tchistiakov, V. (2017). Monotone optimal binning algorithm for credit risk modeling.
//'         Workshop on Data Science and Advanced Analytics (DSAA).
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_sab(Rcpp::IntegerVector target,
                                          Rcpp::StringVector feature,
                                          int min_bins = 3,
                                          int max_bins = 5,
                                          double bin_cutoff = 0.05,
                                          int max_n_prebins = 20,
                                          double initial_temperature = 1.0,
                                          double cooling_rate = 0.995,
                                          int max_iterations = 1000) {
 // Input validation
 if (target.size() != feature.size()) {
   Rcpp::stop("Target and feature vectors must have the same length");
 }
 if (min_bins < 2) {
   Rcpp::stop("min_bins must be at least 2");
 }
 if (max_bins < min_bins) {
   Rcpp::stop("max_bins must be greater than or equal to min_bins");
 }
 if (bin_cutoff <= 0 || bin_cutoff >= 1) {
   Rcpp::stop("bin_cutoff must be between 0 and 1");
 }
 if (max_n_prebins < max_bins) {
   Rcpp::stop("max_n_prebins must be greater than or equal to max_bins");
 }
 
 // Convert Rcpp vectors to C++ vectors
 std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
 std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
 
 // Create and fit the binning model
 OptimalBinningCategoricalSAB binner(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins,
                                     initial_temperature, cooling_rate, max_iterations);
 binner.fit();
 
 // Retrieve and return the results
 return binner.get_results();
}


// #include <Rcpp.h>
// #include <vector>
// #include <string>
// #include <unordered_map>
// #include <algorithm>
// #include <random>
// #include <cmath>
// 
// #ifdef _OPENMP
// #include <omp.h>
// #endif
// 
// class OptimalBinningCategoricalSAB {
// private:
//   std::vector<std::string> feature;
//   std::vector<int> target;
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
//   
//   std::vector<std::string> unique_categories;
//   std::unordered_map<std::string, int> category_counts;
//   std::unordered_map<std::string, int> positive_counts;
//   int total_count;
//   int total_positive;
//   int actual_bins;
//   
//   // Simulated Annealing parameters
//   double initial_temperature;
//   double cooling_rate;
//   int max_iterations;
//   
//   std::vector<int> current_solution;
//   std::vector<int> best_solution;
//   double current_iv;
//   double best_iv;
//   
//   // Random number generator
//   std::mt19937 gen;
//   
//   void initialize() {
//     std::unordered_set<std::string> unique_set(feature.begin(), feature.end());
//     unique_categories = std::vector<std::string>(unique_set.begin(), unique_set.end());
//     
//     total_count = feature.size();
//     total_positive = std::count(target.begin(), target.end(), 1);
//     
//     for (size_t i = 0; i < feature.size(); ++i) {
//       category_counts[feature[i]]++;
//       if (target[i] == 1) {
//         positive_counts[feature[i]]++;
//       }
//     }
//     
//     // Adjust actual_bins based on the number of unique categories
//     actual_bins = std::min(std::max(static_cast<int>(unique_categories.size()), min_bins), max_bins);
//     
//     // Initialize current solution
//     current_solution.resize(unique_categories.size());
//     for (size_t i = 0; i < current_solution.size(); ++i) {
//       current_solution[i] = i % actual_bins;
//     }
//     std::shuffle(current_solution.begin(), current_solution.end(), gen);
//     
//     best_solution = current_solution;
//     current_iv = calculate_iv(current_solution);
//     best_iv = current_iv;
//   }
//   
//   double calculate_iv(const std::vector<int>& solution) {
//     std::vector<int> bin_counts(actual_bins, 0);
//     std::vector<int> bin_positives(actual_bins, 0);
//     
//     for (size_t i = 0; i < unique_categories.size(); ++i) {
//       const std::string& category = unique_categories[i];
//       int bin = solution[i];
//       bin_counts[bin] += category_counts[category];
//       bin_positives[bin] += positive_counts[category];
//     }
//     
//     double iv = 0.0;
//     for (int i = 0; i < actual_bins; ++i) {
//       if (bin_counts[i] > 0) {
//         double bin_rate = static_cast<double>(bin_positives[i]) / bin_counts[i];
//         double overall_rate = static_cast<double>(total_positive) / total_count;
//         if (bin_rate > 0 && bin_rate < 1) {
//           double woe = std::log(bin_rate / (1 - bin_rate) * (1 - overall_rate) / overall_rate);
//           iv += (bin_rate - (1 - bin_rate)) * woe;
//         }
//       }
//     }
//     
//     return std::max(0.0, iv);  // Ensure we always return a non-negative value
//   }
//   
//   bool is_monotonic(const std::vector<int>& solution) {
//     std::vector<double> bin_rates(actual_bins, 0.0);
//     std::vector<int> bin_counts(actual_bins, 0);
//     
//     for (size_t i = 0; i < unique_categories.size(); ++i) {
//       const std::string& category = unique_categories[i];
//       int bin = solution[i];
//       bin_counts[bin] += category_counts[category];
//       bin_rates[bin] += positive_counts[category];
//     }
//     
//     for (int i = 0; i < actual_bins; ++i) {
//       if (bin_counts[i] > 0) {
//         bin_rates[i] /= bin_counts[i];
//       }
//     }
//     
//     // Remove empty bins
//     bin_rates.erase(std::remove(bin_rates.begin(), bin_rates.end(), 0.0), bin_rates.end());
//     
//     for (size_t i = 1; i < bin_rates.size(); ++i) {
//       if (bin_rates[i] < bin_rates[i-1]) {
//         return false;
//       }
//     }
//     
//     return true;
//   }
//   
//   std::vector<int> generate_neighbor(const std::vector<int>& solution) {
//     std::vector<int> neighbor = solution;
//     int idx = std::uniform_int_distribution<>(0, neighbor.size() - 1)(gen);
//     int new_bin = std::uniform_int_distribution<>(0, actual_bins - 1)(gen);
//     neighbor[idx] = new_bin;
//     return neighbor;
//   }
//   
// public:
//   OptimalBinningCategoricalSAB(const std::vector<std::string>& feature,
//                                const std::vector<int>& target,
//                                int min_bins = 3,
//                                int max_bins = 5,
//                                double bin_cutoff = 0.05,
//                                int max_n_prebins = 20,
//                                double initial_temperature = 1.0,
//                                double cooling_rate = 0.995,
//                                int max_iterations = 1000)
//     : feature(feature), target(target), min_bins(min_bins), max_bins(max_bins),
//       bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins),
//       initial_temperature(initial_temperature), cooling_rate(cooling_rate),
//       max_iterations(max_iterations), gen(std::random_device()()) {
//     
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
//     if (max_n_prebins < max_bins) {
//       Rcpp::stop("max_n_prebins must be greater than or equal to max_bins");
//     }
//     
//     initialize();
//   }
//   
//   void fit() {
//     double temperature = initial_temperature;
//     
// #pragma omp parallel
// {
//   std::vector<int> local_best_solution = best_solution;
//   double local_best_iv = best_iv;
//   
// #pragma omp for
//   for (int iter = 0; iter < max_iterations; ++iter) {
//     std::vector<int> neighbor = generate_neighbor(local_best_solution);
//     double neighbor_iv = calculate_iv(neighbor);
//     
//     if (neighbor_iv > local_best_iv && is_monotonic(neighbor)) {
//       local_best_solution = neighbor;
//       local_best_iv = neighbor_iv;
//     } else {
//       double acceptance_probability = std::exp((neighbor_iv - local_best_iv) / temperature);
//       if (std::uniform_real_distribution<>(0, 1)(gen) < acceptance_probability && is_monotonic(neighbor)) {
//         local_best_solution = neighbor;
//         local_best_iv = neighbor_iv;
//       }
//     }
//     
//     temperature *= cooling_rate;
//   }
//   
// #pragma omp critical
// {
//   if (local_best_iv > best_iv) {
//     best_solution = local_best_solution;
//     best_iv = local_best_iv;
//   }
// }
// }
// 
// // Ensure final solution is monotonic
// while (!is_monotonic(best_solution)) {
//   best_solution = generate_neighbor(best_solution);
// }
// best_iv = calculate_iv(best_solution);  // Recalculate best_iv to ensure consistency
//   }
//   
//   Rcpp::List get_results() {
//     std::vector<std::string> bins;
//     std::vector<double> woe;
//     std::vector<double> iv;
//     std::vector<int> count;
//     std::vector<int> count_pos;
//     std::vector<int> count_neg;
//     
//     std::vector<std::vector<std::string>> bin_categories(actual_bins);
//     std::vector<int> bin_counts(actual_bins, 0);
//     std::vector<int> bin_positives(actual_bins, 0);
//     
//     for (size_t i = 0; i < unique_categories.size(); ++i) {
//       const std::string& category = unique_categories[i];
//       int bin = best_solution[i];
//       bin_categories[bin].push_back(category);
//       bin_counts[bin] += category_counts[category];
//       bin_positives[bin] += positive_counts[category];
//     }
//     
//     for (int i = 0; i < actual_bins; ++i) {
//       if (bin_counts[i] > 0) {
//         std::string bin_name = "";
//         for (const auto& category : bin_categories[i]) {
//           if (!bin_name.empty()) bin_name += "+";
//           bin_name += category;
//         }
//         bins.push_back(bin_name);
//         count.push_back(bin_counts[i]);
//         count_pos.push_back(bin_positives[i]);
//         count_neg.push_back(bin_counts[i] - bin_positives[i]);
//         
//         double bin_rate = static_cast<double>(bin_positives[i]) / bin_counts[i];
//         double overall_rate = static_cast<double>(total_positive) / total_count;
//         double woe_value = std::log(bin_rate / (1 - bin_rate) * (1 - overall_rate) / overall_rate);
//         woe.push_back(woe_value);
//         
//         double iv_value = (bin_rate - (1 - bin_rate)) * woe_value;
//         iv.push_back(iv_value);
//       }
//     }
//     
//     // Ensure all vectors have the same length
//     size_t n = bins.size();
//     if (n == 0) {
//       // Return empty DataFrame if no bins
//       return Rcpp::List::create(
//         Rcpp::Named("woefeature") = Rcpp::NumericVector(feature.size()),
//         Rcpp::Named("woebin") = Rcpp::DataFrame::create()
//       );
//     }
//     
//     Rcpp::DataFrame woebin = Rcpp::DataFrame::create(
//       Rcpp::Named("bin") = bins,
//       Rcpp::Named("woe") = woe,
//       Rcpp::Named("iv") = iv,
//       Rcpp::Named("count") = count,
//       Rcpp::Named("count_pos") = count_pos,
//       Rcpp::Named("count_neg") = count_neg
//     );
//     
//     std::vector<double> woefeature(feature.size());
//     for (size_t i = 0; i < feature.size(); ++i) {
//       auto it = std::find(unique_categories.begin(), unique_categories.end(), feature[i]);
//       if (it != unique_categories.end()) {
//         int bin = best_solution[it - unique_categories.begin()];
//         woefeature[i] = woe[bin];
//       } else {
//         woefeature[i] = 0.0;  // Or some other default value
//       }
//     }
//     
//     return Rcpp::List::create(
//       Rcpp::Named("woefeature") = woefeature,
//       Rcpp::Named("woebin") = woebin
//     );
//   }
// };
// 
// //' @title
// //' Optimal Binning for Categorical Variables using Simulated Annealing
// //'
// //' @description
// //' This function performs optimal binning for categorical variables using a Simulated Annealing approach.
// //'
// //' @param target An integer vector of binary target values (0 or 1).
// //' @param feature A character vector of categorical feature values.
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff Minimum proportion of observations in a bin (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins (default: 20).
// //' @param initial_temperature Initial temperature for Simulated Annealing (default: 1.0).
// //' @param cooling_rate Cooling rate for Simulated Annealing (default: 0.995).
// //' @param max_iterations Maximum number of iterations for Simulated Annealing (default: 1000).
// //'
// //' @return A list containing two elements:
// //' \itemize{
// //'   \item woefeature: A numeric vector of Weight of Evidence (WoE) values for each observation
// //'   \item woebin: A data frame containing binning information, including bin names, WoE, Information Value (IV), and counts
// //' }
// //'
// //' @examples
// //' \dontrun{
// //' # Create sample data
// //' set.seed(123)
// //' target <- sample(0:1, 1000, replace = TRUE)
// //' feature <- sample(LETTERS[1:5], 1000, replace = TRUE)
// //'
// //' # Run optimal binning
// //' result <- optimal_binning_categorical_sab(target, feature)
// //'
// //' # View results
// //' print(result$woebin)
// //' }
// //'
// //' @details
// //' This algorithm performs optimal binning for categorical variables using Simulated Annealing (SA). 
// //' The process aims to maximize the Information Value (IV) while maintaining monotonicity in the bins.
// //'
// //' The algorithm works as follows:
// //' \enumerate{
// //'   \item Initialize by assigning each unique category to a random bin.
// //'   \item Calculate the initial IV.
// //'   \item In each iteration of SA:
// //'     \enumerate{
// //'       \item Generate a neighbor solution by randomly reassigning a category to a different bin.
// //'       \item Calculate the IV of the new solution.
// //'       \item Accept the new solution if it improves IV and maintains monotonicity.
// //'       \item If the new solution is worse, accept it with a probability based on the current temperature.
// //'     }
// //'   \item Repeat step 3 for a specified number of iterations, reducing the temperature each time.
// //'   \item Ensure the final solution is monotonic.
// //' }
// //'
// //' The Information Value (IV) is calculated as:
// //' \deqn{IV = \sum(\text{% of non-events} - \text{% of events}) \times WoE}
// //'
// //' Where Weight of Evidence (WoE) is:
// //' \deqn{WoE = \ln(\frac{\text{% of events}}{\text{% of non-events}})}
// //'
// //' The algorithm uses OpenMP for parallel processing to improve performance.
// //'
// //' @references
// //' \itemize{
// //'   \item Bertsimas, D., & Dunn, J. (2017). Optimal classification trees. Machine Learning, 106(7), 1039-1082.
// //'   \item Mironchyk, P., & Tchistiakov, V. (2017). Monotone optimal binning algorithm for credit risk modeling. 
// //'         In Workshop on Data Science and Advanced Analytics (DSAA). 
// //' }
// //'
// //' @author Lopes, J. E.
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_categorical_sab(Rcpp::IntegerVector target,
//                                            Rcpp::StringVector feature,
//                                            int min_bins = 3,
//                                            int max_bins = 5,
//                                            double bin_cutoff = 0.05,
//                                            int max_n_prebins = 20,
//                                            double initial_temperature = 1.0,
//                                            double cooling_rate = 0.995,
//                                            int max_iterations = 1000) {
//   std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
//   std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
//   
//   OptimalBinningCategoricalSAB binner(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins,
//                                       initial_temperature, cooling_rate, max_iterations);
//   binner.fit();
//   return binner.get_results();
// }
