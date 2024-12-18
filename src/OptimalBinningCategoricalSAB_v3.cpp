// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <random>
#include <cmath>
#include <limits>
#include <stdexcept>

using namespace Rcpp;

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
 std::string bin_separator;
 
 // Simulated Annealing parameters
 double initial_temperature;
 double cooling_rate;
 int max_iterations;
 double convergence_threshold;
 
 // Category statistics
 std::vector<std::string> unique_categories;
 std::unordered_map<std::string, int> category_counts;
 std::unordered_map<std::string, int> positive_counts;
 int total_count;
 int total_positive;
 int total_negative;
 
 // Binning results
 int actual_bins;
 std::vector<int> best_solution;
 double best_iv;
 bool converged;
 int iterations_run;
 
 // Random number generator
 std::mt19937 gen;
 
 void initialize() {
   // Extract unique categories
   std::unordered_set<std::string> unique_set(feature.begin(), feature.end());
   unique_categories = std::vector<std::string>(unique_set.begin(), unique_set.end());
   
   // Count totals
   total_count = (int)feature.size();
   total_positive = std::count(target.begin(), target.end(), 1);
   total_negative = total_count - total_positive;
   
   // Count per category
   for (size_t i = 0; i < feature.size(); ++i) {
     category_counts[feature[i]]++;
     if (target[i] == 1) {
       positive_counts[feature[i]]++;
     }
   }
   
   int n_categories = (int)unique_categories.size();
   actual_bins = std::min(std::max(min_bins, std::min(max_bins, n_categories)), n_categories);
   
   // Initial solution: round-robin assignment
   best_solution.resize(n_categories);
   for (size_t i = 0; i < best_solution.size(); ++i) {
     best_solution[i] = (int)(i % actual_bins);
   }
   
   // Shuffle initial solution
   std::shuffle(best_solution.begin(), best_solution.end(), gen);
   
   best_iv = calculate_iv(best_solution);
   converged = false;
   iterations_run = 0;
 }
 
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
   
   for (int i = 0; i < actual_bins; ++i) {
     if (bin_counts[i] > 0) {
       int bin_negatives = bin_counts[i] - bin_positives[i];
       double pos_dist = (double)bin_positives[i] / total_positive;
       double neg_dist = (double)bin_negatives / total_negative;
       
       double bin_proportion = (double)bin_counts[i] / total_count;
       if (bin_proportion < bin_cutoff) {
         // Penalize small bins
         iv -= 1000.0;
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
 
 bool is_monotonic(const std::vector<int>& solution) const {
   std::vector<double> bin_rates(actual_bins, 0.0);
   std::vector<int> bin_counts(actual_bins, 0);
   
   for (size_t i = 0; i < unique_categories.size(); ++i) {
     const std::string& category = unique_categories[i];
     int bin = solution[i];
     bin_counts[bin] += category_counts.at(category);
     bin_rates[bin] += positive_counts.at(category);
   }
   
   for (int i = 0; i < actual_bins; ++i) {
     if (bin_counts[i] > 0) {
       bin_rates[i] /= bin_counts[i];
     }
   }
   
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
 
 std::vector<int> generate_neighbor(const std::vector<int>& solution) {
   std::vector<int> neighbor = solution;
   if (neighbor.size() <= 1) return neighbor;
   
   std::uniform_int_distribution<> dis(0, (int)neighbor.size() - 1);
   int idx1 = dis(gen);
   int idx2 = dis(gen);
   
   while (idx2 == idx1) {
     idx2 = dis(gen);
   }
   
   std::swap(neighbor[idx1], neighbor[idx2]);
   return neighbor;
 }
 
public:
 OptimalBinningCategoricalSAB(const std::vector<std::string>& feature_,
                              const std::vector<int>& target_,
                              int min_bins_ = 3,
                              int max_bins_ = 5,
                              double bin_cutoff_ = 0.05,
                              int max_n_prebins_ = 20,
                              std::string bin_separator_ = "%;%",
                              double initial_temperature_ = 1.0,
                              double cooling_rate_ = 0.995,
                              int max_iterations_ = 1000,
                              double convergence_threshold_ = 1e-6)
   : feature(feature_), target(target_), min_bins(min_bins_), max_bins(max_bins_),
     bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_), bin_separator(bin_separator_),
     initial_temperature(initial_temperature_), cooling_rate(cooling_rate_),
     max_iterations(max_iterations_), convergence_threshold(convergence_threshold_) {
   
   if (feature.size() != target.size()) {
     throw std::invalid_argument("Feature and target vectors must have the same length");
   }
   if (min_bins < 2) {
     throw std::invalid_argument("min_bins must be at least 2");
   }
   if (max_bins < min_bins) {
     throw std::invalid_argument("max_bins must be >= min_bins");
   }
   if (bin_cutoff <= 0 || bin_cutoff >= 1) {
     throw std::invalid_argument("bin_cutoff must be between 0 and 1");
   }
   if (max_n_prebins < max_bins) {
     throw std::invalid_argument("max_n_prebins must be >= max_bins");
   }
   
   std::random_device rd;
   gen.seed(rd());
   initialize();
 }
 
 void fit() {
   double temperature = initial_temperature;
   double prev_best_iv = best_iv;
   
   for (int iter = 0; iter < max_iterations; ++iter) {
     std::vector<int> neighbor = generate_neighbor(best_solution);
     double neighbor_iv = calculate_iv(neighbor);
     
     if (actual_bins > min_bins && !is_monotonic(neighbor)) {
       continue; // Skip non-monotonic
     }
     
     if (neighbor_iv > best_iv) {
       best_solution = neighbor;
       best_iv = neighbor_iv;
     } else {
       double acceptance_probability = std::exp((neighbor_iv - best_iv) / temperature);
       std::uniform_real_distribution<> dis(0.0, 1.0);
       if (dis(gen) < acceptance_probability) {
         best_solution = neighbor;
         best_iv = neighbor_iv;
       }
     }
     
     temperature *= cooling_rate;
     
     if (std::abs(best_iv - prev_best_iv) < convergence_threshold) {
       converged = true;
       iterations_run = iter + 1;
       break;
     }
     
     prev_best_iv = best_iv;
   }
   
   if (!converged) {
     iterations_run = max_iterations;
   }
   
   // Ensure monotonicity post-hoc if needed
   if (actual_bins > min_bins) {
     int max_monotonic_attempts = 100;
     for (int attempt = 0; attempt < max_monotonic_attempts; ++attempt) {
       if (is_monotonic(best_solution)) {
         break;
       }
       best_solution = generate_neighbor(best_solution);
       best_iv = calculate_iv(best_solution);
     }
   }
 }
 
 List get_results() const {
   std::vector<std::string> bins;
   std::vector<double> woe;
   std::vector<double> iv;
   std::vector<int> count;
   std::vector<int> count_pos;
   std::vector<int> count_neg;
   
   std::vector<std::vector<std::string>> bin_categories(actual_bins);
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
   for (int i = 0; i < actual_bins; ++i) {
     if (bin_counts[i] > 0) {
       std::string bin_name = "";
       for (const auto& category : bin_categories[i]) {
         if (!bin_name.empty()) bin_name += bin_separator;
         bin_name += category;
       }
       bins.push_back(bin_name);
       count.push_back(bin_counts[i]);
       count_pos.push_back(bin_positives[i]);
       int bin_negatives = bin_counts[i] - bin_positives[i];
       count_neg.push_back(bin_negatives);
       
       double pos_dist = (double)bin_positives[i] / total_positive;
       double neg_dist = (double)bin_negatives / total_negative;
       
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
   
   Rcpp::NumericVector ids(bins.size());
   for(int i = 0; i < bins.size(); i++) {
      ids[i] = i + 1;
   }
   
   return Rcpp::List::create(
     Named("id") = ids,
     Named("bin") = bins,
     Named("woe") = woe,
     Named("iv") = iv,
     Named("count") = count,
     Named("count_pos") = count_pos,
     Named("count_neg") = count_neg,
     Named("converged") = converged,
     Named("iterations") = iterations_run
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
//' @param bin_separator Separator string for merging categories (default: "%;%").
//' @param initial_temperature Initial temperature for Simulated Annealing (default: 1.0).
//' @param cooling_rate Cooling rate for Simulated Annealing (default: 0.995).
//' @param max_iterations Maximum number of iterations for Simulated Annealing (default: 1000).
//' @param convergence_threshold Threshold for convergence (default: 1e-6).
//'
//' @return A list containing the following elements:
//' \itemize{
//'   \item bins: A character vector of bin names
//'   \item woe: A numeric vector of Weight of Evidence (WoE) values for each bin
//'   \item iv: A numeric vector of Information Value (IV) for each bin
//'   \item count: An integer vector of total counts for each bin
//'   \item count_pos: An integer vector of positive counts for each bin
//'   \item count_neg: An integer vector of negative counts for each bin
//'   \item converged: A logical value indicating whether the algorithm converged
//'   \item iterations: An integer value indicating the number of iterations run
//' }
//'
//' @examples
//' \dontrun{
//' set.seed(123)
//' target <- sample(0:1, 1000, replace = TRUE)
//' feature <- sample(LETTERS[1:5], 1000, replace = TRUE)
//' result <- optimal_binning_categorical_sab(target, feature)
//' print(result)
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
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_sab(Rcpp::IntegerVector target,
                                          Rcpp::StringVector feature,
                                          int min_bins = 3,
                                          int max_bins = 5,
                                          double bin_cutoff = 0.05,
                                          int max_n_prebins = 20,
                                          std::string bin_separator = "%;%",
                                          double initial_temperature = 1.0,
                                          double cooling_rate = 0.995,
                                          int max_iterations = 1000,
                                          double convergence_threshold = 1e-6) {
 try {
   if (target.size() != feature.size()) {
     Rcpp::stop("Target and feature vectors must have the same length");
   }
   if (min_bins < 2) {
     Rcpp::stop("min_bins must be at least 2");
   }
   if (max_bins < min_bins) {
     Rcpp::stop("max_bins must be >= min_bins");
   }
   if (bin_cutoff <= 0 || bin_cutoff >= 1) {
     Rcpp::stop("bin_cutoff must be between 0 and 1");
   }
   if (max_n_prebins < max_bins) {
     Rcpp::stop("max_n_prebins must be >= max_bins");
   }
   
   std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
   std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
   
   OptimalBinningCategoricalSAB binner(feature_vec, target_vec,
                                       min_bins, max_bins, bin_cutoff,
                                       max_n_prebins, bin_separator,
                                       initial_temperature, cooling_rate,
                                       max_iterations, convergence_threshold);
   binner.fit();
   
   return binner.get_results();
 } catch (const std::exception& e) {
   Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
 }
}


// #include <Rcpp.h>
// #include <vector>
// #include <string>
// #include <unordered_map>
// #include <unordered_set>
// #include <algorithm>
// #include <random>
// #include <cmath>
// #include <limits>
// #include <stdexcept>
// 
// using namespace Rcpp;
// 
// // Constants
// static constexpr double EPSILON = 1e-10;
// 
// // Class for Optimal Binning using Simulated Annealing
// class OptimalBinningCategoricalSAB {
// private:
//   // Input data
//   std::vector<std::string> feature;
//   std::vector<int> target;
//   
//   // Binning parameters
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
//   std::string bin_separator;
//   
//   // Simulated Annealing parameters
//   double initial_temperature;
//   double cooling_rate;
//   int max_iterations;
//   double convergence_threshold;
//   
//   // Category statistics
//   std::vector<std::string> unique_categories;
//   std::unordered_map<std::string, int> category_counts;
//   std::unordered_map<std::string, int> positive_counts;
//   int total_count;
//   int total_positive;
//   int total_negative;
//   
//   // Binning results
//   int actual_bins;
//   std::vector<int> best_solution;
//   double best_iv;
//   bool converged;
//   int iterations_run;
//   
//   // Random number generator
//   std::mt19937 gen;
//   
//   // Helper function to initialize category statistics and initial solution
//   void initialize() {
//     // Extract unique categories
//     std::unordered_set<std::string> unique_set(feature.begin(), feature.end());
//     unique_categories = std::vector<std::string>(unique_set.begin(), unique_set.end());
//     
//     // Count total observations and positives/negatives
//     total_count = feature.size();
//     total_positive = std::count(target.begin(), target.end(), 1);
//     total_negative = total_count - total_positive;
//     
//     // Count occurrences and positives per category
//     for (size_t i = 0; i < feature.size(); ++i) {
//       category_counts[feature[i]]++;
//       if (target[i] == 1) {
//         positive_counts[feature[i]]++;
//       }
//     }
//     
//     int n_categories = unique_categories.size();
//     
//     // Adjust actual_bins based on the number of unique categories
//     actual_bins = std::min(std::max(min_bins, std::min(max_bins, n_categories)), n_categories);
//     
//     // Initialize best solution: assign categories to bins in a round-robin fashion
//     best_solution.resize(n_categories);
//     for (size_t i = 0; i < best_solution.size(); ++i) {
//       best_solution[i] = i % actual_bins;
//     }
//     
//     // Shuffle the initial solution to start with a random assignment
//     std::shuffle(best_solution.begin(), best_solution.end(), gen);
//     
//     best_iv = calculate_iv(best_solution);
//     converged = false;
//     iterations_run = 0;
//   }
//   
//   // Calculate Information Value (IV) for a given binning solution
//   double calculate_iv(const std::vector<int>& solution) const {
//     std::vector<int> bin_counts(actual_bins, 0);
//     std::vector<int> bin_positives(actual_bins, 0);
//     
//     // Aggregate counts per bin
//     for (size_t i = 0; i < unique_categories.size(); ++i) {
//       const std::string& category = unique_categories[i];
//       int bin = solution[i];
//       bin_counts[bin] += category_counts.at(category);
//       bin_positives[bin] += positive_counts.at(category);
//     }
//     
//     double iv = 0.0;
//     
//     // Calculate IV based on WoE for each bin
//     for (int i = 0; i < actual_bins; ++i) {
//       if (bin_counts[i] > 0) {
//         int bin_negatives = bin_counts[i] - bin_positives[i];
//         double pos_dist = static_cast<double>(bin_positives[i]) / total_positive;
//         double neg_dist = static_cast<double>(bin_negatives) / total_negative;
//         
//         // Enforce bin_cutoff
//         double bin_proportion = static_cast<double>(bin_counts[i]) / total_count;
//         if (bin_proportion < bin_cutoff) {
//           // Penalize bins that do not meet the bin_cutoff
//           iv -= 1000.0; // Large negative penalty
//           continue;
//         }
//         
//         if (pos_dist > EPSILON && neg_dist > EPSILON) {
//           double woe = std::log(pos_dist / neg_dist);
//           iv += (pos_dist - neg_dist) * woe;
//         }
//       }
//     }
//     
//     return iv;
//   }
//   
//   // Check if the binning solution is monotonic based on positive rates
//   bool is_monotonic(const std::vector<int>& solution) const {
//     std::vector<double> bin_rates(actual_bins, 0.0);
//     std::vector<int> bin_counts(actual_bins, 0);
//     
//     // Aggregate counts and positive rates per bin
//     for (size_t i = 0; i < unique_categories.size(); ++i) {
//       const std::string& category = unique_categories[i];
//       int bin = solution[i];
//       bin_counts[bin] += category_counts.at(category);
//       bin_rates[bin] += positive_counts.at(category);
//     }
//     
//     // Calculate positive rate for each bin
//     for (int i = 0; i < actual_bins; ++i) {
//       if (bin_counts[i] > 0) {
//         bin_rates[i] /= bin_counts[i];
//       }
//     }
//     
//     // Check for monotonic increasing or decreasing order
//     bool increasing = true;
//     bool decreasing = true;
//     for (int i = 1; i < actual_bins; ++i) {
//       if (bin_rates[i] < bin_rates[i - 1] - EPSILON) {
//         increasing = false;
//       }
//       if (bin_rates[i] > bin_rates[i - 1] + EPSILON) {
//         decreasing = false;
//       }
//     }
//     
//     return increasing || decreasing;
//   }
//   
//   // Generate a neighboring solution by swapping two category assignments
//   std::vector<int> generate_neighbor(const std::vector<int>& solution) {
//     std::vector<int> neighbor = solution;
//     if (neighbor.size() <= 1) return neighbor;
//     
//     std::uniform_int_distribution<> dis(0, neighbor.size() - 1);
//     int idx1 = dis(gen);
//     int idx2 = dis(gen);
//     
//     // Ensure two different indices
//     while (idx2 == idx1) {
//       idx2 = dis(gen);
//     }
//     
//     std::swap(neighbor[idx1], neighbor[idx2]);
//     return neighbor;
//   }
//   
// public:
//   // Constructor
//   OptimalBinningCategoricalSAB(const std::vector<std::string>& feature_,
//                                const std::vector<int>& target_,
//                                int min_bins_ = 3,
//                                int max_bins_ = 5,
//                                double bin_cutoff_ = 0.05,
//                                int max_n_prebins_ = 20,
//                                std::string bin_separator_ = "%;%",
//                                double initial_temperature_ = 1.0,
//                                double cooling_rate_ = 0.995,
//                                int max_iterations_ = 1000,
//                                double convergence_threshold_ = 1e-6)
//     : feature(feature_), target(target_), min_bins(min_bins_), max_bins(max_bins_),
//       bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_), bin_separator(bin_separator_),
//       initial_temperature(initial_temperature_), cooling_rate(cooling_rate_),
//       max_iterations(max_iterations_), convergence_threshold(convergence_threshold_) {
//     
//     // Input validation
//     if (feature.size() != target.size()) {
//       throw std::invalid_argument("Feature and target vectors must have the same length");
//     }
//     if (min_bins < 2) {
//       throw std::invalid_argument("min_bins must be at least 2");
//     }
//     if (max_bins < min_bins) {
//       throw std::invalid_argument("max_bins must be greater than or equal to min_bins");
//     }
//     if (bin_cutoff <= 0 || bin_cutoff >= 1) {
//       throw std::invalid_argument("bin_cutoff must be between 0 and 1");
//     }
//     if (max_n_prebins < max_bins) {
//       throw std::invalid_argument("max_n_prebins must be greater than or equal to max_bins");
//     }
//     
//     // Initialize random number generator
//     std::random_device rd;
//     gen.seed(rd());
//     
//     // Initialize category statistics and initial solution
//     initialize();
//   }
//   
//   // Fit the binning solution using Simulated Annealing
//   void fit() {
//     double temperature = initial_temperature;
//     double prev_best_iv = best_iv;
//     
//     for (int iter = 0; iter < max_iterations; ++iter) {
//       // Generate a neighboring solution
//       std::vector<int> neighbor = generate_neighbor(best_solution);
//       double neighbor_iv = calculate_iv(neighbor);
//       
//       // Check for monotonicity if we have more than min_bins
//       if (actual_bins > min_bins && !is_monotonic(neighbor)) {
//         continue; // Skip non-monotonic solutions
//       }
//       
//       // Acceptance criteria
//       if (neighbor_iv > best_iv) {
//         best_solution = neighbor;
//         best_iv = neighbor_iv;
//       } else {
//         double acceptance_probability = std::exp((neighbor_iv - best_iv) / temperature);
//         std::uniform_real_distribution<> dis(0.0, 1.0);
//         if (dis(gen) < acceptance_probability) {
//           best_solution = neighbor;
//           best_iv = neighbor_iv;
//         }
//       }
//       
//       // Cool down the temperature
//       temperature *= cooling_rate;
//       
//       // Check for convergence
//       if (std::abs(best_iv - prev_best_iv) < convergence_threshold) {
//         converged = true;
//         iterations_run = iter + 1;
//         break;
//       }
//       
//       prev_best_iv = best_iv;
//     }
//     
//     if (!converged) {
//       iterations_run = max_iterations;
//     }
//     
//     // Ensure monotonicity for solutions with more than min_bins
//     if (actual_bins > min_bins) {
//       int max_monotonic_attempts = 100;
//       for (int attempt = 0; attempt < max_monotonic_attempts; ++attempt) {
//         if (is_monotonic(best_solution)) {
//           break;
//         }
//         best_solution = generate_neighbor(best_solution);
//         best_iv = calculate_iv(best_solution);
//       }
//     }
//   }
//   
//   // Retrieve the binning results
//   List get_results() const {
//     std::vector<std::string> bins;
//     std::vector<double> woe;
//     std::vector<double> iv;
//     std::vector<int> count;
//     std::vector<int> count_pos;
//     std::vector<int> count_neg;
//     
//     // Aggregate categories into bins
//     std::vector<std::vector<std::string>> bin_categories(actual_bins);
//     std::vector<int> bin_counts(actual_bins, 0);
//     std::vector<int> bin_positives(actual_bins, 0);
//     
//     for (size_t i = 0; i < unique_categories.size(); ++i) {
//       const std::string& category = unique_categories[i];
//       int bin = best_solution[i];
//       bin_categories[bin].push_back(category);
//       bin_counts[bin] += category_counts.at(category);
//       bin_positives[bin] += positive_counts.at(category);
//     }
//     
//     double total_iv = 0.0;
//     
//     // Calculate WoE and IV for each bin
//     for (int i = 0; i < actual_bins; ++i) {
//       if (bin_counts[i] > 0) {
//         // Create bin name by concatenating categories with bin_separator
//         std::string bin_name = "";
//         for (const auto& category : bin_categories[i]) {
//           if (!bin_name.empty()) bin_name += bin_separator;
//           bin_name += category;
//         }
//         bins.push_back(bin_name);
//         count.push_back(bin_counts[i]);
//         count_pos.push_back(bin_positives[i]);
//         int bin_negatives = bin_counts[i] - bin_positives[i];
//         count_neg.push_back(bin_negatives);
//         
//         double pos_dist = static_cast<double>(bin_positives[i]) / total_positive;
//         double neg_dist = static_cast<double>(bin_negatives) / total_negative;
//         
//         double woe_value = 0.0;
//         double iv_value = 0.0;
//         
//         if (pos_dist > EPSILON && neg_dist > EPSILON) {
//           woe_value = std::log(pos_dist / neg_dist);
//           iv_value = (pos_dist - neg_dist) * woe_value;
//           total_iv += iv_value;
//         }
//         
//         woe.push_back(woe_value);
//         iv.push_back(iv_value);
//       }
//     }
//     
//     return List::create(
//       Named("bin") = bins,
//       Named("woe") = woe,
//       Named("iv") = iv,
//       Named("count") = count,
//       Named("count_pos") = count_pos,
//       Named("count_neg") = count_neg,
//       Named("converged") = converged,
//       Named("iterations") = iterations_run
//     );
//   }
// };
// 
// //' @title
// //' Optimal Binning for Categorical Variables using Simulated Annealing
// //'
// //' @description
// //' This function performs optimal binning for categorical variables using a Simulated Annealing approach.
// //' It maximizes the Information Value (IV) while maintaining monotonicity in the bins.
// //'
// //' @param target An integer vector of binary target values (0 or 1).
// //' @param feature A character vector of categorical feature values.
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff Minimum proportion of observations in a bin (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins (default: 20).
// //' @param bin_separator Separator string for merging categories (default: "%;%").
// //' @param initial_temperature Initial temperature for Simulated Annealing (default: 1.0).
// //' @param cooling_rate Cooling rate for Simulated Annealing (default: 0.995).
// //' @param max_iterations Maximum number of iterations for Simulated Annealing (default: 1000).
// //' @param convergence_threshold Threshold for convergence (default: 1e-6).
// //'
// //' @return A list containing the following elements:
// //' \itemize{
// //'   \item bins: A character vector of bin names
// //'   \item woe: A numeric vector of Weight of Evidence (WoE) values for each bin
// //'   \item iv: A numeric vector of Information Value (IV) for each bin
// //'   \item count: An integer vector of total counts for each bin
// //'   \item count_pos: An integer vector of positive counts for each bin
// //'   \item count_neg: An integer vector of negative counts for each bin
// //'   \item converged: A logical value indicating whether the algorithm converged
// //'   \item iterations: An integer value indicating the number of iterations run
// //' }
// //'
// //' @examples
// //' \dontrun{
// //' set.seed(123)
// //' target <- sample(0:1, 1000, replace = TRUE)
// //' feature <- sample(LETTERS[1:5], 1000, replace = TRUE)
// //' result <- optimal_binning_categorical_sab(target, feature)
// //' print(result)
// //' }
// //'
// //' @details
// //' The algorithm uses Simulated Annealing to find an optimal binning solution that maximizes
// //' the Information Value while maintaining monotonicity. It respects the specified constraints
// //' on the number of bins and bin sizes.
// //'
// //' The Weight of Evidence (WoE) is calculated as:
// //' \deqn{WoE_i = \ln(\frac{\text{Distribution of positives}_i}{\text{Distribution of negatives}_i})}
// //'
// //' Where:
// //' \deqn{\text{Distribution of positives}_i = \frac{\text{Number of positives in bin } i}{\text{Total Number of positives}}}
// //' \deqn{\text{Distribution of negatives}_i = \frac{\text{Number of negatives in bin } i}{\text{Total Number of negatives}}}
// //'
// //' The Information Value (IV) is calculated as:
// //' \deqn{IV = \sum_{i=1}^{N} (\text{Distribution of positives}_i - \text{Distribution of negatives}_i) \times WoE_i}
// //'
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_categorical_sab(Rcpp::IntegerVector target,
//                                           Rcpp::StringVector feature,
//                                           int min_bins = 3,
//                                           int max_bins = 5,
//                                           double bin_cutoff = 0.05,
//                                           int max_n_prebins = 20,
//                                           std::string bin_separator = "%;%",
//                                           double initial_temperature = 1.0,
//                                           double cooling_rate = 0.995,
//                                           int max_iterations = 1000,
//                                           double convergence_threshold = 1e-6) {
//  try {
//    // Input validation
//    if (target.size() != feature.size()) {
//      Rcpp::stop("Target and feature vectors must have the same length");
//    }
//    if (min_bins < 2) {
//      Rcpp::stop("min_bins must be at least 2");
//    }
//    if (max_bins < min_bins) {
//      Rcpp::stop("max_bins must be greater than or equal to min_bins");
//    }
//    if (bin_cutoff <= 0 || bin_cutoff >= 1) {
//      Rcpp::stop("bin_cutoff must be between 0 and 1");
//    }
//    if (max_n_prebins < max_bins) {
//      Rcpp::stop("max_n_prebins must be greater than or equal to max_bins");
//    }
//    
//    // Convert Rcpp vectors to C++ vectors
//    std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
//    std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
//    
//    // Create and fit the binning model
//    OptimalBinningCategoricalSAB binner(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, 
//                                        max_n_prebins, bin_separator, initial_temperature, 
//                                        cooling_rate, max_iterations, convergence_threshold);
//    binner.fit();
//    
//    // Retrieve and return the results
//    return binner.get_results();
//  } catch (const std::exception& e) {
//    Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
//  }
// }
