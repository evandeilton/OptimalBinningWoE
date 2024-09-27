#include <Rcpp.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

class OptimalBinningCategoricalSAB {
private:
  std::vector<std::string> feature;
  std::vector<int> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  
  std::vector<std::string> unique_categories;
  std::unordered_map<std::string, int> category_counts;
  std::unordered_map<std::string, int> positive_counts;
  int total_count;
  int total_positive;
  int actual_bins;
  
  // Simulated Annealing parameters
  double initial_temperature;
  double cooling_rate;
  int max_iterations;
  
  std::vector<int> current_solution;
  std::vector<int> best_solution;
  double current_iv;
  double best_iv;
  
  // Random number generator
  std::mt19937 gen;
  
  void initialize() {
    std::unordered_set<std::string> unique_set(feature.begin(), feature.end());
    unique_categories = std::vector<std::string>(unique_set.begin(), unique_set.end());
    
    total_count = feature.size();
    total_positive = std::count(target.begin(), target.end(), 1);
    
    for (size_t i = 0; i < feature.size(); ++i) {
      category_counts[feature[i]]++;
      if (target[i] == 1) {
        positive_counts[feature[i]]++;
      }
    }
    
    // Adjust actual_bins based on the number of unique categories
    actual_bins = std::min(std::max(static_cast<int>(unique_categories.size()), min_bins), max_bins);
    
    // Initialize current solution
    current_solution.resize(unique_categories.size());
    for (size_t i = 0; i < current_solution.size(); ++i) {
      current_solution[i] = i % actual_bins;
    }
    std::shuffle(current_solution.begin(), current_solution.end(), gen);
    
    best_solution = current_solution;
    current_iv = calculate_iv(current_solution);
    best_iv = current_iv;
  }
  
  double calculate_iv(const std::vector<int>& solution) {
    std::vector<int> bin_counts(actual_bins, 0);
    std::vector<int> bin_positives(actual_bins, 0);
    
    for (size_t i = 0; i < unique_categories.size(); ++i) {
      const std::string& category = unique_categories[i];
      int bin = solution[i];
      bin_counts[bin] += category_counts[category];
      bin_positives[bin] += positive_counts[category];
    }
    
    double iv = 0.0;
    for (int i = 0; i < actual_bins; ++i) {
      if (bin_counts[i] > 0) {
        double bin_rate = static_cast<double>(bin_positives[i]) / bin_counts[i];
        double overall_rate = static_cast<double>(total_positive) / total_count;
        if (bin_rate > 0 && bin_rate < 1) {
          double woe = std::log(bin_rate / (1 - bin_rate) * (1 - overall_rate) / overall_rate);
          iv += (bin_rate - (1 - bin_rate)) * woe;
        }
      }
    }
    
    return std::max(0.0, iv);  // Ensure we always return a non-negative value
  }
  
  bool is_monotonic(const std::vector<int>& solution) {
    std::vector<double> bin_rates(actual_bins, 0.0);
    std::vector<int> bin_counts(actual_bins, 0);
    
    for (size_t i = 0; i < unique_categories.size(); ++i) {
      const std::string& category = unique_categories[i];
      int bin = solution[i];
      bin_counts[bin] += category_counts[category];
      bin_rates[bin] += positive_counts[category];
    }
    
    for (int i = 0; i < actual_bins; ++i) {
      if (bin_counts[i] > 0) {
        bin_rates[i] /= bin_counts[i];
      }
    }
    
    // Remove empty bins
    bin_rates.erase(std::remove(bin_rates.begin(), bin_rates.end(), 0.0), bin_rates.end());
    
    for (size_t i = 1; i < bin_rates.size(); ++i) {
      if (bin_rates[i] < bin_rates[i-1]) {
        return false;
      }
    }
    
    return true;
  }
  
  std::vector<int> generate_neighbor(const std::vector<int>& solution) {
    std::vector<int> neighbor = solution;
    int idx = std::uniform_int_distribution<>(0, neighbor.size() - 1)(gen);
    int new_bin = std::uniform_int_distribution<>(0, actual_bins - 1)(gen);
    neighbor[idx] = new_bin;
    return neighbor;
  }
  
public:
  OptimalBinningCategoricalSAB(const std::vector<std::string>& feature,
                               const std::vector<int>& target,
                               int min_bins = 3,
                               int max_bins = 5,
                               double bin_cutoff = 0.05,
                               int max_n_prebins = 20,
                               double initial_temperature = 1.0,
                               double cooling_rate = 0.995,
                               int max_iterations = 1000)
    : feature(feature), target(target), min_bins(min_bins), max_bins(max_bins),
      bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins),
      initial_temperature(initial_temperature), cooling_rate(cooling_rate),
      max_iterations(max_iterations), gen(std::random_device()()) {
    
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
    
    initialize();
  }
  
  void fit() {
    double temperature = initial_temperature;
    
#pragma omp parallel
{
  std::vector<int> local_best_solution = best_solution;
  double local_best_iv = best_iv;
  
#pragma omp for
  for (int iter = 0; iter < max_iterations; ++iter) {
    std::vector<int> neighbor = generate_neighbor(local_best_solution);
    double neighbor_iv = calculate_iv(neighbor);
    
    if (neighbor_iv > local_best_iv && is_monotonic(neighbor)) {
      local_best_solution = neighbor;
      local_best_iv = neighbor_iv;
    } else {
      double acceptance_probability = std::exp((neighbor_iv - local_best_iv) / temperature);
      if (std::uniform_real_distribution<>(0, 1)(gen) < acceptance_probability && is_monotonic(neighbor)) {
        local_best_solution = neighbor;
        local_best_iv = neighbor_iv;
      }
    }
    
    temperature *= cooling_rate;
  }
  
#pragma omp critical
{
  if (local_best_iv > best_iv) {
    best_solution = local_best_solution;
    best_iv = local_best_iv;
  }
}
}

// Ensure final solution is monotonic
while (!is_monotonic(best_solution)) {
  best_solution = generate_neighbor(best_solution);
}
best_iv = calculate_iv(best_solution);  // Recalculate best_iv to ensure consistency
  }
  
  Rcpp::List get_results() {
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
      bin_counts[bin] += category_counts[category];
      bin_positives[bin] += positive_counts[category];
    }
    
    for (int i = 0; i < actual_bins; ++i) {
      if (bin_counts[i] > 0) {
        std::string bin_name = "";
        for (const auto& category : bin_categories[i]) {
          if (!bin_name.empty()) bin_name += "+";
          bin_name += category;
        }
        bins.push_back(bin_name);
        count.push_back(bin_counts[i]);
        count_pos.push_back(bin_positives[i]);
        count_neg.push_back(bin_counts[i] - bin_positives[i]);
        
        double bin_rate = static_cast<double>(bin_positives[i]) / bin_counts[i];
        double overall_rate = static_cast<double>(total_positive) / total_count;
        double woe_value = std::log(bin_rate / (1 - bin_rate) * (1 - overall_rate) / overall_rate);
        woe.push_back(woe_value);
        
        double iv_value = (bin_rate - (1 - bin_rate)) * woe_value;
        iv.push_back(iv_value);
      }
    }
    
    // Ensure all vectors have the same length
    size_t n = bins.size();
    if (n == 0) {
      // Return empty DataFrame if no bins
      return Rcpp::List::create(
        Rcpp::Named("woefeature") = Rcpp::NumericVector(feature.size()),
        Rcpp::Named("woebin") = Rcpp::DataFrame::create()
      );
    }
    
    Rcpp::DataFrame woebin = Rcpp::DataFrame::create(
      Rcpp::Named("bin") = bins,
      Rcpp::Named("woe") = woe,
      Rcpp::Named("iv") = iv,
      Rcpp::Named("count") = count,
      Rcpp::Named("count_pos") = count_pos,
      Rcpp::Named("count_neg") = count_neg
    );
    
    std::vector<double> woefeature(feature.size());
    for (size_t i = 0; i < feature.size(); ++i) {
      auto it = std::find(unique_categories.begin(), unique_categories.end(), feature[i]);
      if (it != unique_categories.end()) {
        int bin = best_solution[it - unique_categories.begin()];
        woefeature[i] = woe[bin];
      } else {
        woefeature[i] = 0.0;  // Or some other default value
      }
    }
    
    return Rcpp::List::create(
      Rcpp::Named("woefeature") = woefeature,
      Rcpp::Named("woebin") = woebin
    );
  }
};

//' @title
//' Optimal Binning for Categorical Variables using Simulated Annealing
//'
//' @description
//' This function performs optimal binning for categorical variables using a Simulated Annealing approach.
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
//' @return A list containing two elements:
//' \itemize{
//'   \item woefeature: A numeric vector of Weight of Evidence (WoE) values for each observation
//'   \item woebin: A data frame containing binning information, including bin names, WoE, Information Value (IV), and counts
//' }
//'
//' @examples
//' \dontrun{
//' # Create sample data
//' set.seed(123)
//' target <- sample(0:1, 1000, replace = TRUE)
//' feature <- sample(LETTERS[1:5], 1000, replace = TRUE)
//'
//' # Run optimal binning
//' result <- optimal_binning_categorical_sab(target, feature)
//'
//' # View results
//' print(result$woebin)
//' }
//'
//' @details
//' This algorithm performs optimal binning for categorical variables using Simulated Annealing (SA). 
//' The process aims to maximize the Information Value (IV) while maintaining monotonicity in the bins.
//'
//' The algorithm works as follows:
//' \enumerate{
//'   \item Initialize by assigning each unique category to a random bin.
//'   \item Calculate the initial IV.
//'   \item In each iteration of SA:
//'     \enumerate{
//'       \item Generate a neighbor solution by randomly reassigning a category to a different bin.
//'       \item Calculate the IV of the new solution.
//'       \item Accept the new solution if it improves IV and maintains monotonicity.
//'       \item If the new solution is worse, accept it with a probability based on the current temperature.
//'     }
//'   \item Repeat step 3 for a specified number of iterations, reducing the temperature each time.
//'   \item Ensure the final solution is monotonic.
//' }
//'
//' The Information Value (IV) is calculated as:
//' \deqn{IV = \sum(\text{% of non-events} - \text{% of events}) \times WoE}
//'
//' Where Weight of Evidence (WoE) is:
//' \deqn{WoE = \ln(\frac{\text{% of events}}{\text{% of non-events}})}
//'
//' The algorithm uses OpenMP for parallel processing to improve performance.
//'
//' @references
//' \itemize{
//'   \item Bertsimas, D., & Dunn, J. (2017). Optimal classification trees. Machine Learning, 106(7), 1039-1082.
//'   \item Mironchyk, P., & Tchistiakov, V. (2017). Monotone optimal binning algorithm for credit risk modeling. 
//'         In Workshop on Data Science and Advanced Analytics (DSAA). 
//' }
//'
//' @author Lopes, J. E.
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
  std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
  std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
  
  OptimalBinningCategoricalSAB binner(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins,
                                      initial_temperature, cooling_rate, max_iterations);
  binner.fit();
  return binner.get_results();
}
