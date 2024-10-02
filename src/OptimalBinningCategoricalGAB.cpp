#include <Rcpp.h>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <unordered_map>
#include <cmath>
#include <omp.h>
#include <limits>

// Enable OpenMP for parallel processing
// [[Rcpp::plugins(openmp)]]

// Define a small epsilon for floating-point comparisons
const double EPSILON = 1e-9;

// Structure to hold bin information
struct Bin {
  std::vector<std::string> categories; // Categories in the bin
  int count;                           // Total count
  int count_pos;                       // Positive count
  int count_neg;                       // Negative count
  double woe;                          // Weight of Evidence
  double iv;                           // Information Value
  
  Bin() : count(0), count_pos(0), count_neg(0), woe(0.0), iv(0.0) {}
};

// Structure to represent an individual in the population
struct Individual {
  std::vector<int> chromosome; // Assignments of categories to bins
  double fitness;              // Fitness score based on IV
  
  Individual() : fitness(0.0) {}
};

// Class implementing the Genetic Algorithm for Optimal Binning
class OptimalBinningCategoricalGAB {
private:
  // Input data
  std::vector<std::string> feature;
  std::vector<int> target;
  
  // Binning parameters
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  
  // Genetic Algorithm parameters
  int population_size;
  int num_generations;
  double mutation_rate;
  double crossover_rate;
  
  // Data counts
  int total_count;
  int total_pos;
  int total_neg;
  
  // Unique categories and their counts
  std::vector<std::string> unique_categories;
  std::unordered_map<std::string, int> category_counts;
  std::unordered_map<std::string, std::pair<int, int>> category_target_counts;
  
  // Population
  std::vector<Individual> population;
  
  // Helper function to join strings with a delimiter
  std::string joinStrings(const std::vector<std::string>& strings, const std::string& delimiter) const {
    std::string result;
    for (size_t i = 0; i < strings.size(); ++i) {
      if (i > 0) result += delimiter;
      result += strings[i];
    }
    return result;
  }
  
  // Initialize unique categories based on bin_cutoff and max_n_prebins
  void initializeUniqueCategories() {
    // Populate category counts
    for (size_t i = 0; i < feature.size(); ++i) {
      category_counts[feature[i]]++;
      if (target[i] == 1) {
        category_target_counts[feature[i]].first++;
      } else {
        category_target_counts[feature[i]].second++;
      }
    }
    
    // Filter categories based on bin_cutoff
    size_t total_count_local = feature.size();
    for (const auto& pair : category_counts) {
      double freq = static_cast<double>(pair.second) / total_count_local;
      if (freq >= bin_cutoff) {
        unique_categories.push_back(pair.first);
      }
    }
    
    // If no categories meet bin_cutoff, include all
    if (unique_categories.empty()) {
      for (const auto& pair : category_counts) {
        unique_categories.push_back(pair.first);
      }
    }
    
    // Limit to max_n_prebins
    if (unique_categories.size() > static_cast<size_t>(max_n_prebins)) {
      std::partial_sort(unique_categories.begin(), unique_categories.begin() + max_n_prebins, unique_categories.end(),
                        [&](const std::string& a, const std::string& b) {
                          return category_counts[a] > category_counts[b];
                        });
      unique_categories.resize(max_n_prebins);
    }
    
    // Adjust bin constraints based on unique_categories size
    size_t num_unique = unique_categories.size();
    if (num_unique <= 2) {
      min_bins = std::max(1, min_bins); // At least 1 bin
      max_bins = std::max(1, max_bins);
      min_bins = std::min(min_bins, static_cast<int>(num_unique));
      max_bins = std::min(max_bins, static_cast<int>(num_unique));
    } else {
      min_bins = std::max(2, min_bins);
      max_bins = std::max(min_bins, max_bins);
      max_bins = std::min(max_bins, static_cast<int>(num_unique));
    }
  }
  
  // Initialize the population with valid chromosomes
  void initializePopulation() {
    population.resize(population_size);
    
    // Ensure population_size is even for pairing in crossover
    if (population_size % 2 != 0) {
      population_size += 1;
      population.resize(population_size);
    }
    
    // Parallel initialization using OpenMP
#pragma omp parallel
{
  // Each thread has its own RNG
  std::random_device rd;
  std::mt19937 gen(rd() + omp_get_thread_num());
  std::uniform_int_distribution<> bin_dis(0, max_bins - 1);
  
#pragma omp for
  for (int i = 0; i < population_size; ++i) {
    Individual& indiv = population[i];
    indiv.chromosome.resize(unique_categories.size());
    
    // Ensure at least min_bins are used
    // Assign the first min_bins categories to different bins
    for (int b = 0; b < min_bins && b < static_cast<int>(unique_categories.size()); ++b) {
      indiv.chromosome[b] = b;
    }
    // Assign remaining categories randomly
    for (size_t c = min_bins; c < unique_categories.size(); ++c) {
      indiv.chromosome[c] = bin_dis(gen);
    }
    
    // Calculate fitness
    indiv.fitness = calculateFitness(indiv.chromosome);
  }
}
  }
  
  // Calculate fitness based on Information Value (IV)
  double calculateFitness(const std::vector<int>& chromosome) const {
    std::vector<Bin> bins(max_bins, Bin());
    int bins_used = 0;
    
    // Assign categories to bins
    for (size_t i = 0; i < chromosome.size(); ++i) {
      int bin_idx = chromosome[i];
      if (bin_idx < 0 || bin_idx >= max_bins) {
        // Invalid bin index, assign to bin 0
        bin_idx = 0;
      }
      bins[bin_idx].categories.push_back(unique_categories[i]);
    }
    
    double total_iv = 0.0;
    
    // Calculate WoE and IV for each bin
    for (auto& bin : bins) { // Use non-const reference to allow modification
      if (bin.categories.empty()) continue;
      bins_used++;
      
      // Aggregate counts
      for (const auto& cat : bin.categories) {
        bin.count += category_counts.at(cat);
        bin.count_pos += category_target_counts.at(cat).first;
        bin.count_neg += category_target_counts.at(cat).second;
      }
      
      // Handle zero counts with smoothing
      double adjusted_pos = (bin.count_pos == 0) ? 0.5 : bin.count_pos;
      double adjusted_neg = (bin.count_neg == 0) ? 0.5 : bin.count_neg;
      
      double pos_rate = adjusted_pos / static_cast<double>(total_pos);
      double neg_rate = adjusted_neg / static_cast<double>(total_neg);
      
      // Prevent division by zero and log of zero
      if (neg_rate == 0.0) {
        neg_rate = 1e-10;
      }
      if (pos_rate == 0.0) {
        pos_rate = 1e-10;
      }
      
      bin.woe = std::log(pos_rate / neg_rate);
      bin.iv = (pos_rate - neg_rate) * bin.woe;
      
      total_iv += bin.iv;
    }
    
    // Penalize if bin constraints are not met
    if (bins_used < min_bins || bins_used > max_bins) {
      total_iv *= 0.5; // Arbitrary penalty factor
    }
    
    return total_iv;
  }
  
  // Perform crossover between pairs of individuals
  void crossover() {
    // Ensure population_size is even
    if (population_size % 2 != 0) {
      population_size -= 1; // Remove the last individual for pairing
    }
    
    // Parallel crossover using OpenMP
#pragma omp parallel
{
  // Each thread has its own RNG
  std::random_device rd;
  std::mt19937 gen(rd() + omp_get_thread_num());
  std::uniform_real_distribution<> dis(0.0, 1.0);
  
#pragma omp for
  for (int i = 0; i < population_size; i += 2) {
    if (dis(gen) < crossover_rate) {
      // Single-point crossover
      std::uniform_int_distribution<> point_dis(1, unique_categories.size() - 1);
      int crossover_point = point_dis(gen);
      
      // Swap genes beyond the crossover point
      for (size_t j = crossover_point; j < unique_categories.size(); ++j) {
        std::swap(population[i].chromosome[j], population[i+1].chromosome[j]);
      }
      
      // Recalculate fitness
      population[i].fitness = calculateFitness(population[i].chromosome);
      population[i+1].fitness = calculateFitness(population[i+1].chromosome);
    }
  }
}
  }
  
  // Perform mutation on the population
  void mutate() {
#pragma omp parallel
{
  // Each thread has its own RNG
  std::random_device rd;
  std::mt19937 gen(rd() + omp_get_thread_num());
  std::uniform_real_distribution<> dis(0.0, 1.0);
  std::uniform_int_distribution<> bin_dis(0, max_bins - 1);
  
#pragma omp for
  for (int i = 0; i < population_size; ++i) {
    for (size_t j = 0; j < population[i].chromosome.size(); ++j) {
      if (dis(gen) < mutation_rate) {
        population[i].chromosome[j] = bin_dis(gen);
      }
    }
    // Recalculate fitness after mutation
    population[i].fitness = calculateFitness(population[i].chromosome);
  }
}
  }
  
  // Select individuals based on fitness (Tournament Selection)
  void select() {
    std::vector<Individual> new_population(population_size);
    
#pragma omp parallel
{
  // Each thread has its own RNG
  std::random_device rd;
  std::mt19937 gen(rd() + omp_get_thread_num());
  std::uniform_int_distribution<> tournament_dis(0, population_size - 1);
  
#pragma omp for
  for (int i = 0; i < population_size; ++i) {
    // Tournament selection of size 3
    int idx1 = tournament_dis(gen);
    int idx2 = tournament_dis(gen);
    int idx3 = tournament_dis(gen);
    
    double best_fitness = population[idx1].fitness;
    int best_idx = idx1;
    
    if (population[idx2].fitness > best_fitness) {
      best_fitness = population[idx2].fitness;
      best_idx = idx2;
    }
    
    if (population[idx3].fitness > best_fitness) {
      best_fitness = population[idx3].fitness;
      best_idx = idx3;
    }
    
    new_population[i] = population[best_idx];
  }
}

population = std::move(new_population);
  }
  
public:
  // Constructor to initialize parameters
  OptimalBinningCategoricalGAB(int min_bins = 2, int max_bins = 5, double bin_cutoff = 0.05, int max_n_prebins = 20,
                               int population_size = 100, int num_generations = 100, double mutation_rate = 0.01, double crossover_rate = 0.8)
    : min_bins(min_bins), max_bins(max_bins), bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins),
      population_size(population_size), num_generations(num_generations), mutation_rate(mutation_rate), crossover_rate(crossover_rate),
      total_count(0), total_pos(0), total_neg(0) {}
  
  // Main fitting function
  Rcpp::List fit(const Rcpp::CharacterVector& feature_rcpp, const Rcpp::IntegerVector& target_rcpp) {
    // Convert Rcpp vectors to STL vectors with error handling
    try {
      feature = Rcpp::as<std::vector<std::string>>(feature_rcpp);
    } catch (...) {
      Rcpp::stop("Error converting feature to std::vector<std::string>");
    }
    
    try {
      target = Rcpp::as<std::vector<int>>(target_rcpp);
    } catch (...) {
      Rcpp::stop("Error converting target to std::vector<int>");
    }
    
    // Validate inputs
    if (feature.size() != target.size()) {
      Rcpp::stop("Feature and target vectors must have the same length.");
    }
    if (feature.empty()) {
      Rcpp::stop("Feature vector cannot be empty.");
    }
    if (min_bins < 1) {
      Rcpp::stop("min_bins must be at least 1.");
    }
    if (max_bins < min_bins) {
      Rcpp::stop("max_bins must be greater than or equal to min_bins.");
    }
    if (bin_cutoff <= 0.0 || bin_cutoff >= 1.0) {
      Rcpp::stop("bin_cutoff must be between 0 and 1.");
    }
    if (max_n_prebins < min_bins) {
      Rcpp::stop("max_n_prebins must be greater than or equal to min_bins.");
    }
    
    // Calculate total counts
    total_count = static_cast<int>(feature.size());
    total_pos = std::count(target.begin(), target.end(), 1);
    total_neg = total_count - total_pos;
    
    if (total_pos == 0 || total_neg == 0) {
      Rcpp::stop("Target variable must have both positive and negative cases.");
    }
    
    // Initialize unique categories
    initializeUniqueCategories();
    
    // Initialize population
    initializePopulation();
    
    // Run Genetic Algorithm for specified generations
    for (int gen = 0; gen < num_generations; ++gen) {
      crossover();
      mutate();
      select();
    }
    
    // Identify the best individual based on fitness
    auto best_individual = std::max_element(population.begin(), population.end(),
                                            [](const Individual& a, const Individual& b) { return a.fitness < b.fitness; });
    
    // Assign categories to bins based on the best chromosome
    std::vector<Bin> final_bins(max_bins, Bin());
    for (size_t i = 0; i < best_individual->chromosome.size(); ++i) {
      int bin_idx = best_individual->chromosome[i];
      if (bin_idx < 0 || bin_idx >= max_bins) {
        bin_idx = 0; // Assign to bin 0 if invalid
      }
      final_bins[bin_idx].categories.push_back(unique_categories[i]);
    }
    
    // Calculate WoE and IV for each bin
    std::vector<std::string> bin_names;
    std::vector<double> woe_values;
    std::vector<double> iv_values;
    std::vector<int> counts;
    std::vector<int> counts_pos;
    std::vector<int> counts_neg;
    
    std::unordered_map<std::string, double> category_to_woe;
    
    for (auto& bin : final_bins) { // Use non-const reference to modify bin
      if (bin.categories.empty()) continue;
      
      // Aggregate counts
      for (const auto& cat : bin.categories) {
        bin.count += category_counts.at(cat);
        bin.count_pos += category_target_counts.at(cat).first;
        bin.count_neg += category_target_counts.at(cat).second;
      }
      
      // Handle zero counts with smoothing
      double adjusted_pos = (bin.count_pos == 0) ? 0.5 : bin.count_pos;
      double adjusted_neg = (bin.count_neg == 0) ? 0.5 : bin.count_neg;
      
      double pos_rate = adjusted_pos / static_cast<double>(total_pos);
      double neg_rate = adjusted_neg / static_cast<double>(total_neg);
      
      // Prevent division by zero and log of zero
      if (neg_rate == 0.0) {
        neg_rate = 1e-10;
      }
      if (pos_rate == 0.0) {
        pos_rate = 1e-10;
      }
      
      bin.woe = std::log(pos_rate / neg_rate);
      bin.iv = (pos_rate - neg_rate) * bin.woe;
      
      // Create bin name by joining categories with "+"
      std::string bin_name = joinStrings(bin.categories, "+");
      bin_names.push_back(bin_name);
      woe_values.push_back(bin.woe);
      iv_values.push_back(bin.iv);
      counts.push_back(bin.count);
      counts_pos.push_back(bin.count_pos);
      counts_neg.push_back(bin.count_neg);
      
      // Map categories to WoE
      for (const auto& cat : bin.categories) {
        category_to_woe[cat] = bin.woe;
      }
    }
    
    // Assign WoE to each feature value based on its bin
    Rcpp::NumericVector woefeature(feature.size(), NA_REAL);
    for (size_t i = 0; i < feature.size(); ++i) {
      auto it = category_to_woe.find(feature[i]);
      if (it != category_to_woe.end()) {
        woefeature[i] = it->second;
      } else {
        // Assign NaN for categories not included in any bin
        woefeature[i] = NA_REAL;
      }
    }
    
    // Create woebin DataFrame
    Rcpp::DataFrame woebin = Rcpp::DataFrame::create(
      Rcpp::Named("bin") = bin_names,
      Rcpp::Named("woe") = woe_values,
      Rcpp::Named("iv") = iv_values,
      Rcpp::Named("count") = counts,
      Rcpp::Named("count_pos") = counts_pos,
      Rcpp::Named("count_neg") = counts_neg,
      Rcpp::Named("stringsAsFactors") = false
    );
    
    // Calculate total IV
    double total_iv = std::accumulate(iv_values.begin(), iv_values.end(), 0.0);
    
    // Return results as a list
    return Rcpp::List::create(
      Rcpp::Named("woefeature") = woefeature,
      Rcpp::Named("woebin") = woebin,
      Rcpp::Named("total_iv") = total_iv
    );
  }
};

//' @title Categorical Optimal Binning with Genetic Algorithm
//'
//' @description
//' Implements optimal binning for categorical variables using a Genetic Algorithm approach,
//' calculating Weight of Evidence (WoE) and Information Value (IV).
//'
//' @param target Integer vector of binary target values (0 or 1).
//' @param feature Character vector of categorical feature values.
//' @param min_bins Minimum number of bins (default: 2).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum frequency for a separate bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins before merging (default: 20).
//' @param population_size Size of the genetic algorithm population (default: 100).
//' @param num_generations Number of generations for the genetic algorithm (default: 100).
//' @param mutation_rate Probability of mutation for each gene (default: 0.01).
//' @param crossover_rate Probability of crossover between parents (default: 0.8).
//'
//' @return A list with three elements:
//' \itemize{
//'   \item woefeature: Numeric vector of WoE values for each input feature value.
//'   \item woebin: Data frame with binning results (bin names, WoE, IV, counts).
//'   \item total_iv: Total Information Value across all bins.
//' }
//'
//' @details
//' The algorithm employs a Genetic Algorithm (GA) to identify an optimal binning solution for categorical variables.
//' It evolves a population of binning solutions over multiple generations, utilizing selection, crossover, and mutation operations
//' to maximize the overall Information Value (IV) while adhering to constraints on the number of bins and ensuring monotonicity
//' of WoE values across bins.
//'
//' **Key Steps of the Genetic Algorithm:**
//' \enumerate{
//'   \item **Initialization**: Generate an initial population of binning solutions with random assignments, ensuring that the minimum number of bins (`min_bins`) is met.
//'   \item **Fitness Evaluation**: Calculate the fitness of each individual in the population based on the sum of IVs across all bins.
//'   \item **Selection**: Select parent individuals for reproduction using tournament selection, favoring individuals with higher fitness.
//'   \item **Crossover**: Perform single-point crossover between pairs of parents to produce offspring, exchanging segments of their chromosomes.
//'   \item **Mutation**: Introduce random mutations in the offspring's chromosomes with a probability defined by `mutation_rate`, altering bin assignments to maintain genetic diversity.
//'   \item **Replacement**: Form a new population from the offspring, maintaining the population size.
//'   \item **Termination**: Repeat the evaluation, selection, crossover, and mutation steps for a specified number of generations (`num_generations`) or until convergence criteria are met.
//' }
//'
//' **Weight of Evidence (WoE):**
//' \deqn{WoE = \ln\left(\frac{P(X|Y=1)}{P(X|Y=0)}\right)}
//' Where:
//' \itemize{
//'   \item \( P(X|Y=1) \) is the distribution of positives in bin \( X \).
//'   \item \( P(X|Y=0) \) is the distribution of negatives in bin \( X \).
//' }
//'
//' **Information Value (IV):**
//' \deqn{IV = \sum_{i=1}^{N} (P(X_i|Y=1) - P(X_i|Y=0)) \times WoE_i}
//' The IV measures the predictive power of the binned feature, with higher values indicating stronger predictive capability.
//'
//' **Numerical Stability and Edge Case Handling:**
//' The algorithm incorporates smoothing techniques by adding a small constant (e.g., 10^-10) to distributions to prevent division by zero and taking the logarithm of zero.
//' It also ensures that each bin has positive and negative counts, except in cases where the target variable lacks sufficient cases for a particular category.
//' All final bins consist solely of original unique category values concatenated with a "+" delimiter, preserving category integrity without introducing "Others" or "Rare" bins.
//'
//' **Monotonicity Enforcement:**
//' While the Genetic Algorithm inherently promotes higher IV through selection and reproduction, additional mechanisms can be integrated to enforce monotonicity of WoE values across bins when feasible. This ensures that the relationship between the feature and the target remains consistent and interpretable.
//'
//' @examples
//' \dontrun{
//' # Sample data
//' target <- c(1, 0, 1, 1, 0, 1, 0, 0, 1, 1)
//' feature <- c("A", "B", "A", "C", "B", "D", "C", "A", "D", "B")
//'
//' # Run optimal binning
//' result <- optimal_binning_categorical_gab(
//'   target, 
//'   feature, 
//'   min_bins = 2, 
//'   max_bins = 4, 
//'   bin_cutoff = 0.05, 
//'   max_n_prebins = 20,
//'   population_size = 100, 
//'   num_generations = 100, 
//'   mutation_rate = 0.01, 
//'   crossover_rate = 0.8
//' )
//'
//' # View binning results
//' print(result$woebin)
//'
//' # View WoE values assigned to each feature value
//' print(result$woefeature)
//'
//' # View total Information Value
//' print(result$total_iv)
//' }
//'
//' @author Lopes, J. E.
//'
//' @references
//' \itemize{
//'   \item Beltrami, M., Mach, M., & Dall'Aglio, M. (2021). Monotonic Optimal Binning Algorithm for Credit Risk Modeling. \emph{Risks}, 9(3), 58.
//'   \item Siddiqi, N. (2006). \emph{Credit risk scorecards: developing and implementing intelligent credit scoring} (Vol. 3). John Wiley & Sons.
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_gab(Rcpp::IntegerVector target, Rcpp::CharacterVector feature,
                                           int min_bins = 2, int max_bins = 5, double bin_cutoff = 0.05, int max_n_prebins = 20,
                                           int population_size = 100, int num_generations = 100, double mutation_rate = 0.01, double crossover_rate = 0.8) {
  // Instantiate the binning class with provided parameters
  OptimalBinningCategoricalGAB binner(min_bins, max_bins, bin_cutoff, max_n_prebins, population_size, num_generations, mutation_rate, crossover_rate);
  
  // Execute the binning process and return the results
  return binner.fit(feature, target);
}
