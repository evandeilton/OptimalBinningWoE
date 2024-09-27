#include <Rcpp.h>
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::plugins(openmp)]]
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

// Added for better floating-point comparisons
const double EPSILON = 1e-9;

// Define the OptimalBinningCategoricalGAB class
class OptimalBinningCategoricalGAB {
private:
  // Class members
  std::vector<std::string> feature;
  std::vector<int> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  size_t max_n_prebins;
  size_t population_size;
  size_t num_generations;
  double mutation_rate;
  double crossover_rate;
  std::chrono::seconds time_limit;

  struct Bin {
    std::vector<std::string> categories;
    double woe;
    double iv;
    int count;
    int count_pos;
    int count_neg;

    // Constructor to initialize variables
    Bin() : woe(0.0), iv(0.0), count(0), count_pos(0), count_neg(0) {}
  };

  struct Individual {
    std::vector<Bin> bins;
    double fitness;

    // Constructor to initialize fitness
    Individual() : fitness(0.0) {}
  };

  std::vector<Individual> population;

  // To store all unique categories
  std::unordered_set<std::string> all_categories;

  // Genetic Algorithm methods
  void initializePopulation();
  void evaluateFitness(Individual& ind);
  Individual selectParent(std::mt19937& gen);
  void crossover(const Individual& parent1, const Individual& parent2, Individual& offspring, std::mt19937& gen);
  void mutate(Individual& ind, std::mt19937& gen);
  void sortPopulation();
  bool isMonotonic(const Individual& ind);
  void mergeBins(Individual& ind);

  // Helper function for approximate floating-point comparison
  bool isApproximatelyGreaterEqual(double a, double b) const {
    return (a > b) || (std::abs(a - b) < EPSILON);
  }

public:
  // Constructor
  OptimalBinningCategoricalGAB(
    const std::vector<std::string>& feature,
    const std::vector<int>& target,
    int min_bins = 3,
    int max_bins = 5,
    double bin_cutoff = 0.05,
    size_t max_n_prebins = 20,
    size_t population_size = 100,
    size_t num_generations = 100,
    double mutation_rate = 0.1,
    double crossover_rate = 0.8,
    int time_limit_seconds = 300  // Added time limit parameter
  );

  // Main fitting function
  Rcpp::List fit();
};

// Constructor Implementation
OptimalBinningCategoricalGAB::OptimalBinningCategoricalGAB(
  const std::vector<std::string>& feature,
  const std::vector<int>& target,
  int min_bins,
  int max_bins,
  double bin_cutoff,
  size_t max_n_prebins,
  size_t population_size,
  size_t num_generations,
  double mutation_rate,
  double crossover_rate,
  int time_limit_seconds
) : feature(feature), target(target), min_bins(min_bins), max_bins(max_bins),
bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins), population_size(population_size),
num_generations(num_generations), mutation_rate(mutation_rate), crossover_rate(crossover_rate),
time_limit(std::chrono::seconds(time_limit_seconds)) {

  // Input validations
  if (feature.size() != target.size()) {
    Rcpp::stop("Feature and target vectors must have the same length");
  }
  if (min_bins < 2 || max_bins < min_bins) {
    Rcpp::stop("Invalid bin constraints: Ensure that min_bins >= 2 and max_bins >= min_bins");
  }
  if (bin_cutoff <= 0 || bin_cutoff >= 1) {
    Rcpp::stop("bin_cutoff must be between 0 and 1");
  }
  if (max_n_prebins < static_cast<size_t>(max_bins)) {
    Rcpp::stop("max_n_prebins must be greater than or equal to max_bins");
  }

  for (int t : target) {
    if (t != 0 && t != 1) {
      Rcpp::stop("Target must be binary (0 or 1)");
    }
  }

  // Initialize all_categories
  all_categories = std::unordered_set<std::string>(feature.begin(), feature.end());
}

// Initialize Population
void OptimalBinningCategoricalGAB::initializePopulation() {
  // Count the frequency of each category
  std::unordered_map<std::string, int> category_counts;
  for (const auto& cat : feature) {
    category_counts[cat]++;
  }

  // Sort categories by frequency in descending order
  std::vector<std::pair<std::string, int>> sorted_categories(category_counts.begin(), category_counts.end());
  std::sort(sorted_categories.begin(), sorted_categories.end(),
            [](const std::pair<std::string, int>& a, const std::pair<std::string, int>& b) {
              return a.second > b.second;
            });

  size_t total_count = feature.size();
  std::vector<std::string> valid_categories;
  for (const auto& pair : sorted_categories) {
    const std::string& cat = pair.first;
    int count = pair.second;
    if (static_cast<double>(count) / total_count >= bin_cutoff) {
      valid_categories.push_back(cat);
    }
    if (valid_categories.size() >= max_n_prebins) break;
  }

  if (valid_categories.empty()) {
    Rcpp::stop("No valid categories after applying bin_cutoff and max_n_prebins.");
  }

  population.resize(population_size);

  // Parallel initialization of the population
#pragma omp parallel
{
  // Each thread gets its own random number generator
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> bin_size_dis(min_bins, std::min(static_cast<int>(max_bins), static_cast<int>(valid_categories.size())));

#pragma omp for
  for (size_t i = 0; i < population_size; ++i) {
    std::vector<std::string> shuffled_categories = valid_categories;
    std::shuffle(shuffled_categories.begin(), shuffled_categories.end(), gen);

    size_t num_bins = bin_size_dis(gen);
    population[i].bins.resize(num_bins);

    for (size_t j = 0; j < shuffled_categories.size(); ++j) {
      population[i].bins[j % num_bins].categories.push_back(shuffled_categories[j]);
    }

    // Assign any missing categories to random bins
    for (const auto& cat : all_categories) {
      bool assigned = false;
      for (const auto& bin : population[i].bins) {
        if (std::find(bin.categories.begin(), bin.categories.end(), cat) != bin.categories.end()) {
          assigned = true;
          break;
        }
      }
      if (!assigned) {
        std::uniform_int_distribution<size_t> bin_dis(0, num_bins - 1);
        size_t bin_idx = bin_dis(gen);
        population[i].bins[bin_idx].categories.push_back(cat);
      }
    }

    evaluateFitness(population[i]);
  }
}
}

// Evaluate Fitness of an Individual
void OptimalBinningCategoricalGAB::evaluateFitness(Individual& ind) {
  double total_iv = 0.0;
  size_t total_count = feature.size();
  int total_pos = std::count(target.begin(), target.end(), 1);
  int total_neg = total_count - total_pos;

  if (total_pos == 0 || total_neg == 0) {
    Rcpp::stop("Target variable must have both positive and negative classes.");
  }

  for (auto& bin : ind.bins) {
    bin.count = 0;
    bin.count_pos = 0;
    bin.count_neg = 0;

    for (size_t i = 0; i < feature.size(); ++i) {
      if (std::find(bin.categories.begin(), bin.categories.end(), feature[i]) != bin.categories.end()) {
        bin.count++;
        if (target[i] == 1) {
          bin.count_pos++;
        } else {
          bin.count_neg++;
        }
      }
    }

    double pos_rate = static_cast<double>(bin.count_pos) / total_pos;
    double neg_rate = static_cast<double>(bin.count_neg) / total_neg;

    if (pos_rate > 0 && neg_rate > 0) {
      bin.woe = std::log(pos_rate / neg_rate);
      bin.iv = (pos_rate - neg_rate) * bin.woe;
    } else {
      bin.woe = 0.0;
      bin.iv = 0.0;
    }

    total_iv += bin.iv;
  }

  ind.fitness = total_iv;
}

// Select Parent using Fitness Proportionate Selection (Roulette Wheel)
OptimalBinningCategoricalGAB::Individual OptimalBinningCategoricalGAB::selectParent(std::mt19937& gen) {
  double total_fitness = 0.0;
  for (const auto& ind : population) {
    total_fitness += ind.fitness;
  }

  if (total_fitness == 0.0) {
    // If total fitness is zero, return a random individual
    std::uniform_int_distribution<size_t> index_dis(0, population.size() - 1);
    return population[index_dis(gen)];
  }

  std::uniform_real_distribution<> dis_fitness(0, total_fitness);
  double r = dis_fitness(gen);
  double sum = 0.0;

  for (const auto& ind : population) {
    sum += ind.fitness;
    if (sum >= r) {
      return ind;
    }
  }

  return population.back();
}

// Crossover between two parents to produce an offspring
void OptimalBinningCategoricalGAB::crossover(const Individual& parent1, const Individual& parent2, Individual& offspring, std::mt19937& gen) {
  size_t bins_p1 = parent1.bins.size();
  size_t bins_p2 = parent2.bins.size();

  if (bins_p1 == 0 || bins_p2 == 0) {
    // If any parent has no bins, copy the other parent
    offspring = (bins_p1 == 0) ? parent2 : parent1;
    return;
  }

  size_t min_bins_size = std::min(bins_p1, bins_p2);

  if (min_bins_size < 2) {
    // If min_bins_size is less than 2, cannot perform crossover
    std::uniform_int_distribution<int> parent_choice(0, 1);
    offspring = (parent_choice(gen) == 0) ? parent1 : parent2;
    return;
  }

  std::uniform_int_distribution<size_t> dis(1, min_bins_size - 1); // Ensure at least one bin from each parent
  size_t crossover_point = dis(gen);

  offspring.bins.clear();
  offspring.bins.reserve(bins_p1 + bins_p2);

  // Copy first part from parent1
  for (size_t i = 0; i < crossover_point; ++i) {
    offspring.bins.push_back(parent1.bins[i]);
  }

  // Copy remaining part from parent2
  for (size_t i = crossover_point; i < bins_p2; ++i) {
    offspring.bins.push_back(parent2.bins[i]);
  }

  // Remove duplicate categories and ensure all categories are assigned
  std::unordered_set<std::string> used_categories;
  for (auto& bin : offspring.bins) {
    std::vector<std::string> unique_categories;
    for (const auto& cat : bin.categories) {
      if (used_categories.find(cat) == used_categories.end()) {
        unique_categories.push_back(cat);
        used_categories.insert(cat);
      }
    }
    bin.categories = unique_categories;
  }

  // Assign any missing categories to random bins
  for (const auto& cat : all_categories) {
    if (used_categories.find(cat) == used_categories.end()) {
      std::uniform_int_distribution<size_t> bin_dis(0, offspring.bins.size() - 1);
      size_t bin_idx = bin_dis(gen);
      offspring.bins[bin_idx].categories.push_back(cat);
      used_categories.insert(cat);
    }
  }
}

// Mutation: Move a category from one bin to another
void OptimalBinningCategoricalGAB::mutate(Individual& ind, std::mt19937& gen) {
  std::uniform_real_distribution<> dis(0, 1);

  if (ind.bins.size() < 2) {
    // Cannot perform mutation if there is only one bin
    return;
  }

  std::uniform_int_distribution<size_t> bin_dis(0, ind.bins.size() - 1);

  for (size_t bin_idx = 0; bin_idx < ind.bins.size(); ++bin_idx) {
    if (dis(gen) < mutation_rate) {
      if (!ind.bins[bin_idx].categories.empty()) {
        std::uniform_int_distribution<size_t> cat_dis(0, ind.bins[bin_idx].categories.size() - 1);
        size_t category_index = cat_dis(gen);
        std::string category = ind.bins[bin_idx].categories[category_index];

        // Only remove if there's more than one category in the bin
        if (ind.bins[bin_idx].categories.size() > 1) {
          ind.bins[bin_idx].categories.erase(ind.bins[bin_idx].categories.begin() + category_index);

          // Select a new bin different from the current one
          size_t new_bin_index;
          do {
            new_bin_index = bin_dis(gen);
          } while (new_bin_index == bin_idx);

          ind.bins[new_bin_index].categories.push_back(category);
        }
      }
    }
  }
}

// Sort population based on fitness in descending order
void OptimalBinningCategoricalGAB::sortPopulation() {
  std::sort(population.begin(), population.end(),
            [](const Individual& a, const Individual& b) {
              return a.fitness > b.fitness;
            });
}

// Check if the WOE values are monotonically increasing or decreasing
bool OptimalBinningCategoricalGAB::isMonotonic(const Individual& ind) {
  if (ind.bins.empty()) return true;
  double prev_woe = ind.bins[0].woe;
  for (size_t i = 1; i < ind.bins.size(); ++i) {
    if (!isApproximatelyGreaterEqual(ind.bins[i].woe, prev_woe)) {
      return false;
    }
    prev_woe = ind.bins[i].woe;
  }
  return true;
}

// Merge bins to ensure the number of bins does not exceed max_bins
void OptimalBinningCategoricalGAB::mergeBins(Individual& ind) {
  while (ind.bins.size() > static_cast<size_t>(max_bins)) {
    if (ind.bins.size() < 2) break; // Safety check

    double min_iv_loss = std::numeric_limits<double>::max();
    size_t merge_index = 0;

    // Find the pair of adjacent bins with the smallest combined IV
    for (size_t i = 0; i < ind.bins.size() - 1; ++i) {
      double iv_loss = ind.bins[i].iv + ind.bins[i + 1].iv;
      if (iv_loss < min_iv_loss) {
        min_iv_loss = iv_loss;
        merge_index = i;
      }
    }

    // Merge bin at merge_index with the next bin
    ind.bins[merge_index].categories.insert(
        ind.bins[merge_index].categories.end(),
        ind.bins[merge_index + 1].categories.begin(),
        ind.bins[merge_index + 1].categories.end()
    );
    ind.bins.erase(ind.bins.begin() + merge_index + 1);

    // Recalculate WOE and IV after merging
    evaluateFitness(ind);
  }
}

// Main Genetic Algorithm Fit Function
Rcpp::List OptimalBinningCategoricalGAB::fit() {
  initializePopulation();

  auto start_time = std::chrono::steady_clock::now();

  // Main genetic algorithm loop
  for (size_t gen = 0; gen < num_generations; ++gen) {
    std::vector<Individual> new_population;
    new_population.reserve(population_size);

    // Check time limit
    auto current_time = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time) > time_limit) {
      Rcpp::warning("Time limit reached. Terminating optimization early.");
      break;
    }

    // Parallel generation of new population
#pragma omp parallel
{
  // Each thread gets its own random number generator
  std::random_device rd;
  std::mt19937 gen_thread(rd());

  std::vector<Individual> local_new_population;

#pragma omp for nowait
  for (size_t i = 0; i < population_size; ++i) {
    // Periodically check for user interrupts
    if (i % (population_size / 10 + 1) == 0) {
#pragma omp critical
      Rcpp::checkUserInterrupt();
    }

    // Select parents
    Individual parent1 = selectParent(gen_thread);
    Individual parent2 = selectParent(gen_thread);
    Individual offspring;

    // Crossover
    std::uniform_real_distribution<> dis(0, 1);
    if (dis(gen_thread) < crossover_rate) {
      crossover(parent1, parent2, offspring, gen_thread);
    } else {
      offspring = parent1;
    }

    // Mutation
    mutate(offspring, gen_thread);

    // Merge bins to respect max_bins constraint
    mergeBins(offspring);

    // Evaluate fitness
    evaluateFitness(offspring);

    // Check for monotonicity
    if (isMonotonic(offspring)) {
#pragma omp critical
      local_new_population.push_back(offspring);
    }
  }

  // Merge local populations into the global new_population
#pragma omp critical
{
  new_population.insert(new_population.end(), local_new_population.begin(), local_new_population.end());
}
}

// If no new individuals were added, retain the best individuals from the current population
if (new_population.empty()) {
  sortPopulation();
  if (population.size() < population_size) {
    Rcpp::stop("Population size is smaller than expected.");
  }
  new_population = std::vector<Individual>(population.begin(), population.begin() + population_size);
} else if (new_population.size() < population_size) {
  sortPopulation();
  size_t to_add = population_size - new_population.size();
  to_add = std::min(to_add, population.size());
  new_population.insert(new_population.end(), population.begin(), population.begin() + to_add);
}

population = new_population;
sortPopulation();

// Early stopping if the best solution hasn't improved for a while
if (gen > 10 && std::abs(population[0].fitness - population[population_size - 1].fitness) < EPSILON) {
  break;
}
  }

  // After all generations, select the best individual
  if (population.empty()) {
    Rcpp::stop("No valid individuals found in the population.");
  }
  Individual best = population[0];

  // Prepare vectors for woebin DataFrame
  Rcpp::CharacterVector bin_col;
  Rcpp::NumericVector woe_col;
  Rcpp::NumericVector iv_col;
  Rcpp::IntegerVector count_col;
  Rcpp::IntegerVector count_pos_col;
  Rcpp::IntegerVector count_neg_col;

  // Prepare woefeature vector
  Rcpp::NumericVector woefeature(feature.size());

  // Create a map from category to WOE for efficient assignment
  std::unordered_map<std::string, double> category_to_woe;
  for (const auto& bin : best.bins) {
    for (const auto& cat : bin.categories) {
      category_to_woe[cat] = bin.woe;
    }
  }

  // Populate woebin DataFrame columns
  for (const auto& bin : best.bins) {
    // Concatenate categories with a comma separator
    std::string joined_categories;
    for (size_t i = 0; i < bin.categories.size(); ++i) {
      joined_categories += bin.categories[i];
      if (i != bin.categories.size() - 1) {
        joined_categories += "+";
      }
    }

    bin_col.push_back(joined_categories);
    woe_col.push_back(bin.woe);
    iv_col.push_back(bin.iv);
    count_col.push_back(bin.count);
    count_pos_col.push_back(bin.count_pos);
    count_neg_col.push_back(bin.count_neg);
  }

  // Assign WOE values to each feature based on their bin
  for (size_t i = 0; i < feature.size(); ++i) {
    auto it = category_to_woe.find(feature[i]);
    if (it != category_to_woe.end()) {
      woefeature[i] = it->second;
    } else {
      woefeature[i] = R_NaN; // Assign NaN if category not found in any bin
    }
  }

  // Create the woebin DataFrame
  Rcpp::DataFrame woebin = Rcpp::DataFrame::create(
    Rcpp::Named("bin") = bin_col,
    Rcpp::Named("woe") = woe_col,
    Rcpp::Named("iv") = iv_col,
    Rcpp::Named("count") = count_col,
    Rcpp::Named("count_pos") = count_pos_col,
    Rcpp::Named("count_neg") = count_neg_col,
    Rcpp::Named("stringsAsFactors") = false
  );

  return Rcpp::List::create(
    Rcpp::Named("woefeature") = woefeature,
    Rcpp::Named("woebin") = woebin
  );
}


//' @title Categorical Optimal Binning with Genetic Algorithm
//'
//' @description
//' Implements optimal binning for categorical variables using a Genetic Algorithm approach,
//' calculating Weight of Evidence (WoE) and Information Value (IV).
//'
//' @param target Integer vector of binary target values (0 or 1).
//' @param feature Character vector of categorical feature values.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum frequency for a separate bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins before merging (default: 20).
//' @param population_size Size of the genetic algorithm population (default: 100).
//' @param num_generations Number of generations for the genetic algorithm (default: 100).
//' @param mutation_rate Probability of mutation for each bin (default: 0.1).
//' @param crossover_rate Probability of crossover between parents (default: 0.8).
//' @param time_limit_seconds Maximum execution time in seconds (default: 300).
//'
//' @return A list with two elements:
//' \itemize{
//'   \item woefeature: Numeric vector of WoE values for each input feature value.
//'   \item woebin: Data frame with binning results (bin names, WoE, IV, counts).
//' }
//'
//' @details
//' The algorithm uses a genetic algorithm approach to find an optimal binning solution.
//' It evolves a population of binning solutions over multiple generations, using
//' selection, crossover, and mutation operations.
//'
//' Weight of Evidence (WoE) for each bin is calculated as:
//'
//' \deqn{WoE = \ln\left(\frac{P(X|Y=1)}{P(X|Y=0)}\right)}
//'
//' Information Value (IV) for each bin is calculated as:
//'
//' \deqn{IV = (P(X|Y=1) - P(X|Y=0)) \times WoE}
//'
//' The fitness of each individual (binning solution) is the sum of IVs across all bins.
//' The algorithm aims to maximize this fitness while respecting constraints on the
//' number of bins and ensuring monotonicity of WoE values across bins.
//'
//' The genetic algorithm includes the following key steps:
//' \enumerate{
//'   \item Initialize population with random binning solutions.
//'   \item Evaluate fitness of each individual.
//'   \item Select parents based on fitness (using roulette wheel selection).
//'   \item Create offspring through crossover and mutation.
//'   \item Ensure offspring respect constraints (number of bins, monotonicity).
//'   \item Replace population with offspring.
//'   \item Repeat steps 2-6 for the specified number of generations or until time limit is reached.
//' }
//'
//' @examples
//' \dontrun{
//' # Sample data
//' target <- c(1, 0, 1, 1, 0, 1, 0, 0, 1, 1)
//' feature <- c("A", "B", "A", "C", "B", "D", "C", "A", "D", "B")
//'
//' # Run optimal binning
//' result <- optimal_binning_categorical_gab(target, feature, min_bins = 2, max_bins = 4)
//'
//' # View results
//' print(result$woebin)
//' print(result$woefeature)
//' }
//'
//' @author Lopes, J. E.
//'
//' @references
//' \itemize{
//'   \item Holland, J. H. (1992). Adaptation in natural and artificial systems: an introductory analysis with applications to biology, control, and artificial intelligence. MIT press.
//'   \item Whitley, D. (1994). A genetic algorithm tutorial. Statistics and computing, 4(2), 65-85.
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_gab(
  Rcpp::IntegerVector target,
  Rcpp::CharacterVector feature,
  int min_bins = 3,
  int max_bins = 5,
  double bin_cutoff = 0.05,
  size_t max_n_prebins = 20,
  size_t population_size = 100,
  size_t num_generations = 100,
  double mutation_rate = 0.1,
  double crossover_rate = 0.8,
  int time_limit_seconds = 300
) {
std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);

OptimalBinningCategoricalGAB binner(
    feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins,
    population_size, num_generations, mutation_rate, crossover_rate, time_limit_seconds
);

return binner.fit();
}
