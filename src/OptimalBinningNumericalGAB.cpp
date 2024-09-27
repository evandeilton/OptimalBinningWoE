#include <Rcpp.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <limits>
#include <sstream>
#include <numeric>

// [[Rcpp::plugins(openmp)]]

class OptimalBinningNumericalGAB {
public:
  OptimalBinningNumericalGAB(const std::vector<double>& target,
                             const std::vector<double>& feature,
                             int min_bins,
                             int max_bins,
                             double bin_cutoff,
                             int max_n_prebins);
  
  void fit();
  
  std::vector<double> get_woefeature();
  Rcpp::DataFrame get_woebin();
  
private:
  std::vector<double> feature_;
  std::vector<double> target_;
  int min_bins_;
  int max_bins_;
  double bin_cutoff_;
  int max_n_prebins_;
  
  std::vector<double> woefeature_;
  Rcpp::DataFrame woebin_;
  
  // Additional private methods and variables
  void prebin_data();
  void initialize_population();
  void evaluate_fitness();
  void selection();
  void crossover();
  void mutation();
  bool check_monotonicity(const std::vector<double>& cut_points);
  double calculate_iv(const std::vector<double>& cut_points);
  void adjust_bins_for_cutoff(std::vector<double>& cut_points);
  
  // Data structures for GA
  struct Individual {
    std::vector<double> cut_points;
    double fitness;
  };
  
  std::vector<Individual> population_;
  int population_size_;
  int max_generations_;
  double mutation_rate_;
  
  std::mt19937 rng_;
  
  std::vector<double> prebin_cut_points_;
};

OptimalBinningNumericalGAB::OptimalBinningNumericalGAB(const std::vector<double>& target,
                                                       const std::vector<double>& feature,
                                                       int min_bins,
                                                       int max_bins,
                                                       double bin_cutoff,
                                                       int max_n_prebins)
  : feature_(feature),
    target_(target),
    min_bins_(min_bins),
    max_bins_(max_bins),
    bin_cutoff_(bin_cutoff),
    max_n_prebins_(max_n_prebins),
    population_size_(50),
    max_generations_(100),
    mutation_rate_(0.1)
{
  // Validate inputs
  if (min_bins_ < 2) {
    throw std::invalid_argument("min_bins must be >= 2.");
  }
  if (max_bins_ < min_bins_) {
    throw std::invalid_argument("max_bins must be >= min_bins.");
  }
  if (bin_cutoff_ <= 0 || bin_cutoff_ >= 1) {
    throw std::invalid_argument("bin_cutoff must be between 0 and 1.");
  }
  if (max_n_prebins_ <= 0) {
    throw std::invalid_argument("max_n_prebins must be > 0.");
  }
  if (feature_.size() != target_.size()) {
    throw std::invalid_argument("feature and target must have the same length.");
  }
  // Initialize random number generator
  std::random_device rd;
  rng_ = std::mt19937(rd());
  
  prebin_data();
}

void OptimalBinningNumericalGAB::prebin_data() {
  // Pre-bin data into max_n_prebins bins
  std::vector<double> sorted_feature = feature_;
  std::sort(sorted_feature.begin(), sorted_feature.end());
  
  std::vector<double> cut_points;
  for (int i = 1; i < max_n_prebins_; ++i) {
    size_t idx = i * sorted_feature.size() / max_n_prebins_;
    cut_points.push_back(sorted_feature[idx]);
  }
  
  // Remove duplicates
  std::sort(cut_points.begin(), cut_points.end());
  cut_points.erase(std::unique(cut_points.begin(), cut_points.end()), cut_points.end());
  
  prebin_cut_points_ = cut_points;
}

void OptimalBinningNumericalGAB::initialize_population() {
  population_.clear();
  std::uniform_int_distribution<int> bins_dist(min_bins_, max_bins_);
  
  for (int i = 0; i < population_size_; ++i) {
    int n_bins = bins_dist(rng_);
    std::vector<double> cut_points = prebin_cut_points_;
    
    // Randomly select n_bins - 1 cut points
    if (cut_points.size() > static_cast<size_t>(n_bins - 1)) {
      std::shuffle(cut_points.begin(), cut_points.end(), rng_);
      cut_points.resize(n_bins - 1);
    } else if (cut_points.size() < static_cast<size_t>(n_bins - 1)) {
      // If not enough prebin cut points, generate random ones
      std::uniform_real_distribution<double> cut_point_dist(*std::min_element(feature_.begin(), feature_.end()),
                                                            *std::max_element(feature_.begin(), feature_.end()));
      while (cut_points.size() < static_cast<size_t>(n_bins - 1)) {
        double cp = cut_point_dist(rng_);
        // Ensure uniqueness
        if (std::find(cut_points.begin(), cut_points.end(), cp) == cut_points.end()) {
          cut_points.push_back(cp);
        }
      }
    }
    std::sort(cut_points.begin(), cut_points.end());
    adjust_bins_for_cutoff(cut_points);
    Individual ind = { cut_points, 0.0 };
    population_.push_back(ind);
  }
}

void OptimalBinningNumericalGAB::evaluate_fitness() {
#pragma omp parallel for
  for (size_t i = 0; i < population_.size(); ++i) {
    Individual& ind = population_[i];
    // Check monotonicity
    if (!check_monotonicity(ind.cut_points)) {
      ind.fitness = -std::numeric_limits<double>::infinity();
      continue;
    }
    // Calculate IV
    double iv = calculate_iv(ind.cut_points);
    ind.fitness = iv;
  }
}

void OptimalBinningNumericalGAB::selection() {
  // Sort population by fitness
  std::sort(population_.begin(), population_.end(), [](const Individual& a, const Individual& b) {
    return a.fitness > b.fitness;
  });
  
  // Keep the top individuals (elitism)
  int num_elites = population_size_ / 2;
  if (population_.size() > static_cast<size_t>(num_elites)) {
    population_.resize(num_elites);
  }
}

void OptimalBinningNumericalGAB::crossover() {
  std::uniform_int_distribution<int> parent_dist(0, population_.size() - 1);
  std::uniform_real_distribution<double> crossover_point_dist(0.0, 1.0);
  
  std::vector<Individual> new_population = population_;
  
  while (new_population.size() < static_cast<size_t>(population_size_)) {
    int parent1_idx = parent_dist(rng_);
    int parent2_idx = parent_dist(rng_);
    const Individual& parent1 = population_[parent1_idx];
    const Individual& parent2 = population_[parent2_idx];
    
    // Perform single-point crossover
    double crossover_point = crossover_point_dist(rng_);
    size_t split = static_cast<size_t>(crossover_point * parent1.cut_points.size());
    
    std::vector<double> child_cut_points;
    child_cut_points.insert(child_cut_points.end(), parent1.cut_points.begin(), parent1.cut_points.begin() + split);
    child_cut_points.insert(child_cut_points.end(), parent2.cut_points.begin() + split, parent2.cut_points.end());
    
    // Remove duplicates and sort
    std::sort(child_cut_points.begin(), child_cut_points.end());
    child_cut_points.erase(std::unique(child_cut_points.begin(), child_cut_points.end()), child_cut_points.end());
    
    // Ensure number of bins within limits
    if (child_cut_points.size() >= static_cast<size_t>(min_bins_ - 1) &&
        child_cut_points.size() <= static_cast<size_t>(max_bins_ - 1)) {
      adjust_bins_for_cutoff(child_cut_points);
      // Final check to ensure bin count is within constraints
      if (child_cut_points.size() >= static_cast<size_t>(min_bins_ - 1) &&
          child_cut_points.size() <= static_cast<size_t>(max_bins_ - 1)) {
        Individual child = { child_cut_points, 0.0 };
        new_population.push_back(child);
      }
    }
  }
  
  population_ = new_population;
}

void OptimalBinningNumericalGAB::mutation() {
  std::uniform_real_distribution<double> mutation_dist(0.0, 1.0);
  std::uniform_real_distribution<double> cut_point_dist(*std::min_element(feature_.begin(), feature_.end()),
                                                        *std::max_element(feature_.begin(), feature_.end()));
  
  for (Individual& ind : population_) {
    if (mutation_dist(rng_) < mutation_rate_) {
      // Mutate individual
      if (!ind.cut_points.empty()) {
        std::uniform_int_distribution<size_t> index_dist(0, ind.cut_points.size() - 1);
        size_t idx = index_dist(rng_);
        double new_cp = cut_point_dist(rng_);
        // Ensure the new cut point maintains order
        ind.cut_points[idx] = new_cp;
        std::sort(ind.cut_points.begin(), ind.cut_points.end());
        // Remove duplicates
        ind.cut_points.erase(std::unique(ind.cut_points.begin(), ind.cut_points.end()), ind.cut_points.end());
        adjust_bins_for_cutoff(ind.cut_points);
      }
    }
  }
}

bool OptimalBinningNumericalGAB::check_monotonicity(const std::vector<double>& cut_points) {
  // Bin the feature
  std::vector<size_t> bins(feature_.size());
  for (size_t i = 0; i < feature_.size(); ++i) {
    double val = feature_[i];
    size_t bin = 0;
    while (bin < cut_points.size() && val > cut_points[bin]) {
      ++bin;
    }
    bins[i] = bin;
  }
  
  // Calculate mean target in each bin
  std::vector<double> bin_means(cut_points.size() + 1, 0.0);
  std::vector<int> bin_counts(cut_points.size() + 1, 0);
  for (size_t i = 0; i < bins.size(); ++i) {
    size_t bin = bins[i];
    bin_means[bin] += target_[i];
    bin_counts[bin] += 1;
  }
  for (size_t i = 0; i < bin_means.size(); ++i) {
    if (bin_counts[i] > 0) {
      bin_means[i] /= bin_counts[i];
    } else {
      // Empty bin, cannot assess monotonicity
      return false;
    }
  }
  
  // Check if bin_means are monotonic
  bool increasing = true;
  bool decreasing = true;
  for (size_t i = 1; i < bin_means.size(); ++i) {
    if (bin_means[i] < bin_means[i - 1]) {
      increasing = false;
    }
    if (bin_means[i] > bin_means[i - 1]) {
      decreasing = false;
    }
  }
  return increasing || decreasing;
}

double OptimalBinningNumericalGAB::calculate_iv(const std::vector<double>& cut_points) {
  std::vector<size_t> bins(feature_.size());
  for (size_t i = 0; i < feature_.size(); ++i) {
    double val = feature_[i];
    size_t bin = 0;
    while (bin < cut_points.size() && val > cut_points[bin]) {
      ++bin;
    }
    bins[i] = bin;
  }
  
  // Calculate counts in each bin
  size_t n_bins = cut_points.size() + 1;
  std::vector<double> count_pos(n_bins, 0.0);
  std::vector<double> count_neg(n_bins, 0.0);
  double total_pos = 0.0;
  double total_neg = 0.0;
  for (size_t i = 0; i < bins.size(); ++i) {
    size_t bin = bins[i];
    if (target_[i] == 1.0) {
      count_pos[bin] += 1.0;
      total_pos += 1.0;
    } else if (target_[i] == 0.0) {
      count_neg[bin] += 1.0;
      total_neg += 1.0;
    }
  }
  
  // Calculate WoE and IV
  double iv = 0.0;
  for (size_t i = 0; i < n_bins; ++i) {
    double dist_pos = count_pos[i] / total_pos;
    double dist_neg = count_neg[i] / total_neg;
    if (dist_pos == 0.0 || dist_neg == 0.0) {
      continue;
    }
    double woe = std::log(dist_pos / dist_neg);
    iv += (dist_pos - dist_neg) * woe;
  }
  return iv;
}


void OptimalBinningNumericalGAB::adjust_bins_for_cutoff(std::vector<double>& cut_points) {
  // Implement bin merging based on bin_cutoff
  bool bins_adjusted = false;
  do {
    bins_adjusted = false;
    std::vector<size_t> bins(feature_.size());
    for (size_t i = 0; i < feature_.size(); ++i) {
      double val = feature_[i];
      size_t bin = 0;
      while (bin < cut_points.size() && val > cut_points[bin]) {
        ++bin;
      }
      bins[i] = bin;
    }
    
    size_t n_bins = cut_points.size() + 1;
    std::vector<double> bin_counts(n_bins, 0.0);
    for (size_t i = 0; i < bins.size(); ++i) {
      size_t bin = bins[i];
      bin_counts[bin] += 1.0;
    }
    
    double total_count = feature_.size();
    for (size_t i = 0; i < n_bins; ++i) {
      double bin_freq = bin_counts[i] / total_count;
      if (bin_freq < bin_cutoff_) {
        // Check if merging this bin would violate min_bins constraint
        if (cut_points.size() < static_cast<size_t>(min_bins_ - 1)) {
          // Cannot merge further without violating min_bins
          continue;
        }
        
        bins_adjusted = true;
        // Merge bin with smallest neighbor
        if (i == 0) {
          // Merge with next bin
          if (!cut_points.empty()) {
            cut_points.erase(cut_points.begin());
          }
        } else if (i == n_bins - 1) {
          // Merge with previous bin
          if (!cut_points.empty()) {
            cut_points.erase(cut_points.end() - 1);
          }
        } else {
          // Merge with neighbor with fewer counts
          if (bin_counts[i - 1] < bin_counts[i + 1]) {
            cut_points.erase(cut_points.begin() + i - 1);
          } else {
            cut_points.erase(cut_points.begin() + i);
          }
        }
        break; // After adjusting, re-evaluate from the beginning
      }
    }
  } while (bins_adjusted && cut_points.size() >= static_cast<size_t>(min_bins_ - 1));
}

void OptimalBinningNumericalGAB::fit() {
  initialize_population();
  Individual best_individual;
  double best_fitness = -std::numeric_limits<double>::infinity();
  
  for (int gen = 0; gen < max_generations_; ++gen) {
    evaluate_fitness();
    
    // Update best individual
    for (const Individual& ind : population_) {
      if (ind.fitness > best_fitness) {
        best_fitness = ind.fitness;
        best_individual = ind;
      }
    }
    
    selection();
    crossover();
    mutation();
  }
  
  // After GA, set the best solution
  // Create woefeature and woebin
  
  // Bin the feature using best cut points
  std::vector<size_t> bins(feature_.size());
  for (size_t i = 0; i < feature_.size(); ++i) {
    double val = feature_[i];
    size_t bin = 0;
    while (bin < best_individual.cut_points.size() && val > best_individual.cut_points[bin]) {
      ++bin;
    }
    bins[i] = bin;
  }
  
  // Calculate counts in each bin
  size_t n_bins = best_individual.cut_points.size() + 1;
  std::vector<double> count_pos(n_bins, 0.0);
  std::vector<double> count_neg(n_bins, 0.0);
  double total_pos = 0.0;
  double total_neg = 0.0;
  for (size_t i = 0; i < bins.size(); ++i) {
    size_t bin = bins[i];
    if (target_[i] == 1.0) {
      count_pos[bin] += 1.0;
      total_pos += 1.0;
    } else if (target_[i] == 0.0) {
      count_neg[bin] += 1.0;
      total_neg += 1.0;
    }
  }
  
  // Calculate WoE and IV for each bin
  woefeature_.resize(feature_.size());
  std::vector<std::string> bin_names;
  std::vector<double> woe_values(n_bins);
  std::vector<double> iv_values(n_bins);
  std::vector<double> counts(n_bins);
  std::vector<double> count_pos_vec(n_bins);
  std::vector<double> count_neg_vec(n_bins);
  
  for (size_t i = 0; i < n_bins; ++i) {
    double dist_pos = count_pos[i] / total_pos;
    double dist_neg = count_neg[i] / total_neg;
    if (dist_pos == 0.0 || dist_neg == 0.0) {
      woe_values[i] = 0.0;
    } else {
      woe_values[i] = std::log(dist_pos / dist_neg);
    }
    iv_values[i] = (dist_pos - dist_neg) * woe_values[i];
    counts[i] = count_pos[i] + count_neg[i];
    count_pos_vec[i] = count_pos[i];
    count_neg_vec[i] = count_neg[i];
    
    // Create bin names
    std::string bin_name;
    std::ostringstream oss_lower, oss_upper;
    oss_lower.precision(4);
    oss_upper.precision(4);
    oss_lower << std::fixed;
    oss_upper << std::fixed;
    if (i == 0) {
      oss_upper << best_individual.cut_points[0];
      bin_name = "(-Inf;" + oss_upper.str() + "]";
    } else if (i == n_bins - 1) {
      oss_lower << best_individual.cut_points[i - 1];
      bin_name = "(" + oss_lower.str() + ";+Inf]";
    } else {
      oss_lower << best_individual.cut_points[i - 1];
      oss_upper << best_individual.cut_points[i];
      bin_name = "(" + oss_lower.str() + ";" + oss_upper.str() + "]";
    }
    bin_names.push_back(bin_name);
  }
  
  // Assign WoE values to the feature
  for (size_t i = 0; i < bins.size(); ++i) {
    size_t bin = bins[i];
    woefeature_[i] = woe_values[bin];
  }
  
  // Set outputs
  woebin_ = Rcpp::DataFrame::create(
    Rcpp::Named("bin") = bin_names,
    Rcpp::Named("woe") = woe_values,
    Rcpp::Named("iv") = iv_values,
    Rcpp::Named("count") = counts,
    Rcpp::Named("count_pos") = count_pos_vec,
    Rcpp::Named("count_neg") = count_neg_vec
  );
  
  // Ensure that the number of bins respects min_bins and max_bins
  if (n_bins < static_cast<size_t>(min_bins_)) {
    throw std::runtime_error("Number of bins after fitting is less than min_bins.");
  }
  if (n_bins > static_cast<size_t>(max_bins_)) {
    throw std::runtime_error("Number of bins after fitting exceeds max_bins.");
  }
}

std::vector<double> OptimalBinningNumericalGAB::get_woefeature() {
  return woefeature_;
}

Rcpp::DataFrame OptimalBinningNumericalGAB::get_woebin() {
  return woebin_;
}

//' @title Optimal Binning for Numerical Variables using Genetic Algorithm
//' 
//' @description
//' This function implements an optimal binning algorithm for numerical variables using a genetic algorithm approach. It aims to find the best binning strategy that maximizes the Information Value (IV) while ensuring monotonicity in the Weight of Evidence (WoE) values.
//' 
//' @param target A numeric vector of binary target values (0 or 1).
//' @param feature A numeric vector of feature values to be binned.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum fraction of total observations in each bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins (default: 20).
//' 
//' @return A list containing:
//' \item{woefeature}{A numeric vector of Weight of Evidence (WoE) values for each observation}
//' \item{woebin}{A data frame with binning information, including bin ranges, WoE, IV, and counts}
//' 
//' @details
//' The optimal binning algorithm using a genetic algorithm approach consists of several steps:
//' 
//' 1. Pre-binning: The feature is initially divided into a maximum number of bins specified by \code{max_n_prebins}.
//' 2. Genetic Algorithm:
//'    a. Initialization: Create a population of potential binning solutions.
//'    b. Evaluation: Calculate the fitness (Information Value) of each solution.
//'    c. Selection: Choose the best solutions for reproduction.
//'    d. Crossover: Create new solutions by combining existing ones.
//'    e. Mutation: Introduce small random changes to maintain diversity.
//'    f. Repeat b-e for a specified number of generations.
//' 3. Monotonicity check: Ensure that the WoE values are either monotonically increasing or decreasing across the bins.
//' 4. Bin adjustment: Merge bins that have fewer observations than specified by \code{bin_cutoff}.
//' 
//' The Weight of Evidence (WoE) for each bin is calculated as:
//' 
//' \deqn{WoE = \ln\left(\frac{P(X|Y=1)}{P(X|Y=0)}\right)}
//' 
//' where \eqn{P(X|Y=1)} is the probability of the feature being in a particular bin given a positive target, and \eqn{P(X|Y=0)} is the probability given a negative target.
//' 
//' The Information Value (IV) for each bin is calculated as:
//' 
//' \deqn{IV = (P(X|Y=1) - P(X|Y=0)) * WoE}
//' 
//' The total IV, which is used as the fitness function in the genetic algorithm, is the sum of IVs for all bins:
//' 
//' \deqn{Total IV = \sum_{i=1}^{n} IV_i}
//' 
//' The genetic algorithm approach allows for a global optimization of the binning strategy, potentially finding better solutions than greedy or local search methods.
//' 
//' @examples
//' \dontrun{
//' set.seed(123)
//' target <- sample(0:1, 1000, replace = TRUE)
//' feature <- rnorm(1000)
//' result <- optimal_binning_numerical_gab(target, feature)
//' print(result$woebin)
//' }
//' 
//' @references
//' \itemize{
//'   \item Kotsiantis, S., & Kanellopoulos, D. (2006). Discretization techniques: A recent survey. GESTS International Transactions on Computer Science and Engineering, 32(1), 47-58.
//'   \item Dougherty, J., Kohavi, R., & Sahami, M. (1995). Supervised and unsupervised discretization of continuous features. In Machine Learning Proceedings 1995 (pp. 194-202). Morgan Kaufmann.
//' }
//' 
//' @author Lopes, J. E.
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_gab(const Rcpp::NumericVector& target,
                                         const Rcpp::NumericVector& feature,
                                         int min_bins = 3,
                                         int max_bins = 5,
                                         double bin_cutoff = 0.05,
                                         int max_n_prebins = 20) {
  std::vector<double> target_vec = Rcpp::as<std::vector<double>>(target);
  std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
  
  OptimalBinningNumericalGAB obng(target_vec, feature_vec, min_bins, max_bins, bin_cutoff, max_n_prebins);
  obng.fit();
  std::vector<double> woefeature = obng.get_woefeature();
  Rcpp::DataFrame woebin = obng.get_woebin();
  return Rcpp::List::create(
    Rcpp::Named("woefeature") = woefeature,
    Rcpp::Named("woebin") = woebin
  );
}
