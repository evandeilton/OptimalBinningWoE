#include <Rcpp.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <cmath>
#include <omp.h>

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

    // Initialize current solution
    current_solution.resize(unique_categories.size());
    for (size_t i = 0; i < current_solution.size(); ++i) {
      current_solution[i] = i % max_bins;
    }
    std::shuffle(current_solution.begin(), current_solution.end(), gen);

    best_solution = current_solution;
    current_iv = calculate_iv(current_solution);
    best_iv = current_iv;
  }

  double calculate_iv(const std::vector<int>& solution) {
    std::vector<int> bin_counts(max_bins, 0);
    std::vector<int> bin_positives(max_bins, 0);

    for (size_t i = 0; i < unique_categories.size(); ++i) {
      const std::string& category = unique_categories[i];
      int bin = solution[i];
      bin_counts[bin] += category_counts[category];
      bin_positives[bin] += positive_counts[category];
    }

    double iv = 0.0;
    for (int i = 0; i < max_bins; ++i) {
      if (bin_counts[i] > 0) {
        double bin_rate = static_cast<double>(bin_positives[i]) / bin_counts[i];
        double overall_rate = static_cast<double>(total_positive) / total_count;
        if (bin_rate > 0 && bin_rate < 1) {
          double woe = std::log(bin_rate / (1 - bin_rate) * (1 - overall_rate) / overall_rate);
          iv += (bin_rate - (1 - bin_rate)) * woe;
        }
      }
    }

    return iv;
  }

  bool is_monotonic(const std::vector<int>& solution) {
    std::vector<double> bin_rates(max_bins, 0.0);
    std::vector<int> bin_counts(max_bins, 0);

    for (size_t i = 0; i < unique_categories.size(); ++i) {
      const std::string& category = unique_categories[i];
      int bin = solution[i];
      bin_counts[bin] += category_counts[category];
      bin_rates[bin] += positive_counts[category];
    }

    for (int i = 0; i < max_bins; ++i) {
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
    int new_bin = std::uniform_int_distribution<>(0, max_bins - 1)(gen);
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
    if (min_bins < 2 || max_bins < min_bins) {
      Rcpp::stop("Invalid min_bins or max_bins values");
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
  }

  Rcpp::List get_results() {
    std::vector<std::string> bins;
    std::vector<double> woe;
    std::vector<double> iv;
    std::vector<int> count;
    std::vector<int> count_pos;
    std::vector<int> count_neg;

    std::vector<std::vector<std::string>> bin_categories(max_bins);
    std::vector<int> bin_counts(max_bins, 0);
    std::vector<int> bin_positives(max_bins, 0);

    for (size_t i = 0; i < unique_categories.size(); ++i) {
      const std::string& category = unique_categories[i];
      int bin = best_solution[i];
      bin_categories[bin].push_back(category);
      bin_counts[bin] += category_counts[category];
      bin_positives[bin] += positive_counts[category];
    }

    for (int i = 0; i < max_bins; ++i) {
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

    // Sort bins by WoE to ensure monotonicity
    std::vector<size_t> sorted_indices(woe.size());
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [&woe](size_t i1, size_t i2) { return woe[i1] < woe[i2]; });

    std::vector<std::string> sorted_bins;
    std::vector<double> sorted_woe;
    std::vector<double> sorted_iv;
    std::vector<int> sorted_count;
    std::vector<int> sorted_count_pos;
    std::vector<int> sorted_count_neg;

    for (size_t i : sorted_indices) {
      sorted_bins.push_back(bins[i]);
      sorted_woe.push_back(woe[i]);
      sorted_iv.push_back(iv[i]);
      sorted_count.push_back(count[i]);
      sorted_count_pos.push_back(count_pos[i]);
      sorted_count_neg.push_back(count_neg[i]);
    }

    Rcpp::DataFrame woebin = Rcpp::DataFrame::create(
      Rcpp::Named("bin") = sorted_bins,
      Rcpp::Named("woe") = sorted_woe,
      Rcpp::Named("iv") = sorted_iv,
      Rcpp::Named("count") = sorted_count,
      Rcpp::Named("count_pos") = sorted_count_pos,
      Rcpp::Named("count_neg") = sorted_count_neg
    );

    std::vector<double> woefeature(feature.size());
    for (size_t i = 0; i < feature.size(); ++i) {
      int bin = best_solution[std::find(unique_categories.begin(), unique_categories.end(), feature[i]) - unique_categories.begin()];
      woefeature[i] = woe[bin];
    }

    return Rcpp::List::create(
      Rcpp::Named("woefeature") = woefeature,
      Rcpp::Named("woebin") = woebin
    );
  }
};

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
