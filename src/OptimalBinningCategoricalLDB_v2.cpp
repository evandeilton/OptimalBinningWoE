// [[Rcpp::plugins(cpp11)]]
// Remova qualquer dependencia de paralelismo. eg. OPENMP e/ou RcppParallel
#include <Rcpp.h>
#include <vector>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

using namespace Rcpp;

// Constants for numerical stability
const double EPSILON = 1e-10;
const double MAX_WOE = 20.0;

struct CategoryStats {
  std::string category;
  int count;
  int count_pos;
  int count_neg;
  double event_rate;
  double woe;
  double iv;
};

class OptimalBinningCategoricalLDB {
private:
  IntegerVector target;
  CharacterVector feature;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  std::string bin_separator;
  double convergence_threshold;
  int max_iterations;
  
  std::vector<CategoryStats> category_stats;
  bool converged;
  int iterations_run;
  
  void validate_inputs() {
    if (feature.size() != target.size()) {
      throw std::invalid_argument("Feature and target must have the same length.");
    }
    IntegerVector unique_targets = sort_unique(target);
    if (unique_targets.size() != 2 || !(unique_targets[0] == 0 && unique_targets[1] == 1)) {
      throw std::invalid_argument("Target must be binary (0 and 1).");
    }
    if (min_bins < 2) {
      throw std::invalid_argument("min_bins must be at least 2.");
    }
    if (max_bins < min_bins) {
      throw std::invalid_argument("max_bins must be greater than or equal to min_bins.");
    }
    if (max_n_prebins < max_bins) {
      throw std::invalid_argument("max_n_prebins must be greater than or equal to max_bins.");
    }
    // Adjust max_bins if it exceeds the number of unique categories
    CharacterVector unique_categories = unique(feature);
    max_bins = std::min(max_bins, (int)unique_categories.size());
  }
  
  void compute_category_stats() {
    std::unordered_map<std::string, CategoryStats> stats_map;
    int total_pos = 0;
    int total_neg = 0;
    
    int n = target.size();
    for (int i = 0; i < n; ++i) {
      std::string cat = as<std::string>(feature[i]);
      int tar = target[i];
      if (stats_map.find(cat) == stats_map.end()) {
        stats_map[cat] = {cat, 0, 0, 0, 0.0, 0.0, 0.0};
      }
      stats_map[cat].count += 1;
      if (tar == 1) {
        stats_map[cat].count_pos += 1;
        total_pos += 1;
      } else {
        stats_map[cat].count_neg += 1;
        total_neg += 1;
      }
    }
    
    for (auto& kv : stats_map) {
      CategoryStats& cs = kv.second;
      cs.event_rate = (double)cs.count_pos / cs.count;
      category_stats.push_back(cs);
    }
    
    recalculate_woe_iv();
  }
  
  void handle_rare_categories() {
    int total_count = target.size();
    double cutoff_count = bin_cutoff * total_count;
    
    std::vector<CategoryStats> non_rare_categories;
    std::vector<CategoryStats> rare_categories;
    
    for (auto& cat_stat : category_stats) {
      if (cat_stat.count < cutoff_count) {
        rare_categories.push_back(cat_stat);
      } else {
        non_rare_categories.push_back(cat_stat);
      }
    }
    
    for (auto& rare_cat : rare_categories) {
      double min_diff = std::numeric_limits<double>::max();
      int min_idx = -1;
      for (size_t i = 0; i < non_rare_categories.size(); ++i) {
        double diff = std::fabs(non_rare_categories[i].woe - rare_cat.woe);
        if (diff < min_diff) {
          min_diff = diff;
          min_idx = i;
        }
      }
      if (min_idx != -1) {
        non_rare_categories[min_idx].category += bin_separator + rare_cat.category;
        non_rare_categories[min_idx].count += rare_cat.count;
        non_rare_categories[min_idx].count_pos += rare_cat.count_pos;
        non_rare_categories[min_idx].count_neg += rare_cat.count_neg;
      }
    }
    
    category_stats = non_rare_categories;
    recalculate_woe_iv();
  }
  
  void limit_prebins() {
    if ((int)category_stats.size() > max_n_prebins) {
      std::sort(category_stats.begin(), category_stats.end(),
                [](const CategoryStats& a, const CategoryStats& b) {
                  return a.woe < b.woe;
                });
      int n_bins = max_n_prebins;
      int n_categories = category_stats.size();
      int bin_size = std::ceil((double)n_categories / n_bins);
      std::vector<CategoryStats> new_category_stats;
      
      for (int i = 0; i < n_bins; ++i) {
        int start_idx = i * bin_size;
        int end_idx = std::min(start_idx + bin_size, n_categories);
        if (start_idx >= end_idx) break;
        
        CategoryStats cs = category_stats[start_idx];
        for (int j = start_idx + 1; j < end_idx; ++j) {
          cs.category += bin_separator + category_stats[j].category;
          cs.count += category_stats[j].count;
          cs.count_pos += category_stats[j].count_pos;
          cs.count_neg += category_stats[j].count_neg;
        }
        new_category_stats.push_back(cs);
      }
      category_stats = new_category_stats;
      recalculate_woe_iv();
    }
  }
  
  void merge_bins() {
    std::sort(category_stats.begin(), category_stats.end(),
              [](const CategoryStats& a, const CategoryStats& b) {
                return a.woe < b.woe;
              });
    
    converged = false;
    iterations_run = 0;
    
    while ((int)category_stats.size() > max_bins || !is_monotonic()) {
      if ((int)category_stats.size() <= min_bins) {
        break;
      }
      
      double min_iv_loss = std::numeric_limits<double>::max();
      int merge_idx = -1;
      
      for (size_t i = 0; i < category_stats.size() - 1; ++i) {
        CategoryStats merged_bin = merge_two_bins(category_stats[i], category_stats[i + 1]);
        double iv_loss = category_stats[i].iv + category_stats[i + 1].iv - merged_bin.iv;
        
        if (iv_loss < min_iv_loss) {
          min_iv_loss = iv_loss;
          merge_idx = i;
        }
      }
      
      if (merge_idx != -1) {
        CategoryStats merged_bin = merge_two_bins(category_stats[merge_idx], category_stats[merge_idx + 1]);
        category_stats[merge_idx] = merged_bin;
        category_stats.erase(category_stats.begin() + merge_idx + 1);
        
        std::sort(category_stats.begin(), category_stats.end(),
                  [](const CategoryStats& a, const CategoryStats& b) {
                    return a.woe < b.woe;
                  });
      } else {
        break;
      }
      
      iterations_run++;
      if (iterations_run >= max_iterations) {
        break;
      }
      
      if (is_monotonic() && (int)category_stats.size() <= max_bins) {
        converged = true;
        break;
      }
    }
  }
  
  bool is_monotonic() {
    if (category_stats.empty()) return true;
    
    bool increasing = true;
    bool decreasing = true;
    for (size_t i = 1; i < category_stats.size(); ++i) {
      if (category_stats[i].woe < category_stats[i - 1].woe - convergence_threshold) {
        increasing = false;
      }
      if (category_stats[i].woe > category_stats[i - 1].woe + convergence_threshold) {
        decreasing = false;
      }
    }
    return increasing || decreasing;
  }
  
  void recalculate_woe_iv() {
    int total_pos = 0;
    int total_neg = 0;
    for (auto& cs : category_stats) {
      total_pos += cs.count_pos;
      total_neg += cs.count_neg;
    }
    for (auto& cs : category_stats) {
      double dist_pos = (double)cs.count_pos / total_pos;
      double dist_neg = (double)cs.count_neg / total_neg;
      
      // Improved numerical stability
      dist_pos = std::max(dist_pos, EPSILON);
      dist_neg = std::max(dist_neg, EPSILON);
      
      cs.woe = std::log(dist_pos / dist_neg);
      // Capping WoE for numerical stability
      cs.woe = std::max(std::min(cs.woe, MAX_WOE), -MAX_WOE);
      cs.iv = (dist_pos - dist_neg) * cs.woe;
    }
  }
  
  CategoryStats merge_two_bins(const CategoryStats& a, const CategoryStats& b) {
    CategoryStats merged;
    merged.category = a.category + bin_separator + b.category;
    merged.count = a.count + b.count;
    merged.count_pos = a.count_pos + b.count_pos;
    merged.count_neg = a.count_neg + b.count_neg;
    
    int total_pos = 0;
    int total_neg = 0;
    for (auto& cs : category_stats) {
      total_pos += cs.count_pos;
      total_neg += cs.count_neg;
    }
    
    double dist_pos = (double)merged.count_pos / total_pos;
    double dist_neg = (double)merged.count_neg / total_neg;
    
    // Improved numerical stability
    dist_pos = std::max(dist_pos, EPSILON);
    dist_neg = std::max(dist_neg, EPSILON);
    
    merged.woe = std::log(dist_pos / dist_neg);
    // Capping WoE for numerical stability
    merged.woe = std::max(std::min(merged.woe, MAX_WOE), -MAX_WOE);
    merged.iv = (dist_pos - dist_neg) * merged.woe;
    
    return merged;
  }
  
public:
  OptimalBinningCategoricalLDB(IntegerVector target, CharacterVector feature,
                               int min_bins = 3, int max_bins = 5,
                               double bin_cutoff = 0.05, int max_n_prebins = 20,
                               std::string bin_separator = "%;%",
                               double convergence_threshold = 1e-6,
                               int max_iterations = 1000)
    : target(target), feature(feature), min_bins(min_bins), max_bins(max_bins),
      bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins),
      bin_separator(bin_separator), convergence_threshold(convergence_threshold),
      max_iterations(max_iterations), converged(false), iterations_run(0) {
  }
  
  List fit() {
    try {
      validate_inputs();
      compute_category_stats();
      handle_rare_categories();
      limit_prebins();
      merge_bins();
      
      CharacterVector bins;
      NumericVector woe_values;
      NumericVector iv_values;
      IntegerVector counts;
      IntegerVector counts_pos;
      IntegerVector counts_neg;
      
      for (auto& cs : category_stats) {
        bins.push_back(cs.category);
        woe_values.push_back(cs.woe);
        iv_values.push_back(cs.iv);
        counts.push_back(cs.count);
        counts_pos.push_back(cs.count_pos);
        counts_neg.push_back(cs.count_neg);
      }
      
      return List::create(
        Named("bins") = bins,
        Named("woe") = woe_values,
        Named("iv") = iv_values,
        Named("count") = counts,
        Named("count_pos") = counts_pos,
        Named("count_neg") = counts_neg,
        Named("converged") = converged,
        Named("iterations") = iterations_run
      );
    } catch (const std::exception& e) {
      Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
    }
  }
};


//' @title Categorical Optimal Binning with Local Distance-Based Algorithm
//'
//' @description
//' This function performs optimal binning for categorical variables using a Local Distance-Based (LDB) algorithm,
//' which merges categories based on their Weight of Evidence (WoE) similarity and Information Value (IV) loss.
//'
//' @param target An integer vector of binary target values (0 or 1).
//' @param feature A character vector of categorical feature values.
//' @param min_bins Minimum number of bins to create (default: 3).
//' @param max_bins Maximum number of bins to create (default: 5).
//' @param bin_cutoff Minimum frequency for a category to be considered as a separate bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins before merging (default: 20).
//' @param bin_separator Separator used when merging category names (default: "%;%").
//' @param convergence_threshold Threshold for considering WoE values equal (default: 1e-6).
//' @param max_iterations Maximum number of iterations for the binning process (default: 1000).
//'
//' @return A list containing the following elements:
//' \itemize{
//'   \item bins: A character vector of bin names.
//'   \item woe: A numeric vector of Weight of Evidence (WoE) values for each bin.
//'   \item iv: A numeric vector of Information Value (IV) for each bin.
//'   \item count: An integer vector of total count for each bin.
//'   \item count_pos: An integer vector of positive class count for each bin.
//'   \item count_neg: An integer vector of negative class count for each bin.
//'   \item converged: A logical value indicating whether the algorithm converged.
//'   \item iterations: An integer indicating the number of iterations performed.
//' }
//'
//' @details
//' The LDB algorithm works as follows:
//' \enumerate{
//'   \item Compute initial statistics for each category.
//'   \item Handle rare categories by merging them with the most similar (in terms of WoE) non-rare category.
//'   \item Limit the number of pre-bins to max_n_prebins.
//'   \item Iteratively merge bins with the lowest IV loss until the desired number of bins is reached or monotonicity is achieved.
//'   \item Ensure monotonicity of WoE across bins.
//' }
//'
//' Weight of Evidence (WoE) for each bin is calculated as:
//'
//' \deqn{WoE = \ln\left(\frac{P(X|Y=1)}{P(X|Y=0)}\right)}
//'
//' Information Value (IV) for each bin is calculated as:
//'
//' \deqn{IV = (P(X|Y=1) - P(X|Y=0)) \times WoE}
//'
//' @examples
//' \dontrun{
//' # Sample data
//' target <- c(1, 0, 1, 1, 0, 1, 0, 0, 1, 1)
//' feature <- c("A", "B", "A", "C", "B", "D", "C", "A", "D", "B")
//'
//' # Run optimal binning
//' result <- optimal_binning_categorical_ldb(target, feature, min_bins = 2, max_bins = 4)
//'
//' # View results
//' print(result)
//' }
//'
//' @export
//'
// [[Rcpp::export]]
List optimal_binning_categorical_ldb(IntegerVector target, CharacterVector feature,
                                     int min_bins = 3, int max_bins = 5,
                                     double bin_cutoff = 0.05, int max_n_prebins = 20,
                                     std::string bin_separator = "%;%",
                                     double convergence_threshold = 1e-6,
                                     int max_iterations = 1000) {
  OptimalBinningCategoricalLDB binning(target, feature, min_bins, max_bins,
                                       bin_cutoff, max_n_prebins, bin_separator,
                                       convergence_threshold, max_iterations);
  return binning.fit();
}
