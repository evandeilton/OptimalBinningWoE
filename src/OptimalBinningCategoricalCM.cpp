// [[Rcpp::plugins(openmp)]]
#include <Rcpp.h>
#include <unordered_map>
#include <vector>
#include <string>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

// Define the OptimalBinningCategoricalCM class
class OptimalBinningCategoricalCM {
private:
  // Input data
  std::vector<std::string> feature;
  std::vector<int> target;

  // Parameters
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;

  // Binning results
  std::vector<double> woefeature;
  DataFrame woebin;

  // Internal structures
  struct Bin {
    std::vector<std::string> categories;
    int count_pos;
    int count_neg;
    double woe;
    double iv;
    int total_count;
  };

  std::vector<Bin> bins;

  // Total counts
  int total_pos;
  int total_neg;

public:
  // Constructor
  OptimalBinningCategoricalCM(const std::vector<std::string>& feature,
                              const std::vector<int>& target,
                              int min_bins = 3,
                              int max_bins = 5,
                              double bin_cutoff = 0.05,
                              int max_n_prebins = 20)
    : feature(feature), target(target), min_bins(min_bins),
      max_bins(max_bins), bin_cutoff(bin_cutoff),
      max_n_prebins(max_n_prebins) {
    // Initialization
    total_pos = 0;
    total_neg = 0;
  }

  // Main function to perform binning
  void fit() {
    // Input validation
    validate_inputs();

    // Initialize bins
    initialize_bins();

    // Apply bin_cutoff to handle rare categories
    handle_rare_categories();

    // Limit the number of prebins
    limit_prebins();

    // Ensure we have at least min_bins bins
    if (bins.size() < static_cast<size_t>(min_bins)) {
      // Split bins if possible (not typical in ChiMerge, but necessary here)
      split_bins_to_min_bins();
    }

    // Merge bins based on Chi-square
    merge_bins();

    // Enforce monotonicity
    enforce_monotonicity();

    // Calculate WoE and IV
    calculate_woe_iv();

    // Assign WoE to feature values
    assign_woe_feature();

    // Prepare the output DataFrame
    prepare_output();
  }

  // Getters for the results
  std::vector<double> get_woefeature() {
    return woefeature;
  }

  DataFrame get_woebin() {
    return woebin;
  }

private:
  // Validate inputs
  void validate_inputs() {
    if (feature.size() != target.size()) {
      stop("Feature and target vectors must be of the same length.");
    }

    // Check target values are binary (0 or 1)
    for (size_t i = 0; i < target.size(); ++i) {
      if (target[i] != 0 && target[i] != 1) {
        stop("Target vector must be binary (0 or 1).");
      }
    }

    // Check min_bins and max_bins
    if (min_bins < 2) {
      stop("min_bins must be at least 2.");
    }
    if (max_bins < min_bins) {
      stop("max_bins must be greater than or equal to min_bins.");
    }
  }

  // Initialize bins with each category as its own bin
  void initialize_bins() {
    // Count frequencies for each category
    std::unordered_map<std::string, int> count_pos_map;
    std::unordered_map<std::string, int> count_neg_map;
    std::unordered_map<std::string, int> total_count_map;

    for (size_t i = 0; i < feature.size(); ++i) {
      const std::string& cat = feature[i];
      int tar = target[i];

      total_count_map[cat]++;
      if (tar == 1) {
        count_pos_map[cat]++;
        total_pos++;
      } else {
        count_neg_map[cat]++;
        total_neg++;
      }
    }

    // Create bins
    for (const auto& item : total_count_map) {
      Bin bin;
      bin.categories.push_back(item.first);
      bin.count_pos = count_pos_map[item.first];
      bin.count_neg = count_neg_map[item.first];
      bin.total_count = item.second;
      bins.push_back(bin);
    }
  }

  // Handle rare categories by merging them into 'Others'
  void handle_rare_categories() {
    int total_count = total_pos + total_neg;
    std::vector<Bin> updated_bins;
    Bin others_bin;
    others_bin.count_pos = 0;
    others_bin.count_neg = 0;
    others_bin.total_count = 0;

    for (auto& bin : bins) {
      double freq = static_cast<double>(bin.total_count) / total_count;
      if (freq < bin_cutoff) {
        // Merge into others
        others_bin.categories.insert(others_bin.categories.end(),
                                     bin.categories.begin(),
                                     bin.categories.end());
        others_bin.count_pos += bin.count_pos;
        others_bin.count_neg += bin.count_neg;
        others_bin.total_count += bin.total_count;
      } else {
        updated_bins.push_back(bin);
      }
    }

    if (others_bin.total_count > 0) {
      others_bin.categories.push_back("Others");
      updated_bins.push_back(others_bin);
    }

    bins = updated_bins;
  }

  // Limit the number of prebins to max_n_prebins
  void limit_prebins() {
    while (bins.size() > static_cast<size_t>(max_n_prebins)) {
      // Merge bins with the smallest total count
      auto min_it = std::min_element(bins.begin(), bins.end(),
                                     [](const Bin& a, const Bin& b) {
                                       return a.total_count < b.total_count;
                                     });

      // Merge with the next smallest bin
      auto next_min_it = std::min_element(bins.begin(), bins.end(),
                                          [min_it](const Bin& a, const Bin& b) {
                                            if (&a == &(*min_it)) return false;
                                            if (&b == &(*min_it)) return true;
                                            return a.total_count < b.total_count;
                                          });

      // Merge min_it and next_min_it
      merge_two_bins(min_it, next_min_it);
    }
  }

  // Split bins to reach min_bins if necessary
  void split_bins_to_min_bins() {
    // This is a non-standard step, but necessary to respect min_bins
    // We will split the largest bins until we reach min_bins
    while (bins.size() < static_cast<size_t>(min_bins)) {
      // Find the bin with the largest total_count
      auto max_it = std::max_element(bins.begin(), bins.end(),
                                     [](const Bin& a, const Bin& b) {
                                       return a.total_count < b.total_count;
                                     });

      if (max_it->categories.size() <= 1) {
        // Cannot split bins with only one category
        break;
      }

      // Split the bin into two bins
      Bin bin1, bin2;
      size_t split_point = max_it->categories.size() / 2;

      // Prepare bin1
      bin1.categories.insert(bin1.categories.end(),
                             max_it->categories.begin(),
                             max_it->categories.begin() + split_point);
      bin1.count_pos = 0;
      bin1.count_neg = 0;
      bin1.total_count = 0;

      // Prepare bin2
      bin2.categories.insert(bin2.categories.end(),
                             max_it->categories.begin() + split_point,
                             max_it->categories.end());
      bin2.count_pos = 0;
      bin2.count_neg = 0;
      bin2.total_count = 0;

      // Recalculate counts for bin1 and bin2
      for (const auto& cat : bin1.categories) {
        bin1.count_pos += get_category_count_pos(cat);
        bin1.count_neg += get_category_count_neg(cat);
      }
      bin1.total_count = bin1.count_pos + bin1.count_neg;

      for (const auto& cat : bin2.categories) {
        bin2.count_pos += get_category_count_pos(cat);
        bin2.count_neg += get_category_count_neg(cat);
      }
      bin2.total_count = bin2.count_pos + bin2.count_neg;

      // Remove the original bin and add the two new bins
      *max_it = bin1;
      bins.push_back(bin2);
    }
  }

  // Get positive count for a category
  int get_category_count_pos(const std::string& category) {
    int count = 0;
    for (size_t i = 0; i < feature.size(); ++i) {
      if (feature[i] == category && target[i] == 1) {
        count++;
      }
    }
    return count;
  }

  // Get negative count for a category
  int get_category_count_neg(const std::string& category) {
    int count = 0;
    for (size_t i = 0; i < feature.size(); ++i) {
      if (feature[i] == category && target[i] == 0) {
        count++;
      }
    }
    return count;
  }

  // Merge bins based on Chi-square statistics
  void merge_bins() {
    while (static_cast<int>(bins.size()) > max_bins) {
      // Compute Chi-square for each adjacent pair
      std::vector<double> chi_squares(bins.size() - 1);
      size_t num_pairs = bins.size() - 1;

#pragma omp parallel for
      for (size_t i = 0; i < num_pairs; ++i) {
        chi_squares[i] = compute_chi_square_between_bins(bins[i], bins[i + 1]);
      }

      // Find the pair with minimum Chi-square
      auto min_it = std::min_element(chi_squares.begin(), chi_squares.end());
      size_t min_index = std::distance(chi_squares.begin(), min_it);

      // Merge the bins at min_index and min_index + 1
      merge_adjacent_bins(min_index);
    }
  }

  // Enforce monotonicity of WoE
  void enforce_monotonicity() {
    // Calculate initial WoE
    calculate_woe_iv_bins();

    // Check monotonicity
    bool monotonic = false;
    while (!monotonic) {
      monotonic = true;
      for (size_t i = 0; i < bins.size() - 1; ++i) {
        if (bins[i].woe > bins[i + 1].woe) {
          // Only merge if bins.size() > min_bins
          if (bins.size() <= static_cast<size_t>(min_bins)) {
            break;
          }
          // Merge bins[i] and bins[i + 1]
          merge_adjacent_bins(i);
          calculate_woe_iv_bins();
          monotonic = false;
          break;
        }
      }
    }
  }

  // Calculate WoE and IV for bins
  void calculate_woe_iv_bins() {
    for (auto& bin : bins) {
      double dist_pos = static_cast<double>(bin.count_pos) / total_pos;
      double dist_neg = static_cast<double>(bin.count_neg) / total_neg;
      if (dist_pos == 0) dist_pos = 0.0001;
      if (dist_neg == 0) dist_neg = 0.0001;
      bin.woe = std::log(dist_pos / dist_neg);
      bin.iv = (dist_pos - dist_neg) * bin.woe;
    }
  }

  // Calculate WoE and IV for output
  void calculate_woe_iv() {
    calculate_woe_iv_bins();
  }

  // Assign WoE values to feature
  void assign_woe_feature() {
    std::unordered_map<std::string, double> category_to_woe;
    for (const auto& bin : bins) {
      for (const auto& cat : bin.categories) {
        category_to_woe[cat] = bin.woe;
      }
    }

    woefeature.resize(feature.size());

#pragma omp parallel for
    for (size_t i = 0; i < feature.size(); ++i) {
      const std::string& cat = feature[i];
      auto it = category_to_woe.find(cat);
      if (it != category_to_woe.end()) {
        woefeature[i] = it->second; // Corrected from it->woe to it->second
      } else {
        // Assign WoE of 'Others' if category not found
        auto others_it = category_to_woe.find("Others");
        if (others_it != category_to_woe.end()) {
          woefeature[i] = others_it->second;
        } else {
          // If 'Others' bin doesn't exist, assign 0
          woefeature[i] = 0.0;
        }
      }
    }
  }

  // Prepare the output DataFrame
  void prepare_output() {
    std::vector<std::string> bin_names;
    std::vector<double> woe_values;
    std::vector<double> iv_values;
    std::vector<int> counts;
    std::vector<int> count_pos;
    std::vector<int> count_neg;

    for (const auto& bin : bins) {
      std::string bin_name = join_categories(bin.categories);
      bin_names.push_back(bin_name);
      woe_values.push_back(bin.woe);
      iv_values.push_back(bin.iv);
      counts.push_back(bin.total_count);
      count_pos.push_back(bin.count_pos);
      count_neg.push_back(bin.count_neg);
    }

    woebin = DataFrame::create(
      Named("bin") = bin_names,
      Named("woe") = woe_values,
      Named("iv") = iv_values,
      Named("count") = counts,
      Named("count_pos") = count_pos,
      Named("count_neg") = count_neg
    );
  }

  // Helper functions
  double compute_chi_square_between_bins(const Bin& bin1, const Bin& bin2) {
    double chi_square = 0.0;

    int observed[2][2] = {
      { bin1.count_pos, bin1.count_neg },
      { bin2.count_pos, bin2.count_neg }
    };

    int total_bin[2] = { bin1.total_count, bin2.total_count };
    int total_class[2] = { bin1.count_pos + bin2.count_pos,
                           bin1.count_neg + bin2.count_neg };

    int total = total_bin[0] + total_bin[1];

    double expected[2][2];

    for (int i = 0; i < 2; ++i) {
      expected[i][0] = static_cast<double>(total_bin[i]) * total_class[0] / total;
      expected[i][1] = static_cast<double>(total_bin[i]) * total_class[1] / total;

      if (expected[i][0] == 0) expected[i][0] = 0.0001;
      if (expected[i][1] == 0) expected[i][1] = 0.0001;

      chi_square += std::pow(observed[i][0] - expected[i][0], 2) / expected[i][0];
      chi_square += std::pow(observed[i][1] - expected[i][1], 2) / expected[i][1];
    }

    return chi_square;
  }

  void merge_adjacent_bins(size_t index) {
    if (index >= bins.size() - 1) return;

    // Merge bins[index] and bins[index + 1]
    Bin& bin1 = bins[index];
    Bin& bin2 = bins[index + 1];

    bin1.categories.insert(bin1.categories.end(),
                           bin2.categories.begin(),
                           bin2.categories.end());
    bin1.count_pos += bin2.count_pos;
    bin1.count_neg += bin2.count_neg;
    bin1.total_count += bin2.total_count;

    bins.erase(bins.begin() + index + 1);
  }

  void merge_two_bins(std::vector<Bin>::iterator bin_it1,
                      std::vector<Bin>::iterator bin_it2) {
    if (bin_it1 == bins.end() || bin_it2 == bins.end()) return;

    bin_it1->categories.insert(bin_it1->categories.end(),
                               bin_it2->categories.begin(),
                               bin_it2->categories.end());
    bin_it1->count_pos += bin_it2->count_pos;
    bin_it1->count_neg += bin_it2->count_neg;
    bin_it1->total_count += bin_it2->total_count;

    bins.erase(bin_it2);
  }

  std::string join_categories(const std::vector<std::string>& categories) {
    std::string result;
    for (size_t i = 0; i < categories.size(); ++i) {
      result += categories[i];
      if (i != categories.size() - 1) {
        result += "+";
      }
    }
    return result;
  }
};

//' @title Categorical Optimal Binning with Chi-Merge
//'
//' @description 
//' Implements optimal binning for categorical variables using the Chi-Merge algorithm,
//' calculating Weight of Evidence (WoE) and Information Value (IV) for resulting bins.
//'
//' @param target Integer vector of binary target values (0 or 1).
//' @param feature Character vector of categorical feature values.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum frequency for a separate bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins before merging (default: 20).
//'
//' @return A list with two elements:
//' \itemize{
//'   \item woefeature: Numeric vector of WoE values for each input feature value.
//'   \item woebin: Data frame with binning results (bin names, WoE, IV, counts).
//' }
//'
//' @details
//' The Chi-Merge algorithm uses chi-square statistics to merge adjacent bins:
//'
//' \deqn{\chi^2 = \sum_{i=1}^{2}\sum_{j=1}^{2} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}}
//'
//' where \eqn{O_{ij}} is the observed frequency and \eqn{E_{ij}} is the expected frequency
//' for bin i and class j.
//'
//' Weight of Evidence (WoE) for each bin:
//'
//' \deqn{WoE = \ln(\frac{P(X|Y=1)}{P(X|Y=0)})}
//'
//' Information Value (IV) for each bin:
//'
//' \deqn{IV = (P(X|Y=1) - P(X|Y=0)) * WoE}
//'
//' The algorithm initializes bins for each category, merges rare categories based on
//' bin_cutoff, and then iteratively merges bins with the lowest chi-square statistic
//' until reaching max_bins. It enforces WoE monotonicity and handles edge cases like
//' zero frequencies using small constant values.
//'
//' @examples
//' \dontrun{
//' # Sample data
//' target <- c(1, 0, 1, 1, 0, 1, 0, 0, 1, 1)
//' feature <- c("A", "B", "A", "C", "B", "D", "C", "A", "D", "B")
//'
//' # Run optimal binning
//' result <- optimal_binning_categorical_cm(target, feature, min_bins = 2, max_bins = 4)
//'
//' # View results
//' print(result$woebin)
//' print(result$woefeature)
//' }
//'
//' @author Lopes, J. E.
//'
//' @references
//' Kerber, R. (1992). ChiMerge: Discretization of numeric attributes. In Proceedings of the tenth national conference on Artificial intelligence (pp. 123-128). AAAI Press.
//'
//' Beltrami, M., Mach, M., & Dall'Aglio, M. (2021). Monotonic Optimal Binning Algorithm for Credit Risk Modeling. Risks, 9(3), 58.
//'
//' @export
// [[Rcpp::export]]
List optimal_binning_categorical_cm(IntegerVector target,
                                    CharacterVector feature,
                                    int min_bins = 3,
                                    int max_bins = 5,
                                    double bin_cutoff = 0.05,
                                    int max_n_prebins = 20) {
  // Convert Rcpp vectors to std::vector
  std::vector<std::string> feature_vec = as<std::vector<std::string>>(feature);
  std::vector<int> target_vec = as<std::vector<int>>(target);

  // Create an instance of the class
  OptimalBinningCategoricalCM obcm(feature_vec, target_vec, min_bins, max_bins,
                                   bin_cutoff, max_n_prebins);

  // Fit the binning algorithm
  obcm.fit();

  // Get the results
  std::vector<double> woefeature = obcm.get_woefeature();
  DataFrame woebin = obcm.get_woebin();

  // Return the results as a list
  return List::create(
    Named("woefeature") = woefeature,
    Named("woebin") = woebin
  );
}
