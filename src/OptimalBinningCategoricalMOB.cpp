#include <Rcpp.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cmath>

using namespace Rcpp;

// Define the OptimalBinningCategoricalMOB class
class OptimalBinningCategoricalMOB {
private:
  // Data members
  std::vector<std::string> feature;
  std::vector<int> target;

  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;

  // Computed members
  std::unordered_map<std::string, double> category_counts;
  std::unordered_map<std::string, double> category_good;
  std::unordered_map<std::string, double> category_bad;
  double total_good;
  double total_bad;

  struct Bin {
    std::vector<std::string> categories;
    double good_count;
    double bad_count;
    double woe;
    double iv;
  };

  std::vector<Bin> bins;

  // Methods
  void calculateCategoryStats();
  void mergeRareCategories();
  void calculateInitialBins();
  void enforceMonotonicity();
  void computeWoEandIV();
  void limitBins();

public:
  OptimalBinningCategoricalMOB(const std::vector<std::string>& feature_,
                               const std::vector<int>& target_,
                               int min_bins_,
                               int max_bins_,
                               double bin_cutoff_,
                               int max_n_prebins_);

  List fit();
};

// Constructor
OptimalBinningCategoricalMOB::OptimalBinningCategoricalMOB(
  const std::vector<std::string>& feature_,
  const std::vector<int>& target_,
  int min_bins_,
  int max_bins_,
  double bin_cutoff_,
  int max_n_prebins_) :
  feature(feature_),
  target(target_),
  min_bins(min_bins_),
  max_bins(max_bins_),
  bin_cutoff(bin_cutoff_),
  max_n_prebins(max_n_prebins_),
  total_good(0),
  total_bad(0) {}

// Calculate counts and distributions for each category
void OptimalBinningCategoricalMOB::calculateCategoryStats() {
  category_counts.clear();
  category_good.clear();
  category_bad.clear();
  total_good = 0;
  total_bad = 0;

  int n = feature.size();

  for (int i = 0; i < n; ++i) {
    std::string cat = feature[i];
    int tgt = target[i];

    category_counts[cat]++;
    if (tgt == 1) {
      category_good[cat]++;
      total_good++;
    } else {
      category_bad[cat]++;
      total_bad++;
    }
  }
}

// Merge categories with frequencies below the bin_cutoff threshold
void OptimalBinningCategoricalMOB::mergeRareCategories() {
  int n = feature.size();
  double min_count = bin_cutoff * n;

  // Identify rare categories
  std::vector<std::string> rare_categories;
  for (auto& kv : category_counts) {
    if (kv.second < min_count) {
      rare_categories.push_back(kv.first);
    }
  }

  if (!rare_categories.empty()) {
    // Merge rare categories into 'Other'
    std::string other_category = "Other";
    double other_count = 0;
    double other_good = 0;
    double other_bad = 0;

    for (const auto& cat : rare_categories) {
      other_count += category_counts[cat];
      other_good += category_good[cat];
      other_bad += category_bad[cat];

      // Remove the category
      category_counts.erase(cat);
      category_good.erase(cat);
      category_bad.erase(cat);
    }

    // Add 'Other' category
    category_counts[other_category] = other_count;
    category_good[other_category] = other_good;
    category_bad[other_category] = other_bad;

    // Replace occurrences in the feature vector
    for (size_t i = 0; i < feature.size(); ++i) {
      if (std::find(rare_categories.begin(), rare_categories.end(), feature[i]) != rare_categories.end()) {
        feature[i] = other_category;
      }
    }
  }
}

// Create initial bins by sorting categories based on WoE
void OptimalBinningCategoricalMOB::calculateInitialBins() {
  // Calculate WoE for each category
  struct CategoryWoE {
    std::string category;
    double woe;
    double good;
    double bad;
  };

  std::vector<CategoryWoE> cat_woe_vec;

  for (auto& kv : category_counts) {
    std::string cat = kv.first;
    double good = category_good[cat];
    double bad = category_bad[cat];

    double rate_good = good / total_good;
    double rate_bad = bad / total_bad;

    double woe = 0.0;

    if (rate_good > 0 && rate_bad > 0) {
      woe = std::log(rate_good / rate_bad);
    } else if (rate_good == 0 && rate_bad > 0) {
      woe = -999.0; // Convention for zero events
    } else if (rate_good > 0 && rate_bad == 0) {
      woe = 999.0; // Convention for zero non-events
    }

    cat_woe_vec.push_back({cat, woe, good, bad});
  }

  // Sort categories based on WoE
  std::sort(cat_woe_vec.begin(), cat_woe_vec.end(),
            [](const CategoryWoE& a, const CategoryWoE& b) {
              return a.woe < b.woe;
            });

  // Create pre-bins
  bins.clear();

  for (const auto& c : cat_woe_vec) {
    Bin bin;
    bin.categories.push_back(c.category);
    bin.good_count = c.good;
    bin.bad_count = c.bad;
    bin.woe = c.woe;
    bin.iv = 0.0; // To be computed later
    bins.push_back(bin);
  }

  // Limit the number of pre-bins
  if (bins.size() > max_n_prebins) {
    // Merge bins to reduce number of pre-bins
    int num_bins_to_merge = bins.size() - max_n_prebins;
    for (int i = 0; i < num_bins_to_merge; ++i) {
      // Merge the two bins with the smallest difference in WoE
      double min_diff = std::numeric_limits<double>::max();
      int min_index = -1;
      for (size_t j = 0; j < bins.size() - 1; ++j) {
        double diff = std::abs(bins[j + 1].woe - bins[j].woe);
        if (diff < min_diff) {
          min_diff = diff;
          min_index = j;
        }
      }
      // Merge bins[min_index] and bins[min_index + 1]
      bins[min_index].categories.insert(bins[min_index].categories.end(),
                                        bins[min_index + 1].categories.begin(),
                                        bins[min_index + 1].categories.end());
      bins[min_index].good_count += bins[min_index + 1].good_count;
      bins[min_index].bad_count += bins[min_index + 1].bad_count;
      bins.erase(bins.begin() + min_index + 1);
    }
  }
}

// Enforce monotonicity of WoE values across bins
void OptimalBinningCategoricalMOB::enforceMonotonicity() {
  bool is_increasing = false;
  bool is_decreasing = false;
  for (size_t i = 0; i < bins.size() - 1; ++i) {
    if (bins[i + 1].woe > bins[i].woe) {
      is_increasing = true;
    } else if (bins[i + 1].woe < bins[i].woe) {
      is_decreasing = true;
    }
  }

  // Enforce decreasing monotonicity
  if (is_increasing && is_decreasing) {
    bool monotonic = false;

    while (!monotonic) {
      monotonic = true;
      for (size_t i = 0; i < bins.size() - 1; ++i) {
        double woe_current = bins[i].woe;
        double woe_next = bins[i + 1].woe;

        if (woe_next > woe_current) {
          // Violation of decreasing monotonicity
          // Merge bins[i] and bins[i + 1]
          bins[i].categories.insert(bins[i].categories.end(),
                                    bins[i + 1].categories.begin(),
                                    bins[i + 1].categories.end());
          bins[i].good_count += bins[i + 1].good_count;
          bins[i].bad_count += bins[i + 1].bad_count;
          bins.erase(bins.begin() + i + 1);
          // Recompute WoE for the merged bin
          double rate_good = bins[i].good_count / total_good;
          double rate_bad = bins[i].bad_count / total_bad;

          if (rate_good > 0 && rate_bad > 0) {
            bins[i].woe = std::log(rate_good / rate_bad);
          } else if (rate_good == 0 && rate_bad > 0) {
            bins[i].woe = -999.0;
          } else if (rate_good > 0 && rate_bad == 0) {
            bins[i].woe = 999.0;
          }
          monotonic = false;
          break;
        }
      }
    }
  }
}

// Ensure the number of bins is within the specified limits
void OptimalBinningCategoricalMOB::limitBins() {
  while (bins.size() > max_bins) {
    // Merge the two bins with the smallest difference in WoE
    double min_diff = std::numeric_limits<double>::max();
    int min_index = -1;
    for (size_t i = 0; i < bins.size() - 1; ++i) {
      double diff = std::abs(bins[i + 1].woe - bins[i].woe);
      if (diff < min_diff) {
        min_diff = diff;
        min_index = i;
      }
    }
    // Merge bins[min_index] and bins[min_index + 1]
    bins[min_index].categories.insert(bins[min_index].categories.end(),
                                      bins[min_index + 1].categories.begin(),
                                      bins[min_index + 1].categories.end());
    bins[min_index].good_count += bins[min_index + 1].good_count;
    bins[min_index].bad_count += bins[min_index + 1].bad_count;
    bins.erase(bins.begin() + min_index + 1);

    // Recompute WoE for the merged bin
    double rate_good = bins[min_index].good_count / total_good;
    double rate_bad = bins[min_index].bad_count / total_bad;

    if (rate_good > 0 && rate_bad > 0) {
      bins[min_index].woe = std::log(rate_good / rate_bad);
    } else if (rate_good == 0 && rate_bad > 0) {
      bins[min_index].woe = -999.0;
    } else if (rate_good > 0 && rate_bad == 0) {
      bins[min_index].woe = 999.0;
    }
  }

  // Ensure at least min_bins
  if (bins.size() < min_bins) {
    // Cannot split bins further; accept the current number of bins
  }
}

// Compute WoE and IV for each bin
void OptimalBinningCategoricalMOB::computeWoEandIV() {
  for (auto& bin : bins) {
    double rate_good = bin.good_count / total_good;
    double rate_bad = bin.bad_count / total_bad;

    // WoE is already computed
    double woe = bin.woe;

    // Compute IV for the bin
    double iv_bin = (rate_good - rate_bad) * woe;
    bin.iv = iv_bin;
  }
}

// Fit the model and return the results
List OptimalBinningCategoricalMOB::fit() {
  calculateCategoryStats();
  mergeRareCategories();
  calculateInitialBins();
  enforceMonotonicity();
  limitBins();
  computeWoEandIV();

  // Prepare output
  // woefeature: WoE values applied to each observation
  std::vector<double> woefeature(feature.size());
  std::unordered_map<std::string, double> category_to_woe;
  for (const auto& bin : bins) {
    for (const auto& cat : bin.categories) {
      category_to_woe[cat] = bin.woe;
    }
  }

  for (size_t i = 0; i < feature.size(); ++i) {
    if (category_to_woe.find(feature[i]) != category_to_woe.end()) {
      woefeature[i] = category_to_woe[feature[i]];
    } else {
      woefeature[i] = NA_REAL; // Assign NA if category not found
    }
  }

  // woebin: DataFrame with bin metrics
  std::vector<std::string> bin_names;
  std::vector<double> woe_values;
  std::vector<double> iv_values;
  std::vector<int> count_values;
  std::vector<int> count_pos_values;
  std::vector<int> count_neg_values;

  for (const auto& bin : bins) {
    // Create bin name
    std::string bin_name = "";
    for (size_t i = 0; i < bin.categories.size(); ++i) {
      bin_name += bin.categories[i];
      if (i != bin.categories.size() - 1) {
        bin_name += "+";
      }
    }
    bin_names.push_back(bin_name);
    woe_values.push_back(bin.woe);
    iv_values.push_back(bin.iv);
    count_values.push_back(static_cast<int>(bin.good_count + bin.bad_count));
    count_pos_values.push_back(static_cast<int>(bin.good_count));
    count_neg_values.push_back(static_cast<int>(bin.bad_count));
  }

  DataFrame woebin = DataFrame::create(
    Named("bin") = bin_names,
    Named("woe") = woe_values,
    Named("iv") = iv_values,
    Named("count") = count_values,
    Named("count_pos") = count_pos_values,
    Named("count_neg") = count_neg_values
  );

  return List::create(
    Named("woefeature") = woefeature,
    Named("woebin") = woebin
  );
}

// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_mob(Rcpp::IntegerVector target,
                                           Rcpp::StringVector feature,
                                           int min_bins = 3,
                                           int max_bins = 5,
                                           double bin_cutoff = 0.05,
                                           int max_n_prebins = 20) {
  // Converter Rcpp::IntegerVector para std::vector<int>
  std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);

  // Converter Rcpp::StringVector para std::vector<std::string>
  std::vector<std::string> feature_vec;
  feature_vec.reserve(feature.size());
  for (int i = 0; i < feature.size(); ++i) {
    if (feature[i] == NA_STRING) {
      feature_vec.push_back("NA"); // Lidar com valores NA
    } else {
      feature_vec.push_back(Rcpp::as<std::string>(feature[i]));
    }
  }

  // Verificar se o target é binário
  int count_1 = std::count(target_vec.begin(), target_vec.end(), 1);
  int count_0 = std::count(target_vec.begin(), target_vec.end(), 0);
  if (count_1 + count_0 != target_vec.size()) {
    Rcpp::stop("Target vector must be binary (0 and 1).");
  }

  // Criar uma instância da classe OptimalBinningCategoricalMOB
  OptimalBinningCategoricalMOB mob(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins);

  // Ajustar o modelo e retornar os resultados
  return mob.fit();
}

// List optimal_binning_categorical_mob(std::vector<int> target,
//                                      RObject feature_obj,
//                                      int min_bins = 3,
//                                      int max_bins = 5,
//                                      double bin_cutoff = 0.05,
//                                      int max_n_prebins = 20) {
//   // Initialize a vector of strings to hold the feature
//   std::vector<std::string> feature;
//
//   // Check if feature_obj is a factor
//   if (Rf_isFactor(feature_obj)) {
//     // It's a factor; extract levels
//     IntegerVector factor_feature(feature_obj);
//     CharacterVector levels = factor_feature.attr("levels");
//     int n = factor_feature.size();
//
//     feature.reserve(n);
//     for (int i = 0; i < n; ++i) {
//       int code = factor_feature[i] - 1; // R factors are 1-based
//       if (code >= 0 && code < levels.size()) {
//         feature.push_back(as<std::string>(levels[code]));
//       } else {
//         feature.push_back("NA"); // Handle NA or invalid codes
//       }
//     }
//   } else if (TYPEOF(feature_obj) == STRSXP) {
//     // It's a character vector
//     CharacterVector char_feature(feature_obj);
//     feature.reserve(char_feature.size());
//     for (size_t i = 0; i < char_feature.size(); ++i) {
//       if (String(char_feature[i]) == NA_STRING) {
//         feature.emplace_back("NA"); // Handle NA values
//       } else {
//         feature.emplace_back(as<std::string>(char_feature[i]));
//       }
//     }
//   } else {
//     Rcpp::stop("Feature must be a character vector or a factor.");
//   }
//
//   // Check if target is binary
//   int count_1 = std::count(target.begin(), target.end(), 1);
//   int count_0 = std::count(target.begin(), target.end(), 0);
//   if (count_1 + count_0 != target.size()) {
//     Rcpp::stop("Target vector must be binary (0 and 1).");
//   }
//
//   // Create an instance of the OptimalBinningCategoricalMOB class
//   OptimalBinningCategoricalMOB mob(feature, target, min_bins, max_bins, bin_cutoff, max_n_prebins);
//
//   // Fit the model and return the results
//   return mob.fit();
// }
//
