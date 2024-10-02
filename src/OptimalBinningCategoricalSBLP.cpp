// [[Rcpp::depends(Rcpp)]]
// [[Rcpp::plugins(openmp)]]
#include <Rcpp.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;
using namespace std;

class OptimalBinningCategoricalSBLP {
public:
  OptimalBinningCategoricalSBLP(const IntegerVector& target,
                                const CharacterVector& feature,
                                int min_bins = 3,
                                int max_bins = 5,
                                double bin_cutoff = 0.05,
                                int max_n_prebins = 20);
  
  void fit();
  List get_results() const;
  
private:
  const IntegerVector& target;
  const CharacterVector& feature;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  
  vector<string> unique_categories;
  vector<int> count_total;
  vector<int> count_pos;
  vector<int> count_neg;
  vector<double> category_target_rate;
  vector<int> sorted_indices;
  
  unordered_map<string, double> woe_map;
  vector<double> woefeature;
  DataFrame woebin;
  
  void compute_initial_counts();
  void handle_rare_categories();
  void compute_woe_iv();
  void perform_binning();
  double calculate_total_iv(const vector<vector<int>>& bins) const;
  void apply_binning(const vector<vector<int>>& bins);
  static string merge_category_names(const vector<string>& categories);
};

OptimalBinningCategoricalSBLP::OptimalBinningCategoricalSBLP(const IntegerVector& target,
                                                             const CharacterVector& feature,
                                                             int min_bins,
                                                             int max_bins,
                                                             double bin_cutoff,
                                                             int max_n_prebins)
  : target(target), feature(feature),
    min_bins(min_bins), max_bins(max_bins),
    bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins) {
  if (min_bins < 2) {
    stop("min_bins must be at least 2");
  }
  if (max_bins < min_bins) {
    stop("max_bins must be greater than or equal to min_bins");
  }
  if (target.size() != feature.size()) {
    stop("target and feature must have the same length");
  }
}

void OptimalBinningCategoricalSBLP::fit() {
  compute_initial_counts();
  handle_rare_categories();
  compute_woe_iv();
  perform_binning();
  
  woefeature.resize(feature.size());
#pragma omp parallel for
  for (int i = 0; i < feature.size(); ++i) {
    string cat = as<string>(feature[i]);
    auto it = woe_map.find(cat);
    woefeature[i] = (it != woe_map.end()) ? it->second : std::numeric_limits<double>::quiet_NaN();
  }
}

void OptimalBinningCategoricalSBLP::compute_initial_counts() {
  unordered_map<string, size_t> category_indices;
  
  for (int i = 0; i < feature.size(); ++i) {
    string cat = as<string>(feature[i]);
    auto [it, inserted] = category_indices.emplace(cat, unique_categories.size());
    if (inserted) {
      unique_categories.push_back(cat);
      count_total.push_back(0);
      count_pos.push_back(0);
      count_neg.push_back(0);
    }
    size_t idx = it->second;
    count_total[idx]++;
    if (target[i] == 1) {
      count_pos[idx]++;
    } else if (target[i] == 0) {
      count_neg[idx]++;
    } else {
      stop("target must be binary (0 or 1)");
    }
  }
  
  category_target_rate.resize(unique_categories.size());
  for (size_t i = 0; i < unique_categories.size(); ++i) {
    category_target_rate[i] = static_cast<double>(count_pos[i]) / count_total[i];
  }
}

void OptimalBinningCategoricalSBLP::handle_rare_categories() {
  int total_count = accumulate(count_total.begin(), count_total.end(), 0);
  vector<bool> is_rare(unique_categories.size(), false);
  
  for (size_t i = 0; i < unique_categories.size(); ++i) {
    double proportion = static_cast<double>(count_total[i]) / total_count;
    if (proportion < bin_cutoff) {
      is_rare[i] = true;
    }
  }
  
  if (none_of(is_rare.begin(), is_rare.end(), [](bool b) { return b; })) {
    return;
  }
  
  vector<size_t> rare_indices;
  for (size_t i = 0; i < unique_categories.size(); ++i) {
    if (is_rare[i]) {
      rare_indices.push_back(i);
    }
  }
  
  sort(rare_indices.begin(), rare_indices.end(),
       [this](size_t i, size_t j) { return category_target_rate[i] < category_target_rate[j]; });
  
  for (size_t i = 1; i < rare_indices.size(); ++i) {
    size_t prev_idx = rare_indices[i - 1];
    size_t curr_idx = rare_indices[i];
    count_total[prev_idx] += count_total[curr_idx];
    count_pos[prev_idx] += count_pos[curr_idx];
    count_neg[prev_idx] += count_neg[curr_idx];
    unique_categories[prev_idx] = merge_category_names({unique_categories[prev_idx], unique_categories[curr_idx]});
    unique_categories[curr_idx].clear();
  }
  
  auto it = remove_if(unique_categories.begin(), unique_categories.end(),
                      [](const string& s) { return s.empty(); });
  size_t new_size = distance(unique_categories.begin(), it);
  
  unique_categories.resize(new_size);
  count_total.resize(new_size);
  count_pos.resize(new_size);
  count_neg.resize(new_size);
  
  category_target_rate.resize(new_size);
  for (size_t i = 0; i < new_size; ++i) {
    category_target_rate[i] = static_cast<double>(count_pos[i]) / count_total[i];
  }
}

void OptimalBinningCategoricalSBLP::compute_woe_iv() {
  int total_pos = accumulate(count_pos.begin(), count_pos.end(), 0);
  int total_neg = accumulate(count_neg.begin(), count_neg.end(), 0);
  
  vector<double> woe(unique_categories.size());
  vector<double> iv(unique_categories.size());
  
  const double epsilon = 1e-10;
  for (size_t i = 0; i < unique_categories.size(); ++i) {
    double dist_pos = static_cast<double>(count_pos[i]) / total_pos;
    double dist_neg = static_cast<double>(count_neg[i]) / total_neg;
    dist_pos = max(dist_pos, epsilon);
    dist_neg = max(dist_neg, epsilon);
    woe[i] = log(dist_pos / dist_neg);
    iv[i] = (dist_pos - dist_neg) * woe[i];
  }
}

void OptimalBinningCategoricalSBLP::perform_binning() {
  sorted_indices.resize(unique_categories.size());
  iota(sorted_indices.begin(), sorted_indices.end(), 0);
  sort(sorted_indices.begin(), sorted_indices.end(),
       [this](size_t i, size_t j) { return category_target_rate[i] < category_target_rate[j]; });
  
  vector<vector<int>> bins(unique_categories.size());
  for (size_t i = 0; i < bins.size(); ++i) {
    bins[i].push_back(sorted_indices[i]);
  }
  
  while (bins.size() > max_bins) {
    double min_iv_decrease = numeric_limits<double>::max();
    size_t merge_idx = -1;
    for (size_t i = 0; i < bins.size() - 1; ++i) {
      vector<int> merged_bin = bins[i];
      merged_bin.insert(merged_bin.end(), bins[i + 1].begin(), bins[i + 1].end());
      
      double iv_before = calculate_total_iv(bins);
      
      vector<vector<int>> new_bins = bins;
      new_bins[i] = merged_bin;
      new_bins.erase(new_bins.begin() + i + 1);
      
      double iv_after = calculate_total_iv(new_bins);
      
      double iv_decrease = iv_before - iv_after;
      if (iv_decrease < min_iv_decrease) {
        min_iv_decrease = iv_decrease;
        merge_idx = i;
      }
    }
    if (merge_idx == static_cast<size_t>(-1)) {
      break;
    }
    bins[merge_idx].insert(bins[merge_idx].end(), bins[merge_idx + 1].begin(), bins[merge_idx + 1].end());
    bins.erase(bins.begin() + merge_idx + 1);
  }
  
  while (bins.size() < static_cast<size_t>(min_bins)) {
    auto max_size_it = max_element(bins.begin(), bins.end(),
                                   [](const vector<int>& a, const vector<int>& b) { return a.size() < b.size(); });
    size_t max_size_idx = distance(bins.begin(), max_size_it);
    if (bins[max_size_idx].size() == 1) {
      break;
    }
    vector<int> new_bin = {bins[max_size_idx].back()};
    bins[max_size_idx].pop_back();
    bins.push_back(new_bin);
  }
  
  apply_binning(bins);
}

double OptimalBinningCategoricalSBLP::calculate_total_iv(const vector<vector<int>>& bins) const {
  int total_pos = accumulate(count_pos.begin(), count_pos.end(), 0);
  int total_neg = accumulate(count_neg.begin(), count_neg.end(), 0);
  
  double total_iv = 0.0;
  const double epsilon = 1e-10;
  for (const auto& bin : bins) {
    int bin_pos = 0, bin_neg = 0;
    for (int idx : bin) {
      bin_pos += count_pos[idx];
      bin_neg += count_neg[idx];
    }
    double dist_pos = static_cast<double>(bin_pos) / total_pos;
    double dist_neg = static_cast<double>(bin_neg) / total_neg;
    dist_pos = max(dist_pos, epsilon);
    dist_neg = max(dist_neg, epsilon);
    double woe = log(dist_pos / dist_neg);
    total_iv += (dist_pos - dist_neg) * woe;
  }
  return total_iv;
}

void OptimalBinningCategoricalSBLP::apply_binning(const vector<vector<int>>& bins) {
  int total_pos = accumulate(count_pos.begin(), count_pos.end(), 0);
  int total_neg = accumulate(count_neg.begin(), count_neg.end(), 0);
  
  vector<string> bin_labels;
  vector<double> bin_woe;
  vector<double> bin_iv;
  vector<int> bin_count;
  vector<int> bin_count_pos;
  vector<int> bin_count_neg;
  
  const double epsilon = 1e-10;
  for (const auto& bin : bins) {
    vector<string> bin_categories;
    int bin_pos = 0, bin_neg = 0, bin_total = 0;
    for (int idx : bin) {
      bin_categories.push_back(unique_categories[idx]);
      bin_pos += count_pos[idx];
      bin_neg += count_neg[idx];
      bin_total += count_total[idx];
    }
    double dist_pos = static_cast<double>(bin_pos) / total_pos;
    double dist_neg = static_cast<double>(bin_neg) / total_neg;
    dist_pos = max(dist_pos, epsilon);
    dist_neg = max(dist_neg, epsilon);
    double woe = log(dist_pos / dist_neg);
    double iv = (dist_pos - dist_neg) * woe;
    
    string bin_label = merge_category_names(bin_categories);
    
    for (const auto& cat : bin_categories) {
      vector<string> cat_parts;
      stringstream ss(cat);
      string part;
      while (getline(ss, part, '+')) {
        cat_parts.push_back(part);
      }
      for (const auto& original_cat : cat_parts) {
        woe_map[original_cat] = woe;
      }
    }
    
    bin_labels.push_back(bin_label);
    bin_woe.push_back(woe);
    bin_iv.push_back(iv);
    bin_count.push_back(bin_total);
    bin_count_pos.push_back(bin_pos);
    bin_count_neg.push_back(bin_neg);
  }
  
  woebin = DataFrame::create(
    Named("bin") = bin_labels,
    Named("woe") = bin_woe,
    Named("iv") = bin_iv,
    Named("count") = bin_count,
    Named("count_pos") = bin_count_pos,
    Named("count_neg") = bin_count_neg
  );
}

string OptimalBinningCategoricalSBLP::merge_category_names(const vector<string>& categories) {
  vector<string> unique_cats;
  for (const auto& cat : categories) {
    stringstream ss(cat);
    string part;
    while (getline(ss, part, '+')) {
      if (find(unique_cats.begin(), unique_cats.end(), part) == unique_cats.end()) {
        unique_cats.push_back(part);
      }
    }
  }
  sort(unique_cats.begin(), unique_cats.end());
  return accumulate(unique_cats.begin(), unique_cats.end(), string(),
                    [](const string& a, const string& b) {
                      return a.empty() ? b : a + "+" + b;
                    });
}

List OptimalBinningCategoricalSBLP::get_results() const {
  return List::create(
    Named("woefeature") = woefeature,
    Named("woebin") = woebin
  );
}


//' Optimal Binning for Categorical Variables using Similarity-Based Logistic Partitioning (SBLP)
//'
//' @description
//' This function performs optimal binning for categorical variables using a Similarity-Based Logistic Partitioning (SBLP) approach,
//' which combines Weight of Evidence (WOE) and Information Value (IV) methods with a similarity-based merging strategy.
//'
//' @param target An integer vector of binary target values (0 or 1).
//' @param feature A character vector of categorical feature values.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum frequency for a category to be considered as a separate bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins before merging (default: 20).
//'
//' @return A list containing two elements:
//' \itemize{
//'   \item woefeature: A numeric vector of WOE values for each input feature value
//'   \item woebin: A data frame containing bin information, including bin labels, WOE, IV, and counts
//' }
//'
//' @details
//' The algorithm performs the following steps:
//' \enumerate{
//'   \item Compute initial counts and target rates for each category
//'   \item Handle rare categories by merging them based on similarity in target rates
//'   \item Compute initial Weight of Evidence (WOE) and Information Value (IV)
//'   \item Perform iterative binning by merging similar categories
//'   \item Apply final binning and calculate WOE and IV for each bin
//' }
//'
//' The Weight of Evidence (WOE) is calculated as:
//' \deqn{WOE = \ln(\frac{\text{Proportion of Events}}{\text{Proportion of Non-Events}})}
//'
//' The Information Value (IV) for each bin is calculated as:
//' \deqn{IV = (\text{Proportion of Events} - \text{Proportion of Non-Events}) \times WOE}
//'
//' @references
//' \itemize{
//'   \item Blanco-Justicia, A., & Domingo-Ferrer, J. (2019). Machine learning explainability through microaggregation and shallow decision trees. Knowledge-Based Systems, 174, 200-212.
//'   \item Zhu, L., Qiu, D., Ergu, D., Ying, C., & Liu, K. (2019). A study on predicting loan default based on the random forest algorithm. Procedia Computer Science, 162, 503-513.
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
//' result <- optimal_binning_categorical_sblp(target, feature)
//'
//' # View results
//' print(result$woebin)
//' hist(result$woefeature)
//' }
//' @author Lopes, J. E.
//' @export
// [[Rcpp::export]]
List optimal_binning_categorical_sblp(const IntegerVector& target,
                                     const CharacterVector& feature,
                                     int min_bins = 3,
                                     int max_bins = 5,
                                     double bin_cutoff = 0.05,
                                     int max_n_prebins = 20) {
 OptimalBinningCategoricalSBLP optbin(target, feature, min_bins, max_bins, bin_cutoff, max_n_prebins);
 optbin.fit();
 return optbin.get_results();
}




// // [[Rcpp::depends(Rcpp)]]
// // [[Rcpp::plugins(openmp)]]
// #include <Rcpp.h>
// #include <vector>
// #include <string>
// #include <unordered_map>
// #include <algorithm>
// #include <cmath>
// #include <numeric>
// 
// #ifdef _OPENMP
// #include <omp.h>
// #endif
// 
// using namespace Rcpp;
// using namespace std;
// 
// class OptimalBinningCategoricalSBLP {
// public:
//   OptimalBinningCategoricalSBLP(IntegerVector target,
//                                 CharacterVector feature,
//                                 int min_bins = 3,
//                                 int max_bins = 5,
//                                 double bin_cutoff = 0.05,
//                                 int max_n_prebins = 20);
// 
//   void fit();
//   List get_results();
// 
// private:
//   IntegerVector target;
//   CharacterVector feature;
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
// 
//   vector<string> categories;
//   unordered_map<string, double> woe_map;
//   vector<double> woefeature;
//   DataFrame woebin;
// 
//   void compute_initial_counts();
//   void handle_rare_categories();
//   void compute_woe_iv();
//   void perform_binning();
//   double calculate_total_iv(const vector<vector<int>>& bins);
//   void apply_binning(const vector<vector<int>>& bins);
//   string merge_category_names(const vector<string>& categories);
// 
//   vector<string> unique_categories;
//   vector<int> count_total;
//   vector<int> count_pos;
//   vector<int> count_neg;
//   vector<double> category_target_rate;
//   vector<int> sorted_indices;
// };
// 
// OptimalBinningCategoricalSBLP::OptimalBinningCategoricalSBLP(IntegerVector target,
//                                                              CharacterVector feature,
//                                                              int min_bins,
//                                                              int max_bins,
//                                                              double bin_cutoff,
//                                                              int max_n_prebins)
//   : target(target), feature(feature),
//     min_bins(min_bins), max_bins(max_bins),
//     bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins) {
// }
// 
// void OptimalBinningCategoricalSBLP::fit() {
//   if (min_bins < 2) {
//     stop("min_bins must be at least 2");
//   }
//   if (max_bins < min_bins) {
//     stop("max_bins must be greater than or equal to min_bins");
//   }
//   if (target.size() != feature.size()) {
//     stop("target and feature must have the same length");
//   }
// 
//   compute_initial_counts();
//   handle_rare_categories();
//   compute_woe_iv();
//   perform_binning();
// 
//   woefeature.resize(feature.size());
// #pragma omp parallel for
//   for (int i = 0; i < feature.size(); ++i) {
//     string cat = as<string>(feature[i]);
//     if (woe_map.find(cat) != woe_map.end()) {
//       woefeature[i] = woe_map[cat];
//     } else {
//       woefeature[i] = NAN;
//     }
//   }
// }
// 
// void OptimalBinningCategoricalSBLP::compute_initial_counts() {
//   unordered_map<string, int> category_indices;
// 
//   int index = 0;
//   for (int i = 0; i < feature.size(); ++i) {
//     string cat = as<string>(feature[i]);
//     if (category_indices.find(cat) == category_indices.end()) {
//       category_indices[cat] = index++;
//       unique_categories.push_back(cat);
//       count_total.push_back(0);
//       count_pos.push_back(0);
//       count_neg.push_back(0);
//     }
//     int idx = category_indices[cat];
//     count_total[idx]++;
//     if (target[i] == 1) {
//       count_pos[idx]++;
//     } else if (target[i] == 0) {
//       count_neg[idx]++;
//     } else {
//       stop("target must be binary (0 or 1)");
//     }
//   }
// 
//   category_target_rate.resize(unique_categories.size());
//   for (int i = 0; i < unique_categories.size(); ++i) {
//     category_target_rate[i] = (double)count_pos[i] / count_total[i];
//   }
// }
// 
// void OptimalBinningCategoricalSBLP::handle_rare_categories() {
//   int total_count = accumulate(count_total.begin(), count_total.end(), 0);
//   vector<bool> is_rare(unique_categories.size(), false);
// 
//   for (int i = 0; i < unique_categories.size(); ++i) {
//     double proportion = (double)count_total[i] / total_count;
//     if (proportion < bin_cutoff) {
//       is_rare[i] = true;
//     }
//   }
// 
//   if (find(is_rare.begin(), is_rare.end(), true) == is_rare.end()) {
//     return;
//   }
// 
//   vector<int> rare_indices;
//   for (int i = 0; i < unique_categories.size(); ++i) {
//     if (is_rare[i]) {
//       rare_indices.push_back(i);
//     }
//   }
// 
//   sort(rare_indices.begin(), rare_indices.end(),
//        [&](int i, int j) { return category_target_rate[i] < category_target_rate[j]; });
// 
//   for (int i = 1; i < rare_indices.size(); ++i) {
//     int prev_idx = rare_indices[i - 1];
//     int curr_idx = rare_indices[i];
//     count_total[prev_idx] += count_total[curr_idx];
//     count_pos[prev_idx] += count_pos[curr_idx];
//     count_neg[prev_idx] += count_neg[curr_idx];
//     unique_categories[prev_idx] = merge_category_names({unique_categories[prev_idx], unique_categories[curr_idx]});
//     unique_categories[curr_idx] = "";
//   }
// 
//   vector<string> new_unique_categories;
//   vector<int> new_count_total;
//   vector<int> new_count_pos;
//   vector<int> new_count_neg;
//   for (int i = 0; i < unique_categories.size(); ++i) {
//     if (!unique_categories[i].empty()) {
//       new_unique_categories.push_back(unique_categories[i]);
//       new_count_total.push_back(count_total[i]);
//       new_count_pos.push_back(count_pos[i]);
//       new_count_neg.push_back(count_neg[i]);
//     }
//   }
//   unique_categories = new_unique_categories;
//   count_total = new_count_total;
//   count_pos = new_count_pos;
//   count_neg = new_count_neg;
// 
//   category_target_rate.resize(unique_categories.size());
//   for (int i = 0; i < unique_categories.size(); ++i) {
//     category_target_rate[i] = (double)count_pos[i] / count_total[i];
//   }
// }
// 
// void OptimalBinningCategoricalSBLP::compute_woe_iv() {
//   int total_pos = accumulate(count_pos.begin(), count_pos.end(), 0);
//   int total_neg = accumulate(count_neg.begin(), count_neg.end(), 0);
// 
//   vector<double> woe(unique_categories.size());
//   vector<double> iv(unique_categories.size());
// 
//   for (int i = 0; i < unique_categories.size(); ++i) {
//     double dist_pos = (double)count_pos[i] / total_pos;
//     double dist_neg = (double)count_neg[i] / total_neg;
//     if (dist_pos == 0) dist_pos = 0.0001;
//     if (dist_neg == 0) dist_neg = 0.0001;
//     woe[i] = log(dist_pos / dist_neg);
//     iv[i] = (dist_pos - dist_neg) * woe[i];
//   }
// }
// 
// void OptimalBinningCategoricalSBLP::perform_binning() {
//   sorted_indices.resize(unique_categories.size());
//   iota(sorted_indices.begin(), sorted_indices.end(), 0);
//   sort(sorted_indices.begin(), sorted_indices.end(),
//        [&](int i, int j) { return category_target_rate[i] < category_target_rate[j]; });
// 
//   vector<vector<int>> bins(unique_categories.size());
//   for (int i = 0; i < bins.size(); ++i) {
//     bins[i].push_back(sorted_indices[i]);
//   }
// 
//   while (bins.size() > max_bins) {
//     double min_iv_decrease = numeric_limits<double>::max();
//     int merge_idx = -1;
//     for (int i = 0; i < bins.size() - 1; ++i) {
//       vector<int> merged_bin = bins[i];
//       merged_bin.insert(merged_bin.end(), bins[i + 1].begin(), bins[i + 1].end());
// 
//       double iv_before = calculate_total_iv(bins);
// 
//       vector<vector<int>> new_bins = bins;
//       new_bins[i] = merged_bin;
//       new_bins.erase(new_bins.begin() + i + 1);
// 
//       double iv_after = calculate_total_iv(new_bins);
// 
//       double iv_decrease = iv_before - iv_after;
//       if (iv_decrease < min_iv_decrease) {
//         min_iv_decrease = iv_decrease;
//         merge_idx = i;
//       }
//     }
//     if (merge_idx == -1) {
//       break;
//     }
//     bins[merge_idx].insert(bins[merge_idx].end(), bins[merge_idx + 1].begin(), bins[merge_idx + 1].end());
//     bins.erase(bins.begin() + merge_idx + 1);
//   }
// 
//   while (bins.size() < min_bins) {
//     int max_size_idx = max_element(bins.begin(), bins.end(),
//                                    [](const vector<int>& a, const vector<int>& b) { return a.size() < b.size(); }) - bins.begin();
//     if (bins[max_size_idx].size() == 1) {
//       break;
//     }
//     vector<int> new_bin = {bins[max_size_idx].back()};
//     bins[max_size_idx].pop_back();
//     bins.push_back(new_bin);
//   }
// 
//   apply_binning(bins);
// }
// 
// double OptimalBinningCategoricalSBLP::calculate_total_iv(const vector<vector<int>>& bins) {
//   int total_pos = accumulate(count_pos.begin(), count_pos.end(), 0);
//   int total_neg = accumulate(count_neg.begin(), count_neg.end(), 0);
// 
//   double total_iv = 0.0;
//   for (const auto& bin : bins) {
//     int bin_pos = 0, bin_neg = 0;
//     for (int idx : bin) {
//       bin_pos += count_pos[idx];
//       bin_neg += count_neg[idx];
//     }
//     double dist_pos = (double)bin_pos / total_pos;
//     double dist_neg = (double)bin_neg / total_neg;
//     if (dist_pos == 0) dist_pos = 0.0001;
//     if (dist_neg == 0) dist_neg = 0.0001;
//     double woe = log(dist_pos / dist_neg);
//     total_iv += (dist_pos - dist_neg) * woe;
//   }
//   return total_iv;
// }
// 
// void OptimalBinningCategoricalSBLP::apply_binning(const vector<vector<int>>& bins) {
//   int total_pos = accumulate(count_pos.begin(), count_pos.end(), 0);
//   int total_neg = accumulate(count_neg.begin(), count_neg.end(), 0);
// 
//   vector<string> bin_labels;
//   vector<double> bin_woe;
//   vector<double> bin_iv;
//   vector<int> bin_count;
//   vector<int> bin_count_pos;
//   vector<int> bin_count_neg;
// 
//   for (const auto& bin : bins) {
//     vector<string> bin_categories;
//     int bin_pos = 0, bin_neg = 0, bin_total = 0;
//     for (int idx : bin) {
//       bin_categories.push_back(unique_categories[idx]);
//       bin_pos += count_pos[idx];
//       bin_neg += count_neg[idx];
//       bin_total += count_total[idx];
//     }
//     double dist_pos = (double)bin_pos / total_pos;
//     double dist_neg = (double)bin_neg / total_neg;
//     if (dist_pos == 0) dist_pos = 0.0001;
//     if (dist_neg == 0) dist_neg = 0.0001;
//     double woe = log(dist_pos / dist_neg);
//     double iv = (dist_pos - dist_neg) * woe;
// 
//     string bin_label = merge_category_names(bin_categories);
// 
//     for (const auto& cat : bin_categories) {
//       vector<string> cat_parts;
//       stringstream ss(cat);
//       string part;
//       while (getline(ss, part, '+')) {
//         cat_parts.push_back(part);
//       }
//       for (const auto& original_cat : cat_parts) {
//         woe_map[original_cat] = woe;
//       }
//     }
// 
//     bin_labels.push_back(bin_label);
//     bin_woe.push_back(woe);
//     bin_iv.push_back(iv);
//     bin_count.push_back(bin_total);
//     bin_count_pos.push_back(bin_pos);
//     bin_count_neg.push_back(bin_neg);
//   }
// 
//   woebin = DataFrame::create(
//     Named("bin") = bin_labels,
//     Named("woe") = bin_woe,
//     Named("iv") = bin_iv,
//     Named("count") = bin_count,
//     Named("count_pos") = bin_count_pos,
//     Named("count_neg") = bin_count_neg
//   );
// }
// 
// string OptimalBinningCategoricalSBLP::merge_category_names(const vector<string>& categories) {
//   vector<string> unique_cats;
//   for (const auto& cat : categories) {
//     stringstream ss(cat);
//     string part;
//     while (getline(ss, part, '+')) {
//       if (find(unique_cats.begin(), unique_cats.end(), part) == unique_cats.end()) {
//         unique_cats.push_back(part);
//       }
//     }
//   }
//   sort(unique_cats.begin(), unique_cats.end());
//   string result;
//   for (size_t i = 0; i < unique_cats.size(); ++i) {
//     if (i > 0) result += "+";
//     result += unique_cats[i];
//   }
//   return result;
// }
// 
// List OptimalBinningCategoricalSBLP::get_results() {
//   return List::create(
//     Named("woefeature") = woefeature,
//     Named("woebin") = woebin
//   );
// }
// 
// //' Optimal Binning for Categorical Variables using Similarity-Based Logistic Partitioning (SBLP)
// //'
// //' @description
// //' This function performs optimal binning for categorical variables using a Similarity-Based Logistic Partitioning (SBLP) approach,
// //' which combines Weight of Evidence (WOE) and Information Value (IV) methods with a similarity-based merging strategy.
// //'
// //' @param target An integer vector of binary target values (0 or 1).
// //' @param feature A character vector of categorical feature values.
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff Minimum frequency for a category to be considered as a separate bin (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins before merging (default: 20).
// //'
// //' @return A list containing two elements:
// //' \itemize{
// //'   \item woefeature: A numeric vector of WOE values for each input feature value
// //'   \item woebin: A data frame containing bin information, including bin labels, WOE, IV, and counts
// //' }
// //'
// //' @details
// //' The algorithm performs the following steps:
// //' \enumerate{
// //'   \item Compute initial counts and target rates for each category
// //'   \item Handle rare categories by merging them based on similarity in target rates
// //'   \item Compute initial Weight of Evidence (WOE) and Information Value (IV)
// //'   \item Perform iterative binning by merging similar categories
// //'   \item Apply final binning and calculate WOE and IV for each bin
// //' }
// //'
// //' The Weight of Evidence (WOE) is calculated as:
// //' \deqn{WOE = \ln(\frac{\text{Proportion of Events}}{\text{Proportion of Non-Events}})}
// //'
// //' The Information Value (IV) for each bin is calculated as:
// //' \deqn{IV = (\text{Proportion of Events} - \text{Proportion of Non-Events}) \times WOE}
// //'
// //' @references
// //' \itemize{
// //'   \item Blanco-Justicia, A., & Domingo-Ferrer, J. (2019). Machine learning explainability through microaggregation and shallow decision trees. Knowledge-Based Systems, 174, 200-212.
// //'   \item Zhu, L., Qiu, D., Ergu, D., Ying, C., & Liu, K. (2019). A study on predicting loan default based on the random forest algorithm. Procedia Computer Science, 162, 503-513.
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
// //' result <- optimal_binning_categorical_sblp(target, feature)
// //'
// //' # View results
// //' print(result$woebin)
// //' hist(result$woefeature)
// //' }
// //' @author Lopes, J. E.
// //' @export
// // [[Rcpp::export]]
// List optimal_binning_categorical_sblp(IntegerVector target,
//                                       CharacterVector feature,
//                                       int min_bins = 3,
//                                       int max_bins = 5,
//                                       double bin_cutoff = 0.05,
//                                       int max_n_prebins = 20) {
//   OptimalBinningCategoricalSBLP optbin(target, feature, min_bins, max_bins, bin_cutoff, max_n_prebins);
//   optbin.fit();
//   return optbin.get_results();
// }
