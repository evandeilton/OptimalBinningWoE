// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::plugins(openmp)]]
#include <Rcpp.h>
#include <vector>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;
using namespace std;

// Constants for numerical stability
const double EPSILON = 1e-10;
const double MAX_WOE = 20.0;

class OptimalBinningCategoricalLDB {
private:
  IntegerVector target;
  CharacterVector feature;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  
  struct CategoryStats {
    string category;
    int count;
    int count_pos;
    int count_neg;
    double event_rate;
    double woe;
    double iv;
  };
  
  vector<CategoryStats> category_stats;
  
  void validate_inputs() {
    if (feature.size() != target.size()) {
      stop("Feature and target must have the same length.");
    }
    IntegerVector unique_targets = sort_unique(target);
    if (unique_targets.size() != 2 || !(unique_targets[0] == 0 && unique_targets[1] == 1)) {
      stop("Target must be binary (0 and 1).");
    }
    if (min_bins < 2) {
      stop("min_bins must be at least 2.");
    }
    if (max_bins < min_bins) {
      stop("max_bins must be greater than or equal to min_bins.");
    }
    // Additional check for max_n_prebins
    if (max_n_prebins < max_bins) {
      stop("max_n_prebins must be greater than or equal to max_bins.");
    }
  }
  
  void compute_category_stats() {
    unordered_map<string, CategoryStats> stats_map;
    int total_pos = 0;
    int total_neg = 0;
    
    int n = target.size();
    for (int i = 0; i < n; ++i) {
      string cat = as<string>(feature[i]);
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
      
      double dist_pos = (double)cs.count_pos / total_pos;
      double dist_neg = (double)cs.count_neg / total_neg;
      
      // Improved numerical stability
      dist_pos = max(dist_pos, EPSILON);
      dist_neg = max(dist_neg, EPSILON);
      
      cs.woe = log(dist_pos / dist_neg);
      // Capping WoE for numerical stability
      cs.woe = max(min(cs.woe, MAX_WOE), -MAX_WOE);
      cs.iv = (dist_pos - dist_neg) * cs.woe;
      
      category_stats.push_back(cs);
    }
  }
  
  void handle_rare_categories() {
    int total_count = target.size();
    double cutoff_count = bin_cutoff * total_count;
    
    vector<CategoryStats> non_rare_categories;
    vector<CategoryStats> rare_categories;
    
    for (auto& cat_stat : category_stats) {
      if (cat_stat.count < cutoff_count) {
        rare_categories.push_back(cat_stat);
      } else {
        non_rare_categories.push_back(cat_stat);
      }
    }
    
    for (auto& rare_cat : rare_categories) {
      double min_diff = numeric_limits<double>::max();
      int min_idx = -1;
      for (size_t i = 0; i < non_rare_categories.size(); ++i) {
        double diff = fabs(non_rare_categories[i].woe - rare_cat.woe);
        if (diff < min_diff) {
          min_diff = diff;
          min_idx = i;
        }
      }
      if (min_idx != -1) {
        non_rare_categories[min_idx].category += "+" + rare_cat.category;
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
      sort(category_stats.begin(), category_stats.end(),
           [](const CategoryStats& a, const CategoryStats& b) {
             return a.woe < b.woe;
           });
      int n_bins = max_n_prebins;
      int n_categories = category_stats.size();
      int bin_size = ceil((double)n_categories / n_bins);
      vector<CategoryStats> new_category_stats;
      
      for (int i = 0; i < n_bins; ++i) {
        int start_idx = i * bin_size;
        int end_idx = min(start_idx + bin_size, n_categories);
        if (start_idx >= end_idx) break;
        
        CategoryStats cs = category_stats[start_idx];
        for (int j = start_idx + 1; j < end_idx; ++j) {
          cs.category += "+" + category_stats[j].category;
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
    sort(category_stats.begin(), category_stats.end(),
         [](const CategoryStats& a, const CategoryStats& b) {
           return a.woe < b.woe;
         });
    
    while ((int)category_stats.size() > max_bins || !is_monotonic()) {
      if ((int)category_stats.size() <= min_bins) {
        break;
      }
      
      double min_iv_loss = numeric_limits<double>::max();
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
        
        sort(category_stats.begin(), category_stats.end(),
             [](const CategoryStats& a, const CategoryStats& b) {
               return a.woe < b.woe;
             });
      } else {
        break;
      }
      
      if (is_monotonic() && (int)category_stats.size() <= max_bins) {
        break;
      }
    }
  }
  
  bool is_monotonic() {
    if (category_stats.empty()) return true;
    
    bool increasing = true;
    bool decreasing = true;
    for (size_t i = 1; i < category_stats.size(); ++i) {
      if (category_stats[i].woe < category_stats[i - 1].woe) {
        increasing = false;
      }
      if (category_stats[i].woe > category_stats[i - 1].woe) {
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
      dist_pos = max(dist_pos, EPSILON);
      dist_neg = max(dist_neg, EPSILON);
      
      cs.woe = log(dist_pos / dist_neg);
      // Capping WoE for numerical stability
      cs.woe = max(min(cs.woe, MAX_WOE), -MAX_WOE);
      cs.iv = (dist_pos - dist_neg) * cs.woe;
    }
  }
  
  CategoryStats merge_two_bins(const CategoryStats& a, const CategoryStats& b) {
    CategoryStats merged;
    merged.category = a.category + "+" + b.category;
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
    dist_pos = max(dist_pos, EPSILON);
    dist_neg = max(dist_neg, EPSILON);
    
    merged.woe = log(dist_pos / dist_neg);
    // Capping WoE for numerical stability
    merged.woe = max(min(merged.woe, MAX_WOE), -MAX_WOE);
    merged.iv = (dist_pos - dist_neg) * merged.woe;
    
    return merged;
  }
  
public:
  OptimalBinningCategoricalLDB(IntegerVector target, CharacterVector feature,
                               int min_bins = 3, int max_bins = 5,
                               double bin_cutoff = 0.05, int max_n_prebins = 20)
    : target(target), feature(feature), min_bins(min_bins), max_bins(max_bins),
      bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins) {}
  
  List fit() {
    validate_inputs();
    compute_category_stats();
    handle_rare_categories();
    limit_prebins();
    merge_bins();
    
    unordered_map<string, double> category_woe_map;
    for (auto& cs : category_stats) {
      vector<string> categories = split(cs.category, '+');
      for (auto& cat : categories) {
        category_woe_map[cat] = cs.woe;
      }
    }
    
    NumericVector woefeature(target.size());
    for (int i = 0; i < target.size(); ++i) {
      string cat = as<string>(feature[i]);
      auto it = category_woe_map.find(cat);
      if (it != category_woe_map.end()) {
        woefeature[i] = it->second;
      } else {
        woefeature[i] = NA_REAL;
      }
    }
    
    // Check WoE consistency
    unordered_map<string, double> bin_woe_map;
    for (auto& cs : category_stats) {
      bin_woe_map[cs.category] = cs.woe;
    }
    
    bool woe_consistent = true;
    for (int i = 0; i < target.size(); ++i) {
      string cat = as<string>(feature[i]);
      auto it = category_woe_map.find(cat);
      if (it != category_woe_map.end()) {
        for (auto& bin : bin_woe_map) {
          if (bin.first.find(cat) != string::npos) {
            if (abs(it->second - bin.second) > EPSILON) {
              woe_consistent = false;
              break;
            }
          }
        }
        if (!woe_consistent) break;
      }
    }
    
    if (!woe_consistent) {
      warning("WoE consistency check failed. The binning process may have introduced inconsistencies.");
    }
    
    vector<string> bins;
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
    
    DataFrame woebin = DataFrame::create(
      Named("bin") = bins,
      Named("woe") = woe_values,
      Named("iv") = iv_values,
      Named("count") = counts,
      Named("count_pos") = counts_pos,
      Named("count_neg") = counts_neg
    );
    
    double total_iv = sum(iv_values);
    
    return List::create(
      Named("woefeature") = woefeature,
      Named("woebin") = woebin,
      Named("total_iv") = total_iv,
      Named("woe_consistent") = woe_consistent
    );
  }
  
private:
  vector<string> split(const string& s, char delimiter) {
    vector<string> tokens;
    string token;
    istringstream tokenStream(s);
    while (getline(tokenStream, token, delimiter)) {
      tokens.push_back(token);
    }
    return tokens;
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
//'
//' @return A list containing two elements:
//' \itemize{
//'   \item woefeature: A numeric vector of WoE values for each input feature value.
//'   \item woebin: A data frame with binning results, including bin names, WoE, IV, and counts.
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
//' print(result$woebin)
//' print(result$woefeature)
//' }
//'
//' @author Lopes, J. E.
//'
//' @references
//' \itemize{
//'   \item Beltrami, M., Mach, M., & Dall'Aglio, M. (2021). Monotonic Optimal Binning Algorithm for Credit Risk Modeling. Risks, 9(3), 58.
//'   \item Zeng, G. (2014). A necessary condition for a good binning algorithm in credit scoring. Applied Mathematical Sciences, 8(65), 3229-3242.
//' }
//'
//' @export
// [[Rcpp::export]]
List optimal_binning_categorical_ldb(IntegerVector target, CharacterVector feature,
                                    int min_bins = 3, int max_bins = 5,
                                    double bin_cutoff = 0.05, int max_n_prebins = 20) {
 OptimalBinningCategoricalLDB obc(target, feature, min_bins, max_bins, bin_cutoff, max_n_prebins);
 return obc.fit();
}


// // [[Rcpp::depends(RcppParallel)]]
// // [[Rcpp::plugins(openmp)]]
// #include <Rcpp.h>
// #include <vector>
// #include <map>
// #include <unordered_map>
// #include <string>
// #include <algorithm>
// #include <cmath>
// 
// #ifdef _OPENMP
// #include <omp.h>
// #endif
// 
// #include <sstream>
// #include <cfloat>
// 
// using namespace Rcpp;
// using namespace std;
// 
// class OptimalBinningCategoricalLDB {
// private:
//   IntegerVector target;
//   CharacterVector feature;
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
// 
//   struct CategoryStats {
//     string category;
//     int count;
//     int count_pos;
//     int count_neg;
//     double event_rate;
//     double woe;
//     double iv;
//   };
// 
//   vector<CategoryStats> category_stats;
// 
//   void validate_inputs() {
//     if (min_bins < 2) {
//       stop("min_bins must be at least 2.");
//     }
//     if (max_bins < min_bins) {
//       stop("max_bins must be greater than or equal to min_bins.");
//     }
//     IntegerVector unique_targets = sort_unique(target);
//     if (unique_targets.size() != 2 || !(unique_targets[0] == 0 && unique_targets[1] == 1)) {
//       stop("Target must be binary (0 and 1).");
//     }
//     if (feature.size() != target.size()) {
//       stop("Feature and target must have the same length.");
//     }
//   }
// 
//   void compute_category_stats() {
//     unordered_map<string, CategoryStats> stats_map;
//     int total_pos = 0;
//     int total_neg = 0;
// 
//     int n = target.size();
//     for (int i = 0; i < n; ++i) {
//       string cat = as<string>(feature[i]);
//       int tar = target[i];
//       if (stats_map.find(cat) == stats_map.end()) {
//         stats_map[cat] = {cat, 0, 0, 0, 0.0, 0.0, 0.0};
//       }
//       stats_map[cat].count += 1;
//       if (tar == 1) {
//         stats_map[cat].count_pos += 1;
//         total_pos += 1;
//       } else {
//         stats_map[cat].count_neg += 1;
//         total_neg += 1;
//       }
//     }
// 
//     for (auto& kv : stats_map) {
//       double rate = (double)kv.second.count_pos / kv.second.count;
//       kv.second.event_rate = rate;
// 
//       double dist_pos = (double)kv.second.count_pos / total_pos;
//       double dist_neg = (double)kv.second.count_neg / total_neg;
// 
//       if (dist_pos == 0) dist_pos = 1e-10;
//       if (dist_neg == 0) dist_neg = 1e-10;
// 
//       kv.second.woe = log(dist_pos / dist_neg);
//       kv.second.iv = (dist_pos - dist_neg) * kv.second.woe;
// 
//       category_stats.push_back(kv.second);
//     }
//   }
// 
//   void handle_rare_categories() {
//     int total_count = target.size();
//     double cutoff_count = bin_cutoff * total_count;
// 
//     vector<string> rare_categories;
//     for (auto& cat_stat : category_stats) {
//       if (cat_stat.count < cutoff_count) {
//         rare_categories.push_back(cat_stat.category);
//       }
//     }
// 
//     if (!rare_categories.empty()) {
//       for (auto& rare_cat : rare_categories) {
//         auto it = find_if(category_stats.begin(), category_stats.end(),
//                           [&](const CategoryStats& cs) { return cs.category == rare_cat; });
//         if (it != category_stats.end()) {
//           double min_diff = DBL_MAX;
//           int min_idx = -1;
//           for (size_t i = 0; i < category_stats.size(); ++i) {
//             if (category_stats[i].category != rare_cat &&
//                 find(rare_categories.begin(), rare_categories.end(), category_stats[i].category) == rare_categories.end()) {
//               double diff = fabs(category_stats[i].woe - it->woe);
//               if (diff < min_diff) {
//                 min_diff = diff;
//                 min_idx = i;
//               }
//             }
//           }
//           if (min_idx != -1) {
//             category_stats[min_idx].category += "+" + it->category;
//             category_stats[min_idx].count += it->count;
//             category_stats[min_idx].count_pos += it->count_pos;
//             category_stats[min_idx].count_neg += it->count_neg;
//             category_stats.erase(it);
//           }
//         }
//       }
// 
//       int total_pos = 0;
//       int total_neg = 0;
//       for (auto& cs : category_stats) {
//         total_pos += cs.count_pos;
//         total_neg += cs.count_neg;
//       }
//       for (auto& cs : category_stats) {
//         double dist_pos = (double)cs.count_pos / total_pos;
//         double dist_neg = (double)cs.count_neg / total_neg;
// 
//         if (dist_pos == 0) dist_pos = 1e-10;
//         if (dist_neg == 0) dist_neg = 1e-10;
// 
//         cs.woe = log(dist_pos / dist_neg);
//         cs.iv = (dist_pos - dist_neg) * cs.woe;
//       }
//     }
//   }
// 
//   void limit_prebins() {
//     if ((int)category_stats.size() > max_n_prebins) {
//       sort(category_stats.begin(), category_stats.end(),
//            [](const CategoryStats& a, const CategoryStats& b) {
//              return a.woe < b.woe;
//            });
//       int n_bins = max_n_prebins;
//       int n_categories = category_stats.size();
//       int bin_size = ceil((double)n_categories / n_bins);
//       vector<CategoryStats> new_category_stats;
// 
//       int total_pos = 0;
//       int total_neg = 0;
//       for (auto& cs : category_stats) {
//         total_pos += cs.count_pos;
//         total_neg += cs.count_neg;
//       }
// 
//       for (int i = 0; i < n_bins; ++i) {
//         int start_idx = i * bin_size;
//         int end_idx = min(start_idx + bin_size, n_categories);
//         if (start_idx >= end_idx) break;
// 
//         CategoryStats cs = category_stats[start_idx];
//         for (int j = start_idx + 1; j < end_idx; ++j) {
//           cs.category += "+" + category_stats[j].category;
//           cs.count += category_stats[j].count;
//           cs.count_pos += category_stats[j].count_pos;
//           cs.count_neg += category_stats[j].count_neg;
//         }
//         double dist_pos = (double)cs.count_pos / total_pos;
//         double dist_neg = (double)cs.count_neg / total_neg;
//         if (dist_pos == 0) dist_pos = 1e-10;
//         if (dist_neg == 0) dist_neg = 1e-10;
//         cs.woe = log(dist_pos / dist_neg);
//         cs.iv = (dist_pos - dist_neg) * cs.woe;
// 
//         new_category_stats.push_back(cs);
//       }
//       category_stats = new_category_stats;
//     }
//   }
// 
//   void merge_bins() {
//     sort(category_stats.begin(), category_stats.end(),
//          [](const CategoryStats& a, const CategoryStats& b) {
//            return a.woe < b.woe;
//          });
// 
//     while ((int)category_stats.size() > max_bins || !is_monotonic()) {
//       double min_iv_loss = DBL_MAX;
//       int merge_idx = -1;
// 
//       int total_pos = 0;
//       int total_neg = 0;
//       for (auto& cs : category_stats) {
//         total_pos += cs.count_pos;
//         total_neg += cs.count_neg;
//       }
// 
//       for (size_t i = 0; i < category_stats.size() - 1; ++i) {
//         CategoryStats merged_bin = category_stats[i];
//         merged_bin.category += "+" + category_stats[i + 1].category;
//         merged_bin.count += category_stats[i + 1].count;
//         merged_bin.count_pos += category_stats[i + 1].count_pos;
//         merged_bin.count_neg += category_stats[i + 1].count_neg;
// 
//         double dist_pos = (double)merged_bin.count_pos / total_pos;
//         double dist_neg = (double)merged_bin.count_neg / total_neg;
//         if (dist_pos == 0) dist_pos = 1e-10;
//         if (dist_neg == 0) dist_neg = 1e-10;
// 
//         merged_bin.woe = log(dist_pos / dist_neg);
//         merged_bin.iv = (dist_pos - dist_neg) * merged_bin.woe;
// 
//         double iv_loss = category_stats[i].iv + category_stats[i + 1].iv - merged_bin.iv;
// 
//         if (iv_loss < min_iv_loss) {
//           min_iv_loss = iv_loss;
//           merge_idx = i;
//         }
//       }
// 
//       if (merge_idx != -1) {
//         CategoryStats merged_bin = category_stats[merge_idx];
//         merged_bin.category += "+" + category_stats[merge_idx + 1].category;
//         merged_bin.count += category_stats[merge_idx + 1].count;
//         merged_bin.count_pos += category_stats[merge_idx + 1].count_pos;
//         merged_bin.count_neg += category_stats[merge_idx + 1].count_neg;
// 
//         double dist_pos = (double)merged_bin.count_pos / total_pos;
//         double dist_neg = (double)merged_bin.count_neg / total_neg;
//         if (dist_pos == 0) dist_pos = 1e-10;
//         if (dist_neg == 0) dist_neg = 1e-10;
//         merged_bin.woe = log(dist_pos / dist_neg);
//         merged_bin.iv = (dist_pos - dist_neg) * merged_bin.woe;
// 
//         category_stats[merge_idx] = merged_bin;
//         category_stats.erase(category_stats.begin() + merge_idx + 1);
// 
//         sort(category_stats.begin(), category_stats.end(),
//              [](const CategoryStats& a, const CategoryStats& b) {
//                return a.woe < b.woe;
//              });
//       } else {
//         break;
//       }
// 
//       if (is_monotonic() && (int)category_stats.size() <= max_bins) {
//         break;
//       }
// 
//       if ((int)category_stats.size() <= min_bins) {
//         break;
//       }
//     }
//   }
// 
//   bool is_monotonic() {
//     if (category_stats.empty()) return true;
// 
//     bool increasing = true;
//     bool decreasing = true;
//     for (size_t i = 1; i < category_stats.size(); ++i) {
//       if (category_stats[i].woe < category_stats[i - 1].woe) {
//         increasing = false;
//       }
//       if (category_stats[i].woe > category_stats[i - 1].woe) {
//         decreasing = false;
//       }
//     }
//     return increasing || decreasing;
//   }
// 
// public:
//   OptimalBinningCategoricalLDB(IntegerVector target, CharacterVector feature,
//                                int min_bins = 3, int max_bins = 5,
//                                double bin_cutoff = 0.05, int max_n_prebins = 20)
//     : target(target), feature(feature), min_bins(min_bins), max_bins(max_bins),
//       bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins) {}
// 
//   List fit() {
//     validate_inputs();
//     compute_category_stats();
//     handle_rare_categories();
//     limit_prebins();
//     merge_bins();
// 
//     unordered_map<string, double> category_woe_map;
//     for (auto& cs : category_stats) {
//       vector<string> categories = split(cs.category, '+');
//       for (auto& cat : categories) {
//         category_woe_map[cat] = cs.woe;
//       }
//     }
// 
//     NumericVector woefeature(target.size());
//     for (int i = 0; i < target.size(); ++i) {
//       string cat = as<string>(feature[i]);
//       auto it = category_woe_map.find(cat);
//       if (it != category_woe_map.end()) {
//         woefeature[i] = it->second;
//       } else {
//         woefeature[i] = NA_REAL;
//       }
//     }
// 
//     vector<string> bins;
//     NumericVector woe_values;
//     NumericVector iv_values;
//     IntegerVector counts;
//     IntegerVector counts_pos;
//     IntegerVector counts_neg;
// 
//     for (auto& cs : category_stats) {
//       bins.push_back(cs.category);
//       woe_values.push_back(cs.woe);
//       iv_values.push_back(cs.iv);
//       counts.push_back(cs.count);
//       counts_pos.push_back(cs.count_pos);
//       counts_neg.push_back(cs.count_neg);
//     }
// 
//     DataFrame woebin = DataFrame::create(
//       Named("bin") = bins,
//       Named("woe") = woe_values,
//       Named("iv") = iv_values,
//       Named("count") = counts,
//       Named("count_pos") = counts_pos,
//       Named("count_neg") = counts_neg
//     );
// 
//     return List::create(
//       Named("woefeature") = woefeature,
//       Named("woebin") = woebin
//     );
//   }
// 
// private:
//   vector<string> split(const string& s, char delimiter) {
//     vector<string> tokens;
//     string token;
//     stringstream tokenStream(s);
//     while (getline(tokenStream, token, delimiter)) {
//       tokens.push_back(token);
//     }
//     return tokens;
//   }
// };
// 
// 
// 
// //' @title Categorical Optimal Binning with Local Distance-Based Algorithm
// //'
// //' @description
// //' This function performs optimal binning for categorical variables using a Local Distance-Based (LDB) algorithm,
// //' which merges categories based on their Weight of Evidence (WoE) similarity and Information Value (IV) loss.
// //'
// //' @param target An integer vector of binary target values (0 or 1).
// //' @param feature A character vector of categorical feature values.
// //' @param min_bins Minimum number of bins to create (default: 3).
// //' @param max_bins Maximum number of bins to create (default: 5).
// //' @param bin_cutoff Minimum frequency for a category to be considered as a separate bin (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins before merging (default: 20).
// //'
// //' @return A list containing two elements:
// //' \itemize{
// //'   \item woefeature: A numeric vector of WoE values for each input feature value.
// //'   \item woebin: A data frame with binning results, including bin names, WoE, IV, and counts.
// //' }
// //'
// //' @details
// //' The LDB algorithm works as follows:
// //' \enumerate{
// //'   \item Compute initial statistics for each category.
// //'   \item Handle rare categories by merging them with the most similar (in terms of WoE) non-rare category.
// //'   \item Limit the number of pre-bins to max_n_prebins.
// //'   \item Iteratively merge bins with the lowest IV loss until the desired number of bins is reached or monotonicity is achieved.
// //'   \item Ensure monotonicity of WoE across bins.
// //' }
// //'
// //' Weight of Evidence (WoE) for each bin is calculated as:
// //'
// //' \deqn{WoE = \ln\left(\frac{P(X|Y=1)}{P(X|Y=0)}\right)}
// //'
// //' Information Value (IV) for each bin is calculated as:
// //'
// //' \deqn{IV = (P(X|Y=1) - P(X|Y=0)) \times WoE}
// //'
// //' @examples
// //' \dontrun{
// //' # Sample data
// //' target <- c(1, 0, 1, 1, 0, 1, 0, 0, 1, 1)
// //' feature <- c("A", "B", "A", "C", "B", "D", "C", "A", "D", "B")
// //'
// //' # Run optimal binning
// //' result <- optimal_binning_categorical_ldb(target, feature, min_bins = 2, max_bins = 4)
// //'
// //' # View results
// //' print(result$woebin)
// //' print(result$woefeature)
// //' }
// //'
// //' @author Lopes, J. E.
// //'
// //' @references
// //' \itemize{
// //'   \item Beltrami, M., Mach, M., & Dall'Aglio, M. (2021). Monotonic Optimal Binning Algorithm for Credit Risk Modeling. Risks, 9(3), 58.
// //'   \item Zeng, G. (2014). A necessary condition for a good binning algorithm in credit scoring. Applied Mathematical Sciences, 8(65), 3229-3242.
// //' }
// //'
// //' @export
// // [[Rcpp::export]]
// List optimal_binning_categorical_ldb(IntegerVector target, CharacterVector feature,
//                                    int min_bins = 3, int max_bins = 5,
//                                    double bin_cutoff = 0.05, int max_n_prebins = 20) {
// OptimalBinningCategoricalLDB obc(target, feature, min_bins, max_bins, bin_cutoff, max_n_prebins);
// return obc.fit();
// }
