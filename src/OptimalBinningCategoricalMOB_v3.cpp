// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <limits>

using namespace Rcpp;

class OptimalBinningCategoricalMOB {
private:
 std::vector<std::string> feature;
 std::vector<bool> target;
 
 int min_bins;
 int max_bins;
 double bin_cutoff;
 int max_n_prebins;
 std::string bin_separator;
 double convergence_threshold;
 int max_iterations;
 
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
 
 void calculateCategoryStats();
 void calculateInitialBins();
 void enforceMonotonicity();
 void limitBins();
 void computeWoEandIV();
 
public:
 OptimalBinningCategoricalMOB(std::vector<std::string>&& feature_,
                              std::vector<bool>&& target_,
                              int min_bins_,
                              int max_bins_,
                              double bin_cutoff_,
                              int max_n_prebins_,
                              std::string bin_separator_,
                              double convergence_threshold_,
                              int max_iterations_);
 
 List fit();
};

OptimalBinningCategoricalMOB::OptimalBinningCategoricalMOB(
 std::vector<std::string>&& feature_,
 std::vector<bool>&& target_,
 int min_bins_,
 int max_bins_,
 double bin_cutoff_,
 int max_n_prebins_,
 std::string bin_separator_,
 double convergence_threshold_,
 int max_iterations_)
 : feature(std::move(feature_)),
   target(std::move(target_)),
   min_bins(min_bins_),
   max_bins(max_bins_),
   bin_cutoff(bin_cutoff_),
   max_n_prebins(max_n_prebins_),
   bin_separator(bin_separator_),
   convergence_threshold(convergence_threshold_),
   max_iterations(max_iterations_),
   total_good(0),
   total_bad(0) {
 if (feature.size() != target.size()) {
   throw std::invalid_argument("Feature and target vectors must have the same length");
 }
 if (min_bins <= 0 || max_bins <= 0 || min_bins > max_bins) {
   throw std::invalid_argument("Invalid min_bins or max_bins values");
 }
 if (bin_cutoff <= 0 || bin_cutoff >= 1) {
   throw std::invalid_argument("bin_cutoff must be between 0 and 1");
 }
 if (max_n_prebins <= 0) {
   throw std::invalid_argument("max_n_prebins must be positive");
 }
 if (convergence_threshold <= 0) {
   throw std::invalid_argument("convergence_threshold must be positive");
 }
 if (max_iterations <= 0) {
   throw std::invalid_argument("max_iterations must be positive");
 }
}

void OptimalBinningCategoricalMOB::calculateCategoryStats() {
 category_counts.clear();
 category_good.clear();
 category_bad.clear();
 total_good = 0;
 total_bad = 0;
 
 for (size_t i = 0; i < feature.size(); ++i) {
   const auto& cat = feature[i];
   category_counts[cat]++;
   if (target[i]) {
     category_good[cat]++;
     total_good++;
   } else {
     category_bad[cat]++;
     total_bad++;
   }
 }
}

void OptimalBinningCategoricalMOB::calculateInitialBins() {
 struct CategoryWoE {
   std::string category;
   double woe;
   double good;
   double bad;
 };
 
 std::vector<CategoryWoE> cat_woe_vec;
 cat_woe_vec.reserve(category_counts.size());
 
 for (const auto& [cat, count] : category_counts) {
   double good = category_good[cat];
   double bad = category_bad[cat];
   
   double rate_good = good / total_good;
   double rate_bad = bad / total_bad;
   
   if (rate_good == 0) {
     rate_good = std::numeric_limits<double>::min();
   }
   if (rate_bad == 0) {
     rate_bad = std::numeric_limits<double>::min();
   }
   
   double woe = std::log(rate_good / rate_bad);
   
   cat_woe_vec.push_back({cat, woe, good, bad});
 }
 
 // Sort categories by WoE
 std::sort(cat_woe_vec.begin(), cat_woe_vec.end(),
           [](const CategoryWoE& a, const CategoryWoE& b) {
             return a.woe < b.woe;
           });
 
 bins.clear();
 bins.reserve(cat_woe_vec.size());
 
 for (const auto& c : cat_woe_vec) {
   bins.push_back({{c.category}, c.good, c.bad, c.woe, 0.0});
 }
 
 // Merge bins to limit the number of prebins
 while (bins.size() > static_cast<size_t>(max_n_prebins)) {
   // Find two adjacent bins with minimum difference in WoE
   size_t min_index = 0;
   double min_diff = std::abs(bins[1].woe - bins[0].woe);
   
   for (size_t i = 1; i < bins.size() - 1; ++i) {
     double diff = std::abs(bins[i + 1].woe - bins[i].woe);
     if (diff < min_diff) {
       min_diff = diff;
       min_index = i;
     }
   }
   
   // Merge bins[min_index] and bins[min_index + 1]
   auto& bin1 = bins[min_index];
   auto& bin2 = bins[min_index + 1];
   
   bin1.categories.insert(bin1.categories.end(),
                          bin2.categories.begin(),
                          bin2.categories.end());
   bin1.good_count += bin2.good_count;
   bin1.bad_count += bin2.bad_count;
   
   double rate_good = bin1.good_count / total_good;
   double rate_bad = bin1.bad_count / total_bad;
   
   if (rate_good == 0) {
     rate_good = std::numeric_limits<double>::min();
   }
   if (rate_bad == 0) {
     rate_bad = std::numeric_limits<double>::min();
   }
   
   bin1.woe = std::log(rate_good / rate_bad);
   
   bins.erase(bins.begin() + min_index + 1);
 }
}

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
 
 if (is_increasing && is_decreasing) {
   bool monotonic = false;
   int iterations = 0;
   
   while (!monotonic && bins.size() > static_cast<size_t>(min_bins) && iterations < max_iterations) {
     monotonic = true;
     for (size_t i = 0; i < bins.size() - 1; ++i) {
       if (bins[i + 1].woe > bins[i].woe) {
         bins[i].categories.insert(bins[i].categories.end(),
                                   bins[i + 1].categories.begin(),
                                   bins[i + 1].categories.end());
         bins[i].good_count += bins[i + 1].good_count;
         bins[i].bad_count += bins[i + 1].bad_count;
         
         double rate_good = bins[i].good_count / total_good;
         double rate_bad = bins[i].bad_count / total_bad;
         
         if (rate_good == 0) {
           rate_good = std::numeric_limits<double>::min();
         }
         if (rate_bad == 0) {
           rate_bad = std::numeric_limits<double>::min();
         }
         
         bins[i].woe = std::log(rate_good / rate_bad);
         
         bins.erase(bins.begin() + i + 1);
         monotonic = false;
         break;
       }
     }
     iterations++;
   }
 }
}

void OptimalBinningCategoricalMOB::limitBins() {
 while (bins.size() > static_cast<size_t>(max_bins)) {
   size_t min_index = 0;
   double min_diff = std::abs(bins[1].woe - bins[0].woe);
   
   for (size_t i = 1; i < bins.size() - 1; ++i) {
     double diff = std::abs(bins[i + 1].woe - bins[i].woe);
     if (diff < min_diff) {
       min_diff = diff;
       min_index = i;
     }
   }
   
   auto& bin1 = bins[min_index];
   auto& bin2 = bins[min_index + 1];
   
   bin1.categories.insert(bin1.categories.end(),
                          bin2.categories.begin(),
                          bin2.categories.end());
   bin1.good_count += bin2.good_count;
   bin1.bad_count += bin2.bad_count;
   
   double rate_good = bin1.good_count / total_good;
   double rate_bad = bin1.bad_count / total_bad;
   
   if (rate_good == 0) {
     rate_good = std::numeric_limits<double>::min();
   }
   if (rate_bad == 0) {
     rate_bad = std::numeric_limits<double>::min();
   }
   
   bin1.woe = std::log(rate_good / rate_bad);
   
   bins.erase(bins.begin() + min_index + 1);
 }
}

void OptimalBinningCategoricalMOB::computeWoEandIV() {
 for (auto& bin : bins) {
   double rate_good = bin.good_count / total_good;
   double rate_bad = bin.bad_count / total_bad;
   
   if (rate_good == 0) {
     rate_good = std::numeric_limits<double>::min();
   }
   if (rate_bad == 0) {
     rate_bad = std::numeric_limits<double>::min();
   }
   
   bin.woe = std::log(rate_good / rate_bad);
   bin.iv = (rate_good - rate_bad) * bin.woe;
 }
}

List OptimalBinningCategoricalMOB::fit() {
 calculateCategoryStats();
 
 int ncat = category_counts.size();
 bool converged = true;
 int iterations_run = 0;
 
 if (ncat > max_bins) {
   calculateInitialBins();
   
   double prev_total_iv = 0.0;
   int iterations = 0;
   bool monotonic = false;
   
   while (!monotonic && iterations < max_iterations) {
     enforceMonotonicity();
     limitBins();
     computeWoEandIV();
     
     double total_iv = 0.0;
     for (const auto& bin : bins) {
       total_iv += bin.iv;
     }
     
     if (std::abs(total_iv - prev_total_iv) < convergence_threshold) {
       monotonic = true;
     }
     
     prev_total_iv = total_iv;
     iterations++;
   }
   
   converged = monotonic;
   iterations_run = iterations;
 } else {
   bins.clear();
   for (const auto& [cat, count] : category_counts) {
     double good = category_good[cat];
     double bad = category_bad[cat];
     double rate_good = good / total_good;
     double rate_bad = bad / total_bad;
     
     if (rate_good == 0) {
       rate_good = std::numeric_limits<double>::min();
     }
     if (rate_bad == 0) {
       rate_bad = std::numeric_limits<double>::min();
     }
     
     double woe = std::log(rate_good / rate_bad);
     double iv = (rate_good - rate_bad) * woe;
     
     bins.push_back({{cat}, good, bad, woe, iv});
   }
 }
 
 std::vector<std::string> bin_names;
 std::vector<double> woe_values;
 std::vector<double> iv_values;
 std::vector<int> count_values;
 std::vector<int> count_pos_values;
 std::vector<int> count_neg_values;
 
 for (auto& bin : bins) {
   std::vector<std::string> sorted_categories = bin.categories;
   std::sort(sorted_categories.begin(), sorted_categories.end());
   
   std::string bin_name = sorted_categories[0];
   for (size_t i = 1; i < sorted_categories.size(); ++i) {
     bin_name += bin_separator + sorted_categories[i];
   }
   bin_names.push_back(bin_name);
   woe_values.push_back(bin.woe);
   iv_values.push_back(bin.iv);
   count_values.push_back(static_cast<int>(bin.good_count + bin.bad_count));
   count_pos_values.push_back(static_cast<int>(bin.good_count));
   count_neg_values.push_back(static_cast<int>(bin.bad_count));
 }
 
 Rcpp::NumericVector ids(bin_names.size());
 for(int i = 0; i < bin_names.size(); i++) {
    ids[i] = i + 1;
 }
 
 return Rcpp::List::create(
   Named("id") = ids,
   Named("bin") = bin_names,
   Named("woe") = woe_values,
   Named("iv") = iv_values,
   Named("count") = count_values,
   Named("count_pos") = count_pos_values,
   Named("count_neg") = count_neg_values,
   Named("converged") = converged,
   Named("iterations") = iterations_run
 );
}

//' @title Optimal Binning for Categorical Variables using Monotonic Optimal Binning (MOB)
//'
//' @description
//' This function performs optimal binning for categorical variables using the Monotonic Optimal Binning (MOB) approach.
//'
//' @param target An integer vector of binary target values (0 or 1).
//' @param feature A character vector of categorical feature values.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum proportion of observations in a bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins (default: 20).
//' @param bin_separator Separator used for merging category names (default: "%;%").
//' @param convergence_threshold Convergence threshold for the algorithm (default: 1e-6).
//' @param max_iterations Maximum number of iterations for the algorithm (default: 1000).
//'
//' @return A list containing the following elements:
//' \itemize{
//'   \item bin: A character vector of bin names (merged categories)
//'   \item woe: A numeric vector of Weight of Evidence (WoE) values for each bin
//'   \item iv: A numeric vector of Information Value (IV) for each bin
//'   \item count: An integer vector of total counts for each bin
//'   \item count_pos: An integer vector of positive target counts for each bin
//'   \item count_neg: An integer vector of negative target counts for each bin
//'   \item converged: A logical value indicating whether the algorithm converged
//'   \item iterations: An integer value indicating the number of iterations run
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
//' result <- optimal_binning_categorical_mob(target, feature)
//'
//' # View results
//' print(result)
//' }
//'
//' @details
//' Este algoritmo aplica o Monotonic Optimal Binning (MOB) para variáveis categóricas.
//' O processo visa maximizar o IV (Information Value) mantendo a monotonicidade no WoE (Weight of Evidence).
//'
//' Passos do algoritmo:
//' 1. Cálculo das estatísticas por categoria.
//' 2. Pré-binagem e ordenação por WoE.
//' 3. Aplicação da monotonicidade e ajuste de bins.
//' 4. Limitação do número de bins a max_bins.
//' 5. Cálculo dos valores finais de WoE e IV.
//'
//' @references
//' \itemize{
//'    \item Belotti, T., Crook, J. (2009). Credit Scoring with Macroeconomic Variables Using Survival Analysis.
//'          *Journal of the Operational Research Society*, 60(12), 1699-1707.
//'    \item Mironchyk, P., Tchistiakov, V. (2017). Monotone optimal binning algorithm for credit risk modeling.
//'          *arXiv preprint* arXiv:1711.05095.
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_mob(Rcpp::IntegerVector target,
                                          Rcpp::StringVector feature,
                                          int min_bins = 3,
                                          int max_bins = 5,
                                          double bin_cutoff = 0.05,
                                          int max_n_prebins = 20,
                                          std::string bin_separator = "%;%",
                                          double convergence_threshold = 1e-6,
                                          int max_iterations = 1000) {
 if (target.size() != feature.size()) {
   Rcpp::stop("Target and feature vectors must have the same length.");
 }
 
 std::vector<bool> target_vec;
 target_vec.reserve(target.size());
 for (int i = 0; i < target.size(); ++i) {
   if (target[i] != 0 && target[i] != 1) {
     Rcpp::stop("Target vector must be binary (0 and 1).");
   }
   target_vec.push_back(target[i] == 1);
 }
 
 std::vector<std::string> feature_vec;
 feature_vec.reserve(feature.size());
 for (int i = 0; i < feature.size(); ++i) {
   feature_vec.push_back(feature[i] == NA_STRING ? "NA" : Rcpp::as<std::string>(feature[i]));
 }
 
 std::unordered_set<std::string> unique_categories(feature_vec.begin(), feature_vec.end());
 max_bins = std::min(max_bins, static_cast<int>(unique_categories.size()));
 min_bins = std::min(min_bins, max_bins);
 
 try {
   OptimalBinningCategoricalMOB mob(std::move(feature_vec), std::move(target_vec),
                                    min_bins, max_bins, bin_cutoff, max_n_prebins,
                                    bin_separator, convergence_threshold, max_iterations);
   return mob.fit();
 } catch (const std::exception& e) {
   Rcpp::stop("Error in OptimalBinningCategoricalMOB: " + std::string(e.what()));
 }
}


// #include <Rcpp.h>
// #include <vector>
// #include <string>
// #include <unordered_map>
// #include <unordered_set>
// #include <algorithm>
// #include <cmath>
// #include <stdexcept>
// #include <limits>
// 
// using namespace Rcpp;
// 
// class OptimalBinningCategoricalMOB {
// private:
//   std::vector<std::string> feature;
//   std::vector<bool> target;
//   
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
//   std::string bin_separator;
//   double convergence_threshold;
//   int max_iterations;
//   
//   std::unordered_map<std::string, double> category_counts;
//   std::unordered_map<std::string, double> category_good;
//   std::unordered_map<std::string, double> category_bad;
//   double total_good;
//   double total_bad;
//   
//   struct Bin {
//     std::vector<std::string> categories;
//     double good_count;
//     double bad_count;
//     double woe;
//     double iv;
//   };
//   
//   std::vector<Bin> bins;
//   
//   void calculateCategoryStats();
//   void calculateInitialBins();
//   void enforceMonotonicity();
//   void limitBins();
//   void computeWoEandIV();
//   
// public:
//   OptimalBinningCategoricalMOB(std::vector<std::string>&& feature_,
//                                std::vector<bool>&& target_,
//                                int min_bins_,
//                                int max_bins_,
//                                double bin_cutoff_,
//                                int max_n_prebins_,
//                                std::string bin_separator_,
//                                double convergence_threshold_,
//                                int max_iterations_);
//   
//   List fit();
// };
// 
// OptimalBinningCategoricalMOB::OptimalBinningCategoricalMOB(
//   std::vector<std::string>&& feature_,
//   std::vector<bool>&& target_,
//   int min_bins_,
//   int max_bins_,
//   double bin_cutoff_,
//   int max_n_prebins_,
//   std::string bin_separator_,
//   double convergence_threshold_,
//   int max_iterations_)
//   : feature(std::move(feature_)),
//     target(std::move(target_)),
//     min_bins(min_bins_),
//     max_bins(max_bins_),
//     bin_cutoff(bin_cutoff_),
//     max_n_prebins(max_n_prebins_),
//     bin_separator(bin_separator_),
//     convergence_threshold(convergence_threshold_),
//     max_iterations(max_iterations_),
//     total_good(0),
//     total_bad(0) {
//   if (feature.size() != target.size()) {
//     throw std::invalid_argument("Feature and target vectors must have the same length");
//   }
//   if (min_bins <= 0 || max_bins <= 0 || min_bins > max_bins) {
//     throw std::invalid_argument("Invalid min_bins or max_bins values");
//   }
//   if (bin_cutoff <= 0 || bin_cutoff >= 1) {
//     throw std::invalid_argument("bin_cutoff must be between 0 and 1");
//   }
//   if (max_n_prebins <= 0) {
//     throw std::invalid_argument("max_n_prebins must be positive");
//   }
//   if (convergence_threshold <= 0) {
//     throw std::invalid_argument("convergence_threshold must be positive");
//   }
//   if (max_iterations <= 0) {
//     throw std::invalid_argument("max_iterations must be positive");
//   }
// }
// 
// void OptimalBinningCategoricalMOB::calculateCategoryStats() {
//   category_counts.clear();
//   category_good.clear();
//   category_bad.clear();
//   total_good = 0;
//   total_bad = 0;
//   
//   for (size_t i = 0; i < feature.size(); ++i) {
//     const auto& cat = feature[i];
//     category_counts[cat]++;
//     if (target[i]) {
//       category_good[cat]++;
//       total_good++;
//     } else {
//       category_bad[cat]++;
//       total_bad++;
//     }
//   }
// }
// 
// void OptimalBinningCategoricalMOB::calculateInitialBins() {
//   struct CategoryWoE {
//     std::string category;
//     double woe;
//     double good;
//     double bad;
//   };
//   
//   std::vector<CategoryWoE> cat_woe_vec;
//   cat_woe_vec.reserve(category_counts.size());
//   
//   for (const auto& [cat, count] : category_counts) {
//     double good = category_good[cat];
//     double bad = category_bad[cat];
//     
//     double rate_good = good / total_good;
//     double rate_bad = bad / total_bad;
//     
//     // Handle zero rates
//     if (rate_good == 0) {
//       rate_good = std::numeric_limits<double>::min();
//     }
//     if (rate_bad == 0) {
//       rate_bad = std::numeric_limits<double>::min();
//     }
//     
//     double woe = std::log(rate_good / rate_bad);
//     
//     cat_woe_vec.push_back({cat, woe, good, bad});
//   }
//   
//   // Sort categories by WoE
//   std::sort(cat_woe_vec.begin(), cat_woe_vec.end(),
//             [](const CategoryWoE& a, const CategoryWoE& b) {
//               return a.woe < b.woe;
//             });
//   
//   bins.clear();
//   bins.reserve(cat_woe_vec.size());
//   
//   for (const auto& c : cat_woe_vec) {
//     bins.push_back({{c.category}, c.good, c.bad, c.woe, 0.0});
//   }
//   
//   // Merge bins to limit the number of prebins
//   while (bins.size() > static_cast<size_t>(max_n_prebins)) {
//     // Find two adjacent bins with minimum difference in WoE
//     size_t min_index = 0;
//     double min_diff = std::abs(bins[1].woe - bins[0].woe);
//     
//     for (size_t i = 1; i < bins.size() - 1; ++i) {
//       double diff = std::abs(bins[i + 1].woe - bins[i].woe);
//       if (diff < min_diff) {
//         min_diff = diff;
//         min_index = i;
//       }
//     }
//     
//     // Merge bins[min_index] and bins[min_index + 1]
//     auto& bin1 = bins[min_index];
//     auto& bin2 = bins[min_index + 1];
//     
//     bin1.categories.insert(bin1.categories.end(),
//                            bin2.categories.begin(),
//                            bin2.categories.end());
//     bin1.good_count += bin2.good_count;
//     bin1.bad_count += bin2.bad_count;
//     
//     // Recalculate WoE for merged bin
//     double rate_good = bin1.good_count / total_good;
//     double rate_bad = bin1.bad_count / total_bad;
//     
//     if (rate_good == 0) {
//       rate_good = std::numeric_limits<double>::min();
//     }
//     if (rate_bad == 0) {
//       rate_bad = std::numeric_limits<double>::min();
//     }
//     
//     bin1.woe = std::log(rate_good / rate_bad);
//     
//     bins.erase(bins.begin() + min_index + 1);
//   }
// }
// 
// void OptimalBinningCategoricalMOB::enforceMonotonicity() {
//   bool is_increasing = false;
//   bool is_decreasing = false;
//   for (size_t i = 0; i < bins.size() - 1; ++i) {
//     if (bins[i + 1].woe > bins[i].woe) {
//       is_increasing = true;
//     } else if (bins[i + 1].woe < bins[i].woe) {
//       is_decreasing = true;
//     }
//   }
//   
//   if (is_increasing && is_decreasing) {
//     bool monotonic = false;
//     int iterations = 0;
//     
//     while (!monotonic && bins.size() > static_cast<size_t>(min_bins) && iterations < max_iterations) {
//       monotonic = true;
//       for (size_t i = 0; i < bins.size() - 1; ++i) {
//         if (bins[i + 1].woe > bins[i].woe) {
//           // Merge bins[i] and bins[i + 1]
//           bins[i].categories.insert(bins[i].categories.end(),
//                                     bins[i + 1].categories.begin(),
//                                     bins[i + 1].categories.end());
//           bins[i].good_count += bins[i + 1].good_count;
//           bins[i].bad_count += bins[i + 1].bad_count;
//           
//           // Recalculate WoE
//           double rate_good = bins[i].good_count / total_good;
//           double rate_bad = bins[i].bad_count / total_bad;
//           
//           if (rate_good == 0) {
//             rate_good = std::numeric_limits<double>::min();
//           }
//           if (rate_bad == 0) {
//             rate_bad = std::numeric_limits<double>::min();
//           }
//           
//           bins[i].woe = std::log(rate_good / rate_bad);
//           
//           bins.erase(bins.begin() + i + 1);
//           monotonic = false;
//           break;
//         }
//       }
//       iterations++;
//     }
//   }
// }
// 
// void OptimalBinningCategoricalMOB::limitBins() {
//   // Merge bins to limit number of bins to max_bins
//   while (bins.size() > static_cast<size_t>(max_bins)) {
//     // Find two adjacent bins with minimum difference in WoE
//     size_t min_index = 0;
//     double min_diff = std::abs(bins[1].woe - bins[0].woe);
//     
//     for (size_t i = 1; i < bins.size() - 1; ++i) {
//       double diff = std::abs(bins[i + 1].woe - bins[i].woe);
//       if (diff < min_diff) {
//         min_diff = diff;
//         min_index = i;
//       }
//     }
//     
//     // Merge bins[min_index] and bins[min_index + 1]
//     auto& bin1 = bins[min_index];
//     auto& bin2 = bins[min_index + 1];
//     
//     bin1.categories.insert(bin1.categories.end(),
//                            bin2.categories.begin(),
//                            bin2.categories.end());
//     bin1.good_count += bin2.good_count;
//     bin1.bad_count += bin2.bad_count;
//     
//     // Recalculate WoE for merged bin
//     double rate_good = bin1.good_count / total_good;
//     double rate_bad = bin1.bad_count / total_bad;
//     
//     if (rate_good == 0) {
//       rate_good = std::numeric_limits<double>::min();
//     }
//     if (rate_bad == 0) {
//       rate_bad = std::numeric_limits<double>::min();
//     }
//     
//     bin1.woe = std::log(rate_good / rate_bad);
//     
//     bins.erase(bins.begin() + min_index + 1);
//   }
// }
// 
// void OptimalBinningCategoricalMOB::computeWoEandIV() {
//   for (auto& bin : bins) {
//     double rate_good = bin.good_count / total_good;
//     double rate_bad = bin.bad_count / total_bad;
//     
//     if (rate_good == 0) {
//       rate_good = std::numeric_limits<double>::min();
//     }
//     if (rate_bad == 0) {
//       rate_bad = std::numeric_limits<double>::min();
//     }
//     
//     bin.woe = std::log(rate_good / rate_bad);
//     bin.iv = (rate_good - rate_bad) * bin.woe;
//   }
// }
// 
// List OptimalBinningCategoricalMOB::fit() {
//   calculateCategoryStats();
//   
//   // Check if optimization is needed
//   int ncat = category_counts.size();
//   bool converged = true;
//   int iterations_run = 0;
//   
//   if (ncat > max_bins) {
//     calculateInitialBins();
//     
//     double prev_total_iv = 0.0;
//     int iterations = 0;
//     bool monotonic = false;
//     
//     while (!monotonic && iterations < max_iterations) {
//       enforceMonotonicity();
//       limitBins();
//       computeWoEandIV();
//       
//       double total_iv = 0.0;
//       for (const auto& bin : bins) {
//         total_iv += bin.iv;
//       }
//       
//       if (std::abs(total_iv - prev_total_iv) < convergence_threshold) {
//         monotonic = true;
//       }
//       
//       prev_total_iv = total_iv;
//       iterations++;
//     }
//     
//     converged = monotonic;
//     iterations_run = iterations;
//   } else {
//     // No optimization needed, use original categories as bins
//     bins.clear();
//     for (const auto& [cat, count] : category_counts) {
//       double good = category_good[cat];
//       double bad = category_bad[cat];
//       double rate_good = good / total_good;
//       double rate_bad = bad / total_bad;
//       
//       if (rate_good == 0) {
//         rate_good = std::numeric_limits<double>::min();
//       }
//       if (rate_bad == 0) {
//         rate_bad = std::numeric_limits<double>::min();
//       }
//       
//       double woe = std::log(rate_good / rate_bad);
//       double iv = (rate_good - rate_bad) * woe;
//       
//       bins.push_back({{cat}, good, bad, woe, iv});
//     }
//   }
//   
//   // Prepare output data
//   std::vector<std::string> bin_names;
//   std::vector<double> woe_values;
//   std::vector<double> iv_values;
//   std::vector<int> count_values;
//   std::vector<int> count_pos_values;
//   std::vector<int> count_neg_values;
//   
//   for (auto& bin : bins) {
//     // Sort categories within bin
//     std::vector<std::string> sorted_categories = bin.categories;
//     std::sort(sorted_categories.begin(), sorted_categories.end());
//     
//     std::string bin_name = sorted_categories[0];
//     for (size_t i = 1; i < sorted_categories.size(); ++i) {
//       bin_name += bin_separator + sorted_categories[i];
//     }
//     bin_names.push_back(bin_name);
//     woe_values.push_back(bin.woe);
//     iv_values.push_back(bin.iv);
//     count_values.push_back(static_cast<int>(bin.good_count + bin.bad_count));
//     count_pos_values.push_back(static_cast<int>(bin.good_count));
//     count_neg_values.push_back(static_cast<int>(bin.bad_count));
//   }
//   
//   return List::create(
//     Named("bin") = bin_names,
//     Named("woe") = woe_values,
//     Named("iv") = iv_values,
//     Named("count") = count_values,
//     Named("count_pos") = count_pos_values,
//     Named("count_neg") = count_neg_values,
//     Named("converged") = converged,
//     Named("iterations") = iterations_run
//   );
// }
// 
// //' @title
// //' Optimal Binning for Categorical Variables using Monotonic Optimal Binning (MOB)
// //'
// //' @description
// //' This function performs optimal binning for categorical variables using the Monotonic Optimal Binning (MOB) approach.
// //'
// //' @param target An integer vector of binary target values (0 or 1).
// //' @param feature A character vector of categorical feature values.
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff Minimum proportion of observations in a bin (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins (default: 20).
// //' @param bin_separator Separator used for merging category names (default: "%;%").
// //' @param convergence_threshold Convergence threshold for the algorithm (default: 1e-6).
// //' @param max_iterations Maximum number of iterations for the algorithm (default: 1000).
// //'
// //' @return A list containing the following elements:
// //' \itemize{
// //'   \item bin: A character vector of bin names (merged categories)
// //'   \item woe: A numeric vector of Weight of Evidence (WoE) values for each bin
// //'   \item iv: A numeric vector of Information Value (IV) for each bin
// //'   \item count: An integer vector of total counts for each bin
// //'   \item count_pos: An integer vector of positive target counts for each bin
// //'   \item count_neg: An integer vector of negative target counts for each bin
// //'   \item converged: A logical value indicating whether the algorithm converged
// //'   \item iterations: An integer value indicating the number of iterations run
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
// //' result <- optimal_binning_categorical_mob(target, feature)
// //'
// //' # View results
// //' print(result)
// //' }
// //'
// //' @details
// //' This algorithm performs optimal binning for categorical variables using the Monotonic Optimal Binning (MOB) approach.
// //' The process aims to maximize the Information Value (IV) while maintaining monotonicity in the Weight of Evidence (WoE) across bins.
// //'
// //' The algorithm works as follows:
// //'
// //' \enumerate{
// //'   \item **Category Statistics Calculation**:
// //'         For each category, calculate the total count, count of positive instances, and count of negative instances.
// //'
// //'   \item **Initial Binning**:
// //'         Categories are sorted based on their initial Weight of Evidence (WoE).
// //'
// //'   \item **Monotonicity Enforcement**:
// //'         The algorithm enforces decreasing monotonicity of WoE across bins.
// //'         If this condition is violated, adjacent bins are merged.
// //'
// //'   \item **Bin Limiting**:
// //'         The number of bins is limited to the specified `max_bins`.
// //'         When merging is necessary, the algorithm chooses the two adjacent bins with the smallest WoE difference.
// //'
// //'   \item **Information Value (IV) Computation**:
// //'         For each bin, the IV is calculated, and the total IV is computed.
// //' }
// //'
// //' The MOB approach ensures that the resulting bins have monotonic WoE values, which is often desirable in credit scoring and risk modeling applications.
// //'
// //' @references
// //' \itemize{
// //'    \item Belotti, T., Crook, J. (2009). Credit Scoring with Macroeconomic Variables Using Survival Analysis.
// //'          *Journal of the Operational Research Society*, 60(12), 1699-1707.
// //'    \item Mironchyk, P., Tchistiakov, V. (2017). Monotone optimal binning algorithm for credit risk modeling.
// //'          *arXiv preprint* arXiv:1711.05095.
// //' }
// //'
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_categorical_mob(Rcpp::IntegerVector target,
//                                           Rcpp::StringVector feature,
//                                           int min_bins = 3,
//                                           int max_bins = 5,
//                                           double bin_cutoff = 0.05,
//                                           int max_n_prebins = 20,
//                                           std::string bin_separator = "%;%",
//                                           double convergence_threshold = 1e-6,
//                                           int max_iterations = 1000) {
//  if (target.size() != feature.size()) {
//    Rcpp::stop("Target and feature vectors must have the same length.");
//  }
//  
//  std::vector<bool> target_vec;
//  target_vec.reserve(target.size());
//  for (int i = 0; i < target.size(); ++i) {
//    if (target[i] != 0 && target[i] != 1) {
//      Rcpp::stop("Target vector must be binary (0 and 1).");
//    }
//    target_vec.push_back(target[i] == 1);
//  }
//  
//  std::vector<std::string> feature_vec;
//  feature_vec.reserve(feature.size());
//  for (int i = 0; i < feature.size(); ++i) {
//    feature_vec.push_back(feature[i] == NA_STRING ? "NA" : Rcpp::as<std::string>(feature[i]));
//  }
//  
//  // Ensure max_bins is not greater than the number of unique categories
//  std::unordered_set<std::string> unique_categories(feature_vec.begin(), feature_vec.end());
//  max_bins = std::min(max_bins, static_cast<int>(unique_categories.size()));
//  
//  // Adjust min_bins if necessary
//  min_bins = std::min(min_bins, max_bins);
//  
//  try {
//    OptimalBinningCategoricalMOB mob(std::move(feature_vec), std::move(target_vec),
//                                     min_bins, max_bins, bin_cutoff, max_n_prebins,
//                                     bin_separator, convergence_threshold, max_iterations);
//    return mob.fit();
//  } catch (const std::exception& e) {
//    Rcpp::stop("Error in OptimalBinningCategoricalMOB: " + std::string(e.what()));
//  }
// }
