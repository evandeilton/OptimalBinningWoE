// [[Rcpp::plugins(openmp)]]

#include <Rcpp.h>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <algorithm>
#include <limits>
#include <cmath>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <queue>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

// ---------------------------------------------------------------------------------------------- //
// CATEGORICAL VARIABLES
// ---------------------------------------------------------------------------------------------- //

//' Performs optimal binning of a categorical variable for Weight of Evidence (WoE) and Information Value (IV)
 //'
 //' This function processes a categorical variable by grouping rare categories, ordering them by event rate, and generating bins to maximize WoE monotonicity. It also calculates WoE and IV for the generated bins.
 //'
 //' @param target Integer vector representing the binary target variable, where 1 indicates a positive event (e.g., default) and 0 indicates a negative event (e.g., non-default).
 //' @param feature Character vector representing the categorical variable to be binned.
 //' @param cat_cutoff (Optional) Frequency cutoff value, below which categories are grouped into "Other". Default is 0.05.
 //' @param min_bins (Optional) Minimum number of bins to generate. Default is 2.
 //' @param max_bins (Optional) Maximum number of bins to generate. Default is 5.
 //'
 //' @return A list with the following elements:
 //' \itemize{
 //'   \item \code{feature_woe}: Numeric vector with the WoE assigned to each instance of the processed categorical variable.
 //'   \item \code{bin}: DataFrame with the generated bins, containing the following fields:
 //'     \itemize{
 //'       \item \code{bin}: Names of the categories grouped into each bin.
 //'       \item \code{woe}: Weight of Evidence (WoE) for each bin.
 //'       \item \code{iv}: Information Value (IV) for each bin.
 //'       \item \code{count}: Total number of observations in each bin.
 //'       \item \code{count_pos}: Count of positive events in each bin.
 //'       \item \code{count_neg}: Count of negative events in each bin.
 //'     }
 //'   \item \code{woe}: Numeric vector with the WoE for each bin.
 //'   \item \code{iv}: Total Information Value (IV) calculated for the variable.
 //'   \item \code{pos}: Vector with the count of positive events in each bin.
 //'   \item \code{neg}: Vector with the count of negative events in each bin.
 //' }
 //'
 // [[Rcpp::export]]
 Rcpp::List OptimalBinningCategoricalMIP(Rcpp::IntegerVector target, Rcpp::CharacterVector feature, 
                                         double cat_cutoff = 0.05, int min_bins = 2, int max_bins = 5) {
   int N = target.size();
   if (feature.size() != N) {
     Rcpp::stop("Length of target and feature must be the same.");
   }
   
   // Validate input parameters
   if (cat_cutoff <= 0 || cat_cutoff >= 1) {
     Rcpp::stop("cat_cutoff must be between 0 and 1.");
   }
   if (min_bins <= 0 || max_bins <= 0 || min_bins > max_bins) {
     Rcpp::stop("Invalid min_bins or max_bins. Ensure 0 < min_bins <= max_bins.");
   }
   
   // Validate target values
   for (int i = 0; i < N; ++i) {
     if (target[i] != 0 && target[i] != 1) {
       Rcpp::stop("Target must contain only 0s and 1s.");
     }
   }
   
   // Compute frequencies of each category
   std::map<std::string, int> category_counts;
   for (int i = 0; i < N; ++i) {
     if (feature[i] == NA_STRING) {
       Rcpp::stop("NA values are not allowed in the feature vector.");
     }
     std::string cat = Rcpp::as<std::string>(feature[i]);
     category_counts[cat]++;
   }
   
   // Identify rare categories
   std::vector<std::string> rare_categories;
   for (const auto& pair : category_counts) {
     if ((double)pair.second / N < cat_cutoff) {
       rare_categories.push_back(pair.first);
     }
   }
   
   // Group rare categories
   std::string grouped_rare = "";
   if (!rare_categories.empty()) {
     std::sort(rare_categories.begin(), rare_categories.end());
     grouped_rare = rare_categories[0];
     for (size_t i = 1; i < rare_categories.size(); ++i) {
       grouped_rare += "+" + rare_categories[i];
     }
   }
   
   // Map categories, combining rare categories
   Rcpp::CharacterVector feature_processed(N);
   for (int i = 0; i < N; ++i) {
     std::string cat = Rcpp::as<std::string>(feature[i]);
     if (std::find(rare_categories.begin(), rare_categories.end(), cat) != rare_categories.end()) {
       feature_processed[i] = grouped_rare;
     } else {
       feature_processed[i] = cat;
     }
   }
   
   // Recompute category counts and frequencies
   category_counts.clear();
   for (int i = 0; i < N; ++i) {
     category_counts[Rcpp::as<std::string>(feature_processed[i])]++;
   }
   
   // Compute event rates per category
   std::map<std::string, int> category_positive_count;
   std::map<std::string, int> category_negative_count;
   for (const auto& it : category_counts) {
     category_positive_count[it.first] = 0;
     category_negative_count[it.first] = 0;
   }
   for (int i = 0; i < N; ++i) {
     std::string cat = Rcpp::as<std::string>(feature_processed[i]);
     int tgt = target[i];
     if (tgt == 1) {
       category_positive_count[cat]++;
     } else {
       category_negative_count[cat]++;
     }
   }
   
   // Compute event rates
   std::vector<std::pair<std::string, double>> categories_event_rates;
   for (const auto& it : category_counts) {
     int pos = category_positive_count[it.first];
     int neg = category_negative_count[it.first];
     double event_rate = (double)pos / (pos + neg);
     categories_event_rates.push_back(std::make_pair(it.first, event_rate));
   }
   
   // Sort categories by event rate
   std::sort(categories_event_rates.begin(), categories_event_rates.end(),
             [](const std::pair<std::string, double>& a, const std::pair<std::string, double>& b) {
               return a.second < b.second;
             });
   
   // Initialize bins with each category as a bin
   struct Bin {
     std::vector<std::string> categories;
     int pos_count;
     int neg_count;
     double event_rate;
   };
   std::vector<Bin> bins;
   for (const auto& category : categories_event_rates) {
     Bin bin;
     bin.categories.push_back(category.first);
     bin.pos_count = category_positive_count[category.first];
     bin.neg_count = category_negative_count[category.first];
     bin.event_rate = category.second;
     bins.push_back(bin);
   }
   
   // Merge bins to satisfy max_bins
   while (bins.size() > (size_t)max_bins) {
     double min_diff = std::numeric_limits<double>::max();
     size_t merge_idx = 0;
     for (size_t i = 0; i < bins.size() - 1; ++i) {
       double diff = bins[i + 1].event_rate - bins[i].event_rate;
       if (diff < min_diff) {
         min_diff = diff;
         merge_idx = i;
       }
     }
     // Merge bins
     bins[merge_idx].categories.insert(bins[merge_idx].categories.end(),
                                       bins[merge_idx + 1].categories.begin(),
                                       bins[merge_idx + 1].categories.end());
     bins[merge_idx].pos_count += bins[merge_idx + 1].pos_count;
     bins[merge_idx].neg_count += bins[merge_idx + 1].neg_count;
     bins[merge_idx].event_rate = (double)bins[merge_idx].pos_count /
       (bins[merge_idx].pos_count + bins[merge_idx].neg_count);
     bins.erase(bins.begin() + merge_idx + 1);
   }
   
   // Ensure monotonicity while respecting min_bins
   bool monotonic = false;
   while (!monotonic && bins.size() > (size_t)min_bins) {
     monotonic = true;
     for (size_t i = 0; i < bins.size() - 1; ++i) {
       if (bins[i].event_rate > bins[i + 1].event_rate) {
         bins[i].categories.insert(bins[i].categories.end(),
                                   bins[i + 1].categories.begin(),
                                   bins[i + 1].categories.end());
         bins[i].pos_count += bins[i + 1].pos_count;
         bins[i].neg_count += bins[i + 1].neg_count;
         bins[i].event_rate = (double)bins[i].pos_count /
           (bins[i].pos_count + bins[i].neg_count);
         bins.erase(bins.begin() + i + 1);
         monotonic = false;
         break;
       }
     }
   }
   
   // Force min_bins if necessary
   while (bins.size() < (size_t)min_bins && bins.size() < categories_event_rates.size()) {
     // Find the bin with the largest event rate difference to its neighbor
     size_t split_idx = 0;
     double max_diff = -1;
     for (size_t i = 0; i < bins.size() - 1; ++i) {
       double diff = std::abs(bins[i + 1].event_rate - bins[i].event_rate);
       if (diff > max_diff) {
         max_diff = diff;
         split_idx = i;
       }
     }
     
     // Split the bin
     Bin new_bin;
     size_t split_point = bins[split_idx].categories.size() / 2;
     new_bin.categories.insert(new_bin.categories.end(),
                               bins[split_idx].categories.begin() + split_point,
                               bins[split_idx].categories.end());
     bins[split_idx].categories.erase(bins[split_idx].categories.begin() + split_point,
                                      bins[split_idx].categories.end());
     
     // Recalculate counts and event rates
     new_bin.pos_count = 0;
     new_bin.neg_count = 0;
     bins[split_idx].pos_count = 0;
     bins[split_idx].neg_count = 0;
     
     for (const auto& cat : bins[split_idx].categories) {
       bins[split_idx].pos_count += category_positive_count[cat];
       bins[split_idx].neg_count += category_negative_count[cat];
     }
     for (const auto& cat : new_bin.categories) {
       new_bin.pos_count += category_positive_count[cat];
       new_bin.neg_count += category_negative_count[cat];
     }
     
     bins[split_idx].event_rate = (double)bins[split_idx].pos_count /
       (bins[split_idx].pos_count + bins[split_idx].neg_count);
     new_bin.event_rate = (double)new_bin.pos_count /
       (new_bin.pos_count + new_bin.neg_count);
     
     bins.insert(bins.begin() + split_idx + 1, new_bin);
   }
   
   // Compute WoE and IV
   int total_pos = 0;
   int total_neg = 0;
   for (const auto& bin : bins) {
     total_pos += bin.pos_count;
     total_neg += bin.neg_count;
   }
   std::vector<double> woe(bins.size());
   std::vector<double> iv_bin(bins.size());
   double total_iv = 0.0;
   for (size_t i = 0; i < bins.size(); ++i) {
     double dist_pos = (double)bins[i].pos_count / total_pos;
     double dist_neg = (double)bins[i].neg_count / total_neg;
     if (dist_pos == 0) dist_pos = 1e-10;
     if (dist_neg == 0) dist_neg = 1e-10;
     woe[i] = std::log(dist_pos / dist_neg);
     iv_bin[i] = (dist_pos - dist_neg) * woe[i];
     total_iv += iv_bin[i];
   }
   
   // Map categories to WoE
   std::map<std::string, double> category_woe_map;
   for (size_t i = 0; i < bins.size(); ++i) {
     for (const auto& cat : bins[i].categories) {
       category_woe_map[cat] = woe[i];
     }
   }
   Rcpp::NumericVector feature_woe(N);
   for (int i = 0; i < N; ++i) {
     feature_woe[i] = category_woe_map[Rcpp::as<std::string>(feature_processed[i])];
   }
   
   // Prepare bin output
   Rcpp::CharacterVector bin_names(bins.size());
   Rcpp::IntegerVector count(bins.size());
   Rcpp::IntegerVector pos(bins.size());
   Rcpp::IntegerVector neg(bins.size());
   for (size_t i = 0; i < bins.size(); ++i) {
     bin_names[i] = bins[i].categories[0];
     for (size_t j = 1; j < bins[i].categories.size(); ++j) {
       bin_names[i] += "+" + bins[i].categories[j];
     }
     count[i] = bins[i].pos_count + bins[i].neg_count;
     pos[i] = bins[i].pos_count;
     neg[i] = bins[i].neg_count;
   }
   
   // Create List for bins
   Rcpp::List bin_lst = Rcpp::List::create(
     Rcpp::Named("bin") = bin_names,
     Rcpp::Named("woe") = woe,
     Rcpp::Named("iv") = iv_bin,
     Rcpp::Named("count") = count,
     Rcpp::Named("count_pos") = pos,
     Rcpp::Named("count_neg") = neg);
   
   // Create List for woe vector feature
   Rcpp::List woe_lst = Rcpp::List::create(
     Rcpp::Named("woefeature") = feature_woe
   );
   
   // Attrib class for compatibility with data.table in memory superfast tables
   bin_lst.attr("class") = Rcpp::CharacterVector::create("data.table", "data.frame");
   woe_lst.attr("class") = Rcpp::CharacterVector::create("data.table", "data.frame");
   
   // Return output
   Rcpp::List output_list = Rcpp::List::create(
     Rcpp::Named("woefeature") = woe_lst,
     Rcpp::Named("woebin") = bin_lst
   );
   
   return output_list;
 }
 // Rcpp::List OptimalBinningCategoricalMIP(Rcpp::IntegerVector target, Rcpp::CharacterVector feature, double cat_cutoff = 0.05, int min_bins = 2, int max_bins = 5) {
 //   int N = target.size();
 //   if (feature.size() != N) {
 //     Rcpp::stop("Length of target and feature must be the same.");
 //   }
 //   
 //   // Parameters
 //   
 //   // Compute frequencies of each category
 //   std::map<std::string, int> category_counts;
 //   for (int i = 0; i < N; ++i) {
 //     std::string cat = Rcpp::as<std::string>(feature[i]);
 //     category_counts[cat]++;
 //   }
 //   
 //   // Handle rare categories
 //   std::map<std::string, double> category_freq;
 //   for (auto& it : category_counts) {
 //     category_freq[it.first] = (double)it.second / N;
 //   }
 //   std::set<std::string> rare_categories;
 //   for (auto& it : category_freq) {
 //     if (it.second < cat_cutoff) {
 //       rare_categories.insert(it.first);
 //     }
 //   }
 //   
 //   // Map categories, combining rare categories into 'Other'
 //   std::vector<std::string> feature_processed(N);
 //   for (int i = 0; i < N; ++i) {
 //     std::string cat = Rcpp::as<std::string>(feature[i]);
 //     if (rare_categories.find(cat) != rare_categories.end()) {
 //       feature_processed[i] = "Other";
 //     } else {
 //       feature_processed[i] = cat;
 //     }
 //   }
 //   
 //   // Recompute category counts and frequencies
 //   category_counts.clear();
 //   for (int i = 0; i < N; ++i) {
 //     category_counts[feature_processed[i]]++;
 //   }
 //   
 //   // Compute event rates per category
 //   std::map<std::string, int> category_positive_count;
 //   std::map<std::string, int> category_negative_count;
 //   for (auto& it : category_counts) {
 //     category_positive_count[it.first] = 0;
 //     category_negative_count[it.first] = 0;
 //   }
 //   for (int i = 0; i < N; ++i) {
 //     std::string cat = feature_processed[i];
 //     int tgt = target[i];
 //     if (tgt == 1) {
 //       category_positive_count[cat]++;
 //     } else {
 //       category_negative_count[cat]++;
 //     }
 //   }
 //   
 //   // Compute event rates
 //   std::vector<std::pair<std::string, double>> categories_event_rates;
 //   for (auto& it : category_counts) {
 //     int pos = category_positive_count[it.first];
 //     int neg = category_negative_count[it.first];
 //     double event_rate = (double)pos / (pos + neg);
 //     categories_event_rates.push_back(std::make_pair(it.first, event_rate));
 //   }
 //   
 //   // Sort categories by event rate
 //   std::sort(categories_event_rates.begin(), categories_event_rates.end(),
 //             [](const std::pair<std::string, double>& a, const std::pair<std::string, double>& b) {
 //               return a.second < b.second;
 //             });
 //   
 //   // Initialize bins with each category as a bin
 //   struct Bin {
 //     std::vector<std::string> categories;
 //     int pos_count;
 //     int neg_count;
 //     double event_rate;
 //   };
 //   std::vector<Bin> bins;
 //   for (size_t i = 0; i < categories_event_rates.size(); ++i) {
 //     Bin bin;
 //     bin.categories.push_back(categories_event_rates[i].first);
 //     bin.pos_count = category_positive_count[categories_event_rates[i].first];
 //     bin.neg_count = category_negative_count[categories_event_rates[i].first];
 //     bin.event_rate = categories_event_rates[i].second;
 //     bins.push_back(bin);
 //   }
 //   
 //   // Merge bins to satisfy max_bins
 //   while (bins.size() > (size_t)max_bins) {
 //     double min_diff = std::numeric_limits<double>::max();
 //     size_t merge_idx = 0;
 //     for (size_t i = 0; i < bins.size() - 1; ++i) {
 //       double diff = bins[i + 1].event_rate - bins[i].event_rate;
 //       if (diff < min_diff) {
 //         min_diff = diff;
 //         merge_idx = i;
 //       }
 //     }
 //     // Merge bins
 //     bins[merge_idx].categories.insert(bins[merge_idx].categories.end(),
 //                                       bins[merge_idx + 1].categories.begin(),
 //                                       bins[merge_idx + 1].categories.end());
 //     bins[merge_idx].pos_count += bins[merge_idx + 1].pos_count;
 //     bins[merge_idx].neg_count += bins[merge_idx + 1].neg_count;
 //     bins[merge_idx].event_rate = (double)bins[merge_idx].pos_count /
 //       (bins[merge_idx].pos_count + bins[merge_idx].neg_count);
 //     bins.erase(bins.begin() + merge_idx + 1);
 //   }
 //   
 //   // Ensure monotonicity
 //   bool monotonic = false;
 //   while (!monotonic) {
 //     monotonic = true;
 //     for (size_t i = 0; i < bins.size() - 1; ++i) {
 //       if (bins[i].event_rate > bins[i + 1].event_rate) {
 //         bins[i].categories.insert(bins[i].categories.end(),
 //                                   bins[i + 1].categories.begin(),
 //                                   bins[i + 1].categories.end());
 //         bins[i].pos_count += bins[i + 1].pos_count;
 //         bins[i].neg_count += bins[i + 1].neg_count;
 //         bins[i].event_rate = (double)bins[i].pos_count /
 //           (bins[i].pos_count + bins[i].neg_count);
 //         bins.erase(bins.begin() + i + 1);
 //         monotonic = false;
 //         break;
 //       }
 //     }
 //     if (bins.size() < (size_t)min_bins) {
 //       break;
 //     }
 //   }
 //   
 //   // Compute WoE and IV
 //   int total_pos = 0;
 //   int total_neg = 0;
 //   for (auto& bin : bins) {
 //     total_pos += bin.pos_count;
 //     total_neg += bin.neg_count;
 //   }
 //   std::vector<double> woe(bins.size());
 //   std::vector<double> iv_bin(bins.size());
 //   double total_iv = 0.0;
 //   for (size_t i = 0; i < bins.size(); ++i) {
 //     double dist_pos = (double)bins[i].pos_count / total_pos;
 //     double dist_neg = (double)bins[i].neg_count / total_neg;
 //     if (dist_pos == 0) dist_pos = 1e-10;
 //     if (dist_neg == 0) dist_neg = 1e-10;
 //     woe[i] = log(dist_pos / dist_neg);
 //     iv_bin[i] = (dist_pos - dist_neg) * woe[i];
 //     total_iv += iv_bin[i];
 //   }
 //   
 //   // Map categories to WoE
 //   std::map<std::string, double> category_woe_map;
 //   for (size_t i = 0; i < bins.size(); ++i) {
 //     for (auto& cat : bins[i].categories) {
 //       category_woe_map[cat] = woe[i];
 //     }
 //   }
 //   NumericVector feature_woe(N);
 //   for (int i = 0; i < N; ++i) {
 //     feature_woe[i] = category_woe_map[feature_processed[i]];
 //   }
 //   
 //   // Prepare bin output
 //   std::vector<std::string> bin_names(bins.size());
 //   std::vector<int> count(bins.size());
 //   std::vector<int> pos(bins.size());
 //   std::vector<int> neg(bins.size());
 //   for (size_t i = 0; i < bins.size(); ++i) {
 //     bin_names[i] = bins[i].categories[0];
 //     for (size_t j = 1; j < bins[i].categories.size(); ++j) {
 //       bin_names[i] += "+" + bins[i].categories[j];
 //     }
 //     count[i] = bins[i].pos_count + bins[i].neg_count;
 //     pos[i] = bins[i].pos_count;
 //     neg[i] = bins[i].neg_count;
 //   }
 //   
 //   // Create List for bins
 //   List bin_lst = List::create(
 //     Named("bin") = bin_names,
 //     Named("woe") = woe,
 //     Named("iv") = iv_bin,
 //     Named("count") = count,
 //     Named("count_pos") = pos,
 //     Named("count_neg") = neg);
 //   
 //   // Create List for woe vector feature
 //   List woe_lst = List::create(
 //     Named("woefeature") = feature_woe
 //   );
 //   
 //   // Attrib class for compatibility with data.table in memory superfast tables
 //   bin_lst.attr("class") = CharacterVector::create("data.table", "data.frame");
 //   woe_lst.attr("class") = CharacterVector::create("data.table", "data.frame");
 //   
 //   // Return output
 //   List output_list = List::create(
 //     Named("woefeature") = woe_lst,
 //     Named("woebin") = bin_lst
 //   // Named("woe") = woe,
 //   // Named("iv") = total_iv,
 //   // Named("pos") = pos,
 //   // Named("neg") = neg
 //   );
 //   
 //   return output_list;
 // }
 
 
//' Performs optimal binning of a categorical variable for Weight of Evidence (WoE) and Information Value (IV) using Monotonic Optimal Binning (MOB)
//'
//' This function processes a categorical variable by grouping rare categories, ordering them by event rate, and generating bins to maximize WoE monotonicity. It also applies constraints to ensure that bins have a minimum number of bad events (min_bads) and calculates WoE and IV for the generated bins.
//'
//' @param target Integer vector representing the binary target variable, where 1 indicates a positive event (e.g., default) and 0 indicates a negative event (e.g., non-default).
//' @param feature Character vector representing the categorical variable to be binned.
//' @param min_bins (Optional) Minimum number of bins to generate. Default is 2.
//' @param max_bins (Optional) Maximum number of bins to generate. Default is 7.
//' @param cat_cutoff (Optional) Frequency cutoff value, below which categories are grouped into "Other". Default is 0.05.
//' @param min_bads (Optional) Minimum proportion of bad events that a bin must contain. Default is 0.05.
//' @param max_n_prebins (Optional) Maximum number of pre-bins to consider before final binning. Default is 20.
//'
//' @return A list with the following elements:
//' \itemize{
//'   \item \code{feature_woe}: Numeric vector with the WoE assigned to each instance of the processed categorical variable.
//'   \item \code{bin}: DataFrame with the generated bins, containing the following fields:
//'     \itemize{
//'       \item \code{bin}: Names of the categories grouped into each bin.
//'       \item \code{woe}: Weight of Evidence (WoE) for each bin.
//'       \item \code{iv}: Information Value (IV) for each bin.
//'       \item \code{count}: Total number of observations in each bin.
//'       \item \code{count_pos}: Count of positive events in each bin.
//'       \item \code{count_neg}: Count of negative events in each bin.
//'     }
//'   \item \code{woe}: Numeric vector with the WoE for each bin.
//'   \item \code{iv}: Total Information Value (IV) calculated for the variable.
//'   \item \code{pos}: Vector with the count of positive events in each bin.
//'   \item \code{neg}: Vector with the count of negative events in each bin.
//' }
//'
//'
// [[Rcpp::export]]
Rcpp::List OptimalBinningCategoricalMOB(Rcpp::IntegerVector target, Rcpp::CharacterVector feature, int min_bins = 2, int max_bins = 7, double cat_cutoff = 0.05, double min_bads = 0.05, int max_n_prebins = 20) {
 int N = target.size();
 if (feature.size() != N) {
   Rcpp::stop("Length of target and feature must be the same.");
 }

 // Parameters

 // Compute frequencies of each category
 std::map<std::string, int> category_counts;
 for (int i = 0; i < N; ++i) {
   std::string cat = Rcpp::as<std::string>(feature[i]);
   category_counts[cat]++;
 }

 // Handle rare categories
 std::map<std::string, double> category_freq;
 for (auto& it : category_counts) {
   category_freq[it.first] = (double)it.second / N;
 }
 std::set<std::string> rare_categories;
 for (auto& it : category_freq) {
   if (it.second < cat_cutoff) {
     rare_categories.insert(it.first);
   }
 }

 // Map categories, combining rare categories into 'Other'
 std::vector<std::string> feature_processed(N);
 for (int i = 0; i < N; ++i) {
   std::string cat = Rcpp::as<std::string>(feature[i]);
   if (rare_categories.find(cat) != rare_categories.end()) {
     feature_processed[i] = "Other";
   } else {
     feature_processed[i] = cat;
   }
 }

 // Recompute category counts and frequencies
 category_counts.clear();
 for (int i = 0; i < N; ++i) {
   category_counts[feature_processed[i]]++;
 }

 // Compute event rates per category
 std::map<std::string, int> category_positive_count;
 std::map<std::string, int> category_negative_count;
 for (auto& it : category_counts) {
   category_positive_count[it.first] = 0;
   category_negative_count[it.first] = 0;
 }
 for (int i = 0; i < N; ++i) {
   std::string cat = feature_processed[i];
   int tgt = target[i];
   if (tgt == 1) {
     category_positive_count[cat]++;
   } else {
     category_negative_count[cat]++;
   }
 }

 // Compute event rates
 std::vector<std::pair<std::string, double>> categories_event_rates;
 for (auto& it : category_counts) {
   int pos = category_positive_count[it.first];
   int neg = category_negative_count[it.first];
   double event_rate = (double)pos / (pos + neg);
   categories_event_rates.push_back(std::make_pair(it.first, event_rate));
 }

 // Sort categories by event rate
 std::sort(categories_event_rates.begin(), categories_event_rates.end(),
           [](const std::pair<std::string, double>& a, const std::pair<std::string, double>& b) {
             return a.second < b.second;
           });

 // Initialize bins with each category as a bin
 struct Bin {
   std::vector<std::string> categories;
   int pos_count;
   int neg_count;
   double event_rate;
 };
 std::vector<Bin> bins;
 for (size_t i = 0; i < categories_event_rates.size(); ++i) {
   Bin bin;
   bin.categories.push_back(categories_event_rates[i].first);
   bin.pos_count = category_positive_count[categories_event_rates[i].first];
   bin.neg_count = category_negative_count[categories_event_rates[i].first];
   bin.event_rate = categories_event_rates[i].second;
   bins.push_back(bin);
 }

 // Function to check if bins satisfy monotonicity
 auto is_monotonic = [](const std::vector<Bin>& bins) {
   bool increasing = true, decreasing = true;
   for (size_t i = 1; i < bins.size(); ++i) {
     if (bins[i].event_rate < bins[i - 1].event_rate) {
       increasing = false;
     }
     if (bins[i].event_rate > bins[i - 1].event_rate) {
       decreasing = false;
     }
   }
   return increasing || decreasing;
 };

 // Merge bins to satisfy min_bads constraint
 bool merged = true;
 while (merged) {
   merged = false;
   for (size_t i = 0; i < bins.size(); ++i) {
     double bad_rate = (double)bins[i].pos_count / N;
     if (bad_rate < min_bads && bins.size() > (size_t)min_bins) {
       // Merge with neighbor bin with closest event rate
       size_t merge_idx = (i == 0) ? i + 1 : i - 1;
       bins[merge_idx].categories.insert(bins[merge_idx].categories.end(),
                                         bins[i].categories.begin(),
                                         bins[i].categories.end());
       bins[merge_idx].pos_count += bins[i].pos_count;
       bins[merge_idx].neg_count += bins[i].neg_count;
       bins[merge_idx].event_rate = (double)bins[merge_idx].pos_count /
         (bins[merge_idx].pos_count + bins[merge_idx].neg_count);
       bins.erase(bins.begin() + i);
       merged = true;
       break;
     }
   }
 }

 // Merge bins to reduce to max_bins
 while (bins.size() > (size_t)max_bins) {
   double min_diff = std::numeric_limits<double>::max();
   size_t merge_idx = 0;
   for (size_t i = 0; i < bins.size() - 1; ++i) {
     double diff = bins[i + 1].event_rate - bins[i].event_rate;
     if (diff < min_diff) {
       min_diff = diff;
       merge_idx = i;
     }
   }
   // Merge bins
   bins[merge_idx].categories.insert(bins[merge_idx].categories.end(),
                                     bins[merge_idx + 1].categories.begin(),
                                     bins[merge_idx + 1].categories.end());
   bins[merge_idx].pos_count += bins[merge_idx + 1].pos_count;
   bins[merge_idx].neg_count += bins[merge_idx + 1].neg_count;
   bins[merge_idx].event_rate = (double)bins[merge_idx].pos_count /
     (bins[merge_idx].pos_count + bins[merge_idx].neg_count);
   bins.erase(bins.begin() + merge_idx + 1);
 }

 // Ensure monotonicity
 while (!is_monotonic(bins) && bins.size() > (size_t)min_bins) {
   // Find the bin to merge based on event rate deviation
   double min_deviation = std::numeric_limits<double>::max();
   size_t merge_idx = 0;
   for (size_t i = 0; i < bins.size() - 1; ++i) {
     if (bins[i].event_rate > bins[i + 1].event_rate) {
       double deviation = bins[i].event_rate - bins[i + 1].event_rate;
       if (deviation < min_deviation) {
         min_deviation = deviation;
         merge_idx = i;
       }
     }
   }
   // Merge bins
   bins[merge_idx].categories.insert(bins[merge_idx].categories.end(),
                                     bins[merge_idx + 1].categories.begin(),
                                     bins[merge_idx + 1].categories.end());
   bins[merge_idx].pos_count += bins[merge_idx + 1].pos_count;
   bins[merge_idx].neg_count += bins[merge_idx + 1].neg_count;
   bins[merge_idx].event_rate = (double)bins[merge_idx].pos_count /
     (bins[merge_idx].pos_count + bins[merge_idx].neg_count);
   bins.erase(bins.begin() + merge_idx + 1);
 }

 // Compute WoE and IV
 int total_pos = 0;
 int total_neg = 0;
 for (auto& bin : bins) {
   total_pos += bin.pos_count;
   total_neg += bin.neg_count;
 }
 std::vector<double> woe(bins.size());
 std::vector<double> iv_bin(bins.size());
 double total_iv = 0.0;
 for (size_t i = 0; i < bins.size(); ++i) {
   double dist_pos = (double)bins[i].pos_count / total_pos;
   double dist_neg = (double)bins[i].neg_count / total_neg;
   if (dist_pos == 0) dist_pos = 1e-10;
   if (dist_neg == 0) dist_neg = 1e-10;
   woe[i] = log(dist_pos / dist_neg);
   iv_bin[i] = (dist_pos - dist_neg) * woe[i];
   total_iv += iv_bin[i];
 }

 // Map categories to WoE
 std::map<std::string, double> category_woe_map;
 for (size_t i = 0; i < bins.size(); ++i) {
   for (auto& cat : bins[i].categories) {
     category_woe_map[cat] = woe[i];
   }
 }
 NumericVector feature_woe(N);
 for (int i = 0; i < N; ++i) {
   feature_woe[i] = category_woe_map[feature_processed[i]];
 }

 // Prepare bin output
 std::vector<std::string> bin_names(bins.size());
 std::vector<int> count(bins.size());
 std::vector<int> pos(bins.size());
 std::vector<int> neg(bins.size());
 for (size_t i = 0; i < bins.size(); ++i) {
   bin_names[i] = bins[i].categories[0];
   for (size_t j = 1; j < bins[i].categories.size(); ++j) {
     bin_names[i] += "+" + bins[i].categories[j];
   }
   count[i] = bins[i].pos_count + bins[i].neg_count;
   pos[i] = bins[i].pos_count;
   neg[i] = bins[i].neg_count;
 }

 // Create List for bins
 List bin_lst = List::create(
   Named("bin") = bin_names,
   Named("woe") = woe,
   Named("iv") = iv_bin,
   Named("count") = count,
   Named("count_pos") = pos,
   Named("count_neg") = neg);

 // Create List for woe vector feature
 List woe_lst = List::create(
   Named("woefeature") = feature_woe
 );

 // Attrib class for compatibility with data.table in memory superfast tables
 bin_lst.attr("class") = CharacterVector::create("data.table", "data.frame");
 woe_lst.attr("class") = CharacterVector::create("data.table", "data.frame");

 // Return output
 List output_list = List::create(
   Named("woefeature") = woe_lst,
   Named("woebin") = bin_lst
 // Named("woe") = woe,
 // Named("iv") = total_iv,
 // Named("pos") = pos,
 // Named("neg") = neg
 );
 return output_list;
}
 
 //' Performs optimal binning of a categorical variable for Weight of Evidence (WoE) and Information Value (IV) using the ChiMerge algorithm
 //'
 //' This function processes a categorical variable by grouping rare categories, ordering them by event rate, and applying the ChiMerge algorithm to merge categories into bins based on the chi-square test of independence. It also ensures monotonicity of event rates and calculates WoE and IV for the generated bins.
 //'
 //' @param target Integer vector representing the binary target variable, where 1 indicates a positive event (e.g., default) and 0 indicates a negative event (e.g., non-default).
 //' @param feature Character vector representing the categorical variable to be binned.
 //' @param min_bins (Optional) Minimum number of bins to generate. Default is 2.
 //' @param max_bins (Optional) Maximum number of bins to generate. Default is 7.
 //' @param pvalue_threshold (Optional) P-value threshold for the chi-square test used to determine whether to merge bins. Default is 0.05.
 //' @param cat_cutoff (Optional) Frequency cutoff value, below which categories are grouped into "Other". Default is 0.05.
 //' @param min_bads (Optional) Minimum proportion of bad events that a bin must contain. Default is 0.05.
 //' @param max_n_prebins (Optional) Maximum number of pre-bins to consider before final binning. Default is 20.
 //'
 //' @return A list with the following elements:
 //' \itemize{
 //'   \item \code{feature_woe}: Numeric vector with the WoE assigned to each instance of the processed categorical variable.
 //'   \item \code{bin}: DataFrame with the generated bins, containing the following fields:
 //'     \itemize{
 //'       \item \code{bin}: Names of the categories grouped into each bin.
 //'       \item \code{woe}: Weight of Evidence (WoE) for each bin.
 //'       \item \code{iv}: Information Value (IV) for each bin.
 //'       \item \code{count}: Total number of observations in each bin.
 //'       \item \code{count_pos}: Count of positive events in each bin.
 //'       \item \code{count_neg}: Count of negative events in each bin.
 //'     }
 //'   \item \code{woe}: Numeric vector with the WoE for each bin.
 //'   \item \code{iv}: Total Information Value (IV) calculated for the variable.
 //'   \item \code{pos}: Vector with the count of positive events in each bin.
 //'   \item \code{neg}: Vector with the count of negative events in each bin.
 //' }
 //'
 // [[Rcpp::export]]
 Rcpp::List OptimalBinningCategoricalChiMerge(Rcpp::IntegerVector target, Rcpp::CharacterVector feature, int min_bins = 2, int max_bins = 7, double pvalue_threshold = 0.05, double cat_cutoff = 0.05, double min_bads = 0.05, int max_n_prebins = 20) {
   int N = target.size();
   if (feature.size() != N) {
     Rcpp::stop("Length of target and feature must be the same.");
   }
   
   // Parameters
   
   // Compute frequencies of each category
   std::map<std::string, int> category_counts;
   for (int i = 0; i < N; ++i) {
     std::string cat = Rcpp::as<std::string>(feature[i]);
     category_counts[cat]++;
   }
   
   // Handle rare categories
   std::map<std::string, double> category_freq;
   for (auto& it : category_counts) {
     category_freq[it.first] = (double)it.second / N;
   }
   std::set<std::string> rare_categories;
   for (auto& it : category_freq) {
     if (it.second < cat_cutoff) {
       rare_categories.insert(it.first);
     }
   }
   
   // Map categories, combining rare categories into 'Other'
   std::vector<std::string> feature_processed(N);
   for (int i = 0; i < N; ++i) {
     std::string cat = Rcpp::as<std::string>(feature[i]);
     if (rare_categories.find(cat) != rare_categories.end()) {
       feature_processed[i] = "Other";
     } else {
       feature_processed[i] = cat;
     }
   }
   
   // Recompute category counts and frequencies
   category_counts.clear();
   for (int i = 0; i < N; ++i) {
     category_counts[feature_processed[i]]++;
   }
   
   // Compute positive and negative counts per category
   std::map<std::string, int> category_positive_count;
   std::map<std::string, int> category_negative_count;
   for (auto& it : category_counts) {
     category_positive_count[it.first] = 0;
     category_negative_count[it.first] = 0;
   }
   for (int i = 0; i < N; ++i) {
     std::string cat = feature_processed[i];
     int tgt = target[i];
     if (tgt == 1) {
       category_positive_count[cat]++;
     } else {
       category_negative_count[cat]++;
     }
   }
   
   // Initialize bins with each category as its own bin
   struct Bin {
     std::vector<std::string> categories;
     int pos_count;
     int neg_count;
   };
   std::vector<Bin> bins;
   for (auto& it : category_counts) {
     Bin bin;
     bin.categories.push_back(it.first);
     bin.pos_count = category_positive_count[it.first];
     bin.neg_count = category_negative_count[it.first];
     bins.push_back(bin);
   }
   
   // Sort bins based on event rate
   std::sort(bins.begin(), bins.end(),
             [](const Bin& a, const Bin& b) {
               double rate_a = (double)a.pos_count / (a.pos_count + a.neg_count);
               double rate_b = (double)b.pos_count / (b.pos_count + b.neg_count);
               return rate_a < rate_b;
             });
   
   // Compute initial chi-square statistics between adjacent bins
   struct ChiMergeInfo {
     double chi2;
     int df;
     size_t bin_index; // Index of the first bin in the pair
   };
   std::vector<ChiMergeInfo> chi_infos;
   
   auto compute_chi2 = [](int pos1, int neg1, int pos2, int neg2) {
     int total_pos = pos1 + pos2;
     int total_neg = neg1 + neg2;
     int total = total_pos + total_neg;
     
     double expected_pos1 = ((double)(pos1 + neg1) * total_pos) / total;
     double expected_neg1 = ((double)(pos1 + neg1) * total_neg) / total;
     double expected_pos2 = ((double)(pos2 + neg2) * total_pos) / total;
     double expected_neg2 = ((double)(pos2 + neg2) * total_neg) / total;
     
     double chi2 = 0.0;
     if (expected_pos1 > 0) chi2 += pow(pos1 - expected_pos1, 2) / expected_pos1;
     if (expected_neg1 > 0) chi2 += pow(neg1 - expected_neg1, 2) / expected_neg1;
     if (expected_pos2 > 0) chi2 += pow(pos2 - expected_pos2, 2) / expected_pos2;
     if (expected_neg2 > 0) chi2 += pow(neg2 - expected_neg2, 2) / expected_neg2;
     
     return chi2;
   };
   
   for (size_t i = 0; i < bins.size() - 1; ++i) {
     double chi2 = compute_chi2(bins[i].pos_count, bins[i].neg_count,
                                bins[i + 1].pos_count, bins[i + 1].neg_count);
     ChiMergeInfo chi_info = { chi2, 1, i };
     chi_infos.push_back(chi_info);
   }
   
   // Iteratively merge bins
   while (bins.size() > (size_t)min_bins && !chi_infos.empty()) {
     // Find the pair with the smallest chi-square value
     auto min_chi_it = std::min_element(chi_infos.begin(), chi_infos.end(),
                                        [](const ChiMergeInfo& a, const ChiMergeInfo& b) {
                                          return a.chi2 < b.chi2;
                                        });
     double p_value = R::pchisq(min_chi_it->chi2, min_chi_it->df, false, false);
     
     // Stop if p-value is below threshold or max_bins reached
     if (p_value < pvalue_threshold && bins.size() <= (size_t)max_bins) {
       break;
     }
     
     // Merge the two bins
     size_t idx = min_chi_it->bin_index;
     bins[idx].categories.insert(bins[idx].categories.end(),
                                 bins[idx + 1].categories.begin(),
                                 bins[idx + 1].categories.end());
     bins[idx].pos_count += bins[idx + 1].pos_count;
     bins[idx].neg_count += bins[idx + 1].neg_count;
     bins.erase(bins.begin() + idx + 1);
     
     // Recompute chi-square statistics
     chi_infos.clear();
     for (size_t j = 0; j < bins.size() - 1; ++j) {
       double chi2 = compute_chi2(bins[j].pos_count, bins[j].neg_count,
                                  bins[j + 1].pos_count, bins[j + 1].neg_count);
       ChiMergeInfo chi_info = { chi2, 1, j };
       chi_infos.push_back(chi_info);
     }
   }
   
   // Ensure bins satisfy min_bads constraint
   bool merged = true;
   while (merged && bins.size() > (size_t)min_bins) {
     merged = false;
     for (size_t i = 0; i < bins.size(); ++i) {
       double bad_rate = (double)bins[i].pos_count / N;
       if (bad_rate < min_bads) {
         // Merge with neighbor bin
         size_t merge_idx = (i == 0) ? i + 1 : i - 1;
         bins[merge_idx].categories.insert(bins[merge_idx].categories.end(),
                                           bins[i].categories.begin(),
                                           bins[i].categories.end());
         bins[merge_idx].pos_count += bins[i].pos_count;
         bins[merge_idx].neg_count += bins[i].neg_count;
         bins.erase(bins.begin() + i);
         
         // Recompute chi-square statistics
         chi_infos.clear();
         for (size_t j = 0; j < bins.size() - 1; ++j) {
           double chi2 = compute_chi2(bins[j].pos_count, bins[j].neg_count,
                                      bins[j + 1].pos_count, bins[j + 1].neg_count);
           ChiMergeInfo chi_info = { chi2, 1, j };
           chi_infos.push_back(chi_info);
         }
         merged = true;
         break;
       }
     }
   }
   
   // Ensure monotonicity
   auto is_monotonic = [](const std::vector<Bin>& bins) {
     bool increasing = true, decreasing = true;
     for (size_t i = 1; i < bins.size(); ++i) {
       double rate_prev = (double)bins[i - 1].pos_count / (bins[i - 1].pos_count + bins[i - 1].neg_count);
       double rate_curr = (double)bins[i].pos_count / (bins[i].pos_count + bins[i].neg_count);
       if (rate_curr < rate_prev) increasing = false;
       if (rate_curr > rate_prev) decreasing = false;
     }
     return increasing || decreasing;
   };
   
   while (!is_monotonic(bins) && bins.size() > (size_t)min_bins) {
     // Merge bins to enforce monotonicity
     size_t merge_idx = 0;
     double min_diff = std::numeric_limits<double>::max();
     for (size_t i = 0; i < bins.size() - 1; ++i) {
       double rate_prev = (double)bins[i].pos_count / (bins[i].pos_count + bins[i].neg_count);
       double rate_next = (double)bins[i + 1].pos_count / (bins[i + 1].pos_count + bins[i + 1].neg_count);
       if (rate_next < rate_prev) {
         double diff = rate_prev - rate_next;
         if (diff < min_diff) {
           min_diff = diff;
           merge_idx = i;
         }
       }
     }
     // Merge bins
     bins[merge_idx].categories.insert(bins[merge_idx].categories.end(),
                                       bins[merge_idx + 1].categories.begin(),
                                       bins[merge_idx + 1].categories.end());
     bins[merge_idx].pos_count += bins[merge_idx + 1].pos_count;
     bins[merge_idx].neg_count += bins[merge_idx + 1].neg_count;
     bins.erase(bins.begin() + merge_idx + 1);
     
     // Recompute chi-square statistics
     chi_infos.clear();
     for (size_t j = 0; j < bins.size() - 1; ++j) {
       double chi2 = compute_chi2(bins[j].pos_count, bins[j].neg_count,
                                  bins[j + 1].pos_count, bins[j + 1].neg_count);
       ChiMergeInfo chi_info = { chi2, 1, j };
       chi_infos.push_back(chi_info);
     }
   }
   
   // Compute WoE and IV
   int total_pos = 0;
   int total_neg = 0;
   for (auto& bin : bins) {
     total_pos += bin.pos_count;
     total_neg += bin.neg_count;
   }
   std::vector<double> woe(bins.size());
   std::vector<double> iv_bin(bins.size());
   double total_iv = 0.0;
   for (size_t i = 0; i < bins.size(); ++i) {
     double dist_pos = (double)bins[i].pos_count / total_pos;
     double dist_neg = (double)bins[i].neg_count / total_neg;
     if (dist_pos == 0) dist_pos = 1e-10;
     if (dist_neg == 0) dist_neg = 1e-10;
     woe[i] = log(dist_pos / dist_neg);
     iv_bin[i] = (dist_pos - dist_neg) * woe[i];
     total_iv += iv_bin[i];
   }
   
   // Map categories to WoE
   std::map<std::string, double> category_woe_map;
   for (size_t i = 0; i < bins.size(); ++i) {
     for (auto& cat : bins[i].categories) {
       category_woe_map[cat] = woe[i];
     }
   }
   NumericVector feature_woe(N);
   for (int i = 0; i < N; ++i) {
     feature_woe[i] = category_woe_map[feature_processed[i]];
   }
   
   // Prepare bin output
   std::vector<std::string> bin_names(bins.size());
   std::vector<int> count(bins.size());
   std::vector<int> pos(bins.size());
   std::vector<int> neg(bins.size());
   for (size_t i = 0; i < bins.size(); ++i) {
     if (!bins[i].categories.empty()) {
       bin_names[i] = bins[i].categories[0];
       for (size_t j = 1; j < bins[i].categories.size(); ++j) {
         bin_names[i] += "+" + bins[i].categories[j];
       }
     } else {
       bin_names[i] = "Unknown";
     }
     count[i] = bins[i].pos_count + bins[i].neg_count;
     pos[i] = bins[i].pos_count;
     neg[i] = bins[i].neg_count;
   }
   
   // Create List for bins
   List bin_lst = List::create(
     Named("bin") = bin_names,
     Named("woe") = woe,
     Named("iv") = iv_bin,
     Named("count") = count,
     Named("count_pos") = pos,
     Named("count_neg") = neg);
   
   // Create List for woe vector feature
   List woe_lst = List::create(
     Named("woefeature") = feature_woe
   );
   
   // Attrib class for compatibility with data.table in memory superfast tables
   bin_lst.attr("class") = CharacterVector::create("data.table", "data.frame");
   woe_lst.attr("class") = CharacterVector::create("data.table", "data.frame");
   
   // Return output
   List output_list = List::create(
     Named("woefeature") = woe_lst,
     Named("woebin") = bin_lst
   // Named("woe") = woe,
   // Named("iv") = total_iv,
   // Named("pos") = pos,
   // Named("neg") = neg
   );
   
   // // Create DataFrame for bins
   // DataFrame bin_df = DataFrame::create(
   //   Named("bin") = bin_names,
   //   Named("woe") = woe,
   //   Named("iv") = iv_bin,
   //   Named("count") = count,
   //   Named("count_pos") = pos,
   //   Named("count_neg") = neg
   // );
   // 
   // // Return output
   // List output_list = List::create(
   //   Named("woefeature") = feature_woe,
   //   Named("woebin") = bin_df
   // // Named("woe") = woe,
   // // Named("iv") = total_iv,
   // // Named("pos") = pos,
   // // Named("neg") = neg
   // );
   // 
   // output_list.attr("class") = CharacterVector::create("data.table", "data.frame");
   
   return output_list;
 }
 
 
 //' Performs optimal binning of a categorical variable for Weight of Evidence (WoE) and Information Value (IV) using the Minimum Description Length Principle (MDLP) criterion
 //'
 //' This function processes a categorical variable by grouping rare categories, calculating event rates, and iteratively merging bins using the MDLP criterion to maximize the information gain. The function also ensures monotonicity of event rates and calculates WoE and IV for the generated bins.
 //'
 //' @param target Integer vector representing the binary target variable, where 1 indicates a positive event (e.g., default) and 0 indicates a negative event (e.g., non-default).
 //' @param feature Character vector representing the categorical variable to be binned.
 //' @param min_bins (Optional) Minimum number of bins to generate. Default is 2.
 //' @param max_bins (Optional) Maximum number of bins to generate. Default is 7.
 //' @param cat_cutoff (Optional) Frequency cutoff value, below which categories are grouped into "Other". Default is 0.05.
 //' @param min_bads (Optional) Minimum proportion of bad events that a bin must contain. Default is 0.05.
 //'
 //' @return A list with the following elements:
 //' \itemize{
 //'   \item \code{feature_woe}: Numeric vector with the WoE assigned to each instance of the processed categorical variable.
 //'   \item \code{bin}: DataFrame with the generated bins, containing the following fields:
 //'     \itemize{
 //'       \item \code{bin}: Names of the categories grouped into each bin.
 //'       \item \code{woe}: Weight of Evidence (WoE) for each bin.
 //'       \item \code{iv}: Information Value (IV) for each bin.
 //'       \item \code{count}: Total number of observations in each bin.
 //'       \item \code{count_pos}: Count of positive events in each bin.
 //'       \item \code{count_neg}: Count of negative events in each bin.
 //'     }
 //'   \item \code{woe}: Numeric vector with the WoE for each bin.
 //'   \item \code{iv}: Total Information Value (IV) calculated for the variable.
 //'   \item \code{pos}: Vector with the count of positive events in each bin.
 //'   \item \code{neg}: Vector with the count of negative events in each bin.
 //' }
 //'
 //'
 // [[Rcpp::export]]
 Rcpp::List OptimalBinningCategoricalMDLP(Rcpp::IntegerVector target, Rcpp::CharacterVector feature, int min_bins = 2, int max_bins = 7, double cat_cutoff = 0.05, double min_bads = 0.05) {
   int N = target.size();
   if (feature.size() != N) {
     Rcpp::stop("Length of target and feature must be the same.");
   }
   
   // Parameters
   
   // Handle rare categories
   std::map<std::string, int> category_counts;
   for (int i = 0; i < N; ++i) {
     std::string cat = Rcpp::as<std::string>(feature[i]);
     category_counts[cat]++;
   }
   
   // Identify rare categories
   std::set<std::string> rare_categories;
   for (auto& it : category_counts) {
     if ((double)it.second / N < cat_cutoff) {
       rare_categories.insert(it.first);
     }
   }
   
   // Map categories, combining rare categories into 'Other'
   std::vector<std::string> feature_processed(N);
   for (int i = 0; i < N; ++i) {
     std::string cat = Rcpp::as<std::string>(feature[i]);
     if (rare_categories.find(cat) != rare_categories.end()) {
       feature_processed[i] = "Other";
     } else {
       feature_processed[i] = cat;
     }
   }
   
   // Recompute category counts after processing rare categories
   category_counts.clear();
   for (int i = 0; i < N; ++i) {
     category_counts[feature_processed[i]]++;
   }
   
   // Compute positive and negative counts per category
   std::map<std::string, int> category_positive_count;
   std::map<std::string, int> category_negative_count;
   for (auto& it : category_counts) {
     category_positive_count[it.first] = 0;
     category_negative_count[it.first] = 0;
   }
   for (int i = 0; i < N; ++i) {
     std::string cat = feature_processed[i];
     int tgt = target[i];
     if (tgt == 1) {
       category_positive_count[cat]++;
     } else {
       category_negative_count[cat]++;
     }
   }
   
   // Compute event rates per category
   std::vector<std::pair<std::string, double>> categories_event_rates;
   for (auto& it : category_counts) {
     int pos = category_positive_count[it.first];
     int neg = category_negative_count[it.first];
     double event_rate = (double)pos / (pos + neg);
     categories_event_rates.push_back(std::make_pair(it.first, event_rate));
   }
   
   // Sort categories by event rate
   std::sort(categories_event_rates.begin(), categories_event_rates.end(),
             [](const std::pair<std::string, double>& a, const std::pair<std::string, double>& b) {
               return a.second < b.second;
             });
   
   // Initialize bins: each category is its own bin
   struct Bin {
     std::vector<std::string> categories;
     int pos_count;
     int neg_count;
   };
   std::vector<Bin> bins;
   for (size_t i = 0; i < categories_event_rates.size(); ++i) {
     Bin bin;
     bin.categories.push_back(categories_event_rates[i].first);
     bin.pos_count = category_positive_count[categories_event_rates[i].first];
     bin.neg_count = category_negative_count[categories_event_rates[i].first];
     bins.push_back(bin);
   }
   
   // Function to compute entropy
   auto entropy = [](double p, double n) {
     double total = p + n;
     if (total == 0) return 0.0;
     double p_ratio = p / total;
     double n_ratio = n / total;
     double ent = 0.0;
     if (p_ratio > 0) ent -= p_ratio * std::log2(p_ratio);
     if (n_ratio > 0) ent -= n_ratio * std::log2(n_ratio);
     return ent;
   };
   
   // Compute initial total entropy
   int total_pos = 0;
   int total_neg = 0;
   for (auto& bin : bins) {
     total_pos += bin.pos_count;
     total_neg += bin.neg_count;
   }
   double total_entropy = entropy(total_pos, total_neg);
   
   // Iteratively merge bins using MDLP criterion
   bool bins_merged = true;
   while (bins_merged && bins.size() > (size_t)min_bins) {
     bins_merged = false;
     size_t best_merge_idx = 0;
     double best_delta = std::numeric_limits<double>::infinity();
     for (size_t i = 0; i < bins.size() - 1; ++i) {
       int pos_left = bins[i].pos_count;
       int neg_left = bins[i].neg_count;
       int pos_right = bins[i + 1].pos_count;
       int neg_right = bins[i + 1].neg_count;
       
       int N_total = pos_left + neg_left + pos_right + neg_right;
       double entropy_parent = entropy(pos_left + pos_right, neg_left + neg_right);
       double entropy_left = entropy(pos_left, neg_left);
       double entropy_right = entropy(pos_right, neg_right);
       
       // Compute Information Gain
       double gain = entropy_parent - ((pos_left + neg_left) * entropy_left + (pos_right + neg_right) * entropy_right) / N_total;
       
       // Compute MDL Criterion
       double delta = std::log2(N_total - 1) / N_total - gain;
       
       if (delta < best_delta) {
         best_delta = delta;
         best_merge_idx = i;
         bins_merged = true;
       }
     }
     if (bins_merged) {
       // Merge the pair with the smallest delta
       bins[best_merge_idx].categories.insert(bins[best_merge_idx].categories.end(),
                                              bins[best_merge_idx + 1].categories.begin(),
                                              bins[best_merge_idx + 1].categories.end());
       bins[best_merge_idx].pos_count += bins[best_merge_idx + 1].pos_count;
       bins[best_merge_idx].neg_count += bins[best_merge_idx + 1].neg_count;
       bins.erase(bins.begin() + best_merge_idx + 1);
     }
   }
   
   // Ensure bins do not exceed max_bins
   while (bins.size() > (size_t)max_bins) {
     // Merge the pair of bins with the smallest difference in event rates
     double min_diff = std::numeric_limits<double>::max();
     size_t merge_idx = 0;
     for (size_t i = 0; i < bins.size() - 1; ++i) {
       double rate_left = (double)bins[i].pos_count / (bins[i].pos_count + bins[i].neg_count);
       double rate_right = (double)bins[i + 1].pos_count / (bins[i + 1].pos_count + bins[i + 1].neg_count);
       double diff = std::abs(rate_right - rate_left);
       if (diff < min_diff) {
         min_diff = diff;
         merge_idx = i;
       }
     }
     // Merge bins
     bins[merge_idx].categories.insert(bins[merge_idx].categories.end(),
                                       bins[merge_idx + 1].categories.begin(),
                                       bins[merge_idx + 1].categories.end());
     bins[merge_idx].pos_count += bins[merge_idx + 1].pos_count;
     bins[merge_idx].neg_count += bins[merge_idx + 1].neg_count;
     bins.erase(bins.begin() + merge_idx + 1);
   }
   
   // Ensure monotonicity
   auto is_monotonic = [](const std::vector<Bin>& bins) {
     bool increasing = true, decreasing = true;
     for (size_t i = 1; i < bins.size(); ++i) {
       double rate_prev = (double)bins[i - 1].pos_count / (bins[i - 1].pos_count + bins[i - 1].neg_count);
       double rate_curr = (double)bins[i].pos_count / (bins[i].pos_count + bins[i].neg_count);
       if (rate_curr < rate_prev) increasing = false;
       if (rate_curr > rate_prev) decreasing = false;
     }
     return increasing || decreasing;
   };
   
   while (!is_monotonic(bins) && bins.size() > (size_t)min_bins) {
     // Find the bin to merge to enforce monotonicity
     size_t merge_idx = 0;
     double min_diff = std::numeric_limits<double>::max();
     for (size_t i = 0; i < bins.size() - 1; ++i) {
       double rate_prev = (double)bins[i].pos_count / (bins[i].pos_count + bins[i].neg_count);
       double rate_next = (double)bins[i + 1].pos_count / (bins[i + 1].pos_count + bins[i + 1].neg_count);
       if (rate_next < rate_prev) {
         double diff = rate_prev - rate_next;
         if (diff < min_diff) {
           min_diff = diff;
           merge_idx = i;
         }
       }
     }
     // Merge bins
     bins[merge_idx].categories.insert(bins[merge_idx].categories.end(),
                                       bins[merge_idx + 1].categories.begin(),
                                       bins[merge_idx + 1].categories.end());
     bins[merge_idx].pos_count += bins[merge_idx + 1].pos_count;
     bins[merge_idx].neg_count += bins[merge_idx + 1].neg_count;
     bins.erase(bins.begin() + merge_idx + 1);
   }
   
   // Compute WoE and IV
   total_pos = 0;
   total_neg = 0;
   for (auto& bin : bins) {
     total_pos += bin.pos_count;
     total_neg += bin.neg_count;
   }
   std::vector<double> woe(bins.size());
   std::vector<double> iv_bin(bins.size());
   double total_iv = 0.0;
   for (size_t i = 0; i < bins.size(); ++i) {
     double dist_pos = (double)bins[i].pos_count / total_pos;
     double dist_neg = (double)bins[i].neg_count / total_neg;
     if (dist_pos == 0) dist_pos = 1e-10;
     if (dist_neg == 0) dist_neg = 1e-10;
     woe[i] = log(dist_pos / dist_neg);
     iv_bin[i] = (dist_pos - dist_neg) * woe[i];
     total_iv += iv_bin[i];
   }
   
   // Map categories to WoE
   std::map<std::string, double> category_woe_map;
   for (size_t i = 0; i < bins.size(); ++i) {
     for (auto& cat : bins[i].categories) {
       category_woe_map[cat] = woe[i];
     }
   }
   
   // Map WoE back to feature
   NumericVector feature_woe(N);
   for (int i = 0; i < N; ++i) {
     feature_woe[i] = category_woe_map[feature_processed[i]];
   }
   
   // Prepare bin output
   std::vector<std::string> bin_names(bins.size());
   std::vector<int> count(bins.size());
   std::vector<int> pos(bins.size());
   std::vector<int> neg(bins.size());
   for (size_t i = 0; i < bins.size(); ++i) {
     // Join categories with '+'
     std::sort(bins[i].categories.begin(), bins[i].categories.end());
     bin_names[i] = bins[i].categories[0];
     for (size_t j = 1; j < bins[i].categories.size(); ++j) {
       bin_names[i] += "+" + bins[i].categories[j];
     }
     count[i] = bins[i].pos_count + bins[i].neg_count;
     pos[i] = bins[i].pos_count;
     neg[i] = bins[i].neg_count;
   }
   
   // Create List for bins
   List bin_lst = List::create(
     Named("bin") = bin_names,
     Named("woe") = woe,
     Named("iv") = iv_bin,
     Named("count") = count,
     Named("count_pos") = pos,
     Named("count_neg") = neg);
   
   // Create List for woe vector feature
   List woe_lst = List::create(
     Named("woefeature") = feature_woe
   );
   
   // Attrib class for compatibility with data.table in memory superfast tables
   bin_lst.attr("class") = CharacterVector::create("data.table", "data.frame");
   woe_lst.attr("class") = CharacterVector::create("data.table", "data.frame");
   
   // Return output
   List output_list = List::create(
     Named("woefeature") = woe_lst,
     Named("woebin") = bin_lst
   // Named("woe") = woe,
   // Named("iv") = total_iv,
   // Named("pos") = pos,
   // Named("neg") = neg
   );
   
   // // Create DataFrame for bins
   // DataFrame bin_df = DataFrame::create(
   //   Named("bin") = bin_names,
   //   Named("woe") = woe,
   //   Named("iv") = iv_bin,
   //   Named("count") = count,
   //   Named("count_pos") = pos,
   //   Named("count_neg") = neg
   // );
   // 
   // // Return output
   // List output_list = List::create(
   //   Named("woefeature") = feature_woe,
   //   Named("woebin") = bin_df
   // // Named("woe") = woe,
   // // Named("iv") = total_iv,
   // // Named("pos") = pos,
   // // Named("neg") = neg
   // );
   // 
   // output_list.attr("class") = CharacterVector::create("data.table", "data.frame");
   
   return output_list;
 }
 
 //' Performs optimal binning of a categorical variable for Weight of Evidence (WoE) and Information Value (IV) using the Class-Attribute Interdependence Maximization (CAIM) criterion
 //'
 //' This function processes a categorical variable by grouping rare categories, calculating event rates, and iteratively merging bins to maximize the CAIM criterion. The function also ensures monotonicity of event rates and calculates WoE and IV for the generated bins.
 //'
 //' @param target Integer vector representing the binary target variable, where 1 indicates a positive event (e.g., default) and 0 indicates a negative event (e.g., non-default).
 //' @param feature Character vector representing the categorical variable to be binned.
 //' @param min_bins (Optional) Minimum number of bins to generate. Default is 2.
 //' @param max_bins (Optional) Maximum number of bins to generate. Default is 7.
 //' @param cat_cutoff (Optional) Frequency cutoff value, below which categories are grouped into "Other". Default is 0.05.
 //' @param min_bads (Optional) Minimum proportion of bad events that a bin must contain. Default is 0.05.
 //'
 //' @return A list with the following elements:
 //' \itemize{
 //'   \item \code{feature_woe}: Numeric vector with the WoE assigned to each instance of the processed categorical variable.
 //'   \item \code{bin}: DataFrame with the generated bins, containing the following fields:
 //'     \itemize{
 //'       \item \code{bin}: Names of the categories grouped into each bin.
 //'       \item \code{woe}: Weight of Evidence (WoE) for each bin.
 //'       \item \code{iv}: Information Value (IV) for each bin.
 //'       \item \code{count}: Total number of observations in each bin.
 //'       \item \code{count_pos}: Count of positive events in each bin.
 //'       \item \code{count_neg}: Count of negative events in each bin.
 //'     }
 //'   \item \code{woe}: Numeric vector with the WoE for each bin.
 //'   \item \code{iv}: Total Information Value (IV) calculated for the variable.
 //'   \item \code{pos}: Vector with the count of positive events in each bin.
 //'   \item \code{neg}: Vector with the count of negative events in each bin.
 //' }
 //'
 //'
 // [[Rcpp::export]]
 Rcpp::List OptimalBinningCategoricalCAIM(Rcpp::IntegerVector target, Rcpp::CharacterVector feature, int min_bins = 2, int max_bins = 7, double cat_cutoff = 0.05, double min_bads = 0.05) {
   int N = target.size();
   if (feature.size() != N) {
     Rcpp::stop("Length of target and feature must be the same.");
   }
   
   // Parameters
   
   // Handle rare categories
   std::map<std::string, int> category_counts;
   for (int i = 0; i < N; ++i) {
     std::string cat = Rcpp::as<std::string>(feature[i]);
     category_counts[cat]++;
   }
   
   // Identify rare categories
   std::set<std::string> rare_categories;
   for (auto& it : category_counts) {
     if ((double)it.second / N < cat_cutoff) {
       rare_categories.insert(it.first);
     }
   }
   
   // Map categories, combining rare categories into 'Other'
   std::vector<std::string> feature_processed(N);
   for (int i = 0; i < N; ++i) {
     std::string cat = Rcpp::as<std::string>(feature[i]);
     if (rare_categories.find(cat) != rare_categories.end()) {
       feature_processed[i] = "Other";
     } else {
       feature_processed[i] = cat;
     }
   }
   
   // Recompute category counts after processing rare categories
   category_counts.clear();
   for (int i = 0; i < N; ++i) {
     category_counts[feature_processed[i]]++;
   }
   
   // Compute positive and negative counts per category
   std::map<std::string, int> category_positive_count;
   std::map<std::string, int> category_negative_count;
   for (auto& it : category_counts) {
     category_positive_count[it.first] = 0;
     category_negative_count[it.first] = 0;
   }
   for (int i = 0; i < N; ++i) {
     std::string cat = feature_processed[i];
     int tgt = target[i];
     if (tgt == 1) {
       category_positive_count[cat]++;
     } else {
       category_negative_count[cat]++;
     }
   }
   
   // Compute event rates per category
   std::vector<std::pair<std::string, double>> categories_event_rates;
   for (auto& it : category_counts) {
     int pos = category_positive_count[it.first];
     int neg = category_negative_count[it.first];
     double event_rate = (double)pos / (pos + neg);
     categories_event_rates.push_back(std::make_pair(it.first, event_rate));
   }
   
   // Sort categories by event rate
   std::sort(categories_event_rates.begin(), categories_event_rates.end(),
             [](const std::pair<std::string, double>& a, const std::pair<std::string, double>& b) {
               return a.second < b.second;
             });
   
   // Initialize bins: each category is its own bin
   struct Bin {
     std::vector<std::string> categories;
     int pos_count;
     int neg_count;
   };
   std::vector<Bin> bins;
   for (size_t i = 0; i < categories_event_rates.size(); ++i) {
     Bin bin;
     bin.categories.push_back(categories_event_rates[i].first);
     bin.pos_count = category_positive_count[categories_event_rates[i].first];
     bin.neg_count = category_negative_count[categories_event_rates[i].first];
     bins.push_back(bin);
   }
   
   // Compute initial CAIM criterion
   int total_pos = 0;
   int total_neg = 0;
   for (auto& bin : bins) {
     total_pos += bin.pos_count;
     total_neg += bin.neg_count;
   }
   
   // Function to compute CAIM criterion
   auto compute_caim = [&](const std::vector<Bin>& bins) {
     double caim = 0.0;
     int r = bins.size();
     for (size_t i = 0; i < bins.size(); ++i) {
       int Mi = bins[i].pos_count + bins[i].neg_count;
       int maxi = std::max(bins[i].pos_count, bins[i].neg_count);
       if (Mi > 0) {
         caim += (double)(maxi * maxi) / Mi;
       }
     }
     caim = caim / r;
     return caim;
   };
   
   double current_caim = compute_caim(bins);
   
   // Iteratively merge bins to maximize CAIM
   bool bins_merged = true;
   while (bins_merged && bins.size() > (size_t)min_bins) {
     bins_merged = false;
     size_t best_merge_idx = 0;
     double best_caim = current_caim;
     
     // Try merging each pair of adjacent bins
     for (size_t i = 0; i < bins.size() - 1; ++i) {
       // Create a copy of bins
       std::vector<Bin> bins_temp = bins;
       
       // Merge bins i and i+1
       bins_temp[i].categories.insert(bins_temp[i].categories.end(),
                                      bins_temp[i + 1].categories.begin(),
                                      bins_temp[i + 1].categories.end());
       bins_temp[i].pos_count += bins_temp[i + 1].pos_count;
       bins_temp[i].neg_count += bins_temp[i + 1].neg_count;
       bins_temp.erase(bins_temp.begin() + i + 1);
       
       // Compute CAIM for the new binning
       double new_caim = compute_caim(bins_temp);
       
       // Check if CAIM is improved
       if (new_caim > best_caim) {
         best_caim = new_caim;
         best_merge_idx = i;
         bins_merged = true;
       }
     }
     
     // If merging improves CAIM, perform the merge
     if (bins_merged) {
       bins[best_merge_idx].categories.insert(bins[best_merge_idx].categories.end(),
                                              bins[best_merge_idx + 1].categories.begin(),
                                              bins[best_merge_idx + 1].categories.end());
       bins[best_merge_idx].pos_count += bins[best_merge_idx + 1].pos_count;
       bins[best_merge_idx].neg_count += bins[best_merge_idx + 1].neg_count;
       bins.erase(bins.begin() + best_merge_idx + 1);
       current_caim = best_caim;
       
       // Ensure bins do not exceed max_bins
       if (bins.size() <= (size_t)max_bins) {
         break;
       }
     } else {
       break;
     }
   }
   
   // Ensure monotonicity
   auto is_monotonic = [](const std::vector<Bin>& bins) {
     bool increasing = true, decreasing = true;
     for (size_t i = 1; i < bins.size(); ++i) {
       double rate_prev = (double)bins[i - 1].pos_count / (bins[i - 1].pos_count + bins[i - 1].neg_count);
       double rate_curr = (double)bins[i].pos_count / (bins[i].pos_count + bins[i].neg_count);
       if (rate_curr < rate_prev) increasing = false;
       if (rate_curr > rate_prev) decreasing = false;
     }
     return increasing || decreasing;
   };
   
   while (!is_monotonic(bins) && bins.size() > (size_t)min_bins) {
     // Find the bin to merge to enforce monotonicity
     size_t merge_idx = 0;
     double min_diff = std::numeric_limits<double>::max();
     for (size_t i = 0; i < bins.size() - 1; ++i) {
       double rate_prev = (double)bins[i].pos_count / (bins[i].pos_count + bins[i].neg_count);
       double rate_next = (double)bins[i + 1].pos_count / (bins[i + 1].pos_count + bins[i + 1].neg_count);
       if (rate_next < rate_prev) {
         double diff = rate_prev - rate_next;
         if (diff < min_diff) {
           min_diff = diff;
           merge_idx = i;
         }
       }
     }
     // Merge bins
     bins[merge_idx].categories.insert(bins[merge_idx].categories.end(),
                                       bins[merge_idx + 1].categories.begin(),
                                       bins[merge_idx + 1].categories.end());
     bins[merge_idx].pos_count += bins[merge_idx + 1].pos_count;
     bins[merge_idx].neg_count += bins[merge_idx + 1].neg_count;
     bins.erase(bins.begin() + merge_idx + 1);
   }
   
   // Compute WoE and IV
   total_pos = 0;
   total_neg = 0;
   for (auto& bin : bins) {
     total_pos += bin.pos_count;
     total_neg += bin.neg_count;
   }
   std::vector<double> woe(bins.size());
   std::vector<double> iv_bin(bins.size());
   double total_iv = 0.0;
   for (size_t i = 0; i < bins.size(); ++i) {
     double dist_pos = (double)bins[i].pos_count / total_pos;
     double dist_neg = (double)bins[i].neg_count / total_neg;
     if (dist_pos == 0) dist_pos = 1e-10;
     if (dist_neg == 0) dist_neg = 1e-10;
     woe[i] = log(dist_pos / dist_neg);
     iv_bin[i] = (dist_pos - dist_neg) * woe[i];
     total_iv += iv_bin[i];
   }
   
   // Map categories to WoE
   std::map<std::string, double> category_woe_map;
   for (size_t i = 0; i < bins.size(); ++i) {
     for (auto& cat : bins[i].categories) {
       category_woe_map[cat] = woe[i];
     }
   }
   
   // Map WoE back to feature
   NumericVector feature_woe(N);
   for (int i = 0; i < N; ++i) {
     feature_woe[i] = category_woe_map[feature_processed[i]];
   }
   
   // Prepare bin output
   std::vector<std::string> bin_names(bins.size());
   std::vector<int> count(bins.size());
   std::vector<int> pos(bins.size());
   std::vector<int> neg(bins.size());
   for (size_t i = 0; i < bins.size(); ++i) {
     // Join categories with '+'
     std::sort(bins[i].categories.begin(), bins[i].categories.end());
     bin_names[i] = bins[i].categories[0];
     for (size_t j = 1; j < bins[i].categories.size(); ++j) {
       bin_names[i] += "+" + bins[i].categories[j];
     }
     count[i] = bins[i].pos_count + bins[i].neg_count;
     pos[i] = bins[i].pos_count;
     neg[i] = bins[i].neg_count;
   }
   
   // Create List for bins
   List bin_lst = List::create(
     Named("bin") = bin_names,
     Named("woe") = woe,
     Named("iv") = iv_bin,
     Named("count") = count,
     Named("count_pos") = pos,
     Named("count_neg") = neg);
   
   // Create List for woe vector feature
   List woe_lst = List::create(
     Named("woefeature") = feature_woe
   );
   
   // Attrib class for compatibility with data.table in memory superfast tables
   bin_lst.attr("class") = CharacterVector::create("data.table", "data.frame");
   woe_lst.attr("class") = CharacterVector::create("data.table", "data.frame");
   
   // Return output
   List output_list = List::create(
     Named("woefeature") = woe_lst,
     Named("woebin") = bin_lst
   // Named("woe") = woe,
   // Named("iv") = total_iv,
   // Named("pos") = pos,
   // Named("neg") = neg
   );
   
   // // Create DataFrame for bins
   // DataFrame bin_df = DataFrame::create(
   //   Named("bin") = bin_names,
   //   Named("woe") = woe,
   //   Named("iv") = iv_bin,
   //   Named("count") = count,
   //   Named("count_pos") = pos,
   //   Named("count_neg") = neg
   // );
   // 
   // // Return output
   // List output_list = List::create(
   //   Named("woefeature") = feature_woe,
   //   Named("woebin") = bin_df
   // // Named("woe") = woe,
   // // Named("iv") = total_iv,
   // // Named("pos") = pos,
   // // Named("neg") = neg
   // );
   // 
   // output_list.attr("class") = CharacterVector::create("data.table", "data.frame");
   
   return output_list;
   
 }
 
 //' Performs optimal binning of a categorical variable for Weight of Evidence (WoE) and Information Value (IV), ensuring monotonicity and maximizing IV
 //'
 //' This function processes a categorical variable by grouping rare categories, calculating event rates, and iteratively merging bins to maximize Information Value (IV) while ensuring monotonicity of event rates. It also calculates WoE and IV for the generated bins.
 //'
 //' @param target Integer vector representing the binary target variable, where 1 indicates a positive event (e.g., default) and 0 indicates a negative event (e.g., non-default).
 //' @param feature Character vector representing the categorical variable to be binned.
 //' @param min_bins (Optional) Minimum number of bins to generate. Default is 2.
 //' @param max_bins (Optional) Maximum number of bins to generate. Default is 7.
 //' @param cat_cutoff (Optional) Frequency cutoff value, below which categories are grouped into "Other". Default is 0.05.
 //' @param min_bads (Optional) Minimum proportion of bad events that a bin must contain. Default is 0.05.
 //'
 //' @return A list with the following elements:
 //' \itemize{
 //'   \item \code{feature_woe}: Numeric vector with the WoE assigned to each instance of the processed categorical variable.
 //'   \item \code{bin}: DataFrame with the generated bins, containing the following fields:
 //'     \itemize{
 //'       \item \code{bin}: Names of the categories grouped into each bin.
 //'       \item \code{woe}: Weight of Evidence (WoE) for each bin.
 //'       \item \code{iv}: Information Value (IV) for each bin.
 //'       \item \code{count}: Total number of observations in each bin.
 //'       \item \code{count_pos}: Count of positive events in each bin.
 //'       \item \code{count_neg}: Count of negative events in each bin.
 //'     }
 //'   \item \code{woe}: Numeric vector with the WoE for each bin.
 //'   \item \code{iv}: Total Information Value (IV) calculated for the variable.
 //'   \item \code{pos}: Vector with the count of positive events in each bin.
 //'   \item \code{neg}: Vector with the count of negative events in each bin.
 //' }
 //'
 //'
 // [[Rcpp::export]]
 Rcpp::List OptimalBinningCategoricalIV(Rcpp::IntegerVector target, Rcpp::CharacterVector feature, int min_bins = 2, int max_bins = 7, double cat_cutoff = 0.05, double min_bads = 0.05) {
   int N = target.size();
   if (feature.size() != N) {
     Rcpp::stop("Length of target and feature must be the same.");
   }
   
   // Parameters
   
   // Handle rare categories
   std::map<std::string, int> category_counts;
   for (int i = 0; i < N; ++i) {
     std::string cat = Rcpp::as<std::string>(feature[i]);
     category_counts[cat]++;
   }
   
   // Identify rare categories
   std::set<std::string> rare_categories;
   for (auto& it : category_counts) {
     if ((double)it.second / N < cat_cutoff) {
       rare_categories.insert(it.first);
     }
   }
   
   // Map categories, combining rare categories into 'Other'
   std::vector<std::string> feature_processed(N);
   for (int i = 0; i < N; ++i) {
     std::string cat = Rcpp::as<std::string>(feature[i]);
     if (rare_categories.find(cat) != rare_categories.end()) {
       feature_processed[i] = "Other";
     } else {
       feature_processed[i] = cat;
     }
   }
   
   // Recompute category counts after processing rare categories
   category_counts.clear();
   for (int i = 0; i < N; ++i) {
     category_counts[feature_processed[i]]++;
   }
   
   // Compute positive and negative counts per category
   std::map<std::string, int> category_positive_count;
   std::map<std::string, int> category_negative_count;
   for (auto& it : category_counts) {
     category_positive_count[it.first] = 0;
     category_negative_count[it.first] = 0;
   }
   for (int i = 0; i < N; ++i) {
     std::string cat = feature_processed[i];
     int tgt = target[i];
     if (tgt == 1) {
       category_positive_count[cat]++;
     } else {
       category_negative_count[cat]++;
     }
   }
   
   // Compute total positives and negatives
   int total_pos = 0;
   int total_neg = 0;
   for (auto& it : category_counts) {
     total_pos += category_positive_count[it.first];
     total_neg += category_negative_count[it.first];
   }
   
   // Define CategoryInfo struct
   struct CategoryInfo {
     std::string category;
     int pos_count;
     int neg_count;
     double dist_pos;
     double dist_neg;
     double event_rate;
     double woe;
     double iv;
   };
   
   // Compute initial WoE and IV for each category
   std::vector<CategoryInfo> categories_info;
   for (auto& it : category_counts) {
     CategoryInfo cat_info;
     cat_info.category = it.first;
     cat_info.pos_count = category_positive_count[it.first];
     cat_info.neg_count = category_negative_count[it.first];
     cat_info.dist_pos = (double)cat_info.pos_count / total_pos;
     cat_info.dist_neg = (double)cat_info.neg_count / total_neg;
     if (cat_info.dist_pos == 0) cat_info.dist_pos = 1e-10;
     if (cat_info.dist_neg == 0) cat_info.dist_neg = 1e-10;
     cat_info.event_rate = (double)cat_info.pos_count / (cat_info.pos_count + cat_info.neg_count);
     cat_info.woe = log(cat_info.dist_pos / cat_info.dist_neg);
     cat_info.iv = (cat_info.dist_pos - cat_info.dist_neg) * cat_info.woe;
     categories_info.push_back(cat_info);
   }
   
   // Sort categories by event rate
   std::sort(categories_info.begin(), categories_info.end(), [](const CategoryInfo& a, const CategoryInfo& b) {
     return a.event_rate < b.event_rate;
   });
   
   // Initialize bins: each category is its own bin
   struct Bin {
     std::vector<std::string> categories;
     int pos_count;
     int neg_count;
     double dist_pos;
     double dist_neg;
     double event_rate;
     double woe;
     double iv;
   };
   
   std::vector<Bin> bins;
   for (size_t i = 0; i < categories_info.size(); ++i) {
     Bin bin;
     bin.categories.push_back(categories_info[i].category);
     bin.pos_count = categories_info[i].pos_count;
     bin.neg_count = categories_info[i].neg_count;
     bin.dist_pos = categories_info[i].dist_pos;
     bin.dist_neg = categories_info[i].dist_neg;
     bin.event_rate = categories_info[i].event_rate;
     bin.woe = categories_info[i].woe;
     bin.iv = categories_info[i].iv;
     bins.push_back(bin);
   }
   
   // Compute total IV
   double total_iv_bins = 0.0;
   for (size_t i = 0; i < bins.size(); ++i) {
     total_iv_bins += bins[i].iv;
   }
   
   // Function to merge two bins
   auto mergeBins = [&](const Bin& bin1, const Bin& bin2) {
     Bin merged_bin;
     merged_bin.categories = bin1.categories;
     merged_bin.categories.insert(merged_bin.categories.end(), bin2.categories.begin(), bin2.categories.end());
     merged_bin.pos_count = bin1.pos_count + bin2.pos_count;
     merged_bin.neg_count = bin1.neg_count + bin2.neg_count;
     merged_bin.dist_pos = (double)merged_bin.pos_count / total_pos;
     merged_bin.dist_neg = (double)merged_bin.neg_count / total_neg;
     if (merged_bin.dist_pos == 0) merged_bin.dist_pos = 1e-10;
     if (merged_bin.dist_neg == 0) merged_bin.dist_neg = 1e-10;
     merged_bin.event_rate = (double)merged_bin.pos_count / (merged_bin.pos_count + merged_bin.neg_count);
     merged_bin.woe = log(merged_bin.dist_pos / merged_bin.dist_neg);
     merged_bin.iv = (merged_bin.dist_pos - merged_bin.dist_neg) * merged_bin.woe;
     return merged_bin;
   };
   
   // Function to check monotonicity
   auto is_monotonic = [](const std::vector<Bin>& bins) {
     bool increasing = true;
     bool decreasing = true;
     for (size_t i = 1; i < bins.size(); ++i) {
       if (bins[i].event_rate < bins[i - 1].event_rate) {
         increasing = false;
       }
       if (bins[i].event_rate > bins[i - 1].event_rate) {
         decreasing = false;
       }
     }
     return increasing || decreasing;
   };
   
   // Main merging loop
   bool bins_merged = true;
   while ((bins.size() > (size_t)max_bins || !is_monotonic(bins)) && bins.size() > (size_t)min_bins) {
     bins_merged = false;
     size_t best_merge_idx = 0;
     double best_delta_IV = std::numeric_limits<double>::max();
     Bin best_merged_bin;
     double total_iv_before_merging = total_iv_bins;
     
     for (size_t i = 0; i < bins.size() - 1; ++i) {
       // Merge bins[i] and bins[i+1]
       Bin merged_bin = mergeBins(bins[i], bins[i + 1]);
       // Check min_bads constraint
       double bad_rate = (double)merged_bin.pos_count / N;
       if (bad_rate < min_bads) {
         continue; // Skip this pair
       }
       // Compute total IV after merging
       double total_iv_after_merging = total_iv_bins - bins[i].iv - bins[i + 1].iv + merged_bin.iv;
       double delta_IV = total_iv_after_merging - total_iv_before_merging;
       // We aim to minimize the decrease in total IV
       if (delta_IV < best_delta_IV) {
         best_delta_IV = delta_IV;
         best_merge_idx = i;
         best_merged_bin = merged_bin;
         bins_merged = true;
       }
     }
     
     if (bins_merged) {
       // Merge the best pair
       bins[best_merge_idx] = best_merged_bin;
       bins.erase(bins.begin() + best_merge_idx + 1);
       total_iv_bins = total_iv_before_merging + best_delta_IV;
     } else {
       // Cannot merge further without violating constraints
       break;
     }
   }
   
   // Ensure monotonicity
   while (!is_monotonic(bins) && bins.size() > (size_t)min_bins) {
     size_t merge_idx = 0;
     double min_diff = std::numeric_limits<double>::max();
     for (size_t i = 0; i < bins.size() - 1; ++i) {
       if (bins[i + 1].event_rate < bins[i].event_rate) {
         double diff = bins[i].event_rate - bins[i + 1].event_rate;
         if (diff < min_diff) {
           min_diff = diff;
           merge_idx = i;
         }
       }
     }
     // Merge bins[merge_idx] and bins[merge_idx + 1]
     Bin merged_bin = mergeBins(bins[merge_idx], bins[merge_idx + 1]);
     // Update total_iv_bins
     total_iv_bins = total_iv_bins - bins[merge_idx].iv - bins[merge_idx + 1].iv + merged_bin.iv;
     // Update bins
     bins[merge_idx] = merged_bin;
     bins.erase(bins.begin() + merge_idx + 1);
   }
   
   // Map categories to WoE
   std::map<std::string, double> category_woe_map;
   for (size_t i = 0; i < bins.size(); ++i) {
     for (auto& cat : bins[i].categories) {
       category_woe_map[cat] = bins[i].woe;
     }
   }
   
   // Map WoE back to feature
   NumericVector feature_woe(N);
   for (int i = 0; i < N; ++i) {
     feature_woe[i] = category_woe_map[feature_processed[i]];
   }
   
   // Prepare bin output
   std::vector<std::string> bin_names(bins.size());
   std::vector<double> woe(bins.size());
   std::vector<double> iv_bin(bins.size());
   std::vector<int> count(bins.size());
   std::vector<int> pos(bins.size());
   std::vector<int> neg(bins.size());
   for (size_t i = 0; i < bins.size(); ++i) {
     // Join categories with '+'
     std::sort(bins[i].categories.begin(), bins[i].categories.end());
     bin_names[i] = bins[i].categories[0];
     for (size_t j = 1; j < bins[i].categories.size(); ++j) {
       bin_names[i] += "+" + bins[i].categories[j];
     }
     woe[i] = bins[i].woe;
     iv_bin[i] = bins[i].iv;
     count[i] = bins[i].pos_count + bins[i].neg_count;
     pos[i] = bins[i].pos_count;
     neg[i] = bins[i].neg_count;
   }
   
   // Create List for bins
   List bin_lst = List::create(
     Named("bin") = bin_names,
     Named("woe") = woe,
     Named("iv") = iv_bin,
     Named("count") = count,
     Named("count_pos") = pos,
     Named("count_neg") = neg);
   
   // Create List for woe vector feature
   List woe_lst = List::create(
     Named("woefeature") = feature_woe
   );
   
   // Attrib class for compatibility with data.table in memory superfast tables
   bin_lst.attr("class") = CharacterVector::create("data.table", "data.frame");
   woe_lst.attr("class") = CharacterVector::create("data.table", "data.frame");
   
   // Return output
   List output_list = List::create(
     Named("woefeature") = woe_lst,
     Named("woebin") = bin_lst
   // Named("woe") = woe,
   // Named("iv") = total_iv,
   // Named("pos") = pos,
   // Named("neg") = neg
   );
   
   // Create DataFrame for bins
   // DataFrame bin_df = DataFrame::create(
   //   Named("bin") = bin_names,
   //   Named("woe") = woe,
   //   Named("iv") = iv_bin,
   //   Named("count") = count,
   //   Named("count_pos") = pos,
   //   Named("count_neg") = neg
   // );
   // 
   // // Return output
   // List output_list = List::create(
   //   Named("woefeature") = feature_woe,
   //   Named("woebin") = bin_df
   // // Named("woe") = woe,
   // // Named("iv") = total_iv,
   // // Named("pos") = pos,
   // // Named("neg") = neg
   // );
   // 
   // output_list.attr("class") = CharacterVector::create("data.table", "data.frame");
   
   return output_list;
   
 }
 
 // ---------------------------------------------------------------------------------------------- //
 // NUMERIC VARIABLES
 // ---------------------------------------------------------------------------------------------- //
 
 //' Performs optimal binning of a numeric variable for Weight of Evidence (WoE) and Information Value (IV) using a Mixed-Integer Programming (MIP) approach
 //'
 //' This function processes a numeric variable, removes missing values, and creates pre-bins based on unique values. It iteratively splits bins to maximize Information Value (IV) while ensuring monotonicity. The function also calculates WoE and IV for the generated bins.
 //'
 //' @param target Integer vector representing the binary target variable, where 1 indicates a positive event (e.g., default) and 0 indicates a negative event (e.g., non-default).
 //' @param feature Numeric vector representing the numeric variable to be binned.
 //' @param min_bins (Optional) Minimum number of bins to generate. Default is 2.
 //' @param max_bins (Optional) Maximum number of bins to generate. Default is 7.
 //' @param bin_cutoff (Optional) Cutoff value that determines the frequency of values to define pre-bins. Default is 0.05.
 //' @param max_n_prebins (Optional) Maximum number of pre-bins to consider before final binning. Default is 20.
 //'
 //' @return A list with the following elements:
 //' \itemize{
 //'   \item \code{feature_woe}: Numeric vector with the WoE assigned to each instance of the processed numeric variable.
 //'   \item \code{bin}: DataFrame containing the generated bins, with the following fields:
 //'     \itemize{
 //'       \item \code{bin}: String representing the range of values for each bin.
 //'       \item \code{woe}: Weight of Evidence (WoE) for each bin.
 //'       \item \code{iv}: Information Value (IV) for each bin.
 //'       \item \code{count}: Total number of observations in each bin.
 //'       \item \code{count_pos}: Count of positive events in each bin.
 //'       \item \code{count_neg}: Count of negative events in each bin.
 //'     }
 //'   \item \code{woe}: Numeric vector with the WoE for each bin.
 //'   \item \code{iv}: Total Information Value (IV) calculated for the variable.
 //'   \item \code{pos}: Vector with the count of positive events in each bin.
 //'   \item \code{neg}: Vector with the count of negative events in each bin.
 //' }
 //'
 //'
 // [[Rcpp::export]]
 Rcpp::List OptimalBinningNumericMIP(Rcpp::IntegerVector target, Rcpp::NumericVector feature, 
                                     int min_bins = 2, int max_bins = 7, 
                                     double bin_cutoff = 0.05, int max_n_prebins = 20) {
   int N = target.size();
   if (feature.size() != N) {
     Rcpp::stop("Length of target and feature must be the same.");
   }
   
   // Validate input parameters
   if (min_bins <= 0 || max_bins <= 0 || min_bins > max_bins) {
     Rcpp::stop("Invalid min_bins or max_bins. Ensure 0 < min_bins <= max_bins.");
   }
   
   if (bin_cutoff <= 0 || bin_cutoff >= 1) {
     Rcpp::stop("bin_cutoff must be between 0 and 1.");
   }
   
   // Validate target values
   for (int i = 0; i < N; ++i) {
     if (target[i] != 0 && target[i] != 1) {
       Rcpp::stop("Target must contain only 0s and 1s.");
     }
   }
   
   // Remove missing values
   Rcpp::NumericVector feature_clean;
   Rcpp::IntegerVector target_clean;
   for (int i = 0; i < N; ++i) {
     if (!Rcpp::NumericVector::is_na(feature[i]) && !Rcpp::IntegerVector::is_na(target[i])) {
       feature_clean.push_back(feature[i]);
       target_clean.push_back(target[i]);
     }
   }
   
   int N_clean = feature_clean.size();
   if (N_clean == 0) {
     Rcpp::stop("No valid data after removing missing values.");
   }
   
   // Create pre-bins
   std::set<double> unique_values(feature_clean.begin(), feature_clean.end());
   std::vector<double> sorted_values(unique_values.begin(), unique_values.end());
   std::sort(sorted_values.begin(), sorted_values.end());
   
   int n_prebins = std::min((int)sorted_values.size() - 1, max_n_prebins);
   std::vector<double> candidate_cutpoints;
   
   if (n_prebins > 0) {
     int step = std::max(1, (int)sorted_values.size() / n_prebins);
     for (int i = step; i < (int)sorted_values.size(); i += step) {
       candidate_cutpoints.push_back(sorted_values[i]);
     }
     // Ensure last cutpoint is included
     if (candidate_cutpoints.back() != sorted_values.back()) {
       candidate_cutpoints.push_back(sorted_values.back());
     }
   } else {
     candidate_cutpoints = sorted_values;
   }
   
   int n_cutpoints = candidate_cutpoints.size();
   
   // Prepare data for binning
   std::vector<int> pos_counts(n_cutpoints + 1, 0);
   std::vector<int> neg_counts(n_cutpoints + 1, 0);
   
   for (int i = 0; i < N_clean; ++i) {
     double value = feature_clean[i];
     int tgt = target_clean[i];
     
     // Determine bin index
     int bin_idx = 0;
     while (bin_idx < n_cutpoints && value > candidate_cutpoints[bin_idx]) {
       bin_idx++;
     }
     
     if (tgt == 1) {
       pos_counts[bin_idx]++;
     } else {
       neg_counts[bin_idx]++;
     }
   }
   
   int total_pos = std::accumulate(pos_counts.begin(), pos_counts.end(), 0);
   int total_neg = std::accumulate(neg_counts.begin(), neg_counts.end(), 0);
   
   // Define a structure for bins
   struct Bin {
     double lower_bound;
     double upper_bound;
     int pos_count;
     int neg_count;
     double event_rate;
   };
   
   // Initialize the entire range as the initial bin
   std::vector<Bin> bins;
   Bin initial_bin;
   initial_bin.lower_bound = -std::numeric_limits<double>::infinity();
   initial_bin.upper_bound = std::numeric_limits<double>::infinity();
   initial_bin.pos_count = total_pos;
   initial_bin.neg_count = total_neg;
   initial_bin.event_rate = (double)total_pos / (total_pos + total_neg);
   bins.push_back(initial_bin);
   
   // Function to compute IV
   auto compute_iv = [](const std::vector<Bin>& bins, int total_pos, int total_neg) {
     double iv = 0.0;
     for (const auto& bin : bins) {
       double dist_pos = (double)bin.pos_count / total_pos;
       double dist_neg = (double)bin.neg_count / total_neg;
       if (dist_pos > 0 && dist_neg > 0) {
         double woe = log(dist_pos / dist_neg);
         iv += (dist_pos - dist_neg) * woe;
       }
     }
     return iv;
   };
   
   // Recursive function to split bins
   std::function<void(std::vector<Bin>&, int, int)> split_bins;
   split_bins = [&](std::vector<Bin>& bins, int total_pos, int total_neg) {
     if (bins.size() >= (size_t)max_bins) return;
     
     // Find the bin with the largest IV improvement upon splitting
     double best_iv_increase = 0.0;
     size_t best_bin_idx = 0;
     double best_split_point = 0.0;
     Bin best_left_bin, best_right_bin;
     
     for (size_t i = 0; i < bins.size(); ++i) {
       const Bin& bin = bins[i];
       // Try splitting at each candidate cutpoint within the bin
       for (double split_point : candidate_cutpoints) {
         if (split_point <= bin.lower_bound || split_point >= bin.upper_bound) continue;
         
         // Split the bin at split_point
         Bin left_bin, right_bin;
         left_bin.lower_bound = bin.lower_bound;
         left_bin.upper_bound = split_point;
         right_bin.lower_bound = split_point;
         right_bin.upper_bound = bin.upper_bound;
         left_bin.pos_count = 0;
         left_bin.neg_count = 0;
         right_bin.pos_count = 0;
         right_bin.neg_count = 0;
         
         // Assign data points to left or right bin
         for (int j = 0; j < N_clean; ++j) {
           double value = feature_clean[j];
           int tgt = target_clean[j];
           if (value > left_bin.lower_bound && value <= left_bin.upper_bound) {
             if (tgt == 1) {
               left_bin.pos_count++;
             } else {
               left_bin.neg_count++;
             }
           } else if (value > right_bin.lower_bound && value <= right_bin.upper_bound) {
             if (tgt == 1) {
               right_bin.pos_count++;
             } else {
               right_bin.neg_count++;
             }
           }
         }
         
         // Check if the split violates the bin_cutoff
         int total_count = total_pos + total_neg;
         if ((left_bin.pos_count + left_bin.neg_count) < bin_cutoff * total_count ||
             (right_bin.pos_count + right_bin.neg_count) < bin_cutoff * total_count) {
           continue;
         }
         
         left_bin.event_rate = (double)left_bin.pos_count / (left_bin.pos_count + left_bin.neg_count);
         right_bin.event_rate = (double)right_bin.pos_count / (right_bin.pos_count + right_bin.neg_count);
         
         // Compute IV for the bins
         std::vector<Bin> temp_bins = bins;
         temp_bins.erase(temp_bins.begin() + i);
         temp_bins.push_back(left_bin);
         temp_bins.push_back(right_bin);
         
         double iv_before = compute_iv(bins, total_pos, total_neg);
         double iv_after = compute_iv(temp_bins, total_pos, total_neg);
         double iv_increase = iv_after - iv_before;
         
         if (iv_increase > best_iv_increase) {
           best_iv_increase = iv_increase;
           best_bin_idx = i;
           best_split_point = split_point;
           best_left_bin = left_bin;
           best_right_bin = right_bin;
         }
       }
     }
     
     // Replace the bin with the two new bins if a valid split was found
     if (best_iv_increase > 0) {
       bins.erase(bins.begin() + best_bin_idx);
       bins.push_back(best_left_bin);
       bins.push_back(best_right_bin);
       
       // Recursively split bins
       split_bins(bins, total_pos, total_neg);
     }
   };
   
   // Start splitting bins
   while (bins.size() < (size_t)min_bins) {
     split_bins(bins, total_pos, total_neg);
   }
   
   // Ensure bins are sorted by lower bound
   std::sort(bins.begin(), bins.end(), [](const Bin& a, const Bin& b) {
     return a.lower_bound < b.lower_bound;
   });
   
   // Compute WoE and IV for final bins
   std::vector<double> woe(bins.size());
   std::vector<double> iv_bin(bins.size());
   double total_iv = 0.0;
   for (size_t i = 0; i < bins.size(); ++i) {
     double dist_pos = (double)bins[i].pos_count / total_pos;
     double dist_neg = (double)bins[i].neg_count / total_neg;
     if (dist_pos > 0 && dist_neg > 0) {
       woe[i] = log(dist_pos / dist_neg);
       iv_bin[i] = (dist_pos - dist_neg) * woe[i];
       total_iv += iv_bin[i];
     } else {
       woe[i] = 0;
       iv_bin[i] = 0;
     }
   }
   
   // Map feature values to WoE
   Rcpp::NumericVector feature_woe(N);
   for (int i = 0; i < N; ++i) {
     double value = feature[i];
     if (Rcpp::NumericVector::is_na(value)) {
       feature_woe[i] = NA_REAL;
       continue;
     }
     // Find the bin
     int bin_idx = 0;
     while (bin_idx < (int)bins.size() && value > bins[bin_idx].upper_bound) {
       bin_idx++;
     }
     if (bin_idx >= (int)bins.size()) bin_idx = bins.size() - 1;
     feature_woe[i] = woe[bin_idx];
   }
   
   // Prepare bin output
   std::vector<std::string> bin_names(bins.size());
   std::vector<int> count(bins.size());
   std::vector<int> pos(bins.size());
   std::vector<int> neg(bins.size());
   
   for (size_t i = 0; i < bins.size(); ++i) {
     std::string lower = (bins[i].lower_bound == -std::numeric_limits<double>::infinity()) ? "[-Inf" : "[" + std::to_string(bins[i].lower_bound);
     std::string upper = (bins[i].upper_bound == std::numeric_limits<double>::infinity()) ? "+Inf]" : std::to_string(bins[i].upper_bound) + ")";
     bin_names[i] = lower + ";" + upper;
     count[i] = bins[i].pos_count + bins[i].neg_count;
     pos[i] = bins[i].pos_count;
     neg[i] = bins[i].neg_count;
   }
   
   // Create List for bins
   Rcpp::List bin_lst = Rcpp::List::create(
     Rcpp::Named("bin") = bin_names,
     Rcpp::Named("woe") = woe,
     Rcpp::Named("iv") = iv_bin,
     Rcpp::Named("count") = count,
     Rcpp::Named("count_pos") = pos,
     Rcpp::Named("count_neg") = neg);
   
   // Create List for woe vector feature
   Rcpp::List woe_lst = Rcpp::List::create(
     Rcpp::Named("woefeature") = feature_woe
   );
   
   // Attrib class for compatibility with data.table in memory superfast tables
   bin_lst.attr("class") = Rcpp::CharacterVector::create("data.table", "data.frame");
   woe_lst.attr("class") = Rcpp::CharacterVector::create("data.table", "data.frame");
   
   // Return output
   Rcpp::List output_list = Rcpp::List::create(
     Rcpp::Named("woefeature") = woe_lst,
     Rcpp::Named("woebin") = bin_lst
   );
   return output_list;
 }
 
 // Rcpp::List OptimalBinningNumericMIP(Rcpp::IntegerVector target, Rcpp::NumericVector feature, int min_bins = 2, int max_bins = 7, double bin_cutoff = 0.05, int max_n_prebins = 20) {
 //   int N = target.size();
 //   if (feature.size() != N) {
 //     Rcpp::stop("Length of target and feature must be the same.");
 //   }
 // 
 //   // Parameters
 // 
 //   // Remove missing values
 //   std::vector<double> feature_clean;
 //   std::vector<int> target_clean;
 //   for (int i = 0; i < N; ++i) {
 //     if (!NumericVector::is_na(feature[i]) && !NumericVector::is_na(target[i])) {
 //       feature_clean.push_back(feature[i]);
 //       target_clean.push_back(target[i]);
 //     }
 //   }
 // 
 //   int N_clean = feature_clean.size();
 //   if (N_clean == 0) {
 //     Rcpp::stop("No valid data after removing missing values.");
 //   }
 // 
 //   // Create pre-bins
 //   std::set<double> unique_values(feature_clean.begin(), feature_clean.end());
 //   std::vector<double> sorted_values(unique_values.begin(), unique_values.end());
 //   std::sort(sorted_values.begin(), sorted_values.end());
 // 
 //   int n_prebins = std::min((int)sorted_values.size() - 1, max_n_prebins);
 //   std::vector<double> candidate_cutpoints;
 // 
 //   if (n_prebins > 0) {
 //     int step = std::max(1, (int)sorted_values.size() / n_prebins);
 //     for (int i = step; i < (int)sorted_values.size(); i += step) {
 //       candidate_cutpoints.push_back(sorted_values[i]);
 //     }
 //     // Ensure last cutpoint is included
 //     if (candidate_cutpoints.back() != sorted_values.back()) {
 //       candidate_cutpoints.push_back(sorted_values.back());
 //     }
 //   } else {
 //     candidate_cutpoints = sorted_values;
 //   }
 // 
 //   int n_cutpoints = candidate_cutpoints.size();
 // 
 //   // Prepare data for binning
 //   // For each cutpoint, calculate the bin counts
 //   std::vector<int> pos_counts(n_cutpoints + 1, 0);
 //   std::vector<int> neg_counts(n_cutpoints + 1, 0);
 // 
 //   for (int i = 0; i < N_clean; ++i) {
 //     double value = feature_clean[i];
 //     int tgt = target_clean[i];
 // 
 //     // Determine bin index
 //     int bin_idx = 0;
 //     while (bin_idx < n_cutpoints && value > candidate_cutpoints[bin_idx]) {
 //       bin_idx++;
 //     }
 // 
 //     if (tgt == 1) {
 //       pos_counts[bin_idx]++;
 //     } else {
 //       neg_counts[bin_idx]++;
 //     }
 //   }
 // 
 //   int total_pos = std::accumulate(pos_counts.begin(), pos_counts.end(), 0);
 //   int total_neg = std::accumulate(neg_counts.begin(), neg_counts.end(), 0);
 // 
 //   // Define a structure for bins
 //   struct Bin {
 //     double lower_bound;
 //     double upper_bound;
 //     int pos_count;
 //     int neg_count;
 //     double event_rate;
 //   };
 // 
 //   // Initialize the entire range as the initial bin
 //   std::vector<Bin> bins;
 //   Bin initial_bin;
 //   initial_bin.lower_bound = -std::numeric_limits<double>::infinity();
 //   initial_bin.upper_bound = std::numeric_limits<double>::infinity();
 //   initial_bin.pos_count = total_pos;
 //   initial_bin.neg_count = total_neg;
 //   initial_bin.event_rate = (double)total_pos / (total_pos + total_neg);
 //   bins.push_back(initial_bin);
 // 
 //   // Function to compute IV
 //   auto compute_iv = [](const std::vector<Bin>& bins, int total_pos, int total_neg) {
 //     double iv = 0.0;
 //     for (const auto& bin : bins) {
 //       double dist_pos = (double)bin.pos_count / total_pos;
 //       double dist_neg = (double)bin.neg_count / total_neg;
 //       if (dist_pos == 0) dist_pos = 1e-10;
 //       if (dist_neg == 0) dist_neg = 1e-10;
 //       double woe = log(dist_pos / dist_neg);
 //       iv += (dist_pos - dist_neg) * woe;
 //     }
 //     return iv;
 //   };
 // 
 //   // Recursive function to split bins
 //   std::function<void(std::vector<Bin>&, int, int)> split_bins;
 //   split_bins = [&](std::vector<Bin>& bins, int total_pos, int total_neg) {
 //     if (bins.size() >= (size_t)max_bins) return;
 // 
 //     // Find the bin with the largest IV improvement upon splitting
 //     double best_iv_increase = 0.0;
 //     size_t best_bin_idx = 0;
 //     double best_split_point = 0.0;
 //     Bin best_left_bin, best_right_bin;
 // 
 //     for (size_t i = 0; i < bins.size(); ++i) {
 //       const Bin& bin = bins[i];
 //       // Try splitting at each candidate cutpoint within the bin
 //       for (double split_point : candidate_cutpoints) {
 //         if (split_point <= bin.lower_bound || split_point >= bin.upper_bound) continue;
 // 
 //         // Split the bin at split_point
 //         Bin left_bin, right_bin;
 //         left_bin.lower_bound = bin.lower_bound;
 //         left_bin.upper_bound = split_point;
 //         right_bin.lower_bound = split_point;
 //         right_bin.upper_bound = bin.upper_bound;
 //         left_bin.pos_count = 0;
 //         left_bin.neg_count = 0;
 //         right_bin.pos_count = 0;
 //         right_bin.neg_count = 0;
 // 
 //         // Assign data points to left or right bin
 //         for (int j = 0; j < N_clean; ++j) {
 //           double value = feature_clean[j];
 //           int tgt = target_clean[j];
 //           if (value > left_bin.lower_bound && value <= left_bin.upper_bound) {
 //             if (tgt == 1) {
 //               left_bin.pos_count++;
 //             } else {
 //               left_bin.neg_count++;
 //             }
 //           } else if (value > right_bin.lower_bound && value <= right_bin.upper_bound) {
 //             if (tgt == 1) {
 //               right_bin.pos_count++;
 //             } else {
 //               right_bin.neg_count++;
 //             }
 //           }
 //         }
 // 
 //         if (left_bin.pos_count + left_bin.neg_count == 0 || right_bin.pos_count + right_bin.neg_count == 0)
 //           continue;
 // 
 //         left_bin.event_rate = (double)left_bin.pos_count / (left_bin.pos_count + left_bin.neg_count);
 //         right_bin.event_rate = (double)right_bin.pos_count / (right_bin.pos_count + right_bin.neg_count);
 // 
 //         // Compute IV for the bins
 //         std::vector<Bin> temp_bins = bins;
 //         temp_bins.erase(temp_bins.begin() + i);
 //         temp_bins.push_back(left_bin);
 //         temp_bins.push_back(right_bin);
 // 
 //         double iv_before = compute_iv(bins, total_pos, total_neg);
 //         double iv_after = compute_iv(temp_bins, total_pos, total_neg);
 //         double iv_increase = iv_after - iv_before;
 // 
 //         if (iv_increase > best_iv_increase) {
 //           best_iv_increase = iv_increase;
 //           best_bin_idx = i;
 //           best_split_point = split_point;
 //           best_left_bin = left_bin;
 //           best_right_bin = right_bin;
 //         }
 //       }
 //     }
 // 
 //     if (best_iv_increase > 0) {
 //       // Replace the bin with the two new bins
 //       bins.erase(bins.begin() + best_bin_idx);
 //       bins.push_back(best_left_bin);
 //       bins.push_back(best_right_bin);
 // 
 //       // Recursively split bins
 //       split_bins(bins, total_pos, total_neg);
 //     }
 //   };
 // 
 //   // Start splitting bins
 //   split_bins(bins, total_pos, total_neg);
 // 
 //   // Ensure bins are sorted by lower bound
 //   std::sort(bins.begin(), bins.end(), [](const Bin& a, const Bin& b) {
 //     return a.lower_bound < b.lower_bound;
 //   });
 // 
 //   // Compute WoE and IV for final bins
 //   std::vector<double> woe(bins.size());
 //   std::vector<double> iv_bin(bins.size());
 //   double total_iv = 0.0;
 //   for (size_t i = 0; i < bins.size(); ++i) {
 //     double dist_pos = (double)bins[i].pos_count / total_pos;
 //     double dist_neg = (double)bins[i].neg_count / total_neg;
 //     if (dist_pos == 0) dist_pos = 1e-10;
 //     if (dist_neg == 0) dist_neg = 1e-10;
 //     woe[i] = log(dist_pos / dist_neg);
 //     iv_bin[i] = (dist_pos - dist_neg) * woe[i];
 //     total_iv += iv_bin[i];
 //   }
 // 
 //   // Map feature values to WoE
 //   NumericVector feature_woe(N);
 //   for (int i = 0; i < N; ++i) {
 //     double value = feature[i];
 //     if (NumericVector::is_na(value)) {
 //       feature_woe[i] = NA_REAL;
 //       continue;
 //     }
 //     // Find the bin
 //     int bin_idx = 0;
 //     while (bin_idx < (int)bins.size() && value > bins[bin_idx].upper_bound) {
 //       bin_idx++;
 //     }
 //     if (bin_idx >= (int)bins.size()) bin_idx = bins.size() - 1;
 //     feature_woe[i] = woe[bin_idx];
 //   }
 // 
 //   // Prepare bin output
 //   std::vector<std::string> bin_names(bins.size());
 //   std::vector<int> count(bins.size());
 //   std::vector<int> pos(bins.size());
 //   std::vector<int> neg(bins.size());
 // 
 //   for (size_t i = 0; i < bins.size(); ++i) {
 //     std::string lower = (bins[i].lower_bound == -std::numeric_limits<double>::infinity()) ? "[-Inf" : "[" + std::to_string(bins[i].lower_bound);
 //     std::string upper = (bins[i].upper_bound == std::numeric_limits<double>::infinity()) ? "+Inf]" : std::to_string(bins[i].upper_bound) + ")";
 //     bin_names[i] = lower + ";" + upper;
 //     count[i] = bins[i].pos_count + bins[i].neg_count;
 //     pos[i] = bins[i].pos_count;
 //     neg[i] = bins[i].neg_count;
 //   }
 //   
 //   // Create List for bins
 //   List bin_lst = List::create(
 //     Named("bin") = bin_names,
 //     Named("woe") = woe,
 //     Named("iv") = iv_bin,
 //     Named("count") = count,
 //     Named("count_pos") = pos,
 //     Named("count_neg") = neg);
 //   
 //   // Create List for woe vector feature
 //   List woe_lst = List::create(
 //     Named("woefeature") = feature_woe
 //   );
 //   
 //   // Attrib class for compatibility with data.table in memory superfast tables
 //   bin_lst.attr("class") = CharacterVector::create("data.table", "data.frame");
 //   woe_lst.attr("class") = CharacterVector::create("data.table", "data.frame");
 //   
 //   // Return output
 //   List output_list = List::create(
 //     Named("woefeature") = woe_lst,
 //     Named("woebin") = bin_lst
 //   // Named("woe") = woe,
 //   // Named("iv") = total_iv,
 //   // Named("pos") = pos,
 //   // Named("neg") = neg
 //   );
 //   return output_list;
 // }
 
 // Helper function to calculate quantiles
 std::vector<double> calculate_quantiles(const std::vector<double>& data, const std::vector<double>& probs) {
   std::vector<double> sorted_data = data;
   std::sort(sorted_data.begin(), sorted_data.end());
   
   std::vector<double> result;
   for (double p : probs) {
     double h = (sorted_data.size() - 1) * p;
     int i = static_cast<int>(h);
     double v = sorted_data[i];
     if (h > i) {
       v += (h - i) * (sorted_data[i + 1] - v);
     }
     result.push_back(v);
   }
   return result;
 }
 
 //' Performs optimal binning of a numeric variable for Weight of Evidence (WoE) and Information Value (IV) using the Monotonic Optimal Binning (MOB) approach
 //'
 //' This function processes a numeric variable by removing missing values and creating pre-bins based on unique values. It iteratively merges or splits bins to ensure monotonicity of event rates, with constraints on the minimum number of bad events (\code{min_bads}) and the number of pre-bins. The function also calculates WoE and IV for the generated bins.
 //'
 //' @param target Integer vector representing the binary target variable, where 1 indicates a positive event (e.g., default) and 0 indicates a negative event (e.g., non-default).
 //' @param feature Numeric vector representing the numeric variable to be binned.
 //' @param min_bins (Optional) Minimum number of bins to generate. Default is 2.
 //' @param max_bins (Optional) Maximum number of bins to generate. Default is 7.
 //' @param bin_cutoff (Optional) Cutoff value that determines the frequency of values to define pre-bins. Default is 0.05.
 //' @param min_bads (Optional) Minimum proportion of bad events (positive target events) that a bin must contain. Default is 0.05.
 //' @param max_n_prebins (Optional) Maximum number of pre-bins to consider before final binning. Default is 20.
 //'
 //' @return A list with the following elements:
 //' \itemize{
 //'   \item \code{feature_woe}: Numeric vector with the WoE assigned to each instance of the processed numeric variable.
 //'   \item \code{bin}: DataFrame with the generated bins, containing the following fields:
 //'     \itemize{
 //'       \item \code{bin}: String representing the range of values for each bin.
 //'       \item \code{woe}: Weight of Evidence (WoE) for each bin.
 //'       \item \code{iv}: Information Value (IV) for each bin.
 //'       \item \code{count}: Total number of observations in each bin.
 //'       \item \code{count_pos}: Count of positive events in each bin.
 //'       \item \code{count_neg}: Count of negative events in each bin.
 //'     }
 //'   \item \code{woe}: Numeric vector with the WoE for each bin.
 //'   \item \code{iv}: Total Information Value (IV) calculated for the variable.
 //'   \item \code{pos}: Vector with the count of positive events in each bin.
 //'   \item \code{neg}: Vector with the count of negative events in each bin.
 //' }
 //'
 //'
 // [[Rcpp::export]]
 Rcpp::List OptimalBinningNumericMOB(Rcpp::IntegerVector target, Rcpp::NumericVector feature,
                                     int min_bins = 2, int max_bins = 7, double bin_cutoff = 0.05,
                                     double min_bads = 0.05, int max_n_prebins = 20) {
   // Input validation
   if (target.size() != feature.size()) {
     Rcpp::stop("Length of target and feature must be the same.");
   }
   if (min_bins < 2 || max_bins < min_bins) {
     Rcpp::stop("Invalid min_bins or max_bins values.");
   }
   if (bin_cutoff <= 0 || bin_cutoff >= 1) {
     Rcpp::stop("bin_cutoff must be between 0 and 1.");
   }
   if (min_bads <= 0 || min_bads >= 1) {
     Rcpp::stop("min_bads must be between 0 and 1.");
   }
   if (max_n_prebins < min_bins) {
     Rcpp::stop("max_n_prebins must be at least equal to min_bins.");
   }
   
   int N = target.size();
   
   // Remove missing values
   std::vector<double> feature_clean;
   std::vector<int> target_clean;
   feature_clean.reserve(N);
   target_clean.reserve(N);
   for (int i = 0; i < N; ++i) {
     if (!NumericVector::is_na(feature[i]) && !IntegerVector::is_na(target[i])) {
       feature_clean.push_back(feature[i]);
       target_clean.push_back(target[i]);
     }
   }
   
   int N_clean = feature_clean.size();
   if (N_clean == 0) {
     Rcpp::stop("No valid data after removing missing values.");
   }
   
   // Create pre-bins based on quantiles
   int num_prebins = std::min(max_n_prebins, N_clean);
   std::vector<double> cutpoints;
   
   if (num_prebins > 1) {
     std::vector<double> probs;
     for (int i = 1; i < num_prebins; ++i) {
       probs.push_back(static_cast<double>(i) / num_prebins);
     }
     
     std::vector<double> prebin_edges = calculate_quantiles(feature_clean, probs);
     
     // Ensure unique edges
     std::set<double> unique_edges(prebin_edges.begin(), prebin_edges.end());
     cutpoints.assign(unique_edges.begin(), unique_edges.end());
     std::sort(cutpoints.begin(), cutpoints.end());
   }
   
   // Initialize bins
   struct Bin {
     double lower_bound;
     double upper_bound;
     int pos_count;
     int neg_count;
     double event_rate;
   };
   
   std::vector<Bin> bins;
   
   // Initial bin edges
   std::vector<double> bin_edges;
   bin_edges.push_back(-std::numeric_limits<double>::infinity());
   bin_edges.insert(bin_edges.end(), cutpoints.begin(), cutpoints.end());
   bin_edges.push_back(std::numeric_limits<double>::infinity());
   
   // Initialize bins with counts
   int num_bins = bin_edges.size() - 1;
   bins.resize(num_bins);
   for (int i = 0; i < num_bins; ++i) {
     bins[i].lower_bound = bin_edges[i];
     bins[i].upper_bound = bin_edges[i + 1];
     bins[i].pos_count = 0;
     bins[i].neg_count = 0;
   }
   
   // Assign data to bins
   for (int i = 0; i < N_clean; ++i) {
     double value = feature_clean[i];
     int tgt = target_clean[i];
     int bin_idx = std::lower_bound(bin_edges.begin(), bin_edges.end(), value) - bin_edges.begin() - 1;
     if (bin_idx >= num_bins) bin_idx = num_bins - 1;
     
     if (tgt == 1) {
       bins[bin_idx].pos_count++;
     } else {
       bins[bin_idx].neg_count++;
     }
   }
   
   // Compute event rates
   int total_pos = 0;
   int total_neg = 0;
   for (int i = 0; i < num_bins; ++i) {
     int bin_total = bins[i].pos_count + bins[i].neg_count;
     bins[i].event_rate = (bin_total > 0) ? static_cast<double>(bins[i].pos_count) / bin_total : 0.0;
     total_pos += bins[i].pos_count;
     total_neg += bins[i].neg_count;
   }
   
   // Merge bins based on size constraints
   auto merge_bins = [&](int idx1, int idx2) {
     bins[idx1].upper_bound = bins[idx2].upper_bound;
     bins[idx1].pos_count += bins[idx2].pos_count;
     bins[idx1].neg_count += bins[idx2].neg_count;
     int bin_total = bins[idx1].pos_count + bins[idx1].neg_count;
     bins[idx1].event_rate = (bin_total > 0) ? static_cast<double>(bins[idx1].pos_count) / bin_total : 0.0;
     bins.erase(bins.begin() + idx2);
   };
   
   // Ensure minimum bin size and min_bads
   bool bins_merged = true;
   while (bins_merged && bins.size() > static_cast<size_t>(min_bins)) {
     bins_merged = false;
     for (size_t i = 0; i < bins.size(); ++i) {
       int bin_total = bins[i].pos_count + bins[i].neg_count;
       double bin_size = static_cast<double>(bin_total) / N_clean;
       double bad_rate = static_cast<double>(bins[i].pos_count) / total_pos;
       if (bin_size < bin_cutoff || bad_rate < min_bads) {
         // Merge with adjacent bin
         size_t merge_idx = (i == 0) ? i + 1 : i - 1;
         if (merge_idx >= bins.size()) continue;
         merge_bins(std::min(i, merge_idx), std::max(i, merge_idx));
         bins_merged = true;
         break;
       }
     }
   }
   
   // Ensure monotonicity
   auto is_monotonic = [](const std::vector<Bin>& bins) {
     bool increasing = true, decreasing = true;
     for (size_t i = 1; i < bins.size(); ++i) {
       if (bins[i].event_rate < bins[i - 1].event_rate) increasing = false;
       if (bins[i].event_rate > bins[i - 1].event_rate) decreasing = false;
     }
     return increasing || decreasing;
   };
   
   while (!is_monotonic(bins) && bins.size() > static_cast<size_t>(min_bins)) {
     // Find the bin to merge to enforce monotonicity
     size_t merge_idx = 0;
     double min_diff = std::numeric_limits<double>::max();
     for (size_t i = 0; i < bins.size() - 1; ++i) {
       double diff = std::abs(bins[i].event_rate - bins[i + 1].event_rate);
       if (diff < min_diff) {
         min_diff = diff;
         merge_idx = i;
       }
     }
     // Merge bins[merge_idx] and bins[merge_idx + 1]
     merge_bins(merge_idx, merge_idx + 1);
   }
   
   // Ensure the number of bins is within the specified range
   while (bins.size() > static_cast<size_t>(max_bins)) {
     // Find the pair of adjacent bins with the smallest difference in event rates
     size_t merge_idx = 0;
     double min_diff = std::numeric_limits<double>::max();
     for (size_t i = 0; i < bins.size() - 1; ++i) {
       double diff = std::abs(bins[i].event_rate - bins[i + 1].event_rate);
       if (diff < min_diff) {
         min_diff = diff;
         merge_idx = i;
       }
     }
     // Merge the selected bins
     merge_bins(merge_idx, merge_idx + 1);
   }
   
   // Compute WoE and IV
   std::vector<double> woe(bins.size());
   std::vector<double> iv_bin(bins.size());
   double total_iv = 0.0;
   for (size_t i = 0; i < bins.size(); ++i) {
     double dist_pos = static_cast<double>(bins[i].pos_count) / total_pos;
     double dist_neg = static_cast<double>(bins[i].neg_count) / total_neg;
     dist_pos = std::max(dist_pos, 1e-10);
     dist_neg = std::max(dist_neg, 1e-10);
     woe[i] = std::log(dist_pos / dist_neg);
     iv_bin[i] = (dist_pos - dist_neg) * woe[i];
     total_iv += iv_bin[i];
   }
   
   // Map feature values to WoE
   NumericVector feature_woe(N);
   for (int i = 0; i < N; ++i) {
     double value = feature[i];
     if (NumericVector::is_na(value)) {
       feature_woe[i] = NA_REAL;
       continue;
     }
     int bin_idx = std::lower_bound(bin_edges.begin(), bin_edges.end(), value) - bin_edges.begin() - 1;
     if (bin_idx >= static_cast<int>(bins.size())) bin_idx = bins.size() - 1;
     feature_woe[i] = woe[bin_idx];
   }
   
   // Prepare bin output
   std::vector<std::string> bin_names(bins.size());
   std::vector<int> count(bins.size());
   std::vector<int> pos(bins.size());
   std::vector<int> neg(bins.size());
   
   for (size_t i = 0; i < bins.size(); ++i) {
     std::string lower = (bins[i].lower_bound == -std::numeric_limits<double>::infinity()) ? "[-Inf" : "[" + std::to_string(bins[i].lower_bound);
     std::string upper = (bins[i].upper_bound == std::numeric_limits<double>::infinity()) ? "+Inf]" : std::to_string(bins[i].upper_bound) + ")";
     bin_names[i] = lower + ";" + upper;
     count[i] = bins[i].pos_count + bins[i].neg_count;
     pos[i] = bins[i].pos_count;
     neg[i] = bins[i].neg_count;
   }
   
   // Create List for bins
   List bin_lst = List::create(
     Named("bin") = bin_names,
     Named("woe") = woe,
     Named("iv") = iv_bin,
     Named("count") = count,
     Named("count_pos") = pos,
     Named("count_neg") = neg
   );
   
   // Create List for woe vector feature
   List woe_lst = List::create(
     Named("woefeature") = feature_woe
   );
   
   // Attrib class for compatibility with data.table in memory superfast tables
   bin_lst.attr("class") = CharacterVector::create("data.table", "data.frame");
   woe_lst.attr("class") = CharacterVector::create("data.table", "data.frame");
   
   // Return output
   List output_list = List::create(
     Named("woefeature") = woe_lst,
     Named("woebin") = bin_lst,
     Named("total_iv") = total_iv
   );
   return output_list;
 }
 // Rcpp::List OptimalBinningNumericMOB(Rcpp::IntegerVector target, Rcpp::NumericVector feature, int min_bins = 2, int max_bins = 7, double bin_cutoff = 0.05, double min_bads = 0.05, int max_n_prebins = 20) {
 //   int N = target.size();
 //   if (feature.size() != N) {
 //     Rcpp::stop("Length of target and feature must be the same.");
 //   }
 //   
 //   // Parameters
 //   
 //   // Remove missing values
 //   NumericVector feature_clean;
 //   IntegerVector target_clean;
 //   for (int i = 0; i < N; ++i) {
 //     if (!NumericVector::is_na(feature[i]) && !NumericVector::is_na(target[i])) {
 //       feature_clean.push_back(feature[i]);
 //       target_clean.push_back(target[i]);
 //     }
 //   }
 //   
 //   int N_clean = feature_clean.size();
 //   if (N_clean == 0) {
 //     Rcpp::stop("No valid data after removing missing values.");
 //   }
 //   
 //   // Create pre-bins based on quantiles
 //   int num_prebins = std::min(max_n_prebins, N_clean);
 //   std::vector<double> cutpoints;
 //   
 //   if (num_prebins > 1) {
 //     double step = 1.0 / num_prebins;
 //     // Exclude 0 and 1
 //     std::vector<double> probs;
 //     for (int i = 1; i < num_prebins; ++i) {
 //       probs.push_back(i * step);
 //     }
 //     NumericVector quantiles = wrap(probs);
 //     
 //     // Call R's quantile function
 //     Environment stats("package:stats");
 //     Function quantile_func = stats["quantile"];
 //     NumericVector prebin_edges = quantile_func(feature_clean, quantiles);
 //     
 //     // Ensure unique edges
 //     std::set<double> unique_edges(prebin_edges.begin(), prebin_edges.end());
 //     cutpoints.assign(unique_edges.begin(), unique_edges.end());
 //     std::sort(cutpoints.begin(), cutpoints.end());
 //   }
 //   
 //   // Initialize bins
 //   struct Bin {
 //     double lower_bound;
 //     double upper_bound;
 //     int pos_count;
 //     int neg_count;
 //     double event_rate;
 //   };
 //   
 //   std::vector<Bin> bins;
 //   
 //   // Initial bin edges
 //   std::vector<double> bin_edges;
 //   bin_edges.push_back(-std::numeric_limits<double>::infinity());
 //   bin_edges.insert(bin_edges.end(), cutpoints.begin(), cutpoints.end());
 //   bin_edges.push_back(std::numeric_limits<double>::infinity());
 //   
 //   // Initialize bins with counts
 //   int num_bins = bin_edges.size() - 1;
 //   bins.resize(num_bins);
 //   for (int i = 0; i < num_bins; ++i) {
 //     bins[i].lower_bound = bin_edges[i];
 //     bins[i].upper_bound = bin_edges[i + 1];
 //     bins[i].pos_count = 0;
 //     bins[i].neg_count = 0;
 //   }
 //   
 //   // Assign data to bins
 //   for (int i = 0; i < N_clean; ++i) {
 //     double value = feature_clean[i];
 //     int tgt = target_clean[i];
 //     int bin_idx = 0;
 //     while (bin_idx < num_bins && value > bins[bin_idx].upper_bound) {
 //       bin_idx++;
 //     }
 //     if (bin_idx >= num_bins) bin_idx = num_bins - 1;
 //     
 //     if (tgt == 1) {
 //       bins[bin_idx].pos_count++;
 //     } else {
 //       bins[bin_idx].neg_count++;
 //     }
 //   }
 //   
 //   // Compute event rates
 //   int total_pos = 0;
 //   int total_neg = 0;
 //   for (int i = 0; i < num_bins; ++i) {
 //     int bin_total = bins[i].pos_count + bins[i].neg_count;
 //     if (bin_total > 0) {
 //       bins[i].event_rate = (double)bins[i].pos_count / bin_total;
 //     } else {
 //       bins[i].event_rate = 0.0;
 //     }
 //     total_pos += bins[i].pos_count;
 //     total_neg += bins[i].neg_count;
 //   }
 //   
 //   // Merge bins based on size constraints
 //   auto merge_bins = [&](int idx1, int idx2) {
 //     bins[idx1].upper_bound = bins[idx2].upper_bound;
 //     bins[idx1].pos_count += bins[idx2].pos_count;
 //     bins[idx1].neg_count += bins[idx2].neg_count;
 //     int bin_total = bins[idx1].pos_count + bins[idx1].neg_count;
 //     if (bin_total > 0) {
 //       bins[idx1].event_rate = (double)bins[idx1].pos_count / bin_total;
 //     } else {
 //       bins[idx1].event_rate = 0.0;
 //     }
 //     bins.erase(bins.begin() + idx2);
 //   };
 //   
 //   // Ensure minimum bin size and min_bads
 //   bool bins_merged = true;
 //   while (bins_merged && bins.size() > (size_t)min_bins) {
 //     bins_merged = false;
 //     for (size_t i = 0; i < bins.size(); ++i) {
 //       int bin_total = bins[i].pos_count + bins[i].neg_count;
 //       double bin_size = (double)bin_total / N_clean;
 //       double bad_rate = (double)bins[i].pos_count / N_clean;
 //       if (bin_size < bin_cutoff || bad_rate < min_bads) {
 //         // Merge with adjacent bin
 //         size_t merge_idx = (i == 0) ? i + 1 : i - 1;
 //         if (merge_idx >= bins.size()) continue;
 //         if (i > merge_idx) {
 //           merge_bins(merge_idx, i);
 //         } else {
 //           merge_bins(i, merge_idx);
 //         }
 //         bins_merged = true;
 //         break;
 //       }
 //     }
 //   }
 //   
 //   // Ensure monotonicity
 //   auto is_monotonic = [](const std::vector<Bin>& bins) {
 //     bool increasing = true, decreasing = true;
 //     for (size_t i = 1; i < bins.size(); ++i) {
 //       if (bins[i].event_rate < bins[i - 1].event_rate) increasing = false;
 //       if (bins[i].event_rate > bins[i - 1].event_rate) decreasing = false;
 //     }
 //     return increasing || decreasing;
 //   };
 //   
 //   while (!is_monotonic(bins) && bins.size() > (size_t)min_bins) {
 //     // Find the bin to merge to enforce monotonicity
 //     size_t merge_idx = 0;
 //     double min_diff = std::numeric_limits<double>::max();
 //     for (size_t i = 0; i < bins.size() - 1; ++i) {
 //       if (bins[i + 1].event_rate < bins[i].event_rate) {
 //         double diff = bins[i].event_rate - bins[i + 1].event_rate;
 //         if (diff < min_diff) {
 //           min_diff = diff;
 //           merge_idx = i;
 //         }
 //       }
 //     }
 //     // Merge bins[merge_idx] and bins[merge_idx + 1]
 //     merge_bins(merge_idx, merge_idx + 1);
 //   }
 //   
 //   // Compute WoE and IV
 //   std::vector<double> woe(bins.size());
 //   std::vector<double> iv_bin(bins.size());
 //   double total_iv = 0.0;
 //   for (size_t i = 0; i < bins.size(); ++i) {
 //     double dist_pos = (double)bins[i].pos_count / total_pos;
 //     double dist_neg = (double)bins[i].neg_count / total_neg;
 //     if (dist_pos == 0) dist_pos = 1e-10;
 //     if (dist_neg == 0) dist_neg = 1e-10;
 //     woe[i] = log(dist_pos / dist_neg);
 //     iv_bin[i] = (dist_pos - dist_neg) * woe[i];
 //     total_iv += iv_bin[i];
 //   }
 //   
 //   // Map feature values to WoE
 //   NumericVector feature_woe(N);
 //   for (int i = 0; i < N; ++i) {
 //     double value = feature[i];
 //     if (NumericVector::is_na(value)) {
 //       feature_woe[i] = NA_REAL;
 //       continue;
 //     }
 //     int bin_idx = 0;
 //     while (bin_idx < (int)bins.size() && value > bins[bin_idx].upper_bound) {
 //       bin_idx++;
 //     }
 //     if (bin_idx >= (int)bins.size()) bin_idx = bins.size() - 1;
 //     feature_woe[i] = woe[bin_idx];
 //   }
 //   
 //   // Prepare bin output
 //   std::vector<std::string> bin_names(bins.size());
 //   std::vector<int> count(bins.size());
 //   std::vector<int> pos(bins.size());
 //   std::vector<int> neg(bins.size());
 //   
 //   for (size_t i = 0; i < bins.size(); ++i) {
 //     std::string lower = (bins[i].lower_bound == -std::numeric_limits<double>::infinity()) ? "[-Inf" : "[" + std::to_string(bins[i].lower_bound);
 //     std::string upper = (bins[i].upper_bound == std::numeric_limits<double>::infinity()) ? "+Inf]" : std::to_string(bins[i].upper_bound) + ")";
 //     bin_names[i] = lower + ";" + upper;
 //     count[i] = bins[i].pos_count + bins[i].neg_count;
 //     pos[i] = bins[i].pos_count;
 //     neg[i] = bins[i].neg_count;
 //   }
 //   
 //   // Create List for bins
 //   List bin_lst = List::create(
 //     Named("bin") = bin_names,
 //     Named("woe") = woe,
 //     Named("iv") = iv_bin,
 //     Named("count") = count,
 //     Named("count_pos") = pos,
 //     Named("count_neg") = neg);
 //   
 //   // Create List for woe vector feature
 //   List woe_lst = List::create(
 //     Named("woefeature") = feature_woe
 //   );
 //   
 //   // Attrib class for compatibility with data.table in memory superfast tables
 //   bin_lst.attr("class") = CharacterVector::create("data.table", "data.frame");
 //   woe_lst.attr("class") = CharacterVector::create("data.table", "data.frame");
 //   
 //   // Return output
 //   List output_list = List::create(
 //     Named("woefeature") = woe_lst,
 //     Named("woebin") = bin_lst
 //   // Named("woe") = woe,
 //   // Named("iv") = total_iv,
 //   // Named("pos") = pos,
 //   // Named("neg") = neg
 //   );
 //   return output_list;
 // }
 // 
 
 // Structure to represent each bin
 struct Bin {
   double lower_bound;
   double upper_bound;
   int count;
   int count_pos;
   int count_neg;
 };
 
 // Function to compute the p-value using the chi-squared test
 // double compute_p_value(Bin& bin1, Bin& bin2) {
 //   double N = bin1.count + bin2.count;
 //   double A = bin1.count_pos;
 //   double B = bin1.count_neg;
 //   double C = bin2.count_pos;
 //   double D = bin2.count_neg;
 //   
 //   double pos_total = A + C;
 //   double neg_total = B + D;
 //   
 //   double bin1_total = A + B;
 //   double bin2_total = C + D;
 //   
 //   double E1_pos = bin1_total * pos_total / N;
 //   double E1_neg = bin1_total * neg_total / N;
 //   double E2_pos = bin2_total * pos_total / N;
 //   double E2_neg = bin2_total * neg_total / N;
 //   
 //   double chi2 = 0.0;
 //   
 //   if (E1_pos > 0) chi2 += pow(A - E1_pos, 2) / E1_pos;
 //   if (E1_neg > 0) chi2 += pow(B - E1_neg, 2) / E1_neg;
 //   if (E2_pos > 0) chi2 += pow(C - E2_pos, 2) / E2_pos;
 //   if (E2_neg > 0) chi2 += pow(D - E2_neg, 2) / E2_neg;
 //   
 //   double p_value = R::pchisq(chi2, 1.0, 0, 0);
 //   return p_value;
 // }
 
 double compute_p_value(const Bin& bin1, const Bin& bin2) {
   int total = bin1.count + bin2.count;
   int total_pos = bin1.count_pos + bin2.count_pos;
   int total_neg = bin1.count_neg + bin2.count_neg;
   
   double expected_pos1 = bin1.count * (double)total_pos / total;
   double expected_neg1 = bin1.count * (double)total_neg / total;
   double expected_pos2 = bin2.count * (double)total_pos / total;
   double expected_neg2 = bin2.count * (double)total_neg / total;
   
   double chi_square = 
     std::pow(bin1.count_pos - expected_pos1, 2) / expected_pos1 +
     std::pow(bin1.count_neg - expected_neg1, 2) / expected_neg1 +
     std::pow(bin2.count_pos - expected_pos2, 2) / expected_pos2 +
     std::pow(bin2.count_neg - expected_neg2, 2) / expected_neg2;
   
   // Approximate p-value using chi-square distribution with 1 degree of freedom
   return 1 - R::pchisq(chi_square, 1, 1, 0);
 }
 
 //' Performs optimal binning of a numeric variable for Weight of Evidence (WoE) and Information Value (IV) using the ChiMerge algorithm
 //'
 //' This function processes a numeric variable by removing missing values and creating pre-bins based on unique values. It iteratively merges bins using the Chi-square test of independence, with a p-value threshold to determine whether bins should be merged. It ensures the minimum number of bad events (\code{min_bads}) is respected, while also calculating WoE and IV for the generated bins.
 //'
 //' @param target Integer vector representing the binary target variable, where 1 indicates a positive event (e.g., default) and 0 indicates a negative event (e.g., non-default).
 //' @param feature Numeric vector representing the numeric variable to be binned.
 //' @param min_bins (Optional) Minimum number of bins to generate. Default is 2.
 //' @param max_bins (Optional) Maximum number of bins to generate. Default is 7.
 //' @param pvalue_threshold (Optional) P-value threshold for the chi-square test used to determine whether to merge bins. Default is 0.05.
 //' @param bin_cutoff (Optional) Cutoff value that determines the frequency of values to define pre-bins. Default is 0.05.
 //' @param min_bads (Optional) Minimum proportion of bad events (positive target events) that a bin must contain. Default is 0.05.
 //' @param max_n_prebins (Optional) Maximum number of pre-bins to consider before final binning. Default is 20.
 //'
 //' @return A list with the following elements:
 //' \itemize{
 //'   \item \code{feature_woe}: Numeric vector with the WoE assigned to each instance of the processed numeric variable.
 //'   \item \code{bin}: DataFrame with the generated bins, containing the following fields:
 //'     \itemize{
 //'       \item \code{bin}: String representing the range of values for each bin.
 //'       \item \code{woe}: Weight of Evidence (WoE) for each bin.
 //'       \item \code{iv}: Information Value (IV) for each bin.
 //'       \item \code{count}: Total number of observations in each bin.
 //'       \item \code{count_pos}: Count of positive events in each bin.
 //'       \item \code{count_neg}: Count of negative events in each bin.
 //'     }
 //'   \item \code{woe}: Numeric vector with the WoE for each bin.
 //'   \item \code{iv}: Total Information Value (IV) calculated for the variable.
 //'   \item \code{pos}: Vector with the count of positive events in each bin.
 //'   \item \code{neg}: Vector with the count of negative events in each bin.
 //' }
 //'
 //'
 // [[Rcpp::export]]
 List OptimalBinningNumericChiMerge(IntegerVector target, NumericVector feature,
                                    int min_bins = 2, int max_bins = 7,
                                    double pvalue_threshold = 0.05, double bin_cutoff = 0.05,
                                    double min_bads = 0.05, int max_n_prebins = 20) {
   if (target.size() != feature.size()) {
     throw std::invalid_argument("Target and feature must have the same length");
   }
   
   if (min_bins < 2 || max_bins < min_bins) {
     throw std::invalid_argument("Invalid min_bins or max_bins");
   }
   
   int n = feature.size();
   int total_bads = std::accumulate(target.begin(), target.end(), 0);
   int total_goods = n - total_bads;
   
   if (total_bads == 0 || total_goods == 0) {
     throw std::invalid_argument("Target must contain both 0s and 1s");
   }
   
   double min_bads_count = min_bads * total_bads;
   double bin_cutoff_count = bin_cutoff * n;
   
   // Create indices and sort them based on feature values
   std::vector<int> indices(n);
   std::iota(indices.begin(), indices.end(), 0);
   
   std::sort(indices.begin(), indices.end(), [&](int i1, int i2) {
     return feature[i1] < feature[i2];
   });
   
   // Determine initial bins
   std::vector<double> unique_values = as<std::vector<double>>(unique(feature));
   int num_unique_values = unique_values.size();
   
   std::vector<Bin> bins;
   if (num_unique_values <= max_n_prebins) {
     // Each unique value is its own bin
     for (int i = 0; i < n; ++i) {
       int idx = indices[i];
       if (bins.empty() || feature[idx] != bins.back().upper_bound) {
         Bin bin;
         bin.lower_bound = feature[idx];
         bin.upper_bound = feature[idx];
         bin.count = 1;
         bin.count_pos = (target[idx] == 1) ? 1 : 0;
         bin.count_neg = (target[idx] == 0) ? 1 : 0;
         bins.push_back(bin);
       } else {
         bins.back().count++;
         if (target[idx] == 1) {
           bins.back().count_pos++;
         } else {
           bins.back().count_neg++;
         }
       }
     }
   } else {
     // Pre-bin into max_n_prebins bins
     int bin_size = n / max_n_prebins;
     int remainder = n % max_n_prebins;
     
     for (int i = 0; i < max_n_prebins; ++i) {
       int start = i * bin_size + std::min(i, remainder);
       int end = (i + 1) * bin_size + std::min(i + 1, remainder) - 1;
       
       Bin bin;
       bin.lower_bound = feature[indices[start]];
       bin.upper_bound = feature[indices[end]];
       bin.count = end - start + 1;
       bin.count_pos = 0;
       bin.count_neg = 0;
       
       for (int j = start; j <= end; ++j) {
         if (target[indices[j]] == 1)
           bin.count_pos++;
         else
           bin.count_neg++;
       }
       bins.push_back(bin);
     }
   }
   
   // Compute initial p-values between bins
   std::vector<double> p_values(bins.size() - 1);
#pragma omp parallel for
   for (size_t i = 0; i < bins.size() - 1; ++i) {
     p_values[i] = compute_p_value(bins[i], bins[i + 1]);
   }
   
   // Initial merging to enforce constraints
   bool merged;
   do {
     merged = false;
     for (size_t i = 0; i < bins.size(); ++i) {
       if (bins[i].count_pos < min_bads_count || bins[i].count < bin_cutoff_count) {
         if (i == 0 && bins.size() > 1) {
           // Merge with next bin
           bins[0].upper_bound = bins[1].upper_bound;
           bins[0].count += bins[1].count;
           bins[0].count_pos += bins[1].count_pos;
           bins[0].count_neg += bins[1].count_neg;
           bins.erase(bins.begin() + 1);
           p_values.erase(p_values.begin());
         } else if (i == bins.size() - 1 && bins.size() > 1) {
           // Merge with previous bin
           bins[i-1].upper_bound = bins[i].upper_bound;
           bins[i-1].count += bins[i].count;
           bins[i-1].count_pos += bins[i].count_pos;
           bins[i-1].count_neg += bins[i].count_neg;
           bins.erase(bins.begin() + i);
           p_values.erase(p_values.end() - 1);
         } else if (i > 0 && i < bins.size() - 1) {
           // Merge with the bin that has higher p-value
           if (p_values[i-1] > p_values[i]) {
             bins[i-1].upper_bound = bins[i].upper_bound;
             bins[i-1].count += bins[i].count;
             bins[i-1].count_pos += bins[i].count_pos;
             bins[i-1].count_neg += bins[i].count_neg;
             bins.erase(bins.begin() + i);
             p_values.erase(p_values.begin() + i - 1);
           } else {
             bins[i].upper_bound = bins[i+1].upper_bound;
             bins[i].count += bins[i+1].count;
             bins[i].count_pos += bins[i+1].count_pos;
             bins[i].count_neg += bins[i+1].count_neg;
             bins.erase(bins.begin() + i + 1);
             p_values.erase(p_values.begin() + i);
           }
         }
         merged = true;
         break;
       }
     }
   } while (merged);
   
   // Main merging process based on p-values
   while (bins.size() > (size_t)min_bins) {
     auto max_p_iter = std::max_element(p_values.begin(), p_values.end());
     double max_p_value = *max_p_iter;
     size_t idx = std::distance(p_values.begin(), max_p_iter);
     
     if (bins.size() <= (size_t)max_bins && max_p_value <= pvalue_threshold) {
       break;
     }
     
     // Merge bins[idx] and bins[idx+1]
     bins[idx].upper_bound = bins[idx + 1].upper_bound;
     bins[idx].count += bins[idx + 1].count;
     bins[idx].count_pos += bins[idx + 1].count_pos;
     bins[idx].count_neg += bins[idx + 1].count_neg;
     
     bins.erase(bins.begin() + idx + 1);
     p_values.erase(p_values.begin() + idx);
     
     // Update p-values
     if (idx > 0) {
       p_values[idx - 1] = compute_p_value(bins[idx - 1], bins[idx]);
     }
     if (idx < p_values.size()) {
       p_values[idx] = compute_p_value(bins[idx], bins[idx + 1]);
     }
   }
   
   // Enforce monotonicity while respecting min_bins
   std::vector<double> event_rates(bins.size());
   for (size_t i = 0; i < bins.size(); ++i) {
     event_rates[i] = (double)bins[i].count_pos / bins[i].count;
   }
   
   bool increasing = event_rates.front() < event_rates.back();
   bool monotonic = false;
   
   while (!monotonic && bins.size() > (size_t)min_bins) {
     monotonic = true;
     for (size_t i = 0; i < bins.size() - 1; ++i) {
       if ((increasing && event_rates[i] > event_rates[i + 1]) ||
           (!increasing && event_rates[i] < event_rates[i + 1])) {
         // Merge bins[i] and bins[i+1]
         bins[i].upper_bound = bins[i + 1].upper_bound;
         bins[i].count += bins[i + 1].count;
         bins[i].count_pos += bins[i + 1].count_pos;
         bins[i].count_neg += bins[i + 1].count_neg;
         
         bins.erase(bins.begin() + i + 1);
         event_rates[i] = (double)bins[i].count_pos / bins[i].count;
         event_rates.erase(event_rates.begin() + i + 1);
         
         monotonic = false;
         break;
       }
     }
   }
   
   // Compute WoE and IV
   std::vector<double> woe(bins.size());
   double iv = 0.0;
   
#pragma omp parallel for reduction(+:iv)
   for (size_t i = 0; i < bins.size(); ++i) {
     double dist_pos = std::max(1e-10, (double)bins[i].count_pos / total_bads);
     double dist_neg = std::max(1e-10, (double)bins[i].count_neg / total_goods);
     
     woe[i] = std::log(dist_pos / dist_neg);
     iv += (dist_pos - dist_neg) * woe[i];
   }
   
   // Map feature values to WoE
   std::vector<double> bin_upper_bounds(bins.size());
   for (size_t i = 0; i < bins.size(); ++i) {
     bin_upper_bounds[i] = bins[i].upper_bound;
   }
   
   NumericVector feature_woe(n);
#pragma omp parallel for
   for (int i = 0; i < n; ++i) {
     double val = feature[i];
     auto it = std::upper_bound(bin_upper_bounds.begin(), bin_upper_bounds.end(), val);
     int idx = it - bin_upper_bounds.begin();
     if (idx >= (int)bins.size()) idx = bins.size() - 1;
     feature_woe[i] = woe[idx];
   }
   
   // Prepare output
   std::vector<std::string> bin_names(bins.size());
   NumericVector bin_lower_bounds(bins.size());
   NumericVector bin_upper_bounds_output(bins.size());
   IntegerVector bin_count(bins.size());
   IntegerVector bin_count_pos(bins.size());
   IntegerVector bin_count_neg(bins.size());
   
   for (size_t i = 0; i < bins.size(); ++i) {
     bin_lower_bounds[i] = bins[i].lower_bound;
     bin_upper_bounds_output[i] = bins[i].upper_bound;
     bin_count[i] = bins[i].count;
     bin_count_pos[i] = bins[i].count_pos;
     bin_count_neg[i] = bins[i].count_neg;
     std::string lower_str = (i == 0) ? "[-Inf" : "[" + std::to_string(bin_lower_bounds[i]);
     std::string upper_str = (i == bins.size() - 1) ? "+Inf]" : std::to_string(bin_upper_bounds_output[i]) + ")";
     bin_names[i] = lower_str + ";" + upper_str;
   }
   
   DataFrame bin_df = DataFrame::create(
     Named("bin") = bin_names,
     Named("woe") = woe,
     Named("iv") = iv / bins.size(),  // IV per bin
     Named("count") = bin_count,
     Named("count_pos") = bin_count_pos,
     Named("count_neg") = bin_count_neg
   );
   
   // Create DataFrame for WoE vector feature
   DataFrame woe_df = DataFrame::create(
     Named("woefeature") = feature_woe
   );
   
   // Set class attributes for compatibility with data.table
   bin_df.attr("class") = CharacterVector::create("data.table", "data.frame");
   woe_df.attr("class") = CharacterVector::create("data.table", "data.frame");
   
   List output_list = List::create(
     Named("woefeature") = woe_df,
     Named("woebin") = bin_df,
     Named("total_iv") = iv
   );
   
   return output_list;
}
 
//  Rcpp::List OptimalBinningNumericChiMerge(Rcpp::IntegerVector target, Rcpp::NumericVector feature, int min_bins = 2, int max_bins = 7, double pvalue_threshold = 0.05, double bin_cutoff = 0.05, double min_bads = 0.05, int max_n_prebins = 20) {
//    int n = feature.size();
//    int total_bads = std::accumulate(target.begin(), target.end(), 0);
//    int total_goods = n - total_bads;
//    
//    double min_bads_count = min_bads * total_bads;
//    double bin_cutoff_count = bin_cutoff * n;
//    
//    // Create indices and sort them based on feature values
//    std::vector<int> indices(n);
//    for (int i = 0; i < n; ++i) indices[i] = i;
//    
//    std::sort(indices.begin(), indices.end(), [&](int i1, int i2) {
//      return feature[i1] < feature[i2];
//    });
//    
//    // Determine initial bins
//    NumericVector unique_values = Rcpp::unique(feature);
//    int num_unique_values = unique_values.size();
//    
//    std::vector<Bin> bins;
//    if (num_unique_values <= max_n_prebins) {
//      // Each unique value is its own bin
//      for (size_t idx = 0; idx < indices.size(); ++idx) {
//        Bin bin;
//        bin.lower_bound = feature[indices[idx]];
//        bin.upper_bound = feature[indices[idx]];
//        bin.count = 1;
//        if (target[indices[idx]] == 1) {
//          bin.count_pos = 1;
//          bin.count_neg = 0;
//        } else {
//          bin.count_pos = 0;
//          bin.count_neg = 1;
//        }
//        bins.push_back(bin);
//      }
//    } else {
//      // Pre-bin into max_n_prebins bins
//      int bin_size = n / max_n_prebins;
//      int remainder = n % max_n_prebins;
//      
//      std::vector<int> bin_start(max_n_prebins);
//      std::vector<int> bin_end(max_n_prebins);
//      int start = 0;
//      for (int i = 0; i < max_n_prebins; ++i) {
//        int b_size = bin_size + (i < remainder ? 1 : 0);
//        bin_start[i] = start;
//        bin_end[i] = start + b_size - 1;
//        start += b_size;
//      }
//      
//      for (int i = 0; i < max_n_prebins; ++i) {
//        int start_idx = bin_start[i];
//        int end_idx = bin_end[i];
//        
//        Bin bin;
//        bin.lower_bound = feature[indices[start_idx]];
//        bin.upper_bound = feature[indices[end_idx]];
//        bin.count = end_idx - start_idx + 1;
//        bin.count_pos = 0;
//        bin.count_neg = 0;
//        
//        for (int j = start_idx; j <= end_idx; ++j) {
//          if (target[indices[j]] == 1)
//            bin.count_pos++;
//          else
//            bin.count_neg++;
//        }
//        bins.push_back(bin);
//      }
//    }
//    
//    // Compute initial p-values between bins
//    std::vector<double> p_values(bins.size() - 1);
//    for (size_t i = 0; i < bins.size() - 1; ++i) {
//      double p_value = compute_p_value(bins[i], bins[i + 1]);
//      p_values[i] = p_value;
//    }
//    
//    // Initial merging to enforce constraints
//    bool merged = true;
//    while (merged) {
//      merged = false;
//      for (size_t i = 0; i < bins.size(); ++i) {
//        bool need_merge = false;
//        if (bins[i].count_pos < min_bads_count || bins[i].count < bin_cutoff_count) {
//          need_merge = true;
//        }
//        
//        if (need_merge) {
//          size_t idx = i;
//          if (idx == 0) {
//            // Merge with next bin
//            bins[idx].upper_bound = bins[idx + 1].upper_bound;
//            bins[idx].count += bins[idx + 1].count;
//            bins[idx].count_pos += bins[idx + 1].count_pos;
//            bins[idx].count_neg += bins[idx + 1].count_neg;
//            
//            bins.erase(bins.begin() + idx + 1);
//            p_values.erase(p_values.begin() + idx);
//            
//            if (idx < p_values.size()) {
//              double p_value = compute_p_value(bins[idx], bins[idx + 1]);
//              p_values[idx] = p_value;
//            }
//          } else if (idx == bins.size() - 1) {
//            // Merge with previous bin
//            bins[idx - 1].upper_bound = bins[idx].upper_bound;
//            bins[idx - 1].count += bins[idx].count;
//            bins[idx - 1].count_pos += bins[idx].count_pos;
//            bins[idx - 1].count_neg += bins[idx].count_neg;
//            
//            bins.erase(bins.begin() + idx);
//            p_values.erase(p_values.begin() + idx - 1);
//            
//            if (idx - 1 > 0) {
//              double p_value = compute_p_value(bins[idx - 2], bins[idx - 1]);
//              p_values[idx - 2] = p_value;
//            }
//          } else {
//            // Decide whether to merge with left or right bin
//            double p_left = p_values[idx - 1];
//            double p_right = p_values[idx];
//            
//            if (p_left >= p_right) {
//              // Merge with previous bin
//              bins[idx - 1].upper_bound = bins[idx].upper_bound;
//              bins[idx - 1].count += bins[idx].count;
//              bins[idx - 1].count_pos += bins[idx].count_pos;
//              bins[idx - 1].count_neg += bins[idx].count_neg;
//              
//              bins.erase(bins.begin() + idx);
//              p_values.erase(p_values.begin() + idx - 1);
//              
//              if (idx - 1 > 0) {
//                double p_value = compute_p_value(bins[idx - 2], bins[idx - 1]);
//                p_values[idx - 2] = p_value;
//              }
//              if (idx - 1 < p_values.size()) {
//                double p_value = compute_p_value(bins[idx - 1], bins[idx]);
//                p_values[idx - 1] = p_value;
//              }
//            } else {
//              // Merge with next bin
//              bins[idx].upper_bound = bins[idx + 1].upper_bound;
//              bins[idx].count += bins[idx + 1].count;
//              bins[idx].count_pos += bins[idx + 1].count_pos;
//              bins[idx].count_neg += bins[idx + 1].count_neg;
//              
//              bins.erase(bins.begin() + idx + 1);
//              p_values.erase(p_values.begin() + idx);
//              
//              if (idx < p_values.size()) {
//                double p_value = compute_p_value(bins[idx], bins[idx + 1]);
//                p_values[idx] = p_value;
//              }
//            }
//          }
//          merged = true;
//          break;
//        }
//      }
//    }
//    
//    // Main merging process based on p-values
//    while (true) {
//      if (bins.size() <= (size_t)min_bins) {
//        break;
//      }
//      
//      auto max_p_iter = std::max_element(p_values.begin(), p_values.end());
//      double max_p_value = *max_p_iter;
//      size_t idx = std::distance(p_values.begin(), max_p_iter);
//      
//      if (bins.size() <= (size_t)max_bins && max_p_value <= pvalue_threshold) {
//        break;
//      }
//      
//      // Merge bins[idx] and bins[idx+1]
//      bins[idx].upper_bound = bins[idx + 1].upper_bound;
//      bins[idx].count += bins[idx + 1].count;
//      bins[idx].count_pos += bins[idx + 1].count_pos;
//      bins[idx].count_neg += bins[idx + 1].count_neg;
//      
//      bins.erase(bins.begin() + idx + 1);
//      p_values.erase(p_values.begin() + idx);
//      
//      // Update p-values
//      if (idx > 0) {
//        double p_value = compute_p_value(bins[idx - 1], bins[idx]);
//        p_values[idx - 1] = p_value;
//      }
//      if (idx < p_values.size()) {
//        double p_value = compute_p_value(bins[idx], bins[idx + 1]);
//        p_values[idx] = p_value;
//      }
//    }
//    
//    // Enforce monotonicity
//    std::vector<double> event_rates(bins.size());
//    for (size_t i = 0; i < bins.size(); ++i) {
//      event_rates[i] = (double)bins[i].count_pos / bins[i].count;
//    }
//    
//    bool increasing = event_rates.front() < event_rates.back();
//    bool monotonic = false;
//    
//    while (!monotonic) {
//      monotonic = true;
//      for (size_t i = 0; i < bins.size() - 1; ++i) {
//        if (increasing) {
//          if (event_rates[i] > event_rates[i + 1]) {
//            // Merge bins[i] and bins[i+1]
//            bins[i].upper_bound = bins[i + 1].upper_bound;
//            bins[i].count += bins[i + 1].count;
//            bins[i].count_pos += bins[i + 1].count_pos;
//            bins[i].count_neg += bins[i + 1].count_neg;
//            
//            bins.erase(bins.begin() + i + 1);
//            event_rates[i] = (double)bins[i].count_pos / bins[i].count;
//            event_rates.erase(event_rates.begin() + i + 1);
//            
//            monotonic = false;
//            break;
//          }
//        } else {
//          if (event_rates[i] < event_rates[i + 1]) {
//            // Merge bins[i] and bins[i+1]
//            bins[i].upper_bound = bins[i + 1].upper_bound;
//            bins[i].count += bins[i + 1].count;
//            bins[i].count_pos += bins[i + 1].count_pos;
//            bins[i].count_neg += bins[i + 1].count_neg;
//            
//            bins.erase(bins.begin() + i + 1);
//            event_rates[i] = (double)bins[i].count_pos / bins[i].count;
//            event_rates.erase(event_rates.begin() + i + 1);
//            
//            monotonic = false;
//            break;
//          }
//        }
//      }
//    }
//    
//    // Compute WoE and IV
//    std::vector<double> woe(bins.size());
//    double iv = 0.0;
//    
//    for (size_t i = 0; i < bins.size(); ++i) {
//      double dist_pos = bins[i].count_pos / (double)total_bads;
//      double dist_neg = bins[i].count_neg / (double)total_goods;
//      
//      if (dist_pos == 0.0) dist_pos = 1e-10;
//      if (dist_neg == 0.0) dist_neg = 1e-10;
//      
//      woe[i] = std::log(dist_pos / dist_neg);
//      iv += (dist_pos - dist_neg) * woe[i];
//    }
//    
//    // Map feature values to WoE
//    std::vector<double> bin_upper_bounds(bins.size());
//    for (size_t i = 0; i < bins.size(); ++i) {
//      bin_upper_bounds[i] = bins[i].upper_bound;
//    }
//    
//    NumericVector feature_woe(n);
// #ifdef _OPENMP
// #pragma omp parallel for
// #endif
//    for (int i = 0; i < n; ++i) {
//      double val = feature[i];
//      auto it = std::upper_bound(bin_upper_bounds.begin(), bin_upper_bounds.end(), val);
//      int idx = it - bin_upper_bounds.begin();
//      if (idx >= (int)bins.size()) idx = bins.size() - 1;
//      feature_woe[i] = woe[idx];
//    }
//    
//    // Prepare output
//    std::vector<std::string> bin_names(bins.size());
//    std::vector<double> bin_lower_bounds(bins.size());
//    std::vector<double> bin_upper_bounds_output(bins.size());
//    std::vector<int> bin_count(bins.size());
//    std::vector<int> bin_count_pos(bins.size());
//    std::vector<int> bin_count_neg(bins.size());
//    
//    for (size_t i = 0; i < bins.size(); ++i) {
//      bin_lower_bounds[i] = bins[i].lower_bound;
//      bin_upper_bounds_output[i] = bins[i].upper_bound;
//      bin_count[i] = bins[i].count;
//      bin_count_pos[i] = bins[i].count_pos;
//      bin_count_neg[i] = bins[i].count_neg;
//      
//      std::string lower_str = (i == 0) ? "[-Inf" : "[" + std::to_string(bin_lower_bounds[i]);
//      std::string upper_str = (i == bins.size() - 1) ? "+Inf]" : std::to_string(bin_upper_bounds_output[i]) + ")";
//      bin_names[i] = lower_str + ";" + upper_str;
//    }
//    
//    List bin_lst = List::create(
//      Named("bin") = bin_names,
//      Named("woe") = woe,
//      Named("iv") = iv,
//      Named("count") = bin_count,
//      Named("count_pos") = bin_count_pos,
//      Named("count_neg") = bin_count_neg
//    );
//    
//    // Create List for woe vector feature
//    List woe_lst = List::create(
//      Named("woefeature") = feature_woe
//    );
//    
//    // Attrib class for compatibility with data.table in memory superfast tables
//    bin_lst.attr("class") = CharacterVector::create("data.table", "data.frame");
//    woe_lst.attr("class") = CharacterVector::create("data.table", "data.frame");
//    
//    List output_list = List::create(
//      Named("woefeature") = woe_lst,
//      Named("woebin") = bin_lst
//    // Named("woe") = woe,
//    // Named("iv") = total_iv,
//    // Named("pos") = pos,
//    // Named("neg") = neg
//    );
//    
//    return output_list;
//  }
 
 
 // Structure to represent an interval
 struct Interval {
   int start;
   int end;
   double entropy;
   double cut_point = std::numeric_limits<double>::infinity();
   int cut_index = -1;
   std::vector<Interval> children;
 };
 
 // Function to compute entropy
 double compute_entropy(const IntegerVector& target, int start, int end) {
   int count = end - start + 1;
   if (count == 0) return 0.0;
   
   int count_pos = 0;
   for (int i = start; i <= end; ++i) {
     if (target[i] == 1) ++count_pos;
   }
   int count_neg = count - count_pos;
   
   double p_pos = static_cast<double>(count_pos) / count;
   double p_neg = static_cast<double>(count_neg) / count;
   
   double entropy = 0.0;
   if (p_pos > 0.0) entropy -= p_pos * std::log2(p_pos);
   if (p_neg > 0.0) entropy -= p_neg * std::log2(p_neg);
   return entropy;
 }
 
 // Function to compute information gain
 double compute_info_gain(const IntegerVector& target, int start, int end, int split) {
   double entropy_parent = compute_entropy(target, start, end);
   int count_parent = end - start + 1;
   
   int count_left = split - start + 1;
   int count_right = end - split;
   
   double entropy_left = compute_entropy(target, start, split);
   double entropy_right = compute_entropy(target, split + 1, end);
   
   double info_gain = entropy_parent - 
     ((static_cast<double>(count_left) / count_parent) * entropy_left +
     (static_cast<double>(count_right) / count_parent) * entropy_right);
   
   return info_gain;
 }
 
 // Function to find optimal split
 bool find_optimal_split(const NumericVector& feature, const IntegerVector& target, int start, int end, Interval& interval) {
   if (start >= end) return false;
   
   std::vector<int> split_candidates;
   for (int i = start; i < end; ++i) {
     if (feature[i] != feature[i + 1]) {
       split_candidates.push_back(i);
     }
   }
   if (split_candidates.empty()) return false;
   
   double best_gain = -std::numeric_limits<double>::infinity();
   int best_split = -1;
   
#pragma omp parallel for reduction(max:best_gain)
   for (size_t i = 0; i < split_candidates.size(); ++i) {
     int split = split_candidates[i];
     double gain = compute_info_gain(target, start, end, split);
     if (gain > best_gain) {
#pragma omp critical
{
  if (gain > best_gain) {
    best_gain = gain;
    best_split = split;
  }
}
     }
   }
   
   if (best_split == -1) return false;
   
   // Apply MDL stopping criterion
   int count_parent = end - start + 1;
   double entropy_parent = compute_entropy(target, start, end);
   double entropy_left = compute_entropy(target, start, best_split);
   double entropy_right = compute_entropy(target, best_split + 1, end);
   
   int k = 2; // Number of classes (binary classification)
   double delta = std::log2(std::pow(3.0, k) - 2.0) - (k * entropy_parent - k * entropy_left - k * entropy_right);
   double threshold = (std::log2(count_parent - 1) + delta) / count_parent;
   
   if (best_gain > threshold) {
     interval.cut_point = (feature[best_split] + feature[best_split + 1]) / 2.0;
     interval.cut_index = best_split;
     return true;
   } else {
     return false;
   }
 }
 
 // Recursive partitioning function
 void mdlp_partition(const NumericVector& feature, const IntegerVector& target, int start, int end, Interval& interval, double bin_cutoff_count, double min_bads_count) {
   int count = end - start + 1;
   if (count <= 1) return;
   
   int count_pos = 0;
   for (int i = start; i <= end; ++i) {
     if (target[i] == 1) ++count_pos;
   }
   int count_neg = count - count_pos;
   
   if (count < bin_cutoff_count || count_pos < min_bads_count) {
     return;
   }
   
   if (find_optimal_split(feature, target, start, end, interval)) {
     Interval left_child;
     left_child.start = start;
     left_child.end = interval.cut_index;
     left_child.entropy = compute_entropy(target, start, interval.cut_index);
     
     Interval right_child;
     right_child.start = interval.cut_index + 1;
     right_child.end = end;
     right_child.entropy = compute_entropy(target, interval.cut_index + 1, end);
     
     mdlp_partition(feature, target, left_child.start, left_child.end, left_child, bin_cutoff_count, min_bads_count);
     mdlp_partition(feature, target, right_child.start, right_child.end, right_child, bin_cutoff_count, min_bads_count);
     
     interval.children.push_back(left_child);
     interval.children.push_back(right_child);
   }
 }
 
 void collect_intervals(const Interval& node, std::vector<Interval>& intervals) {
   if (node.children.empty()) {
     intervals.push_back(node);
   } else {
     for (const auto& child : node.children) {
       collect_intervals(child, intervals);
     }
   }
 }
 
 // Function to compute WoE and IV
 void compute_woe_iv(const IntegerVector& target, const std::vector<Interval>& intervals, NumericVector& woe_values, double& iv, IntegerVector& pos_counts, IntegerVector& neg_counts) {
   int total_bads = std::accumulate(target.begin(), target.end(), 0);
   int total_goods = target.size() - total_bads;
   
   woe_values = NumericVector(intervals.size());
   pos_counts = IntegerVector(intervals.size());
   neg_counts = IntegerVector(intervals.size());
   
   iv = 0.0;
   
   for (size_t idx = 0; idx < intervals.size(); ++idx) {
     const Interval& interval = intervals[idx];
     int count_pos = 0;
     for (int i = interval.start; i <= interval.end; ++i) {
       if (target[i] == 1) ++count_pos;
     }
     int count_neg = interval.end - interval.start + 1 - count_pos;
     
     pos_counts[idx] = count_pos;
     neg_counts[idx] = count_neg;
     
     double dist_pos = static_cast<double>(count_pos) / total_bads;
     double dist_neg = static_cast<double>(count_neg) / total_goods;
     
     if (dist_pos == 0.0) dist_pos = 1e-10;
     if (dist_neg == 0.0) dist_neg = 1e-10;
     
     woe_values[idx] = std::log(dist_pos / dist_neg);
     iv += (dist_pos - dist_neg) * woe_values[idx];
   }
 }

 
 // [[Rcpp::export]]
 List OptimalBinningNumericMDLP(IntegerVector target, NumericVector feature, int min_bins = 2, int max_bins = 7, double bin_cutoff = 0.05, double min_bads = 0.05, int max_n_prebins = 20) {
   // Input validation
   // Validate target values
   int N = target.size();
   for (int i = 0; i < N; ++i) {
     if (target[i] != 0 && target[i] != 1) {
       Rcpp::stop("Target must contain only 0s and 1s.");
     }
   }
   
   if (target.size() != feature.size()) {
     stop("Target and feature must have the same length");
   }
   if (min_bins > max_bins) {
     stop("min_bins must be less than or equal to max_bins");
   }
   
   int n = feature.size();
   int total_bads = sum(target);
   
   double min_bads_count = min_bads * total_bads;
   double bin_cutoff_count = bin_cutoff * n;
   
   // Create indices and sort them based on feature values
   IntegerVector indices = seq(0, n - 1);
   std::sort(indices.begin(), indices.end(), [&](int i1, int i2) {
     return feature[i1] < feature[i2];
   });
   
   // Sort feature and target accordingly
   NumericVector sorted_feature = feature[indices];
   IntegerVector sorted_target = target[indices];
   
   // Create root interval
   Interval root;
   root.start = 0;
   root.end = n - 1;
   root.entropy = compute_entropy(sorted_target, 0, n - 1);
   
   // Perform MDLP partitioning
   mdlp_partition(sorted_feature, sorted_target, 0, n - 1, root, bin_cutoff_count, min_bads_count);
   
   // Collect the final intervals
   std::vector<Interval> intervals;
   collect_intervals(root, intervals);
   
   // Sort intervals by start index
   std::sort(intervals.begin(), intervals.end(), [](const Interval& a, const Interval& b) {
     return a.start < b.start;
   });
   
   // Adjust number of bins if necessary
   while (intervals.size() > max_bins && intervals.size() > 2) {
     // Find the pair of adjacent intervals with the smallest difference in WoE
     double min_woe_diff = std::numeric_limits<double>::infinity();
     size_t merge_index = 0;
     for (size_t i = 0; i < intervals.size() - 1; ++i) {
       double woe1 = std::log(static_cast<double>(intervals[i].entropy) / (1 - intervals[i].entropy));
       double woe2 = std::log(static_cast<double>(intervals[i+1].entropy) / (1 - intervals[i+1].entropy));
       double woe_diff = std::abs(woe1 - woe2);
       if (woe_diff < min_woe_diff) {
         min_woe_diff = woe_diff;
         merge_index = i;
       }
     }
     
     // Merge the two intervals
     intervals[merge_index].end = intervals[merge_index + 1].end;
     intervals[merge_index].entropy = compute_entropy(sorted_target, intervals[merge_index].start, intervals[merge_index].end);
     intervals.erase(intervals.begin() + merge_index + 1);
   }
   
   // Ensure minimum number of bins
   while (intervals.size() < min_bins && intervals.size() < n) {
     // Find the interval with the highest entropy to split
     size_t split_index = 0;
     double max_entropy = -std::numeric_limits<double>::infinity();
     for (size_t i = 0; i < intervals.size(); ++i) {
       if (intervals[i].entropy > max_entropy) {
         max_entropy = intervals[i].entropy;
         split_index = i;
       }
     }
     
     // Split the interval
     int mid = (intervals[split_index].start + intervals[split_index].end) / 2;
     Interval new_interval;
     new_interval.start = mid + 1;
     new_interval.end = intervals[split_index].end;
     new_interval.entropy = compute_entropy(sorted_target, new_interval.start, new_interval.end);
     intervals[split_index].end = mid;
     intervals[split_index].entropy = compute_entropy(sorted_target, intervals[split_index].start, intervals[split_index].end);
     intervals.insert(intervals.begin() + split_index + 1, new_interval);
   }
   
   // Compute WoE and IV
   NumericVector woe_values;
   double iv = 0.0;
   IntegerVector pos_counts;
   IntegerVector neg_counts;
   
   compute_woe_iv(sorted_target, intervals, woe_values, iv, pos_counts, neg_counts);
   
   // Map WoE back to feature vector
   NumericVector bin_cut_points;
   for (const auto& interval : intervals) {
     if (interval.cut_point != std::numeric_limits<double>::infinity()) {
       bin_cut_points.push_back(interval.cut_point);
     }
   }
   std::sort(bin_cut_points.begin(), bin_cut_points.end());
   
   NumericVector feature_woe(n);
#pragma omp parallel for
   for (int i = 0; i < n; ++i) {
     double val = sorted_feature[i];
     auto it = std::upper_bound(bin_cut_points.begin(), bin_cut_points.end(), val);
     int idx = it - bin_cut_points.begin();
     feature_woe[i] = woe_values[idx];
   }
   
   // Map feature_woe back to original order
   NumericVector feature_woe_original(n);
   for (int i = 0; i < n; ++i) {
     feature_woe_original[indices[i]] = feature_woe[i];
   }

   // Prepare bin information
   CharacterVector bin_names(intervals.size());
   NumericVector bin_lower_bounds(intervals.size());
   NumericVector bin_upper_bounds(intervals.size());
   IntegerVector bin_count(intervals.size());
   IntegerVector bin_count_pos(intervals.size());
   IntegerVector bin_count_neg(intervals.size());
   
   for (size_t i = 0; i < intervals.size(); ++i) {
     const Interval& interval = intervals[i];
     bin_lower_bounds[i] = sorted_feature[interval.start];
     bin_upper_bounds[i] = sorted_feature[interval.end];
     bin_count[i] = interval.end - interval.start + 1;
     
     int count_pos = 0;
     for (int j = interval.start; j <= interval.end; ++j) {
       if (sorted_target[j] == 1)
         ++count_pos;
     }
     int count_neg = bin_count[i] - count_pos;
     bin_count_pos[i] = count_pos;
     bin_count_neg[i] = count_neg;
     
     std::stringstream ss;
     if (i == 0) {
       ss << "[-Inf;";
     } else {
       ss << "[" << bin_lower_bounds[i] << ";";
     }
     if (i == intervals.size() - 1) {
       ss << "+Inf]";
     } else {
       ss << bin_upper_bounds[i] << ")";
     }
     bin_names[i] = ss.str();
   }
   
   // Create DataFrame for bin information
   DataFrame bin_df = DataFrame::create(
     Named("bin") = bin_names,
     Named("woe") = woe_values,
     Named("iv") = rep(iv, intervals.size()),
     Named("count") = bin_count,
     Named("count_pos") = bin_count_pos,
     Named("count_neg") = bin_count_neg
   );
   
   // Create DataFrame for WoE vector feature
   DataFrame woe_df = DataFrame::create(
     Named("woefeature") = feature_woe_original
   );
   
   // Set class attributes for compatibility with data.table
   bin_df.attr("class") = CharacterVector::create("data.table", "data.frame");
   woe_df.attr("class") = CharacterVector::create("data.table", "data.frame");
   
   // Create output list
   List output_list = List::create(
     Named("woefeature") = woe_df,
     Named("woebin") = bin_df
     // Named("woe") = woe_values,
     // Named("iv") = iv,
     // Named("pos") = pos_counts,
     // Named("neg") = neg_counts
   );
   
   return output_list;
 }
 
 // // Structure to represent an interval
 // struct Interval {
 //   int start;
 //   int end;
 //   double entropy;
 //   double cut_point = std::numeric_limits<double>::infinity();
 //   int cut_index = -1;
 //   std::vector<Interval> children;
 // };
 // 
 // // Function to compute entropy
 // double compute_entropy(const IntegerVector& target, int start, int end) {
 //   int count = end - start + 1;
 //   if (count == 0) return 0.0;
 //   
 //   int count_pos = 0;
 //   int count_neg = 0;
 //   for (int i = start; i <= end; ++i) {
 //     if (target[i] == 1)
 //       ++count_pos;
 //     else
 //       ++count_neg;
 //   }
 //   double p_pos = (double)count_pos / count;
 //   double p_neg = (double)count_neg / count;
 //   
 //   double entropy = 0.0;
 //   if (p_pos > 0.0)
 //     entropy -= p_pos * std::log2(p_pos);
 //   if (p_neg > 0.0)
 //     entropy -= p_neg * std::log2(p_neg);
 //   return entropy;
 // }
 // 
 // // Function to compute information gain
 // double compute_info_gain(const IntegerVector& target, int start, int end, int split) {
 //   double entropy_parent = compute_entropy(target, start, end);
 //   int count_parent = end - start + 1;
 //   
 //   int count_left = split - start + 1;
 //   int count_right = end - split;
 //   
 //   double entropy_left = compute_entropy(target, start, split);
 //   double entropy_right = compute_entropy(target, split + 1, end);
 //   
 //   double info_gain = entropy_parent - ((double)count_left / count_parent) * entropy_left - ((double)count_right / count_parent) * entropy_right;
 //   
 //   return info_gain;
 // }
 // 
 // // Function to find optimal split
 // bool find_optimal_split(const NumericVector& feature, const IntegerVector& target, int start, int end, Interval& interval) {
 //   if (start >= end) return false;
 //   
 //   // Identify potential split points
 //   std::vector<int> split_candidates;
 //   for (int i = start; i < end; ++i) {
 //     if (feature[i] != feature[i + 1]) {
 //       split_candidates.push_back(i);
 //     }
 //   }
 //   if (split_candidates.empty()) return false;
 //   
 //   double best_gain = -std::numeric_limits<double>::infinity();
 //   int best_split = -1;
 //   
 //   for (int split : split_candidates) {
 //     double gain = compute_info_gain(target, start, end, split);
 //     if (gain > best_gain) {
 //       best_gain = gain;
 //       best_split = split;
 //     }
 //   }
 //   if (best_split == -1) return false;
 //   
 //   // Apply MDL stopping criterion
 //   int count_parent = end - start + 1;
 //   int count_left = best_split - start + 1;
 //   int count_right = end - best_split;
 //   
 //   double entropy_parent = compute_entropy(target, start, end);
 //   double entropy_left = compute_entropy(target, start, best_split);
 //   double entropy_right = compute_entropy(target, best_split + 1, end);
 //   
 //   int k = 2; // Number of classes (binary classification)
 //   double delta = std::log2(std::pow(3.0, k) - 2.0) - (k * entropy_parent - k * entropy_left - k * entropy_right);
 //   double threshold = (std::log2(count_parent - 1) + delta) / count_parent;
 //   
 //   if (best_gain > threshold) {
 //     interval.cut_point = (feature[best_split] + feature[best_split + 1]) / 2.0;
 //     interval.cut_index = best_split;
 //     return true;
 //   } else {
 //     return false;
 //   }
 // }
 // 
 // // Recursive partitioning function
 // void mdlp_partition(const NumericVector& feature, const IntegerVector& target, int start, int end, Interval& interval, double bin_cutoff_count, double min_bads_count) {
 //   // Base case: check constraints
 //   int count = end - start + 1;
 //   if (count <= 1) return;
 //   
 //   int count_pos = 0;
 //   for (int i = start; i <= end; ++i) {
 //     if (target[i] == 1)
 //       ++count_pos;
 //   }
 //   int count_neg = count - count_pos;
 //   
 //   if (count < bin_cutoff_count || count_pos < min_bads_count) {
 //     return;
 //   }
 //   
 //   // Try to find an optimal split
 //   if (find_optimal_split(feature, target, start, end, interval)) {
 //     // Split accepted
 //     Interval left_child;
 //     left_child.start = start;
 //     left_child.end = interval.cut_index;
 //     left_child.entropy = compute_entropy(target, start, interval.cut_index);
 //     Interval right_child;
 //     right_child.start = interval.cut_index + 1;
 //     right_child.end = end;
 //     right_child.entropy = compute_entropy(target, interval.cut_index + 1, end);
 //     
 //     mdlp_partition(feature, target, left_child.start, left_child.end, left_child, bin_cutoff_count, min_bads_count);
 //     mdlp_partition(feature, target, right_child.start, right_child.end, right_child, bin_cutoff_count, min_bads_count);
 //     
 //     interval.children.push_back(left_child);
 //     interval.children.push_back(right_child);
 //   } else {
 //     // No further split
 //     return;
 //   }
 // }
 // 
 // void collect_intervals(const Interval& node, std::vector<Interval>& intervals) {
 //   if (node.children.empty()) {
 //     intervals.push_back(node);
 //   } else {
 //     for (const auto& child : node.children) {
 //       collect_intervals(child, intervals);
 //     }
 //   }
 // }
 // 
 // // Function to compute WoE and IV
 // void compute_woe_iv(const IntegerVector& target, const std::vector<Interval>& intervals, std::vector<double>& woe_values, double& iv, std::vector<int>& pos_counts, std::vector<int>& neg_counts) {
 //   int total_bads = std::accumulate(target.begin(), target.end(), 0);
 //   int total_goods = target.size() - total_bads;
 //   
 //   woe_values.resize(intervals.size());
 //   pos_counts.resize(intervals.size());
 //   neg_counts.resize(intervals.size());
 //   
 //   iv = 0.0;
 //   
 //   for (size_t idx = 0; idx < intervals.size(); ++idx) {
 //     const Interval& interval = intervals[idx];
 //     int count_pos = 0;
 //     for (int i = interval.start; i <= interval.end; ++i) {
 //       if (target[i] == 1)
 //         ++count_pos;
 //     }
 //     int count_neg = interval.end - interval.start + 1 - count_pos;
 //     
 //     pos_counts[idx] = count_pos;
 //     neg_counts[idx] = count_neg;
 //     
 //     double dist_pos = (double)count_pos / total_bads;
 //     double dist_neg = (double)count_neg / total_goods;
 //     
 //     if (dist_pos == 0.0) dist_pos = 1e-10;
 //     if (dist_neg == 0.0) dist_neg = 1e-10;
 //     
 //     woe_values[idx] = std::log(dist_pos / dist_neg);
 //     iv += (dist_pos - dist_neg) * woe_values[idx];
 //   }
 // }
 
//' Performs optimal binning of a numeric variable for Weight of Evidence (WoE) and Information Value (IV) using the Minimum Description Length Principle (MDLP) criterion
//'
//' This function processes a numeric variable and creates pre-bins based on unique values. It iteratively splits bins using the MDLP criterion to maximize information gain, ensuring monotonicity of event rates while respecting the minimum number of bad events (\code{min_bads}) per bin. It also calculates WoE and IV for the generated bins.
//'
//' @param target Integer vector representing the binary target variable, where 1 indicates a positive event (e.g., default) and 0 indicates a negative event (e.g., non-default).
//' @param feature Numeric vector representing the numeric variable to be binned.
//' @param min_bins (Optional) Minimum number of bins to generate. Default is 2.
//' @param max_bins (Optional) Maximum number of bins to generate. Default is 7.
//' @param bin_cutoff (Optional) Cutoff value that determines the frequency of values to define pre-bins. Default is 0.05.
//' @param min_bads (Optional) Minimum proportion of bad events (positive target events) that a bin must contain. Default is 0.05.
//' @param max_n_prebins (Optional) Maximum number of pre-bins to consider before final binning. Default is 20.
//'
//' @return A list with the following elements:
//' \itemize{
//'   \item \code{feature_woe}: Numeric vector with the WoE assigned to each instance of the processed numeric variable.
//'   \item \code{bin}: DataFrame with the generated bins, containing the following fields:
//'     \itemize{
//'       \item \code{bin}: String representing the range of values for each bin.
//'       \item \code{woe}: Weight of Evidence (WoE) for each bin.
//'       \item \code{iv}: Information Value (IV) for each bin.
//'       \item \code{count}: Total number of observations in each bin.
//'       \item \code{count_pos}: Count of positive events in each bin.
//'       \item \code{count_neg}: Count of negative events in each bin.
//'     }
//'   \item \code{woe}: Numeric vector with the WoE for each bin.
//'   \item \code{iv}: Total Information Value (IV) calculated for the variable.
//'   \item \code{pos}: Vector with the count of positive events in each bin.
//'   \item \code{neg}: Vector with the count of negative events in each bin.
//' }
//'
//'

//  Rcpp::List OptimalBinningNumericMDLP(Rcpp::IntegerVector target, Rcpp::NumericVector feature, int min_bins = 2, int max_bins = 7, double bin_cutoff = 0.05, double min_bads = 0.05, int max_n_prebins = 20) {
//    int n = feature.size();
//    int total_bads = std::accumulate(target.begin(), target.end(), 0);
//    
//    double min_bads_count = min_bads * total_bads;
//    double bin_cutoff_count = bin_cutoff * n;
//    
//    // Create indices and sort them based on feature values
//    std::vector<int> indices(n);
//    for (int i = 0; i < n; ++i) indices[i] = i;
//    
//    std::sort(indices.begin(), indices.end(), [&](int i1, int i2) {
//      return feature[i1] < feature[i2];
//    });
//    
//    // Sort feature and target accordingly
//    NumericVector sorted_feature(n);
//    IntegerVector sorted_target(n);
//    for (int i = 0; i < n; ++i) {
//      sorted_feature[i] = feature[indices[i]];
//      sorted_target[i] = target[indices[i]];
//    }
//    
//    // Create root interval
//    Interval root;
//    root.start = 0;
//    root.end = n - 1;
//    root.entropy = compute_entropy(sorted_target, 0, n - 1);
//    
//    // Perform MDLP partitioning
//    mdlp_partition(sorted_feature, sorted_target, 0, n - 1, root, bin_cutoff_count, min_bads_count);
//    
//    // Collect the final intervals
//    std::vector<Interval> intervals;
//    collect_intervals(root, intervals);
//    
//    // Sort intervals by start index
//    std::sort(intervals.begin(), intervals.end(), [](const Interval& a, const Interval& b) {
//      return a.start < b.start;
//    });
//    
//    // Compute WoE and IV
//    std::vector<double> woe_values;
//    double iv = 0.0;
//    std::vector<int> pos_counts;
//    std::vector<int> neg_counts;
//    
//    compute_woe_iv(sorted_target, intervals, woe_values, iv, pos_counts, neg_counts);
//    
//    // Map WoE back to feature vector
//    std::vector<double> bin_cut_points;
//    for (const auto& interval : intervals) {
//      if (interval.cut_point != std::numeric_limits<double>::infinity()) {
//        bin_cut_points.push_back(interval.cut_point);
//      }
//    }
//    std::sort(bin_cut_points.begin(), bin_cut_points.end());
//    
//    NumericVector feature_woe(n);
// #ifdef _OPENMP
// #pragma omp parallel for
// #endif
//    for (int i = 0; i < n; ++i) {
//      double val = sorted_feature[i];
//      auto it = std::upper_bound(bin_cut_points.begin(), bin_cut_points.end(), val);
//      int idx = it - bin_cut_points.begin();
//      feature_woe[i] = woe_values[idx];
//    }
//    
//    // Map feature_woe back to original order
//    NumericVector feature_woe_original(n);
//    for (int i = 0; i < n; ++i) {
//      feature_woe_original[indices[i]] = feature_woe[i];
//    }
//    
//    // Prepare bin information
//    std::vector<std::string> bin_names(intervals.size());
//    std::vector<double> bin_lower_bounds(intervals.size());
//    std::vector<double> bin_upper_bounds(intervals.size());
//    std::vector<int> bin_count(intervals.size());
//    std::vector<int> bin_count_pos(intervals.size());
//    std::vector<int> bin_count_neg(intervals.size());
//    
//    for (size_t i = 0; i < intervals.size(); ++i) {
//      const Interval& interval = intervals[i];
//      bin_lower_bounds[i] = sorted_feature[interval.start];
//      bin_upper_bounds[i] = sorted_feature[interval.end];
//      bin_count[i] = interval.end - interval.start + 1;
//      
//      int count_pos = 0;
//      for (int j = interval.start; j <= interval.end; ++j) {
//        if (sorted_target[j] == 1)
//          ++count_pos;
//      }
//      int count_neg = bin_count[i] - count_pos;
//      bin_count_pos[i] = count_pos;
//      bin_count_neg[i] = count_neg;
//      
//      std::string lower_str;
//      std::string upper_str;
//      if (i == 0) {
//        lower_str = "[-Inf";
//      } else {
//        lower_str = "[" + std::to_string(bin_lower_bounds[i]);
//      }
//      if (i == intervals.size() - 1) {
//        upper_str = "+Inf]";
//      } else {
//        upper_str = std::to_string(bin_upper_bounds[i]) + ")";
//      }
//      bin_names[i] = lower_str + ";" + upper_str;
//    }
//    
//    
//    List bin_lst = List::create(
//      Named("bin") = bin_names,
//      Named("woe") = woe_values,
//      Named("iv") = iv,
//      Named("count") = bin_count,
//      Named("count_pos") = bin_count_pos,
//      Named("count_neg") = bin_count_neg
//    );
//    
//    // Create List for woe vector feature
//    List woe_lst = List::create(
//      Named("woefeature") = feature_woe
//    );
//    
//    // Attrib class for compatibility with data.table in memory superfast tables
//    bin_lst.attr("class") = CharacterVector::create("data.table", "data.frame");
//    woe_lst.attr("class") = CharacterVector::create("data.table", "data.frame");
//    
//    List output_list = List::create(
//      Named("woefeature") = woe_lst,
//      Named("woebin") = bin_lst
//    // Named("woe") = woe,
//    // Named("iv") = total_iv,
//    // Named("pos") = pos,
//    // Named("neg") = neg
//    );
//    return output_list;
//  }
 
// Function to compute the CAIM criterion
double compute_caim(const std::vector<int>& bin_starts,
                   const std::vector<int>& bin_ends,
                   const std::vector<int>& sorted_target) {
 int r = bin_starts.size();
 double caim = 0.0;
 
 for (int i = 0; i < r; ++i) {
   int start = bin_starts[i];
   int end = bin_ends[i];
   
   int count_pos = 0;
   int count_neg = 0;
   for (int j = start; j <= end; ++j) {
     if (sorted_target[j] == 1)
       ++count_pos;
     else
       ++count_neg;
   }
   
   int M_i = count_pos + count_neg;
   int max_i = std::max(count_pos, count_neg);
   
   if (M_i > 0) {
     caim += (max_i * max_i) / static_cast<double>(M_i);
   }
 }
 
 caim /= r;
 return caim;
}

//' Performs optimal binning of a numeric variable for Weight of Evidence (WoE) and Information Value (IV) using the Class-Attribute Interdependence Maximization (CAIM) criterion
//'
//' This function processes a numeric variable by creating pre-bins based on unique values. It iteratively merges or splits bins to maximize the CAIM criterion, ensuring monotonicity of event rates while respecting the minimum number of bad events (\code{min_bads}) per bin. It also calculates WoE and IV for the generated bins.
//'
//' @param target Integer vector representing the binary target variable, where 1 indicates a positive event (e.g., default) and 0 indicates a negative event (e.g., non-default).
//' @param feature Numeric vector representing the numeric variable to be binned.
//' @param min_bins (Optional) Minimum number of bins to generate. Default is 2.
//' @param max_bins (Optional) Maximum number of bins to generate. Default is 7.
//' @param bin_cutoff (Optional) Cutoff value that determines the frequency of values to define pre-bins. Default is 0.05.
//' @param min_bads (Optional) Minimum proportion of bad events (positive target events) that a bin must contain. Default is 0.05.
//' @param max_n_prebins (Optional) Maximum number of pre-bins to consider before final binning. Default is 20.
//'
//' @return A list with the following elements:
//' \itemize{
//'   \item \code{feature_woe}: Numeric vector with the WoE assigned to each instance of the processed numeric variable.
//'   \item \code{bin}: DataFrame with the generated bins, containing the following fields:
//'     \itemize{
//'       \item \code{bin}: String representing the range of values for each bin.
//'       \item \code{woe}: Weight of Evidence (WoE) for each bin.
//'       \item \code{iv}: Information Value (IV) for each bin.
//'       \item \code{count}: Total number of observations in each bin.
//'       \item \code{count_pos}: Count of positive events in each bin.
//'       \item \code{count_neg}: Count of negative events in each bin.
//'     }
//'   \item \code{woe}: Numeric vector with the WoE for each bin.
//'   \item \code{iv}: Total Information Value (IV) calculated for the variable.
//'   \item \code{pos}: Vector with the count of positive events in each bin.
//'   \item \code{neg}: Vector with the count of negative events in each bin.
//' }
//'
// [[Rcpp::export]]
Rcpp::List OptimalBinningNumericCAIM(Rcpp::IntegerVector target, Rcpp::NumericVector feature,
                                    int min_bins = 2, int max_bins = 7, double bin_cutoff = 0.05,
                                    double min_bads = 0.05, int max_n_prebins = 20) {
 
 // Input validation
 if (target.size() != feature.size()) {
   stop("'target' and 'feature' must have the same length");
 }
 
 int n = feature.size();
 std::vector<int> target_vec = as<std::vector<int>>(target);
 std::vector<double> feature_vec = as<std::vector<double>>(feature);
 
 // Check for invalid values in target and feature
 int total_bads = 0;
 int total_goods = 0;
 std::vector<bool> valid_indices(n, true);
 for (int i = 0; i < n; ++i) {
   if (target_vec[i] != 0 && target_vec[i] != 1) {
     stop("'target' must contain only 0s and 1s");
   }
   if (std::isnan(feature_vec[i]) || std::isinf(feature_vec[i])) {
     valid_indices[i] = false;
   } else {
     total_bads += target_vec[i];
     total_goods += (1 - target_vec[i]);
   }
 }
 
 // Remove invalid indices
 feature_vec.erase(
   std::remove_if(feature_vec.begin(), feature_vec.end(), 
                  [&](double) { return !valid_indices[&feature_vec[0] - &*feature_vec.begin()]; }),
                  feature_vec.end()
 );
 target_vec.erase(
   std::remove_if(target_vec.begin(), target_vec.end(), 
                  [&](int) { return !valid_indices[&target_vec[0] - &*target_vec.begin()]; }),
                  target_vec.end()
 );
 n = feature_vec.size();
 
 double min_bads_count = min_bads * total_bads;
 double bin_cutoff_count = bin_cutoff * n;
 
 // Create indices and sort them based on feature values
 std::vector<int> indices(n);
 std::iota(indices.begin(), indices.end(), 0);
 std::sort(indices.begin(), indices.end(), [&](int i1, int i2) {
   return feature_vec[i1] < feature_vec[i2];
 });
 
 // Sort feature and target accordingly
 std::vector<double> sorted_feature(n);
 std::vector<int> sorted_target(n);
 for (int i = 0; i < n; ++i) {
   sorted_feature[i] = feature_vec[indices[i]];
   sorted_target[i] = target_vec[indices[i]];
 }
 
 // Initialize bins
 std::vector<int> bin_starts = {0};
 std::vector<int> bin_ends = {n - 1};
 
 // Compute initial CAIM
 double current_caim = compute_caim(bin_starts, bin_ends, sorted_target);
 
 bool improvement = true;
 
 while (improvement && bin_starts.size() < static_cast<size_t>(max_bins)) {
   improvement = false;
   double best_caim = current_caim;
   int best_bin = -1;
   int best_split = -1;
   
   // For each bin, consider all possible splits
   for (size_t b = 0; b < bin_starts.size(); ++b) {
     int start = bin_starts[b];
     int end = bin_ends[b];
     
     // Potential split points are where the feature value changes
     std::vector<int> split_candidates;
     split_candidates.reserve(end - start);
     for (int i = start; i < end; ++i) {
       if (sorted_feature[i] != sorted_feature[i + 1]) {
         split_candidates.push_back(i);
       }
     }
     
     // Consider each split candidate
     for (int split : split_candidates) {
       // Check bin constraints
       int left_count = split - start + 1;
       int right_count = end - split;
       
       if (left_count < bin_cutoff_count || right_count < bin_cutoff_count)
         continue;
       
       // Count bads in each bin
       int left_bads = std::accumulate(sorted_target.begin() + start, sorted_target.begin() + split + 1, 0);
       int right_bads = std::accumulate(sorted_target.begin() + split + 1, sorted_target.begin() + end + 1, 0);
       
       if (left_bads < min_bads_count || right_bads < min_bads_count)
         continue;
       
       // Temporarily add split
       std::vector<int> temp_bin_starts = bin_starts;
       std::vector<int> temp_bin_ends = bin_ends;
       
       temp_bin_starts.erase(temp_bin_starts.begin() + b);
       temp_bin_ends.erase(temp_bin_ends.begin() + b);
       
       temp_bin_starts.insert(temp_bin_starts.begin() + b, start);
       temp_bin_starts.insert(temp_bin_starts.begin() + b + 1, split + 1);
       
       temp_bin_ends.insert(temp_bin_ends.begin() + b, split);
       temp_bin_ends.insert(temp_bin_ends.begin() + b + 1, end);
       
       // Compute CAIM
       double new_caim = compute_caim(temp_bin_starts, temp_bin_ends, sorted_target);
       
       if (new_caim > best_caim) {
         best_caim = new_caim;
         best_bin = b;
         best_split = split;
         improvement = true;
       }
     }
   }
   
   if (improvement) {
     // Accept the best split
     int start = bin_starts[best_bin];
     int end = bin_ends[best_bin];
     
     bin_starts.erase(bin_starts.begin() + best_bin);
     bin_ends.erase(bin_ends.begin() + best_bin);
     
     bin_starts.insert(bin_starts.begin() + best_bin, start);
     bin_starts.insert(bin_starts.begin() + best_bin + 1, best_split + 1);
     
     bin_ends.insert(bin_ends.begin() + best_bin, best_split);
     bin_ends.insert(bin_ends.begin() + best_bin + 1, end);
     
     current_caim = best_caim;
   }
 }
 
 // Enforce minimum number of bins by splitting bins
 while (bin_starts.size() < static_cast<size_t>(min_bins)) {
   // Find the bin with the largest count to split
   size_t max_bin_idx = 0;
   int max_bin_count = bin_ends[0] - bin_starts[0] + 1;
   for (size_t i = 1; i < bin_starts.size(); ++i) {
     int bin_count = bin_ends[i] - bin_starts[i] + 1;
     if (bin_count > max_bin_count) {
       max_bin_idx = i;
       max_bin_count = bin_count;
     }
   }
   
   int start = bin_starts[max_bin_idx];
   int end = bin_ends[max_bin_idx];
   
   // Potential split points are where the feature value changes
   std::vector<int> split_candidates;
   for (int i = start; i < end; ++i) {
     if (sorted_feature[i] != sorted_feature[i + 1]) {
       split_candidates.push_back(i);
     }
   }
   
   if (split_candidates.empty()) {
     // Cannot split further
     break;
   }
   
   // Choose the middle split candidate
   int split = split_candidates[split_candidates.size() / 2];
   
   // Check bin constraints
   int left_count = split - start + 1;
   int right_count = end - split;
   
   if (left_count < bin_cutoff_count || right_count < bin_cutoff_count) {
     // Cannot split due to bin size constraint
     break;
   }
   
   // Split the bin
   bin_starts.insert(bin_starts.begin() + max_bin_idx + 1, split + 1);
   bin_ends.insert(bin_ends.begin() + max_bin_idx, split);
 }
 
 // Compute WoE and IV
 int num_bins = bin_starts.size();
 std::vector<double> woe_values(num_bins);
 double iv = 0.0;
 std::vector<int> pos_counts(num_bins);
 std::vector<int> neg_counts(num_bins);
 std::vector<double> event_rates(num_bins);
 
 for (int i = 0; i < num_bins; ++i) {
   int start = bin_starts[i];
   int end = bin_ends[i];
   
   int count_pos = 0;
   int count_neg = 0;
   for (int j = start; j <= end; ++j) {
     if (sorted_target[j] == 1)
       ++count_pos;
     else
       ++count_neg;
   }
   
   pos_counts[i] = count_pos;
   neg_counts[i] = count_neg;
   
   double dist_pos = count_pos / static_cast<double>(total_bads);
   double dist_neg = count_neg / static_cast<double>(total_goods);
   
   if (dist_pos == 0.0) dist_pos = 1e-10;
   if (dist_neg == 0.0) dist_neg = 1e-10;
   
   woe_values[i] = std::log(dist_pos / dist_neg);
   iv += (dist_pos - dist_neg) * woe_values[i];
   
   event_rates[i] = count_pos / static_cast<double>(count_pos + count_neg);
 }
 
 // Enforce monotonicity
 bool increasing = event_rates.front() < event_rates.back();
 bool monotonic = false;
 
 while (!monotonic && num_bins > min_bins) {
   monotonic = true;
   for (int i = 0; i < num_bins - 1; ++i) {
     if ((increasing && event_rates[i] > event_rates[i + 1]) ||
         (!increasing && event_rates[i] < event_rates[i + 1])) {
       // Merge bins i and i+1
       bin_ends[i] = bin_ends[i + 1];
       bin_starts.erase(bin_starts.begin() + i + 1);
       bin_ends.erase(bin_ends.begin() + i + 1);
       
       pos_counts[i] += pos_counts[i + 1];
       neg_counts[i] += neg_counts[i + 1];
       pos_counts.erase(pos_counts.begin() + i + 1);
       neg_counts.erase(neg_counts.begin() + i + 1);
       
       event_rates[i] = pos_counts[i] / static_cast<double>(pos_counts[i] + neg_counts[i]);
       event_rates.erase(event_rates.begin() + i + 1);
       
       woe_values.erase(woe_values.begin() + i + 1);
       
       num_bins -= 1;
       monotonic = false;
       break;
     }
   }
 }
 
 // Map WoE back to feature vector
 std::vector<double> feature_woe(n);
 std::vector<double> bin_upper_bounds(num_bins);
 for (int i = 0; i < num_bins; ++i) {
   bin_upper_bounds[i] = sorted_feature[bin_ends[i]];
 }
 
#ifdef _OPENMP
#pragma omp parallel for
#endif
 for (int i = 0; i < n; ++i) {
   double val = sorted_feature[i];
   auto it = std::upper_bound(bin_upper_bounds.begin(), bin_upper_bounds.end(), val);
   int idx = it - bin_upper_bounds.begin();
   if (idx >= num_bins) idx = num_bins - 1;
   feature_woe[i] = woe_values[idx];
 }
 
 // Map feature_woe back to original order
 std::vector<double> feature_woe_original(feature.size(), std::numeric_limits<double>::quiet_NaN());
 for (int i = 0; i < n; ++i) {
   feature_woe_original[indices[i]] = feature_woe[i];
 }
 
 // Prepare bin information
 std::vector<std::string> bin_names(num_bins);
 std::vector<double> bin_lower_bounds(num_bins);
 std::vector<double> bin_upper_bounds_output(num_bins);
 std::vector<int> bin_counts(num_bins);
 
 for (int i = 0; i < num_bins; ++i) {
   bin_lower_bounds[i] = sorted_feature[bin_starts[i]];
   bin_upper_bounds_output[i] = sorted_feature[bin_ends[i]];
   bin_counts[i] = bin_ends[i] - bin_starts[i] + 1;
   
   std::ostringstream oss;
   oss << std::fixed << std::setprecision(4);
   oss << (i == 0 ? "[-Inf;" : "[") << bin_lower_bounds[i] << ";";
   oss << (i == num_bins - 1 ? "+Inf]" : std::to_string(bin_upper_bounds_output[i]) + ")");
   bin_names[i] = oss.str();
 }
 
 DataFrame bin_df = DataFrame::create(
   Named("bin") = bin_names,
   Named("woe") = woe_values,
   Named("iv") = iv,
   Named("count") = bin_counts,
   Named("count_pos") = pos_counts,
   Named("count_neg") = neg_counts
 );
 
 // Create DataFrame for woe vector feature
 DataFrame woe_df = DataFrame::create(
   Named("woefeature") = feature_woe_original
 );
 
 // Set class attributes for compatibility with data.table
 bin_df.attr("class") = CharacterVector::create("data.table", "data.frame");
 woe_df.attr("class") = CharacterVector::create("data.table", "data.frame");
 
 List output_list = List::create(
   Named("woefeature") = woe_df,
   Named("woebin") = bin_df,
   Named("total_iv") = iv
 );
 
 return output_list;
}
 
//  // Function to compute the CAIM criterion
//  double compute_caim(const std::vector<int>& bin_starts,
//                      const std::vector<int>& bin_ends,
//                      const IntegerVector& sorted_target) {
//    int r = bin_starts.size();
//    double caim = 0.0;
//    
//    for (int i = 0; i < r; ++i) {
//      int start = bin_starts[i];
//      int end = bin_ends[i];
//      
//      int count_pos = 0;
//      int count_neg = 0;
//      for (int j = start; j <= end; ++j) {
//        if (sorted_target[j] == 1)
//          ++count_pos;
//        else
//          ++count_neg;
//      }
//      
//      int M_i = count_pos + count_neg;
//      int max_i = std::max(count_pos, count_neg);
//      
//      if (M_i > 0) {
//        caim += (max_i * max_i) / static_cast<double>(M_i);
//      }
//    }
//    
//    caim /= r;
//    return caim;
//  }
//  
//  //' Performs optimal binning of a numeric variable for Weight of Evidence (WoE) and Information Value (IV) using the Class-Attribute Interdependence Maximization (CAIM) criterion
//  //'
//  //' This function processes a numeric variable by creating pre-bins based on unique values. It iteratively merges or splits bins to maximize the CAIM criterion, ensuring monotonicity of event rates while respecting the minimum number of bad events (\code{min_bads}) per bin. It also calculates WoE and IV for the generated bins.
//  //'
//  //' @param target Integer vector representing the binary target variable, where 1 indicates a positive event (e.g., default) and 0 indicates a negative event (e.g., non-default).
//  //' @param feature Numeric vector representing the numeric variable to be binned.
//  //' @param min_bins (Optional) Minimum number of bins to generate. Default is 2.
//  //' @param max_bins (Optional) Maximum number of bins to generate. Default is 7.
//  //' @param bin_cutoff (Optional) Cutoff value that determines the frequency of values to define pre-bins. Default is 0.05.
//  //' @param min_bads (Optional) Minimum proportion of bad events (positive target events) that a bin must contain. Default is 0.05.
//  //' @param max_n_prebins (Optional) Maximum number of pre-bins to consider before final binning. Default is 20.
//  //'
//  //' @return A list with the following elements:
//  //' \itemize{
//  //'   \item \code{feature_woe}: Numeric vector with the WoE assigned to each instance of the processed numeric variable.
//  //'   \item \code{bin}: DataFrame with the generated bins, containing the following fields:
//  //'     \itemize{
//  //'       \item \code{bin}: String representing the range of values for each bin.
//  //'       \item \code{woe}: Weight of Evidence (WoE) for each bin.
//  //'       \item \code{iv}: Information Value (IV) for each bin.
//  //'       \item \code{count}: Total number of observations in each bin.
//  //'       \item \code{count_pos}: Count of positive events in each bin.
//  //'       \item \code{count_neg}: Count of negative events in each bin.
//  //'     }
//  //'   \item \code{woe}: Numeric vector with the WoE for each bin.
//  //'   \item \code{iv}: Total Information Value (IV) calculated for the variable.
//  //'   \item \code{pos}: Vector with the count of positive events in each bin.
//  //'   \item \code{neg}: Vector with the count of negative events in each bin.
//  //' }
//  //'
//  //'
//  Rcpp::List OptimalBinningNumericCAIM(Rcpp::IntegerVector target, Rcpp::NumericVector feature, int min_bins = 2, int max_bins = 7, double bin_cutoff = 0.05, double min_bads = 0.05, int max_n_prebins = 20) {
//    int n = feature.size();
//    int total_bads = std::accumulate(target.begin(), target.end(), 0);
//    int total_goods = n - total_bads;
//    
//    double min_bads_count = min_bads * total_bads;
//    double bin_cutoff_count = bin_cutoff * n;
//    
//    // Create indices and sort them based on feature values
//    std::vector<int> indices(n);
//    for (int i = 0; i < n; ++i) indices[i] = i;
//    
//    std::sort(indices.begin(), indices.end(), [&](int i1, int i2) {
//      return feature[i1] < feature[i2];
//    });
//    
//    // Sort feature and target accordingly
//    NumericVector sorted_feature(n);
//    IntegerVector sorted_target(n);
//    for (int i = 0; i < n; ++i) {
//      sorted_feature[i] = feature[indices[i]];
//      sorted_target[i] = target[indices[i]];
//    }
//    
//    // Initialize bins
//    std::vector<int> bin_starts = {0};
//    std::vector<int> bin_ends = {n - 1};
//    
//    // Compute initial CAIM
//    double current_caim = compute_caim(bin_starts, bin_ends, sorted_target);
//    
//    bool improvement = true;
//    
//    while (improvement) {
//      improvement = false;
//      double best_caim = current_caim;
//      int best_bin = -1;
//      int best_split = -1;
//      
//      // For each bin, consider all possible splits
//      for (size_t b = 0; b < bin_starts.size(); ++b) {
//        int start = bin_starts[b];
//        int end = bin_ends[b];
//        
//        // Potential split points are where the feature value changes
//        std::vector<int> split_candidates;
//        for (int i = start; i < end; ++i) {
//          if (sorted_feature[i] != sorted_feature[i + 1]) {
//            split_candidates.push_back(i);
//          }
//        }
//        
//        // Consider each split candidate
//        for (int split : split_candidates) {
//          // Temporarily add split
//          std::vector<int> temp_bin_starts = bin_starts;
//          std::vector<int> temp_bin_ends = bin_ends;
//          
//          temp_bin_starts.erase(temp_bin_starts.begin() + b);
//          temp_bin_ends.erase(temp_bin_ends.begin() + b);
//          
//          temp_bin_starts.insert(temp_bin_starts.begin() + b, start);
//          temp_bin_starts.insert(temp_bin_starts.begin() + b + 1, split + 1);
//          
//          temp_bin_ends.insert(temp_bin_ends.begin() + b, split);
//          temp_bin_ends.insert(temp_bin_ends.begin() + b + 1, end);
//          
//          // Check bin constraints
//          int left_count = split - start + 1;
//          int right_count = end - split;
//          
//          if (left_count < bin_cutoff_count || right_count < bin_cutoff_count)
//            continue;
//          
//          // Count bads in each bin
//          int left_bads = std::accumulate(sorted_target.begin() + start, sorted_target.begin() + split + 1, 0);
//          int right_bads = std::accumulate(sorted_target.begin() + split + 1, sorted_target.begin() + end + 1, 0);
//          
//          if (left_bads < min_bads_count || right_bads < min_bads_count)
//            continue;
//          
//          // Compute CAIM
//          double new_caim = compute_caim(temp_bin_starts, temp_bin_ends, sorted_target);
//          
//          if (new_caim > best_caim) {
//            best_caim = new_caim;
//            best_bin = b;
//            best_split = split;
//            improvement = true;
//          }
//        }
//      }
//      
//      if (improvement && bin_starts.size() < static_cast<size_t>(max_bins)) {
//        // Accept the best split
//        int start = bin_starts[best_bin];
//        int end = bin_ends[best_bin];
//        
//        bin_starts.erase(bin_starts.begin() + best_bin);
//        bin_ends.erase(bin_ends.begin() + best_bin);
//        
//        bin_starts.insert(bin_starts.begin() + best_bin, start);
//        bin_starts.insert(bin_starts.begin() + best_bin + 1, best_split + 1);
//        
//        bin_ends.insert(bin_ends.begin() + best_bin, best_split);
//        bin_ends.insert(bin_ends.begin() + best_bin + 1, end);
//        
//        current_caim = best_caim;
//      } else {
//        improvement = false;
//      }
//      
//      // Check if maximum number of bins reached
//      if (bin_starts.size() >= static_cast<size_t>(max_bins)) {
//        break;
//      }
//    }
//    
//    // Enforce minimum number of bins by splitting bins
//    while (bin_starts.size() < static_cast<size_t>(min_bins)) {
//      // Find the bin with the largest count to split
//      size_t max_bin_idx = 0;
//      int max_bin_count = bin_ends[0] - bin_starts[0] + 1;
//      for (size_t i = 1; i < bin_starts.size(); ++i) {
//        int bin_count = bin_ends[i] - bin_starts[i] + 1;
//        if (bin_count > max_bin_count) {
//          max_bin_idx = i;
//          max_bin_count = bin_count;
//        }
//      }
//      
//      int start = bin_starts[max_bin_idx];
//      int end = bin_ends[max_bin_idx];
//      
//      // Potential split points are where the feature value changes
//      std::vector<int> split_candidates;
//      for (int i = start; i < end; ++i) {
//        if (sorted_feature[i] != sorted_feature[i + 1]) {
//          split_candidates.push_back(i);
//        }
//      }
//      
//      if (split_candidates.empty()) {
//        // Cannot split further
//        break;
//      }
//      
//      // Choose the middle split candidate
//      int split = split_candidates[split_candidates.size() / 2];
//      
//      // Check bin constraints
//      int left_count = split - start + 1;
//      int right_count = end - split;
//      
//      if (left_count < bin_cutoff_count || right_count < bin_cutoff_count) {
//        // Cannot split due to bin size constraint
//        break;
//      }
//      
//      // Split the bin
//      bin_starts.erase(bin_starts.begin() + max_bin_idx);
//      bin_ends.erase(bin_ends.begin() + max_bin_idx);
//      
//      bin_starts.insert(bin_starts.begin() + max_bin_idx, start);
//      bin_starts.insert(bin_starts.begin() + max_bin_idx + 1, split + 1);
//      
//      bin_ends.insert(bin_ends.begin() + max_bin_idx, split);
//      bin_ends.insert(bin_ends.begin() + max_bin_idx + 1, end);
//    }
//    
//    // Compute WoE and IV
//    int num_bins = bin_starts.size();
//    std::vector<double> woe_values(num_bins);
//    double iv = 0.0;
//    std::vector<int> pos_counts(num_bins);
//    std::vector<int> neg_counts(num_bins);
//    std::vector<double> event_rates(num_bins);
//    
//    for (int i = 0; i < num_bins; ++i) {
//      int start = bin_starts[i];
//      int end = bin_ends[i];
//      
//      int count_pos = 0;
//      int count_neg = 0;
//      for (int j = start; j <= end; ++j) {
//        if (sorted_target[j] == 1)
//          ++count_pos;
//        else
//          ++count_neg;
//      }
//      
//      pos_counts[i] = count_pos;
//      neg_counts[i] = count_neg;
//      
//      double dist_pos = count_pos / static_cast<double>(total_bads);
//      double dist_neg = count_neg / static_cast<double>(total_goods);
//      
//      if (dist_pos == 0.0) dist_pos = 1e-10;
//      if (dist_neg == 0.0) dist_neg = 1e-10;
//      
//      woe_values[i] = std::log(dist_pos / dist_neg);
//      iv += (dist_pos - dist_neg) * woe_values[i];
//      
//      event_rates[i] = count_pos / static_cast<double>(count_pos + count_neg);
//    }
//    
//    // Enforce monotonicity
//    bool increasing = event_rates.front() < event_rates.back();
//    bool monotonic = false;
//    
//    while (!monotonic) {
//      monotonic = true;
//      for (int i = 0; i < num_bins - 1; ++i) {
//        if (increasing) {
//          if (event_rates[i] > event_rates[i + 1]) {
//            // Merge bins i and i+1
//            bin_ends[i] = bin_ends[i + 1];
//            bin_starts.erase(bin_starts.begin() + i + 1);
//            bin_ends.erase(bin_ends.begin() + i + 1);
//            
//            pos_counts[i] += pos_counts[i + 1];
//            neg_counts[i] += neg_counts[i + 1];
//            pos_counts.erase(pos_counts.begin() + i + 1);
//            neg_counts.erase(neg_counts.begin() + i + 1);
//            
//            event_rates[i] = pos_counts[i] / static_cast<double>(pos_counts[i] + neg_counts[i]);
//            event_rates.erase(event_rates.begin() + i + 1);
//            
//            woe_values.erase(woe_values.begin() + i + 1);
//            
//            num_bins -= 1;
//            monotonic = false;
//            break;
//          }
//        } else {
//          if (event_rates[i] < event_rates[i + 1]) {
//            // Merge bins i and i+1
//            bin_ends[i] = bin_ends[i + 1];
//            bin_starts.erase(bin_starts.begin() + i + 1);
//            bin_ends.erase(bin_ends.begin() + i + 1);
//            
//            pos_counts[i] += pos_counts[i + 1];
//            neg_counts[i] += neg_counts[i + 1];
//            pos_counts.erase(pos_counts.begin() + i + 1);
//            neg_counts.erase(neg_counts.begin() + i + 1);
//            
//            event_rates[i] = pos_counts[i] / static_cast<double>(pos_counts[i] + neg_counts[i]);
//            event_rates.erase(event_rates.begin() + i + 1);
//            
//            woe_values.erase(woe_values.begin() + i + 1);
//            
//            num_bins -= 1;
//            monotonic = false;
//            break;
//          }
//        }
//      }
//    }
//    
//    // Map WoE back to feature vector
//    NumericVector feature_woe(n);
//    std::vector<double> bin_upper_bounds(num_bins);
//    for (int i = 0; i < num_bins; ++i) {
//      bin_upper_bounds[i] = sorted_feature[bin_ends[i]];
//    }
//    
// #ifdef _OPENMP
// #pragma omp parallel for
// #endif
//    for (int i = 0; i < n; ++i) {
//      double val = sorted_feature[i];
//      auto it = std::upper_bound(bin_upper_bounds.begin(), bin_upper_bounds.end(), val);
//      int idx = it - bin_upper_bounds.begin();
//      if (idx >= num_bins) idx = num_bins - 1;
//      feature_woe[i] = woe_values[idx];
//    }
//    
//    // Map feature_woe back to original order
//    NumericVector feature_woe_original(n);
//    for (int i = 0; i < n; ++i) {
//      feature_woe_original[indices[i]] = feature_woe[i];
//    }
//    
//    // Prepare bin information
//    std::vector<std::string> bin_names(num_bins);
//    std::vector<double> bin_lower_bounds(num_bins);
//    std::vector<double> bin_upper_bounds_output(num_bins);
//    std::vector<int> bin_counts(num_bins);
//    
//    for (int i = 0; i < num_bins; ++i) {
//      bin_lower_bounds[i] = sorted_feature[bin_starts[i]];
//      bin_upper_bounds_output[i] = sorted_feature[bin_ends[i]];
//      bin_counts[i] = bin_ends[i] - bin_starts[i] + 1;
//      
//      std::string lower_str = (i == 0) ? "[-Inf" : "[" + std::to_string(bin_lower_bounds[i]);
//      std::string upper_str = (i == num_bins - 1) ? "+Inf]" : std::to_string(bin_upper_bounds_output[i]) + ")";
//      bin_names[i] = lower_str + ";" + upper_str;
//    }
//    
//    List bin_lst = List::create(
//      Named("bin") = bin_names,
//      Named("woe") = woe_values,
//      Named("iv") = iv,
//      Named("count") = bin_counts,
//      Named("count_pos") = pos_counts,
//      Named("count_neg") = neg_counts
//    );
//    
//    // Create List for woe vector feature
//    List woe_lst = List::create(
//      Named("woefeature") = feature_woe
//    );
//    
//    // Attrib class for compatibility with data.table in memory superfast tables
//    bin_lst.attr("class") = CharacterVector::create("data.table", "data.frame");
//    woe_lst.attr("class") = CharacterVector::create("data.table", "data.frame");
//    
//    List output_list = List::create(
//      Named("woefeature") = woe_lst,
//      Named("woebin") = bin_lst
//    // Named("woe") = woe,
//    // Named("iv") = total_iv,
//    // Named("pos") = pos,
//    // Named("neg") = neg
//    );
//    return output_list;
//  }
 
 
 
 
 
 
 
 
 // Function to format doubles to six decimal places
 std::string format_double(double val) {
   std::ostringstream oss;
   oss << std::fixed << std::setprecision(6) << val;
   return oss.str();
 }
 
 //' Performs optimal binning of a numeric variable for Weight of Evidence (WoE) and Information Value (IV) using the Pool Adjacent Violators Algorithm (PAVA) to enforce monotonicity
 //'
 //' This function processes a numeric variable by removing missing values and creating pre-bins based on unique values. It then applies the Pool Adjacent Violators Algorithm (PAVA) to ensure the monotonicity of event rates, either increasing or decreasing based on the specified direction. The function respects the minimum number of bad events (\code{min_bads}) per bin and calculates WoE and IV for the generated bins.
 //'
 //' @param target Integer vector representing the binary target variable, where 1 indicates a positive event (e.g., default) and 0 indicates a negative event (e.g., non-default).
 //' @param feature Numeric vector representing the numeric variable to be binned.
 //' @param max_bins (Optional) Maximum number of bins to generate. Default is 7.
 //' @param bin_cutoff (Optional) Cutoff value that determines the frequency of values to define pre-bins. Default is 0.05.
 //' @param min_bads (Optional) Minimum proportion of bad events (positive target events) that a bin must contain. Default is 0.05.
 //' @param max_n_prebins (Optional) Maximum number of pre-bins to consider before final binning. Default is 20.
 //' @param monotonicity_direction (Optional) String that defines the monotonicity direction of event rates, either "increase" for increasing monotonicity or "decrease" for decreasing monotonicity. Default is "increase".
 //'
 //' @return A list with the following elements:
 //' \itemize{
 //'   \item \code{feature_woe}: Numeric vector with the WoE assigned to each instance of the processed numeric variable.
 //'   \item \code{bin}: DataFrame with the generated bins, containing the following fields:
 //'     \itemize{
 //'       \item \code{bin}: String representing the range of values for each bin.
 //'       \item \code{woe}: Weight of Evidence (WoE) for each bin.
 //'       \item \code{iv}: Information Value (IV) for each bin.
 //'       \item \code{count}: Total number of observations in each bin.
 //'       \item \code{count_pos}: Count of positive events in each bin.
 //'       \item \code{count_neg}: Count of negative events in each bin.
 //'     }
 //'   \item \code{woe}: Numeric vector with the WoE for each bin.
 //'   \item \code{iv}: Total Information Value (IV) calculated for the variable.
 //'   \item \code{pos}: Vector with the count of positive events in each bin.
 //'   \item \code{neg}: Vector with the count of negative events in each bin.
 //' }
 //'
 //'
 // [[Rcpp::export]]
 List OptimalBinningNumericPAVA(Rcpp::IntegerVector target, Rcpp::NumericVector feature, int max_bins = 7, double bin_cutoff = 0.05, double min_bads = 0.05, int max_n_prebins = 20, std::string monotonicity_direction = "increase") {
   // Ensure input vectors are of the same length
   int N = feature.size();
   if (N != target.size()) {
     stop("feature and target must be the same length.");
   }
   
   // Total counts of positives and negatives
   int total_pos = std::accumulate(target.begin(), target.end(), 0);
   int total_neg = N - total_pos;
   
   // Sort feature and get sorted indices
   IntegerVector indices = seq(0, N - 1);
   NumericVector feature_copy = clone(feature);
   
   std::sort(indices.begin(), indices.end(), [&](int i, int j) {
     return feature_copy[i] < feature_copy[j];
   });
   
   // Initial binning
   int bin_size = std::max(1, N / max_n_prebins);
   IntegerVector bin_indices(N);
   int current_bin = 0;
   
   for (int i = 0; i < N; ++i) {
     if (i >= (current_bin + 1) * bin_size && current_bin < max_n_prebins - 1) {
       current_bin++;
     }
     bin_indices[indices[i]] = current_bin;
   }
   
   int num_bins = current_bin + 1;
   
   // Initialize bins
   struct Bin {
     int count_pos = 0;
     int count_neg = 0;
     double event_rate = 0.0;
     int total_count = 0;
     double lower_bound = R_PosInf;
     double upper_bound = R_NegInf;
     double woe = 0.0;
   };
   
   std::vector<Bin> bins(num_bins);
   
   for (int i = 0; i < N; ++i) {
     int b = bin_indices[i];
     bins[b].count_pos += target[i];
     bins[b].count_neg += 1 - target[i];
     bins[b].total_count += 1;
     if (feature[i] < bins[b].lower_bound) bins[b].lower_bound = feature[i];
     if (feature[i] > bins[b].upper_bound) bins[b].upper_bound = feature[i];
   }
   
   for (int i = 0; i < num_bins; ++i) {
     bins[i].event_rate = (double)bins[i].count_pos / bins[i].total_count;
   }
   
   // Apply PAVA
   bool merged = true;
   while (merged) {
     merged = false;
     for (size_t i = 0; i < bins.size() - 1; ++i) {
       bool violation = false;
       if (monotonicity_direction == "increase") {
         if (bins[i].event_rate > bins[i + 1].event_rate) {
           violation = true;
         }
       } else if (monotonicity_direction == "decrease") {
         if (bins[i].event_rate < bins[i + 1].event_rate) {
           violation = true;
         }
       }
       if (violation) {
         // Merge bins i and i+1
         bins[i].count_pos += bins[i + 1].count_pos;
         bins[i].count_neg += bins[i + 1].count_neg;
         bins[i].total_count += bins[i + 1].total_count;
         bins[i].event_rate = (double)bins[i].count_pos / bins[i].total_count;
         bins[i].upper_bound = bins[i + 1].upper_bound;
         bins.erase(bins.begin() + i + 1);
         merged = true;
         break;
       }
     }
   }
   
   // Enforce bin size constraints
   for (size_t i = 0; i < bins.size(); ++i) {
     while (bins[i].total_count < N * bin_cutoff && bins.size() > 1) {
       if (i == bins.size() - 1) {
         bins[i - 1].count_pos += bins[i].count_pos;
         bins[i - 1].count_neg += bins[i].count_neg;
         bins[i - 1].total_count += bins[i].total_count;
         bins[i - 1].event_rate = (double)bins[i - 1].count_pos / bins[i - 1].total_count;
         bins[i - 1].upper_bound = bins[i].upper_bound;
         bins.erase(bins.begin() + i);
         i--;
       } else {
         bins[i].count_pos += bins[i + 1].count_pos;
         bins[i].count_neg += bins[i + 1].count_neg;
         bins[i].total_count += bins[i + 1].total_count;
         bins[i].event_rate = (double)bins[i].count_pos / bins[i].total_count;
         bins[i].upper_bound = bins[i + 1].upper_bound;
         bins.erase(bins.begin() + i + 1);
       }
     }
   }
   
   // Limit the number of bins to max_bins
   while ((int)bins.size() > max_bins) {
     // Merge the two bins with the smallest total_count
     size_t min_idx = 0;
     int min_total = bins[0].total_count;
     for (size_t i = 1; i < bins.size(); ++i) {
       if (bins[i].total_count < min_total) {
         min_total = bins[i].total_count;
         min_idx = i;
       }
     }
     if (min_idx == bins.size() - 1) {
       bins[min_idx - 1].count_pos += bins[min_idx].count_pos;
       bins[min_idx - 1].count_neg += bins[min_idx].count_neg;
       bins[min_idx - 1].total_count += bins[min_idx].total_count;
       bins[min_idx - 1].event_rate = (double)bins[min_idx - 1].count_pos / bins[min_idx - 1].total_count;
       bins[min_idx - 1].upper_bound = bins[min_idx].upper_bound;
       bins.erase(bins.begin() + min_idx);
     } else {
       bins[min_idx].count_pos += bins[min_idx + 1].count_pos;
       bins[min_idx].count_neg += bins[min_idx + 1].count_neg;
       bins[min_idx].total_count += bins[min_idx + 1].total_count;
       bins[min_idx].event_rate = (double)bins[min_idx].count_pos / bins[min_idx].total_count;
       bins[min_idx].upper_bound = bins[min_idx + 1].upper_bound;
       bins.erase(bins.begin() + min_idx + 1);
     }
   }
   
   // Compute WoE and IV
   NumericVector woe(bins.size());
   double iv = 0.0;
   
   for (size_t i = 0; i < bins.size(); ++i) {
     double dist_pos = bins[i].count_pos / (double)total_pos;
     double dist_neg = bins[i].count_neg / (double)total_neg;
     
     // Avoid division by zero
     if (dist_pos == 0) dist_pos = 0.0001;
     if (dist_neg == 0) dist_neg = 0.0001;
     
     bins[i].woe = log(dist_pos / dist_neg);
     woe[i] = bins[i].woe;
     iv += (dist_pos - dist_neg) * bins[i].woe;
   }
   
   // Map WoE back to feature values
   NumericVector feature_woe(N);
   
   // Parallel processing
#pragma omp parallel for if(N > 1000)
   for (int i = 0; i < N; ++i) {
     double x = feature[i];
     for (size_t j = 0; j < bins.size(); ++j) {
       if (x >= bins[j].lower_bound && x <= bins[j].upper_bound) {
         feature_woe[i] = bins[j].woe;
         break;
       }
     }
   }
   
   // Prepare binning output
   CharacterVector bin_names(bins.size());
   IntegerVector bin_count(bins.size());
   IntegerVector bin_count_pos(bins.size());
   IntegerVector bin_count_neg(bins.size());
   
   for (size_t i = 0; i < bins.size(); ++i) {
     double lower = bins[i].lower_bound;
     double upper = bins[i].upper_bound;
     
     std::string lower_bracket, upper_bracket;
     
     if (i == 0) {
       // First bin
       lower_bracket = "[";
     } else {
       lower_bracket = "(";
     }
     
     if (i == bins.size() - 1) {
       // Last bin
       upper_bracket = "]";
     } else {
       upper_bracket = ")";
     }
     
     std::string lower_str = (lower == R_NegInf) ? "-Inf" : format_double(lower);
     std::string upper_str = (upper == R_PosInf) ? "+Inf" : format_double(upper);
     
     bin_names[i] = lower_bracket + lower_str + ";" + upper_str + upper_bracket;
     
     bin_count[i] = bins[i].total_count;
     bin_count_pos[i] = bins[i].count_pos;
     bin_count_neg[i] = bins[i].count_neg;
   }
   
   List bin_lst = List::create(
     Named("bin") = bin_names,
     Named("woe") = woe,
     Named("iv") = iv,
     Named("count") = bin_count,
     Named("count_pos") = bin_count_pos,
     Named("count_neg") = bin_count_neg
   );
   
   // Create List for woe vector feature
   List woe_lst = List::create(
     Named("woefeature") = feature_woe
   );
   
   // Attrib class for compatibility with data.table in memory superfast tables
   bin_lst.attr("class") = CharacterVector::create("data.table", "data.frame");
   woe_lst.attr("class") = CharacterVector::create("data.table", "data.frame");
   
   List output_list = List::create(
     Named("woefeature") = woe_lst,
     Named("woebin") = bin_lst
   // Named("woe") = woe,
   // Named("iv") = total_iv,
   // Named("pos") = pos,
   // Named("neg") = neg
   );
   
   // DataFrame bin_df = DataFrame::create(
   //   Named("bin") = bin_names,
   //   Named("woe") = woe,
   //   Named("count") = bin_count,
   //   Named("count_pos") = bin_count_pos,
   //   Named("count_neg") = bin_count_neg
   // );
   // 
   // List output_list = List::create(
   //   Named("woefeature") = feature_woe,
   //   Named("woebin") = bin_df
   // // Named("woe") = woe,
   // // Named("iv") = total_iv,
   // // Named("pos") = pos,
   // // Named("neg") = neg
   // );
   // 
   // output_list.attr("class") = CharacterVector::create("data.table", "data.frame");
   // 
   return output_list;
 }
 
 // Structure to hold bin information, renamed to BinTree
 struct BinTree {
   double start;
   double end;
   int count;
   int count_pos;
   int count_neg;
   double woe;
   double iv;
 };
 
 // Structure for tree nodes
 struct TreeNode {
   double split_point;
   TreeNode* left;
   TreeNode* right;
   BinTree bin; // Only for leaf nodes
   
   TreeNode() : split_point(0.0), left(nullptr), right(nullptr) {}
 };
 
 // Function to calculate WoE
 double calculateWoE(double perc_good, double perc_bad) {
   // To handle division by zero, add a small epsilon
   const double epsilon = 1e-10;
   perc_good = perc_good < epsilon ? epsilon : perc_good;
   perc_bad = perc_bad < epsilon ? epsilon : perc_bad;
   return std::log(perc_good / perc_bad);
 }
 
 // Function to calculate IV
 double calculateIV(double perc_good, double perc_bad, double woe) {
   return (perc_good - perc_bad) * woe;
 }
 
 // Function to build the decision tree
 TreeNode* buildTree(const std::vector<double>& feature,
                     const std::vector<int>& target,
                     int start,
                     int end,
                     int total_pos,
                     int total_neg,
                     double lambda,
                     double min_iv_gain,
                     double min_bin_size,
                     int max_depth,
                     int current_depth,
                     std::string monotonicity_direction) {
   
   // Initialize a new tree node
   TreeNode* node = new TreeNode();
   
   // Calculate total positives and negatives in this segment
   int count_pos = 0;
   int count_neg = 0;
   for(int i = start; i < end; ++i){
     if(target[i] == 1) count_pos++;
     else count_neg++;
   }
   
   // Calculate WoE and IV for this node
   double perc_good = (double)count_pos / total_pos;
   double perc_bad = (double)count_neg / total_neg;
   double woe = calculateWoE(perc_good, perc_bad);
   double iv = calculateIV(perc_good, perc_bad, woe);
   
   // Check stopping criteria (removed iv < min_iv_gain)
   double bin_size = (double)(end - start) / feature.size();
   if(bin_size < min_bin_size || current_depth >= max_depth){
     // Assign bin information to this leaf node
     node->bin.start = feature[start];
     node->bin.end = feature[end - 1];
     node->bin.count = end - start;
     node->bin.count_pos = count_pos;
     node->bin.count_neg = count_neg;
     node->bin.woe = woe;
     node->bin.iv = iv;
     return node;
   }
   
   // Try to find the best split
   double best_split = feature[start];
   double best_gain = -std::numeric_limits<double>::infinity();
   int best_split_idx = -1;
   
   // Precompute cumulative positives and negatives for efficiency
   std::vector<int> cum_pos(end - start, 0);
   std::vector<int> cum_neg(end - start, 0);
   cum_pos[0] = (target[start] == 1) ? 1 : 0;
   cum_neg[0] = (target[start] == 0) ? 1 : 0;
   for(int i = start +1; i < end; ++i){
     cum_pos[i - start] = cum_pos[i - start -1] + ((target[i] ==1) ? 1 : 0);
     cum_neg[i - start] = cum_neg[i - start -1] + ((target[i] ==0) ? 1 : 0);
   }
   
   for(int i = start + 1; i < end; ++i){
     if(feature[i] == feature[i-1]){
       continue; // Skip identical values
     }
     
     // Calculate left counts using cumulative sums
     int left_pos = cum_pos[i - start -1];
     int left_neg = cum_neg[i - start -1];
     int right_pos = count_pos - left_pos;
     int right_neg = count_neg - left_neg;
     
     // Ensure minimum bin size
     double left_bin_size = (double)(i - start) / feature.size();
     double right_bin_size = (double)(end - i) / feature.size();
     if(left_bin_size < min_bin_size || right_bin_size < min_bin_size){
       continue;
     }
     
     // Calculate WoE and IV for left and right
     double perc_good_left = (double)left_pos / total_pos;
     double perc_bad_left = (double)left_neg / total_neg;
     double woe_left = calculateWoE(perc_good_left, perc_bad_left);
     double iv_left = calculateIV(perc_good_left, perc_bad_left, woe_left);
     
     double perc_good_right = (double)right_pos / total_pos;
     double perc_bad_right = (double)right_neg / total_neg;
     double woe_right = calculateWoE(perc_good_right, perc_bad_right);
     double iv_right = calculateIV(perc_good_right, perc_bad_right, woe_right);
     
     // Use Information Value gain as the splitting criterion
     double iv_gain = iv_left + iv_right - iv;
     
     if(iv_gain > best_gain){
       best_gain = iv_gain;
       best_split = (feature[i-1] + feature[i]) / 2.0;
       best_split_idx = i;
     }
   }
   
   // If no valid split is found or gain is insufficient, make this a leaf node
   if(best_split_idx == -1 || best_gain < min_iv_gain){
     node->bin.start = feature[start];
     node->bin.end = feature[end - 1];
     node->bin.count = end - start;
     node->bin.count_pos = count_pos;
     node->bin.count_neg = count_neg;
     node->bin.woe = woe;
     node->bin.iv = iv;
     return node;
   }
   
   // Assign the best split point
   node->split_point = best_split;
   
   // Recursively build left and right subtrees
   node->left = buildTree(feature, target, start, best_split_idx, total_pos, total_neg,
                          lambda, min_iv_gain, min_bin_size, max_depth, current_depth + 1, monotonicity_direction);
   node->right = buildTree(feature, target, best_split_idx, end, total_pos, total_neg,
                           lambda, min_iv_gain, min_bin_size, max_depth, current_depth + 1, monotonicity_direction);
   
   return node;
 }
 
 // Function to collect bins from the tree
 void collectBins(TreeNode* node, std::vector<BinTree>& bins){
   if(node->left == nullptr && node->right == nullptr){
     bins.push_back(node->bin);
   }
   else{
     if(node->left != nullptr) collectBins(node->left, bins);
     if(node->right != nullptr) collectBins(node->right, bins);
   }
 }
 
 // Function to enforce monotonicity
 void enforceMonotonicity(std::vector<BinTree>& bins, std::string direction){
   bool is_monotonic = false;
   while(!is_monotonic){
     is_monotonic = true;
     for(int i = 0; i < bins.size() - 1; ++i){
       if(direction == "increase" && bins[i].woe > bins[i+1].woe){
         // Merge bins i and i+1
         bins[i].end = bins[i+1].end;
         bins[i].count += bins[i+1].count;
         bins[i].count_pos += bins[i+1].count_pos;
         bins[i].count_neg += bins[i+1].count_neg;
         bins[i].woe = calculateWoE((double)bins[i].count_pos / bins[i].count,
                                    (double)bins[i].count_neg / bins[i].count);
         bins[i].iv = calculateIV((double)bins[i].count_pos / bins[i].count,
                                  (double)bins[i].count_neg / bins[i].count,
                                  bins[i].woe);
         bins.erase(bins.begin() + i + 1);
         is_monotonic = false;
         break;
       }
       else if(direction == "decrease" && bins[i].woe < bins[i+1].woe){
         // Merge bins i and i+1
         bins[i].end = bins[i+1].end;
         bins[i].count += bins[i+1].count;
         bins[i].count_pos += bins[i+1].count_pos;
         bins[i].count_neg += bins[i+1].count_neg;
         bins[i].woe = calculateWoE((double)bins[i].count_pos / bins[i].count,
                                    (double)bins[i].count_neg / bins[i].count);
         bins[i].iv = calculateIV((double)bins[i].count_pos / bins[i].count,
                                  (double)bins[i].count_neg / bins[i].count,
                                  bins[i].woe);
         bins.erase(bins.begin() + i + 1);
         is_monotonic = false;
         break;
       }
     }
   }
 }
 
 // Function to prune bins based on IV
 void pruneBins(std::vector<BinTree>& bins, double min_iv_gain){
   bool pruned = false;
   do{
     pruned = false;
     double max_gain = -std::numeric_limits<double>::infinity();
     int merge_idx = -1;
     
     for(int i = 0; i < bins.size() - 1; ++i){
       // Calculate combined IV if bins i and i+1 are merged
       int combined_count = bins[i].count + bins[i+1].count;
       int combined_pos = bins[i].count_pos + bins[i+1].count_pos;
       int combined_neg = bins[i].count_neg + bins[i+1].count_neg;
       double perc_good = (double)combined_pos / combined_count;
       double perc_bad = (double)combined_neg / combined_count;
       double woe = calculateWoE(perc_good, perc_bad);
       double iv = calculateIV(perc_good, perc_bad, woe);
       double gain = iv - (bins[i].iv + bins[i+1].iv);
       
       if(gain > max_gain && gain > min_iv_gain){
         max_gain = gain;
         merge_idx = i;
       }
     }
     
     if(merge_idx != -1){
       // Merge bins at merge_idx and merge_idx +1
       bins[merge_idx].end = bins[merge_idx +1].end;
       bins[merge_idx].count += bins[merge_idx +1].count;
       bins[merge_idx].count_pos += bins[merge_idx +1].count_pos;
       bins[merge_idx].count_neg += bins[merge_idx +1].count_neg;
       bins[merge_idx].woe = calculateWoE((double)bins[merge_idx].count_pos / bins[merge_idx].count,
                                          (double)bins[merge_idx].count_neg / bins[merge_idx].count);
       bins[merge_idx].iv = calculateIV((double)bins[merge_idx].count_pos / bins[merge_idx].count,
                                        (double)bins[merge_idx].count_neg / bins[merge_idx].count,
                                        bins[merge_idx].woe);
       bins.erase(bins.begin() + merge_idx +1);
       pruned = true;
     }
   } while(pruned);
 }
 
 //' Performs optimal binning of a numeric variable for Weight of Evidence (WoE) and Information Value (IV) using a decision tree-based approach
 //'
 //' This function processes a numeric variable and applies a decision tree algorithm to iteratively split the data into bins. The tree splits are based on Information Value (IV) gain, with constraints on minimum bin size and maximum depth. The function ensures monotonicity in event rates, merging bins if necessary to reduce the number of bins to the specified maximum. It also calculates WoE and IV for the generated bins.
 //'
 //' @param target Integer vector representing the binary target variable, where 1 indicates a positive event (e.g., default) and 0 indicates a negative event (e.g., non-default).
 //' @param feature Numeric vector representing the numeric variable to be binned.
 //' @param max_bins (Optional) Maximum number of bins to generate. Default is 7.
 //' @param lambda (Optional) Regularization parameter to penalize tree splits. Default is 0.1.
 //' @param min_bin_size (Optional) Minimum size a bin must have as a proportion of the total data. Default is 0.05.
 //' @param min_iv_gain (Optional) Minimum Information Value (IV) gain required to perform a split. Default is 0.01.
 //' @param max_depth (Optional) Maximum depth of the decision tree. Default is 10.
 //' @param monotonicity_direction (Optional) String that defines the monotonicity direction of event rates, either "increase" for increasing monotonicity or "decrease" for decreasing monotonicity. Default is "increase".
 //'
 //' @return A list with the following elements:
 //' \itemize{
 //'   \item \code{feature_woe}: Numeric vector with the WoE assigned to each instance of the processed numeric variable.
 //'   \item \code{bin}: DataFrame with the generated bins, containing the following fields:
 //'     \itemize{
 //'       \item \code{bin}: String representing the range of values for each bin.
 //'       \item \code{woe}: Weight of Evidence (WoE) for each bin.
 //'       \item \code{count}: Total number of observations in each bin.
 //'       \item \code{count_pos}: Count of positive events in each bin.
 //'       \item \code{count_neg}: Count of negative events in each bin.
 //'     }
 //'   \item \code{woe}: Numeric vector with the WoE for each bin.
 //'   \item \code{iv}: Total Information Value (IV) calculated for the variable.
 //'   \item \code{pos}: Vector with the count of positive events in each bin.
 //'   \item \code{neg}: Vector with the count of negative events in each bin.
 //' }
 //'
 //'
 // [[Rcpp::export]]
 List OptimalBinningNumericTree(IntegerVector target, NumericVector feature, int max_bins = 7, double lambda = 0.1, double min_bin_size = 0.05, double min_iv_gain = 0.01, int max_depth = 10, std::string monotonicity_direction = "increase"){
   
   // Check input lengths
   int n = feature.size();
   if(n != target.size()){
     stop("Feature and target vectors must be of the same length.");
   }
   
   // Check target is binary
   for(int i = 0; i < n; ++i){
     if(target[i] != 0 && target[i] != 1){
       stop("Target vector must be binary (0 and 1).");
     }
   }
   
   // Create a vector of original indices
   std::vector<int> original_indices(n);
   for(int i = 0; i < n; ++i){
     original_indices[i] = i;
   }
   
   // Sort the indices based on feature values
   std::sort(original_indices.begin(), original_indices.end(),
             [&](int a, int b) -> bool{
               return feature[a] < feature[b];
             });
   
   // Create sorted_feature and sorted_target based on sorted indices
   std::vector<double> sorted_feature(n);
   std::vector<int> sorted_target(n);
   for(int i = 0; i < n; ++i){
     sorted_feature[i] = feature[original_indices[i]];
     sorted_target[i] = target[original_indices[i]];
   }
   
   // Calculate total positives and negatives
   int total_pos = 0;
   int total_neg = 0;
   for(int i = 0; i < n; ++i){
     if(sorted_target[i] == 1) total_pos++;
     else total_neg++;
   }
   
   // Handle case where total_pos or total_neg is zero
   if(total_pos == 0 || total_neg == 0){
     stop("The target variable must have both positive and negative classes.");
   }
   
   // Build the initial decision tree
   TreeNode* root = buildTree(sorted_feature, sorted_target, 0, n, total_pos, total_neg,
                              lambda, min_iv_gain, min_bin_size, max_depth, 0, monotonicity_direction);
   
   // Collect bins from the tree
   std::vector<BinTree> bins;
   collectBins(root, bins);
   
   // Ensure maximum number of bins
   while(bins.size() > max_bins){
     // Find the pair of adjacent bins with the smallest IV and merge them
     double min_iv = std::numeric_limits<double>::infinity();
     int merge_idx = -1;
     for(int i = 0; i < bins.size() -1; ++i){
       if(bins[i].iv < min_iv){
         min_iv = bins[i].iv;
         merge_idx = i;
       }
     }
     if(merge_idx == -1){
       break;
     }
     // Merge bins at merge_idx and merge_idx +1
     bins[merge_idx].end = bins[merge_idx +1].end;
     bins[merge_idx].count += bins[merge_idx +1].count;
     bins[merge_idx].count_pos += bins[merge_idx +1].count_pos;
     bins[merge_idx].count_neg += bins[merge_idx +1].count_neg;
     bins[merge_idx].woe = calculateWoE((double)bins[merge_idx].count_pos / bins[merge_idx].count,
                                        (double)bins[merge_idx].count_neg / bins[merge_idx].count);
     bins[merge_idx].iv = calculateIV((double)bins[merge_idx].count_pos / bins[merge_idx].count,
                                      (double)bins[merge_idx].count_neg / bins[merge_idx].count,
                                      bins[merge_idx].woe);
     bins.erase(bins.begin() + merge_idx +1);
   }
   
   // Enforce monotonicity
   enforceMonotonicity(bins, monotonicity_direction);
   
   // Final pruning based on IV gain
   pruneBins(bins, min_iv_gain);
   
   // If after pruning, bins exceed max_bins, merge the least IV gain bins
   while(bins.size() > max_bins){
     // Find the pair of adjacent bins with the smallest IV and merge them
     double min_iv = std::numeric_limits<double>::infinity();
     int merge_idx = -1;
     for(int i = 0; i < bins.size() -1; ++i){
       if(bins[i].iv < min_iv){
         min_iv = bins[i].iv;
         merge_idx = i;
       }
     }
     if(merge_idx == -1){
       break;
     }
     // Merge bins at merge_idx and merge_idx +1
     bins[merge_idx].end = bins[merge_idx +1].end;
     bins[merge_idx].count += bins[merge_idx +1].count;
     bins[merge_idx].count_pos += bins[merge_idx +1].count_pos;
     bins[merge_idx].count_neg += bins[merge_idx +1].count_neg;
     bins[merge_idx].woe = calculateWoE((double)bins[merge_idx].count_pos / bins[merge_idx].count,
                                        (double)bins[merge_idx].count_neg / bins[merge_idx].count);
     bins[merge_idx].iv = calculateIV((double)bins[merge_idx].count_pos / bins[merge_idx].count,
                                      (double)bins[merge_idx].count_neg / bins[merge_idx].count,
                                      bins[merge_idx].woe);
     bins.erase(bins.begin() + merge_idx +1);
   }
   
   // Create WoE mapping for each feature value based on sorted order
   std::vector<double> feature_woe_sorted(n, 0.0);
   int bin_idx = 0;
   for(int i = 0; i < n; ++i){
     if(bin_idx < bins.size() -1 && sorted_feature[i] > bins[bin_idx].end){
       bin_idx++;
     }
     feature_woe_sorted[i] = bins[bin_idx].woe;
   }
   
   // Map the sorted WOE back to the original feature order
   std::vector<double> feature_woe_original(n, 0.0);
   for(int i = 0; i < n; ++i){
     feature_woe_original[original_indices[i]] = feature_woe_sorted[i];
   }
   
   // Prepare bin names with the desired interval format
   std::vector<std::string> bin_names;
   for(int i = 0; i < bins.size(); ++i){
     std::string bin_name;
     if(i == 0){
       // First bin: [-Inf; end)
       bin_name = "[-Inf;" + std::to_string(bins[i].end) + ")";
     }
     else if(i == bins.size() -1){
       // Last bin: (start;+Inf]
       bin_name = "(" + std::to_string(bins[i].start) + "; +Inf]";
     }
     else{
       // Middle bins: [start; end)
       bin_name = "[" + std::to_string(bins[i].start) + "; " + std::to_string(bins[i].end) + ")";
     }
     bin_names.push_back(bin_name);
   }
   
   // Prepare vectors for bin DataFrame
   std::vector<double> woe_values;
   std::vector<double> iv_values;
   std::vector<int> bin_counts;
   std::vector<int> pos_counts;
   std::vector<int> neg_counts;
   for(int i = 0; i < bins.size(); ++i){
     woe_values.push_back(bins[i].woe);
     iv_values.push_back(bins[i].iv);
     bin_counts.push_back(bins[i].count);
     pos_counts.push_back(bins[i].count_pos);
     neg_counts.push_back(bins[i].count_neg);
   }
   
   // Calculate total IV
   double total_iv = 0.0;
   for(int i = 0; i < bins.size(); ++i){
     total_iv += bins[i].iv;
   }
   
   // total_iv
   
   List bin_lst = List::create(
     Named("bin") = bin_names,
     Named("woe") = woe_values,
     Named("iv") = iv_values,
     Named("count") = bin_counts,
     Named("count_pos") = pos_counts,
     Named("count_neg") = neg_counts
   );
   
   // Create List for woe vector feature
   List woe_lst = List::create(
     Named("woefeature") = feature_woe_original
   );
   
   // Attrib class for compatibility with data.table in memory superfast tables
   bin_lst.attr("class") = CharacterVector::create("data.table", "data.frame");
   woe_lst.attr("class") = CharacterVector::create("data.table", "data.frame");
   
   List output_list = List::create(
     Named("woefeature") = woe_lst,
     Named("woebin") = bin_lst
   // Named("woe") = woe,
   // Named("iv") = total_iv,
   // Named("pos") = pos,
   // Named("neg") = neg
   );
   
   // // Create bin DataFrame with updated bin names
   // DataFrame bin_df = DataFrame::create(
   //   Named("bin") = bin_names,
   //   Named("woe") = woe_values,
   //   Named("count") = bin_counts,
   //   Named("count_pos") = pos_counts,
   //   Named("count_neg") = neg_counts
   // );
   // 
   // // Return as Rcpp List with the desired structure
   // List output_list = List::create(
   //   Named("woefeature") = feature_woe_original,
   //   Named("woebin") = bin_df
   //   // Named("woe") = woe_values,
   //   // Named("iv") = total_iv,
   //   // Named("pos") = pos_counts,
   //   // Named("neg") = neg_counts
   // );
   // 
   // output_list.attr("class") = CharacterVector::create("data.table", "data.frame");
   
   return output_list;
   
 }
 
 // [[Rcpp::export]]
 Rcpp::List OptimalBinningCategoricalBreakList(Rcpp::IntegerVector target, 
                                               Rcpp::CharacterVector feature,
                                               Rcpp::List predefined_bins) {
   int N = target.size();
   if (feature.size() != N) {
     Rcpp::stop("Length of target and feature must be the same.");
   }
   
   // Validate target values
   for (int i = 0; i < N; ++i) {
     if (target[i] != 0 && target[i] != 1) {
       Rcpp::stop("Target must contain only 0s and 1s.");
     }
   }
   
   // Create a map to store the bin assignment for each category
   std::map<std::string, int> category_to_bin;
   int num_bins = predefined_bins.size();
   
   for (int i = 0; i < num_bins; ++i) {
     Rcpp::CharacterVector bin_categories = predefined_bins[i];
     for (int j = 0; j < bin_categories.size(); ++j) {
       std::string category = Rcpp::as<std::string>(bin_categories[j]);
       category_to_bin[category] = i;
     }
   }
   
   // Initialize bins
   struct Bin {
     Rcpp::CharacterVector categories;
     int pos_count;
     int neg_count;
     double event_rate;
   };
   std::vector<Bin> bins(num_bins);
   
   for (int i = 0; i < num_bins; ++i) {
     bins[i].categories = predefined_bins[i];
     bins[i].pos_count = 0;
     bins[i].neg_count = 0;
   }
   
   // Process data and assign to bins
   Rcpp::CharacterVector feature_processed(N);
   int unassigned_count = 0;
   
   for (int i = 0; i < N; ++i) {
     if (feature[i] == NA_STRING) {
       Rcpp::stop("NA values are not allowed in the feature vector.");
     }
     
     std::string cat = Rcpp::as<std::string>(feature[i]);
     auto it = category_to_bin.find(cat);
     
     if (it != category_to_bin.end()) {
       int bin_index = it->second;
       feature_processed[i] = Rcpp::as<Rcpp::CharacterVector>(predefined_bins[bin_index])[0];  // Use the first category as the bin name
       
       if (target[i] == 1) {
         bins[bin_index].pos_count++;
       } else {
         bins[bin_index].neg_count++;
       }
     } else {
       // Category not found in predefined bins
       feature_processed[i] = "Unassigned";
       unassigned_count++;
     }
   }
   
   if (unassigned_count > 0) {
     Rcpp::warning(std::to_string(unassigned_count) + " observations were unassigned to any predefined bin.");
   }
   
   // Calculate event rates and sort bins
   for (auto& bin : bins) {
     int total = bin.pos_count + bin.neg_count;
     bin.event_rate = total > 0 ? (double)bin.pos_count / total : 0.0;
   }
   
   std::sort(bins.begin(), bins.end(), 
             [](const Bin& a, const Bin& b) { return a.event_rate < b.event_rate; });
   
   // Compute WoE and IV
   int total_pos = 0;
   int total_neg = 0;
   for (const auto& bin : bins) {
     total_pos += bin.pos_count;
     total_neg += bin.neg_count;
   }
   
   std::vector<double> woe(num_bins);
   std::vector<double> iv_bin(num_bins);
   double total_iv = 0.0;
   
   for (int i = 0; i < num_bins; ++i) {
     double dist_pos = (double)bins[i].pos_count / total_pos;
     double dist_neg = (double)bins[i].neg_count / total_neg;
     if (dist_pos == 0) dist_pos = 1e-10;
     if (dist_neg == 0) dist_neg = 1e-10;
     woe[i] = std::log(dist_pos / dist_neg);
     iv_bin[i] = (dist_pos - dist_neg) * woe[i];
     total_iv += iv_bin[i];
   }
   
   // Map categories to WoE
   std::map<std::string, double> category_woe_map;
   for (int i = 0; i < num_bins; ++i) {
     for (int j = 0; j < bins[i].categories.size(); ++j) {
       std::string cat = Rcpp::as<std::string>(bins[i].categories[j]);
       category_woe_map[cat] = woe[i];
     }
   }
   
   Rcpp::NumericVector feature_woe(N);
   for (int i = 0; i < N; ++i) {
     std::string cat = Rcpp::as<std::string>(feature_processed[i]);
     auto it = category_woe_map.find(cat);
     if (it != category_woe_map.end()) {
       feature_woe[i] = it->second;
     } else {
       feature_woe[i] = NA_REAL;
     }
   }
   
   // Prepare bin output
   Rcpp::CharacterVector bin_names(num_bins);
   Rcpp::IntegerVector count(num_bins);
   Rcpp::IntegerVector pos(num_bins);
   Rcpp::IntegerVector neg(num_bins);
   
   for (int i = 0; i < num_bins; ++i) {
     std::string bin_name = "";
     for (size_t j = 0; j < bins[i].categories.size(); ++j) {
       if (j > 0) bin_name += "+";
       bin_name += bins[i].categories[j];
     }
     bin_names[i] = bin_name;
     count[i] = bins[i].pos_count + bins[i].neg_count;
     pos[i] = bins[i].pos_count;
     neg[i] = bins[i].neg_count;
   }
   
   // Prepare bin output
   // Rcpp::CharacterVector bin_names(num_bins);
   // Rcpp::IntegerVector count(num_bins);
   // Rcpp::IntegerVector pos(num_bins);
   // Rcpp::IntegerVector neg(num_bins);
   // 
   // for (int i = 0; i < num_bins; ++i) {
   //   bin_names[i] = Rcpp::collapse(bins[i].categories, "+");
   //   count[i] = bins[i].pos_count + bins[i].neg_count;
   //   pos[i] = bins[i].pos_count;
   //   neg[i] = bins[i].neg_count;
   // }
   
   // Create List for bins
   Rcpp::List bin_lst = Rcpp::List::create(
     Rcpp::Named("bin") = bin_names,
     Rcpp::Named("woe") = woe,
     Rcpp::Named("iv") = iv_bin,
     Rcpp::Named("count") = count,
     Rcpp::Named("count_pos") = pos,
     Rcpp::Named("count_neg") = neg);
   
   // Create List for woe vector feature
   Rcpp::List woe_lst = Rcpp::List::create(
     Rcpp::Named("woefeature") = feature_woe
   );
   
   // Attrib class for compatibility with data.table in memory superfast tables
   bin_lst.attr("class") = Rcpp::CharacterVector::create("data.table", "data.frame");
   woe_lst.attr("class") = Rcpp::CharacterVector::create("data.table", "data.frame");
   
   // Return output
   Rcpp::List output_list = Rcpp::List::create(
     Rcpp::Named("woefeature") = woe_lst,
     Rcpp::Named("woebin") = bin_lst
   );
   
   return output_list;
 }
 
 // Rcpp::List OptimalBinningCategoricalBreakList(Rcpp::IntegerVector target, 
 //                                               Rcpp::CharacterVector feature,
 //                                               Rcpp::List predefined_bins) {
 //   int N = target.size();
 //   if (feature.size() != N) {
 //     Rcpp::stop("Length of target and feature must be the same.");
 //   }
 //   
 //   // Validate target values
 //   for (int i = 0; i < N; ++i) {
 //     if (target[i] != 0 && target[i] != 1) {
 //       Rcpp::stop("Target must contain only 0s and 1s.");
 //     }
 //   }
 //   
 //   // Create a map to store the bin assignment for each category
 //   std::map<std::string, int> category_to_bin;
 //   int num_bins = predefined_bins.size();
 //   
 //   for (int i = 0; i < num_bins; ++i) {
 //     Rcpp::CharacterVector bin_categories = predefined_bins[i];
 //     for (int j = 0; j < bin_categories.size(); ++j) {
 //       std::string category = Rcpp::as<std::string>(bin_categories[j]);
 //       category_to_bin[category] = i;
 //     }
 //   }
 //   
 //   // Initialize bins
 //   struct Bin {
 //     std::vector<std::string> categories;
 //     int pos_count;
 //     int neg_count;
 //     double event_rate;
 //   };
 //   std::vector<Bin> bins(num_bins);
 //   
 //   for (int i = 0; i < num_bins; ++i) {
 //     bins[i].categories = Rcpp::as<std::vector<std::string>>(predefined_bins[i]);
 //     bins[i].pos_count = 0;
 //     bins[i].neg_count = 0;
 //   }
 //   
 //   // Process data and assign to bins
 //   Rcpp::CharacterVector feature_processed(N);
 //   int unassigned_count = 0;
 //   
 //   for (int i = 0; i < N; ++i) {
 //     if (feature[i] == NA_STRING) {
 //       Rcpp::stop("NA values are not allowed in the feature vector.");
 //     }
 //     
 //     std::string cat = Rcpp::as<std::string>(feature[i]);
 //     auto it = category_to_bin.find(cat);
 //     
 //     if (it != category_to_bin.end()) {
 //       int bin_index = it->second;
 //       feature_processed[i] = Rcpp::as<std::string>(predefined_bins[bin_index]);
 //       
 //       if (target[i] == 1) {
 //         bins[bin_index].pos_count++;
 //       } else {
 //         bins[bin_index].neg_count++;
 //       }
 //     } else {
 //       // Category not found in predefined bins
 //       feature_processed[i] = "Unassigned";
 //       unassigned_count++;
 //     }
 //   }
 //   
 //   if (unassigned_count > 0) {
 //     Rcpp::warning(std::to_string(unassigned_count) + " observations were unassigned to any predefined bin.");
 //   }
 //   
 //   // Calculate event rates and sort bins
 //   for (auto& bin : bins) {
 //     bin.event_rate = (double)bin.pos_count / (bin.pos_count + bin.neg_count);
 //   }
 //   
 //   std::sort(bins.begin(), bins.end(), 
 //             [](const Bin& a, const Bin& b) { return a.event_rate < b.event_rate; });
 //   
 //   // Compute WoE and IV
 //   int total_pos = 0;
 //   int total_neg = 0;
 //   for (const auto& bin : bins) {
 //     total_pos += bin.pos_count;
 //     total_neg += bin.neg_count;
 //   }
 //   
 //   std::vector<double> woe(num_bins);
 //   std::vector<double> iv_bin(num_bins);
 //   double total_iv = 0.0;
 //   
 //   for (int i = 0; i < num_bins; ++i) {
 //     double dist_pos = (double)bins[i].pos_count / total_pos;
 //     double dist_neg = (double)bins[i].neg_count / total_neg;
 //     if (dist_pos == 0) dist_pos = 1e-10;
 //     if (dist_neg == 0) dist_neg = 1e-10;
 //     woe[i] = std::log(dist_pos / dist_neg);
 //     iv_bin[i] = (dist_pos - dist_neg) * woe[i];
 //     total_iv += iv_bin[i];
 //   }
 //   
 //   // Map categories to WoE
 //   std::map<std::string, double> category_woe_map;
 //   for (int i = 0; i < num_bins; ++i) {
 //     for (const auto& cat : bins[i].categories) {
 //       category_woe_map[cat] = woe[i];
 //     }
 //   }
 //   
 //   Rcpp::NumericVector feature_woe(N);
 //   for (int i = 0; i < N; ++i) {
 //     auto it = category_woe_map.find(Rcpp::as<std::string>(feature_processed[i]));
 //     if (it != category_woe_map.end()) {
 //       feature_woe[i] = it->second;
 //     } else {
 //       feature_woe[i] = NA_REAL;
 //     }
 //   }
 //   
 //   // Prepare bin output
 //   Rcpp::CharacterVector bin_names(num_bins);
 //   Rcpp::IntegerVector count(num_bins);
 //   Rcpp::IntegerVector pos(num_bins);
 //   Rcpp::IntegerVector neg(num_bins);
 //   
 //   for (int i = 0; i < num_bins; ++i) {
 //     std::string bin_name = "";
 //     for (size_t j = 0; j < bins[i].categories.size(); ++j) {
 //       if (j > 0) bin_name += "+";
 //       bin_name += bins[i].categories[j];
 //     }
 //     bin_names[i] = bin_name;
 //     count[i] = bins[i].pos_count + bins[i].neg_count;
 //     pos[i] = bins[i].pos_count;
 //     neg[i] = bins[i].neg_count;
 //   }
 //   
 //   // Create List for bins
 //   Rcpp::List bin_lst = Rcpp::List::create(
 //     Rcpp::Named("bin") = bin_names,
 //     Rcpp::Named("woe") = woe,
 //     Rcpp::Named("iv") = iv_bin,
 //     Rcpp::Named("count") = count,
 //     Rcpp::Named("count_pos") = pos,
 //     Rcpp::Named("count_neg") = neg);
 //   
 //   // Create List for woe vector feature
 //   Rcpp::List woe_lst = Rcpp::List::create(
 //     Rcpp::Named("woefeature") = feature_woe
 //   );
 //   
 //   // Attrib class for compatibility with data.table in memory superfast tables
 //   bin_lst.attr("class") = Rcpp::CharacterVector::create("data.table", "data.frame");
 //   woe_lst.attr("class") = Rcpp::CharacterVector::create("data.table", "data.frame");
 //   
 //   // Return output
 //   Rcpp::List output_list = Rcpp::List::create(
 //     Rcpp::Named("woefeature") = woe_lst,
 //     Rcpp::Named("woebin") = bin_lst
 //   );
 //   
 //   return output_list;
 // }
 
 // [[Rcpp::export]]
 Rcpp::List OptimalBinningNumericalBreakList(Rcpp::IntegerVector target, 
                                             Rcpp::NumericVector feature,
                                             Rcpp::NumericVector break_points) {
   int N = target.size();
   if (feature.size() != N) {
     Rcpp::stop("Length of target and feature must be the same.");
   }
   
   // Validate target values
   for (int i = 0; i < N; ++i) {
     if (target[i] != 0 && target[i] != 1) {
       Rcpp::stop("Target must contain only 0s and 1s.");
     }
   }
   
   // Ensure break_points are sorted
   std::vector<double> sorted_breaks = Rcpp::as<std::vector<double>>(break_points);
   std::sort(sorted_breaks.begin(), sorted_breaks.end());
   
   int num_bins = sorted_breaks.size() + 1;
   
   // Initialize bins
   struct Bin {
     double lower_bound;
     double upper_bound;
     int pos_count;
     int neg_count;
     double event_rate;
   };
   std::vector<Bin> bins(num_bins);
   
   bins[0].lower_bound = -std::numeric_limits<double>::infinity();
   bins[0].upper_bound = sorted_breaks[0];
   for (int i = 1; i < num_bins - 1; ++i) {
     bins[i].lower_bound = sorted_breaks[i-1];
     bins[i].upper_bound = sorted_breaks[i];
   }
   bins[num_bins-1].lower_bound = sorted_breaks.back();
   bins[num_bins-1].upper_bound = std::numeric_limits<double>::infinity();
   
   // Process data and assign to bins
   Rcpp::NumericVector feature_processed(N);
   int unassigned_count = 0;
   
   for (int i = 0; i < N; ++i) {
     if (Rcpp::NumericVector::is_na(feature[i])) {
       Rcpp::stop("NA values are not allowed in the feature vector.");
     }
     
     double value = feature[i];
     int bin_index = std::lower_bound(sorted_breaks.begin(), sorted_breaks.end(), value) - sorted_breaks.begin();
     
     feature_processed[i] = bin_index;
     
     if (target[i] == 1) {
       bins[bin_index].pos_count++;
     } else {
       bins[bin_index].neg_count++;
     }
   }
   
   // Calculate event rates
   for (auto& bin : bins) {
     int total = bin.pos_count + bin.neg_count;
     bin.event_rate = total > 0 ? (double)bin.pos_count / total : 0.0;
   }
   
   // Compute WoE and IV
   int total_pos = 0;
   int total_neg = 0;
   for (const auto& bin : bins) {
     total_pos += bin.pos_count;
     total_neg += bin.neg_count;
   }
   
   std::vector<double> woe(num_bins);
   std::vector<double> iv_bin(num_bins);
   double total_iv = 0.0;
   
   for (int i = 0; i < num_bins; ++i) {
     double dist_pos = (double)bins[i].pos_count / total_pos;
     double dist_neg = (double)bins[i].neg_count / total_neg;
     if (dist_pos == 0) dist_pos = 1e-10;
     if (dist_neg == 0) dist_neg = 1e-10;
     woe[i] = std::log(dist_pos / dist_neg);
     iv_bin[i] = (dist_pos - dist_neg) * woe[i];
     total_iv += iv_bin[i];
   }
   
   // Map feature values to WoE
   Rcpp::NumericVector feature_woe(N);
   for (int i = 0; i < N; ++i) {
     int bin_index = feature_processed[i];
     feature_woe[i] = woe[bin_index];
   }
   
   // Prepare bin output
   Rcpp::CharacterVector bin_names(num_bins);
   Rcpp::IntegerVector count(num_bins);
   Rcpp::IntegerVector pos(num_bins);
   Rcpp::IntegerVector neg(num_bins);
   
   for (int i = 0; i < num_bins; ++i) {
     std::ostringstream oss;
     if (i == 0) {
       oss << "(-Inf," << bins[i].upper_bound << "]";
     } else if (i == num_bins - 1) {
       oss << "(" << bins[i].lower_bound << ",Inf)";
     } else {
       oss << "(" << bins[i].lower_bound << "," << bins[i].upper_bound << "]";
     }
     bin_names[i] = oss.str();
     count[i] = bins[i].pos_count + bins[i].neg_count;
     pos[i] = bins[i].pos_count;
     neg[i] = bins[i].neg_count;
   }
   
   // Create List for bins
   Rcpp::List bin_lst = Rcpp::List::create(
     Rcpp::Named("bin") = bin_names,
     Rcpp::Named("woe") = woe,
     Rcpp::Named("iv") = iv_bin,
     Rcpp::Named("count") = count,
     Rcpp::Named("count_pos") = pos,
     Rcpp::Named("count_neg") = neg);
   
   // Create List for woe vector feature
   Rcpp::List woe_lst = Rcpp::List::create(
     Rcpp::Named("woefeature") = feature_woe
   );
   
   // Attrib class for compatibility with data.table in memory superfast tables
   bin_lst.attr("class") = Rcpp::CharacterVector::create("data.table", "data.frame");
   woe_lst.attr("class") = Rcpp::CharacterVector::create("data.table", "data.frame");
   
   // Return output
   Rcpp::List output_list = Rcpp::List::create(
     Rcpp::Named("woefeature") = woe_lst,
     Rcpp::Named("woebin") = bin_lst
   );
   
   return output_list;
 }
 