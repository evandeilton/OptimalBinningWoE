// ---------------------------------------------------------------------------------------------- //
// CATEGORICAL VARIABLES
// ---------------------------------------------------------------------------------------------- //

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

// [[Rcpp::plugins(openmp)]]

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

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
 
 