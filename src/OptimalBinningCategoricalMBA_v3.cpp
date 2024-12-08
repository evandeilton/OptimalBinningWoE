// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>
#include <stdexcept>

using namespace Rcpp;

namespace utils {
inline double safe_log(double x) {
 return x > 1e-10 ? std::log(x) : std::log(1e-10);
}

inline std::string join(const std::vector<std::string>& v, const std::string& delimiter) {
 std::string result;
 for (size_t i = 0; i < v.size(); ++i) {
   if (i > 0) result += delimiter;
   result += v[i];
 }
 return result;
}
}

class OptimalBinningCategoricalMBA {
private:
 std::vector<std::string> feature;
 std::vector<int> target;
 int min_bins;
 int max_bins;
 double bin_cutoff;
 int max_n_prebins;
 std::string bin_separator;
 double convergence_threshold;
 int max_iterations;
 
 int total_good;
 int total_bad;
 
 struct Bin {
   std::vector<std::string> categories;
   double woe;
   double iv;
   int count;
   int count_pos;
   int count_neg;
   
   Bin() : woe(0), iv(0), count(0), count_pos(0), count_neg(0) {}
 };
 
 std::vector<Bin> bins;
 
 void validate_inputs();
 void prebinning();
 void enforce_bin_cutoff();
 void calculate_initial_woe();
 void enforce_monotonicity();
 void optimize_bins();
 bool try_merge_bins(size_t index1, size_t index2); // função auxiliar para tentar mesclar sem violar min_bins
 void update_bin_statistics(Bin& bin);
 void check_consistency() const;
 
public:
 OptimalBinningCategoricalMBA(const std::vector<std::string>& feature_,
                              const Rcpp::IntegerVector& target_,
                              int min_bins_ = 3,
                              int max_bins_ = 5,
                              double bin_cutoff_ = 0.05,
                              int max_n_prebins_ = 20,
                              std::string bin_separator_ = "%;%",
                              double convergence_threshold_ = 1e-6,
                              int max_iterations_ = 1000);
 
 Rcpp::List fit();
};

OptimalBinningCategoricalMBA::OptimalBinningCategoricalMBA(
 const std::vector<std::string>& feature_,
 const Rcpp::IntegerVector& target_,
 int min_bins_,
 int max_bins_,
 double bin_cutoff_,
 int max_n_prebins_,
 std::string bin_separator_,
 double convergence_threshold_,
 int max_iterations_
) : feature(feature_), target(Rcpp::as<std::vector<int>>(target_)), 
min_bins(min_bins_), max_bins(max_bins_), 
bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_),
bin_separator(bin_separator_),
convergence_threshold(convergence_threshold_),
max_iterations(max_iterations_) {
 
 total_good = (int)std::count(target.begin(), target.end(), 0);
 total_bad = (int)std::count(target.begin(), target.end(), 1);
}

void OptimalBinningCategoricalMBA::validate_inputs() {
 if (feature.size() != target.size()) {
   throw std::invalid_argument("Feature e target devem ter o mesmo tamanho.");
 }
 if (feature.empty()) {
   throw std::invalid_argument("Feature e target não podem ser vazios.");
 }
 if (min_bins < 2) {
   throw std::invalid_argument("min_bins deve ser >= 2.");
 }
 if (max_bins < min_bins) {
   throw std::invalid_argument("max_bins deve ser >= min_bins.");
 }
 if (bin_cutoff <= 0 || bin_cutoff >= 1) {
   throw std::invalid_argument("bin_cutoff deve estar entre 0 e 1.");
 }
 if (max_n_prebins < max_bins) {
   throw std::invalid_argument("max_n_prebins >= max_bins.");
 }
 for (int val : target) {
   if (val != 0 && val != 1) {
     throw std::invalid_argument("Target deve conter apenas 0 e 1.");
   }
 }
 
 std::unordered_set<std::string> unique_categories(feature.begin(), feature.end());
 int ncat = (int)unique_categories.size();
 if (max_bins > ncat) {
   max_bins = ncat;
 }
}

void OptimalBinningCategoricalMBA::prebinning() {
 std::unordered_map<std::string, int> category_counts;
 std::unordered_map<std::string, int> category_pos_counts;
 
 for (size_t i = 0; i < feature.size(); ++i) {
   const auto& cat = feature[i];
   category_counts[cat]++;
   if (target[i] == 1) {
     category_pos_counts[cat]++;
   }
 }
 
 std::vector<std::pair<std::string,int>> sorted_categories;
 sorted_categories.reserve(category_counts.size());
 for (const auto& pair : category_counts) {
   sorted_categories.emplace_back(pair);
 }
 std::sort(sorted_categories.begin(), sorted_categories.end(),
           [](const auto& a, const auto& b) { return a.second > b.second; });
 
 bins.clear();
 bins.reserve(sorted_categories.size());
 
 for (const auto& pair : sorted_categories) {
   Bin bin;
   bin.categories.push_back(pair.first);
   bin.count = pair.second;
   bin.count_pos = category_pos_counts[pair.first];
   bin.count_neg = bin.count - bin.count_pos;
   bins.push_back(std::move(bin));
 }
 
 while ((int)bins.size() > max_n_prebins) {
   auto min_it = std::min_element(bins.begin(), bins.end(),
                                  [](const Bin& a, const Bin& b) {
                                    return a.count < b.count;
                                  });
   
   size_t idx = (size_t)(min_it - bins.begin());
   if (idx > 0) {
     if (!try_merge_bins(idx - 1, idx)) break;
   } else {
     if (!try_merge_bins(0, 1)) break;
   }
 }
}

void OptimalBinningCategoricalMBA::enforce_bin_cutoff() {
 int min_count = (int)std::ceil(bin_cutoff * (double)feature.size());
 int min_count_pos = (int)std::ceil(bin_cutoff * (double)total_bad);
 
 bool merged;
 do {
   merged = false;
   for (size_t i = 0; i < bins.size(); ++i) {
     if (bins.size() <= (size_t)min_bins) { // Nunca descer abaixo de min_bins
       break;
     }
     if (bins[i].count < min_count || bins[i].count_pos < min_count_pos) {
       size_t merge_index = (i > 0) ? i - 1 : i + 1;
       if (merge_index < bins.size()) {
         if (!try_merge_bins(std::min(i, merge_index), std::max(i, merge_index))) {
           // Se não foi possível mesclar sem violar min_bins, paramos
           break;
         }
         merged = true;
         break;
       }
     }
   }
 } while (merged && (int)bins.size() > min_bins);
}

void OptimalBinningCategoricalMBA::calculate_initial_woe() {
 for (auto& bin : bins) {
   update_bin_statistics(bin);
 }
}

void OptimalBinningCategoricalMBA::enforce_monotonicity() {
 if (bins.empty()) {
   throw std::runtime_error("Nenhum bin disponível para impor monotonicidade.");
 }
 
 std::sort(bins.begin(), bins.end(),
           [](const Bin& a, const Bin& b) { return a.woe < b.woe; });
 
 bool increasing = true;
 for (size_t i = 1; i < bins.size(); ++i) {
   if (bins[i].woe < bins[i-1].woe) {
     increasing = false;
     break;
   }
 }
 
 bool merged;
 do {
   merged = false;
   for (size_t i = 0; i + 1 < bins.size(); ++i) {
     if ((int)bins.size() <= min_bins) { // Não reduzir abaixo de min_bins
       break;
     }
     if ((increasing && bins[i].woe > bins[i+1].woe) ||
         (!increasing && bins[i].woe < bins[i+1].woe)) {
       if (!try_merge_bins(i, i+1)) {
         break;
       }
       merged = true;
       break;
     }
   }
 } while (merged && (int)bins.size() > min_bins);
 
 // Se por algum motivo ficamos abaixo, não mesclar mais.
 // Ajuste final para garantir min_bins.
 while ((int)bins.size() < min_bins && bins.size() > 1) {
   // Neste caso, já estamos abaixo de min_bins. Tentaremos não mesclar mais.
   // Podemos ignorar este passo, pois o usuário quer min_bins. Não mesclar mais.
   // Para garantir o min_bins, não mesclamos se isso diminuir mais os bins.
   // Assim, removemos este bloco de mescla que ocorria aqui.
   break; 
 }
}

void OptimalBinningCategoricalMBA::optimize_bins() {
 int iterations = 0;
 double prev_total_iv = 0.0;
 double total_iv = 0.0;
 
 while (bins.size() > (size_t)max_bins && iterations < max_iterations) {
   if ((int)bins.size() <= min_bins) { // Não reduzir abaixo de min_bins
     break;
   }
   
   double min_combined_iv = std::numeric_limits<double>::max();
   size_t min_iv_index = 0;
   
   for (size_t i = 0; i < bins.size() - 1; ++i) {
     double combined_iv = std::fabs(bins[i].iv) + std::fabs(bins[i+1].iv);
     if (combined_iv < min_combined_iv) {
       min_combined_iv = combined_iv;
       min_iv_index = i;
     }
   }
   
   // Tentar mesclar sem violar min_bins
   if (bins.size() > (size_t)min_bins) {
     if (!try_merge_bins(min_iv_index, min_iv_index + 1)) {
       // Se não pôde mesclar, paramos a otimização.
       break;
     }
   } else {
     break;
   }
   
   total_iv = 0.0;
   for (const auto& bin : bins) {
     total_iv += std::fabs(bin.iv);
   }
   
   if (std::fabs(total_iv - prev_total_iv) < convergence_threshold) {
     break;
   }
   
   prev_total_iv = total_iv;
   iterations++;
 }
 
 if ((int)bins.size() > max_bins) {
   Rcpp::warning("Não foi possível reduzir o número de bins até max_bins sem violar min_bins ou convergência.");
 }
}

bool OptimalBinningCategoricalMBA::try_merge_bins(size_t index1, size_t index2) {
 // Verifica se mesclar resultará em menos bins que min_bins
 if ((int)bins.size() <= min_bins) {
   return false; // Não mesclar, pois já estamos no mínimo.
 }
 
 if (index1 >= bins.size() || index2 >= bins.size() || index1 == index2) {
   return false;
 }
 
 if (index2 < index1) std::swap(index1, index2);
 
 Bin& bin1 = bins[index1];
 Bin& bin2 = bins[index2];
 
 bin1.categories.insert(bin1.categories.end(),
                        std::make_move_iterator(bin2.categories.begin()),
                        std::make_move_iterator(bin2.categories.end()));
 bin1.count += bin2.count;
 bin1.count_pos += bin2.count_pos;
 bin1.count_neg += bin2.count_neg;
 
 update_bin_statistics(bin1);
 
 bins.erase(bins.begin() + index2);
 
 if (bins.empty()) {
   throw std::runtime_error("Todos os bins foram mesclados. Impossível continuar.");
 }
 
 return true;
}

void OptimalBinningCategoricalMBA::update_bin_statistics(Bin& bin) {
 double prop_event = (double)bin.count_pos / std::max(total_bad, 1);
 double prop_non_event = (double)bin.count_neg / std::max(total_good, 1);
 
 prop_event = std::max(prop_event, 1e-10);
 prop_non_event = std::max(prop_non_event, 1e-10);
 
 bin.woe = utils::safe_log(prop_event / prop_non_event);
 bin.iv = (prop_event - prop_non_event) * bin.woe;
}

void OptimalBinningCategoricalMBA::check_consistency() const {
 int total_count = 0;
 int total_count_pos = 0;
 int total_count_neg = 0;
 
 for (const auto& bin : bins) {
   total_count += bin.count;
   total_count_pos += bin.count_pos;
   total_count_neg += bin.count_neg;
 }
 
 if ((size_t)total_count != feature.size()) {
   throw std::runtime_error("Contagem total inconsistente após binagem.");
 }
 
 if (total_count_pos != total_bad || total_count_neg != total_good) {
   throw std::runtime_error("Contagens positivas/negativas inconsistentes após binagem.");
 }
}

Rcpp::List OptimalBinningCategoricalMBA::fit() {
 try {
   validate_inputs();
   prebinning();
   enforce_bin_cutoff();
   calculate_initial_woe();
   enforce_monotonicity();
   
   // Se já temos menos ou igual a max_bins, não precisa otimizar
   bool converged_flag = false;
   int iterations_done = 0;
   
   if (bins.size() <= (size_t)max_bins) {
     converged_flag = true;
   } else {
     double prev_total_iv = 0.0;
     for (int i = 0; i < max_iterations; ++i) {
       optimize_bins();
       double total_iv = 0.0;
       for (const auto& bin : bins) {
         total_iv += std::fabs(bin.iv);
       }
       
       if (std::fabs(total_iv - prev_total_iv) < convergence_threshold) {
         converged_flag = true;
         iterations_done = i + 1;
         break;
       }
       
       prev_total_iv = total_iv;
       iterations_done = i + 1;
       
       if (bins.size() <= (size_t)max_bins) {
         break;
       }
     }
   }
   
   check_consistency();
   
   CharacterVector bin_names;
   NumericVector bin_woe;
   NumericVector bin_iv;
   IntegerVector bin_count;
   IntegerVector bin_count_pos;
   IntegerVector bin_count_neg;
   
   for (const auto& bin : bins) {
     std::string bn = utils::join(bin.categories, bin_separator);
     bin_names.push_back(bn);
     bin_woe.push_back(bin.woe);
     bin_iv.push_back(bin.iv);
     bin_count.push_back(bin.count);
     bin_count_pos.push_back(bin.count_pos);
     bin_count_neg.push_back(bin.count_neg);
   }
   
   return List::create(
     Named("bin") = bin_names,
     Named("woe") = bin_woe,
     Named("iv") = bin_iv,
     Named("count") = bin_count,
     Named("count_pos") = bin_count_pos,
     Named("count_neg") = bin_count_neg,
     Named("converged") = converged_flag,
     Named("iterations") = iterations_done
   );
 } catch (const std::exception& e) {
   Rcpp::stop("Erro no binning ótimo: " + std::string(e.what()));
 }
}


//' @title Optimal Binning for Categorical Variables using Monotonic Binning Algorithm (MBA)
//'
//' @description
//' This function performs optimal binning for categorical variables using a Monotonic Binning Algorithm (MBA) approach,
//' which combines Weight of Evidence (WOE) and Information Value (IV) methods with monotonicity constraints.
//'
//' @param feature A character vector of categorical feature values.
//' @param target An integer vector of binary target values (0 or 1).
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum frequency for a category to be considered as a separate bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins before merging (default: 20).
//' @param bin_separator String used to separate category names when merging bins (default: "%;%").
//' @param convergence_threshold Threshold for convergence in optimization (default: 1e-6).
//' @param max_iterations Maximum number of iterations for optimization (default: 1000).
//'
//' @return A list containing:
//' \itemize{
//'   \item bins: A character vector of bin labels
//'   \item woe: A numeric vector of Weight of Evidence values for each bin
//'   \item iv: A numeric vector of Information Value for each bin
//'   \item count: An integer vector of total counts for each bin
//'   \item count_pos: An integer vector of positive target counts for each bin
//'   \item count_neg: An integer vector of negative target counts for each bin
//'   \item converged: A logical value indicating whether the algorithm converged
//'   \item iterations: An integer indicating the number of iterations run
//' }
//'
//' @details
//' The algorithm performs the following steps:
//' \enumerate{
//'   \item Input validation and preprocessing
//'   \item Initial pre-binning based on frequency
//'   \item Enforcing minimum bin size (bin_cutoff)
//'   \item Calculating initial Weight of Evidence (WOE) and Information Value (IV)
//'   \item Enforcing monotonicity of WOE across bins
//'   \item Optimizing the number of bins through iterative merging
//' }
//'
//' The Weight of Evidence (WOE) is calculated as:
//' \deqn{WOE = \ln\left(\frac{\text{Proportion of Events}}{\text{Proportion of Non-Events}}\right)}
//'
//' The Information Value (IV) for each bin is calculated as:
//' \deqn{IV = (\text{Proportion of Events} - \text{Proportion of Non-Events}) \times WOE}
//'
//' @examples
//' \dontrun{
//' # Create sample data
//' set.seed(123)
//' target <- sample(0:1, 1000, replace = TRUE)
//' feature <- sample(LETTERS[1:5], 1000, replace = TRUE)
//'
//' # Run optimal binning
//' result <- optimal_binning_categorical_mba(feature, target)
//'
//' # View results
//' print(result)
//' }
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_mba(
   Rcpp::IntegerVector target,
   Rcpp::CharacterVector feature,
   int min_bins = 3,
   int max_bins = 5,
   double bin_cutoff = 0.05,
   int max_n_prebins = 20,
   std::string bin_separator = "%;%",
   double convergence_threshold = 1e-6,
   int max_iterations = 1000
) {
 std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
 
 OptimalBinningCategoricalMBA mba(
     feature_vec, target, min_bins, max_bins, bin_cutoff, max_n_prebins,
     bin_separator, convergence_threshold, max_iterations
 );
 
 return mba.fit();
}



// #include <Rcpp.h>
// #include <vector>
// #include <string>
// #include <unordered_map>
// #include <algorithm>
// #include <cmath>
// #include <numeric>
// #include <limits>
// #include <stdexcept>
// 
// using namespace Rcpp;
// 
// // Utility functions
// namespace utils {
// double safe_log(double x) {
//   return x > 0 ? std::log(x) : std::log(1e-10);
// }
// 
// template<typename T>
// double calculate_entropy(const std::vector<T>& counts) {
//   double total = std::accumulate(counts.begin(), counts.end(), 0.0);
//   double entropy = 0.0;
//   for (const auto& count : counts) {
//     if (count > 0) {
//       double p = count / total;
//       entropy -= p * safe_log(p);
//     }
//   }
//   return entropy;
// }
// 
// std::string join(const std::vector<std::string>& v, const std::string& delimiter) {
//   std::string result;
//   for (size_t i = 0; i < v.size(); ++i) {
//     if (i > 0) {
//       result += delimiter;
//     }
//     result += v[i];
//   }
//   return result;
// }
// }
// 
// // OptimalBinningCategoricalMBA class definition
// class OptimalBinningCategoricalMBA {
// private:
//   // Member variables
//   std::vector<std::string> feature;
//   std::vector<int> target;
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
//   std::string bin_separator;
//   double convergence_threshold;
//   int max_iterations;
//   
//   // Total counts of good and bad instances
//   int total_good;
//   int total_bad;
//   
//   // Bin structure
//   struct Bin {
//     std::vector<std::string> categories;
//     double woe;
//     double iv;
//     int count;
//     int count_pos;
//     int count_neg;
//     
//     Bin() : woe(0), iv(0), count(0), count_pos(0), count_neg(0) {}
//   };
//   std::vector<Bin> bins;
//   
//   // Private member functions
//   void validate_inputs();
//   void prebinning();
//   void enforce_bin_cutoff();
//   void calculate_initial_woe();
//   void enforce_monotonicity();
//   void optimize_bins();
//   void merge_bins(size_t index1, size_t index2);
//   void update_bin_statistics(Bin& bin);
//   bool check_monotonicity() const;
//   void check_consistency() const;
//   
// public:
//   // Constructor
//   OptimalBinningCategoricalMBA(const std::vector<std::string>& feature_,
//                                const Rcpp::IntegerVector& target_,
//                                int min_bins_ = 2,
//                                int max_bins_ = 5,
//                                double bin_cutoff_ = 0.05,
//                                int max_n_prebins_ = 20,
//                                std::string bin_separator_ = "%;%",
//                                double convergence_threshold_ = 1e-6,
//                                int max_iterations_ = 1000);
//   
//   // Function to execute binning
//   Rcpp::List fit();
// };
// 
// OptimalBinningCategoricalMBA::OptimalBinningCategoricalMBA(
//   const std::vector<std::string>& feature_,
//   const Rcpp::IntegerVector& target_,
//   int min_bins_,
//   int max_bins_,
//   double bin_cutoff_,
//   int max_n_prebins_,
//   std::string bin_separator_,
//   double convergence_threshold_,
//   int max_iterations_
// ) : feature(feature_), target(Rcpp::as<std::vector<int>>(target_)), 
// min_bins(min_bins_), max_bins(max_bins_), 
// bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_),
// bin_separator(bin_separator_),
// convergence_threshold(convergence_threshold_),
// max_iterations(max_iterations_) {
//   total_good = std::count(target.begin(), target.end(), 0);
//   total_bad = std::count(target.begin(), target.end(), 1);
// }
// 
// void OptimalBinningCategoricalMBA::validate_inputs() {
//   if (target.empty() || feature.empty()) {
//     throw std::invalid_argument("Input vectors cannot be empty.");
//   }
//   if (target.size() != feature.size()) {
//     throw std::invalid_argument("Feature and target vectors must have the same size.");
//   }
//   if (min_bins < 2) {
//     throw std::invalid_argument("min_bins must be at least 2.");
//   }
//   if (max_bins < min_bins) {
//     throw std::invalid_argument("max_bins must be greater than or equal to min_bins.");
//   }
//   if (bin_cutoff <= 0 || bin_cutoff >= 1) {
//     throw std::invalid_argument("bin_cutoff must be between 0 and 1.");
//   }
//   if (max_n_prebins < max_bins) {
//     throw std::invalid_argument("max_n_prebins must be greater than or equal to max_bins.");
//   }
//   
//   // Validate target values
//   for (int val : target) {
//     if (val != 0 && val != 1) {
//       throw std::invalid_argument("Target vector must contain only 0 and 1.");
//     }
//   }
//   
//   // Ensure max_bins is not greater than the number of unique categories
//   std::unordered_set<std::string> unique_categories(feature.begin(), feature.end());
//   int ncat = unique_categories.size();
//   if (max_bins > ncat) {
//     max_bins = ncat;
//   }
// }
// 
// void OptimalBinningCategoricalMBA::prebinning() {
//   std::unordered_map<std::string, int> category_counts;
//   std::unordered_map<std::string, int> category_pos_counts;
//   
//   // Count occurrences of each category and positives
//   for (size_t i = 0; i < feature.size(); ++i) {
//     const auto& cat = feature[i];
//     category_counts[cat]++;
//     if (target[i] == 1) {
//       category_pos_counts[cat]++;
//     }
//   }
//   
//   // Sort categories by frequency
//   std::vector<std::pair<std::string, int>> sorted_categories;
//   sorted_categories.reserve(category_counts.size());
//   for (const auto& pair : category_counts) {
//     sorted_categories.emplace_back(pair);
//   }
//   std::sort(sorted_categories.begin(), sorted_categories.end(),
//             [](const auto& a, const auto& b) { return a.second > b.second; });
//   
//   // Initialize bins
//   bins.clear();
//   for (const auto& pair : sorted_categories) {
//     Bin bin;
//     bin.categories.push_back(pair.first);
//     bin.count = pair.second;
//     bin.count_pos = category_pos_counts[pair.first];
//     bin.count_neg = bin.count - bin.count_pos;
//     bins.push_back(bin);
//   }
//   
//   // Merge less frequent categories if exceeding max_n_prebins
//   while (bins.size() > static_cast<size_t>(max_n_prebins)) {
//     auto min_it = std::min_element(bins.begin(), bins.end(),
//                                    [](const Bin& a, const Bin& b) {
//                                      return a.count < b.count;
//                                    });
//     
//     if (min_it != bins.begin()) {
//       auto prev_it = std::prev(min_it);
//       merge_bins(prev_it - bins.begin(), min_it - bins.begin());
//     } else {
//       merge_bins(0, 1);
//     }
//   }
// }
// 
// void OptimalBinningCategoricalMBA::enforce_bin_cutoff() {
//   int min_count = static_cast<int>(std::ceil(bin_cutoff * feature.size()));
//   int min_count_pos = static_cast<int>(std::ceil(bin_cutoff * total_bad));
//   
//   bool merged;
//   do {
//     merged = false;
//     for (size_t i = 0; i < bins.size(); ++i) {
//       if (bins[i].count < min_count || bins[i].count_pos < min_count_pos) {
//         size_t merge_index = (i > 0) ? i - 1 : i + 1;
//         if (merge_index < bins.size()) {
//           merge_bins(std::min(i, merge_index), std::max(i, merge_index));
//           merged = true;
//           break;
//         }
//       }
//     }
//   } while (merged && bins.size() > static_cast<size_t>(min_bins));
// }
// 
// void OptimalBinningCategoricalMBA::calculate_initial_woe() {
//   for (auto& bin : bins) {
//     update_bin_statistics(bin);
//   }
// }
// 
// void OptimalBinningCategoricalMBA::enforce_monotonicity() {
//   if (bins.empty()) {
//     throw std::runtime_error("No bins available to enforce monotonicity.");
//   }
//   
//   // Sort bins by WoE
//   std::sort(bins.begin(), bins.end(),
//             [](const Bin& a, const Bin& b) { return a.woe < b.woe; });
//   
//   // Determine monotonicity direction
//   bool increasing = true;
//   for (size_t i = 1; i < bins.size(); ++i) {
//     if (bins[i].woe < bins[i - 1].woe) {
//       increasing = false;
//       break;
//     }
//   }
//   
//   // Merge bins to enforce monotonicity
//   bool merged;
//   do {
//     merged = false;
//     for (size_t i = 0; i < bins.size() - 1; ++i) {
//       if ((increasing && bins[i].woe > bins[i + 1].woe) ||
//           (!increasing && bins[i].woe < bins[i + 1].woe)) {
//         merge_bins(i, i + 1);
//         merged = true;
//         break;
//       }
//     }
//   } while (merged && bins.size() > static_cast<size_t>(min_bins));
//   
//   // Ensure minimum number of bins
//   while (bins.size() < static_cast<size_t>(min_bins) && bins.size() > 1) {
//     size_t min_iv_index = std::min_element(bins.begin(), bins.end(),
//                                            [](const Bin& a, const Bin& b) {
//                                              return std::abs(a.iv) < std::abs(b.iv);
//                                            }) - bins.begin();
//     size_t merge_index = (min_iv_index == 0) ? 0 : min_iv_index - 1;
//     merge_bins(merge_index, merge_index + 1);
//   }
// }
// 
// void OptimalBinningCategoricalMBA::optimize_bins() {
//   int iterations = 0;
//   double prev_total_iv = 0.0;
//   double total_iv = 0.0;
//   bool converged = false;
//   
//   while (bins.size() > static_cast<size_t>(max_bins) && iterations < max_iterations) {
//     double min_combined_iv = std::numeric_limits<double>::max();
//     size_t min_iv_index = 0;
//     
//     for (size_t i = 0; i < bins.size() - 1; ++i) {
//       double combined_iv = std::abs(bins[i].iv) + std::abs(bins[i + 1].iv);
//       if (combined_iv < min_combined_iv) {
//         min_combined_iv = combined_iv;
//         min_iv_index = i;
//       }
//     }
//     
//     merge_bins(min_iv_index, min_iv_index + 1);
//     
//     // Calculate total IV
//     total_iv = 0.0;
//     for (const auto& bin : bins) {
//       total_iv += std::abs(bin.iv);
//     }
//     
//     // Check for convergence
//     if (std::abs(total_iv - prev_total_iv) < convergence_threshold) {
//       converged = true;
//       break;
//     }
//     
//     prev_total_iv = total_iv;
//     iterations++;
//   }
//   
//   if (!converged && iterations == max_iterations) {
//     Rcpp::warning("Maximum iterations reached without convergence.");
//   }
// }
// 
// void OptimalBinningCategoricalMBA::merge_bins(size_t index1, size_t index2) {
//   if (index1 >= bins.size() || index2 >= bins.size() || index1 == index2) {
//     throw std::invalid_argument("Invalid bin indices for merging.");
//   }
//   
//   Bin& bin1 = bins[index1];
//   Bin& bin2 = bins[index2];
//   
//   bin1.categories.insert(bin1.categories.end(), 
//                          std::make_move_iterator(bin2.categories.begin()),
//                          std::make_move_iterator(bin2.categories.end()));
//   bin1.count += bin2.count;
//   bin1.count_pos += bin2.count_pos;
//   bin1.count_neg += bin2.count_neg;
//   
//   update_bin_statistics(bin1);
//   
//   bins.erase(bins.begin() + index2);
//   
//   if (bins.empty()) {
//     throw std::runtime_error("All bins have been merged. Unable to continue.");
//   }
// }
// 
// void OptimalBinningCategoricalMBA::update_bin_statistics(Bin& bin) {
//   // Calculate proportions
//   double prop_event = static_cast<double>(bin.count_pos) / total_bad;
//   double prop_non_event = static_cast<double>(bin.count_neg) / total_good;
//   
//   // Avoid division by zero
//   prop_event = std::max(prop_event, 1e-10);
//   prop_non_event = std::max(prop_non_event, 1e-10);
//   
//   // Calculate WOE
//   bin.woe = utils::safe_log(prop_event / prop_non_event);
//   
//   // Calculate IV
//   bin.iv = (prop_event - prop_non_event) * bin.woe;
// }
// 
// bool OptimalBinningCategoricalMBA::check_monotonicity() const {
//   if (bins.empty()) return true;
//   
//   bool increasing = true;
//   bool decreasing = true;
//   for (size_t i = 1; i < bins.size(); ++i) {
//     if (bins[i].woe < bins[i - 1].woe) {
//       increasing = false;
//     }
//     if (bins[i].woe > bins[i - 1].woe) {
//       decreasing = false;
//     }
//   }
//   return increasing || decreasing;
// }
// 
// void OptimalBinningCategoricalMBA::check_consistency() const {
//   int total_count = 0;
//   int total_count_pos = 0;
//   int total_count_neg = 0;
//   
//   for (const auto& bin : bins) {
//     total_count += bin.count;
//     total_count_pos += bin.count_pos;
//     total_count_neg += bin.count_neg;
//   }
//   
//   if (static_cast<size_t>(total_count) != feature.size()) {
//     throw std::runtime_error("Inconsistent total count after binning.");
//   }
//   
//   if (total_count_pos != total_bad || total_count_neg != total_good) {
//     throw std::runtime_error("Inconsistent positive/negative counts after binning.");
//   }
// }
// 
// Rcpp::List OptimalBinningCategoricalMBA::fit() {
//   try {
//     validate_inputs();
//     prebinning();
//     enforce_bin_cutoff();
//     calculate_initial_woe();
//     enforce_monotonicity();
//     
//     bool converged = false;
//     int iterations_run = 0;
//     
//     // If number of unique categories is less than or equal to max_bins,
//     // skip optimization and calculate statistics
//     if (bins.size() <= static_cast<size_t>(max_bins)) {
//       converged = true;
//     } else {
//       double prev_total_iv = 0.0;
//       for (int i = 0; i < max_iterations; ++i) {
//         optimize_bins();
//         
//         double total_iv = 0.0;
//         for (const auto& bin : bins) {
//           total_iv += std::abs(bin.iv);
//         }
//         
//         if (std::abs(total_iv - prev_total_iv) < convergence_threshold) {
//           converged = true;
//           iterations_run = i + 1;
//           break;
//         }
//         
//         prev_total_iv = total_iv;
//         iterations_run = i + 1;
//         
//         if (bins.size() <= static_cast<size_t>(max_bins)) {
//           break;
//         }
//       }
//     }
//     
//     check_consistency();
//     
//     // Prepare output
//     Rcpp::CharacterVector bin_names;
//     Rcpp::NumericVector bin_woe;
//     Rcpp::NumericVector bin_iv;
//     Rcpp::IntegerVector bin_count;
//     Rcpp::IntegerVector bin_count_pos;
//     Rcpp::IntegerVector bin_count_neg;
//     
//     for (const auto& bin : bins) {
//       std::string bin_name = utils::join(bin.categories, bin_separator);
//       bin_names.push_back(bin_name);
//       bin_woe.push_back(bin.woe);
//       bin_iv.push_back(bin.iv);
//       bin_count.push_back(bin.count);
//       bin_count_pos.push_back(bin.count_pos);
//       bin_count_neg.push_back(bin.count_neg);
//     }
//     
//     return Rcpp::List::create(
//       Rcpp::Named("bin") = bin_names,
//       Rcpp::Named("woe") = bin_woe,
//       Rcpp::Named("iv") = bin_iv,
//       Rcpp::Named("count") = bin_count,
//       Rcpp::Named("count_pos") = bin_count_pos,
//       Rcpp::Named("count_neg") = bin_count_neg,
//       Rcpp::Named("converged") = converged,
//       Rcpp::Named("iterations") = iterations_run
//     );
//   } catch (const std::exception& e) {
//     Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
//   }
// }
// 
// //' @title Optimal Binning for Categorical Variables using Monotonic Binning Algorithm (MBA)
// //'
// //' @description
// //' This function performs optimal binning for categorical variables using a Monotonic Binning Algorithm (MBA) approach,
// //' which combines Weight of Evidence (WOE) and Information Value (IV) methods with monotonicity constraints.
// //'
// //' @param feature A character vector of categorical feature values.
// //' @param target An integer vector of binary target values (0 or 1).
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff Minimum frequency for a category to be considered as a separate bin (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins before merging (default: 20).
// //' @param bin_separator String used to separate category names when merging bins (default: "%;%").
// //' @param convergence_threshold Threshold for convergence in optimization (default: 1e-6).
// //' @param max_iterations Maximum number of iterations for optimization (default: 1000).
// //'
// //' @return A list containing:
// //' \itemize{
// //'   \item bins: A character vector of bin labels
// //'   \item woe: A numeric vector of Weight of Evidence values for each bin
// //'   \item iv: A numeric vector of Information Value for each bin
// //'   \item count: An integer vector of total counts for each bin
// //'   \item count_pos: An integer vector of positive target counts for each bin
// //'   \item count_neg: An integer vector of negative target counts for each bin
// //'   \item converged: A logical value indicating whether the algorithm converged
// //'   \item iterations: An integer indicating the number of iterations run
// //' }
// //'
// //' @details
// //' The algorithm performs the following steps:
// //' \enumerate{
// //'   \item Input validation and preprocessing
// //'   \item Initial pre-binning based on frequency
// //'   \item Enforcing minimum bin size (bin_cutoff)
// //'   \item Calculating initial Weight of Evidence (WOE) and Information Value (IV)
// //'   \item Enforcing monotonicity of WOE across bins
// //'   \item Optimizing the number of bins through iterative merging
// //' }
// //'
// //' The Weight of Evidence (WOE) is calculated as:
// //' \deqn{WOE = \ln\left(\frac{\text{Proportion of Events}}{\text{Proportion of Non-Events}}\right)}
// //'
// //' The Information Value (IV) for each bin is calculated as:
// //' \deqn{IV = (\text{Proportion of Events} - \text{Proportion of Non-Events}) \times WOE}
// //'
// //' @examples
// //' \dontrun{
// //' # Create sample data
// //' set.seed(123)
// //' target <- sample(0:1, 1000, replace = TRUE)
// //' feature <- sample(LETTERS[1:5], 1000, replace = TRUE)
// //'
// //' # Run optimal binning
// //' result <- optimal_binning_categorical_mba(feature, target)
// //'
// //' # View results
// //' print(result)
// //' }
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_categorical_mba(
//    Rcpp::IntegerVector target,
//    Rcpp::CharacterVector feature,
//    int min_bins = 3,
//    int max_bins = 5,
//    double bin_cutoff = 0.05,
//    int max_n_prebins = 20,
//    std::string bin_separator = "%;%",
//    double convergence_threshold = 1e-6,
//    int max_iterations = 1000
// ) {
//  std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
//  
//  // Instantiate the binning class
//  OptimalBinningCategoricalMBA mba(
//      feature_vec, target, min_bins, max_bins, bin_cutoff, max_n_prebins,
//      bin_separator, convergence_threshold, max_iterations
//  );
//  
//  // Execute binning and return the result
//  return mba.fit();
// }