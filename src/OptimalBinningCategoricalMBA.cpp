#include <Rcpp.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <numeric>

// Utility functions
namespace utils {
double safe_log(double x) {
  return x > 0 ? std::log(x) : std::log(1e-10);
}

template<typename T>
double calculate_entropy(const std::vector<T>& counts) {
  double total = std::accumulate(counts.begin(), counts.end(), 0.0);
  double entropy = 0.0;
  for (const auto& count : counts) {
    if (count > 0) {
      double p = count / total;
      entropy -= p * safe_log(p);
    }
  }
  return entropy;
}

std::string join(const std::vector<std::string>& v, const std::string& delimiter) {
  std::string result;
  for (size_t i = 0; i < v.size(); ++i) {
    if (i > 0) {
      result += delimiter;
    }
    result += v[i];
  }
  return result;
}
}

// OptimalBinningCategoricalMBA class definition
class OptimalBinningCategoricalMBA {
private:
  // Member variables
  std::vector<std::string> feature;
  const std::vector<int>& target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;

  // Total counts of good and bad instances
  int total_good;
  int total_bad;

  // Bin structure
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

  // Output vector for WoE values
  std::vector<double> woefeature;

  // Private member functions
  void validate_inputs();
  void prebinning();
  void enforce_bin_cutoff();
  void calculate_initial_woe();
  void enforce_monotonicity();
  void optimize_bins();
  void assign_woe();
  void merge_bins(size_t index1, size_t index2);
  void update_bin_statistics(Bin& bin);

public:
  // Constructor
  OptimalBinningCategoricalMBA(const std::vector<std::string>& feature_,
                               const std::vector<int>& target_,
                               int min_bins_ = 2,
                               int max_bins_ = 5,
                               double bin_cutoff_ = 0.05,
                               int max_n_prebins_ = 20);

  // Function to execute binning
  Rcpp::List fit();
};

OptimalBinningCategoricalMBA::OptimalBinningCategoricalMBA(const std::vector<std::string>& feature_,
                                                           const std::vector<int>& target_,
                                                           int min_bins_,
                                                           int max_bins_,
                                                           double bin_cutoff_,
                                                           int max_n_prebins_)
  : feature(feature_), target(target_), min_bins(min_bins_),
    max_bins(max_bins_), bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_) {
  total_good = std::count(target.begin(), target.end(), 0);
  total_bad = std::count(target.begin(), target.end(), 1);
}

void OptimalBinningCategoricalMBA::validate_inputs() {
  if (target.empty() || feature.empty()) {
    throw std::invalid_argument("Input vectors cannot be empty.");
  }
  if (target.size() != feature.size()) {
    throw std::invalid_argument("Feature and target vectors must have the same size.");
  }
  if (min_bins < 2) {
    throw std::invalid_argument("min_bins must be at least 2.");
  }
  if (max_bins < min_bins) {
    throw std::invalid_argument("max_bins must be greater than or equal to min_bins.");
  }
  if (bin_cutoff <= 0 || bin_cutoff >= 1) {
    throw std::invalid_argument("bin_cutoff must be between 0 and 1.");
  }
  if (max_n_prebins < max_bins) {
    throw std::invalid_argument("max_n_prebins must be greater than or equal to max_bins.");
  }
}

void OptimalBinningCategoricalMBA::prebinning() {
  std::unordered_map<std::string, int> category_counts;
  std::unordered_map<std::string, int> category_pos_counts;

  // Count occurrences of each category and positives
  for (size_t i = 0; i < feature.size(); ++i) {
    const auto& cat = feature[i];
    category_counts[cat]++;
    if (target[i] == 1) {
      category_pos_counts[cat]++;
    }
  }

  // Sort categories by frequency
  std::vector<std::pair<std::string, int>> sorted_categories(category_counts.begin(), category_counts.end());
  std::sort(sorted_categories.begin(), sorted_categories.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

  // Initialize bins
  bins.clear();
  for (const auto& pair : sorted_categories) {
    Bin bin;
    bin.categories.push_back(pair.first);
    bin.count = pair.second;
    bin.count_pos = category_pos_counts[pair.first];
    bin.count_neg = bin.count - bin.count_pos;
    bins.push_back(bin);
  }

  // Merge less frequent categories if exceeding max_n_prebins
  while (bins.size() > static_cast<size_t>(max_n_prebins)) {
    auto min_it = std::min_element(bins.begin(), bins.end(),
                                   [](const Bin& a, const Bin& b) {
                                     return a.count < b.count;
                                   });

    if (min_it != bins.begin()) {
      auto prev_it = std::prev(min_it);
      merge_bins(prev_it - bins.begin(), min_it - bins.begin());
    } else {
      merge_bins(0, 1);
    }
  }

  // Update feature with grouped categories
  std::unordered_map<std::string, std::string> category_mapping;
  for (const auto& bin : bins) {
    std::string bin_name = utils::join(bin.categories, "+");
    for (const auto& cat : bin.categories) {
      category_mapping[cat] = bin_name;
    }
  }

  for (auto& cat : feature) {
    cat = category_mapping[cat];
  }
}

void OptimalBinningCategoricalMBA::enforce_bin_cutoff() {
  int min_count = static_cast<int>(std::ceil(bin_cutoff * feature.size()));
  int min_count_pos = static_cast<int>(std::ceil(bin_cutoff * total_bad));

  bool merged;
  do {
    merged = false;
    for (size_t i = 0; i < bins.size(); ++i) {
      if (bins[i].count < min_count || bins[i].count_pos < min_count_pos) {
        size_t merge_index = (i > 0) ? i - 1 : i + 1;
        if (merge_index < bins.size()) {
          merge_bins(std::min(i, merge_index), std::max(i, merge_index));
          merged = true;
          break;
        }
      }
    }
  } while (merged && bins.size() > static_cast<size_t>(min_bins));
}

void OptimalBinningCategoricalMBA::calculate_initial_woe() {
  for (auto& bin : bins) {
    update_bin_statistics(bin);
  }
}

void OptimalBinningCategoricalMBA::enforce_monotonicity() {
  if (bins.empty()) {
    throw std::runtime_error("No bins available to enforce monotonicity.");
  }

  // Sort bins by WoE
  std::sort(bins.begin(), bins.end(),
            [](const Bin& a, const Bin& b) { return a.woe < b.woe; });

  // Determine monotonicity direction
  bool increasing = true;
  for (size_t i = 1; i < bins.size(); ++i) {
    if (bins[i].woe < bins[i - 1].woe) {
      increasing = false;
      break;
    }
  }

  // Merge bins to enforce monotonicity
  bool merged;
  do {
    merged = false;
    for (size_t i = 0; i < bins.size() - 1; ++i) {
      if ((increasing && bins[i].woe > bins[i + 1].woe) ||
          (!increasing && bins[i].woe < bins[i + 1].woe)) {
        merge_bins(i, i + 1);
        merged = true;
        break;
      }
    }
  } while (merged && bins.size() > static_cast<size_t>(min_bins));

  // Ensure minimum number of bins
  while (bins.size() < static_cast<size_t>(min_bins) && bins.size() > 1) {
    size_t min_iv_index = std::min_element(bins.begin(), bins.end(),
                                           [](const Bin& a, const Bin& b) {
                                             return std::abs(a.iv) < std::abs(b.iv);
                                           }) - bins.begin();
    size_t merge_index = (min_iv_index == 0) ? 0 : min_iv_index - 1;
    merge_bins(merge_index, merge_index + 1);
  }
}

void OptimalBinningCategoricalMBA::optimize_bins() {
  while (bins.size() > static_cast<size_t>(max_bins)) {
    double min_combined_iv = std::numeric_limits<double>::max();
    size_t min_iv_index = 0;

    for (size_t i = 0; i < bins.size() - 1; ++i) {
      double combined_iv = std::abs(bins[i].iv) + std::abs(bins[i + 1].iv);
      if (combined_iv < min_combined_iv) {
        min_combined_iv = combined_iv;
        min_iv_index = i;
      }
    }

    merge_bins(min_iv_index, min_iv_index + 1);
  }
}

void OptimalBinningCategoricalMBA::assign_woe() {
  std::unordered_map<std::string, double> category_woe_map;
  for (const auto& bin : bins) {
    for (const auto& cat : bin.categories) {
      category_woe_map[cat] = bin.woe;
    }
  }

  woefeature.resize(feature.size());

  for (size_t i = 0; i < feature.size(); ++i) {
    auto it = category_woe_map.find(feature[i]);
    woefeature[i] = (it != category_woe_map.end()) ? it->second : 0.0;
  }
}

void OptimalBinningCategoricalMBA::merge_bins(size_t index1, size_t index2) {
  if (index1 >= bins.size() || index2 >= bins.size() || index1 == index2) {
    throw std::runtime_error("Invalid bin indices for merging.");
  }

  Bin& bin1 = bins[index1];
  Bin& bin2 = bins[index2];

  bin1.categories.insert(bin1.categories.end(), bin2.categories.begin(), bin2.categories.end());
  bin1.count += bin2.count;
  bin1.count_pos += bin2.count_pos;
  bin1.count_neg += bin2.count_neg;

  update_bin_statistics(bin1);

  bins.erase(bins.begin() + index2);
}

void OptimalBinningCategoricalMBA::update_bin_statistics(Bin& bin) {
  // Calculate proportions
  double prop_event = static_cast<double>(bin.count_pos) / total_bad;
  double prop_non_event = static_cast<double>(bin.count_neg) / total_good;

  // Avoid division by zero
  prop_event = std::max(prop_event, 1e-10);
  prop_non_event = std::max(prop_non_event, 1e-10);

  // Calculate WOE (corrected)
  bin.woe = utils::safe_log(prop_event / prop_non_event);

  // Calculate IV
  bin.iv = (prop_event - prop_non_event) * bin.woe;
}

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
//   bin.woe = utils::safe_log(prop_non_event / prop_event);
//
//   // Calculate IV
//   bin.iv = (prop_non_event - prop_event) * bin.woe;
// }

Rcpp::List OptimalBinningCategoricalMBA::fit() {
  validate_inputs();
  prebinning();
  enforce_bin_cutoff();
  calculate_initial_woe();
  enforce_monotonicity();
  optimize_bins();
  assign_woe();

  // Prepare output
  Rcpp::NumericVector woefeature_rcpp(woefeature.begin(), woefeature.end());

  Rcpp::CharacterVector bin_names;
  Rcpp::NumericVector woe_values;
  Rcpp::NumericVector iv_values;
  Rcpp::IntegerVector count_values;
  Rcpp::IntegerVector count_pos_values;
  Rcpp::IntegerVector count_neg_values;

  double total_iv = 0.0;

  for (const auto& bin : bins) {
    std::string bin_name = utils::join(bin.categories, "+");
    bin_names.push_back(bin_name);
    woe_values.push_back(bin.woe);
    iv_values.push_back(bin.iv);
    count_values.push_back(bin.count);
    count_pos_values.push_back(bin.count_pos);
    count_neg_values.push_back(bin.count_neg);
    total_iv += bin.iv;
  }

  Rcpp::DataFrame woebin = Rcpp::DataFrame::create(
    Rcpp::Named("bin") = bin_names,
    Rcpp::Named("woe") = woe_values,
    Rcpp::Named("iv") = iv_values,
    Rcpp::Named("count") = count_values,
    Rcpp::Named("count_pos") = count_pos_values,
    Rcpp::Named("count_neg") = count_neg_values
  );

  return Rcpp::List::create(
    Rcpp::Named("woefeature") = woefeature_rcpp,
    Rcpp::Named("woebin") = woebin,
    Rcpp::Named("total_iv") = total_iv
  );
}

// R wrapper function
// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_mba(Rcpp::IntegerVector target,
                                           Rcpp::CharacterVector feature,
                                           int min_bins = 3,
                                           int max_bins = 5,
                                           double bin_cutoff = 0.05,
                                           int max_n_prebins = 20) {
  // Input validation
  if (target.size() != feature.size()) {
    throw std::invalid_argument("Feature and target vectors must have the same size.");
  }
  if (target.size() == 0) {
    throw std::invalid_argument("Input vectors cannot be empty.");
  }

  // Convert Rcpp vectors to std::vectors
  std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
  std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);

  // Instantiate the binning class
  OptimalBinningCategoricalMBA mba(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins);

  // Execute binning and return the result
  return mba.fit();
}

// #include <Rcpp.h>
// #include <vector>
// #include <string>
// #include <unordered_map>
// #include <algorithm>
// #include <stdexcept>
// #include <cmath>
// #include <numeric>
//
// // Funções utilitárias
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
// // Definição da classe OptimalBinningCategoricalMBA
// class OptimalBinningCategoricalMBA {
// private:
//   // Variáveis membro
//   std::vector<std::string> feature;
//   const std::vector<int>& target;
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
//
//   // Armazenamento de total de bons e ruins
//   int total_good;
//   int total_bad;
//
//   // Estrutura para armazenar informações do bin
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
//   // Vetor de saída para valores WoE
//   std::vector<double> woefeature;
//
//   // Funções membro privadas
//   void validate_inputs();
//   void prebinning();
//   void calculate_initial_woe();
//   void enforce_monotonicity();
//   void optimize_bins();
//   void assign_woe();
//   void merge_bins(size_t index1, size_t index2);
//   void update_bin_statistics(Bin& bin);
//
// public:
//   // Construtor
//   OptimalBinningCategoricalMBA(const std::vector<std::string>& feature_,
//                                const std::vector<int>& target_,
//                                int min_bins_ = 2,
//                                int max_bins_ = 5,
//                                double bin_cutoff_ = 0.05,
//                                int max_n_prebins_ = 20);
//
//   // Função para executar o binning
//   Rcpp::List fit();
// };
//
// OptimalBinningCategoricalMBA::OptimalBinningCategoricalMBA(const std::vector<std::string>& feature_,
//                                                            const std::vector<int>& target_,
//                                                            int min_bins_,
//                                                            int max_bins_,
//                                                            double bin_cutoff_,
//                                                            int max_n_prebins_)
//   : feature(feature_), target(target_), min_bins(min_bins_),
//     max_bins(max_bins_), bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_) {
//   // Cálculo único de total_good e total_bad
//   total_good = std::count(target.begin(), target.end(), 0);
//   total_bad = std::count(target.begin(), target.end(), 1);
// }
//
// void OptimalBinningCategoricalMBA::validate_inputs() {
//   if (target.empty() || feature.empty()) {
//     throw std::invalid_argument("Os vetores de entrada não podem ser vazios.");
//   }
//   if (target.size() != feature.size()) {
//     throw std::invalid_argument("Os vetores feature e target devem ter o mesmo tamanho.");
//   }
//   if (min_bins < 2) {
//     throw std::invalid_argument("min_bins deve ser pelo menos 2.");
//   }
//   if (max_bins < min_bins) {
//     throw std::invalid_argument("max_bins deve ser maior ou igual a min_bins.");
//   }
//   if (bin_cutoff <= 0 || bin_cutoff >= 1) {
//     throw std::invalid_argument("bin_cutoff deve estar entre 0 e 1.");
//   }
//   if (max_n_prebins < max_bins) {
//     throw std::invalid_argument("max_n_prebins deve ser maior ou igual a max_bins.");
//   }
// }
//
// void OptimalBinningCategoricalMBA::prebinning() {
//   std::unordered_map<std::string, int> category_counts;
//
//   // Contagem das ocorrências de cada categoria
//   for (const auto& cat : feature) {
//     category_counts[cat]++;
//   }
//
//   int total_count = feature.size();
//
//   // Ordenar categorias por frequência
//   std::vector<std::pair<std::string, int>> sorted_categories(category_counts.begin(), category_counts.end());
//   std::sort(sorted_categories.begin(), sorted_categories.end(),
//             [](const auto& a, const auto& b) { return a.second > b.second; });
//
//   // Lista de bins iniciais (cada categoria é um bin)
//   bins.clear();
//   for (const auto& pair : sorted_categories) {
//     Bin bin;
//     bin.categories.push_back(pair.first);
//     bins.push_back(bin);
//   }
//
//   // Mesclar categorias menos frequentes se exceder max_n_prebins
//   while (bins.size() > static_cast<size_t>(max_n_prebins)) {
//     // Encontrar o bin com menor frequência
//     auto min_it = std::min_element(bins.begin(), bins.end(),
//                                    [](const Bin& a, const Bin& b) {
//                                      return a.count < b.count;
//                                    });
//
//     // Mesclar com o bin adjacente
//     if (min_it != bins.begin()) {
//       auto prev_it = std::prev(min_it);
//       prev_it->categories.insert(prev_it->categories.end(), min_it->categories.begin(), min_it->categories.end());
//       bins.erase(min_it);
//     } else {
//       auto next_it = std::next(min_it);
//       next_it->categories.insert(next_it->categories.end(), min_it->categories.begin(), min_it->categories.end());
//       bins.erase(min_it);
//     }
//   }
//
//   // Atualizar feature com as categorias agrupadas
//   std::unordered_map<std::string, std::string> category_mapping;
//   for (const auto& bin : bins) {
//     std::string bin_name = utils::join(bin.categories, "+");
//     for (const auto& cat : bin.categories) {
//       category_mapping[cat] = bin_name;
//     }
//   }
//
//   for (auto& cat : feature) {
//     cat = category_mapping[cat];
//   }
// }
//
// void OptimalBinningCategoricalMBA::calculate_initial_woe() {
//   std::unordered_map<std::string, Bin> bin_map;
//
//   // Contagem das ocorrências e cálculo das estatísticas iniciais
//   for (size_t i = 0; i < feature.size(); ++i) {
//     const std::string& cat = feature[i];
//     int t = target[i];
//
//     if (bin_map.find(cat) == bin_map.end()) {
//       bin_map[cat] = Bin();
//       bin_map[cat].categories = {cat};
//     }
//
//     Bin& bin = bin_map[cat];
//     bin.count++;
//     if (t == 1) {
//       bin.count_pos++;
//     } else {
//       bin.count_neg++;
//     }
//   }
//
//   // Limpar bins anteriores e preencher com novos bins
//   bins.clear();
//
//   // Calcular WoE e IV para cada bin
//   for (auto& pair : bin_map) {
//     Bin& bin = pair.second;
//     update_bin_statistics(bin);
//     bins.push_back(bin);
//   }
// }
//
// void OptimalBinningCategoricalMBA::enforce_monotonicity() {
//   if (bins.empty()) {
//     throw std::runtime_error("Não há bins disponíveis para impor monotonicidade.");
//   }
//
//   // Ordenar bins por WoE
//   std::sort(bins.begin(), bins.end(),
//             [](const Bin& a, const Bin& b) { return a.woe < b.woe; });
//
//   // Determinar direção da monotonicidade
//   bool increasing = true;
//   for (size_t i = 1; i < bins.size(); ++i) {
//     if (bins[i].woe < bins[i - 1].woe) {
//       increasing = false;
//       break;
//     }
//   }
//
//   // Mesclar bins para impor monotonicidade
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
//   // Garantir número mínimo de bins
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
//   while (bins.size() > static_cast<size_t>(max_bins)) {
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
//   }
// }
//
// void OptimalBinningCategoricalMBA::assign_woe() {
//   std::unordered_map<std::string, double> category_woe_map;
//   for (const auto& bin : bins) {
//     for (const auto& cat : bin.categories) {
//       category_woe_map[cat] = bin.woe;
//     }
//   }
//
//   woefeature.resize(feature.size());
//
//   for (size_t i = 0; i < feature.size(); ++i) {
//     auto it = category_woe_map.find(feature[i]);
//     woefeature[i] = (it != category_woe_map.end()) ? it->second : 0.0;
//   }
// }
//
// void OptimalBinningCategoricalMBA::merge_bins(size_t index1, size_t index2) {
//   if (index1 >= bins.size() || index2 >= bins.size() || index1 == index2) {
//     throw std::runtime_error("Índices de bins inválidos para mesclagem.");
//   }
//
//   Bin& bin1 = bins[index1];
//   Bin& bin2 = bins[index2];
//
//   bin1.categories.insert(bin1.categories.end(), bin2.categories.begin(), bin2.categories.end());
//   bin1.count += bin2.count;
//   bin1.count_pos += bin2.count_pos;
//   bin1.count_neg += bin2.count_neg;
//
//   update_bin_statistics(bin1);
//
//   bins.erase(bins.begin() + index2);
// }
//
// void OptimalBinningCategoricalMBA::update_bin_statistics(Bin& bin) {
//   double dist_good = static_cast<double>(bin.count_neg) / total_good;
//   double dist_bad = static_cast<double>(bin.count_pos) / total_bad;
//
//   dist_good = std::max(dist_good, 1e-5);
//   dist_bad = std::max(dist_bad, 1e-5);
//
//   bin.woe = utils::safe_log(dist_good / dist_bad);
//   bin.iv = (dist_good - dist_bad) * bin.woe;
// }
//
// Rcpp::List OptimalBinningCategoricalMBA::fit() {
//   validate_inputs();
//   prebinning();
//   calculate_initial_woe();
//   enforce_monotonicity();
//   optimize_bins();
//   assign_woe();
//
//   // Preparar saída
//   Rcpp::NumericVector woefeature_rcpp(woefeature.begin(), woefeature.end());
//
//   Rcpp::CharacterVector bin_names;
//   Rcpp::NumericVector woe_values;
//   Rcpp::NumericVector iv_values;
//   Rcpp::IntegerVector count_values;
//   Rcpp::IntegerVector count_pos_values;
//   Rcpp::IntegerVector count_neg_values;
//
//   double total_iv = 0.0;
//
//   for (const auto& bin : bins) {
//     std::string bin_name = utils::join(bin.categories, "+");
//     bin_names.push_back(bin_name);
//     woe_values.push_back(bin.woe);
//     iv_values.push_back(bin.iv);
//     count_values.push_back(bin.count);
//     count_pos_values.push_back(bin.count_pos);
//     count_neg_values.push_back(bin.count_neg);
//     total_iv += bin.iv;
//   }
//
//   Rcpp::DataFrame woebin = Rcpp::DataFrame::create(
//     Rcpp::Named("bin") = bin_names,
//     Rcpp::Named("woe") = woe_values,
//     Rcpp::Named("iv") = iv_values,
//     Rcpp::Named("count") = count_values,
//     Rcpp::Named("count_pos") = count_pos_values,
//     Rcpp::Named("count_neg") = count_neg_values
//   );
//
//   return Rcpp::List::create(
//     Rcpp::Named("woefeature") = woefeature_rcpp,
//     Rcpp::Named("woebin") = woebin,
//     Rcpp::Named("total_iv") = total_iv
//   );
// }
//
// // Função R wrapper
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_categorical_mba(Rcpp::CharacterVector feature,
//                                            Rcpp::IntegerVector target,
//                                            int min_bins = 3,
//                                            int max_bins = 5,
//                                            double bin_cutoff = 0.05,
//                                            int max_n_prebins = 20) {
//   // Validação de entrada
//   if (target.size() != feature.size()) {
//     throw std::invalid_argument("Os vetores feature e target devem ter o mesmo tamanho.");
//   }
//   if (target.size() == 0) {
//     throw std::invalid_argument("Os vetores de entrada não podem ser vazios.");
//   }
//
//   // Converter Rcpp vectors para std::vectors
//   std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
//   std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
//
//   // Instanciar a classe de binning
//   OptimalBinningCategoricalMBA mba(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins);
//
//   // Executar o binning e retornar o resultado
//   return mba.fit();
// }
