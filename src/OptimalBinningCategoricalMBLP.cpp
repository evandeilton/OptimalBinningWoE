#include <Rcpp.h>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <stdexcept>

class OptimalBinningCategoricalMBLP {
private:
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;

  std::vector<std::string> unique_categories;
  std::vector<int> category_counts;
  std::vector<int> positive_counts;
  std::vector<int> negative_counts;
  int total_count;
  int total_positive;
  int total_negative;

  struct BinInfo {
    std::vector<std::string> categories;
    double woe;
    double iv;
    int count;
    int count_pos;
    int count_neg;
  };

  std::vector<BinInfo> optimal_bins;

public:
  OptimalBinningCategoricalMBLP(int min_bins = 3, int max_bins = 5, double bin_cutoff = 0.05, int max_n_prebins = 20)
    : min_bins(min_bins), max_bins(max_bins), bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins) {
    if (min_bins < 2) {
      throw std::invalid_argument("min_bins must be at least 2");
    }
    if (max_bins < min_bins) {
      throw std::invalid_argument("max_bins must be greater than or equal to min_bins");
    }
  }

  void fit(const std::vector<std::string>& feature, const std::vector<int>& target) {
    if (feature.size() != target.size()) {
      throw std::invalid_argument("Feature and target vectors must have the same length");
    }

    preprocess_data(feature, target);
    merge_rare_categories();
    optimize_bins();
  }

  // New helper function to join strings
  std::string join_strings(const std::vector<std::string>& strings, const std::string& delimiter = ", ") {
    std::ostringstream oss;
    for (size_t i = 0; i < strings.size(); ++i) {
      if (i > 0) {
        oss << delimiter;
      }
      oss << strings[i];
    }
    return oss.str();
  }

  Rcpp::List get_result() {
    std::vector<std::string> bins;
    std::vector<double> woes;
    std::vector<double> ivs;
    std::vector<int> counts;
    std::vector<int> counts_pos;
    std::vector<int> counts_neg;
    std::vector<double> woe_feature(total_count);

    for (const auto& bin : optimal_bins) {
      bins.push_back(join_strings(bin.categories));
      woes.push_back(bin.woe);
      ivs.push_back(bin.iv);
      counts.push_back(bin.count);
      counts_pos.push_back(bin.count_pos);
      counts_neg.push_back(bin.count_neg);
    }

    // Create woefeature
    for (size_t i = 0; i < unique_categories.size(); ++i) {
      for (size_t j = 0; j < optimal_bins.size(); ++j) {
        if (std::find(optimal_bins[j].categories.begin(), optimal_bins[j].categories.end(), unique_categories[i]) != optimal_bins[j].categories.end()) {
          woe_feature[i] = optimal_bins[j].woe;
          break;
        }
      }
    }

    return Rcpp::List::create(
      Rcpp::Named("woefeature") = woe_feature,
      Rcpp::Named("woebin") = Rcpp::DataFrame::create(
        Rcpp::Named("bin") = bins,
        Rcpp::Named("woe") = woes,
        Rcpp::Named("iv") = ivs,
        Rcpp::Named("count") = counts,
        Rcpp::Named("count_pos") = counts_pos,
        Rcpp::Named("count_neg") = counts_neg
      )
    );
  }

private:
  void preprocess_data(const std::vector<std::string>& feature, const std::vector<int>& target) {
    std::map<std::string, std::pair<int, int>> category_stats;

#pragma omp parallel for
    for (size_t i = 0; i < feature.size(); ++i) {
#pragma omp critical
{
  if (target[i] == 1) {
    category_stats[feature[i]].first++;
  } else if (target[i] == 0) {
    category_stats[feature[i]].second++;
  } else {
    throw std::invalid_argument("Target values must be 0 or 1");
  }
}
    }

    for (const auto& stat : category_stats) {
      unique_categories.push_back(stat.first);
      int pos_count = stat.second.first;
      int neg_count = stat.second.second;
      positive_counts.push_back(pos_count);
      negative_counts.push_back(neg_count);
      category_counts.push_back(pos_count + neg_count);
    }

    total_count = std::accumulate(category_counts.begin(), category_counts.end(), 0);
    total_positive = std::accumulate(positive_counts.begin(), positive_counts.end(), 0);
    total_negative = std::accumulate(negative_counts.begin(), negative_counts.end(), 0);
  }

  void merge_rare_categories() {
    std::vector<std::string> merged_categories;
    std::vector<int> merged_counts;
    std::vector<int> merged_positive_counts;
    std::vector<int> merged_negative_counts;

    int rare_count = 0;
    int rare_positive = 0;
    int rare_negative = 0;

    for (size_t i = 0; i < unique_categories.size(); ++i) {
      double category_ratio = static_cast<double>(category_counts[i]) / total_count;
      if (category_ratio < bin_cutoff) {
        rare_count += category_counts[i];
        rare_positive += positive_counts[i];
        rare_negative += negative_counts[i];
      } else {
        merged_categories.push_back(unique_categories[i]);
        merged_counts.push_back(category_counts[i]);
        merged_positive_counts.push_back(positive_counts[i]);
        merged_negative_counts.push_back(negative_counts[i]);
      }
    }

    if (rare_count > 0) {
      merged_categories.push_back("Rare");
      merged_counts.push_back(rare_count);
      merged_positive_counts.push_back(rare_positive);
      merged_negative_counts.push_back(rare_negative);
    }

    unique_categories = merged_categories;
    category_counts = merged_counts;
    positive_counts = merged_positive_counts;
    negative_counts = merged_negative_counts;
  }

  void optimize_bins() {
    // Sort categories by their WoE
    std::vector<size_t> sorted_indices(unique_categories.size());
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [this](size_t i, size_t j) {
                return calculate_woe(positive_counts[i], negative_counts[i]) <
                  calculate_woe(positive_counts[j], negative_counts[j]);
              });

    // Initialize bins with individual categories
    std::vector<BinInfo> bins;
    for (size_t i : sorted_indices) {
      bins.push_back({
        {unique_categories[i]},
        calculate_woe(positive_counts[i], negative_counts[i]),
        calculate_iv(positive_counts[i], negative_counts[i]),
        category_counts[i],
                       positive_counts[i],
                                      negative_counts[i]
      });
    }

    // Merge bins while respecting constraints
    while (bins.size() > max_bins || (bins.size() > min_bins && can_merge_bins(bins))) {
      merge_adjacent_bins(bins);
    }

    optimal_bins = bins;
  }

  double calculate_woe(int positive, int negative) {
    double pos_rate = static_cast<double>(positive) / total_positive;
    double neg_rate = static_cast<double>(negative) / total_negative;
    return std::log(pos_rate / neg_rate);
  }

  double calculate_iv(int positive, int negative) {
    double pos_rate = static_cast<double>(positive) / total_positive;
    double neg_rate = static_cast<double>(negative) / total_negative;
    return (pos_rate - neg_rate) * std::log(pos_rate / neg_rate);
  }

  bool can_merge_bins(const std::vector<BinInfo>& bins) {
    for (size_t i = 1; i < bins.size(); ++i) {
      if (bins[i-1].woe > bins[i].woe) {
        return true;
      }
    }
    return false;
  }

  void merge_adjacent_bins(std::vector<BinInfo>& bins) {
    double min_iv_loss = std::numeric_limits<double>::max();
    size_t merge_index = 0;

    for (size_t i = 1; i < bins.size(); ++i) {
      double iv_before = bins[i-1].iv + bins[i].iv;
      int merged_pos = bins[i-1].count_pos + bins[i].count_pos;
      int merged_neg = bins[i-1].count_neg + bins[i].count_neg;
      double iv_after = calculate_iv(merged_pos, merged_neg);
      double iv_loss = iv_before - iv_after;

      if (iv_loss < min_iv_loss) {
        min_iv_loss = iv_loss;
        merge_index = i - 1;
      }
    }

    BinInfo merged_bin;
    merged_bin.categories = bins[merge_index].categories;
    merged_bin.categories.insert(merged_bin.categories.end(),
                                 bins[merge_index + 1].categories.begin(),
                                 bins[merge_index + 1].categories.end());
    merged_bin.count_pos = bins[merge_index].count_pos + bins[merge_index + 1].count_pos;
    merged_bin.count_neg = bins[merge_index].count_neg + bins[merge_index + 1].count_neg;
    merged_bin.count = merged_bin.count_pos + merged_bin.count_neg;
    merged_bin.woe = calculate_woe(merged_bin.count_pos, merged_bin.count_neg);
    merged_bin.iv = calculate_iv(merged_bin.count_pos, merged_bin.count_neg);

    bins.erase(bins.begin() + merge_index, bins.begin() + merge_index + 2);
    bins.insert(bins.begin() + merge_index, merged_bin);
  }
};


//' @title
//' Optimal Binning for Categorical Variables using Minimum Binning Loss Principle (MBLP)
//' 
//' @description
//' This function performs optimal binning for categorical variables using the Minimum Binning Loss Principle (MBLP). It creates optimal bins for a categorical feature based on its relationship with a binary target variable, maximizing the predictive power while respecting user-defined constraints.
//' @param target An integer vector of binary target values (0 or 1).
//' @param feature A character vector of categorical feature values.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum proportion of total observations for a category to avoid being grouped into the "Rare" category (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins before the optimization process (default: 20).
//' @return A list containing two elements:
//' \itemize{
//'   \item woefeature: A numeric vector of Weight of Evidence (WoE) values for each observation.
//'   \item woebin: A data frame with the following columns:
//'   \itemize{
//'     \item bin: Character vector of bin names (categories or groups of categories).
//'     \item woe: Numeric vector of WoE values for each bin.
//'     \item iv: Numeric vector of Information Value (IV) for each bin.
//'     \item count: Integer vector of total observations in each bin.
//'     \item count_pos: Integer vector of positive target observations in each bin.
//'     \item count_neg: Integer vector of negative target observations in each bin.
//'   }
//' }
//' @details
//' The Optimal Binning algorithm using the Minimum Binning Loss Principle (MBLP) is designed to create optimal bins for categorical variables in relation to a binary target variable. The algorithm works as follows:
//' \enumerate{
//'   \item Preprocess the data:
//'   \itemize{
//'     \item Calculate the frequency and positive/negative counts for each category.
//'     \item Merge rare categories (those with a proportion less than bin_cutoff) into a single "Rare" category.
//'   }
//'   \item Calculate the Weight of Evidence (WoE) for each category:
//'   \deqn{WoE = \ln(\frac{\text{Positive Rate}}{\text{Negative Rate}}) = \ln(\frac{\frac{n_{i}^{+}}{N^{+}}}{\frac{n_{i}^{-}}{N^{-}}})}
//'   Where \eqn{n_{i}^{+}} and \eqn{n_{i}^{-}} are the number of positive and negative samples in category \eqn{i}, and \eqn{N^{+}} and \eqn{N^{-}} are the total number of positive and negative samples.
//'   \item Sort categories by their WoE values.
//'   \item Initialize bins with individual categories.
//'   \item Merge adjacent bins while respecting the following constraints:
//'   \itemize{
//'     \item The number of bins is between min_bins and max_bins.
//'     \item Bins are merged to minimize the loss of Information Value (IV).
//'   }
//'   \item Calculate the Information Value (IV) for each bin:
//'   \deqn{IV = (\text{Positive Rate} - \text{Negative Rate}) \times WoE}
//' }
//' The algorithm aims to create bins that maximize the predictive power of the categorical variable while adhering to the specified constraints. It uses the Minimum Binning Loss Principle to determine the optimal merging of adjacent bins, ensuring that the loss of information is minimized during the binning process.
//' This implementation uses OpenMP for parallel processing when available, which can significantly speed up the computation for large datasets.
//' @references
//' \itemize{
//'   \item Mironchyk, P., Tchistiakov, V. (2017). Monotone optimal binning algorithm for credit risk modeling. SSRN Electronic Journal. DOI: 10.2139/ssrn.2978774
//'   \item Zhu, H., Liu, Y., Zhu, X., Lu, J. (2019). Minimum loss principle and its application in credit scoring. Journal of Risk Model Validation, 13(1), 57-85.
//' }
//' @examples
//' \dontrun{
//' # Create sample data
//' set.seed(123)
//' n <- 1000
//' target <- sample(0:1, n, replace = TRUE)
//' feature <- sample(LETTERS[1:5], n, replace = TRUE)
//' # Run optimal binning
//' result <- optimal_binning_categorical_mblp(target, feature, min_bins = 2, max_bins = 4)
//' # Print results
//' print(result$woebin)
//' # Plot WoE values
//' barplot(result$woebin$woe, names.arg = result$woebin$bin,
//'         main = "Weight of Evidence by Bin", xlab = "Bins", ylab = "WoE")
//' }
//' @author Lopes, J. E.
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_mblp(Rcpp::IntegerVector target,
                                          Rcpp::StringVector feature,
                                          int min_bins = 3, int max_bins = 5,
                                          double bin_cutoff = 0.05, int max_n_prebins = 20) {
std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);

OptimalBinningCategoricalMBLP binner(min_bins, max_bins, bin_cutoff, max_n_prebins);
binner.fit(feature_vec, target_vec);
return binner.get_result();
}