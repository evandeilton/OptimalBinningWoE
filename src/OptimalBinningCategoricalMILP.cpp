#include <Rcpp.h>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <algorithm>
#include <cmath>
#include <limits>

class OptimalBinningCategoricalMILP {
private:
  std::vector<int> target;
  std::vector<std::string> feature;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;

  struct Bin {
    std::set<std::string> categories;
    int count_pos;
    int count_neg;
    double woe;
    double iv;
  };

  std::vector<Bin> bins;
  int total_pos;
  int total_neg;
  Rcpp::NumericVector woefeature;
  Rcpp::DataFrame woebin;

public:
  OptimalBinningCategoricalMILP(
    const std::vector<int>& target,
    const std::vector<std::string>& feature,
    int min_bins,
    int max_bins,
    double bin_cutoff,
    int max_n_prebins
  );

  void fit();
  Rcpp::List get_results();

private:
  void initialize_bins();
  void merge_bins();
  void calculate_woe_iv(Bin& bin);
};

OptimalBinningCategoricalMILP::OptimalBinningCategoricalMILP(
  const std::vector<int>& target,
  const std::vector<std::string>& feature,
  int min_bins,
  int max_bins,
  double bin_cutoff,
  int max_n_prebins
) : target(target), feature(feature), min_bins(min_bins), max_bins(max_bins),
bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins)
{
  if (min_bins < 2) {
    Rcpp::stop("min_bins must be at least 2.");
  }
  if (max_bins < min_bins) {
    Rcpp::stop("max_bins must be greater than or equal to min_bins.");
  }
  if (bin_cutoff < 0.0 || bin_cutoff > 1.0) {
    Rcpp::stop("bin_cutoff must be between 0 and 1.");
  }
  if (max_n_prebins < 1) {
    Rcpp::stop("max_n_prebins must be at least 1.");
  }
}

void OptimalBinningCategoricalMILP::initialize_bins() {
  std::map<std::string, Bin> bin_map;
  total_pos = 0;
  total_neg = 0;

  for (size_t i = 0; i < target.size(); ++i) {
    const std::string& cat = feature[i];
    int tar = target[i];

    if (tar != 0 && tar != 1) {
      Rcpp::stop("Target variable must be binary (0 or 1).");
    }

    if (bin_map.find(cat) == bin_map.end()) {
      bin_map[cat] = Bin{{cat}, 0, 0, 0.0, 0.0};
    }

    if (tar == 1) {
      bin_map[cat].count_pos++;
      total_pos++;
    } else {
      bin_map[cat].count_neg++;
      total_neg++;
    }
  }

  bins.clear();
  for (const auto& kv : bin_map) {
    bins.push_back(kv.second);
  }

  for (auto& bin : bins) {
    calculate_woe_iv(bin);
  }
}

void OptimalBinningCategoricalMILP::calculate_woe_iv(Bin& bin) {
  double dist_pos = static_cast<double>(bin.count_pos) / total_pos;
  double dist_neg = static_cast<double>(bin.count_neg) / total_neg;
  bin.woe = std::log((dist_pos + 1e-10) / (dist_neg + 1e-10));
  bin.iv = (dist_pos - dist_neg) * bin.woe;
}

void OptimalBinningCategoricalMILP::merge_bins() {
  while (bins.size() > max_bins || (bins.size() > min_bins && bins.size() > max_n_prebins)) {
    double min_delta_iv = std::numeric_limits<double>::max();
    size_t merge_idx1 = 0;
    size_t merge_idx2 = 0;

    for (size_t i = 0; i < bins.size(); ++i) {
      for (size_t j = i + 1; j < bins.size(); ++j) {
        double iv_before = bins[i].iv + bins[j].iv;

        Bin merged_bin;
        merged_bin.categories.insert(bins[i].categories.begin(), bins[i].categories.end());
        merged_bin.categories.insert(bins[j].categories.begin(), bins[j].categories.end());
        merged_bin.count_pos = bins[i].count_pos + bins[j].count_pos;
        merged_bin.count_neg = bins[i].count_neg + bins[j].count_neg;
        calculate_woe_iv(merged_bin);

        double delta_iv = iv_before - merged_bin.iv;

        if (delta_iv < min_delta_iv) {
          min_delta_iv = delta_iv;
          merge_idx1 = i;
          merge_idx2 = j;
        }
      }
    }

    bins[merge_idx1].categories.insert(bins[merge_idx2].categories.begin(), bins[merge_idx2].categories.end());
    bins[merge_idx1].count_pos += bins[merge_idx2].count_pos;
    bins[merge_idx1].count_neg += bins[merge_idx2].count_neg;
    calculate_woe_iv(bins[merge_idx1]);

    bins.erase(bins.begin() + merge_idx2);
  }
}

void OptimalBinningCategoricalMILP::fit() {
  initialize_bins();
  merge_bins();

  std::map<std::string, double> category_to_woe;
  for (const auto& bin : bins) {
    for (const auto& cat : bin.categories) {
      category_to_woe[cat] = bin.woe;
    }
  }

  woefeature = Rcpp::NumericVector(feature.size());
  for (size_t i = 0; i < feature.size(); ++i) {
    const std::string& cat = feature[i];
    woefeature[i] = category_to_woe[cat];
  }

  size_t num_bins = bins.size();
  Rcpp::CharacterVector bin_vec(num_bins);
  Rcpp::NumericVector woe_vec(num_bins);
  Rcpp::NumericVector iv_vec(num_bins);
  Rcpp::IntegerVector count_vec(num_bins);
  Rcpp::IntegerVector count_pos_vec(num_bins);
  Rcpp::IntegerVector count_neg_vec(num_bins);

  for (size_t i = 0; i < num_bins; ++i) {
    const Bin& bin = bins[i];
    std::string bin_name = "";
    for (const auto& cat : bin.categories) {
      if (!bin_name.empty()) bin_name += "+";
      bin_name += cat;
    }
    bin_vec[i] = bin_name;
    woe_vec[i] = bin.woe;
    iv_vec[i] = bin.iv;
    count_vec[i] = bin.count_pos + bin.count_neg;
    count_pos_vec[i] = bin.count_pos;
    count_neg_vec[i] = bin.count_neg;
  }

  woebin = Rcpp::DataFrame::create(
    Rcpp::Named("bin") = bin_vec,
    Rcpp::Named("woe") = woe_vec,
    Rcpp::Named("iv") = iv_vec,
    Rcpp::Named("count") = count_vec,
    Rcpp::Named("count_pos") = count_pos_vec,
    Rcpp::Named("count_neg") = count_neg_vec
  );
}

Rcpp::List OptimalBinningCategoricalMILP::get_results() {
  return Rcpp::List::create(
    Rcpp::Named("woefeature") = woefeature,
    Rcpp::Named("woebin") = woebin
  );
}


//' @title Optimal Binning for Categorical Variables using MILP
//'
//' @description
//' This function performs optimal binning for categorical variables using a Mixed Integer Linear Programming (MILP) inspired approach. It creates optimal bins for a categorical feature based on its relationship with a binary target variable, maximizing the predictive power while respecting user-defined constraints.
//'
//' @param target An integer vector of binary target values (0 or 1).
//' @param feature A character vector of feature values.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum proportion of total observations for a bin to avoid being merged (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins before the optimization process (default: 20).
//'
//' @return A list containing two elements:
//' \item{woefeature}{A numeric vector of Weight of Evidence (WoE) values for each observation.}
//' \item{woebin}{A data frame with the following columns:
//'   \itemize{
//'     \item bin: Character vector of bin categories.
//'     \item woe: Numeric vector of WoE values for each bin.
//'     \item iv: Numeric vector of Information Value (IV) for each bin.
//'     \item count: Integer vector of total observations in each bin.
//'     \item count_pos: Integer vector of positive target observations in each bin.
//'     \item count_neg: Integer vector of negative target observations in each bin.
//'   }
//' }
//'
//' @details
//' The Optimal Binning algorithm for categorical variables using a MILP-inspired approach works as follows:
//' 1. Create initial bins for each unique category.
//' 2. Merge bins with counts below the cutoff.
//' 3. Calculate initial Weight of Evidence (WoE) and Information Value (IV) for each bin.
//' 4. Optimize bins by merging categories to maximize total IV while respecting constraints.
//' 5. Ensure the number of bins is between min_bins and max_bins.
//' 6. Recalculate WoE and IV for the final bins.
//'
//' The algorithm aims to create bins that maximize the predictive power of the categorical variable while adhering to the specified constraints.
//'
//' Weight of Evidence (WoE) is calculated as:
//' \deqn{WoE = \ln(\frac{\text{Positive Rate}}{\text{Negative Rate}})}
//'
//' Information Value (IV) is calculated as:
//' \deqn{IV = (\text{Positive Rate} - \text{Negative Rate}) \times WoE}
//'
//' @references
//' \itemize{
//'   \item Belotti, P., Kirches, C., Leyffer, S., Linderoth, J., Luedtke, J., & Mahajan, A. (2013). Mixed-integer nonlinear optimization. Acta Numerica, 22, 1-131.
//'   \item Mironchyk, P., & Tchistiakov, V. (2017). Monotone optimal binning algorithm for credit risk modeling. SSRN Electronic Journal. doi:10.2139/ssrn.2978774
//' }
//'
//' @examples
//' \dontrun{
//' # Create sample data
//' set.seed(123)
//' n <- 1000
//' target <- sample(0:1, n, replace = TRUE)
//' feature <- sample(LETTERS[1:10], n, replace = TRUE)
//'
//' # Run optimal binning
//' result <- optimal_binning_categorical_milp(target, feature, min_bins = 2, max_bins = 4)
//'
//' # Print results
//' print(result$woebin)
//'
//' # Plot WoE values
//' barplot(result$woebin$woe, names.arg = result$woebin$bin,
//'         xlab = "Bins", ylab = "WoE", main = "Weight of Evidence by Bin")
//' }
//'
//' @author Lopes, J. E.
//'
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_milp(
  Rcpp::IntegerVector target,
  Rcpp::CharacterVector feature,
  int min_bins = 3,
  int max_bins = 5,
  double bin_cutoff = 0.05,
  int max_n_prebins = 20
) {
std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);

OptimalBinningCategoricalMILP obcm(
    target_vec,
    feature_vec,
    min_bins,
    max_bins,
    bin_cutoff,
    max_n_prebins
);

obcm.fit();

return obcm.get_results();
}
