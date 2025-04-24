// [[Rcpp::plugins(openmp)]]
#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <sstream>
#include <iomanip>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

// Helper function to calculate WoE and IV
void calculate_woe_iv(std::vector<int>& count_pos, std::vector<int>& count_neg, 
                      std::vector<double>& woe, std::vector<double>& iv,
                      int total_pos, int total_neg) {
  for (size_t i = 0; i < count_pos.size(); ++i) {
    double pos_rate = static_cast<double>(count_pos[i]) / total_pos;
    double neg_rate = static_cast<double>(count_neg[i]) / total_neg;
    
    // Handling edge cases to avoid log(0)
    if (pos_rate == 0) pos_rate = 0.0001;
    if (neg_rate == 0) neg_rate = 0.0001;
    
    woe[i] = std::log(pos_rate / neg_rate);
    iv[i] = (pos_rate - neg_rate) * woe[i];
  }
}

// Helper function to format bin ranges
std::string format_bin_range(double lower, double upper) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2);
  
  if (std::isinf(lower) && lower < 0) {
    oss << "[-Inf;";
  } else {
    oss << "[" << lower << ";";
  }
  
  if (std::isinf(upper) && upper > 0) {
    oss << "+Inf]";
  } else {
    oss << upper << ")";
  }
  
  return oss.str();
}

//' @title 
//' Binning Numerical Variables using Custom Cutpoints
//'
//' @description
//' This function performs optimal binning of a numerical variable based on predefined cutpoints,
//' calculates the Weight of Evidence (WoE) and Information Value (IV) for each bin, and transforms
//' the feature accordingly.
//'
//' @param feature A numeric vector representing the numerical feature to be binned.
//' @param target An integer vector representing the binary target variable (0 or 1).
//' @param cutpoints A numeric vector containing the cutpoints to define the bin boundaries.
//'
//' @return A list with two elements:
//' \item{woefeature}{A numeric vector representing the transformed feature with WoE values for each observation.}
//' \item{woebin}{A data frame containing detailed statistics for each bin, including counts, WoE, and IV.}
//'
//' @details
//' Binning is a preprocessing step that groups continuous values of a numerical feature into a smaller number of bins.
//' This function performs binning based on user-defined cutpoints, which allows you to define how the numerical
//' feature should be split into intervals. The resulting bins are evaluated using the WoE and IV metrics, which
//' are often used in predictive modeling, especially in credit risk modeling.
//'
//' The Weight of Evidence (WoE) is calculated as:
//' \deqn{\text{WoE} = \log\left(\frac{\text{Positive Rate}}{\text{Negative Rate}}\right)}
//' where the Positive Rate is the proportion of positive observations (target = 1) within the bin, and the Negative
//' Rate is the proportion of negative observations (target = 0) within the bin.
//'
//' The Information Value (IV) measures the predictive power of the numerical feature and is calculated as:
//' \deqn{IV = \sum (\text{Positive Rate} - \text{Negative Rate}) \times \text{WoE}}
//'
//' The IV metric provides insight into how well the binned feature predicts the target variable:
//' \itemize{
//'   \item IV < 0.02: Not predictive
//'   \item 0.02 <= IV < 0.1: Weak predictive power
//'   \item 0.1 <= IV < 0.3: Medium predictive power
//'   \item IV >= 0.3: Strong predictive power
//' }
//'
//' The WoE transformation helps to convert the numerical variable into a continuous numeric feature,
//' which can be directly used in logistic regression and other predictive models, improving model interpretability and performance.
//'
//' @examples
//' \dontrun{
//' # Example usage
//' feature <- c(23, 45, 34, 25, 56, 48, 35, 29, 53, 41)
//' target <- c(1, 0, 1, 1, 0, 0, 0, 1, 1, 0)
//' cutpoints <- c(30, 40, 50)
//' result <- binning_numerical_cutpoints(feature, target, cutpoints)
//' print(result$woefeature)  # WoE-transformed feature
//' print(result$woebin)      # WoE and IV statistics for each bin
//' }
//'
//' @references
//' \itemize{
//'   \item Siddiqi, N. (2006). Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring. 
//'         John Wiley & Sons.
//' }
//'
//' @author Lopes, J. E.
//'
//' @export
// [[Rcpp::export]]
List binning_numerical_cutpoints(NumericVector feature, IntegerVector target, 
                                 NumericVector cutpoints) {
  int n = feature.size();
  int num_bins = cutpoints.size() + 1;
  
  std::vector<int> count(num_bins, 0);
  std::vector<int> count_pos(num_bins, 0);
  std::vector<int> count_neg(num_bins, 0);
  std::vector<double> woe(num_bins);
  std::vector<double> iv(num_bins);
  std::vector<double> bin_edges(num_bins + 1);
  
  int total_pos = 0, total_neg = 0;
  
  // Sort cutpoints
  std::sort(cutpoints.begin(), cutpoints.end());
  
  // Set bin edges
  bin_edges[0] = -INFINITY;
  for (int i = 0; i < cutpoints.size(); ++i) {
    bin_edges[i + 1] = cutpoints[i];
  }
  bin_edges[num_bins] = INFINITY;
  
  // Assign observations to bins
  for (int i = 0; i < n; ++i) {
    int bin = std::upper_bound(bin_edges.begin(), bin_edges.end(), feature[i]) - bin_edges.begin() - 1;
    count[bin]++;
    if (target[i] == 1) {
      count_pos[bin]++;
      total_pos++;
    } else {
      count_neg[bin]++;
      total_neg++;
    }
  }
  
  // Calculate WoE and IV
  calculate_woe_iv(count_pos, count_neg, woe, iv, total_pos, total_neg);
  
  // Format bin ranges
  CharacterVector bin_ranges(num_bins);
  for (int i = 0; i < num_bins; ++i) {
    bin_ranges[i] = format_bin_range(bin_edges[i], bin_edges[i + 1]);
  }
  
  // Create woebin DataFrame
  DataFrame woebin = DataFrame::create(
    Named("bin") = bin_ranges,
    Named("count") = count,
    Named("count_pos") = count_pos,
    Named("count_neg") = count_neg,
    Named("woe") = woe,
    Named("iv") = iv
  );
  
  // Assign WoE to feature values
  NumericVector woefeature(n);
  for (int i = 0; i < n; ++i) {
    int bin = std::upper_bound(bin_edges.begin(), bin_edges.end(), feature[i]) - bin_edges.begin() - 1;
    woefeature[i] = woe[bin];
  }
  
  return List::create(
    Named("woefeature") = woefeature,
    Named("woebin") = woebin
  );
}

//' Binning Categorical Variables using Custom Cutpoints
//'
//' This function performs optimal binning of categorical variables based on predefined cutpoints, 
//' calculates the Weight of Evidence (WoE) and Information Value (IV) for each bin, 
//' and transforms the feature accordingly.
//'
//' @param feature A character vector representing the categorical feature to be binned.
//' @param target An integer vector representing the binary target variable (0 or 1).
//' @param cutpoints A character vector containing the bin definitions, with categories separated by '+' (e.g., "A+B+C").
//' @return A list with two elements:
//' \item{woefeature}{A numeric vector representing the transformed feature with WoE values for each observation.}
//' \item{woebin}{A data frame containing detailed statistics for each bin, including counts, WoE, and IV.}
//' 
//' @details
//' Binning is a preprocessing step that groups categories of a categorical feature into a smaller number of bins. 
//' This function performs binning based on user-defined cutpoints, where each cutpoint specifies a group of categories 
//' that should be combined into a single bin. The resulting bins are evaluated using the WoE and IV metrics, which 
//' are often used in predictive modeling, especially in credit risk modeling.
//' 
//' The Weight of Evidence (WoE) is calculated as:
//' \deqn{\text{WoE} = \log\left(\frac{\text{Positive Rate}}{\text{Negative Rate}}\right)}
//' where the Positive Rate is the proportion of positive observations (target = 1) within the bin, and the Negative Rate is the proportion of negative observations (target = 0) within the bin. 
//' 
//' The Information Value (IV) measures the predictive power of the categorical feature and is calculated as:
//' \deqn{IV = \sum (\text{Positive Rate} - \text{Negative Rate}) \times \text{WoE}}
//' 
//' The IV metric provides insight into how well the binned feature predicts the target variable:
//' \itemize{
//'   \item IV < 0.02: Not predictive
//'   \item 0.02 <= IV < 0.1: Weak predictive power
//'   \item 0.1 <= IV < 0.3: Medium predictive power
//'   \item IV >= 0.3: Strong predictive power
//' }
//' 
//' WoE is used to transform the categorical variable into a continuous numeric variable, which can be used directly in logistic regression and other predictive models.
//'
//' @examples
//' \dontrun{
//' # Example usage
//' feature <- c("A", "B", "C", "A", "B", "C", "A", "C", "C", "B")
//' target <- c(1, 0, 1, 1, 0, 0, 0, 1, 1, 0)
//' cutpoints <- c("A+B", "C")
//' result <- binning_categorical_cutpoints(feature, target, cutpoints)
//' print(result$woefeature)  # WoE-transformed feature
//' print(result$woebin)      # WoE and IV statistics for each bin
//' }
//' 
//' @references
//' Siddiqi, N. (2006). Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring. 
//' John Wiley & Sons.
//'
//' @export
// [[Rcpp::export]]
List binning_categorical_cutpoints(CharacterVector feature, IntegerVector target, 
                                   CharacterVector cutpoints) {
  int n = feature.size();
  int num_bins = cutpoints.size();
  
  std::unordered_map<std::string, int> category_to_bin;
  for (int i = 0; i < num_bins; ++i) {
    std::string bin_categories = as<std::string>(cutpoints[i]);
    size_t pos = 0;
    while ((pos = bin_categories.find("+")) != std::string::npos) {
      std::string category = bin_categories.substr(0, pos);
      category_to_bin[category] = i;
      bin_categories.erase(0, pos + 1);
    }
    category_to_bin[bin_categories] = i;
  }
  
  std::vector<int> count(num_bins, 0);
  std::vector<int> count_pos(num_bins, 0);
  std::vector<int> count_neg(num_bins, 0);
  std::vector<double> woe(num_bins);
  std::vector<double> iv(num_bins);
  
  int total_pos = 0, total_neg = 0;
  
  // Assign observations to bins
  for (int i = 0; i < n; ++i) {
    std::string cat = as<std::string>(feature[i]);
    int bin = category_to_bin[cat];
    count[bin]++;
    if (target[i] == 1) {
      count_pos[bin]++;
      total_pos++;
    } else {
      count_neg[bin]++;
      total_neg++;
    }
  }
  
  // Calculate WoE and IV
  calculate_woe_iv(count_pos, count_neg, woe, iv, total_pos, total_neg);
  
  // Create woebin DataFrame
  DataFrame woebin = DataFrame::create(
    Named("bin") = cutpoints,
    Named("count") = count,
    Named("count_pos") = count_pos,
    Named("count_neg") = count_neg,
    Named("woe") = woe,
    Named("iv") = iv
  );
  
  // Assign WoE to feature values
  NumericVector woefeature(n);
  for (int i = 0; i < n; ++i) {
    std::string cat = as<std::string>(feature[i]);
    int bin = category_to_bin[cat];
    woefeature[i] = woe[bin];
  }
  
  return List::create(
    Named("woefeature") = woefeature,
    Named("woebin") = woebin
  );
}

