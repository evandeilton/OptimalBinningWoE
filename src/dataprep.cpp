// OptimallBinningDataPreprocessor.cpp

#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <vector>
#include <string>
#include <numeric>
#include <unordered_set>

using namespace Rcpp;

// Helper function to compute summary statistics
List compute_summary(NumericVector data) {
  // Remove NA values
  std::vector<double> vec;
  vec.reserve(data.size());
  for(auto val : data) {
    if(!NumericVector::is_na(val)) {
      vec.push_back(val);
    }
  }
  int n = vec.size();
  if(n == 0) {
    return List::create(Named("min") = NA_REAL,
                        Named("Q1") = NA_REAL,
                        Named("median") = NA_REAL,
                        Named("mean") = NA_REAL,
                        Named("Q3") = NA_REAL,
                        Named("max") = NA_REAL);
  }

  std::sort(vec.begin(), vec.end());

  double min = vec.front();
  double Q1, median, Q3, max = vec.back();

  auto get_percentile = [&](double p) -> double {
    if(n == 1) return vec[0];
    double pos = p * (n + 1) / 100.0;
    if(pos < 1.0) return vec[0];
    if(pos >= n) return vec[n-1];
    int idx = std::floor(pos) - 1;
    double frac = pos - std::floor(pos);
    return vec[idx] + frac * (vec[idx + 1] - vec[idx]);
  };

  Q1 = get_percentile(25.0);
  median = get_percentile(50.0);
  Q3 = get_percentile(75.0);

  double mean = std::accumulate(vec.begin(), vec.end(), 0.0) / n;

  return List::create(Named("min") = min,
                      Named("Q1") = Q1,
                      Named("median") = median,
                      Named("mean") = mean,
                      Named("Q3") = Q3,
                      Named("max") = max);
}

// Helper function to compute Grubbs' critical value
double grubbs_critical(int n, double alpha) {
  // Using the formula for two-sided Grubbs' test
  double t = R::qt(1 - alpha / (2 * n), n - 2, 1, 0);
  double numerator = (n - 1) * std::sqrt(std::pow(t, 2));
  double denominator = std::sqrt(n) * std::sqrt(n - 2 + std::pow(t, 2));
  return numerator / denominator;
}

// Helper function to check if a CharacterVector contains a specific string
bool vector_contains(const CharacterVector& vec, const String& str) {
  for(int i = 0; i < vec.size(); i++) {
    if(!CharacterVector::is_na(vec[i])) {
      if(vec[i] == str) {
        return true;
      }
    }
  }
  return false;
}

// Helper function to serialize a List to a string
std::string list_to_string(const List& lst) {
  std::string result = "{ ";
  CharacterVector names = lst.names();
  for(int i = 0; i < lst.size(); i++) {
    std::string name = as<std::string>(names[i]);
    std::string value;
    if(TYPEOF(lst[i]) == REALSXP) {
      double num = as<double>(lst[i]);
      value = std::to_string(num);
    } else if(TYPEOF(lst[i]) == STRSXP) {
      if(lst[i] == NA_STRING) {
        value = "NA";
      } else {
        value = as<std::string>(lst[i]);
      }
    } else {
      value = "NA";
    }
    result += name + ": " + value + ", ";
  }
  if(result.size() > 2) {
    result = result.substr(0, result.size() - 2); // remove last ", "
  }
  result += " }";
  return result;
}

//' Preprocesses a numeric or categorical variable for optimal binning with handling of missing values and outliers
//'
//' This function preprocesses a given numeric or categorical feature, handling missing values and outliers based on the specified method. It can process both numeric and categorical features and supports outlier detection through various methods, including IQR, Z-score, and Grubbs' test. The function also generates summary statistics before and after preprocessing.
//'
//' @param target Numeric vector representing the binary target variable, where 1 indicates a positive event (e.g., default) and 0 indicates a negative event (e.g., non-default).
//' @param feature Numeric or character vector representing the feature to be binned.
//' @param num_miss_value (Optional) Numeric value to replace missing values in numeric features. Default is -999.0.
//' @param char_miss_value (Optional) String value to replace missing values in categorical features. Default is "N/A".
//' @param outlier_method (Optional) Method to detect outliers. Choose from "iqr", "zscore", or "grubbs". Default is "iqr".
//' @param outlier_process (Optional) Boolean flag indicating whether outliers should be processed. Default is FALSE.
//' @param preprocess (Optional) Character vector specifying what to return: "feature", "report", or "both". Default is "both".
//' @param iqr_k (Optional) The multiplier for the interquartile range (IQR) when using the IQR method to detect outliers. Default is 1.5.
//' @param zscore_threshold (Optional) The threshold for Z-score to detect outliers. Default is 3.0.
//' @param grubbs_alpha (Optional) The significance level for Grubbs' test to detect outliers. Default is 0.05.
//'
//' @return A list containing the following elements based on the \code{preprocess} parameter:
//' \itemize{
//'   \item \code{preprocess}: A DataFrame containing the original and preprocessed feature values.
//'   \item \code{report}: A DataFrame summarizing the variable type, number of missing values, number of outliers (for numeric features), and statistics before and after preprocessing.
//' }
//'
//' @details
//' The function can handle both numeric and categorical features. For numeric features, it replaces missing values with \code{num_miss_value} and can apply outlier detection using different methods. For categorical features, it replaces missing values with \code{char_miss_value}. The function can return the preprocessed feature and/or a report with summary statistics.
//'
//' @examples
//' \dontrun{
//' target <- c(0, 1, 1, 0, 1)
//' feature_numeric <- c(10, 20, NA, 40, 50)
//' feature_categorical <- c("A", "B", NA, "B", "A")
//' result <- OptimalBinningDataPreprocessor(target, feature_numeric, outlier_process = TRUE)
//' result <- OptimalBinningDataPreprocessor(target, feature_categorical)
//' }
//' @export
// [[Rcpp::export]]
List OptimalBinningDataPreprocessor(
    NumericVector target,
    SEXP feature,
    double num_miss_value = -999.0,
    std::string char_miss_value = "N/A",
    std::string outlier_method = "iqr",
    bool outlier_process = false,
    CharacterVector preprocess = CharacterVector::create("both"),
    double iqr_k = 1.5,
    double zscore_threshold = 3.0,
    double grubbs_alpha = 0.05)
{
  // Initialize report variables
  std::string variable_type = "unknown";
  int missing_count = 0;
  int outlier_count = 0;
  List original_stats;
  List preprocessed_stats;

  // Check if target is binary
  std::unordered_set<double> target_unique;
  for(auto val : target) {
    if(!NumericVector::is_na(val)) {
      target_unique.insert(val);
    }
  }
  if(target_unique.size() != 2) {
    stop("Target variable must be binary.");
  }

  // Determine feature type
  bool is_numeric = false;
  bool is_character = false;
  NumericVector feature_numeric;
  CharacterVector feature_character;

  if(TYPEOF(feature) == REALSXP || Rf_isInteger(feature) || Rf_isReal(feature)) {
    is_numeric = true;
    feature_numeric = as<NumericVector>(feature);
    variable_type = "numeric";
  } else if(TYPEOF(feature) == STRSXP || Rf_isFactor(feature) || Rf_isString(feature)) {
    is_character = true;
    feature_character = as<CharacterVector>(feature);
    variable_type = "categorical";
  } else {
    stop("Feature must be either numeric or categorical (string).");
  }

  // Handle missing values
  if(is_numeric) {
    // Compute original stats
    original_stats = compute_summary(feature_numeric);

    // Replace NA with num_miss_value
    for(int i = 0; i < feature_numeric.size(); ++i) {
      if(NumericVector::is_na(feature_numeric[i])) {
        feature_numeric[i] = num_miss_value;
        missing_count++;
      }
    }
  } else if(is_character) {
    // Compute original stats: for categorical, we can skip detailed stats
    original_stats = List::create(Named("min") = NA_STRING,
                                  Named("Q1") = NA_STRING,
                                  Named("median") = NA_STRING,
                                  Named("mean") = NA_STRING,
                                  Named("Q3") = NA_STRING,
                                  Named("max") = NA_STRING);

    // Replace NA with char_miss_value
    for(int i = 0; i < feature_character.size(); ++i) {
      if(feature_character[i] == NA_STRING) {
        feature_character[i] = char_miss_value;
        missing_count++;
      }
    }
  }

  // Outlier detection and handling for numeric variables
  if(is_numeric && outlier_process) {
    if(outlier_method == "iqr") {
      // Compute Q1 and Q3
      NumericVector sorted = clone(feature_numeric);
      std::sort(sorted.begin(), sorted.end());
      int n = sorted.size();
      double Q1 = sorted[std::floor(0.25 * (n + 1)) - 1];
      double Q3 = sorted[std::floor(0.75 * (n + 1)) - 1];
      double IQR = Q3 - Q1;
      double lower_bound = Q1 - iqr_k * IQR;
      double upper_bound = Q3 + iqr_k * IQR;

      // Handle outliers by capping
      for(int i = 0; i < feature_numeric.size(); ++i) {
        double val = feature_numeric[i];
        if(val < lower_bound || val > upper_bound) {
          outlier_count++;
          if(val < lower_bound) {
            feature_numeric[i] = lower_bound;
          } else {
            feature_numeric[i] = upper_bound;
          }
        }
      }

    } else if(outlier_method == "zscore") {
      // Compute mean and standard deviation
      double sum = 0.0;
      int count = 0;
      for(auto val : feature_numeric) {
        if(!NumericVector::is_na(val)) {
          sum += val;
          count++;
        }
      }
      double mean = sum / count;
      double sq_sum = 0.0;
      for(auto val : feature_numeric) {
        if(!NumericVector::is_na(val)) {
          sq_sum += std::pow(val - mean, 2);
        }
      }
      double sd = std::sqrt(sq_sum / (count - 1));

      // Handle outliers by capping
      double lower_bound = mean - zscore_threshold * sd;
      double upper_bound = mean + zscore_threshold * sd;
      for(int i = 0; i < feature_numeric.size(); ++i) {
        double val = feature_numeric[i];
        if(val < lower_bound || val > upper_bound) {
          outlier_count++;
          if(val < lower_bound) {
            feature_numeric[i] = lower_bound;
          } else {
            feature_numeric[i] = upper_bound;
          }
        }
      }

    } else if(outlier_method == "grubbs") {
      // Iteratively apply Grubbs' test
      std::vector<double> data;
      for(auto val : feature_numeric) {
        if(!NumericVector::is_na(val)) {
          data.push_back(val);
        }
      }
      bool continue_test = true;
      while(continue_test && data.size() > 2) {
        // Compute mean and standard deviation
        double sum = std::accumulate(data.begin(), data.end(), 0.0);
        double mean = sum / data.size();
        double sq_sum = 0.0;
        for(auto val : data) {
          sq_sum += std::pow(val - mean, 2);
        }
        double sd = std::sqrt(sq_sum / (data.size() - 1));
        if(sd == 0) break;

        // Find the maximum absolute deviation
        double max_dev = 0.0;
        int max_idx = -1;
        for(int i = 0; i < data.size(); ++i) {
          double dev = std::abs(data[i] - mean);
          if(dev > max_dev) {
            max_dev = dev;
            max_idx = i;
          }
        }

        // Compute Grubbs' statistic
        double G = max_dev / sd;

        // Compute critical value
        double t_dist = R::qt(1 - grubbs_alpha / (2 * data.size()), data.size() - 2, 1, 0);
        double G_critical = ((data.size() - 1) * std::sqrt(std::pow(t_dist, 2))) /
          (std::sqrt(data.size()) * std::sqrt(data.size() - 2 + std::pow(t_dist, 2)));

        if(G > G_critical) {
          // Remove the outlier
          double outlier = data[max_idx];
          // Find and replace in feature_numeric
          for(int i = 0; i < feature_numeric.size(); ++i) {
            if(feature_numeric[i] == outlier) {
              feature_numeric[i] = NA_REAL; // Mark as missing
              outlier_count++;
              break;
            }
          }
          data.erase(data.begin() + max_idx);
        } else {
          continue_test = false;
        }
      }
      // After Grubbs' test, replace NA with num_miss_value
      for(int i = 0; i < feature_numeric.size(); ++i) {
        if(NumericVector::is_na(feature_numeric[i])) {
          feature_numeric[i] = num_miss_value;
        }
      }
    } else {
      stop("Invalid outlier_method. Choose from 'iqr', 'zscore', or 'grubbs'.");
    }
  }

  // Compute preprocessed stats
  if(is_numeric) {
    preprocessed_stats = compute_summary(feature_numeric);
  } else if(is_character) {
    // For categorical, stats can be count of unique categories
    std::unordered_set<std::string> unique_cats;
    for(auto val : feature_character) {
      if(String(val) != NA_STRING) {
        unique_cats.insert(std::string(val));
      }
    }
    preprocessed_stats = List::create(Named("unique_count") = unique_cats.size());
  }

  // Serialize original_stats and preprocessed_stats to strings
  std::string original_stats_str = list_to_string(original_stats);
  std::string preprocessed_stats_str = list_to_string(preprocessed_stats);

  // Prepare preprocess DataFrame
  DataFrame preprocess_df;
  if(vector_contains(preprocess, "feature") || vector_contains(preprocess, "both")) {
    if(is_numeric) {
      preprocess_df = DataFrame::create(
        Named("feature") = feature_numeric,
        Named("feature_preprocessed") = feature_numeric
      );
    } else if(is_character) {
      preprocess_df = DataFrame::create(
        Named("feature") = feature_character,
        Named("feature_preprocessed") = feature_character
      );
    }
  }
  
  // Prepare report DataFrame with serialized stats
  DataFrame report_df = DataFrame::create(
    Named("variable_type") = variable_type,
    Named("missing_count") = missing_count,
    Named("outlier_count") = (is_numeric ? outlier_count : NA_INTEGER),
    Named("original_stats") = original_stats_str,
    Named("preprocessed_stats") = preprocessed_stats_str
  );
  
  preprocess_df.attr("class") = CharacterVector::create("data.table", "data.frame");
  report_df.attr("class") = CharacterVector::create("data.table", "data.frame");

  // Prepare output List
  List output;
  if(vector_contains(preprocess, "feature") || vector_contains(preprocess, "both")) {
    output["preprocess"] = preprocess_df;
  }
  if(vector_contains(preprocess, "report") || vector_contains(preprocess, "both")) {
    output["report"] = report_df;
  }
  
  output.attr("class") = CharacterVector::create("data.table", "data.frame");
  
  return output;
}
