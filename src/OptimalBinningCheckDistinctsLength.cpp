#include <Rcpp.h>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <string>

// [[Rcpp::plugins(cpp11)]]

// Function to create frequency table and return dimensions
Rcpp::IntegerVector create_frequency_table(SEXP x, Rcpp::IntegerVector target) {
  // Using nested unordered_map to avoid out-of-bounds access
  std::unordered_map<std::string, std::unordered_map<int, int>> freq_table;
  
  // Check if inputs are valid
  if (Rf_isNull(x) || target.size() == 0) {
    Rcpp::warning("Input is null or empty");
    return Rcpp::IntegerVector::create(0, 0);
  }
  
  // Check if lengths match
  R_xlen_t x_len = Rf_length(x);
  if (x_len != target.size()) {
    Rcpp::stop("Length of feature and target do not match");
  }
  
  // Iterate based on the type of x
  switch (TYPEOF(x)) {
  case INTSXP: {
    Rcpp::IntegerVector vec(x);
    for (R_xlen_t i = 0; i < x_len; ++i) {
      if (vec[i] != NA_INTEGER && target[i] != NA_INTEGER) {
        std::string key = std::to_string(vec[i]);
        int target_val = target[i];
        freq_table[key][target_val]++;
      }
    }
    break;
  }
  case REALSXP: {
    Rcpp::NumericVector vec(x);
    for (R_xlen_t i = 0; i < x_len; ++i) {
      if (!Rcpp::NumericVector::is_na(vec[i]) && target[i] != NA_INTEGER) {
        std::string key = std::to_string(vec[i]);
        int target_val = target[i];
        freq_table[key][target_val]++;
      }
    }
    break;
  }
  case STRSXP: {
    Rcpp::CharacterVector vec(x);
    for (R_xlen_t i = 0; i < x_len; ++i) {
      if (vec[i] != NA_STRING && target[i] != NA_INTEGER) {
        std::string key = Rcpp::as<std::string>(vec[i]);
        int target_val = target[i];
        freq_table[key][target_val]++;
      }
    }
    break;
  }
  case LGLSXP: {
    Rcpp::LogicalVector vec(x);
    for (R_xlen_t i = 0; i < x_len; ++i) {
      if (vec[i] != NA_LOGICAL && target[i] != NA_INTEGER) {
        std::string key = std::to_string(static_cast<int>(vec[i]));
        int target_val = target[i];
        freq_table[key][target_val]++;
      }
    }
    break;
  }
  default:
    Rcpp::stop("Unsupported type");
  }
  
  // Determine the number of categories and target classes
  int num_categories = freq_table.size();
  int num_target_classes = 0;
  
  for (const auto& entry : freq_table) {
    num_target_classes = std::max(num_target_classes, static_cast<int>(entry.second.size()));
  }
  
  return Rcpp::IntegerVector::create(num_categories, num_target_classes);
}

// [[Rcpp::export]]
Rcpp::IntegerVector OptimalBinningCheckDistinctsLength(SEXP x, Rcpp::IntegerVector target) {
  try {
    return create_frequency_table(x, target);
  } catch (std::exception& e) {
    Rcpp::stop("Error in OptimalBinningCheckDistinctsLength: %s", e.what());
  } catch (...) {
    Rcpp::stop("Unknown error in OptimalBinningCheckDistinctsLength");
  }
}