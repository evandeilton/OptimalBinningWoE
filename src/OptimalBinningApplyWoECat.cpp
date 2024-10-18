// [[Rcpp::plugins(cpp11)]]
#include <Rcpp.h>
#include <unordered_map>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <cctype>

using namespace Rcpp;

// Helper function to trim whitespace from both ends of a string
std::string trim(const std::string& s) {
  auto start = s.begin();
  while (start != s.end() && std::isspace(*start)) {
    ++start;
  }
  
  auto end = s.end();
  do {
    --end;
  } while (std::distance(start, end) > 0 && std::isspace(*end));
  
  return std::string(start, end + 1);
}

// Helper function to split a string by a delimiter string
std::vector<std::string> split(const std::string& s, const std::string& delimiter) {
  std::vector<std::string> tokens;
  size_t pos = 0, prev = 0;
  while ((pos = s.find(delimiter, prev)) != std::string::npos) {
    tokens.push_back(s.substr(prev, pos - prev));
    prev = pos + delimiter.size();
  }
  tokens.push_back(s.substr(prev));
  return tokens;
}

//' @title Apply Optimal Weight of Evidence (WoE) to a Categorical Feature
//'
//' @description
//' This function applies optimal Weight of Evidence (WoE) values to an original categorical feature based on the results from an optimal binning algorithm. It assigns each category in the feature to its corresponding optimal bin and maps the associated WoE value.
//'
//' @param obresults A list containing the output from an optimal binning algorithm for categorical variables. It must include at least the following elements:
//' \itemize{
//'   \item \code{bins}: A character vector where each element represents a merged bin of categories, with categories separated by \code{bin_separator}.
//'   \item \code{woe}: A numeric vector of WoE values corresponding to each bin.
//' }
//' @param feature A character vector containing the original categorical feature data to which WoE values will be applied.
//' @param bin_separator A string representing the separator used in \code{bins} to separate categories within merged bins (default is \code{";"}).
//'
//' @return A data frame with three columns:
//' \itemize{
//'   \item \code{feature}: Original feature values.
//'   \item \code{featurebins}: Optimal merged bins to which each feature value belongs.
//'   \item \code{featurewoe}: Optimal WoE values corresponding to each feature value.
//' }
//'
//' @details
//' The function processes the \code{bins} from \code{obresults} by splitting each merged bin into individual categories using \code{bin_separator}. It then creates a mapping from each category to its corresponding bin index and WoE value.
//'
//' For each value in \code{feature}, the function assigns the appropriate bin and WoE value based on the category-to-bin mapping. If a category in \code{feature} is not found in any bin, \code{NA} is assigned to both \code{featurebins} and \code{featurewoe}.
//'
//' The function handles missing values (\code{NA}) in \code{feature} by assigning \code{NA} to both \code{featurebins} and \code{featurewoe} for those entries.
//'
//' @examples
//' \dontrun{
//' # Example usage with hypothetical obresults and feature vector
//' obresults <- list(
//'   bins = c("business;repairs;car (used);retraining",
//'            "car (new);furniture/equipment;domestic appliances;education;others",
//'            "radio/television"),
//'   woe = c(-0.2000211, 0.2892885, -0.4100628)
//' )
//' feature <- c("business", "education", "radio/television", "unknown_category")
//' result <- OptimalBinningApplyWoECat(obresults, feature, bin_separator = ";")
//' print(result)
//' }
//'
//' @export
// [[Rcpp::export]]
DataFrame OptimalBinningApplyWoECat(const List& obresults,
                                   const CharacterVector& feature,
                                   const std::string& bin_separator = "%;%") {
 // Validate input parameters
 if (!obresults.containsElementNamed("bins")) {
   stop("The 'obresults' list must contain a 'bins' element.");
 }
 if (!obresults.containsElementNamed("woe")) {
   stop("The 'obresults' list must contain a 'woe' element.");
 }
 if (bin_separator.empty()) {
   stop("The 'bin_separator' must be a non-empty string.");
 }
 
 // Extract bins and WoE values
 CharacterVector bins_cv = obresults["bins"];
 NumericVector woe_nv = obresults["woe"];
 
 // Convert to std::vector for efficiency
 std::vector<std::string> bin_labels = as<std::vector<std::string>>(bins_cv);
 std::vector<double> woe_values = as<std::vector<double>>(woe_nv);
 
 // Ensure bins and WoE values have the same size
 if (bin_labels.size() != woe_values.size()) {
   stop("The number of bins must match the number of WoE values.");
 }
 
 // Build category-to-bin mapping
 std::unordered_map<std::string, size_t> category_to_bin_index;
 
 for (size_t i = 0; i < bin_labels.size(); ++i) {
   std::string bin = bin_labels[i];
   // Split the bin into categories
   std::vector<std::string> categories = split(bin, bin_separator);
   for (auto& category : categories) {
     // Trim leading/trailing whitespace
     category = trim(category);
     // Check for duplicates
     auto it = category_to_bin_index.find(category);
     if (it != category_to_bin_index.end()) {
       stop("Category '" + category + "' appears in multiple bins.");
     }
     // Add to the mapping
     category_to_bin_index[category] = i;
   }
 }
 
 // Prepare output vectors
 size_t n = feature.size();
 CharacterVector featurebins(n, NA_STRING);
 NumericVector featurewoe(n, NA_REAL);
 CharacterVector feature_values = clone(feature); // To preserve original feature values
 
 // Apply WoE values to feature
 for (size_t i = 0; i < n; ++i) {
   if (CharacterVector::is_na(feature[i])) {
     // Assign NA to featurebins and featurewoe
     featurebins[i] = NA_STRING;
     featurewoe[i] = NA_REAL;
   } else {
     std::string category = as<std::string>(feature[i]);
     // Trim leading/trailing whitespace
     category = trim(category);
     // Look up the category in the mapping
     auto it = category_to_bin_index.find(category);
     if (it != category_to_bin_index.end()) {
       size_t bin_index = it->second;
       featurebins[i] = bin_labels[bin_index];
       featurewoe[i] = woe_values[bin_index];
     } else {
       // Category not found in any bin
       featurebins[i] = NA_STRING;
       featurewoe[i] = NA_REAL;
     }
   }
 }
 
 // Construct the output DataFrame
 DataFrame result = DataFrame::create(
   Named("feature") = feature_values,
   Named("bin") = featurebins,
   Named("woe") = featurewoe
 );
 
 return result;
}
