// [[Rcpp::plugins(cpp11)]]
#include <Rcpp.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <limits>

using namespace Rcpp;

// Include shared headers
#include "common/optimal_binning_common.h"
#include "common/bin_structures.h"

using namespace Rcpp;
using namespace OptimalBinning;


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

// [[Rcpp::export]]
DataFrame OBApplyWoECat(const List& obresults,
                       const CharacterVector& feature,
                       const std::string& bin_separator = "%;%",
                       Nullable<CharacterVector> missing_values = R_NilValue) {
 
 // Validate input parameters
 if (!obresults.containsElementNamed("bin")) {
   stop("The 'obresults' list must contain a 'bin' element.");
 }
 if (!obresults.containsElementNamed("woe")) {
   stop("The 'obresults' list must contain a 'woe' element.");
 }
 if (!obresults.containsElementNamed("id")) {
   stop("The 'obresults' list must contain an 'id' element.");
 }
 if (bin_separator.empty()) {
   stop("The 'bin_separator' must be a non-empty string.");
 }
 
 // Extract bins, WoE values, and IDs
 CharacterVector bins_cv = obresults["bin"];
 NumericVector woe_nv = obresults["woe"];
 NumericVector id_nv = obresults["id"];
 
 // Check for NA values in critical components
 if (any(is_na(woe_nv))) {
   stop("WoE values cannot contain NA values.");
 }
 if (any(is_na(id_nv))) {
   stop("ID values cannot contain NA values.");
 }
 
 // Convert to std::vector for efficiency
 std::vector<std::string> bin_labels = as<std::vector<std::string>>(bins_cv);
 std::vector<double> woe_values = as<std::vector<double>>(woe_nv);
 std::vector<double> bin_ids = as<std::vector<double>>(id_nv);
 
 // Ensure bins, WoE values, and IDs have the same size
 size_t num_bins = bin_labels.size();
 if (woe_values.size() != num_bins || bin_ids.size() != num_bins) {
   stop("The number of bins must match the number of WoE values and IDs.");
 }
 
 // Calculate the ID for missing values (number of bins + 1)
 double missing_id = static_cast<double>(num_bins + 1);
 
 // Process missing values specification
 std::unordered_set<std::string> missing_set;
 if (missing_values.isNotNull()) {
   CharacterVector mv = as<CharacterVector>(missing_values);
   for (int i = 0; i < mv.size(); ++i) {
     if (!CharacterVector::is_na(mv[i])) {
       missing_set.insert(as<std::string>(mv[i]));
     }
   }
 } else {
   // Default missing values
   missing_set.insert("NA");
   missing_set.insert("Missing");
   missing_set.insert("");
 }
 
 // Build category-to-bin mapping
 std::unordered_map<std::string, size_t> category_to_bin_index;
 
 for (size_t i = 0; i < num_bins; ++i) {
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
 CharacterVector featurebins(n);
 NumericVector featurewoe(n);
 NumericVector featureid(n);
 IntegerVector ismissing(n);
 CharacterVector feature_values = clone(feature); // To preserve original feature values
 
 // Apply WoE values and IDs to feature
 for (size_t i = 0; i < n; ++i) {
   // Check if value is NA first
   if (CharacterVector::is_na(feature[i])) {
     // Handle NA values
     featurebins[i] = "Special";
     featurewoe[i] = NA_REAL;
     featureid[i] = missing_id;
     ismissing[i] = 1;
   } else {
     std::string category = as<std::string>(feature[i]);
     // Trim leading/trailing whitespace
     category = trim(category);
     
     // Check if value is in missing set
     bool is_missing = (missing_set.find(category) != missing_set.end());
     
     if (is_missing) {
       // Handle missing values with "Special" category
       featurebins[i] = "Special";
       featurewoe[i] = NA_REAL;
       featureid[i] = missing_id;
       ismissing[i] = 1;
     } else {
       // Look up the category in the mapping
       auto it = category_to_bin_index.find(category);
       if (it != category_to_bin_index.end()) {
         // Category found in mapping
         size_t bin_index = it->second;
         featurebins[i] = bin_labels[bin_index];
         featurewoe[i] = woe_values[bin_index];
         featureid[i] = bin_ids[bin_index];
         ismissing[i] = 0;
       } else {
         // Category not found in any bin - treat as unknown/special
         featurebins[i] = "Special";
         featurewoe[i] = NA_REAL;
         featureid[i] = missing_id;
         ismissing[i] = 1;
         
         // Optional: Issue a warning for unknown categories
         // Rcpp::warning("Unknown category found: " + category);
       }
     }
   }
 }
 
 // Construct the output DataFrame
 DataFrame result = DataFrame::create(
   Named("feature") = feature_values,
   Named("bin") = featurebins,
   Named("woe") = featurewoe,
   Named("idbin") = featureid,
   Named("ismissing") = ismissing
 );
 
 return result;
}


