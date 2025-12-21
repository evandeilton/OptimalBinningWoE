// [[Rcpp::plugins(cpp11)]]
#include <Rcpp.h>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>
#include <sstream>
#include <unordered_set>
#include <cmath>

using namespace Rcpp;

// Include shared headers
#include "common/optimal_binning_common.h"
#include "common/bin_structures.h"

using namespace Rcpp;
using namespace OptimalBinning;


// [[Rcpp::export]]
DataFrame OBApplyWoENum(const List& obresults,
                       const NumericVector& feature,
                       bool include_upper_bound = true,
                       Nullable<NumericVector> missing_values = R_NilValue) {
 
 // Input validation for obresults components
 if (!obresults.containsElementNamed("cutpoints")) {
   stop("The 'obresults' list must contain a 'cutpoints' element.");
 }
 if (!obresults.containsElementNamed("woe")) {
   stop("The 'obresults' list must contain a 'woe' element.");
 }
 if (!obresults.containsElementNamed("id")) {
   stop("The 'obresults' list must contain an 'id' element.");
 }
 
 // Extract cutpoints, WoE values and IDs
 NumericVector cutpoints_nv = obresults["cutpoints"];
 NumericVector woe_nv = obresults["woe"];
 NumericVector id_nv = obresults["id"];
 
 // Check for NA values in critical components
 if (any(is_na(cutpoints_nv))) {
   stop("Cutpoints cannot contain NA values.");
 }
 if (any(is_na(woe_nv))) {
   stop("WoE values cannot contain NA values.");
 }
 if (any(is_na(id_nv))) {
   stop("ID values cannot contain NA values.");
 }
 
 // Convert to std::vector for efficiency
 std::vector<double> cutpoints = as<std::vector<double>>(cutpoints_nv);
 std::vector<double> woe_values = as<std::vector<double>>(woe_nv);
 std::vector<double> bin_ids = as<std::vector<double>>(id_nv);
 
 // Ensure cutpoints are sorted in ascending order
 if (!std::is_sorted(cutpoints.begin(), cutpoints.end())) {
   stop("Cutpoints must be sorted in ascending order.");
 }
 
 // Number of bins must be one more than the number of cutpoints
 size_t num_bins = cutpoints.size() + 1;
 if (woe_values.size() != num_bins || bin_ids.size() != num_bins) {
   stop("The number of WoE values and IDs must be equal to the number of bins (cutpoints + 1).");
 }
 
 // Calculate the ID for missing values (max ID + 1 or num_bins + 1)
 double missing_id = static_cast<double>(num_bins + 1);
 // Alternative: use max of existing IDs + 1
 // double max_id = *std::max_element(bin_ids.begin(), bin_ids.end());
 // double missing_id = max_id + 1;
 
 // Process missing values specification
 std::unordered_set<double> missing_set;
 if (missing_values.isNotNull()) {
   NumericVector mv = as<NumericVector>(missing_values);
   for (int i = 0; i < mv.size(); ++i) {
     if (!NumericVector::is_na(mv[i])) {
       missing_set.insert(mv[i]);
     }
   }
 } else {
   // Default missing value
   missing_set.insert(-999.0);
 }
 
 // Define negative and positive infinity for clarity
 const double NEG_INF = -std::numeric_limits<double>::infinity();
 const double POS_INF = std::numeric_limits<double>::infinity();
 
 // Precompute intervals and bin labels
 std::vector<std::pair<double, double>> intervals(num_bins);
 std::vector<std::string> bin_labels(num_bins);
 
 // Helper function to format numbers with appropriate precision
 auto format_number = [](double val) -> std::string {
   std::ostringstream oss;
   if (std::isinf(val)) {
     if (val < 0) return "-Inf";
     else return "+Inf";
   }
   oss.precision(6);
   oss << std::noshowpoint << val;
   return oss.str();
 };
 
 if (include_upper_bound) {
   // Include upper bound in intervals
   for (size_t i = 0; i < num_bins; ++i) {
     double lower, upper;
     if (i == 0) {
       lower = NEG_INF;
       upper = cutpoints[0];
     } else if (i == num_bins - 1) {
       lower = cutpoints[i - 1];
       upper = POS_INF;
     } else {
       lower = cutpoints[i - 1];
       upper = cutpoints[i];
     }
     intervals[i] = std::make_pair(lower, upper);
     
     // Create interval notation
     std::ostringstream oss;
     oss << "(" << format_number(lower) << ";" << format_number(upper) << "]";
     bin_labels[i] = oss.str();
   }
 } else {
   // Exclude upper bound in intervals
   for (size_t i = 0; i < num_bins; ++i) {
     double lower, upper;
     if (i == 0) {
       lower = NEG_INF;
       upper = cutpoints[0];
     } else if (i == num_bins - 1) {
       lower = cutpoints[i - 1];
       upper = POS_INF;
     } else {
       lower = cutpoints[i - 1];
       upper = cutpoints[i];
     }
     intervals[i] = std::make_pair(lower, upper);
     
     // Create interval notation
     std::ostringstream oss;
     oss << "[" << format_number(lower) << ";" << format_number(upper) << ")";
     bin_labels[i] = oss.str();
   }
 }
 
 // Prepare output vectors
 size_t n = feature.size();
 NumericVector featurewoe(n);
 CharacterVector featurebins(n);
 NumericVector featureid(n);
 IntegerVector ismissing(n);
 NumericVector feature_values = clone(feature);
 
 // Apply WoE values and IDs to feature
 for (size_t i = 0; i < n; ++i) {
   double x = feature[i];
   
   // Check if value is missing (NA or in missing_set)
   bool is_missing = NumericVector::is_na(x) || (missing_set.find(x) != missing_set.end());
   
   if (is_missing) {
     // Handle missing values with "Special" category
     featurewoe[i] = NA_REAL;
     featurebins[i] = "Special";  // Changed from NA_STRING to "Special"
     featureid[i] = missing_id;   // Changed from NA_REAL to missing_id
     ismissing[i] = 1;
   } else {
     // Process non-missing values
     size_t bin_index;
     
     if (include_upper_bound) {
       // Use std::upper_bound for (a, b] intervals
       auto it = std::upper_bound(cutpoints.begin(), cutpoints.end(), x);
       bin_index = std::distance(cutpoints.begin(), it);
       
       // Ensure bin_index is within valid range
       if (bin_index >= num_bins) {
         bin_index = num_bins - 1;
       }
     } else {
       // Use std::lower_bound for [a, b) intervals
       auto it = std::lower_bound(cutpoints.begin(), cutpoints.end(), x);
       bin_index = std::distance(cutpoints.begin(), it);
       
       // Adjust for the last bin
       if (bin_index >= num_bins) {
         bin_index = num_bins - 1;
       }
     }
     
     // Assign values
     featurewoe[i] = woe_values[bin_index];
     featurebins[i] = bin_labels[bin_index];
     featureid[i] = bin_ids[bin_index];
     ismissing[i] = 0;
   }
 }
 
 // Build the result DataFrame
 DataFrame result = DataFrame::create(
   Named("feature") = feature_values,
   Named("bin") = featurebins,
   Named("woe") = featurewoe,
   Named("idbin") = featureid,
   Named("ismissing") = ismissing
 );
 
 return result;
}









// // [[Rcpp::plugins(cpp11)]]
// #include <Rcpp.h>
// #include <vector>
// #include <string>
// #include <algorithm>
// #include <limits>
// #include <sstream>
// 
// using namespace Rcpp;

// Include shared headers
#include "common/optimal_binning_common.h"
#include "common/bin_structures.h"

using namespace Rcpp;
using namespace OptimalBinning;

// 
// //' @title Apply Optimal Weight of Evidence (WoE) to a Numerical Feature
// //'
// //' @description
// //' This function applies optimal Weight of Evidence (WoE) values to an original numerical feature based on the results from an optimal binning algorithm. It assigns each value in the feature to a bin according to the specified cutpoints and interval inclusion rule, and maps the corresponding WoE value to it.
// //'
// //' @param obresults A list containing the output from an optimal binning algorithm for numerical variables. It must include at least the following elements:
// //' \itemize{
// //'   \item \code{cutpoints}: A numeric vector of cutpoints used to define the bins.
// //'   \item \code{woe}: A numeric vector of WoE values corresponding to each bin.
// //'   \item \code{id}: A numeric vector of bin IDs indicating the optimal order of the bins.
// //' }
// //' @param feature A numeric vector containing the original feature data to which WoE values will be applied.
// //' @param include_upper_bound A logical value indicating whether the upper bound of the interval should be included (default is \code{TRUE}).
// //'
// //' @return A data frame with four columns:
// //' \itemize{
// //'   \item \code{feature}: Original feature values.
// //'   \item \code{bin}: Optimal bins represented as interval notation.
// //'   \item \code{woe}: Optimal WoE values corresponding to each feature value.
// //'   \item \code{idbin}: ID of the bin to which each feature value belongs.
// //' }
// //'
// //' @details
// //' The function assigns each value in \code{feature} to a bin based on the \code{cutpoints} and the \code{include_upper_bound} parameter. The intervals are defined mathematically as follows:
// //'
// //' Let \eqn{C = \{c_1, c_2, ..., c_n\}} be the set of cutpoints.
// //'
// //' If \code{include_upper_bound = TRUE}:
// //' \deqn{
// //' I_1 = (-\infty, c_1]
// //' }
// //' \deqn{
// //' I_i = (c_{i-1}, c_i], \quad \text{for } i = 2, ..., n
// //' }
// //' \deqn{
// //' I_{n+1} = (c_n, +\infty)
// //' }
// //'
// //' If \code{include_upper_bound = FALSE}:
// //' \deqn{
// //' I_1 = (-\infty, c_1)
// //' }
// //' \deqn{
// //' I_i = [c_{i-1}, c_i), \quad \text{for } i = 2, ..., n
// //' }
// //' \deqn{
// //' I_{n+1} = [c_n, +\infty)
// //' }
// //'
// //' The function uses efficient algorithms and data structures to handle large datasets. It implements binary search to assign bins, minimizing computational complexity.
// //'
// //'
// //' @examples
// //' \dontrun{
// //' # Example usage with hypothetical obresults and feature vector
// //' obresults <- list(
// //'   cutpoints = c(1.5, 3.0, 4.5),
// //'   woe = c(-0.2, 0.0, 0.2, 0.4),
// //'   id = c(1, 2, 3, 4)  # IDs for each bin
// //' )
// //' feature <- c(1.0, 2.0, 3.5, 5.0)
// //' result <- OBApplyWoENum(obresults, feature, include_upper_bound = TRUE)
// //' print(result)
// //' }
// //'
// //' @export