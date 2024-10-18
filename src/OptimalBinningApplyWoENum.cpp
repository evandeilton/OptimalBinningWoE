// [[Rcpp::plugins(cpp11)]]
#include <Rcpp.h>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>
#include <sstream>

using namespace Rcpp;

//' @title Apply Optimal Weight of Evidence (WoE) to a Numerical Feature
//'
//' @description
//' This function applies optimal Weight of Evidence (WoE) values to an original numerical feature based on the results from an optimal binning algorithm. It assigns each value in the feature to a bin according to the specified cutpoints and interval inclusion rule, and maps the corresponding WoE value to it.
//'
//' @param obresults A list containing the output from an optimal binning algorithm for numerical variables. It must include at least the following elements:
//' \itemize{
//'   \item \code{cutpoints}: A numeric vector of cutpoints used to define the bins.
//'   \item \code{woe}: A numeric vector of WoE values corresponding to each bin.
//' }
//' @param feature A numeric vector containing the original feature data to which WoE values will be applied.
//' @param include_upper_bound A logical value indicating whether the upper bound of the interval should be included (default is \code{TRUE}).
//'
//' @return A data frame with three columns:
//' \itemize{
//'   \item \code{feature}: Original feature values.
//'   \item \code{featurebins}: Optimal bins represented as interval notation.
//'   \item \code{featurewoe}: Optimal WoE values corresponding to each feature value.
//' }
//'
//' @details
//' The function assigns each value in \code{feature} to a bin based on the \code{cutpoints} and the \code{include_upper_bound} parameter. The intervals are defined mathematically as follows:
//'
//' Let \eqn{C = \{c_1, c_2, ..., c_n\}} be the set of cutpoints.
//'
//' If \code{include_upper_bound = TRUE}:
//' \deqn{
//' I_1 = (-\infty, c_1]
//' }
//' \deqn{
//' I_i = (c_{i-1}, c_i], \quad \text{for } i = 2, ..., n
//' }
//' \deqn{
//' I_{n+1} = (c_n, +\infty)
//' }
//'
//' If \code{include_upper_bound = FALSE}:
//' \deqn{
//' I_1 = (-\infty, c_1)
//' }
//' \deqn{
//' I_i = [c_{i-1}, c_i), \quad \text{for } i = 2, ..., n
//' }
//' \deqn{
//' I_{n+1} = [c_n, +\infty)
//' }
//'
//' The function uses efficient algorithms and data structures to handle large datasets. It implements binary search to assign bins, minimizing computational complexity.
//'
//' @examples
//' \dontrun{
//' # Example usage with hypothetical obresults and feature vector
//' obresults <- list(
//'   cutpoints = c(1.5, 3.0, 4.5),
//'   woe = c(-0.2, 0.0, 0.2, 0.4)
//' )
//' feature <- c(1.0, 2.0, 3.5, 5.0)
//' result <- OptimalBinningApplyWoENum(obresults, feature, include_upper_bound = TRUE)
//' print(result)
//' }
//'
//' @export
// [[Rcpp::export]]
DataFrame OptimalBinningApplyWoENum(const List& obresults,
                                const NumericVector& feature,
                                bool include_upper_bound = true) {
 // Validate input parameters
 if (!obresults.containsElementNamed("cutpoints")) {
   stop("The 'obresults' list must contain a 'cutpoints' element.");
 }
 if (!obresults.containsElementNamed("woe")) {
   stop("The 'obresults' list must contain a 'woe' element.");
 }
 
 // Extract cutpoints and WoE values
 NumericVector cutpoints_nv = obresults["cutpoints"];
 NumericVector woe_nv = obresults["woe"];
 
 // Convert to std::vector for efficiency
 std::vector<double> cutpoints = as<std::vector<double>>(cutpoints_nv);
 std::vector<double> woe_values = as<std::vector<double>>(woe_nv);
 
 // Ensure cutpoints are sorted in ascending order
 if (!std::is_sorted(cutpoints.begin(), cutpoints.end())) {
   stop("Cutpoints must be sorted in ascending order.");
 }
 
 // Number of bins should be one more than the number of cutpoints
 size_t num_bins = cutpoints.size() + 1;
 if (woe_values.size() != num_bins) {
   stop("The number of WoE values must be equal to the number of bins (cutpoints + 1).");
 }
 
 // Precompute intervals and bin labels
 std::vector<std::pair<double, double>> intervals(num_bins);
 std::vector<std::string> bin_labels(num_bins);
 
 // Define negative and positive infinity for clarity
 const double NEG_INF = -std::numeric_limits<double>::infinity();
 const double POS_INF = std::numeric_limits<double>::infinity();
 
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
     oss << "(";
     if (std::isinf(lower)) {
       oss << "-Inf";
     } else {
       oss << lower;
     }
     oss << ";";
     if (upper == POS_INF) {
       oss << "+Inf";
     } else {
       oss << upper;
     }
     oss << "]";
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
     oss << "[";
     if (std::isinf(lower)) {
       oss << "-Inf";
     } else {
       oss << lower;
     }
     oss << ";";
     if (upper == POS_INF) {
       oss << "+Inf";
     } else {
       oss << upper;
     }
     oss << ")";
     bin_labels[i] = oss.str();
   }
 }
 
 // Prepare output vectors
 size_t n = feature.size();
 NumericVector featurewoe(n);
 CharacterVector featurebins(n);
 NumericVector feature_values = clone(feature); // To preserve original feature values
 
 // Apply WoE values to feature
 for (size_t i = 0; i < n; ++i) {
   double x = feature[i];
   size_t bin_index;
   
   if (include_upper_bound) {
     // Use std::lower_bound to find the bin index
     auto it = std::lower_bound(cutpoints.begin(), cutpoints.end(), x);
     bin_index = std::distance(cutpoints.begin(), it);
     
     // Adjust for x > last cutpoint
     if (bin_index >= num_bins) {
       bin_index = num_bins - 1;
     }
   } else {
     // Use std::upper_bound to find the bin index
     auto it = std::upper_bound(cutpoints.begin(), cutpoints.end(), x);
     bin_index = std::distance(cutpoints.begin(), it);
     
     // Adjust for x >= last cutpoint
     if (bin_index >= num_bins) {
       bin_index = num_bins - 1;
     }
   }
   
   featurewoe[i] = woe_values[bin_index];
   featurebins[i] = bin_labels[bin_index];
 }
 
 // Construct the output DataFrame
 DataFrame result = DataFrame::create(
   Named("feature") = feature_values,
   Named("featurebins") = featurebins,
   Named("featurewoe") = featurewoe
 );
 
 return result;
}

 