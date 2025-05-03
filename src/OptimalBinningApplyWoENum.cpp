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
//'   \item \code{id}: A numeric vector of bin IDs indicating the optimal order of the bins.
//' }
//' @param feature A numeric vector containing the original feature data to which WoE values will be applied.
//' @param include_upper_bound A logical value indicating whether the upper bound of the interval should be included (default is \code{TRUE}).
//'
//' @return A data frame with four columns:
//' \itemize{
//'   \item \code{feature}: Original feature values.
//'   \item \code{bin}: Optimal bins represented as interval notation.
//'   \item \code{woe}: Optimal WoE values corresponding to each feature value.
//'   \item \code{idbin}: ID of the bin to which each feature value belongs.
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
//'
//' @examples
//' \dontrun{
//' # Example usage with hypothetical obresults and feature vector
//' obresults <- list(
//'   cutpoints = c(1.5, 3.0, 4.5),
//'   woe = c(-0.2, 0.0, 0.2, 0.4),
//'   id = c(1, 2, 3, 4)  # IDs for each bin
//' )
//' feature <- c(1.0, 2.0, 3.5, 5.0)
//' result <- OBApplyWoENum(obresults, feature, include_upper_bound = TRUE)
//' print(result)
//' }
//'
//' @export
// [[Rcpp::export]]
DataFrame OBApplyWoENum(const List& obresults,
                                  const NumericVector& feature,
                                  bool include_upper_bound = true) {
 // Validação dos parâmetros de entrada, agora incluindo id
 if (!obresults.containsElementNamed("cutpoints")) {
    stop("The 'obresults' list must contain a 'cutpoints' element.");
 }
 if (!obresults.containsElementNamed("woe")) {
    stop("The 'obresults' list must contain a 'woe' element.");
 }
 if (!obresults.containsElementNamed("id")) {
    stop("The 'obresults' list must contain an 'id' element.");
 }
 
 // Extrair cutpoints, WoE values e IDs
 NumericVector cutpoints_nv = obresults["cutpoints"];
 NumericVector woe_nv = obresults["woe"];
 NumericVector id_nv = obresults["id"];  // Novo: extrair IDs das bins
 
 // Converter para std::vector para eficiência
 std::vector<double> cutpoints = as<std::vector<double>>(cutpoints_nv);
 std::vector<double> woe_values = as<std::vector<double>>(woe_nv);
 std::vector<double> bin_ids = as<std::vector<double>>(id_nv);  // Novo: converter IDs para vector
 
 // Garantir que os cutpoints estão ordenados de forma crescente
 if (!std::is_sorted(cutpoints.begin(), cutpoints.end())) {
    stop("Cutpoints must be sorted in ascending order.");
 }
 
 // Número de bins deve ser um a mais que o número de cutpoints
 size_t num_bins = cutpoints.size() + 1;
 if (woe_values.size() != num_bins || bin_ids.size() != num_bins) {  // Atualizado: verificar tamanho dos IDs
    stop("The number of WoE values and IDs must be equal to the number of bins (cutpoints + 1).");
 }
 
 // [O código para precompute intervals e bin labels permanece igual]
 // Define negative and positive infinity for clarity
 const double NEG_INF = -std::numeric_limits<double>::infinity();
 const double POS_INF = std::numeric_limits<double>::infinity();
 
 // Precompute intervals and bin labels
 std::vector<std::pair<double, double>> intervals(num_bins);
 std::vector<std::string> bin_labels(num_bins);
 
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
 
 // Preparar vetores de output, agora incluindo featureid
 size_t n = feature.size();
 NumericVector featurewoe(n);
 CharacterVector featurebins(n);
 NumericVector featureid(n);  // Novo: vetor para armazenar os IDs
 NumericVector feature_values = clone(feature);
 
 // Aplicar WoE values e IDs à feature
 for (size_t i = 0; i < n; ++i) {
    double x = feature[i];
    size_t bin_index;
    
    if (include_upper_bound) {
       // Use std::lower_bound para encontrar o índice do bin
       auto it = std::lower_bound(cutpoints.begin(), cutpoints.end(), x);
       bin_index = std::distance(cutpoints.begin(), it);
       
       // Ajustar para x > último cutpoint
       if (bin_index >= num_bins) {
          bin_index = num_bins - 1;
       }
    } else {
       // Use std::upper_bound para encontrar o índice do bin
       auto it = std::upper_bound(cutpoints.begin(), cutpoints.end(), x);
       bin_index = std::distance(cutpoints.begin(), it);
       
       // Ajustar para x >= último cutpoint
       if (bin_index >= num_bins) {
          bin_index = num_bins - 1;
       }
    }
    
    featurewoe[i] = woe_values[bin_index];
    featurebins[i] = bin_labels[bin_index];
    featureid[i] = bin_ids[bin_index];
 }
 
 // Construir o DataFrame de resultado, agora incluindo idbin
 DataFrame result = DataFrame::create(
    Named("feature") = feature_values,
    Named("bin") = featurebins,
    Named("woe") = featurewoe,
    Named("idbin") = featureid
 );
 
 return result;
}
