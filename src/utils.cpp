#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <map>

using namespace Rcpp;

// Função auxiliar: calcula divergência KL
// KL(p||q) = p * log(p/q)
// Observações:
// - Se p = 0, o termo não contribui para a KL (retorna 0).
// - Se q = 0 e p > 0, a KL tende a infinito (retornamos R_PosInf).
// - p e q devem representar probabilidades (0 <= p,q <= 1).
double kl_divergence(double p, double q) {
  if (p == 0.0) return 0.0;
  if (q == 0.0) return R_PosInf;
  return p * std::log(p / q);
}

//' @title Generate a Detailed Gains Table from Optimal Binning Results
//'
//' @description
//' This function processes the results of optimal binning and generates a comprehensive gains table,
//' including evaluation metrics and characteristics for each bin. It provides insights into the
//' performance and information value of the binned feature within the context of binary classification models.
//'
//' @param binning_result A list containing the binning results, which must include
//' a DataFrame with the following columns:
//' \itemize{
//'   \item \code{id}: Numeric bin identifier.
//'   \item \code{bin}: Bin label where feature values were grouped.
//'   \item \code{count}: Total count of observations in the bin.
//'   \item \code{count_pos}: Count of positive cases (target=1) in the bin.
//'   \item \code{count_neg}: Count of negative cases (target=0) in the bin.
//' }
//'
//' @return A DataFrame containing, for each bin, a detailed breakdown of metrics and characteristics. 
//' Columns include:
//' \itemize{
//'   \item \code{id}: Numeric identifier of the bin.
//'   \item \code{bin}: Label of the bin.
//'   \item \code{count}: Total observations in the bin.
//'   \item \code{pos}: Number of positive cases in the bin.
//'   \item \code{neg}: Number of negative cases in the bin.
//'   \item \code{woe}: Weight of Evidence (\eqn{WoE_i = \ln\frac{P(X_i|Y=1)}{P(X_i|Y=0)}}).
//'   \item \code{iv}: Information Value contribution for the bin (\eqn{IV_i = (P(X_i|Y=1) - P(X_i|Y=0)) \cdot WoE_i}).
//'   \item \code{total_iv}: Total IV across all bins.
//'   \item \code{cum_pos}, \code{cum_neg}: Cumulative counts of positives and negatives up to the current bin.
//'   \item \code{pos_rate}, \code{neg_rate}: Positive and negative rates within the bin.
//'   \item \code{pos_perc}, \code{neg_perc}: Percentage of total positives/negatives represented by the bin.
//'   \item \code{count_perc}, \code{cum_count_perc}: Percentage of total observations and cumulative percentages.
//'   \item \code{cum_pos_perc}, \code{cum_neg_perc}: Cumulative percentages of positives and negatives relative to their totals.
//'   \item \code{cum_pos_perc_total}, \code{cum_neg_perc_total}: Cumulative percentages of positives and negatives relative to total observations.
//'   \item \code{odds_pos}: Odds of positives in the bin (\eqn{\frac{pos}{neg}}).
//'   \item \code{odds_ratio}: Ratio of bin odds to total odds (\eqn{OR_i = \frac{(P(Y=1|X_i)/P(Y=0|X_i))}{(P(Y=1)/P(Y=0))}}).
//'   \item \code{lift}: Lift of the bin (\eqn{Lift_i = \frac{P(Y=1|X_i)}{P(Y=1)}}).
//'   \item \code{ks}: Kolmogorov-Smirnov statistic (\eqn{KS_i = |F_1(i) - F_0(i)|}).
//'   \item \code{gini_contribution}: Contribution to the Gini index (\eqn{Gini_i = P(X_i|Y=1)F_0(i) - P(X_i|Y=0)F_1(i)}).
//'   \item \code{precision}: Precision for the bin (\eqn{Precision_i = \frac{TP}{TP + FP}}).
//'   \item \code{recall}: Recall for the bin (\eqn{Recall_i = \frac{\sum_{j=1}^i TP_j}{\sum_{j=1}^n TP_j}}).
//'   \item \code{f1_score}: F1 Score (\eqn{F1_i = 2 \cdot \frac{Precision_i \cdot Recall_i}{Precision_i + Recall_i}}).
//'   \item \code{log_likelihood}: Log-Likelihood (\eqn{LL_i = n_{1i}\ln(p_i) + n_{0i}\ln(1-p_i)}).
//'   \item \code{kl_divergence}: Kullback-Leibler divergence (\eqn{KL_i = p_i \ln\frac{p_i}{p} + (1-p_i)\ln\frac{1-p_i}{1-p}}).
//'   \item \code{js_divergence}: Jensen-Shannon divergence (\eqn{JS_i = \frac{1}{2}KL(P||M) + \frac{1}{2}KL(Q||M)}).
//' }
//'
//' @details
//' This function organizes the bins and computes essential metrics that help evaluate the quality of optimal binning
//' applied to a binary classification problem. These metrics include measures of separation, information gain, 
//' and performance lift, aiding in model performance analysis.
//'
//' @references
//' \itemize{
//'   \item Siddiqi, N. (2006). Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring. John Wiley & Sons.
//'   \item Hand, D. J., & Till, R. J. (2001). A Simple Generalisation of the Area Under the ROC Curve for Multiple Class Classification Problems. Machine Learning, 45(2), 171-186.
//'   \item Kullback, S., & Leibler, R. A. (1951). On Information and Sufficiency. The Annals of Mathematical Statistics, 22(1), 79-86.
//'   \item Lin, J. (1991). Divergence measures based on the Shannon entropy. IEEE Transactions on Information Theory, 37(1), 145-151.
//' }
//'
//' @examples
//' \dontrun{
//' # Generate optimal binning results
//' binning_result <- OptimalBinning(target, feature)
//' # Create a gains table
//' gains_table <- OptimalBinningGainsTable(binning_result)
//' print(gains_table)
//' }
//'
//' @export
// [[Rcpp::export]]
DataFrame OptimalBinningGainsTable(List binning_result) {
 // Extrair dados do resultado do binning
 NumericVector bin_ids = binning_result["id"];
 CharacterVector bin_labels = binning_result["bin"];
 NumericVector counts = binning_result["count"];
 NumericVector count_pos = binning_result["count_pos"];
 NumericVector count_neg = binning_result["count_neg"];
 
 // Criar vetor de índices para ordenação
 IntegerVector indices = seq_len(bin_ids.length()) - 1;
 
 // Ordenar índices baseado no ID
 std::sort(indices.begin(), indices.end(),
           [&bin_ids](int i, int j) { return bin_ids[i] < bin_ids[j]; });
 
 // Reordenar todos os vetores
 bin_ids = bin_ids[indices];
 bin_labels = bin_labels[indices];
 counts = counts[indices];
 count_pos = count_pos[indices];
 count_neg = count_neg[indices];
 
 int total_count = sum(counts);
 int total_pos = sum(count_pos);
 int total_neg = sum(count_neg);
 double overall_pos_rate = (double)total_pos / (double)total_count;
 
 int n = bin_labels.size();
 
 NumericVector count_perc(n), cum_count_perc(n);
 NumericVector pos_rate(n), neg_rate(n);
 NumericVector pos_perc(n), neg_perc(n);
 NumericVector cum_pos(n), cum_neg(n);
 NumericVector cum_pos_perc(n), cum_neg_perc(n);
 NumericVector cum_pos_perc_total(n), cum_neg_perc_total(n);
 NumericVector ks(n), woe(n), iv(n);
 NumericVector odds_pos(n), odds_ratio(n), lift(n), gini_contribution(n);
 NumericVector precision(n), recall(n), f1_score(n);
 NumericVector log_likelihood(n);
 NumericVector kl_divergence_metric(n), js_divergence_metric(n);
 
 double total_iv = 0.0;
 double cum_pos_sum = 0.0, cum_neg_sum = 0.0;
 
 for (int i = 0; i < n; i++) {
   // Percentuais de contagem
   count_perc[i] = (double)counts[i] / (double)total_count;
   cum_count_perc[i] = (i == 0) ? count_perc[i] : cum_count_perc[i - 1] + count_perc[i];
   
   // Taxas de positivos e negativos no bin
   pos_rate[i] = (double)count_pos[i] / (double)counts[i];
   neg_rate[i] = (double)count_neg[i] / (double)counts[i];
   
   // Percentuais relativos ao total de pos/neg
   pos_perc[i] = (double)count_pos[i] / (double)total_pos;
   neg_perc[i] = (double)count_neg[i] / (double)total_neg;
   
   // Cumulativos
   cum_pos_sum += count_pos[i];
   cum_neg_sum += count_neg[i];
   cum_pos[i] = cum_pos_sum;
   cum_neg[i] = cum_neg_sum;
   
   cum_pos_perc[i] = cum_pos[i] / (double)total_pos;
   cum_neg_perc[i] = cum_neg[i] / (double)total_neg;
   
   cum_pos_perc_total[i] = cum_pos[i] / (double)total_count;
   cum_neg_perc_total[i] = cum_neg[i] / (double)total_count;
   
   // KS
   ks[i] = std::abs(cum_pos_perc[i] - cum_neg_perc[i]);
   
   // WoE
   if (count_pos[i] == 0 || total_pos == 0 || count_neg[i] == 0 || total_neg == 0) {
     woe[i] = 0.0;
   } else {
     woe[i] = std::log(((double)count_pos[i] * (double)total_neg) /
       ((double)count_neg[i] * (double)total_pos));
   }
   
   // IV
   iv[i] = (pos_perc[i] - neg_perc[i]) * woe[i];
   total_iv += iv[i];
   
   // Odds e Odds Ratio
   odds_pos[i] = (count_neg[i] == 0) ? R_PosInf : (double)count_pos[i] / (double)count_neg[i];
   double total_odds = (double)total_pos / (double)total_neg;
   odds_ratio[i] = odds_pos[i] / total_odds;
   
   // Lift
   lift[i] = pos_rate[i] / overall_pos_rate;
   
   // Gini Contribution
   // Gini_i = P(X_i|Y=1)*F_0(i) - P(X_i|Y=0)*F_1(i)
   gini_contribution[i] = pos_perc[i] * cum_neg_perc[i] - neg_perc[i] * cum_pos_perc[i];
   
   // Precision e Recall (TP = count_pos[i], FP = count_neg[i])
   precision[i] = (double)count_pos[i] / (double)counts[i];
   recall[i] = cum_pos_perc[i]; // cumulativo de positivos identificados
   f1_score[i] = (precision[i] + recall[i] == 0.0) ? 0.0 : 
     2.0 * (precision[i] * recall[i]) / (precision[i] + recall[i]);
   
   // Log-Likelihood: LL_i = n_{1i}ln(p_i) + n_{0i}ln(1-p_i)
   // Aqui p_i = pos_rate[i], 1-p_i = neg_rate[i]
   // neg_rate[i] = 1 - pos_rate[i]
   if (pos_rate[i] > 0.0 && neg_rate[i] > 0.0) {
     log_likelihood[i] = count_pos[i] * std::log(pos_rate[i]) + count_neg[i] * std::log(neg_rate[i]);
   } else {
     // Caso limite onde pos_rate[i] ou neg_rate[i] = 0
     // Evita log(0)
     log_likelihood[i] = NA_REAL;
   }
   
   // KL Divergence: KL_i = p_i ln(p_i/p) + (1 - p_i) ln((1 - p_i)/(1 - p))
   double kl_pos = kl_divergence(pos_rate[i], overall_pos_rate);
   double kl_neg_val = kl_divergence(neg_rate[i], 1.0 - overall_pos_rate);
   kl_divergence_metric[i] = kl_pos + kl_neg_val;
   
   // Jensen-Shannon Divergence
   // JS = 1/2 * KL(P||M) + 1/2 * KL(Q||M)
   // P=(p_i, 1-p_i), Q=(p, 1-p)
   // M=(P+Q)/2 = ((p_i+p)/2, (1 - p_i + 1 - p)/2)
   double m_pos = (pos_rate[i] + overall_pos_rate) / 2.0;
   double m_neg = (neg_rate[i] + (1.0 - overall_pos_rate)) / 2.0;
   
   // KL(P||M)
   double kl_p_m = kl_divergence(pos_rate[i], m_pos) + kl_divergence(neg_rate[i], m_neg);
   // KL(Q||M)
   double kl_q_m = kl_divergence(overall_pos_rate, m_pos) + kl_divergence((1.0 - overall_pos_rate), m_neg);
   
   js_divergence_metric[i] = 0.5 * (kl_p_m + kl_q_m);
 }
 
 return DataFrame::create(
   Named("id") = bin_ids,           
   Named("bin") = bin_labels,
   Named("count") = counts,
   Named("pos") = count_pos,
   Named("neg") = count_neg,
   Named("woe") = woe,
   Named("iv") = iv,
   Named("total_iv") = rep(total_iv, n),
   Named("cum_pos") = cum_pos,
   Named("cum_neg") = cum_neg,
   Named("pos_rate") = pos_rate,
   Named("neg_rate") = neg_rate,
   Named("pos_perc") = pos_perc,
   Named("neg_perc") = neg_perc,
   Named("count_perc") = count_perc,
   Named("cum_count_perc") = cum_count_perc,
   Named("cum_pos_perc") = cum_pos_perc,
   Named("cum_neg_perc") = cum_neg_perc,
   Named("cum_pos_perc_total") = cum_pos_perc_total,
   Named("cum_neg_perc_total") = cum_neg_perc_total,
   Named("odds_pos") = odds_pos,
   Named("odds_ratio") = odds_ratio,
   Named("lift") = lift,
   Named("ks") = ks,
   Named("gini_contribution") = gini_contribution,
   Named("precision") = precision,
   Named("recall") = recall,
   Named("f1_score") = f1_score,
   Named("log_likelihood") = log_likelihood,
   Named("kl_divergence") = kl_divergence_metric,
   Named("js_divergence") = js_divergence_metric
 );
}


//' @title Generate Gains Table for a Binned Feature
//'
//' @description
//' This function computes various statistical and performance metrics for a feature that has already been binned,
//' considering a binary target (0/1). It is useful for evaluating the quality of bins generated by
//' optimal binning methods. The calculated metrics include Weight of Evidence (WoE), Information Value (IV),
//' accuracy rates, information divergences, Kolmogorov-Smirnov (KS), Lift, and others.
//'
//' @param binned_df A DataFrame containing the following columns, resulting from a binning process (e.g., using `OptimalBinningApplyWoENum` or `OptimalBinningApplyWoECat`):
//' \itemize{
//'   \item \code{feature}: Original values of the variable.
//'   \item \code{bin}: Bin label where the feature value was classified.
//'   \item \code{woe}: Weight of Evidence associated with the bin.
//'   \item \code{idbin}: Numeric bin identifier used to optimally order the bins.
//' }
//' @param target A numeric binary vector (0 and 1) representing the target. It must have the same length as \code{binned_df}.
//' @param group_var A string indicating which variable to use for grouping data and calculating metrics.
//' Options: \code{"bin"}, \code{"woe"}, or \code{"idbin"}. Default: \code{"idbin"}.
//'
//' @return A DataFrame containing, for each group (bin) defined by \code{group_var}, the following columns:
//' \itemize{
//'   \item \code{group}: Name or value of the group selected by \code{group_var}.
//'   \item \code{id}: Numeric bin identifier, ordered.
//'   \item \code{count}: Total count of observations in the group.
//'   \item \code{pos}: Count of positive cases (target=1) in the group.
//'   \item \code{neg}: Count of negative cases (target=0) in the group.
//'   \item \code{woe}: Weight of Evidence for the group, calculated as \eqn{WoE = ln\frac{P(X|Y=1)}{P(X|Y=0)}}.
//'   \item \code{iv}: Contribution of the group to the Information Value: \eqn{IV = (P(X|Y=1)-P(X|Y=0))*WoE}.
//'   \item \code{total_iv}: Total IV value, sum of the IV from all groups.
//'   \item \code{cum_pos}, \code{cum_neg}: Cumulative counts of positive and negative cases up to the current group.
//'   \item \code{pos_rate}, \code{neg_rate}: Positive and negative rates within the group.
//'   \item \code{pos_perc}, \code{neg_perc}: Percentage of total positives/negatives represented by the group.
//'   \item \code{count_perc}, \code{cum_count_perc}: Percentage of total observations and their cumulative percentage.
//'   \item \code{cum_pos_perc}, \code{cum_neg_perc}: Cumulative percentage of positives/negatives relative to the total positives/negatives.
//'   \item \code{cum_pos_perc_total}, \code{cum_neg_perc_total}: Cumulative percentage of positives/negatives relative to total observations.
//'   \item \code{odds_pos}: Odds of positives in the group (\eqn{\frac{pos}{neg}}).
//'   \item \code{odds_ratio}: Ratio of group odds to overall odds (\eqn{odds_{group}/odds_{total}}).
//'   \item \code{lift}: \eqn{\frac{P(Y=1|X_{group})}{P(Y=1)}}.
//'   \item \code{ks}: Kolmogorov-Smirnov statistic at the group level: \eqn{|F_1(i)-F_0(i)|}.
//'   \item \code{gini_contribution}: Contribution of the bin to the Gini index, given by \eqn{P(X|Y=1)*F_0(i) - P(X|Y=0)*F_1(i)}.
//'   \item \code{precision}: Precision for the group (\eqn{\frac{TP}{TP+FP}}), where TP = pos, FP = neg, considering the bin as "positive".
//'   \item \code{recall}: \eqn{\frac{\sum_{j=1}^i TP_j}{\sum_{j=1}^n TP_j}}, cumulative percentage of true positives.
//'   \item \code{f1_score}: \eqn{2 * \frac{Precision * Recall}{Precision + Recall}}.
//'   \item \code{log_likelihood}: Log-likelihood for the group \eqn{LL = n_{pos} ln(p_i) + n_{neg} ln(1-p_i)}, with \eqn{p_i = pos_rate}.
//'   \item \code{kl_divergence}: Kullback-Leibler divergence between the group distribution and the global distribution of positives.
//'   \item \code{js_divergence}: Jensen-Shannon divergence between the group and global distributions, a symmetric and finite measure.
//' }
//'
//' @details
//' The function organizes the bins defined by \code{group_var} and computes essential performance metrics
//' for the applied binning in a binary classification model. These metrics assist in evaluating the
//' bins' discrimination capability to separate positives from negatives and the information added by each bin to the model.
//'
//' @references
//' \itemize{
//'   \item Siddiqi, N. (2006). Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring. John Wiley & Sons.
//'   \item Hand, D. J., & Till, R. J. (2001). A Simple Generalisation of the Area Under the ROC Curve for Multiple Class Classification Problems. Machine Learning, 45(2), 171-186.
//'   \item Kullback, S., & Leibler, R. A. (1951). On Information and Sufficiency. The Annals of Mathematical Statistics, 22(1), 79-86.
//'   \item Lin, J. (1991). Divergence measures based on the Shannon entropy. IEEE Transactions on Information Theory, 37(1), 145-151.
//' }
//'
//' @examples
//' \dontrun{
//' # Hypothetical example:
//' # Assume binned_df is the result of OptimalBinningApplyWoENum(...) and target is a 0/1 vector.
//' # gains_table <- OptimalBinningGainsTableFeature(binned_df, target, group_var = "idbin")
//' # print(gains_table)
//' }
//'
//' @export
// [[Rcpp::export]]
DataFrame OptimalBinningGainsTableFeature(DataFrame binned_df, 
                                          NumericVector target,
                                          std::string group_var = "bin") {
  // Validação dos parâmetros de entrada
  if ((int)target.size() != binned_df.nrows()) {
    stop("binned_df and target must have the same length");
  }
  
  // Verificação do target binário
  std::set<double> unique_targets(target.begin(), target.end());
  if (unique_targets.size() != 2 || !unique_targets.count(0) || !unique_targets.count(1)) {
    stop("target must contain only binary values (0 and 1)");
  }
  
  // Verificar se as colunas necessárias existem
  if (!binned_df.containsElementNamed("feature") ||
      !binned_df.containsElementNamed("bin") ||
      !binned_df.containsElementNamed("woe") ||
      !binned_df.containsElementNamed("idbin")) {
      stop("binned_df must contain columns: feature, bin, woe, and idbin");
  }
  
  // Verificar se group_var é válida
  if (group_var != "bin" && group_var != "woe" && group_var != "idbin") {
    stop("group_var must be one of: 'bin', 'woe', or 'idbin'");
  }
  
  CharacterVector feature_bins = binned_df["bin"];
  NumericVector feature_woes = binned_df["woe"];
  NumericVector feature_ids = binned_df["idbin"];
  
  // Map para contagens por grupo
  std::map<std::string, std::pair<int,int>> group_counts; 
  std::map<std::string, double> group_id_map; 
  // Armazenamos apenas para ordenação. Não precisamos mais do WoE original para IV, mas
  // precisamos do id para ordenação.
  
  for (int i = 0; i < binned_df.nrows(); ++i) {
    // Determinar chave do grupo como string
    std::string group_key;
    if (group_var == "bin") {
      group_key = Rcpp::as<std::string>(feature_bins[i]);
    } else if (group_var == "woe") {
      // converter woe para string
      group_key = std::to_string((double)feature_woes[i]);
    } else { // "idbin"
      // converter id para string
      group_key = std::to_string((double)feature_ids[i]);
    }
    
    group_id_map[group_key] = (double)feature_ids[i];
    
    if (target[i] == 1.0) {
      group_counts[group_key].first += 1;  // pos
    } else {
      group_counts[group_key].second += 1; // neg
    }
  }
  
  // Ordenar grupos pelo id (para manter coerência)
  std::vector<std::pair<double,std::string>> ordered_groups;
  for (const auto &entry : group_counts) {
    ordered_groups.push_back({group_id_map[entry.first], entry.first});
  }
  std::sort(ordered_groups.begin(), ordered_groups.end(),
            [](const std::pair<double,std::string> &a, const std::pair<double,std::string> &b){
              return a.first < b.first;
            });
  
  int n_bins = (int)ordered_groups.size();
  
  // Vetores de saída
  CharacterVector bin_labels(n_bins); // Sempre string
  NumericVector group_ids(n_bins);
  NumericVector counts(n_bins);
  NumericVector count_pos(n_bins);
  NumericVector count_neg(n_bins);
  NumericVector count_perc(n_bins);
  NumericVector cum_count_perc(n_bins);
  NumericVector pos_rate(n_bins);
  NumericVector neg_rate(n_bins);
  NumericVector pos_perc(n_bins);
  NumericVector neg_perc(n_bins);
  NumericVector cum_pos(n_bins);
  NumericVector cum_neg(n_bins);
  NumericVector cum_pos_perc(n_bins);
  NumericVector cum_neg_perc(n_bins);
  NumericVector cum_pos_perc_total(n_bins);
  NumericVector cum_neg_perc_total(n_bins);
  NumericVector ks(n_bins);
  NumericVector woe(n_bins);
  NumericVector iv(n_bins);
  NumericVector odds_pos(n_bins);
  NumericVector odds_ratio(n_bins);
  NumericVector lift(n_bins);
  NumericVector gini_contribution(n_bins);
  NumericVector precision(n_bins);
  NumericVector recall(n_bins);
  NumericVector f1_score(n_bins);
  NumericVector log_likelihood(n_bins);
  NumericVector kl_divergence_metric(n_bins);
  NumericVector js_divergence_metric(n_bins);
  
  int total_count = 0;
  int total_pos = 0;
  int total_neg = 0;
  double total_iv = 0.0;
  double cum_pos_sum = 0.0;
  double cum_neg_sum = 0.0;
  
  // Preenchendo contagens
  for (int i = 0; i < n_bins; ++i) {
    std::string gkey = ordered_groups[i].second;
    bin_labels[i] = gkey; // Já é string
    group_ids[i] = ordered_groups[i].first;
    
    count_pos[i] = group_counts[gkey].first;
    count_neg[i] = group_counts[gkey].second;
    counts[i] = count_pos[i] + count_neg[i];
    
    total_pos += count_pos[i];
    total_neg += count_neg[i];
    total_count += (int)counts[i];
  }
  
  double overall_pos_rate = (double)total_pos / (double)total_count;
  
  // Cálculo das métricas
  for (int i = 0; i < n_bins; ++i) {
    count_perc[i] = (double)counts[i] / (double)total_count;
    cum_count_perc[i] = (i == 0) ? count_perc[i] : (cum_count_perc[i - 1] + count_perc[i]);
    
    pos_rate[i] = (counts[i] > 0) ? ((double)count_pos[i] / (double)counts[i]) : 0.0;
    neg_rate[i] = (counts[i] > 0) ? ((double)count_neg[i] / (double)counts[i]) : 0.0;
    
    pos_perc[i] = (double)count_pos[i] / (double)total_pos; // P(X|Y=1)
    neg_perc[i] = (double)count_neg[i] / (double)total_neg; // P(X|Y=0)
    
    cum_pos_sum += (double)count_pos[i];
    cum_neg_sum += (double)count_neg[i];
    cum_pos[i] = cum_pos_sum;
    cum_neg[i] = cum_neg_sum;
    
    cum_pos_perc[i] = (double)cum_pos[i] / (double)total_pos;
    cum_neg_perc[i] = (double)cum_neg[i] / (double)total_neg;
    
    cum_pos_perc_total[i] = (double)cum_pos[i] / (double)total_count;
    cum_neg_perc_total[i] = (double)cum_neg[i] / (double)total_count;
    
    ks[i] = std::abs(cum_pos_perc[i] - cum_neg_perc[i]);
    
    // Recalcular WoE e IV
    if (pos_perc[i] > 0.0 && neg_perc[i] > 0.0) {
      woe[i] = std::log(pos_perc[i]/neg_perc[i]);
      iv[i] = (pos_perc[i] - neg_perc[i]) * woe[i];
    } else {
      woe[i] = 0.0;
      iv[i] = 0.0;
    }
    total_iv += iv[i];
    
    odds_pos[i] = (count_neg[i] > 0) ? ((double)count_pos[i]/(double)count_neg[i]) : R_PosInf;
    double total_odds = (double)total_pos/(double)total_neg;
    odds_ratio[i] = (total_odds > 0.0) ? (odds_pos[i]/total_odds) : NA_REAL;
    
    lift[i] = (overall_pos_rate > 0.0) ? (pos_rate[i]/overall_pos_rate) : NA_REAL;
    
    gini_contribution[i] = (pos_perc[i]*cum_neg_perc[i]) - (neg_perc[i]*cum_pos_perc[i]);
    
    precision[i] = (counts[i] > 0) ? ((double)count_pos[i]/(double)counts[i]) : 0.0;
    recall[i] = (total_pos > 0) ? ((double)cum_pos[i]/(double)total_pos) : 0.0;
    
    f1_score[i] = (precision[i] + recall[i] > 0.0) ? 
    (2.0 * precision[i]*recall[i])/(precision[i] + recall[i]) : 0.0;
    
    log_likelihood[i] = (count_pos[i] > 0 ? (double)count_pos[i]*std::log(pos_rate[i] + 1e-10) : 0.0) +
      (count_neg[i] > 0 ? (double)count_neg[i]*std::log(neg_rate[i] + 1e-10) : 0.0);
    
    double kl_pos = kl_divergence(pos_rate[i], overall_pos_rate);
    double kl_neg = kl_divergence(neg_rate[i], 1.0 - overall_pos_rate);
    kl_divergence_metric[i] = kl_pos + kl_neg;
    
    double m_pos = (pos_rate[i] + overall_pos_rate)/2.0;
    double m_neg = (neg_rate[i] + (1.0 - overall_pos_rate))/2.0;
    js_divergence_metric[i] = (kl_divergence(pos_rate[i], m_pos) +
      kl_divergence(neg_rate[i], m_neg))/2.0;
  }
  
  // Conforme solicitado:
  // Esse bloco deve ter sempre o nome "bin" (Named("bin") = ...)
  // porém os valores mudam conforme group_var. Já temos bin_labels como string.
  // Precisamos garantir que bin_labels seja string representando o valor do group_var.
  //
  // bin_labels já tem o valor do group_key, que dependia de group_var:
  // - Se group_var = "bin", já é o nome do bin (string).
  // - Se group_var = "woe" ou "idbin", bin_labels[i] é a string do valor numérico do woe ou idbin.
  //
  // Temos a garantia que bin_labels[i] é sempre string.
  // Logo já satisfazemos o requerimento: sempre "Named("bin") = bin_labels"
  
  return DataFrame::create(
    Named("bin") = bin_labels,  // Sempre string
    Named("id") = group_ids,
    Named("count") = counts,
    Named("pos") = count_pos,
    Named("neg") = count_neg,
    Named("woe") = woe,
    Named("iv") = iv,
    Named("total_iv") = rep(total_iv, n_bins),
    Named("cum_pos") = cum_pos,
    Named("cum_neg") = cum_neg,
    Named("pos_rate") = pos_rate,
    Named("neg_rate") = neg_rate,
    Named("pos_perc") = pos_perc,
    Named("neg_perc") = neg_perc,
    Named("count_perc") = count_perc,
    Named("cum_count_perc") = cum_count_perc,
    Named("cum_pos_perc") = cum_pos_perc,
    Named("cum_neg_perc") = cum_neg_perc,
    Named("cum_pos_perc_total") = cum_pos_perc_total,
    Named("cum_neg_perc_total") = cum_neg_perc_total,
    Named("odds_pos") = odds_pos,
    Named("odds_ratio") = odds_ratio,
    Named("lift") = lift,
    Named("ks") = ks,
    Named("gini_contribution") = gini_contribution,
    Named("precision") = precision,
    Named("recall") = recall,
    Named("f1_score") = f1_score,
    Named("log_likelihood") = log_likelihood,
    Named("kl_divergence") = kl_divergence_metric,
    Named("js_divergence") = js_divergence_metric
  );
}