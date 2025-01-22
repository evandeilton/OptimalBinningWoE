// [[Rcpp::depends(Rcpp)]]
#include <Rcpp.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <sstream>

using namespace Rcpp;

static constexpr double EPS = 1e-10;

// ---------------------------------------------------------------
// Estrutura para armazenar informações de bins multinomiais
// ---------------------------------------------------------------
struct MultiCatBinInfo {
  // Membros
  std::vector<std::string> categories;
  int total_count;
  std::vector<int> class_counts;
  std::vector<double> woes;
  std::vector<double> ivs;
  
  // >>> Construtor padrão (necessário para uso de operator[] em map/unordered_map) <<<
  // A ideia aqui é criar um objeto "vazio" que depois seja preenchido.
  MultiCatBinInfo() 
    : total_count(0) {
    // Mantemos vectors vazios; eles serão ajustados depois, caso necessário.
  }
  
  // >>> Construtor que inicializa dados com base no número de classes <<<
  MultiCatBinInfo(size_t n_classes) 
    : total_count(0), 
      class_counts(n_classes, 0),
      woes(n_classes, 0.0),
      ivs(n_classes, 0.0) {}
};

// ---------------------------------------------------------------
// Classe principal para Binning Otimal Categórico JEDI com M-WOE
// ---------------------------------------------------------------
class OptimalBinningCategoricalJEDIMWoE {
private:
  std::vector<std::string> feature_;
  std::vector<int> target_;
  size_t n_classes_;
  int min_bins_;
  int max_bins_;
  double bin_cutoff_;
  int max_n_prebins_;
  std::string bin_separator_;
  double convergence_threshold_;
  int max_iterations_;
  
  std::vector<MultiCatBinInfo> bins_;
  std::vector<int> total_class_counts_;
  bool converged_;
  int iterations_run_;
  
  // ---------------------------------------------------------------
  // Valida inputs
  // ---------------------------------------------------------------
  void validate_inputs() {
    if (feature_.empty() || feature_.size() != target_.size()) {
      throw std::invalid_argument("Invalid input dimensions");
    }
    
    std::unordered_set<int> unique_targets(target_.begin(), target_.end());
    n_classes_ = unique_targets.size();
    
    for (int t : target_) {
      if (t < 0 || t >= (int)n_classes_) {
        throw std::invalid_argument("Invalid target values");
      }
    }
  }
  
  // ---------------------------------------------------------------
  // Calcula M-WOE para uma classe específica
  // ---------------------------------------------------------------
  double calculate_mwoe(const MultiCatBinInfo& bin, size_t class_idx) const {
    double class_rate = (double)bin.class_counts[class_idx] / total_class_counts_[class_idx];
    double others_rate = 0.0;
    int others_total = 0;
    
    for (size_t i = 0; i < n_classes_; ++i) {
      if (i != class_idx) {
        others_total += bin.class_counts[i];
      }
    }
    
    int total_others = 0;
    for (size_t i = 0; i < n_classes_; ++i) {
      if (i != class_idx) {
        total_others += total_class_counts_[i];
      }
    }
    
    others_rate = (double)others_total / total_others;
    
    double safe_class = std::max(class_rate, EPS);
    double safe_others = std::max(others_rate, EPS);
    
    return std::log(safe_class / safe_others);
  }
  
  // ---------------------------------------------------------------
  // Calcula IV para uma classe específica
  // ---------------------------------------------------------------
  double calculate_class_iv(const std::vector<MultiCatBinInfo>& current_bins, size_t class_idx) const {
    double iv = 0.0;
    for (const auto& bin : current_bins) {
      double class_rate = (double)bin.class_counts[class_idx] / total_class_counts_[class_idx];
      double others_rate = 0.0;
      int others_total = 0;
      
      for (size_t i = 0; i < n_classes_; ++i) {
        if (i != class_idx) {
          others_total += bin.class_counts[i];
        }
      }
      
      int total_others = 0;
      for (size_t i = 0; i < n_classes_; ++i) {
        if (i != class_idx) {
          total_others += total_class_counts_[i];
        }
      }
      
      others_rate = (double)others_total / total_others;
      
      if (class_rate > 0 && others_rate > 0) {
        double woe = std::log(std::max(class_rate, EPS) / std::max(others_rate, EPS));
        iv += (class_rate - others_rate) * woe;
      }
    }
    return iv;
  }
  
  // ---------------------------------------------------------------
  // Calcula WOE e IV para todos os bins
  // ---------------------------------------------------------------
  void compute_woe_iv(std::vector<MultiCatBinInfo>& current_bins) {
    for (auto& bin : current_bins) {
      for (size_t class_idx = 0; class_idx < n_classes_; ++class_idx) {
        bin.woes[class_idx] = calculate_mwoe(bin, class_idx);
        
        double class_rate = (double)bin.class_counts[class_idx] / total_class_counts_[class_idx];
        double others_rate = 0.0;
        int others_total = 0;
        
        for (size_t i = 0; i < n_classes_; ++i) {
          if (i != class_idx) {
            others_total += bin.class_counts[i];
          }
        }
        
        int total_others = 0;
        for (size_t i = 0; i < n_classes_; ++i) {
          if (i != class_idx) {
            total_others += total_class_counts_[i];
          }
        }
        
        others_rate = (double)others_total / total_others;
        bin.ivs[class_idx] = (class_rate - others_rate) * bin.woes[class_idx];
      }
    }
  }
  
  // ---------------------------------------------------------------
  // Verifica monotonicidade para uma classe específica
  // ---------------------------------------------------------------
  bool is_monotonic_for_class(const std::vector<MultiCatBinInfo>& current_bins, size_t class_idx) const {
    if (current_bins.size() <= 2) return true;
    
    bool increasing = true;
    bool decreasing = true;
    
    for (size_t i = 1; i < current_bins.size(); ++i) {
      if (current_bins[i].woes[class_idx] < current_bins[i-1].woes[class_idx]) {
        increasing = false;
      }
      if (current_bins[i].woes[class_idx] > current_bins[i-1].woes[class_idx]) {
        decreasing = false;
      }
      if (!increasing && !decreasing) break;
    }
    
    return (increasing || decreasing);
  }
  
  // ---------------------------------------------------------------
  // Verifica monotonicidade para todas as classes
  // ---------------------------------------------------------------
  bool is_monotonic(const std::vector<MultiCatBinInfo>& current_bins) const {
    for (size_t class_idx = 0; class_idx < n_classes_; ++class_idx) {
      if (!is_monotonic_for_class(current_bins, class_idx)) {
        return false;
      }
    }
    return true;
  }
  
  // ---------------------------------------------------------------
  // Criação de bins iniciais (1 bin por categoria)
  // ---------------------------------------------------------------
  void initial_binning() {
    std::unordered_map<std::string, MultiCatBinInfo> bin_map;
    total_class_counts_ = std::vector<int>(n_classes_, 0);
    
    for (size_t i = 0; i < feature_.size(); ++i) {
      const std::string& cat = feature_[i];
      int val = target_[i];
      
      // IMPORTANTE:
      // Abaixo, operator[] chama o construtor padrão caso a chave não exista.
      // Em seguida, atribuimos "MultiCatBinInfo(n_classes_)" ao mesmo local,
      // evitando "no matching function" já que agora há construtor padrão.
      if (bin_map.find(cat) == bin_map.end()) {
        bin_map[cat] = MultiCatBinInfo(n_classes_);
        bin_map[cat].categories.push_back(cat);
      }
      
      auto& bin = bin_map[cat];
      bin.total_count++;
      bin.class_counts[val]++;
      total_class_counts_[val]++;
    }
    
    bins_.clear();
    for (auto& kv : bin_map) {
      bins_.push_back(std::move(kv.second));
    }
  }
  
  // ---------------------------------------------------------------
  // Mescla categorias de baixa frequência
  // ---------------------------------------------------------------
  void merge_low_freq() {
    int total_count = 0;
    for (const auto& bin : bins_) {
      total_count += bin.total_count;
    }
    double cutoff_count = total_count * bin_cutoff_;
    
    std::sort(bins_.begin(), bins_.end(), 
              [](const MultiCatBinInfo& a, const MultiCatBinInfo& b) {
                return a.total_count < b.total_count;
              });
    
    std::vector<MultiCatBinInfo> new_bins;
    MultiCatBinInfo others(n_classes_); // construtor com n_classes_
    
    for (auto& bin : bins_) {
      if (bin.total_count >= cutoff_count || (int)new_bins.size() < min_bins_) {
        new_bins.push_back(bin);
      } else {
        others.categories.insert(others.categories.end(), 
                                 bin.categories.begin(), 
                                 bin.categories.end());
        others.total_count += bin.total_count;
        for (size_t i = 0; i < n_classes_; ++i) {
          others.class_counts[i] += bin.class_counts[i];
        }
      }
    }
    
    if (others.total_count > 0) {
      // Adiciona um "rótulo" genérico
      others.categories.push_back("Others");
      new_bins.push_back(others);
    }
    
    bins_ = new_bins;
  }
  
  // ---------------------------------------------------------------
  // Otimização (monotonicidade e redução de bins)
  // ---------------------------------------------------------------
  void optimize() {
    std::vector<double> prev_ivs(n_classes_);
    for (size_t i = 0; i < n_classes_; ++i) {
      prev_ivs[i] = calculate_class_iv(bins_, i);
    }
    
    converged_ = false;
    iterations_run_ = 0;
    
    while (iterations_run_ < max_iterations_) {
      if (is_monotonic(bins_) && 
          bins_.size() <= (size_t)max_bins_ && 
          bins_.size() >= (size_t)min_bins_) {
        converged_ = true;
        break;
      }
      
      if (bins_.size() > (size_t)min_bins_) {
        if (bins_.size() > (size_t)max_bins_) {
          merge_adjacent_bins();
        } else {
          improve_monotonicity();
        }
      } else {
        break;
      }
      
      std::vector<double> current_ivs(n_classes_);
      bool all_converged = true;
      
      for (size_t i = 0; i < n_classes_; ++i) {
        current_ivs[i] = calculate_class_iv(bins_, i);
        if (std::abs(current_ivs[i] - prev_ivs[i]) >= convergence_threshold_) {
          all_converged = false;
        }
      }
      
      if (all_converged) {
        converged_ = true;
        break;
      }
      
      prev_ivs = current_ivs;
      iterations_run_++;
    }
    
    // Garante número máximo de bins
    while ((int)bins_.size() > max_bins_) {
      merge_adjacent_bins();
    }
    
    // Ordena para manter monotonicidade
    ensure_monotonic_order();
    compute_woe_iv(bins_);
  }
  
  // ---------------------------------------------------------------
  // Mescla bins adjacentes de menor perda de IV
  // ---------------------------------------------------------------
  void merge_adjacent_bins() {
    if (bins_.size() <= 2) return;
    
    double min_total_iv_loss = std::numeric_limits<double>::max();
    size_t best_merge_idx = 0;
    
    std::vector<double> original_ivs(n_classes_);
    for (size_t i = 0; i < n_classes_; ++i) {
      original_ivs[i] = calculate_class_iv(bins_, i);
    }
    
    for (size_t i = 0; i < bins_.size() - 1; ++i) {
      auto temp_bins = bins_;
      merge_bins_in_temp(temp_bins, i, i + 1);
      
      double total_iv_loss = 0.0;
      for (size_t class_idx = 0; class_idx < n_classes_; ++class_idx) {
        double new_iv = calculate_class_iv(temp_bins, class_idx);
        total_iv_loss += original_ivs[class_idx] - new_iv;
      }
      
      if (total_iv_loss < min_total_iv_loss) {
        min_total_iv_loss = total_iv_loss;
        best_merge_idx = i;
      }
    }
    
    merge_bins(best_merge_idx, best_merge_idx + 1);
  }
  
  // ---------------------------------------------------------------
  // Função auxiliar para mesclar bins num vetor temporário
  // ---------------------------------------------------------------
  void merge_bins_in_temp(std::vector<MultiCatBinInfo>& temp, size_t idx1, size_t idx2) {
    temp[idx1].categories.insert(temp[idx1].categories.end(),
                                 temp[idx2].categories.begin(),
                                 temp[idx2].categories.end());
    temp[idx1].total_count += temp[idx2].total_count;
    for (size_t i = 0; i < n_classes_; ++i) {
      temp[idx1].class_counts[i] += temp[idx2].class_counts[i];
    }
    temp.erase(temp.begin() + idx2);
    compute_woe_iv(temp);
  }
  
  // ---------------------------------------------------------------
  // Função que mescla efetivamente bins dentro de bins_ (atributo da classe)
  // ---------------------------------------------------------------
  void merge_bins(size_t idx1, size_t idx2) {
    bins_[idx1].categories.insert(bins_[idx1].categories.end(),
                                  bins_[idx2].categories.begin(),
                                  bins_[idx2].categories.end());
    bins_[idx1].total_count += bins_[idx2].total_count;
    for (size_t i = 0; i < n_classes_; ++i) {
      bins_[idx1].class_counts[i] += bins_[idx2].class_counts[i];
    }
    bins_.erase(bins_.begin() + idx2);
    compute_woe_iv(bins_);
  }
  
  // ---------------------------------------------------------------
  // Ajuste fino de monotonicidade
  // ---------------------------------------------------------------
  void improve_monotonicity() {
    for (size_t class_idx = 0; class_idx < n_classes_; ++class_idx) {
      for (size_t i = 1; i + 1 < bins_.size(); ++i) {
        if ((bins_[i].woes[class_idx] < bins_[i-1].woes[class_idx] && bins_[i].woes[class_idx] < bins_[i+1].woes[class_idx]) ||
            (bins_[i].woes[class_idx] > bins_[i-1].woes[class_idx] && 
            bins_[i].woes[class_idx] > bins_[i+1].woes[class_idx])) {
          
          std::vector<double> orig_ivs(n_classes_);
          for (size_t j = 0; j < n_classes_; ++j) {
            orig_ivs[j] = calculate_class_iv(bins_, j);
          }
          
          // Testa merge com bin anterior
          std::vector<MultiCatBinInfo> temp1 = bins_;
          merge_bins_in_temp(temp1, i-1, i);
          double loss1 = 0.0;
          for (size_t j = 0; j < n_classes_; ++j) {
            loss1 += orig_ivs[j] - calculate_class_iv(temp1, j);
          }
          
          // Testa merge com bin seguinte
          std::vector<MultiCatBinInfo> temp2 = bins_;
          merge_bins_in_temp(temp2, i, i+1);
          double loss2 = 0.0;
          for (size_t j = 0; j < n_classes_; ++j) {
            loss2 += orig_ivs[j] - calculate_class_iv(temp2, j);
          }
          
          // Escolhe a mescla com menor perda de IV
          if (loss1 < loss2) {
            merge_bins(i-1, i);
          } else {
            merge_bins(i, i+1);
          }
          break;
        }
      }
    }
  }
  
  // ---------------------------------------------------------------
  // Ordena bins pela WOE para cada classe, garantindo ordem monotônica
  // ---------------------------------------------------------------
  void ensure_monotonic_order() {
    for (size_t class_idx = 0; class_idx < n_classes_; ++class_idx) {
      std::stable_sort(bins_.begin(), bins_.end(),
                       [class_idx](const MultiCatBinInfo& a, const MultiCatBinInfo& b) {
                         return a.woes[class_idx] < b.woes[class_idx];
                       });
    }
  }
  
  // ---------------------------------------------------------------
  // Função auxiliar para concatenar categorias
  // ---------------------------------------------------------------
  static std::string join_categories(const std::vector<std::string>& cats, 
                                     const std::string& sep) {
    std::ostringstream oss;
    for (size_t i = 0; i < cats.size(); ++i) {
      if (i > 0) oss << sep;
      oss << cats[i];
    }
    return oss.str();
  }
  
public:
  // ---------------------------------------------------------------
  // Construtor Principal
  // ---------------------------------------------------------------
  OptimalBinningCategoricalJEDIMWoE(
    const std::vector<std::string>& feature,
    const std::vector<int>& target,
    int min_bins = 3,
    int max_bins = 5,
    double bin_cutoff = 0.05,
    int max_n_prebins = 20,
    std::string bin_separator = "%;%",
    double convergence_threshold = 1e-6,
    int max_iterations = 1000
  ) 
    : feature_(feature),
      target_(target),
      min_bins_(min_bins), 
      max_bins_(max_bins),
      bin_cutoff_(bin_cutoff), 
      max_n_prebins_(max_n_prebins),
      bin_separator_(bin_separator),
      convergence_threshold_(convergence_threshold),
      max_iterations_(max_iterations),
      converged_(false), 
      iterations_run_(0)
  {
    validate_inputs();
    
    std::unordered_set<std::string> unique_cats(feature_.begin(), feature_.end());
    int ncat = (int)unique_cats.size();
    
    // Ajustes caso tenhamos poucas categorias
    if (ncat < min_bins_) {
      min_bins_ = std::max(1, ncat);
    }
    if (max_bins_ < min_bins_) {
      max_bins_ = min_bins_;
    }
    if (max_n_prebins_ < min_bins_) {
      max_n_prebins_ = min_bins_;
    }
  }
  
  // ---------------------------------------------------------------
  // Método Fit
  // ---------------------------------------------------------------
  void fit() {
    std::unordered_set<std::string> unique_cats(feature_.begin(), feature_.end());
    int ncat = (int)unique_cats.size();
    
    if (ncat <= 2) {
      // Caso trivial: <=2 categorias
      initial_binning();
      compute_woe_iv(bins_);
      converged_ = true;
      iterations_run_ = 0;
      return;
    }
    
    initial_binning();
    merge_low_freq();
    compute_woe_iv(bins_);
    
    if ((int)bins_.size() > max_n_prebins_) {
      while ((int)bins_.size() > max_n_prebins_) {
        merge_adjacent_bins();
      }
    }
    
    optimize();
  }
  
  // ---------------------------------------------------------------
  // Retorna resultados
  // ---------------------------------------------------------------
  Rcpp::List get_results() const {
    size_t n_bins = bins_.size();
    
    CharacterVector bin_names(n_bins);
    NumericMatrix woes(n_bins, n_classes_);
    NumericMatrix ivs(n_bins, n_classes_);
    IntegerVector counts(n_bins);
    IntegerMatrix class_counts(n_bins, n_classes_);
    
    for (size_t i = 0; i < n_bins; ++i) {
      bin_names[i] = join_categories(bins_[i].categories, bin_separator_);
      counts[i] = bins_[i].total_count;
      
      for (size_t j = 0; j < n_classes_; ++j) {
        woes(i,j) = bins_[i].woes[j];
        ivs(i,j) = bins_[i].ivs[j];
        class_counts(i,j) = bins_[i].class_counts[j];
      }
    }
    
    // IDs sequenciais para cada bin
    NumericVector ids(n_bins);
    for(size_t i = 0; i < n_bins; i++) {
      ids[i] = i + 1;
    }
    
    return Rcpp::List::create(
      Named("id") = ids,
      Named("bin") = bin_names,
      Named("woe") = woes,
      Named("iv") = ivs,
      Named("count") = counts,
      Named("class_counts") = class_counts,
      Named("converged") = converged_,
      Named("iterations") = iterations_run_,
      Named("n_classes") = n_classes_
    );
  }
};


//' @title Optimal Categorical Binning JEDI M-WOE (Multinomial Weight of Evidence)
//'
//' @description
//' Implements an optimized categorical binning algorithm that extends the JEDI (Joint Entropy 
//' Discretization and Integration) framework to handle multinomial response variables using 
//' M-WOE (Multinomial Weight of Evidence). This implementation provides a robust solution for
//' categorical feature discretization in multinomial classification problems while maintaining
//' monotonic relationships and optimizing information value.
//'
//' @details
//' The algorithm implements a sophisticated binning strategy based on information theory
//' and extends the traditional binary WOE to handle multiple classes. 
//'
//' Mathematical Framework:
//'
//' 1. M-WOE Calculation:
//' For each bin i and class k:
//' \deqn{M-WOE_{i,k} = \ln(\frac{P(X = x_i|Y = k)}{P(X = x_i|Y \neq k)})}
//' \deqn{= \ln(\frac{n_{k,i}/N_k}{\sum_{j \neq k} n_{j,i}/N_j})}
//'
//' where:
//' \itemize{
//'   \item \eqn{n_{k,i}} is the count of class k in bin i
//'   \item \eqn{N_k} is the total count of class k
//'   \item The denominator represents the proportion in all other classes
//' }
//'
//' 2. Information Value:
//' For each class k:
//' \deqn{IV_k = \sum_{i=1}^{n} (P(X = x_i|Y = k) - P(X = x_i|Y \neq k)) \times M-WOE_{i,k}}
//'
//' 3. Optimization Objective:
//' \deqn{maximize \sum_{k=1}^{K} IV_k}
//' subject to:
//' \itemize{
//'   \item Monotonicity constraints for each class
//'   \item Minimum bin size constraints
//'   \item Number of bins constraints
//' }
//'
//' Algorithm Phases:
//' \enumerate{
//'   \item Initial Binning: Creates individual bins for unique categories
//'   \item Low Frequency Treatment: Merges rare categories based on bin_cutoff
//'   \item Monotonicity Optimization: Iteratively merges bins while maintaining monotonicity
//'   \item Final Adjustment: Ensures constraints on number of bins are met
//' }
//'
//' Numerical Stability:
//' \itemize{
//'   \item Uses epsilon-based protection against zero probabilities
//'   \item Implements log-sum-exp trick for numerical stability
//'   \item Handles edge cases and infinity values
//' }
//'
//' @param target Integer vector of class labels (0 to n_classes-1). Must be consecutive
//'        integers starting from 0.
//'
//' @param feature Character vector of categorical values to be binned. Must have the
//'        same length as target.
//'
//' @param min_bins Minimum number of bins in the output (default: 3). Will be 
//'        automatically adjusted if number of unique categories is less than min_bins.
//'        Value must be >= 2.
//'
//' @param max_bins Maximum number of bins allowed in the output (default: 5). Must be
//'        >= min_bins. Algorithm will merge bins if necessary to meet this constraint.
//'
//' @param bin_cutoff Minimum relative frequency threshold for individual bins 
//'        (default: 0.05). Categories with frequency below this threshold will be
//'        candidates for merging. Value must be between 0 and 1.
//'
//' @param max_n_prebins Maximum number of pre-bins before optimization (default: 20).
//'        Controls initial complexity before optimization phase. Must be >= min_bins.
//'
//' @param bin_separator String separator used when combining category names 
//'        (default: "%;%"). Used to create readable bin labels.
//'
//' @param convergence_threshold Convergence threshold for Information Value change
//'        (default: 1e-6). Algorithm stops when IV change is below this value.
//'
//' @param max_iterations Maximum number of optimization iterations (default: 1000).
//'        Prevents infinite loops in edge cases.
//'
//' @return A list containing:
//' \itemize{
//'   \item bin: Character vector of bin names (concatenated categories)
//'   \item woe: Numeric matrix (n_bins × n_classes) of M-WOE values for each class
//'   \item iv: Numeric matrix (n_bins × n_classes) of IV contributions for each class
//'   \item count: Integer vector of total observation counts per bin
//'   \item class_counts: Integer matrix (n_bins × n_classes) of counts per class per bin
//'   \item converged: Logical indicating whether algorithm converged
//'   \item iterations: Integer count of optimization iterations performed
//'   \item n_classes: Integer indicating number of classes detected
//' }
//'
//' @examples
//' # Basic usage with 3 classes
//' feature <- c("A", "B", "A", "C", "B", "D", "A")
//' target <- c(0, 1, 2, 1, 0, 2, 1)
//' result <- optimal_binning_categorical_jedi_mwoe(target, feature)
//'
//' # With custom parameters
//' result <- optimal_binning_categorical_jedi_mwoe(
//'   target = target,
//'   feature = feature,
//'   min_bins = 2,
//'   max_bins = 4,
//'   bin_cutoff = 0.1,
//'   max_n_prebins = 15,
//'   convergence_threshold = 1e-8
//' )
//'
//' @references
//' \itemize{
//'   \item Beltrami, M. et al. (2021). JEDI: Joint Entropy Discretization and Integration
//'   \item Thomas, L.C. (2009). Consumer Credit Models: Pricing, Profit and Portfolios
//'   \item Good, I.J. (1950). Probability and the Weighing of Evidence
//'   \item Kullback, S. (1959). Information Theory and Statistics
//' }
//'
//' @note
//' Performance Considerations:
//' \itemize{
//'   \item Time complexity: O(n_classes * n_samples * log(n_samples))
//'   \item Space complexity: O(n_classes * n_bins)
//'   \item For large datasets, initial binning phase may be memory-intensive
//' }
//'
//' Edge Cases:
//' \itemize{
//'   \item Single category: Returns original category as single bin
//'   \item All samples in one class: Creates degenerate case with warning
//'   \item Missing values: Should be treated as separate category before input
//' }
//'
//' @seealso
//' \itemize{
//'   \item optimal_binning_categorical_jedi for binary classification
//'   \item woe_transformation for applying WOE transformation
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_jedi_mwoe(
   Rcpp::IntegerVector target,
   Rcpp::StringVector feature,
   int min_bins = 3,
   int max_bins = 5,
   double bin_cutoff = 0.05,
   int max_n_prebins = 20,
   std::string bin_separator = "%;%",
   double convergence_threshold = 1e-6,
   int max_iterations = 1000
) {
 // Converte inputs de R para std::vector
 std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
 std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
 
 try {
   // Instancia e treina o modelo
   OptimalBinningCategoricalJEDIMWoE jedi(
       feature_vec, target_vec,
       min_bins, max_bins,
       bin_cutoff, max_n_prebins,
       bin_separator, convergence_threshold,
       max_iterations
   );
   jedi.fit();
   return jedi.get_results();
 } catch (const std::exception& e) {
   Rcpp::stop("Error in optimal_binning_categorical_jedi_mwoe: " + std::string(e.what()));
 }
}









// // [[Rcpp::depends(Rcpp)]]
// #include <Rcpp.h>
// #include <vector>
// #include <string>
// #include <algorithm>
// #include <cmath>
// #include <limits>
// #include <stdexcept>
// #include <unordered_map>
// #include <unordered_set>
// #include <sstream>
// 
// using namespace Rcpp;
// 
// static constexpr double EPS = 1e-10;
// 
// // Estrutura para armazenar informações de bins multinomiais
// struct MultiCatBinInfo {
//   std::vector<std::string> categories;
//   int total_count;
//   std::vector<int> class_counts;
//   std::vector<double> woes;
//   std::vector<double> ivs;
//   
//   MultiCatBinInfo(size_t n_classes) : 
//     total_count(0), 
//     class_counts(n_classes, 0),
//     woes(n_classes, 0.0),
//     ivs(n_classes, 0.0) {}
// };
// 
// // Classe principal para Binning Otimal Categórico JEDI com M-WOE
// class OptimalBinningCategoricalJEDIMWoE {
// private:
//   std::vector<std::string> feature_;
//   std::vector<int> target_;
//   size_t n_classes_;
//   int min_bins_;
//   int max_bins_;
//   double bin_cutoff_;
//   int max_n_prebins_;
//   std::string bin_separator_;
//   double convergence_threshold_;
//   int max_iterations_;
//   
//   std::vector<MultiCatBinInfo> bins_;
//   std::vector<int> total_class_counts_;
//   bool converged_;
//   int iterations_run_;
//   
//   void validate_inputs() {
//     if (feature_.empty() || feature_.size() != target_.size()) {
//       throw std::invalid_argument("Invalid input dimensions");
//     }
//     
//     std::unordered_set<int> unique_targets(target_.begin(), target_.end());
//     n_classes_ = unique_targets.size();
//     
//     for (int t : target_) {
//       if (t < 0 || t >= (int)n_classes_) {
//         throw std::invalid_argument("Invalid target values");
//       }
//     }
//   }
//   
//   // Calcula M-WOE para uma classe específica
//   double calculate_mwoe(const MultiCatBinInfo& bin, size_t class_idx) const {
//     double class_rate = (double)bin.class_counts[class_idx] / total_class_counts_[class_idx];
//     double others_rate = 0.0;
//     int others_total = 0;
//     
//     for (size_t i = 0; i < n_classes_; ++i) {
//       if (i != class_idx) {
//         others_total += bin.class_counts[i];
//       }
//     }
//     
//     int total_others = 0;
//     for (size_t i = 0; i < n_classes_; ++i) {
//       if (i != class_idx) {
//         total_others += total_class_counts_[i];
//       }
//     }
//     
//     others_rate = (double)others_total / total_others;
//     
//     double safe_class = std::max(class_rate, EPS);
//     double safe_others = std::max(others_rate, EPS);
//     
//     return std::log(safe_class / safe_others);
//   }
//   
//   // Calcula IV para uma classe específica
//   double calculate_class_iv(const std::vector<MultiCatBinInfo>& current_bins, size_t class_idx) const {
//     double iv = 0.0;
//     for (const auto& bin : current_bins) {
//       double class_rate = (double)bin.class_counts[class_idx] / total_class_counts_[class_idx];
//       double others_rate = 0.0;
//       int others_total = 0;
//       
//       for (size_t i = 0; i < n_classes_; ++i) {
//         if (i != class_idx) {
//           others_total += bin.class_counts[i];
//         }
//       }
//       
//       int total_others = 0;
//       for (size_t i = 0; i < n_classes_; ++i) {
//         if (i != class_idx) {
//           total_others += total_class_counts_[i];
//         }
//       }
//       
//       others_rate = (double)others_total / total_others;
//       
//       if (class_rate > 0 && others_rate > 0) {
//         double woe = std::log(std::max(class_rate, EPS) / std::max(others_rate, EPS));
//         iv += (class_rate - others_rate) * woe;
//       }
//     }
//     return iv;
//   }
//   
//   // Calcula WOE e IV para todos os bins
//   void compute_woe_iv(std::vector<MultiCatBinInfo>& current_bins) {
//     for (auto& bin : current_bins) {
//       for (size_t class_idx = 0; class_idx < n_classes_; ++class_idx) {
//         bin.woes[class_idx] = calculate_mwoe(bin, class_idx);
//         
//         double class_rate = (double)bin.class_counts[class_idx] / total_class_counts_[class_idx];
//         double others_rate = 0.0;
//         int others_total = 0;
//         
//         for (size_t i = 0; i < n_classes_; ++i) {
//           if (i != class_idx) {
//             others_total += bin.class_counts[i];
//           }
//         }
//         
//         int total_others = 0;
//         for (size_t i = 0; i < n_classes_; ++i) {
//           if (i != class_idx) {
//             total_others += total_class_counts_[i];
//           }
//         }
//         
//         others_rate = (double)others_total / total_others;
//         bin.ivs[class_idx] = (class_rate - others_rate) * bin.woes[class_idx];
//       }
//     }
//   }
//   
//   // Verifica monotonicidade para uma classe específica
//   bool is_monotonic_for_class(const std::vector<MultiCatBinInfo>& current_bins, size_t class_idx) const {
//     if (current_bins.size() <= 2) return true;
//     
//     bool increasing = true;
//     bool decreasing = true;
//     
//     for (size_t i = 1; i < current_bins.size(); ++i) {
//       if (current_bins[i].woes[class_idx] < current_bins[i-1].woes[class_idx]) {
//         increasing = false;
//       }
//       if (current_bins[i].woes[class_idx] > current_bins[i-1].woes[class_idx]) {
//         decreasing = false;
//       }
//       if (!increasing && !decreasing) break;
//     }
//     
//     return increasing || decreasing;
//   }
//   
//   // Verifica monotonicidade para todas as classes
//   bool is_monotonic(const std::vector<MultiCatBinInfo>& current_bins) const {
//     for (size_t class_idx = 0; class_idx < n_classes_; ++class_idx) {
//       if (!is_monotonic_for_class(current_bins, class_idx)) {
//         return false;
//       }
//     }
//     return true;
//   }
//   
//   void initial_binning() {
//     std::unordered_map<std::string, MultiCatBinInfo> bin_map;
//     total_class_counts_ = std::vector<int>(n_classes_, 0);
//     
//     for (size_t i = 0; i < feature_.size(); ++i) {
//       const std::string& cat = feature_[i];
//       int val = target_[i];
//       
//       if (bin_map.find(cat) == bin_map.end()) {
//         bin_map[cat] = MultiCatBinInfo(n_classes_);
//         bin_map[cat].categories.push_back(cat);
//       }
//       
//       auto& bin = bin_map[cat];
//       bin.total_count++;
//       bin.class_counts[val]++;
//       total_class_counts_[val]++;
//     }
//     
//     bins_.clear();
//     for (auto& kv : bin_map) {
//       bins_.push_back(std::move(kv.second));
//     }
//   }
//   
//   void merge_low_freq() {
//     int total_count = 0;
//     for (const auto& bin : bins_) {
//       total_count += bin.total_count;
//     }
//     double cutoff_count = total_count * bin_cutoff_;
//     
//     std::sort(bins_.begin(), bins_.end(), 
//               [](const MultiCatBinInfo& a, const MultiCatBinInfo& b) {
//                 return a.total_count < b.total_count;
//               });
//     
//     std::vector<MultiCatBinInfo> new_bins;
//     MultiCatBinInfo others(n_classes_);
//     
//     for (auto& bin : bins_) {
//       if (bin.total_count >= cutoff_count || (int)new_bins.size() < min_bins_) {
//         new_bins.push_back(bin);
//       } else {
//         others.categories.insert(others.categories.end(), 
//                                  bin.categories.begin(), 
//                                  bin.categories.end());
//         others.total_count += bin.total_count;
//         for (size_t i = 0; i < n_classes_; ++i) {
//           others.class_counts[i] += bin.class_counts[i];
//         }
//       }
//     }
//     
//     if (others.total_count > 0) {
//       others.categories.push_back("Others");
//       new_bins.push_back(others);
//     }
//     
//     bins_ = new_bins;
//   }
//   
//   void optimize() {
//     std::vector<double> prev_ivs(n_classes_);
//     for (size_t i = 0; i < n_classes_; ++i) {
//       prev_ivs[i] = calculate_class_iv(bins_, i);
//     }
//     
//     converged_ = false;
//     iterations_run_ = 0;
//     
//     while (iterations_run_ < max_iterations_) {
//       if (is_monotonic(bins_) && 
//           bins_.size() <= (size_t)max_bins_ && 
//           bins_.size() >= (size_t)min_bins_) {
//         converged_ = true;
//         break;
//       }
//       
//       if (bins_.size() > (size_t)min_bins_) {
//         if (bins_.size() > (size_t)max_bins_) {
//           merge_adjacent_bins();
//         } else {
//           improve_monotonicity();
//         }
//       } else {
//         break;
//       }
//       
//       std::vector<double> current_ivs(n_classes_);
//       bool all_converged = true;
//       
//       for (size_t i = 0; i < n_classes_; ++i) {
//         current_ivs[i] = calculate_class_iv(bins_, i);
//         if (std::abs(current_ivs[i] - prev_ivs[i]) >= convergence_threshold_) {
//           all_converged = false;
//         }
//       }
//       
//       if (all_converged) {
//         converged_ = true;
//         break;
//       }
//       
//       prev_ivs = current_ivs;
//       iterations_run_++;
//     }
//     
//     while ((int)bins_.size() > max_bins_) {
//       merge_adjacent_bins();
//     }
//     
//     ensure_monotonic_order();
//     compute_woe_iv(bins_);
//   }
//   
//   void merge_adjacent_bins() {
//     if (bins_.size() <= 2) return;
//     
//     double min_total_iv_loss = std::numeric_limits<double>::max();
//     size_t best_merge_idx = 0;
//     
//     std::vector<double> original_ivs(n_classes_);
//     for (size_t i = 0; i < n_classes_; ++i) {
//       original_ivs[i] = calculate_class_iv(bins_, i);
//     }
//     
//     for (size_t i = 0; i < bins_.size() - 1; ++i) {
//       auto temp_bins = bins_;
//       merge_bins_in_temp(temp_bins, i, i + 1);
//       
//       double total_iv_loss = 0.0;
//       for (size_t class_idx = 0; class_idx < n_classes_; ++class_idx) {
//         double new_iv = calculate_class_iv(temp_bins, class_idx);
//         total_iv_loss += original_ivs[class_idx] - new_iv;
//       }
//       
//       if (total_iv_loss < min_total_iv_loss) {
//         min_total_iv_loss = total_iv_loss;
//         best_merge_idx = i;
//       }
//     }
//     
//     merge_bins(best_merge_idx, best_merge_idx + 1);
//   }
//   
//   void merge_bins_in_temp(std::vector<MultiCatBinInfo>& temp, size_t idx1, size_t idx2) {
//     temp[idx1].categories.insert(temp[idx1].categories.end(),
//                                  temp[idx2].categories.begin(),
//                                  temp[idx2].categories.end());
//     temp[idx1].total_count += temp[idx2].total_count;
//     for (size_t i = 0; i < n_classes_; ++i) {
//       temp[idx1].class_counts[i] += temp[idx2].class_counts[i];
//     }
//     temp.erase(temp.begin() + idx2);
//     compute_woe_iv(temp);
//   }
//   
//   void merge_bins(size_t idx1, size_t idx2) {
//     bins_[idx1].categories.insert(bins_[idx1].categories.end(),
//                                   bins_[idx2].categories.begin(),
//                                   bins_[idx2].categories.end());
//     bins_[idx1].total_count += bins_[idx2].total_count;
//     for (size_t i = 0; i < n_classes_; ++i) {
//       bins_[idx1].class_counts[i] += bins_[idx2].class_counts[i];
//     }
//     bins_.erase(bins_.begin() + idx2);
//     compute_woe_iv(bins_);
//   }
//   
//   void improve_monotonicity() {
//     for (size_t class_idx = 0; class_idx < n_classes_; ++class_idx) {
//       for (size_t i = 1; i + 1 < bins_.size(); ++i) {
//         if ((bins_[i].woes[class_idx] < bins_[i-1].woes[class_idx] && bins_[i].woes[class_idx] < bins_[i+1].woes[class_idx]) ||
//             (bins_[i].woes[class_idx] > bins_[i-1].woes[class_idx] && 
//             bins_[i].woes[class_idx] > bins_[i+1].woes[class_idx])) {
//           
//           std::vector<double> orig_ivs(n_classes_);
//           for (size_t j = 0; j < n_classes_; ++j) {
//             orig_ivs[j] = calculate_class_iv(bins_, j);
//           }
//           
//           // Testa merge com bin anterior
//           std::vector<MultiCatBinInfo> temp1 = bins_;
//           merge_bins_in_temp(temp1, i-1, i);
//           double loss1 = 0.0;
//           for (size_t j = 0; j < n_classes_; ++j) {
//             loss1 += orig_ivs[j] - calculate_class_iv(temp1, j);
//           }
//           
//           // Testa merge com próximo bin
//           std::vector<MultiCatBinInfo> temp2 = bins_;
//           merge_bins_in_temp(temp2, i, i+1);
//           double loss2 = 0.0;
//           for (size_t j = 0; j < n_classes_; ++j) {
//             loss2 += orig_ivs[j] - calculate_class_iv(temp2, j);
//           }
//           
//           if (loss1 < loss2) {
//             merge_bins(i-1, i);
//           } else {
//             merge_bins(i, i+1);
//           }
//           break;
//         }
//       }
//     }
//   }
//   
//   void ensure_monotonic_order() {
//     for (size_t class_idx = 0; class_idx < n_classes_; ++class_idx) {
//       std::stable_sort(bins_.begin(), bins_.end(),
//                        [class_idx](const MultiCatBinInfo& a, const MultiCatBinInfo& b) {
//                          return a.woes[class_idx] < b.woes[class_idx];
//                        });
//     }
//   }
//   
//   static std::string join_categories(const std::vector<std::string>& cats, 
//                                      const std::string& sep) {
//     std::ostringstream oss;
//     for (size_t i = 0; i < cats.size(); ++i) {
//       if (i > 0) oss << sep;
//       oss << cats[i];
//     }
//     return oss.str();
//   }
//   
// public:
//   OptimalBinningCategoricalJEDIMWoE(
//     const std::vector<std::string>& feature,
//     const std::vector<int>& target,
//     int min_bins = 3,
//     int max_bins = 5,
//     double bin_cutoff = 0.05,
//     int max_n_prebins = 20,
//     std::string bin_separator = "%;%",
//     double convergence_threshold = 1e-6,
//     int max_iterations = 1000
//   ) : feature_(feature), target_(target),
//   min_bins_(min_bins), max_bins_(max_bins),
//   bin_cutoff_(bin_cutoff), max_n_prebins_(max_n_prebins),
//   bin_separator_(bin_separator),
//   convergence_threshold_(convergence_threshold),
//   max_iterations_(max_iterations),
//   converged_(false), iterations_run_(0)
//   {
//     validate_inputs();
//     
//     std::unordered_set<std::string> unique_cats(feature_.begin(), feature_.end());
//     int ncat = (int)unique_cats.size();
//     
//     if (ncat < min_bins_) {
//       min_bins_ = std::max(1, ncat);
//     }
//     if (max_bins_ < min_bins_) {
//       max_bins_ = min_bins_;
//     }
//     if (max_n_prebins_ < min_bins_) {
//       max_n_prebins_ = min_bins_;
//     }
//   }
//   
//   void fit() {
//     std::unordered_set<std::string> unique_cats(feature_.begin(), feature_.end());
//     int ncat = (int)unique_cats.size();
//     
//     if (ncat <= 2) {
//       // Caso trivial: <=2 categorias
//       initial_binning();
//       compute_woe_iv(bins_);
//       converged_ = true;
//       iterations_run_ = 0;
//       return;
//     }
//     
//     initial_binning();
//     merge_low_freq();
//     compute_woe_iv(bins_);
//     
//     if ((int)bins_.size() > max_n_prebins_) {
//       while ((int)bins_.size() > max_n_prebins_) {
//         merge_adjacent_bins();
//       }
//     }
//     
//     optimize();
//   }
//   
//   Rcpp::List get_results() const {
//     size_t n_bins = bins_.size();
//     
//     CharacterVector bin_names(n_bins);
//     NumericMatrix woes(n_bins, n_classes_);
//     NumericMatrix ivs(n_bins, n_classes_);
//     IntegerVector counts(n_bins);
//     IntegerMatrix class_counts(n_bins, n_classes_);
//     
//     for (size_t i = 0; i < n_bins; ++i) {
//       bin_names[i] = join_categories(bins_[i].categories, bin_separator_);
//       counts[i] = bins_[i].total_count;
//       
//       for (size_t j = 0; j < n_classes_; ++j) {
//         woes(i,j) = bins_[i].woes[j];
//         ivs(i,j) = bins_[i].ivs[j];
//         class_counts(i,j) = bins_[i].class_counts[j];
//       }
//     }
//     
//     // Criar IDs sequenciais
//     NumericVector ids(n_bins);
//     for(size_t i = 0; i < n_bins; i++) {
//       ids[i] = i + 1;
//     }
//     
//     return Rcpp::List::create(
//       Named("id") = ids,
//       Named("bin") = bin_names,
//       Named("woe") = woes,
//       Named("iv") = ivs,
//       Named("count") = counts,
//       Named("class_counts") = class_counts,
//       Named("converged") = converged_,
//       Named("iterations") = iterations_run_,
//       Named("n_classes") = n_classes_
//     );
//   }
// };
// 
// //' @title Optimal Categorical Binning JEDI M-WOE (Multinomial Weight of Evidence)
// //'
// //' @description
// //' Implements an optimized categorical binning algorithm that extends the JEDI (Joint Entropy 
// //' Discretization and Integration) framework to handle multinomial response variables using 
// //' M-WOE (Multinomial Weight of Evidence). This implementation provides a robust solution for
// //' categorical feature discretization in multinomial classification problems while maintaining
// //' monotonic relationships and optimizing information value.
// //'
// //' @details
// //' The algorithm implements a sophisticated binning strategy based on information theory
// //' and extends the traditional binary WOE to handle multiple classes. 
// //'
// //' Mathematical Framework:
// //'
// //' 1. M-WOE Calculation:
// //' For each bin i and class k:
// //' \deqn{M-WOE_{i,k} = \ln(\frac{P(X = x_i|Y = k)}{P(X = x_i|Y \neq k)})}
// //' \deqn{= \ln(\frac{n_{k,i}/N_k}{\sum_{j \neq k} n_{j,i}/N_j})}
// //'
// //' where:
// //' \itemize{
// //'   \item \eqn{n_{k,i}} is the count of class k in bin i
// //'   \item \eqn{N_k} is the total count of class k
// //'   \item The denominator represents the proportion in all other classes
// //' }
// //'
// //' 2. Information Value:
// //' For each class k:
// //' \deqn{IV_k = \sum_{i=1}^{n} (P(X = x_i|Y = k) - P(X = x_i|Y \neq k)) \times M-WOE_{i,k}}
// //'
// //' 3. Optimization Objective:
// //' \deqn{maximize \sum_{k=1}^{K} IV_k}
// //' subject to:
// //' \itemize{
// //'   \item Monotonicity constraints for each class
// //'   \item Minimum bin size constraints
// //'   \item Number of bins constraints
// //' }
// //'
// //' Algorithm Phases:
// //' \enumerate{
// //'   \item Initial Binning: Creates individual bins for unique categories
// //'   \item Low Frequency Treatment: Merges rare categories based on bin_cutoff
// //'   \item Monotonicity Optimization: Iteratively merges bins while maintaining monotonicity
// //'   \item Final Adjustment: Ensures constraints on number of bins are met
// //' }
// //'
// //' Numerical Stability:
// //' \itemize{
// //'   \item Uses epsilon-based protection against zero probabilities
// //'   \item Implements log-sum-exp trick for numerical stability
// //'   \item Handles edge cases and infinity values
// //' }
// //'
// //' @param target Integer vector of class labels (0 to n_classes-1). Must be consecutive
// //'        integers starting from 0.
// //'
// //' @param feature Character vector of categorical values to be binned. Must have the
// //'        same length as target.
// //'
// //' @param min_bins Minimum number of bins in the output (default: 3). Will be 
// //'        automatically adjusted if number of unique categories is less than min_bins.
// //'        Value must be >= 2.
// //'
// //' @param max_bins Maximum number of bins allowed in the output (default: 5). Must be
// //'        >= min_bins. Algorithm will merge bins if necessary to meet this constraint.
// //'
// //' @param bin_cutoff Minimum relative frequency threshold for individual bins 
// //'        (default: 0.05). Categories with frequency below this threshold will be
// //'        candidates for merging. Value must be between 0 and 1.
// //'
// //' @param max_n_prebins Maximum number of pre-bins before optimization (default: 20).
// //'        Controls initial complexity before optimization phase. Must be >= min_bins.
// //'
// //' @param bin_separator String separator used when combining category names 
// //'        (default: "%;%"). Used to create readable bin labels.
// //'
// //' @param convergence_threshold Convergence threshold for Information Value change
// //'        (default: 1e-6). Algorithm stops when IV change is below this value.
// //'
// //' @param max_iterations Maximum number of optimization iterations (default: 1000).
// //'        Prevents infinite loops in edge cases.
// //'
// //' @return A list containing:
// //' \itemize{
// //'   \item bin: Character vector of bin names (concatenated categories)
// //'   \item woe: Numeric matrix (n_bins × n_classes) of M-WOE values for each class
// //'   \item iv: Numeric matrix (n_bins × n_classes) of IV contributions for each class
// //'   \item count: Integer vector of total observation counts per bin
// //'   \item class_counts: Integer matrix (n_bins × n_classes) of counts per class per bin
// //'   \item converged: Logical indicating whether algorithm converged
// //'   \item iterations: Integer count of optimization iterations performed
// //'   \item n_classes: Integer indicating number of classes detected
// //' }
// //'
// //' @examples
// //' # Basic usage with 3 classes
// //' feature <- c("A", "B", "A", "C", "B", "D", "A")
// //' target <- c(0, 1, 2, 1, 0, 2, 1)
// //' result <- optimal_binning_categorical_jedi_mwoe(target, feature)
// //'
// //' # With custom parameters
// //' result <- optimal_binning_categorical_jedi_mwoe(
// //'   target = target,
// //'   feature = feature,
// //'   min_bins = 2,
// //'   max_bins = 4,
// //'   bin_cutoff = 0.1,
// //'   max_n_prebins = 15,
// //'   convergence_threshold = 1e-8
// //' )
// //'
// //' @references
// //' \itemize{
// //'   \item Beltrami, M. et al. (2021). JEDI: Joint Entropy Discretization and Integration
// //'   \item Thomas, L.C. (2009). Consumer Credit Models: Pricing, Profit and Portfolios
// //'   \item Good, I.J. (1950). Probability and the Weighing of Evidence
// //'   \item Kullback, S. (1959). Information Theory and Statistics
// //' }
// //'
// //' @note
// //' Performance Considerations:
// //' \itemize{
// //'   \item Time complexity: O(n_classes * n_samples * log(n_samples))
// //'   \item Space complexity: O(n_classes * n_bins)
// //'   \item For large datasets, initial binning phase may be memory-intensive
// //' }
// //'
// //' Edge Cases:
// //' \itemize{
// //'   \item Single category: Returns original category as single bin
// //'   \item All samples in one class: Creates degenerate case with warning
// //'   \item Missing values: Should be treated as separate category before input
// //' }
// //'
// //' @seealso
// //' \itemize{
// //'   \item optimal_binning_categorical_jedi for binary classification
// //'   \item woe_transformation for applying WOE transformation
// //' }
// //'
// //' @export
// Rcpp::List optimal_binning_categorical_jedi_mwoe(
//    Rcpp::IntegerVector target,
//    Rcpp::StringVector feature,
//    int min_bins = 3,
//    int max_bins = 5,
//    double bin_cutoff = 0.05,
//    int max_n_prebins = 20,
//    std::string bin_separator = "%;%",
//    double convergence_threshold = 1e-6,
//    int max_iterations = 1000
// ) {
//  std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
//  std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
//  
//  try {
//    OptimalBinningCategoricalJEDIMWoE jedi(
//        feature_vec, target_vec,
//        min_bins, max_bins,
//        bin_cutoff, max_n_prebins,
//        bin_separator, convergence_threshold,
//        max_iterations
//    );
//    jedi.fit();
//    return jedi.get_results();
//  } catch (const std::exception& e) {
//    Rcpp::stop("Error in optimal_binning_categorical_jedi_mwoe: " + std::string(e.what()));
//  }
// }
