#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <string>
#include <limits>
#include <stdexcept>

/*
 * optimal_binning_numerical_fastMDLPM
 *
 * Implementa discretização numérica baseada no critério MDLP (Fayyad & Irani, 1993)
 * com prioridade de monotonicidade WOE opcional.
 *
 * Fluxo:
 * 1. Pré-processamento:
 *    - Remove NAs do feature e target
 *    - Ordena dados pelo feature
 *    - Calcula prefix sums de positivos e negativos
 * 2. MDLP:
 *    - Aplica recursivamente o critério MDLP para encontrar splits ótimos.
 *    - O MDLP busca o melhor ponto de corte que minimiza a entropia condicional, 
 *      penalizando complexidade.
 * 3. Ajuste de Bins:
 *    - Se houver menos bins que min_bins, tentar aumentar (divisão uniforme)
 *    - Se houver mais bins que max_bins, opcionalmente unir bins mais próximos (simples)
 *      (Esta etapa não é o foco principal, mas aqui faremos merges básicos, se necessário)
 * 4. Monotonicidade do WOE:
 *    - Calcula WOE para cada bin.
 *    - Verifica monotonicidade estrita (crescente ou decrescente).
 *    - Se não monotônico:
 *       * Se force_monotonicity = true, mescla bins adjacentes até monotonicidade.
 *         Pode reduzir o nº de bins abaixo de min_bins se min_bins > 2.
 *       * Se force_monotonicity = false, prioriza manter min_bins mesmo que não haja monotonicidade.
 * 5. Convergência:
 *    - Itera o processo de checagem de monotonicidade até convergência ou max_iterations.
 *
 * Saída:
 * - bin: intervalos no formato (a;b], com -Inf e +Inf
 * - woe, iv, count, count_pos, count_neg, cutpoints, converged, iterations
 *
 * Complexidade esperada: O(n log n) devido à ordenação e busca de pontos de corte.
 *
 * Referências:
 * - Fayyad, U.M. & Irani, K.B. (1993). "Multi-Interval Discretization of Continuous-Valued Attributes for Classification Learning."
 * - Implementações práticas inspiradas em métodos MDLP disponíveis em pacotes de discretização.
 */

// ----------------------------------------------------
// Funções auxiliares
// ----------------------------------------------------

/*
 * Função para checar se um vetor é estritamente crescente.
 */
bool is_strictly_increasing(const std::vector<double> &x) {
  for (size_t i = 1; i < x.size(); i++) {
    if (x[i] <= x[i-1]) return false;
  }
  return true;
}

/*
 * Função para checar se um vetor é estritamente decrescente.
 */
bool is_strictly_decreasing(const std::vector<double> &x) {
  for (size_t i = 1; i < x.size(); i++) {
    if (x[i] >= x[i-1]) return false;
  }
  return true;
}

/*
 * Calcula a entropia binária dada contagem de positivos e negativos.
 * Entropia: E = - p_+ * log2(p_+) - p_- * log2(p_-)
 * Caso p_+ ou p_- sejam zero, considerar esse termo como 0.
 */
double entropy(int count_pos, int count_neg) {
  int total = count_pos + count_neg;
  if (total == 0) return 0.0;
  double p_pos = (double)count_pos / (double)total;
  double p_neg = (double)count_neg / (double)total;
  double E = 0.0;
  if (p_pos > 0) E -= p_pos * std::log2(p_pos);
  if (p_neg > 0) E -= p_neg * std::log2(p_neg);
  return E;
}

/*
 * Calcula a entropia condicional após um corte.
 * Suponha um vetor ordenado por feature e um corte no índice c (entre c e c+1).
 * Usando prefix sums:
 *  - total_pos, total_neg: contagem global
 *  - pos_left, neg_left: contagem no lado esquerdo do corte
 *  - pos_right, neg_right: contagem no lado direito do corte
 */
double conditional_entropy(int pos_left, int neg_left, int pos_right, int neg_right) {
  int total_left = pos_left + neg_left;
  int total_right = pos_right + neg_right;
  int total = total_left + total_right;
  if (total == 0) return 0.0;
  double E_left = entropy(pos_left, neg_left);
  double E_right = entropy(pos_right, neg_right);
  double E_cond = ((double)total_left/(double)total)*E_left + ((double)total_right/(double)total)*E_right;
  return E_cond;
}

/*
 * Critério MDLP:
 * MDLP define um teste para decidir se um corte é válido ou não.
 * Definição (Fayyad & Irani, 1993):
 * O ganho de informação (IG) = E(parent) - [ (N_L/N)*E(L) + (N_R/N)*E(R) ].
 * Há um critério de parada baseado no MDL principle.
 * Se o ganho de informação for maior que um certo limiar MDL, então o corte é feito.
 * Caso contrário, não.
 * Simplificado: Se o split reduz significativamente a entropia, faz o split.
 */

bool mdlp_stop_criterion(int pos_left, int neg_left, int pos_right, int neg_right, int total_pos, int total_neg) {
  // Calcular entropias
  double E_parent = entropy(total_pos, total_neg);
  double E_child = conditional_entropy(pos_left, neg_left, pos_right, neg_right);
  
  double IG = E_parent - E_child; // Gain de informação
  
  // Número de classes = 2 (binário)
  // MDLP threshold:
  // Delta = log2(3^k - 2) - [k*E(parent)] 
  // para binário k=1: log2(3^1 - 2)=log2(1)=0
  // Adaptando do paper (Fayyad & Irani):
  // Usaremos a implementação típica do MDLP: 
  // Delta = log2(3^k - 2) - [k * E_parent]
  // Para k=1: Delta = 0 - E_parent
  // Mas essa formulação simplificada pode variar. Ver implementações do MDLP.
  // Neste caso, k = número de partições resultantes do split = 2
  // O original sugere:
  // Delta = log2((3^k) - 2) - [k*sum(pi)*E(S')]
  // Para 2 classes: 3^2 -2 = 7. log2(7) ~ 2.8074
  // Ajuste do critério:
  // We can use the standard MDLP stopping criterion as implemented widely.
  // Let's define a commonly used threshold:
  // Delta = log2((double)std::pow(3,2)-2) - (2*E_parent) = log2(7)-2*E_parent
  double Delta = std::log2(7.0) - 2.0*E_parent;
  int N = total_pos + total_neg;
  double criterion = (IG - (std::log2((double)N - 1.0)/N)) - (Delta/N);
  
  // Note: Original MDLP uses a more complex formula. A common approximation:
  // Stop if IG <= (log2(N - 1) / N) + (Delta / N)
  // If IG > (log2(N - 1) / N) + (Delta / N), then we do the split.
  // Ajustando:
  double threshold = (std::log2((double)N - 1.0)/N) + (Delta/(double)N);
  
  return (IG <= threshold);
}

/*
 * Função recursiva MDLP:
 * Aplica o critério MDLP para encontrar splits ótimos.
 * Retorna um vetor de índices de corte.
 */
void mdlp_recursion(
    const std::vector<int> &target_sorted,
    const std::vector<double> &feature_sorted,
    int start,
    int end,
    const std::vector<int> &prefix_pos,
    const std::vector<int> &prefix_neg,
    std::vector<int> &splits
) {
  // Contagens globais no segmento [start, end]
  int pos_total = prefix_pos[end] - prefix_pos[start];
  int neg_total = prefix_neg[end] - prefix_neg[start];
  if ((end - start) <= 1) {
    // Muito poucos pontos para subdividir
    return;
  }
  
  // Tentar encontrar melhor split que maximize IG
  double best_IG = -std::numeric_limits<double>::infinity();
  int best_split = -1;
  
  double E_parent = entropy(pos_total, neg_total);
  int N = pos_total + neg_total;
  
  // Buscar split
  for (int i = start; i < end - 1; i++) {
    // Apenas split se feature mudar de valor
    if (feature_sorted[i] == feature_sorted[i+1]) continue;
    
    int pos_left = prefix_pos[i+1] - prefix_pos[start];
    int neg_left = prefix_neg[i+1] - prefix_neg[start];
    int pos_right = pos_total - pos_left;
    int neg_right = neg_total - neg_left;
    
    double E_child = conditional_entropy(pos_left, neg_left, pos_right, neg_right);
    double IG = E_parent - E_child;
    
    if (IG > best_IG) {
      best_IG = IG;
      best_split = i;
    }
  }
  
  if (best_split == -1) {
    // Nenhum split possível (valores iguais?)
    return;
  }
  
  // Testar critério MDLP
  int pos_left = prefix_pos[best_split+1] - prefix_pos[start];
  int neg_left = prefix_neg[best_split+1] - prefix_neg[start];
  int pos_right = pos_total - pos_left;
  int neg_right = neg_total - neg_left;
  
  // Verificar se o split atende critério MDLP
  if (!mdlp_stop_criterion(pos_left, neg_left, pos_right, neg_right, pos_total, neg_total)) {
    // Critério não satisfeito, não faz split
    return;
  }
  
  // Se sim, adiciona o split
  splits.push_back(best_split);
  // Recursão esquerda
  mdlp_recursion(target_sorted, feature_sorted, start, best_split+1, prefix_pos, prefix_neg, splits);
  // Recursão direita
  mdlp_recursion(target_sorted, feature_sorted, best_split+1, end, prefix_pos, prefix_neg, splits);
}

/*
 * Dada uma partição, calcular as métricas dos bins:
 * count, count_pos, count_neg, woe, iv
 * cutpoints: pontos internos
 */
void calc_bins_metrics(
    const std::vector<double> &feature_sorted,
    const std::vector<int> &target_sorted,
    const std::vector<int> &prefix_pos,
    const std::vector<int> &prefix_neg,
    const std::vector<int> &splits,
    std::vector<int> &counts,
    std::vector<int> &pos_counts,
    std::vector<int> &neg_counts,
    std::vector<double> &woe,
    std::vector<double> &iv,
    std::vector<double> &cutpoints
) {
  int start = 0;
  int total_pos = prefix_pos.back();
  int total_neg = prefix_neg.back();
  
  // Se não houver splits, um único bin
  std::vector<int> boundaries;
  boundaries.push_back(0);
  for (size_t i = 0; i < splits.size(); i++) {
    boundaries.push_back(splits[i]+1);
  }
  boundaries.push_back((int)feature_sorted.size());
  
  for (size_t i = 0; i < boundaries.size()-1; i++) {
    int s = boundaries[i];
    int e = boundaries[i+1];
    int c_pos = prefix_pos[e] - prefix_pos[s];
    int c_neg = prefix_neg[e] - prefix_neg[s];
    int c_total = c_pos + c_neg;
    
    counts.push_back(c_total);
    pos_counts.push_back(c_pos);
    neg_counts.push_back(c_neg);
    
    double pct_pos = (total_pos > 0) ? (double)c_pos / (double)total_pos : 1e-12;
    double pct_neg = (total_neg > 0) ? (double)c_neg / (double)total_neg : 1e-12;
    
    double w = std::log(pct_pos/pct_neg);
    woe.push_back(w);
    
    double iv_part = (pct_pos - pct_neg)*w;
    iv.push_back(iv_part);
    
    // cutpoint: valor final do bin i (exceto último)
    // Definiremos cutpoints como o limite superior do bin (o maior valor no bin)
    // Exceto o último bin, que terá +Inf
    if (i < boundaries.size()-2) {
      // Ponto de corte é o valor feature_sorted no último elemento do bin atual
      double cp = feature_sorted[e-1];
      cutpoints.push_back(cp);
    }
  }
}

/*
 * Função para forçar um certo número mínimo de bins se MDLP retornar menos que min_bins:
 * Divisão uniforme (equal frequency) simples.
 */
void force_min_bins(
    const std::vector<double> &feature_sorted,
    const std::vector<int> &target_sorted,
    int min_bins,
    const std::vector<int> &prefix_pos,
    const std::vector<int> &prefix_neg,
    std::vector<int> &splits
) {
  // Se o número de bins atual for menor que min_bins, dividir uniformemente.
  // Suponha que o total seja N, dividimos em quantis aproximadamente iguais.
  int current_bins = (int)splits.size() + 1;
  if (current_bins >= min_bins) return;
  
  int N = (int)feature_sorted.size();
  splits.clear();
  // Criar min_bins a partir de quantis aproximados
  for (int i = 1; i < min_bins; i++) {
    int idx = (int)std::floor((double)i* (double)N/(double)min_bins);
    if (idx > 0 && idx < N) {
      splits.push_back(idx-1);
    }
  }
}

/*
 * Função para reduzir número de bins se exceder max_bins:
 * Estratégia simples: unir bins adjacentes com menor diferença em woe ou low count
 */
void enforce_max_bins(
    int max_bins,
    const std::vector<double> &feature_sorted,
    const std::vector<int> &target_sorted,
    const std::vector<int> &prefix_pos,
    const std::vector<int> &prefix_neg,
    std::vector<int> &splits
) {
  int current_bins = (int)splits.size() + 1;
  if (current_bins <= max_bins) return;
  // Unir bins adjacentes simples: cada união remove um bin
  while (current_bins > max_bins && current_bins > 1) {
    // Escolher a fusão mais simples: unir os dois menores bins consecutivos
    // ou unir pelo critério do menor tamanho total
    // Simplesmente unir o último bin (mais a direita)
    // (Implementação simplificada)
    splits.pop_back();
    current_bins--;
  }
}

/*
 * Função para checar e impor monotonicidade do WOE.
 * Tenta mesclar bins adjacentes até obter monotonicidade.
 * Se force_monotonicity = true, monotonicidade é prioridade,
 * podendo reduzir abaixo de min_bins se min_bins > 2.
 * Se min_bins = 2, não pode ir abaixo de 2 bins.
 *
 * Retorna true se convergiu (monotonicidade obtida ou desistência), false se não.
 */
bool enforce_monotonicity(
    std::vector<int> &splits,
    std::vector<int> &counts,
    std::vector<int> &pos_counts,
    std::vector<int> &neg_counts,
    std::vector<double> &woe,
    std::vector<double> &iv,
    std::vector<double> &cutpoints,
    bool force_monotonicity,
    int min_bins
) {
  // Verificar monotonicidade
  if (woe.size() <= 1) {
    // com 0 ou 1 bin, já é monotônico
    return true;
  }
  
  bool is_mono_inc = is_strictly_increasing(woe);
  bool is_mono_dec = is_strictly_decreasing(woe);
  
  if (is_mono_inc || is_mono_dec) {
    return true; // Já monotônico
  }
  
  if (!force_monotonicity) {
    // Não forçar monotonicidade além do min_bins
    // Então paramos, sem mexer
    return true; // Retorna true pois não tentaremos mais (é uma forma de convergência)
  }
  
  // Forçar monotonicidade mesclando bins adjacentes
  // Estratégia:
  // - Enquanto não monotônico, unir o par de bins adjacentes com menor impacto
  //   no total, por ex. unir o par de bins mais próximo em termos de índice.
  // - Refazer o WOE e checar novamente.
  // - Se o número de bins cai abaixo de min_bins e min_bins > 2, permitir. 
  //   Se min_bins = 2, não pode ir abaixo de 2 bins.
  
  // Monotonicidade estrita requer que cada passo mescle um par até monotonia.
  // Aqui fazemos um loop até monotonia ou mínimo de bins atingir um limite.
  
  // Função interna para refazer métricas após fusão:
  auto recompute_metrics = [&](const std::vector<int> &boundaries) {
    // Recalcular woe, iv e cutpoints
    std::vector<int> new_counts;
    std::vector<int> new_pos_counts;
    std::vector<int> new_neg_counts;
    std::vector<double> new_woe;
    std::vector<double> new_iv;
    std::vector<double> new_cutpoints;
    
    int total_pos = 0; for (auto &x: pos_counts) total_pos += x;
    int total_neg = 0; for (auto &x: neg_counts) total_neg += x;
    
    for (size_t i = 0; i < boundaries.size()-1; i++) {
      int c_pos = 0;
      int c_neg = 0;
      for (int b = boundaries[i]; b < boundaries[i+1]; b++) {
        c_pos += pos_counts[b];
        c_neg += neg_counts[b];
      }
      int c_total = c_pos + c_neg;
      new_counts.push_back(c_total);
      new_pos_counts.push_back(c_pos);
      new_neg_counts.push_back(c_neg);
      
      double pct_pos = (total_pos > 0) ? (double)c_pos / (double)total_pos : 1e-12;
      double pct_neg = (total_neg > 0) ? (double)c_neg / (double)total_neg : 1e-12;
      double w = std::log(pct_pos / pct_neg);
      new_woe.push_back(w);
      double iv_part = (pct_pos - pct_neg)*w;
      new_iv.push_back(iv_part);
      if (i < boundaries.size()-2) {
        // último ponto do grupo i
        new_cutpoints.push_back(cutpoints[boundaries[i+1]-2 < 0 ? 0 : boundaries[i+1]-2]);
      }
    }
    
    counts = new_counts;
    pos_counts = new_pos_counts;
    neg_counts = new_neg_counts;
    woe = new_woe;
    iv = new_iv;
    cutpoints = new_cutpoints;
  };
  
  // Precisamos mapear splits para bins, depois mesclar bins adjacentes.
  // splits define as fronteiras internas:
  // se há k splits, temos k+1 bins.
  // Cada bin corresponde a um único índice em counts, pos_counts, etc.
  // Ao mesclar bins i e i+1, unimos seus counts.
  // Aqui, porém, já temos as métricas calculadas, então uniremos direto.
  
  // Cria um vetor de índices [0,1,2,...,n_bins-1] representando bins
  // Ao mesclar bins, atualizar essas estruturas.
  int n_bins = (int)woe.size();
  std::vector<int> bin_map(n_bins);
  for (int i = 0; i < n_bins; i++) {
    bin_map[i] = i;
  }
  
  // boundaries: cada elemento corresponde a um bin original
  // Ao mesclar, um bin final conterá vários bins originais
  // No momento, cada bin final é formado por um único bin original.
  // Ao mesclar, uniremos adjacentes.
  std::vector<int> boundaries;
  for (int i = 0; i <= n_bins; i++) {
    boundaries.push_back(i);
  }
  
  // Tentar mesclar até monotonia ou limite
  for (int iteration = 0; iteration < 1000; iteration++) {
    if (woe.size() <= 1) {
      // Já monotônico
      return true;
    }
    
    bool inc = is_strictly_increasing(woe);
    bool dec = is_strictly_decreasing(woe);
    
    if (inc || dec) {
      return true;
    }
    
    // Não monotônico, tentar mesclar um par
    int current_bins = (int)woe.size();
    if (current_bins <= 2 && min_bins == 2) {
      // Não podemos ir abaixo de 2 se min_bins=2
      // parar
      return true;
    }
    
    // Se min_bins > 2, podemos reduzir abaixo se necessário
    // Mesclar o par com menor diferença em WOE ou simplesmente o par i,i+1
    // mais "suave".
    double min_diff = std::numeric_limits<double>::infinity();
    int merge_pos = -1;
    for (int i = 0; i < current_bins - 1; i++) {
      double diff = std::fabs(woe[i+1] - woe[i]);
      if (diff < min_diff) {
        min_diff = diff;
        merge_pos = i;
      }
    }
    
    // Mesclar bins merge_pos e merge_pos+1
    // boundaries descreve os bins finais:
    // bin i é [boundaries[i], boundaries[i+1]) em termos de bins originais
    // após mescla, boundaries ficará com um bin a menos
    boundaries.erase(boundaries.begin() + merge_pos + 1);
    
    // Recalcular métricas
    recompute_metrics(boundaries);
    
    // Checar se não violamos min_bins
    if ((int)woe.size() < min_bins && min_bins == 2) {
      // Precisamos pelo menos 2 bins
      // Se caiu para 1 bin, paramos.
      return true;
    }
    
    // Se force_monotonicity=true e min_bins>2, podemos aceitar menos bins se preciso
    // Portanto, não checamos se caiu abaixo de min_bins nessa situação.
  }
  
  return false; // Se passar do loop, dizer que não convergiu, mas isso é improvável.
}

/*
 * Função principal:
 * - target: vetor binário 0/1
 * - feature: numérico contínuo (NAs removidos)
 * - min_bins, max_bins
 * - force_monotonicity
 * Retorna lista com bins, woe, iv, etc.
 */

// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_fastMDLPM(
    Rcpp::IntegerVector target,
    Rcpp::NumericVector feature,
    int min_bins = 2,
    int max_bins = 5,
    double bin_cutoff = 0.05, // não utilizado aqui, mas deixado para extensões
    int max_n_prebins = 100, // não usado detalhadamente, espaço para prebinagem
    double convergence_threshold = 1e-6,
    int max_iterations = 1000,
    bool force_monotonicity = true
) {
  // Verificações iniciais
  if (target.size() != feature.size()) {
    Rcpp::stop("target and feature must have the same length.");
  }
  if (min_bins < 2) {
    Rcpp::stop("min_bins must be >= 2.");
  }
  if (max_bins < min_bins) {
    Rcpp::stop("max_bins must be >= min_bins.");
  }
  
  // Remover NAs
  std::vector<double> feat_tmp;
  std::vector<int> targ_tmp;
  for (int i = 0; i < target.size(); i++) {
    if (!Rcpp::NumericVector::is_na(feature[i])) {
      feat_tmp.push_back(feature[i]);
      targ_tmp.push_back(target[i]);
    }
  }
  
  int N = (int)feat_tmp.size();
  if (N == 0) {
    // Sem dados após remover NAs
    return Rcpp::List::create(
      Rcpp::Named("id") = Rcpp::NumericVector(),
      Rcpp::Named("bin") = Rcpp::CharacterVector(),
      Rcpp::Named("woe") = Rcpp::NumericVector(),
      Rcpp::Named("iv") = Rcpp::NumericVector(),
      Rcpp::Named("count") = Rcpp::IntegerVector(),
      Rcpp::Named("count_pos") = Rcpp::IntegerVector(),
      Rcpp::Named("count_neg") = Rcpp::IntegerVector(),
      Rcpp::Named("cutpoints") = Rcpp::NumericVector(),
      Rcpp::Named("converged") = false,
      Rcpp::Named("iterations") = 0
    );
  }
  
  // Ordenar pelo feature
  std::vector<int> idx(N);
  for (int i=0; i<N; i++) idx[i]=i;
  std::sort(idx.begin(), idx.end(), [&](int a, int b) {
    return feat_tmp[a]<feat_tmp[b];
  });
  
  std::vector<double> feature_sorted(N);
  std::vector<int> target_sorted(N);
  for (int i=0; i<N; i++) {
    feature_sorted[i] = feat_tmp[idx[i]];
    target_sorted[i] = targ_tmp[idx[i]];
  }
  
  // Calcular prefix sums
  std::vector<int> prefix_pos(N+1,0);
  std::vector<int> prefix_neg(N+1,0);
  for (int i=0; i<N; i++) {
    prefix_pos[i+1] = prefix_pos[i] + (target_sorted[i]==1?1:0);
    prefix_neg[i+1] = prefix_neg[i] + (target_sorted[i]==0?1:0);
  }
  
  // Verificar número de valores únicos:
  // Se for 1 valor único, um único bin
  // Se precisar min_bins > 1, dividir uniformemente
  {
    bool all_equal = true;
    for (int i = 1; i < N; i++) {
      if (feature_sorted[i] != feature_sorted[0]) {
        all_equal = false;
        break;
      }
    }
    if (all_equal) {
      // Todos iguais
      // Teremos um único bin
      // Se min_bins > 1, dividir uniformemente (não faz sentido, mas exigido)
      std::vector<int> splits;
      if (min_bins > 1) {
        for (int i = 1; i < min_bins; i++) {
          int idx_cut = (int)floor((double)i*N/(double)min_bins);
          if (idx_cut > 0 && idx_cut < N) splits.push_back(idx_cut-1);
        }
      }
      
      // Calcular métricas
      std::vector<int> counts, pos_counts, neg_counts;
      std::vector<double> woe, iv, cutpoints;
      calc_bins_metrics(feature_sorted,target_sorted,prefix_pos,prefix_neg,splits,
                        counts,pos_counts,neg_counts,woe,iv,cutpoints);
      
      // Sem monotonicidade a checar (é um só bin ou poucos)
      // Construir saída
      Rcpp::CharacterVector bin_names;
      {
        // Montar intervalos (a partir do splits)
        // Se for k splits, k+1 bins
        int nb = (int)woe.size();
        double lower = -std::numeric_limits<double>::infinity();
        for (int b = 0; b < nb; b++) {
          double upper = (b < (int)cutpoints.size()) ? cutpoints[b] : std::numeric_limits<double>::infinity();
          std::string interval = "(" + (std::isinf(lower)?"-Inf":std::to_string(lower)) + ";" + (std::isinf(upper)?"+Inf":std::to_string(upper)) + "]";
          bin_names.push_back(interval);
          lower = upper;
        }
      }
      
      // Somar IV total
      double total_iv=0.0;
      for (auto &x: iv) total_iv+=x;
      
      Rcpp::NumericVector ids(bin_names.size());
      for(int i = 0; i < bin_names.size(); i++) {
        ids[i] = i + 1;
      }
      
      return Rcpp::List::create(
        Rcpp::Named("id") = ids,
        Rcpp::Named("bin")=bin_names,
        Rcpp::Named("woe")=Rcpp::wrap(woe),
        Rcpp::Named("iv")=Rcpp::wrap(iv),
        Rcpp::Named("count")=Rcpp::wrap(counts),
        Rcpp::Named("count_pos")=Rcpp::wrap(pos_counts),
        Rcpp::Named("count_neg")=Rcpp::wrap(neg_counts),
        Rcpp::Named("cutpoints")=Rcpp::wrap(cutpoints),
        Rcpp::Named("converged")=true,
        Rcpp::Named("iterations")=0
      );
    }
  }
  
  // Aplicar MDLP recursivo
  std::vector<int> splits;
  mdlp_recursion(target_sorted, feature_sorted, 0, N, prefix_pos, prefix_neg, splits);
  std::sort(splits.begin(), splits.end());
  
  // Ajustar min_bins
  {
    int current_bins = (int)splits.size()+1;
    if (current_bins < min_bins) {
      force_min_bins(feature_sorted,target_sorted,min_bins,prefix_pos,prefix_neg,splits);
    }
  }
  
  // Ajustar max_bins
  {
    int current_bins = (int)splits.size()+1;
    if (current_bins > max_bins) {
      enforce_max_bins(max_bins,feature_sorted,target_sorted,prefix_pos,prefix_neg,splits);
    }
  }
  
  // Calcular métricas iniciais
  std::vector<int> counts, pos_counts, neg_counts;
  std::vector<double> woe, iv, cutpoints;
  calc_bins_metrics(feature_sorted,target_sorted,prefix_pos,prefix_neg,splits,
                    counts,pos_counts,neg_counts,woe,iv,cutpoints);
  
  // Verificar monotonicidade
  bool converged = false;
  int iterations = 0;
  // Guardar WOE anterior para convergência
  std::vector<double> old_woe = woe;
  
  for (iterations = 0; iterations < max_iterations; iterations++) {
    bool mono_res = enforce_monotonicity(splits, counts, pos_counts, neg_counts, woe, iv, cutpoints, force_monotonicity, min_bins);
    // Calcular diferença média absoluta woe
    double diff = 0.0;
    {
      size_t len = std::min(old_woe.size(), woe.size());
      for (size_t i = 0; i < len; i++) {
        diff += std::fabs(old_woe[i]-woe[i]);
      }
      if (len>0) diff/= (double)len;
    }
    if (mono_res && diff < convergence_threshold) {
      converged = true;
      break;
    }
    old_woe = woe;
  }
  
  // Montar intervalos finais
  Rcpp::CharacterVector bin_names;
  {
    int nb = (int)woe.size();
    double lower = -std::numeric_limits<double>::infinity();
    for (int b = 0; b < nb; b++) {
      double upper = (b < (int)cutpoints.size()) ? cutpoints[b] : std::numeric_limits<double>::infinity();
      std::string interval = "(" + (std::isinf(lower)?"-Inf":std::to_string(lower)) + ";" + (std::isinf(upper)?"+Inf":std::to_string(upper)) + "]";
      bin_names.push_back(interval);
      lower = upper;
    }
  }
  
  // IV total
  double total_iv=0.0;
  for (auto &x: iv) total_iv+=x;
  
  Rcpp::NumericVector ids(bin_names.size());
  for(int i = 0; i < bin_names.size(); i++) {
    ids[i] = i + 1;
  }
  
  return Rcpp::List::create(
    Rcpp::Named("id") = ids,
    Rcpp::Named("bin")=bin_names,
    Rcpp::Named("woe")=Rcpp::wrap(woe),
    Rcpp::Named("iv")=Rcpp::wrap(iv),
    Rcpp::Named("count")=Rcpp::wrap(counts),
    Rcpp::Named("count_pos")=Rcpp::wrap(pos_counts),
    Rcpp::Named("count_neg")=Rcpp::wrap(neg_counts),
    Rcpp::Named("cutpoints")=Rcpp::wrap(cutpoints),
    Rcpp::Named("converged")=converged,
    Rcpp::Named("iterations")=iterations
  );
}
