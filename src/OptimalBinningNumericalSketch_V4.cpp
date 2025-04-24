// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>
#include <stdexcept>
#include <memory>
#include <random>
#include <queue>

using namespace Rcpp;

// Constantes globais para melhor legibilidade e consistência
static constexpr double EPSILON = 1e-10;
static constexpr double NEG_INFINITY = -std::numeric_limits<double>::infinity();

// Namespace com funções auxiliares otimizadas
namespace utils {
// Função logaritmo segura e otimizada
inline double safe_log(double x) {
  return x > EPSILON ? std::log(x) : std::log(EPSILON);
}
}

// Estrutura KLL Sketch para aproximação de quantis em streams
class KLLSketch {
private:
  struct Item {
    double value;
    int weight;
    
    Item(double v, int w = 1) : value(v), weight(w) {}
    
    bool operator<(const Item& other) const {
      return value < other.value;
    }
  };
  
  using Compactor = std::vector<Item>;
  std::vector<Compactor> compactors;
  int k;  // Parâmetro que controla a acurácia
  int n;  // Número de itens processados
  double min_value;
  double max_value;
  int max_level;  // Limite de níveis para evitar overflow de pilha
  
  // Função para compactar um nível do sketch - versão não recursiva
  void compact_level(size_t level) {
    std::queue<size_t> levels_to_compact;
    levels_to_compact.push(level);
    
    while (!levels_to_compact.empty()) {
      size_t current_level = levels_to_compact.front();
      levels_to_compact.pop();
      
      // Verificar limite de níveis para evitar expansão infinita
      if (current_level >= static_cast<size_t>(max_level)) {
        continue;
      }
      
      // Se o nível atual não precisa de compactação, continuar
      if (current_level >= compactors.size() || compactors[current_level].size() <= static_cast<size_t>(k)) {
        continue;
      }
      
      // Ordenar compactor
      std::sort(compactors[current_level].begin(), compactors[current_level].end());
      
      // Selecionar itens para o próximo nível (amostragem ponderada)
      Compactor& compactor = compactors[current_level];
      std::vector<Item> next_level;
      next_level.reserve(compactor.size() / 2 + 1);
      
      // Garantir que o próximo nível existe
      if (current_level + 1 >= compactors.size()) {
        compactors.push_back(Compactor());
      }
      
      // Compactação: mesclar pares adjacentes
      for (size_t i = 0; i < compactor.size() - 1; i += 2) {
        // Mesclar par em um único item com peso combinado
        // Para nível par, inclui itens em posição par
        // Para nível ímpar, inclui itens em posição ímpar
        bool include = ((current_level % 2 == 0 && i % 2 == 0) || 
                        (current_level % 2 == 1 && i % 2 == 1));
        
        if (include) {
          next_level.push_back(Item(compactor[i].value, 
                                    compactor[i].weight + compactor[i+1].weight));
        } else {
          next_level.push_back(Item(compactor[i+1].value,
                                    compactor[i].weight + compactor[i+1].weight));
        }
      }
      
      // Se sobrar um elemento (tamanho ímpar)
      if (compactor.size() % 2 == 1) {
        next_level.push_back(compactor.back());
      }
      
      // Limpar nível atual
      compactor.clear();
      
      // Adicionar itens ao próximo nível
      for (const auto& item : next_level) {
        compactors[current_level + 1].push_back(item);
      }
      
      // Adicionar próximo nível à fila se necessário
      if (compactors[current_level + 1].size() > static_cast<size_t>(k)) {
        levels_to_compact.push(current_level + 1);
      }
    }
  }
  
public:
  KLLSketch(int k_param = 200) : k(k_param), n(0), 
    min_value(std::numeric_limits<double>::max()), 
    max_value(std::numeric_limits<double>::lowest()),
    max_level(20) { // Limite razoável para evitar problemas
    compactors.push_back(Compactor());
    compactors[0].reserve(k + 1); // Pré-alocação para melhor desempenho
  }
  
  // Adicionar um valor ao sketch
  void update(double value) {
    // Atualizar min/max exatos
    min_value = std::min(min_value, value);
    max_value = std::max(max_value, value);
    
    // Adicionar ao primeiro compactor
    compactors[0].push_back(Item(value));
    n++;
    
    // Compactar se necessário
    if (compactors[0].size() > static_cast<size_t>(k)) {
      compact_level(0);
    }
  }
  
  // Estimar o q-ésimo quantil (0 <= q <= 1)
  double get_quantile(double q) const {
    if (n == 0) return 0;
    
    // Validação de entrada com correção segura
    if (q <= 0) return min_value;
    if (q >= 1) return max_value;
    
    // Construir representação "achatada" do sketch para consulta
    std::vector<Item> flattened;
    flattened.reserve(n / 2); // Estimativa conservadora
    
    for (const auto& compactor : compactors) {
      flattened.insert(flattened.end(), compactor.begin(), compactor.end());
    }
    
    // Verificação de segurança
    if (flattened.empty()) {
      return min_value;
    }
    
    // Ordenar itens
    std::sort(flattened.begin(), flattened.end());
    
    // Calcular peso total
    int total_weight = 0;
    for (const auto& item : flattened) {
      total_weight += item.weight;
    }
    
    // Verificação de segurança
    if (total_weight <= 0) {
      return min_value;
    }
    
    // Encontrar item que corresponde ao quantil
    int target_weight = static_cast<int>(q * total_weight);
    int cumulative_weight = 0;
    
    for (const auto& item : flattened) {
      cumulative_weight += item.weight;
      if (cumulative_weight >= target_weight) {
        return item.value;
      }
    }
    
    // Fallback para o último item (não deveria chegar aqui)
    return flattened.back().value;
  }
  
  // Retorna o número de itens vistos
  int count() const {
    return n;
  }
  
  // Retorna o valor mínimo (exato)
  double get_min() const {
    return min_value;
  }
  
  // Retorna o valor máximo (exato)
  double get_max() const {
    return max_value;
  }
  
  // Retorna os itens compactados para inspeção
  std::vector<double> get_items() const {
    std::vector<double> result;
    for (const auto& compactor : compactors) {
      for (const auto& item : compactor) {
        result.push_back(item.value);
      }
    }
    std::sort(result.begin(), result.end());
    return result;
  }
};

// Estrutura para armazenar estatísticas de target por ponto de corte
struct CutpointStats {
  double cutpoint;
  int count_below;     // Contagem total abaixo do ponto de corte
  int count_pos_below; // Contagem de positivos abaixo do ponto de corte
  int count_neg_below; // Contagem de negativos abaixo do ponto de corte
  int count_above;     // Contagem total acima do ponto de corte
  int count_pos_above; // Contagem de positivos acima do ponto de corte
  int count_neg_above; // Contagem de negativos acima do ponto de corte
  double iv;           // Information Value calculado para este ponto de corte
  
  CutpointStats(double cp = 0.0) : cutpoint(cp), count_below(0), count_pos_below(0),
    count_neg_below(0), count_above(0), count_pos_above(0),
    count_neg_above(0), iv(0.0) {}
};

// Estrutura Bin para variáveis numéricas
struct NumericBin {
  double lower_bound;
  double upper_bound;
  int count;
  int count_pos;
  int count_neg;
  double woe;
  double iv;
  double event_rate;
  
  NumericBin(double lb = -std::numeric_limits<double>::infinity(), 
             double ub = std::numeric_limits<double>::infinity()) 
    : lower_bound(lb), upper_bound(ub), count(0), count_pos(0), count_neg(0),
      woe(0.0), iv(0.0), event_rate(0.0) {}
  
  // Adicionar um valor ao bin
  void add_value(int is_positive) {
    count++;
    if (is_positive) {
      count_pos++;
    } else {
      count_neg++;
    }
    update_event_rate();
  }
  
  // Adicionar múltiplos valores com contagens
  void add_counts(int pos_count, int neg_count) {
    count += (pos_count + neg_count);
    count_pos += pos_count;
    count_neg += neg_count;
    update_event_rate();
  }
  
  // Combinar com outro bin
  void merge_with(const NumericBin& other) {
    lower_bound = std::min(lower_bound, other.lower_bound);
    upper_bound = std::max(upper_bound, other.upper_bound);
    count += other.count;
    count_pos += other.count_pos;
    count_neg += other.count_neg;
    update_event_rate();
  }
  
  // Atualizar taxa de eventos
  void update_event_rate() {
    event_rate = count > 0 ? static_cast<double>(count_pos) / count : 0.0;
  }
  
  // Cálculo de WoE e IV
  void calculate_metrics(int total_good, int total_bad) {
    double prop_event = static_cast<double>(count_pos) / std::max(total_bad, 1);
    double prop_non_event = static_cast<double>(count_neg) / std::max(total_good, 1);
    
    prop_event = std::max(prop_event, EPSILON);
    prop_non_event = std::max(prop_non_event, EPSILON);
    
    woe = utils::safe_log(prop_event / prop_non_event);
    iv = (prop_event - prop_non_event) * woe;
  }
  
  // Verificar se o valor está no bin
  bool contains(double value) const {
    return value >= lower_bound && value <= upper_bound;
  }
};

// Classe para optimizar o binning com programação dinâmica
class DynamicProgramming {
private:
  std::vector<double> sorted_values;
  std::vector<int> target_values;
  std::vector<std::vector<double>> dp_table;
  std::vector<std::vector<int>> split_points;
  int total_good;
  int total_bad;
  std::unordered_map<std::string, double> iv_cache; // Cache para cálculos de IV
  
  // Função para calcular chave de cache
  std::string get_cache_key(int i, int j) const {
    return std::to_string(i) + "_" + std::to_string(j);
  }
  
  // Calcular o IV de um bin entre os índices i e j
  double calculate_bin_iv(int i, int j) {
    // Usar cache para evitar recálculos
    std::string key = get_cache_key(i, j);
    if (iv_cache.find(key) != iv_cache.end()) {
      return iv_cache[key];
    }
    
    // Verificação de limites
    if (i < 0 || j < i || j >= static_cast<int>(sorted_values.size())) {
      return 0.0;
    }
    
    int bin_count_pos = 0;
    int bin_count_neg = 0;
    
    for (int k = i; k <= j; ++k) {
      if (target_values[k] == 1) {
        bin_count_pos++;
      } else {
        bin_count_neg++;
      }
    }
    
    double prop_event = static_cast<double>(bin_count_pos) / std::max(total_bad, 1);
    double prop_non_event = static_cast<double>(bin_count_neg) / std::max(total_good, 1);
    
    prop_event = std::max(prop_event, EPSILON);
    prop_non_event = std::max(prop_non_event, EPSILON);
    
    double woe = utils::safe_log(prop_event / prop_non_event);
    double iv = (prop_event - prop_non_event) * woe;
    
    // Armazenar no cache
    iv_cache[key] = std::fabs(iv);
    return iv_cache[key];
  }
  
public:
  DynamicProgramming(const std::vector<double>& values, const std::vector<int>& targets) 
    : total_good(0), total_bad(0) {
    
    // Verificação de tamanho vazio
    if (values.empty()) {
      return;
    }
    
    // Criar vetores temporários de pares ordenáveis
    std::vector<std::pair<double, int>> paired_data;
    paired_data.reserve(values.size());
    
    for (size_t i = 0; i < values.size(); ++i) {
      paired_data.push_back(std::make_pair(values[i], targets[i]));
    }
    
    // Ordenar os pares
    std::sort(paired_data.begin(), paired_data.end());
    
    // Preencher os vetores ordenados
    sorted_values.reserve(paired_data.size());
    target_values.reserve(paired_data.size());
    
    // Extrair dados ordenados
    for (const auto& pair : paired_data) {
      sorted_values.push_back(pair.first);
      target_values.push_back(pair.second);
      
      if (pair.second == 1) {
        total_bad++;
      } else {
        total_good++;
      }
    }
  }
  
  // Encontrar os k-1 pontos de corte ótimos para k bins
  std::vector<double> optimize(int k) {
    int n = sorted_values.size();
    
    // Verificação de segurança para entradas inválidas
    if (n <= 1 || k <= 1) {
      return std::vector<double>();
    }
    
    // Limitar k para evitar problemas de memória
    k = std::min(k, std::min(n, 50));
    
    // Inicializar tabela DP e pontos de corte
    try {
      dp_table.resize(n + 1);
      split_points.resize(n + 1);
      
      for (int i = 0; i <= n; ++i) {
        dp_table[i].resize(k + 1, -1.0);
        split_points[i].resize(k + 1, -1);
      }
    } catch (const std::bad_alloc& e) {
      // Fallback para o caso de problema de alocação de memória
      Rcpp::warning("Problema de alocação de memória na programação dinâmica. Usando abordagem simplificada.");
      return fallback_optimize(k);
    }
    
    // Caso base: 0 valores ou 1 bin
    for (int i = 0; i <= n; ++i) {
      dp_table[i][1] = i > 0 ? calculate_bin_iv(0, i - 1) : 0.0;
    }
    
    // Preencher tabela DP
    for (int j = 2; j <= k; ++j) {
      for (int i = j; i <= n; ++i) {
        dp_table[i][j] = -1.0;
        
        for (int l = j - 1; l < i; ++l) {
          double current_iv = dp_table[l][j - 1] + calculate_bin_iv(l, i - 1);
          
          if (current_iv > dp_table[i][j]) {
            dp_table[i][j] = current_iv;
            split_points[i][j] = l;
          }
        }
      }
    }
    
    // Recuperar pontos de corte ótimos
    std::vector<int> optimal_splits;
    int i = n;
    int j = k;
    
    while (j > 1 && i > 0) {
      if (split_points[i][j] < 0) break; // Verificação de segurança
      
      optimal_splits.push_back(split_points[i][j]);
      i = split_points[i][j];
      j--;
    }
    
    // Converter índices para valores reais de pontos de corte
    std::vector<double> cutpoints;
    for (int split : optimal_splits) {
      if (split > 0 && split < n) {
        // Ponto médio entre valores adjacentes como ponto de corte
        cutpoints.push_back((sorted_values[split - 1] + sorted_values[split]) / 2.0);
      }
    }
    
    std::sort(cutpoints.begin(), cutpoints.end());
    return cutpoints;
  }
  
  // Método alternativo (fallback) para quando a DP falha
  std::vector<double> fallback_optimize(int k) {
    std::vector<double> cutpoints;
    int n = sorted_values.size();
    
    if (n <= 1 || k <= 1) {
      return cutpoints;
    }
    
    // Usar quantis uniformes quando a DP falha
    int step = n / k;
    if (step < 1) step = 1;
    
    for (int i = 1; i < k && (i * step) < n; ++i) {
      int idx = i * step;
      cutpoints.push_back(sorted_values[idx]);
    }
    
    return cutpoints;
  }
};

// Classe principal para binning ótimo numérico com sketch
class OptimalBinningNumericalSketch {
private:
  std::vector<double> feature;
  std::vector<int> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  std::string special_codes;
  bool monotonic;
  double convergence_threshold;
  int max_iterations;
  int sketch_k;
  int dp_size_limit; // Limite para usar programação dinâmica
  
  int total_good;
  int total_bad;
  
  std::unique_ptr<KLLSketch> sketch;
  std::vector<NumericBin> bins;
  std::vector<double> cutpoints;
  
  // Validação de entradas
  void validate_inputs() {
    if (feature.size() != target.size()) {
      throw std::invalid_argument("Feature e target devem ter o mesmo tamanho.");
    }
    if (feature.empty()) {
      throw std::invalid_argument("Feature e target não podem ser vazios.");
    }
    if (min_bins < 2) {
      throw std::invalid_argument("min_bins deve ser >= 2.");
    }
    if (max_bins < min_bins) {
      throw std::invalid_argument("max_bins deve ser >= min_bins.");
    }
    if (bin_cutoff <= 0 || bin_cutoff >= 1) {
      throw std::invalid_argument("bin_cutoff deve estar entre 0 e 1.");
    }
    
    // Verificação eficiente dos valores de target
    bool has_zero = false;
    bool has_one = false;
    
    for (int val : target) {
      if (val == 0) has_zero = true;
      else if (val == 1) has_one = true;
      else throw std::invalid_argument("Target deve conter apenas 0 e 1.");
      
      // Early termination
      if (has_zero && has_one) break;
    }
    
    if (!has_zero || !has_one) {
      throw std::invalid_argument("Target deve conter tanto 0 quanto 1.");
    }
  }
  
  // Construir sketch KLL
  void build_sketch() {
    sketch = std::make_unique<KLLSketch>(sketch_k);
    
    total_good = 0;
    total_bad = 0;
    
    // Atualizar sketch em uma única passagem
    for (size_t i = 0; i < feature.size(); ++i) {
      sketch->update(feature[i]);
      
      if (target[i] == 1) {
        total_bad++;
      } else {
        total_good++;
      }
    }
  }
  
  // Extração de candidatos a pontos de corte do sketch
  std::vector<double> extract_candidates() {
    // Verificar se sketch foi inicializado
    if (!sketch || sketch->count() == 0) {
      throw std::runtime_error("Sketch não inicializado ou vazio.");
    }
    
    // Extrair quantis aproximados do sketch
    std::vector<double> candidates;
    candidates.reserve(100);
    
    // Extrair quantis em uma grade mais fina nas extremidades
    for (double q = 0.01; q <= 0.1; q += 0.01) {
      candidates.push_back(sketch->get_quantile(q));
      candidates.push_back(sketch->get_quantile(1.0 - q));
    }
    
    // Extrair quantis no meio da distribuição
    for (double q = 0.1; q <= 0.9; q += 0.05) {
      candidates.push_back(sketch->get_quantile(q));
    }
    
    // Remover duplicações e ordenar
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    
    // Verificação de segurança
    if (candidates.empty()) {
      // Fallback para caso o sketch tenha problemas
      double min_val = sketch->get_min();
      double max_val = sketch->get_max();
      double range = max_val - min_val;
      
      for (int i = 1; i < 10; ++i) {
        candidates.push_back(min_val + (range * i / 10.0));
      }
    }
    
    return candidates;
  }
  
  // Calcula estatísticas de target para cada candidato a ponto de corte
  std::vector<CutpointStats> calculate_cutpoint_stats(const std::vector<double>& candidates) {
    // Precisamos passar sobre os dados novamente para calcular estatísticas
    std::vector<CutpointStats> stats;
    stats.reserve(candidates.size());
    
    for (double cutpoint : candidates) {
      CutpointStats cs(cutpoint);
      stats.push_back(cs);
    }
    
    // Passar sobre os dados uma única vez para todas as estatísticas
    for (size_t i = 0; i < feature.size(); ++i) {
      double value = feature[i];
      int is_positive = target[i];
      
      for (auto& cs : stats) {
        if (value <= cs.cutpoint) {
          cs.count_below++;
          if (is_positive) {
            cs.count_pos_below++;
          } else {
            cs.count_neg_below++;
          }
        } else {
          cs.count_above++;
          if (is_positive) {
            cs.count_pos_above++;
          } else {
            cs.count_neg_above++;
          }
        }
      }
    }
    
    // Calcular IV para cada ponto de corte
    for (auto& cs : stats) {
      // Verificação de segurança para divisão por zero
      if (cs.count_below == 0 || cs.count_above == 0) {
        cs.iv = 0.0;
        continue;
      }
      
      // IV para parte abaixo
      double prop_event_below = static_cast<double>(cs.count_pos_below) / std::max(total_bad, 1);
      double prop_non_event_below = static_cast<double>(cs.count_neg_below) / std::max(total_good, 1);
      
      prop_event_below = std::max(prop_event_below, EPSILON);
      prop_non_event_below = std::max(prop_non_event_below, EPSILON);
      
      double woe_below = utils::safe_log(prop_event_below / prop_non_event_below);
      double iv_below = (prop_event_below - prop_non_event_below) * woe_below;
      
      // IV para parte acima
      double prop_event_above = static_cast<double>(cs.count_pos_above) / std::max(total_bad, 1);
      double prop_non_event_above = static_cast<double>(cs.count_neg_above) / std::max(total_good, 1);
      
      prop_event_above = std::max(prop_event_above, EPSILON);
      prop_non_event_above = std::max(prop_non_event_above, EPSILON);
      
      double woe_above = utils::safe_log(prop_event_above / prop_non_event_above);
      double iv_above = (prop_event_above - prop_non_event_above) * woe_above;
      
      // IV total é a soma dos IVs
      cs.iv = std::fabs(iv_below) + std::fabs(iv_above);
    }
    
    return stats;
  }
  
  // Selecionar os melhores pontos de corte usando greedy ou DP
  void select_optimal_cutpoints(const std::vector<double>& candidates) {
    // Verificação de segurança
    if (candidates.empty()) {
      Rcpp::warning("Nenhum candidato a ponto de corte disponível.");
      cutpoints.clear();
      create_initial_bins();
      return;
    }
    
    // Para conjuntos pequenos, podemos usar programação dinâmica exata
    if (feature.size() <= static_cast<size_t>(dp_size_limit)) {
      try {
        // Usar programação dinâmica para encontrar cutpoints ótimos
        DynamicProgramming dp(feature, target);
        cutpoints = dp.optimize(max_bins);
      } catch (const std::exception& e) {
        Rcpp::warning(std::string("Erro na programação dinâmica: ") + e.what() + ". Usando método greedy.");
        cutpoints.clear(); // Fallback para método greedy
      }
    } 
    
    // Se DP falhou ou conjunto é grande, usar abordagem greedy
    if (cutpoints.empty()) {
      // Para conjuntos grandes, usar abordagem greedy com estatísticas calculadas
      std::vector<CutpointStats> stats = calculate_cutpoint_stats(candidates);
      
      // Ordenar por IV (decrescente)
      std::sort(stats.begin(), stats.end(),
                [](const CutpointStats& a, const CutpointStats& b) {
                  return a.iv > b.iv;
                });
      
      // Selecionar max_bins-1 melhores pontos de corte
      cutpoints.clear();
      for (size_t i = 0; i < std::min(stats.size(), static_cast<size_t>(max_bins - 1)); ++i) {
        cutpoints.push_back(stats[i].cutpoint);
      }
      
      // Ordenar cutpoints
      std::sort(cutpoints.begin(), cutpoints.end());
    }
    
    // Criar bins iniciais baseados nos pontos de corte
    create_initial_bins();
  }
  
  // Criar bins iniciais a partir dos pontos de corte
  void create_initial_bins() {
    bins.clear();
    
    // Verificação de segurança
    if (!sketch) {
      throw std::runtime_error("Sketch não inicializado.");
    }
    
    double min_val = sketch->get_min();
    double max_val = sketch->get_max();
    
    if (cutpoints.empty()) {
      // Caso de um único bin
      NumericBin bin(min_val, max_val);
      bin.add_counts(total_bad, total_good);
      bins.push_back(bin);
      return;
    }
    
    // Primeiro bin (de -inf até o primeiro cutpoint)
    NumericBin first_bin(min_val, cutpoints[0]);
    bins.push_back(first_bin);
    
    // Bins intermediários
    for (size_t i = 0; i < cutpoints.size() - 1; ++i) {
      NumericBin bin(cutpoints[i], cutpoints[i + 1]);
      bins.push_back(bin);
    }
    
    // Último bin (do último cutpoint até +inf)
    NumericBin last_bin(cutpoints.back(), max_val);
    bins.push_back(last_bin);
    
    // Preencher contagens para cada bin
    for (size_t i = 0; i < feature.size(); ++i) {
      double value = feature[i];
      int is_positive = target[i];
      
      // Otimização: busca binária para bins ordenados
      auto bin_it = std::find_if(bins.begin(), bins.end(), 
                                 [value](const NumericBin& bin) {
                                   return bin.contains(value);
                                 });
      
      if (bin_it != bins.end()) {
        bin_it->add_value(is_positive);
      } else {
        // Fallback se a busca falhar
        for (auto& bin : bins) {
          if (bin.contains(value)) {
            bin.add_value(is_positive);
            break;
          }
        }
      }
    }
  }
  
  // Enforcement de bin_cutoff
  void enforce_bin_cutoff() {
    int min_count = static_cast<int>(std::ceil(bin_cutoff * static_cast<double>(feature.size())));
    
    // Identificar bins com baixa frequência
    bool any_change = true;
    while (any_change && bins.size() > static_cast<size_t>(min_bins)) {
      any_change = false;
      
      for (size_t i = 0; i < bins.size(); ++i) {
        if (bins[i].count < min_count) {
          // Mesclar com o vizinho mais próximo em termos de taxa de eventos
          int best_neighbor = -1;
          double min_diff = std::numeric_limits<double>::max();
          
          if (i > 0) {
            double diff = std::fabs(bins[i].event_rate - bins[i - 1].event_rate);
            if (diff < min_diff) {
              min_diff = diff;
              best_neighbor = i - 1;
            }
          }
          
          if (i + 1 < bins.size()) {
            double diff = std::fabs(bins[i].event_rate - bins[i + 1].event_rate);
            if (diff < min_diff) {
              min_diff = diff;
              best_neighbor = i + 1;
            }
          }
          
          if (best_neighbor >= 0) {
            bins[best_neighbor].merge_with(bins[i]);
            bins.erase(bins.begin() + i);
            any_change = true;
            break;
          }
        }
      }
    }
    
    // Recalcular cutpoints depois de mesclar bins
    update_cutpoints_from_bins();
  }
  
  // Cálculo inicial de WoE
  void calculate_initial_woe() {
    for (auto& bin : bins) {
      bin.calculate_metrics(total_good, total_bad);
    }
  }
  
  // Enforcement de monotonicidade
  void enforce_monotonicity() {
    if (!monotonic || bins.size() <= 1) {
      return; // Monotonicidade não requerida ou não aplicável
    }
    
    // Determinar direção da monotonicidade (crescente ou decrescente)
    bool increasing = true;
    double first_woe = bins.front().woe;
    double last_woe = bins.back().woe;
    
    if (last_woe < first_woe) {
      increasing = false;
    }
    
    // Enforçar monotonicidade usando PAVA (Pool Adjacent Violators Algorithm)
    bool any_change = true;
    while (any_change && bins.size() > static_cast<size_t>(min_bins)) {
      any_change = false;
      
      for (size_t i = 0; i + 1 < bins.size(); ++i) {
        bool violation = (increasing && bins[i].woe > bins[i+1].woe) ||
          (!increasing && bins[i].woe < bins[i+1].woe);
        
        if (violation) {
          // Mesclar bins que violam monotonicidade
          bins[i].merge_with(bins[i+1]);
          bins.erase(bins.begin() + i + 1);
          bins[i].calculate_metrics(total_good, total_bad);
          any_change = true;
          break;
        }
      }
    }
    
    // Recalcular cutpoints depois de mesclar bins
    update_cutpoints_from_bins();
  }
  
  // Atualizar cutpoints baseado nos bins atuais
  void update_cutpoints_from_bins() {
    cutpoints.clear();
    for (size_t i = 1; i < bins.size(); ++i) {
      cutpoints.push_back(bins[i].lower_bound);
    }
  }
  
  // Otimização de bins
  void optimize_bins() {
    if (static_cast<int>(bins.size()) <= max_bins) {
      return; // Já estamos dentro do limite
    }
    
    // Otimizar número de bins através de mesclagem iterativa
    int iterations = 0;
    while (static_cast<int>(bins.size()) > max_bins && iterations < max_iterations) {
      if (static_cast<int>(bins.size()) <= min_bins) {
        break;
      }
      
      // Encontrar par de bins adjacentes com menor perda de IV
      double min_iv_loss = std::numeric_limits<double>::max();
      size_t merge_idx = 0;
      
      for (size_t i = 0; i + 1 < bins.size(); ++i) {
        double current_iv = std::fabs(bins[i].iv) + std::fabs(bins[i+1].iv);
        
        // Simular mesclagem
        NumericBin merged = bins[i];
        merged.merge_with(bins[i+1]);
        merged.calculate_metrics(total_good, total_bad);
        
        double merged_iv = std::fabs(merged.iv);
        double iv_loss = current_iv - merged_iv;
        
        if (iv_loss < min_iv_loss) {
          min_iv_loss = iv_loss;
          merge_idx = i;
        }
      }
      
      // Mesclar bins com menor perda de IV
      bins[merge_idx].merge_with(bins[merge_idx+1]);
      bins[merge_idx].calculate_metrics(total_good, total_bad);
      bins.erase(bins.begin() + merge_idx + 1);
      
      iterations++;
    }
    
    // Recalcular cutpoints depois de otimizar
    update_cutpoints_from_bins();
  }
  
  // Verificação de consistência
  void check_consistency() const {
    if (bins.empty()) {
      Rcpp::warning("Nenhum bin criado. Verifique os parâmetros e dados de entrada.");
      return;
    }
    
    int total_count = 0;
    int total_count_pos = 0;
    int total_count_neg = 0;
    
    for (const auto& bin : bins) {
      total_count += bin.count;
      total_count_pos += bin.count_pos;
      total_count_neg += bin.count_neg;
    }
    
    // Validação com tolerância devido à aproximação do sketch
    double count_tolerance = 0.05; // 5% de tolerância
    
    double count_ratio = static_cast<double>(total_count) / feature.size();
    if (std::fabs(count_ratio - 1.0) > count_tolerance) {
      Rcpp::warning(
        "Possível inconsistência após binning devido à aproximação do sketch. "
        "Contagem total: " + std::to_string(total_count) + ", esperado: " + 
          std::to_string(feature.size()) + ". Razão: " + std::to_string(count_ratio)
      );
    }
    
    double pos_ratio = static_cast<double>(total_count_pos) / total_bad;
    double neg_ratio = static_cast<double>(total_count_neg) / total_good;
    if (std::fabs(pos_ratio - 1.0) > count_tolerance || std::fabs(neg_ratio - 1.0) > count_tolerance) {
      Rcpp::warning(
        "Possível inconsistência em contagens positivas/negativas após binning devido à aproximação do sketch. "
        "Positivos: " + std::to_string(total_count_pos) + " vs " + std::to_string(total_bad) + 
          ", Negativos: " + std::to_string(total_count_neg) + " vs " + std::to_string(total_good)
      );
    }
  }
  
public:
  // Construtor
  OptimalBinningNumericalSketch(
    const std::vector<double>& feature_,
    const Rcpp::IntegerVector& target_,
    int min_bins_ = 3,
    int max_bins_ = 5,
    double bin_cutoff_ = 0.05,
    std::string special_codes_ = "",
    bool monotonic_ = true,
    double convergence_threshold_ = 1e-6,
    int max_iterations_ = 1000,
    int sketch_k_ = 200
  ) : feature(feature_), target(Rcpp::as<std::vector<int>>(target_)), 
  min_bins(min_bins_), max_bins(max_bins_), 
  bin_cutoff(bin_cutoff_), special_codes(special_codes_),
  monotonic(monotonic_),
  convergence_threshold(convergence_threshold_),
  max_iterations(max_iterations_),
  sketch_k(sketch_k_),
  dp_size_limit(1000), // Limite mais conservador
  total_good(0), total_bad(0) {
    // Pré-alocação
    bins.reserve(max_bins_);
    cutpoints.reserve(max_bins_ - 1);
  }
  
  // Método fit
  Rcpp::List fit() {
    try {
      // Processo de binagem
      validate_inputs();
      build_sketch();
      std::vector<double> candidates = extract_candidates();
      select_optimal_cutpoints(candidates);
      enforce_bin_cutoff();
      calculate_initial_woe();
      enforce_monotonicity();
      
      // Otimização de bins se necessário
      bool converged_flag = false;
      int iterations_done = 0;
      
      if (static_cast<int>(bins.size()) <= max_bins) {
        converged_flag = true;
      } else {
        double prev_total_iv = 0.0;
        for (const auto& bin : bins) {
          prev_total_iv += std::fabs(bin.iv);
        }
        
        for (int i = 0; i < max_iterations; ++i) {
          // Ponto de partida para esta iteração
          size_t start_bins = bins.size();
          
          optimize_bins();
          
          // Se não reduziu bins ou atingiu max_bins, verificar convergência
          if (bins.size() == start_bins || static_cast<int>(bins.size()) <= max_bins) {
            double total_iv = 0.0;
            for (const auto& bin : bins) {
              total_iv += std::fabs(bin.iv);
            }
            
            if (std::fabs(total_iv - prev_total_iv) < convergence_threshold) {
              converged_flag = true;
              iterations_done = i + 1;
              break;
            }
            
            prev_total_iv = total_iv;
          }
          
          iterations_done = i + 1;
          
          if (static_cast<int>(bins.size()) <= max_bins) {
            break;
          }
        }
      }
      
      // Verificação final de consistência
      check_consistency();
      
      // Verificação de segurança antes de preparar resultados
      if (bins.empty()) {
        return Rcpp::List::create(
          Named("error") = "Falha ao criar bins. Verifique os dados de entrada."
        );
      }
      
      // Preparação de resultados
      const size_t n_bins = bins.size();
      
      NumericVector bin_lower(n_bins);
      NumericVector bin_upper(n_bins);
      NumericVector bin_woe(n_bins);
      NumericVector bin_iv(n_bins);
      IntegerVector bin_count(n_bins);
      IntegerVector bin_count_pos(n_bins);
      IntegerVector bin_count_neg(n_bins);
      NumericVector ids(n_bins);
      
      for (size_t i = 0; i < n_bins; ++i) {
        bin_lower[i] = bins[i].lower_bound;
        bin_upper[i] = bins[i].upper_bound;
        bin_woe[i] = bins[i].woe;
        bin_iv[i] = bins[i].iv;
        bin_count[i] = bins[i].count;
        bin_count_pos[i] = bins[i].count_pos;
        bin_count_neg[i] = bins[i].count_neg;
        ids[i] = i + 1;
      }
      
      return Rcpp::List::create(
        Named("id") = ids,
        Named("bin_lower") = bin_lower,
        Named("bin_upper") = bin_upper,
        Named("woe") = bin_woe,
        Named("iv") = bin_iv,
        Named("count") = bin_count,
        Named("count_pos") = bin_count_pos,
        Named("count_neg") = bin_count_neg,
        Named("cutpoints") = cutpoints,
        Named("converged") = converged_flag,
        Named("iterations") = iterations_done
      );
    } catch (const std::exception& e) {
      Rcpp::stop("Erro no binning ótimo numérico com sketch: " + std::string(e.what()));
    }
  }
};

//' @title Optimal Binning for Numerical Variables using Sketch-based Algorithm
//'
//' @description
//' This function performs optimal binning for numerical variables using a sketch-based approach,
//' combining KLL Sketch for quantile approximation with Weight of Evidence (WOE) and 
//' Information Value (IV) methods.
//'
//' @param feature A numeric vector of feature values.
//' @param target An integer vector of binary target values (0 or 1).
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum frequency for a bin (default: 0.05).
//' @param special_codes String with special codes to be treated separately, separated by comma (default: "").
//' @param monotonic Whether to enforce monotonicity of WOE across bins (default: TRUE).
//' @param convergence_threshold Threshold for convergence in optimization (default: 1e-6).
//' @param max_iterations Maximum number of iterations for optimization (default: 1000).
//' @param sketch_k Parameter controlling the accuracy of the KLL sketch (default: 200).
//'
//' @return A list containing:
//' \itemize{
//'   \item id: Numeric identifiers for each bin
//'   \item bin_lower: Lower bounds of bins
//'   \item bin_upper: Upper bounds of bins
//'   \item woe: Weight of Evidence for each bin
//'   \item iv: Information Value for each bin
//'   \item count: Total counts for each bin
//'   \item count_pos: Positive target counts for each bin
//'   \item count_neg: Negative target counts for each bin
//'   \item cutpoints: Selected cutting points between bins
//'   \item converged: Logical value indicating whether the algorithm converged
//'   \item iterations: Number of iterations run
//' }
//'
//' @details
//' The algorithm uses a KLL (Karnin-Lang-Liberty) Sketch data structure to efficiently approximate
//' the quantiles of numerical data, making it suitable for very large datasets or streaming scenarios.
//' The sketch-based approach allows processing data in a single pass with sublinear memory usage.
//'
//' The algorithm performs the following steps:
//' \enumerate{
//'   \item Input validation and preprocessing
//'   \item Building a KLL sketch for the data
//'   \item Extracting candidate cutpoints from the sketch
//'   \item Selecting optimal cutpoints using either dynamic programming (for smaller datasets)
//'         or a greedy approach based on Information Value
//'   \item Enforcing minimum bin size (bin_cutoff)
//'   \item Calculating initial Weight of Evidence (WOE) and Information Value (IV)
//'   \item Enforcing monotonicity of WOE across bins (if requested)
//'   \item Optimizing the number of bins through iterative merging
//' }
//'
//' @examples
//' \dontrun{
//' # Create sample data
//' set.seed(123)
//' target <- sample(0:1, 1000, replace = TRUE)
//' feature <- rnorm(1000)
//'
//' # Run optimal binning with sketch
//' result <- optimal_binning_numerical_sketch(feature, target)
//'
//' # View results
//' print(result)
//' }
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_sketch(
   Rcpp::IntegerVector target,
   Rcpp::NumericVector feature,
   int min_bins = 3,
   int max_bins = 5,
   double bin_cutoff = 0.05,
   std::string special_codes = "",
   bool monotonic = true,
   double convergence_threshold = 1e-6,
   int max_iterations = 1000,
   int sketch_k = 200
) {
 // Verificações preliminares
 if (feature.size() == 0 || target.size() == 0) {
   Rcpp::stop("Feature e target não podem ser vazios.");
 }
 
 if (feature.size() != target.size()) {
   Rcpp::stop("Feature e target devem ter o mesmo tamanho.");
 }
 
 // Conversão otimizada de R para C++
 std::vector<double> feature_vec;
 feature_vec.reserve(feature.size());
 
 for (R_xlen_t i = 0; i < feature.size(); ++i) {
   // Tratamento de NAs em feature
   if (NumericVector::is_na(feature[i])) {
     Rcpp::stop("Feature não pode conter valores ausentes. Pré-processe os dados primeiro.");
   } else {
     feature_vec.push_back(feature[i]);
   }
 }
 
 // Validação de NAs em target
 for (R_xlen_t i = 0; i < target.size(); ++i) {
   if (IntegerVector::is_na(target[i])) {
     Rcpp::stop("Target não pode conter valores ausentes.");
   }
 }
 
 // Verificação de segurança para valores constantes
 double min_val = *std::min_element(feature_vec.begin(), feature_vec.end());
 double max_val = *std::max_element(feature_vec.begin(), feature_vec.end());
 if (std::fabs(max_val - min_val) < EPSILON) {
   Rcpp::warning("Feature tem valor constante, criando um único bin.");
   NumericVector ids = NumericVector::create(1);
   NumericVector bin_lower = NumericVector::create(min_val);
   NumericVector bin_upper = NumericVector::create(max_val);
   
   // Contagens
   int count_pos = 0;
   for (int t : target) {
     if (t == 1) count_pos++;
   }
   int count = feature_vec.size();
   int count_neg = count - count_pos;
   
   // Usar totais globais para cálculo de WoE e IV
   double prop_event = static_cast<double>(count_pos) / std::max(count_pos, 1);
   double prop_non_event = static_cast<double>(count_neg) / std::max(count_neg, 1);
   double woe = utils::safe_log(prop_event / prop_non_event);
   double iv = (prop_event - prop_non_event) * woe;
   
   IntegerVector count_vec = IntegerVector::create(count);
   IntegerVector count_pos_vec = IntegerVector::create(count_pos);
   IntegerVector count_neg_vec = IntegerVector::create(count_neg);
   NumericVector woe_vec = NumericVector::create(woe);
   NumericVector iv_vec = NumericVector::create(iv);
   NumericVector cutpoints = NumericVector::create();
   
   return Rcpp::List::create(
     Named("id") = ids,
     Named("bin_lower") = bin_lower,
     Named("bin_upper") = bin_upper,
     Named("woe") = woe_vec,
     Named("iv") = iv_vec,
     Named("count") = count_vec,
     Named("count_pos") = count_pos_vec,
     Named("count_neg") = count_neg_vec,
     Named("cutpoints") = cutpoints,
     Named("converged") = true,
     Named("iterations") = 0
   );
 }
 
 // Executa o algoritmo otimizado com sketch
 OptimalBinningNumericalSketch sketch_binner(
     feature_vec, target, min_bins, max_bins, bin_cutoff, special_codes,
     monotonic, convergence_threshold, max_iterations, sketch_k
 );
 
 return sketch_binner.fit();
}
