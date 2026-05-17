# TODO.md — OptimalBinningWoE C++ Audit & Improvement Tracker

**Criado em:** 2026-05-16  
**Auditor:** Claude Code (claude-sonnet-4-6)  
**Versão base auditada:** ≥ 1.0.8 (branch main, commit 8b411a8)  
**Política:** CRAN-safe — toda correção mantém compatibilidade total de API pública (mesmos parâmetros R, mesmos nomes de output)

---

## LEGENDA DE STATUS

| Símbolo | Significado |
|---------|-------------|
| `[ ]` | Pendente |
| `[~]` | Em andamento |
| `[x]` | Concluído |
| `[!]` | Bloqueado / requer decisão |

---

## FASE 1 — CORREÇÕES CRÍTICAS (Crash / UB / Corretude)

> Objetivo: resolver antes do próximo envio ao CRAN.

### P1.1 — Out-of-bounds no backtracking DP `OBC_DP_v5.cpp` ✓
- **Arquivo:** `src/OBC_DP_v5.cpp`
- **Função:** `backtrack_optimal_bins()`, linha ~793
- **Problema:** `prev_bin` inicializado com -1. Se o DP não preencher completamente `dp[n][best_k]`, a conversão `static_cast<size_t>(-1)` resulta em `2^64-1`, causando acesso fora dos limites e crash (UB).
- **Correção:** Checar `prev_j >= 0` antes de usar; lançar exceção descritiva se -1.
```cpp
int prev_j = prev_bin[idx][k];
if (prev_j < 0 || static_cast<size_t>(prev_j) > n) {
    throw std::runtime_error("DP backtracking failed: invalid predecessor index.");
}
idx = static_cast<size_t>(prev_j);
```
- **Status:** `[x]`
- **Aplicado em:** 2026-05-16
- **Testado em:** R CMD INSTALL OK

---

### P1.2 — Push em target_vec antes de validar valor — `OBN_DP_v5.cpp`
- **Arquivo:** `src/OBN_DP_v5.cpp`
- **Função:** `optimal_binning_numerical_dp()`, linhas ~787–801
- **Status:** `[x]`
- **Aplicado em:** 2026-05-16
- **Testado em:** R CMD INSTALL OK

---

### P1.3 — Hessiana singular em `OB_LogisticRegression.cpp`
- **Arquivo:** `src/OB_LogisticRegression.cpp`
- **Função:** `fit_logistic_regression_template()`, linha ~78
- **Problema 1:** `if (det != 0)` usa comparação exata de ponto flutuante. Matrizes quase-singulares passam pelo guard e produzem SE instáveis via `hessian.inverse()`.
- **Problema 2:** `sqrt()` sobre `inverse().diagonal()` pode gerar `NaN` se algum diagonal for negativo por erro numérico.
- **Problema 3:** `iter_count = maxit` é sempre reportado como número de iterações, mesmo após convergência antecipada.
- **Correção:**
```cpp
double threshold = 1e-10 * hessian.norm();
if (std::abs(det) > threshold) {
    Eigen::LDLT<MatrixXd> ldlt(hessian);
    if (ldlt.info() == Eigen::Success) {
        MatrixXd H_inv = ldlt.solve(MatrixXd::Identity(hessian.rows(), hessian.cols()));
        VectorXd diag = H_inv.diagonal().cwiseMax(0.0);
        VectorXd se = diag.array().sqrt();
        ...
    }
}
// iter_count: verificar se optim_lbfgs expõe contagem real
```
- **Status:** `[ ]`
- **Aplicado em:** —
- **Testado em:** —

---

### P1.4 — `log2(0)` em MDL com 1 bin — `OBN_MDLP_v5.cpp`
- **Arquivo:** `src/OBN_MDLP_v5.cpp`
- **Função:** `calculate_mdl_cost()`, linha ~128
- **Status:** `[x]`
- **Aplicado em:** 2026-05-16
- **Testado em:** R CMD INSTALL OK

---

### P1.5 — `constexpr` com `std::log`/`std::exp` — `safe_math.h`
- **Arquivo:** `src/common/safe_math.h`
- **Status:** `[x]`
- **Aplicado em:** 2026-05-16 — todas as 6 funções convertidas de `constexpr` para `inline`
- **Testado em:** R CMD INSTALL OK

---

## FASE 2 — UNIFICAÇÃO DE WoE/IV (Consistência entre algoritmos)

> Objetivo: todos os algoritmos produzem valores WoE/IV equivalentes para os mesmos dados.

### P2.1 — Funções locais `compute_woe_iv` / `safe_log` / `safe_divide` / `clamp` duplicadas
- **Arquivos afetados:**
  - `src/OBC_DP_v5.cpp` — define `compute_woe_iv()` local (sem smoothing, epsilon=1e-10)
  - `src/OBC_CM_v5.cpp` — define `safe_log()`, `safe_divide()`, `clamp()` locais (epsilon=1e-12)
  - `src/OBN_KMB_v5.cpp` — `calculateWOE()` com alpha=0.5 hardcoded
  - `src/OBN_IR_v5.cpp` — `ALPHA = 0.5` Laplace local
  - `src/OBN_MDLP_v5.cpp` — Laplace manual com `bins.size() * laplace_smoothing` no denominador
- **Problema:** 6+ implementações divergentes com epsilons e smoothings distintos. Mesmo bin, resultados WoE diferentes dependendo do algoritmo.
- **Correção:** Incluir `woe_iv_utils.h` e usar `compute_woe()` / `compute_iv()` em todos. Remover duplicatas locais.
- **Status:** `[ ]`
- **Aplicado em:** —
- **Observação:** Compatibilidade de output — verificar se unificação altera valores WoE reportados ao R; se sim, incrementar versão e documentar em NEWS.md.

---

### P2.2 — `is_monotonic()` só verifica direção ascendente — `OBN_MDLP_v5.cpp`
- **Arquivo:** `src/OBN_MDLP_v5.cpp`
- **Função:** `is_monotonic()`, linha ~242
- **Problema:** Nenhum parâmetro de direção; sempre verifica ascendente. Dados com relação negativa disparam merges desnecessários tentando forçar ascendência.
- **Correção:** Adicionar parâmetro `bool ascending = true` e direcionar o check accordingly.
- **Status:** `[ ]`
- **Aplicado em:** —

---

### P2.3 — "auto" monotonicity não implementado em `OBC_DP_v5.cpp`
- **Arquivo:** `src/OBC_DP_v5.cpp`
- **Função:** `compute_and_sort_event_rates()`, linha ~642
- **Problema:** Comentário explícito "This would be implemented here - for now we default to ascending". Funcionalidade pendente — usuário que passa `monotonic_trend="auto"` na versão categórica DP não tem detecção automática.
- **Correção:** Chamar `detect_trend_from_correlation()` de `monotonicity_utils.h` e usar o resultado para ordenar `sorted_categories`.
- **Status:** `[ ]`
- **Aplicado em:** —

---

## FASE 3 — CORREÇÕES DE PERFORMANCE

> Objetivo: melhorar comportamento em datasets maiores sem alterar output.

### P3.1 — Loop de iteração desnecessário no DP — `OBC_DP_v5.cpp` ✓
- **Arquivo:** `src/OBC_DP_v5.cpp`
- **Função:** `perform_dynamic_programming()`, linha ~682
- **Problema:** O DP é executado `max_iterations` vezes (padrão: 1000) em loop externo. DP correto é determinístico e exato em **uma única passagem**. A "convergência" detectada sempre ocorre na 1ª iteração porque o DP não tem estado externo que mude entre iterações.
- **Complexidade atual:** O(max_iterations × n² × k) com alocação O(n²) por iteração.
- **Complexidade correta:** O(n² × k).
- **Correção:** Remover o `for (iterations_run = 1; ...)` externo; executar o DP diretamente:
```cpp
// Construir bin_woe_iv uma vez (fora do loop que não existe mais)
converged = true;
iterations_run = 1;
// ... código DP original sem loop externo ...
```
- **Status:** `[x]`
- **Aplicado em:** 2026-05-16
- **Impacto estimado:** até 1000x speedup para casos com `max_iterations` alto.
- **Testado em:** R CMD INSTALL OK

---

### P3.2 — Re-sort O(n log n) a cada merge — `OBC_DP_v5.cpp::ensure_max_prebins()`
- **Arquivo:** `src/OBC_DP_v5.cpp`
- **Função:** `ensure_max_prebins()`, linha ~563
- **Problema:** `std::sort` completo chamado após cada merge dentro do while loop. Complexidade total: O(m² log m) para m bins.
- **Correção:** Usar `std::priority_queue` ou inserção ordenada após merge:
```cpp
// Após o merge, apenas inserir o novo bin na posição correta (O(m))
// em vez de re-sort completo O(m log m)
```
- **Status:** `[ ]`
- **Aplicado em:** —

---

### P3.3 — Cópia de vetor O(k) em loop interno O(k) — `OBN_MDLP_v5.cpp::apply_mdl_merging()`
- **Arquivo:** `src/OBN_MDLP_v5.cpp`
- **Função:** `apply_mdl_merging()`, linha ~897
- **Problema:** `std::vector<NumericalBin> temp_bins = bins` dentro do inner loop cria cópia completa por candidato de merge. Complexidade: O(k³) cópias.
- **Correção:** Calcular delta MDL diretamente, sem cópia do vetor:
```cpp
// Apenas combinar estatísticas de bins[i] e bins[i+1] temporariamente
// e calcular log2(k-2) - log2(k-1) + entropia_delta
```
- **Status:** `[ ]`
- **Aplicado em:** —

---

### P3.4 — `quantile()` re-ordena dados a cada chamada — `OBN_BB_v5.cpp`
- **Arquivo:** `src/OBN_BB_v5.cpp`
- **Função:** `quantile()`, linha ~127
- **Problema:** Cria cópia e ordena `data` toda vez. Se chamada `max_n_prebins` vezes, custo total: O(max_n_prebins × n log n).
- **Correção:** Ordenar uma vez em `prebinning()` e passar vetor já ordenado para uma variante de `quantile()` que não re-ordena.
- **Status:** `[ ]`
- **Aplicado em:** —

---

### P3.5 — Alocação desnecessária de `indices` em Welford — `monotonicity_utils.h`
- **Status:** `[x]`
- **Aplicado em:** 2026-05-16 — `indices[]` eliminado, loop usa `static_cast<double>(i)` direto
- **Testado em:** R CMD INSTALL OK

---

### P3.6 — `determine_monotonic_direction()` usa fórmula Pearson instável — `OBN_DP_v5.cpp`
- **Status:** `[x]`
- **Aplicado em:** 2026-05-16 — substituído por `detect_trend_from_correlation()` de monotonicity_utils.h
- **Testado em:** R CMD INSTALL OK

---

## FASE 4 — ODR / HEADERS (CRAN Safety)

### P4.1 — `CHI_SQUARE_CRITICAL_VALUES` como `const` em header — `chi_square_utils.h`
- **Status:** `[x]`
- **Aplicado em:** 2026-05-16 — convertido para `chi_square_critical_values_table()` com static local
- **Testado em:** R CMD INSTALL OK

---

### P4.2 — `ENTROPY_LUT` estático duplicado por TU — `entropy_utils.h`
- **Status:** `[x]`
- **Aplicado em:** 2026-05-16 — convertido para `entropy_lut_instance()` com static local
- **Testado em:** R CMD INSTALL OK

---

### P4.3 — `ChiSquareCache` duplicada em dois locais
- **Arquivos:** `src/common/chi_square_utils.h` (namespace `OptimalBinning`) e `src/OBC_CM_v5.cpp` (namespace global)
- **Problema:** Duas implementações. Qualquer `#include "chi_square_utils.h"` em CM geraria conflito de nomes.
- **Correção:** Fazer `OBC_CM_v5.cpp` incluir `chi_square_utils.h` e remover a definição local. Ajustar qualificação de namespace onde necessário.
- **Status:** `[ ]`
- **Aplicado em:** —

---

### P4.4 — `using namespace Rcpp` duplicado em múltiplos arquivos
- **Status:** `[x]`
- **Aplicado em:** 2026-05-16 — 35 arquivos corrigidos via script awk
- **Testado em:** R CMD INSTALL OK

---

## FASE 5 — FUNCIONALIDADES INCOMPLETAS

### P5.1 — "auto" não implementado em `OBC_DP_v5.cpp` *(já listado em P2.3)*
- Ver P2.3

### P5.2 — `iter_count` sempre igual a `maxit` — `OB_LogisticRegression.cpp` *(já listado em P1.3)*
- Ver P1.3

---

## FASE 6 — NORMALIZAÇÃO / LIMPEZA

### P6.1 — Shadowing de `total_count` em `OBN_DP_v5.cpp`
- **Status:** `[x]`
- **Aplicado em:** 2026-05-16 — renomeado para `rare_total`
- **Testado em:** R CMD INSTALL OK

### P6.2 — Comentários de código morto em `OBC_DP_v5.cpp`
- Blocos comentados: `// struct CategoryStats { ... }`, `// Local CategoricalBin definition removed`, etc.
- **Correção:** Remover completamente.
- **Status:** `[ ]`

### P6.3 — `NumericalBin::total()` vs `count` — potencial inconsistência
- **Arquivo:** `src/common/bin_structures.h`
- **Problema:** `total()` retorna `count_pos + count_neg` mas `count` deveria ser idêntico. Construtores definem `count` explicitamente sem garantir a invariante.
- **Correção:** Adicionar `assert(count == count_pos + count_neg)` em `is_valid()` e garantir invariante em construtores.
- **Status:** `[ ]`

### P6.4 — Padronizar `[[Rcpp::plugins(cpp11)]]`
- **Status:** `[x]`
- **Aplicado em:** 2026-05-16 — `OBN_IR_v5.cpp` cpp17→cpp11
- **Testado em:** R CMD INSTALL OK

### P6.5 — `using namespace` antes de `#include` em vários arquivos
- **Status:** `[x]` — resolvido como parte do P4.4 (remoção da linha duplicada era a que aparecia antes dos includes)
- **Aplicado em:** 2026-05-16

---

## REGISTRO DE ALTERAÇÕES APLICADAS

| Data | ID | Arquivo(s) | Descrição | Testado |
|------|----|-----------|-----------|---------|
| 2026-05-16 | P1.4 | `OBN_MDLP_v5.cpp` | Guard `log2(0)` em `calculate_mdl_cost()` | R CMD INSTALL OK |
| 2026-05-16 | P1.5 | `common/safe_math.h` | `constexpr` → `inline` em 6 funções (CRAN/SOLARIS compat) | R CMD INSTALL OK |
| 2026-05-16 | P1.2 | `OBN_DP_v5.cpp` | Validar target antes do push (stop-before-insert) | R CMD INSTALL OK |
| 2026-05-16 | P1.1 | `OBC_DP_v5.cpp` | Guard `prev_j < 0` antes de `static_cast<size_t>` no backtracking | R CMD INSTALL OK |
| 2026-05-16 | P3.5 | `common/monotonicity_utils.h` | Eliminar alocação `indices[]` desnecessária no Welford | R CMD INSTALL OK |
| 2026-05-16 | P3.6 | `OBN_DP_v5.cpp` | Substituir Pearson naïve por `detect_trend_from_correlation()` Welford | R CMD INSTALL OK |
| 2026-05-16 | P6.1 | `OBN_DP_v5.cpp` | Renomear `total_count` local → `rare_total` (shadowing fix) | R CMD INSTALL OK |
| 2026-05-16 | P4.4 | 35 arquivos `.cpp` | Remover `using namespace Rcpp` duplicado (antes dos includes) | R CMD INSTALL OK |
| 2026-05-16 | P4.1 | `common/chi_square_utils.h` | `CHI_SQUARE_CRITICAL_VALUES` const→função com static local (C++11-safe) | R CMD INSTALL OK |
| 2026-05-16 | P4.2 | `common/entropy_utils.h` | `ENTROPY_LUT` static→`entropy_lut_instance()` (um cópia compartilhada) | R CMD INSTALL OK |
| 2026-05-16 | P6.4 | `OBN_IR_v5.cpp` | `[[Rcpp::plugins(cpp17)]]` → `cpp11` (padronização) | R CMD INSTALL OK |
| 2026-05-16 | P3.1 | `OBC_DP_v5.cpp` | Remover loop externo desnecessário em `perform_dynamic_programming()` | R CMD INSTALL OK |
| 2026-05-16 | P2.3 | `OBC_DP_v5.cpp` | Implementar detecção "auto" em `compute_and_sort_event_rates()` via Welford | R CMD INSTALL OK |

---

## PRÓXIMA AÇÃO RECOMENDADA

**Sessão 2026-05-16 concluída.** 13 itens aplicados e compilados com sucesso.

**Próxima sessão — continuar por:**

1. **P1.3** — `OB_LogisticRegression.cpp`: Hessiana singular (`det != 0` → threshold) + SE via LDLT + iter_count real
2. **P2.1** — Unificação WoE/IV: remover `compute_woe_iv()` local de `OBC_DP_v5.cpp` e usar `woe_iv_utils.h`
3. **P2.2** — `is_monotonic()` em `OBN_MDLP_v5.cpp` só verifica ascending → adicionar parâmetro de direção
4. **P3.2** — `ensure_max_prebins()` em `OBC_DP_v5.cpp`: trocar re-sort por inserção ordenada
5. **P3.3** — `apply_mdl_merging()` em `OBN_MDLP_v5.cpp`: cálculo delta MDL sem cópia de vetor
6. **P3.4** — `quantile()` em `OBN_BB_v5.cpp`: pré-ordenar uma vez
7. **P4.3** — `ChiSquareCache` duplicada: consolidar em `chi_square_utils.h`
8. **P6.2** — Remover comentários de código morto em `OBC_DP_v5.cpp`
9. **P6.3** — `NumericalBin::total()` vs `count`: garantir invariante

Após P1.3 → executar `R CMD check --as-cran` completo.

---

## NOTAS GERAIS

- **Política de versão:** Fases 1 e 4 são patch releases (ex: 1.0.9). Fases 2 e 5 podem alterar valores numéricos de WoE/IV — devem ser minor releases (ex: 1.1.0) com documentação em `NEWS.md`.
- **Testes:** Após cada fase, rodar `tests/testthat/` + `R CMD check --as-cran`.
- **CRAN policy:** Não usar `Rf_error` diretamente (já corrigido em commit 8b411a8). Usar apenas `Rcpp::stop()` e `Rcpp::warning()`.
- **Compatibilidade de plataforma:** Windows (MSVC/MinGW), Linux (GCC), macOS (CLANG), SOLARIS (Studio). Testar `constexpr` math especialmente em SOLARIS.
