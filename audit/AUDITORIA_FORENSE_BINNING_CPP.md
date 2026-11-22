# Auditoria forense dos algoritmos de binning em C++ — OptimalBinningWoE 1.14.1

> **Escopo:** 37 algoritmos em `src/OB[NC]_*_v5.cpp` (21 numéricos e 16 categóricos), os headers em `src/common/*.h` e a infraestrutura de aplicação (`OB_ApplyWoE*.cpp`, `OB_Cutpoints.cpp`, `Makevars`), cerca de 34 mil linhas de C++.
> **Natureza:** auditoria forense, só leitura. **Nenhum arquivo de código foi alterado.** Este documento propõe correções para o mantenedor avaliar.
> **Data:** 2026-11-22 · **Branch:** `claude/binning-algorithms-cpp-review-7f7noc`

---

## Sumário

0. [Como ler este documento](#0-como-ler-este-documento)
1. [Sumário executivo](#1-sumário-executivo)
2. [Metodologia](#2-metodologia)
3. [Veredito por algoritmo (37 algoritmos)](#3-veredito-por-algoritmo)
4. [Temas transversais (consenso da equipe)](#4-temas-transversais-consenso-da-equipe)
5. [Registro consolidado de achados (Crítico e Alto)](#5-registro-consolidado-de-achados-crítico-e-alto)
6. [Achados Médios e Baixos (índice)](#6-achados-médios-e-baixos-índice)
7. [Arquitetura de correção proposta (código comum)](#7-arquitetura-de-correção-proposta)
8. [Plano de ação em ondas](#8-plano-de-ação-em-ondas)
9. [Protocolo obrigatório de não-regressão (baseline × candidato)](#9-protocolo-obrigatório-de-não-regressão-baseline--candidato)
10. [Suíte de testes de propriedade proposta](#10-suíte-de-testes-de-propriedade-proposta)
11. [O que está correto (validado)](#11-o-que-está-correto-validado)
12. [Decisões que cabem ao mantenedor](#12-decisões-que-cabem-ao-mantenedor)
13. [Limitações desta auditoria](#13-limitações-desta-auditoria)
14. [Referências](#14-referências)
- [Anexos A–E: relatórios integrais dos agentes](#anexos--relatórios-integrais-dos-agentes)

---

## 0. Como ler este documento

- **IDs:** cada achado leva o prefixo do agente que o encontrou (`A-`, `B-`, `C-`, `D-`, `E-`). A seção 5 consolida os achados graves e agrupa as duplicatas (por exemplo, o mesmo defeito em três arquivos vira um item `X-n`).
- **Severidade:**
  - **Crítico:** resultado estatisticamente errado em uso normal, ou crash/UB.
  - **Alto:** desvio teórico material, ou erro em edge case plausível.
  - **Médio:** performance grave ou instabilidade em casos raros.
  - **Baixo:** melhoria.
- **Status:**
  - **Confirmado:** há experimento reprodutível ou leitura inequívoca do código.
  - **Suspeito:** é inferência que ainda precisa de teste.
- **Verificação do Juiz:** os itens marcados ✔︎ foram reabertos e conferidos linha a linha no código pelo coordenador, independentemente do agente que os reportou.
- Os relatórios integrais, com snippets, cenários numéricos e testes sugeridos, estão nos Anexos A–E. Esta parte principal é a síntese.

---

## 1. Sumário executivo

**Diagnóstico geral.** O pacote está **sólido em engenharia básica**:
- não houve UB, crash, loop infinito nem dependência da ordem das linhas em nenhum dos fuzzers com ASan/UBSan;
- a convenção de fronteira `(a, b]` é a mesma em treino e apply;
- o RNG é o do R;
- o p-valor exato de Fisher, o núcleo Fayyad–Irani do `FastMDLPM`, o PAVA do `IR` e a DP do `OBC_IVB` estão corretos.

O problema dominante **não é de crash, é de fidelidade estatística**. Há um descompasso sistemático entre o que cada algoritmo **promete** (nome, roxygen, referência citada) e o que **executa**, e há critérios de fusão sem fundamento teórico. Na prática, isso deixa uma parte material do IV na mesa e, em alguns casos, produz WoE errado ou não monotônico rotulado como monotônico.

**Números-chave**, medidos em réplicas C++ fiéis ao código e com dados sintéticos:

| Fenômeno | Algoritmos | Perda/erro medido |
|---|---|---|
| Critério de fusão "menor IV individual", "soma de IV", "\|ΔIV\|" ou "\|divergência\|" em vez da **perda de IV** | BB, OSLP, MRBLP, DMIV-num, JEDI, JEDI-MWoE | IV médio de **55%** (BB), **80%** (OSLP), **94%** (MRBLP) e **50–70%** (DMIV) do ótimo. A troca pela perda de IV recupera **98–99,9%** |
| Bin puro (0 eventos ou 0 não-eventos) recebe **WoE = IV = 0** | OBC_DP, OBC_FETB | O DP descarta a melhor partição (IV 0,71 vs 2,59) e o WoE fica não monotônico |
| Seleção gulosa de cortes por IV de corte único | OBN_Sketch (n > 50) | IV **−29%** vs DP sobre os mesmos candidatos |
| Simulated annealing para na iteração ~200 | OBC_SAB | IV **11–17%** abaixo do ótimo, com `converged=TRUE` |
| Monotonicidade "garantida" falha em silêncio quando `min_bins` bloqueia a fusão | MOB, JEDI, DMIV, MBLP, MRBLP, OSLP, UDT, KMB, EWB, BB, LDB… | 53% (MOB) e 83% (MBLP) de saídas não monotônicas em dados em U, todas com `converged=TRUE` |
| O critério estatístico nunca decide (é sobrescrito por `max_bins`) | ChiMerge num., Chi2 num./cat., FETB, MDLP | α = 0,5, 0,05 e 0,001 dão resultado **idêntico** |
| Colapso de variáveis fortíssimas | OBN_CM (`merge_zero_bins`) | `y = 1[x>0]` vira **1 bin com IV = 0** |
| "Raro" definido também pela fração de **eventos** | OBC_MBA, OBC_Sketch | Bins grandes de baixo risco (os mais valiosos) são fundidos |
| Pelo menos 6 definições de WoE/IV coexistem | todo o pacote | IV total **não comparável** entre algoritmos (e `obwoe` ranqueia variáveis por IV) |

**Contagem consolidada** (após deduplicação; detalhes na seção 5):

| Severidade | Nº |
|---|---|
| Crítico | **1** (B-01) |
| Alto | **24** itens de agente, agrupados em 16 entradas consolidadas (X-02…X-17) |
| Médio | 29 |
| Baixo | 33 |

**Recomendação do Juiz:**
1. Corrigir já um conjunto pequeno de *quick wins* de 1 a 10 linhas e baixo risco (Onda 0, §8), que eliminam o Crítico e a maior parte dos Altos.
2. Em seguida, criar **dois componentes comuns**: um backend único de WoE/IV e um motor de fusão por perda de IV com DP monotônico exato (§7, código já validado contra força bruta). Com eles, 10 a 12 algoritmos convergem para o ótimo sob as mesmas restrições.
3. Por último, alinhar nomes e documentação às implementações reais.
4. **Toda** alteração, de qualquer onda, só é aceita depois de passar pelo **protocolo diferencial obrigatório de §9**: comparação lado a lado com o baseline 1.14.1 congelado, paridade de bit fora do escopo, não-inferioridade de IV, performance e generalização fora da amostra, e fuzzing diferencial. Assim nenhuma "melhoria" introduz bugs, regressões ou fraquezas.

---

## 2. Metodologia

### 2.1 Equipe

Cinco agentes autônomos com especialidades distintas, cada um responsável por uma família de algoritmos afim à sua área. Todos aplicaram as mesmas quatro lentes, e o coordenador atuou como Juiz.

| Agente | Especialidade | Escopo |
|---|---|---|
| **A** | Estatístico PhD: inferência, teoria da informação, testes | MDLP, FastMDLPM, ChiMerge/Chi2 (num./cat.), Fisher (num./cat.), `entropy_utils.h`, `chi_square_utils.h` |
| **B** | Otimização / Pesquisa Operacional | DP (num./cat.), MILP, LPDB, MBLP, MRBLP, OSLP, SBLP |
| **C** | ML e meta-heurísticas | UDT (num./cat.), IR (PAVA), KMB, SAB, BB, EWB |
| **D** | Risco de crédito e scorecards | MOB, JEDI, JEDI-MWoE, DMIV (num./cat.), IVB, `woe_iv_utils.h`, `monotonicity_utils.h` |
| **E** | C++ científico, análise numérica, HPC | Sketch (num./cat.), GMB, MBA, SWB, UBSD, LDB, todos os headers, Apply/Cutpoints, OpenMP, varredura transversal |

### 2.2 As quatro lentes

1. **Fidelidade teórica:** o critério, a estatística, a função objetivo e a regra de parada são os do paper?
2. **Robustez/corretude:** edge cases (n pequeno, bins puros, um único valor, NA/Inf, `min_bins > max_bins`), UB, terminação, determinismo.
3. **Performance:** complexidade atual vs. a que a teoria permite.
4. **Estabilidade numérica:** log(0), cancelamento, epsilons, clamps que mascaram, sensibilidade a perturbações.

### 2.3 Evidência

- Não há R no container. Para provar achados, os agentes extraíram trechos literais do código para programas C++ standalone (com um stub de `Rcpp.h`), compilaram com `g++ -fsanitize=address,undefined` e compararam contra:
  - força bruta (enumeração de todas as partições contíguas);
  - DP exato;
  - recursões de referência;
  - valores tabelados (`fisher.test`, `qchisq`).
- Os experimentos ficaram em diretórios temporários de trabalho e não fazem parte do repositório. Os cenários numéricos estão descritos nos anexos de forma reproduzível.

### 2.4 Consenso

O Juiz consolidou os achados de quatro formas:
- **(i) reverificação no código-fonte** dos itens marcados ✔︎;
- **(ii) deduplicação**: o mesmo defeito achado por dois ou mais agentes vira um item consolidado `X-n`, e a convergência independente aumenta a confiança;
- **(iii) recalibração de severidade** quando necessário;
- **(iv) validação da correção-chave**: o motor de DP monotônico proposto em §7.2 foi testado contra força bruta em 3000 instâncias aleatórias, com 0 divergências e ASan/UBSan limpos.

**Correção de localização feita pelo Juiz:** E-04 (GMB) está em `src/OBC_GMB_v5.cpp:330-345` (`absorber`), não em 285-305 como no relatório original.

---

## 3. Veredito por algoritmo

Legenda de fidelidade: **Alta**, **Parcial** ou **Baixa**. A coluna "Nome ≠ algoritmo" marca quando o rótulo promete um método que não está implementado. A coluna "Pior achado" traz o achado mais grave de cada algoritmo.

### 3.1 Numéricos (21)

| Algoritmo | Arquivo | Referência declarada | Fidelidade | Nome ≠ algoritmo? | Pior achado | Prioridade |
|---|---|---|---|---|---|---|
| MDLP | `OBN_MDLP_v5.cpp` | Fayyad & Irani 1993 | **Baixa** | Sim (critério MDL próprio, praticamente inerte) | A-01 (Alto) | P2 |
| FastMDLPM | `OBN_FastMDLPM_v5.cpp` | F&I 1993 + 1992 | **Alta** (núcleo exato) | Não | A-04: `force_min_bins` descarta cortes legítimos (Alto) | P1 |
| ChiMerge/Chi2 | `OBN_CM_v5.cpp` | Kerber 1992; Liu & Setiono 1995 | **Baixa** | Parcial | A-06: colapso para 1 bin (Alto) | **P0** |
| Fisher (FETB) | `OBN_FETB_v5.cpp` | Fisher exato | Alta (p-valor) / Parcial | Não | A-11/A-12 (Médio) | P2 |
| DP | `OBN_DP_v5.cpp` | Navas-Palencia 2020 | **Baixa** (é guloso) | Sim | B-05/B-06 | P1 |
| LPDB | `OBN_LPDB_v5.cpp` | Densidade polinomial local | **Baixa** (KDE; `polynomial_degree` morto) | Sim | B-05/B-11 | P2 |
| MBLP | `OBN_MBLP_v5.cpp` | MILP monotônico | **Baixa** | Sim (sem LP) | B-05: 83% não monotônico em U (Alto) | P1 |
| MRBLP | `OBN_MRBLP_v5.cpp` | — | **Baixa** | Sim (sem LP) | B-04: \|ΔIV\| (Alto) | **P0** |
| OSLP | `OBN_OSLP_v5.cpp` | — | **Baixa** | Sim (sem LP) | B-03: soma de IV, 80% do ótimo (Alto) ✔︎ | **P0** |
| Branch & Bound | `OBN_BB_v5.cpp` | Land & Doig 1960 | **Baixa** | Sim (sem B&B) | C-05: menor IV individual, 55% do ótimo (Alto) ✔︎ | **P0** |
| Isotonic (IR) | `OBN_IR_v5.cpp` | Barlow 1972; Best & Chakravarti 1990 | Parcial (PAVA correto) | Não | C-10 (Médio) | P2 |
| K-means (KMB) | `OBN_KMB_v5.cpp` | Lloyd 1982; Wang & Song 2011 | **Baixa** (é equal-width) | Sim | C-07/C-14 (Médio) | P2 |
| Equal-width (EWB) | `OBN_EWB_v5.cpp` | Dougherty et al. 1995 | Alta no pré-bin | Não | C-06: um outlier gera 2 bins < `min_bins` (Alto) | P1 |
| UDT | `OBN_UDT_v5.cpp` | CART / C4.5 | **Baixa** (top-K de ganhos na raiz) | Sim | C-01: perde pontos de mudança (Alto) | P1 |
| MOB | `OBN_MOB_v5.cpp` | Mironchyk & Tchistiakov 2017 | Parcial (IV quase ótimo: 99,5%) | Não | D-04: monotonia silenciosa (Alto) | P1 |
| JEDI | `OBN_JEDI_v5.cpp` | — (sem entropia) | Baixa/Parcial | Parcial | D-02: soma de IV + piso EPS (Alto) | **P0** |
| JEDI-MWoE | `OBN_JEDIMWoE_v5.cpp` | WoE multinomial one-vs-rest | Parcial | Não | D-02 / D-03: monotonia forçada em K classes (Alto) | P1 |
| DMIV | `OBN_DMIV_v5.cpp` | Zeng 2013/2014 | Baixa na otimização (fórmulas corretas) | Não | D-01: IV 30–50% abaixo do MOB (Alto) ✔︎ | **P0** |
| Sketch (KLL) | `OBN_Sketch_v5.cpp` | Karnin–Lang–Liberty 2016 | **Parcial/Baixa** | Sim (KLL determinístico, sem streaming real) | E-01: −29% de IV (Alto) ✔︎ | P1 |
| UBSD | `OBN_UBSD_v5.cpp` | Média ± kσ | Alta (ao que documenta) | Não | E-15 (Baixo) | P3 |
| LDB | `OBN_LDB_v5.cpp` | KDE (Silverman 1986) | Parcial | Não | E-06/E-07 (Médio) | P2 |

### 3.2 Categóricos (16)

| Algoritmo | Arquivo | Referência declarada | Fidelidade | Nome ≠ algoritmo? | Pior achado | Prioridade |
|---|---|---|---|---|---|---|
| ChiMerge/Chi2 | `OBC_CM_v5.cpp` | Kerber; Liu & Setiono | Parcial | Parcial (Chi2 só roda α=0,5) | A-07: parada indevida com χ²=0 (Alto) ✔︎ | **P0** |
| Fisher (FETB) | `OBC_FETB_v5.cpp` | Fisher exato | Alta (p-valor) | Não | A-09: categoria pura com WoE=0 (Alto) ✔︎ | **P0** |
| DP | `OBC_DP_v5.cpp` | Bellman; Fisher 1958 / Breiman 1984 | Parcial (DP exato do objetivo errado) | Não | **B-01: CRÍTICO** ✔︎ | **P0** |
| MILP | `OBC_MILP_v5.cpp` | Navas-Palencia 2020 | Baixa (guloso; doc honesta) | Sim | B-09/B-10 | P3 |
| SBLP | `OBC_SBLP_v5.cpp` | DP em ordem de taxa | Parcial | Não | B-02: objetivo do DP ≠ IV reportado (Alto) | P1 |
| SAB | `OBC_SAB_v5.cpp` | Kirkpatrick 1983 | Parcial | Parcial | C-02: para na iteração ~200 (Alto) ✔︎ | P1 |
| UDT | `OBC_UDT_v5.cpp` | "User-Defined Technique" | Parcial (doc honesta) | Não | C-16 (Baixo) | P3 |
| MOB | `OBC_MOB_v5.cpp` | Mironchyk & Tchistiakov | Parcial | Não | D-05: `bin_cutoff` ignorado se ncat ≤ max_bins (Alto) ✔︎ | P1 |
| JEDI | `OBC_JEDI_v5.cpp` | — | Parcial | — | D-11 (Médio) | P2 |
| JEDI-MWoE | `OBC_JEDIMWoE_v5.cpp` | WoE multinomial | **Baixa** | — | D-08: adjacência por contagem (Médio) | P2 |
| DMIV | `OBC_DMIV_v5.cpp` | Zeng 2013 | Parcial | — | D-06: `bin_cutoff` morto, O(B²) (Alto) | P1 |
| IVB | `OBC_IVB_v5.cpp` | Siddiqi / Zeng (DP) | **Alta** (DP exata) | Não | D-15 (Baixo) | P3 |
| GMB | `OBC_GMB_v5.cpp` | Greedy merge de IV | Parcial | Não | E-04: excedente dobrado no menor bin, −22% de IV (Alto) ✔︎ | P1 |
| MBA | `OBC_MBA_v5.cpp` | Mironchyk & Tchistiakov | Parcial | — | E-03: cutoff sobre positivos (Alto) ✔︎ | **P0** |
| SWB | `OBC_SWB_v5.cpp` | "Sliding window" | **Baixa** (não há janela) | Sim | E-17 (Baixo) | P3 |
| Sketch (Count-Min) | `OBC_Sketch_v5.cpp` | Cormode & Muthukrishnan 2005 | **Baixa** (CMS decorativo) | Sim | E-03 (Alto) ✔︎ / E-02 (Médio) ✔︎ | **P0** |

**Leitura rápida:**
- Dos 37 algoritmos, só **4 têm fidelidade Alta** ao que prometem: FastMDLPM, IVB, UBSD e o núcleo de Fisher.
- **15 têm nome ou rótulo que promete um método não implementado.**
- O **MOB numérico** é o melhor heurístico do pacote: 99,5% do ótimo monotônico.

---

## 4. Temas transversais (consenso da equipe)

Os cinco agentes, trabalhando de forma independente sobre famílias diferentes, convergiram para os mesmos padrões. Essa convergência é a evidência mais forte desta auditoria.

### T1 — Critério de fusão sem fundamento teórico (achado por B, C, D e E)

**Teoria.** Na fusão gulosa (*agglomerative*) de bins adjacentes, o par a fundir é o que minimiza a **perda de informação**:

$$
\Delta \mathrm{IV}_{i,i+1} \;=\; \mathrm{IV}_i + \mathrm{IV}_{i+1} - \mathrm{IV}_{i\cup i+1} \;\ge 0,
$$

em que $\mathrm{IV}_j = (p_j - q_j)\ln(p_j/q_j)$, com $p_j$ e $q_j$ as frações de eventos e não-eventos no bin $j$. A desigualdade vem da convexidade da f-divergência de Jeffreys (desigualdade log-soma; Cover & Thomas, Teo. 2.7.1).

Por isso, critérios como $\min_i \mathrm{IV}_i$, $\min (\mathrm{IV}_i+\mathrm{IV}_{i+1})$ ou $\min |\mathrm{IV}_i-\mathrm{IV}_{i+1}|$ são **inválidos**:
- **Soma de IV:** dois bins puros vizinhos com o mesmo WoE extremo têm soma de IV alta, mas perda zero. São exatamente os que deveriam ser fundidos primeiro.
- **|ΔIV|:** dá zero para bins com WoE de sinais opostos e IVs iguais, que são os mais distintos.

**Onde ocorre:**

| Algoritmo | Critério atual | Local |
|---|---|---|
| OBN_BB | bin de menor IV, fundido sempre com o **anterior** | `OBN_BB_v5.cpp:512-532` ✔︎ |
| OBN_OSLP | menor soma IV_i + IV_{i+1} (a função `computeIVLoss` já existe no arquivo) | `OBN_OSLP_v5.cpp:372, 406` ✔︎ |
| OBN_MRBLP | menor \|IV_i − IV_{i+1}\| | `OBN_MRBLP_v5.cpp:496, 509` |
| OBN_DMIV | menor \|divergência\| individual, fundido com o anterior | `OBN_DMIV_v5.cpp:724` ✔︎ |
| OBN_JEDI / OBN_JEDIMWoE | menor soma de IV | `OBN_JEDI_v5.cpp:417`, `OBN_JEDIMWoE_v5.cpp:536` |
| OBN_MOB (fusão de raros) | "perda" = IV_a + IV_b, sem subtrair o fundido | `OBN_MOB_v5.cpp:383-385` |
| OBN_DP | menor \|ΔWoE\| (ignora a massa) | `OBN_DP_v5.cpp` (`find_smallest_woe_diff`) |
| OBC_Sketch | alterna JS ↔ \|IV\| a cada 5 iterações | `OBC_Sketch_v5.cpp:632-668` |
| OBC_GMB (pré-bin) | dobra todo o excedente no menor bin | `OBC_GMB_v5.cpp:330-345` ✔︎ |
| OBN_Sketch | top-(k−1) cortes por IV de corte único | `OBN_Sketch_v5.cpp:455-461` ✔︎ |

**Evidência.** Trocar só o critério pela perda de IV elevou o IV médio para **98,4%** (BB) e **99,9%** (MRBLP) do ótimo exaustivo. Correção: §7.1 (`iv_merge_loss`) e §7.2 (DP exato).

### T2 — Bins puros: WoE = IV = 0 ou colapso (achado por A, B e D)

**Teoria.** Um bin sem eventos tem $p_j = 0$, logo $\mathrm{WoE}_j \to -\infty$ e $\mathrm{IV}_j \to +\infty$. Ele é o bin **mais** informativo, nunca o neutro. O tratamento correto é a suavização aditiva normalizada (§7.1) seguida de clamp finito.

**Defeitos encontrados:**
- **`OBC_DP_v5.cpp:76-82`** ✔︎ e **`OBC_FETB_v5.cpp:213`** ✔︎ devolvem `woe = iv = 0` para bins puros. O DP passa a otimizar um objetivo que *penaliza* segmentos puros (no contraexemplo mínimo, IV escolhido 0,71 vs 2,59 descartado). O WoE reportado deixa de ser monotônico, e a ordenação inicial por WoE do FETB fica corrompida.
- **`OBN_CM_v5.cpp:219-247`** ✔︎ (`merge_zero_bins`) funde todo bin puro **ignorando `min_bins`**. Uma flag binária fortíssima, ou uma variável separável, colapsa para **1 bin com IV = 0** e é descartada na seleção de variáveis.
- `OBN_JEDI` / `OBN_JEDIMWoE` usam piso EPS = 1e-10, sem suavização nem clamp. O WoE chega a ≈ −20,6…−23 e o IV de bins puros vira artefato (IV total de 3,7 no caso de D-02).
- `OB_Cutpoints.cpp:23-24` aplica o piso 1e-4 **só** quando a contagem é exatamente zero. O WoE deixa de ser monótono na contagem: 0 eventos dá −2,30 e 1 evento dá −6,91.

### T3 — Pelo menos seis definições de WoE/IV (achado por A, B, C, D e E)

| Família | Fórmula | Σ distribuições = 1? |
|---|---|---|
| `bin_structures.h::calculate_metrics` | Bayes, prior 0,5·taxa global | Não |
| `woe_iv_utils.h` Laplace | $(c+\alpha)/(T+2\alpha)$ | **Não** (para K > 2) |
| OBN_CM / OBC_CM | $(c+0{,}5)/(T+1)$; IV com distribuição **crua** × WoE suavizado | Não |
| FETB, MDLP, MOB, DMIV, UBSD | $(c+\alpha)/(T+K\alpha)$ | **Sim** |
| LDB, KMB, LPDB, UDT-NA | WoE suavizado × distribuição **crua** no IV | Não; o IV pode ser **negativo** |
| JEDI, Sketch-num, Cutpoints | sem suavização, com pisos diferentes | — |

**Consequências:**
- O mesmo particionamento produz IV diferente por algoritmo, e `obwoe` ranqueia variáveis por IV.
- A "perda de IV" mistura escalas quando o K da suavização muda entre o antes e o depois da fusão.
- `MAX_WOE` / `MIN_WOE` existem em `optimal_binning_common.h`, mas só três arquivos os usam.

Correção: §7.1.

### T4 — Monotonicidade "garantida" e `converged` que não informam a verdade (achado por B, C, D e E)

Em praticamente todos os algoritmos numéricos com restrição de monotonicidade, o laço que funde violações para em `bins.size() == min_bins` e devolve o WoE não monotônico com `converged = TRUE` e `monotonicity = "increasing"`. A documentação diz "Guaranteed" (`R/obn_mob.R`, `R/obn_mblp.R:5`).

**Medido em dados em U:**

| Algoritmo | Saídas não monotônicas |
|---|---|
| MOB | 106/200 |
| DMIV | 52/100 |
| MBLP | 82,6% |
| MRBLP / OSLP (dados monótonos ruidosos) | 11,9% |

**Agravante.** Em KMB e EWB a direção é decidida pelos **dois primeiros bins** (erra 16% das vezes com tendência fraca). Em `monotonicity_utils.h` a direção sai de Pearson sobre o valor bruto, e NaN vira DESCENDING.

**Correção:**
- **(a) Imediata:** campo `monotonic` no resultado e `Rcpp::warning` quando a monotonia não for atingida.
- **(b) Estrutural:** o DP monotônico exato de §7.2. Ele devolve a melhor partição monotônica com K ∈ [min_bins, max_bins] ou declara inviabilidade explicitamente.
- **(c) Direção:** escolher a direção do maior IV entre as duas soluções exatas (ou, no mínimo, a de menor SSE isotônico).

### T5 — O critério estatístico declarado nunca decide (achado por A)

| Algoritmo | Defeito |
|---|---|
| OBN_CM | `enforce_bin_limits()` roda **antes** do laço; α nunca é comparado (α = 0,5, 0,05 e 0,001 dão resultado idêntico) ✔︎ `OBN_CM_v5.cpp:1098-1101` |
| Chi2 (num.) | Regra de inconsistência **invertida** (`< 0,05 → break`) |
| Chi2 (cat.) | Só roda o nível α = 0,5 |
| OBC_CM | Para quando \|Δ min χ²\| < 1e-6. Com dois χ² = 0 consecutivos (Yates truncado), 10 categorias de taxa idêntica ficam sem fundir ✔︎ `OBC_CM_v5.cpp:615-620` |
| FETB (num./cat.) | O p-valor só ordena os pares; não há α |
| MDLP | O critério `log2(k−1) + Σ nH` é praticamente inerte: 41–43 de 50 pré-bins sobrevivem até em ruído puro. Não é Fayyad–Irani |

**Teoria (Kerber 1992).** Fundir enquanto $\min \chi^2 < \chi^2_{1-\alpha,\,df}$, com `max_bins` apenas como **teto**. No Chi2 de Liu & Setiono (1995), cada fase reduz α e o algoritmo **para e reverte** quando a inconsistência ultrapassa δ.

### T6 — Nome ≠ algoritmo (achado por todos)

Nenhum dos oito algoritmos rotulados "LP/MILP/DP/B&B" resolve programação matemática: um grep por `simplex|solver|branch|lp_` não retorna nada. A lista completa de algoritmos cujo nome promete um método que não está implementado:

| Algoritmo | O que de fato faz |
|---|---|
| OBN_DP | guloso |
| OBC_MILP, MBLP, MRBLP, OSLP | gulosos |
| OBN_BB | guloso |
| OBN_LPDB | KDE gaussiano; `polynomial_degree` é parâmetro morto |
| OBN_KMB | equal-width, sem iteração de Lloyd |
| OBN_UDT | não é árvore recursiva |
| OBC_SAB | quase não faz annealing |
| OBC_SWB | não há janela deslizante |
| OBN_Sketch | KLL determinístico, com os dados inteiros em memória |
| OBC_Sketch | Count-Min nunca lido |
| OBN_MDLP | não é F&I |

Parte da documentação já admite isso; outra parte promete otimalidade:
- `R/obc_dp.R` fala em "global optimality";
- `R/obc_sblp.R` diz "guarantees an optimal partition";
- `R/obn_bb.R` diz "optimally";
- `R/obn_mblp.R` diz "guaranteed monotonicity".

### T7 — `min_bins` e `bin_cutoff` não são pós-condições (achado por A, C, D e E)

| Algoritmo | Defeito |
|---|---|
| OBN_EWB | Um outlier gera 2 bins (99,9% / 0,1%) com `min_bins=3`; `drop_empty_bins` não recompõe `min_bins` |
| OBN_CM | `merge_zero_bins` viola `min_bins` |
| OBN_FastMDLPM | `force_min_bins` faz `splits.clear()` e troca o corte F&I legítimo por cortes uniformes |
| OBN_IR | Pré-bins limitados a `max_bins`; o PAVA só reduz |
| OBN_UDT | Deixa um bin de 0,12% |
| OBC_MOB | Ignora `bin_cutoff` quando ncat ≤ max_bins (✔︎ `OBC_MOB_v5.cpp:584`) |
| OBC_DMIV | `bin_cutoff` é parâmetro morto (constante fixa 5) |
| OBC_MBA / OBC_Sketch | Aplicam o cutoff também à fração de **positivos** (✔︎ `OBC_MBA_v5.cpp:235-242`, `OBC_Sketch_v5.cpp:504-510`) |
| OBC_Sketch | Índices obsoletos em `enforce_bin_cutoff`, o mesmo bug já corrigido no MBA (✔︎ `OBC_Sketch_v5.cpp:~543-547`) |

### T8 — Performance em alta cardinalidade

O custo é quadrático no número de categorias em três algoritmos categóricos, e nenhum deles chama `checkUserInterrupt`:

| Algoritmo | Medido |
|---|---|
| OBC_MBA (pré-binning) | 2,6 s com 40k categorias; estimativa de ~7 min com 500k |
| OBC_MOB (fusão de raros) | 11 s com 64k categorias |
| OBC_DMIV | 3,7 s com 16k categorias; `max_n_prebins` não é respeitado (14 371 pré-bins) |

Em todo o pacote só existem 4 chamadas a `Rcpp::checkUserInterrupt`. A correção é ordenar por taxa e fundir adjacentes com lista ligada e heap, em O(B log B).

### T9 — Infraestrutura

| Item | Problema |
|---|---|
| (a) ODR | `utils::join_categories` tem **corpos diferentes** em TUs diferentes, o que é UB latente |
| (b) Makevars | `CXX_STD = CXX17` está comentado, mas o código usa structured bindings e o `DESCRIPTION` declara C++17 |
| (c) Sentinela `-999` | `ob_apply_woe_num` usa `missing_values = -999` por padrão, e nenhum binner trata −999 como missing |
| (d) NA | O wrapper troca NA por `"NA"`, o que colide com uma categoria literal "NA" |
| (e) Rótulos | Formatação diferente entre treino (`fixed 6`), apply (`%g`) e cutpoints (2 casas); `'+'` é usado como separador irreservável |
| (f) Código morto | `validation_utils.h` e `cutpoints_validator.h` não são incluídos em lugar nenhum; `get_chi_critical_value(0.05)` devolve **0,004** (armadilha se alguém passar a usá-la) |

---

## 5. Registro consolidado de achados (Crítico e Alto)

Cada item consolidado `X-n` agrupa os IDs originais.

| ID | Sev. | Algoritmo(s) / local | Defeito | Correção proposta (resumo) | Esforço | Origem |
|---|---|---|---|---|---|---|
| **X-01** | **Crítico** | OBC_DP `OBC_DP_v5.cpp:76-91` ✔︎ | Bin puro com WoE=IV=0. O DP maximiza o objetivo errado, o WoE fica não monotônico, e em 134/3000 casos o resultado não é nem o ótimo do próprio objetivo | Estimador suavizado normalizado com K fixo por rodada do DP (§7.1); rodar a DP uma vez por K | P | B-01 |
| X-02 | Alto | OBC_FETB `:213` ✔︎ | Categoria pura com WoE=IV=0 corrompe a ordenação e as fusões | Mesmo estimador de §7.1 | P | A-09 |
| X-03 | Alto | OBN_CM `:219-247` ✔︎ | `merge_zero_bins` ignora `min_bins`: `y=1[x>0]` vira 1 bin com IV=0 | Guarda `bins.size() > max(min_bins,1)`; não chamar no caminho de ≤ 2 valores; confiar na suavização | P | A-06 |
| X-04 | Alto | OBN_CM `:1098`, `:616-650` ✔︎; Chi2 num./cat. | α inoperante; Chi2 invertido e com fases inalcançáveis | Laço de Kerber com `max_bins` como teto (snippet em A-05); reimplementar o Chi2 com snapshot e reversão (A-08) | M | A-05, A-08 |
| X-05 | Alto | OBC_CM `:615-620` ✔︎ | Para quando \|Δmin χ²\| < tol com χ²=0 consecutivos (11 bins em vez de 3) | Remover o bloco | P | A-07 |
| X-06 | Alto | OBN_MDLP `:752-870` | Não é F&I; o critério MDL é inerte | Reusar o motor F&I do FastMDLPM (`Blocks`, `best_mdlp_cut`) e corrigir o roxygen | M | A-01 (+A-02) |
| X-07 | Alto | FastMDLPM `:361-445` | `force_min_bins` descarta os cortes F&I | Manter os `splits` e acrescentar o corte de maior ganho sem o teste MDL (snippet em A-04) | P | A-04 |
| X-08 | Alto | OBN_BB ✔︎, OBN_OSLP ✔︎, OBN_MRBLP, OBN_DMIV ✔︎, OBN_JEDI, OBN_JEDIMWoE, OBN_MOB (raros) | Critério de fusão inválido (T1). IV de 55–94% do ótimo | `iv_merge_loss` (§7.1) em todos; DP exato (§7.2) como meta | P → M | C-05, B-03, B-04, D-01, D-02, D-12 |
| X-09 | Alto | OBN_Sketch `:455-461` ✔︎ | Top-K de cortes por IV de corte único; a DP só roda com n ≤ 50; −29% de IV | DP (§7.2, sem monotonia) sobre os ≤ 37 candidatos, com suporte mínimo | P | E-01 (+E-05) |
| X-10 | Alto | OBC_SBLP `:493, :535-552` | Suavização no DP com `e−s` ≠ IV reportado com K; ordena por taxa suavizada; "chunked" é arbitrário | K fixo por rodada; ordem pela razão exata em inteiros; remover `chunked` | M | B-02 |
| X-11 | Alto | MOB, JEDI, DMIV, MBLP, MRBLP, OSLP, DP, LPDB, UDT, KMB, EWB, BB, LDB, Sketch | Monotonia "garantida" falha em silêncio; `converged=TRUE` (T4) | (a) campo `monotonic` + `warning`; (b) DP monotônico (§7.2); (c) direção pelo maior IV | P → M | B-05, D-04, C-09, E-14 |
| X-12 | Alto | OBC_SAB `:241, :531-550` ✔︎ | Parada na iteração ~200; penalidade de −500 dependente do rótulo | Substituir por DP exato em ordem de taxa (§7.2), O(k·m²); se mantiver o SA: janela `max(200, 20m)` e sem penalidade | P | C-02 (+C-03, C-04) |
| X-13 | Alto | OBN_UDT `:425-520` | Pré-bin com os top-K ganhos calculados na raiz: 20 cortes em [0,697; 0,701] | Best-first recursivo com `min_leaf` (esboço em C-01) | M | C-01 |
| X-14 | Alto | OBN_EWB `:300-370, 514-526` | Um outlier gera 2 bins < `min_bins` | Bisseção do bin mais populoso, como o `removeEmptyBins` do KMB; função comum `restore_min_bins` | P | C-06 |
| X-15 | Alto | OBC_MBA `:235-260` ✔︎, OBC_Sketch `:504-521` ✔︎, OBC_Sketch `:543-547` ✔︎ | Cutoff aplicado sobre positivos (não documentado); índices obsoletos | Remover `min_count_pos` (ou criar `min_events` opt-in); correção de índices `r==erased → kept` | P | E-03, E-02 |
| X-16 | Alto | OBC_GMB `:330-345` ✔︎, OBC_MOB `:584` ✔︎, OBC_DMIV `:88,140,228` | Excedente dobrado sem olhar a taxa (−22% de IV); `bin_cutoff` ignorado ou morto; `max_n_prebins` não respeitado | Fusão adjacente por perda de IV até `max_n_prebins`; sempre executar `handleRareCategories`; aplicar `bin_cutoff` no DMIV | P | E-04, D-05, D-06 |
| X-17 | Alto | OBN_JEDIMWoE `:481-505` | Monotonia imposta a todas as K classes (um bin com 70% dos dados; −20% de IV) | Monotonia só para K=2 ou para uma classe de referência; desligada por padrão para K ≥ 3 | P | D-03 |

> Esforço: **P** = até ~30 linhas, local; **M** = refatoração de uma classe ou reaproveitamento de motor comum.

---

## 6. Achados Médios e Baixos (índice)

Os detalhes completos, com cenário e snippet de correção, estão nos anexos.

**Médios:**

| Tema | IDs |
|---|---|
| Suavização inconsistente | A-10, B-09, C-14, D-10, E-06 |
| Direção de monotonia frágil | A-11, C-08, C-11, D-16 |
| FETB sem α | A-12 |
| OBN_DP guloso | B-06 |
| Pré-bin do OBC_DP / SBLP | B-07, B-08 |
| SAB: penalidade e schedule | C-03 |
| KMB não é k-means | C-07 |
| IR: pré-bins ≤ `max_bins` | C-10 |
| UDT: bin NA com outra normalização | C-13 |
| OBC_MOB O(B²) | D-07 |
| OBC_JEDIMWoE: adjacência por contagem | D-08 |
| DMIV: `woe1` é log-odds | D-09 |
| `keep_rare` e loop morto em JEDI-cat | D-11 |
| Fusão de raros | D-12 |
| KL ≡ KLJ; L1/L2/L∞ equivalentes | D-13 |
| Categorias não vistas → NA | D-14 |
| Sketch-num com n ≤ 50 | E-05 |
| KDE do LDB some com outliers | E-07 |
| Piso 1e-4 do Cutpoints | E-08 |
| Sentinela −999 | E-09 |

**Baixos:**

| Tema | IDs |
|---|---|
| Pré-bins com `floor(N/P)` | A-03, B-12 |
| Desvios menores de FastMDLPM / CM | A-13 |
| Performance de Fisher-cat e XLogX O(N) de memória | A-14 |
| Tabela χ² morta e incorreta | A-15 |
| NaN em `feature.size()` | A-16 |
| Rótulos com 6 casas | A-17, C-18, E-19 |
| Nomes "LP/MILP" | B-10 |
| Parâmetros mortos | B-11, C-15 |
| NA → "NA" | B-13, D-14 |
| Metropolis com p > 1 e reaquecimento | C-04 |
| IR | C-11, C-12 |
| OBC_UDT | C-16 |
| `std::sort` instável e `(a+b)/2` | C-17 |
| Equal-width com poucos distintos | C-19 |
| OBC_IVB | D-15 |
| Menores de MOB / JEDI | D-17 |
| Performance O(B²) de MBA / Sketch | E-10 |
| Count-Min decorativo e hashes correlacionados | E-11 |
| Critério alternado no Sketch | E-12 |
| "KLL" não é KLL | E-13 |
| Flags | E-14 |
| Tolerâncias absolutas | E-15 |
| LDB com poucos distintos | E-16 |
| SWB | E-17 |
| `""` inconsistente | E-18 |
| ODR / C++17 / interrupt / encoding | E-20 |
| Monotonia categórica trivial | E-21 |

---

## 7. Arquitetura de correção proposta

A maior parte dos achados Altos tem a **mesma raiz**: cada arquivo reimplementa WoE/IV e o critério de fusão. Em vez de remendar 37 arquivos, o Juiz recomenda criar **um header novo**, `src/common/ob_core.h`, com três peças.

### 7.1 Backend único de WoE/IV e perda de fusão

Suavização aditiva **normalizada** (as distribuições somam 1):

$$
\hat p_j = \frac{c^{+}_j + \alpha}{C^{+} + K\alpha},\qquad
\hat q_j = \frac{c^{-}_j + \alpha}{C^{-} + K\alpha},\qquad
\mathrm{WoE}_j = \ln\frac{\hat p_j}{\hat q_j},\qquad
\mathrm{IV}_j = (\hat p_j - \hat q_j)\,\mathrm{WoE}_j \ge 0
$$

Variáveis:
- $c^{\pm}_j$: contagem de eventos e não-eventos no bin $j$;
- $C^{\pm}$: totais;
- $K$: número de bins **da partição avaliada** (constante dentro de uma comparação);
- $\alpha$: pseudo-contagem, padrão 0,5 (Jeffreys).

Com $\hat p$ e $\hat q$ estritamente positivos não há log(0), e com $\Sigma \hat p = \Sigma \hat q = 1$ o IV é uma divergência de Jeffreys genuína (≥ 0).

```cpp
// src/common/ob_core.h  (proposal)
#pragma once
#include <cmath>
#include <cstdint>
#include <algorithm>

namespace ob {

constexpr double kWoeCap = 20.0;   // single clamp for the whole package

struct WoeIv { double woe; double iv; };

/// Normalized additive (Laplace/Jeffreys) smoothing.
/// @param pos,neg  events / non-events in the bin
/// @param Tp,Tn    totals over all bins of the partition
/// @param K        number of bins of the partition being evaluated
/// @param alpha    pseudo-count per bin (0.5 = Jeffreys prior)
inline WoeIv woe_iv(double pos, double neg, double Tp, double Tn,
                    int K, double alpha = 0.5) {
  const double dp = (pos + alpha) / (Tp + alpha * K);
  const double dn = (neg + alpha) / (Tn + alpha * K);
  const double w  = std::clamp(std::log(dp / dn), -kWoeCap, kWoeCap);
  return { w, (dp - dn) * w };          // same dp/dn in WoE and IV => IV >= 0
}

/// Information loss of merging adjacent bins a and b, evaluated with the
/// SAME K on both sides (K = current number of bins) so the comparison is
/// on a single scale. Always >= 0 up to rounding (log-sum inequality).
inline double iv_merge_loss(double pa, double na, double pb, double nb,
                            double Tp, double Tn, int K, double alpha = 0.5) {
  return woe_iv(pa, na, Tp, Tn, K, alpha).iv
       + woe_iv(pb, nb, Tp, Tn, K, alpha).iv
       - woe_iv(pa + pb, na + nb, Tp, Tn, K, alpha).iv;
}

/// Exact comparison of event rates pa/ca <= pb/cb using 128-bit integers
/// (no floating point, deterministic ties).
inline bool rate_le(int64_t pa, int64_t ca, int64_t pb, int64_t cb) {
  return static_cast<__int128>(pa) * cb <= static_cast<__int128>(pb) * ca;
}

} // namespace ob
```

> **Nota de portabilidade:** `__int128` existe em GCC e Clang, mas não em MSVC. Para o CRAN (Windows usa GCC via Rtools) isso é suficiente. Uma alternativa portátil é comparar em `long double` com desempate determinístico, já que as contagens são menores que 2⁵³.

### 7.2 DP monotônico exato (motor único de otimização)

**Teoria.**
- Para alvo binário e objetivo aditivo convexo (IV com K fixo), a partição ótima de categorias é **contígua na ordem da taxa de evento** (Fisher 1958; Breiman et al. 1984, Teo. 4.5). Logo, uma DP O(K·m²) resolve **exatamente** os categóricos (DP, SBLP, SAB, MILP, GMB, MBA, MOB-cat, DMIV-cat).
- Para numéricos com restrição de monotonicidade, o estado precisa guardar o último segmento: $f_k(i, j)$ é o melhor IV com $k$ bins cobrindo os pré-bins $[0, i)$, sendo o último bin $[j, i)$. A transição admite $[i, e)$ só quando a taxa respeita a tendência, o que dá complexidade **O(K·m³)**. Com m ≤ 50 pré-bins isso custa microssegundos.
- A mesma rotina **sem** a checagem de ordem resolve OBN_Sketch, OBN_BB e o redutor do DMIV.
- Ela incorpora `min_count` (o `bin_cutoff`) **dentro** da otimização, e não como pós-processamento guloso que destrói a otimalidade. Também declara inviabilidade de forma explícita (IV = −∞), em vez de devolver silenciosamente uma saída não monotônica.

**Validação.** O código abaixo foi comparado com força bruta em **3000 instâncias aleatórias** (m ∈ [4,10], K ∈ [2,4], tendência e `min_count` aleatórios): **0 divergências**, compilado com `-fsanitize=address,undefined` e sem nenhum aviso.

```cpp
// src/common/ob_core.h (continued) -- validated against brute force
#include <vector>
#include <limits>

namespace ob {

struct DpResult { double iv; std::vector<int> cuts; };  // cuts = pre-bin indices

/// Exact maximum-IV partition of m ordered pre-bins into exactly K contiguous
/// bins, each with at least `min_count` observations, and (if `monotone`)
/// with non-decreasing (asc=true) or non-increasing event rate.
/// Complexity O(K m^3) time, O(K m^2) memory. Returns iv = -inf if infeasible.
inline DpResult partition_dp(const std::vector<int64_t>& pos,
                             const std::vector<int64_t>& neg,
                             int K, bool monotone, bool asc,
                             int64_t min_count, double alpha = 0.5) {
  const int m = static_cast<int>(pos.size());
  std::vector<int64_t> P(m + 1, 0), N(m + 1, 0);
  for (int i = 0; i < m; ++i) { P[i+1] = P[i] + pos[i]; N[i+1] = N[i] + neg[i]; }
  const double Tp = P[m], Tn = N[m];
  const double NEG = -std::numeric_limits<double>::infinity();

  auto cnt = [&](int j, int i) { return (P[i]-P[j]) + (N[i]-N[j]); };
  auto val = [&](int j, int i) {
    return woe_iv(P[i]-P[j], N[i]-N[j], Tp, Tn, K, alpha).iv; };
  auto ordered = [&](int j, int i, int e) {            // [j,i) then [i,e)
    if (!monotone) return true;
    const int64_t p1 = P[i]-P[j], c1 = cnt(j,i), p2 = P[e]-P[i], c2 = cnt(i,e);
    return asc ? rate_le(p1, c1, p2, c2) : rate_le(p2, c2, p1, c1); };

  using M2 = std::vector<std::vector<double>>;
  using B2 = std::vector<std::vector<int>>;
  std::vector<M2> f(K + 1, M2(m + 1, std::vector<double>(m + 1, NEG)));
  std::vector<B2> bp(K + 1, B2(m + 1, std::vector<int>(m + 1, -1)));

  for (int i = 1; i <= m; ++i) if (cnt(0, i) >= min_count) f[1][i][0] = val(0, i);
  for (int k = 2; k <= K; ++k)
    for (int e = 1; e <= m; ++e)
      for (int i = 1; i < e; ++i) {
        if (cnt(i, e) < min_count) continue;
        const double v = val(i, e);
        for (int j = 0; j < i; ++j) {
          if (f[k-1][i][j] == NEG || !ordered(j, i, e)) continue;
          const double c = f[k-1][i][j] + v;
          if (c > f[k][e][i]) { f[k][e][i] = c; bp[k][e][i] = j; }
        }
      }

  DpResult r{NEG, {}};
  int bj = -1;
  for (int j = 0; j < m; ++j) if (f[K][m][j] > r.iv) { r.iv = f[K][m][j]; bj = j; }
  if (bj < 0) return r;                                // infeasible: caller warns
  for (int k = K, e = m, i = bj; k > 1; --k) {
    r.cuts.push_back(i); const int j = bp[k][e][i]; e = i; i = j;
  }
  std::reverse(r.cuts.begin(), r.cuts.end());
  return r;
}

/// Driver: best over K in [min_bins, max_bins] and, for monotone=AUTO, over
/// both directions. The IV values are comparable across K because each one
/// is the smoothed IV of the *reported* partition (same K used in output).
inline DpResult best_partition(const std::vector<int64_t>& pos,
                               const std::vector<int64_t>& neg,
                               int min_bins, int max_bins,
                               bool monotone, int64_t min_count,
                               bool* asc_out = nullptr) {
  DpResult best{-std::numeric_limits<double>::infinity(), {}};
  const int m = static_cast<int>(pos.size());
  for (int K = std::max(1, min_bins); K <= std::min(max_bins, m); ++K)
    for (int d = 0; d < (monotone ? 2 : 1); ++d) {
      DpResult r = partition_dp(pos, neg, K, monotone, d == 0, min_count);
      if (r.iv > best.iv) { best = r; if (asc_out) *asc_out = (d == 0); }
    }
  return best;
}

} // namespace ob
```

**Observação de modelagem (importante para o mantenedor).** Maximizar o IV em K sem penalidade sempre favorece o maior K viável, porque o IV é não decrescente em refinamentos. Isso é o comportamento esperado de "optimal binning" com `max_bins` como teto, e é o mesmo do OptBinning de Navas-Palencia. Se o objetivo for parcimônia, há duas opções: adicionar um termo de penalidade por bin, ou um critério de parada do tipo "ganho marginal de IV < τ".

**Uso nos algoritmos:**

| Algoritmo | Como usar |
|---|---|
| OBN_BB, OBN_DP | `best_partition(..., monotone=is_monotonic, ...)` |
| OBN_MBLP, MRBLP, OSLP, DMIV, JEDI | Substitui o par "fusão gulosa + enforce" |
| OBN_Sketch | `monotone=false`, `min_count = ceil(bin_cutoff·n)` |
| Categóricos | Ordenar por `rate_le` e chamar com `monotone=false` (a contiguidade na ordem de taxa já garante a monotonia) |
| Alta cardinalidade | Pré-agrupar até `max_n_prebins` com fusão adjacente pela menor `iv_merge_loss` (heap, O(B log B)) antes da DP |

### 7.3 Pós-condições comuns (contrato de saída)

Criar uma função `ob::finalize(bins, min_bins, max_bins, min_count, monotone)` que:
1. restaura `min_bins` por bisseção do bin mais populoso (generalizando o `removeEmptyBins` do KMB), quando há valores distintos suficientes;
2. recalcula WoE/IV **só** com §7.1;
3. preenche os campos `monotonic` (verificado), `converged` (semântica única: "nenhum limite de iteração interrompeu o trabalho") e `feasible`;
4. emite `Rcpp::warning` quando uma restrição solicitada não pôde ser satisfeita.

---

## 8. Plano de ação em ondas

> **Gate obrigatório:** cada item de cada onda é entregue como PR independente, com manifesto de diferenças esperadas e relatório diferencial contra o baseline congelado (§9). A Onda 0 só começa depois de criar a tag `v1.14.1-baseline` e o snapshot do corpus.

### Onda 0 — *Quick wins* (≈1–2 dias; baixo risco; elimina o Crítico e ~70% dos Altos)

| # | Ação | Arquivo:linha | Linhas | Itens resolvidos |
|---|---|---|---|---|
| 1 | Estimador suavizado no lugar de `woe=iv=0` | `OBC_DP_v5.cpp:76-91`, `OBC_FETB_v5.cpp:213` | ~10 | X-01, X-02 |
| 2 | Guarda `min_bins` em `merge_zero_bins`; não chamar no caminho de ≤ 2 valores | `OBN_CM_v5.cpp:224, 1058-1064` | ~3 | X-03 |
| 3 | Remover a parada por \|Δmin χ²\| | `OBC_CM_v5.cpp:615-620` | −6 | X-05 |
| 4 | Critério de fusão = `iv_merge_loss` | `OBN_BB:512-532`, `OBN_OSLP:406`, `OBN_MRBLP:509`, `OBN_DMIV:724`, `OBN_JEDI:417`, `OBN_JEDIMWoE:536`, `OBN_MOB:383` | ~10 cada | X-08 |
| 5 | Remover `min_count_pos`; corrigir índices | `OBC_MBA:235-260`, `OBC_Sketch:504-547` | ~8 | X-15 |
| 6 | Sempre tratar categorias raras | `OBC_MOB_v5.cpp:584` | ~3 | parte de X-16 |
| 7 | SAB: remover a penalidade −500 e a janela fixa | `OBC_SAB_v5.cpp:233, 241, 546` | ~10 | X-12 (paliativo) |
| 8 | Campo `monotonic` + `warning` | todos de T4 | ~5 cada | X-11 (honestidade) |
| 9 | Manter os cortes F&I em `force_min_bins` | `OBN_FastMDLPM_v5.cpp:361-445` | ~20 | X-07 |
| 10 | `CXX_STD = CXX17`; default `missing_values = NULL` | `Makevars`, `R/ob_apply_woe_num.R:113` | 2 | T9(b), E-09 |

### Onda 1 — Infraestrutura comum (≈1 semana)

- Criar `src/common/ob_core.h` (§7.1–7.3).
- Migrar **todos** os algoritmos para `ob::woe_iv` (resolve T3: A-10, B-09, C-14, D-10, E-06, E-08, C-13).
- Remover os headers mortos (`validation_utils.h`, `cutpoints_validator.h`, a tabela χ² de `chi_square_utils.h`).
- Unificar `safe_log`, `join_categories` e a formatação de rótulos num header único (ODR, E-19).

### Onda 2 — Algoritmos exatos (≈1–2 semanas)

- `ob::best_partition` em BB, DP-num, MBLP, MRBLP, OSLP, DMIV-num, JEDI, Sketch-num e SAB (X-08, X-09, X-11, X-12).
- Categóricos (DP, SBLP, MILP, GMB, MBA) com DP em ordem de taxa exata (X-10, X-16, B-07).
- Kerber/Chi2 fiéis (X-04); MDLP via motor F&I (X-06); UDT best-first (X-13); `restore_min_bins` no EWB, IR e LDB (X-14, C-10).
- Alta cardinalidade com heap O(B log B) e `checkUserInterrupt` (T8).

### Onda 3 — Contrato e documentação (≈3 dias)

Para cada algoritmo cujo nome não corresponde à implementação (T6), escolher uma de duas opções:
- **(i)** implementar o método prometido;
- **(ii)** renomear ou documentar honestamente, por exemplo "greedy IV-loss merging" em vez de "Branch and Bound", ou "equal-width + supervised merging" em vez de "K-means".

Também nesta onda:
- remover "Guaranteed", "global optimality" e "optimal partition" onde ainda não forem verdade;
- registrar no `NEWS.md` todas as mudanças de resultado (*breaking changes* numéricas).

---

## 9. Protocolo obrigatório de não-regressão (baseline × candidato)

> **Regra registrada a pedido do mantenedor:** **toda** alteração derivada desta auditoria deve ser comparada sistematicamente com a implementação atual, a versão 1.14.1 no commit `007672b`, chamada aqui de *baseline*. O objetivo é detectar discrepâncias e impedir que uma "melhoria" introduza bugs, falhas, regressões numéricas ou fraquezas (overfitting, instabilidade, perda de performance, novas superfícies de UB).
> **Nenhuma correção é aceita sem o relatório diferencial descrito abaixo.** O protocolo vale para cada onda de §8 e para cada PR individual.

### 9.1 Princípios

1. **Baseline congelado.** A referência é imutável: tag `v1.14.1-baseline` no commit `007672b`. Nunca se compara contra o *branch* em evolução, senão uma regressão introduzida na onda N vira a "nova verdade" da onda N+1.
2. **Execução lado a lado e isolada.** Baseline e candidato são instalados em bibliotecas R separadas e executados em processos R distintos (`callr`). Os dois pacotes nunca são carregados na mesma sessão, o que evita conflito de DLL e de namespace.
3. **Mudança declarada antes de medida.** Todo PR traz um *manifesto de diferenças esperadas* (§9.4): quais algoritmos podem mudar, em quais cenários e em que direção. Qualquer diferença **fora** do manifesto é tratada como regressão até prova em contrário.
4. **Não basta aumentar o IV de treino.** Uma DP que maximiza o IV pode sobreajustar. Toda mudança é avaliada também **fora da amostra** (§9.3, item E), para que "melhorar" no treino não signifique piorar a generalização.
5. **Paridade de bit fora do escopo.** O que não foi alvo da mudança deve produzir resultado **idêntico** ao baseline: mesmos cutpoints, contagens, rótulos, e WoE/IV com tolerância de 1e-12.

### 9.2 Corpus de cenários (fixo, versionado, com seeds)

O corpus é gerado por código determinístico e é o mesmo para baseline e candidato. Ele cobre deliberadamente os casos em que esta auditoria achou defeitos, para que as correções sejam verificadas e os casos sadios continuem sadios.

| Família | Cenários | Por que está aqui |
|---|---|---|
| **Sinais canônicos** | logístico monótono (forte/fraco), degrau único, múltiplos degraus, U, ruído puro (alvo independente) | Fidelidade de critério; T1, T4, T5 |
| **Separação** | separador perfeito `y=1[x>c]`, quase perfeito (1–5 rótulos trocados), flag binária com nível sem eventos | X-01, X-02, X-03 |
| **Desbalanceamento** | taxa de evento 50%, 10%, 1%, 0,2% | T3 (suavização), E-03, E-08 |
| **Discretização dos dados** | contínuo, com muitos empates, 2/3/5 valores distintos, constante, 90% em um único valor | edge cases, C-19, E-16 |
| **Escala e valores extremos** | escala 1e-12 e 1e12, outlier 1e5, ±Inf, NA/NaN (0%, 5%, 30%) | E-07, E-15, C-06, A-16 |
| **Tamanho** | n ∈ {30, 50, 51, 200, 5 000, 100 000, 1 000 000} | ramos `n ≤ 50` (Sketch), escalabilidade |
| **Categóricos** | 2, 5, 20, 200, 5 000, 50 000 categorias; categorias puras; raras; literal "NA"; `""`; UTF-8/latin1 | X-15, X-16, T8, B-13, E-18 |
| **Multiclasse** (JEDI-MWoE) | K = 3 ordinal e nominal | D-03, D-08 |
| **Dados reais** | `germancredit` (helper já existente em `tests/testthat`) e, se possível, uma base interna anonimizada | realismo |
| **Combinações de parâmetros** | `min_bins` ∈ {2,3,5}, `max_bins` ∈ {3,5,8,15}, `bin_cutoff` ∈ {0,01; 0,05; 0,1}, `max_n_prebins` ∈ {10,20,50}, monotonia on/off | contratos de restrição |

Cada cenário roda com **R = 20 seeds** quando é aleatório. São os 37 algoritmos × cerca de 60 cenários × combinações relevantes de parâmetros.

### 9.3 O que é comparado e critérios de aceite

Para cada par (algoritmo, cenário), o harness grava as saídas do baseline (B) e do candidato (C) e aplica as cinco camadas abaixo. **Basta uma falha para bloquear o merge.**

**A. Integridade (vale para qualquer mudança, alvo ou não).** O candidato **nunca** pode ser pior que o baseline em:

| Verificação | Critério |
|---|---|
| Crash / erro R / warning novo não declarado | proibido |
| UB / memória | `R CMD check --as-cran` com `-fsanitize=address,undefined` (ASan/UBSan, via `R-devel` com sanitizers ou rhub `gcc-asan`) e `valgrind --tool=memcheck` sem novos erros |
| Linhas perdidas | `sum(count) == n` (sem NA) |
| Invariantes de §10 (suíte de propriedades) | 100% |
| Determinismo | duas execuções do candidato com a mesma seed geram resultados idênticos |
| Invariância à ordem das linhas | permutar linhas não altera cutpoints/bins |
| Consistência treino ↔ apply | `ob_apply_woe_*` sobre os dados de treino reproduz exatamente o WoE do bin atribuído no fit |

**B. Paridade fora do escopo.** Algoritmos e cenários fora do manifesto devem ser **idênticos** ao baseline: cutpoints e contagens exatos, WoE/IV com `|C−B| ≤ 1e-12`. Qualquer diferença bloqueia o merge, mesmo que pareça uma "melhoria" acidental.

**C. Diferenças dentro do escopo: direção correta e magnitude explicada.** Para cada algoritmo-alvo:

| Métrica | Critério de aceite |
|---|---|
| `total_iv` (com o **mesmo** estimador nos dois lados; §9.5) | Para correções de otimização (T1, X-08, X-09, X-12): `IV_C ≥ IV_B − 1e-9` em **todos** os cenários. Uma redução só é aceita se o manifesto a justificar (ex.: o baseline violava `min_bins` ou `bin_cutoff`) |
| Gap ao ótimo (força bruta, cenários com ≤ 12 pré-bins) | `gap_C ≤ gap_B` e, nos alvos do motor exato (§7.2), `gap_C ≤ 1e-9` |
| Restrições (`min_bins`, `max_bins`, `bin_cutoff`, monotonia) | Taxa de violação de C ≤ taxa de B; violações remanescentes **sinalizadas** (`monotonic=FALSE` / `feasible=FALSE` + warning) |
| Flags (`converged`, `monotonic`) | Devem refletir a verdade verificada: 0 discordâncias entre a flag e a verificação independente |
| Nº de bins | A mudança é reportada por cenário; um aumento sistemático (efeito esperado da DP exata) é declarado no manifesto |

**D. Performance.**

| Métrica | Critério |
|---|---|
| Tempo mediano (10 repetições, `bench::mark`) | `t_C ≤ 1,2 · t_B` por cenário; regressões maiores exigem justificativa explícita no manifesto. Em alta cardinalidade (T8), a meta é `t_C ≤ t_B` |
| Memória máxima (`bench::mark(memory=TRUE)` / `peakRAM`) | `mem_C ≤ 1,2 · mem_B` |
| Escalabilidade | Ajuste log-log de tempo × n (ou × nº de categorias): o expoente empírico de C não pode exceder o de B + 0,1 |
| Interrupção | Laços longos respondem a `Ctrl-C` (`checkUserInterrupt`) em menos de 1 s nos cenários de 50 000 categorias e 10⁶ linhas |

**E. Generalização e estabilidade (proteção contra "melhorias" que sobreajustam).** Para cada cenário aleatório: amostra de treino/teste 70/30, com 20 repetições.

| Métrica | Critério |
|---|---|
| IV fora da amostra (bins de treino aplicados ao teste) | `median(IV_test_C) ≥ median(IV_test_B) − δ`, com δ = 0,01 ou 2%; teste pareado de Wilcoxon sobre as 20 réplicas sem piora significativa (α = 0,05) |
| AUC/KS de uma logística univariada sobre o WoE (teste) | Mesma regra de não-inferioridade |
| Razão IV_test/IV_train (shrinkage) | `ratio_C ≥ ratio_B − 0,05`. Uma queda indica sobreajuste introduzido |
| PSI entre a distribuição de bins do treino e do teste | `PSI_C ≤ max(PSI_B, 0,1)` |
| Estabilidade de cutpoints (bootstrap, 50 réplicas) | Dispersão (IQR) de cada cutpoint em C ≤ 1,25 × a de B; número de bins por réplica com variância ≤ a de B + 10% |
| Sensibilidade a perturbação mínima | Jitter de 1 ulp nos valores e troca de 1 rótulo: a mudança em `total_iv` de C não pode exceder a de B por mais de 2× |

**F. Robustez adversarial (fuzzing diferencial).** Um gerador aleatório de entradas extremas (n de 1 a 10⁴, empates, NA, ±Inf, `min_bins > max_bins`, alvo constante, pesos extremos, strings vazias ou longas, encodings mistos) alimenta B e C em paralelo com no mínimo 10 000 casos por algoritmo, compilados com sanitizers. Critérios:
- C não pode falhar onde B não falha;
- onde B falha com erro R controlado, C deve falhar com erro controlado ou funcionar e satisfazer os invariantes;
- qualquer divergência de classe ("B ok, C erro", "C NaN") é reportada.

### 9.4 Manifesto de diferenças esperadas (obrigatório em cada PR)

Arquivo `audit/diffs/<PR>-manifest.yml`, mantido localmente (a pasta `audit/diffs/` é material de trabalho e fica fora do repositório, por decisão do mantenedor):

```yaml
pr: "onda0-01-obc-dp-pure-bins"
findings: [B-01, A-09]                 # IDs from this audit
targets:                               # only these may differ from baseline
  - algorithm: ob_categorical_dp
    scenarios: ["separation/*", "imbalance/*", "categorical/pure_*"]
    expected:
      total_iv: ">= baseline"          # direction of change
      woe_monotone_violations: "== 0"
      n_bins: "may change"
  - algorithm: ob_categorical_fetb
    scenarios: ["categorical/pure_*"]
    expected:
      woe_of_pure_category: "most extreme of all bins"
non_targets: "bit-identical"           # every other algorithm/scenario
performance_budget: { time_ratio: 1.2, memory_ratio: 1.2 }
breaking_change_note: "NEWS.md entry: OBC_DP pure categories no longer get WoE=0"
```

### 9.5 Comparabilidade das métricas

Como o baseline tem pelo menos seis estimadores de WoE/IV (T3), comparar o `total_iv` reportado por B com o reportado por C seria comparar réguas diferentes. Por isso, **toda métrica de qualidade é recalculada pelo harness**, a partir das **contagens por bin** (`count_pos` / `count_neg`, ou da reaplicação dos cutpoints aos dados), com um único estimador de referência: o estimador de §7.1 com α = 0,5 e K = nº de bins, e também o IV cru quando não há bins puros. Assim a comparação mede a **partição**, não a fórmula de suavização. O `total_iv` reportado por cada versão é registrado à parte, só para documentar a mudança de estimador.

### 9.6 Harness de referência (proposta)

Arquivos propostos: `audit/harness/` (excluído do build pelo `.Rbuildignore`, que já ignora `^audit$`).

```r
# audit/harness/run_differential.R
# Differential harness: runs the frozen baseline and the candidate build of
# OptimalBinningWoE in isolated R processes on the same scenario corpus and
# compares partitions, constraints, performance and out-of-sample behaviour.

# ---- 0. One-time setup ------------------------------------------------------
# git worktree add ../obwoe-baseline v1.14.1-baseline
# R CMD INSTALL --library=lib_baseline ../obwoe-baseline
# R CMD INSTALL --library=lib_candidate .

run_side <- function(lib, scenarios, engines) {
  # Executes all engines on all scenarios inside a fresh R process whose
  # library path points only to `lib`, so the two builds never share a session.
  callr::r(function(lib, scenarios, engines) {
    .libPaths(c(lib, .libPaths()))
    suppressPackageStartupMessages(library(OptimalBinningWoE))
    out <- list()
    for (sc in scenarios) for (en in engines) {
      key  <- paste(en, sc$id, sep = "::")
      fn   <- get(en, envir = asNamespace("OptimalBinningWoE"))
      t0   <- proc.time()[["elapsed"]]
      res  <- tryCatch(
        withCallingHandlers(
          do.call(fn, c(list(feature = sc$x, target = sc$y), sc$params)),
          warning = function(w) { attr(key, "warn") <<- conditionMessage(w)
                                  invokeRestart("muffleWarning") }),
        error = function(e) structure(list(error = conditionMessage(e)),
                                      class = "engine_error"))
      out[[key]] <- list(res = res, time = proc.time()[["elapsed"]] - t0,
                         warn = attr(key, "warn"))
    }
    out
  }, args = list(lib = lib, scenarios = scenarios, engines = engines))
}

# Reference IV computed from per-bin counts with ONE estimator (Section 9.5).
ref_iv <- function(pos, neg, alpha = 0.5) {
  K <- length(pos); dp <- (pos + alpha) / (sum(pos) + K * alpha)
  dn <- (neg + alpha) / (sum(neg) + K * alpha)
  sum((dp - dn) * log(dp / dn))
}

compare_pair <- function(b, c, manifest_entry = NULL, tol = 1e-12) {
  # Returns a one-row data.frame with the verdict for one (engine, scenario).
  if (inherits(c$res, "engine_error") && !inherits(b$res, "engine_error"))
    return(data.frame(verdict = "FAIL", reason = "candidate errors, baseline ok"))
  same_partition <- identical(b$res$cutpoints, c$res$cutpoints) &&
                    identical(b$res$count, c$res$count)
  same_woe <- isTRUE(all.equal(b$res$woe, c$res$woe, tolerance = tol))
  if (is.null(manifest_entry))                       # non-target => bit parity
    return(data.frame(verdict = if (same_partition && same_woe) "PASS" else "FAIL",
                      reason  = "non-target parity"))
  ivb <- ref_iv(b$res$count_pos, b$res$count_neg)
  ivc <- ref_iv(c$res$count_pos, c$res$count_neg)
  ok_iv   <- ivc >= ivb - 1e-9 || identical(manifest_entry$total_iv, "may decrease")
  ok_time <- c$time <= 1.2 * b$time + 0.01
  data.frame(verdict = if (ok_iv && ok_time) "PASS" else "FAIL",
             iv_baseline = ivb, iv_candidate = ivc,
             time_ratio = c$time / max(b$time, 1e-6),
             reason = paste(c(if (!ok_iv) "IV regression", if (!ok_time) "slower"),
                            collapse = "; "))
}
# The out-of-sample block (Section 9.3-E), the bootstrap stability block and
# the differential fuzzer follow the same pattern: generate once, run on both
# sides through run_side(), compare with a dedicated compare_* function.
```

**Saída obrigatória:** `audit/diffs/<PR>-report.md`, gerado automaticamente, contendo:
- resumo PASS/FAIL por camada (A–F);
- tabela de todas as diferenças, com algoritmo, cenário, métrica, B, C, Δ e a classificação "esperada (manifesto)" ou "**inesperada**";
- gráficos de IV de treino × teste e de tempo × n;
- lista de warnings novos.

O relatório é anexado ao PR e revisado antes do merge.

### 9.7 Integração no fluxo de trabalho

| Etapa | Gate |
|---|---|
| Antes de começar | Criar a tag `v1.14.1-baseline`; gerar e versionar o corpus (`audit/harness/corpus.R`) e o snapshot do baseline (`audit/harness/baseline_snapshot.rds`, com hash SHA-256 registrado) |
| Cada PR | Manifesto + relatório diferencial + `R CMD check --as-cran` + sanitizers + suíte de propriedades (§10) |
| CI (GitHub Actions) | Job `differential` que reconstrói o baseline da tag, roda as camadas A–D e F a cada push, e as camadas E e de performance completas em execução noturna ou manual |
| Fim de cada onda | Relatório consolidado da onda × baseline (não × onda anterior) e atualização do `NEWS.md` com as mudanças numéricas intencionais |
| Rollback | Toda mudança de resultado sai inicialmente atrás de `legacy = FALSE` (novo comportamento) com `legacy = TRUE` disponível por uma versão, para comparação em produção pelos usuários |

### 9.8 Testes de regressão permanentes

Para cada achado corrigido, o cenário mínimo de falha descrito nos anexos (ex.: `0/100, 0/20, 25/75, 20/20` para B-01; `y = 1[x>0]` para A-06; 10 categorias com χ² = 0 para A-07) vira um teste `testthat` permanente com o **resultado esperado explícito**. Assim, uma regressão futura que reintroduza o defeito é detectada mesmo depois que o baseline 1.14.1 deixar de ser a referência.

---

## 10. Suíte de testes de propriedade proposta

Em vez de testes por exemplo, a equipe recomenda **invariantes** verificados em todos os 37 algoritmos, num `tests/testthat/test-invariants.R` parametrizado:

```r
# Property-based invariants applied to every binning engine.
algos_num <- c("mdlp","fast_mdlp","cm","fetb","dp","lpdb","mblp","mrblp","oslp",
               "bb","ir","kmb","ewb","udt","mob","jedi","dmiv","sketch","ubsd","ldb")

sim_logistic <- function(n = 5000, seed = 1) {
  set.seed(seed); x <- rnorm(n); y <- rbinom(n, 1, plogis(-1 + 1.5 * x))
  list(x = x, y = y)
}

for (a in algos_num) {
  test_that(paste(a, "satisfies the output contract"), {
    d   <- sim_logistic()
    res <- obwoe_engine_call(a, d$x, d$y, min_bins = 3, max_bins = 6)  # helper
    k   <- length(res$count)
    expect_equal(sum(res$count), length(d$x))                 # no row lost
    expect_true(all(res$iv >= -1e-12))                        # IV_j >= 0
    expect_true(all(is.finite(res$woe)))                      # no +-Inf
    expect_true(k >= 3 || isFALSE(res$feasible))              # min_bins honoured
    expect_true(k <= 6)                                       # max_bins honoured
    expect_false(is.unsorted(res$cutpoints, strictly = TRUE)) # strictly increasing
    if (isTRUE(res$monotonic))                                # flag tells the truth
      expect_true(!is.unsorted(res$woe) || !is.unsorted(rev(res$woe)))
  })
}

test_that("pure bins are maximally informative, not neutral (X-01, X-02)", {
  feat <- c(rep("A", 50), rep("B", 200), rep("C", 200), rep("D", 200))
  y    <- c(rep(0, 50), rbinom(200, 1, .05), rbinom(200, 1, .3), rbinom(200, 1, .6))
  r    <- ob_categorical_dp(feat, y, max_bins = 4)
  expect_lt(r$woe[grepl("A", r$bin)], min(r$woe[!grepl("A", r$bin)]))
})

test_that("ChiMerge does not collapse a perfect separator (X-03)", {
  x <- rnorm(2000); y <- as.integer(x > 0)
  expect_gte(length(ob_numerical_cm(x, y, min_bins = 3)$count), 3)
})

test_that("greedy engines reach >= 95% of brute-force IV (X-08)", {
  # 12 pre-bins -> exhaustive search over contiguous partitions in R is cheap
  # compare total_iv of BB/OSLP/MRBLP/DMIV/JEDI against the exhaustive optimum
})

test_that("results are invariant to row order", {
  d <- sim_logistic(); p <- sample(length(d$x))
  for (a in algos_num) expect_equal(obwoe_engine_call(a, d$x, d$y)$cutpoints,
                                    obwoe_engine_call(a, d$x[p], d$y[p])$cutpoints)
})
```

Testes específicos e os cenários numéricos de cada achado estão nos anexos (campo "Risco da correção / testes sugeridos").

---

## 11. O que está correto (validado)

O registro do que **não** precisa ser mexido também é parte da auditoria:

| Item | Evidência |
|---|---|
| Núcleo Fayyad–Irani do `OBN_FastMDLPM` (ganho, limiar $\log_2(N-1)/N + \Delta/N$, $\Delta=\log_2(3^k-2)-[kE-k_1E_1-k_2E_2]$, boundary points) | Coincidiu 100% com uma recursão de referência em 3000 datasets (A) |
| p-valor bicaudal exato de Fisher (num. e cat.) | 20 000 tabelas vs enumeração `long double`: erro máximo em log p = 8,9e-9; iguala `fisher.test` (A) |
| Quantil χ² via `R::qchisq` | Exato (A) |
| PAVA do `OBN_IR` (pilha, O(B), pooling ponderado, comparação exata em inteiros) | Leitura + réplica (C) |
| `OBC_IVB`: DP exata em ordem de taxa | Leitura + réplica (D) |
| `OBN_MOB`: IV 1,5465 vs ótimo monotônico 1,5539 (99,5%) | Força bruta (D) |
| `OBC_SWB::greedy_similarity_merge` ≡ varredura completa | 3000 instâncias, 0 divergências (E) |
| Fronteira `(a, b]` consistente entre os 37 algoritmos, `OB_ApplyWoENum` (`lower_bound`, `include_upper_bound=TRUE`) e `OB_Cutpoints` | Leitura (C, E) |
| RNG via R (`unif_rand` + `RNGScope`), reprodutível com `set.seed`; nenhum `rand()` ou `random_device` | grep (C, E) |
| Sinal do WoE consistente: ln(%evento/%não-evento), documentado em `obwoe.R` | Leitura (D) |
| IV_j ≥ 0 em todos os algoritmos que usam as **mesmas** distribuições no WoE e no IV | Leitura (D) |
| OpenMP só em `OB_Correlation.cpp`, região sem race condition | Leitura (E) |
| Nenhum UB em MOB-num, DMIV-num, KLL, DP-Sketch, SWB e KDE | ASan/UBSan, ~8000 execuções de fuzz (D, E) |

---

## 12. Decisões que cabem ao mantenedor

1. **Mudanças de resultado.** Quase toda correção de fidelidade muda cutpoints e WoE para os usuários atuais. Opções:
   - **(a)** corrigir e anunciar como *breaking* numérico no `NEWS.md` (recomendado pelo Juiz, porque os resultados atuais estão estatisticamente errados em casos documentados);
   - **(b)** manter o comportamento antigo atrás de `legacy = TRUE` por uma versão.
2. **Renomear ou implementar.** Para os 15 algoritmos com nome ≠ método: implementar (ex.: KMB com Lloyd 1D ou Ckmeans, LPDB com polinômio local) ou renomear e documentar? Uma alternativa pragmática é consolidar vários "LP" no motor exato §7.2 e manter os nomes antigos como aliases deprecados.
3. **Parcimônia vs IV máximo.** Com a DP exata, a solução tende a usar `max_bins`. O pacote quer um termo de penalidade por bin (ou "ganho marginal < τ")?
4. **Suavização padrão.** Jeffreys (α = 0,5, K-normalizada) para todos? Isso altera o IV de bins pequenos e deve ser documentado ("o IV total depende da suavização").
5. **Monotonia multiclasse (JEDI-MWoE).** Desligar por padrão para K ≥ 3?
6. **Categorias não vistas no apply.** Política `unseen = c("na", "zero", "other")`?
7. **Convenção de sinal.** O pacote usa ln(evento/não-evento), o oposto de Siddiqi. É consistente e documentado, mas convém um aviso explícito na vinheta de scorecard.

---

## 13. Limitações desta auditoria

- **Sem R no container:** não foi possível rodar a suíte `testthat` nem chamar as funções exportadas. As evidências vêm de **réplicas C++ fiéis** do código (trechos extraídos literalmente, com stub de `Rcpp.h`). As magnitudes de perda de IV dependem dos geradores sintéticos usados e devem ser lidas como ordem de grandeza.
- **Achados "Suspeito"** (por exemplo B-08, C-12, C-19, D-15.1, E-20(e)) precisam de confirmação em R antes de qualquer correção.
- **Fora do escopo:** `OB_LogisticRegression.cpp`, `OB_Correlation.cpp` (exceto o OpenMP), `OB_DataPrep.cpp`, `OB_CheckDistinctsLength.cpp` e a camada R (`obwoe*`, `step_obwoe`) além do necessário para entender os contratos.
- **Os snippets de correção dos anexos são propostas.** Só o motor de §7.2 foi validado formalmente (contra força bruta). Os demais devem passar pela suíte de §10 e pelo protocolo diferencial de §9 antes do merge.

---

## 14. Referências

- Barlow, R.E., Bartholomew, D.J., Bremner, J.M. & Brunk, H.D. (1972). *Statistical Inference under Order Restrictions*. Wiley.
- Best, M.J. & Chakravarti, N. (1990). Active set algorithms for isotonic regression. *Math. Programming* 47.
- Breiman, L., Friedman, J., Olshen, R. & Stone, C. (1984). *Classification and Regression Trees*. Wadsworth (Teo. 4.5).
- Cormode, G. & Muthukrishnan, S. (2005). An improved data stream summary: the count-min sketch. *J. Algorithms* 55.
- Cover, T.M. & Thomas, J.A. (2006). *Elements of Information Theory*, 2nd ed. Wiley (desigualdade log-soma, Teo. 2.7.1).
- Dougherty, J., Kohavi, R. & Sahami, M. (1995). Supervised and unsupervised discretization of continuous features. *ICML*.
- Fayyad, U. & Irani, K. (1992). On the handling of continuous-valued attributes in decision tree generation. *Machine Learning* 8.
- Fayyad, U. & Irani, K. (1993). Multi-interval discretization of continuous-valued attributes. *IJCAI*.
- Fisher, W.D. (1958). On grouping for maximum homogeneity. *JASA* 53.
- Karnin, Z., Lang, K. & Liberty, E. (2016). Optimal quantile approximation in streams. *FOCS*.
- Kerber, R. (1992). ChiMerge: discretization of numeric attributes. *AAAI*.
- Kirkpatrick, S., Gelatt, C.D. & Vecchi, M.P. (1983). Optimization by simulated annealing. *Science* 220.
- Liu, H. & Setiono, R. (1995). Chi2: feature selection and discretization of numeric attributes. *IEEE ICTAI*.
- Mironchyk, P. & Tchistiakov, V. (2017). Monotone optimal binning algorithm for credit risk modeling. Working paper.
- Navas-Palencia, G. (2020). Optimal binning: mathematical programming formulation. arXiv:2001.08025.
- Siddiqi, N. (2006). *Credit Risk Scorecards*. Wiley.
- Silverman, B.W. (1986). *Density Estimation for Statistics and Data Analysis*. Chapman & Hall.
- Wang, H. & Song, M. (2011). Ckmeans.1d.dp: optimal k-means clustering in one dimension by dynamic programming. *R Journal* 3(2).
- Zeng, G. (2014). A necessary condition for a good binning algorithm in credit scoring. *Applied Mathematical Sciences* 8.

---

## 15. Execução das correções: status e evidência

Esta seção registra o que foi implementado após a auditoria. Toda mudança passou pelo protocolo da §9, sempre comparada com o baseline congelado `v1.14.1-baseline` (commit `007672b`). Os manifestos e relatórios de cada mudança são gerados em `audit/diffs/`, que é material de trabalho local e não é versionado.

### 15.1 Gate de integração final

Relatórios `audit/diffs/90-*` e `91-integration-full-*`. Veredito: **PASS** em todas as camadas.

| Camada | Resultado |
|---|---|
| A–C (integridade, paridade, escopo) | 32 100 execuções por lado (20 seeds); **16 320 / 16 320** execuções fora do escopo **bit-idênticas**; 0 falhas não declaradas |
| D (performance) | 70 / 70 dentro do orçamento (t_C ≤ 1,2·t_B + 0,05 s); nenhuma piora de expoente de escala |
| E (generalização) | log-loss fora da amostra não-inferior em todas as famílias, exceto os desvios declarados no manifesto 90 (todos ligados a cumprir contrato ou ao B-01) |
| F (fuzzing ASan/UBSan) | 7 000 entradas adversariais: 0 erros novos, 0 violações novas; **19 violações pré-existentes de IV < 0 eliminadas** |
| testthat | 630 testes; 0 falhas novas; 10 falhas pré-existentes, todas de ambiente (pacotes opcionais) |

### 15.2 Emenda ao protocolo (camada E)

Por indicação do G1, o critério vinculante da camada E passou a ser a **log-loss fora da amostra** do modelo de taxa por bin (com prior de Jeffreys), que é uma regra de pontuação própria. A margem é de 0,5% relativo + 1e-4. O IV fora da amostra continua reportado, mas não reprova: ele só pode cair quando bins são fundidos, então penalizaria por construção qualquer fusão por significância ou por MDL.

Outras duas mudanças no harness:
- quando mais de uma declaração de manifesto casa com uma execução, vence a mais específica;
- a integração roda com 20 seeds, porque com 3 seeds as medianas da camada E são ruidosas.

### 15.3 Corrigido e integrado

| Achado | Algoritmo | Correção |
|---|---|---|
| B-01 (Crítico), A-09 | OBC_DP, OBC_FETB | Bins puros recebem o WoE finito mais extremo (Jeffreys normalizado); o DP exato roda uma vez por K sobre o IV reportado |
| X-03 / A-06 | OBN_CM | `merge_zero_bins` respeita `min_bins` (antes, um separador perfeito virava 1 bin com IV = 0) |
| X-05 / A-07 | OBC_CM | Removida a parada indevida por \|Δ min χ²\| < tol |
| E-04 | OBC_GMB | Redução de pré-bins por fusões adjacentes em ordem de taxa |
| D-05 | OBC_MOB | Categorias raras tratadas também quando ncat ≤ max_bins |
| D-06 | OBC_DMIV | `bin_cutoff` passa a ter efeito; `max_n_prebins` é respeitado |
| B-02 | OBC_SBLP | O DP otimiza o IV reportado; ordenação exata; removido o modo "chunked" |
| D-03 | OBN_JEDIMWoE | Monotonicidade exigida só com K = 2; fusão pela perda de IV somada entre classes |
| C-06 | OBN_EWB | `min_bins` restaurado quando um outlier esvazia intervalos |
| X-11 | 9 algoritmos numéricos | Campo `monotonic` verificado e aviso quando a monotonia não é atingida |
| E-06 / C-14 / B-09 | DP, KMB, LDB, LPDB (num.) | O IV usa as mesmas distribuições do WoE, então IV ≥ 0 sempre |
| T2 | `obwoe_gains_score` | Bins puros deixam de ter WoE = IV = 0 na seleção de variáveis |
| E-20b | Makevars | `CXX_STD = CXX17` |

### 15.4 Regularização por evidência mínima (readmissões)

As correções retidas na primeira integração foram reavaliadas com duas ferramentas comuns de `ob_core.h`: o IV com shrinkage empírico-bayesiano (`eb_woe_iv`) e a escolha de K pelo BIC (`bic_score`). Cada uma foi calibrada por algoritmo, com o gate pareado de 20 seeds, e entrou a variante mais simples que deixa o algoritmo não-inferior em todas as famílias. A camada E foi corrigida para o teste pareado (Wilcoxon por seed) que a §9.3 especifica: a primeira versão comparava medianas sem pareamento.

| Algoritmo | Correção readmitida | Regra de evidência escolhida | Saldo (log-loss fora da amostra, média das células) |
|---|---|---|---|
| OBN_BB | X-08 / C-05 | nenhuma necessária (passou no teste pareado) | −11,6% (31 células melhores, 0 piores) |
| OBN_JEDI | X-08 / D-02 | IV empírico-bayesiano nas decisões (m = 50) | −11,2% (33 / 0) |
| OBN_MOB | D-12 | IV empírico-bayesiano (m = 50) | +0,1% (3 / 0) |
| OBN_Sketch | X-09 / E-01 | DP exato por K + **K pelo BIC** | −4,8% (30 / 1, declarada: sinal em U com monotonia exigida) |
| OBN_UDT | X-13 / C-01 | árvore best-first + K pelo BIC + folha mínima `max(ceil(bin_cutoff·n), min(7, n/min_bins))` | −2,0% (18 / 0) |
| OBN_FastMDLPM | X-07 / A-04 | o corte MDL só fica se pagar o próprio custo de código, log2(N−1) bits | −2,8% (2 / 0) |
| OBN_KMB, OBN_EWB | C-08 | nenhuma necessária | −2,5% / −6,9% |
| OBC_MBA, OBC_Sketch | E-03, E-02 | ordem contígua pela taxa a posteriori (pos + 20·r₀)/(n + 20) + perda de IV EB | −0,9% / −0,9% |
| OBC_SAB | X-12 / C-02..C-04 | DP exato em ordem de taxa encolhida, objetivo EB, pré-bins | −0,75% (8 / 0) |

**Aprendizados** (registrados pelos grupos R1–R3):
- **Nos numéricos gulosos, o problema não era o número de bins**, e escolher K pelo BIC piorou as amostras pequenas. O problema era onde os cortes caem: o IV cru premia isolar poucas linhas com taxa extrema, e o shrinkage ataca isso diretamente.
- **Nos algoritmos de partição (Sketch, UDT), escolher K pelo BIC resolve.**
- **Nos categóricos de alta cardinalidade, a causa era a "maldição do vencedor"** ao ordenar categorias pela taxa crua. Num caso, um bin de 57 categorias teve taxa de 0,3% no treino e 10,4% no teste. A ordem pela média a posteriori resolve, e é a forma bayesiana do argumento de contiguidade de Fisher e Breiman.
- **A regra oculta sobre positivos do MBA funcionava, sem querer, como regularizador.** Agora foi substituída por uma regularização documentada.

### 15.5 Gate final (estado integrado, `audit/diffs/93-final-*`)

| Camada | Resultado |
|---|---|
| A–C | 32 100 execuções por lado (20 seeds); **9 900 / 9 900** execuções fora do escopo bit-idênticas; 0 falhas não declaradas |
| D | 70 / 70 dentro do orçamento de performance |
| E | 1 035 células família × parâmetros × algoritmo: **174 significativamente melhores, 8 piores**. As 8 estão todas declaradas: cumprimento de `min_bins`/`bin_cutoff` (CM, MOB-cat, DMIV-cat), o B-01 (DP categórico) e o sinal em U com monotonia exigida (Sketch) |
| F | 7 000 entradas adversariais sob ASan/UBSan: 0 regressões; 19 violações pré-existentes (IV < 0) eliminadas |
| testthat | 651 testes; 0 falhas novas (as 10 falhas restantes são pré-existentes e de ambiente) |

### 15.6 Readmissões finais: OSLP, MRBLP, DMIV

| Algoritmo | Correção | Regra de evidência | Saldo (log-loss fora da amostra) |
|---|---|---|---|
| OBN_OSLP | B-03 / X-08 | Funde bins puros adjacentes da mesma classe; um bin puro só fica isolado se o teste binomial exato de zero ocorrências rejeitar o acaso a 1%, ou seja, (1 − r₀)ⁿ ≤ 0,01 | −10,7% (0 células piores) |
| OBN_MRBLP | B-04 / X-08 | Mesma regra | −3,5% (0 piores) |
| OBN_DMIV | D-01 / X-08 | Significância do bin puro a 5%; em L1, L2 e L∞, as fusões do mesmo lado de p = q vão primeiro, pelo menor χ² de Pearson (a distância não consegue escolher entre elas) | −7,0% (default `l2`); de −7% a −11% nos nove métodos (0 piores) |

**Ressalva metodológica.**
- O α = 0,01 do OSLP e do MRBLP foi escolhido entre cerca de 15 variantes avaliadas no mesmo corpus do gate, então há risco de ajuste ao corpus.
- Três pontos atenuam esse risco:
  - 0,01 estava entre os níveis definidos antes do teste (0,05, 0,01 e 0,10);
  - os níveis 0,005, 0,0025 e 0,001 também passam, então o resultado não depende de um valor exato de α;
  - a regra tem fundamento independente do corpus: o teste exato de zero ocorrências (Clopper & Pearson, 1934).
- Uma validação externa, em dados reais do mantenedor, é recomendada.
- O harness só exercita o DMIV com o método default (`l2`). Os outros oito métodos foram verificados em rodadas à parte, feitas pelo agente H2.

**Gate final** (estado integrado; relatórios mantidos localmente, fora do repositório):

| Camada | Resultado |
|---|---|
| A–C | 6 300 / 6 300 execuções fora do escopo bit-idênticas (20 seeds) |
| D | 70 / 70 dentro do orçamento |
| E | 1 206 células: **248 significativamente melhores, 8 piores**, as mesmas 8 declaradas antes |
| F | 0 regressões sob ASan/UBSan; 19 violações pré-existentes de IV < 0 eliminadas |
| testthat | 658 testes; 0 falhas novas |

### 15.7 Ainda retido (`audit/pending/`)

| Correção | Motivo |
|---|---|
| OBN_CM (X-04, Kerber/Chi2) | +1% a +3% de log-loss em amostras pequenas e eventos raros |
| OBN_MDLP (X-06, Fayyad–Irani) | Fragmenta menos que a versão atual em sinais suaves |

Decisões de API ainda pendentes do mantenedor: E-09 (`missing_values = -999`), força dos priors como parâmetro, e `max_bins` como teto no Sketch e no UDT.

---

# Anexos — relatórios integrais dos agentes

Os anexos abaixo são os relatórios originais, sem edição de conteúdo. Eles trazem para cada achado: teoria, trecho do código, cenário numérico de falha, snippet de correção e teste sugerido.
Referências a `work_X/*.cpp` apontam para os experimentos temporários usados durante a auditoria; eles não fazem parte do repositório.


---

## Anexo A — Agente A — Estatístico (inferência, teoria da informação)


Escopo: `OBN_MDLP_v5.cpp`, `OBN_FastMDLPM_v5.cpp`, `OBN_CM_v5.cpp`, `OBC_CM_v5.cpp`, `OBN_FETB_v5.cpp`, `OBC_FETB_v5.cpp`, `common/entropy_utils.h`, `common/chi_square_utils.h` e os wrappers R correspondentes.
Experimentos: todos em `scratchpad/work_A/` (extração literal do código do repo com um stub de `Rcpp.h`; nada do repo foi editado).

#### Resumo executivo

- **Verificado como correto (e testado):** (i) `OBN_FastMDLPM` implementa Fayyad & Irani (1993) fielmente — fórmula do ganho, limiar `log2(N-1)/N + Δ/N` com `Δ = log2(3^k−2) − [kE − k1E1 − k2E2]`, boundary points (F&I 1992), tudo em log2. Em 3000 datasets aleatórios (com e sem empates) o conjunto de cortes coincide 100% com uma recursão ingênua de referência. (ii) O p-valor bicaudal de Fisher (`OBN_FETB` e `OBC_FETB`) é correto: 20 000 tabelas aleatórias comparadas a enumeração exata em `long double`/`lgammal`: erro máximo em log p = 8.9e-9; tea-tasting (3,1,1,3) → 0.4857; (10,0,0,10) → 1.0825e-05 (iguais a `fisher.test`). (iii) O quantil χ² do ChiMerge usa `R::qchisq` (exato; a tabela de `chi_square_utils.h` está morta).
- **Problema dominante:** vários algoritmos **não executam a teoria que anunciam**. O "MDLP" de `OBN_MDLP` não é Fayyad-Irani (critério `log2(k−1)+ΣnH` praticamente inerte: 41–43 de 50 pré-bins sobrevivem, inclusive em ruído puro; o nº de bins é decidido por `max_bins`/monotonicidade). O limiar de significância do ChiMerge/Chi2 numérico nunca decide nada (documentado); o "Chi2" categórico só executa o 1º nível α=0.5; a regra de parada do ChiMerge categórico dispara indevidamente quando dois χ²=0 consecutivos.
- **Robustez:** `OBN_CM` colapsa para 1 bin (ou < `min_bins`) em variáveis (quase) perfeitamente separáveis e em flags binárias com um nível sem eventos (IV=0 em variável fortíssima). `OBN_FastMDLPM::force_min_bins` **descarta** o corte MDLP legítimo quando `min_bins` > nº de cortes aceitos. `OBC_FETB` atribui WoE=IV=0 a categorias puras (sem suavização), corrompendo a ordenação por WoE.
- **Numérica:** Fisher e χ² estão estáveis (int overflow já corrigido). Suavização de Laplace é inconsistente entre algoritmos (`total+1.0` em CM vs `total+0.5k` em FETB/MDLP), gerando viés de WoE material com poucos eventos.
- Nenhum achado Crítico (nada produz crash/UB em uso normal); 7 Altos, 4 Médios, 6 Baixos.

#### Tabela de conformidade por algoritmo

| Algoritmo | Arquivo | Referência teórica | Fidelidade | Robustez | Performance (atual → ideal) | Estab. numérica | Nº achados (C/A/M/B) |
|---|---|---|---|---|---|---|---|
| MDLP numérico | `OBN_MDLP_v5.cpp` | Fayyad & Irani 1993 (declarada) | **Baixa** (heurística MDL própria, não F&I) | Média (NaN conta em `feature.size()`, prebins > P) | O(k²·I) merge ingênuo → O(N log N) com F&I recursivo sobre blocos | Boa (LUT/log2 ok) | 0/2/1/1 |
| FastMDLPM | `OBN_FastMDLPM_v5.cpp` | F&I 1993 + 1992 | **Alta** (núcleo exato) / Parcial (min_bins, monotonia) | Boa, exceto `force_min_bins` | O(m·#cortes) + tabela XLogX O(N) mem → ok | Boa (A(p,n)=t log t−p log p−n log n) | 0/1/0/1 |
| ChiMerge/Chi2 numérico | `OBN_CM_v5.cpp` | Kerber 1992; Liu & Setiono 1995 | **Baixa** (limiar χ² inoperante; Chi2 invertido) | Média/Baixa (colapso por merge_zero_bins) | O(B) por merge (chi_adj) → ok | Boa (double nos produtos) | 0/3/1/2 |
| ChiMerge/Chi2 categórico | `OBC_CM_v5.cpp` | idem | **Parcial** (ChiMerge ok até parada indevida; Chi2 só α=0.5) | Média | Cache `ChiSquareCache` é zerado a cada merge → O(B) por iteração (equivale a sem cache) | Boa (doubles) | 0/2/1/1 |
| FETB numérico | `OBN_FETB_v5.cpp` | Fisher exato (hipergeométrica) | **Alta** (p-valor) / Parcial (p-valor só ordena, sem α; tendência pelo 1º par) | Boa | O(B²) ok; p-valor O(log N + cauda) | **Excelente** (log-espaço, busca binária, tolerância R) | 0/0/2/0 |
| FETB categórico | `OBC_FETB_v5.cpp` | idem | **Alta** (p-valor) / Parcial (WoE de puras = 0) | Média | p-valor O(N) por chamada (soma plena do suporte) → O(log N) como o numérico | Boa (p-valor); **ruim** (WoE puro) | 0/1/1/0 |
| Headers | `entropy_utils.h`, `chi_square_utils.h` | — | n/a | n/a | n/a | n/a | 0/0/0/1 (+transversais) |

---

#### Achados

##### A-01 — `OBN_MDLP` não é MDLP de Fayyad-Irani; o critério MDL é praticamente inerte
- **Algoritmo/arquivo:** `src/OBN_MDLP_v5.cpp:752-805` (`apply_mdl_merging`), `:825-870` (`enforce_max_bins`); wrapper `R/obn_mdlp.R` (seção "Phase 3", "Theoretical Guarantee").
- **Severidade:** Alto
- **Lente:** Teoria
- **Status:** Confirmado (experimento `work_A/mdlp_test.cpp`, `mdlp_test2.cpp`)
- **Teoria:** F&I (1993): corte aceito sse `Gain > [log2(N−1) + Δ]/N`, `Δ = log2(3^k−2) − [kE(S) − k1E(S1) − k2E(S2)]`, aplicado recursivamente (top-down) sobre *boundary points* (F&I 1992). O custo do modelo é `log2(3^k−2)`, não `log2(k−1)`. Não existe "garantia de estimador consistente" em F&I 1993 (a documentação R afirma isso).
- **Implementação:** bottom-up greedy com `MDL(k) = log2(k−1) + Σ n_i·H(S_i)`; `delta = model_delta + data_delta` (l. 765-787). Como `n·H` é côncava, `data_delta ≥ 0` sempre; `model_delta ∈ [−1, 0)` bit. Logo só se aceita um merge se a perda de informação for < 1 bit — i.e., só merges de bins praticamente idênticos. Depois, `enforce_max_bins` (l. 825+) funde à força até `max_bins`.
- **Impacto / cenário de falha:** N=5000, 50 pré-bins: a fase MDL deixa 42/41/43/43 bins (cenários sinal logístico / ruído puro p=0.3 / não-linear / degrau). Ou seja, **a fase "MDLP" quase não decide nada**; o nº final vem de `max_bins` + monotonicidade (ruído puro com `max_bins=2,5,8` → 2 bins em todos, por `min_bins`+monotonia). Variável independente do alvo não é rejeitada como no MDLP real (que devolveria 1 bin). Usuários que escolhem "MDLP" acreditando em F&I recebem outra heurística.
- **Correção proposta:** (a) preferida: reutilizar o motor F&I de `OBN_FastMDLPM` (`Blocks`, `best_mdlp_cut`, `mdlp_cuts`) em `OBN_MDLP` e só então aplicar bin_cutoff/monotonia; (b) mínima: trocar o critério de merge por teste F&I sobre o par adjacente. Snippet de (b):
```cpp
// Aceita o merge (b_i, b_{i+1}) se o corte entre eles NÃO passa no teste MDLP de F&I
static bool fi_cut_accepted(int p1,int n1,int p2,int n2){
  const int N=p1+n1+p2+n2, P=p1+p2, Q=n1+n2; if(P==0||Q==0||N<2) return false;
  auto A=[](int p,int n){ auto xl=[](int c){return c>0? c*std::log2((double)c):0.0;}; return xl(p+n)-xl(p)-xl(n);};
  const double ES=A(P,Q)/N, E1=A(p1,n1)/(p1+n1), E2=A(p2,n2)/(p2+n2);
  const double gain=ES-((p1+n1)*E1+(p2+n2)*E2)/N;
  const int k=2,k1=(p1>0)+(n1>0),k2=(p2>0)+(n2>0);
  const double delta=std::log2(std::pow(3.0,k)-2.0)-(k*ES-k1*E1-k2*E2);
  return gain > (std::log2(N-1.0)+delta)/N;
}
// loop: while (bins.size()>min_bins) { escolher o par com menor (gain - threshold)·N;
//        if (fi_cut_accepted(par)) break; merge; }
```
  Além disso corrigir a documentação (R/obn_mdlp.R) removendo a "Theoretical Guarantee" atribuída a F&I.
- **Risco da correção / testes sugeridos:** muda resultados de todos os usuários de `ob_numerical_mdlp`. Teste: `x <- rnorm(5000); y <- rbinom(5000,1,.3)`; esperar 1 bin (ou `min_bins`) com o critério F&I; `y <- as.integer(x>0.2)` esperar corte em ≈0.2.

##### A-02 — Pré-binning por quantis impede localizar boundary points (teoria F&I 1992)
- **Algoritmo/arquivo:** `src/OBN_MDLP_v5.cpp:688-750` (`create_prebins`).
- **Severidade:** Médio
- **Lente:** Teoria
- **Status:** Confirmado (`mdlp_test2.cpp`)
- **Teoria:** o corte que minimiza a entropia está sempre num boundary point (F&I 1992, Teorema 1); restringir candidatos a quantis perde essa propriedade.
- **Implementação:** candidatos = bordas de 20 pré-bins equifrequentes (ignorando onde o alvo muda).
- **Impacto:** alvo = 1[x>0.2], N=5000 (separação perfeita): saída com cortes 0.199 e 0.258 (3 bins; o bin (0.199;0.258] é misto e contém o degrau real) em vez de 1 corte em 0.2. Perda de poder/IV e cutpoint fora do ponto ótimo.
- **Correção proposta:** pré-binning baseado em boundary points (blocos por valor distinto, como `build_blocks` do FastMDLPM) quando nº de blocos de fronteira ≤ limite; só quantizar se exceder.
- **Risco/testes:** `expect_equal(res$cutpoints[1], 0.2, tolerance=0.01)` no caso de degrau perfeito.

##### A-03 — `records_per_bin = floor(N/P)` gera até 2P−1 pré-bins e último bin de 1 registro
- **Algoritmo/arquivo:** `src/OBN_MDLP_v5.cpp:692`.
- **Severidade:** Baixo
- **Lente:** Teoria / Robustez
- **Status:** Confirmado (`mdlp_test.cpp`: N=39, P=20 → 39 pré-bins; N=1000, P=3 → 4 pré-bins, último com 1 registro)
- **Teoria/Implementação:** `max_n_prebins` é documentado como máximo; o piso viola isso (o mesmo defeito foi corrigido em `OBN_FETB::make_prebins`).
- **Correção:** `end = i + (k+1)*N/P - ...` ou usar o esquema de `OBN_FETB_v5.cpp:178-190` (`floor(k*n/P)`).
- **Teste:** `expect_lte(nrow(res), max_n_prebins)` antes das fases de merge (via parâmetro de depuração) ou checar `count` mínimo.

##### A-04 — `force_min_bins` descarta os cortes MDLP legítimos e os substitui por cortes uniformes
- **Algoritmo/arquivo:** `src/OBN_FastMDLPM_v5.cpp:361-445` (`splits.clear()` na l. 363), chamada na l. 595.
- **Severidade:** Alto
- **Lente:** Teoria / Robustez
- **Status:** Confirmado (`work_A/fm_min.cpp`)
- **Teoria:** F&I não define `min_bins`; qualquer extensão razoável deve *manter* os cortes aceitos e acrescentar os de maior ganho.
- **Implementação:** se `splits.size()+1 < min_bins` → `splits.clear()` e recria cortes igualmente espaçados em valores distintos.
- **Impacto:** dados com degrau verdadeiro em 0.5 (MDLP acha 0.5006), `min_bins=4` → cortes finais 0.1635, 0.3319, 0.6596: **o único corte significativo some**. Em ruído puro e `min_bins=2` (default do R) o corte forçado cai em 0.4920 enquanto o de maior ganho é 0.9769 (cortes arbitrários, IV subótimo). `obwoe()` documenta que `min_bins` costuma ser o fator vinculante, então o caminho é comum.
- **Correção proposta:** manter `splits`; adicionar iterativamente o corte de maior ganho (sem aplicar o teste MDL) em qualquer intervalo corrente:
```cpp
// best_mdlp_cut(..., bool apply_test): quando false, pula o `if (!(gain > threshold)) return false;`
void force_min_bins(const Blocks& b,int min_bins,std::vector<int>& splits){
  const XLogX xl(b.n());
  while ((int)splits.size()+1 < min_bins){
    std::vector<int> bnd{0}; for(int s:splits) bnd.push_back(s+1); bnd.push_back(b.m());
    Cut best{-1.0,0,0,-1}; 
    for(size_t i=0;i+1<bnd.size();++i){ Cut c; if(best_mdlp_cut(b,xl,bnd[i],bnd[i+1],c,/*apply_test=*/false) && c.gain_bits>best.gain_bits) best=c; }
    if(best.cut<0) break;                // sem boundary points: cair no fallback atual (bisseção do maior intervalo)
    splits.push_back(best.cut); std::sort(splits.begin(),splits.end());
  }
}
```
- **Risco/testes:** muda saída quando `min_bins` > nº MDLP. Teste: degrau em 0.5 + `min_bins=3` → `0.5` deve permanecer em `cutpoints`.

##### A-05 — ChiMerge numérico: limiar χ² (α) completamente inoperante
- **Algoritmo/arquivo:** `src/OBN_CM_v5.cpp:1098-1102` (pré-enforce), `:616-650` (`chi_merge`), `R/obn_cm.R:33-45,86-92` (documentado).
- **Severidade:** Alto
- **Lente:** Teoria
- **Status:** Confirmado (`work_A/cm_test.cpp`: ruído puro, `max_bins=15`, α∈{0.5,0.05,0.001} → resultado idêntico, `iterations=0`)
- **Teoria:** Kerber (1992): fundir enquanto `min χ² < χ²_{1−α,df}`; `max_interval` é apenas teto adicional. A significância é o critério de parada principal.
- **Implementação:** `fit()` chama `enforce_bin_limits()` *antes* de `chi_merge()`, deixando `bins ≤ max_bins`; dentro de `chi_merge`, `if (bins.size() <= max_bins) break` precede o teste de significância. O χ² crítico (`R::qchisq(1−α,1)`, l. 965) nunca é comparado.
- **Impacto:** (i) pares estatisticamente indistinguíveis não são fundidos se `bins ≤ max_bins`; (ii) pares significativamente diferentes são fundidos à força se `bins > max_bins` (igual ao categórico, mas o categórico ainda aplica α depois). O nº final de bins vem de `bin_cutoff` + monotonia (ruído puro → 4 bins de 500/600/800/100 obs.). O wrapper R admite o problema, mas o contrato "ChiMerge de Kerber" não é cumprido.
- **Correção proposta:** remover o pré-enforce em `fit()` (l. ~1100) e reescrever o laço:
```cpp
const double crit = chi_square_critical_value(chi_merge_threshold);
while (bins.size() > (size_t)min_bins && iterations_run < max_iterations) {
  auto [chi, idx] = find_min_chi_square_pair();
  const bool forced = bins.size() > (size_t)max_bins;     // teto de max_bins (max_interval de Kerber)
  if (!forced && chi >= crit) { converged = true; break; } // todos os pares significativos
  merge_bins(idx); ++iterations_run;
}
```
  Sem a parada por |ΔIV|<tol (ver A-13).
- **Risco/testes:** altera nº de bins (mais bins quando pares significativos + `bins ≤ max_bins`; menos quando não significativos). Teste: ruído puro com `max_bins=15` deve fundir até `min_bins`; sinal forte com `max_bins=15` deve manter mais bins do que ruído.

##### A-06 — `merge_zero_bins` ignora `min_bins`: separação (quase) perfeita ou flag com nível sem eventos colapsa em 1 bin (IV=0)
- **Algoritmo/arquivo:** `src/OBN_CM_v5.cpp:219-249` (`merge_zero_bins`), chamado em `:605, :661, :1088-1094` e no caminho `unique_values.size() <= 2` (l. 1058-1064).
- **Severidade:** Alto
- **Lente:** Robustez
- **Status:** Confirmado (`work_A/cm_test.cpp`, `cm_test3.cpp`, `cm_test4.cpp`)
- **Teoria:** bins sem eventos produzem WoE→−∞, mas isso é tratado por suavização/cap (as demais engines do pacote usam `+0.5`); fundir bins puros é opcional e **não pode violar `min_bins`**.
- **Implementação:** o laço funde qualquer bin com `count_pos==0 || count_neg==0` com vizinhos até não haver nenhum — sem olhar `min_bins`. Quando tudo é puro, sobra 1 bin.
- **Impacto (medido):** `y = 1[x>0]` (n=2000, `min_bins=3`): **1 bin**, IV=0. Com 1 rótulo trocado: 2 bins (< min_bins=3). Flag binária onde `x=0` nunca tem evento e `x=1` tem 10%: **1 bin, WoE≈0, IV=0** (variável fortemente preditiva descartada). Variável ordinal de 4 níveis com nível 0 sem eventos: nível 0 é fundido ao 1.
- **Correção proposta:** não fundir por pureza quando `bins.size() <= min_bins`; confiar na suavização para o WoE:
```cpp
while (merged && bins.size() > static_cast<size_t>(std::max(min_bins, 1)) && merge_attempts < max_merge_attempts) { ... }
```
  e, no caminho `unique_values.size() <= 2`, **não** chamar `merge_zero_bins()`.
- **Risco/testes:** bins puros passam a existir (WoE grande porém finito via +0.5, clamp ±20). Teste: `x <- c(rep(0,500), rep(1,500)); y <- c(rep(0,500), rbinom(500,1,.1))` → 2 bins, IV>0.

##### A-07 — ChiMerge categórico: parada por |Δ(min χ²)| < tol interrompe a fusão quando dois χ²=0 consecutivos
- **Algoritmo/arquivo:** `src/OBC_CM_v5.cpp:615-621` (dentro de `merge_bins_chimerge`).
- **Severidade:** Alto
- **Lente:** Teoria / Robustez
- **Status:** Confirmado (`work_A/obccm_test.cpp`)
- **Teoria:** Kerber: parar apenas quando todos os pares forem significativamente distintos (min χ² ≥ crítico) ou ao atingir o teto de intervalos. A diferença entre o menor χ² de duas iterações não é critério de convergência.
- **Implementação:** `if (prev_min_chi >= 0 && fabs(min_chi − prev_min_chi) < convergence_threshold) { converged = true; break; }`. Com Yates truncado em zero (`max(0, |O−E|−½)`), muitos pares têm χ²=0 exato → após o 1º merge, o 2º min χ² também é 0 → |0−0|<1e-6 → para.
- **Impacto:** 10 categorias com taxa idêntica (20%) + 2 com taxa 80%, `max_bins=12`, α=0.05: **1 iteração, 11 bins** (as 9 categorias idênticas restantes não são fundidas; o esperado é 3 bins).
- **Correção proposta:** remover o bloco (l. 616-621); a convergência real é `min_chi ≥ crítico` ou `!can_merge_further()`.
- **Risco/testes:** muda nº de bins em categóricas com muitos pares de χ² baixo (para menos, como esperado). Teste: o cenário acima deve retornar 3 bins.

##### A-08 — "Chi2" (numérico e categórico) não é o Chi2 de Liu & Setiono: fases inalcançáveis e regra de inconsistência invertida
- **Algoritmo/arquivo:** `src/OBN_CM_v5.cpp:693-760`; `src/OBC_CM_v5.cpp:458-495` (+ `:477-489` marcado `# nocov`).
- **Severidade:** Alto
- **Lente:** Teoria
- **Status:** Confirmado por leitura + `cm_test.cpp` (Chi2 ≡ ChiMerge, `iterations=0`) e `obccm`-logic (a checagem `bins<=max_bins` encerra após o 1º nível, o código admite em comentário)
- **Teoria:** Chi2 (1995): fase *k* com α_k decrescente; em cada fase aplica ChiMerge com limiar χ²(α_k) até não haver merge; **após cada fase mede a taxa de inconsistência e PARA (revertendo à fase anterior) se ela exceder δ**. Inconsistência *baixa* autoriza mais merging; aqui o sentido está invertido.
- **Implementação:** (numérico) `if (calculate_inconsistency_rate() < 0.05) break;` — para exatamente quando ainda seria permitido fundir mais. (categórico) cada nível chama `merge_bins_chimerge()`, que já funde até `max_bins`; o teste `bins.size() <= max_bins → break` sempre dispara após o 1º nível (α=0.5, crítico 0.455); níveis 0.25…0.001 e a inconsistência nunca rodam. Resultado: "Chi2 categórico" = ChiMerge com α=0.5 (o menos agressivo dos níveis).
- **Correção proposta:** reimplementar:
```cpp
state_ok = bins;                                   // snapshot
for (double a : {0.5,0.1,0.05,0.01,0.005,0.001}) {
  crit = qchisq(1-a,1); while (min_chi < crit && bins.size()>min_bins) merge_min();
  if (inconsistency_rate() > delta) { bins = state_ok; break; }   // reverte e para
  state_ok = bins;
}
enforce_max_bins();   // só depois, como pós-condição
```
- **Risco/testes:** muda resultados de `use_chi2_algorithm=TRUE`. Teste: dados com sinal forte → Chi2 deve parar em mais bins que ruído puro.

##### A-09 — `OBC_FETB`: categorias puras recebem WoE=IV=0 (sem suavização); ordenação por WoE corrompida
- **Algoritmo/arquivo:** `src/OBC_FETB_v5.cpp:210-218` (`computeWoeIv`), usado em `makeBin`, `initialiseBins` (sort por WoE, l. ~297-302), `mergeBins`.
- **Severidade:** Alto
- **Lente:** Numérica / Teoria
- **Status:** Confirmado por leitura (l. 213: `if (pos <= EPSILON || neg <= EPSILON) { bin.woe = bin.iv = 0.0; return; }`)
- **Teoria:** categorias/bins puros têm |WoE| máximo (→±∞, ou grande finito com suavização); atribuir 0 declara "sem informação" justamente ao grupo mais informativo.
- **Implementação:** sem suavização e com "clamp a zero". `OBN_FETB` e demais algoritmos usam `+0.5`.
- **Impacto:** categorias A(0/100 eventos), B(5/100), C(20/100), D(60/100) (85 eventos, 315 não-eventos): WoE real A≪B<C<D, mas A recebe 0 e fica **entre B (−1.63) e C**; a sequência de merges adjacentes (e `enforceLocalMonotonicity`) opera sobre ordem errada, fundindo A com categorias de taxa muito diferente; IV da categoria pura = 0 (subestima IV total). O wrapper R não documenta esse comportamento.
- **Correção proposta:**
```cpp
inline void computeWoeIv(CategoricalBin& bin, size_t nbins_hint) const {
  const double nb = std::max<double>(1.0, nbins_hint);
  const double dp = (bin.count_pos + 0.5) / (total_pos + 0.5*nb);
  const double dn = (bin.count_neg + 0.5) / (total_neg + 0.5*nb);
  bin.woe = std::clamp(std::log(dp/dn), MIN_WOE, MAX_WOE);
  bin.iv  = (dp - dn) * bin.woe;
}
// ordenação inicial/merges: usar WoE suavizado só para ordenar; recomputar IV final com nbins = bins.size()
```
- **Risco/testes:** muda WoE/IV de saída. Teste: categoria com 0 eventos → `woe < 0` e menor do que todas as demais.

##### A-10 — Suavização de Laplace inconsistente entre algoritmos (viés de WoE/IV com poucos eventos)
- **Algoritmo/arquivo:** `src/OBN_CM_v5.cpp:274-276`, `src/OBC_CM_v5.cpp:844-845` e `:798-801` (IV), vs. `OBN_FETB_v5.cpp:156-160`, `OBN_MDLP_v5.cpp:148-157`, `OBN_FastMDLPM_v5.cpp` (`ALPHA*(n_bnd)` com n_bnd = bins+1).
- **Severidade:** Médio
- **Lente:** Numérica
- **Status:** Confirmado por leitura/cálculo
- **Teoria:** a suavização aditiva coerente é `(n_i+α)/(N+kα)`, de modo que as distribuições somem 1.
- **Implementação:** CM: `(n_i+0.5)/(N+1.0)` — independe do nº de bins `k`; soma das proporções = `(N+0.5k)/(N+1)`. Com 30 eventos e k=12: `36/31` → todos os WoE deslocados em +0.149 e IV viesado. OBC_CM ainda calcula `iv = (dist_pos_não_suavizada − dist_neg_não_suavizada)·woe_suavizado`. FETB/MDLP usam `kα`; FastMDLPM usa `(k+1)α`.
- **Impacto:** o mesmo particionamento gera WoE/IV diferentes por algoritmo; eventos raros (≤50) produzem viés comparável a diferenças reais entre bins.
- **Correção:** centralizar em helper (`woe_iv_utils.h`): `dp=(pos+α)/(Tpos+kα)`, `dn=(neg+α)/(Tneg+kα)`, `iv=(dp−dn)·log(dp/dn)` com os mesmos `dp/dn` na IV.
- **Risco/testes:** muda WoE de saída de CM/OBC_CM. Teste de propriedade: `sum(dp)==1` e igualdade de WoE entre algoritmos para a mesma partição.

##### A-11 — `OBN_FETB::enforce_monotone`: tendência definida pelo primeiro par discordante → super-fusão
- **Algoritmo/arquivo:** `src/OBN_FETB_v5.cpp:290-318`.
- **Severidade:** Médio
- **Lente:** Teoria
- **Status:** Confirmado (`work_A/fetb_mono.cpp`)
- **Teoria:** a restrição monotônica ótima (regressão isotônica / PAVA) usa a tendência global; os demais engines usam inclinação (Welford/OLS).
- **Implementação:** `trend` = sinal do 1º par com |ΔWoE|>eps; qualquer par posterior em sentido oposto é fundido.
- **Impacto:** taxas de evento 20%,5%,10%,30%,40%,50% (tendência crescente): resultado **3 bins** (20%, 21.2%, 50%); com PAVA/slope crescente restariam 4 (11.7%,30%,40%,50%). O 1º par decide a direção → fusões desnecessárias, perda de IV. Não existe parâmetro para desligar a monotonicidade, que é aplicada incondicionalmente (inclusive na pré-etapa, l. 413).
- **Correção:** usar `detect_monotonic_direction(woe_)` (Welford, já em `monotonicity_utils.h`) uma vez por chamada; opcionalmente PAVA.
- **Risco/testes:** `expect_gte(nrow(res), 4)` no cenário acima.

##### A-12 — FETB (num. e cat.): p-valor só ordena pares; não há limiar α → "teste exato" sem decisão inferencial
- **Algoritmo/arquivo:** `src/OBN_FETB_v5.cpp:322-341` (`fisher_merge_loop`), `src/OBC_FETB_v5.cpp:340-368` (`mergeLoop`).
- **Severidade:** Médio
- **Lente:** Teoria
- **Status:** Confirmado
- **Teoria:** o uso do teste de Fisher como critério de fusão (análogo ao ChiMerge) exige parar quando o maior p-valor entre pares adjacentes for < α, i.e., quando todos os pares adjacentes diferem significativamente.
- **Implementação:** funde até `max_bins` sempre (inclusive pares com p≈1e-30) e para exatamente em `max_bins` (nunca funde pares com p≈1 se `bins ≤ max_bins`).
- **Impacto:** o nº de bins é ditado por `max_bins`/monotonia; a "significância" é irrelevante. A documentação ("fusão de bins estatisticamente similares") sugere o contrário, embora descreva o critério de escolha.
- **Correção:** adicionar parâmetro `alpha` (default 0.05) e usar `if (best_lp < log(alpha) && cnt_.size() <= max_bins) break;` mantendo `max_bins` como teto.
- **Risco/testes:** mudança de contrato (novo argumento exportado). Teste: ruído puro com `max_bins=8` deve colapsar até `min_bins`.

##### A-13 — Pontos menores de FastMDLPM e CM numérico (heurísticas/flags)
- **Severidade:** Baixo
- **Lente:** Teoria / Robustez
- **Status:** Confirmado por leitura
- (a) `OBN_FastMDLPM_v5.cpp:472-509`: a monotonicidade funde o **par globalmente mais próximo em WoE** (não o par violador) e exige monotonia *estrita*; ao atingir `min_bins` retorna `true` mesmo com WoE não monotônico (teste `fm_min.cpp` caso C: `woe = −1.07 0.68 −1.07 0.68` devolvido sem aviso). `bin_cutoff` é ignorado (bins de poucos registros nas caudas). Laço `converged` (l. 612-625) compara WoE de vetores de tamanhos diferentes após merges: sem efeito prático.
- (b) `OBN_CM_v5.cpp:660-680`: convergência por `|ΔIV_total| < tol` com `prev_total_iv=0` inicial: uma variável com IV total < 1e-6 para na 1ª iteração; fusões de pares quase idênticos (justamente os de menor χ²) mudam IV < tol e interrompem o laço. `:683-686` força `converged=true` ao atingir `max_iterations` (contraria o contrato em `bin_structures.h`).
- (c) Yates em ChiMerge: Kerber (1992) não usa correção de continuidade e substitui E<0.5 por 0.1; aqui `max(0,|O−E|−0.5)` zera muitos χ² (empates em 0), e `find_min_chi_square_pair` escolhe sempre o primeiro (viés para esquerda). Documentado, mas é desvio do paper.
- **Correção:** (a) violador local + aviso se não monotônico; (b) remover critério por ΔIV e `converged=true` em `max_iterations`; (c) opção `continuity_correction=FALSE` (default Kerber) e desempate por menor soma de contagens.

##### A-14 — Performance: Fisher categórico O(N) por chamada; XLogX O(N) memória; cache χ² zerado a cada merge
- **Algoritmo/arquivo:** `src/OBC_FETB_v5.cpp:160-200` (soma plena sobre `[lo,hi]`), `src/OBN_FastMDLPM_v5.cpp:160-175, 289` (tabela de N+1 doubles), `src/common/chi_square_utils.h:112-116` + `OBC_CM_v5.cpp:926-934` (`chi_cache->resize` em todo merge reatribui `INVALID`).
- **Severidade:** Baixo
- **Lente:** Performance
- **Status:** Confirmado por leitura
- **Impacto:** (i) cada p-valor novo custa O(min(r1,c1)) exp/lgamma; `OBN_FETB` já resolve com busca binária + cauda geométrica (O(log N + cauda)). (ii) N=1e8 → 800 MB só para `c·log2(c)`; basta indexar por contagens distintas ou `std::lgamma`. (iii) `ChiSquareCache::resize` (`assign(n−1, −1)`) apaga todos os valores → cada iteração recalcula todos os pares (O(B)), equivalente a não ter cache; `OBN_CM` usa `chi_adj` incremental correto.
- **Correção:** reutilizar `fisher_two_sided_log_p` de `OBN_FETB` em `OBC_FETB` (mover para header comum); em `merge_adjacent_bins` usar `erase` do vetor de cache em vez de `resize`.

##### A-15 — Código morto e armadilhas nos headers
- **Algoritmo/arquivo:** `src/common/chi_square_utils.h:25-62`, `entropy_utils.h` (`calculate_mdl_cost`, `EntropyCache`, `entropy_classes`).
- **Severidade:** Baixo
- **Lente:** Teoria
- **Status:** Confirmado
- `get_chi_critical_value(0.05)` devolve **0.004** (tabela chaveada por nível de confiança/quantil inferior, rotulada "DF=1"; também `unordered_map<double,…>` com chave `double` exata e fallback "mais próximo"). Não é chamada (CM usa `R::qchisq`), mas é uma armadilha para qualquer novo uso. `calculate_mdl_cost` usa modelo `log2(n−1)` e não é usada. Valor do χ² crítico correto: `R::qchisq(alpha, 1, lower_tail=0, log=0)` (melhor que `1−alpha` para α<1e-12: `OBN_CM_v5.cpp:965-967`).
- **Correção:** remover a tabela e `get_chi_critical_value`, ou substituí-las por `R::qchisq(alpha,df,0,0)`; apagar `calculate_mdl_cost` ou alinhá-la a F&I.

##### A-16 — NaN e contagens: `OBN_MDLP` usa `feature.size()` (inclui NaN) em frequências
- **Algoritmo/arquivo:** `src/OBN_MDLP_v5.cpp:300` (`merge_rare_bins`), `:550` (`count` no caso alvo constante), `:165` (stop se alvo reduzido a 1 classe após remoção de NaN).
- **Severidade:** Baixo
- **Lente:** Robustez
- **Status:** Confirmado por leitura
- **Impacto:** com 30% de NaN, a frequência de cada bin é subestimada em 30% → mais bins marcados "raros" e fundidos; no alvo constante, `count` inclui linhas NaN contra a convenção "NaN excluído" (outros engines). Se todas as linhas de uma classe têm NaN, erro enganoso "All target values are the same".
- **Correção:** `total = sorted_data.size()` (pós-remoção) e contar o caso de alvo constante após filtrar NaN.

##### A-17 — Rótulos com precisão fixa e NA categórico
- **Algoritmo/arquivo:** `OBN_FastMDLPM_v5.cpp:511-526` (`std::to_string` = 6 casas), `OBN_MDLP_v5.cpp:895-905`/`OBN_CM_v5.cpp:1166-1182` (`setprecision(6)` fixo), `OBN_FETB_v5.cpp` (6 dígitos significativos), `OBC_CM_v5.cpp:1085` (`"NA"`) e `OBC_FETB_v5.cpp:538` (`"__NA__"`).
- **Severidade:** Baixo
- **Lente:** Robustez
- **Status:** Confirmado por leitura
- **Impacto:** features em escala <1e-6 geram rótulos `(0.000000;0.000000]` duplicados (cutpoints numéricos permanecem exatos); categoria literal `"NA"` colide com ausentes. Só afeta se alguém usar o rótulo como chave.
- **Correção:** `%.*g` com 15–17 dígitos significativos; usar sentinela impossível ou tratar NA fora do mapa.

---

#### Problemas transversais observados

1. **Padrão "pré-enforce de `max_bins`" apaga o critério estatístico** (CM numérico, Chi2 categórico, FETB): o teste inferencial só ordena candidatos. Convém tratar `max_bins` como teto aplicado *após* o laço estatístico.
2. **Suavização de Laplace não centralizada** (A-10): cada engine reimplementa `+0.5` com denominadores distintos.
3. **`min_bins` não é pós-condição uniforme**: `merge_zero_bins` (CM) e `force_min_bins` (FastMDLPM) o violam/contornam.
4. **Código duplicado entre num./cat.**: `fisher_*` duplicado (versões diferentes, uma O(N)), `safe_log/clamp/safe_divide` redefinidos em cada TU; mover para `common/`.
5. **Precisão de `log-factorial` acumulado** (`lf[i] = lf[i−1] + log i`): erro vs `lgamma` de 3e-9 (N=2e5), 6e-7 (N=2e6) e 3.6e-6 (N=1e7) — maior que a tolerância relativa de 1e-7 do teste de Fisher. Experimento (`tie_test.cpp`) com tabelas simétricas até r=3e6 mostrou diferença de log-prob < 4e-9 (as parcelas simétricas continuam incluídas), logo **sem impacto observado**; recomenda-se `std::lgamma` ou soma compensada para N>1e6.
6. **`R::qchisq(1−α)`** perde precisão relativa para α < ~1e-12; usar `lower_tail=0`.

#### Priorização recomendada (top 5)

1. **A-06** (`OBN_CM`): colapso para 1 bin / < `min_bins` em flags e variáveis separáveis — destrói as variáveis mais preditivas.
2. **A-07** (`OBC_CM`): parada prematura do ChiMerge categórico por `|Δχ²|<tol` (χ²=0 consecutivos).
3. **A-04** (`FastMDLPM`): `force_min_bins` descarta cortes F&I válidos (default `min_bins=2` em ruído e `min_bins≥3` em sinal).
4. **A-01 + A-05 + A-08**: alinhar MDLP numérico a F&I e restaurar o papel do limiar χ²/Chi2 (ou rebatizar/documentar os métodos como heurísticas de `max_bins`).
5. **A-09** (`OBC_FETB`): WoE=0 para categorias puras e ordenação inicial corrompida; **A-10** (suavização) logo em seguida.

---

## Anexo B — Agente B — Otimização / Pesquisa Operacional


Escopo: `OBN_DP`, `OBC_DP`, `OBC_MILP`, `OBN_LPDB`, `OBN_MBLP`, `OBN_MRBLP`, `OBN_OSLP`, `OBC_SBLP` (src/*_v5.cpp) e wrappers R correspondentes. Somente leitura; experimentos standalone em `.../scratchpad/work_B/` (`exp1.cpp`, `exp1b.cpp`, `ex_search.cpp`, `exp2.cpp`, `exp3.cpp`, `exp3b.cpp`). Rcpp não está disponível: os experimentos usam **réplicas fiéis** das funções (copiadas/portadas linha a linha), com contagens inteiras e a mesma suavização; isso é dito em cada achado.

#### Resumo executivo

- **Nenhum** dos oito algoritmos faz LP/MILP/branch-and-bound. Apenas `OBC_DP` e `OBC_SBLP` têm DP de verdade (Bellman sobre categorias ordenadas por taxa de eventos, O(n²k)). Os demais são heurísticas gulosas de fusão. A documentação R/C++ é honesta para OBN_DP, OBC_MILP, MBLP, MRBLP, OSLP, LPDB, mas **superestima** OBC_DP ("global optimality") e OBC_SBLP ("guarantees an optimal partition"), e MBLP ("guaranteed monotonicity").
- **B-01 (Crítico)**: `OBC_DP` usa `compute_woe_iv()` que devolve WoE=IV=0 para bin sem eventos ou sem não-eventos. O DP então maximiza um objetivo errado (bin puro "vale" 0), reporta WoE=0 para o bin mais extremo e quebra a monotonicidade que o código afirma garantir.
- **B-02 (Alto)**: `OBC_SBLP` — o DP otimiza um IV cuja suavização depende do nº de pré-bins do segmento (`e-s`), mas o IV reportado usa o nº final de bins; ordena por taxa suavizada (não a razão de verossimilhança exata); pós-processamento "monotonic_bins"/"chunked" é arbitrário. Réplica: partição do DP ≠ ótimo do IV reportado em 25% (amostras pequenas) / 6,8% (médias) dos casos, gap máx. 80%.
- **B-03/B-04 (Alto)**: critérios de redução para `max_bins` sem fundamento teórico — `OSLP` usa a *soma* dos IV (`findMinIVMerge`) e `MRBLP` a *diferença absoluta* dos IV (`findMinIVDiffMerge`) em vez da **perda de IV** (`IV_i+IV_{i+1}-IV_merged`). Contra o ótimo exaustivo (12 pré-bins → 3..5 bins monotônicos), OSLP-like atinge em média 80% do IV ótimo (90% dos casos < 0,9·ótimo) e MRBLP-like 94% (12,7% dos casos < 0,9·ótimo); o mesmo laço com perda de IV real sobe para 99,9%.
- **B-05 (Alto)**: MBLP/MRBLP/OSLP (e OBN_DP/LPDB em grau menor) param a fusão de violações em `min_bins` e retornam `converged=TRUE` com WoE não monotônico, mesmo existindo partição monotônica viável (provada por busca exaustiva). MBLP: 82,6% de saídas não monotônicas em dados em U; MRBLP/OSLP: 11,9% em dados monótonos ruidosos.
- OBN_DP é guloso apesar do nome (admitido no código); ~45% subótimo vs. exaustivo mas IV médio 99,7% do ótimo em dados monótonos (pior em U: 90%).
- Pontos menores: `ensure_max_prebins` do OBC_DP funde os dois menores por contagem ignorando a taxa; o de OBC_SBLP pode não atingir `max_n_prebins`; prebinning de OBN_DP com divisão inteira desbalanceia bins para n pequeno; parâmetros mortos (`polynomial_degree`, `convergence_threshold`); NA→"NA" colide com categoria real "NA".

#### Tabela de conformidade por algoritmo

| Algoritmo | Arquivo | Referência teórica | Fidelidade | Robustez | Performance (atual → ideal) | Estab. numérica | Nº achados (C/A/M/B) |
|---|---|---|---|---|---|---|---|
| OBN_DP | src/OBN_DP_v5.cpp | Navas-Palencia 2020 (prometido); implementa merge guloso + PAVA-like | Baixa (nome "DP"; honesto no código) | Média: para em min_bins com violação; prebin desbalanceado p/ n pequeno | O(n log n + k²·ger. de rótulos) → DP O(m³k) com monotonia | Boa (Bayes por bin, não normalizado) | 0/1/2/1 |
| OBC_DP | src/OBC_DP_v5.cpp | Bellman 1954; Fisher 1958/Breiman 1984 (ordenar por taxa); Navas-Palencia | Parcial (DP exato do objetivo *errado*) | Boa; pré-bin heurístico | O(n²k) memória O(n²) → ok (pode ser O(nk log n)) | **Ruim** (WoE=IV=0 em bin puro) | 1/0/2/2 |
| OBC_MILP | src/OBC_MILP_v5.cpp | Navas-Palencia 2020 (MILP) | Baixa (guloso; honesto) | Boa | O(B²) → ok | Média (prior por bin não normaliza; \|iv\|) | 0/0/2/1 |
| OBN_LPDB | src/OBN_LPDB_v5.cpp | Local polynomial density (prometido) → KDE gaussiano | Baixa (sem polinômio; honesto) | Boa | O(n+g²) | Média (IV com WoE suavizado e dist. crua) | 0/0/2/2 |
| OBN_MBLP | src/OBN_MBLP_v5.cpp | MILP monotônico (Navas-Palencia) | Baixa | Média: stall em min_bins com `converged=TRUE` | O(k²·it) → ok | Boa | 0/1/1/1 |
| OBN_MRBLP | src/OBN_MRBLP_v5.cpp | Monotonic risk / LR pre-binning (não implementado) | Baixa | Média | O(k²) | Boa (cap ±20 só com laplace=0) | 0/2/1/1 |
| OBN_OSLP | src/OBN_OSLP_v5.cpp | "Optimal supervised learning partitioning" (heurística) | Baixa | Média | O(k log u) por avaliação (bom) | Boa | 0/2/0/1 |
| OBC_SBLP | src/OBC_SBLP_v5.cpp | DP sobre categorias ordenadas (Fisher/Breiman) | Parcial (DP com objetivo inconsistente) | Média | O(n²k) ok | Média (suavização depende de `e-s`) | 0/1/3/1 |

#### Experimentos (resumo)

| ID | Experimento | Resultado |
|---|---|---|
| E1 | `ex_search.cpp`: contra-exemplo mínimo OBC_DP | categorias (pos/neg) ordenadas `0/100, 0/20, 25/75, 20/20`, k=2: DP(código) corta após 3ª cat. (IV_código 0,722); cortar após 2ª isola grupo puro 0/120 (IV suavizado 2,595 vs 0,714; IV verdadeiro → ∞) e é descartado |
| E2 | `exp1b.cpp` (3000 inst., 7 cats, k=3) | DP(código) < ótimo do *mesmo* objetivo sobre todas as partições em 134 casos (contiguidade falha com o hack do zero); partição < ótimo do IV suavizado em 948 casos (gap médio 10,8%, máx 89,8%) |
| E3 | `exp2.cpp` (replica SBLP) | DP ≠ ótimo do IV reportado: 24,9% (n≈10–20/cat, gap médio 2,8%, máx 80%) e 6,8% (n≈60–120/cat, gap máx 4,35%); ordem por taxa suavizada ≠ ordem por taxa crua em 61% / 13,5% das instâncias |
| E4 | `exp3.cpp` (12 pré-bins → [3,5] bins, ótimo exaustivo monotônico, 1500 inst./cenário) | ver tabela abaixo |

Tabela E4 (IV do resultado ÷ IV do ótimo exaustivo; "não-monot." = saída não monotônica apesar de existir solução viável):

| Cenário | Heurística | subótimo | IV<0,9·ótimo | não-monot. | IV/ótimo médio |
|---|---|---|---|---|---|
| monótono | MBLP-like | 23,9% | 0,1% | 0% | 0,998 |
| monótono | MRBLP-like (\|ΔIV\|) | 97,8% | 12,7% | 0% | 0,944 |
| monótono | OSLP-like (ΣIV) | 100% | 90,3% | 0% | 0,799 |
| monótono | MRBLP + perda de IV (correção proposta) | 22,0% | 0% | 0% | 0,999 |
| monótono | OBN_DP-like (\|ΔWoE\|) | 45,1% | 0% | 0% | 0,997 |
| monótono ruidoso | MBLP-like | 95,7% | 30,8% | 2,8% | 0,918 |
| monótono ruidoso | MRBLP-like | 86,7% | 31,5% | 11,9% | 0,929 |
| monótono ruidoso | OSLP-like | 90,7% | 43,6% | 11,9% | 0,903 |
| monótono ruidoso | MRBLP + perda de IV | 52,3% | 26,3% | 11,9% | 0,952 |
| monótono ruidoso | OBN_DP-like | 22,8% | 0% | 0,1% | 0,999 |
| em U | MBLP-like | 37,1% | 14,5% | **82,6%** | 0,932 |
| em U | MRBLP/OSLP-like | 55,1% | 41,6% | 0,3% | 0,833 |
| em U | OBN_DP-like | 42,8% | 29,1% | 0,1% | 0,898 |

Limitações: réplicas (não o código compilado), dados sintéticos, objetivo de referência = IV com Laplace α=0,5 dependente de K (o das famílias MBLP/MRBLP/OSLP); a comparação é justa entre heurísticas mas as magnitudes absolutas dependem do gerador.

#### Achados

##### B-01 — OBC_DP: bin puro recebe WoE=IV=0 (objetivo do DP e saída incorretos; monotonicidade quebrada)
- **Algoritmo/arquivo:** src/OBC_DP_v5.cpp:76-91 (`compute_woe_iv`), usado em :377 (caminho sem otimização), :588 (`seg_iv` do DP), :683 (bins finais); R/obc_dp.R:66-103 (promessa)
- **Severidade:** Crítico
- **Lente:** Teoria / Numérica
- **Status:** Confirmado (E1, E2 e raciocínio algébrico)
- **Teoria:** IV é aditivo por bin: IV = Σ (p_i − q_i)·ln(p_i/q_i), uma f-divergência convexa; para alvo binário a partição ótima é contígua na ordem da razão de verossimilhança (Fisher 1958; Breiman et al. 1984, Thm 4.5; Burshtein et al. 1992). Em bin puro o IV é +∞ (ou, com suavização, grande), nunca 0. WoE de bin puro deve ser o extremo da escala, não 0.
- **Implementação:** `if (count_pos <= EPSILON || count_neg <= EPSILON) { woe = 0.0; iv = 0.0; return; }`
- **Impacto / cenário de falha:** (i) o DP *penaliza* segmentos puros (valem 0), logo prefere diluí-los; no exemplo `0/100, 0/20, 25/75, 20/20` (k=2) descarta o corte que isola o grupo 0/120 (IV suavizado 2,59 vs 0,71 escolhido). (ii) O WoE reportado para o bin sem eventos é 0 ("neutro"), quando é o mais seguro; com bins ordenados por taxa (0, 5%, 30%…), a sequência de WoE fica `0, −1,74, …` — **não monotônica**, contradizendo o comentário em :563-580 ("every contiguous partition has monotone event rates -- and WoE is an increasing function of the event rate") e a doc R. Ex.: cats `0/50, 10/190, 60/140, 80/120` com 4 bins → WoE reportado `0, −1,74, …`. (iii) Categorias sem eventos são rotina em scorecards. (iv) A contiguidade deixa de valer com o hack: 134/3000 instâncias o DP não é nem o ótimo global do próprio objetivo.
- **Correção proposta:** usar o mesmo estimador suavizado das demais famílias, com K fixo por candidato (aditivo dado K): 
  ```cpp
  // WoE/IV suavizado para um bin; Kbins = nº final de bins (constante dentro do DP para dado k)
  inline void compute_woe_iv(double p, double n, double Tp, double Tn, int Kbins, double a, double &woe, double &iv){
    double dp = (p + a) / (Tp + a*Kbins), dn = (n + a) / (Tn + a*Kbins);
    woe = std::log(dp/dn); iv = (dp - dn) * woe; }
  ```
  e no DP rodar uma tabela por k (ou recalcular `seg_iv` por k) usando `Kbins=k`; `backtrack` já escolhe k. Alternativa mínima: manter IV bruto com clamp `std::max(dist, EPSILON)` em vez de zerar (mantém WoE no extremo ±MAX_WOE).
- **Risco / testes:** altera resultados para qualquer feature com categoria pura. Teste testthat: `ob_categorical_dp(c(rep("A",50),rep("B",200),rep("C",200),rep("D",200)), y)` onde A é 100% negativa → `expect_lt(res$woe[res$bin=="A"], min(res$woe[res$bin!="A"]))` e `expect_true(!is.unsorted(res$woe))`.

##### B-02 — OBC_SBLP: DP sobre objetivo diferente do reportado; promessa de otimalidade falsa
- **Algoritmo/arquivo:** src/OBC_SBLP_v5.cpp:493 e :535-552 (`iv_from_counts(..., e - s)`), :278/:313 (chave `(pos+0,5)/(n+1)`), :717 (`converged = max_iterations > 1`), :757-780 (pós-processamento), prepare_output (suavização com `nb`); R/obc_sblp.R:93-95
- **Severidade:** Alto
- **Lente:** Teoria
- **Status:** Confirmado (E3, réplica do DP + força bruta sobre todas as partições)
- **Teoria:** Para o DP de Bellman ser exato, o custo do segmento deve ser independente das outras decisões. Com Laplace por bin, a normalização é `T + α·K` com K = nº *final* de bins (constante para dado k), não o nº de pré-bins membros do segmento. A ordem de varredura correta é a da razão de verossimilhança (taxa crua `pos/n`), não a taxa suavizada `(pos+0,5)/(n+1)` (que reordena categorias pequenas: 61% das instâncias com n≈10–20/cat).
- **Implementação:** `smoothed_total_pos = total_pos_all + alpha * n_members` com `n_members = e - s` (nº de pré-bins do segmento); `prepare_output` usa `nb`. O DP maximiza ΣIV_seg(m=e−s) mas o IV reportado é ΣIV(K=nb).
- **Impacto / cenário de falha:** partição "ótima" do DP não é a ótima do IV reportado em 24,9% (n≈10–20/cat; gap médio 2,8%, máx 80%) e 6,8% (n≈60–120/cat; gap máx 4,35%). Ainda: o DP usa k = `max_bins` fixo (não busca k∈[min,max]); se a saída não for monotônica, o código cai em fusões ad hoc (`monotonic_bins`) e, por fim, em "chunks" de igual número de *categorias* (não de observações) — destrói qualquer otimalidade; `converged = max_iterations > 1` é sem sentido.
- **Correção proposta:** (a) `segment_iv(s,e)` → `iv_from_counts(pos,neg,K)` com K = j (o índice externo do laço de k) — executar o DP uma vez por j∈[min_bins,max_bins] e escolher o melhor IV reportado; (b) ordenar por `count_pos*(n_b) < count_pos_b*n_a` (inteiro exato, como em OBC_DP) com desempate por contagem; (c) remover `chunked` e o ramo `monotonic_bins` (com ordenação exata a monotonicidade é automática); (d) `converged = true`.
  ```cpp
  for (size_t j = 2; j <= k; ++j) { /* mesmo laço, mas */ 
     auto seg = [&](size_t s,size_t e){ return iv_from_counts(pre_pos[e]-pre_pos[s], pre_neg[e]-pre_neg[s], j); };
     ...}
  ```
- **Risco / testes:** muda resultados em categorias esparsas. Teste: comparar `total_iv` com força bruta em R para ≤8 categorias (`expect_gte(res$total_iv, best_brute - 1e-9)`).

##### B-03 — OBN_OSLP: redução para max_bins pela *soma* dos IV
- **Algoritmo/arquivo:** src/OBN_OSLP_v5.cpp:372 chamando `findMinIVMerge()` (:406)
- **Severidade:** Alto
- **Lente:** Teoria
- **Status:** Confirmado (E4, réplica)
- **Teoria:** o par a fundir deve minimizar a **perda** de IV: ΔIV = IV_i + IV_{i+1} − IV(i∪i+1) (critério gulosos clássico do ChiMerge/ MDLP ascendente; Kerber 1992 usa estatística; Navas-Palencia usa IV). A soma IV_i+IV_{i+1} é pequena para bins de WoE≈0 (o centro da escala), mesmo que tenham sinais opostos, e grande para dois bins extremos com WoE idêntico (perda ≈0).
- **Implementação:** `iv_sum = iv_values[i] + iv_values[i+1]; if (iv_sum < min_iv_sum) ...`. Note que `computeIVLoss()` já existe e é usada no merge de bins raros (:~320), mas não aqui.
- **Impacto:** em cenário monótono IV/ótimo médio 0,799; 90% dos casos < 0,9·ótimo; 100% subótimo. É o caminho padrão (20 pré-bins → 5).
- **Correção proposta:**
  ```cpp
  size_t findMinIVMerge() const {
    double best = std::numeric_limits<double>::max(); size_t idx = 0;
    for (size_t i = 0; i + 1 < iv_values.size(); ++i) {
      const double loss = computeIVLoss(i, i + 1);
      if (loss < best) { best = loss; idx = i; }
    }
    return idx; }
  ```
  (computeIVLoss recalcula `accumulate` por chamada: pré-calcular totais fora do laço → O(k)).
- **Risco / testes:** `expect_gte(oslp$total_iv, 0.95 * brute_force_iv)` em dados simulados monótonos; a variante com perda de IV recuperou 0,999 do ótimo na réplica (MRBLP+perda).

##### B-04 — OBN_MRBLP: redução para max_bins por |IV_i − IV_{i+1}|
- **Algoritmo/arquivo:** src/OBN_MRBLP_v5.cpp:496 e `findMinIVDiffMerge` (:509)
- **Severidade:** Alto
- **Lente:** Teoria
- **Status:** Confirmado (E4)
- **Teoria:** idem B-03; |ΔIV| é zero para bins com WoE de sinais opostos e IV iguais (justamente os dois lados do zero), os mais distintos.
- **Impacto:** IV/ótimo médio 0,944 (monótono), 0,833 (em U); 12,7% < 0,9·ótimo (monótono) e 41,6% (U).
- **Correção:** mesmo snippet de B-03 (perda de IV com K−1 na suavização do bin fundido).
- **Risco / testes:** idem B-03.

##### B-05 — Monotonicidade "garantida" não é garantida; `converged=TRUE` mente (MBLP, MRBLP, OSLP; OBN_DP/LPDB em menor grau)
- **Algoritmo/arquivo:** src/OBN_MBLP_v5.cpp:~560-600 (`enforce_monotonicity` retorna false em `size<=min_bins`; laço sai em :590 com `converged = true` :592); OBN_MRBLP_v5.cpp:`mergeViolations` + :506; OBN_OSLP_v5.cpp:`mergeViolations` + :383 (`pending` exige `k > min_bins`); OBN_DP_v5.cpp:590/612; OBN_LPDB_v5.cpp `merge_monotonicity_violations`; R/obn_mblp.R:5 ("guaranteed monotonicity")
- **Severidade:** Alto
- **Lente:** Teoria / Robustez
- **Status:** Confirmado (E4: instâncias com ótimo monotônico viável provado por busca exaustiva)
- **Teoria:** com a restrição de monotonicidade e min_bins, o conjunto viável é não vazio (aqui provado) e um algoritmo que se diz "monotônico" deve achá-lo (PAVA/DP). Fusão "primeira violação" para em min_bins sem retroceder.
- **Impacto:** exemplo da réplica MRBLP: pré-bins `50/585 34/474 22/388 221/403 130/590 28/183 74/293 223/559 137/223 248/435 123/146 115/215` → saída 3 bins `50/585 | 34/474 | 1321/3435` com WoE `−1,289, −1,459, +0,207` (descendo no começo, direção "ascendente"); existe solução monotônica com 5 bins. MBLP em dados em U: 82,6% saídas não monotônicas, `converged=TRUE`, `monotonicity="increasing"`. O usuário que usa WoE em regressão logística recebe uma variável não monotônica rotulada como monotônica.
- **Correção proposta:** (a) curto prazo: se ao final `!isMonotonic`, `converged=false` + `Rcpp::warning`; (b) longo prazo: PAVA sobre a taxa crua (fusão do primeiro par violador usando taxa *agrupada*, K-independente, termina monotônico por construção para qualquer K≥1) e depois redução de bins escolhendo a fusão de menor perda *entre as que mantêm a monotonia* (para pares adjacentes com taxa agrupada, a monotonia é preservada para qualquer fusão — a fusão de vizinhos em sequência monotônica de taxas crus nunca cria violação); (c) ou DP exato O(m³k) (ver "Priorização").
  ```cpp
  // MRBLP/OSLP: ao final de enforceMonotonicity()
  if (!isMonotonic(increasing)) { converged = false; Rcpp::warning("monotonicity not reached within min_bins"); }
  ```
  Nota: a suavização K-dependente faz o WoE de um bin fundido não ficar entre os pais; monotonicidade deve ser testada na taxa crua pos/n (equivalente à razão de verossimilhança), que é aditiva sob fusão.
- **Risco / testes:** `expect_true(!is.unsorted(res$woe) || !is.unsorted(rev(res$woe)))` para `monotonic` em dados em U simulados (min_bins=3).

##### B-06 — OBN_DP: "DP" é guloso; sem garantia de otimalidade; promessa de Navas-Palencia 2020 não cumprida
- **Algoritmo/arquivo:** src/OBN_DP_v5.cpp:23-40 (comentário assume ser guloso), `ensure_bin_constraints`/`find_smallest_woe_diff` (~:660, critério |ΔWoE| ignora contagens), `enforce_monotonicity` (:590). R/obn_dp.R:1-7 diz "resembles Dynamic Programming".
- **Severidade:** Médio (honestamente rotulado)
- **Lente:** Teoria
- **Status:** Confirmado
- **Teoria:** a formulação exata (Navas-Palencia 2020) é MIP; com pré-bins ordenados e IV aditivo (K fixo), é resolvida por DP O(m²k) sem monotonia e O(m³k) com monotonia (estado = (fim, início do último segmento, k)).
- **Implementação:** pré-bins quantílicos → fusão da primeira violação → |ΔWoE| mínimo até max_bins → fusão de raros (que para em min_bins deixando bins < `bin_cutoff`) → reaplica monotonia uma vez. Direção "auto" vem da correlação de Pearson feature×target (sensível a outliers; força colapso para min_bins em relações em U).
- **Impacto:** réplica E4: subótimo em 45% (monótono) / 23% (ruidoso) / 43% (U); IV médio 99,7–99,9% do ótimo nos dois primeiros e 89,8% em U. Moderado, não grave. Também: o modo `auto` reporta "ascending" quando `woe_values.size()<=2` mesmo com WoE decrescente (rótulo, não efeito).
- **Correção proposta:** substituir pelo DP monotônico sobre pré-bins:
  ```cpp
  // dp[k][j][i]: melhor IV com k bins, último segmento (j,i], taxa(j,i) >= taxa(prev_start,j) (ascendente)
  // transição O(m) -> O(m^3 k); m<=max_n_prebins (≈20-50) é trivial.
  ```
  K fixo por rodada (suavização). Manter o guloso como fallback para m>200.
- **Risco / testes:** `expect_gte(dp$total_iv, greedy_iv)` sobre simulações.

##### B-07 — OBC_DP: pré-binagem heurística invalida "global optimality"
- **Algoritmo/arquivo:** src/OBC_DP_v5.cpp:407-465 (`merge_rare_categories`), :475-500 (`ensure_max_prebins`), :402 (`ncat<=max_bins` ignora min_bins/cutoff); R/obc_dp.R:66, :103
- **Severidade:** Médio
- **Lente:** Teoria
- **Status:** Confirmado (leitura)
- **Teoria:** a otimalidade do DP vale sobre o conjunto de pré-bins dado; juntar raros por ordem de contagem (independente da taxa) e depois fundir "os dois menores" até `max_n_prebins` descarta partições candidatas sem critério de IV.
- **Implementação:** raros acumulados em grupo por ordem crescente de contagem (mistura taxas muito distintas num mesmo pré-bin); `ensure_max_prebins` funde `merged_bins[0]` e `[1]` (menores por contagem) ignorando taxa — com `bin_cutoff` pequeno (ex. 0,001) e 500 categorias, as duas categorias menores (0% e 100%) viram um pré-bin.
- **Correção proposta:** fundir vizinhos **na ordem de taxa** com menor perda de IV (ordenar primeiro, depois fusão adjacente), ou agrupar raros por similaridade de taxa. Ajustar doc: "otimo dado o conjunto de pré-bins".
- **Risco / testes:** comparar IV com e sem a mudança em alta cardinalidade.

##### B-08 — OBC_SBLP: `ensure_max_prebins` pode não atingir `max_n_prebins`
- **Algoritmo/arquivo:** src/OBC_SBLP_v5.cpp:410-455
- **Severidade:** Médio
- **Lente:** Robustez
- **Status:** Suspeito (lógica; não simulado)
- **Implementação:** candidatos = pares adjacentes fixos ordenados por diferença de taxa; `if (merged[idx1] || merged[idx2]) continue;` e uma única passada. Uma vez que um par (a,b) absorve b, o par (b,c) é pulado; se a ordem de diferença for desfavorável, restam pré-bins acima do limite (o limite máximo de fusões é < n−1).
- **Impacto:** `max_n_prebins` vira limite soft; DP O(n²k) com n inesperadamente grande em alta cardinalidade (memória `split` (n+1)×(k+1) size_t).
- **Correção:** laço `while (size > max_n_prebins)` com recomputação das diferenças (heap) após cada fusão.
- **Teste:** 200 categorias, `max_n_prebins=5` → `expect_lte(nrow(res), 5)`.

##### B-09 — Suavização/IV inconsistentes entre algoritmos (aditividade e K-dependência)
- **Algoritmo/arquivo:** OBC_MILP_v5.cpp:226-250 (`dist = (pos+prior_pos)/(Tp+0,5)` por bin → Σ_bins dist ≠ 1 e depende de K); shared `compute_woe(..., BAYESIAN)` (woe_iv_utils.h:54-60) usado por OBN_DP; OBN_LPDB_v5.cpp:~1117-1145 (WoE Laplace-K, IV com `pos/total_pos` cru); OBN_MBLP_v5.cpp `find_min_iv_loss_merge` (`iv_before` na suavização de K, `iv_merged` em K−1); OBN_MRBLP/OSLP (cap ±20 só com `laplace_smoothing=0`)
- **Severidade:** Médio/Baixo
- **Lente:** Numérica
- **Status:** Confirmado (leitura)
- **Teoria:** IV é aditivo apenas se a distribuição suavizada soma 1 e a suavização é a mesma antes e depois da fusão; senão a "perda de IV" mistura escalas (viés a favor/contra fundir).
- **Impacto:** pequeno para n grande (prior 0,5 vs milhares), relevante para categorias esparsas / alta cardinalidade (OBC_MILP com K=1000 e Tp=500: Σ dist_pos ≈ 1,1); IV por bin em OBN_DP/LPDB = (p_cru−q_cru)·WoE_suav pode ser negativo para bins minúsculos.
- **Correção:** `dist = (c + α)/(T + α·K)` com K=nº de bins *do candidato* para ambos os lados da comparação (antes/depois); IV=(dist_p−dist_n)·woe com as mesmas dist.
- **Risco/teste:** `expect_gte(min(res$iv), 0)`.

##### B-10 — OBC_MILP e família "LP": nenhum solver; rótulos e títulos "Optimal"
- **Algoritmo/arquivo:** src/OBC_MILP_v5.cpp:30-46; OBN_MBLP:51-60; OBN_MRBLP:46-60; OBN_OSLP:44-60; R/*.R (títulos "Optimal Binning ... Linear Programming")
- **Severidade:** Baixo (documentação já admite)
- **Lente:** Teoria
- **Status:** Confirmado — não há constraint matrix, simplex, relaxação, branch-and-bound em nenhum arquivo do escopo (grep por `simplex|solver|branch|lp_` = 0).
- **Impacto:** nomes de função/classes sugerem otimalidade; vale padronizar (`ob_*_greedy`) ou implementar DP exato (B-06). OBC_MILP: redução de pré-bins por similaridade de taxa de qualquer par (`reduce_prebins_by_similarity`) ignora contagens; ordem `std::sort` por nome influencia empates.

##### B-11 — Parâmetros mortos e `converged` sem significado
- **Algoritmo/arquivo:** OBN_LPDB_v5.cpp:118,126,147 (`polynomial_degree` validado e nunca usado; densidade = KDE gaussiano com bandwidth de Silverman); OBN_DP/MRBLP/OSLP/MBLP/OBC_DP `convergence_threshold` (validado, nunca usado); OBC_DP `converged = true` constante; OBC_SBLP `converged = max_iterations > 1`
- **Severidade:** Baixo
- **Lente:** Teoria
- **Status:** Confirmado
- **Impacto:** usuário ajusta parâmetros sem efeito; `converged` não informa nada. LPDB: cortes por mínimos/inflexões da densidade *ignoram o alvo*; em densidades unimodais cai para quantis (Suspeito; não executado) e, quando há >`max_n_prebins−1` mínimos, só os mais à esquerda são usados (`find_density_critical_points` trunca em ordem de x).
- **Correção:** remover/ignorar com `warning`, ou implementar; reportar `converged` real.

##### B-12 — OBN_DP: pré-binning por divisão inteira desbalanceia bins para n pequeno
- **Algoritmo/arquivo:** src/OBN_DP_v5.cpp:371 (`bin_size = max(1, n / max_n_prebins)`, `idx = i*bin_size`)
- **Severidade:** Baixo
- **Lente:** Robustez
- **Status:** Confirmado (aritmética): n=39, prebins=20 → `bin_size=1`, cortes nos índices 1..19: 19 pré-bins com 1 obs e o último com 20.
- **Correção:** `idx = (long long)i * n / max_n_prebins`.
- **Teste:** `n=39` → contagens dos pré-bins diferem ≤ 1 (antes de fusões).

##### B-13 — Wrappers: NA→"NA" colide com categoria real "NA"; caminho "MISSING" do SBLP é código morto
- **Algoritmo/arquivo:** R/obc_dp.R:248, R/obc_milp.R:138, R/obc_sblp.R (`feature[is.na(feature)] <- "NA"`); OBC_SBLP_v5.cpp `compute_initial_counts` ("MISSING")
- **Severidade:** Baixo
- **Lente:** Robustez
- **Status:** Confirmado (leitura)
- **Impacto:** um NA e a string literal "NA" são contados como a mesma categoria; em C++ direto (sem wrapper) NA vira "MISSING" e colide com uma categoria "MISSING" real.
- **Correção:** deixar NA chegar ao C++ (DP/MILP já tratam `NA_STRING` no export do DP) e usar um rótulo sentinela reservado; avisar na colisão.

#### Problemas transversais observados

1. **IV aditivo ≠ objetivo K-independente.** Todas as famílias usam Laplace/Bayes com K na normalização; correto só se K for fixado no DP/comparação (B-02, B-09).
2. **Ordenação por taxa suavizada vs crua** (SBLP) — a contiguidade do ótimo exige a razão de verossimilhança exata; OBC_DP faz bem (`rate_less` em inteiros de 64 bits, com empates determinísticos).
3. **Pós-processamento guloso** (monotonia depois de max_bins, raros depois de monotonia) sem retroceder: cada passo pode desfazer o anterior; o código tenta "re-aplicar uma vez".
4. **`min_bins` como piso rígido que prevalece sobre a restrição de monotonia/cutoff**, sem sinalizar (`converged=TRUE`).
5. **Critérios de fusão ad hoc** (|ΔWoE|, |ΔIV|, ΣIV) em vez de perda de IV (B-03, B-04, B-06).
6. Mixed: contagens `int` em C++ (overflow só para n>2³¹, irrelevante); determinismo é bom (ordem por nome, empates determinísticos) nos categóricos.

#### Priorização recomendada (top 5 do seu escopo)

1. **B-01** (OBC_DP: bin puro WoE=IV=0) — corrigir estimador; é erro de resultado, não de heurística.
2. **B-02** (OBC_SBLP: objetivo do DP ≠ IV reportado; remover `chunked`).
3. **B-03/B-04** (OSLP/MRBLP: usar perda de IV; 3 linhas; ganho de ~20%/6% de IV médio).
4. **B-05** (monotonicidade garantida: PAVA em taxa crua + flag `converged=false` honesta).
5. **B-06** (DP monotônico exato O(m³k) para OBN_*; reaproveita o `seg_iv` do OBC_DP) — e renomear/documentar os "LP/MILP" que ficarem gulosos (B-10).

---

## Anexo C — Agente C — ML e meta-heurísticas


Escopo: `src/OBN_UDT_v5.cpp`, `src/OBC_UDT_v5.cpp`, `src/OBN_IR_v5.cpp`, `src/OBN_KMB_v5.cpp`, `src/OBC_SAB_v5.cpp`, `src/OBN_BB_v5.cpp`, `src/OBN_EWB_v5.cpp` e wrappers R correspondentes. Somente leitura no repo. Experimentos standalone (g++, sem Rcpp) em `scratchpad/work_C/` (`sab.cpp`, `udt.cpp`, `ewb.cpp`, `dir.cpp`, `bb.cpp`, `ir_bb.cpp`). Cada experimento é uma réplica fiel do trecho citado (RNG do R trocado por `mt19937_64`).

#### Resumo executivo

- O código já passou por uma rodada de correções (comentários "used to..." por todo lado): não encontrei UB, loops infinitos, RNG não semeado, `rand()`/`random_device`, nem desencontro de ordem de argumentos `.Call` vs. `RcppExports`. A convenção de borda `(lo; hi]` (x == cutpoint vai para o bin inferior) é **consistente** entre os 7 algoritmos e `OB_ApplyWoENum.cpp`/`OB_Cutpoints.cpp` (`std::lower_bound` / `countLE` / `<=`).
- **PAVA (IR) está correto**: pilha, O(bins), comparação exata em inteiros, pooling ponderado por contagem (soma de pos/cnt), direção decrescente por reversão. O problema do IR é de desenho (pré-bins limitados a `max_bins`), não do PAVA.
- **SAB**: RNG do R (`R_unif_index`/`unif_rand` + `RNGScope`) — reprodutível com `set.seed` e independente da ordem das linhas. Porém a busca **para prematuramente na iteração ~200** (de 1000) e devolve praticamente a solução inicial; o IV final fica em média 11% (m=40 categorias) a 17% (m=150) abaixo do ótimo exato (DP). A penalidade de "monotonicidade" é um artefato de rótulo e só atrapalha.
- **UDT numérico**: o pré-binning "top-K ganhos de informação" escolhe K cortes **adjacentes ao melhor split** (todos os 20 cortes em 0.697–0.701 num exemplo de 3 patamares), e a fusão posterior deixa um bin de 0.12% das observações e perde totalmente o segundo ponto de mudança.
- **BB**: não há branch-and-bound (nem bound, nem poda); é *greedy* que funde o bin de **menor IV** no vizinho anterior — critério que, em simulação, perde 45% do IV em média vs. DP exato (1.6% com o critério correto "menor perda de IV").
- **EWB**: um único outlier (x=1000 numa N(0,1), n=1000) produz **2 bins (99.9% / 0.1%)** com `min_bins=3`, violando `min_bins` e `bin_cutoff`.
- **KMB** não executa k-means (nenhuma iteração de Lloyd; é EWB + fusões); `max_n_prebins` é ignorado acima de `max_bins`.
- Padrão transversal: os "enforcers" de monotonicidade (UDT, KMB, EWB, BB) param silenciosamente em `min_bins` e podem devolver WoE não-monótono sem sinalizar; a direção é escolhida a partir dos 2 primeiros bins (KMB, EWB).

#### Tabela de conformidade por algoritmo

| Algoritmo | Arquivo | Referência teórica | Fidelidade | Robustez | Performance (atual → ideal) | Estab. numérica | Nº achados (C/A/M/B) |
|---|---|---|---|---|---|---|---|
| UDT numérico | OBN_UDT_v5.cpp | Quinlan 1993 (C4.5), Breiman 1984 (CART), Fayyad & Irani 1993 | Baixa (não é árvore; top-K de ganhos independentes) | Parcial | O(n log n + B·n) → ok | Boa (EPSILON floor; Kahan desnecessário) | 0/1/2/3 |
| UDT categórico | OBC_UDT_v5.cpp | (nome genérico "User-Defined Technique"; JS-divergence merge) | Parcial (não é árvore; doc honesta) | Boa | O(B² log) greedy já otimizado → ok | Boa | 0/0/0/2 |
| Isotonic (IR) | OBN_IR_v5.cpp | Barlow et al. 1972; Best & Chakravarti 1990 | Parcial (PAVA ok; pré-bins ≤ max_bins; direção por Pearson) | Boa | O(n log n) → ótimo | Boa (inteiros exatos); WoE suavizado pode inverter | 0/0/2/2 |
| K-means (KMB) | OBN_KMB_v5.cpp | Lloyd 1982; Wang & Song 2011 (Ckmeans.1d.dp) | Baixa (sem k-means; equal-width) | Parcial | O(n log n) → ok | IV inconsistente (WoE suav. × dist. crua) | 0/0/3/1 |
| Simulated Annealing (SAB) | OBC_SAB_v5.cpp | Kirkpatrick et al. 1983; Metropolis 1953 | Parcial (Metropolis ok; schedule/convergência frágeis) | Boa (sem UB; RNG ok) | O(it·m) heurístico → DP exato O(k·m²) | Ok | 0/1/1/2 |
| Branch & Bound (BB) | OBN_BB_v5.cpp | Land & Doig 1960 | Baixa (não há B&B) | Parcial | O(B²) → ok | Ok | 0/1/1/2 |
| Equal-width (EWB) | OBN_EWB_v5.cpp | Dougherty, Kohavi & Sahami 1995 | Alta para o pré-bin; Parcial no pipeline | Parcial (outliers) | O(n log n) → ok | Ok | 0/1/1/2 |

Contagem de achados na tabela agrupa os itens abaixo por arquivo principal.

#### Achados

##### C-01 — UDT numérico: pré-binning "top-K ganho" produz cortes redundantes e perde pontos de mudança
- **Algoritmo/arquivo:** `src/OBN_UDT_v5.cpp:425-520` (`get_entropy_based_cutpoints`, sort em :508), usado em :123; fusão em :597-660; split de rescue em :794.
- **Severidade:** Alto
- **Lente:** Teoria / Robustez
- **Status:** Confirmado (experimento `work_C/udt.cpp`)
- **Teoria:** Em CART/C4.5 (e MDLP de Fayyad & Irani 1993) o ganho é recalculado **recursivamente em cada partição** (top-down), de modo que o segundo split é escolhido condicionalmente ao primeiro. Os ganhos de splits "adjacentes" ao melhor split têm ganho quase igual, logo escolher os K maiores ganhos do nó raiz não fornece K pontos de corte informativos.
- **Implementação:** calcula o ganho de cada candidato **uma única vez na raiz**, ordena e fica com os `max_n_prebins` maiores (`std::sort(info_gains…)` :508).
- **Impacto / cenário de falha:** n=5000, x~U(0,1), P(y=1)=0.1 (x<0.3), 0.5 (0.3≤x<0.7), 0.9 (x≥0.7). Os 20 cortes escolhidos ficam **todos em [0.697, 0.701]** (pré-bins: 3418, 1, 1, …, 1, 1563). Após `merge_rare_bins` com `min_bins=3` (para em `bins.size()==min_bins`) restam:
  `(-Inf,0.700] n=3431 (68.6%) | (0.700,0.701] n=6 (0.12%) taxa=1.0 | (0.701,Inf) n=1563`.
  O ponto de mudança em 0.3 é perdido, existe um bin com 0.12% (< `bin_cutoff=0.05`) com WoE de ruído, e `max_bins=5` nunca é explorado (só `split_optimal_bin` :794 poderia recuperar, mas só é chamado quando `bins.size() < min_bins`).
- **Correção proposta:** pré-binning recursivo best-first: manter uma fila de partições, a cada passo dividir a partição cujo melhor split tem maior ganho absoluto (ganho × tamanho) respeitando `min_leaf = ceil(bin_cutoff*n)` (poda por tamanho mínimo, não por IV), até `max_n_prebins` folhas. Reaproveita o sweep O(n) já existente por partição:
  ```cpp
  // esboço
  struct Node{size_t lo,hi; double gain; size_t split;}; // [lo,hi) em data_pairs ordenado
  auto best_split=[&](size_t lo,size_t hi){ /* sweep de counts cumulativos, ignora empates,
        exige min_leaf em ambos os lados, retorna (gain, idx) */ };
  std::priority_queue<Node> pq; pq.push(make_node(0,n));
  while(cuts.size()<max_cuts && !pq.empty()){ auto nd=pq.top(); pq.pop(); if(nd.gain<=0)break;
      cuts.push_back(split_point_between(...)); pq.push(make_node(nd.lo,nd.split)); pq.push(make_node(nd.split,nd.hi)); }
  ```
  Complexidade O(n log n · profundidade) (ou O(n) por nível), igual à atual quando K é pequeno.
- **Risco / testes:** muda resultados (para melhor) em todos os casos com ≥2 pontos de mudança. Teste testthat: `x<-runif(5000); y<-rbinom(5000,1,ifelse(x<.3,.1,ifelse(x<.7,.5,.9))); r<-ob_numerical_udt(x,y); expect_true(all(r$count/sum(r$count) >= 0.05)); expect_true(any(abs(r$cutpoints-0.3)<0.05)) ; expect_true(any(abs(r$cutpoints-0.7)<0.05))`.

##### C-02 — SAB: parada prematura (~iteração 200) devolve quase a solução inicial; IV 11–17% abaixo do ótimo
- **Algoritmo/arquivo:** `src/OBC_SAB_v5.cpp:241` (`kConvergenceWindow = 200`), :531-550 (critério), :469-560 (`fit`).
- **Severidade:** Alto
- **Lente:** Teoria / Robustez
- **Status:** Confirmado (experimento `work_C/sab.cpp`, 200 instâncias por tamanho; DP exato como referência)
- **Teoria:** Kirkpatrick et al. 1983: a busca só se justifica com resfriamento até temperatura baixa; critérios de parada devem ser relativos ao orçamento/tamanho do espaço (aqui o espaço é k^m).
- **Implementação:** `last_improvement_iter` parte de 0 e a parada ocorre em `iter - last_improvement_iter >= 200 && is_monotonic(best)` (checado a cada 10). Para m≳40 categorias, melhorias ≥1e-6 raramente ocorrem em 200 iterações de vizinhança aleatória (≈0.6% de chance por iteração de consertar uma categoria mal alocada em m=150) → para em it=200–260 de 1000 e `converged=TRUE`.
- **Impacto:** gap relativo de IV vs. ótimo exato (partição contígua em ordem de taxa de evento, DP), parâmetros default (3–5 bins, 1000 it):

  | m categorias | configuração default | sem penalidade de monotonicidade | + sem janela de parada | + sem janela, 20 000 it |
  |---|---|---|---|---|
  | 12  | gap médio 2.6% (161/200 subótimos) | 0.7% | 0.3% | ~0% |
  | 40  | 11.1% (máx 47%) | 10.3% | 5.0% | 0.5% |
  | 150 | **17.0%** (máx 43%) — idêntico ao da solução inicial em 200/200 | 17.0% | 16.5% | 3.2% |

  Em m=150 o resultado do SA é o da inicialização (rate-sorted em chunks de igual nº de categorias, +10% de ruído) em todas as 200 instâncias.
- **Correção proposta:** (a) preferencial: substituir o SA por **DP exato**: para o IV (soma de termos convexos nas somas de grupo, Fisher 1958; Walter Fisher "On grouping for maximum homogeneity"), a partição ótima em k grupos é contígua na ordenação por taxa de evento → O(k·m²) após ordenar (m=1000 categorias: ~5M operações). Manter o SA apenas como opção. (b) se mantiver SA: janela = `max(200, 20*m)`, não parar enquanto `iter < max_iterations/2`, e escalar `max_iterations` com m.
  ```cpp
  const int window = std::max(kConvergenceWindow, 20 * m);
  if (iter % 10 == 0 && iter >= max_iterations/2 && iter - last_improvement_iter >= window) {...}
  ```
- **Risco / testes:** (a) muda saída (melhora IV) e remove dependência de `set.seed`; manter `converged=TRUE`. Teste: comparar `total_iv` com enumeração força-bruta para m≤10 e com DP para m grande (`expect_gte(iv_sab, 0.99*iv_dp)`).

##### C-03 — SAB: penalidade de "monotonicidade" depende do rótulo do bin (sem significado) e degrada a busca
- **Algoritmo/arquivo:** `src/OBC_SAB_v5.cpp:233` (`iv -= 500.0`), :244-280 (`is_monotonic`), :567-620 (`ensure_monotonicity`), :722 (ordenação final por WoE).
- **Severidade:** Médio
- **Lente:** Teoria / Robustez
- **Status:** Confirmado (`sab.cpp`, coluna "sem penalidade")
- **Teoria:** categorias nominais não têm ordem natural; "WoE monótono" só faz sentido após ordenar os bins por WoE — o que `get_results` já faz sempre. O IV é invariante a permutação de rótulos.
- **Implementação:** `is_monotonic` examina a taxa de evento na **ordem do índice do bin** (rótulo arbitrário). Qualquer solução com rótulos fora de ordem recebe −500, mesmo sendo idêntica (a menos de rótulos) a uma solução "monótona". A solução inicial (+10% de reatribuições aleatórias) quase sempre é penalizada; swaps/reatribuições aleatórias quebram a ordem de rótulos. `ensure_monotonicity` apenas re-rotula por taxa, sempre aceito (`monotonic_iv > best_iv` pois best carrega a penalidade); o ramo `else` (100 tentativas) é código morto. A doc ("Guaranteed monotonic WoE trend") é verdadeira trivialmente.
- **Impacto:** m=12: subótimo em 161/200 com penalidade vs. 93/200 sem (gap médio 2.6% → 0.7%); a penalidade também impede a parada correta (`is_monotonic(best)` na condição de parada).
- **Correção:** remover a penalidade e `is_monotonic` da função objetivo; manter só a ordenação final por WoE (já existe). `-` linhas :232-234, :244-280 (condição em :546), :567-620.
- **Risco / testes:** nenhum de comportamento externo além de melhorar IV. `expect_true(!is.unsorted(r$woe))` continua válido.

##### C-04 — SAB: schedule de resfriamento é reaquecido (T=0.5·T0) a cada 100 rejeições e Metropolis com probabilidade >1
- **Algoritmo/arquivo:** `src/OBC_SAB_v5.cpp:381` (`probability *= 1.5`), :524-529 (restart `temperature = initial_temperature*0.5`), :479-520.
- **Severidade:** Baixo
- **Lente:** Teoria
- **Status:** Confirmado (leitura); impacto não isolado experimentalmente
- **Teoria:** Metropolis: aceita Δ<0 com p=exp(Δ/T)≤1; resfriamento monotônico (geométrico) T_{k+1}=αT_k. O fator 1.5 pode dar p>1 (inócuo, mas quebra o balanço detalhado) e o reaquecimento a 0.5·T0 impede a temperatura de chegar a valores baixos (IV típico tem deltas ~0.01–0.1, T=0.5 aceita quase tudo).
- **Correção:** remover o fator 1.5 e o reset de temperatura (reiniciar da melhor solução, mantendo T); ou reaquecimento decrescente.
- **Risco:** baixo.

##### C-05 — BB: não há Branch-and-Bound; critério de fusão (menor IV → fundir com o anterior) é ruim
- **Algoritmo/arquivo:** `src/OBN_BB_v5.cpp:508-540` (loop `min_element` por `iv`), cabeçalho/`R/obn_bb.R:1-6` ("optimally", "Branch and Bound-style").
- **Severidade:** Alto
- **Lente:** Teoria
- **Status:** Confirmado (experimento `work_C/bb.cpp`, 2000 instâncias, 20 pré-bins → 5 bins, sem monotonicidade)
- **Teoria:** B&B (Land & Doig 1960) exige árvore de busca com limite superior válido e poda; o problema de partição contígua maximizando IV tem DP exato O(k·B²) (B=nº de pré-bins). O critério correto de fusão gulosa é a **perda de IV da fusão** (como fazem EWB `ensure_max_bins` e UDT `merge_optimal_bins`).
- **Implementação:** a cada passo funde o bin de menor IV *individual* (não a perda da fusão) com o vizinho **anterior** (ou o seguinte se for o primeiro), ignorando qual vizinho é mais parecido. Um bin de IV baixo (taxa ≈ média) entre dois bins de WoE alto é absorvido pelo anterior, destruindo o IV do vizinho.
- **Impacto:** IV final vs. DP exato: critério do BB perde **45% em média** (máx 85%; subótimo em 99.8% das instâncias); o critério "menor perda de IV do par adjacente" perde 1.6% (máx 16.7%). (Com `is_monotonic=TRUE` a monotonicidade reduz o espaço, mas o critério continua o mesmo.)
- **Correção:** (1) trocar o critério por perda de IV do par adjacente (código igual ao de `OBN_EWB_v5.cpp:380-415`); (2) opcional/ideal: DP exato sobre os pré-bins (para monotonicidade, DP com restrição de taxa monótona crescente/decrescente, melhor dos dois sentidos); (3) renomear/documentar como "greedy" se não houver B&B.
  ```cpp
  for (size_t i=0;i+1<bins.size();++i){ NumericalBin m=bins[i]; m.merge_with(bins[i+1]);
      double loss=(bins[i].iv+bins[i+1].iv)-iv_of(m); if(loss<best){best=loss;idx=i;} }
  ```
- **Risco / testes:** muda resultado. Teste: para 20 pré-bins sintéticos comparar `total_iv` com o ótimo por força-bruta (C(19,4)=3876 partições).

##### C-06 — EWB: um outlier colapsa o binning (2 bins, 99.9%/0.1%), violando `min_bins` e `bin_cutoff`
- **Algoritmo/arquivo:** `src/OBN_EWB_v5.cpp:219-260` (`create_prebins`), :300-370 (`merge_rare_bins` para em `min_bins`), :514-526 (`drop_empty_bins`), :571 (`fit`).
- **Severidade:** Alto
- **Lente:** Robustez
- **Status:** Confirmado (`work_C/ewb.cpp`, réplica de create_prebins + merge_rare_bins + drop_empty_bins)
- **Teoria:** Dougherty, Kohavi & Sahami (1995, §3.1) notam a sensibilidade do equal-width a outliers; um método "optimal binning" não deve devolver menos que `min_bins` quando há ≥`min_bins` valores distintos.
- **Implementação:** com x~N(0,1), n=1000 e um valor 1000: 20 pré-bins de largura 50; `merge_rare_bins` para quando `size==min_bins=3` deixando `(-Inf,47.1] n=999 | (47.1,949.8] n=0 | (949.8,Inf) n=1`; `drop_empty_bins` remove o vazio sem recompor `min_bins`.
- **Impacto:** saída final `(-Inf,949.8] n=999 | (949.8,Inf) n=1`: 2 bins < `min_bins`, bin de 0.1% < `bin_cutoff`, IV ≈ 0. (O KMB tem `removeEmptyBins()` que bisseciona o bin mais populoso; o EWB não.)
- **Correção:** (a) remover bins vazios **antes** de contar contra `min_bins` e restaurar `min_bins` bisseccionando o bin mais populoso no ponto médio de seus valores (como `OBN_KMB_v5.cpp:~395-440`); (b) alternativamente fazer o pré-binning equal-width sobre `[q_{0.01}, q_{0.99}]` com caudas no primeiro/último bin (ressalva: muda a definição clássica).
- **Risco / testes:** `x<-c(rnorm(999),1000); y<-rbinom(1000,1,plogis(x)); r<-ob_numerical_ewb(x,y); expect_gte(length(r$count),3)`.

##### C-07 — KMB não executa k-means; `max_n_prebins` ignorado acima de `max_bins`; `adjustBinCount` é código morto
- **Algoritmo/arquivo:** `src/OBN_KMB_v5.cpp:273-310` (`createRegularBins`), :279 (`n_bins = min(n_bins, max_bins)`), :607-640.
- **Severidade:** Médio
- **Lente:** Teoria
- **Status:** Confirmado (leitura)
- **Teoria:** Lloyd 1982 itera atribuição/atualização de centróides até convergir; em 1D o ótimo exato é obtido por DP (Wang & Song 2011, Ckmeans.1d.dp, O(k·n²) ou O(k·n log n)).
- **Implementação:** centróides igualmente espaçados `min+(i+0.5)·range/n_bins` e cortes nos pontos médios ≡ **equal-width**; nenhuma iteração. Como `n_bins ≤ max_bins`, `adjustBinCount()` (que só funde se `size>max_bins`) nunca age, e `max_n_prebins>max_bins` não tem efeito. Os centróides reportados são recalculados como média dos bins (:`computeCentroids`) apenas para exibição. A doc é honesta ("K-means style initialization"), mas o nome/`@seealso` sugerem clustering.
- **Impacto:** em features assimétricas/com outliers o resultado é pior que EWB (mesmos cortes, menos pré-bins) e idêntico a um EWB com `max_n_prebins=max_bins`.
- **Correção:** executar Lloyd 1D (ou DP ótimo) sobre os valores/quantis com k = min(max_n_prebins, distintos); depois fusões supervisionadas até `max_bins`. Esboço Lloyd 1D com dados ordenados: atribuição por busca binária nos pontos médios, atualização por somas prefixadas — O(k log n) por iteração; semente determinística (quantis).
- **Risco:** muda resultados.

##### C-08 — Direção de monotonicidade decidida pelos 2 primeiros bins (KMB, EWB)
- **Algoritmo/arquivo:** `src/OBN_KMB_v5.cpp:566` (`increasing = (bins[1].woe >= bins[0].woe)`), `src/OBN_EWB_v5.cpp:449` (`prefer_increasing = ...`).
- **Severidade:** Médio
- **Lente:** Teoria / Robustez
- **Status:** Confirmado (`work_C/dir.cpp`, réplica do enforcement, 5000 reps, 5 bins, 200 obs/bin)
- **Teoria:** a direção da regressão isotônica deve ser a de menor SSE ponderado (Robertson, Wright & Dykstra 1988, cap. 1) ou o sinal da tendência global; usar 2 pontos é estimar um coeficiente angular com n=2.
- **Impacto:** tendência verdadeira crescente com inclinação fraca (logit −0.5+0.2·i): direção errada em **15.9%** das execuções, resultado final não-monótono em 2.5% (vs. 0.4% com tendência global), IV médio 0.0895 vs 0.0931. Com inclinação forte (0.5/bin) o erro cai a 0.9%. Direção errada → fusões em cascata até `min_bins`.
- **Correção:** escolher a direção por IV/SSE como o BB já faz (`OBN_BB_v5.cpp:356-400`) ou por SSE isotônico; mínimo: sinal de correlação (índice do bin, WoE) ponderada por contagem.
- **Risco:** baixo.

##### C-09 — Enforcers de monotonicidade param silenciosamente em `min_bins` (UDT, KMB, EWB, BB)
- **Algoritmo/arquivo:** `src/OBN_UDT_v5.cpp:754-775` (`bins.size() > min_bins`), `OBN_KMB_v5.cpp:550-590`, `OBN_EWB_v5.cpp:440-505`, `OBN_BB_v5.cpp:356-440`.
- **Severidade:** Médio
- **Lente:** Robustez
- **Status:** Confirmado por leitura e por simulação (C-08: 2.5% de saídas não-monótonas)
- **Teoria:** a restrição de monotonicidade é "hard" no PAVA; aqui `min_bins` tem precedência sem aviso, e nenhum campo (`monotonic = TRUE/FALSE`) informa se o resultado é monótono. Só o BB documenta ("monotonic unless that would require fewer than min_bins").
- **Correção:** (i) retornar `is_monotonic_achieved`; (ii) preferir PAVA (O(B)) com pooling ponderado como no IR e só então verificar `min_bins`; (iii) documentar.
- **Risco:** baixo.

##### C-10 — IR: pré-bins limitados a `max_bins`; PAVA só pode reduzir e `min_bins` não é garantido
- **Algoritmo/arquivo:** `src/OBN_IR_v5.cpp:313-314`, :663-740; doc `R/obn_ir.R` (@param max_n_prebins / min_bins, já admite).
- **Severidade:** Médio
- **Lente:** Teoria
- **Status:** Confirmado (leitura; comportamento documentado)
- **Teoria:** PAVA (Barlow et al. 1972) deve ser aplicado a pré-bins **finos** (aqui `max_n_prebins=20`); o nº final de blocos é resultado do ajuste, depois limitado a `[min_bins,max_bins]` por fusão (se >max) — nunca por truncar a resolução inicial.
- **Implementação:** `n_prebins = min(max_n_prebins, unique, max_bins)`, `max(.., min_bins)`; PAVA depois pode pooling para <`min_bins` (nenhum re-split após :663). Com 5 pré-bins ruidosos e tendência fraca, PAVA frequentemente colapsa para 2 bins.
- **Correção:** usar `n_prebins = min(max_n_prebins, unique)` (quantis de observações), rodar PAVA, e então fundir até `max_bins` pelo critério de menor perda de IV; se `<min_bins`, dividir o bloco com maior N.
- **Risco:** muda resultados; **testes:** `expect_lte(length(r$count), max_bins)` e IV ≥ IV atual.

##### C-11 — IR: direção por Pearson em pontos médios (sensível à escala) em vez de SSE isotônico; pré-bins por quantil dos valores *distintos*
- **Algoritmo/arquivo:** `src/OBN_IR_v5.cpp:572-602` (+ `calculateCorrelation`), :320-330 (idx por `unique_vals`).
- **Severidade:** Baixo
- **Lente:** Teoria
- **Status:** Confirmado (leitura; `work_C/ir_bb.cpp`: em perfis aleatórios de 5 bins, Pearson-por-ponto-médio e SSE-mínimo discordam em 11.9% dos casos, o SSE da direção escolhida sendo em média 37% maior que o ótimo quando discordam — perfis ruidosos, logo indicativo)
- **Detalhes:** pontos médios dos bins extremos são inventados (`±1.0` do corte vizinho), e Pearson depende da escala/espaçamento; Spearman ou SSE isotônico não dependem. Os cortes por quantil de valores distintos (não de observações) ignoram a massa de empates (ex.: 90% de zeros) e produzem bins desbalanceados, mitigado só por `mergeLowFrequencyBins`.
- **Correção:** calcular SSE ponderado das duas direções com o mesmo PAVA (custo O(B)) e escolher o menor; quantis por observação (`pos_sorted`/`neg_sorted` já permitem).
- **Risco:** baixo.

##### C-12 — IR: WoE suavizado pode inverter a ordem mesmo com taxas isotônicas
- **Algoritmo/arquivo:** `src/OBN_IR_v5.cpp:727-760`.
- **Severidade:** Baixo
- **Lente:** Numérica
- **Status:** Suspeito (aritmética simples, não rodado)
- **Detalhe:** PAVA garante p/(p+n) monótono; o WoE usa (p+0.5)/(n+0.5), que não é função monótona da taxa quando as contagens diferem muito (ex.: bin (p=0,n=1) taxa 0 → WoE −1.1; bin (p=1,n=40) taxa 0.024 → WoE −3.3). Só ocorre com bins minúsculos; `bin_cutoff` mitiga.
- **Correção:** rodar PAVA sobre as razões suavizadas (ou checar `woe` monótono pós-hoc).

##### C-13 — UDT numérico: normalização do bin "NA" inconsistente com os demais bins (IV total não é um IV)
- **Algoritmo/arquivo:** `src/OBN_UDT_v5.cpp:163-210` (`calculate_woe_iv` — totais só dos bins regulares) vs. :263-305 (`calculate_woe_iv_all` — totais incluindo o bin de missing); `handle_few_unique_values` (:1020-1190) usa totais com missing para **todos** os bins.
- **Severidade:** Médio
- **Lente:** Numérica / Teoria
- **Status:** Confirmado (leitura)
- **Teoria:** WoE_i = ln(p_i/n_i) onde p_i = pos_i/P_total é a distribuição sobre **todos** os bins (Siddiqi 2006; Thomas 2009), logo Σp_i=1 e IV=Σ(p_i−n_i)·WoE_i.
- **Implementação:** bins regulares: denominador `P_reg + α·nb`; missing: `P_reg+P_miss + α(nb+1)`. As distribuições somam >1 e WoE(missing) não é comparável a WoE(regular). `total_iv` (:`calculate_total_iv`) soma ambos. Além disso, ≤2 valores distintos e ≥3 valores distintos usam normalizações diferentes.
- **Correção:** calcular totais incluindo missing para todos os bins (como no ramo de poucos valores únicos) ou deixar o bin NA fora do IV.
- **Risco:** altera WoE/IV dos bins regulares quando há NA; `testthat`: `expect_equal(sum(exp(woe)*...))` — mais direto: com α=0, Σ(count_pos/P)=1 sobre todas as linhas.

##### C-14 — KMB: IV mistura WoE suavizado com distribuições cruas
- **Algoritmo/arquivo:** `src/OBN_KMB_v5.cpp:70-112` (`calculateWOE` suavizado, `calculateIV` com `pos/total_pos` cru).
- **Severidade:** Médio
- **Lente:** Numérica
- **Status:** Confirmado (leitura)
- **Detalhe:** IV = (pos/P − neg/N)·WoE_suavizado: não é a identidade usada nos outros 6 algoritmos (`(p_s−n_s)·WoE_s`). Ex.: P=100, N=900, 5 bins, bin (pos=1,neg=0): WoE_s=3.27; IV_KMB=(0.01−0)·3.27=0.033 vs. IV_suavizado=(0.0146−0.00055)·3.27=0.046 (−29%). Pode também ter sinal oposto ao esperado em bins minúsculos quando a ordem crua e a suavizada discordam; `total_iv` não é comparável entre algoritmos.
- **Correção:** `iv = (pos_rate_s − neg_rate_s) * woe` com as mesmas taxas suavizadas de `calculateWOE`.

##### C-15 — Parâmetros de convergência ignorados / doc inconsistente (EWB, IR, KMB, BB)
- **Algoritmo/arquivo:** `OBN_EWB_v5.cpp` (construtor: `double /* convergence_threshold: unused */`), `R/obn_ewb.R` (@param convergence_threshold: "threshold for determining convergence"), `OBN_IR_v5.cpp:41`, `OBN_KMB_v5.cpp` (`(void) convergence_threshold_`), `OBN_BB_v5.cpp`.
- **Severidade:** Baixo
- **Lente:** Teoria (contrato)
- **Status:** Confirmado
- **Detalhe:** só o EWB documenta o parâmetro como ativo quando não é; IR/KMB/BB documentam como não usado. O flag `converged` nesses algoritmos significa apenas "não estourou max_iterations".
- **Correção:** corrigir a doc do EWB; ou deprecar o parâmetro.

##### C-16 — OBC_UDT: não é árvore de decisão; `max_n_prebins` ignorado; `ensure_monotonicity` é só um sort
- **Algoritmo/arquivo:** `src/OBC_UDT_v5.cpp:326, 366, 458-462, 85-110 (calculate_divergence não usado)`.
- **Severidade:** Baixo
- **Lente:** Teoria
- **Status:** Confirmado
- **Detalhe:** o algoritmo é "agrupar raras + merge guloso por divergência JS (inclusive de bins não-adjacentes) até `max_bins` + ordenar por WoE". Não há critério de split (Gini/entropia), poda ou `min_leaf`. `max_n_prebins` só é validado. A doc é honesta ("User-Defined Technique"). Para categorias, a fusão ótima de IV é contígua na ordem de taxa (ver C-02) — o merge por JS entre bins não-adjacentes no espaço de taxa é uma heurística de qualidade inferior mas válida.
- **Correção:** documentar; opcional: usar o DP de C-02.

##### C-17 — UDT: seleção de cortes empatados depende de `std::sort` instável; `count < 10` fixo; cutpoint de 2 valores com `(a+b)/2`
- **Algoritmo/arquivo:** `src/OBN_UDT_v5.cpp:508` (sort por ganho), :807 (`current_bin.count < 10`), :1034 (`(min_val+max_val)/2.0`).
- **Severidade:** Baixo
- **Lente:** Robustez
- **Status:** Confirmado (leitura)
- **Detalhe:** (1) ganhos exatamente iguais (comum com contagens inteiras simétricas) são ordenados de forma dependente da implementação da STL → corte escolhido pode mudar entre plataformas; use `stable_sort` com desempate por posição. (2) limiar 10 hardcoded no split de rescue, independente de `bin_cutoff`/n. (3) `handle_few_unique_values` usa `(min+max)/2` em vez de `split_point_between`: com |valores|~1e308 vira `Inf` e com doubles adjacentes iguala o maior → `x ≤ cut` do `ob_apply` enviaria ambos para o bin esquerdo (a contagem interna usa `<= min_val` e discordaria).
- **Correção:** usar `split_point_between(min_val,max_val)`; `std::stable_sort`.

##### C-18 — BB: validação e rótulos
- **Algoritmo/arquivo:** `src/OBN_BB_v5.cpp:46-80` (sem checagem de duas classes, `bin_cutoff` aceita 0 e 1), :576-577 (`std::to_string`: 6 casas fixas).
- **Severidade:** Baixo
- **Lente:** Robustez / Numérica
- **Status:** Confirmado (leitura)
- **Detalhe:** alvo todo 0 (ou 1) não gera erro e devolve WoE/IV sem sentido; `bin_cutoff=1` funde tudo. Rótulos de bins truncam cortes em escalas pequenas (feature ~1e-8 → "0.000000;0.000000"); `cutpoints` numérico permanece correto.
- **Correção:** validar ambas as classes e `0<bin_cutoff<1`; usar `%.6g`.

##### C-19 — EWB/KMB: pré-bins equal-width com poucos valores distintos fundem valores distintos
- **Algoritmo/arquivo:** `src/OBN_EWB_v5.cpp:219-260`, `src/OBN_KMB_v5.cpp:273-310`.
- **Severidade:** Baixo
- **Lente:** Teoria / Robustez
- **Status:** Suspeito (aritmética direta)
- **Detalhe:** valores {0,1,100} com 3 bins de largura 33.3 dão {0,1} | ∅ | {100}; 0 e 1 não são separados e depois do `drop_empty_bins` ficam <`min_bins` bins, embora existam 3 valores distintos. Em EWB, `create_unique_bins` só é usado para ≤2 distintos.
- **Correção:** se `unique_count <= max(min_bins, max_n_prebins)` usar um bin por valor distinto (como IR/BB).

#### Problemas transversais observados

1. **Convenção de borda** `(lo; hi]` está uniforme (UDT `find_bin_index`/`in_bin`, IR `countLE`, KMB/EWB/BB `std::lower_bound` nos limites superiores) e coincide com `OB_ApplyWoENum.cpp` (`include_upper_bound=TRUE`) e `OB_Cutpoints.cpp` (`lower_bound` em `cuts`). Verificado; nenhum desvio. (Único ponto frágil: cortes calculados por aritmética de ponto flutuante em EWB, p.ex. `min+(i+1)*w` — 0.3 vs 0.30000000000000004 — podem diferir de uma ulp do corte "exato" teórico, mas é consistente entre fit e apply porque o vetor `cutpoints` numérico é o usado.)
2. **Reprodutibilidade/RNG:** apenas o SAB usa RNG, via `R_unif_index`/`R::unif_rand` com `Rcpp::RNGScope`; nenhum `rand()`, `random_device` ou `mt19937` no escopo (grep). A ordem das categorias é determinística (`std::sort` por nome). Sem dependência da ordem das linhas.
3. **Terminação:** todos os `while` têm guarda de progresso ou `max_iterations`; sem acesso fora de faixa encontrado (`rate_greater` do PAVA, `bin_edges`, `bins[i-1]` com `size>min_bins≥2`).
4. **`min_bins` "soft" em quase todos os pipelines** (C-06, C-09, C-10): as fases de fusão respeitam `min_bins` mas as fases de limpeza (`drop_empty_bins`, PAVA) podem violá-lo sem aviso. Recomenda-se uma passada final de "restaurar `min_bins` por bisseção" comum (existe em KMB:`removeEmptyBins`).
5. **Suavização de Laplace dependente do nº de bins** (`α·nb` no denominador) em IR/UDT/KMB/EWB/BB: o WoE de um bin muda quando outros bins são fundidos; ok, mas a comparação de IV antes/depois de fusão (EWB:380-415) mistura `nb`.
6. **Desalinhamentos nome ↔ algoritmo:** BB (sem B&B), KMB (sem k-means), UDT (sem árvore recursiva), SAB (SA que quase não annealing). Doc dos wrappers já é honesta em UDT/KMB; BB descreve "optimally" (R/obn_bb.R:1-6).

#### Priorização recomendada (top 5 do escopo)

1. **C-02/C-03 (SAB)** — trocar por DP exato (ou, no mínimo, remover a penalidade de monotonicidade e a janela de parada de 200 it): IV 11–17% abaixo do ótimo com `converged=TRUE`.
2. **C-05 (BB)** — corrigir o critério de fusão (perda de IV do par) e documentar como greedy; ganho de ~45% de IV nos testes.
3. **C-01 (UDT numérico)** — pré-binning recursivo (best-first) com `min_leaf`; hoje produz bins degenerados e perde pontos de mudança.
4. **C-06 (EWB)** — restaurar `min_bins` e `bin_cutoff` com outliers (bisseção do bin mais populoso, como no KMB).
5. **C-08/C-09/C-10 (direção e monotonicidade)** — direção por SSE/tendência global, flag de monotonicidade atingida, e pré-bins finos no IR; baixo risco e melhora robustez geral.

---

## Anexo D — Agente D — Risco de crédito e scorecards


Escopo: MOB, JEDI, JEDI-MWoE, DMIV (numéricos e categóricos), OBC_IVB, `woe_iv_utils.h`, `monotonicity_utils.h` e os wrappers R correspondentes. Nada no repositório foi editado. Os experimentos estão em `scratchpad/work_D/`: trechos C++ extraídos com stub de `Rcpp.h`, compilados com g++. Os fuzzers rodaram com ASan/UBSan.

#### Resumo executivo

- **Convenção de sinal.** Todos os algoritmos do escopo usam WoE = ln(%evento/%não-evento), com evento = target 1 e "bad" na documentação de `obwoe.R`. Isso é o oposto de Siddiqi (ln(%good/%bad)), mas é consistente entre algoritmos e documentado. `OB_ApplyWoE*` não depende de sinal: lê o `woe` do resultado.
- **IV_i ≥ 0.** Verificado em todos: (p−q)·ln(p/q) tem fatores de mesmo sinal, e o clamp ±20 preserva o sinal. Não encontrei violação.
- **Robustez básica.** O fuzz de 4000 execuções (n pequeno, empates, ±Inf, NaN, valores 1e-12) em MOB-num e DMIV-num não gerou UB, perda de observações nem cutpoints não crescentes.
- **Melhor implementação do escopo.** MOB numérico atinge IV 1,5465 contra o ótimo monotônico de 1,5539 (brute force sobre os mesmos 20 pré-bins). `OBC_IVB` usa DP exata (correta).
- **Problemas principais, todos de fidelidade/critério de merge, não de crash:**
  1. O redutor de bins do DMIV numérico (`OBN_DMIV_v5.cpp:724`) produz IV 30–50% abaixo do que MOB obtém nos mesmos dados.
  2. JEDI-num e JEDI-MWoE escolhem merge pela *soma* de IV dos dois bins, não pela *perda* de IV. Isso mantém bins puros duplicados e funde bins informativos.
  3. JEDI-MWoE exige monotonicidade simultânea em todas as K classes, o que é conceitualmente inadequado para K≥3.
  4. A monotonicidade "garantida" falha em silêncio quando `min_bins` impede o merge: 106/200 execuções em U-shape no MOB, 52/100 no DMIV, todas com `converged=TRUE`.
  5. `bin_cutoff` é ignorado em `OBC_MOB` (quando ncat ≤ max_bins) e em `OBC_DMIV` (parâmetro morto).
- **Categóricos.** A "monotonicidade" é trivial por construção: as categorias são ordenadas por WoE/taxa de evento antes de agrupar. O que importa é o critério de agrupamento e o tratamento de raras, e aí há falhas (D-05, D-06, D-08, D-13).
- **Smoothing incoerente.** Cada algoritmo reimplementa WoE/IV com um esquema diferente (D-10), e nenhum dos algoritmos do escopo inclui `woe_iv_utils.h` (só `OBN_DP` o usa). Os IVs totais não são comparáveis entre algoritmos.

#### Tabela de conformidade por algoritmo

| Algoritmo | Arquivo | Referência teórica | Fidelidade | Robustez | Performance (atual → ideal) | Estab. numérica | Achados (C/A/M/B) |
|---|---|---|---|---|---|---|---|
| OBN MOB | OBN_MOB_v5.cpp | Mironchyk & Tchistiakov 2017 (monotone binning; equivale a isotônica/PAV) | Parcial (heurística greedy, IV quase ótimo nos testes) | Boa (fuzz limpo); monotonia falha silenciosa em `min_bins` | O(N log N + k³) → ok | Boa (Laplace k-dependente, clamp ±20) | 0/1/1/2 |
| OBC MOB | OBC_MOB_v5.cpp | idem + Siddiqi (categorias ordenadas por taxa) | Parcial | `bin_cutoff` ignorado se ncat ≤ max_bins | O(B²) em rara-merge (64k cat = 11 s) → O(B log B) | Boa | 0/1/2/1 |
| OBN JEDI | OBN_JEDI_v5.cpp | Sem paper; nome sugere entropia, mas não há entropia | Baixa/Parcial | OK; monotonia silenciosa | O(N log N) (radix) ok | Fraca: piso EPS=1e-10 dá WoE ≈ −20,6…−23, sem clamp | 0/2/2/1 |
| OBC JEDI | OBC_JEDI_v5.cpp | idem | Parcial | OK; loop de otimização morto | O(B²) só com cutoff minúsculo | Boa (Bayes 0,5) | 0/0/1/1 |
| OBN JEDI-MWoE | OBN_JEDIMWoE_v5.cpp | WoE multinomial one-vs-rest (definição correta) | Parcial | Colapso de bins para K≥3; sem smoothing | O(N log N) ok | Fraca (EPS 1e-10, sem clamp) | 0/2/1/0 |
| OBC JEDI-MWoE | OBC_JEDIMWoE_v5.cpp | idem | Baixa (adjacência por ordem de contagem) | Ordem final arbitrária | O(B²) no pré-bin, otimizado | Laplace 2α, OK | 0/0/2/1 |
| OBN DMIV | OBN_DMIV_v5.cpp | Zeng 2013/2014 (fórmulas das divergências corretas) | Baixa na otimização (fórmulas OK) | Fuzz limpo; monotonia silenciosa | O(k²) irrelevante | Boa (Laplace 0,5k) | 0/2/2/0 |
| OBC DMIV | OBC_DMIV_v5.cpp | Zeng 2013 | Parcial | `bin_cutoff` morto, `max_n_prebins` não respeitado | O(B²) (16k cat = 3,7 s) | Piso EPS sem clamp em modo "woe" | 0/1/2/0 |
| OBC IVB | OBC_IVB_v5.cpp | Siddiqi / Zeng 2014 (DP ótima sobre ordem por taxa) | Alta | Boa | O(n²·k), n ≤ prebins | Boa | 0/0/0/3 |
| woe_iv_utils.h | header | Good 1950, Siddiqi 2006 | Parcial | n/a | n/a | Laplace com 2α e Bayes não normalizados | (D-10) |
| monotonicity_utils.h | header | — | Parcial | Pearson não robusto, NaN vira DESCENDING | n/a | ok | (D-17) |

#### Achados

##### D-01 — DMIV numérico: redução de bins pelo menor |contribuição| da divergência, sempre fundindo com o bin anterior
- **Algoritmo/arquivo:** src/OBN_DMIV_v5.cpp:724-757 (min_element) e 346-440 (`compute_bin_metrics`).
- **Severidade:** Alto. **Lente:** Teoria. **Status:** Confirmado (experimento).
- **Teoria:** Zeng (2013/2014) define as medidas como divergências entre as distribuições de bons e maus, a serem maximizadas. O critério correto para reduzir bins é fundir o par adjacente que causa a menor perda da divergência total. Zeng (2014) mostra que o ótimo exige ordem monotônica da taxa de evento, e portanto cabe DP (como em `OBC_IVB`).
- **Implementação:** escolhe o *bin* com menor `|bin.divergence|` (contribuição individual) e o funde sempre com o anterior (`prev`), ou com o próximo se for o primeiro:
  ```cpp
  auto min_div_it = std::min_element(bins.begin(), bins.end(),
      [](const NumericalBin &a, const NumericalBin &b){ return std::abs(a.divergence) < std::abs(b.divergence); });
  ```
  Isso ignora a divergência que o par fundido teria. A contribuição KL por bin, p·ln(p/q), pode ser negativa, e o `abs` não tem significado. Em "ln" (L∞) todas as contribuições são 0, exceto a do máximo. O min_element pega sempre o primeiro bin, e o algoritmo consome os bins mais à esquerda de forma independente dos dados.
- **Impacto:** teste com N=20000, x~U(0,10), logit(p) = −3+0,5x, 20 pré-bins, 3–5 bins, `bin_method="woe"`:

  | Resultado | IV total |
  |---|---|
  | Ótimo monotônico (brute force) | 1,554 |
  | MOB | 1,546 |
  | DMIV he/kl/tr/klj/sc/js/ln | 0,752 |
  | DMIV l1/l2 | 0,868 |

  Com x~N(0,1), 8 sementes (3–5 bins, `cutoff` 0,02), o IV do MOB foi 0,30–1,99 e o do DMIV (he/tr/klj/sc/js) ficou 29–38% abaixo em todas. kl/ln variaram, mas também ficaram abaixo do MOB. He, tr, klj, sc e js deram o mesmo IV na maioria das sementes: a escolha de `divergence_method` quase não altera o resultado. No caso uniforme, o cutpoint de x≈8,0 deixa 80% das observações em um único bin.
- **Correção proposta:** substituir o passo 4 por DP sobre os pré-bins (O(k²·max_bins)), maximizando a soma de contribuições. Isso vale para he/kl/tr/klj/sc/js/l1 (aditivos). Para l2, a DP maximiza Σd². Para "ln" (não aditivo), usar greedy por perda real:
  ```cpp
  // greedy por perda real (fallback / "ln")
  double base = total_div(bins);
  size_t best=0; double best_loss=1e300;
  for (size_t i=0;i+1<bins.size();++i){
    auto tmp = merge_copy(bins,i);          // soma contagens, recalcula métricas em tmp
    double loss = base - total_div(tmp);
    if (loss < best_loss){best_loss=loss;best=i;}
  }
  merge_bins(best);
  ```
  Requer refatorar `compute_bin_metrics` para operar sobre um vetor passado como argumento.
- **Risco / testes:** muda os resultados de todos os usuários de DMIV (para melhor). Teste testthat: `expect_gte(dmiv$total_iv, 0.95*mob$total_iv)` em dado logístico monotônico, para cada `divergence_method`. Teste que `he` e `ln` produzem cutpoints diferentes em dados assimétricos.

##### D-02 — JEDI numérico e JEDI-MWoE numérico: merge por soma de IV, não por perda de IV
- **Algoritmo/arquivo:** src/OBN_JEDI_v5.cpp:417-440; src/OBN_JEDIMWoE_v5.cpp:536-556.
- **Severidade:** Alto. **Lente:** Teoria. **Status:** Confirmado (experimento).
- **Teoria:** o par a fundir é o de menor perda `IV_i + IV_{i+1} − IV_merge`. A doc de `obn_jedi.R` admite "proxy greedy", mas o proxy não é monotônico em relação à perda real.
- **Implementação:** `iv_sum = bins[i].iv + bins[i+1].iv` e `min_iv_sum`.
- **Impacto:** dois bins puros adjacentes (0 eventos) têm IV artificial alto (~1,8 cada, por causa do piso EPS) e perda de merge zero (mesmo WoE). Nunca são fundidos, enquanto bins informativos de IV pequeno são engolidos. Caso testado: 1000 pontos, x<10 sem eventos, taxa crescente depois, `max_bins=4`:
  - JEDI mantém `(-Inf;4.9]` e `(4.9;9.9]`, ambos WoE −20,56 e n=50 cada.
  - Funde tudo entre 9,9 e 94,9 em um bin de 850 observações.
  - O bin informativo `(89.9;94.9]` (WoE 1,19) se perde.
  - O total_iv de 3,7 é dominado pelo artefato EPS (D-10).
- **Correção proposta** (JEDI numérico; no MWoE somar sobre k):
  ```cpp
  size_t find_min_iv_merge() const {
    double best=1e300; size_t idx=bins.size();
    int tp=0,tn=0; for(auto&b:bins){tp+=b.count_pos;tn+=b.count_neg;}
    auto ivf=[&](int p,int n){double a=p>0?(double)p/tp:EPS,b=n>0?(double)n/tn:EPS;return (a-b)*std::log(a/b);};
    for(size_t i=0;i+1<bins.size();++i){
      double loss=bins[i].iv+bins[i+1].iv-ivf(bins[i].count_pos+bins[i+1].count_pos,bins[i].count_neg+bins[i+1].count_neg);
      if(loss<best){best=loss;idx=i;}
    }
    return idx;
  }
  ```
- **Risco / testes:** muda a saída de JEDI. Teste: com dois bins puros adjacentes, o algoritmo com `max_bins` menor deve fundi-los primeiro. Comparar `total_iv` antes e depois.

##### D-03 — JEDI-MWoE numérico: monotonicidade imposta simultaneamente a todas as K classes
- **Algoritmo/arquivo:** src/OBN_JEDIMWoE_v5.cpp:481-505 (`enforce_monotonicity`), 536-556 (soma de IV sobre classes).
- **Severidade:** Alto. **Lente:** Teoria. **Status:** Confirmado (experimento).
- **Teoria:** a monotonicidade de scorecards vale para um alvo binário (Siddiqi). Com K≥3 e classes ordinais ou nominais, o WoE one-vs-rest da classe do meio é tipicamente unimodal em x (sobe e desce). Forçar a monotonia nele não tem justificativa teórica.
- **Implementação:** loop sobre k, funde o primeiro par violador de qualquer classe e recomeça.
- **Impacto:** K=3, x~N(0,1), N(1,5;1), N(3;1), 2000 cada, 30 pré-bins, `max_bins=8`, `min_bins=3`:
  - o último bin acumulou 4200 das 6000 observações (70%);
  - com `enforce_monotonicity` o IV total (soma sobre classes) foi 7,88, e sem a chamada, 9,86 (−20%).
  - Mesmo sem enforce, o redutor de D-02 também forma um bin de 4400 observações.
- **Correção:** tornar a monotonia opcional por classe (ex.: apenas para K=2, ou para uma classe de referência indicada pelo usuário). Para K≥3, desligar por padrão e documentar. Usar perda de IV (D-02) também aqui.
- **Testes:** K=3 ordinal; verificar que nenhum bin passa de ~40% das observações e que o IV total com K=3 não é pior que o do greedy sem monotonia.

##### D-04 — Monotonicidade "garantida" falha silenciosamente quando `min_bins` impede o merge (MOB-num, JEDI-num, DMIV-num)
- **Algoritmo/arquivo:** src/OBN_MOB_v5.cpp:340-372 (`optimize_bins`; `converged` só é FALSE por `max_iterations`), 408-458; src/OBN_JEDI_v5.cpp:393-410; src/OBN_DMIV_v5.cpp:457-640 (`enforce_monotonicity(true)`); R/obn_mob.R (tabela "MOB: Guaranteed").
- **Severidade:** Alto. **Lente:** Robustez/Teoria. **Status:** Confirmado.
- **Teoria:** a promessa de produto (obn_mob.R, "regulatory requirement") é WoE monotônico. Mironchyk e Tchistiakov tratam monotonia como restrição dura.
- **Implementação:** os loops param em `bins.size() > min_bins` sem sinalizar nada. `converged` permanece TRUE. A doc em `@return` admite "no warning", mas a tabela comparativa diz "Guaranteed".
- **Impacto** (U-shape: p=0,60 em x<0,2 ou x>0,8, p=0,10 no meio; N=3000):
  - MOB com `min_bins=3,max_bins=3`: 106 de 200 sementes retornaram WoE não monotônico (ex.: 0,94; 1,05; −0,13) com `converged=TRUE`.
  - DMIV com `is_monotonic=TRUE`: 52 de 100 sementes com WoE não monotônico (`min_bins=3, max_bins=3`).
  - Isso acontece com os defaults `min_bins=3` em qualquer variável em U.
- **Correção:**
  1. Adicionar `monotonic` (bool) ao resultado e `Rcpp::warning` quando a monotonia não for atingida.
  2. Opcionalmente, relaxar `min_bins` quando `monotonic_strict=TRUE`.
  ```cpp
  bool mono = is_monotonic_woe();          // após o último enforce
  if (!mono) Rcpp::warning("min_bins=%d impede WoE monotônico; resultado não monotônico.", min_bins);
  // ... Named("monotonic") = mono
  ```
- **Testes:** em dado em U com `min_bins=3`, `expect_warning` ou `expect_false(res$monotonic)`; com `min_bins=2`, `expect_true(all(diff(res$woe) >= -1e-10)) || all(diff(res$woe) <= 1e-10)`.

##### D-05 — OBC_MOB: `bin_cutoff` e categorias raras ignorados quando ncat ≤ max_bins
- **Algoritmo/arquivo:** src/OBC_MOB_v5.cpp:584-657 (ramo `ncat <= max_bins`); wrapper em 718-723 (`max_bins = min(max_bins, nº de categorias)`).
- **Severidade:** Alto. **Lente:** Teoria/Robustez. **Status:** Confirmado (leitura).
- **Teoria:** Siddiqi (2006, cap. 6): agrupar categorias raras antes de calcular WoE para estabilidade. A doc de `obc_mob.R` (linha 10) promete "Rare category handling: categories below bin_cutoff".
- **Implementação:** o wrapper limita `max_bins` ao número de categorias. Para variável de baixa cardinalidade (ex.: 4 categorias, default `max_bins=5`), `ncat > max_bins` é falso. O ramo cria uma bin por categoria sem chamar `handleRareCategories` e reporta `converged = isMonotonic(bins)`.
- **Impacto:** categoria com 3 observações em 10 000 vira bin própria com WoE ≈ ln(3,25/0,25 · N/P). O resultado é instável e ignora `bin_cutoff` e `min_bins`.
- **Correção:** sempre executar `calculateInitialBins()` (ou ao menos `handleRareCategories()`), mesmo quando ncat ≤ max_bins:
  ```cpp
  calculateInitialBins();  // inclui handleRareCategories e limitBins, para qualquer ncat
  ```
  respeitando `min_bins` (a rotina já faz `bins.size() <= min_bins → break`).
- **Testes:** feature com 4 categorias, uma com 0,1%: espera-se que ela seja fundida (`expect_true(all(res$count/sum(res$count) >= 0.05))` salvo min_bins).

##### D-06 — OBC_DMIV: `bin_cutoff` é parâmetro morto; `max_n_prebins` não é respeitado; O(B²)
- **Algoritmo/arquivo:** src/OBC_DMIV_v5.cpp:88 (`min_prebin_count = 5`), 140, 228-260; `perform_binning` 536-590; `initialize_neighbours` 465.
- **Severidade:** Alto. **Lente:** Teoria/Performance. **Status:** Confirmado (grep + experimento).
- **Teoria/doc:** R/obc_dmiv.R:20 documenta `bin_cutoff` como limiar de frequência para categorias raras.
- **Implementação:** `bin_cutoff` só é validado. O único pré-agrupamento é `cat_total >= 5` (constante fixa) e só quando `ncat > max_n_prebins`. O número de bins iniciais é livre (ex.: 14 371 pré-bins com `max_n_prebins=20`, B=16 000). O vizinho-mais-próximo é O(B²).
- **Impacto:** com 16 000 níveis, 3,7 s. Estimado para 64 000 níveis, ~1 min. Categorias com 5–50 observações influenciam as divergências de forma ruidosa e `bin_cutoff` não tem efeito algum.
- **Correção:** aplicar `bin_cutoff` (proporção) em `initialize_bins` (agrupar categorias com `count < bin_cutoff·N` em "other", como `OBC_IVB::merge_rare_categories`) e então `ensure_max_prebins`. Isso também limita o custo O(B²).
- **Testes:** `expect_lte(nrow(res), 20)` não se aplica aos bins finais. Testar que alterar `bin_cutoff` muda o resultado.

##### D-07 — OBC_MOB: rara-merge O(B²) e limitBins por menor |ΔWoE| (ignora massa)
- **Algoritmo/arquivo:** src/OBC_MOB_v5.cpp:247-300 (`handleRareCategories`, `findBestMergeCandidate` O(B) por categoria + `erase` + ajuste de índices), 533-558 (`limitBins`).
- **Severidade:** Médio. **Lente:** Performance/Teoria. **Status:** Confirmado.
- **Impacto de performance:** medido (extração da classe): B=1000 → 0,003 s; 4000 → 0,04 s; 16 000 → 0,63 s; 64 000 → 11,1 s (crescimento quadrático). Alta cardinalidade (CEP, CNAE, IDs) pode custar minutos.
- **Impacto estatístico:** `limitBins` funde o par adjacente (em ordem de WoE) de menor |ΔWoE|, sem considerar quantas observações há em cada bin. Em simulação (12 categorias, 300 sementes, 5 bins finais, WoE Bayes 0,5), a perda média de IV frente à DP ótima foi 0,48% e a pior, 6,3%. Efeito pequeno na média, mas nunca é ótimo.
- **Correção:** pré-agrupar raras via ordenação por taxa e varredura linear O(B log B), como em `OBC_IVB`; usar DP (O(B²·k) só sobre ≤ max_n_prebins) para o agrupamento final em vez de greedy.

##### D-08 — OBC_JEDIMWoE: "adjacência" definida pela ordem de contagem; ordem final arbitrária
- **Algoritmo/arquivo:** src/OBC_JEDIMWoE_v5.cpp:290-295 (stable_sort por `total_count`), 586-605 (`merge_adjacent_bins`), 671-684 (`ensure_monotonic_order`).
- **Severidade:** Médio. **Lente:** Teoria. **Status:** Confirmado (experimento).
- **Teoria:** para categórica nominal multiclasse não há ordem natural. O agrupamento deve ser por similaridade dos perfis de classe (qualquer par), não por adjacência em um vetor ordenado por frequência.
- **Implementação:** `merge_low_freq` deixa os bins ordenados por contagem crescente. O pré-bin por divergência (qualquer par, `merge_most_similar_bins`) alterna com `merge_adjacent_bins` (apenas vizinhos na ordem de contagem). Ao final, `ensure_monotonic_order` ordena por WoE de uma classe de cada vez, de modo que a ordem reportada é a da última classe não monotônica.
- **Impacto:** 3 classes, 9 categorias em 3 grupos com perfis [0,8;0,1;0,1], [0,1;0,8;0,1], [0,1;0,1;0,8], tamanhos 300,310,320 / 500,510,520 / 900,910,920, `min_bins=max_bins=3`. Resultado: {cat0,cat3,cat6}, {cat1,cat2,cat4,cat7}, {cat5,cat8}. A categoria 2 (perfil da classe 2) caiu no grupo da classe 1, e cat5/cat8 ficaram sem ela.
- **Correção:** em `merge_adjacent_bins`, avaliar a perda de IV de qualquer par (como o pré-bin já faz). Eliminar a pseudo-monotonicidade para K≥3 (ver D-03).

##### D-09 — DMIV: `woe` padrão ("woe1") é log-odds, não WoE; `iv` não corresponde à coluna `woe`
- **Algoritmo/arquivo:** src/OBN_DMIV_v5.cpp:373-380 e 777-790; src/OBC_DMIV_v5.cpp:~655-660; R/obn_dmiv.R:34, R/obc_dmiv.R.
- **Severidade:** Médio. **Lente:** Teoria. **Status:** Confirmado.
- **Teoria:** WoE_i = ln[(g_i/G)/(b_i/B)] (Siddiqi). "woe1" = ln(g_i/b_i) difere por constante ln(G/B). A doc cita Zeng, mas não pude verificar que essa seja a definição do paper.
- **Implementação:** padrão `bin_method="woe1"`, com `woe = ln((pos+0,5)/(neg+0,5))`. O `iv` é recalculado a partir das distribuições (corretamente), e portanto `iv_i ≠ (p_i−q_i)·woe_i`. `obwoe(algorithm="dmiv")` não passa `bin_method` e herda o padrão.
- **Impacto:** com evento raro (5%), o WoE reportado vem deslocado de ~−2,9 em todos os bins. O modelo logístico absorve a constante no intercepto, mas qualquer comparação/diagnóstico entre algoritmos fica inconsistente.
- **Correção:** padrão `"woe"` (alinha com os outros algoritmos), ou renomear a coluna para `log_odds` quando `woe1`. Documentar na seção de retorno que `iv` não é derivado de `woe` neste caso.

##### D-10 — Smoothing/clamp heterogêneos; IV total não comparável entre algoritmos; piso EPS sem clamp
- **Algoritmo/arquivo:** src/common/woe_iv_utils.h:44-86 (Laplace com 2α, Bayes), 109-125; OBN_JEDI_v5.cpp:374-377; OBN_JEDIMWoE_v5.cpp:~500; OBN_MOB_v5.cpp:508, 525 e 538 (α·k com k = nº atual de bins); categóricos via `CategoricalBin::calculate_metrics` (Bayes 0,5, bin_structures.h:225-237).
- **Severidade:** Médio. **Lente:** Numérica. **Status:** Confirmado.
- **Implementação, por família:**
  - OBN_JEDI / OBN_JEDIMWoE: sem smoothing, piso EPS=1e-10. Para share 5–30%, WoE ≈ −20,6…−23 (documentado em obn_jedi.R, sem clamp ±20).
  - OBN_MOB / OBN_DMIV: Laplace (c+α)/(T+kα), com k variando ao longo do algoritmo.
  - `woe_iv_utils.h` Laplace: (c+α)/(T+2α), que não soma 1 entre bins (soma (T+kα)/(T+2α) > 1 se k>2 e infla o IV). Bayes: (c+prior)/(T+α), também não normalizado.
  - Categóricos: Bayes α=0,5 com `BAYESIAN_PRIOR_STRENGTH` (R doc de `obc_ivb.R:98` diz α_total=1,0, mas a constante é 0,5).
  - O header `woe_iv_utils.h` não é incluído por nenhum algoritmo do escopo.
- **Impacto:** o mesmo conjunto de bins tem IV total diferente em cada algoritmo; o IV de JEDI com bin puro (3,7 no caso de D-02) é artefato. O `obwoe` usa IV para ranking de variáveis.
- **Correção:** um único backend de WoE/IV (Laplace com kα, normalizado) com clamp ±20 chamado por todos os algoritmos; documentar que o IV total depende do smoothing.

##### D-11 — Categóricos: `merge_low_freq` de JEDI e JEDI-MWoE mantém até `min_bins` categorias raras separadas; loop `optimize` de OBC_JEDI é código morto
- **Algoritmo/arquivo:** src/OBC_JEDI_v5.cpp:273-290, 483-545; src/OBC_JEDIMWoE_v5.cpp:~250-285.
- **Severidade:** Médio. **Lente:** Teoria/Robustez. **Status:** Confirmado (leitura).
- **Implementação 1:** `keep_rare = min(n_rare, min_bins)`: até `min_bins` (padrão 3) categorias abaixo de `bin_cutoff` permanecem como bins próprios.
- **Implementação 2 (OBC_JEDI):** `optimize()` guarda `best_bins = bins_` inicial e só atualiza se o IV *aumentar*, o que merges não fazem. No fim sempre restaura o estado inicial e chama `reduce_bins(max_bins, 1)`. O trabalho do loop é descartado. Dentro do loop, `merge_adjacent_bins_for_prebins` reduz até `max_n_prebins` (20) e não até `max_bins`. Então o IV não muda, o loop declara `converged_=true` e sai.
- **Impacto:** o resultado final é razoável (redução por perda de IV até `max_bins`), mas `bin_cutoff` não é garantido e `converged/iterations` não informam nada. `is_monotonic` olha só os dois primeiros bins para a direção (ordem já garantida por `ensure_monotonic_order`).
- **Correção:** remover `keep_rare`, ou documentar. Simplificar `optimize()` para um único `reduce_bins`.

##### D-12 — Merge de bins raros: critério por soma de IV (MOB), primeiro bin em ordem (JEDI), sempre com o anterior (DMIV)
- **Algoritmo/arquivo:** src/OBN_MOB_v5.cpp:383-385 (`iv_loss_left = IV_a + IV_b`, não é perda); src/OBN_JEDI_v5.cpp:333-355; src/OBN_DMIV_v5.cpp:304-340.
- **Severidade:** Médio. **Lente:** Teoria. **Status:** Confirmado (leitura).
- **Implementação:** a doc de obn_mob.R promete "direction chosen to minimize IV loss", mas o código soma os IVs sem subtrair o do bin fundido. JEDI escolhe o vizinho pela menor contagem. DMIV funde sempre com o anterior (e o primeiro com o próximo). Em todos, `bin_cutoff` deixa de ser garantido quando `bins.size() == min_bins`.
- **Correção:** usar a perda real `IV_a + IV_b − IV(a∪b)` para escolher o vizinho (MOB e JEDI).

##### D-13 — OBC_DMIV: medidas duplicadas ou equivalentes; critério ignora massa
- **Algoritmo/arquivo:** src/OBC_DMIV_v5.cpp:350-366 (KL e KLJ idênticos), 394-420 (L1/L2/LN).
- **Severidade:** Médio. **Lente:** Teoria. **Status:** Confirmado (derivação).
- **Implementação:**
  - `kl` e `klj` retornam a mesma expressão (kl12+kl21 = (p1−p2)ln(p1/p2) + (n1−n2)ln(n1/n2)). O comentário "seems different" é incorreto: a fórmula literal de Zeng é igual à implementada.
  - L1, L2 e L∞ usam a taxa local (r, 1−r): L1=2|Δr|, L2=√2|Δr|, L∞=|Δr|, todos equivalentes em ordenação e portanto dão o mesmo agrupamento.
  - A distância entre bins usa vetores (p, n) que não somam 1.
  - Nenhum critério pondera a massa dos bins (perda de IV).
- **Impacto:** das 9 opções documentadas, na prática há 6 ou 7 comportamentos distintos. Teste de L1/L2/L∞ vai dar resultados idênticos.
- **Correção:** documentar equivalências; para "kl" usar a KL unilateral ou removê-la. Avaliar o critério pelo IV perdido na fusão.

##### D-14 — Categorias não vistas, "NA" e o `apply`
- **Algoritmo/arquivo:** R wrappers (`feature[is.na(feature)] <- "NA"`), src/OB_ApplyWoECat.cpp:199-233.
- **Severidade:** Médio. **Lente:** Robustez. **Status:** Confirmado (leitura).
- **Implementação:** NA no treino vira a categoria "NA" (colide com uma categoria real "NA", sem aviso). Na aplicação, categoria nova → `Special`, WoE = `NA_REAL`. Todos os 7 categóricos incluem as categorias raras agrupadas no rótulo do bin (labels completos), então o `apply` as encontra. Não há bin "other" para categorias nunca vistas.
- **Impacto:** score `NA` para clientes com categorias novas, sem política de fallback (ex.: WoE do bin "other" ou 0).
- **Correção:** parâmetro `unseen = c("na", "zero", "other")` em `OBApplyWoECat`, e `Rcpp::warning` no fit quando existir categoria literal "NA".

##### D-15 — OBC_IVB: ordenação por taxa bruta × WoE suavizado, absorver excesso no menor bin, convergência vazia, doc α
- **Algoritmo/arquivo:** src/OBC_IVB_v5.cpp:403-427 (`ensure_max_prebins`), 429-436, 763-780 (loop de convergência), R/obc_ivb.R:98.
- **Severidade:** Baixo. **Lente:** Teoria. **Status:** Suspeito (item 1) / Confirmado (itens 2–4).
  1. Ordena por taxa de evento bruta, mas o IV usa WoE suavizado (Bayes 0,5). Bins pequenos têm WoE encolhido para a média e podem inverter a ordem em relação à taxa (ex.: 1/0 → taxa 1,0, WoE≈1,6; 90/10 → taxa 0,9, WoE≈2,2 para totais ~1000). Só afeta categorias minúsculas; o comentário diz que `enforce_monotonicity` nunca foi acionado nas simulações.
  2. `ensure_max_prebins` funde todo o excesso na menor categoria retida, sem olhar a taxa de evento.
  3. O loop de "convergência" é vazio (dp constante) e dá `converged=FALSE` se `max_iterations=1`.
  4. A doc diz α_total = 1,0; o código usa 0,5.
- **Correção:** ordenar pelo WoE suavizado; agrupar excesso por taxa; remover o loop; corrigir a doc.

##### D-16 — Headers (`monotonicity_utils.h`): tendência por Pearson, NaN → DESCENDING, slope sem pesos, merge por menor |ΔWoE|
- **Algoritmo/arquivo:** src/common/monotonicity_utils.h:38-100, 120-150, 232-244.
- **Severidade:** Baixo. **Lente:** Robustez. **Status:** Confirmado (leitura; não usados pelos algoritmos do escopo — só OBN_DP/OBC_DP/OBN_MDLP).
- **Implementação:**
  - `detect_trend_from_correlation` usa correlação de Pearson do valor bruto, sensível a outliers. NaN no feature → correlação NaN → `(NaN >= 0)` falso → DESCENDING (já contornado no chamador `OBN_DP_v5.cpp:556-570`).
  - Em relação U, correlação ≈ 0 dá ASCENDING arbitrário.
  - `detect_trend_welford_woe` é um OLS não ponderado sobre o índice do bin.
  - `find_smallest_woe_diff` ignora massa.
- **Correção:** usar Spearman ou slope ponderado por contagem; filtrar não finitos dentro do header.

##### D-17 — Menores (Baixo)
- `OBN_MOB`: `convergence_threshold` não é usado (src/OBN_MOB_v5.cpp:72,79,176,198); `calculate_bin_woe` usa `bins.size()` como k na avaliação de candidatos de merge (k em vez de k−1; lines 508, 525), um desvio pequeno de smoothing na perda de IV.
- `OBC_MOB`: `converged = monotonic` (linha 609-616) na verdade indica "variação de IV < tolerância"; `best_bins` guarda o estado pré-enforce (IV só cai com merges).
- JEDI-num/JEDI-MWoE: o pré-bin usa quantis dos valores *distintos* (documentado), não das observações. Em variáveis com massa em poucos valores e cauda longa, quase todas as bordas caem na cauda (bins pequenos, depois fundidos por `merge_small_bins`).
- `OBN_JEDI`/`OBN_JEDIMWoE`: `min_bins=max(min_b,2)` com `max_bins=max(max_b,min_b)` permite `max_bins < min_bins` se o usuário passa `min_bins=1`.
- OBC_JEDIMWoE: Laplace com 2α no denominador (`mwoe_iv`, linhas 56-64), não normalizado entre bins.
- C++ `OBN_DMIV` aceita `bin_cutoff` ∈ [0,1], a doc diz (0,1).
- Comentário `// target == 1 as "good"` em OBC_MOB:~42 contradiz Siddiqi (good = não evento).

#### Problemas transversais observados

1. **Monotonicidade em categórico é trivial por construção.** As categorias são ordenadas por taxa/WoE e agrupadas em blocos contíguos, o que dá WoE monotônico (salvo artefatos de smoothing em bins minúsculos). A função `enforceMonotonicity` de OBC_MOB, os `is_monotonic` de OBC_JEDI e OBC_JEDI-MWoE são, na prática, no-ops ou apenas reordenam. O que determina a qualidade é o critério de agrupamento (DP ótima em IVB; greedy nos demais) e o tratamento de raras. Para MWoE multiclasse, a ordenação por uma classe não faz sentido para as demais (D-03/D-08).
2. **Convenção de sinal.** Consistente: ln(evento/não-evento). Documentada em obwoe.R:79 e :2136. Não segue Siddiqi. Risco só de interpretação (WoE alto = risco alto), e o código de gains table (obwoe.R:2577) já comenta isso.
3. **Smoothing e IV total.** Ver D-10. O ideal é um backend único, e documentar que o IV total muda com a suavização.
4. **Estado de "converged".** Significa coisas diferentes em cada algoritmo (D-04, D-11, D-15, D-17).
5. **Parâmetros mortos ou parcialmente usados:** `convergence_threshold` (MOB-num, DMIV-num), `bin_cutoff` (OBC_DMIV), `max_n_prebins` (OBC_DMIV).
6. **Positivos:** `OBN_MOB` produz IV quase ótimo; `OBC_IVB` usa DP exata; o fuzz com ASan/UBSan (MOB-num e DMIV-num, 4000 execuções cada) foi limpo; contagens somam n; cutpoints finitos e crescentes.

#### Priorização recomendada (top 5 do escopo)

1. **D-01** — DMIV numérico: trocar o redutor por DP ou greedy de perda real (IV 30–50% abaixo do MOB hoje).
2. **D-04** — Sinalizar (`monotonic`, `warning`) a falha de monotonicidade por `min_bins` em MOB/JEDI/DMIV numéricos; corrigir a doc "Guaranteed".
3. **D-02 + D-03** — JEDI/JEDI-MWoE numéricos: merge por perda de IV; desligar monotonia multi-classe para K≥3.
4. **D-05 + D-06** — Aplicar `bin_cutoff` em OBC_MOB (ramo ncat ≤ max_bins) e OBC_DMIV (parâmetro morto, `max_n_prebins` não respeitado).
5. **D-10** — Unificar smoothing/clamp (backend único) e corrigir inconsistências de documentação (D-09, D-15).

---

## Anexo E — Agente E — C++ científico / numérico / HPC


Escopo auditado (somente leitura): `OBN_Sketch_v5.cpp`, `OBC_Sketch_v5.cpp`, `OBC_GMB_v5.cpp`, `OBC_MBA_v5.cpp`, `OBC_SWB_v5.cpp`, `OBN_UBSD_v5.cpp`, `OBN_LDB_v5.cpp`, headers `src/common/*.h`, `OB_ApplyWoENum.cpp`, `OB_ApplyWoECat.cpp`, `OB_Cutpoints.cpp`, wrappers R correspondentes e `src/Makevars`.
Experimentos (standalone, g++ 13, ASAN/UBSAN onde indicado) em `scratchpad/work_E/` (arquivos citados em cada achado).

#### Resumo executivo

- Não encontrei **nenhum Crítico** (crash/UB em uso normal) no escopo. O código já passou por várias rodadas de correção (comentários "used to…" por todo lado): índices `size_t`, `erase` em loop, `front()/back()` e NaN em `static_cast<size_t>` estão em geral protegidos. Extratos rodados sob ASAN/UBSAN (KLL, DP, `greedy_similarity_merge` do SWB, KDE) ficaram limpos.
- O problema dominante é **fidelidade/qualidade estatística**, não UB: (i) `OBN_Sketch` escolhe cutpoints por ranking guloso de IV de corte único (cortes agrupados; IV 29% menor que o DP sobre os mesmos candidatos); (ii) `OBC_MBA`/`OBC_Sketch` aplicam `bin_cutoff` também sobre a *parcela de positivos*, o que não está documentado e funde bins grandes de baixo risco; (iii) `OBC_GMB` dobra categorias excedentes no bin "menor" ignorando taxa de evento (−22% de IV num caso de brinquedo).
- Há um bug de índice residual confirmado em `OBC_Sketch::enforce_bin_cutoff` (o mesmo que já foi corrigido em `OBC_MBA`), que deixa bins abaixo do cutoff.
- O "sketch" não é sketch: `OBN_Sketch` mantém todos os dados em memória e usa o KLL só para 37 quantis (que um `nth_element` daria exatos); `OBC_Sketch` agrega contagens exatas por categoria *antes* de construir o Count-Min, e as tabelas `sketch_pos/neg` nunca são lidas. A garantia de erro do KLL é determinística/heurística (não é KLL de Karnin et al.), mas o erro medido (~1–2,2/k) é coerente com o que a documentação admite.
- OpenMP: usado **somente** em `OB_Correlation.cpp` (região paralela sem race condition; escreve em slots distintos, nada do R API). Nenhum dos algoritmos de binning usa OpenMP; as flags em `Makevars` apenas o habilitam globalmente.
- Transversal: epsilons inconsistentes (1e-10 / 1e-14 / 1e-4 / Laplace / Bayes), definições de WoE/IV diferentes entre algoritmos e entre `fit` e `binning_*_cutpoints`, zero `checkUserInterrupt` nos arquivos do escopo, default silencioso `-999` como missing em `ob_apply_woe_num`.

#### Tabela de conformidade por algoritmo

| Algoritmo | Arquivo | Referência teórica | Fidelidade | Robustez | Performance (atual → ideal) | Estab. numérica | Nº achados (C/A/M/B) |
|---|---|---|---|---|---|---|---|
| Sketch numérico (KLL) | `OBN_Sketch_v5.cpp` | Karnin-Lang-Liberty 2016; (cita GK 2001) | **Parcial/Baixa**: compactação determinística sem capacidade geométrica nem coin aleatório; seleção de cortes gulosa | Boa (Inf/NaN/ties tratados) | O(n log k + n·C) → ok (mas não é streaming de fato: O(n) memória) | Média (clamp EPS no DP favorece bins pontuais) | 0/1/1/2 |
| Sketch categórico (Count-Min) | `OBC_Sketch_v5.cpp` | Cormode-Muthukrishnan 2005 | **Baixa**: CMS decorativo (contagens exatas pré-calculadas) | 1 bug de índice confirmado | O(B³) no pior caso (B=max_n_prebins) → ok p/ B≤20 | Boa | 0/0/2/3 |
| GMB | `OBC_GMB_v5.cpp` | Greedy merge de IV (Zeng 2014 / Mironchyk 2017) | Parcial | Boa; "" lança erro | O(k³) com k≤20 → ok | Boa | 0/1/0/2 |
| MBA | `OBC_MBA_v5.cpp` | Mironchyk & Tchistiakov 2017 (monotone) | Parcial (monotonia é trivial: bins reordenados por WoE) | Boa; "" lança erro | pré-binning O(B²) (B=nº categorias) → O(B log B) | Boa | 0/1/0/2 |
| SWB | `OBC_SWB_v5.cpp` | Sem paper; doc diz "sliding window" | **Baixa** (não há janela deslizante; JS entre todos os pares) | Boa | `greedy_similarity_merge` equivalente ao scan completo (testado) | Boa | 0/0/0/2 |
| UBSD | `OBN_UBSD_v5.cpp` | Média ± k·σ + equal-width (heurística; sem paper) | Alta (ao que documenta) | Boa | O(n + k²·it) → ok | Boa (WoE/IV coerentes com Laplace) | 0/0/0/2 |
| LDB | `OBN_LDB_v5.cpp` + `optimal_binning_common.h` (KDE) | Silverman 1986 (bw); KDE por linear binning (Silverman 1982 / R `density`) | Parcial | Boa; densidade some com outliers | O(n log n + G²) → ok | **Média**: IV mistura dist. bruta com WoE suavizado | 0/0/2/2 |
| Headers comuns | `src/common/*.h` | — | — | `cutpoints_validator.h`, `validation_utils.h` sem uso | — | EPS absolutos dependentes de escala | 0/0/0/3 |
| Apply num/cat | `OB_ApplyWoENum/Cat.cpp` | — | Consistente com treino (right-closed, `lower_bound`) | Boa | O(n log k) / O(n) → ótimo | n/a | 0/0/1/2 |
| Cutpoints | `OB_Cutpoints.cpp` | — | WoE não suavizado c/ piso 1e-4 | Boa | O(n log k) | **Média** (piso só se contagem == 0) | 0/0/1/1 |


Contagem total: **Críticos 0 · Altos 3 (E-01, E-03, E-04) · Médios 6 (E-02, E-05..E-09) · Baixos 12 (E-10..E-21)**.

---

#### Achados

##### E-01 — OBN_Sketch (n > 50): cortes escolhidos por ranking guloso de IV de corte único → cutpoints agrupados, IV muito abaixo do ótimo
- **Algoritmo/arquivo:** `src/OBN_Sketch_v5.cpp:455-473` (`select_optimal_cutpoints`, ramo `else`), `389-440` (`calculate_cutpoint_stats`)
- **Severidade:** Alto
- **Lente:** Teoria
- **Status:** Confirmado (experimento `work_E/greedy_sel.cpp`)
- **Teoria:** "Optimal binning" maximiza IV total sob restrições (Mironchyk & Tchistiakov 2017; a própria doc R fala em DP para a seleção). O IV de uma partição em k bins **não** é separável por corte: cortes vizinhos têm IV de corte único quase idêntico e são redundantes.
- **Implementação:** cada candidato recebe o IV de *um* corte (`|iv_below|+|iv_above|`), os candidatos são ordenados por IV decrescente e os `max_bins-1` primeiros são usados:
  ```cpp
  std::sort(stats.begin(), stats.end(), [](a,b){ return a.iv > b.iv; });
  const size_t n_take = std::min(stats.size(), (size_t)(max_bins - 1));
  for (i<n_take) cutpoints.push_back(stats[i].cutpoint);
  ```
- **Impacto / cenário de falha:** n=20 000, x~N(0,1), P(y)=σ(1.5x−1), 37 candidatos, max_bins=5. Cortes escolhidos: `0.001 0.127 0.257 0.385` (quatro cortes adjacentes de 5%) → contagens `10000 1000 1001 1000 6999`, IV=1.134. DP exato sobre os mesmos candidatos: cortes `1.467 0.670 0.001 −0.838`, contagens `4001 5999 5001 3600 1399`, **IV=1.461 (+29%)**. Cenário "step" e "bump": IV 0.236 vs 0.273 (+15%). O DP exato só roda para n ≤ 50 (`dp_size_limit`), ou seja, **nunca no uso real**. Os bins minúsculos (1000/20000 = 5% = exatamente o cutoff) ainda disparam o `enforce_bin_cutoff` quando o cutoff é maior.
- **Correção proposta:** rodar o DP já existente sobre os C≤37 candidatos usando contagens por *slot* (as contagens já são calculadas em O(n log C)); custo O(K·C²) desprezível. Esboço:
  ```cpp
  // slot_pos/slot_neg: contagens por intervalo entre candidatos (já existem em calculate_cutpoint_stats)
  std::vector<long> pp(C+2,0), pn(C+2,0);
  for (size_t b=0;b<=C;++b){ pp[b+1]=pp[b]+slot_pos[b]; pn[b+1]=pn[b]+slot_neg[b]; }
  const long min_cnt = (long)std::ceil(bin_cutoff * feature.size());
  auto bin_iv = [&](size_t l,size_t r)->double{            // slots [l, r)
    long p=pp[r]-pp[l], q=pn[r]-pn[l];
    if (p+q < min_cnt) return -1e18;                        // suporte mínimo embutido no DP
    double pe=(p+0.5*tp/(tp+tn))/(tp+0.5), pn_=(q+0.5*tn/(tp+tn))/(tn+0.5);
    return (pe-pn_)*std::log(pe/pn_);                       // mesma suavização de NumericalBin
  };
  // D[k][r] = max IV com k bins cobrindo slots [0,r); recuperar cortes por backpointer
  ```
- **Risco / testes:** muda resultados de todos os usos (esperado: IV ≥ antes). Teste: `ob_numerical_sketch(x,y,max_bins=5)$total_iv >= (valor antigo)` em N(0,1)+logística; `all(res$count >= ceiling(0.05*n))`.

##### E-02 — OBC_Sketch: índices obsoletos em `enforce_bin_cutoff` deixam bins abaixo de `bin_cutoff`
- **Algoritmo/arquivo:** `src/OBC_Sketch_v5.cpp:500-550` (atualização em 543-547)
- **Severidade:** Médio
- **Lente:** Robustez
- **Status:** Confirmado (cópia literal do laço em `work_E/sk_cutoff.cpp`, `sk_cutoff2.cpp`; ASAN/UBSAN limpo)
- **Teoria:** n/a (invariante do algoritmo: ao fim, todo bin ≥ `bin_cutoff` quando `bins.size() > min_bins`).
- **Implementação:** `try_merge_bins(min(idx,m), max(idx,m))` apaga o índice **maior**, mas depois só decrementa `r > merge_idx`:
  ```cpp
  for (auto &remaining_idx : low_freq_bins) { if (remaining_idx > merge_idx) remaining_idx--; }
  ```
  Quando o parceiro `m > idx` também está na lista (`r == m`), a entrada continua apontando para `m`, que agora é o bin seguinte; o bin fundido (índice `idx`) não é reavaliado. O mesmo defeito foi corrigido em `OBC_MBA_v5.cpp:289-297` (`if (r==erased) r=kept; else if (r>erased) r--;`), mas não aqui.
- **Impacto / cenário de falha:** instância mínima (n=1112, min_count=56): bins `A(n=9) B(381) C(268) D(453) E(n=1)`; original → `[AE n=10] B C D` (bin de 10 obs. mantido com 4 bins > min_bins=2); corrigido → `[AEC n=278] B D`. Alcançável quando ≥2 bins "heavy hitters" ficam entre `cutoff/2` e `cutoff` (ou abaixo do limiar de positivos de E-03). Em 20 000 sorteios aleatórios de bins, o original deixou mais bins sub-cutoff que a versão corrigida em 88% dos casos (instâncias sintéticas, sem passar pelo pré-binning, portanto não é frequência de campo).
- **Correção proposta:**
  ```cpp
  const size_t kept = std::min(idx, merge_idx), erased = std::max(idx, merge_idx);
  if (!try_merge_bins(kept, erased)) continue;
  for (auto &r : low_freq_bins) { if (r == erased) r = kept; else if (r > erased) --r; }
  ```
- **Risco / testes:** baixo. testthat: simular 200 bases com 15 categorias (algumas com 3–4,9%) e verificar `all(res$count >= ceiling(cutoff*N)) || nrow(res)==min_bins`.

##### E-03 — OBC_MBA e OBC_Sketch: `bin_cutoff` também é aplicado à parcela de **positivos** (não documentado) → bins grandes de baixo risco são fundidos
- **Algoritmo/arquivo:** `src/OBC_MBA_v5.cpp:233-243, 259-261`; `src/OBC_Sketch_v5.cpp:502-511, 520-522`
- **Severidade:** Alto
- **Lente:** Teoria
- **Status:** Confirmado (leitura do código; aritmética direta). Os dois únicos arquivos do pacote com esse padrão (`grep min_count_pos`).
- **Teoria:** a restrição de frequência mínima (Mironchyk & Tchistiakov 2017; Siddiqi 2006) é sobre a fração de *observações* por bin. A doc R (`R/obc_mba.R`, `@param bin_cutoff`) diz "Minimum proportion of total observations required for a category to remain separate".
- **Implementação:**
  ```cpp
  int min_count     = ceil(bin_cutoff * feature.size());
  int min_count_pos = ceil(bin_cutoff * total_bad);
  if (bins[i].count < min_count || bins[i].count_pos < min_count_pos) low_freq_bins.push_back(i);
  ```
- **Impacto / cenário de falha:** total_bad=1000, N=20 000, cutoff 5%: um bin com 10 000 obs. e 40 eventos (taxa 0,4%, WoE fortemente negativo — o bin mais valioso de um scorecard) tem `count_pos=40 < 50` → é fundido com o bin de taxa mais próxima, destruindo IV. Bins puramente negativos (0 eventos) são *sempre* fundidos. Quanto menor a taxa de eventos global, maior o efeito.
- **Correção proposta:** remover o termo de positivos (ou torná-lo parâmetro opt-in, ex. `min_events`), mantendo apenas `bins[i].count < min_count`.
- **Risco / testes:** mudará resultados de MBA/Sketch categóricos. Teste: base com categoria de 50% das linhas e 0,2% de eventos deve permanecer como bin próprio (`"LOW" %in% res$bin`).

##### E-04 — OBC_GMB: excedente de pré-bins é "dobrado" no bin menor ignorando taxa de evento
- **Algoritmo/arquivo:** `src/OBC_GMB_v5.cpp:285-305`
- **Severidade:** Alto (condicional: requer > `max_n_prebins` bins após `mergeRare`, i.e. `bin_cutoff` < 1/`max_n_prebins` ou `max_n_prebins` pequeno)
- **Lente:** Teoria / Robustez
- **Status:** Confirmado por experimento numérico (script Python inline; contas reproduzíveis a partir do código)
- **Teoria:** reduzir pré-bins num binning supervisionado deve fundir bins **adjacentes na ordenação de risco** com menor perda de IV (Zeng 2014; é exatamente o que o próprio `greedyMerge` faz depois).
- **Implementação:** ordena por contagem decrescente, mantém os `max_n_prebins` maiores e funde **todos** os demais no `bins[keep-1]` (o menor dos mantidos), sem olhar taxa:
  ```cpp
  CategoricalBin& absorber = bins[keep - 1];
  for (size_t i = keep; i < n_bins; ++i) absorber.merge_with(bins[i]);
  ```
- **Impacto / cenário de falha:** 60 categorias de 1 000 obs., taxas de evento lineares de 2% a 60%, `bin_cutoff=0.01`, `max_n_prebins=20`: com contagens iguais o `stable_sort` mantém as 20 de menor taxa e as 40 de maior taxa (2/3 dos dados) viram **um** bin. IV: 60 categorias = 0.795; após o dobramento (20 bins) = **0.615 (−22%)**; agrupamento adjacente em 20 bins = 0.789. A perda é silenciosa e anterior ao `greedyMerge`.
- **Correção proposta:** substituir o bloco por fusões gulosas de pares adjacentes (já existe `mergeScore`/`contributionPrefix`) até `bins.size() == max_n_prebins`; custo O(B²) para B pequeno, ou O(B log B) com heap se B for grande.
  ```cpp
  while ((int)bins.size() > max_n_prebins) {      // bins já ordenados por taxa de evento
    const auto prefix = contributionPrefix();
    size_t best = 0; double best_s = NEG_INFINITY;
    for (size_t i=0;i+1<bins.size();++i){ double s = mergeScore(prefix,i); if (s>best_s){best_s=s;best=i;} }
    bins[best].merge_with(bins[best+1]); bins[best].calculate_metrics(total_pos,total_neg);
    bins.erase(bins.begin()+best+1);
  }
  ```
  (os `calculate_metrics` dos bins devem existir antes; mover o loop de métricas para cima.)
- **Risco / testes:** testthat com o cenário acima: `res$total_iv` ≥ 0.9·IV(60 categorias) com `max_bins=5`.

##### E-05 — OBN_Sketch, ramo DP (n ≤ 50): objetivo Σ|IV| com clamp EPS favorece bins de 1–2 observações
- **Algoritmo/arquivo:** `src/OBN_Sketch_v5.cpp:194-205, 237-299`
- **Severidade:** Médio
- **Lente:** Numérica / Teoria
- **Status:** Confirmado (`work_E/dp_test.cpp`, ASAN/UBSAN limpo)
- **Teoria:** IV não suavizado cresce sem limite quando um bin isola uma classe (`log(1e-10/p)` ≈ −23). Maximizar IV sem suporte mínimo é overfitting trivial.
- **Implementação:** `prop_event = max(prop_event, EPSILON)`; o DP ignora `bin_cutoff`; só depois `enforce_bin_cutoff` junta bins.
- **Impacto:** 40 obs., sinal fraco, max_bins=5: tamanhos de bins escolhidos `1 8 2 26 3`, `30 2 4 1 3`, `2 5 22 8 3`, `11 1 4 2 22`. As fusões posteriores (vizinho de taxa mais próxima) produzem uma partição arbitrária, não a ótima sob o cutoff.
- **Correção proposta:** impor `count ≥ ceil(bin_cutoff·n)` e usar IV suavizado (mesma fórmula de `NumericalBin::calculate_metrics`) dentro de `calculate_bin_iv` (retornar `-INF`/inviável quando o bin é pequeno); é o mesmo molde do snippet de E-01.
- **Risco / testes:** n pequeno é raro; testar `n=30` com `bin_cutoff=0.1` → todos os bins ≥ 3 obs.

##### E-06 — OBN_LDB: IV calculado com distribuição **bruta** × WoE **suavizado** (pode ser negativo; total_iv ≠ Σ(p−q)·ln(p/q))
- **Algoritmo/arquivo:** `src/OBN_LDB_v5.cpp:118-157` (`calculateWOE` suaviza com K·α no denominador; `calculateIV` usa `pos/total_pos` bruto)
- **Severidade:** Médio
- **Lente:** Numérica
- **Status:** Confirmado (aritmética: P=10, N=1000, K=3, bin com 0 eventos e 20 não-eventos → WoE=+0.753, contribuição de IV = **−0.0151**)
- **Teoria:** IV = Σ(p_i − q_i)·WoE_i com p,q as *mesmas* distribuições usadas no WoE (Siddiqi 2006); é ≥ 0 por construção.
- **Implementação:** `good=(pos+α)/(total_pos+K·α)` no WoE, mas `dist_good = pos/total_pos` no IV.
- **Impacto:** em dados muito desbalanceados (fraude, ≤1% eventos) com cutoff pequeno, bins podem ter IV negativo e `total_iv` subestimado; o merge de max_bins usa `iv_loss` com essas quantidades, distorcendo a escolha. Também diverge de `UBSD` (que usa p,q suavizados coerentes).
- **Correção proposta:** retornar `(good_s - bad_s) * woe` com `good_s,bad_s` suavizados (ou usar `NumericalBin::calculate_metrics`).
- **Risco / testes:** `all(res$iv >= 0)` para base com 1% de eventos e `bin_cutoff=0.01`.

##### E-07 — OBN_LDB: piso de bandwidth `range/1000` + grade fixa de 512 pontos fazem a densidade "sumir" com outliers extremos
- **Algoritmo/arquivo:** `src/OBN_LDB_v5.cpp:440-447`; `src/common/optimal_binning_common.h:121-195`
- **Severidade:** Médio
- **Lente:** Robustez
- **Status:** Confirmado (`work_E/kde.cpp`)
- **Teoria:** Silverman (1986, §3.4.2): h = 0.9·min(σ, IQR/1.34)·n^(−1/5) — a forma está correta. Mas a KDE por linear binning (Silverman 1982; R `density`) exige que `h` seja resolvido pela grade (h ≳ 2·dx); R estende o range por 3h e avisa.
- **Implementação:** `h = max(h, range/1000)` e grade `G=512` sobre `[min,max]`.
- **Impacto:** amostra bimodal bem separada (±3, n=5000). Mínimos locais de densidade encontrados: sem outlier **1**; outlier=50 → 1; outlier=1e3 → 1 (h sobe a 1.01); outlier=**1e5 → 0** (dx=196, h=100: todo o bulk cai em 1–2 células da grade). O algoritmo cai silenciosamente nos quantis (`min_bins`), perdendo a razão de ser do LDB ("local density").
- **Correção proposta:** calcular a KDE no intervalo robusto `[Q1−k·IQR, Q3+k·IQR]` (ou winsorizar para a grade) e tratar caudas separadamente; ou subir G proporcionalmente a `range/h` (G = min(2^20, ceil(8·range/h)) com convolução por FFT/ou truncando o kernel em 4h, O(n·G_local)).
- **Risco / testes:** `x <- c(rnorm(5000,-3,.8), rnorm(5000,3,.8), 1e5)`; esperar um corte entre −1 e 1.

##### E-08 — OB_Cutpoints: piso 1e-4 só quando a contagem é exatamente 0 → WoE não monótono na contagem
- **Algoritmo/arquivo:** `src/OB_Cutpoints.cpp:16-29`
- **Severidade:** Médio
- **Lente:** Numérica
- **Status:** Confirmado (aritmética)
- **Teoria:** WoE deve ser monótono na taxa de eventos do bin (Siddiqi 2006); suavização/clamp precisa preservar a ordem.
- **Implementação:** `pos_rate = count_pos/total_pos; if (pos_rate == 0) pos_rate = 0.0001;`
- **Impacto:** total_pos=total_neg=1e6, bin com 1000 não-eventos: 0 eventos → WoE=−2.303; **1 evento → −6.908**; 50 eventos → −2.996; 100 eventos → −2.303. O bin com *mais* eventos recebe WoE maior (correto) só a partir de ≥100; abaixo disso, 0 eventos é "melhor" que 1. O piso 1e-4 é absoluto (escala: = 100 eventos em 1e6) e inconsistente com o resto do pacote (EPSILON 1e-10, Laplace, Bayes).
- **Correção proposta:** usar `std::max(rate, EPSILON)` ou, melhor, a mesma suavização de `NumericalBin::calculate_metrics`.
- **Risco / testes:** `binning_numerical_cutpoints` vs WoE do `fit` sobre os mesmos cortes devem coincidir.

##### E-09 — `ob_apply_woe_num`: default `missing_values = c(-999)` trata o sentinela como NA, enquanto nenhum binner do escopo o trata como tal
- **Algoritmo/arquivo:** `R/ob_apply_woe_num.R:113`; `src/OB_ApplyWoENum.cpp:106-113, 179-196`
- **Severidade:** Médio
- **Lente:** Robustez
- **Status:** Confirmado (grep: `-999`/`999` não aparece em nenhum `OB*_v5.cpp`)
- **Teoria:** treino e aplicação devem tratar o mesmo valor da mesma forma.
- **Implementação:** `x == -999` ⇒ `ismissing=1`, WoE do bin "NA" ou `NA`/"Special".
- **Impacto:** se `-999` é um valor legítimo (ou o sentinela foi binado como número no treino, caindo no primeiro bin), o apply o reclassifica silenciosamente → WoE do treino ≠ WoE do apply. (`obwoe()` não chama essa função; vale para uso standalone.)
- **Correção proposta:** default `NULL`/`numeric(0)` (só NA/NaN) e sentinela explícito; documentar.

##### E-10 — OBC_MBA e OBC_Sketch: pré-binning quadrático / O(B³)
- **Algoritmo/arquivo:** `src/OBC_MBA_v5.cpp:158-205`; `src/OBC_Sketch_v5.cpp:590-612`
- **Severidade:** Baixo
- **Lente:** Performance
- **Status:** Confirmado (`work_E/mba_perf.cpp`)
- **Implementação:** para cada categoria excedente, varre todos os bins (similaridade `1/(|Δtaxa|+ε)`) e faz `vector::erase` (O(B)). Medido: B=10k → 0.16 s; 20k → 0.63 s; 40k → 2.56 s (×4 por dobro). Com 500k categorias (IDs) → ~7 min só nesta etapa. `OBC_Sketch::optimize_bins` é O(B²) por fusão (O(B³)) com B=`max_n_prebins`; só dói se o usuário elevar `max_n_prebins` para milhares.
- **Correção proposta:** ordenar por taxa e fundir adjacentes com lista ligada + heap (O(B log B)), ou limitar a busca ao vizinho em ordem de taxa; `std::list`/índices `prev/next`.

##### E-11 — OBC_Sketch: o Count-Min é decorativo, tem rows correlacionadas e `check_consistency` é código morto
- **Algoritmo/arquivo:** `src/OBC_Sketch_v5.cpp:66-77, 141-215, 292-305, 631-668, 906-989`
- **Severidade:** Baixo
- **Lente:** Teoria
- **Status:** Confirmado (`work_E/cm.cpp` para o hash)
- **Teoria:** Cormode & Muthukrishnan (2005): erro ≤ εN com prob. ≥ 1−δ exige hashes 2-universais independentes por linha; ε=e/w, δ=e^(−d).
- **Implementação:** (a) `aggregate` exato por categoria é feito antes (`str_index`), logo o CMS apenas gera falsos positivos de "heavy hitter"; (b) `sketch_pos` e `sketch_neg` são atualizados e nunca lidos; (c) todas as linhas usam FNV-1a com `OFFSET ^ seed` — não independentes: 5000 chaves, w=2000, d=5, pares que colidem em 2 linhas: **132–432 observados vs 31 esperados** (4–14×); (d) `check_consistency()` compara contagens exatas com elas mesmas (nunca avisa).
- **Impacto:** nenhum numérico (limiar bin_cutoff/2 ≫ εN), mas custo/complexidade sem benefício e uma doc ("single-pass, sublinear") que não corresponde ao código (vetores completos já em memória).
- **Correção proposta:** ou remover o CMS (usar a contagem exata) ou aplicá-lo de fato em streaming; se mantido, dupla-hash `h_i = (a_i·h + b_i) mod p mod w` com `a_i,b_i` do mt19937 (seed fixa); apagar `sketch_pos/neg`.

##### E-12 — OBC_Sketch::optimize_bins alterna critério (divergência ↔ |IV|) a cada 5 iterações
- **Algoritmo/arquivo:** `src/OBC_Sketch_v5.cpp:632-668` (`use_divergence = !use_divergence`)
- **Severidade:** Baixo
- **Lente:** Teoria
- **Status:** Confirmado (leitura)
- **Impacto:** a partição final depende do nº de fusões (e do estado persistente do membro entre chamadas do laço externo); `|iv_i|+|iv_j|` (soma de IVs, sem olhar similaridade) não é critério de fusão em nenhuma referência. Determinístico, mas não justificável.
- **Correção proposta:** um único critério (perda de IV por fusão, ou JS), sem alternância.

##### E-13 — OBN_Sketch: "KLL" não é KLL; a garantia ε≈1/k vale ≈1–2,2/k na prática; estrutura é inútil (n em memória)
- **Algoritmo/arquivo:** `src/OBN_Sketch_v5.cpp:44-161, 349-383`; `R/obn_sketch.R` (doc "Provable approximation guarantees")
- **Severidade:** Baixo
- **Lente:** Teoria
- **Status:** Confirmado (`work_E/kll_test.cpp`, ASAN/UBSAN limpo)
- **Teoria:** KLL (Karnin-Lang-Liberty 2016): capacidades geométricas por nível (k·c^(H−h)), coin aleatório por compactação, erro ε≈O(1/k) com prob. ≥1−δ e espaço O(k).
- **Implementação:** capacidade `k` constante em todos os níveis, compactação par/ímpar determinística alternando lower/upper por nível, `weight` int. Espaço O(k log(n/k)); erro de rank pior caso O(n·log(n/k)/k) (a doc admite).
- **Medido (erro máximo de rank / n, quantis 0.01..0.99; k=50/200/1000):** uniforme 0.023–0.029 / 0.0037–0.0053 / 0.0008–0.0012; serrote (`i%1000`) até **0.034 / 0.0109 / 0.0007** = 1.7/k, 2.2/k, 0.7/k. Ou seja, ≈1/k em média, até ~2,2/k em padrões periódicos, sem crescimento visível com n até 5·10^6. Viés sistemático negativo em alguns casos (−0.012 para k=50 zigzag).
- **Impacto:** os quantis só geram candidatos; os IVs são exatos depois → efeito prático nulo. Mas: (i) como `feature_vec` já está em memória, o sketch (O(n log k)) não economiza nada frente a ordenar/`nth_element` (exato, O(n)); (ii) a doc "provável" supera o que a estrutura garante.
- **Correção proposta:** ajustar a doc ("determinístico, ε≈1–2/k empírico") ou usar quantis exatos para n < 10^7 e KLL de verdade só em modo streaming.

##### E-14 — LDB/UBSD/Sketch: flag/rótulo de monotonicidade e `converged` não refletem o resultado
- **Algoritmo/arquivo:** `OBN_LDB_v5.cpp:524-546, 828-849` (`monotonicity` = direção escolhida, não o resultado); `OBN_Sketch_v5.cpp:558-574` (`converged_flag = true` constante); `OBN_UBSD_v5.cpp:101-126` (`iterations_run` sobrescrito por `iter+1` e usado também como contador interno)
- **Severidade:** Baixo
- **Lente:** Robustez
- **Status:** Confirmado (leitura)
- **Impacto:** quando o laço de monotonicidade para em `min_bins` com WoE ainda não monótono, LDB devolve `monotonicity="increasing"`; Sketch devolve `converged=TRUE` sempre. Usuário não consegue saber que a restrição falhou.
- **Correção proposta:** calcular `is_monotonic()` ao final e expor `monotonic_achieved`.

##### E-15 — Tolerâncias absolutas dependentes de escala
- **Algoritmo/arquivo:** `OBN_UBSD_v5.cpp:299 (sd < EPSILON)` e `301 (max-min >= EPSILON)`; `OBN_LDB_v5.cpp:384 (iqr < EPSILON)`; `common/optimal_binning_common.h:148 (span > EPSILON)`; `common/cutpoints_validator.h:42 (|a−b| < 1e-10)`; `OBN_LDB_v5.cpp:137 (epsilon=1e-14)`
- **Severidade:** Baixo
- **Lente:** Numérica
- **Status:** Confirmado (leitura)
- **Impacto:** variáveis em escala ≤1e-10 (SI, frações de ppm) são tratadas como constantes (UBSD gera 1 bin ou `sd<EPS` com 3+ valores distintos e range<1e-10 → um único bin apesar de `min_bins≥2`; KDE colapsa para 1 ponto). `OBN_Sketch` já removeu isso (comentário "tolerance of 1e-10 used to collapse…").
- **Correção proposta:** tolerâncias relativas (`eps*max(1,|scale|)` ou `std::nextafter`) ou comparar com `DBL_MIN`; remover o header morto `cutpoints_validator.h` (sem includes).

##### E-16 — LDB: `handle_few_unique_values` ignora `bin_cutoff`/monotonicidade
- **Algoritmo/arquivo:** `src/OBN_LDB_v5.cpp:240-286`
- **Severidade:** Baixo · **Lente:** Teoria · **Status:** Confirmado (leitura)
- **Impacto:** feature com 3 valores distintos e `min_bins=3` → 3 bins mesmo que um deles tenha 0,1% das linhas e WoE não monótono (`monotonicity="none"`).
- **Correção:** executar `merge_bins()` e `enforce_monotonicity()` também nesse ramo (respeitando `min_bins`).

##### E-17 — SWB: não há "sliding window"; JS entre todos os pares; IV de convergência ≠ IV reportado
- **Algoritmo/arquivo:** `src/OBC_SWB_v5.cpp:420-455, 458-500, 99-115`; `R/obc_swb.R:8-30`
- **Severidade:** Baixo · **Lente:** Teoria · **Status:** Confirmado (leitura)
- **Implementação:** sem janela deslizante; `merge_most_similar_bins` examina *todos* os pares (viés 0.95 para adjacentes); critério de convergência usa `utils::calculate_total_iv` (Laplace 2α) enquanto os bins reportam `calculate_metrics` (Bayes 0.5) — duas definições de IV no mesmo algoritmo. `min_bins>=1` aqui vs ≥2 nos demais (inconsistência de API).
- **Nota positiva:** `greedy_similarity_merge` (O(B²)) é idêntico ao scan completo: 3000 instâncias aleatórias (B 4–43), **0 divergências**, ASAN/UBSAN limpo (`work_E/swb_test.cpp`).
- **Correção:** renomear/documentar como "greedy JS merge" ou implementar a janela; unificar a definição de IV.

##### E-18 — Categóricos: tratamento de `""` inconsistente entre wrappers e apply
- **Algoritmo/arquivo:** `OBC_GMB_v5.cpp:116-121`, `OBC_MBA_v5.cpp:75-80` lançam erro em string vazia; `OBC_SWB/OBC_Sketch` aceitam `""` como categoria; `OB_ApplyWoECat.cpp:124-126` trata `""` como token de missing por default; wrappers R só mapeiam `NA → "NA"` (`R/obc_gmb.R:339`, `obc_mba.R:362`).
- **Severidade:** Baixo · **Lente:** Robustez · **Status:** Confirmado
- **Impacto:** o mesmo vetor com `""` falha em GMB/MBA e passa em SWB/Sketch; no apply, `""` é missing → pode receber o bin "NA".
- **Correção:** mapear `""` → `"NA"` (ou `"<empty>"`) uniformemente no wrapper.

##### E-19 — Rótulos e separadores: treino vs apply e `binning_categorical_cutpoints`
- **Algoritmo/arquivo:** `OB_ApplyWoENum.cpp:121-129` (`%g` com 6 dígitos: `(−Inf;12.5]`) vs `OBN_LDB/UBSD/Sketch` (`std::fixed 6 casas`: `(−Inf;12.500000]`); `OB_Cutpoints.cpp:66-69` (2 casas); `OB_Cutpoints.cpp:176-196` divide categorias por `'+'` (categoria "C++" ou "A+" impossível de expressar).
- **Severidade:** Baixo · **Lente:** Robustez · **Status:** Confirmado
- **Impacto:** o `bin` devolvido pelo treino e pelo apply para o mesmo bin tem strings diferentes (joins por rótulo falham); duas formatações com 2 casas podem colapsar cortes distintos no rótulo (a atribuição usa o valor, não o rótulo — só visual).
- **Correção:** função de formatação única em header comum; separador configurável em `binning_categorical_cutpoints`.

##### E-20 — Latências/ODR e infraestrutura de build
- **Severidade:** Baixo · **Lente:** Robustez · **Status:** Confirmado (leitura)
- (a) **ODR**: `utils::safe_log(double)` aparece com o mesmo texto em 7 TUs (`OBN_Sketch`, `OBC_Sketch`, `OBC_SWB`, `OBC_UDT`, `OBC_JEDIMWoE`, …); `utils::join_categories(const vector<string>&, const string&)` com **corpos diferentes** (ostringstream em `OBC_UDT:108`, concatenação em `OBC_SWB`/`OBC_JEDIMWoE:36`); `utils::calculate_woe/iv` em 3 TUs. Funções `inline` com definições distintas em TUs diferentes é UB (o linker escolhe uma); hoje equivalem funcionalmente, mas uma edição divergente passaria despercebida. Mover para header comum ou `namespace {}`.
- (b) `src/Makevars:20` só define `PKG_CXXFLAGS = $(SHLIB_OPENMP_CXXFLAGS)`; `CXX_STD` está comentado (linha 15 e 28) enquanto `DESCRIPTION` declara `SystemRequirements: C++17` e `Depends: R (>= 4.1.0)` e o código usa structured bindings (`OBC_Sketch_v5.cpp:84,96`, `OBC_UDT`, `OBC_CM`). Em R 4.1/4.2 (default C++14) isso compila só com `-Wpedantic`-warning (GCC/Clang). Descomentar `CXX_STD = CXX17`.
- (c) Nenhum arquivo do escopo chama `Rcpp::checkUserInterrupt()` (apenas `OBC_JEDI/JEDIMWoE/MILP/MOB` fazem). Os laços do escopo são O(n log n) puros ou O(B²) com B pequeno; risco real só em `OBC_MBA` com ≥10^5 categorias (E-10) e `OBC_SWB`/`OBC_Sketch` com `max_n_prebins` alto.
- (d) `OB_Correlation.cpp:791-798`: mapeamento linear→(i,j) é O(p) por par ⇒ O(p³/…) no total; irrelevante para p ≤ 1000. `int total_pairs = p*(p-1)/2` estoura em p > 65 535.
- (e) Encoding: todos os categóricos usam `CHAR(cs)` sem `translateCharUTF8`; a afirmação em `OBC_SWB`/`OBC_Sketch` ("merges equal byte strings stored under different encodings") só vale para bytes idênticos; "é" em latin1 e UTF-8 são categorias distintas no C++ mas iguais no R. (Suspeita; não testado, sem R no container.)

##### E-21 — Monotonicidade em categóricos é trivial (observação de fidelidade)
- **Algoritmo/arquivo:** `OBC_MBA_v5.cpp:313-398`, `OBC_Sketch_v5.cpp:569-614`, `OBC_SWB_v5.cpp:497-540`
- **Severidade:** Baixo · **Lente:** Teoria · **Status:** Confirmado (leitura)
- **Implementação:** os bins são ordenados por WoE e *depois* verificada a monotonicidade; ela vale por construção (ordem arbitrária de categorias), então `enforce_monotonicity` não funde nada e `monotonicity_threshold = min(EPSILON, 0.01·gap)` (GMB:`ensureMonotonicity`) é efetivamente 1e-10, não "adaptativo" como a doc diz. Documentar que "monotônico" é apenas rótulo de ordenação.

---

#### Problemas transversais observados (headers comuns, padrões repetidos)

**1. Quatro (ou mais) definições de WoE/IV coexistem**
| Onde | Fórmula |
|---|---|
| `NumericalBin/CategoricalBin::calculate_metrics` (`bin_structures.h:97-109, 225-237`) | Bayes, prior 0,5·taxa global; `return` silencioso (WoE=0) se `total_pos<=0` ou `total_neg<=0` |
| OBC_SWB/OBC_Sketch `utils::calculate_*` | Laplace α=0,5, denominador `total+2α` |
| OBN_UBSD | Laplace α configurável, denominador `total+K·α` (consistente, Σp=1) |
| OBN_LDB | WoE Laplace(K·α), IV com distribuição bruta (E-06) |
| OBN_Sketch (DP/ranking) | sem suavização, clamp EPS=1e-10 (E-05) |
| `OB_Cutpoints.cpp` | sem suavização, piso 1e-4 se contagem==0 (E-08) |
`MAX_WOE/MIN_WOE` (`optimal_binning_common.h:33-36`) só são usados por `OBC_CM`, `OBN_CM`, `woe_iv_utils.h`; UBSD usa literal `±20.0`. WoE de bins idênticos difere entre algoritmos e entre `fit` e `binning_*_cutpoints`.

**2. Epsilons (grep literal em `src/OB*_v5.cpp`, `OB_*.cpp`, `common/*.h`)**
- `1e-10`: 19 arquivos (OBC_CM 2, OBN_CM 3, OBC_SBLP 2, OBN_LPDB 2, OB_Correlation 3, `cutpoints_validator.h` 3, `common.h` 2, 1 em cada de OBC_JEDI, OBN_BB/DMIV/EWB/JEDI/JEDIMWoE/KMB/MBLP/Sketch/UBSD/UDT, OB_LogisticRegression)
- `1e-9`: OBN_MBLP 1; `1e-12`: OBC_CM 3, OBN_CM 3, OBN_FETB 1, OB_Correlation 1, OB_LogisticRegression 3, `common.h` 1 (EPSILON_STRICT); `1e-14`: OBN_LDB 1; `1e-7`: OBC_FETB 3, OBN_FETB 4, OB_CheckDistinctsLength 1; `1e-8`/`1e-5`: OB_LogisticRegression; `0.0001`/`1e-4`: OB_Cutpoints 3; `1e-3`: OBN_FETB 1.
- `1e-6` (convergence default): 38 arquivos.
- Nos arquivos do escopo: `OBN_Sketch` 9 usos de `EPSILON`, `OBC_Sketch` 9, `OBC_GMB` 2, `OBC_MBA` 6, `OBC_SWB` 7, `OBN_UBSD` 2, `OBN_LDB` 4 (+1e-14).
- Tolerância de monotonicidade: GMB/MBA `min(EPSILON, 0.01·gap)` (=1e-10 efetivo); SWB/Sketch/OBN_Sketch `EPSILON` absoluto; LDB/UBSD sem tolerância (`diff<0`).

**3. Contagens de padrões em `src/OB*_v5.cpp` (37 arquivos; totais)**
| Padrão | Total | Arquivos com maiores contagens |
|---|---|---|
| `.erase(` | 132 | OBN_DMIV 13, OBN_BB 12, OBN_FastMDLPM 12, OBN_LDB 11, OBN_EWB 8, OBN_MBLP 7 |
| `.back()/.front()` | 59 | OBN_IR 9, OBN_FastMDLPM 8, OBN_Sketch 5, OBN_OSLP 4 |
| `size() - 1` | 108 | OBN_LPDB 13, OBN_CM 7, OBN_LDB 6, OBC_CM 6 |
| `std::log(` cru | 85 (vs `safe_log` 37) | OBC_DMIV 21, OBN_DMIV 6, OBN_UDT 5 |
| `Rcpp::checkUserInterrupt` | **4** | OBC_JEDI, OBC_JEDIMWoE, OBC_MILP, OBC_MOB (nenhum numérico, nenhum do escopo) |
| `#pragma omp` | 0 em OB*_v5 (3 em OB_Correlation.cpp) | — |
| RNG | 5 | OBC_SAB 3 (usa `R::unif_rand`, OK), OBC_Sketch 2 (`mt19937(42)` fixo, determinístico) |
| `unordered_map<SEXP` | 4 | OBC_SBLP, OBC_SWB, OBC_Sketch, OBC_UDT |
| `std::map<` | 0 | — |
| `static_cast<int>(x.size())` | 200 | OBC_MBA 24, OBN_MBLP 20, OBC_Sketch 15, OBC_GMB/IVB 11 — risco de overflow só para >2^31 elementos; `int` para contagens (`count`, `total_pos`) idem |
| comparação `==/!=` de double | 2 em escopo | `OBN_Sketch:248 (lo != hi)`, `OBN_LDB:609 (valid_feature[i] != valid_feature[i-1])` — corretos (detecção de empates exatos) |
| `Rcpp::warning` | 42 | ok (nenhum em laço quente) |
Padrão "`vector::erase` em laço com recomputo": presente em GMB/MBA/Sketch/SWB/LDB/UBSD, todos com B ≤ `max_n_prebins` (≤20 por default), logo aceitável; só MBA (E-10) escala mal.

**4. Headers**
- `safe_math.h`: correto (Kahan/Neumaier, `log_sum_exp`); incluído apenas por `woe_iv_utils.h`/`monotonicity_utils.h`/`entropy_utils.h`; `safe_divide` devolve 0.0 quando |denom| ≤ ε (mascara divisões degeneradas), `compensated_sum` usa soma ingênua para n<10.
- `validation_utils.h`, `cutpoints_validator.h`: **sem includes** em todo o `src/` (código morto); o validador usa tolerância absoluta 1e-10 + `std::unique` com predicado não transitivo.
- `bin_structures.h`: `NumericalBin::merge_with` assume `other` imediatamente à direita (apenas atualiza `upper_bound`); `calculate_metrics` ignora silenciosamente totais ≤ 0; `NumericalBin` ctor "count ignora argumento c" (documentado); `event_rate()` de bin vazio = 0.0 (ordena bins vazios como baixo-risco nos algoritmos que fundem por taxa).
- `optimal_binning_common.h` KDE: `std::exp` por (d·dx/h)² sem truncar o kernel (O(G²)=262k exps/ cálculos por variável, ok); `span > EPSILON` absoluto; usa `M_PI` (não portátil MSVC sem `_USE_MATH_DEFINES`).

**5. Apply (consistência treino ↔ apply)** — verificado e **correto**: bins right-closed `(a,b]`; apply usa `lower_bound` (primeiro corte ≥ x) com `include_upper_bound=TRUE`; ±Inf caem no primeiro/último bin; NaN/NA → bin "NA"/"Special"; cutpoints ordenados exigidos; bin de missing localizado por rótulo "NA"/"Missing" (ou último). Categorias não vistas → `NA` WoE, `idbin = nbins+1`, `ismissing=1` (política documentada). Única inconsistência: E-09 e E-19.

**6. OpenMP / Makevars**: ver E-20(b)/(d). Região paralela em `OB_Correlation.cpp:761-815` é race-free (slots disjuntos, sem R API, `num_threads` limitado a 2 por default para CRAN). `Makevars.win` idêntico. As flags de OpenMP são aplicadas a todos os TUs sem necessidade.

#### Priorização recomendada (top 5 do escopo)

1. **E-01** — substituir o ranking guloso do `OBN_Sketch` por DP sobre os candidatos (ganho de IV até +29%; muda o caminho de uso real, n>50).
2. **E-03** — remover o limiar de positivos em `OBC_MBA`/`OBC_Sketch` (fusão indevida de bins grandes de baixo risco; contradiz a doc).
3. **E-04** — GMB: reduzir pré-bins por fusão adjacente de menor perda de IV em vez de "dobrar" no menor bin.
4. **E-02** — aplicar em `OBC_Sketch::enforce_bin_cutoff` a mesma correção de índices já feita no MBA (3 linhas).
5. **E-06 + E-08 (+E-09)** — unificar a definição de WoE/IV (LDB com IV coerente; `OB_Cutpoints` com suavização/EPS em vez do piso 1e-4; sentinela `-999` do apply opt-in).

Arquivos de experimento (todos em `scratchpad/work_E/`): `kll_test.cpp`, `greedy_sel.cpp`, `dp_test.cpp`, `sk_cutoff.cpp`, `sk_cutoff2.cpp`, `swb_test.cpp`, `mba_perf.cpp`, `kde.cpp`, `bw.cpp`, `cm.cpp`, `js.cpp`, `stub/Rcpp.h` (stub vazio para compilar headers sem R).
