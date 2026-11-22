# Differential report: G1-14: OBN_MDLP runs the Fayyad-Irani MDLP (X-06 / A-01, A-02)

- **Verdict:** **FAIL**
- Findings addressed: X-06, A-01, A-02
- Previously accepted changes in scope (cumulative vs baseline): self-test-identical-builds | onda0-01: pure bins get the most extreme finite WoE (OBC_DP, OBC_FETB) | onda0-10: declare C++17 in Makevars (no numerical change expected) | G1-10: OBN_CM merge_zero_bins respects min_bins (X-03 / A-06) | G1-11: OBC_CM no longer stops on |delta min chi2| < tol (X-05 / A-07) | G1-12: FastMDLPM force_min_bins keeps the MDL cuts (X-07 / A-04) | G1-13: numerical ChiMerge honours alpha (Kerber); Chi2 phases with reversion (X-04 / A-05, A-08)
- Baseline: 1.14.1 | Candidate: 1.14.1
- Runs compared: 180 (180 target, 0 non-target)
- Non-target runs bit-identical: 0 / 0
- Target runs changed: 162 / 180
- Failing checks: 42

## In-scope summary (target runs)

| engine | runs | changed | mean ref IV B | mean ref IV C | mean OOS IV B | mean OOS IV C | non-monotone B→C | min_bins viol. B→C | errors B→C |
|---|---|---|---|---|---|---|---|---|---|
| ob_numerical_mdlp | 180 | 162 | 2.2672 | 2.2591 | 2.0610 | 1.9836 | 20→18 | 0→0 | 0→0 |

## Failing checks

| run | failures | bins B→C | ref IV B → C |
|---|---|---|---|
| `ob_numerical_mdlp::signal/step/s1::default` | C:became_nonmonotone | 4→3 | 0.5479 → 0.5558 |
| `ob_numerical_mdlp::signal/step/s1::wide` | C:became_nonmonotone | 4→3 | 0.5479 → 0.5558 |
| `ob_numerical_mdlp::signal/step/s3::default` | C:became_nonmonotone | 3→3 | 0.6130 → 0.6336 |
| `ob_numerical_mdlp::signal/step/s3::wide` | C:became_nonmonotone | 5→3 | 0.6119 → 0.6336 |
| `ob_numerical_mdlp::signal/u_shape/s1::wide` | C:became_nonmonotone | 3→3 | 0.4572 → 1.5113 |
| `ob_numerical_mdlp::signal/u_shape/s3::default` | C:became_nonmonotone | 3→3 | 0.3835 → 1.4914 |
| `ob_numerical_mdlp::signal/u_shape/s3::wide` | C:became_nonmonotone | 4→3 | 0.3857 → 1.4914 |
| `ob_numerical_mdlp::separation/near/s3::default` | C:became_nonmonotone | 3→3 | 12.4385 → 12.9155 |
| `ob_numerical_mdlp::separation/near/s3::wide` | C:became_nonmonotone | 3→3 | 12.4385 → 12.9155 |
| `ob_numerical_mdlp::discrete/mass_at_zero/s2::default` | C:became_nonmonotone | 3→3 | 0.0970 → 0.1160 |
| `ob_numerical_mdlp::discrete/mass_at_zero/s2::wide` | C:became_nonmonotone | 3→3 | 0.0970 → 0.1160 |

- E:oos_iv_inferior ob_numerical_mdlp discrete/ties default (B=0.7217 C=0.6935)
- E:oos_iv_inferior ob_numerical_mdlp imbalance/rate_02pct default (B=0.5514 C=0.5367)
- E:oos_iv_inferior ob_numerical_mdlp imbalance/rate_1pct default (B=1.6281 C=0.8541)
- E:oos_iv_inferior ob_numerical_mdlp outlier/single_extreme default (B=0.7760 C=0.6989)
- E:oos_iv_inferior ob_numerical_mdlp scale/huge default (B=0.7053 C=0.6477)
- E:oos_iv_inferior ob_numerical_mdlp scale/tiny default (B=0.7053 C=0.6477)
- E:oos_iv_inferior ob_numerical_mdlp separation/near default (B=11.6323 C=11.1447)
- E:oos_iv_inferior ob_numerical_mdlp separation/perfect default (B=12.8605 C=11.7480)
- E:oos_iv_inferior ob_numerical_mdlp signal/logistic_strong default (B=1.4172 C=1.3553)
- E:oos_iv_inferior ob_numerical_mdlp signal/logistic_weak default (B=0.0768 C=0.0579)
- E:oos_iv_inferior ob_numerical_mdlp size/n60 default (B=0.6296 C=0.3879)
- E:oos_iv_inferior ob_numerical_mdlp size/small_n40 default (B=2.2139 C=1.4364)
- E:oos_iv_inferior ob_numerical_mdlp imbalance/rate_1pct tight (B=1.3274 C=0.3184)
- E:oos_iv_inferior ob_numerical_mdlp outlier/single_extreme tight (B=0.6705 C=0.6157)
- E:oos_iv_inferior ob_numerical_mdlp scale/huge tight (B=0.6800 C=0.6262)
- E:oos_iv_inferior ob_numerical_mdlp scale/tiny tight (B=0.6800 C=0.6262)
- E:oos_iv_inferior ob_numerical_mdlp separation/near tight (B=11.8442 C=11.3259)
- E:oos_iv_inferior ob_numerical_mdlp separation/perfect tight (B=12.8605 C=12.4487)
- E:oos_iv_inferior ob_numerical_mdlp signal/logistic_weak tight (B=0.0482 C=0.0303)
- E:oos_iv_inferior ob_numerical_mdlp size/n60 tight (B=0.5664 C=0.3084)
- E:oos_iv_inferior ob_numerical_mdlp discrete/ties wide (B=0.7732 C=0.6935)
- E:oos_iv_inferior ob_numerical_mdlp imbalance/rate_1pct wide (B=1.2245 C=0.8541)
- E:oos_iv_inferior ob_numerical_mdlp missing/na_30pct wide (B=0.7074 C=0.6314)
- E:oos_iv_inferior ob_numerical_mdlp outlier/single_extreme wide (B=0.8554 C=0.6989)
- E:oos_iv_inferior ob_numerical_mdlp scale/huge wide (B=0.7428 C=0.6477)
- E:oos_iv_inferior ob_numerical_mdlp scale/tiny wide (B=0.7428 C=0.6477)
- E:oos_iv_inferior ob_numerical_mdlp separation/near wide (B=11.6323 C=11.1447)
- E:oos_iv_inferior ob_numerical_mdlp separation/perfect wide (B=12.8605 C=11.7480)
- E:oos_iv_inferior ob_numerical_mdlp signal/logistic_strong wide (B=1.4131 C=1.3553)
- E:oos_iv_inferior ob_numerical_mdlp size/n60 wide (B=0.6296 C=0.3879)
- E:oos_iv_inferior ob_numerical_mdlp size/small_n40 wide (B=1.8442 C=1.4364)

## Declared deviations (manifest rationale)

- `ob_categorical_dp` / `cat/pure_category/*` / `tight`: With max_bins = 3 the exact DP now isolates the pure category A (0 events). The population IV of that partition is +infinity (pure bin), so it is the theoretically preferred one; the hold-out IV is finite only because the reference estimator smooths it (alpha = 0.5), which ranks it slightly lower in one seed. This is the intended B-01 behaviour, not a regression.

- `ob_categorical_fetb` / `*` / `*`: FETB merges by Fisher p-value, it does not maximise IV, so training IV is not a monotone target for it. In k200_signal the baseline kept a pool of 47 pure categories (0 events) in the middle of the order because its WoE was reported as 0; it is now placed next to the lowest-risk bin and the Fisher test merges the two. The hold-out IV non-inferiority (layer E) is the binding check and passes for every scenario family.

- `ob_numerical_cm` / `separation/near/*` / `*`: With min_bins = 3 (default/wide) the baseline returned 2 bins, the second one holding 944 events and 1 non-event, i.e. it violated min_bins. The candidate honours min_bins by keeping the pure event bin (525/0) apart from the near-pure one (419/1). The two pieces have practically the same event rate, so the split adds almost no information while the reference estimator smooths with K = 3 instead of K = 2 (a larger pseudo-count mass), which lowers the smoothed IV by 0.07 on 8.16 (s1 only). A drop caused by satisfying a violated min_bins is the exception foreseen in audit Section 9.3-C. Hold-out IV is non-inferior.

- `ob_numerical_cm` / `imbalance/rate_02pct/*` / `*`: 0.2% event rate: about 4 events in 2100 training rows. The baseline collapsed every bin without events and returned 1 or 2 bins although min_bins = 3 (min_bins violation, IV = 0 in s2). The candidate keeps min_bins = 3 bins; with 1 / 1 / 0 events their WoE cannot be monotone, and OBN_CM documents that monotonicity merging stops at min_bins (min_bins has priority). The non-monotone shape is the honest outcome of the requested min_bins on 4 events, not a regression of the merge rule.

- `ob_categorical_cm` / `cat/noise_k10/*` / `*`: cat/noise_k10 has no signal (true IV = 0). With min_bins = 3 / max_bins = 8 (wide, s1) the baseline stopped after two consecutive minimum chi-squares of 0 (Yates term clamped at zero) and kept 7 bins of statistically indistinguishable levels; ChiMerge (Kerber 1992) keeps merging while the smallest chi-square is below the critical value, so the candidate merges them down to min_bins = 3. The training IV of a noise variable shrinks (0.0167 -> 0.0157), which is the intended effect of honouring the test; layer E already treats this family as null-signal.

- `ob_numerical_fast_mdlp` / `imbalance/rate_1pct/*` / `wide`: About 21 events in 2100 training rows. With min_bins = 2 (default / tight) the baseline itself returns the single Fayyad-Irani cut (s1: 0.943638, s2: 2.051675; candidate unchanged there). With min_bins = 3 (wide) the baseline threw that cut away and returned equally spaced cuts; the candidate keeps it and bisects the most populated bin. The hold-out IV of the kept tail bin (s2: 36 training rows, 7 events) is noisy with ~9 test events, so the median over 3 seeds moves 1.08 -> 0.85; it is the IV of the MDL-accepted cut that the baseline already reports for min_bins = 2, not of a cut introduced by this change.

- `ob_numerical_fast_mdlp` / `separation/near/*` / `wide`: The baseline discarded the MDL cut at the true class boundary (s3: cuts -0.017 / 0.660 instead of -0.0024). The candidate keeps it; min_bins = 3 then bisects the pure non-event side (1/531 vs 0/532 events), two bins with practically equal rate whose smoothed WoE differ by count, and monotonicity merging stops at min_bins as documented.

- `ob_numerical_fast_mdlp` / `discrete/mass_at_zero/*` / `wide`: 90% of the mass at x = 0. The MDL cut at 0 is kept and the forced third bin bisects x > 0 by observations (s2: event rates 0.287 / 0.268 / 0.575); the first two bins have nearly equal rate and monotonicity merging stops at min_bins = 3, as documented.

- `ob_numerical_cm` / `imbalance/rate_1pct/*` / `*`: About 21 events in 2100 training rows (expected cell counts below 5 in most 2x2 tables). Adjacent baseline bins such as 4/416 vs 8/412 events (s1) are not significantly different at alpha = 0.05, so Kerber merges them (s1 5 -> 3 bins, s2 4 -> 3, s3 4 -> 3, min_bins = 3). The hold-out IV median falls 1.93 -> 1.56 (default/wide) and 1.56 -> 1.33 (tight) by the coarsening argument above; with ~9 hold-out events it is dominated by sampling noise. Hold-out log-loss changes by +0.0010 / +0.0019 / +0.0012 nats (default / tight / wide, on 0.058).

- `ob_numerical_cm` / `imbalance/rate_02pct/*` / `*`: About 4 events in 2100 training rows. The baseline also violated min_bins here (1-2 bins, see manifest 10); the candidate returns min_bins = 3 bins, whose 1 / 1 / 0 events cannot be monotone, and monotonicity merging stops at min_bins as documented. Hold-out IV with ~1-2 test events is noise (tight 0.48 -> 0.06); hold-out log-loss is unchanged or better (tight -0.00004 nats).

- `ob_numerical_cm` / `signal/step/*` / `*`: The true rate has two levels (0.1 / 0.4 at x = 0.5). Kerber merges the non-significant pieces (s2 tight: a 105-row bin at 32/73 events next to a 945-row bin at 380/565 merges into the 2-bin step), lowering hold-out IV 0.703 -> 0.670 by coarsening; log-loss +0.0009 nats on 0.50. With min_bins = 3 (default/wide) the third bin must split one flat level of the step, its WoE order is noise, and monotonicity merging stops at min_bins (documented), hence the non-monotone runs.

- `ob_numerical_cm` / `discrete/ties/*` / `wide`: round(rnorm, 1) with a weak logistic signal: the pre-bins that are not significantly different at alpha = 0.05 are merged down from up to 8 bins; hold-out IV median 0.773 -> 0.722 by coarsening, hold-out log-loss +0.0028 nats on 0.605 (0.5%).

- `ob_numerical_cm` / `scale/*` / `wide`: scale/tiny and scale/huge are the same logistic (slope 1 on the standardised value) at scales 1e-12 / 1e12; both give identical bins. With max_bins = 8 Kerber merges the adjacent pre-bins whose rates do not differ at alpha = 0.05 (smooth logistic: neighbouring quantile bins differ little), hold-out IV median 0.743 -> 0.718 by coarsening; hold-out log-loss +0.0024 nats on 0.608 (0.4%).

- `ob_numerical_cm` / `signal/u_shape/*` / `*`: The true rate is U-shaped (0.6 / 0.1 / 0.6), so a correct binning is non-monotone. Honouring alpha lets the significant tails survive (s1 wide: ref IV 0.46 -> 1.47; hold-out log-loss -0.042 nats).

- `ob_numerical_cm` / `signal/noise/*` / `*`: No signal (true IV = 0, layer E skips the family): nothing is significant, Kerber merges down to min_bins and the remaining WoE order of min_bins = 3 noise bins is arbitrary.

- `ob_numerical_cm` / `size/small_n40/*` / `*`: n = 40 (28 training rows): few pairs are significant, Kerber merges down to min_bins = 3 and their WoE order is driven by 2-3 events per bin; monotonicity merging stops at min_bins (documented). Hold-out log-loss improves (wide -0.008 nats).

- `ob_numerical_cm` / `*` / `*`: See the common theory above: merging the pairs that are not significantly different lowers training IV by construction.


_Layer E skips null-signal families (signal/noise, cat/noise_k10): true IV = 0, hold-out IV is noise._

## Changed target runs (detail)

| run | bins B→C | ref IV B → C | OOS IV B → C |
|---|---|---|---|
| `ob_numerical_mdlp::signal/logistic_strong/s1::default` | 5→5 | 1.5096 → 1.5972 | 1.7036 → 1.8219 |
| `ob_numerical_mdlp::signal/logistic_strong/s1::tight` | 3→3 | 1.2828 → 1.3201 | 1.4505 → 1.5077 |
| `ob_numerical_mdlp::signal/logistic_strong/s1::wide` | 8→5 | 1.6005 → 1.5972 | 1.8726 → 1.8219 |
| `ob_numerical_mdlp::signal/logistic_strong/s2::default` | 5→5 | 1.6782 → 1.8241 | 1.4172 → 1.3553 |
| `ob_numerical_mdlp::signal/logistic_strong/s2::tight` | 3→3 | 1.3783 → 1.6116 | 1.1537 → 1.2516 |
| `ob_numerical_mdlp::signal/logistic_strong/s2::wide` | 7→5 | 1.8058 → 1.8241 | 1.4131 → 1.3553 |
| `ob_numerical_mdlp::signal/logistic_strong/s3::default` | 5→4 | 1.5427 → 1.5677 | 1.3267 → 1.3098 |
| `ob_numerical_mdlp::signal/logistic_strong/s3::tight` | 3→3 | 1.2860 → 1.3178 | 1.1085 → 1.1096 |
| `ob_numerical_mdlp::signal/logistic_strong/s3::wide` | 8→4 | 1.6199 → 1.5677 | 1.3623 → 1.3098 |
| `ob_numerical_mdlp::signal/logistic_weak/s1::default` | 4→3 | 0.0972 → 0.0982 | 0.0768 → 0.0579 |
| `ob_numerical_mdlp::signal/logistic_weak/s1::tight` | 3→2 | 0.1128 → 0.0918 | 0.0311 → 0.0165 |
| `ob_numerical_mdlp::signal/logistic_weak/s1::wide` | 5→3 | 0.1096 → 0.0982 | 0.0519 → 0.0579 |
| `ob_numerical_mdlp::signal/logistic_weak/s2::default` | 4→3 | 0.0921 → 0.0944 | 0.1571 → 0.1726 |
| `ob_numerical_mdlp::signal/logistic_weak/s2::tight` | 3→2 | 0.0889 → 0.0804 | 0.1433 → 0.1422 |
| `ob_numerical_mdlp::signal/logistic_weak/s2::wide` | 5→3 | 0.0951 → 0.0944 | 0.1634 → 0.1726 |
| `ob_numerical_mdlp::signal/logistic_weak/s3::default` | 3→3 | 0.0851 → 0.1810 | 0.0347 → 0.0303 |
| `ob_numerical_mdlp::signal/logistic_weak/s3::tight` | 3→3 | 0.1512 → 0.1810 | 0.0482 → 0.0303 |
| `ob_numerical_mdlp::signal/logistic_weak/s3::wide` | 3→3 | 0.1339 → 0.1810 | 0.0574 → 0.0303 |
| `ob_numerical_mdlp::signal/step/s1::default` | 4→3 | 0.5479 → 0.5558 | 0.5522 → 0.5452 |
| `ob_numerical_mdlp::signal/step/s1::tight` | 2→2 | 0.5440 → 0.5574 | 0.5376 → 0.5216 |
| `ob_numerical_mdlp::signal/step/s1::wide` | 4→3 | 0.5479 → 0.5558 | 0.5522 → 0.5452 |
| `ob_numerical_mdlp::signal/step/s2::default` | 4→3 | 0.7426 → 0.7599 | 0.6988 → 0.7215 |
| `ob_numerical_mdlp::signal/step/s2::tight` | 2→2 | 0.7358 → 0.7616 | 0.6703 → 0.7279 |
| `ob_numerical_mdlp::signal/step/s2::wide` | 5→3 | 0.7443 → 0.7599 | 0.7016 → 0.7215 |
| `ob_numerical_mdlp::signal/step/s3::default` | 3→3 | 0.6130 → 0.6336 | 0.7455 → 0.8073 |
| `ob_numerical_mdlp::signal/step/s3::tight` | 2→2 | 0.6127 → 0.6333 | 0.7400 → 0.7874 |
| `ob_numerical_mdlp::signal/step/s3::wide` | 5→3 | 0.6119 → 0.6336 | 0.7616 → 0.8073 |
| `ob_numerical_mdlp::signal/multi_step/s1::default` | 5→3 | 2.0473 → 2.0684 | 1.9701 → 1.9952 |
| `ob_numerical_mdlp::signal/multi_step/s1::tight` | 3→3 | 1.9849 → 2.0684 | 1.9600 → 1.9952 |
| `ob_numerical_mdlp::signal/multi_step/s1::wide` | 5→3 | 1.9820 → 2.0684 | 1.8849 → 1.9952 |
| `ob_numerical_mdlp::signal/multi_step/s2::default` | 3→3 | 1.5293 → 2.2003 | 1.5848 → 2.0328 |
| `ob_numerical_mdlp::signal/multi_step/s2::tight` | 3→3 | 2.1649 → 2.2003 | 2.0409 → 2.0328 |
| `ob_numerical_mdlp::signal/multi_step/s2::wide` | 5→3 | 2.0343 → 2.2003 | 1.9603 → 2.0328 |
| `ob_numerical_mdlp::signal/multi_step/s3::default` | 4→3 | 1.1647 → 1.9113 | 1.5195 → 2.0676 |
| `ob_numerical_mdlp::signal/multi_step/s3::tight` | 3→3 | 1.8093 → 1.9113 | 2.0101 → 2.0676 |
| `ob_numerical_mdlp::signal/multi_step/s3::wide` | 5→3 | 1.6017 → 1.9113 | 1.7564 → 2.0676 |
| `ob_numerical_mdlp::signal/u_shape/s1::default` | 3→3 | 1.4722 → 1.5113 | 1.3869 → 1.3486 |
| `ob_numerical_mdlp::signal/u_shape/s1::tight` | 2→2 | 0.4523 → 0.4615 | 0.4929 → 0.4878 |
| `ob_numerical_mdlp::signal/u_shape/s1::wide` | 3→3 | 0.4572 → 1.5113 | 0.4970 → 1.3486 |
| `ob_numerical_mdlp::signal/u_shape/s2::default` | 3→3 | 1.1562 → 1.4424 | 1.2570 → 1.3994 |
| `ob_numerical_mdlp::signal/u_shape/s2::tight` | 2→2 | 0.5196 → 0.5285 | 0.3082 → 0.3136 |
| `ob_numerical_mdlp::signal/u_shape/s2::wide` | 3→3 | 1.1562 → 1.4424 | 1.2570 → 1.3994 |
| `ob_numerical_mdlp::signal/u_shape/s3::default` | 3→3 | 0.3835 → 1.4914 | 0.3595 → 1.3475 |
| `ob_numerical_mdlp::signal/u_shape/s3::tight` | 2→2 | 0.3636 → 0.4448 | 0.3472 → 0.4336 |
| `ob_numerical_mdlp::signal/u_shape/s3::wide` | 4→3 | 0.3857 → 1.4914 | 0.3894 → 1.3475 |
| `ob_numerical_mdlp::signal/noise/s1::default` | 3→3 | 0.0070 → 0.0025 | 0.0050 → 0.0105 |
| `ob_numerical_mdlp::signal/noise/s1::tight` | 2→2 | 0.0086 → 0.0021 | 0.0005 → 0.0006 |
| `ob_numerical_mdlp::signal/noise/s1::wide` | 3→3 | 0.0020 → 0.0025 | 0.0011 → 0.0105 |
| `ob_numerical_mdlp::signal/noise/s2::default` | 3→3 | 0.0039 → 0.0043 | 0.0099 → 0.0144 |
| `ob_numerical_mdlp::signal/noise/s2::tight` | 2→2 | 0.0034 → 0.0008 | 0.0086 → 0.0094 |
| `ob_numerical_mdlp::signal/noise/s2::wide` | 3→3 | 0.0031 → 0.0043 | 0.0046 → 0.0144 |
| `ob_numerical_mdlp::signal/noise/s3::default` | 3→3 | 0.0224 → 0.0173 | 0.0061 → 0.0091 |
| `ob_numerical_mdlp::signal/noise/s3::tight` | 2→2 | 0.0130 → 0.0147 | 0.0007 → 0.0084 |
| `ob_numerical_mdlp::signal/noise/s3::wide` | 3→3 | 0.0224 → 0.0173 | 0.0061 → 0.0091 |
| `ob_numerical_mdlp::separation/perfect/s1::default` | 3→3 | 14.5302 → 14.5865 | 12.7825 → 11.7480 |
| `ob_numerical_mdlp::separation/perfect/s1::tight` | 3→2 | 14.5302 → 15.2855 | 12.7825 → 12.4487 |
| `ob_numerical_mdlp::separation/perfect/s1::wide` | 3→3 | 14.5302 → 14.5865 | 12.7825 → 11.7480 |
| `ob_numerical_mdlp::separation/perfect/s2::default` | 3→3 | 14.4353 → 14.5855 | 13.0549 → 12.8721 |
| `ob_numerical_mdlp::separation/perfect/s2::tight` | 3→2 | 14.4353 → 15.2847 | 13.0549 → 13.5766 |
| `ob_numerical_mdlp::separation/perfect/s2::wide` | 3→3 | 14.4353 → 14.5855 | 13.0549 → 12.8721 |
| `ob_numerical_mdlp::separation/perfect/s3::default` | 3→3 | 14.5485 → 14.5865 | 12.8605 → 10.8549 |
| `ob_numerical_mdlp::separation/perfect/s3::tight` | 3→2 | 14.5485 → 15.2856 | 12.8605 → 11.5527 |
| `ob_numerical_mdlp::separation/perfect/s3::wide` | 3→3 | 14.5485 → 14.5865 | 12.8605 → 10.8549 |
| `ob_numerical_mdlp::separation/near/s1::default` | 4→3 | 13.4296 → 13.4758 | 10.5545 → 11.1447 |
| `ob_numerical_mdlp::separation/near/s1::tight` | 3→2 | 13.5040 → 14.1737 | 10.7491 → 11.3259 |
| `ob_numerical_mdlp::separation/near/s1::wide` | 4→3 | 13.4296 → 13.4758 | 10.5545 → 11.1447 |
| `ob_numerical_mdlp::separation/near/s2::default` | 4→3 | 13.0391 → 12.9153 | 11.6323 → 12.3028 |
| `ob_numerical_mdlp::separation/near/s2::tight` | 3→2 | 12.3902 → 13.0634 | 11.8626 → 12.4501 |
| `ob_numerical_mdlp::separation/near/s2::wide` | 4→3 | 13.0391 → 12.9153 | 11.6323 → 12.3028 |
| `ob_numerical_mdlp::separation/near/s3::default` | 3→3 | 12.4385 → 12.9155 | 11.8442 → 10.5804 |
| `ob_numerical_mdlp::separation/near/s3::tight` | 3→2 | 12.4385 → 13.0636 | 11.8442 → 11.2770 |
| `ob_numerical_mdlp::separation/near/s3::wide` | 3→3 | 12.4385 → 12.9155 | 11.8442 → 10.5804 |
| `ob_numerical_mdlp::imbalance/rate_1pct/s1::default` | 4→3 | 1.8470 → 1.7684 | 2.8063 → 2.0329 |
| `ob_numerical_mdlp::imbalance/rate_1pct/s1::tight` | 3→2 | 1.7236 → 1.3557 | 2.4897 → 2.1962 |
| `ob_numerical_mdlp::imbalance/rate_1pct/s1::wide` | 5→3 | 1.8830 → 1.7684 | 2.4284 → 2.0329 |
| `ob_numerical_mdlp::imbalance/rate_1pct/s2::default` | 4→3 | 1.5258 → 1.4034 | 1.6281 → 0.8541 |
| `ob_numerical_mdlp::imbalance/rate_1pct/s2::tight` | 3→2 | 1.4292 → 1.4521 | 1.0652 → 0.1672 |
| `ob_numerical_mdlp::imbalance/rate_1pct/s2::wide` | 3→3 | 0.5859 → 1.4034 | 0.7034 → 0.8541 |
| `ob_numerical_mdlp::imbalance/rate_1pct/s3::default` | 4→3 | 2.6983 → 2.6254 | 1.2339 → 0.4374 |
| `ob_numerical_mdlp::imbalance/rate_1pct/s3::tight` | 3→2 | 2.5790 → 2.5503 | 1.3274 → 0.3184 |
| `ob_numerical_mdlp::imbalance/rate_1pct/s3::wide` | 5→3 | 2.7172 → 2.6254 | 1.2245 → 0.4374 |
| `ob_numerical_mdlp::imbalance/rate_02pct/s1::default` | 3→3 | 1.4269 → 0.6411 | 1.5609 → 0.9875 |
| `ob_numerical_mdlp::imbalance/rate_02pct/s1::tight` | 2→2 | 1.0795 → 0.5379 | 1.0924 → 0.5365 |
| `ob_numerical_mdlp::imbalance/rate_02pct/s1::wide` | 3→3 | 1.2369 → 0.6411 | 1.3718 → 0.9875 |
| `ob_numerical_mdlp::imbalance/rate_02pct/s2::default` | 3→3 | 1.8938 → 0.5934 | 0.1337 → 0.5367 |
| `ob_numerical_mdlp::imbalance/rate_02pct/s2::tight` | 2→2 | 0.5271 → 0.2752 | 0.0179 → 0.4397 |
| `ob_numerical_mdlp::imbalance/rate_02pct/s2::wide` | 3→3 | 1.8938 → 0.5934 | 0.1337 → 0.5367 |
| `ob_numerical_mdlp::imbalance/rate_02pct/s3::default` | 3→3 | 2.5684 → 1.8952 | 0.5514 → 0.2741 |
| `ob_numerical_mdlp::imbalance/rate_02pct/s3::tight` | 2→2 | 2.5008 → 1.1054 | 0.2452 → 0.0010 |
| `ob_numerical_mdlp::imbalance/rate_02pct/s3::wide` | 5→3 | 2.1185 → 1.8952 | 0.4431 → 0.2741 |
| `ob_numerical_mdlp::discrete/ties/s1::default` | 5→3 | 0.8570 → 0.7578 | 0.8617 → 0.8192 |
| `ob_numerical_mdlp::discrete/ties/s1::tight` | 3→3 | 0.7578 → 0.7578 | 0.8192 → 0.8192 |
| `ob_numerical_mdlp::discrete/ties/s1::wide` | 8→3 | 0.8738 → 0.7578 | 0.8948 → 0.8192 |
| `ob_numerical_mdlp::discrete/ties/s2::default` | 5→4 | 0.8178 → 0.8153 | 0.7217 → 0.6935 |
| `ob_numerical_mdlp::discrete/ties/s2::tight` | 3→3 | 0.7618 → 0.7618 | 0.6678 → 0.6678 |
| `ob_numerical_mdlp::discrete/ties/s2::wide` | 8→4 | 0.8413 → 0.8153 | 0.7732 → 0.6935 |
| `ob_numerical_mdlp::discrete/ties/s3::default` | 5→4 | 0.7999 → 0.7725 | 0.6296 → 0.6137 |
| `ob_numerical_mdlp::discrete/ties/s3::tight` | 3→3 | 0.7095 → 0.6935 | 0.5655 → 0.4954 |
| `ob_numerical_mdlp::discrete/ties/s3::wide` | 7→4 | 0.8328 → 0.7725 | 0.6617 → 0.6137 |
| `ob_numerical_mdlp::discrete/mass_at_zero/s1::default` | 3→3 | 0.1005 → 0.1272 | 0.1846 → 0.1590 |
| `ob_numerical_mdlp::discrete/mass_at_zero/s1::tight` | 3→2 | 0.1005 → 0.1274 | 0.1846 → 0.1454 |
| `ob_numerical_mdlp::discrete/mass_at_zero/s1::wide` | 3→3 | 0.1005 → 0.1272 | 0.1846 → 0.1590 |
| `ob_numerical_mdlp::discrete/mass_at_zero/s2::default` | 3→3 | 0.0970 → 0.1160 | 0.1495 → 0.1529 |
| `ob_numerical_mdlp::discrete/mass_at_zero/s2::tight` | 2→2 | 0.0884 → 0.1161 | 0.1192 → 0.1534 |
| `ob_numerical_mdlp::discrete/mass_at_zero/s2::wide` | 3→3 | 0.0970 → 0.1160 | 0.1495 → 0.1529 |
| `ob_numerical_mdlp::discrete/mass_at_zero/s3::default` | 3→3 | 0.0772 → 0.1326 | 0.0476 → 0.1011 |
| `ob_numerical_mdlp::discrete/mass_at_zero/s3::tight` | 2→2 | 0.0593 → 0.0852 | 0.0103 → 0.0710 |
| `ob_numerical_mdlp::discrete/mass_at_zero/s3::wide` | 3→3 | 0.0772 → 0.1326 | 0.0476 → 0.1011 |
| `ob_numerical_mdlp::scale/tiny/s1::default` | 5→4 | 0.8329 → 0.8192 | 0.8086 → 0.7799 |
| `ob_numerical_mdlp::scale/tiny/s1::tight` | 3→3 | 0.7446 → 0.7638 | 0.7368 → 0.7234 |
| `ob_numerical_mdlp::scale/tiny/s1::wide` | 8→4 | 0.8489 → 0.8192 | 0.8754 → 0.7799 |
| `ob_numerical_mdlp::scale/tiny/s2::default` | 5→4 | 0.8252 → 0.8553 | 0.7053 → 0.6477 |
| `ob_numerical_mdlp::scale/tiny/s2::tight` | 3→3 | 0.7440 → 0.7889 | 0.6800 → 0.6262 |
| `ob_numerical_mdlp::scale/tiny/s2::wide` | 7→4 | 0.8461 → 0.8553 | 0.7428 → 0.6477 |
| `ob_numerical_mdlp::scale/tiny/s3::default` | 5→4 | 0.7907 → 0.7933 | 0.5834 → 0.5997 |
| `ob_numerical_mdlp::scale/tiny/s3::tight` | 3→3 | 0.7236 → 0.6818 | 0.5378 → 0.5008 |
| `ob_numerical_mdlp::scale/tiny/s3::wide` | 8→4 | 0.8260 → 0.7933 | 0.6376 → 0.5997 |
| `ob_numerical_mdlp::scale/huge/s1::default` | 5→4 | 0.8329 → 0.8192 | 0.8086 → 0.7799 |
| `ob_numerical_mdlp::scale/huge/s1::tight` | 3→3 | 0.7446 → 0.7638 | 0.7368 → 0.7234 |
| `ob_numerical_mdlp::scale/huge/s1::wide` | 8→4 | 0.8489 → 0.8192 | 0.8754 → 0.7799 |
| `ob_numerical_mdlp::scale/huge/s2::default` | 5→4 | 0.8252 → 0.8553 | 0.7053 → 0.6477 |
| `ob_numerical_mdlp::scale/huge/s2::tight` | 3→3 | 0.7440 → 0.7889 | 0.6800 → 0.6262 |
| `ob_numerical_mdlp::scale/huge/s2::wide` | 7→4 | 0.8461 → 0.8553 | 0.7428 → 0.6477 |
| `ob_numerical_mdlp::scale/huge/s3::default` | 5→4 | 0.7907 → 0.7933 | 0.5834 → 0.5997 |
| `ob_numerical_mdlp::scale/huge/s3::tight` | 3→3 | 0.7236 → 0.6818 | 0.5378 → 0.5008 |
| `ob_numerical_mdlp::scale/huge/s3::wide` | 8→4 | 0.8260 → 0.7933 | 0.6376 → 0.5997 |
| `ob_numerical_mdlp::outlier/single_extreme/s1::default` | 5→4 | 0.8244 → 0.8159 | 0.7760 → 0.6989 |
| `ob_numerical_mdlp::outlier/single_extreme/s1::tight` | 3→3 | 0.7233 → 0.7631 | 0.6705 → 0.6157 |
| `ob_numerical_mdlp::outlier/single_extreme/s1::wide` | 8→4 | 0.8590 → 0.8159 | 0.8554 → 0.6989 |
| `ob_numerical_mdlp::outlier/single_extreme/s2::default` | 5→5 | 0.8801 → 0.9356 | 0.5772 → 0.6125 |
| `ob_numerical_mdlp::outlier/single_extreme/s2::tight` | 3→3 | 0.7484 → 0.7732 | 0.5375 → 0.5052 |
| `ob_numerical_mdlp::outlier/single_extreme/s2::wide` | 8→5 | 0.9148 → 0.9356 | 0.6496 → 0.6125 |
| `ob_numerical_mdlp::outlier/single_extreme/s3::default` | 5→5 | 0.7802 → 0.8242 | 0.8570 → 0.8421 |
| `ob_numerical_mdlp::outlier/single_extreme/s3::tight` | 3→3 | 0.6819 → 0.6936 | 0.7176 → 0.7294 |
| `ob_numerical_mdlp::outlier/single_extreme/s3::wide` | 8→5 | 0.8069 → 0.8242 | 0.9170 → 0.8421 |
| `ob_numerical_mdlp::missing/na_30pct/s1::default` | 3→3 | 0.7083 → 0.7344 | 0.6008 → 0.5855 |
| `ob_numerical_mdlp::missing/na_30pct/s1::tight` | 3→3 | 0.7083 → 0.7344 | 0.6008 → 0.5855 |
| `ob_numerical_mdlp::missing/na_30pct/s1::wide` | 4→3 | 0.7121 → 0.7344 | 0.6849 → 0.5855 |
| `ob_numerical_mdlp::missing/na_30pct/s2::default` | 5→4 | 0.8165 → 0.8249 | 0.7286 → 0.6922 |
| `ob_numerical_mdlp::missing/na_30pct/s2::tight` | 3→3 | 0.7264 → 0.7399 | 0.6212 → 0.6465 |
| `ob_numerical_mdlp::missing/na_30pct/s2::wide` | 6→4 | 0.8275 → 0.8249 | 0.7353 → 0.6922 |
| `ob_numerical_mdlp::missing/na_30pct/s3::default` | 4→3 | 0.7417 → 0.7354 | 0.6251 → 0.6314 |
| `ob_numerical_mdlp::missing/na_30pct/s3::tight` | 3→3 | 0.7115 → 0.7354 | 0.6245 → 0.6314 |
| `ob_numerical_mdlp::missing/na_30pct/s3::wide` | 6→3 | 0.8050 → 0.7354 | 0.7074 → 0.6314 |
| `ob_numerical_mdlp::size/small_n40/s1::default` | 3→3 | 0.7123 → 0.6097 | 2.2139 → 1.4364 |
| `ob_numerical_mdlp::size/small_n40/s1::tight` | 2→2 | 0.7214 → 0.4667 | 1.2694 → 1.9251 |
| `ob_numerical_mdlp::size/small_n40/s1::wide` | 3→3 | 0.7123 → 0.6097 | 2.2139 → 1.4364 |
| `ob_numerical_mdlp::size/small_n40/s2::default` | 3→3 | 0.9699 → 0.1013 | 1.8442 → 0.4538 |
| `ob_numerical_mdlp::size/small_n40/s2::tight` | 2→2 | 0.6160 → 0.0742 | 1.0666 → 0.6035 |
| `ob_numerical_mdlp::size/small_n40/s2::wide` | 3→3 | 0.6319 → 0.1013 | 1.8442 → 0.4538 |
| `ob_numerical_mdlp::size/small_n40/s3::default` | 3→3 | 0.8933 → 0.6963 | 2.6901 → 2.6901 |
| `ob_numerical_mdlp::size/small_n40/s3::tight` | 3→2 | 1.2608 → 0.7070 | 2.6901 → 2.7916 |
| `ob_numerical_mdlp::size/small_n40/s3::wide` | 3→3 | 0.8323 → 0.6963 | 1.7881 → 2.6901 |
| `ob_numerical_mdlp::size/n60/s1::default` | 3→3 | 0.5706 → 0.7735 | 0.9598 → 1.4822 |
| `ob_numerical_mdlp::size/n60/s1::tight` | 2→2 | 0.5489 → 0.4272 | 1.0909 → 0.6712 |
| `ob_numerical_mdlp::size/n60/s1::wide` | 3→3 | 0.5274 → 0.7735 | 1.1571 → 1.4822 |
| `ob_numerical_mdlp::size/n60/s2::default` | 4→3 | 0.9975 → 0.3954 | 0.6296 → 0.3879 |
| `ob_numerical_mdlp::size/n60/s2::tight` | 3→2 | 1.0159 → 0.3066 | 0.5664 → 0.3084 |
| `ob_numerical_mdlp::size/n60/s2::wide` | 4→3 | 0.9975 → 0.3954 | 0.6296 → 0.3879 |
| `ob_numerical_mdlp::size/n60/s3::default` | 3→3 | 1.2457 → 0.5154 | 0.3349 → 0.2122 |
| `ob_numerical_mdlp::size/n60/s3::tight` | 2→2 | 1.1350 → 0.4272 | 0.1247 → 0.1386 |
| `ob_numerical_mdlp::size/n60/s3::wide` | 3→3 | 1.2457 → 0.5154 | 0.3349 → 0.2122 |
