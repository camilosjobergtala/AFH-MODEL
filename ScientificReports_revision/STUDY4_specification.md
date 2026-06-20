# Study 4 — Pre-specified model specification and cited external evidence

This file is the repository artifact referenced in the manuscript's Data Availability statement for Study 4. Study 4 is **not** an executed retrospective EIS regression: no available retrospective corpus can validly compute the EIS while also supplying replication/generalization outcomes (see manuscript §3.4 and §4.4). Study 4 therefore consists of (Part 1) a pre-registered specification for prospective execution, and (Part 2) cited published external evidence used as the closest available proxy. Both are reproduced here so the claim is verifiable without running anything.

---

## Part 1 — Pre-specified nested model specification (prospective)

Test of Reviewer 1's Major 3 concern: does the EIS add predictive value **beyond binary preregistration**? To be estimated prospectively on studies carrying ECLIPSE-computable artifacts (formal splits, holdout records, stage adherence) **together with** known replication/generalization outcomes.

Outcome: `replication` (binary: replicated / did not replicate, by the prospective study's pre-declared criterion).

| Model | Specification | Purpose |
|---|---|---|
| A | `replication ~ binary_preregistration` | Baseline. |
| B | `replication ~ EIS_total` | EIS alone. |
| C | `replication ~ binary_preregistration + EIS_without_preregistration` | Isolates the contribution of EIS dimensions **other than** preregistration — the specification explicitly requested by Reviewer 1 (Major 3). |
| D | `replication ~ binary_preregistration + EIS_without_preregistration + publication_year_or_era` | Controls for confounding by era / open-science culture. **Primary specification of interest.** |
| E (exploratory) | `replication ~ binary_preregistration + (EIS components individually) + original_effect_size + sample_size`, penalized regression as sensitivity check given the events-per-predictor ratio | Component-level exploration. |

Pre-specified outputs: odds ratios with 95% CIs; AUC; Nagelkerke R²; likelihood-ratio tests comparing nested models; multicollinearity assessment. **All specifications will be reported regardless of outcome.** The test is not executed here because no current retrospective corpus has both EIS-computable artifacts and known outcomes; computing EIS by imputing absent dimensions as constants is not a measurement and is explicitly avoided.

---

## Part 2 — Cited external evidence (proxy: non-preregistration methodological quality → generalization)

Used as the closest available published proxy that recorded methodological quality predicts whether reported performance holds up at independent external validation. This is an **EIS-analog** (PROBAST/TRIPOD-style quality), **not** EIS, and the corpora have essentially no preregistration variance — so it partially addresses, but does not resolve, the incremental-validity question.

| Reference | Domain / corpus | Design | Scale | Finding |
|---|---|---|---|---|
| Venema et al. (2021), *J Clin Epidemiol* 138:32–39, doi:10.1016/j.jclinepi.2021.06.017 | Clinical prediction models, Tufts PACE registry | PROBAST risk-of-bias → discrimination change at external validation | 556 models, 1,147 external validations | High-risk-of-bias models show systematically poorer external discrimination. |
| Retel Helmrich et al. (2022), *Diagn Progn Res* 6:8, doi:10.1186/s41512-022-00122-0 | Traumatic-brain-injury prediction models | PROBAST quality → development-to-validation discrimination change | TBI model set | Reproduces the quality→generalization relationship. |

Manuscript reference numbers: [26] Venema et al. (2021); [27] Retel Helmrich et al. (2022).