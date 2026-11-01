> **Note (October 2026).** The code and data supporting the article *ECLIPSE v2.0* (Scientific Reports, DOI [10.1038/s41598-026-74760-5](https://doi.org/10.1038/s41598-026-74760-5)) correspond to commit `93339b4` (tag `scientific-reports-2026`), archived at Zenodo ([10.5281/zenodo.22760496](https://doi.org/10.5281/zenodo.22760496)), and remain unchanged in `ScientificReports_revision/`. Further development of ECLIPSE takes place in a separate repository, which will be linked here when it is released.

> **Upcoming software update: ECLIPSE 3 (core 3.2).** An updated version of the ECLIPSE software will be released in the repository mentioned above; it is already archived at Zenodo ([10.5281/zenodo.23599462](https://doi.org/10.5281/zenodo.23599462)). It keeps the computations used in the article unchanged — the Eclipse Integrity Score (EIS), the Standardized Test-set Discrepancy Score (STDS) and the code auditor give the same results, and every deposited numerical output of the published studies reproduces identically — and improves the framework's mechanisms so that they do what the article describes:
>
> - **Split record (Stage 1).** The SHA-256 hash covers the development/holdout assignment, the seed, the split ratios and the sample count, not only the seed and the set of identifiers. The split no longer depends on the order of the input, and duplicate identifiers are rejected.
> - **Preregistration (Stage 2).** The criteria hash also covers the registration date and the project name, and registering criteria after validation is refused. A receipt of the hashes (`get_commitment_receipt()`) can be deposited with an external time-stamping service, since local timestamps do not by themselves establish temporal precedence.
> - **Development (Stage 3).** Requires Stage 2. It can check that the identifiers supplied belong to the development set (holdout identifiers are refused), and it trains the final model on all development data.
> - **Single-shot validation (Stage 4).** A lock is written before validation runs, and a second attempt is refused, including after a failed attempt and from a new session. The SHA-256 of the holdout file can be recorded and the file set read-only.
> - **Final assessment (Stage 5).** Requires a completed Stage 4 and takes the development and holdout metrics from the stored records, so they cannot be replaced at this stage. The code auditor can be run as part of it.
> - **Audit trail.** Every stage is recorded in an append-only log, each entry hash-chained to the previous one. Overrides (`force=True`) remain possible, but the previous record is archived and the override is logged and reported.
> - **Integrity check.** `verify_integrity()` also covers the validation record, the final result and the log, not only the split and criteria files.
> - **Code auditor** labelled experimental, given its 0% specificity on legitimate near-miss scripts (Study 3). Its computation is unchanged.
>
> Like any locally stored record, these files can be rewritten by whoever controls them; depositing the receipt externally is what establishes timing. The commit cited in the article (`93339b4`) remains the reference for the published results and will not be modified.


This repository collects the author's research across several projects.

For the Scientific Reports manuscript "ECLIPSE v2.0: A Reproducible Scaffold for Severe Confirmatory Testing through Preregistration, Single-Shot Validation, and Static-Analysis Screening" (C. A. Sjöberg Tala), all code, data, and reproducibility materials are in:

→ /ScientificReports_revision

For the Neuroscience of Consciousness manuscript "La convergencia temporal talámica intralaminar como mecanismo candidato de la presencia fenomenológica" (C. A. Sjöberg Tala), the reviewer-facing package (manuscript, formal model, code, and results) is in:

→ /NeuroscienceOfConsciousness_revision

See the README inside each directory for layout and reproduction steps.

Other directories (e.g. Lab/) contain the author's earlier and separate exploratory research archive from which the two packages above were curated, and are not part of either manuscript.

License: Apache License 2.0 (see LICENSE).