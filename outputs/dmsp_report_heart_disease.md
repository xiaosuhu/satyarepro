# NIH DMSP Compliance Report

**File:** `heart-disease-prediction-notebook.ipynb`
**Report Date:** 2026-05-30
**Prepared by:** NIH DMSP Compliance Officer

---

### 1. Data Description and Format

**NIH Requirement:** Scientific data, associated metadata, and file formats must be described, including data types (tabular, imaging, EHR) and expected volume.

**Status:** PARTIAL

**Evidence from Audit:**
- The file is loaded via a bare path (`path = "../input/heart-disease-prediction/heart_data.csv"`) with no accompanying description of data type (tabular, clinical), expected dimensions, or file format rationale; the only implicit volume reference is a runtime print statement: `print(f'The length of the data now is {len(data)} instead of 303!')`, which is not a formal metadata declaration.
- Column renaming (e.g., `'thal'` → `'thalassemia'`, `'cp'` → `'chest_pain_type'`) demonstrates partial clinical awareness of the 14-attribute Cleveland subset, but no data dictionary, codebook, or variable-level metadata (type, units, valid range) is provided anywhere in the notebook.
- The audit found no description of the broader 76-attribute UCI source dataset, no rationale for using only 14 attributes, and no file format specification (encoding, delimiter, schema version).

**Recommended Action:**
1. Add a dedicated provenance header cell explicitly declaring: data type (tabular clinical data), file format (`CSV`, UTF-8), expected dimensions (303 rows × 14 columns pre-cleaning; 297 × 14 post-cleaning), and the named dataset (UCI Heart Disease Dataset, Cleveland subset, Detrano et al., 1989).
2. Embed or link a formal data dictionary defining each of the 14 variables with clinical units, valid ranges, and encoding schemes (e.g., `thalassemia`: 1 = normal, 2 = fixed defect, 3 = reversible defect), replacing the current implicit column renaming as the sole documentation.
3. Replace the generic variable name `data` with a descriptive identifier such as `uci_heart_df` and add a programmatic shape assertion (e.g., `assert uci_heart_df.shape == (303, 14)`) to formally bind the data description to a verifiable, reproducible check.

---

### 2. Related Tools, Software, and Code

**NIH Requirement:** Software, code, and algorithms needed to access or reproduce the data and analyses must be identified, with version information.

**Status:** NON-COMPLIANT

**Evidence from Audit:**
- No library versions are declared, pinned, or logged anywhere in the notebook; packages including `pandas`, `scikit-learn`, `matplotlib`, and `seaborn` are imported without any `__version__` assertions or a companion `requirements.txt` / `environment.yml` file.
- The `LabelEncoder` objects fitted during preprocessing are never serialised (e.g., via `joblib.dump`), meaning the encoding transformation cannot be reproduced on new data; the audit explicitly flags this as making the preprocessing pipeline **non-reproducible**.
- The random seed (`seed = 0`) is set for `train_test_split` but is not propagated to all stochastic model constructors, and no environment snapshot (e.g., `pip freeze` output, conda lockfile, or Docker image reference) is recorded to reproduce the full computational environment.

**Recommended Action:**
1. Add an environment declaration cell at the top of the notebook that programmatically logs and asserts minimum required versions for all critical dependencies (e.g., `assert sklearn.__version__ >= "1.3.0"`), and commit a `requirements.txt` or `environment.yml` alongside the notebook in version control.
2. Serialise all fitted preprocessing objects (encoders, scalers) using `joblib.dump` with timestamped filenames, and document the full `sklearn` Pipeline object so that the transformation from raw CSV to model-ready arrays is fully reproducible from a single, saved artefact.
3. Adopt a provenance-tracking framework (e.g., **DVC**, **MLflow**, or **Weights & Biases**) to log data lineage, hyperparameters, random seeds, and model artefacts automatically, replacing the current ad-hoc and incomplete manual documentation.

---

### 3. Standards and Data Formats

**NIH Requirement:** Metadata standards, common data elements, ontologies, and terminology used to describe the data must be specified to enable interoperability.

**Status:** NON-COMPLIANT

**Evidence from Audit:**
- No controlled vocabulary, ontology, or terminology standard is referenced anywhere in the notebook; clinical concepts such as thalassemia type, chest pain classification, and ST-segment depression are represented only as integer-encoded Kaggle CSV columns with no mapping to standards such as **SNOMED CT**, **LOINC**, or **ICD-10**.
- There is no reference to community metadata standards applicable to clinical tabular data (e.g., **CDISC CDASH/SDTM**, **HL7 FHIR**, or **OMOP CDM**), and no schema file (e.g., JSON Schema, Frictionless Data `datapackage.json`) is provided to enable machine-readable interoperability.
- The binary target encoding (0/1 for heart disease presence) silently collapses the original 5-class ordinal angiographic scale with no citation to the encoding decision or mapping to a standard clinical outcome definition (e.g., **SNOMED CT** concept for coronary artery stenosis ≥ 50%).

**Recommended Action:**
1. Map all 14 clinical variables to recognised standard terminologies (e.g., LOINC codes for laboratory and physiological measurements, SNOMED CT for clinical findings such as thalassemia and angina type) and include this mapping as a machine-readable metadata file (e.g., a Frictionless Data `datapackage.json` or a `metadata.yaml`) committed alongside the notebook.
2. Document the target variable transformation explicitly, citing the original 5-class label definition from Detrano et al. (1989) and the binary recoding decision with its clinical reference (diameter narrowing > 50% in any major vessel = class 1), to ensure the outcome definition is interoperable with other heart disease datasets using the same standard.
3. Align the dataset's structural metadata with at least one recognised clinical data standard (e.g., CDISC CDASH for data element naming, or an OMOP CDM concept mapping) to support cross-study interoperability as required under the 2023 NIH DMSP policy.

---

### 4. Data Preservation, Access, and Timelines

**NIH Requirement:** The repository where scientific data will be archived, the timeline for deposit (no later than publication or end of award), and persistent identifiers (DOI) must be specified.

**Status:** NON-COMPLIANT

**Evidence from Audit:**
- No archival repository is named or planned anywhere in the notebook or its documentation; the sole data location is a transient Kaggle input path (`"../input/heart-disease-prediction/heart_data.csv"`), which is an ephemeral, session-specific path that does not constitute a persistent archive.
- No persistent identifier (DOI, ARK, or accession number) is assigned to or cited for either the dataset or the notebook itself; the UCI source dataset's canonical DOI is never referenced, and no plan to mint a DOI for derived outputs (e.g., via **Zenodo** or **Figshare**) is present.
- No deposit timeline, embargo period, or end-of-award data release commitment is stated, which directly violates the 2023 NIH DMSP requirement that data be made accessible no later than the time of an associated publication or the end of the performance period.

**Recommended Action:**
1. Designate a NIH-recognised repository for archiving both the analysis notebook and any derived data outputs (e.g., **Zenodo**, **Harvard Dataverse**, **OSF**, or an institutional repository), and document the planned deposit timeline (target: at or before manuscript submission) explicitly in a DMSP header cell or companion `README.md`.
2. Cite the canonical persistent identifier for the source dataset (UCI ML Repository DOI or the Detrano et al. DOI: `10.1016/0002-9149(89)90524-9`) and, upon notebook deposit, obtain and record a DOI for the notebook artefact itself to enable unambiguous citation and long-term access.
3. Record the exact download date of the Kaggle dataset and the Kaggle dataset version integer in the notebook metadata, and commit a SHA-256 checksum of `heart_data.csv` to the repository so that the precise data snapshot used in the analysis can be reconstructed independently of the Kaggle platform's availability.

---

### 5. Access, Distribution, and Reuse Considerations

**NIH Requirement:** Who may access the data, under what conditions (IRB, DUA, HIPAA), any embargo periods, and acceptable reuse terms must be specified.

**Status:** NON-COMPLIANT

**Evidence from Audit:**
- The audit found a **complete absence** of IRB documentation: no IRB approval number, no exemption determination citation (e.g., 45 CFR §46.104(d)(4)), and no institutional affiliation is recorded anywhere in the notebook, despite the data originating from identifiable cardiac patients at the Cleveland Clinic.
- No HIPAA compliance statement, de-identification attestation, or Safe Harbor certification (45 CFR §164.514(b)) is present; the notebook relies entirely on the implicit assumption that the Kaggle-hosted CSV is de-identified, without citing the entity that performed de-identification or the method used.
- No license, Data Use Agreement (DUA), or reuse terms are specified for either the input data or the notebook outputs; the Kaggle Terms of Service are not acknowledged, and no statement governs whether derived models or processed datasets may be redistributed, commercially used, or incorporated into downstream studies.

**Recommended Action:**
1. Add a mandatory data governance header cell that records: (a) IRB status and protocol number or formal exemption category with institutional name; (b) a de-identification attestation citing the method (Safe Harbor or Expert Determination per 45 CFR §164.514) and the responsible custodian (UCI ML Repository); and (c) an explicit Kaggle Terms of Service acknowledgment, treating this cell as a non-negotiable prerequisite before any NIH submission.
2. Specify a machine-readable license for all notebook outputs and any derived data (e.g., **CC BY 4.0** for processed datasets and figures, **MIT** or **Apache 2.0** for code), verify this is compatible with the UCI dataset's current license terms, and include the license file in the repository root.
3. Draft and attach a brief Data Use Statement that defines: who may access the outputs (unrestricted public access vs. credentialed access), any embargo period, prohibited uses (e.g., re-identification attempts), and a point-of-contact (PI name, institutional email, ORCID) for data governance inquiries, in full alignment with the 2023 NIH DMSP controlled-access framework requirements.

---

## Summary Scorecard

| Section | Title | Status |
|---------|-------|--------|
| 1 | Data Description and Format | PARTIAL |
| 2 | Related Tools, Software, and Code | NON-COMPLIANT |
| 3 | Standards and Data Formats | NON-COMPLIANT |
| 4 | Data Preservation, Access, and Timelines | NON-COMPLIANT |
| 5 | Access, Distribution, and Reuse Considerations | NON-COMPLIANT |

**Overall Compliance:** NON-COMPLIANT (0 of 5 sections fully compliant)

**Priority Actions:**
1. **Immediately add a data governance header cell** documenting IRB status (protocol number or 45 CFR §46.104(d)(4) exemption), HIPAA de-identification attestation, and Kaggle Terms of Service acknowledgment — this is a blocking deficiency that prevents any NIH submission.
2. **Formally identify the dataset and assign persistent identifiers**: explicitly name the UCI Heart Disease Dataset (Cleveland subset), cite Detrano et al. (1989) with DOI, record the Kaggle dataset version and download date, and commit a SHA-256 checksum of `heart_data.csv` to enable independent verification.
3. **Designate an archival repository and specify a deposit timeline**: name a NIH-recognised repository (e.g., Zenodo, Harvard Dataverse), commit to depositing no later than manuscript submission, serialise and deposit the full preprocessing pipeline with pinned dependency versions, and obtain a DOI for the notebook artefact.

---

*Generated by SatyaRepro · University of Michigan MIDAS · NIH DAIR3 (5R25GM151182-03)*