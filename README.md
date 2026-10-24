# Byproduct Lab

A local research prototype for screening protein-rich Indian food-processing by-products. It combines sample traceability, assay-entry workflows, evidence-based treatment hypotheses, a batch-held-out NIR protein calibration pilot, and an auditable process-cost comparison.

## Run locally

```powershell
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:8000>. SQLite data is stored in `byproduct_lab.db` beside `app.py`. To use another database path, set `BYPRODUCT_DB_PATH` before starting the server.

Run the tests with:

```powershell
python -m pytest
```

## Import and train from `dataset`

Import the complete `dataset` folder into SQLite and train the separate cereal-grain protein research model:

```powershell
python dataset_training.py
```

The importer stores each source file byte-for-byte in SQLite, and stores parsed CSV rows, labels, spectra, and wavelength axes in queryable tables. Re-running the command refreshes the imported records and model. View the stored-file, row, and model summary at `GET /api/dataset-training`.

The folder contains wheat, barley, and maize data rather than the Indian by-products handled by sample reports. The importer trains a cereal research model and, when enough compatible wheat samples exist, a separate wheat-only research model. Fits use protein-labelled FOSS NIRSystem 5000 spectra on the shared 1100–2498 nm / 2 nm axis. Small FOSS NIRSystem 6500 cohorts and workbook, archive, and notebook files are retained in SQLite, but are not mixed into these fits. The current broad cereal model has a negative grouped-holdout R² and is marked **not suitable for predictive use**. The wheat model is validated by holding out a collection year, not independently documented physical batches, so it also remains research-only. Neither model is installed as a by-product calibration or affects sample predictions. Material-specific by-product predictions still require matching by-product reference samples and instrument settings.

The **Model performance** page displays the stored wheat holdout's Three.js/WebGL measured-versus-predicted scatter, PLSR-vs-training-mean error bars, historical training/holdout protein ranges, and the older cereal-wide model metrics. It documents the SNV preprocessing, five-component PLSR, and year-grouped 20% holdout. The older cereal-wide model uses a different held-out material and is for context only; it is not a direct apples-to-apples benchmark. Illustrative instrument forms are also shown for the named FOSS NIRSystem 5000 and 6500 data sources; they are not manufacturer-accurate models or live devices.

## Research workflow

1. Record the by-product, supplier, batch, and collection date.
2. Enter laboratory or validated rapid-test results. Material-specific priorities are prompts for testing, not detected compounds.
3. Keep composition results separate from food-safety checks.
4. For a sample NIR test, upload the instrument's exported wavelength/value CSV or enter the ordered values and wavelengths manually. The app previews the spectrum and checks its wavelength axis against the active calibration before predicting protein.
5. Import paired NIR spectra and reference protein labels to train a material-specific PLSR calibration. The pilot uses a held-out split by independent batch and reports MAE, RMSE, R², and a lab-reference-versus-prediction plot.
6. Compare treatment hypotheses using confirmatory assays and controlled sensory evaluation.
7. Compare process costs only when dry-feedstock cost, process cost, dry-matter yield, and post-process protein are measured or otherwise justified.
8. Hold missing, positive, or uncertain safety checks for qualified review.

## Hardware & runs

The Overview **Analyze a sample** button opens the Hardware & runs page. Select a previously saved sample and choose **Run automated testing & generate report** to check every configured instrument type and save a run report to that sample's history. The report also includes any results already recorded for the sample. No instrument models, drivers, or connection SDKs are configured in this prototype, so devices are reported as **not connected**, with no measurement values fabricated. Save a sample first using Manual entry, then rerun automated testing when hardware is connected; otherwise fill missing values manually.

## AI research assistant (AWS Bedrock)

Manual intake and results are on their own **Manual entry** page, rather than the Overview landing page. Each submission is saved as a sample and report; opening that sample later shows the original manual values and latest saved AI review. Reports include a section jump bar, a clickable evidence-flow diagram, interactive Three.js/WebGL 3D charts for composition, economics, published ranges, and NIR spectra, and a usable-protein cost comparison when the required inputs are available. The **Estimates** section separately shows prior same-material summaries and cited public composition ranges. Estimated values never overwrite the sample's stored measurements; charts and report cards label measured, historical, and published values separately. Same-material summaries exclude the current sample and records explicitly marked demo or synthetic. Safety results are never estimated. Three.js is loaded as a pinned browser module from jsDelivr; chart values remain in the browser and are not sent to that CDN.

The navigation also includes **Compare samples**, a theoretical **Blend explorer**, a **Cost & treatment hub**, **Reference prices**, and a guided **Story mode**, along with read-only **Safety rules** and **Roadmap** pages. These are front-end additions over the existing sample/report endpoints; they do not add API routes or database tables. Comparison and blending use saved report values only. Blend calculations are temporary, use entered values without dry-matter correction, and retain the most concerning component status. Cost scenarios and comparator quotes are browser-session calculations and are not saved. Story mode can be printed or saved as PDF from the browser. Safety guidance is read-only and deliberately does not import or apply thresholds from the other project.

The **Molecule & matrices** page adds the eight compound profiles and qualitative sorting options from the supplied concept reference, alongside the five supported Byproduct Lab materials. Reference properties, identifiers, sensory and binding hypotheses, and treatment suggestions are explicitly unverified. Three.js renders an interactive ball-and-stick illustration from each reference SMILES string, but it is not an experimentally determined conformer, stereochemical model, or validated geometry. Phytase and esterase are shown as generic conceptual protein folds, not sequence-derived enzyme structures or recommendations for a specific product. Available instruments have interactive, illustrative equipment form factors, not manufacturer-specific models. Material links are screening context only, never detections. The sorting matrix reports qualitative resource bands, not estimated removal or profitability. The profit matrix and chart use only economics calculated from saved sample inputs; missing inputs remain blank, and demo/synthetic sample records are excluded. These views use existing read APIs and do not alter the API, SQLite schema, or sample-entry workflow.

Public references include Feedipedia composition summaries for guar meal, groundnut/peanut meal, rice-bran subtypes, and chickpea bran/chuni, plus an open-access almond extraction study. Groundnut and rice-bran ranges are shown by published material class rather than pooled across different forms. Chickpea chuni is only a small-sample proxy for that specific chickpea by-product, not a generic pulse-residue estimate. No numeric almond press-cake estimate is shown where the source does not support one. Feedipedia is feed-ingredient context, not a human-food specification, safety result, or guarantee for a particular batch; analytical methods and processing history can differ.

Open a saved sample report and choose **Run AI evidence review** to run the bounded agent. It uses AWS Bedrock Converse tool calling to inspect recorded evidence, retrieve the application's treatment hypotheses, and calculate process economics using the same deterministic formula as the report. Where an economic input is missing, it may use the median of that input from earlier saved samples of the same by-product, and each such value is explicitly labelled with its historical source and marked as an assumption. These are scenario estimates, not new measurements. If no same-material history is available, it lists missing values and does not fabricate them. Rough protein potential in the supplied notes is reference context only, not a substitute for post-process protein data.

When costs, dry-matter yield, post-process protein, and sale price are available from the current sample or labelled historical assumptions, the report calculates cost per kg usable protein, projected revenue per kg dry feedstock, gross margin, and return on entered costs. The return is a partial scenario and excludes overheads, taxes, packaging, and logistics not entered. Enter a sale price to calculate revenue and returns. Existing material-specific NIR calibration remains responsible for protein predictions; AI does not generate a replacement prediction. If historical assumptions do not fit the current batch, record measured values manually or use a configured compatible instrument and re-run the assessment.

The assessment is saved with the sample. On Windows PowerShell, configure AWS access before starting the app:

```powershell
$env:AWS_PROFILE = "aws-toolkit"
$env:AWS_REGION = "us-east-1"
$env:BYPRODUCT_BEDROCK_MODEL_ID = "us.anthropic.claude-sonnet-4-6"
python app.py
```

Install the Python dependencies with `python -m pip install -r requirements.txt`; `botocore[crt]` is required for AWS CLI login-based profiles. The app uses `AWS_PROFILE` when set, or automatically selects the configured `aws-toolkit` profile if present, and otherwise uses the default AWS SDK credential chain. It does not store AWS credentials in the code or database. The selected model must be enabled for the AWS account and Region and support the Bedrock Converse API with tool use. When the user explicitly starts an assessment, the app sends the by-product material, reported measurements, findings, safety-test statuses, and model/economics results to Bedrock. It omits sample ID, supplier, batch ID, and raw NIR scans. Do not send data to a provider unless your lab permits it.

The agent has read-only tools and a bounded tool-call loop; it cannot edit sample records, operate devices, or approve food use. Its narrative is not a substitute for reference assays. It must not invent missing measurements, infer safety from NIR, or treat known risks as detections. AWS permissions, service availability, model access, and network connectivity are required; errors are shown to the user.

The **AI workspace** and floating star help button use the same Bedrock assistant. It can summarize workspace counts, search/filter/sort saved reports, find similar reports using same-material and recorded-evidence overlap, summarize imported cereal dataset labels by cohort, explain stored model performance, and retrieve documented treatment hypotheses. Related saved reports open from result cards. Add new samples through Manual entry; the assistant itself is read-only and cannot save/edit records, train models, trigger instrument runs, or certify food safety. For a chat request, the app sends the user message and relevant anonymized report/model summaries to Bedrock; database record IDs, supplier names, batch IDs, and raw spectra are omitted from tool-result summaries. Since the user's message and chat history are sent as entered, avoid typing identifying details. Chat history stays in page memory, clears on reload, and is not saved to SQLite.

## Calibration CSV

Use one independent sample per row, with columns named `sample_id`, `batch_id`, `protein_pct`, and `scan_1`, `scan_2`, `scan_3`, and so on. To enforce instrument wavelength matching, also include `wavelength_1_nm`, `wavelength_2_nm`, and matching wavelength columns for each scan feature. Every sample must use the same ordered wavelength axis. Protein labels should be from reference laboratory measurements on a dry-matter basis. Repeated scans of one physical sample are not independent samples. The prototype requires at least 30 samples across 5 batches before fitting a pilot model; the project notes recommend aiming for 150–300 independent samples and a 20% independent test set. Passing the pilot gate is not proof of scientific validity.

NIR sample files can use a two-column CSV or tab-separated format with `wavelength_nm` and `reflectance` headers. `intensity`, `absorbance`, `signal`, or `value` can be used instead of `reflectance`. The instrument's export is uploaded to the browser; direct live USB acquisition is not implemented because device communication depends on the spectrometer model and its SDK.

## Important limitations

- This app does not contain a validated spectroscopy dataset, benchmark process costs, or food-safety limits.
- A protein model is available only after the user supplies a material-specific reference dataset. Its estimate is research screening output, not a validated food ingredient claim.
- A spectrum is only scored when its feature count and (if provided) wavelength axis match the calibration. Calibration and sample scans must use compatible instrument settings and preprocessing.
- The plotted held-out comparison and MAE/RMSE/R² describe the pilot split only; they are not a prediction interval or independent external validation.
- NIR protein predictions do not test or clear aflatoxins, pathogens, allergens, or anti-nutritional factors.
- Treatment suggestions are hypotheses. Confirm compound reduction, digestibility, flavour, process yield, and resource use with suitable reference methods.
- Economics use user-entered inputs and exclude costs not supplied, such as laboratory testing, packaging, logistics, and capital equipment.
- Sensory evaluation should happen only after safety clearance. A qualified food-safety reviewer makes the final suitability decision.
