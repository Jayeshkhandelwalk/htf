from __future__ import annotations

import csv
import hashlib
import io
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

from app import connect_db, initialize_db, standard_normal_variate, utc_now


ROOT = Path(__file__).parent
DATASET_DIR = ROOT / "dataset"
MODEL_ID = "sensaifood-cereal-protein-research-v1"
WHEAT_MODEL_ID = "sensaifood-wheat-protein-research-v1"
EXPECTED_INSTRUMENT = "foss_nirsystem5000"
EXPECTED_WAVELENGTHS = np.arange(1100, 2500, 2, dtype=float)


def _finite_number(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if np.isfinite(number) else None


def _read_csv_rows(path: Path, content: bytes) -> list[dict]:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path.name} is not a UTF-8 CSV file.") from error
    reader = csv.DictReader(io.StringIO(text))
    fields = reader.fieldnames or []
    spectral_fields = []
    for field in fields:
        try:
            wavelength = float(field)
        except (TypeError, ValueError):
            continue
        if np.isfinite(wavelength):
            spectral_fields.append((wavelength, field))
    spectral_fields.sort()
    wavelengths = np.asarray([item[0] for item in spectral_fields], dtype=float)
    parsed_rows = []
    for row_number, raw in enumerate(reader, start=2):
        spectrum_values = [_finite_number(raw.get(field)) for _, field in spectral_fields]
        spectrum = (
            [float(value) for value in spectrum_values]
            if spectrum_values and all(value is not None for value in spectrum_values)
            else None
        )
        protein = _finite_number(raw.get("Protein"))
        if protein is not None and not 0 < protein <= 100:
            protein = None
        moisture = _finite_number(raw.get("Moisture"))
        cereal = (raw.get("Cereal") or "").strip()
        year = (raw.get("Year") or "").strip()
        instrument = (raw.get("Spectrometer") or "").strip()
        reasons = []
        if protein is None:
            reasons.append("Missing or invalid reference protein label.")
        if spectrum is None:
            reasons.append("Spectrum contains missing or invalid values.")
        if instrument.casefold() != EXPECTED_INSTRUMENT:
            reasons.append("Instrument differs from the FOSS NIRSystem 5000 training cohort.")
        if len(wavelengths) != len(EXPECTED_WAVELENGTHS) or not np.allclose(
            wavelengths, EXPECTED_WAVELENGTHS, rtol=0, atol=0.01
        ):
            reasons.append("Wavelength axis differs from the 1100-2498 nm, 2 nm cohort.")
        parsed_rows.append(
            {
                "row_number": row_number,
                "sample_id": (raw.get("ID") or "").strip() or None,
                "material": cereal or None,
                "instrument": instrument or None,
                "batch_group": f"{cereal.casefold()}:{year}" if cereal and year else None,
                "protein_pct": protein,
                "moisture_pct": moisture,
                "wavelengths": wavelengths.tolist() if len(wavelengths) else None,
                "spectrum": spectrum,
                "raw": raw,
                "training_eligible": not reasons,
                "exclusion_reason": "; ".join(reasons) or None,
            }
        )
    return parsed_rows


def import_and_train_dataset(
    dataset_dir: Path = DATASET_DIR,
    database_path: Path | None = None,
) -> dict:
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")
    paths = sorted(path for path in dataset_dir.rglob("*") if path.is_file())
    if not paths:
        raise ValueError(f"No files found in dataset folder: {dataset_dir}")

    file_records = []
    row_records = []
    training_rows = []
    for path in paths:
        content = path.read_bytes()
        relative_path = path.relative_to(dataset_dir).as_posix()
        file_records.append(
            {
                "path": relative_path,
                "sha256": hashlib.sha256(content).hexdigest(),
                "format": path.suffix.lower().lstrip(".") or "unknown",
                "byte_size": len(content),
                "content": content,
            }
        )
        if path.suffix.lower() != ".csv":
            continue
        parsed = _read_csv_rows(path, content)
        for row in parsed:
            row_records.append((relative_path, row))
            if row["training_eligible"]:
                training_rows.append((relative_path, row))

    if len(training_rows) < 30:
        raise ValueError(
            "At least 30 protein-labelled spectra from the compatible FOSS "
            "NIRSystem 5000 wavelength cohort are required."
        )
    sample_ids = [row["sample_id"] for _, row in training_rows]
    if None in sample_ids or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Eligible training spectra must have unique, non-empty sample IDs.")
    groups = np.asarray([row["batch_group"] for _, row in training_rows])
    if len(set(groups)) < 3:
        raise ValueError("Training data must span at least three cereal/year cohorts.")

    spectra = np.asarray([row["spectrum"] for _, row in training_rows], dtype=float)
    targets = np.asarray([row["protein_pct"] for _, row in training_rows], dtype=float)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_indices, test_indices = next(splitter.split(spectra, targets, groups))
    if len(test_indices) < 6 or set(groups[train_indices]) & set(groups[test_indices]):
        raise ValueError("The grouped validation split is too small or has overlapping cohorts.")

    transformed = standard_normal_variate(spectra)
    components = min(5, transformed.shape[1], len(train_indices) - 1)
    evaluation_model = PLSRegression(n_components=components)
    evaluation_model.fit(transformed[train_indices], targets[train_indices])
    predictions = evaluation_model.predict(transformed[test_indices]).ravel()
    baseline_predictions = np.full(
        len(test_indices), float(targets[train_indices].mean()), dtype=float
    )
    mae = float(mean_absolute_error(targets[test_indices], predictions))
    rmse = float(np.sqrt(mean_squared_error(targets[test_indices], predictions)))
    r2 = float(r2_score(targets[test_indices], predictions))

    final_model = PLSRegression(n_components=components)
    final_model.fit(transformed, targets)
    excluded_counts: dict[str, int] = {}
    for _, row in row_records:
        if row["exclusion_reason"]:
            reason = row["exclusion_reason"].split("; ", maxsplit=1)[0]
            excluded_counts[reason] = excluded_counts.get(reason, 0) + 1
    summary = {
        "model_id": MODEL_ID,
        "scope": "Cereal grain protein research only; not a by-product calibration.",
        "status": (
            "Grouped holdout failed (negative R²); not suitable for predictive use."
            if r2 < 0
            else "Exploratory research only; independent laboratory validation required."
        ),
        "instrument": "FOSS NIRSystem 5000",
        "wavelength_range_nm": [1100, 2498],
        "spectral_resolution_nm": 2,
        "preprocessing": "Standard normal variate (SNV)",
        "training_samples": len(training_rows),
        "training_cohorts": sorted(set(groups.tolist())),
        "holdout_samples": len(test_indices),
        "holdout_cohorts": sorted(set(groups[test_indices].tolist())),
        "features": int(transformed.shape[1]),
        "components": components,
        "mae_pp": round(mae, 3),
        "rmse_pp": round(rmse, 3),
        "r2": round(r2, 3) if np.isfinite(r2) else None,
        "baseline_comparison": {
            "baseline": "Training-set mean (predicts one constant for every holdout sample)",
            "model": {
                "mae_pp": round(mae, 3),
                "rmse_pp": round(rmse, 3),
                "r2": round(r2, 3) if np.isfinite(r2) else None,
            },
            "baseline_metrics": {
                "mae_pp": round(
                    float(mean_absolute_error(targets[test_indices], baseline_predictions)),
                    3,
                ),
                "rmse_pp": round(
                    float(
                        np.sqrt(
                            mean_squared_error(targets[test_indices], baseline_predictions)
                        )
                    ),
                    3,
                ),
                "r2": round(
                    float(r2_score(targets[test_indices], baseline_predictions)), 3
                ),
            },
        },
        "holdout_comparison": [
            {
                "sample_id": str(training_rows[index][1]["sample_id"]),
                "cohort": str(groups[index]),
                "measured": round(float(targets[index]), 3),
                "predicted": round(float(predicted), 3),
            }
            for index, predicted in zip(test_indices, predictions, strict=True)
        ],
        "imported_files": len(file_records),
        "imported_csv_rows": len(row_records),
        "excluded_rows": len(row_records) - len(training_rows),
        "exclusion_reasons": excluded_counts,
        "limitations": [
            "Source samples are wheat, barley, and maize; the dataset contains no Indian by-product reference samples.",
            "Only the FOSS NIRSystem 5000 1100-2498 nm cohort was used for fitting.",
            "Small FOSS NIRSystem 6500 cohorts and other files are retained in SQL but not mixed into this fit.",
            "The grouped holdout is a single exploratory cohort split, not external validation.",
            "A negative grouped-holdout R² indicates poor transfer to the held-out cereal cohort.",
        ],
    }
    wheat_rows = [
        (source_path, row)
        for source_path, row in row_records
        if row["material"] and row["material"].casefold() == "wheat"
        and row["training_eligible"]
    ]
    wheat_model = None
    if len(wheat_rows) >= 30:
        wheat_groups = np.asarray([row["batch_group"] for _, row in wheat_rows])
        if len(set(wheat_groups)) < 2:
            raise ValueError("Wheat training data must span at least two separate year cohorts.")
        wheat_spectra = np.asarray([row["spectrum"] for _, row in wheat_rows], dtype=float)
        wheat_targets = np.asarray([row["protein_pct"] for _, row in wheat_rows], dtype=float)
        wheat_splitter = GroupShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        )
        wheat_train_indices, wheat_test_indices = next(
            wheat_splitter.split(wheat_spectra, wheat_targets, wheat_groups)
        )
        if (
            len(wheat_test_indices) < 6
            or set(wheat_groups[wheat_train_indices]) & set(wheat_groups[wheat_test_indices])
        ):
            raise ValueError("The wheat year-cohort holdout is too small or overlaps training.")
        wheat_transformed = standard_normal_variate(wheat_spectra)
        wheat_components = min(5, wheat_transformed.shape[1], len(wheat_train_indices) - 1)
        wheat_evaluation = PLSRegression(n_components=wheat_components)
        wheat_evaluation.fit(
            wheat_transformed[wheat_train_indices], wheat_targets[wheat_train_indices]
        )
        wheat_predictions = wheat_evaluation.predict(
            wheat_transformed[wheat_test_indices]
        ).ravel()
        wheat_baseline_predictions = np.full(
            len(wheat_test_indices),
            float(wheat_targets[wheat_train_indices].mean()),
            dtype=float,
        )
        wheat_r2 = float(r2_score(wheat_targets[wheat_test_indices], wheat_predictions))
        wheat_final_model = PLSRegression(n_components=wheat_components)
        wheat_final_model.fit(wheat_transformed, wheat_targets)
        all_wheat_rows = [
            row
            for _, row in row_records
            if row["material"] and row["material"].casefold() == "wheat"
        ]
        wheat_training_mean = float(wheat_targets[wheat_train_indices].mean())
        wheat_model_metrics = {
            "mae_pp": float(
                mean_absolute_error(wheat_targets[wheat_test_indices], wheat_predictions)
            ),
            "rmse_pp": float(
                np.sqrt(mean_squared_error(wheat_targets[wheat_test_indices], wheat_predictions))
            ),
            "r2": wheat_r2,
        }
        wheat_baseline_metrics = {
            "mae_pp": float(
                mean_absolute_error(
                    wheat_targets[wheat_test_indices], wheat_baseline_predictions
                )
            ),
            "rmse_pp": float(
                np.sqrt(
                    mean_squared_error(
                        wheat_targets[wheat_test_indices], wheat_baseline_predictions
                    )
                )
            ),
            "r2": float(
                r2_score(wheat_targets[wheat_test_indices], wheat_baseline_predictions)
            ),
        }
        wheat_training_cohort_values = [
            row["protein_pct"]
            for _, row in wheat_rows
            if row["batch_group"] not in set(wheat_groups[wheat_test_indices].tolist())
        ]
        wheat_holdout_cohort_values = wheat_targets[wheat_test_indices].tolist()
        wheat_model = {
            "model_id": WHEAT_MODEL_ID,
            "scope": "Wheat grain protein research only; not a by-product calibration.",
            "status": (
                "Wheat year-cohort holdout failed (negative R²); not suitable for predictive use."
                if wheat_r2 < 0
                else "Exploratory wheat research only; independent laboratory validation required."
            ),
            "instrument": "FOSS NIRSystem 5000",
            "wavelength_range_nm": [1100, 2498],
            "spectral_resolution_nm": 2,
            "preprocessing": "Standard normal variate (SNV)",
            "training_samples": len(wheat_rows),
            "available_wheat_rows": len(all_wheat_rows),
            "excluded_wheat_rows": len(all_wheat_rows) - len(wheat_rows),
            "training_cohorts": sorted(set(wheat_groups.tolist())),
            "holdout_samples": len(wheat_test_indices),
            "holdout_cohorts": sorted(set(wheat_groups[wheat_test_indices].tolist())),
            "features": int(wheat_transformed.shape[1]),
            "components": wheat_components,
            "mae_pp": round(
                float(mean_absolute_error(
                    wheat_targets[wheat_test_indices], wheat_predictions
                )),
                3,
            ),
            "rmse_pp": round(
                float(np.sqrt(mean_squared_error(
                    wheat_targets[wheat_test_indices], wheat_predictions
                ))),
                3,
            ),
            "r2": round(wheat_r2, 3) if np.isfinite(wheat_r2) else None,
            "baseline_comparison": {
                "baseline": "Training-set mean (predicts one constant for every holdout sample)",
                "model": {key: round(value, 3) for key, value in wheat_model_metrics.items()},
                "baseline_metrics": {
                    key: round(value, 3) for key, value in wheat_baseline_metrics.items()
                },
            },
            "holdout_comparison": [
                {
                    "sample_id": str(wheat_rows[index][1]["sample_id"]),
                    "cohort": str(wheat_groups[index]),
                    "measured": round(float(wheat_targets[index]), 3),
                    "predicted": round(float(predicted), 3),
                    "baseline": round(wheat_training_mean, 3),
                }
                for index, predicted in zip(
                    wheat_test_indices, wheat_predictions, strict=True
                )
            ],
            "historical_cohort_comparison": [
                {
                    "cohort": "Training cohorts",
                    "cohorts": sorted(
                        set(wheat_groups[wheat_train_indices].tolist())
                    ),
                    "samples": len(wheat_training_cohort_values),
                    "mean_protein_pct": round(
                        float(np.mean(wheat_training_cohort_values)), 3
                    ),
                    "min_protein_pct": round(
                        float(np.min(wheat_training_cohort_values)), 3
                    ),
                    "max_protein_pct": round(
                        float(np.max(wheat_training_cohort_values)), 3
                    ),
                },
                {
                    "cohort": "Held-out historical cohort",
                    "cohorts": sorted(set(wheat_groups[wheat_test_indices].tolist())),
                    "samples": len(wheat_holdout_cohort_values),
                    "mean_protein_pct": round(
                        float(np.mean(wheat_holdout_cohort_values)), 3
                    ),
                    "min_protein_pct": round(
                        float(np.min(wheat_holdout_cohort_values)), 3
                    ),
                    "max_protein_pct": round(
                        float(np.max(wheat_holdout_cohort_values)), 3
                    ),
                },
            ],
            "limitations": [
                "Only wheat rows with matching FOSS NIRSystem 5000 1100-2498 nm spectra were used.",
                "Wheat2 and ground-wheat spectra use a different instrument and wavelength axis and were excluded.",
                "Validation holds out a collection year, not independently documented physical batches.",
                "This research model is stored separately and is not used for by-product predictions.",
            ],
        }
    else:
        wheat_final_model = None
    if database_path is not None:
        import app

        original_path = app.DB_PATH
        app.DB_PATH = Path(database_path)
    else:
        original_path = None
    try:
        initialize_db()
        with connect_db() as connection:
            for record in file_records:
                connection.execute(
                    """
                    INSERT INTO dataset_files (path, sha256, format, byte_size, content, imported_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(path) DO UPDATE SET
                        sha256 = excluded.sha256,
                        format = excluded.format,
                        byte_size = excluded.byte_size,
                        content = excluded.content,
                        imported_at = excluded.imported_at
                    """,
                    (
                        record["path"],
                        record["sha256"],
                        record["format"],
                        record["byte_size"],
                        record["content"],
                        utc_now(),
                    ),
                )
            for record in file_records:
                relative_path = record["path"]
                if Path(relative_path).suffix.lower() == ".csv":
                    connection.execute(
                        "DELETE FROM dataset_rows WHERE source_path = ?", (relative_path,)
                    )
            for source_path, row in row_records:
                connection.execute(
                    """
                    INSERT INTO dataset_rows (
                        source_path, row_number, sample_id, material, instrument, batch_group,
                        protein_pct, moisture_pct, wavelengths_json, spectrum_json, raw_json,
                        training_eligible, exclusion_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        source_path,
                        row["row_number"],
                        row["sample_id"],
                        row["material"],
                        row["instrument"],
                        row["batch_group"],
                        row["protein_pct"],
                        row["moisture_pct"],
                        json.dumps(row["wavelengths"]),
                        json.dumps(row["spectrum"]),
                        json.dumps(row["raw"], allow_nan=False),
                        int(row["training_eligible"]),
                        row["exclusion_reason"],
                    ),
                )
            connection.execute(
                """
                INSERT INTO dataset_models (model_id, created_at, summary, model)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(model_id) DO UPDATE SET
                    created_at = excluded.created_at,
                    summary = excluded.summary,
                    model = excluded.model
                """,
                (MODEL_ID, utc_now(), json.dumps(summary), pickle.dumps(final_model)),
            )
            if wheat_model is not None and wheat_final_model is not None:
                connection.execute(
                    """
                    INSERT INTO dataset_models (model_id, created_at, summary, model)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(model_id) DO UPDATE SET
                        created_at = excluded.created_at,
                        summary = excluded.summary,
                        model = excluded.model
                    """,
                    (
                        WHEAT_MODEL_ID,
                        utc_now(),
                        json.dumps(wheat_model),
                        pickle.dumps(wheat_final_model),
                    ),
                )
    finally:
        if original_path is not None:
            import app

            app.DB_PATH = original_path
    return {**summary, "wheat_model": wheat_model}


if __name__ == "__main__":
    result = import_and_train_dataset()
    print(json.dumps(result, indent=2))
