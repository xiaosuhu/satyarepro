from __future__ import annotations

import json
from typing import Any

import pandas as pd

from satyarepro.types import ToolSchema

from ..base import Tool

_CA_RANGE = (0, 3)
_THAL_RANGE = (1, 3)


class DataProfiler(Tool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="data_profiler",
            description=(
                "Profile a CSV dataset locally without any LLM calls. Reports basic "
                "statistics, missing values, class imbalance, demographic distributions "
                "(age/sex), IQR-based outliers, and consistency checks for known "
                "clinical columns (ca, thal)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to a CSV file to profile.",
                    },
                    "target_col": {
                        "type": "string",
                        "description": "Name of the target/label column (default: 'target').",
                    },
                },
                "required": ["path"],
            },
        )

    async def execute(self, path: str, target_col: str = "target") -> str:  # type: ignore[override]
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

        result: dict[str, Any] = {}

        # 1. Basic stats
        result["basic"] = {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        }

        # 2. Missing values
        missing = df.isna().sum()
        result["missing"] = {
            col: {
                "count": int(missing[col]),
                "ratio": round(float(missing[col]) / len(df), 4),
            }
            for col in df.columns
            if missing[col] > 0
        }
        result["missing_total"] = int(missing.sum())

        # 3. Class imbalance
        if target_col in df.columns:
            counts = df[target_col].value_counts().sort_index()
            max_count = int(counts.max())
            min_count = int(counts.min())
            result["class_imbalance"] = {
                "column": target_col,
                "distribution": {str(k): int(v) for k, v in counts.items()},
                "imbalance_ratio": round(max_count / min_count, 4) if min_count > 0 else None,
            }
        else:
            result["class_imbalance"] = {"error": f"Column '{target_col}' not found"}

        # 4. Demographic distributions (age / sex)
        demo: dict[str, Any] = {}
        if "sex" in df.columns:
            sex_counts = df["sex"].value_counts().sort_index()
            demo["sex"] = {
                "distribution": {str(k): int(v) for k, v in sex_counts.items()},
                "ratios": {
                    str(k): round(int(v) / len(df), 4)
                    for k, v in sex_counts.items()
                },
                "labels": {
                    str(k): ("male" if float(k) == 1.0 else "female")
                    for k in sex_counts.index
                },
            }
        if "age" in df.columns:
            age = df["age"].dropna()
            demo["age"] = {
                "min": round(float(age.min()), 2),
                "q25": round(float(age.quantile(0.25)), 2),
                "median": round(float(age.median()), 2),
                "mean": round(float(age.mean()), 2),
                "q75": round(float(age.quantile(0.75)), 2),
                "max": round(float(age.max()), 2),
            }
        result["demographics"] = demo

        # 5. Outliers — IQR method per numeric column
        numeric_cols = df.select_dtypes(include="number").columns
        outliers: dict[str, Any] = {}
        for col in numeric_cols:
            series = df[col].dropna()
            if series.empty:
                continue
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            n_out = int(((series < lower) | (series > upper)).sum())
            outliers[col] = {
                "q1": round(q1, 4),
                "q3": round(q3, 4),
                "iqr": round(iqr, 4),
                "lower_fence": round(lower, 4),
                "upper_fence": round(upper, 4),
                "outlier_count": n_out,
                "outlier_ratio": round(n_out / len(series), 4),
            }
        result["outliers"] = outliers

        # 6. Data consistency — known clinical column ranges
        consistency: dict[str, Any] = {}
        if "ca" in df.columns:
            ca_lo, ca_hi = _CA_RANGE
            ca_valid = df["ca"].dropna()
            n_bad = int(((ca_valid < ca_lo) | (ca_valid > ca_hi)).sum())
            consistency["ca"] = {
                "expected_range": [ca_lo, ca_hi],
                "out_of_range_count": n_bad,
            }
        if "thal" in df.columns:
            thal_lo, thal_hi = _THAL_RANGE
            thal_valid = df["thal"].dropna()
            n_bad = int(((thal_valid < thal_lo) | (thal_valid > thal_hi)).sum())
            consistency["thal"] = {
                "expected_range": [thal_lo, thal_hi],
                "out_of_range_count": n_bad,
            }
        result["consistency"] = consistency

        return json.dumps(result, indent=2)
