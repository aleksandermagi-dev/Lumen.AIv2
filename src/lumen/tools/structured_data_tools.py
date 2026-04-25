from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


def load_records(*, input_path: Path | None = None, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    params = params or {}
    inline_records = params.get("records")
    if isinstance(inline_records, list):
        return [_normalize_record(item) for item in inline_records if isinstance(item, dict)]
    if input_path is None:
        return []
    suffix = input_path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        delimiter = "," if suffix == ".csv" else "\t"
        with input_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            return [_normalize_record(dict(row)) for row in reader]
    if suffix == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [_normalize_record(item) for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            for key in ("records", "items", "rows", "data"):
                candidate = payload.get(key)
                if isinstance(candidate, list):
                    return [_normalize_record(item) for item in candidate if isinstance(item, dict)]
    return []


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    columns = sorted({key for record in records for key in record})
    numeric = numeric_columns(records)
    categorical = [column for column in columns if column not in numeric]
    numeric_summary: dict[str, dict[str, float | int]] = {}
    for column in numeric:
        values = numeric_values(records, column)
        if not values:
            continue
        numeric_summary[column] = {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.fmean(values),
        }
        if len(values) > 1:
            numeric_summary[column]["stdev"] = statistics.pstdev(values)
    categorical_summary: dict[str, dict[str, Any]] = {}
    for column in categorical:
        values = [str(record.get(column)).strip() for record in records if str(record.get(column)).strip()]
        if not values:
            continue
        counts: dict[str, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        top_values = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:5]
        categorical_summary[column] = {
            "distinct_count": len(counts),
            "top_values": [{"value": value, "count": count} for value, count in top_values],
        }
    return {
        "row_count": len(records),
        "column_count": len(columns),
        "columns": columns,
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
    }


def numeric_columns(records: list[dict[str, Any]]) -> list[str]:
    columns = sorted({key for record in records for key in record})
    numeric: list[str] = []
    for column in columns:
        values = [coerce_number(record.get(column)) for record in records if record.get(column) not in (None, "")]
        if values and all(value is not None for value in values):
            numeric.append(column)
    return numeric


def numeric_values(records: list[dict[str, Any]], column: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = coerce_number(record.get(column))
        if value is not None:
            values.append(value)
    return values


def correlation_payload(records: list[dict[str, Any]], columns: list[str] | None = None) -> dict[str, Any]:
    usable_columns = columns or numeric_columns(records)
    usable_columns = [column for column in usable_columns if column in numeric_columns(records)]
    pairs: list[dict[str, Any]] = []
    for index, left in enumerate(usable_columns):
        left_values = numeric_values(records, left)
        for right in usable_columns[index + 1 :]:
            right_values = numeric_values(records, right)
            paired = _paired_values(records, left, right)
            if len(paired) < 2:
                continue
            xs = [item[0] for item in paired]
            ys = [item[1] for item in paired]
            coefficient = pearson(xs, ys)
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "coefficient": coefficient,
                    "sample_size": len(paired),
                }
            )
    return {
        "column_count": len(usable_columns),
        "columns": usable_columns,
        "pairs": sorted(pairs, key=lambda item: abs(float(item["coefficient"])), reverse=True),
    }


def regression_payload(records: list[dict[str, Any]], x_column: str | None = None, y_column: str | None = None) -> dict[str, Any]:
    usable = numeric_columns(records)
    if x_column is None or y_column is None:
        if len(usable) < 2:
            return {
                "status": "error",
                "failure_category": "input_failure",
                "failure_reason": "Need at least two numeric columns for regression.",
                "runtime_diagnostics": {"runtime_ready": True, "input_ready": False},
            }
        x_column = usable[0]
        y_column = usable[1]
    paired = _paired_values(records, x_column, y_column)
    if len(paired) < 2:
        return {
            "status": "error",
            "failure_category": "input_failure",
            "failure_reason": "Need at least two paired numeric rows for regression.",
            "runtime_diagnostics": {"runtime_ready": True, "input_ready": False},
        }
    xs = [item[0] for item in paired]
    ys = [item[1] for item in paired]
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denominator = sum((value - x_mean) ** 2 for value in xs)
    if denominator == 0:
        return {
            "status": "error",
            "failure_category": "input_failure",
            "failure_reason": "Regression requires variation in the x column.",
            "runtime_diagnostics": {"runtime_ready": True, "input_ready": False},
        }
    slope = sum((x - x_mean) * (y - y_mean) for x, y in paired) / denominator
    intercept = y_mean - (slope * x_mean)
    predicted = [intercept + (slope * x) for x in xs]
    residual_sum = sum((actual - estimate) ** 2 for actual, estimate in zip(ys, predicted))
    total_sum = sum((actual - y_mean) ** 2 for actual in ys)
    r_squared = 1.0 - (residual_sum / total_sum) if total_sum else 1.0
    return {
        "status": "ok",
        "x_column": x_column,
        "y_column": y_column,
        "sample_size": len(paired),
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "predictions": [
            {x_column: x, y_column: y, "predicted": estimate}
            for x, y, estimate in zip(xs, ys, predicted)
        ],
        "runtime_diagnostics": {"runtime_ready": True, "input_ready": True},
    }


def cluster_payload(records: list[dict[str, Any]], columns: list[str] | None = None, cluster_count: int = 2) -> dict[str, Any]:
    usable = columns or numeric_columns(records)
    usable = usable[:2]
    if not usable:
        return {
            "status": "error",
            "failure_category": "input_failure",
            "failure_reason": "Clustering requires at least one numeric column.",
            "runtime_diagnostics": {"runtime_ready": True, "input_ready": False},
        }
    rows: list[tuple[dict[str, Any], list[float]]] = []
    for record in records:
        point = [coerce_number(record.get(column)) for column in usable]
        if all(value is not None for value in point):
            rows.append((record, [float(value) for value in point if value is not None]))
    if len(rows) < cluster_count:
        return {
            "status": "error",
            "failure_category": "input_failure",
            "failure_reason": "Not enough numeric rows to form clusters.",
            "runtime_diagnostics": {"runtime_ready": True, "input_ready": False},
        }
    centroids = [rows[index][1][:] for index in range(cluster_count)]
    assignments = [0 for _ in rows]
    for _ in range(6):
        for index, (_, point) in enumerate(rows):
            assignments[index] = min(
                range(cluster_count),
                key=lambda centroid_index: euclidean(point, centroids[centroid_index]),
            )
        for centroid_index in range(cluster_count):
            members = [point for assignment, (_, point) in zip(assignments, rows) if assignment == centroid_index]
            if not members:
                continue
            centroids[centroid_index] = [
                statistics.fmean(values)
                for values in zip(*members)
            ]
    cluster_rows: list[dict[str, Any]] = []
    for centroid_index in range(cluster_count):
        member_indices = [index for index, assignment in enumerate(assignments) if assignment == centroid_index]
        cluster_rows.append(
            {
                "cluster_id": centroid_index,
                "size": len(member_indices),
                "centroid": {column: centroids[centroid_index][i] for i, column in enumerate(usable)},
            }
        )
    return {
        "status": "ok",
        "columns": usable,
        "cluster_count": cluster_count,
        "clusters": cluster_rows,
        "runtime_diagnostics": {"runtime_ready": True, "input_ready": True},
    }


def build_chart_svg(
    *,
    title: str,
    points: list[dict[str, float]],
    x_label: str,
    y_label: str,
) -> str:
    width = 640
    height = 360
    margin = 40
    if not points:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
            f'<rect width="100%" height="100%" fill="#ffffff" />'
            f'<text x="24" y="40" font-family="Arial" font-size="18">{escape_xml(title)}</text>'
            f'<text x="24" y="80" font-family="Arial" font-size="14">No plottable data.</text>'
            f"</svg>"
        )
    xs = [point["x"] for point in points]
    ys = [point["y"] for point in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_span = (x_max - x_min) or 1.0
    y_span = (y_max - y_min) or 1.0
    chart_width = width - (2 * margin)
    chart_height = height - (2 * margin)

    circles: list[str] = []
    for point in points:
        cx = margin + ((point["x"] - x_min) / x_span) * chart_width
        cy = height - margin - ((point["y"] - y_min) / y_span) * chart_height
        label = escape_xml(str(point.get("label") or ""))
        circles.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="4" fill="#1f77b4" />'
            + (f'<title>{label}</title>' if label else "")
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<rect width="100%" height="100%" fill="#ffffff" />'
        f'<text x="24" y="30" font-family="Arial" font-size="18">{escape_xml(title)}</text>'
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#333" />'
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#333" />'
        f'<text x="{width / 2:.0f}" y="{height - 8}" font-family="Arial" font-size="12">{escape_xml(x_label)}</text>'
        f'<text x="8" y="{height / 2:.0f}" font-family="Arial" font-size="12">{escape_xml(y_label)}</text>'
        + "".join(circles)
        + "</svg>"
    )


def extract_points_from_records(records: list[dict[str, Any]], x_column: str | None = None, y_column: str | None = None) -> dict[str, Any]:
    usable = numeric_columns(records)
    if x_column is None or y_column is None:
        if len(usable) < 2:
            return {"status": "error", "failure_reason": "Need at least two numeric columns to visualize.", "runtime_diagnostics": {"runtime_ready": True, "input_ready": False}}
        x_column = usable[0]
        y_column = usable[1]
    points: list[dict[str, float | str]] = []
    for index, record in enumerate(records):
        x_value = coerce_number(record.get(x_column))
        y_value = coerce_number(record.get(y_column))
        if x_value is None or y_value is None:
            continue
        points.append({"x": x_value, "y": y_value, "label": str(record.get("label") or record.get("name") or index)})
    return {
        "status": "ok",
        "x_column": x_column,
        "y_column": y_column,
        "points": points,
        "runtime_diagnostics": {"runtime_ready": True, "input_ready": bool(points)},
    }


def coerce_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def pearson(xs: list[float], ys: list[float]) -> float:
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_den = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_den = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if x_den == 0 or y_den == 0:
        return 0.0
    return numerator / (x_den * y_den)


def euclidean(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _paired_values(records: list[dict[str, Any]], left: str, right: str) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for record in records:
        left_value = coerce_number(record.get(left))
        right_value = coerce_number(record.get(right))
        if left_value is None or right_value is None:
            continue
        pairs.append((left_value, right_value))
    return pairs


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, str):
            stripped = value.strip()
            number = coerce_number(stripped)
            normalized[str(key)] = number if number is not None else stripped
        else:
            normalized[str(key)] = value
    return normalized


def escape_xml(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
