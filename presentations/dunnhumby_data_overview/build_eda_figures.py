"""Export the saved, executed EDA charts used by the presentation.

Run the EDA notebook before this script when its analysis changes. The script
does not recompute metrics or read the source CSVs.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "phan_tich_du_lieu_propensity.ipynb"
OUTPUT = Path(__file__).resolve().parent / "eda_figures"

# Cell indices are tied to the published EDA notebook narrative.
CHARTS = {
    "target_scenarios": (23, 0),
    "recency_activity": (26, 0),
    "department_activity": (32, 0),
    "discount_activity": (35, 0),
    "activity_over_time": (41, 0),
    "fuel_activity": (44, 0),
    "rfm_distributions": (58, 0),
    "rf_heatmap": (60, 0),
    "monetary_groups": (62, 0),
}


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, (cell_index, image_index) in CHARTS.items():
        cell = notebook["cells"][cell_index]
        images = [
            block["data"]["image/png"]
            for block in cell.get("outputs", [])
            if "image/png" in block.get("data", {})
        ]
        if image_index >= len(images):
            raise ValueError(f"Missing saved image {image_index} in cell {cell_index}: {name}")
        payload = images[image_index]
        encoded = "".join(payload) if isinstance(payload, list) else payload
        destination = OUTPUT / f"{name}.png"
        destination.write_bytes(base64.b64decode(encoded))
        print(destination.relative_to(ROOT))


if __name__ == "__main__":
    main()
