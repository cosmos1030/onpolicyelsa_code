"""Generate artifact HTML from template + DB."""
from __future__ import annotations
import json
import os

TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "..", "artifact_content.html")
PLACEHOLDER = "__DB_JSON__"


def build_artifact_html(db, template_path: str = TEMPLATE_PATH,
                        output_path: str | None = None) -> str:
    """
    Inject DB JSON into the artifact template and return the HTML string.
    Optionally writes to output_path.
    """
    with open(template_path) as f:
        template = f.read()

    db_data = db.for_artifact()
    db_json = json.dumps(db_data, ensure_ascii=False, separators=(",", ":"))

    if PLACEHOLDER not in template:
        raise ValueError(f"Placeholder '{PLACEHOLDER}' not found in template")

    html = template.replace(PLACEHOLDER, db_json)

    if output_path:
        with open(output_path, "w") as f:
            f.write(html)

    return html
