# Assessment inputs

All four original assessment CSVs are included here so a checkout can run the
pipeline directly after installing dependencies. Filenames were normalized for
the code; their contents are unchanged:

| Original download | Included filename |
| --- | --- |
| `train-test.csv` | `train_test.csv` |
| `validation.csv` | `validation.csv` |
| `validation-predictions-template.csv` | `validation_predictions_template.csv` |
| `december-chart-inputs.csv` | `december_chart_inputs.csv` |

The original assessment brief, README, requirements, and unchanged scorer are
included under `assessment/`. Duplicate copies in `attachments/` and the
downloaded email stay local. Use the root `requirements.txt` to install this
solution's pinned dependencies.

Optionally verify the included CSVs against the saved input audit:

```bash
.venv/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path

expected = json.loads(Path('evaluation/data_audit.json').read_text())['checksums']
for name, digest in expected.items():
    actual = hashlib.sha256((Path('data') / name).read_bytes()).hexdigest()
    if actual != digest:
        raise SystemExit(f'Checksum mismatch: {name}')
    print(f'OK: {name}')
PY
```

The pipeline preserves all source files. Reproduction writes local prediction
diagnostics at `artifacts/prediction_summary.json`; both final prediction CSVs are
included separately in the submission. Setup requires no private email links.
