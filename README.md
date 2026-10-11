# Freight Rate Prediction

Predict freight load prices from 48,000 labeled January–October 2025 loads.
The submission contains predictions for all 12,000 November–December loads
and the supplied fixed December scenario. All original inputs are included.

## Run

Use **Python 3.12** from the repository root. Install the pinned dependencies,
then refit the saved recipe and regenerate both prediction files:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export PYTHONPATH=src
export MPLCONFIGDIR=artifacts/matplotlib

.venv/bin/python -m pytest -q
.venv/bin/python -m freight_rate.train
.venv/bin/python -m freight_rate.predict
.venv/bin/python assessment/score.py \
  --predictions validation_predictions.csv \
  --december-predictions outputs/december_predictions.csv \
  --output-dir reports/assets
```

Verified in a fresh Python 3.12.3 environment: 26 tests pass and both prediction
CSVs reproduce byte for byte.

Training checks the saved recipe, source/input hashes, Python minor version,
and dependency versions before fitting on all 48,000 labels. Models, diagnostics, and caches
are generated locally under ignored `artifacts/`. Reproduction does not repeat model selection.

The unchanged supplied scorer validates both output contracts and creates the
December chart; **it does not measure predictive accuracy**. Main predictions
are joined to the template by ID and exported as exactly `load_id,predicted_rate`.

To regenerate the data audit and EDA figures locally under `artifacts/eda/`:

```bash
.venv/bin/python -m freight_rate.audit
```

## Results

**Development backtests.** September–October results were reviewed during
development, and September overlaps model selection. These scores are not an
independent test. Final November–December labels are unavailable.

The following comparison uses all **9,523 September–October loads**, with
January–August training and identical original targets:

| Model / input contract | MAE ($) | RMSE ($) | WAPE (%) | MAPE (%) |
| --- | ---: | ---: | ---: | ---: |
| Selected main forecast / full | **90.15** | 629.98 | 3.77 | 3.97 |
| Matched CatBoost baseline / full | 118.06 | 637.01 | 4.93 | 5.10 |
| Selected December model / reduced | **109.70** | 633.93 | 4.59 | 4.94 |
| Matched CatBoost baseline / reduced | 113.69 | 635.69 | 4.75 | 5.00 |

MAE is mean absolute dollar error. RMSE gives large errors greater weight;
WAPE divides total absolute error by total actual dollars; MAPE averages each
load's absolute percentage error. Every headline metric retains every label.
Extreme prices dominate squared error; this is not proof that their errors are
unavoidable. The hidden evaluation metric is unspecified.

The main model improves equally weighted May–September rolling MAE by **29.3%**
over its matched baseline ($85.69 versus $121.23). The reduced model improves
its matched rolling baseline by **13.3%** ($111.95 versus $129.16).

## Approach and validation

The main model forecasts `log(price / distance)` using smooth distance, weight,
and coordinate effects; shared city and directed-lane effects; market index;
weekly terms; an extrapolating linear trend; and quarter-position hinges.
Huber-style reweighted ridge limits extreme-label influence while retaining
all labels with positive weights.

A 250-tree CatBoost residual stage adds a **small, inconsistent benefit**:
rolling MAE improves about 1% over the structured model alone ($86.58 → $85.69),
while September–October MAE worsens ($89.43 → $90.15). Most of the improvement
comes from the structured forecast. The frozen selection is retained.

The December file omits market index and coordinates. A separately validated
calendar CatBoost model uses only its available inputs. Both models exclude
the quote signal because its relationship with price reverses across months;
its unknown provenance does not establish leakage.

| Stage | Training months | Evaluation months |
| --- | --- | --- |
| Rolling A–D | Jan–Apr / Jan–May / Jan–Jun / Jan–Jul | May–Jun / Jun–Jul / Jul–Aug / Aug–Sep |
| Development confirmation | Jan–Aug | Sep–Oct |
| Submission refit | Jan–Oct | Nov–Dec inference |

Preprocessing statistics are fitted on training rows only. Repeated fold
errors are averaged within each month, then May–September receive equal weight.
Overlapping windows provide correlated evidence. Candidate eligibility requires
at least 2% macro-MAE improvement against the matched CatBoost baseline,
no month more than 5% worse, and no macro-RMSE increase above 5%.

A target-independent four-city withholding test evaluates unfamiliar endpoints:
main MAE is $51.35 versus baseline $102.55 on the same 1,040 affected loads.
This is relevant to the 1,447 final loads with new cities, without guaranteeing
their accuracy. Nonpositive weights become missing with explicit flags.
Quarter recurrence and longer-horizon price drift remain uncertainties.

## Submission files

- [Final predictions](validation_predictions.csv) and [December predictions](outputs/december_predictions.csv).
- [Final report](reports/report.pdf), including the chart produced by the supplied scorer.
- [Saved protocol](evaluation/protocol.json), [recipe and candidate results](evaluation/selected_recipe.json), and [confirmation metrics](evaluation/final_audit.json).
- [Input audit](evaluation/data_audit.json) and report and EDA assets in `reports/assets/`.
- `src/freight_rate/`: preprocessing, evaluation, training, prediction, and EDA.
- `tests/`: chronology, training-only preprocessing, extrapolation, unseen inputs, serialization, and output alignment.
- `data/` and `assessment/`: original CSVs, brief, instructions, requirements, and scorer; see [input mapping](data/README.md).

**Loom walkthrough:** [Watch the recorded project walkthrough](https://www.loom.com/share/209795baeed34856b5c826b1dc929883).
