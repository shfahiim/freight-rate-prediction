"""Reproducible descriptive figures and a reading guide; no model fitting."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from .common import write_json

TEAL, ORANGE, BLUE = "#126475", "#c57535", "#4767a6"
EQUIPMENT = ["Dry Van", "Flatbed", "Reefer"]
DOLLARS = FuncFormatter(lambda value, _: f"${value:,.0f}")


def build_eda(development: pd.DataFrame, final: pd.DataFrame, output_dir: Path, checksums: dict) -> None:
    """Use final covariates only for descriptive comparisons, never fit a model."""
    figures = output_dir / "figures/eda"
    figures.mkdir(parents=True, exist_ok=True)
    rate_per_mile = development.posted_rate / development.distance
    valid_weight = development.weight.where(development.weight > 0)
    entries = []
    summary = {
        "input_checksums": checksums,
        "development_rows": len(development), "final_rows": len(final),
        "target_quantiles": {str(q): float(development.posted_rate.quantile(q)) for q in [0, .01, .25, .5, .75, .99, 1]},
        "rate_per_mile_quantiles": {str(q): float(rate_per_mile.quantile(q)) for q in [0, .01, .25, .5, .75, .99, 1]},
        "distance_price_pearson": float(development.distance.corr(development.posted_rate)),
        "equipment": {}, "quality": {},
        "note": "Descriptive EDA includes all January-October labels. Final covariates are shown without labels. No models, recipes, predictions, or fitted preprocessing are changed.",
    }

    def save(fig, name, title, method, finding, implication):
        fig.tight_layout()
        path = figures / f"{name}.png"
        fig.savefig(path, dpi=180, facecolor="white")
        plt.close(fig)
        entries.append({"title": title, "path": str(path.relative_to(output_dir)),
                        "method": method, "finding": finding, "implication": implication})

    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "semibold", "axes.labelcolor": "#263238",
                         "legend.fontsize": 9, "axes.grid": False}):
        # 1. All observations, with logarithmic axes making both tails visible.
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
        for axis, values, label in [(axes[0], development.posted_rate, "Total posted price ($)"),
                                     (axes[1], rate_per_mile, "Posted price per mile ($/mi)")]:
            axis.hist(values, bins=np.geomspace(values.min(), values.max(), 65), color=TEAL, alpha=.85)
            axis.set(xscale="log", yscale="log", xlabel=f"{label}, log scale", ylabel="Loads, log scale")
            for q, color, name in [(.5, ORANGE, "Median"), (.99, BLUE, "99th percentile")]:
                axis.axvline(values.quantile(q), color=color, linestyle="--", label=f"{name}: {values.quantile(q):,.2f}")
            axis.legend(loc="upper left")
        fig.suptitle("Price distributions: all 48,000 original labels")
        save(fig, "price_distributions", "1. Price distributions and tails",
             "Both panels include all 48,000 labels. Logarithmic price and count axes reveal sparse extremes; no values are trimmed.",
             f"Median total price is ${development.posted_rate.median():,.2f}; the 99th percentile is ${development.posted_rate.quantile(.99):,.2f}, and the maximum is ${development.posted_rate.max():,.2f}. Median price per mile is ${rate_per_mile.median():.3f}.",
             "Extreme prices can dominate squared error. Robust fitting is reasonable, while all original labels remain in evaluation.")

        # 2. Density rather than overplotting or target-dependent sampling.
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
        density = axes[0].hexbin(development.distance, development.posted_rate, gridsize=42,
                                 bins="log", mincnt=1, cmap="viridis", linewidths=0)
        axes[0].set(xlabel="Distance (miles)", ylabel="Total posted price ($)", title="All loads: distance and price")
        axes[0].yaxis.set_major_formatter(DOLLARS)
        fig.colorbar(density, ax=axes[0], label="Loads per hexagon (log scale)")
        binned = pd.DataFrame({"distance": development.distance, "rpm": rate_per_mile})
        binned["band"] = pd.qcut(binned.distance, 20, duplicates="drop")
        groups = binned.groupby("band", observed=True)
        centers = groups.distance.median().to_numpy()
        median_curve = groups.rpm.median().to_numpy()
        summary["distance_groups"] = [{"rows": int(count), "median_distance": float(distance), "median_rate_per_mile": float(rate)}
                                      for count, distance, rate in zip(groups.size(), centers, median_curve)]
        axes[1].plot(centers, median_curve, color=TEAL, marker="o", label="Median")
        axes[1].fill_between(centers, groups.rpm.quantile(.25).to_numpy(), groups.rpm.quantile(.75).to_numpy(),
                             color=TEAL, alpha=.18, label="Middle 50% (IQR)")
        axes[1].set(xlabel="Distance (miles; 20 quantile groups)", ylabel="Posted price per mile ($/mi)",
                    title="Typical rate per mile by distance")
        axes[1].legend()
        save(fig, "distance_price", "2. Distance, price, and price per mile",
             "The left density plot includes every original load. The right summarizes all loads in distance quantile groups; its band is an interquartile spread, not a confidence interval.",
             f"Distance and total price have Pearson correlation {summary['distance_price_pearson']:.3f}. Observed median price per mile falls from ${median_curve[0]:.2f} near {centers[0]:.0f} miles to ${median_curve[-1]:.2f} near {centers[-1]:.0f} miles; raw load composition also differs between groups.",
             "Distance is a strong predictor, but a single fixed price per mile cannot express every load. Smooth distance effects supplement equipment and route information.")

        # 3. Normalize mileage, retain fliers, and disclose composition effects.
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
        samples = [rate_per_mile.loc[development.equipment.eq(name)].to_numpy() for name in EQUIPMENT]
        axes[0].boxplot(samples, tick_labels=[f"{name}\nn={len(values):,}" for name, values in zip(EQUIPMENT, samples)],
                         showfliers=True, patch_artist=True,
                         boxprops={"facecolor": "#d6e9e9"}, medianprops={"color": ORANGE, "linewidth": 2},
                         flierprops={"marker": ".", "markersize": 2, "alpha": .3})
        axes[0].set(yscale="log", ylabel="Posted price per mile ($/mi), log scale", title="Equipment distributions; all fliers shown")
        months = development.date.dt.to_period("M")
        for name, color in zip(EQUIPMENT, [TEAL, ORANGE, BLUE]):
            mask = development.equipment.eq(name)
            medians = rate_per_mile.loc[mask].groupby(months.loc[mask]).median()
            axes[1].plot([p.strftime("%b") for p in medians.index], medians, marker="o", color=color, label=name)
            summary["equipment"][name] = {"rows": int(mask.sum()), "median_rate_per_mile": float(rate_per_mile.loc[mask].median())}
        axes[1].set(ylabel="Median posted price per mile ($/mi)", xlabel="Month in 2025", title="Equipment and calendar together")
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].legend()
        save(fig, "equipment_calendar", "3. Equipment and monthly rate patterns",
             "Boxplots retain all original fliers. Monthly medians use labeled development loads only; no November-December rates are inferred.",
             "Equipment groups have different observed price distributions, and monthly medians show that their pricing relationships need not stay constant over time.",
             "Include equipment and test equipment-calendar interactions. These raw comparisons are descriptive: distance, routes, weights, and market composition still differ between groups.")

        # 4. Final dates are shown as unlabeled, not fabricated future targets.
        periods = pd.period_range("2025-01", "2025-12", freq="M")
        positions = np.arange(len(periods))
        train_counts = development.groupby(development.date.dt.to_period("M")).size().reindex(periods, fill_value=0)
        final_counts = final.groupby(final.date.dt.to_period("M")).size().reindex(periods, fill_value=0)
        monthly = pd.DataFrame({"month": development.date.dt.to_period("M"), "rpm": rate_per_mile}).groupby("month").rpm
        fig, axes = plt.subplots(2, 1, figsize=(10.5, 6), sharex=True)
        axes[0].bar(positions, train_counts, color=TEAL, label="Labeled development")
        axes[0].bar(positions, final_counts, bottom=train_counts, color=ORANGE, label="Final inputs: labels hidden")
        axes[0].set(ylabel="Load count", title="Data periods and observed monthly pricing")
        axes[0].legend(loc="upper left")
        axes[1].plot(positions[:10], monthly.median().reindex(periods[:10]), marker="o", color=TEAL, label="Median")
        axes[1].fill_between(positions[:10], monthly.quantile(.25).reindex(periods[:10]).to_numpy(),
                             monthly.quantile(.75).reindex(periods[:10]).to_numpy(), color=TEAL, alpha=.18, label="Middle 50% (IQR)")
        for axis in axes:
            axis.axvspan(9.5, 11.5, color=ORANGE, alpha=.1)
            for boundary in [2.5, 5.5, 8.5]:
                axis.axvline(boundary, color="#899499", linestyle="--", linewidth=.8)
        axes[1].text(10.5, .5, "No observed\nfinal prices", ha="center", va="center", transform=axes[1].get_xaxis_transform(), color=ORANGE)
        axes[1].set(ylabel="Posted price per mile ($/mi)", xlabel="Month in 2025", xticks=positions,
                    xticklabels=[p.strftime("%b") for p in periods])
        axes[1].legend(loc="upper left")
        save(fig, "monthly_overview", "4. Time coverage and monthly changes",
             "Counts include all 60,000 inputs. Price summaries use only the 48,000 labeled loads. Dashed lines mark quarter boundaries; the shaded final period has no observed prices.",
             "The final prediction dates follow training rather than mixing randomly with it. Monthly pricing changes and group composition make temporal validation consequential.",
             "Use chronological two-month windows. Recurring quarter effects are hypotheses tested by backtests, not proof of future or annual seasonality.")

        # 5. Compare rates rather than counts between differently sized datasets.
        quality_labels = ["Missing weight", "Invalid weight", "Missing market index"]
        quality_counts = {}
        for name, frame in [("Development", development), ("Final inputs", final)]:
            counts = np.array([frame.weight.isna().sum(), frame.weight.le(0).sum(), frame.market_index.isna().sum()], dtype=int)
            quality_counts[name] = counts
            summary["quality"][name] = {label: {"rows": int(count), "pct": float(100 * count / len(frame))}
                                         for label, count in zip(quality_labels, counts)}
        fig, axis = plt.subplots(figsize=(10.5, 4.8))
        for offset, name, frame, color in [(-.19, "Development", development, TEAL), (.19, "Final inputs", final, ORANGE)]:
            values = quality_counts[name] * 100 / len(frame)
            bars = axis.bar(np.arange(3) + offset, values, width=.36, color=color, label=f"{name} (n={len(frame):,})")
            axis.bar_label(bars, labels=[f"{value:.2f}%\n{count:,} loads" for value, count in zip(values, quality_counts[name])], padding=5, fontsize=10)
        axis.set(xticks=np.arange(3), xticklabels=quality_labels, ylabel="Share of each dataset (%)",
                 ylim=(0, 3), title="Data quality: missingness rises in final inputs")
        axis.legend(loc="upper left")
        save(fig, "missingness", "5. Missing and invalid inputs",
             "Bars show percentages with original row counts. Invalid weight means nonpositive; missing and invalid flags are separate. Numerical values are not silently repaired with absolute values.",
             f"Missing/invalid weight rises from {100 * (development.weight.isna() | development.weight.le(0)).mean():.2f}% to {100 * (final.weight.isna() | final.weight.le(0)).mean():.2f}%. Missing market index rises from {100 * development.market_index.isna().mean():.2f}% to {100 * final.market_index.isna().mean():.2f}%.",
             "Train-only imputation and explicit flags must support more frequent missing inputs at inference. Report errors on these slices separately.")

        # 6. Mutually exclusive row categories; per-city counts can overlap.
        cities = set(development.pickup) | set(development.delivery)
        lanes = set(zip(development.pickup, development.delivery))
        unseen_city = ~final.pickup.isin(cities) | ~final.delivery.isin(cities)
        unseen_lane = pd.Series([lane not in lanes for lane in zip(final.pickup, final.delivery)], index=final.index)
        categories = [int((~unseen_lane).sum()), int((unseen_lane & ~unseen_city).sum()), int(unseen_city.sum())]
        new_cities = sorted((set(final.pickup) | set(final.delivery)) - cities)
        city_counts = {name: int((final.pickup.eq(name) | final.delivery.eq(name)).sum()) for name in new_cities}
        summary["novelty"] = {"seen_route_rows": categories[0], "new_route_known_cities_rows": categories[1],
                               "new_city_rows": categories[2], "loads_per_new_city": city_counts}
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
        left = 0
        for count, color, label in zip(categories, [TEAL, BLUE, ORANGE], ["Seen route", "New route; known cities", "At least one new city"]):
            share = 100 * count / len(final)
            axes[0].barh([0], [share], left=left, color=color, label=f"{label}: {count:,} ({share:.2f}%)", height=.4)
            left += share
        axes[0].set(xlim=(0, 100), ylim=(-1, 1), yticks=[], xlabel="Share of final loads (%)", title="Final route/city novelty")
        axes[0].legend(loc="upper center", bbox_to_anchor=(.5, -.15), frameon=False)
        ordered = sorted(city_counts, key=city_counts.get)
        bars = axes[1].barh(ordered, [city_counts[name] for name in ordered], color=ORANGE)
        axes[1].bar_label(bars, padding=3)
        axes[1].set(xlabel="Loads involving this city", title="The eight new cities", xlim=(0, max(city_counts.values()) * 1.2))
        save(fig, "city_novelty", "6. Unfamiliar cities and routes",
             "The left categories partition final rows exactly once. The right counts a load once per involved new city, so bars can overlap and should not be added.",
             f"{categories[2]:,} final loads ({100 * categories[2] / len(final):.2f}%) involve new cities. Only {categories[1]} new-route loads have two known cities; most route novelty is city novelty.",
             "Chronological splits alone do not exercise unseen cities in this dataset. Use a target-independent city withholding test and geometry-based fallbacks.")

        # 7. Shared display bins are descriptive, not fitted model statistics.
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.4))
        summary["feature_distributions"] = {}
        for axis, column, label in zip(axes.flat, ["distance", "weight", "market_index", "quote_signal"],
                                        ["Distance (miles)", "Valid weight (lb)", "Market index", "Quote signal"]):
            values = []
            for frame in [development, final]:
                series = frame[column]
                if column == "weight":
                    series = series.where(series > 0)
                values.append(series.dropna().to_numpy())
            combined = np.concatenate(values)
            summary["feature_distributions"][column] = {
                name: {"observed_rows": len(data), "median": float(np.median(data)),
                       "q25": float(np.quantile(data, .25)), "q75": float(np.quantile(data, .75))}
                for name, data in zip(["development", "final_inputs"], values)}
            bins = np.linspace(combined.min(), combined.max(), 45)
            for data, color, name in zip(values, [TEAL, ORANGE], ["Development", "Final inputs"]):
                axis.hist(data, bins=bins, density=True, histtype="step", linewidth=2, color=color, label=f"{name}; n={len(data):,}")
            axis.set(xlabel=label, ylabel="Density", title=label)
            axis.legend()
        market = summary["feature_distributions"]["market_index"]
        save(fig, "feature_distributions", "7. Development versus final feature distributions",
             "Each panel uses shared display bins and density normalization. Missing market/weight and nonpositive weight are excluded from these finite-value distributions and counted explicitly in figure 5. No extreme valid values are trimmed.",
             f"Observed median market index changes from {market['development']['median']:.3f} in development to {market['final_inputs']['median']:.3f} in final inputs. Its middle-50% range changes from [{market['development']['q25']:.3f}, {market['development']['q75']:.3f}] to [{market['final_inputs']['q25']:.3f}, {market['final_inputs']['q75']:.3f}]. Distance and valid-weight distributions largely overlap visually. The quote distribution cannot establish its relationship with hidden final prices.",
             "This comparison describes inference conditions. It does not fit imputers, choose a quote regime, or change the selected models using final covariates.")

        # 8. Pairwise complete values, with invalid weights made explicit.
        columns = ["distance", "weight", "market_index", "quote_signal", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "posted_rate"]
        numeric = development[columns].copy()
        numeric["weight"] = valid_weight
        correlation = numeric.corr(method="spearman")
        summary["spearman_correlation"] = correlation.to_dict()
        fig, axis = plt.subplots(figsize=(10.5, 7))
        display = axis.imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1)
        labels = ["Distance", "Valid weight", "Market index", "Quote signal", "Pickup lat", "Pickup lon", "Delivery lat", "Delivery lon", "Total price"]
        axis.set(xticks=np.arange(len(columns)), yticks=np.arange(len(columns)), xticklabels=labels, yticklabels=labels,
                 title="Pooled numerical relationships: Spearman correlation")
        axis.tick_params(axis="x", labelrotation=45)
        plt.setp(axis.get_xticklabels(), ha="right")
        for row in range(len(columns)):
            for column in range(len(columns)):
                value = correlation.iloc[row, column]
                axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=9,
                          color="white" if abs(value) > .6 else "#263238")
        fig.colorbar(display, ax=axis, label="Spearman correlation", fraction=.046, pad=.04)
        save(fig, "correlations", "8. Numerical feature relationships",
             "Spearman correlations use pairwise observed training values; invalid weight is treated as missing, without imputation. The matrix is pooled over all ten labeled months.",
             f"Pooled distance-price Spearman correlation is {correlation.loc['distance', 'posted_rate']:.3f}, while quote-price correlation is {correlation.loc['quote_signal', 'posted_rate']:.3f}. Monthly quote reversals disappear in that pooled number. Weak marginal weight/market correlations do not exclude conditional effects after accounting for distance, routes, and date.",
             "Use correlations as descriptive evidence rather than causality, leakage proof, or a feature-selection rule derived from future labels.")

    summary["figures"] = entries
    write_json(output_dir / "eda_summary.json", summary)
    render_guide(output_dir, entries)
    print(f"Generated {len(entries)} EDA figures, eda.md, and eda_summary.json", flush=True)


def render_guide(output_dir: Path, entries: list[dict]) -> None:
    title = "Freight Rate Prediction — Exploratory Data Analysis"
    introduction = "Explore data quality, target distributions, feature relationships, calendar coverage, and inference conditions. These figures describe the supplied data; they do not measure predictive accuracy. All original labeled prices are retained. Final inputs have no observed prices."
    boundary = "Evidence boundary: the analysis includes the previously inspected January-October development labels. Final covariates are used only for descriptive plots. This appendix does not retrain, select, or alter the saved models or prediction files."
    markdown = [f"# {title}", "", introduction, "", boundary, "", "## Figures", ""]
    markdown.extend([f"- [{item['title']}](#{item['title'][0]}-{item['title'][3:].lower().replace(' ', '-').replace(',', '')})" for item in entries])
    # The contents above are links to Markdown headings; each figure also has a direct PNG link below.
    for item in entries:
        markdown.extend(["", f"## {item['title']}", "", f"![{item['title']}]({item['path']})", "",
                         f"**How to read it:** {item['method']}", "", f"**Finding:** {item['finding']}", "",
                         f"**Modeling implication:** {item['implication']}", ""])
    markdown.extend(["## Reproduce", "", "From the repository root, run `PYTHONPATH=src .venv/bin/python -m freight_rate.audit`. The original inputs are included; see `data/README.md`. Numerical summaries and source input checksums are saved in `eda_summary.json`.", "",
                     "The monthly quote diagnostic is also available in [quote_relationship.png](figures/quote_relationship.png). Forecast ablations and model errors belong to the main report, separate from this descriptive appendix.", ""])
    (output_dir / "eda.md").write_text("\n".join(markdown))
