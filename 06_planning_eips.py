#!/usr/bin/env python3
"""
=============================================================================
Step 6: Planning Decision Analysis, Infrastructure Gap & EIPS Framework
=============================================================================
Project: Explainable AI Framework for Smart EV Charging Infrastructure Planning
Domain: Telangana, India (33 Administrative Districts)

Purpose:
- Integrate Track A station infrastructure, Track B district EV adoption
  context, and Track C verified DISCOM electricity context.
- Compute a transparent district infrastructure-gap measure.
- Compute spatial suitability from observed GIS-derived station attributes.
- Derive AHP weights for five planning criteria and verify AHP consistency.
- Apply TOPSIS to derive the Explainable Infrastructure Priority Score (EIPS).
- Generate traceable district-level planning outputs.

Important data handling:
- The frozen master station dataset is used as the station source.
- The FY18-FY24 district panel is retained for adoption-history context.
- FY2023-24 is the decision year used for the EIPS adoption criterion.
- FY18-FY24 cumulative-EV velocity is reported as contextual evidence and
  is NOT silently substituted for the approved FY24 penetration criterion.
- The 288-record TGSPDCL section-month ledger is used as an independent
  provenance/total check; district-year grid values come from the certified
  district-year panel.
- Missing TGSPDCL observations are never replaced by an invented physical
  grid measurement. A zero grid-throughput score means "no observed TGSPDCL
  throughput evidence in the certified district context", not zero grid
  capacity.
=============================================================================
"""

import os

# Headless backend for Windows command-line execution.
os.environ["MPLBACKEND"] = "Agg"

import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = BASE_DIR / "data"
DATA_FINAL_DIR = DATA_DIR / "final"
DATA_GOV_DIR = DATA_DIR / "government"

RESULTS_DIR = BASE_DIR / "results"
PLANNING_DIR = RESULTS_DIR / "planning"
FIGURES_DIR = RESULTS_DIR / "figures"

for directory in [PLANNING_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


STATION_REQUIRED = [
    "station_id",
    "district",
    "distance_to_highway_km",
    "distance_to_city_center_km",
    "nearby_poi_count",
    "poi_diversity_index",
    "fast_charging_flag",
    "highway_proximity_score",
    "city_center_proximity_score"
]

PANEL_REQUIRED = [
    "district",
    "year",
    "financial_year",
    "projected_population",
    "nh_corridors_crossing_count",
    "district_ev_penetration_rate",
    "estimated_district_cumulative_evs_proxy",
    "measured_discom_annual_ev_kwh"
]

ELECTRICITY_REQUIRED = [
    "measured_electricity_consumed_kwh"
]

AHP_CRITERIA = [
    "C1_EV_Adoption_Context",
    "C2_Infrastructure_Gap",
    "C3_Corridor_Accessibility",
    "C4_Grid_Readiness",
    "C5_Spatial_Suitability"
]

AHP_DESCRIPTIONS = [
    "FY2023-24 district EV penetration context; FY18-FY24 adoption velocity retained as context",
    "Observed fast-charging infrastructure deficit relative to district EV adoption",
    "National Highway corridor connectivity / crossing exposure",
    "Observed TGSPDCL utility throughput evidence with explicit data-availability status",
    "Station-derived highway, city-center and POI spatial suitability"
]


def fail_if_missing_columns(df, required, source_name):
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise RuntimeError(
            "{} is missing required columns: {}".format(
                source_name,
                missing
            )
        )


def load_planning_data():
    print("=" * 80)
    print("[INFO] STEP 6: STARTING MULTI-CRITERIA PLANNING & EIPS FRAMEWORK")
    print("=" * 80)

    station_path = DATA_FINAL_DIR / "FINAL_PROJECT_DATASET.csv"
    panel_path = DATA_FINAL_DIR / "district_year_panel.csv"
    electricity_path = (
        DATA_GOV_DIR /
        "tgspdcl_ev_charging_electricity_standardized.csv"
    )

    for path, label in [
        (station_path, "Frozen master station dataset"),
        (panel_path, "District-year panel"),
        (electricity_path, "TGSPDCL electricity ledger")
    ]:
        if not path.exists():
            raise FileNotFoundError(
                "[ERROR] {} not found at {}".format(
                    label,
                    path
                )
            )

    df_stations = pd.read_csv(station_path)
    df_panel = pd.read_csv(panel_path)
    df_elec = pd.read_csv(electricity_path)

    fail_if_missing_columns(
        df_stations,
        STATION_REQUIRED,
        "Frozen master station dataset"
    )

    fail_if_missing_columns(
        df_panel,
        PANEL_REQUIRED,
        "District-year panel"
    )

    fail_if_missing_columns(
        df_elec,
        ELECTRICITY_REQUIRED,
        "TGSPDCL electricity ledger"
    )

    print(
        "[INFO] Loaded frozen master station dataset: "
        "{} stations x {} columns".format(
            len(df_stations),
            len(df_stations.columns)
        )
    )

    district_count = int(df_stations["district"].nunique())

    if district_count != 33:
        raise RuntimeError(
            "Expected 33 station-covered districts, found {}".format(
                district_count
            )
        )

    print("[INFO] Station districts:", district_count)

    panel_districts = int(df_panel["district"].nunique())

    if panel_districts != 33:
        raise RuntimeError(
            "Expected 33 districts in district-year panel, found {}".format(
                panel_districts
            )
        )

    panel_years = int(
        df_panel[
            ["district", "financial_year"]
        ].drop_duplicates().shape[0]
    )

    print(
        "[INFO] District-year panel: {} rows, {} districts".format(
            len(df_panel),
            panel_districts
        )
    )
    print(
        "[INFO] District-year observations: {} district-year pairs".format(
            panel_years
        )
    )

    fy24 = df_panel[
        df_panel["financial_year"] == "FY2023-24"
    ].copy()

    if len(fy24) != 33:
        raise RuntimeError(
            "Expected one FY2023-24 row for each district, found {}".format(
                len(fy24)
            )
        )

    print(
        "[INFO] FY2023-24 planning slice: {} districts".format(
            len(fy24)
        )
    )

    total_kwh = pd.to_numeric(
        df_elec["measured_electricity_consumed_kwh"],
        errors="coerce"
    ).sum()

    total_gwh = total_kwh / 1e6

    print(
        "[INFO] TGSPDCL ledger: {} section-month records, "
        "statewide ledger total = {:.2f} GWh".format(
            len(df_elec),
            total_gwh
        )
    )

    panel_grid_total = pd.to_numeric(
        fy24["measured_discom_annual_ev_kwh"],
        errors="coerce"
    ).sum()

    panel_grid_total_gwh = panel_grid_total / 1e6

    print(
        "[INFO] FY2023-24 district-panel grid total: {:.2f} GWh".format(
            panel_grid_total_gwh
        )
    )

    if total_kwh > 0:
        discrepancy = abs(
            panel_grid_total - total_kwh
        ) / total_kwh

        print(
            "[INFO] TGSPDCL ledger vs district-panel total discrepancy: "
            "{:.2%}".format(discrepancy)
        )

    return df_stations, df_panel, fy24, df_elec


def compute_adoption_context(df_panel, fy24):
    """
    Compute an auditable FY18-FY24 adoption-velocity context metric.

    The EIPS C1 criterion remains the approved FY2023-24 district EV
    penetration rate. Velocity is reported separately so that the project
    retains its multi-year panel context without silently changing the
    criterion definition.
    """

    print("\n" + "=" * 80)
    print("[INFO] COMPUTING FY18-FY24 EV ADOPTION CONTEXT")
    print("=" * 80)

    panel = df_panel.copy()

    panel["year_numeric"] = pd.to_numeric(
        panel["year"],
        errors="coerce"
    )

    velocity_records = []

    for district, group in panel.groupby("district"):
        valid = group[
            [
                "year_numeric",
                "estimated_district_cumulative_evs_proxy"
            ]
        ].dropna()

        valid = valid.sort_values("year_numeric")

        if len(valid) >= 2:
            slope = float(
                np.polyfit(
                    valid["year_numeric"].values,
                    valid[
                        "estimated_district_cumulative_evs_proxy"
                    ].values,
                    1
                )[0]
            )
        else:
            slope = np.nan

        velocity_records.append(
            {
                "district": district,
                "ev_adoption_velocity_proxy_per_year": slope,
                "velocity_observation_count": int(len(valid))
            }
        )

    velocity_df = pd.DataFrame(velocity_records)

    df_dist = fy24.merge(
        velocity_df,
        on="district",
        how="left",
        validate="one_to_one"
    )

    if len(df_dist) != 33:
        raise RuntimeError(
            "Adoption-context merge did not preserve 33 districts."
        )

    print(
        "[INFO] FY2023-24 penetration is the EIPS adoption criterion."
    )
    print(
        "[INFO] FY18-FY24 cumulative-EV velocity is retained as contextual evidence."
    )

    velocity_available = int(
        df_dist["ev_adoption_velocity_proxy_per_year"]
        .notna()
        .sum()
    )

    print(
        "[INFO] Velocity available for {} of 33 districts".format(
            velocity_available
        )
    )

    return df_dist


def compute_infrastructure_gap(df_stations, df_dist):
    print("\n" + "=" * 80)
    print("[INFO] COMPUTING DISTRICT-LEVEL INFRASTRUCTURE GAP")
    print("=" * 80)

    station_agg = (
        df_stations
        .groupby("district", dropna=False)
        .agg(
            total_stations=(
                "station_id",
                "count"
            ),
            fast_stations=(
                "fast_charging_flag",
                lambda s: int((s == 1).sum())
            ),
            slow_stations=(
                "fast_charging_flag",
                lambda s: int((s == 0).sum())
            ),
            unknown_label_stations=(
                "fast_charging_flag",
                lambda s: int(s.isna().sum())
            ),
            mean_highway_distance=(
                "distance_to_highway_km",
                "mean"
            ),
            mean_city_center_distance=(
                "distance_to_city_center_km",
                "mean"
            ),
            mean_poi_count=(
                "nearby_poi_count",
                "mean"
            ),
            mean_poi_diversity=(
                "poi_diversity_index",
                "mean"
            ),
            mean_highway_proximity_score=(
                "highway_proximity_score",
                "mean"
            ),
            mean_city_center_proximity_score=(
                "city_center_proximity_score",
                "mean"
            )
        )
        .reset_index()
    )

    df_dist = df_dist.merge(
        station_agg,
        on="district",
        how="left",
        validate="one_to_one"
    )

    for column in [
        "total_stations",
        "fast_stations",
        "slow_stations",
        "unknown_label_stations"
    ]:
        df_dist[column] = (
            df_dist[column]
            .fillna(0)
            .astype(int)
        )

    spatial_columns = [
        "mean_highway_distance",
        "mean_city_center_distance",
        "mean_poi_count",
        "mean_poi_diversity",
        "mean_highway_proximity_score",
        "mean_city_center_proximity_score"
    ]

    for column in spatial_columns:
        if df_dist[column].isna().any():
            if df_dist[column].notna().any():
                replacement = float(
                    df_dist[column].median()
                )
                df_dist[column] = df_dist[column].fillna(
                    replacement
                )
            else:
                raise RuntimeError(
                    "No observed values available for {}".format(column)
                )

    # Approved transparent gap formulation.
    df_dist["infrastructure_gap_index"] = (
        df_dist["district_ev_penetration_rate"] /
        (df_dist["fast_stations"] + 1.0)
    )

    df_dist["infrastructure_gap_index"] = (
        df_dist["infrastructure_gap_index"]
        .round(4)
    )

    # Spatial suitability from observed station-level GIS context.
    highway_proximity = (
        1.0 /
        (1.0 + df_dist["mean_highway_distance"])
    )

    city_center_proximity = (
        1.0 /
        (1.0 + df_dist["mean_city_center_distance"])
    )

    poi_max = float(
        df_dist["mean_poi_count"].max()
    )

    if poi_max <= 0:
        poi_richness = np.zeros(
            len(df_dist),
            dtype=float
        )
    else:
        poi_richness = (
            df_dist["mean_poi_count"] /
            poi_max
        )

    df_dist["spatial_suitability_score"] = (
        0.40 * highway_proximity +
        0.35 * city_center_proximity +
        0.25 * poi_richness
    ).round(4)

    # Grid context is sourced from the certified district-year panel.
    grid_kwh = pd.to_numeric(
        df_dist["measured_discom_annual_ev_kwh"],
        errors="coerce"
    )

    grid_available = grid_kwh.notna() & (
        grid_kwh >= 0
    )

    df_dist["grid_data_observed_flag"] = (
        grid_available.astype(int)
    )

    grid_score = np.zeros(
        len(df_dist),
        dtype=float
    )

    if grid_available.any():
        observed_values = grid_kwh[grid_available].clip(
            lower=0
        )

        log_values = np.log1p(
            observed_values
        )

        max_log = float(
            log_values.max()
        )

        if max_log > 0:
            grid_score[
                grid_available.values
            ] = (
                log_values / max_log
            ).values
        else:
            grid_score[
                grid_available.values
            ] = 1.0

    df_dist["grid_readiness_score"] = np.round(
        grid_score,
        4
    )

    df_dist["grid_readiness_interpretation"] = np.where(
        df_dist["grid_data_observed_flag"] == 1,
        "Observed TGSPDCL district throughput evidence",
        "No observed TGSPDCL district throughput value"
    )

    print(
        "[INFO] Stations: {} across {} districts".format(
            len(df_stations),
            df_dist["district"].nunique()
        )
    )

    print(
        "[INFO] Fast-labeled stations: {}".format(
            int(df_dist["fast_stations"].sum())
        )
    )

    print(
        "[INFO] Slow-labeled stations: {}".format(
            int(df_dist["slow_stations"].sum())
        )
    )

    print(
        "[INFO] Unknown fast-charging labels: {}".format(
            int(df_dist["unknown_label_stations"].sum())
        )
    )

    print(
        "[INFO] Districts with observed TGSPDCL context: {}".format(
            int(df_dist["grid_data_observed_flag"].sum())
        )
    )

    gap_columns = [
        "district",
        "projected_population",
        "district_ev_penetration_rate",
        "ev_adoption_velocity_proxy_per_year",
        "total_stations",
        "fast_stations",
        "slow_stations",
        "unknown_label_stations",
        "infrastructure_gap_index",
        "spatial_suitability_score",
        "measured_discom_annual_ev_kwh",
        "grid_data_observed_flag",
        "grid_readiness_score"
    ]

    gap_output = (
        PLANNING_DIR /
        "infrastructure_gap_analysis.csv"
    )

    (
        df_dist[gap_columns]
        .sort_values(
            by="infrastructure_gap_index",
            ascending=False
        )
        .to_csv(
            gap_output,
            index=False
        )
    )

    print(
        "[OK] Infrastructure gap analysis saved: {}".format(
            gap_output
        )
    )

    # Visualization.
    plt.figure(
        figsize=(10, 7),
        dpi=300
    )

    top_gap = (
        df_dist
        .sort_values(
            "infrastructure_gap_index",
            ascending=False
        )
        .head(12)
        .sort_values(
            "infrastructure_gap_index",
            ascending=True
        )
    )

    plt.barh(
        top_gap["district"],
        top_gap["infrastructure_gap_index"]
    )

    plt.xlabel(
        "Infrastructure Gap Index "
        "(EV Penetration / [Fast Stations + 1])"
    )

    plt.ylabel("District")

    plt.title(
        "Top 12 Telangana Districts by Infrastructure Gap"
    )

    plt.grid(
        axis="x",
        linestyle=":",
        alpha=0.5
    )

    plt.tight_layout()

    gap_figure = (
        FIGURES_DIR /
        "infrastructure_gap_vs_adoption.png"
    )

    plt.savefig(
        gap_figure,
        dpi=300
    )

    plt.close()

    print(
        "[OK] Saved: {}".format(
            gap_figure
        )
    )

    return df_dist


def compute_ahp_weights():
    print("\n" + "=" * 80)
    print("[INFO] COMPUTING AHP MULTI-CRITERIA WEIGHTS")
    print("=" * 80)

    # Saaty-style reciprocal pairwise matrix.
    matrix = np.array(
        [
            [1.0, 1 / 1.5, 2.0, 2.0, 2.0],
            [1.5, 1.0, 2.5, 2.5, 2.5],
            [0.5, 1 / 2.5, 1.0, 1.0, 1.0],
            [0.5, 1 / 2.5, 1.0, 1.0, 1.0],
            [0.5, 1 / 2.5, 1.0, 1.0, 1.0]
        ],
        dtype=float
    )

    eigenvalues, eigenvectors = np.linalg.eig(
        matrix
    )

    maximum_index = int(
        np.argmax(
            np.real(eigenvalues)
        )
    )

    lambda_max = float(
        np.real(
            eigenvalues[maximum_index]
        )
    )

    weights = np.real(
        eigenvectors[:, maximum_index]
    )

    weights = weights / np.sum(weights)

    n = len(AHP_CRITERIA)

    consistency_index = (
        lambda_max - n
    ) / (n - 1)

    random_index = {
        1: 0.0,
        2: 0.0,
        3: 0.58,
        4: 0.90,
        5: 1.12
    }[n]

    consistency_ratio = (
        consistency_index / random_index
    )

    print(
        "[INFO] Principal eigenvalue: {:.6f}".format(
            lambda_max
        )
    )

    print(
        "[INFO] Consistency Index: {:.6f}".format(
            consistency_index
        )
    )

    print(
        "[INFO] Consistency Ratio: {:.6f}".format(
            consistency_ratio
        )
    )

    if consistency_ratio >= 0.10:
        raise RuntimeError(
            "AHP consistency ratio is not below 0.10: {:.6f}".format(
                consistency_ratio
            )
        )

    print(
        "[OK] AHP consistency validated: CR < 0.10"
    )

    weight_df = pd.DataFrame(
        {
            "criterion": AHP_CRITERIA,
            "description": AHP_DESCRIPTIONS,
            "weight": np.round(
                weights,
                4
            )
        }
    )

    weights_path = (
        PLANNING_DIR /
        "ahp_weights.csv"
    )

    weight_df.to_csv(
        weights_path,
        index=False
    )

    print(
        "[OK] AHP weights saved: {}".format(
            weights_path
        )
    )

    for _, row in weight_df.iterrows():
        print(
            "  {} = {:.4f} ({:.1f}%)".format(
                row["criterion"],
                row["weight"],
                row["weight"] * 100
            )
        )

    ahp_info = {
        "lambda_max": lambda_max,
        "consistency_index": consistency_index,
        "consistency_ratio": consistency_ratio,
        "random_index": random_index,
        "criteria_count": n,
        "pairwise_matrix": matrix.tolist()
    }

    return weights, weight_df, ahp_info


def compute_topsis_eips(
    df_dist,
    weights,
    weight_df,
    ahp_info
):
    print("\n" + "=" * 80)
    print("[INFO] COMPUTING TOPSIS & EIPS FOR 33 DISTRICTS")
    print("=" * 80)

    criteria_columns = [
        "district_ev_penetration_rate",
        "infrastructure_gap_index",
        "nh_corridors_crossing_count",
        "grid_readiness_score",
        "spatial_suitability_score"
    ]

    for column in criteria_columns:
        if column not in df_dist.columns:
            raise RuntimeError(
                "Missing EIPS criterion column: {}".format(
                    column
                )
            )

    decision_matrix = np.column_stack(
        [
            pd.to_numeric(
                df_dist[column],
                errors="coerce"
            ).values
            for column in criteria_columns
        ]
    )

    if np.isnan(decision_matrix).any():
        missing_counts = np.isnan(
            decision_matrix
        ).sum(axis=0)

        details = dict(
            zip(
                criteria_columns,
                missing_counts.tolist()
            )
        )

        raise RuntimeError(
            "NaN values found in EIPS decision matrix: {}".format(
                details
            )
        )

    # Vector normalization.
    norm = np.sqrt(
        np.sum(
            decision_matrix ** 2,
            axis=0
        )
    )

    norm = np.where(
        norm == 0,
        1e-12,
        norm
    )

    normalized = (
        decision_matrix / norm
    )

    # AHP-weighted matrix.
    weighted = (
        normalized * weights
    )

    # All five criteria are benefit criteria for planning priority.
    positive_ideal = np.max(
        weighted,
        axis=0
    )

    negative_ideal = np.min(
        weighted,
        axis=0
    )

    separation_positive = np.sqrt(
        np.sum(
            (
                weighted -
                positive_ideal
            ) ** 2,
            axis=1
        )
    )

    separation_negative = np.sqrt(
        np.sum(
            (
                weighted -
                negative_ideal
            ) ** 2,
            axis=1
        )
    )

    denominator = (
        separation_positive +
        separation_negative
    )

    denominator = np.where(
        denominator == 0,
        1e-12,
        denominator
    )

    closeness = (
        separation_negative /
        denominator
    )

    eips = (
        closeness * 100.0
    )

    df_dist["EIPS"] = np.round(
        eips,
        2
    )

    df_dist["separation_ideal_S_plus"] = np.round(
        separation_positive,
        4
    )

    df_dist["separation_anti_ideal_S_minus"] = np.round(
        separation_negative,
        4
    )

    criterion_short_names = [
        "ev_adoption",
        "infrastructure",
        "corridor",
        "grid",
        "spatial"
    ]

    for index, name in enumerate(
        criterion_short_names
    ):
        df_dist[
            "criterion_{}_score".format(name)
        ] = np.round(
            normalized[:, index] * 100,
            2
        )

        df_dist[
            "weighted_{}_contribution".format(name)
        ] = np.round(
            weighted[:, index] * 100,
            2
        )

    # Rank-based planning tiers are deterministic presentation categories;
    # they are not model-performance metrics.
    df_dist = (
        df_dist
        .sort_values(
            by="EIPS",
            ascending=False
        )
        .reset_index(drop=True)
    )

    df_dist["rank"] = np.arange(
        1,
        len(df_dist) + 1
    )

    if len(df_dist) != 33:
        raise RuntimeError(
            "EIPS output must contain exactly 33 districts."
        )

    def assign_tier(rank):
        if rank <= 10:
            return (
                "Tier 1: High Priority "
                "(Immediate Fast-Charging Expansion)"
            )

        if rank <= 22:
            return (
                "Tier 2: Medium Priority "
                "(Growth Corridor Provisioning)"
            )

        return (
            "Tier 3: Standard Priority "
            "(Sustained Maintenance)"
        )

    df_dist["priority_tier"] = (
        df_dist["rank"]
        .apply(assign_tier)
    )

    def build_narrative(row):
        grid_status = row[
            "grid_readiness_interpretation"
        ]

        velocity = row[
            "ev_adoption_velocity_proxy_per_year"
        ]

        if pd.isna(velocity):
            velocity_text = "velocity context unavailable"
        else:
            velocity_text = (
                "FY18-FY24 cumulative-EV velocity proxy "
                "{:.1f}/year".format(
                    velocity
                )
            )

        return (
            "Rank #{:d}; EIPS {:.2f}/100. "
            "FY2023-24 EV penetration {:.3f} EV/1,000; "
            "{}. "
            "Infrastructure Gap {:.4f} with {} fast stations; "
            "NH crossings {}; "
            "spatial suitability {:.4f}; "
            "grid score {:.4f} ({})"
        ).format(
            int(row["rank"]),
            float(row["EIPS"]),
            float(row["district_ev_penetration_rate"]),
            velocity_text,
            float(row["infrastructure_gap_index"]),
            int(row["fast_stations"]),
            int(row["nh_corridors_crossing_count"]),
            float(row["spatial_suitability_score"]),
            float(row["grid_readiness_score"]),
            grid_status
        )

    df_dist["eips_explanation"] = (
        df_dist
        .apply(
            build_narrative,
            axis=1
        )
    )

    output_columns = [
        "rank",
        "district",
        "priority_tier",
        "EIPS",
        "district_ev_penetration_rate",
        "ev_adoption_velocity_proxy_per_year",
        "infrastructure_gap_index",
        "total_stations",
        "fast_stations",
        "slow_stations",
        "unknown_label_stations",
        "nh_corridors_crossing_count",
        "spatial_suitability_score",
        "grid_readiness_score",
        "grid_data_observed_flag",
        "grid_readiness_interpretation",
        "criterion_ev_adoption_score",
        "criterion_infrastructure_score",
        "criterion_corridor_score",
        "criterion_grid_score",
        "criterion_spatial_score",
        "weighted_ev_adoption_contribution",
        "weighted_infrastructure_contribution",
        "weighted_corridor_contribution",
        "weighted_grid_contribution",
        "weighted_spatial_contribution",
        "separation_ideal_S_plus",
        "separation_anti_ideal_S_minus",
        "eips_explanation"
    ]

    eips_output = (
        PLANNING_DIR /
        "district_eips_rankings.csv"
    )

    df_dist[
        output_columns
    ].to_csv(
        eips_output,
        index=False
    )

    print(
        "[OK] EIPS rankings saved: {}".format(
            eips_output
        )
    )

    print(
        "\nTop 10 districts by EIPS:"
    )

    for _, row in df_dist.head(10).iterrows():
        print(
            "  #{:<2} {:<30} EIPS={:>6.2f} "
            "Gap={:>8.4f} Fast={:>3d}".format(
                int(row["rank"]),
                str(row["district"])[:30],
                float(row["EIPS"]),
                float(row["infrastructure_gap_index"]),
                int(row["fast_stations"])
            )
        )

    # EIPS ranking figure.
    plt.figure(
        figsize=(12, 9),
        dpi=300
    )

    plot_df = (
        df_dist[
            [
                "district",
                "EIPS",
                "rank"
            ]
        ]
        .sort_values(
            "EIPS",
            ascending=True
        )
    )

    plt.barh(
        plot_df["district"],
        plot_df["EIPS"]
    )

    plt.xlabel(
        "Explainable Infrastructure Priority Score (EIPS)"
    )

    plt.ylabel(
        "District"
    )

    plt.title(
        "Telangana 33 Districts: EIPS via AHP-Weighted TOPSIS"
    )

    plt.grid(
        axis="x",
        linestyle=":",
        alpha=0.5
    )

    plt.tight_layout()

    ranking_figure = (
        FIGURES_DIR /
        "eips_district_rankings.png"
    )

    plt.savefig(
        ranking_figure,
        dpi=300
    )

    plt.close()

    print(
        "[OK] Saved: {}".format(
            ranking_figure
        )
    )

    # Weighted component contribution figure.
    top8 = df_dist.head(8).copy()

    contribution_columns = [
        (
            "weighted_ev_adoption_contribution",
            "EV Adoption Context"
        ),
        (
            "weighted_infrastructure_contribution",
            "Infrastructure Gap"
        ),
        (
            "weighted_corridor_contribution",
            "Corridor Access"
        ),
        (
            "weighted_grid_contribution",
            "Grid Readiness"
        ),
        (
            "weighted_spatial_contribution",
            "Spatial Suitability"
        )
    ]

    plt.figure(
        figsize=(12, 7),
        dpi=300
    )

    x = np.arange(
        len(top8)
    )

    width = 0.15

    for offset_index, (
        column,
        label
    ) in enumerate(
        contribution_columns
    ):
        offset = (
            offset_index - 2
        ) * width

        plt.bar(
            x + offset,
            top8[column],
            width,
            label=label
        )

    plt.xticks(
        x,
        top8["district"],
        rotation=30,
        ha="right"
    )

    plt.ylabel(
        "AHP-Weighted Criterion Contribution"
    )

    plt.xlabel(
        "District"
    )

    plt.title(
        "Traceable AHP-Weighted EIPS Component Contributions"
    )

    plt.legend(
        fontsize=8
    )

    plt.grid(
        axis="y",
        linestyle=":",
        alpha=0.5
    )

    plt.tight_layout()

    component_figure = (
        FIGURES_DIR /
        "eips_component_breakdown.png"
    )

    plt.savefig(
        component_figure,
        dpi=300
    )

    plt.close()

    print(
        "[OK] Saved: {}".format(
            component_figure
        )
    )

    report_path = (
        PLANNING_DIR /
        "planning_summary_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as report:

        report.write(
            "=" * 80 + "\n"
        )

        report.write(
            "EXPLAINABLE AI FRAMEWORK FOR SMART EV CHARGING INFRASTRUCTURE PLANNING\n"
        )

        report.write(
            "EIPS & MULTI-CRITERIA DECISION ANALYSIS SUMMARY REPORT\n"
        )

        report.write(
            "=" * 80 + "\n\n"
        )

        report.write(
            "1. PLANNING FRAMEWORK\n"
        )

        report.write(
            "   AHP-weighted TOPSIS -> EIPS\n"
        )

        report.write(
            "   Districts evaluated: 33\n"
        )

        report.write(
            "   EIPS scale: 0-100\n"
        )

        report.write(
            "   EIPS criterion directions: all benefit criteria\n\n"
        )

        report.write(
            "2. AHP CONSISTENCY\n"
        )

        report.write(
            "   Principal Eigenvalue: {:.6f}\n".format(
                ahp_info["lambda_max"]
            )
        )

        report.write(
            "   Consistency Index: {:.6f}\n".format(
                ahp_info["consistency_index"]
            )
        )

        report.write(
            "   Consistency Ratio: {:.6f}\n".format(
                ahp_info["consistency_ratio"]
            )
        )

        report.write(
            "   Consistency criterion: CR < 0.10\n\n"
        )

        report.write(
            "3. DATA CONTEXT\n"
        )

        report.write(
            "   Station grain: 166 physical stations\n"
        )

        report.write(
            "   Supervised labeled stations: 154\n"
        )

        report.write(
            "   Unknown fast-charging labels: 12\n"
        )

        report.write(
            "   District-year panel: 33 districts x 7 financial years\n"
        )

        report.write(
            "   TGSPDCL utility ledger: 288 section-month records\n\n"
        )

        report.write(
            "4. EV ADOPTION CONTEXT\n"
        )

        report.write(
            "   EIPS C1 uses FY2023-24 district EV penetration.\n"
        )

        report.write(
            "   FY18-FY24 cumulative-EV velocity proxy is retained as context.\n\n"
        )

        report.write(
            "5. GRID CONTEXT\n"
        )

        report.write(
            "   Grid score uses observed district TGSPDCL annual throughput from the certified panel.\n"
        )

        report.write(
            "   Missing TGSPDCL district values are not replaced by fabricated physical measurements.\n"
        )

        report.write(
            "   A grid score of 0 therefore means no observed TGSPDCL throughput evidence in this context.\n\n"
        )

        report.write(
            "6. PRIORITY TIERS\n"
        )

        report.write(
            "   Tier 1: ranks 1-10\n"
        )

        report.write(
            "   Tier 2: ranks 11-22\n"
        )

        report.write(
            "   Tier 3: ranks 23-33\n\n"
        )

        report.write(
            "7. EIPS DISTRICT OUTPUT\n"
        )

        for _, row in df_dist.iterrows():
            report.write(
                "   Rank #{:<2} | {:<30} | EIPS {:>6.2f} | {}\n".format(
                    int(row["rank"]),
                    str(row["district"])[:30],
                    float(row["EIPS"]),
                    row["priority_tier"]
                )
            )

        report.write(
            "\n" + "=" * 80 + "\n"
        )

    print(
        "[OK] Planning summary report saved: {}".format(
            report_path
        )
    )

    metadata = {
        "project_title": (
            "Explainable AI Framework for Smart EV Charging Infrastructure Planning"
        ),
        "step": 6,
        "station_rows": int(len(df_dist)),
        "station_total": int(df_dist["total_stations"].sum()),
        "fast_labeled": int(df_dist["fast_stations"].sum()),
        "slow_labeled": int(df_dist["slow_stations"].sum()),
        "unknown_labels": int(df_dist["unknown_label_stations"].sum()),
        "districts": int(df_dist["district"].nunique()),
        "eips_criteria": criteria_columns,
        "eips_scale": "0-100",
        "ahp_lambda_max": ahp_info["lambda_max"],
        "ahp_ci": ahp_info["consistency_index"],
        "ahp_cr": ahp_info["consistency_ratio"],
        "grid_missingness_policy": (
            "No fabricated physical value; zero score indicates no observed TGSPDCL throughput evidence"
        ),
        "generated_at": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__
    }

    metadata_path = (
        PLANNING_DIR /
        "planning_run_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as metadata_file:
        json.dump(
            metadata,
            metadata_file,
            indent=2,
            default=str
        )

    print(
        "[OK] Planning metadata saved: {}".format(
            metadata_path
        )
    )

    return df_dist


def main():
    (
        df_stations,
        df_panel,
        fy24,
        df_elec
    ) = load_planning_data()

    df_adoption = compute_adoption_context(
        df_panel,
        fy24
    )

    df_dist = compute_infrastructure_gap(
        df_stations,
        df_adoption
    )

    (
        weights,
        weight_df,
        ahp_info
    ) = compute_ahp_weights()

    compute_topsis_eips(
        df_dist,
        weights,
        weight_df,
        ahp_info
    )

    print("\n" + "=" * 80)
    print("[OK] STEP 6 PLANNING & EIPS FRAMEWORK COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
