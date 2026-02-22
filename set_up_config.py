from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

# -----------------------
# Config objects
# -----------------------

@dataclass(frozen=True)
class CaseConfig:
    key: str                       # "future", "future_ccs", "future_ew", ...
    steel_method: str              # "", "ccs", "ew", "ew_lc"
    db_name_out: str               # "steel_db_future", "steel_db_future_ccs", ...
    dict_acts: Dict[str, Dict[str, Any]]  # dict_acts_future_xxx[row["steel_classification"]][tech_key] -> match_activity
    create_db: bool
    calc_lca: bool
    contri: bool

    # optional plant preprocessing hook (e.g., set renewable power)
    plants_transform: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None


# -----------------------
# Caches to speed up
# -----------------------

@dataclass
class BuildCaches:
    database_sel: Dict[str, Any]      # mapping of db_name -> extracted bw2 db object
    bio_db_w: Any
    matched_database: str
    year: int
    iso2_cache: Dict[str, str]


def build_database_sel_once(w, NAME_REF_DB: str, BIOSPHERE_DB: str, future_db_name: str) -> Tuple[Dict[str, Any], Any]:
    """
    Extract BW2 databases once and reuse across cases.
    """
    ref_db_w = w.extract_brightway2_databases(NAME_REF_DB, add_properties=False, add_identifiers=False)
    ref_db_w_future = w.extract_brightway2_databases(future_db_name, add_properties=False, add_identifiers=False)
    bio_db_w = w.extract_brightway2_databases(BIOSPHERE_DB, add_properties=False, add_identifiers=False)

    database_sel = {
        NAME_REF_DB: ref_db_w,
        BIOSPHERE_DB: bio_db_w,
        future_db_name: ref_db_w_future,
    }
    return database_sel, bio_db_w


def make_build_caches(
    *,
    w,
    NAME_REF_DB: str,
    BIOSPHERE_DB: str,
    future_db_name: str,            # e.g. "ecoinvent_image_SSP2-RCP26_2040_base"
    FUTURE_YEAR: int,
    match_year_to_database: Callable[[int], str],
    init_model: str = 'image_SSP2-RCP26',
) -> BuildCaches:
    database_sel, bio_db_w = build_database_sel_once(w, NAME_REF_DB, BIOSPHERE_DB, future_db_name)
    matched_database = match_year_to_database(FUTURE_YEAR, init_model)

    return BuildCaches(
        database_sel=database_sel,
        bio_db_w=bio_db_w,
        matched_database=matched_database,
        year=FUTURE_YEAR,
        iso2_cache={},
    )


# -----------------------
# DB build step (shared)
# -----------------------

def iter_end_uses(row: pd.Series) -> pd.Index:
    """
    Centralizes your logic for choosing end-use columns.
    Returns the column names of non-null, >0 end-uses.
    """
    prod = row.get("steel_or_iron_production", None)
    if prod in ("both", "steel"):
        s = row.filter(regex="steel production")
    elif prod == "iron":
        s = row.filter(like=" production").filter(regex="^(?!.*(steel|Company)).*$")
    else:
        raise ValueError(f"Unexpected steel_or_iron_production='{prod}'")

    s = s.dropna()
    s = s[s > 0]
    return s.index


def make_plant_name_and_tech_key(row: pd.Series, end_use: str) -> Tuple[str, str]:
    """
    Your current rule: if plant name missing, build synthetic.
    """
    plant_name = row.get("Plant name (English)_x", np.nan)
    tech_key = end_use

    if pd.isna(plant_name):
        tech_key = f"{row['steel_or_iron_production']}-{row['steel_decarb_classification']}"
        plant_name = f"{row['Project name']}_{row['Internal ID']}"

    return plant_name, tech_key


def get_iso2_cached(country_name: str, iso2_cache: Dict[str, str], country_to_iso2: Callable[[str], str]) -> str:
    if country_name not in iso2_cache:
        iso2_cache[country_name] = country_to_iso2(country_name)
    return iso2_cache[country_name]


def build_regionalized_steel_db_for_case(
    *,
    plants_df: pd.DataFrame,
    case: CaseConfig,
    caches: BuildCaches,
    bw,
    db_name_base: str,                     # your "db_name" used in create_regionalized_activity signature
    process_import: Callable[..., None],
    annotate_exchanges_with_cbam: Callable[..., None],
    define_scope_cbam: Callable[..., Any],
    create_regionalized_activity: Callable[..., Tuple[dict, Any, str]],
    process_exchanges: Callable[..., list],
    add_transport_exchanges: Callable[..., list],
    match_year_to_database: Callable[[int], str],  # (kept for compatibility; caches already has matched_database)
    country_to_iso2: Callable[[str], str],
    add_transport: bool = False,
    km_transport_train: int = 500,
    start_idx: int = 0,
) -> None:
    """
    Builds BW database for the selected case. Reused for H2/CCS/EW/EW_LC.
    """
    if case.db_name_out in list(bw.databases):
        del bw.databases[case.db_name_out]

    new_acts = []
    seen_names = set()

    matched_database = caches.matched_database
    db_sel = caches.database_sel[matched_database]

    # main loop
    for i, (index, row) in enumerate(plants_df.iterrows(), start=1):
        print(f"Making steel facility activities: {i}/{len(plants_df)}", end="\r", flush=True)
        if index < start_idx:
            continue

        # avoid inplace edits that can trigger chained assignment issues
        row = row.copy()
        row = row.fillna(np.nan)

        for end_use in list(iter_end_uses(row)):
            plant_name, tech_key = make_plant_name_and_tech_key(row, end_use)

            match_activity = case.dict_acts[row["steel_classification"]].get(tech_key)
            if match_activity is None:
                # optional: warn and skip
                # print(f"[{case.key}] No match for tech_key={tech_key} steel_classification={row['steel_classification']}")
                continue

            iso2 = get_iso2_cached(row["Country/Area"], caches.iso2_cache, country_to_iso2)

            new_act, activity_to_adapt, new_code = create_regionalized_activity(
                db_sel,
                match_activity,
                iso2,
                row,
                end_use,
                db_name_base,
                plant_name,
                caches.year,
            )

            if new_act["name"] in seen_names:
                # print(f"WARNING: duplicate activity name {new_act['name']} (Plant ID {row.get('Plant ID')})")
                continue
            seen_names.add(new_act["name"])

            new_act["exchanges"] = process_exchanges(
                activity_to_adapt,
                row,
                matched_database,
                case.db_name_out,
                iso2,
                db_sel,
                new_code,
                caches.bio_db_w,
                plant_type=row["steel_or_iron_production"],
            )

            # do not include transport activities
            if add_transport:
                transport_ex = add_transport_exchanges(
                    row["Continent"],
                    db_sel,
                    matched_database,
                    km_transport_train=km_transport_train,
                )
                new_act["exchanges"].extend(transport_ex)
            new_acts.append(new_act)

    process_import(case.db_name_out, new_acts)

    annotate_exchanges_with_cbam(
        db_name=case.db_name_out,
        define_scope_cbam_func=define_scope_cbam
    )


# -----------------------
# Postprocessing (shared)
# -----------------------

def add_scope_and_cbam_columns(
    df: pd.DataFrame,
    *,
    alpha: float,
    contri_col: str = "lca_impact_contri_climate change",
    sum_scope_contributions: Callable[[Any], Any] = None,
    sum_cbam_contributions: Callable[..., Any] = None,
    sum_cbam_contributions_emission_factor: Optional[Callable[..., Any]] = None,
    production_col: str = "production volume",
    prefix: str = "",
) -> pd.DataFrame:
    """
    Adds scope 1/2/3 and CBAM splits (and optional emission-factor variant).
    prefix lets you avoid having per-case column names like cbam_true_total_ew_lc.
    """
    out = df.copy()

    if sum_scope_contributions is not None:
        out[["Scope 1", "Scope 2", "Scope 3"]] = out[contri_col].apply(sum_scope_contributions).apply(pd.Series)

    if sum_cbam_contributions is not None:
        out[[f"{prefix}cbam_true", f"{prefix}cbam_false"]] = (
            out[contri_col].apply(sum_cbam_contributions, alpha=alpha).apply(pd.Series)
        )
        out[f"{prefix}cbam_true_total_cc"] = out[f"{prefix}cbam_true"] * out[production_col]
        out[f"{prefix}cbam_false_total_cc"] = out[f"{prefix}cbam_false"] * out[production_col]

        denom = out[f"{prefix}cbam_true_total_cc"] + out[f"{prefix}cbam_false_total_cc"]
        out[f"{prefix}share_cbam_covered"] = out[f"{prefix}cbam_true_total_cc"] / denom
        out.loc[out[f"{prefix}share_cbam_covered"] < 0, f"{prefix}share_cbam_covered"] = 1

    if sum_cbam_contributions_emission_factor is not None:
        out[[f"{prefix}cbam_true_efactor", f"{prefix}cbam_false_efactor"]] = (
            out[contri_col].apply(sum_cbam_contributions_emission_factor, alpha=alpha).apply(pd.Series)
        )
        out[f"{prefix}cbam_true_total_cc_efactor"] = out[f"{prefix}cbam_true_efactor"] * out[production_col]
        out[f"{prefix}cbam_false_total_cc_efactor"] = out[f"{prefix}cbam_false_efactor"] * out[production_col]

        denom2 = out[f"{prefix}cbam_true_total_cc_efactor"] + out[f"{prefix}cbam_false_total_cc_efactor"]
        out[f"{prefix}share_cbam_covered_efactor"] = out[f"{prefix}cbam_true_total_cc_efactor"] / denom2
        out.loc[out[f"{prefix}share_cbam_covered_efactor"] < 0, f"{prefix}share_cbam_covered_efactor"] = 1

    return out


def emissions_intensity(
    df: pd.DataFrame,
    *,
    emissions_col: str = "Plant_GHG_emissions_Mt_wo_transport",
    production_col: str = "production volume",
) -> float:
    total_em = float(df[emissions_col].sum())
    total_prod = float(df[production_col].sum())
    return total_em / total_prod if total_prod else np.nan


# -----------------------
# Case runner (single entry point)
# -----------------------

def run_case(
    *,
    plants_future: pd.DataFrame,
    case: CaseConfig,
    caches: BuildCaches,
    bw,
    db_name_base: str,
    # build deps
    process_import,
    annotate_exchanges_with_cbam,
    define_scope_cbam,
    create_regionalized_activity,
    process_exchanges,
    add_transport_exchanges,
    match_year_to_database,
    country_to_iso2,
    # lca deps
    calc_lca_impacts_all_plants,
    MY_METHODS,
    european_countries,
    dict_types,
    sum_exchanges_wo_transport,
    time_tag: str,
    # cbam/scope deps
    alpha: float,
    sum_scope_contributions,
    sum_cbam_contributions,
    sum_cbam_contributions_emission_factor=None,
    db_tag='',
) -> pd.DataFrame:
    """
    Orchestrates: optional transform -> optional db build -> optional LCA -> shared postprocessing.
    """
    plants_df = plants_future
    if case.plants_transform is not None:
        plants_df = case.plants_transform(plants_df)

    if case.create_db:
        build_regionalized_steel_db_for_case(
            plants_df=plants_df,
            case=case,
            caches=caches,
            bw=bw,
            db_name_base=db_name_base,
            process_import=process_import,
            annotate_exchanges_with_cbam=annotate_exchanges_with_cbam,
            define_scope_cbam=define_scope_cbam,
            create_regionalized_activity=create_regionalized_activity,
            process_exchanges=process_exchanges,
            add_transport_exchanges=add_transport_exchanges,
            match_year_to_database=match_year_to_database,
            country_to_iso2=country_to_iso2,
        )

    if not case.calc_lca:
        return pd.DataFrame()
    
    df = calc_lca_impacts_all_plants(
        steel_method=case.steel_method,
        db_name_base="steel_db",  # keep your convention (base without suffix)
        methods=MY_METHODS,
        calc_lca_impacts=case.calc_lca,
        contri=case.contri,
        start_idx=0,
        european_countries=european_countries,
        dict_types=dict_types,
        sum_exchanges_wo_transport=sum_exchanges_wo_transport,
        time_tag=time_tag,
        db_tag=db_tag,
    )

    # If contri is False, you may want to skip CBAM/scope columns
    if case.contri:
        df = add_scope_and_cbam_columns(
            df,
            alpha=alpha,
            sum_scope_contributions=sum_scope_contributions,
            sum_cbam_contributions=sum_cbam_contributions,
            sum_cbam_contributions_emission_factor=sum_cbam_contributions_emission_factor,
            prefix="",  # keep standard names
        )

    return df
