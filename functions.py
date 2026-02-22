from config import *
import pandas as pd
import numpy as np
import uuid
import pycountry
import pycountry_convert as pc
import copy
import rioxarray as ri
import brightway2 as bw
import time
from collections import defaultdict
import io
import bw2data as bd
from contextlib import redirect_stdout
from rapidfuzz import process, fuzz
import wurst as w
import os
import re
import pickle
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Any

import pandas as pd
from tqdm import tqdm

from mapping import CO2_emission_factor_hydrogen, CO2_emission_factor_precursors_kgCO2_kg, CBAM_RELEVANT_PRECURSORS_IRON, CBAM_RELEVANT_PRECURSORS_STEEL, SCOPE_2_EXCHANGES

# Standard VARS
EN_H2 = 120 #MJ/kg h2
AMOUNT_WATER_ELECTROLYSIS = 15+9 # Tonelli et al., 2023, and Terlouw et al., 2024
HOURS_YR = 8760
COST_DATA = pd.read_excel(FILE_NAME_COSTS, index_col = [0,1], usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12])
DF_RATIOS = pd.read_excel(FILE_DF_RATIOS,  index_col=[0,1])

BAR_ACT = {'pem':30,
           'aec':20,
           'soec':1} # dictionary to provide the output pressure of hydrogen after water electrolysis

custom_mappings = {
    "Russia": "RU",
}

FUZZY_THRESHOLD = 90  # Tweak as needed (0–100). Higher = stricter match.
FUZZY_THRESHOLD_SCOPE_2 = 90

def convert_iso3_to_iso2(iso3):
    try:
        country = pycountry.countries.get(alpha_3=iso3)
        return country.alpha_2
    except KeyError:
        return None

def country_to_iso2(country_name):
    if country_name in custom_mappings:
        return custom_mappings[country_name]
    try:
        return pycountry.countries.lookup(country_name).alpha_2
    except LookupError:
        return None  # Handle missing cases gracefully


# -------------------------
# Helpers
# -------------------------

def build_results_columns(methods: Sequence[Tuple]) -> List[str]:
    base = [
        "name", "year", "unit", "country", "location", "reference product",
        "production volume", "database", "initial name", "comment",
        "latitude", "longitude", "power_source",
    ]
    base += [f"lca_impact_{m[1]}" for m in methods]
    base += [f"lca_impact_contri_{m[1]}" for m in methods if "climate change" in str(m)]
    return base


def parse_year_from_name(name: str) -> Optional[int]:
    m = re.search(r"\[(\d{4})\]", name or "")
    return int(m.group(1)) if m else None


def parse_comment_field(comment: str, key: str) -> Optional[str]:
    """
    Extracts e.g. key='latitude:' from comment like "... latitude:47.3, longitude:8.5, ..."
    Returns None if not found.
    """
    if not comment:
        return None
    token = f"{key}:"
    if token not in comment:
        return None
    try:
        return comment.split(token, 1)[1].split(",", 1)[0].strip()
    except Exception:
        return None


def should_do_contri(method: Tuple, contri: bool) -> bool:
    return contri and ("climate change" in str(method))


def compute_contribution_array(act_sel, lca: bw.LCA) -> List[Tuple[Any, ...]]:
    """
    Returns list of tuples: (exc_name, impact_value, scope, cbam, amount)
    - Technosphere: redo LCIA for exc.input with exc['amount']
    - Biosphere: cf * exc['amount']
    """
    result_array = []
    for exc in act_sel.exchanges():
        exc_type = exc.get("type")

        if exc_type == "technosphere":
            lca.redo_lcia({exc.input: exc["amount"]})
            result_array.append((
                exc.get("name"),
                lca.score,
                exc.get("scope"),
                exc.get("cbam"),
                exc.get("amount"),
            ))

        elif exc_type == "biosphere":
            # Multiply amount by its CF (as in your code)
            # Note: this assumes the current lca.method is the climate change method.
            cf = lca.characterization_matrix[lca.biosphere_dict[exc.input], :].sum()
            result_array.append((
                exc.get("name"),
                float(cf) * exc.get("amount", 0.0),
                exc.get("scope"),
                exc.get("cbam"),
                exc.get("amount"),
            ))
    return result_array


def compute_lca_for_activity(
    act_sel,
    methods: Sequence[Tuple],
    contri: bool = True,
) -> Dict[str, Any]:
    """
    Computes LCA scores for one activity across all methods.
    Reuses the same LCA object, switching methods between runs.
    """
    lca_scores: Dict[str, Any] = {}
    lca_scores_contri: Dict[str, Any] = {}

    lca = None
    for j, method in enumerate(methods):
        if j == 0:
            lca = bw.LCA({act_sel: 1}, method=method)
            lca.lci()
            lca.lcia()
        else:
            # switch method and redo LCIA
            lca.switch_method(method)
            lca.redo_lcia({act_sel: 1})

        lca_scores[f"lca_impact_{method[1]}"] = lca.score

        if should_do_contri(method, contri):
            lca_scores_contri[f"lca_impact_contri_{method[1]}"] = compute_contribution_array(act_sel, lca)

    out = {}
    out.update(lca_scores)
    out.update(lca_scores_contri)
    return out


def postprocess_results_df(
    df: pd.DataFrame,
    european_countries: Iterable[str],
    dict_types: Dict[str, Any],
    sum_exchanges_wo_transport: Optional[Callable[[Any], float]] = None,
    exclude_iron: bool = True,
) -> pd.DataFrame:
    df = df.copy()

    # numeric lat/lon
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # region classification
    eu_set = set(european_countries)
    df["region"] = df["location"].apply(lambda x: "European" if x in eu_set else "Non-European")

    # emissions totals
    if "lca_impact_climate change" in df.columns:
        df["Plant_GHG_emissions_Mt"] = df["production volume"] * df["lca_impact_climate change"]

    # climate change without transport (if available)
    if sum_exchanges_wo_transport is not None and "lca_impact_contri_climate change" in df.columns:
        df["lca_impact_climate change_wo_transport"] = df["lca_impact_contri_climate change"].apply(sum_exchanges_wo_transport)
        df["Plant_GHG_emissions_Mt_wo_transport"] = df["production volume"] * df["lca_impact_climate change_wo_transport"]

    # commodity type mapping
    if "initial name" in df.columns:
        df["commodity_type"] = df["initial name"].map(dict_types)

    # exclude iron making
    if exclude_iron and "initial name" in df.columns:
        df = df[~df["initial name"].str.contains("iron", case=False, na=False)]

    # sort
    if "Plant_GHG_emissions_Mt" in df.columns:
        df = df.sort_values(by="Plant_GHG_emissions_Mt", ascending=False)

    return df


# -------------------------
# Main modular function
# -------------------------

def calc_lca_impacts_all_plants(
    steel_method: str,
    db_name_base: str,
    methods: Sequence[Tuple],
    *,
    results_dir: str = "results",
    calc_lca_impacts: bool = True,
    contri: bool = True,
    start_idx: int = 0,
    european_countries: Iterable[str] = (),
    dict_types: Dict[str, Any] = None,
    sum_exchanges_wo_transport: Optional[Callable[[Any], float]] = None,
    exclude_iron: bool = True,
    save_pickle: bool = True,
    tqdm_desc: Optional[str] = "Calculating LCA impacts",
    time_tag = '',
    db_tag: str = '',
) -> pd.DataFrame:
    """
    Modular version of your workflow.

    Parameters
    ----------
    steel_method : str
        Suffix identifying the steel production method, e.g. "ew", "bf_bof", ...
    db_name_base : str
        Base db name without suffix. Final DB name becomes f"{db_name_base}_{steel_method}".
    methods : Sequence[Tuple]
        Brightway LCIA methods (tuples), e.g. MY_METHODS.
    results_dir : str
        Directory to store pickles.
    calc_lca_impacts : bool
        If True: compute and save. If False: load from pickle.
    contri : bool
        If True: compute contribution arrays for climate change methods.
    start_idx : int
        Resume index (skip activities with i < start_idx).
    european_countries : iterable
        Used to classify 'region'.
    dict_types : dict
        Mapping from initial name -> commodity type.
    sum_exchanges_wo_transport : callable
        Function applied to contribution arrays to exclude transport.
    exclude_iron : bool
        Whether to remove plants whose initial name contains 'iron'.
    save_pickle : bool
        Save computed results to pickle if calc_lca_impacts is True.
    tqdm_desc : str
        Description for tqdm progress bar.

    Returns
    -------
    pd.DataFrame
    """
    if dict_types is None:
        dict_types = {}
    
    time_tag = "_future" if time_tag == 'future' else '' 
    db_tag = f"_{db_tag}" if len(db_tag) > 0 else '' 

    db_name = (
        f"{db_name_base}{time_tag}_{steel_method}{db_tag}"
        if steel_method
        else f"{db_name_base}{time_tag}{db_tag}"
    )
    
    os.makedirs(results_dir, exist_ok=True)

    file_path = (
        os.path.join(results_dir, f"results_df{time_tag}_{steel_method}{db_tag}.pkl")
        if steel_method
        else os.path.join(results_dir, f"results_df{time_tag}{db_tag}.pkl")
    )


    if not calc_lca_impacts:
        with open(file_path, "rb") as f:
            df = pickle.load(f)
        return postprocess_results_df(
            df,
            european_countries=european_countries,
            dict_types=dict_types,
            sum_exchanges_wo_transport=sum_exchanges_wo_transport,
            exclude_iron=exclude_iron,
        )

    # compute
    columns = build_results_columns(methods)
    results_df = pd.DataFrame(columns=columns)

    acts = list(bw.Database(db_name))
    iterator = enumerate(tqdm(acts, desc=tqdm_desc or f"LCA {db_name}"))

    for i, act_sel in iterator:
        if i < start_idx:
            continue

        # compute scores for this activity
        scores = compute_lca_for_activity(act_sel, methods=methods, contri=contri)

        name = act_sel.get("name")
        comment = act_sel.get("comment", "")

        row = {
            "name": name,
            "reference product": act_sel.get("reference product"),
            "country": act_sel.get("location"),
            "location": act_sel.get("location"),
            "production volume": act_sel.get("production volume"),
            "unit": act_sel.get("unit"),
            "database": act_sel.get("database"),
            "initial name": (name.split(" for facility in")[0] if isinstance(name, str) else None),
            "comment": comment,
            "year": parse_year_from_name(name or ""),
            "latitude": parse_comment_field(comment, "latitude"),
            "longitude": parse_comment_field(comment, "longitude"),
            "power_source": parse_comment_field(comment, "power source"),
        }
        row.update(scores)

        results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)

    if save_pickle:
        results_df.to_pickle(file_path)

    # postprocess
    return postprocess_results_df(
        results_df,
        european_countries=european_countries,
        dict_types=dict_types,
        sum_exchanges_wo_transport=sum_exchanges_wo_transport,
        exclude_iron=exclude_iron,
    )


# Function to convert country name to continent name
def get_continent(country_name):
    try:
        # Get the ISO alpha-2 code (e.g., 'US' for United States)
        country = pycountry.countries.lookup(country_name)
        country_code = country.alpha_2

        # Convert to continent code
        continent_code = pc.country_alpha2_to_continent_code(country_code)

        # Convert continent code to full name
        continent_name = pc.convert_continent_code_to_continent_name(continent_code)
        return continent_name
    except:
        return 'Unknown'
    
def sum_exchanges_wo_transport(data, exclude_names=[
                                    "market for transport, freight, sea, container ship",
                                    "market group for transport, freight train"
                                ]):
    """
    Sum the second element (index 1) of each tuple in a dataset, excluding entries 
    where the exchange name matches any in the provided list of excluded names.

    Parameters:
    ----------
    data : list of tuples
        A list where each item is a tuple. The first element (index 0) is assumed to be
        the exchange name (a string), and the second element (index 1) is a numeric value to be summed.
        
    exclude_names : list of str, optional
        A list of exchange names to exclude from the summation. By default, this excludes
        common transport-related markets.

    Returns:
    -------
    float
        The sum of the second-place values in the tuples, excluding any tuples where
        the exchange name matches an item in `exclude_names`.
    """
    return sum(
        item[1]
        for item in data
        if item[0] not in exclude_names
    )

def silent_search(db_name, name_search, desired_location):
    """
    Perform a Brightway database search while suppressing console output.

    Parameters:
    - db_name (str): Name of the Brightway2 database to search.
    - name_search (str): Substring to match in activity names.
    - desired_location (str): Location to filter the search results by (exact match).

    Returns:
    - list: List of matching activities (as Brightway2 activity objects).
    """
    f = io.StringIO()
    with redirect_stdout(f):
        results = list(bw.Database(db_name).search(
            name_search,
            limit=1000,
            filter={"location": desired_location}
        ))
    return results

def sum_electricity_and_hydrogen_from_db(
    db_name: str,
    *,
    bw,
    h2_electricity_factor_kWh_per_kg: float,
):
    """
    Sum electricity production and hydrogen use from a Brightway database.

    Parameters
    ----------
    db_name : str
        Name of the Brightway database to scan.
    bw : brightway2 module
    h2_electricity_factor_kWh_per_kg : float
        Electricity required to produce hydrogen (kWh per kg H2),
        e.g. ~50–55 kWh/kg for electrolysis.

    Returns
    -------
    dict with:
        electricity_TWh
        hydrogen_Mt
        hydrogen_electricity_TWh
        total_electricity_TWh
    """

    electricity_kWh = 0.0
    hydrogen_kg = 0.0

    for act in list(bw.Database(db_name)):
        #print(act['name'])

        # ---------------------------
        # Hydrogen and electricity consumption
        # ---------------------------
        production_volume = act.get("production volume", 0)
        for exc in act.exchanges():
            if (
                exc.get("type") == "technosphere" and
                "kilogram" in str(exc.get("unit", "")).lower()
                and (
                    "hydrogen, gaseous" in str(exc.get("reference product", "")).lower()
                    or "hydrogen, gaseous" in str(exc.get("product", "")).lower()
                )
            ):
                amount = exc.get("amount", 0.0)
                if amount is not None:
                    hydrogen_kg += float(amount) * float(production_volume) * 1e9

            if (
                "electricity" in str(exc.get("name", "")).lower()
                and "kilowatt hour" in str(exc.get("unit", "")).lower()
                and (
                    "electricity" in str(exc.get("reference product", "")).lower()
                    or "electricity" in str(exc.get("product", "")).lower()
                )
            ):
                amount_power = exc.get("amount", 0.0)
                if amount_power is not None:
                    electricity_kWh += float(amount_power) * float(production_volume) * 1e9


    electricity_TWh = electricity_kWh / 1e9
    hydrogen_Mt = hydrogen_kg / 1e9

    hydrogen_electricity_TWh = (
        hydrogen_kg * h2_electricity_factor_kWh_per_kg / 1e9
    )

    return {
        "electricity_TWh": electricity_TWh,
        "hydrogen_Mt": hydrogen_Mt,
        "hydrogen_electricity_TWh": hydrogen_electricity_TWh,
        "total_electricity_TWh": electricity_TWh + hydrogen_electricity_TWh,
    }

def define_scope_cbam(exc, cbam_relevant_precursors, scope_2_exchanges=SCOPE_2_EXCHANGES):
    """
    Classifies an exchange (`exc`) into Scope 1, 2, or 3 emissions categories
    AND indicates whether the exchange is included in CBAM (especially for Scope 3),
    using fuzzy matching for synonyms of CBAM goods.

    Parameters
    ----------
    exc : dict
        A dictionary representing an exchange with at least the following keys:
          - 'type' (str): e.g. 'technosphere' or 'biosphere'
          - 'name' (str): Name/description of the exchange
          - 'unit' (str): The unit of measurement
          - 'product' (str, optional): The reference product if known

    Returns
    -------
    dict
        {
          'scope': int  (1, 2, or 3),
          'cbam': bool
        }

    Notes
    -----
    - Scope 1: Direct emissions (onsite fuel combustion, process emissions).
      Biosphere flows are assumed to be direct (Scope 1).
    - Scope 2: Purchased electricity (kWh).
    - Scope 3: Other indirect emissions.
      For CBAM, only certain upstream inputs (themselves CBAM goods) are included.
      We use fuzzy matching to detect these goods in the exchange's name or product.
    """

    result = {
        'scope': None,
        'cbam': False
    }
    if exc['type'] != 'production':
        # 1) Biosphere flows => direct (Scope 1) + NG vented
        if exc['type'] == 'biosphere' or (exc['type'] == 'technosphere' and 'natural gas, vented' == exc['name']):
            result['scope'] = 1
            result['cbam'] = True
            return result
    
        # 2) Technosphere (or other) flows
        ref_product = exc.get('product', '').lower()        
        if isinstance(exc.get('product', {}), dict):
            ref_product = exc.get('reference product', '').lower()
            
        name_lower = exc['name'].lower()
    
        # --- Scope 2: purchased electricity, steam, heat, or cooling ---
        for candidate in scope_2_exchanges:
            candidate_lower = candidate.lower()
            ratio_name = fuzz.token_set_ratio(candidate_lower, name_lower)
            
            if ratio_name >= FUZZY_THRESHOLD_SCOPE_2:
                result['scope'] = 2
                result['cbam'] = True
                return result
    
        # --- Otherwise => Scope 3 ---
        result['scope'] = 3
    
        # For CBAM, check if ref_product or name matches any known CBAM good (synonym list)
        for cbam_good, (synonyms, exclusions) in cbam_relevant_precursors.items():
            # Check for exclusion words
            if any(exclusion in ref_product or exclusion in name_lower for exclusion in exclusions):
                continue  # Skip this CBAM good if an exclusion word is present
    
            # Check fuzzy matches
            for candidate in synonyms:
                candidate_lower = candidate.lower()
                ratio_product = fuzz.token_set_ratio(candidate_lower, ref_product)
                ratio_name = fuzz.token_set_ratio(candidate_lower, name_lower)
    
                if ratio_product >= FUZZY_THRESHOLD or ratio_name >= FUZZY_THRESHOLD:
                    result['cbam'] = True
                    break  # Stop checking further synonyms in this sub-list
            
            if result['cbam']:
                break  # Stop checking further CBAM goods
    
    return result
    
def sum_scope_contributions(data):
    """
    Sums LCA impact contributions by scope (Scope 1, Scope 2, Scope 3).

    Parameters:
    -----------
    data : list of tuples
        A list where each element is a tuple containing:
        - name (str): Identifier for the contribution (not used in summation).
        - value (float or np.float64): Impact value to be summed.
        - scope (int or str): Scope category (e.g., 1, 2, or 3).

    Returns:
    --------
    dict
        A dictionary where keys are formatted as 'Scope {scope_number}' and values 
        are the summed impact contributions for each scope.
    """
    scope_sums = defaultdict(float)
    
    for name, value, scope, is_cbam, exc_amount in data:
        if isinstance(value, (np.float64, float)):  # Ensure numeric values only
            scope_sums[scope] += value
    
    return {f'Scope {key}': val for key, val in sorted(scope_sums.items())}

def sum_cbam_contributions(data, alpha, exc_ccs=True, exclude_scope_2=False):
    """
    Sums LCA impact contributions based on CBAM relevance.

    Parameters:
    -----------
    data : list of tuples
        A list where each element is a tuple containing:cond
        - name (str): Identifier for the contribution (not used in summation).
        - value (float or np.float64): Impact value to be summed.
        - is_cbam (bool): Boolean indicating whether the contribution is CBAM-relevant.

    Returns:
    --------
    dict
        A dictionary with two keys:
        - 'cbam_true': Sum of CBAM-relevant impact contributions.
        - 'cbam_false': Sum of non-CBAM impact contributions.
    """
    cbam_sums = defaultdict(float)
    
    for name, value, scope, is_cbam, exc_amount in data:
        #print(name, value, scope, is_cbam)
        if isinstance(value, (np.float64, float)):  # Ensure numeric values only
            key = 'cbam_true' if is_cbam else 'cbam_false'

            # This is an exception, if CBAM is previously indicated as False, but the impact is negative, for example, due to CCS, \
            # we account for it is as true, as such, direct emissions that should be substracted, note we would also get very strange results...
            # Better would be substract CO2 stored from direct emisions in scope 1 (in original inventories), but the activities for CCS are built in a different way.
            if exc_ccs:
                if key == 'cbam_false' and value<0 and scope == 3 and any(sub in name.lower() for sub in ['ccs', 'capture']):
                    cbam_sums['cbam_true'] += value
                    continue
            
            if key == 'cbam_true' and scope == 3:
                cbam_sums['cbam_true'] += value * alpha # consider that emissions are LCA impacts
                cbam_sums['cbam_false'] += value * (1-alpha)
            elif exclude_scope_2 and scope == 2:
                cbam_sums['cbam_false'] += value # if we want to exclude scope 2 from CBAM, we consider it as non-CBAM, but we still sum it in the total (cbam_false)
            else:
                cbam_sums[key] += value
    
    return {'cbam_true': cbam_sums['cbam_true'], 'cbam_false': cbam_sums['cbam_false']}

def get_best_match(name, choices=CO2_emission_factor_precursors_kgCO2_kg):
    """
    Return the best fuzzy match for a name from the CBAM precursor list using RapidFuzz.
    
    Parameters
    ----------
    name : str
        The input string to match (e.g. 'market for iron scrap').
    cbam_precursors : dict
        Dictionary whose keys are CBAM precursor names.
        
    Returns
    -------
    tuple
        (best_match_str, similarity_score)
    """
    name_clean = name.lower().replace('market for', '').replace(', in pieces, loose',"").replace('sorted, pressed', '').strip()
    match = process.extractOne(
        query=name_clean,
        choices=choices.keys(),
        scorer=fuzz.token_set_ratio
    )
    return match  # e.g., ('steel', 67.5, ...)

def sum_cbam_contributions_emission_factor(data, alpha, log_path='logs\cbam_emission_factor_scope3_corrections.log', exc_ccs=True):
    """
    Calculates the sum of life cycle assessment (LCA) impact values,
    separating CBAM-relevant and non-CBAM contributions, with emission factor adjustments for scope 3 precursors.

    Parameters
    ----------
    data : list of tuples
        Each tuple contains:
        - name (str): Identifier for the contribution.
        - value (float or np.float64): LCA impact value.
        - scope (int): Scope level (used for CBAM filtering).
        - is_cbam (bool): Whether the contribution is CBAM-relevant.
        - exc_amount (float): Quantity of the exchanged amount.

    log_path : str, optional
        File path for logging corrections. Default is 'cbam_corrections.log'.

    Returns
    -------
    dict
        Dictionary with the summed LCA impact values:
        - 'cbam_true_efactor': Total CBAM-relevant impact.
        - 'cbam_false_efactor': Total non-CBAM impact.
    """
    cbam_sums = defaultdict(float)

    with open(log_path, 'a') as log_file:
        for name, value, scope, is_cbam, exc_amount in data:
            if not isinstance(value, (float, np.float64)):
                continue  # Skip non-numeric values

            # This is an exception, if CBAM is previously indicated as False, but the impact is negative, for example, due to CCS, \
            # we account for it is as true, as such, direct emissions that should be substracted, note we would also get very strange results...
            # Better would be substract CO2 stored from direct emisions in scope 1 (in original inventories), but the activities for CCS are built in a different way.
            if exc_ccs:
                if not is_cbam and value<0 and scope == 3 and any(sub in name.lower() for sub in ['ccs', 'capture']):
                    cbam_sums['cbam_true'] += value
                    continue
                
            #normal procedure:
            if is_cbam and scope == 3:
                match = get_best_match(name)

                if match[0] != 'hydrogen':
                    impact_factor = CO2_emission_factor_precursors_kgCO2_kg.get(match[0])
                else:
                    match = get_best_match(name, choices=CO2_emission_factor_hydrogen)
                    ghg_smr = CO2_emission_factor_hydrogen.get('steam methane reforming')
                    calculated_lca_factor = (value / exc_amount)

                    if calculated_lca_factor < 4.4: #emission low-carbon hydrogen threshold Certifhy
                        impact_factor = 0
                    else:
                        impact_factor = ghg_smr
                    print(name, match, calculated_lca_factor,impact_factor)

                calculated_value = exc_amount * impact_factor

                if value - calculated_value < 0:
                    # Log the correction
                    corrected_factor_lca_based = alpha * (value / exc_amount)
                    corrected_value = exc_amount * corrected_factor_lca_based
                    cbam_sums['cbam_true'] += corrected_value
                    cbam_sums['cbam_false'] += value - corrected_value

                    log_file.write(
                        f"Correction applied:\n"
                        f"- Name: {name}\n"
                        f"- Match: {match[0]}\n"
                        f"- Original value: {value}\n"
                        f"- EF-derived value: {calculated_value}\n"
                        f"- Corrected value: {corrected_value}\n"
                        f"- LCA factor: {corrected_factor_lca_based}\n"
                        f"- EF factor used: {impact_factor}\n"
                        f"---\n"
                    )
                else:
                    cbam_sums['cbam_true'] += calculated_value
                    cbam_sums['cbam_false'] += value - calculated_value
            else:
                key = 'cbam_true' if is_cbam else 'cbam_false'
                cbam_sums[key] += value

    return {
        'cbam_true_efactor': cbam_sums['cbam_true'],
        'cbam_false_efactor': cbam_sums['cbam_false']
    }

def find_activity_fast(db_name, name_search, desired_ref_product, desired_location):
    """
    Quickly find a Brightway2 activity by name, location, and reference product using filtered search.

    Parameters:
    - db_name (str): Name of the Brightway2 database.
    - name_search (str): Exact name of the activity to find.
    - desired_ref_product (str or None): Reference product to filter activities by.
    - desired_location (str or None): Location to filter activities by.

    Returns:
    - dict: A single matching Brightway2 activity dictionary.

    Raises:
    - ValueError: If multiple matching activities are found.
    """
    # use search() function first, this is much faster to match activities.
    candidates = silent_search(db_name, name_search, desired_location)
    
    if len(candidates) != 1:
        matching_activities = [
            act for act in candidates
            if act['location']== desired_location and
            act['name'] == name_search
            and act['reference product'] == desired_ref_product
        ]
    else:
        return candidates[0]
        
    # Check if there is more than one matching activity
    if len(matching_activities) > 1:
        raise ValueError(f"More than one activity found for name '{name_search}' in location '{desired_location}'")
            
    elif len(matching_activities) == 0:
        #print(f"Cannot find activity using search function for name '{name_search}' in location '{desired_location}', using list comprehension instead.")
        matching_activities = try_find_bw_act(db_name, name_search, desired_ref_product, desired_location)
        return matching_activities  
    else:
        # Return the single matching activity
        return matching_activities[0]
    
def search_biosphere_entries(database, name, category, unit):
    """
    Search for entries in the biosphere database that match the specified name, category, and unit.

    Parameters:
    database (list of dict): The biosphere database to search, where each entry is a dictionary containing 'name', 'categories', and 'unit'.
    name (str): The name to match in the biosphere entries.
    category (str): The category to match in the biosphere entries.
    unit (str): The unit to match in the biosphere entries.

    Returns:
    list of dict: A list of biosphere entries that match the specified criteria.
    """
    matching_activities = [bio for bio in bw.Database(database)
            if bio['name'] == name and 
               bio['categories'] == category and 
               bio['unit'] == unit]
    
    # Check if there is more than one matching activity
    if len(matching_activities) > 1:
        raise ValueError(f"More than one activity found for name '{name}' with category '{category}'")
    
    # Check if no activity is found
    if len(matching_activities) == 0:
        raise ValueError(f"No activity found for name '{name}' with category '{category}'")
    
    # Return the single matching activity
    return matching_activities[0]
    
def try_find_bw_act(database_name, activity_name, ref_product, location):
    """
    Finds a single activity in a Brightway2 database matching the given name, reference product, and location.

    This function searches through the specified database for activities that match the provided
    name, reference product, and location. If more than one matching activity is found, or if no
    matching activities are found, it tries alternative locations ("GLO", "RoW", "RER", "Europe without Switzerland") in that order.

    Parameters:
    database_name (str): The name of the Brightway2 database to search in.
    activity_name (str): The name of the activity to search for.
    ref_product (str): The reference product of the activity to search for.
    location (str): The location of the activity to search for.

    Returns:
    object: The single matching activity.

    Raises:
    ValueError: If more than one matching activity is found.
    ValueError: If no matching activity is found.
    """

    locations_to_try = [location, "GLO", "RoW", "RER", "Europe without Switzerland", "ES", "FR"] #FR as ultimate proxy for renewables, as it is quite average reg wind and solar PV...

    for loc in locations_to_try:
        if 'market for electricity' in str(activity_name):
            for i, activity_name_l in enumerate([activity_name, activity_name.replace('market for electricity', 'market group for electricity')]):
                #if 'market group for electricity' not in str(activity_name) and i==1:
                    # give a warning since we take another activity than initially targeted fior power.
                    #print(f"WARNING: '{activity_name}' [{location}] not found, sourced electricity from 'market group for electricity' [{loc}] instead.")
                # Filter activities matching the given criteria
                matching_activities = [
                    act for act in bw.Database(database_name)
                    if activity_name_l == act['name'] and loc == act['location']
                    and ref_product == act['reference product']
                ]

                if len(matching_activities) == 1:
                    return matching_activities[0]  # Return the single matching activity

                elif len(matching_activities) > 1:
                    raise ValueError(f"More than one activity found for name '{activity_name}' in location '{loc}'")

        else:   
            # Filter activities matching the given criteria
            matching_activities = [
                act for act in bw.Database(database_name)
                if activity_name == act['name'] and loc == act['location']
                and ref_product == act['reference product']
            ]

        if len(matching_activities) == 1:
            return matching_activities[0]  # Return the single matching activity

        elif len(matching_activities) > 1:
            raise ValueError(f"More than one activity found for name '{activity_name}' in location '{loc}'")
    
    # If no activity is found after trying all locations
    raise ValueError(f"No activity found for name '{activity_name}' with the given locations")

def find_bw_act(database_name, activity_name, ref_product, location):
    """
    Finds a single activity in a Brightway2 database matching the given name, reference product, and location.
    
    This function searches through the specified database for activities that match the provided
    name, reference product, and location. If more than one matching activity is found, or if no
    matching activities are found, an error is raised.
    
    Parameters:
    database_name (str): The name of the Brightway2 database to search in.
    activity_name (str): The name of the activity to search for.
    ref_product (str): The reference product of the activity to search for.
    location (str): The location of the activity to search for.
    
    Returns:
    object: The single matching activity.
    
    Raises:
    ValueError: If more than one matching activity is found.
    ValueError: If no matching activity is found.
    
    Example:
    >>> find_bw_act("db_ei", "electrolyzer production, 1MWe, PEM, Stack", "stack", "RER")
    {
        'name': 'electrolyzer production, 1MWe, PEM, Stack',
        'reference product': 'stack',
        'location': 'RER',
        ...
    }
    """
    # Filter activities matching the given criteria
    matching_activities = [
        act for act in bw.Database(database_name) 
        if activity_name == act['name'] and location == act['location']
        and ref_product == act['reference product']
    ]
    
    # Check if there is more than one matching activity
    if len(matching_activities) > 1:
        raise ValueError(f"More than one activity found for name '{activity_name}' in location '{location}'")
    
    # Check if no activity is found
    if len(matching_activities) == 0:
        raise ValueError(f"No activity found for name '{activity_name}' in location '{location}'")
    
    # Return the single matching activity
    return matching_activities[0]

def annotate_exchanges_with_cbam(db_name, define_scope_cbam_func, cbam_precursors_steel_excl=CBAM_RELEVANT_PRECURSORS_STEEL, cbam_precursors_iron_excl=CBAM_RELEVANT_PRECURSORS_IRON, 
                                 annotation_scope = 'scope', annotation_cbam_included = 'cbam',start_idx=0):
    """
    Annotate exchanges in a Brightway2 database with CBAM-related metadata.

    Parameters:
    - db_name (str): Name of the Brightway2 database to process.
    - define_scope_cbam_func (Callable): A function that takes an exchange and 
      `CBAM_RELEVANT_PRECURSORS` as input, and returns a dictionary with keys 
      `'scope'` and `'cbam'`.
    - cbam_precursors_steel_excl (Iterable): A list or set of CBAM-relevant precursors to be excluded 
      from consideration when defining CBAM scope for steel.
    - cbam_precursors_iron_excl (Iterable): A list or set of CBAM-relevant precursors to be excluded 
      from consideration when defining CBAM scope for iron.
    - annotation_scope (str, optional): The exchange key where the scope value should be saved. 
      Default is `'scope'`.
    - annotation_cbam_included (str, optional): The exchange key where the CBAM inclusion flag 
      should be saved. Default is `'cbam'`.

    Returns:
    None [modified database with annotations of scope and cbam included]
    """
    acts = list(bw.Database(db_name))
    start_time = time.time()

    for i, act in enumerate(acts):
        steel_or_iron_act = parse_comment_field(act.get("comment", ""), "steel or iron production")
        # based on this, we need to decide to obtain the relevant precursors for CBAM scope definition, as some precursors are relevant for steel but not for iron, and vice versa.
        if steel_or_iron_act == 'steel':
            cbam_precursors_excl = cbam_precursors_steel_excl
        elif steel_or_iron_act == 'iron':
            cbam_precursors_excl = cbam_precursors_iron_excl   
        elif steel_or_iron_act == 'both':
            cbam_precursors_excl = cbam_precursors_iron_excl # iron precursors are overlapping but add hydrogen.
        else:
            print(f"WARNING: 'steel or iron production' comment field not found or not correctly filled for activity '{act['name']}' [{act['location']}], precursors relevant for both steel and iron as default.")

        for exc in act.exchanges():
            # Calculate elapsed and estimated remaining time
            elapsed_time = time.time() - start_time
            avg_time_per_iteration = elapsed_time / (i - start_idx + 1)
            remaining_iters = len(acts) - i - 1
            estimated_remaining_time = remaining_iters * avg_time_per_iteration

            print(f"Annotating scope/CBAM: activity {i}/{len(acts)} "
                  f"({(i / len(acts) * 100):.2f}%) | "
                  f"Elapsed: {elapsed_time:.1f}s | "
                  f"Remaining: {estimated_remaining_time:.0f}s", 
                  end='\r', flush=True)

            if exc['type'] != 'production':
                scope_cbam = define_scope_cbam_func(exc, cbam_relevant_precursors=cbam_precursors_excl)
                exc[annotation_scope] = scope_cbam['scope']
                exc[annotation_cbam_included] = scope_cbam['cbam']
                exc.save()
        act.save()

def annotate_act_exchanges_with_cbam(act, define_scope_cbam_func, cbam_precursors_excl, # 
                                 annotation_scope = 'scope', annotation_cbam_included = 'cbam'):
    """
    Annotate exchanges in a Brightway2 database with CBAM-related metadata.

    Parameters:
    - act: Brightway2 activity to process.
    - define_scope_cbam_func (Callable): A function that takes an exchange and 
      `CBAM_RELEVANT_PRECURSORS` as input, and returns a dictionary with keys 
      `'scope'` and `'cbam'`.
    - cbam_precursors_excl (Iterable): A list or set of CBAM-relevant precursors to be excluded 
      from consideration when defining CBAM scope.
    - annotation_scope (str, optional): The exchange key where the scope value should be saved. 
      Default is `'scope'`.
    - annotation_cbam_included (str, optional): The exchange key where the CBAM inclusion flag 
      should be saved. Default is `'cbam'`.

    Returns:
    None [modified database with annotations of scope and cbam included]
    """
    for exc in act.exchanges():
        if exc['type'] != 'production':
            scope_cbam = define_scope_cbam_func(exc, cbam_relevant_precursors=cbam_precursors_excl)
            exc[annotation_scope] = scope_cbam['scope']
            exc[annotation_cbam_included] = scope_cbam['cbam']
            exc.save()
    act.save()

def convert_tuple_to_string(input_tuple):
    """
    Converts a tuple of strings into a single string separated by '::'.
    """
    return "::".join(input_tuple)

def export_activity_to_excel(database_name, activity_code, output_file, comment='', source=''):
    """
    Exports a Brightway2 activity along with metadata and exchanges to an Excel file.
    """
    # Load the database
    db = bd.Database(database_name)
    
    # Retrieve the activity
    activity = db.get(code=activity_code)
    
    # Extract metadata
    metadata = {
        "Activity": activity.get('name', ''),
        "production amount": activity.get('production amount', ''),
        "reference product": activity.get('reference product', ''),
        #"type": activity.get('type', ''),
        "unit": activity.get('unit', ''),
        "location": activity.get('location', ''),
        "comment": comment,
        "source": source,  # Optional
    }
    
    # Convert metadata into a DataFrame
    metadata_df = pd.DataFrame(list(metadata.items()), columns=["Attribute", "Value"])
    
    # Extract exchanges
    exchanges_data = []
    for exc in activity.exchanges():
        exchanges_data.append({
            "name": exc.input['name'],
            "amount": exc['amount'],
            "location": db.get(code=exc.input.key[1])['location'] if exc.get('type', "") != 'biosphere' else "",
            "unit": exc['unit'],
            "categories": convert_tuple_to_string(bd.Database('ecoinvent-3.10-biosphere').get(code=exc.input.key[1])['categories']) if exc.get('type', "") == 'biosphere' else "",
            "type": exc.get('type', ""),
            "reference product": db.get(code=exc.input.key[1])['reference product'] if exc.get('type', "") != 'biosphere' else "",
            "comment": exc.get("comment", ""),  # Optional
        })
    
    # Create a DataFrame for exchanges
    exchanges_df = pd.DataFrame(exchanges_data)
    
    # Write metadata and exchanges to Excel
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        # Convert metadata to a tabular format for top placement
        worksheet = writer.book.add_worksheet("lci")
        writer.sheets["lci"] = worksheet
        
        # Write metadata at the top
        row_offset = 0
        for idx, (key, value) in enumerate(metadata.items()):
            worksheet.write(row_offset + idx, 0, key)
            worksheet.write(row_offset + idx, 1, value)
        
        # NO blank row for separation
        row_offset += len(metadata) #+ 1
        
        # Write "Exchanges" header
        worksheet.write(row_offset, 0, "Exchanges")
        row_offset += 1
        
        # Write exchanges dataframe below the metadata
        for col_idx, column in enumerate(exchanges_df.columns):
            worksheet.write(row_offset, col_idx, column)
        
        for row_idx, row in enumerate(exchanges_df.values):
            for col_idx, cell_value in enumerate(row):
                worksheet.write(row_offset + 1 + row_idx, col_idx, cell_value)
    
    print(f"Activity exported to {output_file}")
    
def round_to_nearest(number, step=0.05, max_value=None):
    """
    Rounds a given float to the nearest specified step and optionally caps the result at a maximum value.

    Parameters:
    number (float): The number to be rounded.
    step (float): The step to which the number should be rounded. Default is 0.05.
    max_value (float or None): The maximum value to cap the result. Default is None (no cap).

    Returns:
    float: The number rounded to the nearest specified step and optionally capped at max_value.
    
    Example:
    >>> round_to_nearest(3.67)
    3.65
    >>> round_to_nearest(3.67, 0.1)
    3.7
    >>> round_to_nearest(0.75, max_value=0.7)
    0.7
    >>> round_to_nearest(0.75, max_value=None)
    0.75
    """
    result = round(number / step) * step
    if max_value is not None:
        return min(result, max_value)
    return result

def iso2_to_country(iso2_code):
    try:
        return pycountry.countries.get(alpha_2=iso2_code).name.replace(
            'Taiwan, Province of China', "Taiwan").replace(
            'Iran, Islamic Republic of', "Iran").replace(
            'Korea, Republic of', "South-Korea")
    except AttributeError:
        return None  # Handle missing cases gracefully
    
def get_value_from_tif(filepath, lon, lat):
    """
    Retrieve the value from a TIFF file at a specified longitude and latitude.

    Parameters:
        filepath (str): The path to the TIFF file.
        lon (float): The longitude of the point of interest.
        lat (float): The latitude of the point of interest.

    Returns:
        float: The value at the specified longitude and latitude.
    """
    # Open the raster TIFF file
    raster_data = ri.open_rasterio(filepath)

    # Slice the first band of the raster
    img_data = raster_data[0, :, :]

    # Use the .sel() method to retrieve the value of the nearest cell close to the POI
    value = img_data.sel(x=lon, y=lat, method="nearest").data.item()

    # the data for PV is in kWh/day/kWp, so we will have to do following modification
    if "PVOUT" in str(filepath):
        value = (value * 365)/HOURS_YR
    
    return value


def curve_fit_f(x, db, info_type):
    """
    Calculate the wind share based on curve fitting from optimization.

    Args:
        x (float): share of x.
        db (str): Database identifier.
        info_type (str): Type of information to retrieve from DF_RATIOS.

    Returns:
        float: ratio share.

    """

    a = DF_RATIOS.loc[(info_type,db)]['a']
    b = DF_RATIOS.loc[(info_type,db)]['b']
    c = DF_RATIOS.loc[(info_type,db)]['c']

    # calculates ratio from predefined curve fit based on cf
    ratio = 1 / (1 + np.exp(-(x - b) * a)) ** c if x>0 else 0

    return round(ratio,3)

def GENERATE_ACTS_GM_PV(db_ei, cf_pv, curtailment_pv, cf_electrolyzer=0.3, electrolyzer="pem", excel_col_name = 'ecoinvent_310_reference'):
    """
    Generates a new activity in the Brightway2 database for hydrogen production
    from electrolysis using photovoltaic (PV) ground-mounted systems.

    Parameters:
    db_ei (str): The name of the Brightway2 database to be used.
    cf_pv (float): capacity factor of solar PV system.
    curtailment_pv (float): curtailment of solar PV to account for oversizing.
    cf_electrolyzer (float): The capacity factor of the PV system, which is 0.3 according to IEA.
    type of electrolyzer (str): Type of electrolyzer (can be 'pem', 'aec', or 'soec')

    Returns:
    None: The function performs operations directly on the Brightway2 database.

    Description:
    This function creates a new activity for hydrogen production using electrolysis powered by
    ground-mounted photovoltaic systems. It calculates the necessary inputs and adds exchanges
    for the activity based on the specified capacity factor of the PV system. The activity is 
    only created if it does not already exist in the database.

    Steps performed:
    1. Check if the activity already exists in the database.
    2. Retrieve necessary technical and cost data.
    3. Create the new activity with the specified parameters.
    4. Add exchanges for:
        - Stack electrolyzer production and treatment
        - Balance of plant electrolyzer production and treatment
        - Water consumption
        - PV infrastructure and water usage
        - Biosphere flows for waste heat, solar energy, hydrogen leakage, and oxygen production.

    Example:
    >>> GENERATE_ACTS_gm_pv('ecoinvent_3.7.1', 0.18)
    Skipped: activity 'hydrogen production, gaseous, 30 bar, from PEM electrolysis, solar PV ground-mounted, global cf [0.18]' already generated!
    """

    capacity = 560 * 0.895 #kWp, 10.5% average degradation, described in activity
    kg_water_unit= 2871.8 * 20 #20 L per m2 module, 2871.8 square meter for open ground construction, on ground, Mont Soleil

    new_name = "hydrogen production, gaseous, {} bar, from {} electrolysis, solar PV ground-mounted, global cf [{}]".format(BAR_ACT[electrolyzer],electrolyzer.upper(), round(cf_pv,3))
    check_act = [act for act in bw.Database(db_ei) if new_name == act['name']]
    
    tech = "solar_pv_gm"
    
    lifetime_pv = COST_DATA.loc[(tech,'lifetime')][excel_col_name].item()
    eff_elect = COST_DATA.loc[('electrolyzer_{}'.format(electrolyzer),'eff')][excel_col_name]
    lifetime_stack = COST_DATA.loc[('electrolyzer_{}'.format(electrolyzer),'stack_lt')][excel_col_name]  
    lifetime_bop = COST_DATA.loc[('electrolyzer_{}'.format(electrolyzer),'bop_lt')][excel_col_name]  
    h2_leakage_factor = COST_DATA.loc[('h2_leakage','-')][excel_col_name] 
    land_m2_kw = COST_DATA.loc[('electrolyzer_{}'.format(electrolyzer),'land_m2_kw')][excel_col_name]  
    
    if len(check_act) > 0:
        print("Skipped: activity '{}' already generated!".format(new_name))
    else:
        db = bw.Database(db_ei)
        code_na = str(uuid.uuid4().hex)
        act = db.new_activity(
            **{
                'name': new_name,
                 "code": code_na,
                'unit': 'kilogram',
                'reference product': "hydrogen, gaseous, {} bar".format(BAR_ACT[electrolyzer]),
                'location' :"GLO",
                'production amount': 1.0,
                'comment': f"Hydrogen production via water electrolysis with a \
                    {electrolyzer} electrolyzer with a capacity factor of {round(cf_electrolyzer,2)} and solar PV capacity factor of {round(cf_pv,3)}"
            }
        )

        act.save()

        # Add production exchange
        act.new_exchange(**{
            'input': (db_ei, code_na),
            'amount': 1,
            'type': 'production',
        }).save()

        # calculate electricity requirements
        amount_electricity = (EN_H2/(3.6*eff_elect))

        # 1.1. Add stack electrolyzer
        if electrolyzer == "pem":
            add_act = find_bw_act(db_ei, 'electrolyzer production, 1MWe, PEM, Stack', 'electrolyzer, 1MWe, PEM, Stack', 'RER')

            amount_elect_unit = 1/(1000*cf_electrolyzer*HOURS_YR*lifetime_stack/amount_electricity)

            act.new_exchange(amount= amount_elect_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()
            
            add_act = find_bw_act(db_ei, 'treatment of fuel cell stack, 1MWe, PEM', 'used fuel cell stack, 1MWe, PEM', 'RER')

            act.new_exchange(amount= -amount_elect_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()

            # 1.2. Add BoP electrolyzer
            add_act = find_bw_act(db_ei, 'electrolyzer production, 1MWe, PEM, Balance of Plant', 'electrolyzer, 1MWe, PEM, Balance of Plant', 'RER')

            amount_bop_unit = (lifetime_stack/lifetime_bop)/(1000*cf_electrolyzer*HOURS_YR*lifetime_stack/amount_electricity)  #lifetime bop is 20 years

            act.new_exchange(amount=amount_bop_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()
            
            add_act = find_bw_act(db_ei, 'treatment of fuel cell balance of plant, 1MWe, PEM', 'used fuel cell balance of plant, 1MWe, PEM', 'RER')

            act.new_exchange(amount= -amount_bop_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()

        elif electrolyzer == "aec":         
            add_act = find_bw_act(db_ei, 'electrolyzer production, 1MWe, AEC, Stack', 'electrolyzer, 1MWe, AEC, Stack', 'RER')

            amount_elect_unit = 1/(1000*cf_electrolyzer*HOURS_YR*lifetime_stack/amount_electricity)

            act.new_exchange(amount= amount_elect_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()

            add_act = find_bw_act(db_ei, 'treatment of fuel cell stack, 1MWe, AEC', 'used fuel cell stack, 1MWe, AEC', 'RER')

            act.new_exchange(amount= -amount_elect_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()

            # 1.2. Add BoP electrolyzer            
            add_act = find_bw_act(db_ei, 'electrolyzer production, 1MWe, AEC, Balance of Plant', 'electrolyzer, 1MWe, AEC, Balance of Plant', 'RER')

            amount_bop_unit = (lifetime_stack/lifetime_bop)/(1000*cf_electrolyzer*HOURS_YR*lifetime_stack/amount_electricity)  #lifetime bop is 20 years

            act.new_exchange(amount=amount_bop_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()

            add_act = find_bw_act(db_ei, 'treatment of fuel cell balance of plant, 1MWe, AEC', 'used fuel cell balance of plant, 1MWe, AEC', 'RER')

            act.new_exchange(amount= -amount_bop_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()

            # 1.2. Add potassium hydroxide
            add_act = find_bw_act(db_ei, 'market for potassium hydroxide', 'potassium hydroxide', 'GLO')

            act.new_exchange(amount=3.70*1e-3, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()

        elif electrolyzer == 'soec':  
            add_act = find_bw_act(db_ei, 'electrolyzer production, 1MWe, SOEC, Stack', 'electrolyzer, 1MWe, SOEC, Stack', 'RER')

            amount_elect_unit = 1/(1000*cf_electrolyzer*HOURS_YR*lifetime_stack/amount_electricity)

            act.new_exchange(amount= amount_elect_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()

            add_act = find_bw_act(db_ei, 'treatment of fuel cell stack, 1MWe, SOEC', 'used fuel cell stack, 1MWe, SOEC', 'RER')

            act.new_exchange(amount= -amount_elect_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()

            # 1.2. Add BoP electrolyzer            
            add_act = find_bw_act(db_ei, 'electrolyzer production, 1MWe, SOEC, Balance of Plant', 'electrolyzer, 1MWe, SOEC, Balance of Plant', 'RER')

            amount_bop_unit = (lifetime_stack/lifetime_bop)/(1000*cf_electrolyzer*HOURS_YR*lifetime_stack/amount_electricity)  #lifetime bop is 20 years

            act.new_exchange(amount=amount_bop_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()

            add_act = find_bw_act(db_ei, 'treatment of fuel cell balance of plant, 1MWe, SOEC', 'used fuel cell balance of plant, 1MWe, SOEC', 'RER')

            act.new_exchange(amount= -amount_bop_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()
        else:
            raise ValueError("Not possible to user electrolyzer exchanges since electrolyzer '{}' doesn't exist in LCI".format(electrolyzer))

        # 1.3. Add direct water consumption        
        add_act = find_bw_act(db_ei, 'market for water, deionised', 'water, deionised', 'RoW')

        # Add the exchange
        act.new_exchange(input=add_act.key,amount=AMOUNT_WATER_ELECTROLYSIS,
                                    unit="kilogram",type='technosphere', location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product']).save()

        # 1.4. Add infrastructure activity, PV unit
        add_act = find_bw_act(db_ei, 'photovoltaic open ground installation, 560 kWp, single-Si, on open ground', 
                              'photovoltaic open ground installation, 560 kWp, single-Si, on open ground', 'CH')
        amount_kwh = 1/(capacity*lifetime_pv*cf_pv*HOURS_YR*(1-curtailment_pv))

        # Add the exchange
        act.new_exchange(input=add_act.key,amount=(amount_kwh*amount_electricity),
                                        unit="unit",type='technosphere').save()

        # 1.5 Add tap water
        #tap water	market for tap water	Europe without Switzerland	kilogram	
        add_act = find_bw_act(db_ei, 'market for tap water', 
                              'tap water', 'Europe without Switzerland')

        amount_kwh = kg_water_unit*(1/(capacity*lifetime_pv*cf_pv*HOURS_YR*(1-curtailment_pv)))

        # Add the exchange
        act.new_exchange(input=add_act.key,amount=(amount_kwh*amount_electricity),
                                        unit="kilogram",type='technosphere', location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product']).save()

        # 1.6 Add wastewater, from residence
        add_act = find_bw_act(db_ei, 'treatment of wastewater, average, wastewater treatment', 
                              'wastewater, average', 'RoW')
        amount_kwh = kg_water_unit*(1/(capacity*lifetime_pv*cf_pv*HOURS_YR*(1-curtailment_pv)))/1000

        # Add the exchange
        act.new_exchange(input=add_act.key,amount=(-amount_kwh*amount_electricity),
                                        unit="cubic meter",type='technosphere').save()

        # 1.7 Add biosphere flows
        bio_flow = search_biosphere_entries(BIOSPHERE_DB, 'Heat, waste', ('air', 'urban air close to ground'), 'megajoule')
        amount_kwh = 0.25027 # curtailment doesn't have influence here, as solar PV is switched off

        # Add the exchange
        act.new_exchange(input=bio_flow.key,amount=(amount_kwh*amount_electricity),
                                        unit="megajoule",type='biosphere').save()

        # 1.8 add solar energy converted
        bio_flow = search_biosphere_entries(BIOSPHERE_DB, 'Energy, solar, converted', ('natural resource', 'in air'), 'megajoule')

        amount_kwh = 3.8503 # curtailment doesn't have influence here, as solar PV is switched off
        # Add the exchange
        act.new_exchange(input=bio_flow.key,amount=(amount_kwh*amount_electricity),
                                        unit="megajoule",type='biosphere').save()
        
        
        # 1.9 Add biosphere flows - H2 Leakage
        bio_flow = search_biosphere_entries(BIOSPHERE_DB, 'Hydrogen', ('air',), 'kilogram')

        # Add the exchange
        act.new_exchange(input=bio_flow.key,amount=h2_leakage_factor,
                                        unit="kilogram",type='biosphere').save()
        
        # 1.10 Oxygen production
        bio_flow = search_biosphere_entries(BIOSPHERE_DB, 'Oxygen', ('air',), 'kilogram')

        # Add the exchange
        act.new_exchange(input=bio_flow.key,amount=8,
                                        unit="kilogram",type='biosphere').save()

        # Finally, add the biosphere flows for land occupation.
        # 1.11 Add biosphere flows - land occupation
        bio_flow = search_biosphere_entries(BIOSPHERE_DB, 'Occupation, industrial area', ('natural resource','land'), 'square meter-year')

        # Electrolyser land footprint: (land_m2_kw [m2/kW]*1000[kW/MW])/(H2_prod[kg H2 per lifetime]/lifetime_system[years])
        act.new_exchange(input=bio_flow.key,amount=((land_m2_kw*1000)/(1000*cf_electrolyzer*HOURS_YR/amount_electricity)),
                                        unit="square meter-year",type='biosphere').save()
        
        bio_flow = search_biosphere_entries(BIOSPHERE_DB, 'Transformation, from industrial area', ('natural resource','land'), 'square meter')   

        # Electrolyser land transformation: (land_m2_kw [m2/kW]*1000 [kW/MW])/H2_prod [kg H2]
        act.new_exchange(input=bio_flow.key,amount=(land_m2_kw*1000)/(1000*cf_electrolyzer*HOURS_YR*lifetime_bop/amount_electricity),
                                        unit="square meter",type='biosphere').save()
        
        bio_flow = search_biosphere_entries(BIOSPHERE_DB, 'Transformation, to industrial area', ('natural resource','land'), 'square meter')   
        # Electrolyser land transformation: (land_m2_kw [m2/kW]*1000 [kW/MW])/H2_prod [kg H2]
        act.new_exchange(input=bio_flow.key,amount=(land_m2_kw*1000)/(1000*cf_electrolyzer*HOURS_YR*lifetime_bop/amount_electricity),
                                        unit="square meter",type='biosphere').save()
        
        act.save()

def GENERATE_ACTS_WIND(db_ei, cf_wind, curtailment_wind, cf_electrolyzer=0.4, electrolyzer = "pem", excel_col_name = 'ecoinvent_310_reference'):
    """
    Generates a new activity in the Brightway2 database for hydrogen production
    from electrolysis using onshore wind energy.

    Parameters:
    db_ei (str): The name of the Brightway2 database to be used.
    cf_wind (float): capacity factor of onshore wind system.
    curtailment_wind (float): curtailment of onshore wind to account for oversizing.
    cf_electrolyzer (float): The capacity factor of the wind energy source, which is 0.4 according to IEA for onshore wind.
    type of electrolyzer (str): Type of electrolyzer (can be 'pem', 'aec', or 'soec')

    Returns:
    None: The function performs operations directly on the Brightway2 database.

    Description:
    This function creates a new activity for hydrogen production using electrolysis powered by
    onshore wind energy. It calculates the necessary inputs and adds exchanges for the activity
    based on the specified capacity factor. The activity is only created if it does not already exist
    in the database.

    Steps performed:
    1. Check if the activity already exists in the database.
    2. Retrieve necessary technical and cost data.
    3. Create the new activity with the specified parameters.
    4. Add exchanges for:
        - Stack electrolyzer production
        - Balance of plant electrolyzer production
        - Water consumption
        - Lubricating oil and its waste treatment
        - Wind turbine network connection
        - Biosphere flows for kinetic wind energy, hydrogen leakage, and oxygen production.
    
    Example:
    >>> GENERATE_ACTS_wind('ecoinvent_3.7.1', 0.25)
    Skipped: activity 'hydrogen production, gaseous, 30 bar, from PEM electrolysis, onshore wind, global cf [0.25]' already generated!
    """

    capacity = 2000 #kWp
    new_name = "hydrogen production, gaseous, {} bar, from {} electrolysis, onshore wind, global cf [{}]".format(BAR_ACT[electrolyzer],electrolyzer.upper(),round(cf_wind,3))
    check_act = [act for act in bw.Database(db_ei) if new_name == act['name']]
    
    tech = "onshore_wind"
    
    lifetime_wind = COST_DATA.loc[(tech,'lifetime')][excel_col_name]
    eff_elect = COST_DATA.loc[('electrolyzer_{}'.format(electrolyzer),'eff')][excel_col_name]
    lifetime_stack = COST_DATA.loc[('electrolyzer_{}'.format(electrolyzer),'stack_lt')][excel_col_name]  
    lifetime_bop = COST_DATA.loc[('electrolyzer_{}'.format(electrolyzer),'bop_lt')][excel_col_name]  
    h2_leakage_factor = COST_DATA.loc[('h2_leakage','-')][excel_col_name] 
    land_m2_kw = COST_DATA.loc[('electrolyzer_{}'.format(electrolyzer),'land_m2_kw')][excel_col_name]  

    if len(check_act) > 0:
        print("Skipped: activity '{}' already generated!".format(new_name))
    else:
        # Generate activity
        db = bw.Database(db_ei)
        code_na = str(uuid.uuid4().hex)
        act = db.new_activity(
            **{
                'name': new_name,
                 "code": code_na,
                'unit': 'kilogram',
                'reference product': "hydrogen, gaseous, {} bar".format(BAR_ACT[electrolyzer]),
                'location' :"GLO",
                'production amount': 1.0,
                'comment': f"Hydrogen production via water electrolysis with a \
                    {electrolyzer} electrolyzer with a capacity factor of {round(cf_electrolyzer,2)} and onshore wind capacity factor of {round(cf_wind,2)}"
            }
        )

        act.save()

        # Add production exchange
        act.new_exchange(**{
            'input': (db_ei, code_na),
            'amount': 1,
            'type': 'production',
        }).save()

        # calculate electricity requirements
        amount_electricity = (EN_H2/(3.6*eff_elect))

        # 1.1. Add stack electrolyzer
        if electrolyzer == "pem":
            add_act = find_bw_act(db_ei, 'electrolyzer production, 1MWe, PEM, Stack', 'electrolyzer, 1MWe, PEM, Stack', 'RER')

            amount_elect_unit = 1/(1000*cf_electrolyzer*HOURS_YR*lifetime_stack/amount_electricity)

            act.new_exchange(amount= amount_elect_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()
            
            add_act = find_bw_act(db_ei, 'treatment of fuel cell stack, 1MWe, PEM', 'used fuel cell stack, 1MWe, PEM', 'RER')

            act.new_exchange(amount= -amount_elect_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()

            # 1.2. Add BoP electrolyzer
            add_act = find_bw_act(db_ei, 'electrolyzer production, 1MWe, PEM, Balance of Plant', 'electrolyzer, 1MWe, PEM, Balance of Plant', 'RER')

            amount_bop_unit = (lifetime_stack/lifetime_bop)/(1000*cf_electrolyzer*HOURS_YR*lifetime_stack/amount_electricity)  #lifetime bop is 20 years

            act.new_exchange(amount=amount_bop_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()
            
            add_act = find_bw_act(db_ei, 'treatment of fuel cell balance of plant, 1MWe, PEM', 'used fuel cell balance of plant, 1MWe, PEM', 'RER')

            act.new_exchange(amount= -amount_bop_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()

        elif electrolyzer == "aec":         
            add_act = find_bw_act(db_ei, 'electrolyzer production, 1MWe, AEC, Stack', 'electrolyzer, 1MWe, AEC, Stack', 'RER')

            amount_elect_unit = 1/(1000*cf_electrolyzer*HOURS_YR*lifetime_stack/amount_electricity)

            act.new_exchange(amount= amount_elect_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()

            add_act = find_bw_act(db_ei, 'treatment of fuel cell stack, 1MWe, AEC', 'used fuel cell stack, 1MWe, AEC', 'RER')

            act.new_exchange(amount= -amount_elect_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()

            # 1.2. Add BoP electrolyzer            
            add_act = find_bw_act(db_ei, 'electrolyzer production, 1MWe, AEC, Balance of Plant', 'electrolyzer, 1MWe, AEC, Balance of Plant', 'RER')

            amount_bop_unit = (lifetime_stack/lifetime_bop)/(1000*cf_electrolyzer*HOURS_YR*lifetime_stack/amount_electricity)  #lifetime bop is 20 years

            act.new_exchange(amount=amount_bop_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()

            add_act = find_bw_act(db_ei, 'treatment of fuel cell balance of plant, 1MWe, AEC', 'used fuel cell balance of plant, 1MWe, AEC', 'RER')

            act.new_exchange(amount= -amount_bop_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()

            # 1.2. Add potassium hydroxide
            add_act = find_bw_act(db_ei, 'market for potassium hydroxide', 'potassium hydroxide', 'GLO')

            act.new_exchange(amount=3.70*1e-3, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()

        elif electrolyzer == 'soec':  
            add_act = find_bw_act(db_ei, 'electrolyzer production, 1MWe, SOEC, Stack', 'electrolyzer, 1MWe, SOEC, Stack', 'RER')

            amount_elect_unit = 1/(1000*cf_electrolyzer*HOURS_YR*lifetime_stack/amount_electricity)

            act.new_exchange(amount= amount_elect_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()

            add_act = find_bw_act(db_ei, 'treatment of fuel cell stack, 1MWe, SOEC', 'used fuel cell stack, 1MWe, SOEC', 'RER')

            act.new_exchange(amount= -amount_elect_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()

            # 1.2. Add BoP electrolyzer            
            add_act = find_bw_act(db_ei, 'electrolyzer production, 1MWe, SOEC, Balance of Plant', 'electrolyzer, 1MWe, SOEC, Balance of Plant', 'RER')

            amount_bop_unit = (lifetime_stack/lifetime_bop)/(1000*cf_electrolyzer*HOURS_YR*lifetime_stack/amount_electricity)  #lifetime bop is 20 years

            act.new_exchange(amount=amount_bop_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()

            add_act = find_bw_act(db_ei, 'treatment of fuel cell balance of plant, 1MWe, SOEC', 'used fuel cell balance of plant, 1MWe, SOEC', 'RER')

            act.new_exchange(amount= -amount_bop_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()
        else:
            raise ValueError("Not possible to user electrolyzer exchanges since electrolyzer '{}' doesn't exist in LCI".format(electrolyzer))

        # 1.3. Add direct water consumption        
        add_act = find_bw_act(db_ei, 'market for water, deionised', 'water, deionised', 'RoW')

        # Add the exchange
        act.new_exchange(input=add_act.key,amount=AMOUNT_WATER_ELECTROLYSIS,
                                    unit="kilogram",type='technosphere', location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product']).save()

        # 1.4. Add infrastructure activity, PV unit
        add_act = find_bw_act(db_ei, 'market for lubricating oil', 'lubricating oil', 'RER')
        
        amount_kwh = 157.5/(capacity*cf_wind*HOURS_YR*(1-curtailment_wind))

        # Add the exchange
        act.new_exchange(input=add_act.key,amount=(amount_kwh*amount_electricity),
                                        unit="kilogram",type='technosphere', location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product']).save()

        # 1.5 Add infrastructure activities, lubricating oil waste mineral oil
        add_act = find_bw_act(db_ei, 'market for waste mineral oil', 'waste mineral oil', 'Europe without Switzerland')

        amount_kwh = 157.5/(capacity*cf_wind*HOURS_YR*(1-curtailment_wind))
        # Add the exchange
        act.new_exchange(input=add_act.key,amount=(-amount_kwh*amount_electricity),
                                        unit="kilogram",type='technosphere', location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product']).save()

        # 1.6 Add infrastructure activities, market for wind turbine network connection
        add_act = find_bw_act(db_ei, 'market for wind turbine network connection, 2MW, onshore', 'wind turbine network connection, 2MW, onshore', 'GLO')

        amount_kwh = 1/(capacity*lifetime_wind*cf_wind*HOURS_YR*(1-curtailment_wind))

        # Add the exchange
        act.new_exchange(input=add_act.key,amount=(amount_kwh*amount_electricity),
                                        unit="unit",type='technosphere').save()

        # 1.7 Add infrastructure activities, market for wind turbine network connection
        add_act = find_bw_act(db_ei, 'market for wind turbine, 2MW, onshore', 'wind turbine, 2MW, onshore', 'GLO')

        amount_kwh = 1/(capacity*lifetime_wind*cf_wind*HOURS_YR*(1-curtailment_wind))

        # Add the exchange
        act.new_exchange(input=add_act.key,amount=(amount_kwh*amount_electricity),
                                        unit="unit",type='technosphere').save()
        
        # 1.8 Add biosphere flows - kinteic wind
        bio_flow = search_biosphere_entries(BIOSPHERE_DB, 'Energy, kinetic (in wind), converted', ('natural resource', 'in air'), 'megajoule')
            
        amount_kwh = 3.87 # curtailment doesn't have influence here, as solar PV is switched off
        # Add the exchange
        act.new_exchange(input=bio_flow.key,amount=(amount_kwh*amount_electricity),
                                        unit="megajoule",type='biosphere').save()

        # 1.8 Add biosphere flows - H2 Leakage
        bio_flow = search_biosphere_entries(BIOSPHERE_DB, 'Hydrogen', ('air',), 'kilogram')

        # Add the exchange
        act.new_exchange(input=bio_flow.key,amount=h2_leakage_factor,
                                        unit="kilogram",type='biosphere').save()
        
        # 1.9 Oxygen production
        bio_flow = search_biosphere_entries(BIOSPHERE_DB, 'Oxygen', ('air',), 'kilogram')

        # Add the exchange
        act.new_exchange(input=bio_flow.key,amount=8,
                                        unit="kilogram",type='biosphere').save()

        # Finally, add the biosphere flows for land occupation.
        # 1.10 Add biosphere flows - land occupation
        bio_flow = search_biosphere_entries(BIOSPHERE_DB, 'Occupation, industrial area', ('natural resource','land'), 'square meter-year')

        # Electrolyser land footprint: (land_m2_kw [m2/kW]*1000[kW/MW])/(H2_prod[kg H2 per lifetime]/lifetime_system[years])
        act.new_exchange(input=bio_flow.key,amount=((land_m2_kw*1000)/(1000*cf_electrolyzer*HOURS_YR/amount_electricity)),
                                        unit="square meter-year",type='biosphere').save()
        
        bio_flow = search_biosphere_entries(BIOSPHERE_DB, 'Transformation, from industrial area', ('natural resource','land'), 'square meter')   

        # Electrolyser land transformation: (land_m2_kw [m2/kW]*1000 [kW/MW])/H2_prod [kg H2]
        act.new_exchange(input=bio_flow.key,amount=(land_m2_kw*1000)/(1000*cf_electrolyzer*HOURS_YR*lifetime_bop/amount_electricity),
                                        unit="square meter",type='biosphere').save()
        
        bio_flow = search_biosphere_entries(BIOSPHERE_DB, 'Transformation, to industrial area', ('natural resource','land'), 'square meter')   
        # Electrolyser land transformation: (land_m2_kw [m2/kW]*1000 [kW/MW])/H2_prod [kg H2]
        act.new_exchange(input=bio_flow.key,amount=(land_m2_kw*1000)/(1000*cf_electrolyzer*HOURS_YR*lifetime_bop/amount_electricity),
                                        unit="square meter",type='biosphere').save()
        
        act.save()

def GENERATE_ACTS_WIND_OFF(db_ei, cf_wind, curtailment_wind, cf_electrolyzer=0.55, electrolyzer = "pem",excel_col_name = 'ecoinvent_310_reference'):
    """
    Generates a new activity in the Brightway2 database for hydrogen production
    from electrolysis using offshore wind energy.

    Parameters:
    db_ei (str): The name of the Brightway2 database to be used.
    cf_wind (float): capacity factor of onshore wind system.
    curtailment_wind (float): curtailment of onshore wind to account for oversizing.
    cf_electrolyzer (float): The capacity factor of the wind energy source, which is 0.55 according to IEA database.
    type of electrolyzer (str): Type of electrolyzer (can be 'pem', 'aec', or 'soec')

    Returns:
    None: The function performs operations directly on the Brightway2 database.

    Description:
    This function creates a new activity for hydrogen production using electrolysis powered by
    offshore wind energy. It calculates the necessary inputs and adds exchanges for the activity
    based on the specified capacity factor. The activity is only created if it does not already exist
    in the database.

    Steps performed:
    1. Check if the activity already exists in the database.
    2. Retrieve necessary technical and cost data.
    3. Create the new activity with the specified parameters.
    4. Add exchanges for:
        - Stack electrolyzer production
        - Balance of plant electrolyzer production
        - Water consumption
        - Lubricating oil and its waste treatment
        - Wind power plant network connection
        - Biosphere flows for kinetic wind energy, hydrogen leakage, and oxygen production.
    
    Example:
    >>> GENERATE_ACTS_wind_off('ecoinvent_3.7.1', 0.25)
    Skipped: activity 'hydrogen production, gaseous, 30 bar, from PEM electrolysis, offshore wind, global cf [0.25]' already generated!
    """

    capacity = 2000 #kWp
    new_name = "hydrogen production, gaseous, {} bar, from {} electrolysis, offshore wind, global cf [{}]".format(BAR_ACT[electrolyzer],electrolyzer.upper(),round(cf_wind,3))
    check_act = [act for act in bw.Database(db_ei) if new_name == act['name']]
    
    tech = "offshore_wind"
    
    lifetime_wind = COST_DATA.loc[(tech,'lifetime')][excel_col_name]
    eff_elect = COST_DATA.loc[('electrolyzer_{}'.format(electrolyzer),'eff')][excel_col_name]
    lifetime_stack = COST_DATA.loc[('electrolyzer_{}'.format(electrolyzer),'stack_lt')][excel_col_name]
    lifetime_bop = COST_DATA.loc[('electrolyzer_{}'.format(electrolyzer),'bop_lt')][excel_col_name]    
    h2_leakage_factor = COST_DATA.loc[('h2_leakage','-')][excel_col_name] 
    land_m2_kw = COST_DATA.loc[('electrolyzer_{}'.format(electrolyzer),'land_m2_kw')][excel_col_name]  

    if len(check_act) > 0:
        print("Skipped: activity '{}' already generated!".format(new_name))
    else:
        # Generate activity
        db = bw.Database(db_ei)
        code_na = str(uuid.uuid4().hex)
        act = db.new_activity(
            **{
                'name': new_name,
                 "code": code_na,
                'unit': 'kilogram',
                'reference product': "hydrogen, gaseous, {} bar".format(BAR_ACT[electrolyzer]),
                'location' :"GLO",
                'production amount': 1.0,
                'comment': f"Hydrogen production via water electrolysis with a \
                    {electrolyzer} electrolyzer with a capacity factor of {round(cf_electrolyzer,2)} and onshore wind of capacity factor {round(cf_wind,2)}"
            }
        )

        act.save()

        # Add production exchange
        act.new_exchange(**{
            'input': (db_ei, code_na),
            'amount': 1,
            'type': 'production',
        }).save()

        # calculate electricity requirements
        amount_electricity = (EN_H2/(3.6*eff_elect))

        # 1.1. Add stack electrolyzer
        if electrolyzer == "pem":
            add_act = find_bw_act(db_ei, 'electrolyzer production, 1MWe, PEM, Stack', 'electrolyzer, 1MWe, PEM, Stack', 'RER')

            amount_elect_unit = 1/(1000*cf_electrolyzer*HOURS_YR*lifetime_stack/amount_electricity)

            act.new_exchange(amount= amount_elect_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()
            
            add_act = find_bw_act(db_ei, 'treatment of fuel cell stack, 1MWe, PEM', 'used fuel cell stack, 1MWe, PEM', 'RER')

            act.new_exchange(amount= -amount_elect_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()

            # 1.2. Add BoP electrolyzer
            add_act = find_bw_act(db_ei, 'electrolyzer production, 1MWe, PEM, Balance of Plant', 'electrolyzer, 1MWe, PEM, Balance of Plant', 'RER')

            amount_bop_unit = (lifetime_stack/lifetime_bop)/(1000*cf_electrolyzer*HOURS_YR*lifetime_stack/amount_electricity)  #lifetime bop is 20 years

            act.new_exchange(amount=amount_bop_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()
            
            add_act = find_bw_act(db_ei, 'treatment of fuel cell balance of plant, 1MWe, PEM', 'used fuel cell balance of plant, 1MWe, PEM', 'RER')

            act.new_exchange(amount= -amount_bop_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()

        elif electrolyzer == "aec":         
            add_act = find_bw_act(db_ei, 'electrolyzer production, 1MWe, AEC, Stack', 'electrolyzer, 1MWe, AEC, Stack', 'RER')

            amount_elect_unit = 1/(1000*cf_electrolyzer*HOURS_YR*lifetime_stack/amount_electricity)

            act.new_exchange(amount= amount_elect_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()

            add_act = find_bw_act(db_ei, 'treatment of fuel cell stack, 1MWe, AEC', 'used fuel cell stack, 1MWe, AEC', 'RER')

            act.new_exchange(amount= -amount_elect_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()

            # 1.2. Add BoP electrolyzer            
            add_act = find_bw_act(db_ei, 'electrolyzer production, 1MWe, AEC, Balance of Plant', 'electrolyzer, 1MWe, AEC, Balance of Plant', 'RER')

            amount_bop_unit = (lifetime_stack/lifetime_bop)/(1000*cf_electrolyzer*HOURS_YR*lifetime_stack/amount_electricity)  #lifetime bop is 20 years

            act.new_exchange(amount=amount_bop_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()

            add_act = find_bw_act(db_ei, 'treatment of fuel cell balance of plant, 1MWe, AEC', 'used fuel cell balance of plant, 1MWe, AEC', 'RER')

            act.new_exchange(amount= -amount_bop_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()

            # 1.2. Add potassium hydroxide
            add_act = find_bw_act(db_ei, 'market for potassium hydroxide', 'potassium hydroxide', 'GLO')

            act.new_exchange(amount=3.70*1e-3, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()

        elif electrolyzer == 'soec':  
            add_act = find_bw_act(db_ei, 'electrolyzer production, 1MWe, SOEC, Stack', 'electrolyzer, 1MWe, SOEC, Stack', 'RER')

            amount_elect_unit = 1/(1000*cf_electrolyzer*HOURS_YR*lifetime_stack/amount_electricity)

            act.new_exchange(amount= amount_elect_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()

            add_act = find_bw_act(db_ei, 'treatment of fuel cell stack, 1MWe, SOEC', 'used fuel cell stack, 1MWe, SOEC', 'RER')

            act.new_exchange(amount= -amount_elect_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()

            # 1.2. Add BoP electrolyzer            
            add_act = find_bw_act(db_ei, 'electrolyzer production, 1MWe, SOEC, Balance of Plant', 'electrolyzer, 1MWe, SOEC, Balance of Plant', 'RER')

            amount_bop_unit = (lifetime_stack/lifetime_bop)/(1000*cf_electrolyzer*HOURS_YR*lifetime_stack/amount_electricity)  #lifetime bop is 20 years

            act.new_exchange(amount=amount_bop_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()

            add_act = find_bw_act(db_ei, 'treatment of fuel cell balance of plant, 1MWe, SOEC', 'used fuel cell balance of plant, 1MWe, SOEC', 'RER')

            act.new_exchange(amount= -amount_bop_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()
        else:
            raise ValueError("Not possible to user electrolyzer exchanges since electrolyzer '{}' doesn't exist in LCI".format(electrolyzer))

        # 1.3. Add direct water consumption        
        add_act = find_bw_act(db_ei, 'market for water, deionised', 'water, deionised', 'RoW')

        # Add the exchange
        act.new_exchange(input=add_act.key,amount=AMOUNT_WATER_ELECTROLYSIS,
                                    unit="kilogram",type='technosphere', location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product']).save()
        
        # 1.4. Add infrastructure activity
        add_act = find_bw_act(db_ei, 'market for lubricating oil', 'lubricating oil', 'RER')
        
        amount_kwh = 157.5/(capacity*cf_wind*HOURS_YR*(1-curtailment_wind))

        # Add the exchange
        act.new_exchange(input=add_act.key,amount=(amount_kwh*amount_electricity),
                                        unit="kilogram",type='technosphere', location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product']).save()

        # 1.5 Add infrastructure activities, lubricating oil waste mineral oil
        add_act = find_bw_act(db_ei, 'market for waste mineral oil', 'waste mineral oil', 'Europe without Switzerland')

        amount_kwh = 157.5/(capacity*cf_wind*HOURS_YR*(1-curtailment_wind))

        # 1.6 Add infrastructure activities, market for wind turbine network connection
        add_act = find_bw_act(db_ei, 'market for wind power plant, 2MW, offshore, fixed parts', 'wind power plant, 2MW, offshore, fixed parts', 'GLO')
        amount_kwh = 1/(capacity*lifetime_wind*cf_wind*HOURS_YR*(1-curtailment_wind))

        # Add the exchange
        act.new_exchange(input=add_act.key,amount=(amount_kwh*amount_electricity),
                                        unit="unit",type='technosphere').save()

        # 1.7 Add infrastructure activities, market for wind turbine network connection
        add_act = find_bw_act(db_ei, 'market for wind power plant, 2MW, offshore, moving parts', 'wind power plant, 2MW, offshore, moving parts', 'GLO')

        amount_kwh = 1/(capacity*lifetime_wind*cf_wind*HOURS_YR*(1-curtailment_wind))
        # Add the exchange
        act.new_exchange(input=add_act.key,amount=(amount_kwh*amount_electricity),
                                        unit="unit",type='technosphere').save()
        
        # 1.8 Add biosphere flows - kinteic wind
        bio_flow = search_biosphere_entries(BIOSPHERE_DB, 'Energy, kinetic (in wind), converted', ('natural resource', 'in air'), 'megajoule')
            
        amount_kwh = 3.87 # curtailment doesn't have influence here, as solar PV is switched off

        # Add the exchange
        act.new_exchange(input=bio_flow.key,amount=(amount_kwh*amount_electricity),
                                        unit="megajoule",type='biosphere').save()

        # 1.9 Add biosphere flows - H2 Leakage
        bio_flow = search_biosphere_entries(BIOSPHERE_DB, 'Hydrogen', ('air',), 'kilogram')

        # Add the exchange
        act.new_exchange(input=bio_flow.key,amount=h2_leakage_factor,
                                        unit="kilogram",type='biosphere').save()
        
        # 1.10 Oxygen production
        bio_flow = search_biosphere_entries(BIOSPHERE_DB, 'Oxygen', ('air',), 'kilogram')

        # Add the exchange
        act.new_exchange(input=bio_flow.key,amount=8,
                                        unit="kilogram",type='biosphere').save()

        # Finally, add the biosphere flows for land occupation.
        # 1.11 Add biosphere flows - land occupation
        bio_flow = search_biosphere_entries(BIOSPHERE_DB, 'Occupation, industrial area', ('natural resource','land'), 'square meter-year')

        # Electrolyser land footprint: (land_m2_kw [m2/kW]*1000[kW/MW])/(H2_prod[kg H2 per lifetime]/lifetime_system[years])
        act.new_exchange(input=bio_flow.key,amount=((land_m2_kw*1000)/(1000*cf_electrolyzer*HOURS_YR/amount_electricity)),
                                        unit="square meter-year",type='biosphere').save()
        
        bio_flow = search_biosphere_entries(BIOSPHERE_DB, 'Transformation, from industrial area', ('natural resource','land'), 'square meter')   

        # Electrolyser land transformation: (land_m2_kw [m2/kW]*1000 [kW/MW])/H2_prod [kg H2]
        act.new_exchange(input=bio_flow.key,amount=(land_m2_kw*1000)/(1000*cf_electrolyzer*HOURS_YR*lifetime_bop/amount_electricity),
                                        unit="square meter",type='biosphere').save()
        
        bio_flow = search_biosphere_entries(BIOSPHERE_DB, 'Transformation, to industrial area', ('natural resource','land'), 'square meter')   
        # Electrolyser land transformation: (land_m2_kw [m2/kW]*1000 [kW/MW])/H2_prod [kg H2]
        act.new_exchange(input=bio_flow.key,amount=(land_m2_kw*1000)/(1000*cf_electrolyzer*HOURS_YR*lifetime_bop/amount_electricity),
                                        unit="square meter",type='biosphere').save()
        
        act.save()

def GENERATE_ACTS_GENERIC_ELECT(db_ei, cf_electrolyzer=0.57, electrolyzer = "pem",
                                generic_electr_act = ["electricity production, hydro, reservoir, alpine region",
                                                      'electricity, high voltage',
                                                      "RoW"], excel_col_name = 'ecoinvent_310_reference'
                                ):
    """
    Generates a new activity in the Brightway2 database for hydrogen production
    from electrolysis using power from a specified power source.

    Parameters
    ----------
    db_ei : str
        Name of the Brightway2 database.
    cf_electrolyzer : float, optional
        Capacity factor for the technology. Default is 0.57.
    electrolyzer : str, optional
        Type of electrolyzer to be used. Can be either "pem" or "aec". Default is "pem".
    generic_electr_act : list, optional
        List containing three elements that specify the electricity production activity:
        1. Name of the electricity production method.
        2. Voltage level.
        3. Region. Default is ["electricity production, hydro, reservoir, alpine region",
                               'electricity, high voltage', "RoW"].

    Returns
    -------
    None
        The function does not return any value. It generates a new activity in the Brightway2 
        database or prints a message if the activity already exists.

    Description
    -----------
    The function performs the following steps:

    1. Activity Name Generation:
       - Constructs a new activity name based on the type of electrolyzer and the electricity 
         production method.

    2. Activity Check:
       - Checks if the activity already exists in the database. If it does, the function prints a 
         message and skips the generation.

    3. Electrolyzer Efficiency and Lifetime:
       - Retrieves efficiency and lifetime values for the selected electrolyzer type from the 
         COST_DATA dataframe.

    4. Activity Creation:
       - Creates a new activity in the database with specified properties.

    5. Adding Exchanges:
       - Adds various exchanges related to the electrolyzer stack, balance of plant (BoP), water 
         consumption, power consumption, and biosphere flows for hydrogen leakage and oxygen.

    Example Usage
    -------------
    GENERATE_ACTS_GENERIC_ELECT("my_database", cf_electrolyzer=0.60, electrolyzer="aec",
                                generic_electr_act=["electricity production, wind, offshore", 
                                                    'electricity, high voltage', "US"])
    
    This will generate a new activity for hydrogen production using an AEC electrolyzer with power 
    from offshore wind electricity production in the US region.
    """

    # This is to make sure that the matching complies for activities based on grid power and having larger aggregated scales.
    if generic_electr_act[0] == 'market group for electricity, medium voltage':
        act_new_name = 'market for electricity, medium voltage'
    else:
        act_new_name = generic_electr_act[0]

    new_name = "hydrogen production, gaseous, {} bar, from {} electrolysis, power from {}".format(BAR_ACT[electrolyzer],electrolyzer.upper(),act_new_name)
    check_act = [act for act in bw.Database(db_ei) if new_name == act['name'] and generic_electr_act[2] == act['location']]
    
    eff_elect = COST_DATA.loc[('electrolyzer_{}'.format(electrolyzer),'eff')][excel_col_name]
    lifetime_stack = COST_DATA.loc[('electrolyzer_{}'.format(electrolyzer),'stack_lt')][excel_col_name]  
    lifetime_bop = COST_DATA.loc[('electrolyzer_{}'.format(electrolyzer),'bop_lt')][excel_col_name]  
    h2_leakage_factor = COST_DATA.loc[('h2_leakage','-')][excel_col_name] 
    land_m2_kw = COST_DATA.loc[('electrolyzer_{}'.format(electrolyzer),'land_m2_kw')][excel_col_name]  

    if len(check_act) > 0:
        print("Skipped: activity '{}' already generated!".format(new_name))
    else:
        # Generate activity
        db = bw.Database(db_ei)
        code_na = str(uuid.uuid4().hex)
        act = db.new_activity(
            **{
                'name': new_name,
                 "code": code_na,
                'unit': 'kilogram',
                'reference product': "hydrogen, gaseous, {} bar".format(BAR_ACT[electrolyzer]),
                'location' :generic_electr_act[2],
                'production amount': 1.0,
            }
        )

        act.save()

        # Add production exchange
        act.new_exchange(**{
            'input': (db_ei, code_na),
            'amount': 1,
            'type': 'production',
        }).save()

        # calculate electricity requirements
        amount_electricity = (EN_H2/(3.6*eff_elect))

        # 1.1. Add stack electrolyzer
        if electrolyzer == "pem":
            add_act = find_bw_act(db_ei, 'electrolyzer production, 1MWe, PEM, Stack', 'electrolyzer, 1MWe, PEM, Stack', 'RER')

            amount_elect_unit = 1/(1000*cf_electrolyzer*HOURS_YR*lifetime_stack/amount_electricity)

            act.new_exchange(amount= amount_elect_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()
            
            add_act = find_bw_act(db_ei, 'treatment of fuel cell stack, 1MWe, PEM', 'used fuel cell stack, 1MWe, PEM', 'RER')

            act.new_exchange(amount= -amount_elect_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()

            # 1.2. Add BoP electrolyzer
            add_act = find_bw_act(db_ei, 'electrolyzer production, 1MWe, PEM, Balance of Plant', 'electrolyzer, 1MWe, PEM, Balance of Plant', 'RER')

            amount_bop_unit = (lifetime_stack/lifetime_bop)/(1000*cf_electrolyzer*HOURS_YR*lifetime_stack/amount_electricity)  #lifetime bop is 20 years

            act.new_exchange(amount=amount_bop_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()
            
            add_act = find_bw_act(db_ei, 'treatment of fuel cell balance of plant, 1MWe, PEM', 'used fuel cell balance of plant, 1MWe, PEM', 'RER')

            act.new_exchange(amount= -amount_bop_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()

        elif electrolyzer == "aec":         
            add_act = find_bw_act(db_ei, 'electrolyzer production, 1MWe, AEC, Stack', 'electrolyzer, 1MWe, AEC, Stack', 'RER')

            amount_elect_unit = 1/(1000*cf_electrolyzer*HOURS_YR*lifetime_stack/amount_electricity)

            act.new_exchange(amount= amount_elect_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()

            add_act = find_bw_act(db_ei, 'treatment of fuel cell stack, 1MWe, AEC', 'used fuel cell stack, 1MWe, AEC', 'RER')

            act.new_exchange(amount= -amount_elect_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()

            # 1.2. Add BoP electrolyzer            
            add_act = find_bw_act(db_ei, 'electrolyzer production, 1MWe, AEC, Balance of Plant', 'electrolyzer, 1MWe, AEC, Balance of Plant', 'RER')

            amount_bop_unit = (lifetime_stack/lifetime_bop)/(1000*cf_electrolyzer*HOURS_YR*lifetime_stack/amount_electricity)  #lifetime bop is 20 years

            act.new_exchange(amount=amount_bop_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()

            add_act = find_bw_act(db_ei, 'treatment of fuel cell balance of plant, 1MWe, AEC', 'used fuel cell balance of plant, 1MWe, AEC', 'RER')

            act.new_exchange(amount= -amount_bop_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()

            # 1.2. Add potassium hydroxide
            add_act = find_bw_act(db_ei, 'market for potassium hydroxide', 'potassium hydroxide', 'GLO')

            act.new_exchange(amount=3.70*1e-3, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()

        elif electrolyzer == 'soec':  
            add_act = find_bw_act(db_ei, 'electrolyzer production, 1MWe, SOEC, Stack', 'electrolyzer, 1MWe, SOEC, Stack', 'RER')

            amount_elect_unit = 1/(1000*cf_electrolyzer*HOURS_YR*lifetime_stack/amount_electricity)

            act.new_exchange(amount= amount_elect_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()

            add_act = find_bw_act(db_ei, 'treatment of fuel cell stack, 1MWe, SOEC', 'used fuel cell stack, 1MWe, SOEC', 'RER')

            act.new_exchange(amount= -amount_elect_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()

            # 1.2. Add BoP electrolyzer            
            add_act = find_bw_act(db_ei, 'electrolyzer production, 1MWe, SOEC, Balance of Plant', 'electrolyzer, 1MWe, SOEC, Balance of Plant', 'RER')

            amount_bop_unit = (lifetime_stack/lifetime_bop)/(1000*cf_electrolyzer*HOURS_YR*lifetime_stack/amount_electricity)  #lifetime bop is 20 years

            act.new_exchange(amount=amount_bop_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()

            add_act = find_bw_act(db_ei, 'treatment of fuel cell balance of plant, 1MWe, SOEC', 'used fuel cell balance of plant, 1MWe, SOEC', 'RER')

            act.new_exchange(amount= -amount_bop_unit, input = add_act.key, type="technosphere", location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product'] ).save()
        else:
            raise ValueError("Not possible to user electrolyzer exchanges since electrolyzer '{}' doesn't exist in LCI".format(electrolyzer))

        # 1.3. Add direct water consumption        
        add_act = find_bw_act(db_ei, 'market for water, deionised', 'water, deionised', 'RoW')

        # Add the exchange
        act.new_exchange(input=add_act.key,amount=AMOUNT_WATER_ELECTROLYSIS,
                                    unit="kilogram",type='technosphere', location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product']).save()

        # Find the power activity as given in the function
        add_act = find_bw_act(db_ei, generic_electr_act[0],
                                  generic_electr_act[1], 
                                  generic_electr_act[2])

        # Add the exchange
        act.new_exchange(input=add_act.key,amount=amount_electricity,
                                    unit="kilogram",type='technosphere', location = add_act['location'],
                             name = add_act['name'], product = add_act['reference product']).save()

        # 1.4 Add biosphere flows - H2 Leakage
        bio_flow = search_biosphere_entries(BIOSPHERE_DB, 'Hydrogen', ('air',), 'kilogram')

        # Add the exchange
        act.new_exchange(input=bio_flow.key,amount=h2_leakage_factor,
                                        unit="kilogram",type='biosphere').save()
        
        # 1.6 Oxygen production
        bio_flow = search_biosphere_entries(BIOSPHERE_DB, 'Oxygen', ('air',), 'kilogram')

        # Add the exchange
        act.new_exchange(input=bio_flow.key,amount=8,
                                        unit="kilogram",type='biosphere').save()

        # Finally, add the biosphere flows for land occupation.
        # 1.7 Add biosphere flows - land occupation
        bio_flow = search_biosphere_entries(BIOSPHERE_DB, 'Occupation, industrial area', ('natural resource','land'), 'square meter-year')

        # Electrolyser land footprint: (land_m2_kw [m2/kW]*1000[kW/MW])/(H2_prod[kg H2 per lifetime]/lifetime_system[years])
        act.new_exchange(input=bio_flow.key,amount=((land_m2_kw*1000)/(1000*cf_electrolyzer*HOURS_YR/amount_electricity)),
                                        unit="square meter-year",type='biosphere').save()
        
        bio_flow = search_biosphere_entries(BIOSPHERE_DB, 'Transformation, from industrial area', ('natural resource','land'), 'square meter')   

        # Electrolyser land transformation: (land_m2_kw [m2/kW]*1000 [kW/MW])/H2_prod [kg H2]
        act.new_exchange(input=bio_flow.key,amount=(land_m2_kw*1000)/(1000*cf_electrolyzer*HOURS_YR*lifetime_bop/amount_electricity),
                                        unit="square meter",type='biosphere').save()
        
        bio_flow = search_biosphere_entries(BIOSPHERE_DB, 'Transformation, to industrial area', ('natural resource','land'), 'square meter')   
        # Electrolyser land transformation: (land_m2_kw [m2/kW]*1000 [kW/MW])/H2_prod [kg H2]
        act.new_exchange(input=bio_flow.key,amount=(land_m2_kw*1000)/(1000*cf_electrolyzer*HOURS_YR*lifetime_bop/amount_electricity),
                                        unit="square meter",type='biosphere').save()
        
        act.save()


def create_market_activity(name_comm, results_df, db_sel, db_name, db_name_markets, 
                           market_name, location_filter_fn, market_location_label, w, max_year=2026):
    """
    Create a new market activity for a specified group of countries based on production volume shares.

    This function filters a results DataFrame for a given group of countries using a user-defined 
    filtering function and constructs a new market activity that combines all relevant steel 
    production activities. The contribution of each region to the market is weighted by its 
    relative production volume. The function is tailored for use in Brightway2.

    Parameters:
    ----------
    name_comm : str
        Name of the commodity (e.g., 'crude steel') for the new market activity.

    results_df : pandas.DataFrame
        DataFrame containing information on existing activities, including columns:
        'initial name', 'name', 'reference product', 'unit', 'location', 
        'production volume', 'cbam_true', 'cbam_false', and 'year'.

    db_sel : str
        Name of the Brightway database containing the original activity datasets.

    db_name : str
        Name of the Brightway database used for input references in technosphere exchanges.

    db_name_markets : str
        Name of the new Brightway database to which the market activity will be added.

    market_name : str
        Descriptive name of the market being created (e.g., 'EU-27', 'Non-EU-27', 'Others').

    location_filter_fn : callable
        Function that accepts the `results_df` and returns a boolean Series 
        indicating which rows to include for this market.

    market_location_label : str
        Label to assign to the market activity's location field.

    w : brightway2.Wrapper
        Brightway2 Wrapper instance to retrieve activity codes using `.get_one()`.

    Returns:
    -------
    dict
        A dictionary representing the new market activity, including production volume, 
        production exchange, and weighted technosphere exchanges.
    """
    df_filtered = results_df[
        location_filter_fn(results_df) &
        (results_df['initial name'].str.contains('steel production', na=False)) &
        (results_df['year'] <= max_year)
    ].copy()

    df_filtered['production_volume_share'] = df_filtered['production volume'] / df_filtered['production volume'].sum()

    all_market_exchanges = []
    sum_volume = 0

    for loc in df_filtered.location.unique():
        df_loc = df_filtered[df_filtered.location == loc]
        if df_loc.empty:
            continue

        for _, row in df_loc.iterrows():
            act = w.get_one(db_sel,
                            w.equals("name", row['name']),
                            w.equals("reference product", row['reference product']),
                            w.equals("unit", row['unit']),
                            w.equals("location", row['location']))

            exc_market = {
                'name': row['name'],
                'type': 'technosphere',
                'location': row['location'],
                'amount': row['production_volume_share'],
                'reference product': row['reference product'],
                'unit': row['unit'],
                'production volume': row['production volume'],
                #'cbam_true': row['cbam_true'],
                #'cbam_false': row['cbam_false'],
                'input': (db_name, act['code'])
            }

            all_market_exchanges.append(exc_market)
            sum_volume += row['production volume']

    unique_code = str(uuid.uuid4().hex)

    # Add production exchange
    all_market_exchanges.append({
        'location': market_location_label,
        'database': db_name_markets,
        'name': f"market for {name_comm}",
        'reference product': name_comm,
        'unit': 'kilogram',
        'type': 'production',
        'amount': 1,
        'input': (db_name_markets, unique_code),
        'output': (db_name_markets, unique_code)
    })

    return {
        'location': market_location_label,
        'database': db_name_markets,
        'code': unique_code,
        'name': f"market for {name_comm}",
        'reference product': name_comm,
        'amount': 1,
        'unit': 'kilogram',
        'production volume': float(sum_volume),
        'exchanges': all_market_exchanges
    }

def create_steel_market_activities(results_df, db_sel, db_name,
                                   db_name_markets, name_comm='steel'):
    """
    Create new market activities for steel production based on production volume shares.

    Parameters:
    - results_df: pd.DataFrame containing production data with required columns
    - db_sel: Brightway2 database object to search original activities
    - db_name: name of the original database (e.g., 'ecoinvent 3.9')
    - name_comm: reference product name for the market activity (default: 'crude steel')
    - db_name_markets: name of the new market database to create
    - create_new_market_db: whether to delete and recreate the database

    Returns:
    - new_market_activities: list of new market activity dictionaries
    """
    new_market_activities = []

    for loc in results_df.location.unique():
        # Filter data for the location and only relevant activities before 2026
        results_df_loc = results_df[
            (results_df.location == loc) &
            (results_df['initial name'].str.contains('steel production', na=False)) & 
            (results_df['year'] < 2026)
        ].copy()

        # Skip if no activities found
        if results_df_loc.empty:
            continue

        # Calculate production volume shares
        results_df_loc['production_volume_share'] = (
            results_df_loc['production volume'] / results_df_loc['production volume'].sum()
        )

        unique_code = str(uuid.uuid4().hex)

        # Start with the production exchange for the market itself
        all_market_exchanges = [{
            'location': loc,
            'database': db_name_markets,
            'name': f"market for {name_comm}",
            'reference product': name_comm,
            'unit': 'kilogram',
            'type': 'production',
            'amount': 1,
            'input': (db_name_markets, unique_code),
            'output': (db_name_markets, unique_code)
        }]

        # Add technosphere exchanges based on production share
        for _, row in results_df_loc.iterrows():
            act = w.get_one(db_sel,
                            w.equals("name", row['name']),
                            w.equals("reference product", row['reference product']),
                            w.equals("unit", row['unit']),
                            w.equals("location", row['location']))

            exc_market = {
                'name': row['name'],
                'type': 'technosphere',
                'location': row['location'],
                'amount': row['production_volume_share'],
                'reference product': row['reference product'],
                'unit': row['unit'],
                'production volume': row['production volume'],
                #'cbam_true': row['cbam_true'],
                #'cbam_false': row['cbam_false'],
                'input': (db_name, act['code']),
            }

            all_market_exchanges.append(exc_market)

        new_act = {
            'location': loc,
            'database': db_name_markets,
            'code': unique_code,
            'name': f"market for {name_comm}",
            'reference product': name_comm,
            'amount': 1,
            'unit': 'kilogram',
            'production volume': float(results_df_loc['production volume'].sum()),
            'exchanges': all_market_exchanges
        }

        new_market_activities.append(new_act)

    return new_market_activities

def check_cbam_coverage(results_df, cbam_precursors_dict):
    #import pandas as pd
    #from collections import defaultdict

    report = []

    # Iterate through each plant (row)
    for idx, row in results_df.iterrows():
        plant_id = row.get("plant_id", idx)  # fallback if no plant_id
        exchanges = row['lca_impact_contri_climate change']

        # Track found precursors
        found_precursors = defaultdict(list)

        # Loop through all precursors
        for precursor_name, (include_keys, exclude_keys) in cbam_precursors_dict.items():
            for exchange in exchanges:
                exchange_name = exchange[0].lower()

                # Check if exchange matches include and not exclude
                included = any(key in exchange_name for key in include_keys)
                excluded = any(key in exchange_name for key in exclude_keys)

                if included and not excluded:
                    # Check if it is marked as CBAM-covered (4th tuple element)
                    is_cbam_flagged = exchange[3]
                    found_precursors[precursor_name].append((exchange_name, is_cbam_flagged))

        # Analyze results for this plant
        for precursor, matches in found_precursors.items():
            for ex_name, is_flagged in matches:
                if not is_flagged:
                    report.append({
                        "plant": plant_id,
                        "precursor": precursor,
                        "exchange": ex_name,
                        "issue": "Should be CBAM-covered but is not flagged"
                    })

    return pd.DataFrame(report)

def add_transport_exchanges(continent, db_sel, matched_database, km_transport_train=500):
    """
    Adds transport exchanges to a steel activity based on the continent of origin.

    Parameters:
        continent: str, continent name ('Asia', 'Africa', etc.)
        db_sel: Brightway2 database selection
        matched_database: Brightway2 database name
        km_transport_train: float, distance by train in km
    """

    # Static transport mode metadata (shared across all continents)
    transport_modes = {
        "train": {
            "name": "market group for transport, freight train",
            "reference product": "transport, freight train",
            "unit": "ton kilometer"
        },
        "ship": {
            "name": "market for transport, freight, sea, container ship",
            "reference product": "transport, freight, sea, container ship",
            "unit": "ton kilometer"
        }
    }

    # Only distances vary by continent
    transport_distances = {
        "Asia": {"train": km_transport_train, "ship": 10000},
        "Africa": {"train": km_transport_train, "ship": 7000},
        "Europe": {"train": km_transport_train},
        "Oceania": {"train": km_transport_train, "ship": 20000},
        "North America": {"train": km_transport_train, "ship": 5000},
        "South America": {"train": km_transport_train, "ship": 12000},
        "Unknown": {"train": km_transport_train, "ship": 5000},
    }

    if continent not in transport_distances:
        raise ValueError(f"Continent '{continent}' not found in transport distances.")

    exchanges_to_add = []

    for mode, km in transport_distances[continent].items():
        mode_data = transport_modes[mode]
        amount = km * 1e-3  # convert kg-km to ton-km 

        exc = {
            'amount': amount,
            'unit': mode_data["unit"],
            'type': 'technosphere',
            'name': mode_data["name"],
            'location': 'GLO'
        }

        code = w.get_one(
            db_sel,
            w.equals("name", mode_data["name"]),
            w.equals("reference product", mode_data["reference product"]),
            w.equals("unit", mode_data["unit"]),
            w.equals("location", "GLO")
        )['code']

        exc['input'] = (matched_database, code)
        exchanges_to_add.append(exc)

    return exchanges_to_add

def calculate_cbam_alpha(
    db_name,
    method,
    cbam_act_to_get_alpha_steel,
    cbam_act_to_get_alpha_iron,
    cbam_precursors_excl_steel=CBAM_RELEVANT_PRECURSORS_STEEL,
    cbam_precursors_excl_iron=CBAM_RELEVANT_PRECURSORS_IRON,
    threshold=90,
    cbam_threshold=0.15
):
    """
    Calculate alpha: the average share of CBAM-covered emissions for matched activities.
    Uses steel/iron-specific CBAM precursor lists depending on the matched activity type.
    """

    stored_data = []
    share_cbams = []

    # Build one combined list of rules but keep the material label
    rules = []
    for dct, material in [(cbam_act_to_get_alpha_steel, "steel"),
                          (cbam_act_to_get_alpha_iron, "iron")]:
        # Expect dict format: {key: (include_terms, exclude_terms)}
        for _, (include_terms, exclude_terms) in dct.items():
            rules.append((material, include_terms, exclude_terms))

    # 1) Identify relevant activities and label them as steel/iron
    matched_rows = []  # each entry: dict with act identifiers + material
    for act in bw.Database(db_name):
        activity_name = (act.get("name") or "").lower()
        ref_product = (act.get("reference product") or "").lower()

        for material, include_terms, exclude_terms in rules:
            inc_ok = (
                any(fuzz.token_set_ratio(activity_name, term.lower()) >= threshold for term in include_terms)
                and any(fuzz.token_set_ratio(ref_product, term.lower()) >= threshold for term in include_terms)
            )
            if not inc_ok:
                continue

            exc_ok = not any(
                fuzz.partial_ratio(activity_name, term.lower()) >= threshold for term in exclude_terms
            )
            if not exc_ok:
                continue

            matched_rows.append({
                "name": act["name"],
                "reference product": act["reference product"],
                "location": act["location"],
                "unit": act["unit"],
                "material": material,  # <-- KEY: steel vs iron
            })
            break  # avoid duplicate matches (first match wins)

    # 2) Analyze emissions for matched activities
    lca = None  # init once, reuse via redo
    for row in matched_rows:

        # pick the exact activity
        act_sel = next(
            act for act in bw.Database(db_name)
            if act["name"] == row["name"]
            and act["reference product"] == row["reference product"]
            and act["location"] == row["location"]
        )

        # Choose precursor list based on material label
        cbam_precursors_excl = (
            cbam_precursors_excl_iron if row["material"] == "iron" else cbam_precursors_excl_steel
        )

        # Init LCA once; then redo for each activity/exchange
        if lca is None:
            lca = bw.LCA({act_sel: 1}, method=method)
            lca.lci()
            lca.lcia()
        else:
            lca.redo_lcia({act_sel: 1})

        result_array = []
        cbam_amount = 0.0
        non_cbam_amount = 0.0

        for exc in act_sel.exchanges():
            # IMPORTANT: pass the *selected* precursor list for this activity
            scope_cbam = define_scope_cbam(exc.as_dict(), cbam_precursors_excl)

            if exc["type"] == "technosphere":
                lca.redo_lcia({exc.input: exc["amount"]})
                amount = float(lca.score)
            elif exc["type"] == "biosphere":
                # biosphere CF extraction (same approach as your original)
                cf = lca.characterization_matrix[lca.biosphere_dict[exc.input], :].sum()
                amount = float(cf * exc["amount"])
            else:
                continue

            result_array.append((exc["name"], amount, scope_cbam.get("scope"), scope_cbam.get("cbam")))

            if scope_cbam.get("cbam"):
                cbam_amount += amount
            else:
                non_cbam_amount += amount

        total = cbam_amount + non_cbam_amount
        if total == 0:
            print("Warning: Zero total emissions for", act_sel["name"])
            continue

        share_cbam = cbam_amount / total
        if share_cbam < cbam_threshold:
            print("Skipping low CBAM share:", share_cbam, act_sel["name"])
            continue

        share_cbams.append(min(float(share_cbam), 1.0))

        stored_data.append({
            "activity_name": act_sel["name"],
            "location": act_sel["location"],
            "reference_product": act_sel["reference product"],
            "unit": act_sel["unit"],
            "material": row["material"],  # <-- keep for traceability
            "cbam_share": float(share_cbam),
            "exchanges": result_array,
            "cbam_precursors_used": "iron" if row["material"] == "iron" else "steel",
        })

    alpha = float(np.mean(share_cbams)) if share_cbams else 0.0
    return alpha, stored_data