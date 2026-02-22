from constructive_geometries import Geomatcher, resolved_row
import json
from copy import deepcopy
from numbers import Number
import math
import wurst as w
import uuid
import copy
import pandas as pd
import pycountry

from config import BIOSPHERE_DB

from mapping import dict_power_acts, dict_hydrogen_acts

# NOTE: many functions are slightly modified but mainly copied from the wurst package (Mutel et al.: https://github.com/polca/wurst/tree/main).
geomatcher = Geomatcher(backwards_compatible=True)

# add definitions from REMIND, if use another IAM for prospective LCA, then we need to change this.
with open(r"data\iam_variables_mapping\topologies\remind-topology.json", "r", encoding="utf-8") as f:
    REMIND_TOPOLOGY = json.load(f)
with open(r"data\iam_variables_mapping\topologies\image-topology.json", "r", encoding="utf-8") as f:
    IMAGE_TOPOLOGY = json.load(f)

DF_CVS = pd.read_csv("data\mean_cfs.csv", index_col="ISO_A3_EH")

def manually_add_definitions(geomatcher, data: dict, namespace: str = "", relative: bool = True):
    """
    Manually add topology definitions to a Geomatcher instance.

    This function replicates the logic of `geomatcher.add_definitions`, with one key difference:
    location keys are added without the IAM model namespace (i.e., just `k` instead of `(model, k)`).

    Parameters:
    - geomatcher (Geomatcher): The Geomatcher instance to modify.
    - data (dict): A dictionary of new topology definitions. If `relative` is True, 
      values should be lists of existing location names. If False, values should be sets of face IDs.
    - namespace (str, optional): Ignored in this function, included for compatibility. Defaults to "".
    - relative (bool, optional): If True, defines new locations by combining existing ones.
      If False, adds raw face ID mappings. Defaults to True.
    """
    if not relative:
        # Direct mapping of keys to face IDs
        geomatcher.topology.update({(namespace, k): v for k, v in data.items()})
        geomatcher.faces.update(set.union(*data.values()))
    else:
        # Definitions relative to existing entries in Geomatcher
        geomatcher.topology.update({
            k: set.union(*[geomatcher[o] for o in v])
            for k, v in data.items()
        })

print("Adding REMIND and IMAGE topology using regionalization.py")
manually_add_definitions(geomatcher, REMIND_TOPOLOGY, namespace="", relative=True)
manually_add_definitions(geomatcher, IMAGE_TOPOLOGY, namespace="", relative=True)

DF_CVS = pd.read_csv("data/mean_cfs.csv", index_col="ISO_A3_EH")

def iso2_to_iso3(iso2):
    country = pycountry.countries.get(alpha_2=iso2.upper())
    return country.alpha_3 if country else None

def round_to_nearest_0025(x):
    return round(x / 0.025) * 0.025

def get_mean_cf_pv_from_iso2(
    iso2,
    df=DF_CVS,
    fallback=0.175,
    minimum=0.1,
):
    """
    Return mean_cf_pv for a given ISO2 country code.
    - Falls back if ISO2/ISO3 not found or value is NaN
    - Rounds to nearest 0.025
    - Enforces minimum value
    """

    iso3 = iso2_to_iso3(iso2)

    if iso3 is None or iso3 not in df.index:
        value = fallback
    else:
        value = df.loc[iso3, "mean_cf_pv"]
        if pd.isna(value):
            value = fallback

    # round + enforce minimum
    value = round_to_nearest_0025(value)
    value = round(max(value, minimum), 3)

    return value

def match_best_location(location, possible_locations, exclusive=True, biggest_first=False, contained=False, fallback='GLO',
                     geomatcher=geomatcher):
    """
    Find the best-matching location from a list of possible locations using spatial rules.

    Parameters:
    - location (str or geometry): The target location to match, either as a string name or geometry object.
    - possible_locations (iterable): Collection of candidate locations to match against.
    - exclusive (bool, optional): If True, excludes overlapping matches (i.e. enforces mutually exclusive regions). Defaults to True.
    - biggest_first (bool, optional): If True, gives preference to larger matching regions. Defaults to False.
    - contained (bool, optional): If True, match only if the target location is fully contained within a candidate.
      If False, match based on any intersection. Defaults to False.
    - fallback (str, optional): Value to return if no match is found. Defaults to 'GLO'.
    - geomatcher (Geomatcher, optional): An instance of Geomatcher used for spatial comparisons.

    Returns:
    - list: A list of matching location(s), or a list containing the fallback if no match is found.
    """
    
    with resolved_row(possible_locations, geomatcher) as g:
        func = g.contained if contained else g.intersects
        locs = func(
            location,
            include_self=True,
            exclusive=exclusive,
            biggest_first=biggest_first,
            only=possible_locations)

        return locs if len(locs)>0 else [fallback] #Fallback is global dataset

# NOTE: This is copied from wurst package (Mutel et al.: https://github.com/polca/wurst/tree/main), full credits to that, however, we need to slightly change it to:
# ensure that RER datasets are also matched since some premise datasets are only in 'RER' location:
#if not kept and "RER" in possible_locations:
#    kept = [obj for obj in possible_datasets if obj["location"] == "RER"]
#    print(f"'RER' used for regionalization, exchange: {exc['name']}, {exc['location']}, dataset loc: {ds['location']}")

def relink_technosphere_exchanges(
    ds, data, exclusive=True, drop_invalid=False, biggest_first=False, contained=True
):
    """Find new technosphere providers based on the location of the dataset.

    Designed to be used when the dataset's location changes, or when new datasets are added.

    Uses the name, reference product, and unit of the exchange to filter possible inputs. These must match exactly. Searches in the list of datasets ``data``.

    Will only search for providers contained within the location of ``ds``, unless ``contained`` is set to ``False``, all providers whose location intersects the location of ``ds`` will be used.

    A ``RoW`` provider will be added if there is a single topological face in the location of ``ds`` which isn't covered by the location of any providing activity.

    If no providers can be found, `relink_technosphere_exchanes` will try to add a `RoW` or `GLO` providers, in that order, if available. If there are still no valid providers, a ``InvalidLink`` exception is raised, unless ``drop_invalid`` is ``True``, in which case the exchange will be deleted.

    Allocation between providers is done using ``allocate_inputs``; results seem strange if ``contained=False``, as production volumes for large regions would be used as allocation factors.

    Input arguments:

        * ``ds``: The dataset whose technosphere exchanges will be modified.
        * ``data``: The list of datasets to search for technosphere product providers.
        * ``exclusive``: Bool, default is ``True``. Don't allow overlapping locations in input providers.
        * ``drop_invalid``: Bool, default is ``False``. Delete exchanges for which no valid provider is available.
        * ``biggest_first``: Bool, default is ``False``. Determines search order when selecting provider locations. Only relevant is ``exclusive`` is ``True``.
        * ``contained``: Bool, default is ``True``. If ture, only use providers whose location is completely within the ``ds`` location; otherwise use all intersecting locations.

    Modifies the dataset in place; returns the modified dataset."""
    new_exchanges = []
    technosphere = lambda x: x["type"] == "technosphere"

    for exc in filter(technosphere, ds["exchanges"]):
        possible_datasets = list(get_possibles(exc, data))
        possible_locations = [obj["location"] for obj in possible_datasets]

        with resolved_row(possible_locations, geomatcher) as g:
            func = g.contained if contained else g.intersects
            gis_match = func(
                ds["location"],
                include_self=True,
                exclusive=exclusive,
                biggest_first=biggest_first,
                only=possible_locations,
            )

        kept = [
            ds for loc in gis_match for ds in possible_datasets if ds["location"] == loc
        ]

        if kept:
            missing_faces = geomatcher[ds["location"]].difference(
                set.union(*[geomatcher[obj["location"]] for obj in kept])
            )
            if missing_faces and "RoW" in possible_locations:
                kept.extend(
                    [obj for obj in possible_datasets if obj["location"] == "RoW"]
                )
        elif "RoW" in possible_locations:
            kept = [obj for obj in possible_datasets if obj["location"] == "RoW"]

        if not kept and "GLO" in possible_locations:
            kept = [obj for obj in possible_datasets if obj["location"] == "GLO"]

        ###
        if not kept and "RER" in possible_locations:
            kept = [obj for obj in possible_datasets if obj["location"] == "RER"]
        #    print(f"'RER' used for regionalization, exchange: {exc['name']}, {exc['location']}, dataset loc: {ds['location']}")
        #if not kept and "EUR" in possible_locations and 'RER' not in possible_locations:
        #    kept = [obj for obj in possible_datasets if obj["location"] == "EUR"]
        #    print(f"'EUR' used for regionalization, exchange: {exc['name']}, {exc['location']}, dataset loc: {ds['location']}")
        ###

        if not kept:
            if drop_invalid:
                continue
            else:
                raise ValueError("Invalidlink")

        allocated = allocate_inputs(exc, kept)

        new_exchanges.extend(allocated)

    ds["exchanges"] = [
        exc for exc in ds["exchanges"] if exc["type"] != "technosphere"
    ] + new_exchanges
    return ds

##NOTE: The following functions are copied from the Wurst package, full credits to Mutel et al.: https://github.com/polca/wurst/tree/main
def reference_product(ds):
    """Get single reference product exchange from a dataset.

    Raises ``wurst.errors.NoResults`` or ``wurst.errors.MultipleResults`` if zero or multiple results are returned."""
    excs = [
        exc for exc in ds["exchanges"] if exc["amount"] and exc["type"] == "production"
    ]
    if not excs:
        raise ValueError("No suitable production exchanges found")
    elif len(excs) > 1:
        raise ValueError("Multiple production exchanges found")
        
    return excs[0]

def allocate_inputs(exc, lst):
    """Allocate the input exchanges in ``lst`` to ``exc``, using production volumes where possible, and equal splitting otherwise.

    Always uses equal splitting if ``RoW`` is present."""
    has_row = any((x["location"] in ("RoW", "GLO") for x in lst))
    pvs = [reference_product(o).get("production volume") or 0 for o in lst]
    if all((x > 0 for x in pvs)) and not has_row:
        # Allocate using production volume
        total = sum(pvs)
    else:
        # Allocate evenly
        total = len(lst)
        pvs = [1 for _ in range(total)]

    def new_exchange(exc, location, factor):
        cp = deepcopy(exc)
        cp["location"] = location
        return rescale_exchange(cp, factor)

    return [
        new_exchange(exc, obj["location"], factor / total)
        for obj, factor in zip(lst, pvs)
    ]


def get_possibles(exchange, data):
    """FIlter a list of datasets ``data``, returning those with the save name, reference product, and unit as in ``exchange``.

    Returns a generator."""
    key = (exchange["name"], exchange["product"], exchange["unit"])
    for ds in data:
        if (ds["name"], ds["reference product"], ds["unit"]) == key:
            yield ds

def rescale_exchange(exc, value, remove_uncertainty=True):
    """Function to rescale exchange amount and uncertainty.

    * ``exc`` is an exchange dataset.
    * ``value`` is a number, to be multiplied by the existing amount.
    * ``remove_uncertainty``: Remove (unscaled) uncertainty data, default is ``True``.
    If ``False``, uncertainty data is scaled by the same factor as the amount
    (except for lognormal distributions, where the ``loc`` parameter is scaled by the log of the factor).
    Currently, does not rescale for Bernoulli, Discrete uniform, Weibull, Gamma, Beta, Generalized Extreme value
    and Student T distributions.

    Returns the modified exchange."""
    assert isinstance(exc, dict), "Must pass exchange dictionary"
    assert isinstance(value, Number), "Constant factor ``value`` must be a number"

    # Scale the amount
    exc["amount"] *= value

    # Scale the uncertainty fields if uncertainty is not being removed
    if not remove_uncertainty and "uncertainty type" in exc:
        uncertainty_type = exc["uncertainty type"]

        # No uncertainty, do nothing
        if uncertainty_type in {0, 6, 7, 8, 9, 10, 11, 12}:
            pass
        elif uncertainty_type in {1, 2, 3, 4, 5}:
            # Scale "loc" by the log of value for lognormal distribution
            if "loc" in exc and uncertainty_type == 2:
                exc["loc"] += math.log(value)
            elif "loc" in exc:
                exc["loc"] *= value

            # "scale" stays the same for lognormal
            # For other distributions, scale "scale" by the absolute value
            if "scale" in exc and uncertainty_type not in {2}:
                exc["scale"] *= abs(value)

            # Scale "minimum" and "maximum" by value
            for bound in ("minimum", "maximum"):
                if bound in exc:
                    exc[bound] *= value

    # If remove_uncertainty is True, then remove all uncertainty info
    elif remove_uncertainty:
        FIELDS = ("scale", "minimum", "maximum", )
        exc["uncertainty type"] = 0
        exc["loc"] = exc["amount"]
        for field in FIELDS:
            if field in exc:
                del exc[field]

    return exc

def create_regionalized_activity(
        db_sel,
        match_activity,
        iso2,
        row,
        end_use,
        db_name,
        plant_name,
        year
    ):
    """
    Create a regionalized version of a Brightway activity based on plant metadata and add it to a new database.

    Parameters
    ----------
    db_sel : str
        Name of the source database from which to copy the activity.
    match_activity : tuple
        Tuple of (activity name, reference product) to match the source activity.
    iso2 : str
        ISO 3166-1 alpha-2 country code of the target plant location.
    row : pd.Series
        Row from the input DataFrame containing plant information.
    end_use : str
        Column name in `row` representing the plant's annual production in Mt.
    db_name : str
        Name of the target database to which the new activity will belong.
    plant_name : str
        Descriptive name of the plant.
    year : int
        Year of plant commissioning or analysis.

    Returns
    -------
    tuple or None
        A tuple of (new_activity_dict, updated_all_names), or None if a duplicate is detected.
    """
    # Step 1: Find candidate activities
    activities_to_adapt_init = w.get_many(
        db_sel,
        w.equals("name", match_activity[0]),
        w.equals("reference product", match_activity[1]),
    )

    possible_locations = [act["location"] for act in activities_to_adapt_init]

    # Step 2: Determine best matching location
    best_matched_loc = (
        match_best_location(iso2, possible_locations)[0]
        if iso2 not in possible_locations
        else iso2
    )

    # Step 3: Get best activity match
    activity_to_adapt_init = w.get_one(
        db_sel,
        w.equals("name", match_activity[0]),
        w.equals("reference product", match_activity[1]),
        w.equals("location", best_matched_loc),
    )
    activity_to_adapt = activity_to_adapt_init.copy()

    # Step 4: Handle known proxies, otherwise this leads to errors due to now available locations
    if iso2 == "UG":
        iso2 = "KE"
        activity_to_adapt["location"] = "KE"
    elif iso2 == "US-PR":
        iso2 = "US"
        activity_to_adapt["location"] = "US"
    else:
        activity_to_adapt["location"] = iso2

    # Step 5: Regionalize technosphere exchanges
    activity_to_adapt = relink_technosphere_exchanges(activity_to_adapt, db_sel, contained=False)

    # Step 6: Create unique identifiers and metadata
    new_name = f"facility in {row['Country/Area']} with name: '{plant_name}' [{year}], having a production of {round(row[end_use],3)} Mt/a"
    new_code = str(uuid.uuid4().hex)

    new_act_dict = {
        "location": activity_to_adapt["location"],
        "database": db_name,
        "code": new_code,
        "name": f"{activity_to_adapt['name']} for {new_name}",
        "reference product": activity_to_adapt["reference product"],
        "unit": activity_to_adapt["unit"],
        "production volume": round(row[end_use], 3),
        "comment": (
            f"{activity_to_adapt.get('comment', '')}, location of plant, "
            f"latitude: {row.get('Latitude', 'N/A')}, longitude: {row.get('Longitude', 'N/A')}, "
            f"power source: {row.get('power_classification', 'N/A')}, "
            f"retired date: {row.get('Retired Date', 'N/A')},"
            f"steel or iron production: {row.get('steel_or_iron_production', 'N/A')}"
        ),
    }

    return new_act_dict, activity_to_adapt, new_code

def process_exchanges(activity_to_adapt, row, matched_database, db_name, iso2, db_sel, new_code, bio_db_w, plant_type,
                      dict_power_acts=dict_power_acts, dict_hydrogen_acts=dict_hydrogen_acts):
    """
    Transform and regionalize exchanges from a base activity.

    This function processes exchanges in a given activity by modifying their location, linking them to external 
    datasets, and adjusting inputs, amounts, and metadata as needed. It handles different types of exchanges 
    (production, technosphere, and biosphere) and applies logic for electricity, hydrogen, and iron-related 
    inputs based on plant characteristics and power source.

    Main operations include:
        - Rebuilding production exchanges with updated references.
        - Regionalizing electricity exchanges (market or onsite generation).
        - Replacing hydrogen inputs with region-specific sources.
        - Updating iron production exchanges and substituting hydrogen flows.
        - Annotating exchanges with CBAM-relevant scope metadata.

    Parameters:
        activity_to_adapt (dict): Activity containing the original exchanges to be adapted.
        row (pandas.Series): Row with plant-specific data (e.g. power classification, location).
        matched_database (str): Name of the external database to link inputs to.
        db_name (str): Name of the current project database (used for production exchanges).
        iso2 (str): ISO 3166-1 alpha-2 country code used for matching regional datasets.
        db_sel (object): Selector or database interface for querying external activities.
        new_code (str): New activity code used for linking production and output exchanges.
        dict_power_acts (dict): Mapping of power classification to generator activity name and product.
        dict_hydrogen_acts (dict): Mapping of power classification to hydrogen production activity name and product.

    Returns:
        list: A list of processed and merged exchange dictionaries.

    Raises:
        ValueError: If an unknown exchange type is encountered.
    """
    new_exchanges = []
    elect_type_searched = 0
    
    for exc in activity_to_adapt['exchanges']:     
        #exc['input'] = None  # Reset before assigning
        if 'production' == exc['type']:
            new_exchanges.append( 
                {'location': exc['location'],
                 'database': db_name,
                 'name': exc['name'],
                 'reference product': exc['product'],
                 'unit': exc['unit'],
                 'type': 'production',
                 'amount': exc['amount'],
                 'input': (db_name, new_code),
                 'output': (db_name, new_code)  })
            
        elif 'technosphere' == exc['type']:
            #1 Power change
            if 'for electricity' in exc['name'] and 'electricity' in exc['product'] and exc['unit'] == 'kilowatt hour' and row['power_classification']!='grid':
                if elect_type_searched == 0:
                    # this is to avoid searching again for the same new electricity exchange if there are multiple power inputs 
                    power_generator = dict_power_acts[row['power_classification']]
                    power_generator_acts = w.get_many(db_sel,
                                                      w.equals("name", power_generator[0]),
                                                      w.equals("reference product", power_generator[1]) )
                    
                    possible_locations = [act['location'] for act in power_generator_acts]
            
                    best_matched_loc = match_best_location(iso2, possible_locations)[0] if iso2 not in possible_locations else iso2
                    #print(db_sel,power_generator[0],power_generator[1],best_matched_loc)

                    # Needed to link to dataset from external datab)ase
                    act_elect = w.get_one(db_sel,
                                    w.equals("name", power_generator[0]),
                                    w.equals("reference product", power_generator[1]),
                                    w.equals("unit", 'kilowatt hour'),
                                    w.equals("location", best_matched_loc) ).copy()

                    elect_type_searched = 1

                exc=exc.copy()
                
                # Note: we will have to provide the input as we link to external database
                exc_new = {'location': act_elect['location'],
                                     'name': act_elect['name'],
                                     'reference product': act_elect['reference product'],
                                     'unit': act_elect['unit'],
                                     'type': 'technosphere',
                                     #'database': db_name,
                                     'amount':exc['amount'],#IMPORTANT: take the initital amount in the exchange
                                     'input': (matched_database, act_elect['code'])  
                          }
                new_exchanges.append(exc_new)
                
            #2 Power change GRID
            elif 'market group for electricity, ' in exc['name'] and iso2 not in ["BR", "IN", "US", "CN", "CA", "UG"] and exc['unit']=='kilowatt hour' and row['power_classification']=='grid':
                # Further regionalize power exchanges
                exc = exc.copy()
                init_name = exc['name']
                voltage_level = exc['name'].split("market group for electricity, ")[-1]
                exc['name'] = f"market for electricity, {voltage_level}"
                #print(exc['name'], exc['location'])
                # Needed to link to dataset from external database
                try:
                    act = w.get_one(db_sel,
                                    w.equals("name", exc['name']),
                                    w.equals("reference product", exc['product']),
                                    w.equals("unit", exc['unit']),
                                    w.equals("location", iso2) ).copy()
                    exc['location'] = iso2
                except:
                    # If an error, then we will have to search for best matching one
                    activities_to_adapt_init = w.get_many(
                                                db_sel,
                                                w.equals("name", exc['name']),
                                                w.equals("reference product", exc['product']),
                                            )
            
                    possible_locations = [act['location'] for act in activities_to_adapt_init]
                    best_matched_loc = match_best_location(iso2, possible_locations)[0] if iso2 not in possible_locations else iso2
                    
                    if best_matched_loc == 'GLO':
                        print(f"Global dataset used for combi: {exc['name'],exc['product'],iso2,best_matched_loc, possible_locations}")
                        exc['name'] = init_name
                
                    act = w.get_one(db_sel, w.equals("name", exc['name']),\
                                    w.equals("reference product", exc['product']),
                                    w.equals("unit", exc['unit']),
                                    w.equals("location", best_matched_loc) )

                    exc['location'] = best_matched_loc
                
                # Note: we will have to provide the input as we link to external database
                exc['input'] = (matched_database, act['code'])          
                new_exchanges.append(exc)

            # We should also modify/replace the hydrogen source for DRI steel/iron, if needed:
            elif ('hydrogen, gaseous' in exc['product']):
                if row['power_classification']!='grid':
                    if row['power_classification'] == 'renewable' or row['power_classification'] == 'pv':
                        # now we have to find the cf of the country
                        cf_rounded = get_mean_cf_pv_from_iso2(iso2)
                        hydrogen_source = (f'hydrogen production, gaseous, 30 bar, from PEM electrolysis, solar PV ground-mounted, global cf [{cf_rounded}]', 
                                           'hydrogen, gaseous, 30 bar')
                    else:
                        hydrogen_source = dict_hydrogen_acts[row['power_classification']]
                else:
                    hydrogen_source = ('hydrogen production, gaseous, 30 bar, from PEM electrolysis, power from market for electricity, medium voltage', 
                                       'hydrogen, gaseous, 30 bar')

                hydrogen_generator_acts = w.get_many(db_sel,
                                                  w.equals("name", hydrogen_source[0]),
                                                  w.equals("reference product", hydrogen_source[1]) )
                
                possible_locations = [act['location'] for act in hydrogen_generator_acts]
        
                best_matched_loc = match_best_location(iso2, possible_locations)[0] if iso2 not in possible_locations else iso2
                #print(hydrogen_source[0],hydrogen_source[1],best_matched_loc)

                # Needed to link to dataset from external database
                act_h2 = w.get_one(db_sel,
                                w.equals("name", hydrogen_source[0]),
                                w.equals("reference product", hydrogen_source[1]),
                                w.equals("unit", 'kilogram'),
                                w.equals("location", best_matched_loc) )

                exc_new = {'location': act_h2['location'],
                                     'name': act_h2['name'],
                                     'reference product': act_h2['reference product'],
                                     'unit': act_h2['unit'],
                                     'type': 'technosphere',
                                     #'database': db_name,
                                     'amount':exc['amount'],#IMPORTANT: take the initital amount in the exchange
                                     'input': (matched_database, act_h2['code'])}
                new_exchanges.append(exc_new)

            # We should also modify/replace the hydrogen source (h2-dri) and regionalize iron production IF produced locally at the plant
            elif (plant_type != 'steel') and ('technosphere' == exc['type'] and "iron" in str(exc['product']) and 
                                              ("iron production" in str(exc['name']) or 'market for pig iron' in str(exc['name']))
                                              and not any(word in exc['name'] for word in ['treatment', 'sludge', 'ash', 'residues', 'waste'])):
                exc = exc.copy()

                if 'market for pig iron' == str(exc['name']):
                    exc['name'] = 'pig iron production' # we do this so that we can better regionalize the iron production exchange, but we will keep the original name in the comment for clarity

                init_amount = exc['amount']
            
                act_iron_init = w.get_one(db_sel,
                                          w.equals("name", exc['name']),
                                          w.equals("reference product", exc['product']),
                                          w.equals("unit", exc['unit']),
                                          w.equals("location", exc['location']) ) 

                act_iron = act_iron_init.copy()
                act_iron['location'] = iso2
                act_iron = relink_technosphere_exchanges(act_iron, db_sel, contained=False) 

                for exc_2 in act_iron['exchanges']:
                    exc_2_copy = exc_2.copy()
                    if 'hydrogen, gaseous' in str(exc_2_copy['product']):
                        if row['power_classification']!='grid':
                            if row['power_classification'] == 'renewable' or row['power_classification'] == 'pv':
                                # now we have to find the cf of the country
                                cf_rounded = get_mean_cf_pv_from_iso2(iso2)
                                hydrogen_source = (f'hydrogen production, gaseous, 30 bar, from PEM electrolysis, solar PV ground-mounted, global cf [{cf_rounded}]', 
                                                'hydrogen, gaseous, 30 bar')
                            else:
                                hydrogen_source = dict_hydrogen_acts[row['power_classification']]
                        else:
                            hydrogen_source = ('hydrogen production, gaseous, 30 bar, from PEM electrolysis, power from market for electricity, medium voltage', 
                                            'hydrogen, gaseous, 30 bar')
                        
                        hydrogen_generator_acts = w.get_many(db_sel, 
                                                          w.equals("name", hydrogen_source[0]),
                                                          w.equals("reference product", hydrogen_source[1]) )
                        
                        possible_locations = [act['location'] for act in hydrogen_generator_acts]
                
                        best_matched_loc = match_best_location(iso2, possible_locations)[0] if iso2 not in possible_locations else iso2
                        #print(hydrogen_source[0],hydrogen_source[1],best_matched_loc)

                        # Needed to link to dataset from external database
                        act_h2 = w.get_one(db_sel,
                                        w.equals("name", hydrogen_source[0]),
                                        w.equals("reference product", hydrogen_source[1]),
                                        w.equals("unit", 'kilogram'),
                                        w.equals("location", best_matched_loc) )

                        exc_new = {'location': act_h2['location'],
                                             'name': act_h2['name'],
                                             'reference product': act_h2['reference product'],
                                             'unit': act_h2['unit'],
                                             'type': 'technosphere',
                                             'amount':exc_2_copy['amount']* init_amount,#IMPORTANT: take the initital amount in the exchange
                                             'input': (matched_database, act_h2['code'])}
                        new_exchanges.append(exc_new)

                    elif 'for electricity' in exc_2_copy['name'] and 'electricity' in exc_2_copy['product'] and exc_2_copy['unit'] == 'kilowatt hour' and row['power_classification']!='grid':
                        if elect_type_searched == 0:
                            # this is to avoid searching again for the same new electricity exchange if there are multiple power inputs 
                            power_generator = dict_power_acts[row['power_classification']]
                            power_generator_acts = w.get_many(db_sel,
                                                            w.equals("name", power_generator[0]),
                                                            w.equals("reference product", power_generator[1]) )
                            
                            possible_locations = [act['location'] for act in power_generator_acts]
                    
                            best_matched_loc = match_best_location(iso2, possible_locations)[0] if iso2 not in possible_locations else iso2
                            #print(db_sel,power_generator[0],power_generator[1],best_matched_loc)

                            # Needed to link to dataset from external datab)ase
                            act_elect = w.get_one(db_sel,
                                            w.equals("name", power_generator[0]),
                                            w.equals("reference product", power_generator[1]),
                                            w.equals("unit", 'kilowatt hour'),
                                            w.equals("location", best_matched_loc) ).copy()

                            elect_type_searched = 1

                        exc_3=exc_2_copy.copy()
                        
                        # Note: we will have to provide the input as we link to external database
                        exc_new = {'location': act_elect['location'],
                                            'name': act_elect['name'],
                                            'reference product': act_elect['reference product'],
                                            'unit': act_elect['unit'],
                                            'type': 'technosphere',
                                            #'database': db_name,
                                            'amount':exc_3['amount']* init_amount,#IMPORTANT: take the initital amount in the exchange
                                            'input': (matched_database, act_elect['code'])  
                                }
                        new_exchanges.append(exc_new)

                    else:
                        # Needed to link to dataset from external database
                        if 'technosphere' == exc_2_copy['type']:
                            #print(exc_2_copy['name'],  init_amount_2, init_amount)
                            act = w.get_one(db_sel,
                                                        w.equals("name", exc_2_copy['name']),
                                                        w.equals("reference product", exc_2_copy['product']),
                                                        w.equals("unit", exc_2_copy['unit']),
                                                        w.equals("location", exc_2_copy['location']) ).copy()
                            
                            exc_2_copy['input'] = (matched_database, act['code'])
                            # Note: we will have to provide the input as we link to external database
                            exc_2_copy['amount'] = exc_2_copy['amount'] * init_amount
                            new_exchanges.append(exc_2_copy)
                        if 'biosphere' == exc_2_copy['type']:
                            act = w.get_one(bio_db_w,
                                                        w.equals("name", exc_2_copy['name']),
                                                        w.equals("unit", exc_2_copy['unit']),
                                                        w.equals("categories", exc_2_copy['categories']) ).copy()
                            exc_2_copy['input'] = (BIOSPHERE_DB, act['code'])
                            # Note: we will have to provide the input as we link to external database
                            exc_2_copy['amount'] = exc_2_copy['amount'] * init_amount
                            new_exchanges.append(exc_2_copy)
          
            else:
                # Needed to link to dataset from external database
                act = w.get_one(db_sel,
                                w.equals("name", exc['name']),
                                w.equals("reference product", exc['product']),
                                w.equals("unit", exc['unit']),
                                w.equals("location", exc['location']) ).copy()
                
                # Note: we will have to provide the input as we link to external database
                exc['input'] = (matched_database, act['code'])             
                new_exchanges.append(exc)
                
        elif 'biosphere' == exc['type']:
            # Needed to link to dataset from external database
            bio_flow = w.get_one(bio_db_w,
                                 w.equals("name", exc['name']),
                                 w.equals("unit", exc['unit']),
                                 w.equals("categories", exc['categories']) ).copy()

            # Note: we will have to provide the input as we link to external database
            exc['input'] = (BIOSPHERE_DB, bio_flow['code'])
            new_exchanges.append(exc)

        else:
            raise ValueError(f"ERROR: not known exchange type {exc['type']}")
            
    return merge_duplicate_exchanges(new_exchanges)

def merge_duplicate_exchanges(exchanges):
    """
    Aggregates exchanges by summing the 'amount' for entries with the same 'input' key.

    This function identifies duplicate exchanges based on the 'input' field and merges them 
    by summing their 'amount' values. Only the first occurrence of each unique 'input' key 
    is retained in the output, with its 'amount' updated to reflect the total sum.

    Parameters:
        exchanges (list of dict): List of exchange dictionaries.

    Returns:
        list of dict: List of exchanges with duplicates merged based on 'input'.
    """
    seen = {}
    merged_exchanges = []

    for ex in exchanges:
        key = ex['input']
        if key in seen:
            seen[key]['amount'] += ex['amount']  # Sum amounts for duplicates
        else:
            new_entry = copy.deepcopy(ex)  # Create a separate copy
            seen[key] = new_entry
            merged_exchanges.append(new_entry)

    # Remove entries where amount is zero
    merged_exchanges = [ex for ex in merged_exchanges if ex['amount'] != 0]

    return merged_exchanges
 