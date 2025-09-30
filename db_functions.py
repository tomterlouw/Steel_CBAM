import brightway2 as bw
from config import *
import premise as pr
from private_keys import KEY_PREMISE
from functools import partial
from bw2io.importers.base_lci import LCIImporter
from bw2io.strategies import add_database_name, csv_restore_tuples

def drop_empty_categories(db):
    """Drop categories with the value ``('',)``"""
    DROP = ('',)
    for ds in db:
        if ds.get('categories') == DROP:
            del ds['categories']
        for exc in ds.get("exchanges", []):
            if exc.get('categories') == DROP:
                del exc['categories']
    return db

def strip_nonsense(db):
    for ds in db:
        for key, value in ds.items():
            if isinstance(value, str):
                ds[key] = value.strip()
            for exc in ds.get('exchanges', []):
                for key, value in exc.items():
                    if isinstance(value, str):
                        exc[key] = value.strip()
    return db

def create_database(db_name):
    """
    Creates and registers a new database in the Brightway2 environment.

    Parameters:
    -----------
    db_name : str
        The name of the database to be created.

    Returns:
    --------
    None
        The function does not return a value but will print a message indicating the result of the operation.

    Description:
    ------------
    This function checks if a database with the given name (`db_name`) already exists in the Brightway2 environment. 
    If the database already exists, the function aborts the operation and prints a message indicating that the database 
    already exists. If the database does not exist, the function proceeds to create and register a new database with 
    the specified name.

    Example:
    --------
    >>> create_database("new_db")
    Database 'new_db' has been created and registered.
    
    >>> create_database("new_db")
    Database 'new_db' already exists. Operation aborted.
    """
    # Check if the database already exists
    if db_name in bw.databases:
        print(f"Database '{db_name}' already exists. Operation aborted.")
        return
    
    # Create and register the new database
    db = bw.Database(db_name)
    db.register()
    print(f"Database '{db_name}' has been created and registered.")

# ### generate the database which we are going to use, as premise include many novel datasets. Add some datasets that we generated ourselves.
def generate_reference_database():
    """
    Generate a reference database based on specified parameters.

    This function generates a reference database based on the specified parameters.
    It deletes the existing reference database with the same name if it exists and then creates a new one.

    Returns:
    None

    Example:
    >>> generate_reference_database()
    """
    # Delete old reference database with the same name
    for db_name in list(bw.databases):
        if NAME_REF_DB in db_name:
            print(db_name)
            del bw.databases[db_name]

    # Create a new reference database using NewDatabase
    ndb = pr.NewDatabase(
        scenarios=[{"model": "remind", "pathway": 'SSP2-Base', "year": "2024"}],
        source_db=DB_NAME_INIT,
        source_version=EI_VERSION,
        key=KEY_PREMISE,
        biosphere_name= BIOSPHERE_DB,
        additional_inventories=[
            {"filepath": r"data\H2-DRI_LCI.xlsx", "ecoinvent version": "3.9.1"},
            {"filepath": r"data\BF-BOF-CCS_Carina.xlsx", "ecoinvent version": "3.9.1"},         
            ]
    )

    # Fix suggested by R.S. since premise expect modifications
    ndb.scenarios[0]["database"] = ndb.database
    ndb.write_db_to_brightway(name=NAME_REF_DB)

def generate_future_ei_dbs(scenarios = ["SSP2-Base", "SSP2-PkBudg1150","SSP2-PkBudg500"], iam = 'remind',
                           start_yr=2020, end_yr = 2050, step = 5, endstring="base"):
    """
    Generate Ecoinvent scenario models with specified parameters.

    This function generates Ecoinvent scenario models based on the specified year, scenario, and additional settings.
    It avoids adding duplicated databases by checking the existing databases in Brightway2.

    Parameters:
    - scenarios (list): The scenarios for which the models are generated. Default is: 
                ["SSP2-Base",
                "SSP2-PkBudg1150",
                "SSP2-PkBudg500"] corresponding to baseline, 2 degrees C, and 1.5 degrees C.
    - iam (str): IAM chosen, can be 'remind' or 'image'.
    - start_yr (int): The starting year for the scenarios.
    - end_yr (int): The end year for the scenarios.
    - step (int): step between scenario years.
    - endstring (str, optional): A suffix to differentiate the generated databases. Default is "base".

    Returns:
    tuple: A tuple containing two lists -
        1. List of dictionaries specifying the models for the scenarios.
        2. List of database names generated based on the specified parameters.

    Example:
    >>> generate_future_ei_dbs("SSP2-Base")
    ([{'model': 'remind', 'pathway': 'SSP2-Base', 'year': 2030},
      {'model': 'remind', 'pathway': 'SSP2-Base', 'year': 2050}],
     ['ecoinvent_remind_SSP2-Base_2030_custom', 'ecoinvent_remind_SSP2-Base_2050_base'])
    """

    list_years = [start_yr + i * step for i in range(1, int((end_yr - start_yr) / step) + 1)]

    list_spec_scenarios = []
    list_names = []

    for pt in scenarios:
        for yr in list_years:
            string_db = "ecoinvent_{}_{}_{}_{}".format(iam, pt, yr, endstring)

            if yr == start_yr and pt == "SSP2-Base":
                dict_spec = {"model": iam, "pathway": pt, "year": yr,
                                "exclude": ["update_electricity", "update_cement", "update_steel", "update_dac",
                                            "update_fuels", "update_emissions", "update_two_wheelers"
                                            "update_cars", "update_trucks", "update_buses"]}

                if string_db not in bw.databases:
                    list_spec_scenarios.append(dict_spec)
                    list_names.append(string_db)
                else:
                    print("Avoid duplicated db and therefore following db not added: '{}'".format(string_db))
            else:
                dict_spec = {"model": iam, "pathway": pt, "year": yr}

                if string_db not in bw.databases:
                    list_spec_scenarios.append(dict_spec)
                    list_names.append(string_db)
                else:
                    print("Avoid duplicated db and therefore following db not added: '{}'".format(string_db))

    return list_spec_scenarios, list_names

def generate_prospective_lca_dbs(list_spec_scenarios, list_names):
    """
    Generate and update future LCA databases for prospective LCA.

    This function generates and updates future LCA databases based on specified scenarios if needed, and writes them to Brightway2.

    Parameters:
    - list_spec_scenarios (list): List of dictionaries specifying scenarios for new databases.
    - list_names (list): List of names specifying scenario names for new databases.

    Returns:
    None

    Example:
    >>> generate_and_update_lca_databases([{"model": "remind", "pathway": 'SSP2-Base', "year": "2035"}], ['ecoinvent_remind_SSP2-Base_2035_base'])
    """

    if len(list_spec_scenarios) > 0:
        ndb = pr.NewDatabase(
            scenarios=list_spec_scenarios,
            source_db=DB_NAME_INIT,
            source_version=EI_VERSION,
            key=KEY_PREMISE,
            biosphere_name=BIOSPHERE_DB,
            additional_inventories=[
                {"filepath": r"data\H2-DRI_LCI.xlsx", "ecoinvent version": "3.9.1"},
                {"filepath": r"data\BF-BOF-CCS_Carina.xlsx", "ecoinvent version": "3.9.1"},                    
                ]
        )

        print("START UPDATING")
        ndb.update()

        print("START WRITING")
        ndb.write_db_to_brightway(name = list_names)

def delete_project(bw, project_name):
    """
    Deletes a specified project if it exists.

    Parameters:
    - bw: The object that manages the projects (brightway2).
    - project_name (str): The name of the project to be deleted.

    Returns:
    - str: A message indicating the result of the deletion attempt.
    """
    # List all existing projects
    print("Existing projects:", list(bw.projects))

    # Check if the project exists before attempting to delete it
    if project_name in bw.projects:
        # Set the project to be deleted as the current project
        bw.projects.set_current(project_name)
        
        # Delete the project
        bw.projects.delete_project(project_name)
        bw.projects.purge_deleted_directories()
        return f"Project '{project_name}' has been deleted."
    else:
        return f"Project '{project_name}' does not exist."

def drop_empty_categories_2(db):
    """Drop categories with the value ``('',)`` or None."""
    DROP = ('',)
    for ds in db:
        if ds.get('categories') in [DROP, None]:
            ds.pop('categories', None)  # Remove 'categories' if it exists
            #del ds['categories']
        for exc in ds.get("exchanges", []):
            if exc.get('categories') in [DROP, None]:
                #del exc['categories']
                exc.pop('categories', None)  # Remove 'categories' if it exists
    return db

def process_import(db_name, new_activities, iam='image'):
    """
    Process and write a database with predefined strategies, match databases, 
    and generate statistics.
    
    Parameters:
        db_name (str): The name of the database to write.
        new_activities (list): The dataset containing new activities.
    
    Returns:
        None
    """
    importer = LCIImporter(db_name)
    importer.data = new_activities
    importer.strategies = [
        partial(add_database_name, name=db_name),
        csv_restore_tuples,
        drop_empty_categories,
        drop_empty_categories_2,
        strip_nonsense,
    ]

    importer.apply_strategies()

    # Match databases
    for db in bw.databases:
        if f'ecoinvent_{iam}' in str(db) or BIOSPHERE_DB in str(db) or NAME_REF_DB in str(db):
            importer.match_database(db)

    # Generate statistics and export results
    importer.statistics()
    importer.write_excel(only_unlinked=True)
    importer.write_database()

def match_year_to_database(year, iam_model='image_SSP2-RCP26', ref_db = NAME_REF_DB):
    """
    Matches a given year to a database based on defined year ranges.
    
    Returns
    -------
    str
        The matched database name.
    """
    database_mapping = [
        (1500, 2027, ref_db),
        (2028, 2032, f"ecoinvent_{iam_model}_2030_base"),
        (2033, 2037, f"ecoinvent_{iam_model}_2035_base"),
        (2038, 2042, f"ecoinvent_{iam_model}_2040_base"),
        (2043, 2047, f"ecoinvent_{iam_model}_2045_base"),
        (2048, 2100, f"ecoinvent_{iam_model}_2050_base"),
    ]

    for start, end, db in database_mapping:
        if start <= year <= end:
            return db

    raise ValueError(f"No database match found for year {year}")

    