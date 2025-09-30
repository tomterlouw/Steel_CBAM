#Standard vars
EI_VERSION = "3.10"
DB_NAME_INIT = 'ecoinvent-{}-cutoff'.format(EI_VERSION)
PROJECT_NAME = "lca_steel_sector_cbam"
NAME_REF_DB = "ecoinvent_{}_reference".format(EI_VERSION).replace(".","")
USER_NAME = "terlouw_t"
BIOSPHERE_DB = 'ecoinvent-3.10-biosphere'
FUTURE_YEAR = 2040 #year chosen for future scenarios

# LCIA methods that we are interested in here.
MY_METHODS = [
                ('EF v3.1 EN15804', 'climate change', 'global warming potential (GWP100) including H2'),
            ]

#FILEPATH_WIND = r"data\tifs\gwa3_250_capacityfactor_IEC1.tif"
#FILEPATH_PV = r"data\tifs\PVOUT.tif"  # Assuming same filepath for both
FILE_NAME_COSTS = r"data\cost_data_init.xlsx"
FILE_DF_RATIOS = r"data\results_curve_fits_wind_hydrogen.xlsx"