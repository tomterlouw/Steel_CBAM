#Standard vars
EI_VERSION = "3.12"
DB_NAME_INIT = 'ecoinvent-{}-cutoff'.format(EI_VERSION)
PROJECT_NAME = "steel_sector_imp"
NAME_REF_DB = "ecoinvent_{}_reference".format(EI_VERSION).replace(".","")
USER_NAME = "terlouw_t"
BIOSPHERE_DB = f'ecoinvent-{EI_VERSION}-biosphere'
FUTURE_YEAR = 2040 #year chosen for future scenarios

# LCIA methods that we are interested in here.
MY_METHODS = [
                ('ecoinvent-3.12', 'IPCC 2021 (incl. biogenic CO2)', 
                 'climate change: total (incl. biogenic CO2, incl. SLCFs)', 'global warming potential (GWP100)'),
            ]

NAME_CC_COL = f"lca_impact_{MY_METHODS[0][1]}" #name of the climate change method we are using, to be used in the contribution arrays and post-processing

FILE_NAME_COSTS = r"data\cost_data_init.xlsx"
FILE_DF_RATIOS = r"data\results_curve_fits_wind_hydrogen.xlsx"