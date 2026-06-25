end_product_colors = {
    'secondary steel (EAF)': 'darkgreen',
    #'iron production (H$_2$-DRI)': 'steelblue',
    'primary steel (DRI-EAF, coal)': 'indianred',
    'primary steel (DRI-EAF, NG)': 'firebrick',
    'primary steel (BF-BOF)': 'black',
    'primary steel (DRI-EAF, H$_2$)': 'royalblue',
    'primary steel (BF-BOF-CCS)': 'gray',
    'primary steel (EW-EAF)': "#04B7F8"
    #'iron production': 'saddlebrown'
}

dict_color_fig_2_c = {'North America':"red", 
                'Africa':"darkgreen", 
                'Asia':"darkblue", 
                'Oceania':"yellow", 
                'Europe':'green', 
                'South America':'turquoise', 
               }

dict_color_fig_2_c = {
    'North America': "#D55E00",      # warm burnt orange
    'Africa': "#009E73",            # soft teal green
    'Asia': "#0072B2",              # calm deep blue
    'Oceania': "#F0E442",           # muted yellow
    'Europe': "turquoise",            # sky blue
    'South America': "darkred",     # muted pink-purple
}

# List of European countries
eu27_countries = [
    'AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR', 'DE', 'GR', 'HU',  
    'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL', 'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE'
]

european_countries = [
    'AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR', 'DE', 'GR', 'HU',
    'IS', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL', 'PL', 'PT', 'RO', 'SK', 'SI',
    'ES', 'SE', 'AL', 'AM', 'AD', 'AZ', 'BY', 'BA', 'GE', 'MD', 'ME', 'MK', 'RS',
    'RU', 'TR', 'UA', 'XK', 'SM', 'VA', 'LI', 'MC', 'CH', 'NO'
]

##ACTUAL relevant precursors based on the document of the EU:
#https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32023R1773#page=9.57
#https://taxation-customs.ec.europa.eu/system/files/2023-11/CBAM%20Guidance_EU%20231121%20for%20web_0.pdf#page=26.06
"""
Relevant precursors, embedded emissions of those should be obtained from the buyer and reported
— crude steel, if used in the process;
— pig iron, DRI, if used in the process;
— FeMn, FeCr, FeNi, if used in the process;
— iron or steel products, if used in the process;
"""

# p. 89:
CBAM_RELEVANT_PRECURSORS_STEEL = {
    # Relevant precursors per Annex II for crude steel include:
    # pig iron or DRI (if used); FeMn/FeCr/FeNi (if used);
    # iron or steel products (if used).
    # Hydrogen is not a standalone precursor for steel under CBAM. Iron scrap is not a relevant precursor for steel under CBAM, as it is considered a waste/recycling flow rather than a primary input.
    "pig iron": (["pig iron", "BF+CCS, iron production", #'market for iron scrap, sorted, pressed', 'market for iron scrap, unsorted',
                  "iron production"],#, 'iron scrap', 'iron scrap, sorted', 'iron scrap, unsorted'], 
                 ["CCS via VPSA for DRI", "iron scrap","waste", "treatment","recycling", "recycled", "ash", "sludge", 'residues', 'slag', "wastewater", 'silicon production, photovoltaics']),
    "ferro-chromium": (["ferro-chromium", "ferro chromium", "ferrochromium", "ferrochromium production", 'market for ferrochromium, high-carbon'], 
                       ["waste", "treatment","recycling","scrap", "recycled", "ash", "sludge", 'residues', 'slag', "wastewater", 'silicon production, photovoltaics']),
    "ferro-manganese": (['market for ferromanganese, high-coal, 74.5% Mn', "ferro-manganese", "ferro manganese", "ferromanganese", "ferromanganese production"], 
                        ["waste", "treatment","recycling","scrap", "recycled", "ash", "sludge", 'residues', 'slag', "wastewater", 'silicon production, photovoltaics']),
    "ferro-nickel": (["ferro-nickel", "ferro nickel", "ferronickel", "ferronickel production"], 
                     ["waste", "treatment","recycling","scrap", "recycled", "ash", "sludge", 'residues', 'slag', "wastewater", 'silicon production, photovoltaics']),
    "direct reduced iron": (["direct reduced iron", "dri", "reduced iron", "sponge iron production", "market for sponge iron", 'hot briquetted iron'], 
                            ["waste", "treatment","recycling","scrap", "recycled", "ash", "sludge", 'residues', 'slag', 'silicon production, photovoltaics']),
    "steel": (["crude steel", "steel production", "low-alloyed steel", "unalloyed steel", "steel manufacturing"], 
              ["waste", "treatment","recycling","scrap", "recycled", "ash", "sludge", 'residues', 'slag', "wastewater", 'metal working', 'silicon production, photovoltaics']),
}

# Relevant precursors per Annex II for Pig iron and DRI include:
# sintered ore; pig iron or DRI (if used); FeMn/FeCr/FeNi (if used); hydrogen (if used).
# :contentReference[oaicite:2]{index=2}
CBAM_RELEVANT_PRECURSORS_IRON = CBAM_RELEVANT_PRECURSORS_STEEL.copy()
# Hydrogen is explicitly a relevant precursor for Pig iron and DRI routes (if used in the process)
CBAM_RELEVANT_PRECURSORS_IRON["hydrogen"] = (
    ["hydrogen production", "hydrogen, gaseous", "hydrogen gaseous", "pem electrolysis",
     "steam methane reforming", "smr", "coal gasification"],
    ["waste", "treatment", "recycling", "scrap", "recycled", "ash", "sludge"]
)
CBAM_RELEVANT_PRECURSORS_IRON["sintered ore"] = (
    ["sintered ore", "iron sinter", 'iron pellet production', "pellet", "pelletising", "pelletizing", 'iron pellet', 'market for iron ore pellet',
        # upstream steps allowed ONLY as part of sinter/pellet chains
        "iron ore concentrate", #https://taxation-customs.ec.europa.eu/system/files/2023-11/CBAM%20Frequently%20Asked%20Questions_November%202023.pdf#page=27.08
        "iron ore beneficiation",
    ], 
    ["waste", "treatment","recycling","scrap", "recycled", "ash", "sludge", 'residues', 'slag', "wastewater", 'silicon production, photovoltaics'])

# those scope 2 exchanges are mainly taken from Minten et al.
SCOPE_2_EXCHANGES =['electricity, medium voltage',
                    'electricity, low voltage',
                    'electricity production,',
                  'diesel, burned in diesel-electric generating set, 10MW, for oil and gas extraction',
                  'heavy fuel oil, burned in refinery furnace',
                  'sweet gas, burned in gas turbine',
                  'heat, district or industrial, other than natural gas',
                  'heat, district or industrial, natural gas',
                  'natural gas, burned in gas turbine',
                  'diesel, burned in building machine',
                  'electricity, high voltage',
                  'electricity, high voltage, for internal use in coal mining',
                  'heat, from steam, in chemical industry',
                 ]

WASTE_EXCHANGES=['hazardous waste, for underground deposit', 
                 'municipal solid waste',
                 'waste natural gas, sweet',
                 'water discharge from petroleum extraction, offshore',
                 'spent catalytic converter NOx reduction',
                 'waste refinery gas',
                 'water discharge from petroleum/natural gas extraction, onshore',
                 'spoil from hard coal mining',
                 'hazardous waste, for incineration',
                 'waste gypsum',
                 'inert waste, for final disposal',
                 'wastewater, average',
                 'spoil from lignite mining',
                 'blast furnace slag, Recycled Content cut-off'
                ]

dict_types = {
            "steel production, electric, low-alloyed, secondary steel, using scrap iron" : "secondary steel (EAF)",
            "steel production, electric, unalloyed, secondary steel, using scrap iron" : "secondary steel (EAF)",

            "pig iron production, hydrogen-based direct reduction iron": "iron production (H$_2$-DRI)",

            "steel production, natural gas-based direct reduction iron-electric arc furnace, low-alloyed": "primary steel (DRI-EAF, NG)",
            "steel production, natural gas-based direct reduction iron-electric arc furnace, unalloyed": "primary steel (DRI-EAF, NG)",

            "steel production, electric, low-alloyed, primary steel, using direct reduced iron": "primary steel (DRI-EAF, coal)",
            "steel production, electric, unalloyed, primary steel, using direct reduced iron": "primary steel (DRI-EAF, coal)",

            "steel production, blast furnace-basic oxygen furnace, low-alloyed": "primary steel (BF-BOF)",
            "steel production, blast furnace-basic oxygen furnace, unalloyed": "primary steel (BF-BOF)",

            "steel production, hydrogen-based direct reduction iron-electric arc furnace, low-alloyed": "primary steel (DRI-EAF, H$_2$)",
            "steel production, hydrogen-based direct reduction iron-electric arc furnace, unalloyed": "primary steel (DRI-EAF, H$_2$)",

            "steel production, blast furnace-basic oxygen furnace, with carbon capture and storage, low-alloyed": "primary steel (BF-BOF-CCS)",
            "steel production, blast furnace-basic oxygen furnace, with carbon capture and storage, unalloyed": "primary steel (BF-BOF-CCS)",

            "pig iron production": "iron production",

            "steel production, electrowinning-electric arc furnace, low-alloyed": "primary steel (EW-EAF)",
            "steel production, electrowinning-electric arc furnace, unalloyed": "primary steel (EW-EAF)",
        }

dict_power_acts = {'natural gas': ('electricity production, natural gas, combined cycle power plant', 'electricity, high voltage'),
                   'pv': ('electricity production, photovoltaic, 570kWp open ground installation, multi-Si', 'electricity, low voltage'), 
                   'wind': ('electricity production, wind, 1-3MW turbine, onshore', 'electricity, high voltage'), 
                   'nuclear': ('electricity production, nuclear, pressure water reactor', 'electricity, high voltage'),
                   'coal': ('electricity production, hard coal', 'electricity, high voltage'),
                   # renewable is assumed to be large-scale solar PV as it the fastest growing renewable power source
                   'renewable': ('electricity production, photovoltaic, 570kWp open ground installation, multi-Si', 'electricity, low voltage'),
                  }

dict_hydrogen_acts = {'natural gas': ('hydrogen production, steam methane reforming', 'hydrogen, gaseous, low pressure'),
                   'pv': ('hydrogen production, gaseous, 30 bar, from PEM electrolysis, solar PV ground-mounted, global cf [0.175]', 'hydrogen, gaseous, 30 bar'),
                   'wind': ('hydrogen production, gaseous, 30 bar, from PEM electrolysis, onshore wind, global cf [0.35]', 'hydrogen, gaseous, 30 bar'), 
                   'nuclear': ('hydrogen production, gaseous, 30 bar, from PEM electrolysis, power from electricity production, nuclear, pressure water reactor', 'hydrogen, gaseous, 30 bar'),
                   'coal': ('hydrogen production, coal gasification', 'hydrogen, gaseous, low pressure'),
                   # renewable is assumed to be large-scale solar PV as it the fastest growing renewable power source
                   'renewable': ('hydrogen production, gaseous, 30 bar, from PEM electrolysis, solar PV ground-mounted, global cf [0.175]', 'hydrogen, gaseous, 30 bar'),
                  }

dict_acts = {
    "unalloyed": {
        # Steel production
        "EAF steel production": (
            "steel production, electric, low-alloyed, secondary steel, using scrap iron" ,
            "steel, low-alloyed",
            "RoW",
        ),
        "EAF steel production (DRI, coal)": (
            "steel production, electric, low-alloyed, primary steel, using direct reduced iron",
            "steel, low-alloyed",
            "RoW",
        ),
        "EAF steel production (DRI, NG)": (
            "steel production, natural gas-based direct reduction iron-electric arc furnace, unalloyed",
            "steel, unalloyed",
            "GLO",
        ),
        "OHF steel production": (
            "steel production, blast furnace-basic oxygen furnace, unalloyed",
            "steel, unalloyed",
            "GLO",
        ),
        "BOF steel production": (
            "steel production, blast furnace-basic oxygen furnace, unalloyed",
            "steel, unalloyed",
            "GLO",
        ),
        "steel-ccs": (
            "steel production, blast furnace-basic oxygen furnace, with carbon capture and storage, unalloyed",
            "steel, unalloyed",
            "GLO",
        ),
        "both-ccs": (
            "steel production, blast furnace-basic oxygen furnace, with carbon capture and storage, unalloyed",
            "steel, unalloyed",
            "GLO",
        ),
        "steel-dri-hydrogen": (
            "steel production, hydrogen-based direct reduction iron-electric arc furnace, unalloyed",
            "steel, unalloyed",
            "GLO",
        ),
        "both-dri-hydrogen": (
            "steel production, hydrogen-based direct reduction iron-electric arc furnace, unalloyed",
            "steel, unalloyed",
            "GLO",
        ),
        "steel-tgr": (
            "steel production, blast furnace-basic oxygen furnace, with top gas recycling, unalloyed",
            "steel, unalloyed",
            "GLO",
        ),
        "both-tgr": (
            "steel production, blast furnace-basic oxygen furnace, with top gas recycling, unalloyed",
            "steel, unalloyed",
            "GLO",
        ),
        "steel-tgr-ccs": (
            "steel production, blast furnace-basic oxygen furnace, with top gas recycling, with carbon capture and storage, unalloyed",
            "steel, unalloyed",
            "GLO",
        ),
        "both-tgr-ccs": (
            "steel production, blast furnace-basic oxygen furnace, with top gas recycling, with carbon capture and storage, unalloyed",
            "steel, unalloyed",
            "GLO",
        ),
        "steel-electrowinning": (
            "steel production, electrowinning-electric arc furnace, unalloyed",
            "steel, unalloyed",
            "GLO",
        ),
        "both-electrowinning": (
            "steel production, electrowinning-electric arc furnace, unalloyed",
            "steel, unalloyed",
            "GLO",
        ),
        "steel-dri-ng-ccs": (
            "steel production, natural gas-based direct reduction iron-electric arc furnace, with carbon capture and storage, unalloyed",
            "steel, unalloyed",
            "GLO",
        ),
        "both-dri-ng-ccs": (
            "steel production, natural gas-based direct reduction iron-electric arc furnace, with carbon capture and storage, unalloyed",
            "steel, unalloyed",
            "GLO",
        ),

        # Iron production
        "iron-dri-hydrogen": (
            "pig iron production, hydrogen-based direct reduction iron",
            "pig iron",
            "GLO",
        ),
        "BF production": (
            "pig iron production",
            "pig iron",
            "RER",
        ),
        "DRI production": (
            "pig iron production, hydrogen-based direct reduction iron",
            "pig iron",
            "GLO",
        ),
        "iron-ccs": (
            "pig iron production, blast furnace, with carbon capture and storage",
            "pig iron",
            "GLO",
        ),
        "iron-tgr": (
            "pig iron production, top gas recycling-blast furnace",
            "pig iron",
            "GLO",
        ),
        "iron-tgr-ccs": (
            "pig iron production, blast furnace, with top gas recycling, with carbon capture and storage",
            "pig iron",
            "GLO",
        ),
        "iron-ng-dri": (
            "pig iron production, with natural gas-based direct reduction",
            "iron",
            "GLO",
        ),
        "iron-ng-dri-ccs": (
            "pig iron production, with natural gas-based direct reduction, with carbon capture and storage",
            "iron",
            "GLO",
        ),
        "iron-electrowinning": (
            "pig iron production, by electrowinning",
            "pig iron",
            "GLO",
        ),
    },

    "low-alloyed": {
        # Steel production
        "EAF steel production": (
            "steel production, electric, low-alloyed, secondary steel, using scrap iron" ,
            "steel, low-alloyed",
            "RoW",
        ),
        "EAF steel production (DRI, coal)": (
            "steel production, electric, low-alloyed, primary steel, using direct reduced iron",
            "steel, low-alloyed",
            "RoW",
        ),
        "EAF steel production (DRI, NG)": (
            "steel production, natural gas-based direct reduction iron-electric arc furnace, low-alloyed",
            "steel, low-alloyed",
            "GLO",
        ),
        "OHF steel production": (
            "steel production, blast furnace-basic oxygen furnace, low-alloyed",
            "steel, low-alloyed",
            "GLO",
        ),
        "BOF steel production": (
            "steel production, blast furnace-basic oxygen furnace, low-alloyed",
            "steel, low-alloyed",
            "GLO",
        ),
        "steel-ccs": (
            "steel production, blast furnace-basic oxygen furnace, with carbon capture and storage, low-alloyed",
            "steel, low-alloyed",
            "GLO",
        ),
        "both-ccs": (
            "steel production, blast furnace-basic oxygen furnace, with carbon capture and storage, low-alloyed",
            "steel, low-alloyed",
            "GLO",
        ),
        "steel-dri-hydrogen": (
            "steel production, hydrogen-based direct reduction iron-electric arc furnace, low-alloyed",
            "steel, low-alloyed",
            "GLO",
        ),
        "both-dri-hydrogen": (
            "steel production, hydrogen-based direct reduction iron-electric arc furnace, low-alloyed",
            "steel, low-alloyed",
            "GLO",
        ),
        "steel-tgr": (
            "steel production, blast furnace-basic oxygen furnace, with top gas recycling, low-alloyed",
            "steel, low-alloyed",
            "GLO",
        ),
        "both-tgr": (
            "steel production, blast furnace-basic oxygen furnace, with top gas recycling, low-alloyed",
            "steel, low-alloyed",
            "GLO",
        ),
        "steel-tgr-ccs": (
            "steel production, blast furnace-basic oxygen furnace, with top gas recycling, with carbon capture and storage, low-alloyed",
            "steel, low-alloyed",
            "GLO",
        ),
        "both-tgr-ccs": (
            "steel production, blast furnace-basic oxygen furnace, with top gas recycling, with carbon capture and storage, low-alloyed",
            "steel, low-alloyed",
            "GLO",
        ),
        "steel-electrowinning": (
            "steel production, electrowinning-electric arc furnace, low-alloyed",
            "steel, low-alloyed",
            "GLO",
        ),
        "both-electrowinning": (
            "steel production, electrowinning-electric arc furnace, low-alloyed",
            "steel, low-alloyed",
            "GLO",
        ),
        "steel-dri-ng-ccs": (
            "steel production, natural gas-based direct reduction iron-electric arc furnace, with carbon capture and storage, low-alloyed",
            "steel, low-alloyed",
            "GLO",
        ),
        "both-dri-ng-ccs": (
            "steel production, natural gas-based direct reduction iron-electric arc furnace, with carbon capture and storage, low-alloyed",
            "steel, low-alloyed",
            "GLO",
        ),

        # Iron production
        "iron-dri-hydrogen": (
            "pig iron production, hydrogen-based direct reduction iron",
            "pig iron",
            "GLO",
        ),
        "BF production": (
            "pig iron production",
            "pig iron",
            "RER",
        ),
        "DRI production": (
            "pig iron production, hydrogen-based direct reduction iron",
            "pig iron",
            "GLO",
        ),
        "iron-ccs": (
            "pig iron production, blast furnace, with carbon capture and storage",
            "pig iron",
            "GLO",
        ),
        "iron-tgr": (
            "pig iron production, top gas recycling-blast furnace",
            "pig iron",
            "GLO",
        ),
        "iron-tgr-ccs": (
            "pig iron production, blast furnace, with top gas recycling, with carbon capture and storage",
            "pig iron",
            "GLO",
        ),
        "iron-ng-dri": (
            "pig iron production, with natural gas-based direct reduction",
            "iron",
            "GLO",
        ),
        "iron-ng-dri-ccs": (
            "pig iron production, with natural gas-based direct reduction, with carbon capture and storage",
            "iron",
            "GLO",
        ),
        "iron-electrowinning": (
            "pig iron production, by electrowinning",
            "pig iron",
            "GLO",
        ),
    },
}



dict_acts_future_dri = {
    "unalloyed": {
        "EAF steel production": (
            "steel production, electric, low-alloyed, secondary steel, using scrap iron" ,
            "steel, low-alloyed",
            "RoW",
        ),  # still fallback because no low-alloyed EAF steel in your setup

        "EAF steel production (DRI, coal)": (
            "steel production, hydrogen-based direct reduction iron-electric arc furnace, unalloyed",
            "steel, unalloyed",
            "GLO",
        ),
        "EAF steel production (DRI, NG)": (
            "steel production, hydrogen-based direct reduction iron-electric arc furnace, unalloyed",
            "steel, unalloyed",
            "GLO",
        ),

        "OHF steel production": (
            "steel production, hydrogen-based direct reduction iron-electric arc furnace, unalloyed",
            "steel, unalloyed",
            "GLO",
        ),
        "BOF steel production": (
            "steel production, hydrogen-based direct reduction iron-electric arc furnace, unalloyed",
            "steel, unalloyed",
            "GLO",
        ),

        "steel-ccs": (
            "steel production, blast furnace-basic oxygen furnace, with carbon capture and storage, unalloyed",
            "steel, unalloyed",
            "GLO",
        ),
        "both-ccs": (
            "steel production, blast furnace-basic oxygen furnace, with carbon capture and storage, unalloyed",
            "steel, unalloyed",
            "GLO",
        ),

        "steel-dri-hydrogen": (
            "steel production, hydrogen-based direct reduction iron-electric arc furnace, unalloyed",
            "steel, unalloyed",
            "GLO",
        ),
        "both-dri-hydrogen": (
            "steel production, hydrogen-based direct reduction iron-electric arc furnace, unalloyed",
            "steel, unalloyed",
            "GLO",
        ),

        "iron-dri-hydrogen": (
            "pig iron production, hydrogen-based direct reduction iron",
            "pig iron",
            "GLO",
        ),
        "BF production": (
            "pig iron production",
            "pig iron",
            "RER",
        ),
        "DRI production": (
            "pig iron production, hydrogen-based direct reduction iron",
            "pig iron",
            "GLO",
        ),
        "iron-ccs": (
            "pig iron production, blast furnace, with carbon capture and storage",
            "pig iron",
            "GLO",
        ),
    },

    "low-alloyed": {
        "EAF steel production": (
            "steel production, electric, low-alloyed, secondary steel, using scrap iron" ,
            "steel, low-alloyed",
            "RoW",
        ),
        "EAF steel production (DRI, coal)": (
            "steel production, hydrogen-based direct reduction iron-electric arc furnace, low-alloyed",
            "steel, low-alloyed",
            "GLO",
        ),
        "EAF steel production (DRI, NG)": (
            "steel production, hydrogen-based direct reduction iron-electric arc furnace, low-alloyed",
            "steel, low-alloyed",
            "GLO",
        ),

        "OHF steel production": (
            "steel production, hydrogen-based direct reduction iron-electric arc furnace, low-alloyed",
            "steel, low-alloyed",
            "GLO",
        ),
        "BOF steel production": (
            "steel production, hydrogen-based direct reduction iron-electric arc furnace, low-alloyed",
            "steel, low-alloyed",
            "GLO",
        ),

        "steel-ccs": (
            "steel production, blast furnace-basic oxygen furnace, with carbon capture and storage, low-alloyed",
            "steel, low-alloyed",
            "GLO",
        ),
        "both-ccs": (
            "steel production, blast furnace-basic oxygen furnace, with carbon capture and storage, low-alloyed",
            "steel, low-alloyed",
            "GLO",
        ),

        "steel-dri-hydrogen": (
            "steel production, hydrogen-based direct reduction iron-electric arc furnace, low-alloyed",
            "steel, low-alloyed",
            "GLO",
        ),
        "both-dri-hydrogen": (
            "steel production, hydrogen-based direct reduction iron-electric arc furnace, low-alloyed",
            "steel, low-alloyed",
            "GLO",
        ),

        "iron-dri-hydrogen": (
            "pig iron production, hydrogen-based direct reduction iron",
            "pig iron",
            "GLO",
        ),
        "BF production": (
            "pig iron production",
            "pig iron",
            "RER",
        ),
        "DRI production": (
            "pig iron production, hydrogen-based direct reduction iron",
            "pig iron",
            "GLO",
        ),
        "iron-ccs": (
            "pig iron production, blast furnace, with carbon capture and storage",
            "pig iron",
            "GLO",
        ),
    },
}

dict_acts_future_ccs = {
    "unalloyed": {
        "EAF steel production": (
            "steel production, electric, low-alloyed, secondary steel, using scrap iron" ,
            "steel, low-alloyed",
            "RoW",
        ),  # fallback kept as in your original logic

        "EAF steel production (DRI, coal)": (
            "steel production, blast furnace-basic oxygen furnace, with carbon capture and storage, unalloyed",
            "steel, unalloyed",
            "GLO",
        ),
        "EAF steel production (DRI, NG)": (
            "steel production, blast furnace-basic oxygen furnace, with carbon capture and storage, unalloyed",
            "steel, unalloyed",
            "GLO",
        ),

        "OHF steel production": (
            "steel production, blast furnace-basic oxygen furnace, with carbon capture and storage, unalloyed",
            "steel, unalloyed",
            "GLO",
        ),
        "BOF steel production": (
            "steel production, blast furnace-basic oxygen furnace, with carbon capture and storage, unalloyed",
            "steel, unalloyed",
            "GLO",
        ),

        "steel-ccs": (
            "steel production, blast furnace-basic oxygen furnace, with carbon capture and storage, unalloyed",
            "steel, unalloyed",
            "GLO",
        ),
        "both-ccs": (
            "steel production, blast furnace-basic oxygen furnace, with carbon capture and storage, unalloyed",
            "steel, unalloyed",
            "GLO",
        ),

        "steel-dri-hydrogen": (
            "steel production, hydrogen-based direct reduction iron-electric arc furnace, unalloyed",
            "steel, unalloyed",
            "GLO",
        ),
        "both-dri-hydrogen": (
            "steel production, hydrogen-based direct reduction iron-electric arc furnace, unalloyed",
            "steel, unalloyed",
            "GLO",
        ),

        "iron-dri-hydrogen": (
            "pig iron production, hydrogen-based direct reduction iron",
            "pig iron",
            "GLO",
        ),
        "BF production": (
            "pig iron production",
            "pig iron",
            "RER",
        ),
        "DRI production": (
            "pig iron production, hydrogen-based direct reduction iron",
            "pig iron",
            "GLO",
        ),
        "iron-ccs": (
            "pig iron production, blast furnace, with carbon capture and storage",
            "pig iron",
            "GLO",
        ),
    },

    "low-alloyed": {
        "EAF steel production": (
            "steel production, electric, low-alloyed, secondary steel, using scrap iron" ,
            "steel, low-alloyed",
            "RoW",
        ),
        "EAF steel production (DRI, coal)": (
            "steel production, blast furnace-basic oxygen furnace, with carbon capture and storage, low-alloyed",
            "steel, low-alloyed",
            "GLO",
        ),
        "EAF steel production (DRI, NG)": (
            "steel production, blast furnace-basic oxygen furnace, with carbon capture and storage, low-alloyed",
            "steel, low-alloyed",
            "GLO",
        ),

        "OHF steel production": (
            "steel production, blast furnace-basic oxygen furnace, with carbon capture and storage, low-alloyed",
            "steel, low-alloyed",
            "GLO",
        ),
        "BOF steel production": (
            "steel production, blast furnace-basic oxygen furnace, with carbon capture and storage, low-alloyed",
            "steel, low-alloyed",
            "GLO",
        ),

        "steel-ccs": (
            "steel production, blast furnace-basic oxygen furnace, with carbon capture and storage, low-alloyed",
            "steel, low-alloyed",
            "GLO",
        ),
        "both-ccs": (
            "steel production, blast furnace-basic oxygen furnace, with carbon capture and storage, low-alloyed",
            "steel, low-alloyed",
            "GLO",
        ),

        "steel-dri-hydrogen": (
            "steel production, hydrogen-based direct reduction iron-electric arc furnace, low-alloyed",
            "steel, low-alloyed",
            "GLO",
        ),
        "both-dri-hydrogen": (
            "steel production, hydrogen-based direct reduction iron-electric arc furnace, low-alloyed",
            "steel, low-alloyed",
            "GLO",
        ),

        "iron-dri-hydrogen": (
            "pig iron production, hydrogen-based direct reduction iron",
            "pig iron",
            "GLO",
        ),
        "BF production": (
            "pig iron production",
            "pig iron",
            "RER",
        ),
        "DRI production": (
            "pig iron production, hydrogen-based direct reduction iron",
            "pig iron",
            "GLO",
        ),
        "iron-ccs": (
            "pig iron production, blast furnace, with carbon capture and storage",
            "pig iron",
            "GLO",
        ),
    },
}

dict_acts_future_ew = {
    "unalloyed": {
        "EAF steel production": (
            "steel production, electric, low-alloyed, secondary steel, using scrap iron" ,
            "steel, low-alloyed",
            "RoW",
        ),  # fallback kept as in your original logic

        "EAF steel production (DRI, coal)": (
            "steel production, electrowinning-electric arc furnace, unalloyed",
            "steel, unalloyed",
            "GLO",
        ),
        "EAF steel production (DRI, NG)": (
            "steel production, electrowinning-electric arc furnace, unalloyed",
            "steel, unalloyed",
            "GLO",
        ),

        "OHF steel production": (
            "steel production, electrowinning-electric arc furnace, unalloyed",
            "steel, unalloyed",
            "GLO",
        ),
        "BOF steel production": (
            "steel production, electrowinning-electric arc furnace, unalloyed",
            "steel, unalloyed",
            "GLO",
        ),

        "steel-ccs": (
            "steel production, blast furnace-basic oxygen furnace, with carbon capture and storage, unalloyed",
            "steel, unalloyed",
            "GLO",
        ),
        "both-ccs": (
            "steel production, blast furnace-basic oxygen furnace, with carbon capture and storage, unalloyed",
            "steel, unalloyed",
            "GLO",
        ),

        "steel-dri-hydrogen": (
            "steel production, hydrogen-based direct reduction iron-electric arc furnace, unalloyed",
            "steel, unalloyed",
            "GLO",
        ),
        "both-dri-hydrogen": (
            "steel production, hydrogen-based direct reduction iron-electric arc furnace, unalloyed",
            "steel, unalloyed",
            "GLO",
        ),

        "iron-dri-hydrogen": (
            "pig iron production, hydrogen-based direct reduction iron",
            "pig iron",
            "GLO",
        ),
        "BF production": (
            "pig iron production",
            "pig iron",
            "RER",
        ),
        "DRI production": (
            "pig iron production, hydrogen-based direct reduction iron",
            "pig iron",
            "GLO",
        ),
        "iron-ccs": (
            "pig iron production, blast furnace, with carbon capture and storage",
            "pig iron",
            "GLO",
        ),
    },

    "low-alloyed": {
        "EAF steel production": (
            "steel production, electric, low-alloyed, secondary steel, using scrap iron" ,
            "steel, low-alloyed",
            "RoW",
        ),
        "EAF steel production (DRI, coal)": (
            "steel production, electrowinning-electric arc furnace, low-alloyed",
            "steel, low-alloyed",
            "GLO",
        ),
        "EAF steel production (DRI, NG)": (
            "steel production, electrowinning-electric arc furnace, low-alloyed",
            "steel, low-alloyed",
            "GLO",
        ),

        "OHF steel production": (
            "steel production, electrowinning-electric arc furnace, low-alloyed",
            "steel, low-alloyed",
            "GLO",
        ),
        "BOF steel production": (
            "steel production, electrowinning-electric arc furnace, low-alloyed",
            "steel, low-alloyed",
            "GLO",
        ),

        "steel-ccs": (
            "steel production, blast furnace-basic oxygen furnace, with carbon capture and storage, low-alloyed",
            "steel, low-alloyed",
            "GLO",
        ),
        "both-ccs": (
            "steel production, blast furnace-basic oxygen furnace, with carbon capture and storage, low-alloyed",
            "steel, low-alloyed",
            "GLO",
        ),

        "steel-dri-hydrogen": (
            "steel production, hydrogen-based direct reduction iron-electric arc furnace, low-alloyed",
            "steel, low-alloyed",
            "GLO",
        ),
        "both-dri-hydrogen": (
            "steel production, hydrogen-based direct reduction iron-electric arc furnace, low-alloyed",
            "steel, low-alloyed",
            "GLO",
        ),

        "iron-dri-hydrogen": (
            "pig iron production, hydrogen-based direct reduction iron",
            "pig iron",
            "GLO",
        ),
        "BF production": (
            "pig iron production",
            "pig iron",
            "RER",
        ),
        "DRI production": (
            "pig iron production, hydrogen-based direct reduction iron",
            "pig iron",
            "GLO",
        ),
        "iron-ccs": (
            "pig iron production, blast furnace, with carbon capture and storage",
            "pig iron",
            "GLO",
        ),
    },
}

db_label_map = {
    "ecoinvent_312_reference": "2025",
    "ecoinvent_image_SSP2-L_2030_base": "2030",
    "ecoinvent_image_SSP2-L_2035_base": "2035",
    "ecoinvent_image_SSP2-L_2040_base": "2040",
    "ecoinvent_image_SSP2-L_2045_base": "2045",
    "ecoinvent_image_SSP2-L_2050_base": "2050"
}

name_mapping_techs = {
    "steel production, blast furnace-basic oxygen furnace, with top gas recycling, with carbon capture and storage, unalloyed": "TGR-BF-BOF + CCS",
    "steel production, electrowinning-electric arc furnace, unalloyed": "EW-EAF",
    "steel production, converter, unalloyed": "BF-BOF",
    "steel production, natural gas-based direct reduction iron-electric arc furnace, unalloyed": "DRI-EAF (NG)",
    "steel production, hydrogen-based direct reduction iron-electric arc furnace, unalloyed": "DRI-EAF (H$_2$)",
    "steel production, electric, low-alloyed, secondary steel, using scrap iron" : "EAF (Secondary, scrap)",
    "steel production, natural gas-based direct reduction iron-electric arc furnace, with carbon capture and storage, unalloyed": "DRI-EAF (NG) + CCS",
    "steel production, electric, low-alloyed, primary steel, using direct reduced iron": "EAF (DRI, coal)",
    "steel production, blast furnace-basic oxygen furnace, with top gas recycling, unalloyed": "TGR-BF-BOF",
    "steel production, blast furnace-basic oxygen furnace, with carbon capture and storage, unalloyed": "BF-BOF + CCS",
}