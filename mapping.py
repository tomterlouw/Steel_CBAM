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

CO2_emission_factor_hydrogen = {'coal gasification': 24, #https://iea.blob.core.windows.net/assets/acc7a642-e42b-4972-8893-2f03bf0bfa03/Towardshydrogendefinitionsbasedontheiremissionsintensity.pdf?utm_source=chatgpt.com
                                'steam methane reforming': 11, #https://iea.blob.core.windows.net/assets/acc7a642-e42b-4972-8893-2f03bf0bfa03/Towardshydrogendefinitionsbasedontheiremissionsintensity.pdf?utm_source=chatgpt.com
                                'hydrogen production, gaseous, 30 bar, from PEM electrolysis': 0,
                                'other': 0}

# https://taxation-customs.ec.europa.eu/document/download/017e46f1-dd1a-4235-b2d7-dafcc6692acf_en?filename=Default%20values%20transitional%20period.pdf
CO2_emission_factor_precursors_kgCO2_kg = {'sintered ore': 0.36, #https://taxation-customs.ec.europa.eu/document/download/017e46f1-dd1a-4235-b2d7-dafcc6692acf_en?filename=Default%20values%20transitional%20period.pdf, 'Weighted average primary', CN 2601 12 00
                                 'pig iron': 2.07, #https://taxation-customs.ec.europa.eu/document/download/017e46f1-dd1a-4235-b2d7-dafcc6692acf_en?filename=Default%20values%20transitional%20period.pdf, 'Weighted average primary', CN 7201
                                  'iron scrap': 0,
                                   'iron scrap, sorted, pressed': 0,
                                    'iron ore concentrate':0,
                                      'iron ore beneficiation':0,
                                  'ferro-chromium': 5.45, #https://taxation-customs.ec.europa.eu/document/download/017e46f1-dd1a-4235-b2d7-dafcc6692acf_en?filename=Default%20values%20transitional%20period.pdf, 'Weighted average primary'

                                 'ferro-manganese': 3.51, #https://taxation-customs.ec.europa.eu/document/download/017e46f1-dd1a-4235-b2d7-dafcc6692acf_en?filename=Default%20values%20transitional%20period.pdf, 'Weighted average primary'

                                 'ferro-nickel': 6.26,  #https://taxation-customs.ec.europa.eu/document/download/017e46f1-dd1a-4235-b2d7-dafcc6692acf_en?filename=Default%20values%20transitional%20period.pdf, 'Weighted average primary'

                                 'direct reduced iron': 0.70, #NOT CLEAR IN JRC publication so average taken from Table 4.1, https://www.ipcc-nggip.iges.or.jp/public/2006gl/pdf/3_Volume3/V3_4_Ch4_Metal_Industry.pdf
                                 'steel': 2.2, #https://taxation-customs.ec.europa.eu/document/download/017e46f1-dd1a-4235-b2d7-dafcc6692acf_en?filename=Default%20values%20transitional%20period.pdf, 'Weighted average primary', 'Other articles of iron or steel'
                                 'hydrogen': "SPECIFIC_HYDROGEN", # not included as precursor here, handled separately
                                      }

dict_types = {'steel production, electric, low-alloyed': 'secondary steel (EAF)',
              'steel production, electric, unalloyed': 'secondary steel (EAF)',
             'H2-DRI, iron production': 'iron production (H$_2$-DRI)',
              'NG-DRI-EAF, steel production, low-alloyed': 'primary steel (DRI-EAF, NG)',
              'NG-DRI-EAF, steel production, unalloyed': 'primary steel (DRI-EAF, NG)',
             'steel production, electric, low-alloyed, primary steel, using direct reduced iron': 'primary steel (DRI-EAF, coal)',
             'steel production, electric, unalloyed, primary steel, using direct reduced iron': 'primary steel (DRI-EAF, coal)',
             'steel production, converter, low-alloyed, only primary steel': 'primary steel (BF-BOF)',
                'steel production, converter, unalloyed, only primary steel': 'primary steel (BF-BOF)',
             'H2-DRI-EAF, steel production, low-alloyed': 'primary steel (DRI-EAF, H$_2$)',
              'H2-DRI-EAF, steel production, unalloyed': 'primary steel (DRI-EAF, H$_2$)',
             'BF-BOF+CCS, steel production, low-alloyed': 'primary steel (BF-BOF-CCS)',
              'BF-BOF+CCS, steel production, unalloyed': 'primary steel (BF-BOF-CCS)',
             'pig iron production': 'iron production', 
             'EW-EAF, steel production, low-alloyed': 'primary steel (EW-EAF)',
             'EW-EAF, steel production, unalloyed': 'primary steel (EW-EAF)',
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


dict_acts = {"unalloyed":

                 {'EAF steel production': ('steel production, electric, low-alloyed', 'steel, low-alloyed', "RoW"), # no low-alloyed EAF steel
                  'EAF steel production (DRI, coal)':  ('steel production, electric, low-alloyed, primary steel, using direct reduced iron', 
                                                  'steel, low-alloyed', "RoW"), # no low-alloyed EAF steel
                          'EAF steel production (DRI, NG)': ('NG-DRI-EAF, steel production, unalloyed',
                                                             'NG-DRI-EAF steel, unalloyed', 'GLO'),
                 
                  'OHF steel production': ('steel production, converter, unalloyed, only primary steel', 'steel, unalloyed', "RER"),
                 'BOF steel production': ('steel production, converter, unalloyed, only primary steel', 'steel, unalloyed', "RER"),
                 'steel-ccs': ('BF-BOF+CCS, steel production, unalloyed', 'steel, unalloyed', 'GLO'),
                 'both-ccs': ('BF-BOF+CCS, steel production, unalloyed', 'steel, unalloyed', 'GLO'),                  
                 'steel-dri-hydrogen': ('H2-DRI-EAF, steel production, unalloyed', 'H2-DRI-EAF steel, unalloyed', 'GLO'),
                 'both-dri-hydrogen': ('H2-DRI-EAF, steel production, unalloyed', 'H2-DRI-EAF steel, unalloyed', 'GLO'),
                  
                 'iron-dri-hydrogen': ('H2-DRI, iron production', 'H2-DRI iron', 'GLO'),
                 'BF production': ('pig iron production', 'pig iron', "RER"), 
                 'DRI production': ('H2-DRI, iron production', 'H2-DRI iron', 'GLO'),
                 'iron-ccs': ('BF+CCS, iron production', 'BF+CCS iron', 'GLO')},
    
            "low-alloyed":
                {'EAF steel production': ('steel production, electric, low-alloyed', 'steel, low-alloyed', "RoW"),
                          'EAF steel production (DRI, coal)':  ('steel production, electric, low-alloyed, primary steel, using direct reduced iron', 'steel, low-alloyed', "RoW"),

                      'EAF steel production (DRI, NG)': ('NG-DRI-EAF, steel production, low-alloyed',
                                                         'NG-DRI-EAF steel, low-alloyed', 'GLO'),
                         'OHF steel production': ('steel production, converter, low-alloyed, only primary steel', 'steel, low-alloyed', "RER"),
                         'BOF steel production': ('steel production, converter, low-alloyed, only primary steel', 'steel, low-alloyed', "RER"),
                         'steel-ccs': ('BF-BOF+CCS, steel production, low-alloyed', 'steel, low-alloyed', 'GLO'),
                         'both-ccs': ('BF-BOF+CCS, steel production, low-alloyed', 'steel, low-alloyed', 'GLO'),
                         'steel-dri-hydrogen': ('H2-DRI-EAF, steel production, low-alloyed', 'H2-DRI-EAF steel, low-alloyed', 'GLO'),
                         'both-dri-hydrogen': ('H2-DRI-EAF, steel production, low-alloyed', 'H2-DRI-EAF steel, low-alloyed', 'GLO'),
                 
                         'iron-dri-hydrogen': ('H2-DRI, iron production', 'H2-DRI iron', 'GLO'),
                         'BF production': ('pig iron production', 'pig iron', "RER"), 
                         'DRI production': ('H2-DRI, iron production', 'H2-DRI iron', 'GLO'),
                         'iron-ccs': ('BF+CCS, iron production', 'BF+CCS iron', 'GLO')}
            
            }

dict_acts_future_dri = {"unalloyed":

                 {'EAF steel production': ('steel production, electric, low-alloyed', 'steel, low-alloyed', "RoW"), # no low-alloyed EAF steel
                  'EAF steel production (DRI, coal)':  ('H2-DRI-EAF, steel production, unalloyed', 'H2-DRI-EAF steel, unalloyed', 'GLO'),
                  'EAF steel production (DRI, NG)': ('H2-DRI-EAF, steel production, unalloyed', 'H2-DRI-EAF steel, unalloyed', 'GLO'),
                 
                  'OHF steel production': ('H2-DRI-EAF, steel production, unalloyed', 'H2-DRI-EAF steel, unalloyed', 'GLO'),
                 'BOF steel production': ('H2-DRI-EAF, steel production, unalloyed', 'H2-DRI-EAF steel, unalloyed', 'GLO'),
                 'steel-ccs': ('BF-BOF+CCS, steel production, unalloyed', 'steel, unalloyed', 'GLO'),
                 'both-ccs': ('BF-BOF+CCS, steel production, unalloyed', 'steel, unalloyed', 'GLO'),                  
                 'steel-dri-hydrogen': ('H2-DRI-EAF, steel production, unalloyed', 'H2-DRI-EAF steel, unalloyed', 'GLO'),
                 'both-dri-hydrogen': ('H2-DRI-EAF, steel production, unalloyed', 'H2-DRI-EAF steel, unalloyed', 'GLO'),
                  
                 'iron-dri-hydrogen': ('H2-DRI, iron production', 'H2-DRI iron', 'GLO'),
                 'BF production': ('pig iron production', 'pig iron', "RER"), 
                 'DRI production': ('H2-DRI, iron production', 'H2-DRI iron', 'GLO'),
                 'iron-ccs': ('BF+CCS, iron production', 'BF+CCS iron', 'GLO')},
    
            "low-alloyed":
                        
                {'EAF steel production': ('steel production, electric, low-alloyed', 'steel, low-alloyed', "RoW"),
                          'EAF steel production (DRI, coal)': ('H2-DRI-EAF, steel production, low-alloyed', 'H2-DRI-EAF steel, low-alloyed', 'GLO'),
                            'EAF steel production (DRI, NG)': ('H2-DRI-EAF, steel production, low-alloyed', 'H2-DRI-EAF steel, low-alloyed', 'GLO'),
                 
                         'OHF steel production': ('H2-DRI-EAF, steel production, low-alloyed', 'H2-DRI-EAF steel, low-alloyed', 'GLO'),
                         'BOF steel production': ('H2-DRI-EAF, steel production, low-alloyed', 'H2-DRI-EAF steel, low-alloyed', 'GLO'),
                         'steel-ccs': ('BF-BOF+CCS, steel production, low-alloyed', 'steel, low-alloyed', 'GLO'),
                         'both-ccs': ('BF-BOF+CCS, steel production, low-alloyed', 'steel, low-alloyed', 'GLO'),
                         'steel-dri-hydrogen': ('H2-DRI-EAF, steel production, low-alloyed', 'H2-DRI-EAF steel, low-alloyed', 'GLO'),
                         'both-dri-hydrogen': ('H2-DRI-EAF, steel production, low-alloyed', 'H2-DRI-EAF steel, low-alloyed', 'GLO'),
                 
                         'iron-dri-hydrogen': ('H2-DRI, iron production', 'H2-DRI iron', 'GLO'),
                         'BF production': ('pig iron production', 'pig iron', "RER"), 
                         'DRI production': ('H2-DRI, iron production', 'H2-DRI iron', 'GLO'),
                         'iron-ccs': ('BF+CCS, iron production', 'BF+CCS iron', 'GLO')},
            
            }

dict_acts_future_ccs = {"unalloyed":

                 {'EAF steel production': ('steel production, electric, low-alloyed', 'steel, low-alloyed', "RoW"), # no low-alloyed EAF steel
                  'EAF steel production (DRI, coal)': ('BF-BOF+CCS, steel production, unalloyed', 'steel, unalloyed', 'GLO'),
                  'EAF steel production (DRI, NG)': ('BF-BOF+CCS, steel production, unalloyed', 'steel, unalloyed', 'GLO'),
                 
                  'OHF steel production': ('BF-BOF+CCS, steel production, unalloyed', 'steel, unalloyed', 'GLO'),
                 'BOF steel production': ('BF-BOF+CCS, steel production, unalloyed', 'steel, unalloyed', 'GLO'),
                 'steel-ccs': ('BF-BOF+CCS, steel production, unalloyed', 'steel, unalloyed', 'GLO'),
                 'both-ccs': ('BF-BOF+CCS, steel production, unalloyed', 'steel, unalloyed', 'GLO'),                  
                 'steel-dri-hydrogen': ('H2-DRI-EAF, steel production, unalloyed', 'H2-DRI-EAF steel, unalloyed', 'GLO'),
                 'both-dri-hydrogen': ('H2-DRI-EAF, steel production, unalloyed', 'H2-DRI-EAF steel, unalloyed', 'GLO'),
                  
                 'iron-dri-hydrogen': ('H2-DRI, iron production', 'H2-DRI iron', 'GLO'),
                 'BF production': ('pig iron production', 'pig iron', "RER"), 
                 'DRI production': ('H2-DRI, iron production', 'H2-DRI iron', 'GLO'),
                 'iron-ccs': ('BF+CCS, iron production', 'BF+CCS iron', 'GLO')},
    
            "low-alloyed":
                        
                {'EAF steel production': ('steel production, electric, low-alloyed', 'steel, low-alloyed', "RoW"),
                          'EAF steel production (DRI, coal)': ('BF-BOF+CCS, steel production, low-alloyed', 'steel, low-alloyed', 'GLO'),
                         'EAF steel production (DRI, NG)': ('BF-BOF+CCS, steel production, low-alloyed', 'steel, low-alloyed', 'GLO'),
                 
                         'OHF steel production': ('BF-BOF+CCS, steel production, low-alloyed', 'steel, low-alloyed', 'GLO'),
                         'BOF steel production': ('BF-BOF+CCS, steel production, low-alloyed', 'steel, low-alloyed', 'GLO'),
                         'steel-ccs': ('BF-BOF+CCS, steel production, low-alloyed', 'steel, low-alloyed', 'GLO'),
                         'both-ccs': ('BF-BOF+CCS, steel production, low-alloyed', 'steel, low-alloyed', 'GLO'),
                         'steel-dri-hydrogen': ('H2-DRI-EAF, steel production, low-alloyed', 'H2-DRI-EAF steel, low-alloyed', 'GLO'),
                         'both-dri-hydrogen': ('H2-DRI-EAF, steel production, low-alloyed', 'H2-DRI-EAF steel, low-alloyed', 'GLO'),
                 
                         'iron-dri-hydrogen': ('H2-DRI, iron production', 'H2-DRI iron', 'GLO'),
                         'BF production': ('pig iron production', 'pig iron', "RER"), 
                         'DRI production': ('H2-DRI, iron production', 'H2-DRI iron', 'GLO'),
                         'iron-ccs': ('BF+CCS, iron production', 'BF+CCS iron', 'GLO')},
            
            }

dict_acts_future_ew = {"unalloyed":

                 {'EAF steel production': ('steel production, electric, low-alloyed', 'steel, low-alloyed', "RoW"), # no low-alloyed EAF steel
                  'EAF steel production (DRI, coal)': ('EW-EAF, steel production, unalloyed', 'EW-EAF steel, unalloyed', 'GLO'),
                  'EAF steel production (DRI, NG)': ('EW-EAF, steel production, unalloyed', 'EW-EAF steel, unalloyed', 'GLO'),
                 
                  'OHF steel production': ('EW-EAF, steel production, unalloyed', 'EW-EAF steel, unalloyed', 'GLO'),
                 'BOF steel production': ('EW-EAF, steel production, unalloyed', 'EW-EAF steel, unalloyed', 'GLO'),
                 'steel-ccs': ('BF-BOF+CCS, steel production, unalloyed', 'steel, unalloyed', 'GLO'),
                 'both-ccs': ('BF-BOF+CCS, steel production, unalloyed', 'steel, unalloyed', 'GLO'),                  
                 'steel-dri-hydrogen': ('H2-DRI-EAF, steel production, unalloyed', 'H2-DRI-EAF steel, unalloyed', 'GLO'),
                 'both-dri-hydrogen': ('H2-DRI-EAF, steel production, unalloyed', 'H2-DRI-EAF steel, unalloyed', 'GLO'),
                  
                 'iron-dri-hydrogen': ('H2-DRI, iron production', 'H2-DRI iron', 'GLO'),
                 'BF production': ('pig iron production', 'pig iron', "RER"), 
                 'DRI production': ('H2-DRI, iron production', 'H2-DRI iron', 'GLO'),
                 'iron-ccs': ('BF+CCS, iron production', 'BF+CCS iron', 'GLO')},
    
            "low-alloyed":

                {'EAF steel production': ('steel production, electric, low-alloyed', 'steel, low-alloyed', "RoW"),
                          'EAF steel production (DRI, coal)': ('EW-EAF, steel production, low-alloyed', 'EW-EAF steel, low-alloyed', 'GLO'),
                         'EAF steel production (DRI, NG)': ('EW-EAF, steel production, low-alloyed', 'EW-EAF steel, low-alloyed', 'GLO'),
                 
                         'OHF steel production': ('EW-EAF, steel production, low-alloyed', 'EW-EAF steel, low-alloyed', 'GLO'),
                         'BOF steel production': ('EW-EAF, steel production, low-alloyed', 'EW-EAF steel, low-alloyed', 'GLO'),
                         'steel-ccs': ('BF-BOF+CCS, steel production, low-alloyed', 'steel, low-alloyed', 'GLO'),
                         'both-ccs': ('BF-BOF+CCS, steel production, low-alloyed', 'steel, low-alloyed', 'GLO'),
                         'steel-dri-hydrogen': ('H2-DRI-EAF, steel production, low-alloyed', 'H2-DRI-EAF steel, low-alloyed', 'GLO'),
                         'both-dri-hydrogen': ('H2-DRI-EAF, steel production, low-alloyed', 'H2-DRI-EAF steel, low-alloyed', 'GLO'),
                 
                         'iron-dri-hydrogen': ('H2-DRI, iron production', 'H2-DRI iron', 'GLO'), #('EW, iron production', 'EW iron', 'GLO'),
                         'BF production': ('pig iron production', 'pig iron', "RER"), 
                         'DRI production': ('H2-DRI, iron production', 'H2-DRI iron', 'GLO'),
                         'iron-ccs': ('BF+CCS, iron production', 'BF+CCS iron', 'GLO')},
            
            }

db_label_map = {
    "ecoinvent_310_reference": "2025",
    "ecoinvent_image_SSP2-RCP26_2030_base": "2030",
    "ecoinvent_image_SSP2-RCP26_2035_base": "2035",
    "ecoinvent_image_SSP2-RCP26_2040_base": "2040",
    "ecoinvent_image_SSP2-RCP26_2045_base": "2045",
    "ecoinvent_image_SSP2-RCP26_2050_base": "2050"
}

name_mapping_techs = {
    'TGR-BF-BOF+CCS, steel production, unalloyed': 'TGR-BF-BOF + CCS',
    'EW-EAF, steel production, unalloyed': 'EW-EAF',
    'steel production, converter, unalloyed, only primary steel': 'BF-BOF',
    'NG-DRI-EAF, steel production, unalloyed': 'DRI-EAF (NG)',
    'H2-DRI-EAF, steel production, unalloyed': 'DRI-EAF (H$_2$)',
    'steel production, electric, low-alloyed, secondary steel, using scrap iron': 'EAF (Secondary, scrap)',
    'NG-DRI-EAF+CCS, steel production, unalloyed': 'DRI-EAF (NG) + CCS',
    'steel production, electric, low-alloyed, primary steel, using direct reduced iron': 'EAF (DRI, coal)',
    'TGR-BF-BOF, steel production, unalloyed': 'TGR-BF-BOF',
    'BF-BOF+CCS, steel production, unalloyed': 'BF-BOF + CCS'
}

