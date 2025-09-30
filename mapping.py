end_product_colors = {
    'secondary steel (EAF)': 'darkgreen',
    #'iron production (H$_2$-DRI)': 'steelblue',
    'primary steel (DRI-EAF, coal)': 'indianred',
    'primary steel (DRI-EAF, NG)': 'firebrick',
    'primary steel (BF-BOF)': 'black',
    'primary steel (DRI-EAF, H$_2$)': 'royalblue',
    'primary steel (BF-BOF-CCS)': 'gray',
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

CO2_emission_factor_hydrogen = {'coal gasification': 24, #https://iea.blob.core.windows.net/assets/acc7a642-e42b-4972-8893-2f03bf0bfa03/Towardshydrogendefinitionsbasedontheiremissionsintensity.pdf?utm_source=chatgpt.com
                                'steam methane reforming': 11, #https://iea.blob.core.windows.net/assets/acc7a642-e42b-4972-8893-2f03bf0bfa03/Towardshydrogendefinitionsbasedontheiremissionsintensity.pdf?utm_source=chatgpt.com
                                'hydrogen production, gaseous, 30 bar, from PEM electrolysis': 0,
                                'other': 0}

CO2_emission_factor_precursors_kgCO2_kg = {'sintered ore': 0.31, #https://publications.jrc.ec.europa.eu/repository/handle/JRC134682, 'Weighted average primary', CN 2601 12 00
                                 'pig iron': 1.90, #https://publications.jrc.ec.europa.eu/repository/handle/JRC134682, 'Weighted average primary', CN 7201
                                  'iron scrap': 0,
                                   'iron scrap, sorted, pressed': 0,
                                    'iron ore concentrate':0,
                                      'iron ore beneficiation':0,
                                  'ferro-chromium': 2.06, #https://publications.jrc.ec.europa.eu/repository/handle/JRC134682, 'Weighted average primary'

                                 'ferro-manganese': 1.44, #https://publications.jrc.ec.europa.eu/repository/handle/JRC134682, 'Weighted average primary'

                                 'ferro-nickel': 3.45,  #https://publications.jrc.ec.europa.eu/repository/handle/JRC134682, 'Weighted average primary'

                                 'direct reduced iron': 0.70, #NOT CLEAR IN JRC publication so average taken from Table 4.1, https://www.ipcc-nggip.iges.or.jp/public/2006gl/pdf/3_Volume3/V3_4_Ch4_Metal_Industry.pdf
                                 'steel': 1.97, #https://publications.jrc.ec.europa.eu/repository/handle/JRC134682, 'Weighted average primary', 'Other articles of iron or steel'
                                 'hydrogen': "SPECIFIC_HYDROGEN",
                                 # THOSE ARE NOT INCLUDED YET IN CBAM:
                                 #'lime': 0.75,#0.75, #2.3.1.2https://www.ipcc-nggip.iges.or.jp/public/2006gl/pdf/3_Volume3/V3_2_Ch2_Mineral_Industry.pdf
                                 #'ferrosilicon': (3.6+2.5)/2, #average tabken from Table 4.5, https://www.ipcc-nggip.iges.or.jp/public/2006gl/pdf/3_Volume3/V3_4_Ch4_Metal_Industry.pdf
                                 # market for coke is in MJ in ei, use 28.6 MJ/kg
                                 #'coke': 0.56/28.6, #0.56 kgCO2/kg Tier 1:https://www.ipcc-nggip.iges.or.jp/public/2006gl/pdf/3_Volume3/V3_4_Ch4_Metal_Industry.pdf
                                 
                                 # market for hard coal is in MJ in ei, use  27.91 MJ/kg          
                                #'coal': 2.4/27.91, #2.4 kgCO2/kg approximated based on https://www.epa.gov/sites/default/files/2020-04/documents/ghg-emission-factors-hub.pdf
                                #'aluminium': 1.65, #average of Table 4.10: https://www.ipcc-nggip.iges.or.jp/public/2006gl/pdf/3_Volume3/V3_4_Ch4_Metal_Industry.pdf}
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
             'pig iron production': 'iron production'}

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
                 'DRI production': ('H2-DRI, iron production', 'H2-DRI iron', 'GLO')},
                 'iron-ccs': ('BF+CCS, iron production', 'BF+CCS iron', 'GLO'),
    
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
                         'DRI production': ('H2-DRI, iron production', 'H2-DRI iron', 'GLO')},
                         'iron-ccs': ('BF+CCS, iron production', 'BF+CCS iron', 'GLO'),
            
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
                 'DRI production': ('H2-DRI, iron production', 'H2-DRI iron', 'GLO')},
                 'iron-ccs': ('BF+CCS, iron production', 'BF+CCS iron', 'GLO'),
    
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
                         'DRI production': ('H2-DRI, iron production', 'H2-DRI iron', 'GLO')},
                         'iron-ccs': ('BF+CCS, iron production', 'BF+CCS iron', 'GLO'),
            
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
                 'DRI production': ('H2-DRI, iron production', 'H2-DRI iron', 'GLO')},
                 'iron-ccs': ('BF+CCS, iron production', 'BF+CCS iron', 'GLO'),
    
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
                         'DRI production': ('H2-DRI, iron production', 'H2-DRI iron', 'GLO')},
                         'iron-ccs': ('BF+CCS, iron production', 'BF+CCS iron', 'GLO'),
            
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

