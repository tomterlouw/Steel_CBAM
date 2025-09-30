import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.patches as patches
import matplotlib.patches as mpatches  # Import for the legend patches
import cartopy

from mapping import * 
from functions import iso2_to_country

def custom_autopct(pct, all_vals):
    absolute = np.round(pct / 1. * np.sum(all_vals), 1)
    if pct > 10:
        return f'{pct:.0f}%\n{absolute / 1e2:.0f} Mt'
    elif pct > 5:
        return f'{pct:.0f}%\n{absolute / 1e2:.1f} Mt'
    return ''

def custom_autopct_2(pct, all_vals):
    absolute = np.round(pct / 1. * np.sum(all_vals), 1)
    if pct > 10:
        return f'{pct:.0f}%\n{absolute / 1e2:.1f} Mt'
    elif pct > 4.5:
        return f'{pct:.0f}%\n{absolute / 1e2:.1f} Mt'
    return ''

def plot_steel_supply_curve(
    results_df,
    get_continent_func,
    color_mapping,
    emission_thresholds=None,
    output_path=None,
    font_size=17,
    cut_axis_lim=6,
    inset_xlim=(0, 250),
    inset_ylim=(0, 1.75),
    inset_max_cumulative=1000,
    lca_impact_col = "lca_impact_climate change_wo_transport",
    zoom=True,
):
    """
    Plots the steel supply curve with climate impact annotations and inset zoom.

    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame containing steel production data. Required columns:
        ['country', 'lca_impact_climate change', 'production volume', 'commodity_type']
    
    get_continent_func : function
        Function that takes a country name and returns its continent.
    
    color_mapping : dict
        Dictionary mapping continent names to plot colors.
    
    emission_thresholds : dict, optional
        Dictionary of emission thresholds to display on the plot.
        Format: {'Label': value}. Example: {'BF-BOF (2025)': 2.2}
    
    output_path : str, optional
        File path to save the resulting plot. If None, the plot is not saved.
    
    font_size : int, default=17
        Base font size for labels and titles.
    
    cut_axis_lim : float, default=6
        Upper limit of the y-axis (climate impact).
    
    inset_xlim : tuple, default=(0, 250)
        x-axis limits for the inset plot.
    
    inset_ylim : tuple, default=(0, 1.75)
        y-axis limits for the inset plot.
    
    inset_max_cumulative : float, default=1000
        Maximum cumulative steel value to include in the inset plot (same unit as production volume).

    Returns:
    --------
    None
    """
    
    # Copy and sort data
    df = results_df.copy().sort_values(lca_impact_col)
    df['continent'] = df['country'].apply(get_continent_func)
    
    # Cumulative steel production
    df['cumulative_steel'] = df['production volume'].cumsum()
    df['next_cumulative_steel'] = df['cumulative_steel'].shift(-1)
    
    # Initialize plot
    fig, ax_main = plt.subplots(figsize=(12, 3.7))
    alpha_s = 1
    
    # Main supply curve
    for _, row in df.iterrows():
        ax_main.fill_betweenx(
            [0, row[lca_impact_col]],
            row['cumulative_steel'], row['next_cumulative_steel'],
            color=color_mapping.get(row['continent'], '#999999'),
            alpha=alpha_s
        )
    
    # Main axis settings
    ax_main.set_ylim(0, cut_axis_lim)
    ax_main.set_xlim(0, df['cumulative_steel'].max())
    ax_main.set_ylabel('Climate impact (kgCO$_2$-eq./kg steel)', fontsize=font_size - 3.8)
    ax_main.set_xlabel('Cumulative steel supply (Mt steel year$^{-1}$)', fontsize=font_size - 3)
    ax_main.set_title('$\mathbf{c}$ Steel supply curve', fontsize=font_size - 1, loc='left')
    ax_main.grid(color='gray', axis='y', linestyle='--', linewidth=0.5, alpha=0.9)

    # Emission thresholds
    if emission_thresholds:
        for label, value in emission_thresholds.items():
            ax_main.axhline(y=value, color='black', linestyle='--', linewidth=1, label=f'{label} ({value} kgCO₂/kg)')
            ax_main.fill_betweenx([0, value], 0, df['cumulative_steel'].max(), color='gray', alpha=0.1)
    
    # Inset plot
    if zoom:
        ax_inset = inset_axes(
            ax_main, width="30%", height="30%", loc='upper left',
            bbox_to_anchor=(0.05, -0.108, 1, 1.06), bbox_transform=ax_main.transAxes
        )
        
        for _, row in df.iterrows():
            if row['cumulative_steel'] <= inset_max_cumulative:
                ax_inset.fill_betweenx(
                    [0, row[lca_impact_col]],
                    row['cumulative_steel'], row['next_cumulative_steel'],
                    color=color_mapping.get(row['continent'], '#999999'),
                    alpha=alpha_s
                )
        
        ax_inset.set_xlim(*inset_xlim)
        ax_inset.set_ylim(*inset_ylim)
        ax_inset.set_title('First 250 Mt steel', fontsize=font_size - 5, fontweight='bold', y=0.965)
        ax_inset.grid(color='gray', axis='y', linestyle='--', linewidth=0.5, alpha=0.9)
        
        mark_inset(ax_main, ax_inset, loc1=2, loc2=4, fc="none", ec="black")
    
    # Legend
    handles = [
        plt.Line2D([0], [0], color=color_mapping[cont], lw=4, label=cont)
        for cont in df['continent'].unique() if cont in color_mapping
    ]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.52, -0.14), ncol=3,
               fontsize=font_size - 4, frameon=False)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_steel_production(import_df, production_col='production volume', max_nr_countries=25, export_figure=False,max_y=1200, cc_name_col="Plant_GHG_emissions_Mt_wo_transport"):
    """
    Plots the production of primary and secondary steel per country as a stacked bar chart,
    with total production values displayed above each bar. If 'production volume' is selected,
    also plots GHG intensity (tCO₂/tonne) on a secondary y-axis.
    
    Parameters:
        import_df (pd.DataFrame): DataFrame containing country-wise steel production data.
        production_col (str): Column name representing production volume.
    """
    df_filtered = import_df[import_df['commodity_type'].isin([i[1] for i in dict_types.items() if 'steel' in str(i)])].copy()

    fontsize=12
    
    # Convert location codes to full country names
    df_filtered['country'] = df_filtered['location'].apply(iso2_to_country)
    
    # Pivot data for stacked bar plot
    df_pivot = df_filtered.pivot_table(index='country', columns='commodity_type', values=production_col, aggfunc='sum').fillna(0)
    
    # Compute total production per country and sort
    df_pivot['total'] = df_pivot.sum(axis=1)
    df_pivot = df_pivot.sort_values(by='total', ascending=False)[:max_nr_countries]

    fig, ax = plt.subplots(figsize=(12, 4))
    bar_width = 0.7
    x = np.arange(len(df_pivot))

    bottom_values = np.zeros(len(df_pivot))
    steel_types = []

    for steel_type in [i[1] for i in dict_types.items() if 'steel' in str(i)]:
        if steel_type not in df_pivot.columns or steel_type in steel_types:
            continue
        ax.bar(x, df_pivot[steel_type], width=bar_width, label=steel_type,
               color=end_product_colors[steel_type], bottom=bottom_values)
        bottom_values += df_pivot[steel_type].values
        steel_types.append(steel_type)

    for i, total in enumerate(df_pivot['total']):
        ax.text(x[i], total * 1.01, f'{int(total):.0f}', ha='center', fontsize=fontsize-1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(df_pivot.index, rotation=45, ha='right', fontsize=fontsize)
    ax.set_ylabel(f'{production_col.replace("_"," ").capitalize()} (Mt per year)', fontsize=fontsize)
    letter='$\mathbf{b}$'
    ax.set_title(f'{letter} {production_col.replace("_"," ").capitalize()} and country-specific climate impact of steel production', 
                 loc='left', fontsize=fontsize+2)
    #ax.legend(frameon=False, loc='best', ncol=5)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0,max_y)
    ax.set_xlim(-0.5,max_nr_countries-0.5)

    ax.tick_params(axis='y', labelsize=fontsize) 

    # Add GHG intensity if plotting production volume
    if production_col == 'production volume':
        df_emissions = import_df[import_df['commodity_type'].isin(steel_types)].copy()
        df_emissions['country'] = df_emissions['location'].apply(iso2_to_country)

        df_prod = df_emissions.pivot_table(index='country', columns='commodity_type', values='production volume', aggfunc='sum').fillna(0)
        df_em = df_emissions.pivot_table(index='country', columns='commodity_type', values=cc_name_col, aggfunc='sum').fillna(0)

        df_prod['total'] = df_prod.sum(axis=1)
        df_em['total'] = df_em.sum(axis=1)

        common_countries = df_pivot.index.intersection(df_prod.index).intersection(df_em.index)
        ghg_intensity = (df_em.loc[common_countries, 'total'] / df_prod.loc[common_countries, 'total'])  # tCO2/t

        ax2 = ax.twinx()
        ax2.scatter(x[:len(ghg_intensity)], ghg_intensity.values, color='black', zorder=5, edgecolor='white')
        ax2.tick_params(axis='y', labelsize=fontsize) 

        for i, val in enumerate(ghg_intensity.values):
            ax2.text(x[i], val * 1.02, f'{val:.1f}', ha='center', va='bottom', fontsize=fontsize-1.5, fontweight='bold', color='black')

        ax2.set_ylabel('GHG intensity (tCO₂/t)', fontsize=fontsize)
        ax2.set_ylim(0,3)
        #ax2.legend(loc='upper right', frameon=False)

    plt.tight_layout()
    if export_figure:
        plt.savefig(f"figs/{export_figure}.png".format(), dpi = 200, bbox_inches = 'tight')
    plt.show()

def plot_steel_map(import_df, pot_cols, subplot=False, titles = [], division_bubble=2, amount_show_bubble=2.2, non_rounding=False, export_figure=False, 
                   pos_legend_false_subplot = (95.6, -9.5), inc_inset_ax_bubble=1.5, inset_ax=True, custom_autopct=custom_autopct,
                   dict_pos = {'Brazil': (-7.5, 0),
                        'China': (+15, 0),
                        'Germany': (+6, 0),
                        'India': (-7.5, 0),
                        'Iran, Islamic Republic of': (0.5, -19),
                        'Japan': (-7.5, -20),
                        'Korea, Republic of': (+5, +45),
                        'Russian Federation': (+10, -40),
                        'Türkiye': (-12, +19),
                        'United States': (-8, 0)}):
    """
    Plot a world map with inset pie charts showing steel production or potential by country and commodity type.

    Parameters
    ----------
    import_df : pd.DataFrame
        Input DataFrame containing country-level steel production or potential data, with required columns:
        ['location', 'commodity_type', 'latitude', 'longitude'] and at least one of `pot_cols`.
    
    pot_cols : list of str
        List of column names indicating different potential or production metrics to visualize.

    subplot : bool, optional (default: False)
        If True, creates subplots for each column in `pot_cols`. If False, all data is shown on one map.

    titles : list of str, optional (default: [])
        Titles for each subplot/map. If not provided, defaults to formatted column names.

    division_bubble : float, optional (default: 2)
        Scaling factor for the size of pie charts on the map.

    amount_show_bubble : float or list, optional (default: 2.2)
        Threshold above which country names and values are shown. Can be a list (for multiple pot_cols).

    non_rounding : bool or int, optional (default: False)
        If False, label values are rounded to integers. If an integer is provided, it is used as decimal precision.

    export_figure : str or bool, optional (default: False)
        If provided as a string, the plot is saved to `figs/{export_figure}.png`.

    dict_pos : dict, optional
        Dictionary mapping country names to manual text offset positions (y, x) for label placement.

    Returns
    -------
    None
        Displays (and optionally saves) a map or subplots of steel data with pie chart insets per country and total summary.
    """

    df_2 = import_df.copy()
    
    # Filter for primary and secondary steel
    df_2 = df_2[df_2['commodity_type'].isin([i[1] for i in list(dict_types.items() ) if 'steel' in str(i) ])]
    
    # Map colors
    df_2['color'] = df_2['commodity_type'].map(end_product_colors)
    
    # Create figure and axis
    if subplot:
        fig, axs = plt.subplots(nrows=len(pot_cols), ncols=1, figsize=(20*len(pot_cols), 8*len(pot_cols)), subplot_kw={'projection': ccrs.PlateCarree()})
        axs = np.ravel(axs)
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 7), subplot_kw={'projection': ccrs.PlateCarree()})
        axs = [ax]
    
    for j, (ax, pot_col) in enumerate(zip(axs, pot_cols)):
        df_filtered = df_2.sort_values(by=pot_col, ascending=True)
        # to remove very high lat and low lats
        ax.set_extent([-180, 180, -55, 80], crs=ccrs.PlateCarree())
        
        # Add land, ocean, and borders
        old = False
        if old:
            ax.add_feature(cfeature.LAND, facecolor='#f0f0f0')
            ax.add_feature(cfeature.OCEAN, facecolor='#e0e0e0')
            ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.3)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.7, color='gray')
        else:
            # Add high-quality land and ocean features with soft, neutral colors
            ax.add_feature(cfeature.LAND, facecolor='#F4F4F4', edgecolor='none', zorder=-2)  
            ax.add_feature(cfeature.OCEAN, facecolor='#D8E1E8', edgecolor='none', zorder=-2)
            
            # Improve borders and coastlines for clarity
            ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.8, color='gray', alpha=0.7)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='black', alpha=0.8)
            ax.add_feature(cfeature.LAKES, facecolor='#D8E1E8', edgecolor='black', zorder=-2, linewidth=0.5)
        # Aggregate data for country-based pie charts
        df_agg = df_filtered.groupby('location').agg({'latitude': 'mean', 'longitude': 'mean', pot_col: 'sum'}).reset_index()

        for _, row in df_agg.iterrows():
            country_data = df_2[df_2['location'] == row['location']]
            if row[pot_col] > 0:
                grouped_df = country_data.groupby('commodity_type').agg({pot_col: 'sum', 'color': 'first'}).reset_index()
                grouped_df = grouped_df[grouped_df[pot_col] > 0]

                pie_ax = inset_axes(ax, width=row[pot_col]**0.4/division_bubble, height=row[pot_col]**0.4/division_bubble, loc='center',
                                    bbox_to_anchor=(row['longitude'], row['latitude']),
                                    bbox_transform=ax.transData, borderpad=0)
                
                wedges = pie_ax.pie(
                    grouped_df[pot_col], 
                    colors=grouped_df['color'], 
                    startangle=90, 
                    wedgeprops={'linewidth': 0.5, 'edgecolor': 'k'}
                )
                pie_ax.set_aspect('equal')

        # Add text labels for significant values
        for _, row in df_agg.iterrows():
            if isinstance(amount_show_bubble, list):
                if abs(row[pot_col]) > amount_show_bubble[j]:
                    #print(f"'{iso2_to_country(row['location'])}'")
                    cor_y, cor_x = dict_pos.get(iso2_to_country(row['location']), (-8, 0))
    
                    new_name = iso2_to_country(row['location']).replace(", Islamic Republic of",""
                                                                       ).replace(", Province of China",""
                                                                       ).replace(", Republic of",""
                                                                                ).replace("n Federation",""
                                                                                         ).replace("Türkiye","Turkey").replace("Korea, Democratic People's Republic","North-Korea"
                                                                                         )
                    ax.text(row['longitude']-cor_x, row['latitude'] + cor_y, 
                            f"{new_name}: {int(round(row[pot_col], 1))}Mt/a" if non_rounding == False else f"{new_name}: {round(row[pot_col], non_rounding)}Mt/a", 
                            fontsize=12.5, ha='center', va='center', color='black',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))
            else:
                if abs(row[pot_col]) > amount_show_bubble:
                    #print(f"'{iso2_to_country(row['location'])}'")
                    cor_y, cor_x = dict_pos.get(iso2_to_country(row['location']), (-8, 0))
    
                    new_name = iso2_to_country(row['location']).replace(", Islamic Republic of",""
                                                                       ).replace(", Province of China",""
                                                                       ).replace(", Republic of",""
                                                                                ).replace("n Federation",""
                                                                                         ).replace("Türkiye","Turkey").replace("Korea, Democratic People's Republic","North-Korea"
                                                                                         )
                    ax.text(row['longitude']-cor_x, row['latitude'] + cor_y, 
                            f"{new_name}: {int(round(row[pot_col], 1))}Mt/a" if non_rounding == False else f"{new_name}: {round(row[pot_col], non_rounding)}Mt/a", 
                            fontsize=12.5, ha='center', va='center', color='black',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))


        if len(titles) > 0:
            ax.set_title(titles[j], fontsize=22, loc='left')
        else:
            ax.set_title(f"Steel analysis for: {pot_col.replace('_',' ')}", fontsize=22, loc='left')
        ax.spines['geo'].set_visible(True)
        ax.spines['geo'].set_linewidth(1.5)
        ax.spines['geo'].set_color('gray')

        # Add inset pie chart
        grouped_df = df_2.groupby(['commodity_type']).agg({pot_col: 'sum', 'color': 'first'}).reset_index()

        ax_inset = inset_axes(ax, width=2.2 if len(pot_cols) ==1 else 1.9, 
                              height=2.2 if len(pot_cols) ==1 else 1.9, bbox_to_anchor=(0.08, 0.920, 0.065, 0.065), 
                              bbox_transform=ax.transAxes, borderpad=0)

        wedges, texts, autotexts = ax_inset.pie(
            grouped_df[pot_col], 
            colors=grouped_df['color'], 
            autopct=lambda pct: custom_autopct(pct, grouped_df[pot_col]),
            startangle=90, wedgeprops={'linewidth': 1, 'edgecolor': 'black'}, pctdistance=0.65
        )

        plt.setp(autotexts, size=12, weight="bold", color="white")

        legend_patches = [mpatches.Patch(color=color, label=label ) for label, color in end_product_colors.items()]
        rect = patches.Rectangle((0.0, 0.60), 0.14, 0.57, transform=ax.transAxes, facecolor='black', edgecolor='black', alpha=0.1)
        ax.add_patch(rect)

        # ========== EUROPEAN ZOOM INSET ==========
        # European extent coordinates
        # Add zoomed-in inset for Europe
        # Add Europe inset map in bottom-left
        if inset_ax:
            europe_ax = inset_axes(
                ax,
                width="35%", height="48%",  # relative to parent
                loc='lower left',
                bbox_to_anchor=(-0.0893, -0.01, 1, 1),
                bbox_transform=ax.transAxes,
                axes_class=cartopy.mpl.geoaxes.GeoAxes,
                axes_kwargs=dict(projection=ccrs.PlateCarree()),
                #borderpad=1.5
            )

            zoom_extent = [-10, 28, 32, 70] # [lon_min, lon_max, lat_min, lat_max]
            europe_ax.set_extent(zoom_extent, crs=ccrs.PlateCarree())

            # Add high-quality land and ocean features with soft, neutral colors
            europe_ax.add_feature(cfeature.LAND, facecolor='#F4F4F4', edgecolor='none', zorder=-2)  
            europe_ax.add_feature(cfeature.OCEAN, facecolor='#D8E1E8', edgecolor='none', zorder=-2)
            
            # Improve borders and coastlines for clarity
            europe_ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.8, color='gray', alpha=0.7)
            europe_ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='black', alpha=0.8)
            europe_ax.add_feature(cfeature.LAKES, facecolor='#D8E1E8', edgecolor='black', zorder=-2, linewidth=0.5)
            europe_ax.set_title("Europe", fontsize=12, loc='left', weight='bold')

            mark_inset(ax, europe_ax, loc1=2, loc2=4, fc="none", ec='darkgray', linestyle='')
            # Then access the connecting lines and hide them
            for line in ax.patches[-1].get_children():
                if hasattr(line, 'set_visible'):
                    line.set_visible(False)
            
            # Aggregate data for country-based pie charts
            df_agg = df_filtered.groupby('location').agg({'latitude': 'mean', 'longitude': 'mean', pot_col: 'sum'}).reset_index()

            for _, row in df_agg.iterrows():
                country_data = df_2[df_2['location'] == row['location']]
                if zoom_extent[0] <= row['longitude'] <= zoom_extent[1] and zoom_extent[2] <= row['latitude'] <= zoom_extent[3]:  # Filter only Europe
                    if row[pot_col] > 0:
                        grouped_df = country_data.groupby('commodity_type').agg({pot_col: 'sum', 'color': 'first'}).reset_index()
                        grouped_df = grouped_df[grouped_df[pot_col] > 0]

                        pie_ax = inset_axes(europe_ax, width=row[pot_col]**0.4/(division_bubble/inc_inset_ax_bubble), 
                                            height=row[pot_col]**0.4/(division_bubble/inc_inset_ax_bubble), loc='center',
                                            bbox_to_anchor=(row['longitude'], row['latitude']),
                                            bbox_transform=europe_ax.transData, borderpad=0)
                        
                        wedges = pie_ax.pie(
                            grouped_df[pot_col], 
                            colors=grouped_df['color'], 
                            startangle=90, 
                            wedgeprops={'linewidth': 0.5, 'edgecolor': 'k'}
                        )
                        pie_ax.set_aspect('equal')

        ax_inset.set_title('Total', fontsize=12.5, weight='bold', y=0.892)

            # ========== EUROPEAN ZOOM INSET ==========
        if j == (len(pot_cols)-1):
            plt.legend(handles=legend_patches, bbox_to_anchor=pos_legend_false_subplot if subplot==False else (95.6, -9.5) , ncol=3, fontsize=15, frameon=False)


    if subplot:
        plt.subplots_adjust(hspace=0.09)
    if export_figure:
        plt.savefig(f"figs/{export_figure}.png".format(), dpi = 200, bbox_inches = 'tight')

    plt.show()

def plot_cbam_covered_stacked(df, 
                               value_cols=['cbam_included_emissions_export_EU', 'cbam_excluded_emissions_export_EU'], 
                               share_col='share_cbam_covered',
                               max_nr_countries=19, 
                               export_figure=False, 
                               max_y=9):
    
    data = df.copy()

    # Ensure 'country' column exists
    data['country'] = data['location'].apply(iso2_to_country)

    # Group and sum values per country
    grouped = data.groupby('country').agg({
        value_cols[0]: 'sum',
        value_cols[1]: 'sum',
        share_col: 'mean'
    })
    grouped['total'] = grouped[value_cols[0]] + grouped[value_cols[1]]
    grouped = grouped.sort_values('total', ascending=False).head(max_nr_countries)

    x = np.arange(len(grouped))
    width = 0.6

    fig, ax = plt.subplots(figsize=(12, 4))

    bar1 = ax.bar(x, grouped[value_cols[0]], width, label='CBAM-covered', color='darkgreen')
    bar2 = ax.bar(x, grouped[value_cols[1]], width, bottom=grouped[value_cols[0]],
                  label='Non-CBAM-covered', color='darkred')

    ax.set_xticks(x)
    ax.set_xticklabels(grouped.index, rotation=45, ha='right', fontsize=12)
    ax.set_ylabel('Climate impact (MtCO₂)', fontsize=12)
    ax.set_title(r'$\mathbf{b}$ CBAM-covered vs non-covered climate impact of imported steel to EU-27 countries', loc='left', fontsize=15)

    if max_y:
        ax.set_ylim(0, max_y)
    ax.set_xlim(-0.5, max_nr_countries-0.5)

    # Add share labels
    for i, total in enumerate(grouped['total']):
        share = grouped.iloc[i]['cbam_included_emissions_export_EU']/ (grouped.iloc[i]['cbam_included_emissions_export_EU']+
                                            grouped.iloc[i]['cbam_excluded_emissions_export_EU'])
        ax.text(x[i], total * 1.01, f'{share:.0%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.legend(frameon=False, fontsize=12, loc='upper center', ncols=2)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    ax.set_xticklabels(grouped.index, rotation=45, ha='right')

    # === Inset Pie Chart ===
    total_cbam_covered = data[value_cols[0]].sum()
    total_non_cbam = data[value_cols[1]].sum()
    total_sum = total_cbam_covered + total_non_cbam
    pie_values = [total_cbam_covered, total_non_cbam]
    pie_labels = ["",""]# ['CBAM-covered', 'Non-CBAM-covered']

    def autopct_func(pct):
        val = pct * total_sum / 100
        return f'{pct:.0f}%\n({val:.1f} Mt)'

    ax_inset = inset_axes(ax, width=1.5, height=1.5,
                          bbox_to_anchor=(0.91, 0.945, 0.1, 0.1),
                          bbox_transform=ax.transAxes, borderpad=0)

    wedges, texts, autotexts = ax_inset.pie(
        pie_values,
        labels=pie_labels,
        colors=['darkgreen', 'darkred'],
        autopct=autopct_func,
        startangle=90,
        wedgeprops={'linewidth': 1, 'edgecolor': 'black'},
        pctdistance=0.42
    )

    plt.setp(autotexts, size=10, weight="bold", color="white")

    if export_figure:
        plt.savefig(f"figs/{export_figure}.png", dpi=200, bbox_inches='tight')

    plt.show()

def plot_cbam_scope_stacked(results_df_ind, ei_dbs, db_label_map, name_mapping_techs, 
                            name_cbam_true = 'cbam_true', name_cbam_false = 'cbam_false', 
                             output_path="figs/steel_cbam_scope_stacked.png", fontsize_n=13.2):
    """
    Plots stacked bar charts for each steel technology showing:
        - CBAM-covered vs non-covered emissions
        - Scope 1, 2, and 3 breakdown
    
    Parameters:
        results_df_ind (pd.DataFrame): DataFrame containing steel emissions data.
        ei_dbs (list): List of database names to include on the x-axis.
        db_label_map (dict): Mapping from database keys to display names for x-axis.
        name_mapping_techs (dict): Mapping from steel technology identifiers to display names.
        output_path (str): File path to save the figure.
        fontsize_n (float): Base font size for labels and annotations.
    """
    steel_names = np.sort(results_df_ind["name"].unique())
    
    # Layout: 5 plots per row
    cols = 5
    rows = int(np.ceil(len(steel_names) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(14, 5 * rows), sharey=True)
    axs = axs.flatten()

    cbam_colors = ['darkgreen', 'darkred']
    scope_colors = ['darkgrey', 'darkblue', '#b07aa1']
    bar_width = 0.35

    for i, steel in enumerate(steel_names):
        ax = axs[i]
        data = results_df_ind[results_df_ind["name"] == steel]
        x = np.arange(len(ei_dbs))

        for j, db in enumerate(ei_dbs):
            row = data[data["database"] == db]
            if row.empty:
                continue
            row = row.iloc[0]

            # CBAM stacked bar
            ax.bar(x[j] - bar_width / 2, row[name_cbam_true], width=bar_width,
                   color=cbam_colors[0], label="CBAM True" if i == 0 and j == 0 else "")
            ax.bar(x[j] - bar_width / 2, row[name_cbam_false], width=bar_width,
                   bottom=row[name_cbam_true], color=cbam_colors[1],
                   label="CBAM False" if i == 0 and j == 0 else "")

            # CBAM share annotation
            cbam_total = row[name_cbam_true] + row[name_cbam_false]
            cbam_share = row[name_cbam_true] / cbam_total if cbam_total > 0 else 0
            ax.text(
                x[j] - bar_width / 2 + 0.2,
                cbam_total + 0.01,
                f"{cbam_share:.0%}",
                ha="center", va="bottom",
                fontsize=9, fontweight='bold', color="#333333"
            )
            ax.set_ylim(0, 4)

            # Scope stacked bar
            bottom = 0
            for k, (scope, color) in enumerate(zip(["Scope 1", "Scope 2", "Scope 3"], scope_colors)):
                ax.bar(x[j] + bar_width / 2, row[scope], width=bar_width,
                       bottom=bottom, color=color,
                       label=scope if i == 0 and j == 0 else "")
                bottom += row[scope]

        # X-axis and titles
        ax.set_xticks(x)
        ax.set_xticklabels([db_label_map.get(db, db) for db in ei_dbs], rotation=90, fontsize=fontsize_n)
        ax.tick_params(axis='y', labelsize=fontsize_n)

        subplot_label = f"{chr(97 + i)}."
        ax.set_title(rf"$\bf{{{subplot_label}}}$ {name_mapping_techs.get(steel, steel)}",
                     fontsize=fontsize_n - 1, pad=10, loc='left')

        ax.yaxis.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)

    # Global y-axis labels
    fig.text(0.07, 0.28, "kg CO$_2$-eq./kg steel", va='center', rotation='vertical', fontsize=fontsize_n)
    fig.text(0.07, 0.72, "kg CO$_2$-eq./kg steel", va='center', rotation='vertical', fontsize=fontsize_n)

    # Remove unused subplots
    for j in range(len(steel_names), len(axs)):
        fig.delaxes(axs[j])

    # Global legend
    handles, labels = axs[0].get_legend_handles_labels()
    labels = [label.replace('CBAM True', 'CBAM included').replace('CBAM False', 'CBAM excluded') for label in labels]
    fig.legend(handles, labels, loc='upper center', ncol=5, frameon=False,
               fontsize=fontsize_n, bbox_to_anchor=(0.5, 0.95))

    plt.subplots_adjust(hspace=0.29, wspace=0.15)

    # Save and show
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.show()