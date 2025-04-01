from flask import Flask, render_template
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

app = Flask(__name__)

def create_total_capacity_plot(operating_df, planned_df):
    def clean_dataframe(df, year_col, capacity_col):
        # Remove rows where year or capacity is missing
        df = df.dropna(subset=[year_col, capacity_col])
        
        # Convert year to integer, removing any non-numeric values
        df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
        df = df.dropna(subset=[year_col])
        df[year_col] = df[year_col].astype(int)
        
        # Convert capacity to numeric, removing any non-numeric values
        df[capacity_col] = pd.to_numeric(df[capacity_col], errors='coerce')
        df = df.dropna(subset=[capacity_col])
        
        # Remove negative or zero capacities
        df = df[df[capacity_col] > 0]
        
        # Remove any unreasonably large capacities (e.g., > 100,000 MW)
        df = df[df[capacity_col] <= 100000]
        
        return df

    # Clean both datasets but don't filter years yet
    operating_df = clean_dataframe(operating_df, 'Operating Year', 'Nameplate Capacity (MW)')
    planned_df = clean_dataframe(planned_df, 'Planned Operation Year', 'Nameplate Capacity (MW)')
    
    # Calculate pre-2000 capacity
    pre_2000_capacity = operating_df[operating_df['Operating Year'] < 2000]['Nameplate Capacity (MW)'].sum()
    
    # Filter for 2000 onwards
    operating_df = operating_df[operating_df['Operating Year'] >= 2000]
    operating_df = operating_df[operating_df['Operating Year'] <= 2030]
    planned_df = planned_df[planned_df['Planned Operation Year'] >= 2000]
    planned_df = planned_df[planned_df['Planned Operation Year'] <= 2030]
    
    # Group by year and sum the capacity for both datasets
    operating_yearly = operating_df.groupby('Operating Year')['Nameplate Capacity (MW)'].sum().reset_index()
    planned_yearly = planned_df.groupby('Planned Operation Year')['Nameplate Capacity (MW)'].sum().reset_index()
    
    # Add pre-2000 capacity to the first year (2000)
    if 2000 not in operating_yearly['Operating Year'].values:
        new_row = pd.DataFrame({'Operating Year': [2000], 'Nameplate Capacity (MW)': [pre_2000_capacity]})
        operating_yearly = pd.concat([new_row, operating_yearly]).reset_index(drop=True)
    else:
        operating_yearly.loc[operating_yearly['Operating Year'] == 2000, 'Nameplate Capacity (MW)'] += pre_2000_capacity
    
    # Rename columns to match
    operating_yearly = operating_yearly.rename(columns={'Operating Year': 'Year'})
    planned_yearly = planned_yearly.rename(columns={'Planned Operation Year': 'Year'})
    
    # Combine operating and planned capacity
    all_years = pd.DataFrame({'Year': range(2000, 2031)})
    operating_yearly = pd.merge(all_years, operating_yearly, on='Year', how='left').fillna(0)
    planned_yearly = pd.merge(all_years, planned_yearly, on='Year', how='left').fillna(0)
    
    # Calculate total capacity for each year
    total_yearly = pd.DataFrame({
        'Year': operating_yearly['Year'],
        'Nameplate Capacity (MW)': operating_yearly['Nameplate Capacity (MW)'] + planned_yearly['Nameplate Capacity (MW)']
    })
    
    # Calculate cumulative capacity
    total_yearly['Cumulative Capacity (MW)'] = total_yearly['Nameplate Capacity (MW)'].cumsum()
    
    # Calculate growth rate
    total_yearly['Growth Rate'] = (total_yearly['Cumulative Capacity (MW)'].pct_change() * 100).round(1)
    
    # Create the visualization with both MW and Growth Rate traces
    fig = go.Figure()
    
    # MW view trace for total capacity plot
    fig.add_trace(go.Scatter(
        x=total_yearly[total_yearly['Year'] < 2026]['Year'],
        y=total_yearly[total_yearly['Year'] < 2026]['Cumulative Capacity (MW)'],
        mode='lines',
        name='Historical Capacity',
        line=dict(color='rgba(0, 184, 184, 1)', width=2),
        visible=True
    ))

    fig.add_trace(go.Scatter(
        x=total_yearly[total_yearly['Year'] >= 2026]['Year'],
        y=total_yearly[total_yearly['Year'] >= 2026]['Cumulative Capacity (MW)'],
        mode='lines',
        name='Projected Capacity',
        line=dict(color='rgba(0, 184, 184, 1)', width=2, dash='dot'),
        visible=True
    ))
    
    # Growth Rate view trace
    fig.add_trace(go.Scatter(
        x=total_yearly[total_yearly['Year'] < 2026]['Year'],
        y=total_yearly[total_yearly['Year'] < 2026]['Growth Rate'],
        mode='lines',
        name='Historical Growth',
        line=dict(color='rgba(0, 184, 184, 1)', width=2),
        visible=False
    ))

    fig.add_trace(go.Scatter(
        x=total_yearly[total_yearly['Year'] >= 2026]['Year'],
        y=total_yearly[total_yearly['Year'] >= 2026]['Growth Rate'],
        mode='lines',
        name='Projected Growth',
        line=dict(color='rgba(0, 184, 184, 1)', width=2, dash='dot'),
        visible=False
    ))
    
    # Update layout with buttons
    fig.update_layout(
        title=dict(
            text='Total Cumulative Power Plant Capacity',
            font=dict(size=24, color='rgb(30, 30, 30)')
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.7,
                y=1.2,
                showactive=True,
                buttons=[
                    dict(
                        label="Capacity (MW)",
                        method="update",
                        args=[{"visible": [True, True, False, False]},
                              {"yaxis.title.text": "Cumulative Capacity (MW)"}]
                    ),
                    dict(
                        label="Growth Rate (%)",
                        method="update",
                        args=[{"visible": [False, False, True, True]},
                              {"yaxis.title.text": "Annual Growth Rate (%)"}]
                    )
                ]
            )
        ],
        xaxis_title='Year',
        yaxis_title='Cumulative Capacity (MW)',
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=600,
        xaxis=dict(
            tickmode='linear',
            tick0=2000,
            dtick=5,
            range=[2000, 2030],
            gridcolor='rgba(200, 200, 200, 0.5)',
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='rgba(200, 200, 200, 1)',
            ticks='outside'
        ),
        yaxis=dict(
            gridcolor='rgba(200, 200, 200, 0.5)',
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='rgba(200, 200, 200, 1)',
            ticks='outside'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255, 255, 255, 0.8)'
        ),
        margin=dict(t=150, b=100),  # Increased top margin for buttons
        shapes=[
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=2024,
                x1=2030,
                y0=0,
                y1=1,
                fillcolor="rgba(240, 248, 255, 0.5)",
                layer="below",
                line_width=0,
            )
        ]
    )
    
    return fig

def create_source_capacity_plot(operating_df, planned_df):
    def clean_dataframe(df, year_col, capacity_col, source_col):
        # Remove rows where year, capacity, or source is missing
        df = df.dropna(subset=[year_col, capacity_col, source_col])
        
        # Convert year to integer, removing any non-numeric values
        df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
        df = df.dropna(subset=[year_col])
        df[year_col] = df[year_col].astype(int)
        
        # Convert capacity to numeric, removing any non-numeric values
        df[capacity_col] = pd.to_numeric(df[capacity_col], errors='coerce')
        df = df.dropna(subset=[capacity_col])
        
        # Remove negative or zero capacities
        df = df[df[capacity_col] > 0]
        
        # Remove any unreasonably large capacities (e.g., > 100,000 MW)
        df = df[df[capacity_col] <= 100000]
        
        return df
    
    def calculate_growth_rates(pivot_data):
        # Calculate year-over-year growth rate of cumulative capacity
        growth_rates = (pivot_data.pct_change() * 100).round(1)
        return growth_rates

    # Clean both datasets but don't filter years yet
    operating_df = clean_dataframe(operating_df, 'Operating Year', 'Nameplate Capacity (MW)', 'Energy Source Code')
    planned_df = clean_dataframe(planned_df, 'Planned Operation Year', 'Nameplate Capacity (MW)', 'Energy Source Code')
    
    # Get total capacity by source for both operating and planned
    operating_totals = operating_df.groupby('Energy Source Code')['Nameplate Capacity (MW)'].sum()
    planned_totals = planned_df.groupby('Energy Source Code')['Nameplate Capacity (MW)'].sum()
    combined_totals = operating_totals.add(planned_totals, fill_value=0).sort_values(ascending=False)
    
    # Get top 5 sources based on combined capacity
    top_5_sources = combined_totals.head(5).index.tolist()
    
    # Create source categories (top 5 + Other) for both datasets
    source_name_mapping = {
        'MWH': 'Batteries',
        'SUN': 'Solar',
        'WND': 'Wind',
        'NG': 'Natural Gas',
        'SUB': 'Coal'
    }
    
    operating_df['Source Category'] = operating_df['Energy Source Code'].apply(
        lambda x: source_name_mapping.get(x, x) if x in top_5_sources else 'Other'
    )
    
    # Calculate pre-2000 capacity by source
    pre_2000_by_source = operating_df[operating_df['Operating Year'] < 2000].groupby('Source Category')['Nameplate Capacity (MW)'].sum()
    
    # Filter for 2000 onwards
    operating_df = operating_df[operating_df['Operating Year'] >= 2000]
    operating_df = operating_df[operating_df['Operating Year'] <= 2030]
    
    planned_df['Source Category'] = planned_df['Energy Source Code'].apply(
        lambda x: source_name_mapping.get(x, x) if x in top_5_sources else 'Other'
    )
    planned_df = planned_df[planned_df['Planned Operation Year'].between(2000, 2030)]
    
    # Combine operating and planned data
    operating_df['Type'] = 'Operating'
    planned_df['Type'] = 'Planned'
    operating_df = operating_df.rename(columns={'Operating Year': 'Year'})
    planned_df = planned_df.rename(columns={'Planned Operation Year': 'Year'})
    
    # Concatenate the dataframes
    combined_df = pd.concat([operating_df, planned_df])
    
    # Group by year and source, calculate yearly capacity
    yearly_by_source = combined_df.groupby(['Year', 'Source Category'])['Nameplate Capacity (MW)'].sum().reset_index()
    
    # Add pre-2000 capacity to the year 2000
    for source in pre_2000_by_source.index:
        source_2000 = yearly_by_source[(yearly_by_source['Year'] == 2000) & (yearly_by_source['Source Category'] == source)]
        if len(source_2000) == 0:
            # If no entry exists for 2000, create one
            new_row = pd.DataFrame({
                'Year': [2000],
                'Source Category': [source],
                'Nameplate Capacity (MW)': [pre_2000_by_source[source]]
            })
            yearly_by_source = pd.concat([yearly_by_source, new_row], ignore_index=True)
        else:
            # If entry exists, add the pre-2000 capacity
            yearly_by_source.loc[
                (yearly_by_source['Year'] == 2000) & (yearly_by_source['Source Category'] == source),
                'Nameplate Capacity (MW)'
            ] += pre_2000_by_source[source]
    
    # Pivot the data for the stacked area chart
    pivot_data = yearly_by_source.pivot(
        index='Year', 
        columns='Source Category', 
        values='Nameplate Capacity (MW)'
    ).fillna(0)
    
    # Calculate cumulative sums
    cumulative_data = pivot_data.copy()
    for col in cumulative_data.columns:
        cumulative_data[col] = cumulative_data[col].cumsum()
    
    # Calculate growth rates
    growth_rates = calculate_growth_rates(cumulative_data)
    
    # Create the visualization
    fig = go.Figure()
    
    # Color palette for the sources (matching the example image style)
    colors = {
        'Natural Gas': 'rgb(158, 202, 225)',      # Light blue for Natural Gas
        'Coal': 'rgb(70, 70, 70)',                # Dark gray for Coal
        'NUC': 'rgb(253, 141, 60)',              # Orange for Nuclear
        'WAT': 'rgb(49, 130, 189)',              # Dark blue for Hydroelectric
        'Wind': 'rgb(49, 163, 84)',              # Green for Wind
        'Solar': 'rgb(255, 215, 0)',             # Gold for Solar
        'Batteries': 'rgb(140, 86, 75)',         # Brown for Batteries
        'Other': 'rgb(180, 180, 180)'            # Light gray for Other
    }
    
    # Add MW view traces for source capacity plot
    for source in cumulative_data.columns[::-1]:
        color = colors.get(source, colors['Other'])
        rgb_values = [int(x.strip()) for x in color.replace('rgb(', '').replace(')', '').split(',')]
        fill_color = f'rgba({rgb_values[0]}, {rgb_values[1]}, {rgb_values[2]}, 0.8)'
        
        # Historical data (before 2026)
        fig.add_trace(go.Scatter(
            x=cumulative_data[cumulative_data.index < 2026].index,
            y=cumulative_data[cumulative_data.index < 2026][source],
            name=source,
            mode='lines',
            stackgroup='one',
            line=dict(width=0.5, color=color),
            fillcolor=fill_color,
            hovertemplate="%{y:,.0f} MW<extra></extra>",
            visible=True,
            showlegend=True
        ))
        
        # Projected data (2026 onwards)
        fig.add_trace(go.Scatter(
            x=cumulative_data[cumulative_data.index >= 2026].index,
            y=cumulative_data[cumulative_data.index >= 2026][source],
            name=source + " (Projected)",
            mode='lines',
            stackgroup='one',
            line=dict(width=0.5, color=color, dash='dot'),
            fillcolor=fill_color,
            hovertemplate="%{y:,.0f} MW<extra></extra>",
            visible=True,
            showlegend=False
        ))
    
    # Add Growth Rate view traces
    for source in growth_rates.columns[::-1]:
        color = colors.get(source, colors['Other'])
        
        # Historical growth rates (before 2026)
        fig.add_trace(go.Scatter(
            x=growth_rates[growth_rates.index < 2026].index,
            y=growth_rates[growth_rates.index < 2026][source],
            name=f"{source} Growth",
            mode='lines',
            line=dict(width=2, color=color),
            hovertemplate="%{y:.1f}%<extra></extra>",
            visible=False,
            showlegend=True
        ))
        
        # Projected growth rates (2026 onwards)
        fig.add_trace(go.Scatter(
            x=growth_rates[growth_rates.index >= 2026].index,
            y=growth_rates[growth_rates.index >= 2026][source],
            name=f"{source} Growth (Projected)",
            mode='lines',
            line=dict(width=2, color=color, dash='dot'),
            hovertemplate="%{y:.1f}%<extra></extra>",
            visible=False,
            showlegend=False
        ))
    
    # Update layout with buttons for source capacity plot
    fig.update_layout(
        title=dict(
            text='Cumulative Power Plant Capacity by Energy Source',
            font=dict(size=24, color='rgb(30, 30, 30)')
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.7,
                y=1.2,
                showactive=True,
                buttons=[
                    dict(
                        label="Capacity (MW)",
                        method="update",
                        args=[{"visible": [True, True] * len(cumulative_data.columns) + [False, False] * len(growth_rates.columns)},
                              {"yaxis.title.text": "Cumulative Capacity (MW)"}]
                    ),
                    dict(
                        label="Growth Rate (%)",
                        method="update",
                        args=[{"visible": [False, False] * len(cumulative_data.columns) + [True, True] * len(growth_rates.columns)},
                              {"yaxis.title.text": "Annual Growth Rate (%)"}]
                    )
                ]
            )
        ],
        xaxis_title='Year',
        yaxis_title='Cumulative Capacity (MW)',
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=600,
        xaxis=dict(
            tickmode='linear',
            tick0=2000,
            dtick=5,
            range=[2000, 2030],
            gridcolor='rgba(200, 200, 200, 0.5)',
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='rgba(200, 200, 200, 1)',
            ticks='outside'
        ),
        yaxis=dict(
            gridcolor='rgba(200, 200, 200, 0.5)',
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='rgba(200, 200, 200, 1)',
            ticks='outside',
            hoverformat=",.0f"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255, 255, 255, 0.8)'
        ),
        margin=dict(t=150, b=100),  # Increased top margin for buttons
        hovermode='x unified'
    )
    
    return fig

def create_installed_vs_queues_plot():
    # Data for 2010
    data_2010_installed = {
        'Hydro': 100,
        'Other': 50,
        'Coal': 300,
        'Nuclear': 100,
        'Gas': 400,
        'Wind': 22
    }
    
    data_2010_queues = {
        'Other': 50,
        'Gas': 150,
        'Wind': 200,
        'Solar': 62
    }
    
    # Data for 2023
    data_2023_installed = {
        'Hydro': 100,
        'Other': 50,
        'Coal': 200,
        'Nuclear': 100,
        'Gas': 800,
        'Wind': 29
    }
    
    data_2023_queues = {
        'Other': 50,
        'Gas': 100,
        'Wind': 400,
        'Storage': 500,
        'Storage (Hybrid)': 500,
        'Solar': 548,
        'Solar (Hybrid)': 500
    }
    
    # Create the visualization
    fig = go.Figure()
    
    # Color mapping
    colors = {
        'Hydro': 'rgb(100, 149, 237)',      # Light blue
        'Other': 'rgb(165, 42, 42)',        # Brown
        'Coal': 'rgb(70, 70, 70)',          # Dark gray
        'Nuclear': 'rgb(253, 141, 60)',     # Orange
        'Gas': 'rgb(158, 202, 225)',        # Light blue
        'Wind': 'rgb(49, 163, 84)',         # Green
        'Solar': 'rgb(255, 215, 0)',        # Gold
        'Storage': 'rgb(135, 206, 235)',    # Sky blue
        'Storage (Hybrid)': 'rgb(65, 105, 225)',  # Royal blue
        'Solar (Hybrid)': 'rgb(218, 165, 32)',    # Golden rod
        'Offshore Wind': 'rgb(144, 238, 144)'     # Light green
    }

    # Define the order of sources for stacking
    installed_order = ['Hydro', 'Other', 'Coal', 'Nuclear', 'Gas', 'Wind']
    queues_order_2010 = ['Other', 'Gas', 'Wind', 'Solar']
    queues_order_2023 = ['Other', 'Gas', 'Wind', 'Storage', 'Storage (Hybrid)', 'Solar', 'Solar (Hybrid)']

    # 2010 Installed Capacity (stacked)
    y_cumsum = 0
    for source in installed_order:
        if source in data_2010_installed:
            capacity = data_2010_installed[source]
            fig.add_trace(go.Bar(
                x=['2010 Installed'],
                y=[capacity],
                name=source,
                marker_color=colors.get(source, 'gray'),
                showlegend=True,
                xaxis='x1',
                yaxis='y1'
            ))
    
    # 2010 Queues (stacked)
    for source in queues_order_2010:
        if source in data_2010_queues:
            capacity = data_2010_queues[source]
            fig.add_trace(go.Bar(
                x=['2010 Queues'],
                y=[capacity],
                name=source,
                marker_color=colors.get(source, 'gray'),
                showlegend=False,
                xaxis='x1',
                yaxis='y1'
            ))

    # 2023 Installed Capacity (stacked)
    for source in installed_order:
        if source in data_2023_installed:
            capacity = data_2023_installed[source]
            fig.add_trace(go.Bar(
                x=['2023 Installed'],
                y=[capacity],
                name=source,
                marker_color=colors.get(source, 'gray'),
                showlegend=False,
                xaxis='x2',
                yaxis='y2'
            ))
    
    # 2023 Queues (stacked)
    for source in queues_order_2023:
        if source in data_2023_queues:
            capacity = data_2023_queues[source]
            fig.add_trace(go.Bar(
                x=['2023 Queues'],
                y=[capacity],
                name=source,
                marker_color=colors.get(source, 'gray'),
                showlegend=False,
                xaxis='x2',
                yaxis='y2'
            ))
    
    # Update layout
    fig.update_layout(
        title=dict(text='Installed Capacity and Queues Comparison', font=dict(size=24)),
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=600,
        barmode='stack',
        grid=dict(
            rows=1,
            columns=2,
            pattern='independent'
        ),
        annotations=[
            dict(
                x=0.25,
                y=1.05,
                xref='paper',
                yref='paper',
                text='2010',
                showarrow=False,
                font=dict(size=20)
            ),
            dict(
                x=0.75,
                y=1.05,
                xref='paper',
                yref='paper',
                text='2023',
                showarrow=False,
                font=dict(size=20)
            )
        ],
        yaxis1=dict(
            title='Capacity (GW)',
            range=[0, 2600],
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.5)',
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='rgba(200, 200, 200, 1)'
        ),
        yaxis2=dict(
            title='Capacity (GW)',
            range=[0, 2600],
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.5)',
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='rgba(200, 200, 200, 1)'
        ),
        xaxis1=dict(
            showgrid=False,
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='rgba(200, 200, 200, 1)'
        ),
        xaxis2=dict(
            showgrid=False,
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='rgba(200, 200, 200, 1)'
        ),
        bargap=0.15,
        margin=dict(t=150, b=100, l=50, r=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255, 255, 255, 0.8)'
        )
    )
    
    return fig

def create_generation_by_source_plot(generation_df):
    def clean_generation_data(df):
        # Convert year to integer
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df = df.dropna(subset=['year'])
        df['year'] = df['year'].astype(int)
        
        # Convert generation columns to numeric
        generation_columns = ['coal', 'natural gas', 'nuclear', 'renewables', 'petroleum and other']
        for col in generation_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    # Clean the data
    generation_df = clean_generation_data(generation_df)
    
    # Calculate percent change for each source
    pct_change_df = generation_df.copy()
    for col in ['coal', 'natural gas', 'nuclear', 'renewables', 'petroleum and other']:
        pct_change_df[col] = generation_df[col].pct_change() * 100
    
    # Create the visualization
    fig = go.Figure()
    
    # Color palette matching the other charts
    colors = {
        'natural gas': 'rgb(158, 202, 225)',  # Light blue
        'coal': 'rgb(70, 70, 70)',           # Dark gray
        'nuclear': 'rgb(253, 141, 60)',      # Orange
        'renewables': 'rgb(49, 163, 84)',    # Green
        'petroleum and other': 'rgb(180, 180, 180)'  # Light gray
    }
    
    # Add absolute value traces for each source
    for source in ['coal', 'natural gas', 'nuclear', 'renewables', 'petroleum and other']:
        color = colors[source]
        rgb_values = [int(x.strip()) for x in color.replace('rgb(', '').replace(')', '').split(',')]
        fill_color = f'rgba({rgb_values[0]}, {rgb_values[1]}, {rgb_values[2]}, 0.8)'
        
        # Capitalize source name for display
        display_name = ' '.join(word.capitalize() for word in source.split())
        
        # Add absolute value trace
        fig.add_trace(go.Scatter(
            x=generation_df['year'],
            y=generation_df[source],
            name=display_name,
            mode='lines',
            stackgroup='one',
            line=dict(width=0.5, color=color),
            fillcolor=fill_color,
            hovertemplate="%{y:,.0f} GWh<extra></extra>",
            visible=True
        ))
        
        # Add percent change trace
        fig.add_trace(go.Scatter(
            x=pct_change_df['year'],
            y=pct_change_df[source],
            name=display_name,
            mode='lines',
            line=dict(width=2, color=color),
            hovertemplate="%{y:.1f}%<extra></extra>",
            visible=False
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Annual Electricity Generation by Source',
            font=dict(size=24, color='rgb(30, 30, 30)')
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.7,
                y=1.2,
                showactive=True,
                buttons=[
                    dict(
                        label="Absolute Values",
                        method="update",
                        args=[{"visible": [True, False] * 5},
                              {"yaxis.title.text": "Generation (GWh)"}]
                    ),
                    dict(
                        label="Percent Change",
                        method="update",
                        args=[{"visible": [False, True] * 5},
                              {"yaxis.title.text": "Annual Change (%)"}]
                    )
                ]
            )
        ],
        xaxis_title='Year',
        yaxis_title='Generation (GWh)',
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=600,
        xaxis=dict(
            tickmode='linear',
            tick0=2000,
            dtick=5,
            range=[2000, 2030],
            gridcolor='rgba(200, 200, 200, 0.5)',
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='rgba(200, 200, 200, 1)',
            ticks='outside'
        ),
        yaxis=dict(
            gridcolor='rgba(200, 200, 200, 0.5)',
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='rgba(200, 200, 200, 1)',
            ticks='outside',
            hoverformat=",.0f"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255, 255, 255, 0.8)'
        ),
        margin=dict(t=150, b=100),
        hovermode='x unified'
    )
    
    return fig

def create_total_generation_plot(generation_df):
    def clean_generation_data(df):
        # Convert year to integer
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df = df.dropna(subset=['year'])
        df['year'] = df['year'].astype(int)
        
        # Convert generation columns to numeric and calculate total
        generation_columns = ['coal', 'natural gas', 'nuclear', 'renewables', 'petroleum and other']
        for col in generation_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate total generation per year
        df['total_generation'] = df[generation_columns].sum(axis=1)
        
        return df
    
    # Clean the data
    generation_df = clean_generation_data(generation_df)
    
    # Calculate percent change
    generation_df['pct_change'] = generation_df['total_generation'].pct_change() * 100
    
    # Get the last actual year's total generation
    last_actual_year = generation_df['year'].max()
    last_total_generation = generation_df.loc[generation_df['year'] == last_actual_year, 'total_generation'].iloc[0]
    
    # Calculate initial data center demand (4% of total)
    initial_dc_demand = last_total_generation * 0.04
    base_generation = last_total_generation - initial_dc_demand  # Non-DC demand
    
    # Create projection years
    projection_years = range(last_actual_year + 1, 2029)
    
    # Calculate projections for different growth rates
    growth_scenarios = {
        '12% Growth': 0.12,
        '20% Growth': 0.20,
        '30% Growth': 0.30
    }
    
    # Create the visualization
    fig = go.Figure()
    
    # Add historical trace
    fig.add_trace(go.Scatter(
        x=generation_df['year'],
        y=generation_df['total_generation'],
        name='Historical Generation',
        mode='lines',
        line=dict(width=2, color='rgb(0, 184, 184)'),
        hovertemplate="%{y:,.0f} GWh<extra></extra>",
        visible=True
    ))
    
    # Add projection traces for each growth scenario
    colors = {
        '12% Growth': 'rgb(255, 127, 14)',  # Orange
        '20% Growth': 'rgb(44, 160, 44)',   # Green
        '30% Growth': 'rgb(214, 39, 40)'    # Red
    }
    
    # Store growth rates for each scenario
    growth_rate_data = []
    
    for scenario, growth_rate in growth_scenarios.items():
        projection_data = []
        dc_demand = initial_dc_demand
        prev_total = last_total_generation
        
        for year in projection_years:
            # Grow data center demand by the scenario rate
            dc_demand *= (1 + growth_rate)
            # Add to base generation (which stays constant)
            total_generation = base_generation + dc_demand
            projection_data.append((year, total_generation))
            
            # Calculate year-over-year growth rate
            growth_rate_val = ((total_generation - prev_total) / prev_total) * 100
            growth_rate_data.append({
                'year': year,
                'growth_rate': growth_rate_val,
                'scenario': scenario
            })
            prev_total = total_generation
        
        # Add projection trace for absolute values
        fig.add_trace(go.Scatter(
            x=[x[0] for x in projection_data],
            y=[x[1] for x in projection_data],
            name=f'Projection ({scenario})',
            mode='lines',
            line=dict(
                width=2,
                color=colors[scenario],
                dash='dot'
            ),
            hovertemplate="%{y:,.0f} GWh<extra></extra>",
            visible=True
        ))
    
    # Add historical percent change trace
    fig.add_trace(go.Scatter(
        x=generation_df['year'],
        y=generation_df['pct_change'],
        name='Historical Growth Rate',
        mode='lines',
        line=dict(width=2, color='rgb(0, 184, 184)'),
        hovertemplate="%{y:.1f}%<extra></extra>",
        visible=False
    ))
    
    # Add projection traces for growth rates
    growth_rate_df = pd.DataFrame(growth_rate_data)
    for scenario in growth_scenarios.keys():
        scenario_data = growth_rate_df[growth_rate_df['scenario'] == scenario]
        fig.add_trace(go.Scatter(
            x=scenario_data['year'],
            y=scenario_data['growth_rate'],
            name=f'Projected Growth ({scenario})',
            mode='lines',
            line=dict(
                width=2,
                color=colors[scenario],
                dash='dot'
            ),
            hovertemplate="%{y:.1f}%<extra></extra>",
            visible=False
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Total Annual Electricity Generation<br><sup>Projections show different data center growth scenarios from 4% base</sup>',
            font=dict(size=24, color='rgb(30, 30, 30)')
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.7,
                y=1.1,
                showactive=True,
                buttons=[
                    dict(
                        label="Absolute Values",
                        method="update",
                        args=[{"visible": [True, True, True, True, False, False, False, False]},
                              {
                                "yaxis.title.text": "Generation (GWh)",
                                "yaxis.range": [3000, None]
                              }]
                    ),
                    dict(
                        label="Percent Change",
                        method="update",
                        args=[{"visible": [False, False, False, False, True, True, True, True]},
                              {
                                "yaxis.title.text": "Annual Growth Rate (%)",
                                "yaxis.range": [None, None]
                              }]
                    )
                ]
            )
        ],
        xaxis_title=dict(
            text='Year',
            standoff=20  # Add some space between axis and title
        ),
        yaxis_title='Generation (GWh)',
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=600,
        xaxis=dict(
            tickmode='linear',
            tick0=2000,
            dtick=5,
            range=[2000, 2030],
            gridcolor='rgba(200, 200, 200, 0.5)',
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='rgba(200, 200, 200, 1)',
            ticks='outside'
        ),
        yaxis=dict(
            range=[3000, None],
            gridcolor='rgba(200, 200, 200, 0.5)',
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='rgba(200, 200, 200, 1)',
            ticks='outside',
            hoverformat=",.0f"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,  # Moved down from -0.2
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255, 255, 255, 0.8)'
        ),
        margin=dict(t=150, b=120),  # Increased bottom margin from 100 to 120
        hovermode='x unified'
    )
    
    return fig

@app.route('/')
def index():
    # Read data files
    operating_df = pd.read_csv('operating_power_plants.csv', low_memory=False)
    planned_df = pd.read_csv('planned_power_plants.csv', low_memory=False)
    
    # Create required plots
    total_capacity_plot = create_total_capacity_plot(operating_df.copy(), planned_df.copy())
    source_capacity_plot = create_source_capacity_plot(operating_df.copy(), planned_df.copy())
    installed_vs_queues_plot = create_installed_vs_queues_plot()
    
    # Try to create generation plots if data is available
    generation_html = ""
    total_generation_html = ""
    try:
        generation_df = pd.read_csv('/Users/maria/Documents/code-work/ai-energy-insights/generation-major-source.csv', low_memory=False)
        generation_plot = create_generation_by_source_plot(generation_df.copy())
        total_generation_plot = create_total_generation_plot(generation_df.copy())
        generation_html = generation_plot.to_html(full_html=False)
        total_generation_html = total_generation_plot.to_html(full_html=False)
    except FileNotFoundError:
        print("Warning: generation-major-source.csv not found. Skipping generation plots.")
    
    # Convert plots to HTML
    total_capacity_html = total_capacity_plot.to_html(full_html=False)
    source_capacity_html = source_capacity_plot.to_html(full_html=False)
    installed_vs_queues_html = installed_vs_queues_plot.to_html(full_html=False)
    
    return render_template('index.html', 
                         plot1=total_capacity_html, 
                         plot2=source_capacity_html,
                         plot3=installed_vs_queues_html,
                         plot4=generation_html,
                         plot5=total_generation_html)

if __name__ == '__main__':
    app.run(debug=True) 