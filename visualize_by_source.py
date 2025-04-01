from flask import Flask, render_template
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    # Read both operating and planned power plants data
    operating_df = pd.read_csv('operating_power_plants.csv', low_memory=False)
    
    def clean_dataframe(df, year_col, capacity_col, source_col):
        # Remove rows where year, capacity, or source is missing
        df = df.dropna(subset=[year_col, capacity_col, source_col])
        
        # Convert year to integer, removing any non-numeric values
        df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
        df = df.dropna(subset=[year_col])
        df[year_col] = df[year_col].astype(int)
        
        # Remove years before 2000 or after 2030
        df = df[df[year_col].between(2000, 2030)]
        
        # Convert capacity to numeric, removing any non-numeric values
        df[capacity_col] = pd.to_numeric(df[capacity_col], errors='coerce')
        df = df.dropna(subset=[capacity_col])
        
        # Remove negative or zero capacities
        df = df[df[capacity_col] > 0]
        
        # Remove any unreasonably large capacities (e.g., > 100,000 MW)
        df = df[df[capacity_col] <= 100000]
        
        return df
    
    # Clean the dataset
    operating_df = clean_dataframe(operating_df, 'Operating Year', 'Nameplate Capacity (MW)', 'Primary Energy Source')
    
    # Get top 5 sources and group others
    source_totals = operating_df.groupby('Primary Energy Source')['Nameplate Capacity (MW)'].sum().sort_values(ascending=False)
    top_5_sources = source_totals.head(5).index.tolist()
    
    # Create source categories (top 5 + Other)
    operating_df['Source Category'] = operating_df['Primary Energy Source'].apply(
        lambda x: x if x in top_5_sources else 'Other'
    )
    
    # Group by year and source, calculate yearly capacity
    yearly_by_source = operating_df.groupby(['Operating Year', 'Source Category'])['Nameplate Capacity (MW)'].sum().reset_index()
    
    # Pivot the data for the stacked area chart
    pivot_data = yearly_by_source.pivot(
        index='Operating Year', 
        columns='Source Category', 
        values='Nameplate Capacity (MW)'
    ).fillna(0)
    
    # Calculate cumulative sums for each source
    for col in pivot_data.columns:
        pivot_data[col] = pivot_data[col].cumsum()
    
    # Create the visualization
    fig = go.Figure()
    
    # Color palette for the sources (using a more diverse color scheme)
    colors = {
        'Natural Gas': 'rgba(70, 130, 180, 0.8)',    # Steel Blue
        'Coal': 'rgba(128, 128, 128, 0.8)',          # Gray
        'Nuclear': 'rgba(255, 165, 0, 0.8)',         # Orange
        'Hydroelectric': 'rgba(0, 128, 128, 0.8)',   # Teal
        'Solar': 'rgba(255, 215, 0, 0.8)',           # Gold
        'Wind': 'rgba(46, 139, 87, 0.8)',            # Sea Green
        'Other': 'rgba(169, 169, 169, 0.6)'          # Dark Gray
    }
    
    # Add traces for each energy source in reverse order (for better stacking)
    for source in pivot_data.columns[::-1]:
        fig.add_trace(go.Scatter(
            x=pivot_data.index,
            y=pivot_data[source],
            name=source,
            mode='lines',
            stackgroup='one',
            line=dict(width=0.5),
            fillcolor=colors.get(source, 'rgba(169, 169, 169, 0.6)')  # Default to gray if color not found
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Cumulative Power Plant Capacity by Energy Source',
            font=dict(size=24, color='rgb(30, 30, 30)')
        ),
        xaxis_title='Year',
        yaxis_title='Cumulative Capacity (MW)',
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
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
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor='rgba(255, 255, 255, 0.8)'
        ),
        shapes=[
            # Add light blue shaded background for future years
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
    
    # Convert the plot to HTML
    plot_html = fig.to_html(full_html=False)
    
    return render_template('index.html', plot=plot_html)

if __name__ == '__main__':
    app.run(port=5001, debug=True) 