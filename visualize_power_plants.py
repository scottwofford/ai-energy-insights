import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

try:
    # Read the operating power plants data
    print("Reading operating power plants data...")
    operating_df = pd.read_csv('operating_power_plants.csv')
    
    # Convert operating year to datetime and extract just the year
    operating_df['Operating Year'] = pd.to_datetime(operating_df['Operating Year'], errors='coerce').dt.year
    
    # Group by operating year and sum the capacity
    print("\nGrouping data by year...")
    yearly_capacity = operating_df.groupby('Operating Year')['Nameplate Capacity (MW)'].sum().reset_index()
    
    print("\nSample of yearly capacity data:")
    print(yearly_capacity.head())
    
    # Set the style for better visualization
    plt.style.use('seaborn')
    
    # Create the visualization
    print("\nCreating visualization...")
    plt.figure(figsize=(12, 6))
    
    # Create bar plot
    plt.bar(yearly_capacity['Operating Year'], yearly_capacity['Nameplate Capacity (MW)'])
    
    # Customize the plot
    plt.title('Total Power Plant Capacity by Year', fontsize=14, pad=20)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Total Capacity (MW)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    output_file = 'power_plants_visualization.png'
    print(f"\nSaving visualization to {output_file}...")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Verify the file was created
    if os.path.exists(output_file):
        print(f"Successfully created {output_file}")
        print(f"File size: {os.path.getsize(output_file)} bytes")
    else:
        print("Error: File was not created!")

    # Display some basic statistics
    print("\nBasic Statistics:")
    print(f"Total number of operating plants: {len(operating_df)}")
    print("\nYearly Capacity Distribution:")
    print(yearly_capacity.sort_values('Operating Year'))

except Exception as e:
    print(f"\nError occurred: {str(e)}")
    import traceback
    print("\nFull error traceback:")
    print(traceback.format_exc()) 