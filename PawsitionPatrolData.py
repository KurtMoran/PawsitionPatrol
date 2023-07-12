import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
import tkinter as tk
from tkinter import filedialog
import matplotlib.animation as animation
import matplotlib.colors as mcolors

def load_data(file_path):
    # Load the CSV file
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    # Remove rows with missing position data
    data_clean = data.dropna(subset=['Position X', 'Position Y'])
    return data_clean

def analyze_data(data):
    # Calculate the time interval between each row
    data_sorted = data.sort_values('Time')
    data_sorted['Time Difference'] = data_sorted['Time'].diff()

    # Sum of seconds per zone
    seconds_per_zone = data_sorted.groupby('Zone')['Time Difference'].sum()

    # Cumulative time in each zone
    cumulative_time = data_sorted.groupby('Zone')['Time Difference'].cumsum()

    # Identify zone entry and exit times
    zone_changes = data_sorted['Zone'].ne(data_sorted['Zone'].shift())
    entry_exit_times = data_sorted.loc[zone_changes, ['Time', 'Zone', 'Position X', 'Position Y']]

    return seconds_per_zone, cumulative_time, entry_exit_times

def plot_data(data, seconds_per_zone, cumulative_time, entry_exit_times):
    # Define colors for each zone
    zones = sorted(data['Zone'].dropna().unique())
    num_zones = len(zones)
    cmap = plt.get_cmap('viridis')
    colors = {zone: cmap(i / num_zones) for i, zone in enumerate(zones)}

    # Scatter plot of positions over time
    plt.figure(figsize=(10, 8))
    plt.scatter(data['Position X'], data['Position Y'], c=data['Time'], cmap='viridis', alpha=0.7)
    
    # Markers for zone entries
    for i in range(len(entry_exit_times)):
        plt.scatter(entry_exit_times.iloc[i]['Position X'], entry_exit_times.iloc[i]['Position Y'], 
                    c='black', s=100, alpha=0.8)
    
    plt.colorbar(label='Time')
    plt.xlabel('Position X')
    plt.ylabel('Position Y')
    plt.title('Path of the rat in the maze over time')
    plt.gca().invert_yaxis()  # Invert the y-axis
    plt.grid(True)

    # Heatmap of positions
    plt.figure(figsize=(10, 8))
    plt.hist2d(data['Position X'], data['Position Y'], bins=[50,50], cmap='coolwarm')
    plt.colorbar(label='Frequency')
    plt.xlabel('Position X')
    plt.ylabel('Position Y')
    plt.title("Heatmap of the rat's positions in the maze")
    plt.gca().invert_yaxis()  # Invert the y-axis
    plt.grid(True)

    # Table of sum of seconds per zone
    fig, ax = plt.subplots(figsize=(8, 4))
    table_data = pd.DataFrame({'Zone': seconds_per_zone.index, 'Seconds': seconds_per_zone.values})
    table_data['Seconds'] = table_data['Seconds'].round(3)  # Round the 'Seconds' column to 3 decimal places
    table = ax.table(cellText=table_data.values, colLabels=table_data.columns, cellLoc='center', loc='center',
                    colWidths=[0.15, 0.15], cellColours=[[mcolors.CSS4_COLORS['lightsteelblue']]*2]*len(table_data))
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    table.auto_set_column_width([0, 1])
    ax.axis('off')

    # Output table data in text form to the terminal
    print("Sum of Seconds in Each Zone:")
    print(table_data.to_string(index=False))

    # Bar plot for sum of seconds per zone
    plt.figure(figsize=(10, 6))
    seconds_per_zone.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.xlabel('Zone')
    plt.ylabel('Seconds')
    plt.title('Sum of Seconds Spent in Each Zone')
    plt.grid(axis='y')

    # Cumulative time in each zone
    plt.figure(figsize=(10, 6))
    for zone in seconds_per_zone.index:
        zone_cumulative_time = cumulative_time[data['Zone'] == zone]
        plt.plot(data.loc[data['Zone'] == zone, 'Time'], zone_cumulative_time, marker='o', linestyle='-', linewidth=2, markersize=5,
                 label='Zone {}'.format(zone))
    plt.xlabel('Time')
    plt.ylabel('Cumulative Time (Seconds)')
    plt.title('Cumulative Time in Each Zone')
    plt.grid(True)
    plt.legend()

    # Timeline of zone entries
    plt.figure(figsize=(15, 6))
    for i in range(len(entry_exit_times) - 1):
        if pd.notna(entry_exit_times.iloc[i, 1]):
            plt.hlines(y=entry_exit_times.iloc[i, 1], xmin=entry_exit_times.iloc[i, 0], xmax=entry_exit_times.iloc[i+1, 0], 
                       colors=colors[entry_exit_times.iloc[i, 1]], linestyles='solid', linewidth=15)
    plt.xlabel('Time')
    plt.ylabel('Zone')
    plt.yticks(zones)
    plt.title('Timeline of Zone Entries')
    plt.grid(True)

    # Show all plots
    plt.show()

    # Animation of positions leaving a trace over time
    fig = plt.figure()
    ax = plt.axes(xlim=(0, max(data['Position X'])), ylim=(0, max(data['Position Y'])))

    scatter, = ax.plot([], [], 'o')

    def init():
        scatter.set_data([], [])
        return scatter,

    def animate(i):
        x = data['Position X'].iloc[:i+1]
        y = max(data['Position Y']) - data['Position Y'].iloc[:i+1]  # Invert the y-coordinates
        scatter.set_data(x, y)
        return scatter,

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(data), interval=2, blit=True)

    plt.show()

# Create a Tkinter root window
root = tk.Tk()
root.withdraw()  # Hide the root window

# Prompt the user to select the CSV file
file_path = filedialog.askopenfilename(title="Select CSV file", filetypes=(("CSV files", "*.csv"),))

# Check if a file was selected
if file_path:
    # Load and clean the data
    data = load_data(file_path)
    data_clean = clean_data(data)

    # Analyze the data
    seconds_per_zone, cumulative_time, entry_exit_times = analyze_data(data_clean)

    # Plot the data
    plot_data(data_clean, seconds_per_zone, cumulative_time, entry_exit_times)
else:
    print("No file selected.")