import os
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import matplotlib.animation as animation
import matplotlib.colors as mcolors

class PawsitionPatrol:
    def __init__(self, file_path):
        self.file_path = file_path
        self.subject = self.extract_subject_from_filename()
        self.data = self.load_data()
        self.data_clean = self.clean_data()
        self.seconds_per_zone, self.cumulative_time, self.entry_exit_times, self.zone_latency = self.analyze_data()

    def extract_subject_from_filename(self):
        file_name = os.path.basename(self.file_path)
        return 'KM' + file_name.split('KM')[1].split('.')[0]

    def load_data(self):
        try:
            return pd.read_csv(self.file_path)
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def clean_data(self):
        return self.data.dropna(subset=['Position X', 'Position Y'])

    def analyze_data(self):
        data_sorted = self.data_clean.sort_values('Time')
        data_sorted['Time Difference'] = data_sorted['Time'].diff()
        seconds_per_zone = data_sorted.groupby('Zone')['Time Difference'].sum()
        cumulative_time = data_sorted.groupby('Zone')['Time Difference'].cumsum()
        zone_changes = data_sorted['Zone'].ne(data_sorted['Zone'].shift())
        entry_exit_times = data_sorted.loc[zone_changes, ['Time', 'Zone']]
        zone_latency = self.calculate_zone_latency(entry_exit_times)
        return seconds_per_zone, cumulative_time, entry_exit_times, zone_latency

    def calculate_zone_latency(self, entry_exit_times):
        zone_latency = entry_exit_times.copy()
        zone_latency['Subject'] = self.subject
        zone_latency.columns = ['Zone Change Time', 'Entering Zone', 'Subject']
        zone_latency['Exiting Zone'] = zone_latency['Entering Zone'].shift()
        zone_latency = zone_latency[['Subject', 'Zone Change Time', 'Entering Zone', 'Exiting Zone']]
        return zone_latency
    
    def write_zone_latency_to_csv(self):
        output_dir = os.path.dirname(self.file_path)
        output_file_path = os.path.join(output_dir, self.subject + '_Zone_Latency.csv')
        self.zone_latency.to_csv(output_file_path, index=False)

    def write_zone_times_to_csv(self):
        table_data = pd.DataFrame({'Zone': self.seconds_per_zone.index, 'Total Time': self.seconds_per_zone.values.round(3)})
        table_data.to_csv(os.path.join(os.path.dirname(self.file_path), self.subject + '_Zone_Times.csv'), index=False)

    def plot_data(self):
        zones = sorted(self.data_clean['Zone'].dropna().unique())
        num_zones = len(zones)
        cmap = plt.get_cmap('viridis')
        colors = {zone: cmap(i / num_zones) for i, zone in enumerate(zones)}

        self.scatter_plot_positions_over_time(colors)
        self.heatmap_of_positions()
        self.table_and_bar_plot_seconds_per_zone()
        self.plot_cumulative_time_per_zone()
        self.timeline_of_zone_entries(colors)
        self.animate_positions_over_time()

    def scatter_plot_positions_over_time(self, colors):
        plt.figure(figsize=(10, 8))
        plt.scatter(self.data_clean['Position X'], self.data_clean['Position Y'], c=self.data_clean['Time'], cmap='viridis', alpha=0.7)
        plt.colorbar(label='Time')
        plt.xlabel('Position X')
        plt.ylabel('Position Y')
        plt.title('Path of the hamster in the maze over time')
        plt.gca().invert_yaxis()  
        plt.grid(True)

    def heatmap_of_positions(self):
        plt.figure(figsize=(10, 8))
        plt.hist2d(self.data_clean['Position X'], self.data_clean['Position Y'], bins=[50,50], cmap='coolwarm')
        plt.colorbar(label='Frequency')
        plt.xlabel('Position X')
        plt.ylabel('Position Y')
        plt.title("Heatmap of the hamster's positions in the maze")
        plt.gca().invert_yaxis()
        plt.grid(True)

    def table_and_bar_plot_seconds_per_zone(self):
        fig, ax = plt.subplots(figsize=(8, 4))
        table_data = pd.DataFrame({'Zone': self.seconds_per_zone.index, 'Seconds': self.seconds_per_zone.values})
        table_data['Seconds'] = table_data['Seconds'].round(3)
        table = ax.table(cellText=table_data.values, colLabels=table_data.columns, cellLoc='center', loc='center',
                        colWidths=[0.15, 0.15], cellColours=[[mcolors.CSS4_COLORS['lightsteelblue']]*2]*len(table_data))
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)
        table.auto_set_column_width([0, 1])
        ax.axis('off')

        print("Sum of Seconds in Each Zone:")
        print(table_data.to_string(index=False))

        plt.figure(figsize=(10, 6))
        self.seconds_per_zone.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.xlabel('Zone')
        plt.ylabel('Seconds')
        plt.title('Sum of Seconds Spent in Each Zone')
        plt.grid(axis='y')

    def plot_cumulative_time_per_zone(self):
        plt.figure(figsize=(10, 6))
        for zone in self.seconds_per_zone.index:
            zone_cumulative_time = self.cumulative_time[self.data_clean['Zone'] == zone]
            plt.plot(self.data_clean.loc[self.data_clean['Zone'] == zone, 'Time'], zone_cumulative_time, marker='o', linestyle='-', linewidth=2, markersize=5,
                    label='Zone {}'.format(zone))
        plt.xlabel('Time')
        plt.ylabel('Cumulative Time (Seconds)')
        plt.title('Cumulative Time in Each Zone')
        plt.grid(True)
        plt.legend()

    def timeline_of_zone_entries(self, colors):
        plt.figure(figsize=(15, 6))
        for i in range(len(self.entry_exit_times) - 1):
            if pd.notna(self.entry_exit_times.iloc[i, 1]):
                plt.hlines(y=self.entry_exit_times.iloc[i, 1], xmin=self.entry_exit_times.iloc[i, 0], xmax=self.entry_exit_times.iloc[i+1, 0], 
                        colors=colors[self.entry_exit_times.iloc[i, 1]], linestyles='solid', linewidth=15)
        plt.xlabel('Time')
        plt.ylabel('Zone')
        plt.yticks(self.seconds_per_zone.index)
        plt.title('Timeline of Zone Entries')
        plt.grid(True)

    def animate_positions_over_time(self):
        fig = plt.figure()
        ax = plt.axes(xlim=(0, max(self.data_clean['Position X'])), ylim=(0, max(self.data_clean['Position Y'])))

        scatter, = ax.plot([], [], 'o')

        def init():
            scatter.set_data([], [])
            return scatter,

        def animate(i):
            x = self.data_clean['Position X'].iloc[:i+1]
            y = max(self.data_clean['Position Y']) - self.data_clean['Position Y'].iloc[:i+1]
            scatter.set_data(x, y)
            return scatter,

        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(self.data_clean), interval=2, blit=True)

        plt.show()

    def run(self):
        if self.data is not None:
            self.write_zone_latency_to_csv()
            self.write_zone_times_to_csv()
            self.plot_data()

def select_file():
    root = tk.Tk()
    root.withdraw() 
    return filedialog.askopenfilename(title="Select CSV file", filetypes=(("CSV files", "*.csv"),))

def main():
    file_path = select_file()
    if file_path:
        patrol = PawsitionPatrol(file_path)
        patrol.run()
    else:
        print("No file selected.")

if __name__ == "__main__":
    main()