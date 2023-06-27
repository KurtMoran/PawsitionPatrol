import cv2  # Import OpenCV, a powerful library for image processing and computer vision
import os  # This module provides a portable way of using operating system dependent functionality
import numpy as np  # NumPy is a general-purpose array-processing package
import csv  # This module implements classes to read and write tabular data in CSV format
from scipy import stats  # SciPy is a scientific computation library that builds on NumPy
import matplotlib.pyplot as plt  # Matplotlib is a plotting library
import tkinter as tk
from tkinter import filedialog
from matplotlib.colors import TwoSlopeNorm
from matplotlib import cm

# Create a root Tkinter window and hide it
root = tk.Tk()
root.withdraw()

# Open the file dialog
video_path = filedialog.askopenfilename()
cap = cv2.VideoCapture(video_path)  # Create a VideoCapture object

# Extract the base file name
base_file_name = os.path.splitext(os.path.basename(video_path))[0]

# The names of the output files
csv_file_name = base_file_name + "_positions.csv"  # The name of the CSV output file
plot_file_name = base_file_name + "_plot.png"  # The name of the plot image file
heatmap_file_name = base_file_name + "_heatmap.png"  # The name of the heatmap image file

# Get the frame rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frames per second
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get the total number of frames

# Create the background subtractor object
fgbg = cv2.createBackgroundSubtractorMOG2()  # This is used to subtract the background from each frame

# Prepare CSV output
csv_file = open(csv_file_name, 'w', newline='')  # Open the CSV output file
writer = csv.writer(csv_file)  # Create a CSV writer object
writer.writerow(["Time", "Position", "Zone"])  # Write the header row

# Define zones
current_rectangle = []  # The current rectangle being drawn by the user
zones = []  # The list of zones
heatmap = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))))  # The heatmap of rat positions

# The mouse click event handler
def on_mouse_click(event, x, y, flags, param):
    global current_rectangle

    if event == cv2.EVENT_LBUTTONDOWN:  # If the left button is pressed
        current_rectangle = [(x, y)]  # Start a new rectangle
    elif event == cv2.EVENT_LBUTTONUP:  # If the left button is released
        current_rectangle.append((x, y))  # Finish the rectangle
        zones.append(current_rectangle)  # Add the rectangle to the list of zones
        cv2.rectangle(param, current_rectangle[0], current_rectangle[1], (255, 0, 0), 2)  # Draw the rectangle on the image
        cv2.imshow('Image', param)  # Show the image

# Ask the user to define the zones
print("Please define zones by clicking and dragging in the image.")
cap.set(cv2.CAP_PROP_POS_MSEC, 60000)  # Skip to 60 seconds
ret, frame = cap.read()  # Read the next frame
cv2.namedWindow('Image')  # Create a window
cv2.setMouseCallback('Image', on_mouse_click, param=frame)  # Set the mouse click event handler
cv2.imshow('Image', frame)  # Show the image
cv2.waitKey(0)  # Wait for the user to press a key
cv2.destroyAllWindows()  # Close all windows
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Go back to the start of the video

# Convert the list of points into a list of rectangles
zones = [cv2.boundingRect(np.array(zone)) for zone in zones]  # This converts each pair of points into a rectangle
zone_times = [0] * len(zones)  # Initialize the time spent in each zone to 0

# The list of rat positions
rat_positions = []

# Ask the user if they want to show the video playback
show_video = input("Do you want to show video playback? (y/n): ").lower() == 'y'

# The last known zone the rat was in
last_known_zone = None
tracked_frames = 0

while True:  # Start an infinite loop
    frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)  # Get the current frame number
    current_time = frame_num / fps  # Calculate the current time

    ret, frame = cap.read()  # Read the next frame
    if not ret:  # If there was an error
        break  # Exit the loop

    fgmask = fgbg.apply(frame)  # Subtract the background from the frame
    _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)  # Threshold the image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find the contours

    current_zone = None  # The current zone the rat is in
    center = None  # The center of the rat

    for contour in contours:  # For each contour
        if cv2.contourArea(contour) > 500:  # If the area of the contour is greater than 500
            (x, y, w, h) = cv2.boundingRect(contour)  # Get the bounding rectangle of the contour
            center = (int(x + w/2), int(y + h/2))  # Calculate the center of the rat
            rat_positions.append(center)  # Add the position to the list of positions

            # Update the heatmap
            heatmap[center[1], center[0]] += 1

            for i, (zx, zy, zw, zh) in enumerate(zones):  # For each zone
                if zx < center[0] < zx + zw and zy < center[1] < zy + zh:  # If the rat is in the zone
                    current_zone = i + 1  # The rat is in this zone
                    break  # Exit the loop

    if current_zone is None and last_known_zone is not None:  # If the rat is not in any zone but was in a zone in the last frame
        current_zone = last_known_zone  # The rat is still in the last known zone
        center = rat_positions[-1] if rat_positions else None  # Get the last known position of the rat
    elif current_zone is not None:  # If the rat is in a zone
        last_known_zone = current_zone  # Update the last known zone
        tracked_frames += 1  # Increase the number of tracked frames by 1
        zone_times[current_zone - 1] += 1  # Increase the time spent in the current zone by 1

    writer.writerow([current_time, center, last_known_zone])  # Write the current time, position, and zone to the CSV file
    print('Time:', current_time, 'Position:', center, 'Zone:', last_known_zone)  # Print the current time, position, and zone

    if show_video and center:  # If the user wants to see the video playback and the center of the rat has been found
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw a rectangle around the rat
        if current_zone is not None:  # If the rat is in a zone
            cv2.putText(frame, "Zone: " + str(current_zone), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)  # Draw the current zone on the frame

    if show_video:  # If the user wants to see the video playback
        for i, zone in enumerate(zones):  # For each zone
            cv2.rectangle(frame, (zone[0], zone[1]), (zone[0] + zone[2], zone[1] + zone[3]), (255, 0, 0), 2)  # Draw the zone
            cv2.putText(frame, str(i + 1), (zone[0] + 5, zone[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)  # Draw the zone number
        cv2.imshow('Frame', frame)  # Show the frame

        if cv2.waitKey(1) & 0xFF == ord('q'):  # If the user presses the 'q' key
            break  # Exit the loop

cap.release()  # Release the video capture object
csv_file.close()  # Close the CSV file

# Calculate the time spent in each zone and the percentage of time spent in each zone
total_time = tracked_frames / fps  # Calculate the total time
zone_times = [(time / fps) for time in zone_times]  # Calculate the time spent in each zone
zone_percentages = [(time / total_time * 100) for time in zone_times]  # Calculate the percentage of time spent in each zone

print("\nTime spent in each zone:")
for i, (time, percentage) in enumerate(zip(zone_times, zone_percentages)):  # For each zone
    print(f"Zone {i + 1} : {time} seconds, {percentage}% of total time")  # Print the time and percentage

# Plot the rat positions
rat_positions = np.array(rat_positions)  # Convert the list of rat positions to a NumPy array
plt.figure()  # Create a new figure
plt.scatter(*zip(*rat_positions), s=1, c=np.linspace(0, 1, len(rat_positions)), cmap='jet')  # Draw a scatter plot of the rat positions
plt.title("Rat Positions Over Time")  # Set the title of the plot
plt.gca().invert_yaxis()  # Invert the y axis
plt.savefig(plot_file_name)  # Save the plot to a file
plt.show()  # Show the plot

# Show the heatmap
plt.figure()  # Create a new figure
plt.imshow(heatmap, cmap='hot', interpolation='nearest')  # Draw the heatmap
plt.title("Heatmap of Rat Positions")  # Set the title of the plot
plt.gca().invert_yaxis()  # Invert the y axis
plt.colorbar()  # Add a colorbar
plt.savefig(heatmap_file_name)  # Save the heatmap to a file
plt.show()  # Show the plot