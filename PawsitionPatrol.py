import cv2  #OpenCV is used for image and video analysis  #Import OpenCV, a powerful library for image processing and computer vision
import os  #os module is used to interact with the operating system  #This module provides a portable way of using operating system dependent functionality
import numpy as np  #NumPy is used for numerical operations  #NumPy is a general-purpose array-processing package
import csv  #csv module is used to handle CSV files  #This module implements classes to read and write tabular data in CSV format
from scipy import stats  #scipy.stats is used for statistical analysis  #SciPy is a scientific computation library that builds on NumPy
import matplotlib.pyplot as plt  #Matplotlib is used for generating plots  #Matplotlib is a plotting library
import tkinter as tk  #tkinter is used for creating GUIs
from tkinter import filedialog  #filedialog is used for opening the file dialog
from matplotlib.colors import TwoSlopeNorm  #TwoSlopeNorm is used for creating a colormap
from matplotlib import cm  #cm is used for handling colormaps

def select_video_file():
    #Create a root Tkinter window and hide it
    root = tk.Tk()
    root.withdraw()

    #Prompt the user to select a video file
    print("Please select a video file.")
    video_path = filedialog.askopenfilename()
    cap = cv2.VideoCapture(video_path)  #Create a VideoCapture object

    #Extract the base file name
    base_file_name = os.path.splitext(os.path.basename(video_path))[0]

    return cap, video_path, base_file_name

#Prompt the user to select a video file
cap, video_path, base_file_name = select_video_file()

#Create a new directory for the outputs
directory = base_file_name
os.makedirs(directory, exist_ok=True)

#The names of the output files
csv_file_name = os.path.join(directory, base_file_name + "_positions.csv")  #The name of the CSV output file
time_series_plot_file_name = os.path.join(directory, base_file_name + "_time_series_plot.png")  #The name of the time series plot image file
scatter_plot_file_name = os.path.join(directory, base_file_name + "_scatter_plot.png")  #The name of the scatter plot image file
heatmap_file_name = os.path.join(directory, base_file_name + "_heatmap.png")  #The name of the heatmap image file

#Get the frame rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)  #Get the frames per second
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  #Get the total number of frames

#Create the background subtractor object
fgbg = cv2.createBackgroundSubtractorMOG2()  #This is used to subtract the background from each frame

#Prepare CSV output
csv_file = open(csv_file_name, 'w', newline='')  #Open the CSV output file
writer = csv.writer(csv_file)  #Create a CSV writer object
writer.writerow(["Time", "Position X", "Position Y", "Zone"])  #Write the header row

#Define zones
current_rectangle = []  #The current rectangle being drawn by the user
zones = []  #The list of zones
heatmap = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))))  #The heatmap of rat positions

def on_mouse_click(event, x, y, flags, param):
    global current_rectangle, zones, frame_copy

    if event == cv2.EVENT_LBUTTONDOWN:  #If the left button is pressed
        current_rectangle = [(x, y)]  #Start a new rectangle
    elif event == cv2.EVENT_LBUTTONUP:  #If the left button is released
        current_rectangle.append((x, y))  #Finish the rectangle
        zones.append(current_rectangle)  #Add the rectangle to the list of zones
        frame_copy = frame.copy()  #Create a copy of the original frame
        for zone in zones:  #For each zone
            cv2.rectangle(frame_copy, zone[0], zone[1], (255, 0, 0), 2)  #Draw the zone on the copy of the frame
        cv2.imshow('Image', frame_copy)  #Show the copy of the frame
    elif event == cv2.EVENT_RBUTTONDOWN:  #If the right button is pressed
        if zones:  #If there are zones
            zones.pop()  #Remove the last zone
            frame_copy = frame.copy()  #Create a copy of the original frame
            for zone in zones:  #For each remaining zone
                cv2.rectangle(frame_copy, zone[0], zone[1], (255, 0, 0), 2)  #Draw the zone on the copy of the frame
            cv2.imshow('Image', frame_copy)  #Show the copy of the frame

print("\nInstructions:")
print("1. Left click and drag to draw a zone.")
print("2. Release the left click to finalize the zone.")
print("3. Right click to remove the most recently added zone.")
print("4. Press any key to proceed after you have finished defining zones.\n")

#Ask user to define zones
print("Please define zones by clicking and dragging in the image.")
cap.set(cv2.CAP_PROP_POS_MSEC, 60000)  #Skip to 60 seconds
ret, frame = cap.read()
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', on_mouse_click, param=frame)
cv2.imshow('Image', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  #Go back to start

#Allow user to decide the first frame
print("Press 'n' for next 1 second, 'p' for previous 1 second, 's' to start analysis.")
frame_jump = int(1 * fps)  #Set the jump to 1 second worth of frames
frame_index = 0
while True:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)  #Move to the selected frame
    ret, frame = cap.read()
    if not ret:
        print("Video ended before starting analysis. Please try again.")
        exit(1)
    
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(0)
    if key == ord('s'):
        break
    elif key == ord('n'):  #Next 1 second
        frame_index += frame_jump
    elif key == ord('p'):  #Previous 1 second
        frame_index = max(0, frame_index - frame_jump)  #Ensure frame_index doesn't go below 0
    else:
        print("Invalid key. Press 'n' for next 1 second, 'p' for previous 1 second, 's' to start analysis.")
cv2.destroyAllWindows()

#Ask the user for the sensitivity of the rat detection
sensitivity = input("Please enter the sensitivity for rat detection (default is 500): ")
if sensitivity == "":
    sensitivity = 500
else:
    try:
        sensitivity = int(sensitivity)
    except ValueError:
        print("Invalid input. Using default sensitivity of 500.")
        sensitivity = 500

#Convert the list of points into a list of rectangles
zones = [cv2.boundingRect(np.array(zone)) for zone in zones]
zone_times = [0] * len(zones)

#Record rat positions
rat_positions = []

#Ask the user if they want to show the video playback
show_video = input("Do you want to show video playback? (y/n): ").lower() == 'y'

#The last known zone the rat was in
last_known_zone = None
tracked_frames = 0

while True:  #Start an infinite loop
    frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)  #Get the current frame number
    current_time = frame_num / fps  #Calculate the current time

    ret, frame = cap.read()  #Read the next frame
    if not ret:  #If there was an error
        break  #Exit the loop

    fgmask = fgbg.apply(frame)  #Subtract the background from the frame
    _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)  #Threshold the image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  #Find the contours

    current_zone = None  #The current zone the rat is in
    center = None  #The center of the rat

    for contour in contours:  #For each contour
        if cv2.contourArea(contour) > sensitivity:  #If the area of the contour is greater than 500
            (x, y, w, h) = cv2.boundingRect(contour)  #Get the bounding rectangle of the contour
            center = (int(x + w/2), int(y + h/2))  #Calculate the center of the rat
            rat_positions.append(center)  #Add the position to the list of positions

            #Update the heatmap
            heatmap[center[1], center[0]] += 1

            zone_areas = []  #list to store areas of zones the rat is in
            for i, (zx, zy, zw, zh) in enumerate(zones):  #For each zone
                if zx < center[0] < zx + zw and zy < center[1] < zy + zh:  #If the rat is in the zone
                    zone_areas.append((i+1, zw*zh))  #store the zone index and its area

            if zone_areas:  #if there are any zones the rat is in
                zone_areas.sort(key=lambda x: x[1])  #sort by area
                current_zone = zone_areas[0][0]  #the current zone is the smallest one

    if current_zone is None and last_known_zone is not None:  #If the rat is not in any zone but was in a zone in the last frame
        current_zone = last_known_zone  #The rat is still in the last known zone
        center = rat_positions[-1] if rat_positions else None  #Get the last known position of the rat
    elif current_zone is not None:  #If the rat is in a zone
        last_known_zone = current_zone  #Update the last known zone
        tracked_frames += 1  #Increase the number of tracked frames by 1
        zone_times[current_zone - 1] += 1  #Increase the time spent in the current zone by 1

    if center is not None:
        writer.writerow([current_time, center[0] if center else None, center[1] if center else None, last_known_zone])
    else:
        writer.writerow([current_time, None, None, last_known_zone])  #Write the current time, position, and zone to the CSV file
    print('Time:', current_time, 'Position:', center, 'Zone:', last_known_zone)  #Print the current time, position, and zone

    if show_video and center:  #If the user wants to see the video playback and the center of the rat has been found
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  #Draw a rectangle around the rat
        if current_zone is not None:  #If the rat is in a zone
            cv2.putText(frame, f"Zone: {current_zone}, Coordinates: {center}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)  #Draw the current zone and coordinates on the frame

    if show_video:  #If the user wants to see the video playback
        for i, zone in enumerate(zones):  #For each zone
            cv2.rectangle(frame, (zone[0], zone[1]), (zone[0] + zone[2], zone[1] + zone[3]), (255, 0, 0), 2)  #Draw the zone
            cv2.putText(frame, str(i + 1), (zone[0] + 5, zone[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)  #Draw the zone number
        cv2.imshow('Frame', frame)  #Show the frame

        if cv2.waitKey(1) & 0xFF == ord('q'):  #If the user presses the 'q' key
            break  #Exit the loop

cap.release()  #Release the video capture object
csv_file.close()  #Close the CSV file

#Calculate time in seconds and percentages
total_time = tracked_frames / fps
zone_times = [(time / fps) for time in zone_times]
zone_percentages = [(time / total_time * 100) for time in zone_times]

print("\nTime spent in each zone:")
for i, (time, percentage) in enumerate(zip(zone_times, zone_percentages)):  #For each zone
    print(f"Zone {i + 1} : {time} seconds, {percentage}% of total time")  #Print the time and percentage

#Time series plot of rat positions
plt.figure(figsize=(10, 6))  #Create a new figure
plt.plot(range(len(rat_positions)), [p[0] for p in rat_positions], label='X Coordinate')  #Draw a line plot of the X coordinate over time
plt.plot(range(len(rat_positions)), [p[1] for p in rat_positions], label='Y Coordinate')  #Draw a line plot of the Y coordinate over time
plt.title("Rat Positions Over Time")
plt.legend(["X Coordinate", "Y Coordinate"])  #Set the title of the plot
plt.xlabel("Time (frames)")
plt.ylabel("Position (pixels)")
plt.legend()
plt.savefig(time_series_plot_file_name)  #Save the time series plot to a file
plt.show()  #Show the plot

#Plot the rat positions
plt.figure(figsize=(10, 6))  #Create a new figure
plt.scatter(*zip(*rat_positions), s=1, c=np.linspace(0, 1, len(rat_positions)), cmap='jet')  #Draw a scatter plot of the rat positions
plt.title("Rat Positions Scatter Plot")
plt.legend(["Position"])  #Set the title of the plot
plt.xlabel("X Coordinate (pixels)")
plt.ylabel("Y Coordinate (pixels)")
plt.gca().invert_yaxis()  #Invert the y axis
plt.legend(["Rat Positions"])
plt.savefig(scatter_plot_file_name)  #Save the scatter plot to a file
plt.show()  #Show the plot

#Show the heatmap
plt.figure(figsize=(5, 3))  #Create a new figure
plt.imshow(heatmap, cmap='viridis', interpolation='nearest')  #Draw the heatmap
plt.title("Heatmap of Rat Positions")  #Set the title of the plot
plt.xlabel("X Coordinate (pixels)")
plt.ylabel("Y Coordinate (pixels)")
plt.gca().invert_yaxis()  #Invert the y axis
plt.colorbar(label='Frequency')  #Add a colorbar
plt.savefig(heatmap_file_name)  #Save the heatmap to a file
plt.show()  #Show the plot