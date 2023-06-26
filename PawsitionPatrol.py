import cv2
import os
import numpy as np
import csv
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib import cm

# Load the video
video_path = r"C:\Users\kurtm\OneDrive\Projects\Rat Tracker\Hab 1 KM213.MTS.re-encoded.960px.13061k.avi"
cap = cv2.VideoCapture(video_path)

# extract base file name
base_file_name = os.path.splitext(os.path.basename(video_path))[0]

# for csv file
csv_file_name = base_file_name + "_positions.csv"

# for plot
plot_file_name = base_file_name + "_plot.png"

# Get the frame rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Create the background subtractor object
fgbg = cv2.createBackgroundSubtractorMOG2()

# Prepare CSV output
csv_file = open(csv_file_name, 'w', newline='')
writer = csv.writer(csv_file)
writer.writerow(["Time", "Position", "Zone"])

# Define zones
current_rectangle = []
zones = []
def on_mouse_click(event, x, y, flags, param):
    global current_rectangle

    if event == cv2.EVENT_LBUTTONDOWN:
        current_rectangle = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        current_rectangle.append((x, y))
        zones.append(current_rectangle)
        cv2.rectangle(param, current_rectangle[0], current_rectangle[1], (255, 0, 0), 2)
        cv2.imshow('Image', param)

# Ask user to define zones
print("Please define zones by clicking and dragging in the image.")
cap.set(cv2.CAP_PROP_POS_MSEC, 60000)  # Skip to 60 seconds
ret, frame = cap.read()
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', on_mouse_click, param=frame)
cv2.imshow('Image', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Go back to start

# Convert the list of points into a list of rectangles
zones = [cv2.boundingRect(np.array(zone)) for zone in zones]
zone_times = [0] * len(zones)

# Record rat positions
rat_positions = []

# User input for video playback
show_video = input("Do you want to show video playback? (y/n): ").lower() == 'y'

last_known_zone = None
tracked_frames = 0

while True:
    frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
    current_time = frame_num / fps

    ret, frame = cap.read()
    if not ret:
        break

    fgmask = fgbg.apply(frame)
    _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_zone = None
    center = None  # Initialize center as None

    for contour in contours:
        if cv2.contourArea(contour) > 500:
            (x, y, w, h) = cv2.boundingRect(contour)
            center = (int(x + w/2), int(y + h/2))  # Update center
            rat_positions.append(center)

            for i, (zx, zy, zw, zh) in enumerate(zones):
                if zx < center[0] < zx + zw and zy < center[1] < zy + zh:
                    current_zone = i + 1
                    break

    if current_zone is None and last_known_zone is not None:
        current_zone = last_known_zone
        center = rat_positions[-1] if rat_positions else None
    elif current_zone is not None:
        last_known_zone = current_zone
        tracked_frames += 1
        zone_times[current_zone - 1] += 1

    writer.writerow([current_time, center, last_known_zone])
    print('Time:', current_time, 'Position:', center, 'Zone:', last_known_zone)

    if show_video and center:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if current_zone is not None:
            cv2.putText(frame, "Zone: " + str(current_zone), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    
    if show_video:
        for i, zone in enumerate(zones):
            cv2.rectangle(frame, (zone[0], zone[1]), (zone[0] + zone[2], zone[1] + zone[3]), (255, 0, 0), 2)
            cv2.putText(frame, str(i + 1), (zone[0] + 5, zone[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
csv_file.close()

# Calculate time in seconds and percentages
total_time = tracked_frames / fps
zone_times = [(time / fps) for time in zone_times]
zone_percentages = [(time / total_time * 100) for time in zone_times]

print("\nTime spent in each zone:")
for i, (time, percentage) in enumerate(zip(zone_times, zone_percentages)):
    print(f"Zone {i + 1} : {time} seconds, {percentage}% of total time")

# Plot rat positions
rat_positions = np.array(rat_positions)
plt.figure()
plt.scatter(*zip(*rat_positions), s=1, c=np.linspace(0, 1, len(rat_positions)), cmap='jet')
plt.title("Rat Positions Over Time")
plt.gca().invert_yaxis()  # invert y axis to match the video orientation
plt.show()
plt.savefig(plot_file_name)