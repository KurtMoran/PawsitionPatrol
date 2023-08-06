#Import required libraries
import cv2
import os
import numpy as np
import csv
import tkinter as tk
from tkinter import filedialog

#Create PawsitionPatrol class
class PawsitionPatrol:
    def __init__(self, root_dir):
        #Initialize instance variables
        self.root_dir = root_dir
        self.zones = []  #List to store defined zones
        self.rat_positions = []  #List to store positions of the detected hamster
        self.last_known_zone = None  #Store the last known zone of the hamster
        self.sensitivity = 500  #Sensitivity for hamster detection
        self.show_video = False  #Boolean to determine if the video playback should be shown
        self.video_path = None  #Path to the video file
        self.cap = None  #VideoCapture object
        self.base_file_name = None  #Base name of the video file
        self.fgbg = cv2.createBackgroundSubtractorMOG2()  #Background subtractor object
        self.fps = None  #Frames per second of the video
        self.csv_file = None  #CSV file to save the hamster's positions and corresponding times
        self.writer = None  #CSV writer
        self.frame_index = 0  #Index of the current frame
        self.frame_jump = 0  #Number of frames to jump for next or previous second of the video

    #Method to select video file
    def select_video_file(self):
        #Open a dialog box to select the video file
        root = tk.Tk()
        root.withdraw()
        print("Please select a video file.")
        self.video_path = filedialog.askopenfilename()
        #Initialize the VideoCapture object and other related instance variables
        self.cap = cv2.VideoCapture(self.video_path)
        self.base_file_name = os.path.splitext(os.path.basename(self.video_path))[0]
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_jump = int(1 * self.fps)
        #Create a directory to save the output and initialize the CSV file and writer
        directory = os.path.join(self.root_dir, self.base_file_name)
        os.makedirs(directory, exist_ok=True)
        csv_file_name = os.path.join(directory, self.base_file_name + "_positions.csv")
        self.csv_file = open(csv_file_name, 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(["Time", "Position X", "Position Y", "Zone"])

    #Method to handle mouse clicks
    def on_mouse_click(self, event, x, y, flags, param):
        #If left button down, start a new zone
        if event == cv2.EVENT_LBUTTONDOWN:
            self.zones.append([(x, y)])
        #If left button up, finalize the current zone
        elif event == cv2.EVENT_LBUTTONUP:
            self.zones[-1].append((x, y))
            self.update_frame_copy()
        #If right button down and there are defined zones, remove the last zone
        elif event == cv2.EVENT_RBUTTONDOWN and self.zones:
            self.zones.pop()
            self.update_frame_copy()

    #Method to update the frame with drawn zones
    def update_frame_copy(self):
        #Copy the current frame and draw the defined zones
        frame_copy = self.frame.copy()
        for zone in self.zones:
            cv2.rectangle(frame_copy, zone[0], zone[1], (255, 0, 0), 2)
        cv2.imshow('Image', frame_copy)

    #Method to define zones
    def define_zones(self):
        #Instructions for defining zones
        print("\\nInstructions:")
        print("1. Left click and drag to draw a zone.")
        print("2. Release the left click to finalize the zone.")
        print("3. Right click to remove the most recently added zone.")
        print("4. Press any key to proceed after you have finished defining zones.\\n")
        print("Please define zones by clicking and dragging in the image.")
        #Set the frame position to 1 minute and display the frame for defining zones
        self.cap.set(cv2.CAP_PROP_POS_MSEC, 60000)
        ret, self.frame = self.cap.read()
        cv2.namedWindow('Image')
        cv2.setMouseCallback('Image', self.on_mouse_click, param=self.frame)
        cv2.imshow('Image', self.frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #Reset the frame position and finalize the defined zones
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.zones = [cv2.boundingRect(np.array(zone)) for zone in self.zones]

    #Method to set sensitivity
    def set_sensitivity(self):
        #Get the sensitivity from the user and validate the input
        sensitivity = input("Please enter the sensitivity for hamster detection (default is 500): ")
        if sensitivity:
            try:
                self.sensitivity = int(sensitivity)
            except ValueError:
                print("Invalid input. Using default sensitivity of 500.")

    #Method to set show_video
    def set_show_video(self):
        #Get the input from the user if they want to show the video playback
        self.show_video = input("Do you want to show video playback? (y/n): ").lower() == 'y'

    #Method to start the analysis
    def run(self):
        #Instructions for navigating the video
        print("Press 'n' for next 1 second, 'p' for previous 1 second, 's' to start analysis.")
        #Loop until 's' is pressed
        while True:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index)
            ret, frame = self.cap.read()
            if not ret:
                print("Video ended before starting analysis. Please try again.")
                exit(1)
            cv2.imshow('Frame', frame)
            key = cv2.waitKey(0)
            if key == ord('s'):
                break
            elif key == ord('n'):
                self.frame_index += self.frame_jump
            elif key == ord('p'):
                self.frame_index = max(0, self.frame_index - self.frame_jump)
            else:
                print("Invalid key. Press 'n' for next 1 second, 'p' for previous 1 second, 's' start analysis.")
        cv2.destroyAllWindows()

        #Loop until the end of the video
        while True:
            #Get the current frame number and time
            frame_num = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            current_time = frame_num / self.fps
            ret, frame = self.cap.read()
            if not ret:
                break
            #Apply background subtraction and thresholding
            fgmask = self.fgbg.apply(frame)
            _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            '''
            1. We have a contour of interest.
            2. Around this shape, draw the smallest possible rectangle.
            3. Find the midpoint of this rectangle; this is our center point.

            4. To determine if the center point is inside a designated area (zone):
            - Each zone has defined boundaries: left, right, top, and bottom edges.
            - The x-coordinate of the center should be between the left and right edges.
            - The y-coordinate of the center should be between the top and bottom edges.
            - If both conditions are met, the center is inside that zone.

            5. If the center point is inside multiple overlapping zones, identify all such zones.
            6. Out of these zones, choose the one with the smallest area as the main zone.
            7. If no shape is identified, we don't have a main zone.
            '''
            #Initialize current zone and center
            current_zone = None
            center = None
            # Find the largest contour with area greater than sensitivity
            largest_contour = max((contour for contour in contours if cv2.contourArea(contour) > self.sensitivity), key=cv2.contourArea, default=None)
            # If a valid largest contour is found, perform the operations
            if largest_contour is not None:
                (x, y, w, h) = cv2.boundingRect(largest_contour)
                center = (int(x + w/2), int(y + h/2))
                self.rat_positions.append(center)
                # Check if the center is inside any zone
                zone_areas = [(i+1, zw*zh) for i, (zx, zy, zw, zh) in enumerate(self.zones) if zx < center[0] < zx + zw and zy < center[1] < zy + zh]
                if zone_areas:
                    zone_areas.sort(key=lambda x: x[1])
                    current_zone = zone_areas[0][0]
            else:
                current_zone = None
            # If no zone is found, use the last known zone and the last known position
            if current_zone is None and self.last_known_zone is not None:
                current_zone = self.last_known_zone
                center = self.rat_positions[-1] if self.rat_positions else None
            # If a zone is found, update the last known zone
            elif current_zone is not None:
                self.last_known_zone = current_zone
            #Write the current time, position and zone to the CSV file
            self.writer.writerow([current_time, center[0] if center else None, center[1] if center else None, self.last_known_zone])
            print('Time:', current_time, 'Position:', center, 'Zone:', self.last_known_zone)
            #If show_video is true and a center is found, draw the bounding box and the zone and coordinates
            if self.show_video and center:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if current_zone is not None:
                    cv2.putText(frame, f"Zone: {current_zone}, Coordinates: {center}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            #If show_video is true, show the thresholded frame and the original frame with zones
            if self.show_video:
                cv2.imshow('Thresholded Frame', thresh)
                for i, zone in enumerate(self.zones):
                    cv2.rectangle(frame, (zone[0], zone[1]), (zone[0] + zone[2], zone[1] + zone[3]), (255, 0, 0), 2)
                    cv2.putText(frame, str(i + 1), (zone[0] + 5, zone[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
                cv2.imshow('Frame', frame)
                #Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        #Release the VideoCapture object and destroy all windows
        self.cap.release()
        cv2.destroyAllWindows()
        #Close the CSV file
        self.csv_file.close()

def main():
    try:
        #Create a PawsitionPatrol object and run the analysis
        pawsition_patrol = PawsitionPatrol('./output')
        pawsition_patrol.select_video_file()
        pawsition_patrol.define_zones()
        pawsition_patrol.set_sensitivity()
        pawsition_patrol.set_show_video()
        pawsition_patrol.run()
    except Exception as e:
        #If an error occurs, print the error and wait for a user input to end
        print("An error occurred:")
        print(e)
        input("Please enter to end")

if __name__ == "__main__":
    main()