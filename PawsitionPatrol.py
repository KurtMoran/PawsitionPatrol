import cv2
import os
import numpy as np
import csv
import tkinter as tk
from tkinter import filedialog

class PawsitionPatrol:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.zones = []
        self.rat_positions = []
        self.last_known_zone = None
        self.tracked_frames = 0
        self.sensitivity = 500
        self.show_video = False
        self.video_path = None
        self.cap = None
        self.base_file_name = None
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.frame_count = None
        self.fps = None
        self.heatmap = None
        self.csv_file = None
        self.writer = None
        self.frame_index = 0
        self.frame_jump = 0

    def select_video_file(self):
        root = tk.Tk()
        root.withdraw()
        print("Please select a video file.")
        self.video_path = filedialog.askopenfilename()
        self.cap = cv2.VideoCapture(self.video_path)
        self.base_file_name = os.path.splitext(os.path.basename(self.video_path))[0]
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.heatmap = np.zeros((int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))))
        self.frame_jump = int(1 * self.fps)
        directory = os.path.join(self.root_dir, self.base_file_name)
        os.makedirs(directory, exist_ok=True)
        csv_file_name = os.path.join(directory, self.base_file_name + "_positions.csv")
        self.csv_file = open(csv_file_name, 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(["Time", "Position X", "Position Y", "Zone"])

    def on_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.zones.append([(x, y)])
        elif event == cv2.EVENT_LBUTTONUP:
            self.zones[-1].append((x, y))
            self.update_frame_copy()
        elif event == cv2.EVENT_RBUTTONDOWN and self.zones:
            self.zones.pop()
            self.update_frame_copy()

    def update_frame_copy(self):
        frame_copy = self.frame.copy()
        for zone in self.zones:
            cv2.rectangle(frame_copy, zone[0], zone[1], (255, 0, 0), 2)
        cv2.imshow('Image', frame_copy)

    def define_zones(self):
        print("\\nInstructions:")
        print("1. Left click and drag to draw a zone.")
        print("2. Release the left click to finalize the zone.")
        print("3. Right click to remove the most recently added zone.")
        print("4. Press any key to proceed after you have finished defining zones.\\n")
        print("Please define zones by clicking and dragging in the image.")
        self.cap.set(cv2.CAP_PROP_POS_MSEC, 60000)
        ret, self.frame = self.cap.read()
        cv2.namedWindow('Image')
        cv2.setMouseCallback('Image', self.on_mouse_click, param=self.frame)
        cv2.imshow('Image', self.frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.zones = [cv2.boundingRect(np.array(zone)) for zone in self.zones]

    def set_sensitivity(self):
        sensitivity = input("Please enter the sensitivity for rat detection (default is 500): ")
        if sensitivity:
            try:
                self.sensitivity = int(sensitivity)
            except ValueError:
                print("Invalid input. Using default sensitivity of 500.")

    def set_show_video(self):
        self.show_video = input("Do you want to show video playback? (y/n): ").lower() == 'y'

    def run(self):
        print("Press 'n' for next 1 second, 'p' for previous 1 second, 's' to start analysis.")
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
        zone_times = [0] * len(self.zones)

        while True:
            frame_num = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            current_time = frame_num / self.fps
            ret, frame = self.cap.read()
            if not ret:
                break
            fgmask = self.fgbg.apply(frame)
            _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            current_zone = None
            center = None
            for contour in contours:
                if cv2.contourArea(contour) > self.sensitivity:
                    (x, y, w, h) = cv2.boundingRect(contour)
                    center = (int(x + w/2), int(y + h/2))
                    self.rat_positions.append(center)
                    self.heatmap[center[1], center[0]] += 1
                    zone_areas = [(i+1, zw*zh) for i, (zx, zy, zw, zh) in enumerate(self.zones) if zx < center[0] < zx + zw and zy < center[1] < zy + zh]
                    if zone_areas:
                        zone_areas.sort(key=lambda x: x[1])
                        current_zone = zone_areas[0][0]
            if current_zone is None and self.last_known_zone is not None:
                current_zone = self.last_known_zone
                center = self.rat_positions[-1] if self.rat_positions else None
            elif current_zone is not None:
                self.last_known_zone = current_zone
                self.tracked_frames += 1
                zone_times[current_zone - 1] += 1
            self.writer.writerow([current_time, center[0] if center else None, center[1] if center else None, self.last_known_zone])
            print('Time:', current_time, 'Position:', center, 'Zone:', self.last_known_zone)
            if self.show_video and center:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if current_zone is not None:
                    cv2.putText(frame, f"Zone: {current_zone}, Coordinates: {center}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            if self.show_video:
                cv2.imshow('Thresholded Frame', thresh)
                for i, zone in enumerate(self.zones):
                    cv2.rectangle(frame, (zone[0], zone[1]), (zone[0] + zone[2], zone[1] + zone[3]), (255, 0, 0), 2)
                    cv2.putText(frame, str(i + 1), (zone[0] + 5, zone[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
                cv2.imshow('Frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        self.cap.release()
        cv2.destroyAllWindows()
        self.csv_file.close()

def main():
    try:
        pawsition_patrol = PawsitionPatrol('./output')
        pawsition_patrol.select_video_file()
        pawsition_patrol.define_zones()
        pawsition_patrol.set_sensitivity()
        pawsition_patrol.set_show_video()
        pawsition_patrol.run()
    except Exception as e:
        print("An error occurred:")
        print(e)
        input("Please enter to end")

if __name__ == "__main__":
    main()