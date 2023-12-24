import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

def select_video():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename()
    root.destroy()
    return file_path

def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, 1)
    abs_cos = abs(M[0,0])
    abs_sin = abs(M[0,1])

    bound_w = int(h * abs_sin + w * abs_cos)
    bound_h = int(h * abs_cos + w * abs_sin)

    M[0, 2] += bound_w/2 - center[0]
    M[1, 2] += bound_h/2 - center[1]

    rotated = cv2.warpAffine(image, M, (bound_w, bound_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return rotated

def process_video(video_path, angle, progress_label):
    file_dir, file_name = os.path.split(video_path)
    name, ext = os.path.splitext(file_name)
    output_file = os.path.join(file_dir, f"{name}_rotated{ext}")

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame = cap.read()
    rotated_frame = rotate_image(frame, angle)
    h, w = rotated_frame.shape[:2]
    out = cv2.VideoWriter(output_file, fourcc, fps, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(frame_count):
        ret, frame = cap.read()
        if ret:
            rotated_frame = rotate_image(frame, angle)
            out.write(rotated_frame)
            progress = (i + 1) / frame_count * 100
            progress_label.config(text=f"Processing: {progress:.2f}%")
            progress_label.update()
        else:
            break

    cap.release()
    out.release()
    progress_label.config(text="Processing Complete!")

def main():
    root = tk.Tk()
    root.withdraw()

    video_path = select_video()
    if not video_path:
        root.destroy()
        return

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 60 * int(cap.get(cv2.CAP_PROP_FPS)))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Unable to read the video frame.")
        root.destroy()
        return

    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv2image)
    tk_image = ImageTk.PhotoImage(image=pil_image)

    preview_window = tk.Toplevel(root)
    preview_window.title("Video Rotation Tool")

    label = tk.Label(preview_window, image=tk_image)
    label.pack()

    confirmed = False
    while not confirmed:
        angle = simpledialog.askfloat("Rotation Angle", "Enter the angle to rotate the video:", parent=preview_window)
        if angle is None:
            break

        rotated_frame = rotate_image(cv2image, angle)
        pil_rotated = Image.fromarray(rotated_frame)
        tk_rotated = ImageTk.PhotoImage(image=pil_rotated)
        label.configure(image=tk_rotated)
        label.image = tk_rotated

        confirmed = messagebox.askyesno("Confirm Rotation", "Do you want to rotate the video with this angle?")

    if confirmed:
        preview_window.destroy()

        processing_window = tk.Toplevel(root)
        processing_window.title("Processing Video")
        progress_label = tk.Label(processing_window, text="Starting processing...")
        progress_label.pack()

        processing_window.update_idletasks()
        process_video(video_path, angle, progress_label)
        processing_window.destroy()

    root.destroy()

if __name__ == "__main__":
    main()
