import tkinter as tk
import cv2
from PIL import Image, ImageTk
import process_frame as pf
class Camera():
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.root = tk.Tk()
        self.root.title("手势检测")
        self.label = tk.Label(self.root)
        self.label.pack()
        self.root.bind("<Escape>", lambda e: self.root.quit())
        gesture_label = tk.Label(self.root, text="手势检测平台", font=("宋体", 20))
        gesture_label.pack()

    def show_frame(self):
        _, frame = self.cap.read()
        self.process = pf.Process(frame)
        processed_frame=self.process.process_frame()
        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(processed_frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.label.imgtk = imgtk
        self.label.configure(image=imgtk)
        self.label.after(1, self.show_frame)

    def run(self):
        self.show_frame()
        self.root.mainloop()

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

