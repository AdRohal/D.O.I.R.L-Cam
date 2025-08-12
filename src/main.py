import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading
import time

class YOLODashboard:
    def __init__(self):
        self.running = False
        self.cap = None
        self.camera_thread = None
        self.net = None
        self.output_layers = None
        self.classes = None
        self.animation_running = False
        
        self.root = tk.Tk()
        self.root.title("D.O.I.R.L - CAM Dashboard")
        self.root.geometry("1400x800")
        self.root.configure(bg="#4b296b")
        self.root.resizable(False, False)
        
        self.create_dashboard()
        self.load_yolo_model()
        
    def create_dashboard(self):
        self.control_panel = tk.Frame(self.root, bg="#4b296b", width=350, height=800)
        self.control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        self.control_panel.pack_propagate(False)
        
        self.camera_panel = tk.Frame(self.root, bg="#4b296b", width=1030, height=780)
        self.camera_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(0, 10), pady=10)
        self.camera_panel.pack_propagate(False)
        
        self.create_control_widgets()
        self.create_camera_widgets()
        
    def create_control_widgets(self):
        title_label = tk.Label(
            self.control_panel,
            text="YOLO Detection\nDashboard",
            font=("Arial", 20, "bold"),
            fg="#fffcf2",
            bg="#4b296b"
        )
        title_label.pack(pady=20)
        
        self.status_frame = tk.Frame(self.control_panel, bg="#4b296b")
        self.status_frame.pack(pady=10)
        
        tk.Label(self.status_frame, text="Status:", font=("Arial", 12, "bold"), 
                fg="#fffcf2", bg="#4b296b").pack()
        
        self.status_label = tk.Label(
            self.status_frame,
            text="● STOPPED",
            font=("Arial", 14, "bold"),
            fg="#e74c3c",
            bg="#4b296b"
        )
        self.status_label.pack()
        
        self.loading_frame = tk.Frame(self.control_panel, bg="#4b296b")
        self.loading_frame.pack(pady=10)
        
        self.loading_label = tk.Label(
            self.loading_frame,
            text="",
            font=("Arial", 12),
            fg="#e9d758",
            bg="#4b296b"
        )
        self.loading_label.pack()
        
        button_frame = tk.Frame(self.control_panel, bg="#4b296b")
        button_frame.pack(pady=30)
        
        self.start_button = tk.Button(
            button_frame,
            text="▶ START DETECTION",
            command=self.start_camera,
            bg="#e9d758",
            fg="#4b296b",
            font=("Arial", 14, "bold"),
            width=20,
            height=2,
            relief="flat",
            cursor="hand2",
            activebackground="#f4e470",
            activeforeground="#4b296b"
        )
        self.start_button.pack(pady=10)
        
        self.stop_button = tk.Button(
            button_frame,
            text="⏸ STOP DETECTION",
            command=self.stop_camera,
            bg="#e74c3c",
            fg="#fffcf2",
            font=("Arial", 14, "bold"),
            width=20,
            height=2,
            relief="flat",
            cursor="hand2",
            state="disabled",
            activebackground="#c0392b",
            activeforeground="#fffcf2"
        )
        self.stop_button.pack(pady=10)
        
        stats_frame = tk.LabelFrame(
            self.control_panel,
            text="Statistics",
            font=("Arial", 12, "bold"),
            fg="#fffcf2",
            bg="#4b296b",
            borderwidth=2,
            relief="groove"
        )
        stats_frame.pack(pady=20, padx=20, fill="x")
        
        self.fps_label = tk.Label(stats_frame, text="FPS: 0", font=("Arial", 10), 
                                 fg="#fffcf2", bg="#4b296b")
        self.fps_label.pack(pady=5)
        
        self.objects_label = tk.Label(stats_frame, text="Objects: 0", font=("Arial", 10), 
                                     fg="#fffcf2", bg="#4b296b")
        self.objects_label.pack(pady=5)
        
        self.frame_count_label = tk.Label(stats_frame, text="Frames: 0", font=("Arial", 10), 
                                         fg="#fffcf2", bg="#4b296b")
        self.frame_count_label.pack(pady=5)
        
        exit_button = tk.Button(
            self.control_panel,
            text="✕ EXIT",
            command=self.exit_app,
            bg="#95a5a6",
            fg="#fffcf2",
            font=("Arial", 12, "bold"),
            width=15,
            height=2,
            relief="flat",
            cursor="hand2",
            activebackground="#7f8c8d",
            activeforeground="#fffcf2"
        )
        exit_button.pack(side=tk.BOTTOM, pady=20)
        
    def create_camera_widgets(self):
        camera_title = tk.Label(
            self.camera_panel,
            text="Live Camera Feed - YOLO Object Detection",
            font=("Arial", 16, "bold"),
            fg="#fffcf2",
            bg="#4b296b"
        )
        camera_title.pack(pady=10)
        
        self.camera_frame = tk.Frame(self.camera_panel, bg="#000000", relief="sunken", borderwidth=3)
        self.camera_frame.pack(pady=10, padx=20, fill="both", expand=True)
        
        self.camera_label = tk.Label(
            self.camera_frame,
            text="",
            bg="#000000"
        )
        self.camera_label.pack(expand=True)
        
        self.show_camera_error_overlay()
        
    def create_camera_error_overlay(self):
        width, height = 800, 600
        
        blur_image = Image.new('RGB', (width, height), color='#333333')
        blur_array = np.array(blur_image)
        
        noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
        blur_array = cv2.addWeighted(blur_array, 0.7, noise, 0.3, 0)
        blur_array = cv2.GaussianBlur(blur_array, (51, 51), 0)
        
        blur_image = Image.fromarray(blur_array)
        draw = ImageDraw.Draw(blur_image)
        
        x_size = 150
        center_x, center_y = width//2, height//2 - 50
        line_width = 8
        
        draw.line([(center_x - x_size//2, center_y - x_size//2), 
                  (center_x + x_size//2, center_y + x_size//2)], 
                 fill='#e74c3c', width=line_width)
        draw.line([(center_x + x_size//2, center_y - x_size//2), 
                  (center_x - x_size//2, center_y + x_size//2)], 
                 fill='#e74c3c', width=line_width)
        
        draw.ellipse([(center_x - x_size//2 - 20, center_y - x_size//2 - 20),
                     (center_x + x_size//2 + 20, center_y + x_size//2 + 20)],
                    outline='#e74c3c', width=line_width)
        
        try:
            font_large = ImageFont.truetype("arial.ttf", 24)
            font_small = ImageFont.truetype("arial.ttf", 16)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        error_text = "CAMERA NOT WORKING"
        instruction_text = "Please press START to try again..."
        
        bbox1 = draw.textbbox((0, 0), error_text, font=font_large)
        text1_width = bbox1[2] - bbox1[0]
        
        bbox2 = draw.textbbox((0, 0), instruction_text, font=font_small)
        text2_width = bbox2[2] - bbox2[0]
        
        text1_y = center_y + x_size//2 + 40
        text2_y = text1_y + 40
        
        for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
            draw.text((center_x - text1_width//2 + dx, text1_y + dy), 
                     error_text, font=font_large, fill='#000000')
            draw.text((center_x - text2_width//2 + dx, text2_y + dy), 
                     instruction_text, font=font_small, fill='#000000')
        
        draw.text((center_x - text1_width//2, text1_y), 
                 error_text, font=font_large, fill='#fffcf2')
        draw.text((center_x - text2_width//2, text2_y), 
                 instruction_text, font=font_small, fill='#e9d758')
        
        return blur_image
        
    def show_camera_error_overlay(self):
        error_image = self.create_camera_error_overlay()
        photo = ImageTk.PhotoImage(error_image)
        self.camera_label.config(image=photo, text="")
        self.camera_label.image = photo
        
    def load_yolo_model(self):
        try:
            weights_path = "yolo/yolov3-tiny.weights"
            config_path = "yolo/yolov3-tiny.cfg"
            names_path = "yolo/coco.names"

            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"YOLO weights file not found: {weights_path}")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"YOLO config file not found: {config_path}")
            if not os.path.exists(names_path):
                raise FileNotFoundError(f"YOLO names file not found: {names_path}")

            print("Loading YOLO model...")
            self.net = cv2.dnn.readNet(weights_path, config_path)
            layer_names = self.net.getLayerNames()
            
            unconnected = self.net.getUnconnectedOutLayers()
            if len(unconnected.shape) > 1:
                self.output_layers = [layer_names[i[0] - 1] for i in unconnected]
            else:
                self.output_layers = [layer_names[i - 1] for i in unconnected]
            
            self.classes = []
            with open(names_path, "r") as f:
                self.classes = [line.strip() for line in f.readlines()]
                
            print("YOLO model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            
    def animate_loading(self):
        loading_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        counter = 0
        
        while self.animation_running:
            if self.running:
                char = loading_chars[counter % len(loading_chars)]
                self.loading_label.config(text=f"{char} Processing...")
                counter += 1
            else:
                self.loading_label.config(text="")
            
            time.sleep(0.1)
            
    def update_status(self, status, color):
        self.status_label.config(text=f"● {status}", fg=color)
        
    def run_camera(self):
        try:
            print("Initializing camera...")
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            
            if not self.cap.isOpened():
                print("ERROR: Could not open camera!")
                self.root.after(0, lambda: self.update_status("CAMERA ERROR", "#e74c3c"))
                self.root.after(0, self.show_camera_error_overlay)
                return
                
            self.root.after(0, lambda: self.update_status("RUNNING", "#27ae60"))
            print("Camera started successfully!")
            
            frame_count = 0
            fps_counter = 0
            fps_start_time = time.time()
            
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    self.root.after(0, self.show_camera_error_overlay)
                    break
                    
                frame_count += 1
                fps_counter += 1
                
                current_time = time.time()
                if current_time - fps_start_time >= 1.0:
                    fps = fps_counter
                    fps_counter = 0
                    fps_start_time = current_time
                    self.root.after(0, lambda f=fps: self.fps_label.config(text=f"FPS: {f}"))
                
                height, width, channels = frame.shape
                
                blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                self.net.setInput(blob)
                outs = self.net.forward(self.output_layers)
                
                class_ids = []
                confidences = []
                boxes = []
                
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        
                        if confidence > 0.5:
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)
                            
                            x_rect = int(center_x - w / 2)
                            y_rect = int(center_y - h / 2)
                            
                            boxes.append([x_rect, y_rect, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)
                
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                
                objects_detected = 0
                if len(indexes) > 0:
                    if isinstance(indexes, np.ndarray):
                        indexes = indexes.flatten()
                    
                    objects_detected = len(indexes)
                    for i in indexes:
                        x_rect, y_rect, w, h = boxes[i]
                        label = str(self.classes[class_ids[i]])
                        confidence = confidences[i]
                        
                        color = (128, 0, 128)
                        cv2.rectangle(frame, (x_rect, y_rect), (x_rect + w, y_rect + h), color, 2)
                        cv2.putText(frame, f"{label} {confidence:.2f}", 
                                  (x_rect, y_rect - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                self.root.after(0, lambda o=objects_detected: self.objects_label.config(text=f"Objects: {o}"))
                self.root.after(0, lambda f=frame_count: self.frame_count_label.config(text=f"Frames: {f}"))
                
                self.display_frame(frame)
                
        except Exception as e:
            print(f"Error in camera loop: {e}")
            self.root.after(0, lambda: self.update_status("ERROR", "#e74c3c"))
            self.root.after(0, self.show_camera_error_overlay)
        finally:
            self.cleanup()
            
    def display_frame(self, frame):
        display_height = 600
        aspect_ratio = frame.shape[1] / frame.shape[0]
        display_width = int(display_height * aspect_ratio)
        
        frame_resized = cv2.resize(frame, (display_width, display_height))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        pil_image = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(pil_image)
        
        self.root.after(0, lambda: self.camera_label.config(image=photo, text=""))
        self.camera_label.image = photo
        
    def cleanup(self):
        if self.cap:
            self.cap.release()
        print("Camera cleaned up.")
        
    def start_camera(self):
        if not self.running and self.net is not None:
            self.running = True
            self.animation_running = True
            
            self.start_button.config(state="disabled", bg="#95a5a6", fg="#fffcf2")
            self.stop_button.config(state="normal", bg="#e74c3c", fg="#fffcf2")
            self.update_status("STARTING...", "#f39c12")
            
            animation_thread = threading.Thread(target=self.animate_loading)
            animation_thread.daemon = True
            animation_thread.start()
            
            self.camera_thread = threading.Thread(target=self.run_camera)
            self.camera_thread.daemon = True
            self.camera_thread.start()
            
    def stop_camera(self):
        if self.running:
            self.running = False
            self.animation_running = False
            
            self.start_button.config(state="normal", bg="#e9d758", fg="#4b296b")
            self.stop_button.config(state="disabled", bg="#95a5a6", fg="#fffcf2")
            self.update_status("STOPPED", "#e74c3c")
            
            self.show_camera_error_overlay()
            
            self.fps_label.config(text="FPS: 0")
            self.objects_label.config(text="Objects: 0")
            self.frame_count_label.config(text="Frames: 0")
            
            print("Camera stopped.")
            
    def exit_app(self):
        self.stop_camera()
        time.sleep(0.5)
        self.root.quit()
        self.root.destroy()
        
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    try:
        from PIL import Image, ImageTk
    except ImportError:
        print("PIL (Pillow) is required for the dashboard. Install with: pip install Pillow")
        exit(1)
    
    app = YOLODashboard()
    app.run()