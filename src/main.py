import cv2
import numpy as np
import os
import tkinter as tk
from screeninfo import get_monitors


def run_camera():
    """Start the webcam and run YOLO object detection."""

    # Paths to the YOLO files
    weights_path = "../yolo/yolov3-tiny.weights"
    config_path = "../yolo/yolov3-tiny.cfg"
    names_path = "../yolo/coco.names"

    # Verify that the paths exist
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"YOLO weights file not found: {weights_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"YOLO config file not found: {config_path}")
    if not os.path.exists(names_path):
        raise FileNotFoundError(f"YOLO names file not found: {names_path}")

    # Load YOLO
    net = cv2.dnn.readNet(weights_path, config_path)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    classes = []
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Create a named window and resize it
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Frame', 1280, 800)

    # Get screen resolution and calculate position to center the window
    monitor = get_monitors()[0]
    screen_width, screen_height = monitor.width, monitor.height
    window_width, window_height = 1280, 900
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2

    # Move the window to the center of the screen
    cv2.moveWindow('Frame', x, y)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing information on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = (128, 0, 128)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Check for quit condition
        if cv2.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_cam = False

    def start_program():
        global start_cam
        start_cam = True
        menu_root.destroy()

    def exit_program():
        menu_root.destroy()

    menu_root = tk.Tk()
    menu_root.title("Menu")

    start_button = tk.Button(
        menu_root,
        text="Start",
        command=start_program,
        bg="red",
        fg="yellow",
        font=("Comic Sans MS", 24, "bold"),
        width=15,
        height=3,
    )
    start_button.pack(pady=20)

    exit_button = tk.Button(
        menu_root,
        text="Exit",
        command=exit_program,
        font=("Comic Sans MS", 24, "bold"),
        width=15,
        height=3,
    )
    exit_button.pack(pady=20)

    menu_root.mainloop()

    if start_cam:
        run_camera()

