import cv2
import numpy as np
import serial  
import time  

# Initialize serial communication with Arduino
ser = serial.Serial('/dev/ttyUSB0', 115200)  # Adjust as per connection

class MovingAverageFilter:
    def init(self, window_size=5):
        self.window_size = window_size
        self.values = []

    def add_value(self, value):
        if len(self.values) >= self.window_size:
            self.values.pop(0)
        self.values.append(value)
        return int(sum(self.values) / len(self.values))

# Constants for distance calculation
FOCAL_LENGTH = 800  
KNOWN_HEIGHT = 1.7  # Known height of an object (e.g., average height of a car)

def estimate_distance(pixel_height):
    return (KNOWN_HEIGHT * FOCAL_LENGTH) / pixel_height if pixel_height > 0 else 0

# Load class names and initialize the filter for classes of interest
classFile = '/home/electronic-brain/Desktop/Project/Model_Data/coco.names'
interested_classes = {'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'traffic light', 'street sign', 'stop sign', 'cat', 'dog'}
classNames = []
with open(classFile, 'rt') as f:
    classNames = [line.strip() for line in f.readlines()]
interested_indices = {i for i, className in enumerate(classNames) if className in interested_classes}

# Initialize the moving average filter
steering_filter = MovingAverageFilter()

# Object detection setup
thres = 0.5
nms_threshold = 0.2
cap = cv2.VideoCapture(0)
configPath = '/home/electronic-brain/Desktop/Project/Model_Data/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = '/home/electronic-brain/Desktop/Project/Model_Data/frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Lane detection setup
vertices = np.array([[(5, 470), (150, 300), (470, 300), (640, 470)]], dtype=np.int32)
src = np.float32(vertices)
dst = np.float32([[100, 470], [100, 100], [500, 100], [500, 470]])
Minv = cv2.getPerspectiveTransform(dst, src)

# Function to calculate the path for lane change
def calculate_lane_change_path(x_start, y_start, x_end, y_end, steps=50):
    x_values = np.linspace(x_start, x_end, steps)
    y_values = np.linspace(y_start, y_end, steps)
    return x_values, y_values

# Function to perform lane change
def perform_lane_change(x_start, y_start, x_end, y_end):
    x_values, y_values = calculate_lane_change_path(x_start, y_start, x_end, y_end)
    for x, y in zip(x_values, y_values):
        steering_angle = x - (img.shape[1] // 2)
        filtered_angle = steering_filter.add_value(steering_angle)
        ser.write(f"{filtered_angle}\n".encode('utf-8'))
        time.sleep(0.1)  # adjust delay for smoothness
    return x_values, y_values

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture frame. Exiting...")
        break

    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    obstacle_detected = False
    if classIds is not None and len(classIds) > 0:  # Check if detections were made
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            if classId - 1 in interested_indices:
                x, y, w, h = box
                cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
                distance = estimate_distance(h)
                label = f"{classNames[classId - 1].upper()} {confidence:.2f} Dist: {distance:.2f}In"
                cv2.putText(img, label, (x + 10, y + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                if classNames[classId - 1] == 'car' and distance < 50:  # Adjust the distance threshold
                    obstacle_detected = True

    if obstacle_detected:
        print("Obstacle detected! Initiating lane change...")
        x_start, y_start = img.shape[1] // 2, img.shape[0]
        x_end, y_end = img.shape[1] // 2 - 100, img.shape[0]  
        perform_lane_change(x_start, y_start, x_end, y_end)
        print("Lane changed. Returning to original lane...")
        time.sleep(2)  # Move forward for a certain distance
        perform_lane_change(x_end, y_end, x_start, y_start)

    image_roi = np.copy(img)
    M = cv2.getPerspectiveTransform(src, dst)
    warped_image = cv2.warpPerspective(image_roi, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
    lines = cv2.HoughLinesP(binary_image, rho=1, theta=np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)

    if lines is not None:
        lane_lines = [((line[0][0] + line[0][2]) // 2) for line in lines]
        if lane_lines:
            lane_center = np.mean(lane_lines)
            steering_angle = lane_center - (img.shape[1] // 2)
            filtered_angle = steering_filter.add_value(steering_angle)
            print(f"Smoothed Steering Angle: {filtered_angle}")
            cv2.line(binary_image, (int(lane_center), 0), (int(lane_center), binary_image.shape[0]), (255, 255, 255), 2)

        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Transform points back to original image space
            pts = np.array([[x1, y1], [x2, y2]], dtype=np.float32).reshape(-1, 1, 2)
            original_pts = cv2.perspectiveTransform(pts, Minv).reshape(-1, 2)
            cv2.line(img, (int(original_pts[0][0]), int(original_pts[0][1])), (int(original_pts[1][0]), int(original_pts[1][1])), (0, 0, 255), 3)

    # Send the steering angle to the Arduino
    ser.write(f"{filtered_angle}\n".encode('utf-8'))

    cv2.imshow("Lane Center Calculation", binary_image)
    cv2.imshow("Lane Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
ser.close()