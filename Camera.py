import cv2
import numpy as np
import math
import time

from object_detection import ObjectDetection


def measure_speed(position_cur, position_prev, time_interval):
    d_pixels = math.sqrt(math.pow(
        position_prev[0] - position_cur[0], 2) + math.pow(position_prev[1] - position_cur[1], 2))
    ppm = 8.8
    d_meters = d_pixels / ppm
    fps = 10
    speed = d_meters * fps * 3.6
    return speed


# Initialize Object Detection
od = ObjectDetection()

start_time = time.time()

# Video capture
cap = cv2.VideoCapture(file_path)

# Initialize count
count = 0
center_points_prv_frame = []

tracking_objects = {}
track_id = 0

class_list = []
classes_path = "dnn_model\classes.txt"
with open(classes_path, "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        class_list.append(class_name)

while True:
    ret, frame = cap.read()
    current_time = time.time()
    count += 1
    if count % 2 != 0:
        continue

    if not ret:
        break

    center_points_cur_frame = []

    # Detect objects on frame
    (class_ids, scores, boxes) = od.detect(frame)

    for class_id, box in zip(class_ids, boxes):
        class_name = class_list[class_id]  # Отримати назву класу за індексом
        (x, y, w, h) = box

        # center point of each box - щоб дати якесь позначення, що це один тз на різних фреймах
        center_point_x = int((x + x + w) / 2)
        center_point_y = int((y + y + h) / 2)
        center_points_cur_frame.append((center_point_x, center_point_y))

        print(f"FRAME {count}:", x, y, w, h)

        if class_name in ["car", "truck", "motorbike", "bus", "bicycle"]:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(frame, class_name, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Only at the begining we compare previous and current frame
    if count <= 2:
        for pt in center_points_cur_frame:
            for pt2 in center_points_prv_frame:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                if distance < 40:
                    tracking_objects[track_id] = pt
                    track_id += 1
    else:
        # Порівнюємо об'єкти, які вже маємо
        tracking_objects_copy = tracking_objects.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()

        for object_id, pt2 in tracking_objects_copy.items():
            object_exists = False

            for pt in center_points_cur_frame_copy:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                # Update obj position
                if distance < 40:
                    tracking_objects[object_id] = pt
                    object_exists = True

                    if pt in center_points_cur_frame:
                        # видаляємо непотрібні - додаємо нові на їх місце
                        center_points_cur_frame.remove(pt)
                    continue

            # if don`t update the position - remove the obj id lost
            if not object_exists:
                tracking_objects.pop(object_id)

        # Додаємо нові знайдені ID
        for pt in center_points_cur_frame:
            tracking_objects[track_id] = pt
            track_id += 1

    for object_id, pt in tracking_objects.items():
        if object_id in tracking_objects and object_id in tracking_objects_copy:
            position_cur = tracking_objects[object_id]
            position_prev = tracking_objects_copy[object_id]
            time_interval = current_time - start_time
            speed = int(measure_speed(
                position_cur, position_prev, time_interval)*3)
            start_time = current_time

            if speed < 110:
                cv2.rectangle(frame, (pt[0], pt[1] + 20),
                              (pt[0]+120, pt[1]), (0, 0, 0), -1)
                cv2.putText(
                    frame, (f"Speed: {speed} km/h"), (pt[0]+5, pt[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            elif speed > 110:
                cv2.rectangle(frame, (pt[0], pt[1] + 20),
                              (pt[0]+120, pt[1]), (0, 0, 0), -1)
                cv2.putText(
                    frame, (f"Speed: {speed} km/h"), (pt[0]+5, pt[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    cv2.imshow("Frame", frame)

    center_points_prv_frame = center_points_cur_frame.copy()

    key = cv2.waitKey()
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
