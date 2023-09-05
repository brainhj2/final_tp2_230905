
import os
import cv2
from PIL import Image
import numpy as np


def is_overlapping(box1, box2):
    # box = (x1, y1, x2, y2)
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Check if one box is to the right or left of the other
    if x1_1 > x2_2 or x1_2 > x2_1:
        return False

    # Check if one box is above or below the other
    if y1_1 > y2_2 or y1_2 > y2_1:
        return False

    return True

def get_center(box):
    # box = (x1, y1, x2, y2)
    return (box[0]+box[2])/2, (box[1]+box[3])/2

def get_distance(box1, box2):
    center1 = get_center(box1)
    center2 = get_center(box2)
    return np.sqrt((center1[0]-center2[0])**2 + (center1[1]-center2[1])**2)

def check_violation(box, traffic_cls, crosswalk_mask, stopline_mask):
    if traffic_cls is not None:
        # print(crosswalk_mask)
        if crosswalk_mask is not None and is_bottom_in_mask(box, crosswalk_mask):
            return "Crosswalk Violation"
        elif stopline_mask is not None and is_bottom_in_mask(box, stopline_mask):
            return "Stopline Violation"
    return None

def is_bottom_in_mask(box, masks_xy):
    x_min, y_min, x_max, y_max = box
    bottom_left = (x_min, y_max)
    bottom_right = (x_max, y_max)

    for mask_xy in masks_xy:
        mask_xy = mask_xy
        mask_xy = mask_xy.reshape(-1, 1, 2)
        if cv2.pointPolygonTest(mask_xy, bottom_left, False) >= 0 and cv2.pointPolygonTest(mask_xy, bottom_right, False) >= 0:
            return True
    return False

# 인도 주행

image_dir = "/content/sample"
count = 1
# # List all files in the directory
image_files = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]

for image_path in image_files:
    image = cv2.imread(image_path)
    results1 = model2.predict(image_path, save=True, conf=0.12)  #bbox
    results2 = model1.predict(image_path, save=True, conf=0.12)  #poly

    for result1, result2 in zip(results1, results2):
        boxes_xyxy = result1.boxes.xyxy.numpy()
        labels = result1.boxes.cls.numpy()
        if len(result2) == 0:
            print(f"{image_path} 폴리곤 객체가 없습니다. 스킵합니다!")
            continue
        masks_xy = result2.masks.xy
        masks_cls = result2.boxes.cls.numpy()

        # 보도 클래스에 대한 마스크만 가져옵니다.
        sidewalk_masks_xy = [masks_xy[i] for i in range(len(masks_cls)) if masks_cls[i] == 0]

        # 차량 클래스에 대한 박스만 가져옵니다.
        vehicle_boxes_xyxy = boxes_xyxy[np.isin(labels, [8, 9, 10])]
        
        for box in vehicle_boxes_xyxy:
            if is_bottom_in_mask(box, sidewalk_masks_xy):
                print(f"The bottom of the box in {image_path} is in a mask!")
                print(f"Coordinates of the box: {box}")
                # print(image)
                x_min, y_min, x_max, y_max = map(int, box)
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)


# 오토바이 헬멧


for image_path in image_files:
    results = model2.predict(image_path, save=True, conf=0.12) #bbox
    for result in results:
        boxes_xyxy = result.boxes.xyxy.numpy()
        labels = result.boxes.cls.numpy()

        riding_motorcycles, helmets = [], []

        for i, box in enumerate(boxes_xyxy):
            label = labels[i]
            if label == 10:
                riding_motorcycles.append(box)
            elif label == 3:
                helmets.append(box)

        for helmet in helmets:
            # find the closest motorcycle rider to the helmet
            distances = [get_distance(helmet, riding_motorcycle) for riding_motorcycle in riding_motorcycles]
            closest_riding_motorcycle = riding_motorcycles[np.argmin(distances)]

            # check if this motorcycle rider is wearing the helmet
            if is_overlapping(closest_riding_motorcycle, helmet):
                print("Okay, pass")
            else:
                print("A motorcycle rider is violating the rules!")
                x1, y1, x2, y2 = closest_riding_motorcycle.astype(int)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)


#정지선 교차로 횡단보도 빨간불일때 탐지

for image_path in image_files:
    
    results1 = model2.predict(image_path, save=True, conf=0.12) #bbox
    results2 = model1.predict(image_path, save=True, conf=0.12) #poly
    

    for result1, result2 in zip(results1, results2):
        boxes_xyxy = result1.boxes.xyxy.numpy()
        labels = result1.boxes.cls.numpy()
        if len(result2) == 0:
            print(f"{image_path} 폴리곤 객체가 없습니다. 스킵합니다!")
            continue
        masks_xy = result2.masks.xy
        masks_cls = result2.boxes.cls.numpy()
        
        # 정지선 및 교차로 클래스에 대한 마스크만 가져옵니다.
        intersection_masks_xy = [masks_xy[i] for i in range(len(masks_cls)) if masks_cls[i] == 3]  #교차로
        stopline_masks_xy = [masks_xy[i] for i in range(len(masks_cls)) if masks_cls[i] == 5] #정지선
        
        # 차량 클래스에 대한 박스만 가져옵니다.
        vehicle_boxes_xyxy = boxes_xyxy[np.isin(labels, [8, 9, 10])]  
        
        for box in vehicle_boxes_xyxy:
            if is_bottom_in_mask(box, intersection_masks_xy):
                print(f"The bottom of the box in {image_path} is in an intersection mask!")
                print(f"Coordinates of the box: {box}")
                x_min, y_min, x_max, y_max = map(int, box)
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            elif is_bottom_in_mask(box, stopline_masks_xy):
                print(f"The bottom of the box in {image_path} is in a stopline mask!")
                print(f"Coordinates of the box: {box}")
                x_min, y_min, x_max, y_max = map(int, box)
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                
#횡단보도 주행 (if 구문 한번 손봐야함)


for image_path in image_files:
    results1 = model2.predict(image_path, save=True, conf=0.12) # bbox
    results2 = model1.predict(image_path, save=True, conf=0.12) # poly


    for result1, result2 in zip(results1, results2): #zip(bbox, poly)
        boxes_xyxy = result1.boxes.xyxy.numpy()
        labels = result1.boxes.cls.numpy()
        if len(result2) == 0:
            print(f"{image_path} 폴리곤 객체가 없습니다. 스킵합니다!")
            continue
        masks_xy = result2.masks.xy
        masks_cls = result2.boxes.cls.numpy() # poly

        # 횡단보도 클래스에 대한 마스크만 가져옵니다.
        sidewalk_masks_xy = [masks_xy[i] for i in range(len(masks_cls)) if masks_cls[i] == 1]

        # 탈거 객체 클래스에 대한 박스만 가져옵니다.
        vehicle_boxes_xyxy = boxes_xyxy[np.isin(labels, [8, 9, 10])]

        # 보행자 신호 클래스에 대한 박스만 가져옵니다.
        pedlight_boxes_xyxy = boxes_xyxy[np.isin(labels, [5, 6])] # 5: pedGreen, 6: pedRed
        for box in vehicle_boxes_xyxy:
            if is_bottom_in_mask(box, sidewalk_masks_xy):
                print(f"The bottom of the box in {image_path} is in a mask!")
                print(f"Coordinates of the box: {box}")
                x_min, y_min, x_max, y_max = map(int, box)
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), 2)
    
    result_image_path = os.path.join("/content/result", os.path.basename(image_path))
    cv2.imwrite(result_image_path, image)