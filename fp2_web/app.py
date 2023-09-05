#-----------------------------------수정전----------------
from flask import Flask, render_template, request, send_from_directory, jsonify, url_for, redirect
from werkzeug.utils import secure_filename
import os
from ultralytics import YOLO
import requests
import cv2
from PIL import Image
import numpy as np
# from algorithm_test import algorithm_exam
#####################
# from algorithm_class import SafetyAlgorithm  # assuming SafetyAlgorithm is in a separate module
#####################

app = Flask(__name__)
 
cards = []
# #####################
# # initialize SafetyAlgorithm with necessary paths
# safety_algorithm = SafetyAlgorithm(
#     model1_path='path_to_model1', 
#     model2_path='path_to_model2', 
#     model_path='path_to_model3', 
#     save_dir='path_to_save_directory'
# )
# #####################

@app.route('/')
def index_before():
    return render_template('index.html')

@app.route('/imageupload')
def index():
    return render_template('imageupload.html')

from PIL import Image
# 수정 전 잘 돌아가는 get_prediction 여기부터 시작-------
@app.route('/predict', methods=['POST'])
def get_prediction():

    img = request.files['img']
    filename = secure_filename(img.filename)

    folder = '/Users/brainhj2/Desktop/final2/fp2_web/predicted'
    img_path = os.path.join(folder, filename)
    img.save(img_path)
    # algorithm_exam()
    
    # predicted_filename = '/Users/brainhj2/Desktop/final2/fp2_web/yolo_predicted_results'

    # model = YOLO('yolov8n-seg.pt')
    # predict = model.predict(source=img_path,
    #                         conf=0.25,
    #                         save=True)

    # #####################
    # # Now use the SafetyAlgorithm instance to run your additional algorithms
    # safety_algorithm.algorithm_crosswalk(img_path)
    # safety_algorithm.algorithm_helmet([img_path])
    # #####################
    model1 = YOLO("/Users/brainhj2/Desktop/final2/fp2_web/polygon_200_best.pt") # model1 이 results2 이고 polygon입니다. 
    model2 = YOLO("/Users/brainhj2/Desktop/final2/fp2_web/epoch_100_bbox_best.pt") # model2 이 results1 이고 bbox입니다.
    ###복붙시작


    # import os
    # import cv2
    # from PIL import Image
    # import numpy as np
    pm_list = []

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
                pm_list.append("Crosswalk Violation")
                return "Crosswalk Violation"
            elif stopline_mask is not None and is_bottom_in_mask(box, stopline_mask):
                pm_list.append("Stopline Violation")
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

    image_dir = '/Users/brainhj2/Desktop/final2/fp2_web/predicted'
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
                    pm_list.append("인도 주행 위반입니다")
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
                    pm_list.append("안전모 규정 위반입니다.")
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
                    pm_list.append("교차로 규정 위반입니다..")
                    print(f"The bottom of the box in {image_path} is in an intersection mask!")
                    print(f"Coordinates of the box: {box}")
                    x_min, y_min, x_max, y_max = map(int, box)
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                elif is_bottom_in_mask(box, stopline_masks_xy):
                    pm_list.append("정지선 규정 위반입니다.")
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
                    pm_list.append("횡단보도 규정 위반입니다.")
                    print(f"The bottom of the box in {image_path} is in a mask!")
                    print(f"Coordinates of the box: {box}")
                    x_min, y_min, x_max, y_max = map(int, box)
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), 2)
        
    result_image_path = os.path.join("/Users/brainhj2/Desktop/final2/fp2_web/yolo_predicted_results", os.path.basename(image_path))
    cv2.imwrite(result_image_path, image)
    # result = predict[0]
    # result_image = Image.fromarray(result.plot()[:,:,::-1])
    # path = os.path.join(predicted_filename, filename)
    # result_image.save(path)
    img_url = url_for('send_predicted_file', filename=os.path.basename(image_path))
    pay = f"{len(pm_list) * 10} 만원입니다"  
    #######복붙끝
    cards.append({
        "img_src": img_url,
        "link": img_url,
        "title": "PM 7항을 위반하셨습니다.",
        "date": "2023/09/05",
        "description": [a for a in pm_list],
        "comment": f"벌칙금은 {pay}"
    })
    
    return redirect(url_for('gallery'))



@app.route('/gallery')
def gallery():
    return render_template('gallery.html')



@app.route('/get-cards')
def get_cards():
    
    return jsonify(cards=cards)



# 수정 전 잘 돌아가는 get_prediction 끝 
@app.route('/predicted/<filename>')
def send_predicted_file(filename):
    return send_from_directory('/Users/brainhj2/Desktop/final2/fp2_web/yolo_predicted_results', filename)
 


@app.route('/get-processed-img')
def get_processed_img():
    # ... determine the filename of the processed image ...
    filename = 'processed_image.jpg'  # replace this with the actual filename
    img_url = '/predicted/' + filename
    return jsonify({'img_url': img_url})


     
if __name__ == '__main__':
    app.run(debug=True,port=5001)

#-------수정후






# 이미지 업로드 버튼 눌르고 
# 확인하기 눌르면 다음 main() 함수 실행? 

# def main():
#     def algo1() 
#     def algo2()
#     def algo3()

#     input_image = input(~~)#업로드된 이미지파일 


#     위반 bbox = []

#     pm위반 사항 total = []

#     algo1(input_image)

#     if algo1:

#         bbox.append(1 위반객체'sbbox)
#         pm위반 사항1 total.append(pm code)

#     algo2(input_image)

#     if algo2:

#         bbox.append(2 위반객체'sbbox)
#         pm위반 사항2 total.append(pm code)
#     algo3(input_image)

#     if algo3:
#         bbox.append(3 위반객체'sbbox)

#         pm위반 사항3 total.append(pm code)


#     # cv2. ~image draw 
#     input 받은 이미지에 위반 bbox_list에 담긴 좌표를 모두 빨간 네모박스치기
#     박스친후 이미지게시 

#     print(f" {pm위반사항1 }, 과 {pm2위반사항2}.... 을 위반하셨습니다")
#     return None

# if __name__ == '__main__':
#     main()
    