import csv
import copy
from collections import Counter
from collections import deque

import cv2 as cv
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

from solve import *

gesture = None

def solve(args):
    # Argument parsing
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True
    
    # Camera preparation
    cap = cv.VideoCapture(cap_device) # cap_device = 0 -> webcam
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    
    # Model load 
    mp_hands = mp.solutions.hands # tạo 1 đối tượng hand với 2 thuộc tính Hand và Handlandmark
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode, # False là chạy webcam
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    
    keypoint_classifier = KeyPointClassifier() # phân lớp

    point_history_classifier = PointHistoryClassifier() # phân loại chuỗi lịch sử điểm chỉ tay 
    
    # Read labels
    with open(
            'model/keypoint_classifier/keypoint_classifier_label.csv',
            encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]
        
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [row[0] for row in point_history_classifier_labels]
        
    # FPS Measurement
    csvFpsCalc = CvFpsCalc(buffer_len=10) # tính Fps: số khung hình mỗi giây
    
    # Coordinate history
    history_length = 16
    point_history = deque(maxlen=history_length) # tạo hàng đợi deque  lưu trữ lịch sử tọa độ
    
    # Finger gesture history 
    finger_gesture_history = deque(maxlen=history_length)
    
    mode = 0

    while True:
        fps = csvFpsCalc.get()
        
        # Process key (ESC: end)
        key = cv.waitKey(10)
        if key == 27:
            break
        number, mode = select_mode(key, mode) # xem xem mã ASCII là bao nhiêu, khớp với ký tự nào
        
        # Camera capture
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image) # tạo bản sao 
        
        # Detection implementation
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB) # chuyển sang RGB 
        
        image.flags.writeable = False # khóa ảnh 
        results = hands.process(image) # dự đoán 
        image.flags.writeable = True # Dừng khóa ảnh 
        
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                
                # Bouding box calculation
                brect = calc_bouding_rect(debug_image, hand_landmarks) # Từ tất cả các điểm trên tay, vẽ bouding box hình cn khớp vừa đủ các điểm đó

                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)
                # # print(pre_processed_point_history_list, len(pre_processed_point_history_list))
                
                # Write to the dataset file
                logging_csv(number, mode, pre_processed_landmark_list, 
                            pre_processed_point_history_list) # thực hiện với từng tác vụ k, h, n
                
                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 0: # Point gesture == Nothing
                    point_history.append(landmark_list[8]) # lưu 16 thông tin lịch sử theo đầu ngón trỏ
                else:
                    point_history.append([0, 0]) # lưu theo gốc bàn tay
                # # print(point_history, len(point_history))
                
                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list) # tim cu chi tay ti le cao nhat theo lich su 
                
                
                # Calculates the gesture IDs in the lastest dedection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common() # đếm số lần xuất hiện mỗi cử chỉ
                
                # Draw part
                debug_image = draw_bounding_rect(
                    use_brect, 
                    debug_image, 
                    brect) # vẽ bouding box
                debug_image = draw_landmarks(
                    debug_image, 
                    landmark_list) # vẽ đường ngón tay và lòng bàn tay
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id], # dự đoán cử chỉ tay theo dự đoán
                    point_history_classifier_labels[most_common_fg_id[0][0]] # labels cử chỉ tay most theo lịch sử
                )
                global gesture
                gesture = keypoint_classifier_labels[hand_sign_id]
        else:
            point_history.append([0, 0])
            
        debug_image = draw_point_history(debug_image, point_history) # Dựa trên lịch sử, nếu là con trỏ thì vẽ hình tròn đầu ngón tay  
        debug_image = draw_info(debug_image, fps, mode, number) # vẽ khu vực FPS
        
        print(gesture, 1)
        # Screen reflection
        cv.imshow('Hand Gesture Recognition', debug_image)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
   
