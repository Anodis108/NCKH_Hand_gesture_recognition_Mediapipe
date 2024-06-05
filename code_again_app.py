#  Returns:
#  |        A NamedTuple object with the following fields:
#  |          1) a "multi_hand_landmarks" field that contains the hand landmarks on
#  |             each detected hand.
#  |          2) a "multi_hand_world_landmarks" field that contains the hand landmarks
#  |             on each detected hand in real-world 3D coordinates that are in meters
#  |             with the origin at the hand's approximate geometric center.
#  |          3) a "multi_handedness" field that contains the handedness (left v.s.
#  |             right hand) of the detected hand

import threading # chạy đa luồng

import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

gesture = None
con_tro = None

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args

def main():
    # Argument parsing################################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True
    
    # Camera preparation################################################################################
    cap = cv.VideoCapture(cap_device) # cap_device = 0 -> webcam
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    
    # Model load################################################################################ 
    mp_hands = mp.solutions.hands # tạo 1 đối tượng hand với 2 thuộc tính Hand và Handlandmark
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode, # False là chạy webcam
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    
    keypoint_classifier = KeyPointClassifier() # phân lớp

    point_history_classifier = PointHistoryClassifier() # phân loại chuỗi lịch sử điểm chỉ tay 
    
    # Read labels################################################################################
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
        
    # FPS Measurement################################################################################
    csvFpsCalc = CvFpsCalc(buffer_len=10) # tính Fps: số khung hình mỗi giây
    
    # Coordinate history################################################################################
    history_length = 16
    point_history = deque(maxlen=history_length) # tạo hàng đợi deque  lưu trữ lịch sử tọa độ
    
    # Finger gesture history ################################################################################
    finger_gesture_history = deque(maxlen=history_length)
    
    mode = 0

    while True:
        fps = csvFpsCalc.get()
        
        # Process key (ESC: end)################################################################################
        key = cv.waitKey(10)
        if key == 27:
            break
        number, mode = select_mode(key, mode) # xem xem mã ASCII là bao nhiêu, khớp với ký tự nào
        
        # Camera capture################################################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image) # tạo bản sao 
        
        # Detection implementation################################################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB) # chuyển sang RGB 
        
        image.flags.writeable = False # khóa ảnh 
        results = hands.process(image) # dự đoán 
        image.flags.writeable = True # Dừng khóa ảnh 
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                
                # Bouding box calculation################################################################################
                brect = calc_bouding_rect(debug_image, hand_landmarks) # Từ tất cả các điểm trên tay, vẽ bouding box hình cn khớp vừa đủ các điểm đó

                # Landmark calculation################################################################################
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # conversion to relative coordinates / normalized coordinates################################################################################
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)
                # # print(pre_processed_point_history_list, len(pre_processed_point_history_list))
                
                # Write to the dataset file################################################################################
                logging_csv(number, mode, pre_processed_landmark_list, 
                            pre_processed_point_history_list) # thực hiện với từng tác vụ k, h, n
                
                # Hand sign classification################################################################################
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 0: # Point gesture == Nothing
                    point_history.append(landmark_list[8]) # lưu 16 thông tin lịch sử theo đầu ngón trỏ vì đây là cử chỉ trỏ
                else:
                    point_history.append([0, 0]) # lưu theo gốc bàn tay
                # # print(point_history, len(point_history))
                
                # Finger gesture classification################################################################################
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list) # tim cu chi tay ti le cao nhat theo lich su 
                
                
                # Calculates the gesture IDs in the lastest dedection################################################################################
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common() # đếm số lần xuất hiện mỗi cử chỉ
                
                # Draw part################################################################################
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
                global gesture, con_tro
                gesture = keypoint_classifier_labels[hand_sign_id]
                con_tro = tuple(landmark_list[8])
        else:
            point_history.append([0, 0])
            
        debug_image = draw_point_history(debug_image, point_history) # Dựa trên lịch sử, nếu là con trỏ thì vẽ hình tròn đầu ngón tay  
        debug_image = draw_info(debug_image, fps, mode, number) # vẽ khu vực FPS
        
            
        # Screen reflection################################################################################
        cv.imshow('Hand Gesture Recognition', debug_image)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
    
def test():
    # test with pictures
    path = 'install.png'
    images = cv.imread(path)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    # help(mp_hands.Hands)
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.7) as hands:
        results = hands.process(cv.flip(cv.cvtColor(images, cv.COLOR_BGR2RGB), 1))

        
        # Print handedness (left v.s. right hand).
        # print(f'Handedness of picture:')
        # print(results.multi_handedness)

        # print(f'multi_hand_landmarks of picture:')
        # print(results.multi_hand_landmarks)
        
        # print(f'multi_hand_world_landmarks of picture:')
        # print(results.multi_hand_world_landmarks)
        

        # Draw hand landmarks of each hand.
        print(f'Hand landmarks of picture:')
        image_hight, image_width, _ = images.shape
        annotated_image = cv.flip(images.copy(), 1)
        
        cv.imshow('Ban đầu', annotated_image)
        cv.waitKey(0)
        cv.destroyAllWindows()
        
        for hand_landmarks in results.multi_hand_landmarks:
        #   for i in mp_hands.HandLandmark: # i gồm 3 thành phần x, y, z
        #       print(hand_landmarks.landmark[i])
        #   print(
        #       f'Index finger tip coordinate: (',
        #       f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, ' # truy cập đến phần tử đầu ngón trỏ
        #       f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight})'
        #   )
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            brect = calc_bouding_rect(annotated_image, hand_landmarks)
            # print(brect)
            
            landmark_list = calc_landmark_list(annotated_image, hand_landmarks)
            # print(landmark_list)
            
            pre_preocessed_landmark_list = pre_process_landmark(landmark_list)
            pre_processed_point_history_list = pre_process_point_history(annotated_image, deque(maxlen=16))
            
            
            
        cv.imshow('Sau khi vẽ', cv.flip(annotated_image, 1))
        cv.waitKey(0)
        cv.destroyAllWindows()

def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57: # 0 ~ 9
        number = key - 48    
    if key == 110:   # n
        mode = 0
    elif key == 107: # k
        mode = 1 
    elif key == 104: # h
        mode = 2
    return number, mode

def calc_bouding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    
    landmark_array = np.empty((0, 2), int)
    
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1) # tính xmin và y min để vẽ bouding box cho cả bàn tay 
        
        landmark_point = [np.array((landmark_x, landmark_y))]
        
        landmark_array = np.append(landmark_array, landmark_point, axis=0)
    
    x, y, w, h = cv.boundingRect(landmark_array)
    
    return [x, y, x + w, y + h, landmark_array]   

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    
    landmark_point = []
    
    # keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x,landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    
    # Convert to relative coordinates (Lấy mốc là gốc bàn tay, tính hiệu x, y từ điểm này đến mốc)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
        
    # Convert to a one-dimensional list # chải phẳng list 2 chiều về 1 chiều
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))        
    # Normalization ( chuẩn hóa theo trị tuyệt đối giá trị lớn nhất)
    max_value = max(list(map(abs, temp_landmark_list)))
    
    def normalize_(n):
        return n / max_value
    
    temp_landmark_list = list(map(normalize_, temp_landmark_list)) # chuẩn hóa về tỉ lệ từ 0 -> 1 so với max value
    return temp_landmark_list

def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]
    
    temp_point_history = copy.deepcopy(point_history)
    # Convert to relative coordinates
    base_x, base_y = 0, 0
    
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height
        
    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history)
    )

    return temp_point_history
   
def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'  
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <=9):
        csv_path = 'model/point_history_classifier/point_history.csv'  
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
       
def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image
 
def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0: # vẽ đốt các ngón
        # # Thumb : vẽ đót ngón cái
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6) # vẽ màu đen
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2) # vẽ màu trắng
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points # vẽ điểm trên bàn tay
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # WRIST 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # THUMB_CMC 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # THUMB_MCP 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # THUMB_IP 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # THUMB_TIP 
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # INDEX_FINGER_MCP 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # INDEX_FINGER_PIP 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # INDEX_FINGER_DIP 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # INDEX_FINGER_TIP 
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # MIDDLE_FINGER_MCP 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # MIDDLE_FINGER_PIP 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # MIDDLE_FINGER_DIP 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # MIDDLE_FINGER_TIP 
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # RING_FINGER_MCP 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # RING_FINGER_PIP 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # RING_FINGER_DIP 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # RING_FINGER_TIP 
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # PINKY_MCP 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # PINKY_PIP   
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # PINKY_DIP 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # PINKY_TIP  
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image 

def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1) # Vẽ khung viền tiêu đề

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)
        # print(finger_gesture_text)

    return image

def draw_point_history(image, point_history): # nếu là con trỏ thì vẽ hình tròn
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image

def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image

import pygame

import random
import math
import game.block as block
import game.constants as constants

class Tetris(object):
    """
    The class with implementation of tetris game logic.
    """

    def __init__(self,bx,by):
        """
        Initialize the tetris object.

        Parameters:
            - bx - number of blocks in x
            - by - number of blocks in y
        """
        # Compute the resolution of the play board based on the required number of blocks.
        self.resx = bx*constants.BWIDTH+2*constants.BOARD_HEIGHT+constants.BOARD_MARGIN
        self.resy = by*constants.BHEIGHT+2*constants.BOARD_HEIGHT+constants.BOARD_MARGIN
        # Prepare the pygame board objects (white lines)
        self.board_up    = pygame.Rect(0,constants.BOARD_UP_MARGIN,self.resx,constants.BOARD_HEIGHT)
        self.board_down  = pygame.Rect(0,self.resy-constants.BOARD_HEIGHT,self.resx,constants.BOARD_HEIGHT)
        self.board_left  = pygame.Rect(0,constants.BOARD_UP_MARGIN,constants.BOARD_HEIGHT,self.resy)
        self.board_right = pygame.Rect(self.resx-constants.BOARD_HEIGHT,constants.BOARD_UP_MARGIN,constants.BOARD_HEIGHT,self.resy)
        # List of used blocks
        self.blk_list    = []
        # Compute start indexes for tetris blocks
        self.start_x = math.ceil(self.resx/2.0)
        self.start_y = constants.BOARD_UP_MARGIN + constants.BOARD_HEIGHT + constants.BOARD_MARGIN
        # Blocka data (shapes and colors). The shape is encoded in the list of [X,Y] points. Each point
        # represents the relative position. The true/false value is used for the configuration of rotation where
        # False means no rotate and True allows the rotation.
        self.block_data = (
            ([[0,0],[1,0],[2,0],[3,0]],constants.RED,True),     # I block 
            ([[0,0],[1,0],[0,1],[-1,1]],constants.GREEN,True),  # S block 
            ([[0,0],[1,0],[2,0],[2,1]],constants.BLUE,True),    # J block
            ([[0,0],[0,1],[1,0],[1,1]],constants.ORANGE,False), # O block
            ([[-1,0],[0,0],[0,1],[1,1]],constants.GOLD,True),   # Z block
            ([[0,0],[1,0],[2,0],[1,1]],constants.PURPLE,True),  # T block
            ([[0,0],[1,0],[2,0],[0,1]],constants.CYAN,True),    # J block
        )
        # Compute the number of blocks. When the number of blocks is even, we can use it directly but 
        # we have to decrese the number of blocks in line by one when the number is odd (because of the used margin).
        self.blocks_in_line = bx if bx%2 == 0 else bx-1
        self.blocks_in_pile = by
        # Score settings
        self.score = 0
        # Remember the current speed 
        self.speed = 1
        # The score level threshold
        self.score_level = constants.SCORE_LEVEL

    def apply_action(self):
        """
        Get the event from the event queue and run the appropriate 
        action.
        """
        # Take the event from the event queue.
        for ev in pygame.event.get():
            # Check if the close button was fired.
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.unicode == 'q'):
                self.done = True
            # Detect the key evevents for game control.
            if gesture is not None:
                print(gesture, 1)
                if gesture == 'Down':
                    self.active_block.move(0,constants.BHEIGHT)
                if gesture == 'Left':
                    self.active_block.move(-constants.BWIDTH,0)
                if gesture == 'Right':
                    self.active_block.move(constants.BWIDTH,0)
                if gesture == 'Rotate':
                    self.active_block.rotate()
                if gesture == 'Click':
                    self.pause()
            
       
            # Detect if the movement event was fired by the timer.
            if ev.type == constants.TIMER_MOVE_EVENT:
                self.active_block.move(0,constants.BHEIGHT)
       
    def pause(self):
        """
        Pause the game and draw the string. This function
        also calls the flip function which draws the string on the screen.
        """
        # Draw the string to the center of the screen.
        self.print_center(["PAUSE","Press \"p\" to continue"])
        pygame.display.flip()
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_p:
                    return
       
    def set_move_timer(self):
        """
        Setup the move timer to the 
        """
        # Setup the time to fire the move event. Minimal allowed value is 1
        speed = math.floor(constants.MOVE_TICK / self.speed)
        speed = max(1,speed)
        pygame.time.set_timer(constants.TIMER_MOVE_EVENT,speed)
 
    def run(self):
        # Initialize the game (pygame, fonts)
        pygame.init()
        pygame.font.init()
        self.myfont = pygame.font.SysFont(pygame.font.get_default_font(),constants.FONT_SIZE)
        self.screen = pygame.display.set_mode((self.resx,self.resy))
        pygame.display.set_caption("Tetris")
        # Setup the time to fire the move event every given time
        self.set_move_timer()
        # Control variables for the game. The done signal is used 
        # to control the main loop (it is set by the quit action), the game_over signal
        # is set by the game logic and it is also used for the detection of "game over" drawing.
        # Finally the new_block variable is used for the requesting of new tetris block. 
        self.done = False
        self.game_over = False
        self.new_block = True
        # Print the initial score
        self.print_status_line()
        while not(self.done) and not(self.game_over):
            # Get the block and run the game logic
            self.get_block()
            self.game_logic()
            self.draw_game()
            print(gesture, 1)
        # Display the game_over and wait for a keypress
        if self.game_over:
            self.print_game_over()
        # Disable the pygame stuff
        pygame.font.quit()
        pygame.display.quit()        
   
    def print_status_line(self):
        """
        Print the current state line
        """
        string = ["SCORE: {0}   SPEED: {1}x".format(self.score,self.speed)]
        self.print_text(string,constants.POINT_MARGIN,constants.POINT_MARGIN)        

    def print_game_over(self):
        """
        Print the game over string.
        """
        # Print the game over text
        self.print_center(["Game Over","Press \"q\" to exit"])
        # Draw the string
        pygame.display.flip()
        # Wait untill the space is pressed
        while True: 
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.unicode == 'q'):
                    return

    def print_text(self,str_lst,x,y):
        """
        Print the text on the X,Y coordinates. 

        Parameters:
            - str_lst - list of strings to print. Each string is printed on new line.
            - x - X coordinate of the first string
            - y - Y coordinate of the first string
        """
        prev_y = 0
        for string in str_lst:
            size_x,size_y = self.myfont.size(string)
            txt_surf = self.myfont.render(string,False,(255,255,255))
            self.screen.blit(txt_surf,(x,y+prev_y))
            prev_y += size_y 

    def print_center(self,str_list):
        """
        Print the string in the center of the screen.
        
        Parameters:
            - str_lst - list of strings to print. Each string is printed on new line.
        """
        max_xsize = max([tmp[0] for tmp in map(self.myfont.size,str_list)])
        self.print_text(str_list,self.resx/2-max_xsize/2,self.resy/2)

    def block_colides(self):
        """
        Check if the block colides with any other block.

        The function returns True if the collision is detected.
        """
        for blk in self.blk_list:
            # Check if the block is not the same
            if blk == self.active_block:
                continue 
            # Detect situations
            if(blk.check_collision(self.active_block.shape)):
                return True
        return False

    def game_logic(self):
        """
        Implementation of the main game logic. This function detects colisions
        and insertion of new tetris blocks.
        """
        # Remember the current configuration and try to 
        # apply the action
        self.active_block.backup()
        self.apply_action()
        # Border logic, check if we colide with down border or any
        # other border. This check also includes the detection with other tetris blocks. 
        down_board  = self.active_block.check_collision([self.board_down])
        any_border  = self.active_block.check_collision([self.board_left,self.board_up,self.board_right])
        block_any   = self.block_colides()
        # Restore the configuration if any collision was detected
        if down_board or any_border or block_any:
            self.active_block.restore()
        # So far so good, sample the previous state and try to move down (to detect the colision with other block). 
        # After that, detect the the insertion of new block. The block new block is inserted if we reached the boarder
        # or we cannot move down.
        self.active_block.backup()
        self.active_block.move(0,constants.BHEIGHT)
        can_move_down = not self.block_colides()  
        self.active_block.restore()
        # We end the game if we are on the respawn and we cannot move --> bang!
        if not can_move_down and (self.start_x == self.active_block.x and self.start_y == self.active_block.y):
            self.game_over = True
        # The new block is inserted if we reached down board or we cannot move down.
        if down_board or not can_move_down:     
            # Request new block
            self.new_block = True
            # Detect the filled line and possibly remove the line from the 
            # screen.
            self.detect_line()   
 
    def detect_line(self):
        """
        Detect if the line is filled. If yes, remove the line and
        move with remaining bulding blocks to new positions.
        """
        # Get each shape block of the non-moving tetris block and try
        # to detect the filled line. The number of bulding blocks is passed to the class
        # in the init function.
        for shape_block in self.active_block.shape:
            tmp_y = shape_block.y
            tmp_cnt = self.get_blocks_in_line(tmp_y)
            # Detect if the line contains the given number of blocks
            if tmp_cnt != self.blocks_in_line:
                continue 
            # Ok, the full line is detected!     
            self.remove_line(tmp_y)
            # Update the score.
            self.score += self.blocks_in_line * constants.POINT_VALUE 
            # Check if we need to speed up the game. If yes, change control variables
            if self.score > self.score_level:
                self.score_level *= constants.SCORE_LEVEL_RATIO
                self.speed       *= constants.GAME_SPEEDUP_RATIO
                # Change the game speed
                self.set_move_timer()

    def remove_line(self,y):
        """
        Remove the line with given Y coordinates. Blocks below the filled
        line are untouched. The rest of blocks (yi > y) are moved one level done.

        Parameters:
            - y - Y coordinate to remove.
        """ 
        # Iterate over all blocks in the list and remove blocks with the Y coordinate.
        for block in self.blk_list:
            block.remove_blocks(y)
        # Setup new block list (not needed blocks are removed)
        self.blk_list = [blk for blk in self.blk_list if blk.has_blocks()]

    def get_blocks_in_line(self,y):
        """
        Get the number of shape blocks on the Y coordinate.

        Parameters:
            - y - Y coordinate to scan.
        """
        # Iteraveovel all block's shape list and increment the counter
        # if the shape block equals to the Y coordinate.
        tmp_cnt = 0
        for block in self.blk_list:
            for shape_block in block.shape:
                tmp_cnt += (1 if y == shape_block.y else 0)            
        return tmp_cnt

    def draw_board(self):
        """
        Draw the white board.
        """
        pygame.draw.rect(self.screen,constants.WHITE,self.board_up)
        pygame.draw.rect(self.screen,constants.WHITE,self.board_down)
        pygame.draw.rect(self.screen,constants.WHITE,self.board_left)
        pygame.draw.rect(self.screen,constants.WHITE,self.board_right)
        # Update the score         
        self.print_status_line()

    def get_block(self):
        """
        Generate new block into the game if is required.
        """
        if self.new_block:
            # Get the block and add it into the block list(static for now)
            tmp = random.randint(0,len(self.block_data)-1)
            data = self.block_data[tmp]
            self.active_block = block.Block(data[0],self.start_x,self.start_y,self.screen,data[1],data[2])
            self.blk_list.append(self.active_block)
            self.new_block = False

    def draw_game(self):
        """
        Draw the game screen.
        """
        # Clean the screen, draw the board and draw
        # all tetris blocks
        self.screen.fill(constants.BLACK)
        self.draw_board()
        for blk in self.blk_list:
            blk.draw()
        # Draw the screen buffer
        pygame.display.flip()

if __name__ == '__main__':
    # test()
    p1 = threading.Thread(target=main, args=()) # thêm dấu (,) ở đuôi args vì nó yêu cầu đuôi là một iterable
    p1.start()
    # p2 = threading.Thread(target=Tetris(16, 30).run, args=())
    # p2.start()

