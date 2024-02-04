#  Returns:
#  |        A NamedTuple object with the following fields:
#  |          1) a "multi_hand_landmarks" field that contains the hand landmarks on
#  |             each detected hand.
#  |          2) a "multi_hand_world_landmarks" field that contains the hand landmarks
#  |             on each detected hand in real-world 3D coordinates that are in meters
#  |             with the origin at the hand's approximate geometric center.
#  |          3) a "multi_handedness" field that contains the handedness (left v.s.
#  |             right hand) of the detected hand

import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp
import tensorflow as tf

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

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
    # Argument parsing
    args = get_args()

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
    
    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier()
    
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
                
        # Screen reflection
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
    print(mode)
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
    print(landmarks.landmark)
    return landmark_point


if __name__ == '__main__':
    # main()
    test()

