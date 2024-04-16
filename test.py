import cv2 as cv
import mediapipe as mp

from caculate import *
from draw import *
from process import *

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
