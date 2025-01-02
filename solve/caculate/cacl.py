import cv2 as cv
import numpy as np

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


   

       