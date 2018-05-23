from enum import IntEnum
import cv2
import numpy as np
from random import randint
from facelib import FaceType


class TrainingDataType(IntEnum):
    IMAGE_SRC = 0 #raw image
    IMAGE_DST = 1 #
    FACE_SRC = 2 #raw aligned face image unsorted
    FACE_DST = 3    
    FACE_SRC_WITH_NEAREST = 4 # as FACE_SRC but samples can return get_random_nearest_target_sample()
    FACE_SRC_ONLY_10_NEAREST_TO_DST_ONLY_1 = 5 #currently unused, idea to get only 10 nearest samples to target one face for PHOTO256 model
    FACE_DST_ONLY_1 = 6  #currently unused, idea to get only 10 nearest samples to target one face for PHOTO256 model
    FACE_SRC_YAW_SORTED = 7 # sorted by yaw
    FACE_DST_YAW_SORTED = 8 # sorted by yaw
    FACE_SRC_YAW_SORTED_AS_DST = 9 #sorted by yaw but included only yaws which exist in DST_YAW_SORTED also automatic mirrored
    FACE_SRC_YAW_SORTED_AS_DST_WITH_NEAREST = 10 #same as SRC_YAW_SORTED_AS_DST but samples can return get_random_nearest_target_sample()
    
    QTY = 11
    
    
class TrainingDataSample(object):

    def __init__(self, filename=None, face_type=None, shape=None, landmarks=None, yaw=None, mirror=None, nearest_target_list=None):
        self.filename = filename
        self.face_type = face_type
        self.shape = shape
        self.landmarks = np.array(landmarks) if landmarks is not None else None
        self.yaw = yaw
        self.mirror = mirror
        self.nearest_target_list = nearest_target_list
    
    def copy_and_set(self, filename=None, face_type=None, shape=None, landmarks=None, yaw=None, mirror=None, nearest_target_list=None):
        return TrainingDataSample( 
            filename=filename if filename is not None else self.filename, 
            face_type=face_type if face_type is not None else self.face_type, 
            shape=shape if shape is not None else self.shape, 
            landmarks=landmarks if landmarks is not None else self.landmarks.copy(), 
            yaw=yaw if yaw is not None else self.yaw, 
            mirror=mirror if mirror is not None else self.mirror, 
            nearest_target_list=nearest_target_list if nearest_target_list is not None else self.nearest_target_list)
    
    def load_bgr(self):
        img = cv2.imread (self.filename).astype(np.float32) / 255.0
        if self.mirror:
            img = img[:,::-1].copy()
        return img

    def get_random_nearest_target_sample(self):
        if self.nearest_target_list is None:
            return None
        return self.nearest_target_list[randint (0, len(self.nearest_target_list)-1)]