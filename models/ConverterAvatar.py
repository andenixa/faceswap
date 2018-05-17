from models import ConverterBase
from facelib import LandmarksProcessor
from facelib import FaceType

import cv2
import numpy as np
from utils import image_utils

'''
predictor_func: 
    input:  [predictor_input_size, predictor_input_size, G]
    output: [predictor_input_size, predictor_input_size, BGR]
'''

class ConverterAvatar(ConverterBase):

    #override
    def __init__(self,  predictor,
                        predictor_input_size=0, 
                        output_size=0,               
                        **in_options):
                        
        super().__init__(predictor)
         
        self.predictor_input_size = predictor_input_size
        self.output_size = output_size   
  
    #override
    def is_avatar_mode(self):
        return True
        
    #override
    def dummy_predict(self):
        self.predictor ( np.zeros ( (self.predictor_input_size, self.predictor_input_size,1), dtype=np.float32) )
        
    #override
    def convert_avatar (self, img_bgr, img_face_landmarks, debug):
        img_size = img_bgr.shape[1], img_bgr.shape[0]

        face_mat = LandmarksProcessor.get_transform_mat (img_face_landmarks, self.output_size, face_type=FaceType.HALF)
        dst_face_bgr = cv2.warpAffine( img_bgr,         face_mat, (self.output_size, self.output_size) )
 
        predictor_input_g = cv2.cvtColor(dst_face_bgr, cv2.COLOR_BGR2GRAY)
        predictor_input_g = cv2.resize (predictor_input_g, (self.predictor_input_size,self.predictor_input_size) )
        predictor_input_g = np.expand_dims(predictor_input_g,-1)
        predicted_bgr = self.predictor ( predictor_input_g )
        
        output = cv2.resize ( predicted_bgr, (self.output_size, self.output_size) )
        if debug:
            return (output,)
        return output
