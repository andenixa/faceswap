import copy
'''
You can implement your own Converter, check example ConverterMasked.py
'''

class ConverterBase(object):

    #overridable
    def __init__(self, predictor):
        self.predictor = predictor
        
    #overridable
    def is_avatar_mode(self):
        #if true, convert_avatar will be called
        return False
        
    #overridable
    def convert (self, img_bgr, img_face_landmarks, debug):
        #return float32 image        
        #if debug , return tuple ( images of any size and channels, ...)
        return image
        
    #overridable
    def convert_avatar (self, img_bgr, img_face_landmarks, debug):
        #return float32 image        
        #if debug , return tuple ( images of any size and channels, ...)
        return image
        
    #overridable
    def dummy_predict(self):
        #do dummy predict here
        pass

    def copy(self):
        return copy.copy(self)
        
    def copy_and_set_predictor(self, predictor):
        result = self.copy()
        result.predictor = predictor
        return result