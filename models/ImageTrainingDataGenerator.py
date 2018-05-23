from facelib import FaceType
from facelib import LandmarksProcessor
import cv2
import numpy as np
from models import TrainingDataGeneratorBase
from utils import image_utils
from utils import random_utils
from enum import IntEnum
from models import TrainingDataType

class ImageTrainingDataGenerator(TrainingDataGeneratorBase):
    class SampleTypeFlags(IntEnum):
        SOURCE               = 0x000001,
        WARPED               = 0x000002,
        WARPED_TRANSFORMED   = 0x000004,
        TRANSFORMED          = 0x000008,

        MODE_BGR         = 0x000100,  #BGR
        MODE_G           = 0x000200,  #Grayscale
        MODE_GGG         = 0x000400,  #3xGrayscale 
        MODE_M           = 0x000800,  #mask only
        MODE_BGR_SHUFFLE = 0x001000,  #BGR shuffle
        
    #overrided
    def onInitialize(self, random_flip=False, normalize_tanh=False, output_sample_types=[], **kwargs):
        self.random_flip = random_flip        
        self.normalize_tanh = normalize_tanh
        self.output_sample_types = output_sample_types
        
        allowed_types = [TrainingDataType.RAW_SRC, TrainingDataType.RAW_DST]
        if self.trainingdatatype not in allowed_types:
            raise Exception ('unsupported %s for ImageTrainingDataGenerator. Allowed types: %s' % (self.trainingdatatype, allowed_types) )
            
    #overrided
    def onProcessSample(self, sample, debug):
        source = sample.load_bgr()
        h,w,c = source.shape

        params = image_utils.gen_warp_params(source, self.random_flip)

        images = [None for _ in range(4)]
            
        outputs = []        
        for t,size in self.output_sample_types:
            if t & self.SampleTypeFlags.SOURCE != 0:
                img_type = 0
            elif t & self.SampleTypeFlags.WARPED != 0:
                img_type = 1
            elif t & self.SampleTypeFlags.WARPED_TRANSFORMED != 0:
                img_type = 2
            elif t & self.SampleTypeFlags.TRANSFORMED != 0:
                img_type = 3
            else:
                raise ValueError ('expected SampleTypeFlags type')
                
            if images[img_type] is None:
                img = source
                images[img_type] = image_utils.warp_by_params (params, img, (img_type==1 or img_type==2), (img_type==2 or img_type==3), img_type != 0)
                
            img = images[img_type]

            img = cv2.resize( img, (size,size), cv2.INTER_LANCZOS4 )
                
            img_bgr  = img[...,0:3]
            img_mask = img[...,3:4]
 
            if t & self.SampleTypeFlags.MODE_BGR != 0:
                img = img
            elif t & self.SampleTypeFlags.MODE_BGR_SHUFFLE != 0:
                img_bgr = np.take (img_bgr, np.random.permutation(img_bgr.shape[-1]), axis=-1)
                img = np.concatenate ( (img_bgr,img_mask) , -1 )
            elif t & self.SampleTypeFlags.MODE_G != 0:
                img = np.concatenate ( (np.expand_dims(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY),-1),img_mask) , -1 )
            elif t & self.SampleTypeFlags.MODE_GGG != 0:
                img = np.concatenate ( ( np.repeat ( np.expand_dims(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY),-1), (3,), -1), img_mask), -1)
            elif t & self.SampleTypeFlags.MODE_M != 0:
                if mask_type== 0:
                    raise ValueError ('no mask mode defined')
                img = img_mask
            else:
                raise ValueError ('expected SampleTypeFlags mode')
     
            if not debug and self.normalize_tanh:
                img = img * 2.0 - 1.0
                
            outputs.append ( img )

        if debug:
            result = ()

            for output in outputs:
                if output.shape[2] < 4:
                    result += (output,)
                elif output.shape[2] == 4:
                    result += (output[...,0:3]*output[...,3:4],)

            return result            
        else:
            return outputs
   
