from facelib import FaceType
from facelib import LandmarksProcessor
import cv2
import numpy as np
from models import TrainingDataGeneratorBase
from utils import image_utils
from utils import random_utils
from enum import IntEnum
'''
generates full face images 
    random_flip 
    output_sample_types_flags is TrainingDataGenerator.SampleTypeFlags
'''   
class TrainingDataGenerator(TrainingDataGeneratorBase):
    class SampleTypeFlags(IntEnum):
        SOURCE               = 0x000001,
        WARPED               = 0x000002,
        WARPED_TRANSFORMED   = 0x000004,
        TRANSFORMED          = 0x000008,
   
        HALF_FACE   = 0x000010,
        FULL_FACE   = 0x000020,
        HEAD_FACE   = 0x000040,
        AVATAR_FACE = 0x000080,
        
        MODE_BGR    = 0x000100,  #BGR
        MODE_G      = 0x000200,  #Grayscale
        MODE_GGG    = 0x000400,  #3xGrayscale 
        MODE_M      = 0x000800,  #mask only

        MASK_FULL   = 0x100000,
        MASK_EYES   = 0x200000,

        SIZE_64     = 0x010000,
        SIZE_128    = 0x020000,
        SIZE_256    = 0x040000,
        
    #overrided
    def onInitialize(self, random_flip=False, output_sample_types_flags=[], **kwargs):
        self.random_flip = random_flip        
        self.output_sample_types_flags = output_sample_types_flags
        
    #overrided
    def onProcessSample(self, sample, debug):
        source = sample.load_bgr()

        if debug:
            LandmarksProcessor.draw_landmarks (source, sample.landmarks, (0, 1, 0))

        #warped_bgrm, target_bgrm, target_bgrm_untransformed = TrainingDataGenerator.warp (s_bgrm, sample.landmarks)

        params = TrainingDataGenerator.gen_params(source, self.random_flip)

        warped = None
        warped_transformed = None
        transformed = None
        
        full_mask = None
        full_mask_warped = None
        full_mask_warped_transformed = None
        full_mask_transformed = None

        images = [[None]*3 for _ in range(4)]
            
        outputs = []        
        for t in self.output_sample_types_flags:
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
                
            mask_type = 0
            if t & self.SampleTypeFlags.MASK_FULL != 0:
                mask_type = 1               
            elif t & self.SampleTypeFlags.MASK_EYES != 0:
                mask_type = 2
                    
            if images[img_type][mask_type] is None:
                img = source
                if mask_type == 1:
                    img = np.concatenate( (img, LandmarksProcessor.get_image_hull_mask (source, sample.landmarks) ), -1 )                    
                elif mask_type == 2:
                    pass
                    
                images[img_type][mask_type] = TrainingDataGenerator.warp (params, img, (img_type==1 or img_type==2), (img_type==2 or img_type==3), img_type != 0)
                
            img = images[img_type][mask_type]
            
            if t & self.SampleTypeFlags.SIZE_64 != 0:
                size = 64
            elif t & self.SampleTypeFlags.SIZE_128 != 0:
                size = 128
            elif t & self.SampleTypeFlags.SIZE_256 != 0:
                size = 256
            else:
                raise ValueError ('expected SampleTypeFlags size')
                
            if t & self.SampleTypeFlags.HALF_FACE != 0:
                target_face_type = FaceType.HALF            
            elif t & self.SampleTypeFlags.FULL_FACE != 0:
                target_face_type = FaceType.FULL
            elif t & self.SampleTypeFlags.HEAD_FACE != 0:
                target_face_type = FaceType.HEAD
            elif t & self.SampleTypeFlags.AVATAR_FACE != 0:
                target_face_type = FaceType.AVATAR
            else:
                raise ValueError ('expected SampleTypeFlags face type')
                
            if target_face_type > sample.face_type:
                raise Exception ('sample %s type %s does not match model requirement %s. Consider extract necessary type of faces.' % (sample.filename, sample.face_type, target_face_type) )

            img = cv2.warpAffine( img, LandmarksProcessor.get_transform_mat (sample.landmarks, size, target_face_type), (size,size), flags=cv2.INTER_LANCZOS4 )
 
            img_bgr  = img[...,0:3]
            img_mask = img[...,3:4]
 
            if t & self.SampleTypeFlags.MODE_BGR != 0:
                img = img
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
     
            outputs.append ( img )

        if debug:
            result = (source,)

            for output in outputs:
                if output.shape[2] < 4:
                    result += (output,)
                elif output.shape[2] == 4:
                    result += (output[...,0:3]*output[...,3:4],)

            return result            
        else:
            return outputs
   
    @staticmethod
    def gen_params (source, flip):
        h,w,c = source.shape
        if (h != w) or (w != 64 and w != 128 and w != 256 and w != 512 and w != 1024):
            raise ValueError ('TrainingDataGenerator accepts only square power of 2 images.')
            
        rotation = np.random.uniform(-10, 10)
        scale = np.random.uniform(1 - 0.05, 1 + 0.05)
        tx = np.random.uniform(-0.05, 0.05)
        ty = np.random.uniform(-0.05, 0.05)    
     
        #random warp by grid
        cell_size = [ w // (2**i) for i in range(1,5) ] [ np.random.randint(4) ]
        cell_count = w // cell_size + 1
        
        grid_points = np.linspace( 0, w, cell_count)
        mapx = np.broadcast_to(grid_points, (cell_count, cell_count)).copy()
        mapy = mapx.T
        
        mapx[1:-1,1:-1] = mapx[1:-1,1:-1] + random_utils.random_normal( size=(cell_count-2, cell_count-2) )*(cell_size*0.24)
        mapy[1:-1,1:-1] = mapy[1:-1,1:-1] + random_utils.random_normal( size=(cell_count-2, cell_count-2) )*(cell_size*0.24)

        half_cell_size = cell_size // 2
        
        mapx = cv2.resize(mapx, (w+cell_size,)*2 )[half_cell_size:-half_cell_size-1,half_cell_size:-half_cell_size-1].astype(np.float32)
        mapy = cv2.resize(mapy, (w+cell_size,)*2 )[half_cell_size:-half_cell_size-1,half_cell_size:-half_cell_size-1].astype(np.float32)
        
        #random transform
        random_transform_mat = cv2.getRotationMatrix2D((w // 2, w // 2), rotation, scale)
        random_transform_mat[:, 2] += (tx*w, ty*w)
        
        params = dict()
        params['mapx'] = mapx
        params['mapy'] = mapy
        params['rmat'] = random_transform_mat
        params['w'] = w        
        params['flip'] = flip and np.random.randint(10) < 4
                
        return params
        
    @staticmethod    
    def warp (params, img, warp, transform, flip):
        if warp:
            img = cv2.remap(img, params['mapx'], params['mapy'], cv2.INTER_LANCZOS4 )
        if transform:
            img = cv2.warpAffine( img, params['rmat'], (params['w'], params['w']), borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_LANCZOS4 )            
        if flip and params['flip']:
            img = img[:,::-1,:]
        return img
