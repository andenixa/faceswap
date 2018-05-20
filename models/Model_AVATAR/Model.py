from models import ModelBase
from models import TrainingDataType
import numpy as np
import cv2

from nnlib import DSSIMMaskLossClass
from nnlib import conv
from nnlib import res
from nnlib import upscale
from nnlib import upscale4
from facelib import FaceType

class Model(ModelBase):

    encoderH5 = 'encoder.h5'
    decoder_srcH5 = 'decoder_src.h5'
    decoder_dstH5 = 'decoder_dst.h5'

    #override
    def get_model_name(self):
        return "AVATAR"
 
    #override
    def onInitialize(self, batch_size=-1, **in_options):
        if self.gpu_total_vram_gb < 4:
            raise Exception ('Sorry, this model works only on 4GB+ GPU')
            
        self.batch_size = batch_size
        if self.batch_size == 0:     
            if self.gpu_total_vram_gb == 4:
                self.batch_size = 8
            elif self.gpu_total_vram_gb <= 6:
                self.batch_size = 4
            elif self.gpu_total_vram_gb < 8: 
                self.batch_size = 32
            else:    
                self.batch_size = 48 #best for all models
                
        ae_input_layer = self.keras.layers.Input(shape=(64, 64, 1))
 
        self.encoder = self.Encoder(ae_input_layer)
        self.decoder_src = self.Decoder()
        self.decoder_dst = self.Decoder()        
        
        if not self.is_first_run():
            self.encoder.load_weights     (self.get_strpath_storage_for_file(self.encoderH5))
            self.decoder_src.load_weights (self.get_strpath_storage_for_file(self.decoder_srcH5))
            self.decoder_dst.load_weights (self.get_strpath_storage_for_file(self.decoder_dstH5))

        self.autoencoder_src = self.keras.models.Model(ae_input_layer, self.decoder_src(self.encoder(ae_input_layer)))
        self.autoencoder_dst = self.keras.models.Model(ae_input_layer, self.decoder_dst(self.encoder(ae_input_layer)))

        if self.is_training_mode:
            self.autoencoder_src, self.autoencoder_dst = self.to_multi_gpu_model_if_possible ( [self.autoencoder_src, self.autoencoder_dst] )
                
        optimizer = self.keras.optimizers.Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)        
        self.autoencoder_src.compile(optimizer=optimizer, loss='mae' )
        self.autoencoder_dst.compile(optimizer=optimizer, loss='mae' )
  
        if self.is_training_mode:
            from models import TrainingDataGenerator
            f = TrainingDataGenerator.SampleTypeFlags 
            self.set_training_data_generators ([            
                    TrainingDataGenerator(self, TrainingDataType.SRC,  batch_size=self.batch_size, output_sample_types_flags=[ f.WARPED_TRANSFORMED | f.HALF_FACE | f.MODE_G | f.SIZE_64, f.TRANSFORMED | f.AVATAR_FACE | f.MODE_BGR | f.SIZE_256, f.SOURCE | f.HALF_FACE | f.MODE_G | f.SIZE_64, f.SOURCE | f.HALF_FACE | f.MODE_GGG | f.SIZE_256] ),
                    TrainingDataGenerator(self, TrainingDataType.DST,  batch_size=self.batch_size, output_sample_types_flags=[ f.WARPED_TRANSFORMED | f.HALF_FACE | f.MODE_G | f.SIZE_64, f.TRANSFORMED | f.AVATAR_FACE | f.MODE_BGR | f.SIZE_256, f.SOURCE | f.HALF_FACE | f.MODE_G | f.SIZE_64, f.SOURCE | f.HALF_FACE | f.MODE_GGG | f.SIZE_256] ),
                ])
    #override
    def onSave(self):        
        self.save_weights_safe( [[self.encoder, self.get_strpath_storage_for_file(self.encoderH5)],
                                [self.decoder_src, self.get_strpath_storage_for_file(self.decoder_srcH5)],
                                [self.decoder_dst, self.get_strpath_storage_for_file(self.decoder_dstH5)]] )
        
    #override
    def onTrainOneEpoch(self, sample):
        warped_src, target_src, target_src_for_predict, target_src_for_preview = sample[0]
        warped_dst, target_dst, target_dst_for_predict, target_dst_for_preview = sample[1]    
  
        loss_src = self.autoencoder_src.train_on_batch( warped_src, target_src )
        loss_dst = self.autoencoder_dst.train_on_batch( warped_dst, target_dst )
        
        return ( ('loss_src', loss_src), ('loss_dst', loss_dst) )
        

    #override
    def onGetPreview(self, sample):
        test_A   = sample[0][2][0:2] #first 2 samples
        test_B   = sample[1][2][0:2]
        
        test_A_pr   = sample[0][3][0:2] #first 2 samples
        test_B_pr   = sample[1][3][0:2]
        
        AA = self.autoencoder_src.predict(test_A)                                       
        AB = self.autoencoder_dst.predict(test_A)
        BB = self.autoencoder_dst.predict(test_B)

        st = []
        for i in range(0, len(test_A)):
            st.append ( np.concatenate ( (
                test_A_pr[i],
                AA[i],
                test_B_pr[i], 
                BB[i],               
                AB[i],
                ), axis=1) )
            
        return [ ('src, dst, src->dst', np.concatenate ( st, axis=0 ) ) ]
    
    def predictor_func (self, face):       
        x = self.autoencoder_dst.predict ( [ np.expand_dims(face,0) ] )
        x = x[0]     
        return x
        
    #override
    def get_converter(self, **in_options):
        from models import ConverterAvatar
        return ConverterAvatar(self.predictor_func, predictor_input_size=64, output_size=256, **in_options)
        
    def Encoder(self, input_layer):
        x = input_layer
        
        x = conv(self.keras, x, 96)
        x = conv(self.keras, x, 192)
        x = conv(self.keras, x, 384)
        x = conv(self.keras, x, 768)
        x = self.keras.layers.Flatten()(x)
        x = self.keras.layers.Dense(384)(x)
        x = self.keras.layers.Dense(8 * 8 * 384)(x)
        x = self.keras.layers.Reshape((8, 8, 384))(x)
        x = upscale(self.keras, x, 384) 
        return self.keras.models.Model(input_layer, x)

    def Decoder(self):
        input_ = self.keras.layers.Input(shape=(16, 16, 384))
        x = input_
        x = upscale(self.keras, x, 1536)
        x = upscale(self.keras, x, 768)
        x = upscale(self.keras, x, 384) 
        x = upscale(self.keras, x, 48)
        x = self.keras.layers.convolutional.Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)

        return self.keras.models.Model(input_, x)
