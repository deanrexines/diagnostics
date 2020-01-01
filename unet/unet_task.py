import os
import cv2
import sys
import math
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
import numpy as np 
from skimage.io import imsave, imread
from skimage.transform import rescale, resize
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras import *
from tensorflow.keras.preprocessing.image import *
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger
from google.cloud import storage

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

LOCAL_TRAIN_IMAGE_PATH = 'unet/chest_xray_data/chest-xray-masks-and-labels/data/lung_segmentation/CXR_png'
LOCAL_TRAIN_MASK_PATH = 'unet/chest_xray_data/chest-xray-masks-and-labels/data/lung_segmentation/masks'

BUCKET_NAME = 'diagnostics-unet-bucket'
GCLOUD_TRAIN_PATH = 'data/chest_xray/lung_segmentation/train/images/CXR_png/'
GCLOUD_MASK_PATH = 'data/chest_xray/lung_segmentation/train/images/masks/'

IMAGE_ROWS, IMAGE_COLS = 256, 256

NUM_TOTAL_IMAGE_PAIRS = 704

class UNet:
    def __init__(self, train_image_path=LOCAL_TRAIN_IMAGE_PATH, train_mask_path=LOCAL_TRAIN_MASK_PATH):
        self.model = None
        self.train_image_path = train_image_path
        self.train_mask_path = train_mask_path

    def get_unet(self, pretrained_weights = None,input_size = (256,256,1)):
        inputs = Input(input_size)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = concatenate([drop4,up6], axis = 3)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([conv3,up7], axis = 3)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([conv2,up8], axis = 3)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([conv1,up9], axis = 3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

        model = Model(inputs = inputs, outputs = conv10)

        model.compile(optimizer = Adam(lr=0.0), loss = 'binary_crossentropy', metrics = ['accuracy'])

        model.summary()

        if(pretrained_weights):
        	model.load_weights(pretrained_weights)

        self.model = model

    @staticmethod
    def step_decay(epoch):
       initial_lrate = 0.1
       drop = 0.5
       epochs_drop = 10.0
       lrate = initial_lrate * math.pow(drop,  
               math.floor((1+epoch)/epochs_drop))
       return lrate
    
    def create_train_data_local(self):
        train_image_path = os.path.abspath(self.train_image_path)
        os.path.abspath(self.train_image_path)
        train_mask_path = os.path.abspath(self.train_mask_path) 
        os.path.abspath(self.train_mask_path)

        _images = os.listdir(train_image_path)
        _masks = os.listdir(train_mask_path)
        total = int(NUM_TOTAL_IMAGE_PAIRS / 2)

        imgs = np.ndarray((total + 1, IMAGE_ROWS, IMAGE_COLS, 1), dtype=np.uint8)
        imgs_mask = np.ndarray((total + 1, IMAGE_ROWS, IMAGE_COLS, 1), dtype=np.uint8)

        i = 0
        for image_name, mask_name in zip(_images, _masks):
            if 'mask' in image_name:
                continue

            img = imread(os.path.join(train_image_path, image_name), as_gray=True)
            img_mask = imread(os.path.join(train_mask_path, mask_name), as_gray=True)

            img = resize(img, (IMAGE_ROWS, IMAGE_COLS, 1))
            img_mask = resize(img_mask, (IMAGE_ROWS, IMAGE_COLS, 1))

            if i < len(imgs) and i < len(imgs_mask):
                imgs[i] = img
                imgs_mask[i] = img_mask
                i += 1

        np.save('imgs_train.npy', imgs)
        np.save('imgs_mask_train.npy', imgs_mask)


    def process_image_for_inference(self):
        train_image_path = os.path.abspath(self.train_image_path)
        os.path.abspath(self.train_image_path)

        _images = os.listdir(train_image_path)

        img = resize(imread(os.path.join(train_image_path, _images[0]), as_gray=True), (IMAGE_ROWS, IMAGE_COLS, 1))

        return img

    def create_train_data_gcloud(self):
        storage_client = storage.Client()

        _image_blobs = storage_client.list_blobs(BUCKET_NAME, prefix=str(GCLOUD_TRAIN_PATH), delimiter='/')
        _mask_blobs = storage_client.list_blobs(BUCKET_NAME, prefix=str(GCLOUD_MASK_PATH), delimiter='/')

        total = int(NUM_TOTAL_IMAGE_PAIRS / 2)

        imgs = np.ndarray((total + 1, IMAGE_ROWS, IMAGE_COLS, 1), dtype=np.uint8)
        imgs_mask = np.ndarray((total + 1, IMAGE_ROWS, IMAGE_COLS, 1), dtype=np.uint8)

        i = 0
        for image_blob, mask_blob in zip(_image_blobs, _mask_blobs):

            img = np.array(
                cv2.imdecode(
                    np.asarray(bytearray(image_blob.download_as_string()), dtype=np.uint8), cv2.IMREAD_GRAYSCALE
                ).flatten()
            )
            img_mask = np.array(
                cv2.imdecode(
                    np.asarray(bytearray(mask_blob.download_as_string()), dtype=np.uint8), cv2.IMREAD_GRAYSCALE
                ).flatten()
            )

            img = resize(img, (IMAGE_ROWS, IMAGE_COLS, 1))
            img_mask = resize(img_mask, (IMAGE_ROWS, IMAGE_COLS, 1))

            if i < len(imgs) and i < len(imgs_mask):
                imgs[i] = img
                imgs_mask[i] = img_mask
                i += 1
                
        np.save('imgs_train.npy', imgs)
        np.save('imgs_mask_train.npy', imgs_mask)


    def load_train_data(self):
        imgs_train = np.load('imgs_train.npy')
        imgs_mask_train = np.load('imgs_mask_train.npy')
        return imgs_train, imgs_mask_train

    def train_no_generator(self):
        tf.executing_eagerly()

        csv_logger = CSVLogger('log.csv', append=True, separator=';')

        imgs_train, imgs_mask_train = self.load_train_data()

        lrate = LearningRateScheduler(UNet.step_decay)

        self.model.fit(imgs_train, imgs_mask_train, batch_size=32, epochs=5, verbose=1, shuffle=True, callbacks=[csv_logger, lrate],
                       validation_split=0.2)

        export_path = 'gs://diagnostics-unet-bucket/saved_models'

        tf.keras.models.save_model(self.model, export_path, save_format='tf')

    def train(self):
        tf.executing_eagerly()

        csv_logger = CSVLogger('log.csv', append=True, separator=';')


        data_gen_args = dict(featurewise_center=True,
                             featurewise_std_normalization=True,
                             rotation_range=90,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             )
        image_datagen = ImageDataGenerator(**data_gen_args)
        mask_datagen = ImageDataGenerator(**data_gen_args)

        seed = 1

        image_generator = image_datagen.flow_from_directory(
            self.train_image_path,
            class_mode=None,
            seed=seed)

        mask_generator = mask_datagen.flow_from_directory(
            self.train_mask_path,
            class_mode=None,
            seed=seed)

        train_generator = zip(image_generator, mask_generator)

        self.model.fit_generator(
            train_generator,
            steps_per_epoch=5,
            epochs=5, callbacks=[csv_logger])

if __name__ == '__main__':
    train_image_path = None
    train_mask_path = None
    if len(sys.argv) > 1:
        train_image_path=sys.argv[1]
        train_mask_path=sys.argv[2]

    if sys.argv[3] and sys.argv[3] == 'local':
        unet = UNet()
        unet.get_unet()
        unet.create_train_data_local()
        unet.train_no_generator()
    else:
        unet = UNet(train_image_path=train_image_path, train_mask_path=train_mask_path)
        unet.get_unet()
        unet.create_train_data_gcloud()
        unet.train_no_generator()