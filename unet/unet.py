import numpy as np 
import os 
from skimage.io import imsave, imread
from skimage.transform import rescale, resize
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras import *
from keras.preprocessing.image import *
import tensorflow as tf
from keras.callbacks import CSVLogger
# from keras.callbacks import ModelCheckpoint, LearningRateScheduler

class UNet:
    def __init__(self):
        self.model = None
        self.image_path = 'chest_xray_data/chest-xray-masks-and-labels/data/lung_segmentation/CXR_png'
        self.mask_path = 'chest_xray_data/chest-xray-masks-and-labels/data/lung_segmentation/masks'
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

        model = Model(input = inputs, output = conv10)

        model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

        model.summary()

        if(pretrained_weights):
        	model.load_weights(pretrained_weights)

        self.model = model

    def create_train_data(self):
        image_rows, image_cols = 256, 256

        train_data_path = os.path.abspath(self.image_path)
        train_mask_path = os.path.abspath(self.mask_path)
        _images = os.listdir(train_data_path)
        _masks = os.listdir(train_mask_path)
        total = int(len(_images) / 2)

        imgs = np.ndarray((total, image_rows, image_cols, 1), dtype=np.uint8)
        imgs_mask = np.ndarray((total, image_rows, image_cols, 1), dtype=np.uint8)

        i = 0

        avg_1 = 0
        avg_2 = 0
        count_1 = 0
        count_2 = 0
        for image_name, mask_name in zip(_images, _masks):
            if 'mask' in image_name:
                continue

            img = imread(os.path.join(train_data_path, image_name), as_gray=True)
            img_mask = imread(os.path.join(train_mask_path, mask_name), as_gray=True)

            # img = resize(np.array([img]), (image_rows, image_cols, 1))
            img = resize(img, (image_rows, image_cols, 1))
            # img_mask = resize(np.array([img_mask]), (image_rows, image_cols, 1))
            img_mask = resize(img_mask, (image_rows, image_cols, 1))

            avg_1 += img.shape[0];count_1 += 1;
            avg_2 += img.shape[1];count_2 += 1;

            if i == 50:
                break
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
        import tensorflow as tf
        tf.executing_eagerly()

        from keras.callbacks import CSVLogger

        csv_logger = CSVLogger('log.csv', append=True, separator=';')

        imgs_train, imgs_mask_train = self.load_train_data()
        self.model.fit(imgs_train, imgs_mask_train, batch_size=32, nb_epoch=20, verbose=1, shuffle=True, callbacks=[csv_logger])
                  # validation_split=0.2,
                  # callbacks=[model_checkpoint])

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

        # seed = 1
        # image_datagen.fit(images, augment=True, seed=seed)
        # mask_datagen.fit(masks, augment=True, seed=seed)

        image_generator = image_datagen.flow_from_directory(
            self.image_path,
            class_mode=None,
            color_mode="grayscale")
            # target_size=(256,256,1))
            # seed=seed)

        mask_generator = mask_datagen.flow_from_directory(
            self.mask_path,
            class_mode=None,)
            # color_mode="grayscale")
            # target_size=(256,256,1))
            # seed=seed)

        train_generator = zip(image_generator, mask_generator)

        self.model.fit_generator(
            train_generator,
            steps_per_epoch=5,
            epochs=5, callbacks=[csv_logger])

if __name__ == '__main__':
    unet = UNet()
    unet.get_unet()
    unet.create_train_data()
    unet.train_no_generator()
    # unet.train()