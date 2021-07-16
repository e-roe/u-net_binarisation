import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.layers import merge
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback,  EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as keras
from data_docs import *
from matplotlib import pyplot as plt
from os import makedirs
import sys
import pandas as pd


USE_GPU = True
IMG_MODEL_SIZE = 256


def loader(batch_size, train_path, image_folder, mask_folder, mask_color_mode="grayscale", target_size=(256, 256), save_to_dir=None):
    image_datagen = ImageDataGenerator(rescale=1. / 255)
    mask_datagen = ImageDataGenerator(rescale=1. / 255)

    image_generator = image_datagen.flow_from_directory(train_path, classes=[image_folder], class_mode=None,
                                                        color_mode=mask_color_mode, target_size=target_size,
                                                        batch_size=batch_size, save_to_dir=save_to_dir, seed=1)

    mask_generator = mask_datagen.flow_from_directory(train_path, classes=[mask_folder], class_mode=None,
                                                      color_mode=mask_color_mode, target_size=target_size,
                                                      batch_size=batch_size, save_to_dir=save_to_dir, seed=1)
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        # img, mask = get_data(img, mask)
        yield (img, mask)

def check_gpu():
    import os
    if USE_GPU:
        import tensorflow as tf
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'



class myUnet(Callback):

    def __init__(self, img_rows=IMG_MODEL_SIZE, img_cols=IMG_MODEL_SIZE):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.counter = 0

    def on_epoch_end(self, epoch, logs=None):
        print('--- on_epoch_end ---')
        #self.save_epoch_results()

    def load_data(self):
        mydata = dataProcess(self.img_rows, self.img_cols)
        imgs_train, imgs_mask_train = mydata.load_train_data()
        imgs_test = mydata.load_test_data()

        return imgs_train, imgs_mask_train, imgs_test

    def get_unet(self):

        inputs = Input((self.img_rows, self.img_cols, 1))

        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        print("conv1 shape:", conv1.shape)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        print("conv1 shape:", conv1.shape)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        print("pool1 shape:", pool1.shape)

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        print("conv2 shape:", conv2.shape)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        print("conv2 shape:", conv2.shape)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        print("pool2 shape:", pool2.shape)

        conv3 = Conv2D(IMG_MODEL_SIZE, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        print("conv3 shape:", conv3.shape)
        conv3 = Conv2D(IMG_MODEL_SIZE, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        print("conv3 shape:", conv3.shape)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        print("pool3 shape:", pool3.shape)

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        print('up6 ' + str(up6))
        print('drop4 ' + str(drop4))
        # merge6 = merge([drop4, up6], mode = 'concat', concat_axis = 3)
        merge6 = concatenate([drop4, up6], axis=3)

        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(IMG_MODEL_SIZE, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        # merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
        merge7 = concatenate([conv3, up7], axis=3)

        conv7 = Conv2D(IMG_MODEL_SIZE, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(IMG_MODEL_SIZE, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        # merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
        merge8 = concatenate([conv2, up8], axis=3)

        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        # merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
        merge9 = concatenate([conv1, up9], axis=3)

        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

        model = Model(inputs, conv10)
        # model = Model()

        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

        return model


    def train(self):

        print("loading data")
        imgs_train, imgs_mask_train, imgs_test = self.load_data()
        print(imgs_test.shape)
        test_img = cv2.imread('D:\\Roe\\Medium\\data\\augmentations\\Originals\\1_17_orig_408.jpg', cv2.IMREAD_GRAYSCALE)
        # cv2.imshow('', test_img)
        # cv2.waitKey(0)
        test_img = test_img.astype('float32')
        test_img /= 255
        imgs_test = np.asarray([test_img])
        imgs_test = imgs_test[..., np.newaxis]
        print(imgs_test.shape)
        print("loading data done")
        model = self.get_unet()
        print("got unet")

        model_checkpoint = ModelCheckpoint('unet_4605_dataset.hdf5', monitor='loss', verbose=1, save_best_only=True)
        early_stopping = EarlyStopping(patience=20, verbose=1)
        reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1)
        print('Fitting model...')

        ld = loader(1, 'D:\\Roe\\Medium\\data\\augmentations', 'Originals', 'GT')

        history = model.fit_generator(ld, epochs=50, verbose=1, validation_steps=0.2, shuffle=True, steps_per_epoch=4605, callbacks=[reduce_lr, early_stopping,model_checkpoint, self])
        # print(history.history.keys())
        # plt.plot(history.history['accuracy'])
        # plt.plot(history.history['loss'])
        # plt.title('model accuracy')
        # plt.ylabel('accuracy')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'validation'], loc='upper left')
        # plt.show()
        # # "Loss"
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title('model loss')
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'validation'], loc='upper left')
        # plt.show()

        pd.DataFrame(history.history).plot(figsize=(8, 5))
        #plt.show()
        plt.savefig('D:\\Roe\\Medium\\prjs\\u_net\\results\\graph.png')

        # model.fit(imgs_train, imgs_mask_train, batch_size=4, epochs=20, verbose=1, validation_split=0.2, shuffle=True,
        #           callbacks=[model_checkpoint, self])


        print('Going to predict test data')
        imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
        result = np.zeros([256, 256, 1], dtype=np.uint8)
        result = imgs_mask_test[0] * 255
        print(imgs_mask_test.shape)
        cv2.imwrite('D:\\Roe\\Medium\\prjs\\u_net\\results\\test.jpg', result)

        # cv2.imshow('res', result)
        # cv2.waitKey(0)
        #np.save('D:\\Roe\\Medium\\prjs\\u_net\\imgs_mask_test.npy', imgs_mask_test)

    def load_predict(self, model_weights, input_image):
        print("loading data")
        imgs_train, imgs_mask_train, imgs_test = self.load_data()
        print("loading data done")
        model = self.get_unet()
        print("got unet")
        print(imgs_test.shape)
        cv2.imshow('', imgs_test[0])
        cv2.waitKey(0)
        model.load_weights('unet_dibco.hdf5')

        print('predicting test data')
        imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
        np.save('D:\\Roe\\Medium\\prjs\\u_net\\dibco\\images\\results\\imgs_mask_test.npy', imgs_mask_test)

    def prepare_image_predict(self, input_image):
        img = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
        width = img.shape[1]
        height = img.shape[0]
        delta_x = width // IMG_MODEL_SIZE
        delta_y = height // IMG_MODEL_SIZE
        remx = width % IMG_MODEL_SIZE
        remy = height % IMG_MODEL_SIZE
        print('remy', remy)
        parts = []
        border = np.zeros([IMG_MODEL_SIZE, IMG_MODEL_SIZE], dtype=np.uint8)
        border.fill(255)  # or img[:] = 255

        for x in range(delta_x):
            xinit = x * IMG_MODEL_SIZE
            for y in range(delta_y):
                yinit = y * IMG_MODEL_SIZE
                part = img[yinit:yinit + IMG_MODEL_SIZE, xinit:xinit+IMG_MODEL_SIZE]
                parts.append(part.astype('float32') / 255)
            if remy > 0:
                border.fill(255)
                border[:remy, :] = img[height-remy:height, xinit:xinit+IMG_MODEL_SIZE]
                parts.append(border.astype('float32') / 255)

        if remx > 0:
            xinit = width - remx
            for y in range(delta_y):
                yinit = y * IMG_MODEL_SIZE
                border.fill(255)
                border[:, :remx] = img[yinit:yinit + IMG_MODEL_SIZE, xinit:width]
                parts.append(border.astype('float32') / 255)
            if remy > 0:
                border.fill(255)
                border[:remy, :remx] = img[height-remy:height, xinit:width]
                parts.append(border.astype('float32') / 255)

        # for part in parts:
        #     cv2.imshow('re', part)
        #     cv2.waitKey(0)

        return np.asarray(parts), (img.shape[0], img.shape[1])


    def restore_image(self, parts, dim):
        width = dim[1]
        height = dim[0]
        result = np.zeros([height, width, 1], dtype=np.uint8)
        result.fill(255)  # or img[:] = 255
        delta_x = width // IMG_MODEL_SIZE
        delta_y = height // IMG_MODEL_SIZE
        remx = width % IMG_MODEL_SIZE
        remy = height % IMG_MODEL_SIZE
        index = 0
        for x in range(delta_x):
            xinit = x * IMG_MODEL_SIZE
            for y in range(delta_y):
                yinit = y * IMG_MODEL_SIZE
                result[yinit:yinit+IMG_MODEL_SIZE, xinit:xinit+IMG_MODEL_SIZE] = parts[index] * 255
                index += 1
            if remy > 0:
                result[height-remy:, xinit:xinit+IMG_MODEL_SIZE] = parts[index][:remy, :] * 255
                index += 1
        if remx > 0:
            xinit = width - remx
            for y in range(delta_y):
                yinit = y * IMG_MODEL_SIZE
                result[yinit:yinit+IMG_MODEL_SIZE, xinit:] = parts[index][:, :remx] * 255
                index += 1
            if remy > 0:
                result[height-remy:, xinit:xinit+IMG_MODEL_SIZE] = parts[index][:remy, :remx] * 255

        cv2.imshow('final', cv2.resize(result, (0, 0), fx=0.25, fy=0.25))
        cv2.waitKey(0)

        return result


    def binarise_image(self, model_weights, input_image):
        print("loading image")
        parts, dim = self.prepare_image_predict(input_image=input_image)
        #imgs_train, imgs_mask_train, imgs_test = self.load_data()
        print("loading data done")

        print(type(parts))
        print(parts.shape)

        model = self.get_unet()
        print("got unet")
        print(parts.shape)

        model.load_weights(model_weights)

        print('predicting test data')
        imgs_mask_test = model.predict(parts, batch_size=1, verbose=1)
        print(imgs_mask_test.shape)
        for part in imgs_mask_test:
            cv2.imshow('', part)
            cv2.waitKey(100)

        result = self.restore_image(imgs_mask_test, dim)
        cv2.imwrite('D:\\Roe\\Medium\\prjs\\u_net\\results\\bin.png', result)
        #np.save('D:\\Roe\\Medium\\prjs\\u_net\\dibco\\images\\results\\imgs_mask_test.npy', imgs_mask_test)

    def save_epoch_results(self):
        print("loading data")
        imgs_train, imgs_mask_train, imgs_test = self.load_data()
        print("loading data done")
        model = self.get_unet()
        print("got unet")

        model.load_weights('unet_dibco.hdf5')

        print('predict test data')
        imgs = model.predict(imgs_test, batch_size=1, verbose=1)
        path = "D:\\Roe\\Medium\\prjs\\u_net\\results\\" + str(self.counter)

        self.counter += 1
        if not os.path.exists(path):
            makedirs(path)

        for i in range(imgs.shape[0]):
            img = imgs[i]
            # self.show_image(img)
            img = array_to_img(img)
            img.save(path + "%d.jpg" % (i))

    def save_img(self):
        print("array to image")
        imgs = np.load('D:\\Roe\\Medium\\prjs\\u_net\\results\\imgs_mask_test.npy')
        for i in range(imgs.shape[0]):
            img = imgs[i]
            # self.show_image(img)
            img = array_to_img(img)
            img.save("D:\\Roe\\Medium\\prjs\\u_net\\results_new1\\%d.jpg" % (i))

    def show_image(self, image, *args, **kwargs):
        title = kwargs.get('title', 'Figure')
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        plt.imshow(image)
        plt.title(title)
        plt.show()


# def add_crf_layer(original_model):
#     original_model.trainable = False
#
#     crf_layer = CrfRnnLayer(image_dims=(224, 224),
#                             num_classes=2,
#                             theta_alpha=3.,
#                             theta_beta=160.,
#                             theta_gamma=3.,
#                             num_iterations=10,
#                             name='crfrnn')([original_model.outputs[0], original_model.inputs[0]])
#
#     new_crf_model = Model(inputs = original_model.input, outputs = crf_layer)
#
#     return(new_crf_model)

if __name__ == '__main__':
    check_gpu()
    myunet = myUnet()

    # myunet = add_crf_layer(myunet)

    myunet.train()


    testing = 'D:\\Roe\\Medium\\prjs\\u_net\\tests\\395.jpg'
    testing = 'D:\\Roe\\Medium\\data\\dibco\\originals\\HW7.png'
    testing = 'D:\\Roe\\Medium\\data\ICDAR\\2017\\Originals\\6-IMG_MAX_1005569.jpg'
    #myunet.binarise_image(model_weights='unet_4605_dataset.hdf5', input_image=testing)

    #myunet.load_predict()
    # myunet.save_img()
