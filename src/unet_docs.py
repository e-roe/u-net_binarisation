from tensorflow.keras.models import *
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, Callback,  EarlyStopping, ReduceLROnPlateau
import numpy as np
from matplotlib import pyplot as plt
from os import makedirs
import pandas as pd
import os
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola
from sklearn.metrics import jaccard_similarity_score
from keras.preprocessing.image import ImageDataGenerator, array_to_img
import cv2

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
        yield (img, mask)


def check_gpu():
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
        # Code here what you want each time an epoch ends
        print('--- on_epoch_end ---')
        #self.save_epoch_results()


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


    def train(self, data_path, checkpoint_file, epochs=50):
        model = self.get_unet()
        print("got unet")

        model_checkpoint = ModelCheckpoint(checkpoint_file, monitor='loss', verbose=1, save_best_only=True)
        early_stopping = EarlyStopping(patience=20, verbose=1)
        reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1)
        print('Fitting model...')

        ld = loader(1, data_path, 'Originals', 'GT')

        history = model.fit_generator(ld, epochs=epochs, verbose=1, validation_steps=0.2, shuffle=True, steps_per_epoch=4605, callbacks=[reduce_lr, early_stopping,model_checkpoint, self])

        pd.DataFrame(history.history).plot(figsize=(8, 5))

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

        return self.restore_image(imgs_mask_test, dim)


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


    def show_image(self, image, *args, **kwargs):
        title = kwargs.get('title', 'Figure')
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        plt.imshow(image)
        plt.title(title)
        plt.show()


def test_predict(u_net, model):
    images = os.listdir(os.path.join('..', 'images'))
    results = []
    for image in images:
        ground_truth = cv2.imread(os.path.join('..', 'GT', image[:-3] + 'png'), cv2.IMREAD_GRAYSCALE)
        current_image = os.path.join('..', 'images', image)
        result_unet = u_net.binarise_image(model_weights=model, input_image=current_image)

        image_read = cv2.imread(current_image, cv2.IMREAD_GRAYSCALE)
        ressult_otsu = threshold_otsu(image_read)
        result_otsu = ((image_read > ressult_otsu) * 255).astype(np.uint8)
        result_sauvola = threshold_sauvola(image_read)
        result_sauvola = ((image_read > result_sauvola) * 255).astype(np.uint8)
        window_size = 25
        result_niblack = threshold_niblack(image_read, window_size=window_size, k=0.8)
        result_niblack = ((image_read > result_niblack) * 255).astype(np.uint8)

        img_true = np.array(ground_truth).ravel()
        img_pred = np.array(result_niblack).ravel()
        iou_niblack = jaccard_similarity_score(img_true, img_pred)
        cv2.imwrite(os.path.join('..', 'results', image[:-4] + '_' + str(iou_niblack)[:5] + '_niblack_.png'),
                    result_niblack)

        img_pred = np.array(result_otsu).ravel()
        iou_otsu = jaccard_similarity_score(img_true, img_pred)
        cv2.imwrite(os.path.join('..', 'results', image[:-4] + '_' + str(iou_otsu)[:5] + '_otsu_.png'), result_otsu)

        img_pred = np.array(result_sauvola).ravel()
        iou_sauvola = jaccard_similarity_score(img_true, img_pred)
        cv2.imwrite(os.path.join('..', 'results', image[:-4] + '_' + str(iou_sauvola)[:5] + '_sauvola_.png'),
                    result_sauvola)

        img_pred = np.array(result_unet).ravel()
        iou_unet = jaccard_similarity_score(img_true, img_pred)
        cv2.imwrite(os.path.join('..', 'results', image[:-4] + '_' + str(iou_unet)[:5] + '_unet_.png'), result_unet)

        results.append([iou_unet, iou_otsu, iou_sauvola, iou_niblack])

    for index, image in enumerate(images):
        print('Image', image, '- U-Net IoU:', results[index][0], 'Otsu IoU:', results[index][1], 'Sauvola IoU:',
              results[index][2], 'Niblack IoU:', results[index][3])


if __name__ == '__main__':
    check_gpu()
    my_unet = myUnet()

    data_path = 'D:\\Roe\\Medium\\data\\augmentations'
    checkpoint_file = 'unet_testing_dataset.hdf5'
    my_unet.train(data_path, checkpoint_file, epochs=2)

    #If you want to test the model just uncomment the following code
    #Pre-trained model
    # model = os.path.join('..', 'model', 'unet_4605_dataset.hdf5')
    # test_predict(my_unet, model)

