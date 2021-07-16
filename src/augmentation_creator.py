
import os
import numpy as np
import cv2
from PIL import Image
from skimage.util import random_noise

IMG_MODEL_SIZE = 256


def rotate_img(img, rt_degr):
    img = Image.fromarray(img)

    return np.asarray(img.rotate(rt_degr, expand=1))


def rotate():
   ...


def invert(image):
    return 255 - image


def cut_image(img, step):
    # print('cutting', img.shape)
    width = img.shape[1]
    height = img.shape[0]
    delta_x = step
    delta_y = step
    cuts = []
    curr_x = 0
    curr_y = 0
    # step = 128
    fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
    out_name = 'D:\\Roe\\Medium\\paper_to\\unet\\figs\\step'+ str(step) + '.mp4'
    out_cut = cv2.VideoWriter(out_name, fourcc, 20, (img.shape[1], img.shape[0]))
    first = True
    while curr_x + IMG_MODEL_SIZE < width:
        curr_y = 0
        while curr_y + IMG_MODEL_SIZE < height:
            cuts.append(img[curr_y:curr_y + IMG_MODEL_SIZE, curr_x:curr_x + IMG_MODEL_SIZE])
            cv2.rectangle(img, (curr_x, curr_y), (curr_x + IMG_MODEL_SIZE, curr_y + IMG_MODEL_SIZE), (127, 127, 127), 2)
            if first and curr_x == 3 * step:
                first = False
                #cv2.rectangle(img, (0, 0), (0 + IMG_MODEL_SIZE, 0 + IMG_MODEL_SIZE), (255, 255, 255), 6)
            cv2.imshow('w', cv2.resize(img, (0,0), fx=0.5, fy=0.5))
            #cv2.imshow('', img[yy:yy+delta_y, xx:xx+delta_x])
            cv2.waitKey(100)
            out_cut.write(img)
            cv2.imwrite('D:\\Roe\\Medium\\paper_to\\unet\\figs\\63-IMG_MAX_881468_cut.jpg', img)
            curr_y += step
        curr_x += step
    out_cut.release()

    return cuts


def myFunc(img):
    print('This is another function.')

def myFunc2():
    print('This is another function2.')


def noise(image):
    amount = 0.01
    res = random_noise(image, mode='s&p', amount=amount)

    return np.array(255 * res, dtype='uint8')


def create_augmentation(root_files_path, destination_path):

    # functions = {'myfoo2': myFunc2, 'myfoo': myFunc}
    # for name in functions.keys():
    #     res = functions[name](image_src)

    cut_step = 256 # 50 -> 241.731 100 -> 64.169
    datasets = os.listdir(root_files_path)
    index_all = 0
    for dataset in datasets:
        if dataset in ['nop', 'backup', 'augmentations', 'dataset', 'ICDAR']:
            continue

        # if dataset not in ['dibco']:
        #     continue

        sub_datas = os.listdir(os.path.join(root_files_path, dataset))
        for sub_data in sub_datas:
            originals_path = 'Originals'
            gt_path = 'GT'

            files = os.listdir(os.path.join(root_files_path, dataset, sub_data, originals_path))
            for file in files:
                print(file)

                image_original = cv2.imread(os.path.join(root_files_path, dataset, sub_data, originals_path, file))
                image_gt = cv2.imread(os.path.join(root_files_path, dataset, sub_data, gt_path, file[:-3] + 'png'))
                #result = eval('cut_image(image_src, 50)')

                images_augmentation_original = []
                images_augmentation_gt = []
                names = ['_orig']

                images_augmentation_original.append(image_original)
                images_augmentation_gt.append(image_gt)

                # images_augmentation_original.append(cv2.flip(image_original, 0))
                # images_augmentation_gt.append(cv2.flip(image_gt, 0))
                # names.append('_fliph')
                # images_augmentation_original.append(cv2.flip(image_original, 1))
                # images_augmentation_gt.append(cv2.flip(image_gt, 1))
                # names.append('_flipv')

                # images_augmentation_original.append(cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY))
                # images_augmentation_gt.append(image_gt)
                # names.append('_grey')

                # images_augmentation_original.append(noise(image_original))
                # images_augmentation_gt.append(image_gt)
                # names.append('_nois')
                #
                #
                # for ang in [90, 180, 270]:
                #     names.append('_ang'+str(ang))
                #     images_augmentation_original.append(rotate_img(image_original, ang))
                #     images_augmentation_gt.append(rotate_img(image_gt, ang))

                print(names, len(images_augmentation_original))

                for index in range(len(images_augmentation_original)):
                    # cv2.imshow('', images_augmentation_original[index])
                    # cv2.imshow('1', images_augmentation_gt[index])
                    # cv2.waitKey(1000)

                    name_augm = names[index]
                    cuts_originals = cut_image(images_augmentation_original[index], cut_step)
                    cuts_gt = cut_image(images_augmentation_gt[index], cut_step)
                    for cut in cuts_originals:
                          name = file[:-4] + name_augm + '_' + str(index_all) + '.jpg'
                          print(name)
                          index_all += 1
                          cv2.imwrite(os.path.join(destination_path, 'Originals', name), cut)
                    for cut in cuts_gt:
                          name = file[:-4] + name_augm + '_' + str(index_all) + '.png'
                          print(name)
                          index_all += 1
                          cv2.imwrite(os.path.join(destination_path, 'GT', name), cut)

                #sys.exit()

                # cuts_src = cut_image(image_original, 50)
                # image_label = cv2.imread(os.path.join(root_files_path, gt_path, file))
                # cuts_label = cut_image(image_label, 50)


                # index = 0
                # for cut in cuts_src:
                #       name = name_src + '_cut_' + str(index) + '.jpg'
                #       index += 1
                #       cv2.imwrite(os.path.join(destination_path, originals_path, name), cut)
                #
                # index = 0
                # for cut in cuts_label:
                #       name = name_src + '_cut_' + str(index) + '.jpg'
                #       index += 1
                #       cv2.imwrite(os.path.join(destination_path, gt_path, name), cut)
                #
                # rot_src = rotate_img(image_original, 90)
                # rot_label = rotate_img(image_label, 90)


                # cv2.imshow('', rot)
                # cv2.waitKey(0)


def convert(root_path):
    datasets = os.listdir(root_path)
    tot_all = 0
    for dataset in datasets:
        if dataset in ['nop', 'backup']:
            continue
        print(dataset)
        sub_datas = os.listdir(os.path.join(root_path, dataset))
        for subdata in sub_datas:
            print('   ', subdata)
            gts = os.listdir(os.path.join(root_path, dataset, subdata, 'GT'))
            for file in gts:
                if 'png' not in file:
                    print('         ', file)
                    tot_all += 1
                    delt = 3
                    if 'tiff' in file:
                        delt = 4
                    image = cv2.imread(os.path.join(root_path, dataset, subdata, 'GT', file))
                    cv2.imwrite(os.path.join(root_path, dataset, subdata, 'GT', file[:-delt]+'png'), image)
                    print('writing', os.path.join(root_path, dataset, subdata, 'GT', file[:-delt]+'png'))
                    os.remove(os.path.join(root_path, dataset, subdata, 'GT', file))
    print(tot_all)

def check_files(root_path):
    # dibco 116
    # ICDAR 4782
    # ICFHR_2016 100
    # nabuco-dataset 15
    # PHIB 15
    # Total 5028
    datasets = os.listdir(root_path)
    tot_all = 0
    for dataset in datasets:
        if dataset in ['nop', 'backup', 'augmentations']:
            continue
        tot_dataset = 0
        sub_datas = os.listdir(os.path.join(root_path, dataset))
        for subdata in sub_datas:
            originals = os.listdir(os.path.join(root_path, dataset, subdata, 'Originals'))
            for original in originals:
                name_gt = original[:-3] + 'png'
                if not os.path.exists(os.path.join(root_path, dataset, subdata, 'GT', name_gt)):
                    print('error', original, name_gt)
        tot_all += tot_dataset
        print(dataset, tot_dataset)
    print(tot_all)



def rename_dataset(root_path):
    # dibco 116
    # ICDAR 4782
    # ICFHR_2016 100
    # nabuco-dataset 15
    # PHIB 15
    # Total 5028
    datasets = os.listdir(root_path)
    tot_all = 0
    for dataset in datasets:
        originals = os.listdir(os.path.join(root_path, 'Originals'))
        gt = os.listdir(os.path.join(root_path, 'GT'))
        for index in range(len(gt)):
            if originals[index] != gt[index]:
                print(originals[index], '---', gt[index])
                os.rename(os.path.join(root_path, 'GT', gt[index]),
                          os.path.join(root_path, 'GT', originals[index][:-3]+'png'))

def invert():
    pp = 'D:\\Roe\\Medium\\data\\ICDAR\\2017\\GTi'
    files = os.listdir(pp)
    for file in files:
        image = cv2.imread(os.path.join(pp, file))
        cv2.imwrite(os.path.join('D:\\Roe\\Medium\\data\\ICDAR\\2017\\GT', file), 255 - image)


import sys
if __name__ == '__main__':
    root = 'D:\\Roe\\Medium\\data'

    img = cv2.imread('D:\\Roe\\Medium\\paper_to\\unet\\figs\\63-IMG_MAX_881468.jpg')
    cut_image(img, 256)
    # rename_dataset('D:\\Roe\\Medium\\data\\ICDAR\\2017')
    # check_files(root)
    # convert(root)
    # sys.exit()
    #create_augmentation(root, 'D:\\Roe\\Medium\\data\\augmentations')