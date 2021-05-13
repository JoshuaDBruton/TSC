import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import nibabel as nift
import cv2
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from experiments.dataset.image_pair import ImageSegmentation as DataSet
from csv import reader
from random import random


def generate_tiles(h, w, tile_width=None, tile_height=None, window_size=100):
    np.seterr(divide='ignore', invalid='ignore')

    if not tile_width:
        tile_width = window_size

    if not tile_height:
        tile_height = window_size

    wTile = tile_width
    hTile = tile_height

    if tile_width > w or tile_height > h:
        raise ValueError("tile dimensions cannot be larger than origin dimensions")

    # Number of tiles
    nTilesX = np.uint16(np.ceil(w / wTile))
    nTilesY = np.uint16(np.ceil(h / hTile))

    # Total remainders
    remainderX = nTilesX * wTile - w
    remainderY = nTilesY * hTile - h

    # Set up remainders per tile
    remaindersX = np.ones((nTilesX-1, 1)) * np.uint16(np.floor(remainderX / (nTilesX-1)))
    remaindersY = np.ones((nTilesY-1, 1)) * np.uint16(np.floor(remainderY / (nTilesY-1)))
    remaindersX[0:np.remainder(remainderX, np.uint16(nTilesX-1))] += 1
    remaindersY[0:np.remainder(remainderY, np.uint16(nTilesY-1))] += 1

    # Initialize array of tile boxes
    tiles = np.zeros((nTilesX * nTilesY, 4), np.uint16)

    k = 0
    x = 0
    for i in range(nTilesX):
        y = 0
        for j in range(nTilesY):
            tiles[k, :] = (x, y, hTile, wTile)
            k += 1
            if j < (nTilesY-1):
                y = y + hTile - remaindersY[j]
        if i < (nTilesX-1):
            x = x + wTile - remaindersX[i]

    return tiles


def tile_image(image, tile_size):
    tiles = generate_tiles(image.shape[0], image.shape[1], window_size=tile_size)
    images = []
    for tile in tiles:
        t = image[tile[0]:tile[0]+tile_size, tile[1]:tile[1]+tile_size]
        if t.shape[0] == tile_size and t.shape[1] == tile_size:
            images.append(image[tile[0]:tile[0]+tile_size, tile[1]:tile[1]+tile_size])
    images = np.array(images, dtype=np.uint8)
    return images


def process_iild():
    data_path = "/home/joshua/Desktop/data/IILD/AerialImageDataset/train"
    oinput_path = os.path.join(data_path, "oinputs")
    otarget_path = os.path.join(data_path, "otargets")
    input_path = os.path.join(data_path, "inputs")
    target_path = os.path.join(data_path, "targets")

    input_filenames = os.listdir(oinput_path)
    curr = 0
    for i_f in tqdm(input_filenames):
        x = np.array(plt.imread(os.path.join(oinput_path, i_f)))
        y = np.array(plt.imread(os.path.join(otarget_path, i_f)))
        y[y != 0] = 1

        tiled_x = tile_image(x, 256)
        tiled_y = tile_image(y, 256)

        for i in range(tiled_x.shape[0]):
            assert (len(tiled_x[i].shape) == 3), "x has more or less than 3 dimensions, it has shape {}".format(tiled_x[i].shape)
            assert (tiled_x[i].shape[2] == 3), "x is still not RGB, it has {} channels".format(tiled_x[i].shape[2])
            assert (tiled_x[i].shape[0] == tiled_y[i].shape[0] and tiled_x[i].shape[1] == tiled_y[i].shape[1]), "x and y have different shapes in dim 0 or 1"
            assert (tiled_x[i].dtype == np.uint8), "x does not have to correct type, its type is {}".format(tiled_x[i].dtype)
            assert (tiled_y[i].dtype == np.uint8), "y does not have the correct type, it is {}".format(tiled_y[i].dtype)
            assert (len(tiled_y[i].shape) == 2), "y has the incorrect number of dimensions, shape is {}".format(tiled_y[i].shape)

            plt.imsave(os.path.join(input_path, str(curr)+".jpg"), tiled_x[i])
            np.save(os.path.join(target_path, str(curr) + ".npy"), tiled_y[i])
            # plt.imsave(os.path.join(target_path, str(curr)+".png"), tiled_y[i])
            curr += 1


def process_landcover():
    data_path = "/home/joshua/Desktop/data/landcover.ai/train"
    oinput_path = os.path.join(data_path, "oinputs")
    otarget_path = os.path.join(data_path, "otargets")
    input_path = os.path.join(data_path, "inputs")
    target_path = os.path.join(data_path, "targets")

    input_filenames = os.listdir(oinput_path)
    curr = 0
    for i_f in tqdm(input_filenames):
        x = np.array(plt.imread(os.path.join(oinput_path, i_f)))
        y = np.array(plt.imread(os.path.join(otarget_path, i_f)))

        tiled_x = tile_image(x, 256)
        tiled_y = tile_image(y, 256)

        for i in range(tiled_x.shape[0]):
            assert (len(tiled_x[i].shape) == 3), "x has more or less than 3 dimensions, it has shape {}".format(tiled_x[i].shape)
            assert (tiled_x[i].shape[2] == 3), "x is still not RGB, it has {} channels".format(tiled_x[i].shape[2])
            assert (tiled_x[i].shape[0] == tiled_y[i].shape[0] and tiled_x[i].shape[1] == tiled_y[i].shape[1]), "x and y have different shapes in dim 0 or 1"
            assert (tiled_x[i].dtype == np.uint8), "x does not have the correct type, it's type is {}".format(tiled_x[i].dtype)
            assert (tiled_y[i].dtype == np.uint8), "y does not have the correct type, it is {}".format(tiled_y[i].dtype)
            assert (len(tiled_y[i].shape) == 2), "y has the incorrect number of dimensions, shape is {}".format(tiled_y[i].shape)

            plt.imsave(os.path.join(input_path, str(curr) + ".jpg"), tiled_x[i])
            np.save(os.path.join(target_path, str(curr)+".npy"), tiled_y[i])
            # plt.imsave(os.path.join(target_path, str(curr) + ".png"), tiled_y[i])
            curr += 1


def process_ircad():
    data_path = "/home/joshua/Desktop/data/ircad-dataset/train"
    oinput_path = os.path.join(data_path, "input_nif")
    otarget_path = os.path.join(data_path, "target_nif")
    input_path = os.path.join(data_path, "inputs")
    target_path = os.path.join(data_path, "targets")

    input_filenames = os.listdir(oinput_path)

    curr = 0
    for i_f in tqdm(input_filenames):
        x = np.array(nift.load(os.path.join(oinput_path, i_f)).get_fdata(), dtype=np.uint8)
        y = np.array(nift.load(os.path.join(otarget_path, i_f)).get_fdata(), dtype=np.uint8)

        for i in range(x.shape[2]):
            plt.imsave(os.path.join(input_path, str(curr)+".jpg"), x[::2, ::2, i])
            # plt.imsave(os.path.join(target_path, str(curr)+".png"), y[::2, ::2, i])
            np.save(os.path.join(target_path, str(curr)+".npy"), y[::2, ::2, i])
            curr += 1


def process_coco():
    data_path = "/home/joshua/Desktop/data/COCO/train"
    oinput_path = os.path.join(data_path, "oinputs")
    otarget_path = os.path.join(data_path, "otargets")
    input_path = os.path.join(data_path, "inputs")
    target_path = os.path.join(data_path, "targets")
    rgb_weights = [0.2989, 0.5870, 0.1140]

    input_filenames = os.listdir(oinput_path)

    all_labels = np.load(os.path.join(data_path, "coco_uniques.npy"))
    labels = {k: v for v, k in enumerate(all_labels)}

    curr = 0
    for i_f in tqdm(input_filenames):
        x = np.array(plt.imread(os.path.join(oinput_path, i_f)))
        y = np.array(plt.imread(os.path.join(otarget_path, i_f.replace("jpg", "png"))))

        if len(x.shape) == 2:
            x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)

        y = np.dot(y[..., :3], rgb_weights)
        gs = y.copy()

        uniques = np.unique(gs)

        for u in uniques:
            y[gs == u] = labels[u]

        y = np.array(y, dtype=np.uint8)

        assert (len(x.shape) == 3), "x has more or less than 3 dimensions, it has shape {}".format(x.shape)
        assert (x.shape[2] == 3), "x is still not RGB, it has {} channels".format(x.shape[2])
        assert (x.shape[0] == y.shape[0] and x.shape[1] == y.shape[1]), "x and y have different shapes in dim 0 or 1"
        assert (y.dtype == np.uint8), "y does not have the correct type, it is {}".format(y.dtype)

        plt.imsave(os.path.join(input_path, str(curr)+".jpg"), x)
        np.save(os.path.join(target_path, str(curr)+".npy"), y)
        # plt.imsave(os.path.join(target_path, str(curr) + ".jpg"), y)
        curr += 1


def process_voc():
    files_dir = "/home/joshua/Desktop/Work/data/VOCdevkit/VOC2012/ImageSets/Segmentation"
    train_fs = os.path.join(files_dir, "train.txt")
    val_fs = os.path.join(files_dir, "val.txt")

    o_data_path = "/home/joshua/Desktop/Work/data/VOCdevkit/VOC2012"
    o_input_path = os.path.join(o_data_path, "JPEGImages")
    o_target_path = os.path.join(o_data_path, "SegmentationClass")

    f_data_path = "/home/joshua/Desktop/data/VOC/validation"
    f_input_path = os.path.join(f_data_path, "inputs")
    f_target_path = os.path.join(f_data_path, "targets")

    f = open(val_fs, "r")

    curr = 0
    for line in tqdm(f.readlines()):
        if len(line) <= 0:
            continue
        line = line.rstrip()
        x = np.asarray(Image.open(os.path.join(o_input_path, line+".jpg")))
        y = np.asarray(Image.open(os.path.join(o_target_path, line+".png")))
        y = y.copy()

        y[y == 255] = 21

        assert (len(x.shape) == 3), "x has more or less than 3 dimensions, it has shape {}".format(x.shape)
        assert (len(y.shape) == 2), "y has more or less than 2 dimensions, it has shape {}".format(y.shape)
        assert (x.shape[2] == 3), "x is still not RGB, it has {} channels".format(x.shape[2])
        assert (x.shape[0] == y.shape[0] and x.shape[1] == y.shape[1]), "x and y have different shapes in dim 0 or 1"
        assert (y.dtype == np.uint8), "y does not have the correct type, it is {}".format(y.dtype)
        assert (x.dtype == np.uint8), "x does not have the correct type, it is {}".format(x.dtype)

        plt.imsave(os.path.join(f_input_path, str(curr) + ".jpg"), x)
        np.save(os.path.join(f_target_path, str(curr) + ".npy"), y)
        # plt.imsave(os.path.join(f_target_path, str(curr) + ".png"), y)
        curr += 1


def process_carla():
    o_data_path = "/home/joshua/Desktop/data/carla/dataE/dataE"
    o_input_path = os.path.join(o_data_path, "CameraRGB")
    o_target_path = os.path.join(o_data_path, "CameraSeg")

    f_data_path = "/home/joshua/Desktop/data/carla/validation"
    f_input_path = os.path.join(f_data_path, "inputs")
    f_target_path = os.path.join(f_data_path, "targets")

    filenames = os.listdir(o_input_path)

    curr = 0

    for f in tqdm(filenames):
        x = plt.imread(os.path.join(o_input_path, f))
        y = plt.imread(os.path.join(o_target_path, f))

        y = y[:, :, 0]
        y = np.array(y*255, dtype=np.uint8)

        x = np.array(x*255, dtype=np.uint8)

        assert (len(x.shape) == 3), "x has more or less than 3 dimensions, it has shape {}".format(x.shape)
        assert (x.shape[2] == 3), "x is still not RGB, it has {} channels".format(x.shape[2])
        assert (x.shape[0] == y.shape[0] and x.shape[1] == y.shape[1]), "x and y have different shapes in dim 0 or 1"
        assert (y.dtype == np.uint8), "y does not have the correct type, it is {}".format(y.dtype)
        assert (x.dtype == np.uint8), "x does not have the correct type, it is {}".format(x.dtype)

        plt.imsave(os.path.join(f_input_path, str(curr) + ".jpg"), x)
        np.save(os.path.join(f_target_path, str(curr) + ".npy"), y)
        # plt.imsave(os.path.join(f_target_path, str(curr) + ".png"), y)
        curr += 1

    print(curr)


def check_voc():
    all_labels = np.load("/home/joshua/Desktop/data/VOC/all_labels.npy")

    print(all_labels)
    print(all_labels.shape)

TRAIN_TRANSFORM = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomHorizontalFlip(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

TARGET_TRANSFORM = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.ToTensor(),
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
])

def count_labels():
    all_data = DataSet(root_dir="/home/joshua/Desktop/data/VOC/validation", transform=TRAIN_TRANSFORM,
                       target_transform=TARGET_TRANSFORM)

    # all_data, _ = torch.utils.data.random_split(all_data, [10, len(all_data)-10])

    test_loader = DataLoader(all_data, shuffle=False, num_workers=1, batch_size=1,)

    bins = np.zeros(22)

    for (x, y) in tqdm(test_loader):
        counts, _ = np.histogram(y, bins=[*range(0, 23, 1)])
        bins += counts

    bins = bins/np.sum(bins)
    print()
    print(bins)
    plt.bar(x=[*range(0, 22, 1)], height=bins, align='center')
    plt.xticks([*range(0, 22, 1)], ['Background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
                                    'chair', 'cow', 'table', 'dog', 'horse', 'bike', 'person', 'pot plant', 'sheep',
                                    'sofa', 'train', 'monitor', 'border'])
    plt.show()
    print(np.argsort(bins))


def process_liver():
    o_data_path = "/home/joshua/Desktop/data/pre_liver/dataset_6/dataset_6"
    train_list_path = "/home/joshua/Desktop/data/pre_liver/lits_train.csv"
    val_list_path = "/home/joshua/Desktop/data/pre_liver/lits_test.csv"

    f_data_path = "/home/joshua/Desktop/data/liver/validation"
    f_input_path = os.path.join(f_data_path, "inputs")
    f_target_path = os.path.join(f_data_path, "targets")

    with open(val_list_path, 'r') as read_obj:
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        for i, row in tqdm(enumerate(csv_reader)):
            x_name = row[0].replace('../input/lits-png/dataset_6/', '')
            y1_name = row[1].replace('../input/lits-png/dataset_6/', '')
            y2_name = row[2].replace('../input/lits-png/dataset_6/', '')

            x = np.asarray(Image.open(os.path.join(o_data_path, x_name)))
            y1 = np.asarray(Image.open(os.path.join(o_data_path, y1_name)))
            y2 = np.asarray(Image.open(os.path.join(o_data_path, y2_name)))

            y = y1[:, :, 0].copy()
            y2 = y2[:, :, 0]

            y[y2 == 1] = 2

            assert (len(x.shape) == 3), "x has more or less than 3 dimensions, it has shape {}".format(x.shape)
            assert (x.shape[2] == 3), "x is still not RGB, it has {} channels".format(x.shape[2])
            assert (x.shape[0] == y.shape[0] and x.shape[1] == y.shape[1]), "x and y have different shapes in dim 0 or 1"
            assert (y.dtype == np.uint8), "y does not have the correct type, it is {}".format(y.dtype)
            assert (x.dtype == np.uint8), "x does not have the correct type, it is {}".format(x.dtype)

            plt.imsave(os.path.join(f_input_path, str(i) + ".jpg"), x)
            np.save(os.path.join(f_target_path, str(i) + ".npy"), y)


def process_covid_channels():
    data_path = "/home/joshua/Desktop/data/pre_covid"
    ct_path = os.path.join(data_path, "ct_scans")
    mask_path = os.path.join(data_path, "lung_and_infection_mask")
    # train_file = os.path.join(data_path, "train.txt")
    # val_file = os.path.join(data_path, "val.txt")
    all_file = os.path.join(data_path, "all.txt")

    train_data_path = "/home/joshua/Desktop/data/covid-2/train"
    train_input_path = os.path.join(train_data_path, "inputs")
    train_target_path = os.path.join(train_data_path, "targets")

    val_data_path = "/home/joshua/Desktop/data/covid-2/validation"
    val_input_path = os.path.join(val_data_path, "inputs")
    val_target_path = os.path.join(val_data_path, "targets")

    f = open(all_file, 'r')

    curr_train = 0
    curr_val = 0
    for row in tqdm(f):
        i_f, t_f = row.rstrip().split(" ")

        scan_ori = np.array(nift.load(os.path.join(ct_path, i_f)).get_fdata(), dtype=np.uint8)
        scan_mask = np.array(nift.load(os.path.join(mask_path, t_f)).get_fdata(), dtype=np.uint8)

        assert scan_ori.shape[2] == scan_mask.shape[2], "Scan and mask mismatch on dim 3"

        indices = np.random.permutation(scan_ori.shape[2])

        train_size = int(scan_ori.shape[2]*0.6)
        val_size = scan_ori.shape[2]-train_size

        # print(scan_ori.shape[2], train_size, val_size)

        train_idx, val_idx = indices[:train_size], indices[train_size:]

        train_ins, train_outs = scan_ori[:, :, train_idx], scan_mask[:, :, train_idx]
        val_ins, val_outs = scan_ori[:, :, val_idx], scan_mask[:, :, val_idx]

        for i in range(train_size):
            plt.imsave(os.path.join(train_input_path, str(curr_train) + ".jpg"), train_ins[:, :, i])
            # plt.imsave(os.path.join(train_target_path, str(curr_train)+".png"), train_outs[:, :, i])
            np.save(os.path.join(train_target_path, str(curr_train) + ".npy"), train_outs[:, :, i])
            curr_train += 1

        for i in range(val_size):
            plt.imsave(os.path.join(val_input_path, str(curr_val) + ".jpg"), val_ins[:, :, i])
            # plt.imsave(os.path.join(val_target_path, str(curr_val) + ".png"), val_outs[:, :, i])
            np.save(os.path.join(val_target_path, str(curr_val) + ".npy"), val_outs[:, :, i])
            curr_val += 1


def process_covid_scans():
    data_path = "/home/joshua/Desktop/data/pre_covid"
    ct_path = os.path.join(data_path, "ct_scans")
    mask_path = os.path.join(data_path, "lung_and_infection_mask")
    # train_file = os.path.join(data_path, "train.txt")
    # val_file = os.path.join(data_path, "val.txt")
    all_file = os.path.join(data_path, "all.txt")

    train_data_path = "/home/joshua/Desktop/data/covid/train"
    train_input_path = os.path.join(train_data_path, "inputs")
    train_target_path = os.path.join(train_data_path, "targets")

    val_data_path = "/home/joshua/Desktop/data/covid/validation"
    val_input_path = os.path.join(val_data_path, "inputs")
    val_target_path = os.path.join(val_data_path, "targets")

    f = open(all_file, 'r')

    scan_count = 0
    curr_train = 0
    curr_val = 0
    for row in tqdm(f):
        i_f, t_f = row.rstrip().split(" ")

        scan_ori = np.array(nift.load(os.path.join(ct_path, i_f)).get_fdata(), dtype=np.uint8)
        scan_mask = np.array(nift.load(os.path.join(mask_path, t_f)).get_fdata(), dtype=np.uint8)

        assert scan_ori.shape[2] == scan_mask.shape[2], "Scan and mask mismatch on dim 3"

        if scan_count % 2 == 0:
            for i in range(scan_ori.shape[2]):
                plt.imsave(os.path.join(train_input_path, str(curr_train) + ".jpg"), scan_ori[:, :, i])
                # plt.imsave(os.path.join(train_target_path, str(curr_train)+".png"), scan_mask[:, :, i])
                np.save(os.path.join(train_target_path, str(curr_train) + ".npy"), scan_mask[:, :, i])
                curr_train += 1
        else:
            for i in range(scan_ori.shape[2]):
                plt.imsave(os.path.join(val_input_path, str(curr_val) + ".jpg"), scan_ori[:, :, i])
                # plt.imsave(os.path.join(val_target_path, str(curr_val) + ".png"), scan_mask[:, :, i])
                np.save(os.path.join(val_target_path, str(curr_val) + ".npy"), scan_mask[:, :, i])
                curr_val += 1

        scan_count += 1


def covid_check():
    f_data_path = "/home/joshua/Desktop/data/covid/train"
    input_path = os.path.join(f_data_path, "inputs")
    target_path = os.path.join(f_data_path, "targets")

    max = 0
    for i in tqdm(range(38523)):
        x = np.array(plt.imread(os.path.join(input_path, str(i) + ".jpg")))
        y = np.load(os.path.join(target_path, str(i) + ".npy"))

        curr = len(np.unique(y))

        if curr > max:
            print(np.unique(y))
            max = curr

    print(max)


if __name__ == '__main__':
    process_covid_scans()
