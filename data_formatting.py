import pathlib
import zipfile

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from tqdm import tqdm_notebook as tqdm

HEIGHT = 137
WIDTH = 236
SIZE = 128

DATA_PATH = "bengaliai-cv19/"

TRAIN = [DATA_PATH + 'train_image_data_0.parquet',
         DATA_PATH + 'train_image_data_1.parquet',
         DATA_PATH + 'train_image_data_2.parquet',
         DATA_PATH + 'train_image_data_3.parquet', ]

OUT_TRAIN = 'train_images/'

DATA_PATH = "bengaliai-cv19/"
#
train_indices = pd.read_csv(DATA_PATH + "train.csv")


def bbox(img):
	rows = np.any(img, axis=1)
	cols = np.any(img, axis=0)
	rmin, rmax = np.where(rows)[0][[0, -1]]
	cmin, cmax = np.where(cols)[0][[0, -1]]
	return rmin, rmax, cmin, cmax


def crop_resize(img0, size=SIZE, pad=16):
	# crop a box around pixels large than the threshold
	# some images contain line at the sides
	ymin, ymax, xmin, xmax = bbox(img0[5:-5, 5:-5] > 80)
	# cropping may cut too much, so we need to add it back
	xmin = xmin - 13 if (xmin > 13) else 0
	ymin = ymin - 10 if (ymin > 10) else 0
	xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
	ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
	img = img0[ymin:ymax, xmin:xmax]
	# remove lo intensity pixels as noise
	img[img < 28] = 0
	lx, ly = xmax - xmin, ymax - ymin
	l = max(lx, ly) + pad
	# make sure that the aspect ratio is kept in rescaling
	img = np.pad(img, [((l - ly) // 2,), ((l - lx) // 2,)], mode='constant')
	return cv2.resize(img, (size, size))


def export_images():
	df = pd.read_parquet(TRAIN[0])
	n_imgs = 8
	fig, axs = plt.subplots(n_imgs, 2, figsize=(10, 5 * n_imgs))

	for idx in range(n_imgs):
		# somehow the original input is inverted
		img0 = df.iloc[idx, 1:].values.reshape(HEIGHT, WIDTH).astype(np.uint8)
		# normalize each image by its max val
		img = (img0 * (255.0 / img0.max())).astype(np.uint8)
		img = crop_resize(img)

		axs[idx, 0].imshow(img0)
		axs[idx, 0].set_title('Original image')
		axs[idx, 0].axis('off')
		axs[idx, 1].imshow(img)
		axs[idx, 1].set_title('Crop & resize')
		axs[idx, 1].axis('off')
	plt.show()

	x_tot, x2_tot = [], []

	for fname in TRAIN:
		df = pd.read_parquet(fname)
		# the input is inverted
		data = 255 - df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)
		for index, idx in enumerate(tqdm(range(len(df)))):
			name = df.iloc[idx, 0]
			# normalize each image by its max val
			img = (data[idx] * (255.0 / data[idx].max())).astype(np.uint8)
			img = crop_resize(img)

			x_tot.append((img / 255.0).mean())
			x2_tot.append(((img / 255.0) ** 2).mean())
			img = cv2.imencode('.png', img)[1]
			current_index = train_indices[index]
			with zipfile.ZipFile(OUT_TRAIN + f'', 'w') as img_out:
				img_out.writestr(name + '.png', img)


def move_images_into_classes():
	file_names = os.walk("train_images/")
	file_names = list(file_names)[0][2]
	print(train_indices)
	for file in file_names:
		indexable = file.replace(".png", '')
		file_class = train_indices.loc[train_indices.image_id == indexable]["grapheme_root"].values[0]
		print(file_class)
		if not os.path.exists(f"train_images/grapheme_root_{file_class}"):
			os.mkdir(f"train_images/grapheme_root_{file_class}")
		os.rename(f"train_images/{file}", f"train_images/grapheme_root_{file_class}/{file}")
		print(file_class)


IMG_WIDTH = 128
IMG_HEIGHT = 128
CLASS_NAMES = [root for root in list(os.walk("train_images"))[0][1]]


def get_label(file_path):
	# convert the path to a list of path components
	parts = tf.strings.split(file_path, os.path.sep)
	# The second to last is the class-directory
	return parts[-2] == CLASS_NAMES


def decode_img(img):
	# convert the compressed string to a 3D uint8 tensor
	img = tf.image.decode_jpeg(img, channels=1)
	# Use `convert_image_dtype` to convert to floats in the [0,1] range.
	img = tf.image.convert_image_dtype(img, tf.float32)
	# resize the image to the desired size.
	return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])


def process_path(file_path):
	label = get_label(file_path)
	# load the raw data from the file as a string
	img = tf.io.read_file(file_path)
	img = decode_img(img)
	return img, label


BATCH_SIZE = 5


def build_tensorflow_dataset(cache=True, shuffle_buffer_size=1000):
	data_dir = pathlib.Path("train_images")
	file_list = tf.data.Dataset.list_files(str(data_dir / '*/*'))
	AUTOTUNE = tf.data.experimental.AUTOTUNE
	labeled_ds = file_list.map(process_path, num_parallel_calls=AUTOTUNE)

	if cache:
		if isinstance(cache, str):
			ds = labeled_ds.cache(cache)
		else:
			ds = labeled_ds.cache()

	ds = labeled_ds.shuffle(buffer_size=shuffle_buffer_size)

	# Repeat forever
	ds = ds.repeat()

	ds = ds.batch(BATCH_SIZE)

	# `prefetch` lets the dataset fetch batches in the background while the model
	# is training.
	ds = ds.prefetch(buffer_size=AUTOTUNE)

	return ds


# move_images_into_classes()
