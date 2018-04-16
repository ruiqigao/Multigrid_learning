import numpy as np
import tensorflow as tf
import skimage
import pprint
from sklearn.externals.joblib import Parallel, delayed
import scipy.misc
from skimage import io

pp = pprint.PrettyPrinter()


def imread_url(url, image_size):
    try:
        buck = connect_s3()
        k = boto.s3.key.Key(buck)
        k.key = url
        jpgdata = k.get_contents_as_string()
        file_name = StringIO(jpgdata)
        #file_name = k.generate_url(expires_in = 2400)
        img = np.array(scipy.misc.imresize(io.imread(file_name), [image_size, image_size]))
        if len(img.shape) < 3 or img.shape[2] != 3:
            img = img[...,1]
            img = np.tile(img[...,None], [1,1,3])
        if img.shape != (image_size, image_size, 3):
            print('Image {} size {} is not equal to {}, replace with an empty image'.format(url, image.shape, (image_size, image_size, 3)))
            img = np.zeros([image_size, image_size, 3]).astype(np.uint8)
    except:
        print("Cannot read image from {}".format(url))
        img = np.zeros([image_size, image_size, 3]).astype(np.uint8)
    return img


def imread_file(file_name, image_size):
    img = np.array(scipy.misc.imresize(skimage.io.imread(file_name), [image_size, image_size]))
    if len(img.shape) < 3 or img.shape[2] != 3:
        print(img.shape)
        img = img[...,1]
        img = np.tile(img[..., None], [1, 1, 3])
    if img.shape != (image_size, image_size, 3):
        print('Image {} size {} is not equal to {}, replace with an empty image'.format(file_name, img.shape, (image_size, image_size, 3)))
        img = np.zeros([image_size, image_size, 3]).astype(np.uint8)
    return img


def par_imread(files, image_size, num_threads=8, from_url=False):
    if from_url:
        return Parallel(n_jobs=num_threads, verbose=5)(
            delayed(imread_url)(f, image_size) for f in files
        )
    else:
        return Parallel(n_jobs=num_threads, verbose=5)(
            delayed(imread_file)(f, image_size) for f in files
        )


def merge_images(images, space=0, mean_img=None):
    num_images = images.shape[0]
    canvas_size = int(np.ceil(np.sqrt(num_images)))
    h = images.shape[1]
    w = images.shape[2]
    canvas = np.zeros((canvas_size * h + (canvas_size-1) * space,  canvas_size * w + (canvas_size-1) * space, 3), np.uint8)

    for idx in xrange(num_images):
        image = images[idx,:,:,:]
        if mean_img:
            image += mean_img
        i = idx % canvas_size
        j = idx // canvas_size
        min_val = np.min(image)
        max_val = np.max(image)
        image = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        canvas[j*(h+space):j*(h+space)+h, i*(w+space):i*(w+space)+w,:] = image
    return canvas


def save_images(images, file_name, space=0, mean_img=None):
    skimage.io.imsave(file_name, merge_images(images, space, mean_img))

