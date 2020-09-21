import numpy as np

import cv2

from .util import prepare_data


def load_mnist(test=False):
    import mnist
    if not test:
        x = mnist.train_images()
        y = mnist.train_labels()
    else:
        x = mnist.test_images()
        y = mnist.test_labels()

    x = x/255
    return x, y


def mnist_256(test=False, **datafmt):
    '''
    Converts 28x28 mnist digits to [16x16]
    [samples x pixels]  ([N X 256])
    '''
    x, y = load_mnist(test)

    x = preprocess(x, (16, 16))
    x = x.reshape(-1, (256))

    return prepare_data(x, y, **datafmt)


def mnist_full(test=False, **datafmt):
    """28x28 mnist digits"""
    x, y = load_mnist(test)

    x = preprocess(x)
    x = x.reshape(-1, (28*28))

    return prepare_data(x, y, **datafmt)


def preprocess(images, size=None):
    """Resizes, and unskewes images. """

    nImg = np.shape(images)[0]

    if size is None:
        procImg = np.empty_like(images, dtype='float')
    else:
        procImg = np.empty((nImg, size[0], size[1]))

    # Unskew and Resize
    for i in range(nImg):
        img = images[i, :, :]
        if size is not None:
            img = cv2.resize(img, size)
        procImg[i, :, :] = deskew(img, img.shape)

    return procImg


def deskew(image, image_shape, negated=True):
    """
    This method deskwes an image using moments
    :param image: a numpy nd array input image
    :param image_shape: a tuple denoting the image`s shape
    :param negated: a boolean flag telling whether the input image is negated
    :returns: a numpy nd array deskewd image
    source: https://github.com/vsvinayak/mnist-helper
    """

    # negate the image
    if not negated:
        image = 255-image
    # calculate the moments of the image
    m = cv2.moments(image)
    if abs(m['mu02']) < 1e-2:
        return image.copy()
    # caclulating the skew
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*image_shape[0]*skew], [0,1,0]])
    img = cv2.warpAffine(image, M, image_shape, \
                         flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR)
    return img
