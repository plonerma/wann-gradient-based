import numpy as np

import cv2


def mnist_256(test=False):
    '''
    Converts 28x28 mnist digits to [16x16]
    [samples x pixels]  ([N X 256])
    '''
    import mnist
    if not test:
        x = mnist.train_images()
        y_true = mnist.train_labels()
    else:
        x = mnist.test_images()
        y_true = mnist.test_labels()

    x = x/255
    x = preprocess(x,(16,16))
    x = x.reshape(-1, (256))

    return x, y_true

def preprocess(img,size, patchCorner=(0,0), patchDim=None, unskew=True):
    """
    Resizes, crops, and unskewes images
    """
    if patchDim is None:
        patchDim = size
    nImg = np.shape(img)[0]
    procImg  = np.empty((nImg,size[0],size[1]))

    # Unskew and Resize
    if unskew:
        for i in range(nImg):
            procImg[i,:,:] = deskew(cv2.resize(img[i,:,:],size),size)

    # Crop
    cropImg  = np.empty((nImg,patchDim[0],patchDim[1]))

    for i in range(nImg):
        cropImg[i,:,:] = procImg[i,patchCorner[0]:patchCorner[0]+patchDim[0],\
                                 patchCorner[1]:patchCorner[1]+patchDim[1]]
    procImg = cropImg

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
