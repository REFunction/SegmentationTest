import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
import numpy as np

def crf(image, pred_mask, num_classes=21):
    '''
    inference with crf
    :param image: shape [height, width, 3]
    :param pred_mask:shape [height, width, num_classes] after softmax
    :return:refined mask
    '''
    height = np.shape(image)[0]
    width = np.shape(image)[1]
    # create densecrf
    d = dcrf.DenseCRF2D(width, height, num_classes)
    # Unary potential
    U = unary_from_softmax(pred_mask)
    U = U.reshape((num_classes, -1))
    d.setUnaryEnergy(U)
    # Pairwise potentials
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image, compat=10)
    # inference
    Q = d.inference(5)
    Q = np.reshape(Q, [1, height, width, num_classes])
    Q = np.reshape(np.argmax(Q, axis=3), [1, height, width])

    return Q
