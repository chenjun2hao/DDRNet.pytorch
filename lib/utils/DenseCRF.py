# -*- coding: utf-8 -*-
#
# Author: Kazuto Nakashima
# URL:    https://kazuto1011.github.io
# Date:   09 January 2019

import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils

class DenseCRF(object):
    def __init__(self, max_epochs=5, delta_aphla=80, delta_beta=3, w1=10, delta_gamma=3, w2=3):
        self.max_epochs = max_epochs
        self.delta_gamma = delta_gamma
        self.delta_alpha = delta_aphla
        self.delta_beta = delta_beta
        self.w1 = w1
        self.w2 = w2

    def __call__(self, image, probmap):
        c, h, w = probmap.shape

        U = utils.unary_from_softmax(probmap)
        U = np.ascontiguousarray(U)

        image = np.ascontiguousarray(image)

        d = dcrf.DenseCRF2D(w, h, c)
        d.setUnaryEnergy(U)

        d.addPairwiseGaussian(sxy=self.delta_gamma, compat=self.w2)
        d.addPairwiseBilateral(sxy=self.delta_alpha, srgb=self.delta_beta, rgbim=image, compat=self.w1)

        Q = d.inference(self.max_epochs)
        Q = np.array(Q).reshape((c, h, w))

        return Q

# import numpy as np
# import pydensecrf.densecrf as dcrf
# import pydensecrf.utils as utils


# class DenseCRF(object):
#     def __init__(self, iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std):
#         self.iter_max = iter_max       # iter num
#         self.pos_w = pos_w             # the weight of the Gaussian kernel which only depends on Pixel Position
#         self.pos_xy_std = pos_xy_std
#         self.bi_w = bi_w               # the weight of bilateral kernel
#         self.bi_xy_std = bi_xy_std
#         self.bi_rgb_std = bi_rgb_std

#     def __call__(self, image, probmap):
#         C, H, W = probmap.shape

#         U = utils.unary_from_softmax(probmap)
#         U = np.ascontiguousarray(U)

#         image = np.ascontiguousarray(image)

#         d = dcrf.DenseCRF2D(W, H, C)
#         d.setUnaryEnergy(U)

#         # the gaussian kernel depends only on pixel position
#         d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)

#         # bilateral kernel depends on both position and color
#         d.addPairwiseBilateral(
#             sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w
#         )

#         Q = d.inference(self.iter_max)
#         Q = np.array(Q).reshape((C, H, W))

#         return Q