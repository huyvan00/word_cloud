from __future__ import division

import warnings
import random
from random import Random
import numpy as np
import cv2
from .query_integral_image import query_integral_image


class IntegralOccupancyMap(object):
    def __init__(self, height, width, mask):
        self.height = height
        self.width = width
        if mask is not None:
            # the order of the cumsum's is important for speed ?!
            self.integral = np.cumsum(np.cumsum(255 * mask, axis=1),
                                      axis=0).astype(np.uint32)
        else:
            self.integral = np.zeros((height, width), dtype=np.uint32)

    def sample_position(self, size_x, size_y, random_state):
        return query_integral_image(self.integral, size_x, size_y,
                                    random_state)

    def update(self, img_array, pos_x, pos_y):
        partial_integral = np.cumsum(np.cumsum(img_array[pos_x:, pos_y:],
                                               axis=1), axis=0)
        # paste recomputed part into old image
        # if x or y is zero it is a bit annoying
        if pos_x > 0:
            if pos_y > 0:
                partial_integral += (self.integral[pos_x - 1, pos_y:]
                                     - self.integral[pos_x - 1, pos_y - 1])
            else:
                partial_integral += self.integral[pos_x - 1, pos_y:]
        if pos_y > 0:
            partial_integral += self.integral[pos_x:, pos_y - 1][:, np.newaxis]

        self.integral[pos_x:, pos_y:] = partial_integral


class ImageCloud(object):
    def __init__(self, shape_mask, list_image=None, max_height=100, min_height=30, random_rotate=True, margin=0, height_step=2):
        self.shape_mask = shape_mask
        self.list_image = list_image
        self.max_height = max_height
        self.min_height = min_height
        self.random_rotate = random_rotate
        self.margin = margin
        self.height_step = height_step

    def generate_from_image(self):
        random_state = Random()
        boolean_mask = self._get_bolean_mask(self.shape_mask)
        width = self.shape_mask.shape[1]
        height = self.shape_mask.shape[0]
        occupancy = IntegralOccupancyMap(height, width, boolean_mask)

        image_idx, height_sizes, positions, rotate_angles = [], [], [], []

        height_size = self.max_height
        while True:
            if self.random_rotate:
                rotate_angle = random.randint(-30, 30)
            else:
                rotate_angle = 0
            index = random.choice(range(len(self.list_image)))

            while True:

                mask_image = self.resize_by_height(self.list_image[index], height_size)
                if self.random_rotate:
                    mask_image = self.rotate_bound(mask_image, rotate_angle)

                result = occupancy.sample_position(mask_image.shape[0] + self.margin,
                                                   mask_image.shape[1] + self.margin,
                                                   random_state)
                if result is not None or height_size < self.min_height:
                    # either we found a place or font-size went too small
                    break
                # if we didn't find a place, make font smaller
                # but first try to rotate!
                height_size -= self.height_step

            if height_size < self.min_height:
                # we were unable to draw any more
                break

            y, x = np.array(result) + self.margin // 2
            ## place image
            tmp_image = boolean_mask[y:y+mask_image.shape[0], x:x+mask_image.shape[1]]
            tmp_boolean_image = mask_image == 255
            tmp_boolean_image = np.logical_or(tmp_image, tmp_boolean_image)
            boolean_mask[y:y + mask_image.shape[0], x:x + mask_image.shape[1]] = tmp_boolean_image

            occupancy.update(boolean_mask, y, x)

            image_idx.append(index)
            height_sizes.append(height_size)
            positions.append((x, y))
            rotate_angles.append(rotate_angle)
        layout_ = list(zip(image_idx, positions, height_sizes, rotate_angles))

        return layout_

    def rotate_bound(self, image, angle):
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w / 2, h / 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (nW, nH))

    def resize_by_height(self, image, height):
        scale = height/image.shape[0]
        output = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        return output

    def _get_bolean_mask(self, mask):
        """Cast to two dimensional boolean mask."""
        if mask.dtype.kind == 'f':
            warnings.warn("mask image should be unsigned byte between 0"
                          " and 255. Got a float array")
        if mask.ndim == 2:
            boolean_mask = mask == 255
        elif mask.ndim == 3:
            # if all channels are white, mask out
            boolean_mask = np.all(mask[:, :, :3] == 255, axis=-1)
        else:
            raise ValueError("Got mask of invalid shape: %s" % str(mask.shape))
        return boolean_mask
