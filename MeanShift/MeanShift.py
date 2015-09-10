
import cv2
import numpy as np


class MeanShift(object):

    def __init__(self):
        pass

    def initialise(self, img, tl, br):
        x = tl[0]
        y = tl[1]
        w = br[0]-tl[0]
        h = br[1]-tl[1]

        self.bb_init = (x, y, w, h)
        self.bb = (x, y, w, h)

        roi = img[tl[1]:br[1], tl[0]:br[0]]
        hsv_roi = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)),
                           np.array((180., 255., 255.)))
        self.roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)

        # Setup the termination criteria, either 10
        # iteration or move by at least 1 pt
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10,
                          1)

    def process_frame(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)

        # apply meanshift to get the new location
        ret, self.bb = cv2.meanShift(dst, self.bb, self.term_crit)
