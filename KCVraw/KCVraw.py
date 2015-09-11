import time

import pylab
import scipy.misc


class KCVraw(object):

    def __init__(self):
        # parameters according to the paper --
        self.padding = 1.0  # extra area surrounding the target
        #spatial bandwidth (proportional to target)
        self.output_sigma_factor = 1 / float(16)
        self.sigma = 0.2  # gaussian kernel bandwidth
        self.lambda_value = 1e-2  # regularization
        # linear interpolation factor for adaptation
        self.interpolation_factor = 0.075

    def initialise(self, img, tl, br):
        self.is_first_frame = True

        self.w = br[0]-tl[0]
        self.h = br[1]-tl[1]

        info = KCVraw.load_info(tl, br)
        img_files, self.pos, target_sz, \
            self.should_resize_image, ground_truth, video_path = info

        # window size, taking padding into account
        self.sz = pylab.floor(target_sz * (1 + self.padding))

        # desired output (gaussian shaped), bandwidth proportional to target
        # size
        output_sigma = pylab.sqrt(pylab.prod(target_sz)) * \
            self.output_sigma_factor

        grid_y = pylab.arange(self.sz[0]) - pylab.floor(self.sz[0]/2)
        grid_x = pylab.arange(self.sz[1]) - pylab.floor(self.sz[1]/2)
        #[rs, cs] = ndgrid(grid_x, grid_y)
        rs, cs = pylab.meshgrid(grid_x, grid_y)
        y = pylab.exp(-0.5 / output_sigma**2 * (rs**2 + cs**2))
        self.yf = pylab.fft2(y)

        # store pre-computed cosine window
        self.cos_window = pylab.outer(pylab.hanning(self.sz[0]),
                                      pylab.hanning(self.sz[1]))

        global z, response
        z = None
        response = None
        self.alphaf = None

    def process_frame(self, im):
        global z, response

        if len(im.shape) == 3 and im.shape[2] > 1:
            im = KCVraw.rgb2gray(im)

        if self.should_resize_image:
            im = scipy.misc.imresize(im, 0.5)

        start_time = time.time()

        # extract and pre-process subwindow
        x = KCVraw.get_subwindow(im, self.pos, self.sz, self.cos_window)

        if not self.is_first_frame:
            # calculate response of the classifier at all locations
            k = KCVraw.dense_gauss_kernel(self.sigma, x, z)
            kf = pylab.fft2(k)
            alphaf_kf = pylab.multiply(self.alphaf, kf)
            response = pylab.real(pylab.ifft2(alphaf_kf))  # Eq. 9

            # target location is at the maximum response
            r = response
            row, col = pylab.unravel_index(r.argmax(), r.shape)
            self.pos = self.pos - pylab.floor(self.sz/2) + [row, col]

        # end "if not first frame"

        # get subwindow at current estimated target position,
        # to train classifer
        x = KCVraw.get_subwindow(im, self.pos, self.sz, self.cos_window)

        # Kernel Regularized Least-Squares,
        # calculate alphas (in Fourier domain)
        k = KCVraw.dense_gauss_kernel(self.sigma, x)
        # Eq. 7
        new_alphaf = pylab.divide(self.yf, (pylab.fft2(k) + self.lambda_value))
        new_z = x

        if self.is_first_frame:
            #first frame, train with a single image
            self.alphaf = new_alphaf
            z = x
        else:
            # subsequent frames, interpolate model
            f = interpolation_factor
            self.alphaf = (1 - f) * self.alphaf + f * new_alphaf
            z = (1 - f) * z + f * new_z
        # end "first frame or not"

        center = (self.pos[1], self.pos[0]) \
            if not self.should_resize_image else \
            (2*self.pos[1], 2*self.pos[0])

        self.bb = (center[0]-self.w/2, center[1]-self.h/2, self.w, self.h)

        self.is_is_first_frame = False

    @staticmethod
    def load_info(tl, br):
        x = tl[0]
        y = tl[1]
        w = br[0]-tl[0]
        h = br[1]-tl[1]

        # set initial position and size
        first_ground_truth = [x, y, w, h]
        # target_sz contains height, width
        target_sz = pylab.array([first_ground_truth[3], first_ground_truth[2]])
        # pos contains y, x center
        pos = [first_ground_truth[1], first_ground_truth[0]] \
            + pylab.floor(target_sz / 2)

        # if the target is too large, use a lower resolution
        # no need for so much detail
        if pylab.sqrt(pylab.prod(target_sz)) >= 100:
            pos = pylab.floor(pos / 2)
            target_sz = pylab.floor(target_sz / 2)
            resize_image = True
        else:
            resize_image = False

        ret = [None, pos, target_sz, resize_image, None, None]
        return ret

    @staticmethod
    def rgb2gray(rgb_image):
        "Based on http://stackoverflow.com/questions/12201577"
        # [0.299, 0.587, 0.144] normalized gives [0.29, 0.57, 0.14]
        return pylab.dot(rgb_image[:, :, :3], [0.29, 0.57, 0.14])

    @staticmethod
    def get_subwindow(im, pos, sz, cos_window):
        """
        Obtain sub-window from image, with replication-padding.
        Returns sub-window of image IM centered at POS ([y, x] coordinates),
        with size SZ ([height, width]). If any pixels are outside of the image,
        they will replicate the values at the borders.

        The subwindow is also normalized to range -0.5 .. 0.5, and the given
        cosine window COS_WINDOW is applied
        (though this part could be omitted to make the function more general).
        """

        if pylab.isscalar(sz):  # square sub-window
            sz = [sz, sz]

        ys = pylab.floor(pos[0]) \
            + pylab.arange(sz[0], dtype=int) - pylab.floor(sz[0]/2)
        xs = pylab.floor(pos[1]) \
            + pylab.arange(sz[1], dtype=int) - pylab.floor(sz[1]/2)

        ys = ys.astype(int)
        xs = xs.astype(int)

        # check for out-of-bounds coordinates,
        # and set them to the values at the borders
        ys[ys < 0] = 0
        ys[ys >= im.shape[0]] = im.shape[0] - 1

        xs[xs < 0] = 0
        xs[xs >= im.shape[1]] = im.shape[1] - 1
        #zs = range(im.shape[2])

        # extract image
        #out = im[pylab.ix_(ys, xs, zs)]
        out = im[pylab.ix_(ys, xs)]

        #pre-process window --
        # normalize to range -0.5 .. 0.5
        # pixels are already in range 0 to 1
        out = out.astype(pylab.float64) - 0.5

        # apply cosine window
        out = pylab.multiply(cos_window, out)

        return out

    @staticmethod
    def dense_gauss_kernel(sigma, x, y=None):
        """
        Gaussian Kernel with dense sampling.
        Evaluates a gaussian kernel with bandwidth SIGMA for all displacements
        between input images X and Y, which must both be MxN. They must also
        be periodic (ie., pre-processed with a cosine window). The result is
        an MxN map of responses.

        If X and Y are the same, ommit the third parameter to re-use some
        values, which is faster.
        """

        xf = pylab.fft2(x)  # x in Fourier domain
        x_flat = x.flatten()
        xx = pylab.dot(x_flat.transpose(), x_flat)  # squared norm of x

        if y is not None:
            # general case, x and y are different
            yf = pylab.fft2(y)
            y_flat = y.flatten()
            yy = pylab.dot(y_flat.transpose(), y_flat)
        else:
            # auto-correlation of x, avoid repeating a few operations
            yf = xf
            yy = xx

        # cross-correlation term in Fourier domain
        xyf = pylab.multiply(xf, pylab.conj(yf))

        # to spatial domain
        xyf_ifft = pylab.ifft2(xyf)
        #xy_complex = circshift(xyf_ifft, floor(x.shape/2))
        row_shift, col_shift = pylab.floor(pylab.array(x.shape)/2).astype(int)
        xy_complex = pylab.roll(xyf_ifft, row_shift, axis=0)
        xy_complex = pylab.roll(xy_complex, col_shift, axis=1)
        xy = pylab.real(xy_complex)

        # calculate gaussian response for all positions
        scaling = -1 / (sigma**2)
        xx_yy = xx + yy
        xx_yy_2xy = xx_yy - 2 * xy
        k = pylab.exp(scaling * pylab.maximum(0, xx_yy_2xy / x.size))

        return k