import logging

import cv2
import numpy as np

import KCVraw


def array_to_int_tuple(X):
    return (int(X[0]), int(X[1]))


def main():
    logging.basicConfig(level=logging.DEBUG)

    with open('images.txt') as f:
        images = [line.strip() for line in f]

    kcv = KCVraw.KCVraw()

    init_region = np.genfromtxt('region.txt', delimiter=',')
    num_frames = len(images)

    results = np.empty((num_frames, 4))
    results[:] = np.nan

    results[0, :] = init_region

    frame = 0

    im0 = cv2.imread(images[frame])
    im_draw = np.copy(im0)

    tl, br = (array_to_int_tuple(init_region[:2]),
              array_to_int_tuple(init_region[:2] + init_region[2:4] - 1))

    try:
        kcv.initialise(im0, tl, br)
        while frame < num_frames:
            im = cv2.imread(images[frame])
            kcv.process_frame(im)
            results[frame, :] = kcv.bb

            # Advance frame number
            frame += 1
    except Exception as e:
        logging.exception(e)

    np.savetxt('output.txt', results, delimiter=',')

if __name__ == '__main__':
    main()
