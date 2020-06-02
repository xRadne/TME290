"""
Created by Jan Schiffeler on 12.05.20
jan.schiffeler[at]gmail.com

Changed by



Python 3.8
Library version:
openCV 4.2

TODO Mask edge image with colour thresholded image. So only complex black areas are detected
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


def find_kiwi_box(img_name):
    img_colour = cv2.imread(img_name, 1)
    img_colour_cut = img_colour[130:310, :, :]
    img = cv2.imread(img_name, 0)
    img_cut = img[130:310, :]
    img_edges = cv2.Canny(img_cut, 100, 200)

    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(img_edges, kernel, iterations=2)
    img_edges = cv2.erode(dilation, kernel, iterations=2)

    horizontal = np.sum(img_edges, 0)
    vertical = np.sum(img_edges, 1)
    horizontal_zero = np.zeros_like(horizontal)
    vertical_zero = np.zeros_like(vertical)

    # smoothing the curve
    for n in range(10):
        horizontal_zero += np.roll(horizontal, n) + np.roll(horizontal, -n)
        vertical_zero += np.roll(vertical, n) + np.roll(vertical, -n)
    horizontal += horizontal_zero
    vertical += vertical_zero

    horizontal_max = np.argmax(horizontal)
    vertical_max = np.argmax(vertical)
    
    # use fixed rect size (Half maximum doesn't work better)
    box_radius = 150
    v_l = vertical_max - box_radius if vertical_max - box_radius > 0 else 0
    v_u = vertical_max + box_radius if vertical_max + box_radius < vertical.shape else vertical.shape[0]
    h_l = horizontal_max - box_radius if horizontal_max - box_radius > 0 else 0
    h_u = horizontal_max + box_radius if horizontal_max + box_radius < horizontal.shape else horizontal.shape[0]

    # print(f'{h_u=}, {h_l=}, {v_u=}, {v_l=}')
    # print(horizontal[horizontal_max]/horizontal.mean())
    # print(np.median(horizontal))

    roi = img_cut[v_l:v_u, h_l:h_u]
    ret, roi = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY_INV)
    # roi = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
    #                       cv2.THRESH_BINARY, 49, 2)

    roi = cv2.erode(roi, kernel, iterations=5)
    roi = cv2.dilate(roi, kernel, iterations=8)

    contours, _ = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_poly = []
    boundRect = []
    for i, c in enumerate(contours):
        contours_poly.append(cv2.approxPolyDP(c, 3, True))
        boundRect.append(cv2.boundingRect(contours_poly[i]))

    drawing = np.repeat(roi[:, :, np.newaxis], 3, 2)
    data_roi = img_colour_cut[v_l:v_u, h_l:h_u].copy()
    drawing = cv2.bitwise_and(drawing, data_roi)

    # filter unlikely bounding boxes
    popper = []
    for i in range(len(boundRect)):
        width = int(boundRect[i][2])
        height = int(boundRect[i][3])
        if width < 80 or height < 80 or height/width > 1.5:
            popper.append(i)

    if popper is not None:
        popper.reverse()

    for i in popper:
        boundRect.pop(i)
        contours_poly.pop(i)

    for i in range(len(boundRect)):
        color = (180, 0, 255)
        # cv2.drawContours(drawing, contours_poly, i, color)
        cv2.rectangle(drawing,
                      (int(boundRect[i][0]), int(boundRect[i][1])),
                      (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])),
                      color, 2)

    if len(boundRect) != 1:
        detection = "No Kiwi Car"
    else:
        detection = "There is a Kiwi Car!"

    img_colour = cv2.rectangle(img_colour, (h_l, v_l + 130), (h_u, v_u + 130), (0, 255, 0), thickness=4)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    ax1.set(title=img_name)
    ax2.set(title="vertical edge density distribution")
    ax3.set(title="horizontal edge density distribution")
    ax4.set(title=detection)
    ax1.imshow(np.flip(img_colour, 2))
    ax3.plot(horizontal)
    ax2.plot(vertical, np.arange(vertical.shape[0]))
    ax4.imshow(np.flip(drawing, 2), cmap='gray')
    plt.show()


if __name__ == "__main__":
    import glob
    import os

    for pathAndFilename in glob.iglob(os.path.join("*.png")):
        # print(pathAndFilename)
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        find_kiwi_box(title + '.png')
