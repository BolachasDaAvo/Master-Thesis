import itertools

import cv2
import numpy as np


def gans():
    root = ""

    gans = ["ACGAN", "ADCGAN", "BigGAN", "ContraGAN", "cStyleGAN2", "MHGAN", "ProjGAN", "ReACGAN"]
    num_images = 10

    for gan in gans:
        save_path = f"{root}/{gan}/images"
        vertical_img = None
        for label in range(0, 4):
            horizontal_imgs = [cv2.imread(f"{save_path}/{label}/{num}.png") for num in range(num_images)]
            horizontal_imgs = cv2.hconcat(horizontal_imgs)

            if vertical_img is not None:
                vertical_img = cv2.vconcat([vertical_img, horizontal_imgs])
            else:
                vertical_img = horizontal_imgs

        cv2.imwrite(f"{save_path}/compiled.png", vertical_img)


images = []
save_path = f""

rows, cols = 3, 3
n = cols * rows
imgs = [cv2.imread(i) for i in images]
img_h, img_w, img_c = imgs[0].shape
m_x = 0
m_y = 0

imgmatrix = np.zeros((img_h * rows + m_y * (rows - 1),
                      img_w * cols + m_x * (cols - 1),
                      img_c),
                     np.uint8)

imgmatrix.fill(255)

positions = itertools.product(range(rows), range(cols))
for (y_i, x_i), img in zip(positions, imgs):
    x = x_i * (img_w + m_x)
    y = y_i * (img_h + m_y)
    imgmatrix[y:y + img_h, x:x + img_w, :] = img

cv2.imwrite(save_path, imgmatrix)
