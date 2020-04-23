import argparse
import os
import requests
import numpy as np
import cv2 as cv
import math

IMG_MAX_DIM = 225

def create_output_dir(dir):
    if dir[-1] != '/':
        dir += '/'

    if not os.path.exists(dir):
        os.mkdir(dir)

    return dir

def resize_image(img):
    if max(img.shape) != IMG_MAX_DIM:
        resize_percent = IMG_MAX_DIM / max(img.shape)
        # For some reason this tuple needs to be (width, height), but img.shape is (height, width)
        resized_size = (int(resize_percent * img.shape[1]), int(resize_percent * img.shape[0]))
        img = cv.resize(img, resized_size, interpolation=cv.INTER_CUBIC)

    top = bottom = left = right = 0
    if img.shape[0] < IMG_MAX_DIM:
        top = math.floor((IMG_MAX_DIM - img.shape[0]) / 2)
        bottom = math.ceil((IMG_MAX_DIM - img.shape[0]) / 2)
    if img.shape[1] < IMG_MAX_DIM:
        left = math.floor((IMG_MAX_DIM - img.shape[1]) / 2)
        right = math.floor((IMG_MAX_DIM - img.shape[1]) / 2)

    return cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0))


def download_image(url, directory_name, index):
    r = requests.get(url)
    img_array = np.asarray(bytearray(r.content))

    img = cv.imdecode(img_array, cv.IMREAD_COLOR)
    img = resize_image(img)
    cv.imwrite(f'{directory_name}/{index}.png', img)
    print("Written")

def main():
    parser = argparse.ArgumentParser(description='Download imagse from a file containing a list of URLs')
    parser.add_argument('--input', '-i')
    parser.add_argument('--output', '-o')
    parser.add_argument('--start', '-s', default=0, type=int)
    args = parser.parse_args()

    dir = create_output_dir(args.output)

    with open(args.input) as f:
        urls = list(filter(lambda x: len(x.strip()) > 0, f.read().split('\n')))
        for i, url in enumerate(urls):
            download_image(url, dir, i + args.start)

if __name__ == '__main__':
    main()