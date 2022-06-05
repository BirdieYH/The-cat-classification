import tensorflow as tf
from PIL import Image
import numpy as np
import argparse
import json
import cv2
from utils.datasets import IMG_FORMATS, VID_FORMATS
from pathlib import Path
import sys
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def parse_args():
    """
    :return:进行参数的解析
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/data.json', help="The path of data.json")
    parser.add_argument('--model', help="The model for test")
    parser.add_argument('--source', help="The source of test")
    opt = parser.parse_args()
    return opt


def run(configs, model_path, source):
    source = str(source)

    model = tf.keras.models.load_model(model_path)
    names = configs['cats']

    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)

    if is_url and is_file:
        pass

    img = np.asarray(Image.open(source))  # 将图片转化为numpy的数组
    img = cv2.resize(img, (224, 224))# 将图片输入模型得到结果
    outputs = model.predict(img.reshape(1, configs['height'], configs['width'], 3))
    result = names[int(np.argmax(outputs))]# 获得对应的名称

    return result

if __name__ == '__main__':
    args = parse_args()
    print(run(json.load(open(args.data, 'r')), args.model, args.source))
