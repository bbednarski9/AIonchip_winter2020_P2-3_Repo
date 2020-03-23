from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

IMG_SIZE = 416
IMG_DIR = "/home/bryanbed/Documents/tvm/Quantization_PJ2/image_preprocessing/"
IMG_NAME = "ILSVRC2012_val_00001110.JPEG"

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)
    return img, pad

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def preprocess_image():
    img_path = IMG_DIR + IMG_NAME
    img = Image.open(img_path)

    transformer = transforms.ToTensor()
    img = transformer(img)
    img, _ = pad_to_square(img, 0)
    print(img)
    img = resize(img, IMG_SIZE)
    img = np.array(img)
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 3)
    np.save('preprocessed.npy', img)

preprocess_image()
