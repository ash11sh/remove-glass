import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from util.prepare_images import *
from torchvision.utils import save_image
import os


os.environ["LRU_CACHE_CAPACITY"] = "1"


def get_potrait(test_image, interpreter,input_details,output_details):

    # get the potrait mask output
    im = np.asarray(test_image)
    h, w, _ = im.shape
    face_rgba = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

    # resize
    image = cv2.resize(face_rgba, (512, 512), interpolation=cv2.INTER_AREA)

    # Preprocess the input image
    test_image = image / 255.0
    test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])

    # Run the interpreter and get the output
    interpreter.set_tensor(input_details["index"], test_image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]

    # Compute mask from segmentaion output
    mask = np.reshape(output, (512, 512)) > 0.5

    mask = (mask * 255).astype(np.uint8)

    # resize the mask output
    bin_mask = cv2.resize(mask, (w, h))

    # extract the potrait
    image = np.dstack((im, bin_mask))

    # make background white
    face = image[:, :, :3].copy()
    mask = image[:, :, 3].copy()[:, :, np.newaxis] / 255.0
    face_white_bg = (face * mask + (1 - mask) * 255).astype(np.uint8)

    # convert image to PIL format
    mask = Image.fromarray(bin_mask)
    im = Image.fromarray(face_white_bg)

    return im, mask


def upscale(img, model_cran_v2):

    # convert pil image to tensor
    img_t = transforms.ToTensor()(img).unsqueeze(0)

    # used to compare the origin
    img = img.resize((img.size[0] // 2, img.size[1] // 2), Image.BICUBIC)
    img_splitter = ImageSplitter(seg_size=64, scale_factor=2, boarder_pad_size=3)
    img_patches = img_splitter.split_img_tensor(img, scale_method=None, img_pad=0)
    with torch.no_grad():
        out = [model_cran_v2(i) for i in img_patches]
    img_upscale = img_splitter.merge_img_tensor(out)
    save_image(img_upscale, "app/removal.png")

    return Image.open("app/removal.png")
