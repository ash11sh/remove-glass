import io
import os
import sys
import cv2
import aiohttp
import asyncio
import uvicorn
import numpy as np
from io import BytesIO
from pathlib import Path
from starlette.applications import Starlette
from starlette.staticfiles import StaticFiles

from starlette.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from starlette.middleware.cors import CORSMiddleware
from PIL import Image, ImageFile, ImageFilter, ImageEnhance, ImageOps

import tflite_runtime.interpreter as tflite
from misc import get_potrait, upscale

import torch
import contextlib
from data.base_dataset import get_transform
from models.cut_model import CUTModel
from util.util import tensor2im
from argparse import Namespace
from pathlib import Path
from copy import deepcopy
from Models import *


os.environ["LRU_CACHE_CAPACITY"] = "1"
port = int(os.environ.get("PORT", 5001))


path = Path(__file__).parent


app = Starlette()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["X-Requested-With", "Content-Type"],
)
app.mount("/static", StaticFiles(directory="app/static"))


# CUTGAN input options

OPT = Namespace(
    batch_size=1,
    checkpoints_dir="cyclegan",
    crop_size=256,
    # dataroot=".",
    dataset_mode="unaligned",
    direction="AtoB",
    display_id=-1,
    display_winsize=256,
    epoch="latest",
    eval=False,
    gpu_ids=[],
    nce_layers="0,4,8,12,16",
    nce_idt=False,
    lambda_NCE=10.0,
    lambda_GAN=1.0,
    init_gain=0.02,
    nce_includes_all_negatives_from_minibatch=False,
    init_type="xavier",
    normG="instance",
    no_antialias=False,
    no_antialias_up=False,
    netF="mlp_sample",
    netF_nc=256,
    nce_T=0.07,
    num_patches=256,
    CUT_mode="FastCUT",
    input_nc=3,
    isTrain=False,
    load_iter=0,
    load_size=256,
    max_dataset_size=float("inf"),
    model="CUT",
    n_layers_D=3,
    name=None,
    ndf=64,
    netD="basic",
    netG="resnet_9blocks",
    ngf=64,
    no_dropout=True,
    no_flip=True,
    num_test=50,
    num_threads=4,
    output_nc=3,
    phase="test",
    preprocess="scale_width",
    random_scale_max=3.0,
    results_dir="./results/",
    serial_batches=True,
    suffix="",
    verbose=False,
)


class SingleImageDataset(torch.utils.data.Dataset):
    """dataset with precisely one image"""

    def __init__(self, img, preprocess):
        img = preprocess(img)
        self.img = img

    def __getitem__(self, i):
        return self.img

    def __len__(self):
        return 1


fp = "app/cyclegan/EyeFastcut/latest_net_G.pth"
opt = deepcopy(OPT)
model_name = "EyeFastcut"
opt.name = model_name
if opt.verbose:
    # model = load_model(opt, model_fp)
    model = CUTModel(opt).netG
    model.load_state_dict(torch.load(fp))
else:
    with contextlib.redirect_stdout(io.StringIO()):
        # model = load_model(opt, model_fp)
        model = CUTModel(opt).netG
        model.load_state_dict(torch.load(fp))


# inference code for single image - cutgan

"""reference inference code:

https://www.jeremyafisher.com/running-cyclegan-programmatically.html

"""
def cutgan(img: Image) -> Image:
    img = img.convert("RGB")
    data_loader = torch.utils.data.DataLoader(
        SingleImageDataset(img, get_transform(opt)), batch_size=1
    )
    data = next(iter(data_loader))
    with torch.no_grad():
        pred = model(data)
    pred_arr = tensor2im(pred)
    pred_img = Image.fromarray(pred_arr)
    return pred_img


# load image upscale model

model_cran_v2 = CARN_V2(
    color_channels=3,
    mid_channels=64,
    conv=nn.Conv2d,
    single_conv_size=3,
    single_conv_group=1,
    scale=2,
    activation=nn.LeakyReLU(0.1),
    SEBlock=True,
    repeat_blocks=3,
    atrous=(1, 1, 1),
)


model_cran_v2 = network_to_half(model_cran_v2)
checkpoint = "app/model_check_points/CRAN_V2/CARN_model_checkpoint.pt"
model_cran_v2.load_state_dict(torch.load(checkpoint, "cpu"))
# if use GPU, then comment out the next line so it can use fp16.
model_cran_v2 = model_cran_v2.float()


# Initialize the tflie interpreter for potrait segmentation
interpreter = tflite.Interpreter(
    model_path="app/model_check_points/slim_reshape_v2.tflite"
)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]


@app.route("/")
async def homepage(request):
    html_file = path / "view" / "index.html"
    if os.path.exists("app/removal.png"):
        os.remove("app/removal.png")
    else:
        print("The file does not exist")
    return HTMLResponse(html_file.open().read())


@app.route("/removal", methods=["POST", "GET"])
async def removal(request):

    img_data = await request.form()
    img_bytes = await (img_data["file"].read())
    im = Image.open(BytesIO(img_bytes))

    if __name__ == "__main__":

        # change img size and orientation (exif remover)
        im = ImageOps.exif_transpose(im)
        width, height = im.size
        ori_im = im.copy()

        # get potrait and mask
        im, mask = get_potrait(im, interpreter, input_details, output_details)

        # send image to model to remove glasses
        im = cutgan(im)

        # composite original image and output based on mask
        w, h = im.size
        ori_im = ori_im.resize((w, h))
        mask = mask.resize((w, h))
        img = Image.composite(im, ori_im, mask)

        # upscale the image
        img = upscale(img, model_cran_v2)

        # scale image to original size
        img = img.resize((width, height))
        img.save("app/removal.png")

    return FileResponse("app/removal.png", media_type="image/png")


if __name__ == "__main__":
    if "serve" in sys.argv:
        uvicorn.run(app=app, host="0.0.0.0", port=port, log_level="info")
