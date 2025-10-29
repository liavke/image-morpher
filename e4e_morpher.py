import argparse

import torch
import numpy as np
import sys
import os
import dlib

sys.path.append(".")
sys.path.append("..")

from configs import data_configs, paths_config
from datasets.inference_dataset import InferenceDataset
from torch.utils.data import DataLoader
from utils.model_utils import setup_model
from utils.common import tensor2im
from utils.alignment import align_face
from PIL import Image
from time import time

from paths import PARAMS

def start():
    net, opts = setup_model(params.ckpt , params.device) #TODO: replace ckpt with checkpoint_path
    generator = net.decoder
    generator.eval()

    st = time()

    w1 = get_latents(net, params.imag1)
    w2 = get_latents(net, params.imag2)

    print(f"---------------------------------------INFERENCE TIME IS {time()-st} SECONDS---------------------------------------")
    
    interpolate2vectors(w1, w2, generator, params.n_steps, params.save_dir)

@torch.no_grad()
def interpolate2vectors(w1, w2, g, n, save_dir):
    interpolated_images = []
    for i in n:
        alpha = i / (n - 1)
        w = (1 - alpha) * w1 + alpha * w2
        imgs, _ = g([w.unsqueeze(0)], input_is_latent=True, randomize_noise=False, return_latents=True)
        img = tensor2im(imgs[0])
        interpolated_images.append(img)
        save_image(img, save_dir, idx=i)


def get_latents(net, image):
    codes = None
    if not image.aligned:
        aligned_image = run_alignment(image_path=image.image_path)
        codes = net.encoder(aligned_image)
    else:
        codes = net.encoder(image)
    if net.opts.start_from_latent_avg:
        if codes.ndim == 2:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        else:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)
    return codes


def save_image(img, save_dir, idx):
    im_save_path = os.path.join(save_dir, f"{idx:05d}.jpg")
    Image.fromarray(np.array(img)).save(im_save_path)


def run_alignment(image_path):
    predictor = dlib.shape_predictor(paths_config.model_paths['shape_predictor'])
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image


if __name__ == "__main__":
    params = PARAMS(image1_dir='', img1_aligned=True, image2_dir='', img2_aligned=False, checkpoint_path='', n_steps=10)
    start(params)
