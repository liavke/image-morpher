from PIL import Image
import PIL
from projector import project
import numpy as np
import torch
import dnnlib
from time import perf_counter
import legacy

def resize(img:Image):
    resized = img.resize((250,250), Image.Resampling.BILINEAR)
    #todo: recenter image
    return resized

def img2LV_stylegan(network_pkl: str,
    target_pil: Image,
    seed: int= 303,
    num_steps: int=500):

    """Project given image to the latent space of pretrained network pickle.

    Examples:

    \b
    python projector.py --outdir=out --target=~/mytargetimg.png \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cpu')
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore

    # Load target image.
    target_pil = target_pil.resize((250,250), Image.Resampling.BILINEAR)
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    size_ = G.img_resolution
    target_pil = target_pil.resize((G.img_resolution, G.img_resolution), Image.Resampling.BILINEAR)
    target_uint8 = np.array(target_pil, dtype=np.uint8)

    target_tensor = torch.tensor(target_uint8.transpose([2, 0, 1]), device=device, dtype=torch.float32) / 127.5 - 1.0
    #target_tensor = target_tensor.unsqueeze(0)  # Add batch dim

    # Optimize projection.
    start_time = perf_counter()
    projected_w_steps = project(
        G,
        target=target_tensor,
        num_steps=num_steps,
        device=device,
        verbose=True
    )
    print (f'Elapsed: {(perf_counter()-start_time):.1f} s')


    projected_w = projected_w_steps[-1]
    return projected_w



def img2LV_cutoms_encoder():
    pass

def test_img2LV():
    image = Image.open('test.png')
    w = img2LV_stylegan(network_pkl='ffhq.pkl',
                        target_pil=image)

if __name__ == '__main__':
    test_img2LV()
