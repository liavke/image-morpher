import torch

class PARAMS:
    def __init__(self, image1_dir, img1_aligned, image2_dir, img2_aligned, checkpoint_path, save_dir='./out', n_steps):
        self.image1=Img(image_path=image1_dir, aligned=img1_aligned)
        self.image2=Img(image_path=image2_dir, aligned=img2_aligned)
        self.ckpt=checkpoint_path
        self.device = 'cuda' #if torch.cuda.is_available() else 'cpu'
        self.save_dir = save_dir
        self.n_steps = n_steps

class Img:
    def __init__(self, image_path, aligned):
        self.image_path = image_path
        self.aligned= aligned
