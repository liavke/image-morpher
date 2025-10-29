#Generative Adversarial Networks

GANs are an unsupervised DNN (Deep Neural Network). They are unsupervised in that their training data are not labeled. GANs use two neural networks to create synthetic data: a generative, and a discriminative network (also known as the Generator and Discriminator). GANs generally work by training a generative network to produce fake data from real data; in turn, both datasets are used to train a Discriminator to differentiate between real and fake data. In more specific terms, the images in the dataset are fed to the Discriminator, while the Generator takes a random noise vector 𝑍 to produce data. Vector 𝑍 is sampled from either a Gaussian or uniform distribution. This allows for an adversarial game to take place between the Generator and the Discriminator. The Discriminator penalizes the Generator if the output is not realistic enough; this pushes the Generator to produce more realistic results until the Discriminator cannot differentiate between real or fake data. As Goodfellow et al. (2014, p. 1) put it:

“Competition in this game drives both teams to improve their methods until the counterfeits are indistinguishable from the genuine articles.”







#Stylegan

While most advances in GAN technology up to this point were focused on improving the Discriminator, few concerned themselves with the Generator. Researchers at NVIDIA aimed to crack the black box that is the Generator, which would in turn solve a major issue that existed with GAN up to this point: that being, the inability to control style direction that a model takes during training. To solve this, Karras, Laine, and Aila (2019) introduced a mapping network that takes the noise vector Z and encodes it into an intermediate vector. By doing so, it essentially disentangles features within Z from each other, and makes the output vector W less dependent on biases in the training data, while also encoding style information. W is then fed to the Synthesis Network (also known as the Generator Network) to produce an image. At each resolution layer of the Synthesis Network, AdaIN (Adaptive Instance Normalization) decodes the features of W that apply to each resolution (such as skin tone, pose, eye color, hair, etc). Additionally, noise is injected at each resolution layer to allow for stochastic variation. This allows the model to produce unique synthetic images with each iteration. These innovations not only resulted in higher resolution photorealistic synthetic images, but they also allowed the researchers to mix styles between different images, essentially allowing for the transferring of the key features of one face onto another (see Karras, Laine, and Aila, 2019).




#Encoder
##Encoder4Editing




#Interpolation

The main idea of interpolation is the transformation of one vector image to another vector image. Many, probably have seen similar interpolation videos like this one here https://www.youtube.com/shorts/AnlUiFMD5lw . The goal of this project is to create a tool that allows for a synthetic interpolation between 2 images. The first step is to invert 2 images to the latent space. To do that, we use a pretrained encoder from the selected encoders (encoder4editing...), which extracts features from the images and reduce its dimensions until it reaches the proximity of the desired latent space. The interpolation function will then iterate over a predefined n-steps, in which each n iteration is normalized between 0 and 1. This produces the alpha weight, where alpha=1=W1 and alpha=0=W2, with W1 and W2 being the respective latent space representation of image1 and image2. The weighted vectors are then multiplied and the result is given to the StyleGAN generator, to generate a synthetic image from the resulted latent vector. 


