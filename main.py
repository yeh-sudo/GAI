import torch
import torchvision.transforms as T
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer1D
from PIL import Image

device = torch.device("cuda")

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)
model.to(device)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000    # number of steps
)
diffusion.to(device)

DIP_model = torch.load("./DIP_models/DIP.pt").to(device)
DIP_model.eval()

image = Image.open("./imgs/barbara.jpg").convert("LA").resize((128, 128))

convert_tensor = T.ToTensor()
image = convert_tensor(image)
transform = T.ToPILImage()
image = image.to(device)

training_images = torch.unsqueeze(image, 0)
training_images = DIP_model(training_images)


loss = diffusion(training_images)
loss.backward()

torch.save(diffusion, "./diffusion_models/diffusion.pt")

# after a lot of training

diff_model = torch.load("./diffusion_models/diffusion.pt").to(device)
diff_model.eval()
noisy_image = diffusion.sample(batch_size=1)
img = transform(noisy_image[0])
img.show()
