import argparse
import torch
from model import VAE
from utils import save_images


def main():
    parser = argparse.ArgumentParser(description="VAE")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_images", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--image_channels", type=int, default=1)
    parser.add_argument("--latent_dim", type=int, default=2)
    opt = parser.parse_args()

    vae = VAE(opt.image_size, opt.image_channels, opt.latent_dim, opt.device).to(opt.device)
    vae.load_state_dict(torch.load("weights/vae.pth"))
    vae.eval()

    with torch.no_grad():
        z = torch.randn(opt.num_images, 2).to(opt.device)
        output = vae.decoder(z)
        output = output.permute(0, 2, 3, 1).cpu().numpy()
        save_images("outputs/test", output)


if __name__ == "__main__":
    main()
