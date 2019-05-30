import argparse
import numpy as np
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.distributions import Normal
from model import VAE
from dataset import get_dataset
from utils import save_latent_variable, concat_image, save_image


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.bce = nn.BCELoss(reduction="sum")

    def forward(self, predict, target, mean, logvar):
        bce = self.bce(predict, target)
        kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

        return bce + kl


def train(model, opt):
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = Loss()
    train_dataset = get_dataset(opt.dataset, opt.data_root, opt.image_size, train=True)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)

    model.train()
    for i, (image, label) in enumerate(train_loader):
        image = image.to(opt.device)

        optimizer.zero_grad()
        output, mean, logvar = model(image)
        loss = criterion(output, image, mean, logvar)
        loss.backward()
        optimizer.step()

    return loss


def eval_encoder(file_name, model, opt):
    eval_dataset = get_dataset(opt.dataset, opt.data_root, opt.image_size, train=False)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    zs, labels = [], []
    model.eval()
    with torch.no_grad():
        for i, (image, label) in enumerate(eval_loader):
            image = image.to(opt.device)
            mean, logvar = model.encoder(image)
            z = Normal(mean, logvar.exp())
            z = z.sample().cpu().tolist()
            label = label.tolist()
            zs.extend(z)
            labels.extend(label)

    zs, labels = np.array(zs), np.array(labels)
    save_latent_variable(file_name, zs, labels)


def eval_decoder(file_name, model, opt):
    model.eval()
    with torch.no_grad():
        z = torch.randn(64, opt.latent_dim, device=opt.device)
        output = model.decoder(z)
        output = output.permute(0, 2, 3, 1).cpu().numpy()

    output = concat_image(output)
    save_image(file_name, output)


def make_directory():
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./outputs", exist_ok=True)
    os.makedirs("./outputs/decode", exist_ok=True)
    os.makedirs("./outputs/encode", exist_ok=True)
    os.makedirs("./outputs/test", exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="VAE")
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default="kmnist")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--image_channels", type=int, default=1)
    parser.add_argument("--latent_dim", type=int, default=2)
    opt = parser.parse_args()

    make_directory()

    vae = VAE(opt.image_size, opt.image_channels, opt.latent_dim, opt.device).to(opt.device)
    for epoch in range(opt.num_epoch):
        loss = train(vae, opt)
        print("epoch: {} train loss: {}".format(epoch, loss))
        eval_encoder("./outputs/encode/{}epoch.jpg".format(epoch), vae, opt)
        eval_decoder("./outputs/decode/{}epoch.jpg".format(epoch), vae, opt)

    torch.save(vae.state_dict(), "./models/vae.pth")


if __name__ == "__main__":
    main()

