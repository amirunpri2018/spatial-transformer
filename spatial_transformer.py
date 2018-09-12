import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from datetime import datetime
from torchvision import datasets, transforms, utils


class SpatialTransformerNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=7)
        self.conv2 = nn.Conv2d(8, 10, kernel_size=5)
        self.fc1 = nn.Linear(10 * 3 * 3, 32)
        self.fc2 = nn.Linear(32, 3 * 2)

        self.conv3 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv4 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc3 = nn.Linear(320, 50)
        self.fc4 = nn.Linear(50, 10)

        # Initialize with identity transformation
        self.fc2.weight.data.zero_()
        self.fc2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, input, classification=True):
        # Transform input
        x = F.relu(F.max_pool2d(self.conv1(input), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = F.relu(self.fc1(x.view(-1, 10 * 3 * 3)))
        theta = self.fc2(x).view(-1, 2, 3)

        # Parametrized sampling grid
        # Alternatively, grid = F.affine_grid(theta, x.size())
        dtype = input.type()
        N, C, H, W = input.size()
        xs = torch.linspace(-1.0, 1.0, W).repeat(1, H)
        ys = torch.linspace(-1.0, 1.0, H).view(-1, 1).repeat(1, W).view(1, -1)
        xy = torch.cat([xs, ys, torch.ones(1, H * W)], dim=0).type(dtype)
        grid = torch.matmul(theta.view(-1, 3), xy).view(N, 2, H, W).permute(0, 2, 3, 1)

        # Differential image sampling
        # Alternatively, x = F.grid_sample(x, grid)
        xs = 0.5 * (W - 1) * (grid[:, :, :, 0] + 1.0)
        x0 = torch.floor(xs).long()
        x1 = x0 + 1

        ys = 0.5 * (H - 1) * (grid[:, :, :, 1] + 1.0)
        y0 = torch.floor(ys).long()
        y1 = y0 + 1

        x0 = torch.clamp(x0, 0, W - 1)
        x1 = torch.clamp(x1, 0, W - 1)
        y0 = torch.clamp(y0, 0, H - 1)
        y1 = torch.clamp(y1, 0, H - 1)

        imga = gather_pixels(input, x0, y0)
        imgb = gather_pixels(input, x0, y1)
        imgc = gather_pixels(input, x1, y0)
        imgd = gather_pixels(input, x1, y1)

        wa = ((x1.type(dtype) - xs) * (y1.type(dtype) - ys)).unsqueeze(dim=1)
        wb = ((x1.type(dtype) - xs) * (ys - y0.type(dtype))).unsqueeze(dim=1)
        wc = ((xs - x0.type(dtype)) * (y1.type(dtype) - ys)).unsqueeze(dim=1)
        wd = ((xs - x0.type(dtype)) * (ys - y0.type(dtype))).unsqueeze(dim=1)

        x = wa * imga + wb * imgb + wc * imgc + wd * imgd
        if not classification:
            return x

        # Classification
        x = F.relu(F.max_pool2d(self.conv3(x), kernel_size=2))
        x = F.relu(F.max_pool2d(F.dropout(self.conv4(x), training=self.training), kernel_size=2))
        x = F.dropout(F.relu(self.fc3(x.view(-1, 320))), training=self.training)
        return F.log_softmax(self.fc4(x), dim=1)


def main():
    args = get_args()
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # Data loaders
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data, download=True, train=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, num_workers=4)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data, download=True, train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Train model
    model = SpatialTransformerNetwork().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    train(args, model, optimizer, train_loader, test_loader, device)

    # Visualize transformer
    with torch.no_grad():
        data = next(iter(test_loader))[0].to(device)
        transformed_data = model(data, classification=False)
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(convert_image_np(utils.make_grid(data.cpu())))
        ax[0].set_title('Original Images')
        ax[0].set_axis_off()
        ax[1].imshow(convert_image_np(utils.make_grid(transformed_data.cpu())))
        ax[1].set_title('Transformed Images')
        ax[1].set_axis_off()
        fig.savefig('sample/sample.png', dpi=200)


def train(args, model, optimizer, train_loader, test_loader, device):
    model.train()
    for epoch in range(1, args.num_epochs + 1):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

        train_loss, train_acc = evaluate(model, train_loader, device)
        test_loss, test_acc = evaluate(model, test_loader, device)
        print('[{}] Epoch {}: train_loss = {:.4f}, test_loss = {:.4f}, train_acc = {:.4f}, test_acc = {:.4f}'.format(
            datetime.now(), epoch, train_loss, test_loss, train_acc, test_acc))


def evaluate(model, data_loader, device):
    model.eval()
    loss, correct = 0, 0

    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss += F.nll_loss(output, target, size_average=False).item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()

    loss = loss / len(data_loader.dataset)
    accuracy = 100.0 * correct / len(data_loader.dataset)
    return loss, accuracy


def gather_pixels(image, x, y):
    """Gather pixel values for coordinate vectors from a 4D tensor"""
    # image: N x C x H x W
    # x:     N x H x W
    N, C, H, W = image.size()
    img = image.permute(0, 2, 3, 1)
    batch_idx = torch.arange(N).long().type(x.type())
    batch_idx = batch_idx.view(N, 1, 1).expand(N, H, W)
    output = img[batch_idx, y, x].permute(0, 3, 1, 2)
    return output


def convert_image_np(tensor):
    """Convert a Tensor to numpy image."""
    inp = tensor.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


def get_args():
    parser = argparse.ArgumentParser('Spatial Transformer Networks (https://arxiv.org/pdf/1506.02025.pdf)')
    parser.add_argument('--data', required=True, metavar='PATH', help='path to dataset')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default 128)')
    parser.add_argument('--learning-rate', type=float, default=0.01, metavar='LR', help='learning rate (default 0.01)')
    parser.add_argument('--num-epochs', type=int, default=20, metavar='N', help='number of epochs to train (default 10)')
    parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default 42)')
    return parser.parse_args()


if __name__ == '__main__':
    main()
