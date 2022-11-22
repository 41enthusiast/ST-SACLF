import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

import net
from sampler import InfiniteSamplerWrapper


from torch.utils.data import Subset
from dataset_processing import pacs
from torchvision.datasets import ImageFolder

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options
# parser.add_argument('--content_dir', type=str, required=True,
#                     help='Directory path to a batch of content images')
# parser.add_argument('--style_dir', type=str, required=True,
#                     help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10000)
args = parser.parse_args()

device = torch.device('cuda')
save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)
log_dir = Path(args.log_dir)
log_dir.mkdir(exist_ok=True, parents=True)
writer = SummaryWriter(log_dir=str(log_dir))

decoder = net.decoder
vgg = net.vgg

# vgg_keys = ['0.weight', '0.bias', '3.weight', '3.bias', '7.weight', '7.bias', '10.weight', '10.bias', '14.weight', '14.bias', '17.weight', '17.bias', '20.weight', '20.bias', '23.weight', '23.bias', '27.weight', '27.bias', '30.weight', '30.bias', '33.weight', '33.bias', '36.weight', '36.bias', '40.weight', '40.bias', '43.weight', '43.bias', '46.weight', '46.bias', '49.weight', '49.bias']
# pretrained_keys = ['0.weight', '0.bias', '2.weight', '2.bias', '5.weight', '5.bias', '7.weight', '7.bias', '10.weight', '10.bias', '12.weight', '12.bias', '14.weight', '14.bias', '16.weight', '16.bias', '19.weight', '19.bias', '21.weight', '21.bias', '23.weight', '23.bias', '25.weight', '25.bias', '28.weight', '28.bias', '30.weight', '30.bias', '32.weight', '32.bias', '34.weight', '34.bias']
# old_pretrained_keys = ['0.weight', '0.bias', '2.weight', '2.bias', '5.weight', '5.bias', '9.weight', '9.bias', '12.weight', '12.bias', '16.weight', '16.bias', '19.weight', '19.bias', '22.weight', '22.bias', '25.weight', '25.bias', '29.weight', '29.bias', '32.weight', '32.bias', '35.weight', '35.bias', '38.weight', '38.bias', '42.weight', '42.bias', '45.weight', '45.bias', '48.weight', '48.bias', '51.weight', '51.bias']
# temp_state_dict = torch.load(args.vgg)
# old_state_dict = torch.load('models/vgg_normalised.pth')
# pretrained_layers = [old_state_dict[i] for i in old_pretrained_keys[:2]]+ [temp_state_dict[i] for i in pretrained_keys]
# vgg_state_dict = vgg.state_dict()
# assert len(vgg_keys) == len(pretrained_keys)
# for i in range(len(old_pretrained_keys)):
#     vgg_state_dict[old_pretrained_keys[i]] = pretrained_layers[i]
# vgg.load_state_dict(vgg_state_dict)

#vgg.load_state_dict(torch.load(args.vgg))

#vgg = nn.Sequential(*list(vgg.children())[:31])
#network = net.Net(vgg, decoder)
network = net.Net(decoder)
network.train()
network.to(device)

content_tf = train_transform()
style_tf = train_transform()

DATASET = 'kaokore'  # pacs
dataset_root = {'kaokore': 'data/kaokore_imagenet_style/status/train',
                'pacs': 'data/pacs_data'}[DATASET]

# content_dataset = FlatFolderDataset(args.content_dir, content_tf)
# style_dataset = FlatFolderDataset(args.style_dir, style_tf)

if DATASET == 'pacs':
    domain_name = 'photo'
    _, train_dataset = pacs.get_domain_dl(domain_name, content_tf)
    indices = list(filter(lambda idx: train_dataset.fileclasses[idx] == '1', range(len(train_dataset.fileclasses))))
    content_dataset = Subset(train_dataset, indices)
    style_ds = ImageFolder(f'pacs-ohem-style/{domain_name}/rare/',
                                       transform=style_tf)
    style_indices = list(
                            filter(lambda idx: style_ds[idx][1] == 0,
                                   range(len(style_ds)))
                    )
    style_dataset = Subset(style_ds, style_indices)
else:
    domain_name = 'kaokore_control'
    train_dataset = ImageFolder(dataset_root, transform=transforms.ToTensor())
    content_dataset = train_dataset
    style_dataset = ImageFolder(dataset_root, transform=transforms.ToTensor())
    # style_ds = ImageFolder('kaokore-ohem-k100/rare/',
    #                                    transform=style_tf)
    # style_indices = list(
    #                         filter(lambda idx: style_ds[idx][1] == 0,
    #                                range(len(style_ds)))
    #                 )
    # style_dataset = Subset(style_ds, style_indices)



content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))

optimizer = torch.optim.Adam(network.decoder.parameters(), lr=args.lr)

for i in tqdm(range(args.max_iter)):
    adjust_learning_rate(optimizer, iteration_count=i)
    content_images, _ = next(content_iter)
    content_images = content_images.to(device)
    style_images, _ = next(style_iter)
    style_images = style_images.to(device)

    content_f, _ = vgg(content_images)
    style_f, _ = vgg(style_images)

    #loss_c, loss_s = network(content_images, style_images)
    loss_c, loss_s = network(content_f, style_f)
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    loss = loss_c + loss_s

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    writer.add_scalar('loss_content', loss_c.item(), i + 1)
    writer.add_scalar('loss_style', loss_s.item(), i + 1)

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = net.decoder.state_dict()
        # for key in state_dict.keys():
        #     state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, save_dir /
                   'decoder_iter_{:d}_{}.pth'.format(i + 1,domain_name))
writer.close()
