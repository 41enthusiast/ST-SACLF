import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

import net
from function import adaptive_instance_normalization, coral

from torch.utils.data import Subset, DataLoader
from dataset_processing import pacs
from torchvision.datasets import ImageFolder
from sampler import InfiniteSamplerWrapper

def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content_f, style_f, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    # content_f = vgg(content)
    # style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f[-1], style_f[-1])
        #feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f[-1] * (1 - alpha)
    #feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


parser = argparse.ArgumentParser()
# Basic options
# parser.add_argument('--content', type=str,
#                     help='File path to the content image')
# parser.add_argument('--content_dir', type=str,
#                     help='Directory path to a batch of content images')
# parser.add_argument('--style', type=str,
#                     help='File path to the style image, or multiple style \
#                     images separated by commas if you want to do style \
#                     interpolation or spatial control')
# parser.add_argument('--style_dir', type=str,
#                     help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='models/decoder.pth')

# Additional options
parser.add_argument('--content_size', type=int, default=512,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style_size', type=int, default=512,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')

# Advanced options
parser.add_argument('--preserve_color', action='store_true',
                    help='If specified, preserve color of the content image')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')
parser.add_argument(
    '--style_interpolation_weights', type=str, default='',
    help='The weight for blending the style of multiple style images')

args = parser.parse_args()

do_interpolation = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Either --content or --contentDir should be given.
# assert (args.content or args.content_dir)
# if args.content:
#     content_paths = [Path(args.content)]
# else:
#     content_dir = Path(args.content_dir)
#     content_paths = [f for f in content_dir.glob('*')]

# # Either --style or --styleDir should be given.
# assert (args.style or args.style_dir)
# if args.style:
#     style_paths = args.style.split(',')
#     if len(style_paths) == 1:
#         style_paths = [Path(args.style)]
#     else:
#         do_interpolation = True
#         assert (args.style_interpolation_weights != ''), \
#             'Please specify interpolation weights'
#         weights = [int(i) for i in args.style_interpolation_weights.split(',')]
#         interpolation_weights = [w / sum(weights) for w in weights]
# else:
#     style_dir = Path(args.style_dir)
#     style_paths = [f for f in style_dir.glob('*')]



decoder = net.decoder
vgg = net.vgg

decoder.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder))
#
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

# vgg.load_state_dict(torch.load(args.vgg))

#vgg = nn.Sequential(*list(vgg.children())[:31])

vgg.to(device)
decoder.to(device)

content_tf = test_transform(args.content_size, args.crop)
style_tf = test_transform(args.style_size, args.crop)
DATASET = 'kaokore'  # pacs
dataset_root = {'kaokore': 'data/kaokore_imagenet_style/status/train',
                'pacs': 'data/pacs_data'}[DATASET]
batch_size, n_threads = 8, 16
for cluster_type in ['control', ]:  # 'centroid', 'rare']:
    print(cluster_type)
    if DATASET == 'pacs':
        domain_name = 'cartoon'
        _, train_dataset = pacs.get_domain_dl(domain_name, content_tf)
        for i, tgt in enumerate('dog  elephant  giraffe  guitar  horse  house  person'.split('  '), 1):
            print(tgt)
            indices = list(
                filter(lambda idx: train_dataset.fileclasses[idx] == f'{i}', range(len(train_dataset.fileclasses))))
            content_dataset = Subset(train_dataset, indices)
            style_ds = ImageFolder(f'pacs_styles/{domain_name}/vgg16_{cluster_type}_classes/',
                                   transform=style_tf)
            style_indices = list(
                filter(lambda idx: style_ds[idx][1] == i - 1,
                       range(len(style_ds)))
            )
            style_dataset = Subset(style_ds, style_indices)

            content_iter = DataLoader(
                content_dataset, batch_size=batch_size,
                num_workers=n_threads)
            style_iter = DataLoader(
                style_dataset, batch_size=batch_size,
                sampler=InfiniteSamplerWrapper(style_dataset),
                num_workers=n_threads)

            output_dir = Path(f'{domain_name}/{cluster_type}_classes/{tgt}')
            output_dir.mkdir(exist_ok=True, parents=True)

            # for content_path in content_paths:
            #     if do_interpolation:  # one content image, N style image
            #         style = torch.stack([style_tf(Image.open(str(p))) for p in style_paths])
            #         content = content_tf(Image.open(str(content_path))) \
            #             .unsqueeze(0).expand_as(style)
            #         style = style.to(device)
            #         content = content.to(device)
            #         with torch.no_grad():
            #             output = style_transfer(vgg, decoder, content, style,
            #                                     args.alpha, interpolation_weights)
            #         output = output.cpu()
            #         output_name = output_dir / '{:s}_interpolation{:s}'.format(
            #             content_path.stem, args.save_ext)
            #         save_image(output, str(output_name))
            #
            #     else:  # process one content and one style
            #         for style_path in style_paths:
            #             content = content_tf(Image.open(str(content_path)))
            #             style = style_tf(Image.open(str(style_path)))
            #             if args.preserve_color:
            #                 style = coral(style, content)
            #             style = style.to(device).unsqueeze(0)
            #             content = content.to(device).unsqueeze(0)
            #             with torch.no_grad():
            #                 output = style_transfer(vgg, decoder, content, style,
            #                                         args.alpha)
            #             output = output.cpu()
            #
            #             output_name = output_dir / '{:s}_stylized_{:s}{:s}'.format(
            #                 content_path.stem, style_path.stem, args.save_ext)
            #             save_image(output, str(output_name))

            for i, ((content, _), (style, _)) in enumerate(zip(content_iter, style_iter)):
                style = style[:content.shape[0], :, :]
                if args.preserve_color:
                    style = coral(style, content)
                style = style.to(device)
                content = content.to(device)
                with torch.no_grad():
                    content_f, _ = vgg(content)
                    style_f, _ = vgg(style)
                    # content_f = vgg(content)
                    # style_f = vgg(style)
                    output = style_transfer(vgg, decoder, content_f, style_f,
                                            args.alpha)
                output = output.cpu()
                for j, o_img in enumerate(output):
                    output_name = output_dir / f'stylized_{i * batch_size + j}{args.save_ext}'
                    save_image(o_img, str(output_name))
    else:
        train_dataset = ImageFolder(dataset_root, transform=transforms.ToTensor())
        for i, tgt in enumerate('commoner  incarnation  noble  warrior'.split('  ')):
            indices = list(
                filter(lambda idx: train_dataset[idx][1] == i,
                       range(len(train_dataset)))
            )
            content_dataset = Subset(train_dataset, indices)
            print(len(content_dataset))
            # style_ds = ImageFolder(f'kaokore-ohem/{cluster_type}/',
            #                        transform=style_tf)
            # style_indices = list(
            #     filter(lambda idx: style_ds[idx][1] == i,
            #            range(len(style_ds)))
            # )
            # style_dataset = Subset(style_ds, style_indices)
            content_iter = DataLoader(
                content_dataset, batch_size=batch_size,
                num_workers=n_threads)
            style_iter = DataLoader(
                content_dataset, batch_size=batch_size,
                shuffle=True, num_workers=n_threads)
            # style_iter = DataLoader(
            #     style_dataset, batch_size=batch_size,
            #     sampler=InfiniteSamplerWrapper(style_dataset),
            #     num_workers=n_threads)

            output_dir = Path(f'{cluster_type}_classes/{tgt}')
            output_dir.mkdir(exist_ok=True, parents=True)

            # for content_path in content_paths:
            #     if do_interpolation:  # one content image, N style image
            #         style = torch.stack([style_tf(Image.open(str(p))) for p in style_paths])
            #         content = content_tf(Image.open(str(content_path))) \
            #             .unsqueeze(0).expand_as(style)
            #         style = style.to(device)
            #         content = content.to(device)
            #         with torch.no_grad():
            #             output = style_transfer(vgg, decoder, content, style,
            #                                     args.alpha, interpolation_weights)
            #         output = output.cpu()
            #         output_name = output_dir / '{:s}_interpolation{:s}'.format(
            #             content_path.stem, args.save_ext)
            #         save_image(output, str(output_name))
            #
            #     else:  # process one content and one style
            #         for style_path in style_paths:
            #             content = content_tf(Image.open(str(content_path)))
            #             style = style_tf(Image.open(str(style_path)))
            #             if args.preserve_color:
            #                 style = coral(style, content)
            #             style = style.to(device).unsqueeze(0)
            #             content = content.to(device).unsqueeze(0)
            #             with torch.no_grad():
            #                 output = style_transfer(vgg, decoder, content, style,
            #                                         args.alpha)
            #             output = output.cpu()
            #
            #             output_name = output_dir / '{:s}_stylized_{:s}{:s}'.format(
            #                 content_path.stem, style_path.stem, args.save_ext)
            #             save_image(output, str(output_name))

            #for i, ((content, _), (style, _)) in enumerate(zip(content_iter, style_iter)):
            for i in range(len(content_dataset)):
                content = content_dataset[torch.randint(len(content_dataset), (1,))][0]
                style = content_dataset[torch.randint(len(content_dataset), (1,))][0]
                # style = style[:content.shape[0], :, :]
                if args.preserve_color:
                    for j in range(len(style)):
                        style[j] = coral(style[j], content[j])
                style = style.to(device).unsqueeze(0)
                content = content.to(device).unsqueeze(0)
                with torch.no_grad():
                    content_f, _ = vgg(content)
                    style_f, _ = vgg(style)
                    # content_f = vgg(content)
                    # style_f = vgg(style)
                    output = style_transfer(vgg, decoder, content_f, style_f,
                                            args.alpha)
                output = output.cpu()
                for j, o_img in enumerate(output):
                    output_name = output_dir / f'stylized_{i * batch_size + j}{args.save_ext}'
                    save_image(o_img, str(output_name))

