import argparse
import os
import shutil
import sys
import time

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision import transforms
import torchvision.utils as tutils

#import utils
#from transformer_net import TransformerNet
#from vgg import Vgg16

import torch

from collections import namedtuple

import torch
from torchvision import models


import torch
from PIL import Image

from utils import  *
from pretrained_models import Vgg16
from dataset_processing.pacs import *

from dataset_processing.vis_style_embeddings import tsne_features_vis

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2) # swapped ch and w*h, transpose share storage with original
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def KMeans(x, K=10, Niter=10, verbose=True, use_cuda=True):
    """Implements Lloyd's algorithm for the Euclidean metric."""

    start = time.time()
    N, D = x.shape  # Number of samples, dimension of the ambient space

    c = x[:K, :].clone()  # Simplistic initialization for the centroids


    x_i = x.view(N, 1, D).clone()  # (N, 1, D) samples
    c_j = c.view(1, K, D).clone()  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):

        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        c_rep = D_ij.argmin(dim=0).long().view(-1) # nearest img per cluster
        #c_rare = D_ij.argmax(dim=0).long().view(-1)  # farthest img per cluster
        rare_candidates = torch.topk(D_ij, K, dim=0).indices #the worst case with equal worst samples per class
        c_rare = torch.unique(torch.cat(tuple(rare_candidates), dim=0), dim=0)[:K]
        print(c_rare)

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x) # clusters updated to K images in x

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        c /= Ncl  # in-place division to compute the average

    if verbose:  # Fancy display -----------------------------------------------
        if use_cuda:
            torch.cuda.synchronize()
        end = time.time()
        print(
            f"K-means for the Euclidean metric with {N:,} points in dimension {D:,}, K = {K:,}:"
        )
        print(
            "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                Niter, end - start, Niter, (end - start) / Niter
            )
        )

    return cl, c_rep, c_rare, c  # (N,) class labels and (K, D) cluster centroids


def train(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print('Getting the style vectors embedding')

    #THE PRETRAINED MODEL
    vgg = Vgg16([1, 8, 13, 20],requires_grad=False).to(device)

    DATASET = 'kaokore'  # pacs
    dataset_root = {'kaokore': 'data/kaokore_imagenet_style/status/train',
                    'pacs': 'data/pacs_data'}[DATASET]

    mini_images_rep = []
    mini_images_rare = []
    tgts_rep = []
    tgts_rare = []
    merged_codes_rep = []
    merged_codes_rare = []

    mini_images = []
    tgts = []

    if DATASET == 'pacs':
        for label_domain in sorted(os.listdir(dataset_root)):
            if os.path.isfile(dataset_root+'/'+label_domain):
                continue
            print(' Processing domain', label_domain)
            _, train_ds = get_domain_dl(label_domain)
            for label_idx, label in enumerate(sorted(os.listdir(f'{dataset_root}/{label_domain}')), start=1):
                print('K-means clustering for class',label,label_idx)
                indices = list(filter(lambda idx: train_ds.fileclasses[idx] == f'{label_idx}', range(len(train_ds.fileclasses))))
                train_classds = Subset(train_ds, indices)
                train_loader = DataLoader(train_classds, batch_size=args.batch_size, shuffle=False)  # to provide a batch loader

                count = 0

                merged_codes = []
                for batch_id, (x, _) in enumerate(train_loader):
                    n_batch = len(x)

                    if n_batch < args.batch_size:
                        break  # skip to next epoch when no enough images left in the last batch of current epoch

                    count += n_batch
                    x = normalize_batch(x)

                    _, features_style = vgg(x.to(device))

                    gram_style = gram_matrix(features_style)#[b,512,512]
                    merged_codes.append(gram_style.mean(dim=1))

                merged_codes = torch.concat(merged_codes, dim=0) # training dataset size, concatenated mean features (N_train,D)
                print('KMeans clustering now')
                cl, c_rep, c_rare, c = KMeans(merged_codes, K = 10)#10 clusters test first

                os.makedirs(f'{label_domain}/{model_arch}_rare_classes/'+label, exist_ok= True)
                os.makedirs(f'{label_domain}/{model_arch}_centroid_classes/' + label, exist_ok=True)
                # for i,t in train_classds:
                #     mini_images.append(transforms.Resize((48, 48))(i).detach().cpu())
                #     tgts.append(t)
                #     merged_codes_combined = [mc.detach().cpu() for mc in merged_codes]
                for i in c_rare:
                    # mini_images_rare.append(transforms.Compose([transforms.Resize((128, 128)),
                    #                                             transforms.ToPILImage(),
                    #                                             transforms.Pad(16, (0.,255,0.)),
                    #                                             transforms.ToTensor()])(train_classds[i][0]).detach().cpu())
                    mini_images_rare.append(transforms.Resize((128, 128)),
                                            (train_classds[i][0]).detach().cpu())
                    tgts_rare.append(train_classds[i][1])
                    merged_codes_rare.append(merged_codes[i].detach().cpu())
                    # tutils.save_image(train_classds[i][0],
                    #                   f'{model_arch}_rare_classes/' + label + '/gen_' + str(i) + '.jpg')
                    shutil.copyfile(train_ds.filenames[indices[i]],f'{label_domain}/{model_arch}_rare_classes/'+label+'/'+str(i)+'.jpg')
                for i in c_rep:
                    # mini_images_rep.append(transforms.Compose([transforms.Resize((128, 128)),
                    #                                            transforms.ToPILImage(),
                    #                                             transforms.Pad(16, (0.,255,0.)),
                    #                                            transforms.ToTensor()])(train_classds[i][0]).detach().cpu())
                    mini_images_rep.append(transforms.Resize((128, 128)),
                                           (train_classds[i][0]).detach().cpu())
                    tgts_rep.append(train_classds[i][1])
                    merged_codes_rep.append(merged_codes[i].detach().cpu())

                    #tutils.save_image(train_classds[i][0], f'{model_arch}_centroid_classes/'+label+'/gen_'+str(i)+'.jpg')
                    shutil.copyfile(train_ds.filenames[indices[i]], f'{label_domain}/{model_arch}_centroid_classes/' + label + '/' + str(i) + '.jpg')
                    shutil.copyfile(train_ds.filenames[indices[i]], f'{label_domain}/{model_arch}_centroid_classes/' + label + '/' + str(i) + '.jpg')
                #print("\nDone Embedding")
        # tsne_features_vis(torch.stack(merged_codes_rep+merged_codes_combined), mini_images_rep+mini_images, 'rep_dogs_domains_pacs',
        #                   tgts_rare+tgts, 2048)
        # tsne_features_vis(torch.stack(merged_codes_rare+merged_codes_combined), mini_images_rare+mini_images, 'rare_dogs_domains_pacs',
        #                   tgts_rare+tgts, 2048)
        tsne_features_vis(torch.stack(merged_codes_rep), mini_images_rep,
                          f'rep_pacs_domains',
                          tgts_rare, 2048)
        tsne_features_vis(torch.stack(merged_codes_rare), mini_images_rare ,
                          f'rare_pacs_domains',
                          tgts_rare, 2048)
    else:
        for label_idx, label in enumerate(os.listdir(f'{dataset_root}')):
            mini_images_rep = []
            mini_images_rare = []
            tgts_rep = []
            tgts_rare = []
            merged_codes_rep = []
            merged_codes_rare = []

            mini_images = []
            tgts = []
            if os.path.isfile(dataset_root + '/' + label):
                continue
            print(' K-means clustering for class', label, label_idx)
            count = 0
            train_ds = ImageFolder(dataset_root, transform=transforms.ToTensor())
            indices = list(
                filter(lambda idx: train_ds[idx][1] == label_idx, range(len(train_ds))))
            train_classds = Subset(train_ds, indices)
            train_loader = DataLoader(train_classds, batch_size=args.batch_size,
                                      shuffle=False)  # to provide a batch loader

            count = 0

            merged_codes = []
            for batch_id, (x, _) in enumerate(train_loader):
                n_batch = len(x)

                if n_batch < args.batch_size:
                    break  # skip to next epoch when no enough images left in the last batch of current epoch

                count += n_batch
                x = normalize_batch(x)

                _, features_style = vgg(x.to(device))
                #print(features_style.shape)

                gram_style = gram_matrix(features_style)  # [b,512,512]
                merged_codes.append(gram_style.mean(dim=1))

            merged_codes = torch.concat(merged_codes,
                                        dim=0)  # training dataset size, concatenated mean features (N_train,D)
            print('KMeans clustering now')
            cl, c_rep, c_rare, c = KMeans(merged_codes, K=10)  # 10 clusters test first

            os.makedirs(f'kaokore-kmeans/rare_classes/' + label, exist_ok=True)
            os.makedirs(f'kaokore-kmeans/centroid_classes/' + label, exist_ok=True)
            for i,t in train_classds:
                mini_images.append(transforms.Resize((48, 48))(i).detach().cpu())
                tgts.append(t)
                merged_codes_combined = [mc.detach().cpu() for mc in merged_codes]
            for i in c_rare:
                mini_images_rare.append(transforms.Compose([transforms.Resize((128, 128)),
                                                            transforms.ToPILImage(),
                                                            transforms.Pad(16, (0.,255,0.)),
                                                            transforms.ToTensor()])(train_classds[i][0]).detach().cpu())
                # mini_images_rare.append(transforms.Resize((128, 128))
                #                         (train_classds[i][0]).detach().cpu())
                tgts_rare.append(train_classds[i][1])
                merged_codes_rare.append(merged_codes[i].detach().cpu())
                shutil.copyfile(train_ds.samples[indices[i]][0],
                                f'kaokore-kmeans/rare_classes/' + label + '/' + str(i) + '.jpg')
            for i in c_rep:
                mini_images_rep.append(transforms.Compose([transforms.Resize((128, 128)),
                                                           transforms.ToPILImage(),
                                                            transforms.Pad(16, (0.,255,0.)),
                                                           transforms.ToTensor()])(train_classds[i][0]).detach().cpu())
                # mini_images_rep.append(transforms.Resize((128, 128))
                #                        (train_classds[i][0]).detach().cpu())
                tgts_rep.append(train_classds[i][1])
                merged_codes_rep.append(merged_codes[i].detach().cpu())
                shutil.copyfile(train_ds.samples[indices[i]][0],
                                f'kaokore-kmeans/centroid_classes/' + label + '/' + str(i) + '.jpg')
            tsne_features_vis(torch.stack(merged_codes_rep+merged_codes_combined), mini_images_rep+mini_images,
                          f'rep_domains_{label}_kaokore',
                          tgts_rare+tgts, 2048)
            tsne_features_vis(torch.stack(merged_codes_rare+merged_codes_combined), mini_images_rare+mini_images,
                          f'rare_domains_{label}_kaokore',
                          tgts_rare+tgts, 2048)
        # tsne_features_vis(torch.stack(merged_codes_rep), mini_images_rep,
        #                   f'rep_kaokore_domains',
        #                   tgts_rare, 2048)
        # tsne_features_vis(torch.stack(merged_codes_rare), mini_images_rare,
        #                   f'rare_kaokore_domains',
        #                   tgts_rare, 2048)

def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for K-means clustering")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("cluster", help="parser for training arguments")
    train_arg_parser.add_argument("--epochs", type=int, default=1,
                                  help="number of training epochs, default is 2")
    train_arg_parser.add_argument("--batch-size", type=int, default=4,
                                  help="batch size for training, default is 4")
    train_arg_parser.add_argument("--image-size", type=int, default=256,
                                  help="size of training images, default is 256 X 256")
    train_arg_parser.add_argument("--style-size", type=int, default=512,
                                  help="size of style-image, default is the original size of style image")
    train_arg_parser.add_argument("--cuda", type=int, required=True,
                                  help="set it to 1 for running on GPU, 0 for CPU")
    train_arg_parser.add_argument("--seed", type=int, default=42,
                                  help="random seed for training")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 1e-3")

    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)  # 1 for all other types of error besides syntax
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if args.subcommand == "cluster":
        train(args)

    print('Done Embedding')

if __name__ == "__main__":
    main()