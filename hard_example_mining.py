import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch import optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import shutil

from dataset_processing import pacs
from pretrained_models import Vgg

def make_scatterplot_images(paths, y, label, zoom=1, num_samples=10):
    def getImage(path, zoom=1):
        return OffsetImage(plt.imread(path), zoom=zoom)
    fig, ax = plt.subplots()
    x = list(range(num_samples))
    ax.scatter(x,y[:num_samples])

    for x0, y0, path in zip(x, y[:num_samples], paths[:num_samples]):
        ab = AnnotationBbox(getImage(path, zoom), (x0,y0), frameon=False)
        ax.add_artist(ab)
    plt.ylabel('Losses')
    plt.xlabel('Ranking')
    plt.title(f'{label} subclass')
    plt.savefig(f'hard_mining_scatter_{label}.jpg')
    plt.close(fig)


class NLL_OHEM(torch.nn.NLLLoss):
    """ Online hard example mining.
    Needs input from nn.LogSotmax() """

    def __init__(self, ratio, reduction='mean'):
        super(NLL_OHEM, self).__init__(None, True)
        self.ratio = ratio
        self.reduction=reduction

    def forward(self, x, y, ratio=None):
        if ratio is not None:
            self.ratio = ratio
        num_inst = x.size(0)
        num_hns = int(self.ratio * num_inst)
        x_ = x.clone()
        inst_losses = torch.autograd.Variable(torch.zeros(num_inst)).cuda()
        for idx, label in enumerate(y.data):
            inst_losses[idx] = -x_.data[idx, label]
            # loss_incs = -x_.sum(1)
        _, idxs = inst_losses.topk(num_hns, largest= False)#change this for rare
        x_hn = x.index_select(0, idxs)
        y_hn = y.index_select(0, idxs)
        return torch.nn.functional.nll_loss(x_hn, y_hn, reduction=self.reduction), idxs

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


DATASET = 'pacs'
tgt_path = 'rare'
model_subtype = 'vgg19st'
batch_size = 16
num_workers = 4
nb_epochs = 10
k=10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


#Dataset - Load data
transform_train = transforms.Compose([
    transforms.ToTensor(),
])
if DATASET == 'kaokore':
    dataset_path = '../kaokore_imagenet_style/status/train'
    train_set = ImageFolder(dataset_path, transform=transform_train)
    #print(train_set.imgs)
    test_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers)
    num_classes = len(train_set.classes)
    class_names = list(train_set.classes)
    print('Loaded data')
    print(class_names, num_classes)

    pm = Vgg([1, 8, 13, 20])  # 30 layers
    pretrained = []
    for model_slice in [pm.slice1, pm.slice2, pm.slice3, pm.slice4, pm.slice5]:
        pretrained += model_slice
    classify = nn.Linear(in_features=512, out_features=num_classes, bias=True)
    model = nn.Sequential(*pretrained, Flatten(), classify, nn.LogSoftmax(dim=1)).to(device)

    name = 'kaokore-style-clustering'
    print('Starting training to get consistent results')
    loss_vals = []
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epochs in range(nb_epochs):
        count = 0
        for batch_id, (x, y) in enumerate(train_loader):
            n_batch = len(x)

            if n_batch < batch_size:
                break  # skip to next epoch when no enough images left in the last batch of current epoch

            count += n_batch
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()

            # print(model)
            outputs = model(x)

            loss = F.nll_loss(outputs, y)
            loss.backward()

            optimizer.step()

            loss_vals.append(loss)
        print(f'epoch: {epochs} loss {loss}')

    print('Finished training\n Inference for supervised clustering')

    # Inference
    count = 0
    test_losses = {cls: [] for cls in class_names}
    test_paths = {cls: [] for cls in class_names}
    print(test_losses)

    loss_fn = NLL_OHEM(0.25, reduction='none')
    img_paths = train_set.imgs
    indices_to_names = {index: name for index, name in zip(list(range(num_classes)), class_names)}
    print(indices_to_names)
    for batch_id, (x, y) in enumerate(test_loader):
        n_batch = len(x)

        if n_batch < batch_size:
            break  # skip to next epoch when no enough images left in the last batch of current epoch

        count += n_batch
        x, y = x.cuda(), y.cuda()

        outputs = model(x)

        loss, idxs = loss_fn(outputs, y)
        # print(outputs.shape, y.shape, loss.shape, idxs)
        path_indices = idxs + (count - n_batch)
        # print(path_indices, loss)
        for i, p_i in enumerate(path_indices):
            name = indices_to_names[img_paths[p_i][1]]
            test_paths[name].append(img_paths[p_i][0])
            test_losses[name].append(loss[i].item())
    for cls in class_names:
        print(cls, ':', len(test_losses[cls]))

    # Loss ranking reorder
    indices_sorted = {cls: sorted(range(len(test_losses[cls])), key=lambda k: test_losses[cls][k], reverse=True) for cls
                      in class_names}  # change this for rare
    # print(indices_sorted)
    test_paths_sorted = {cls: [test_paths[cls][i] for i in indices_sorted[cls]] for cls in class_names}
    test_losses_sorted = {cls: [test_losses[cls][i] for i in indices_sorted[cls]] for cls in class_names}
    # print(test_losses_sorted)

    print('Visualization')

    for cls in class_names:
        make_scatterplot_images(test_paths_sorted[cls], test_losses_sorted[cls], cls, zoom=0.1, num_samples=k)
        for fname in test_paths_sorted[cls][:k]:
            os.makedirs(tgt_path + f'/{cls}', exist_ok=True)
            shutil.copy2(fname, tgt_path + f'/{cls}')
else:
    dataset_root = 'data/pacs_data'
    for label_domain in sorted(os.listdir(dataset_root)):
        if os.path.isfile(dataset_root + '/' + label_domain):
            continue
        print(' Processing domain', label_domain)
        train_loader, train_set= pacs.get_domain_dl(label_domain, transform_train, batch_size, 'train')
        test_loader, _ = pacs.get_domain_dl(label_domain, transform_train, batch_size, 'train')
        num_classes = len(train_set.classes)
        class_names = list(train_set.classes)
        print('Loaded data')
        print(class_names, num_classes)

        pm = Vgg([1, 8, 13, 20])  # 30 layers
        pretrained = []
        for model_slice in [pm.slice1, pm.slice2, pm.slice3, pm.slice4, pm.slice5]:
            pretrained += model_slice
        classify = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        model = nn.Sequential(*pretrained, Flatten(), classify, nn.LogSoftmax(dim=1)).to(device)

        name = 'pacs-style-clustering'
        print('Starting training to get consistent results')
        loss_vals = []
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        for epochs in range(nb_epochs):
            count = 0
            for batch_id, (x, y) in enumerate(train_loader):
                n_batch = len(x)

                if n_batch < batch_size:
                    break  # skip to next epoch when no enough images left in the last batch of current epoch

                count += n_batch
                x, y = x.cuda(), y.cuda()
                y = y-1
                optimizer.zero_grad()

                # print(model)
                outputs = model(x)

                loss = F.nll_loss(outputs, y)
                loss.backward()

                optimizer.step()

                loss_vals.append(loss)
            print(f'epoch: {epochs} loss {loss}')

        print('Finished training\n Inference for supervised clustering')

        # Inference
        count = 0
        test_losses = {cls: [] for cls in class_names}
        test_paths = {cls: [] for cls in class_names}
        print(test_losses)

        loss_fn = NLL_OHEM(0.25, reduction='none')
        img_paths = train_set.filenames
        img_classes = [int(i)-1 for i in train_set.fileclasses]
        indices_to_names = {index: name for index, name in zip(list(range(num_classes)), class_names)}
        print(indices_to_names)
        for batch_id, (x, y) in enumerate(test_loader):
            n_batch = len(x)

            if n_batch < batch_size:
                break  # skip to next epoch when no enough images left in the last batch of current epoch

            count += n_batch
            x, y = x.cuda(), y.cuda()
            y = y-1

            outputs = model(x)

            loss, idxs = loss_fn(outputs, y)
            # print(outputs.shape, y.shape, loss.shape, idxs)
            path_indices = idxs + (count - n_batch)
            # print(path_indices, loss)
            for i, p_i in enumerate(path_indices):
                name = indices_to_names[img_classes[p_i]]
                test_paths[name].append(img_paths[p_i])
                test_losses[name].append(loss[i].item())
        for cls in class_names:
            print(cls, ':', len(test_losses[cls]))

        # Loss ranking reorder
        indices_sorted = {cls: sorted(range(len(test_losses[cls])), key=lambda k: test_losses[cls][k], reverse=True) for
                          cls in class_names}  # change this for rare
        # print(indices_sorted)
        test_paths_sorted = {cls: [test_paths[cls][i] for i in indices_sorted[cls]] for cls in class_names}
        test_losses_sorted = {cls: [test_losses[cls][i] for i in indices_sorted[cls]] for cls in class_names}
        # print(test_losses_sorted)

        print('Visualization')

        for cls in class_names:
            make_scatterplot_images(test_paths_sorted[cls], test_losses_sorted[cls], cls, zoom=0.1, num_samples=k)
            for fname in test_paths_sorted[cls][:k]:
                os.makedirs(f'pacs-ohem-style/{label_domain}/'+tgt_path + f'/{cls}', exist_ok=True)
                shutil.copy2(fname, f'pacs-ohem-style/{label_domain}/'+tgt_path + f'/{cls}')


