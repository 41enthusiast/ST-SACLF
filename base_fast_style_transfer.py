import argparse
import time

import torch.backends.cudnn as cudnn
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from dataset_processing.pacs import *
from sampler import *

import torch
cudnn.benchmark = True
from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True

from pretrained_models import Vgg16, decoder
from utils import *

import cv2

# class Net(nn.Module):
class Net(nn.Module):
    def __init__(self, decoder):#, encoder, decoder):
        super(Net, self).__init__()
        # enc_layers = list(encoder.children())
        # self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        # self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        # self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        # self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()

        # fix the encoder
        # comment- This code part is really cool
        # for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
        #     for param in getattr(self, name).parameters():
        #         param.requires_grad = False

    # # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    # def encode_with_intermediate(self, input):
    #     results = [input]
    #     for i in range(4):
    #         func = getattr(self, 'enc_{:d}'.format(i + 1))
    #         results.append(func(results[-1]))
    #     return results[1:]
    #
    # # extract relu4_1 from input image
    # def encode(self, input):
    #     for i in range(4):
    #         input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
    #     return input

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def forward(self, content_feat, style_feats,alpha=1.0):#content, style, alpha=1.0):
        assert 0 <= alpha <= 1
        # style_feats = self.encode_with_intermediate(style)
        # content_feat = self.encode(content)

        content_feat = [cf for cf in content_feat]
        style_feats = [sf for sf in style_feats]


        t = adaptive_instance_normalization(content_feat[-1], style_feats[-1])
        t = alpha * t + (1. - alpha) * content_feat[-1]

        g_t = self.decoder(t)
        g_t = F.interpolate(g_t, size=(256,256))
        #print(g_t.shape)
        # g_t_feats = self.encode_with_intermediate(g_t)
        vgg = Vgg16([2, 9, 22, 30],requires_grad=False).cuda()
        g_t_feats, g_t_g = vgg(g_t)
        g_t_feats = list(g_t_feats)
        loss_c = self.calc_content_loss(g_t_feats[-1], t)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        return loss_c, loss_s
#
# class TransformerNet(torch.nn.Module): #the arch doesn't change with the style transfer layers
#     def __init__(self, decoder):
#         super(TransformerNet, self).__init__()
#         self.decoder = decoder
#
#         #self.ada_in = AdaIN()
#
#         self.relu = torch.nn.ReLU()
#
#     def forward(self, X_g, Y_g, alpha = 1.0, infer = False):
#         #encoding
#         x_feat = X_g
#         y_feat = Y_g
#
#         #t = self.ada_in(x_feat, y_feat, infer)
#         # t = adaptive_instance_normalization(x_feat, y_feat)
#         # t = alpha * t + (1. - alpha)*x_feat
#         t = adaptive_instance_normalization(x_feat[-1], y_feat[-1])
#         t = alpha * t + (1. - alpha) * x_feat[-1]
#
#
#         #decoder
#         out = self.decoder(t)
#         out = F.interpolate(out, size = (256,256))
#
#         #TEMP
#         vgg = Vgg16([2, 9, 22, 30], requires_grad=False).cuda()
#         out_feats, out_g = vgg(out)
#         out_feats = list(out_feats)
#         loss_c = calc_content_loss(out_feats[-1], t)
#         loss_s = calc_style_loss(out_feats[0], y_feat[0])
#         for i in range(1, 4):
#             loss_s += calc_style_loss(out_feats[i], y_feat[i])
#         return loss_c, loss_s
#
#         #return out, t


def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    # content_f = vgg(content)
    # style_f = vgg(style)
    content_f, content_g =vgg(content)
    style_f, style_g =vgg(style)

    #feat = adaptive_instance_normalization(content_f, style_f)
    feat = adaptive_instance_normalization(content_f.content_lyr, style_f.content_lyr)
    feat = feat * alpha + content_f.content_lyr * (1 - alpha)
    return decoder(feat)

def train(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    #np.random.seed(args.seed)
    #torch.manual_seed(args.seed)

    model_arch = 'vgg16'
    dataset_root = 'data/pacs_data'
    style_dataset_root = 'pacs_styles'
    clusterstyle = f'{model_arch}_rare_classes' #f'{model_arch}_centroid_classes'

    for label_domain in sorted(os.listdir(dataset_root)):
        if os.path.isfile(dataset_root+'/'+label_domain): #.DS_FILE case
            continue
        print(' Processing domain', label_domain)
        transform = transforms.Compose([
            #transforms.Resize(args.image_size),  # the shorter side is resize to match image_size
            #transforms.CenterCrop(args.image_size),
            transforms.Resize(size=(512, 512)),
            transforms.RandomCrop(256),
            transforms.ToTensor(),  # to tensor [0,1]
            # transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            # transforms.Lambda(lambda x: x.mul(255))  # convert back to [0, 255]
        ])

        style_transform = transforms.Compose([
            # transforms.Resize(args.style_size),
            # transforms.CenterCrop(args.style_size),
            transforms.Resize(size=(512, 512)),
            transforms.RandomCrop(256),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # transforms.Lambda(lambda x: x.mul(255))
        ])
        _, train_ds = get_domain_dl(label_domain, transform)

        for label_idx, label in enumerate(sorted(os.listdir(f'{dataset_root}/{label_domain}')), start=1):
            print('Fast style transfer for for class', label, label_idx)

            #content dataset
            indices = list(filter(lambda idx: train_ds.fileclasses[idx] == f'{label_idx}', range(len(train_ds.fileclasses))))
            train_classds = Subset(train_ds, indices)
            #train_loader = DataLoader(train_classds, batch_size=args.batch_size)  # to provide a batch loader
            train_loader = iter(DataLoader(
                train_classds, batch_size=args.batch_size,
                sampler=InfiniteSamplerWrapper(train_classds),
            ))

            #style dataset
            style_ds = ImageFolder(f'{style_dataset_root}/{label_domain}/{clusterstyle}/',
                                   transform=style_transform)
            style_indices = list(
                                    filter(lambda idx: style_ds[idx][1] == label_idx-1,
                                           range(len(style_ds)))
                            )
            style_dataset = Subset(style_ds, style_indices)
            style_loader = iter(DataLoader(
                style_dataset, batch_size=args.batch_size,
                sampler=InfiniteSamplerWrapper(style_dataset),
            ))

            #transformer = TransformerNet(decoder).to(device)
            transformer = Net(decoder).to(device)
            optimizer = Adam(transformer.decoder.parameters(), lr=args.lr)
            #mse_loss = torch.nn.MSELoss()

            vgg = Vgg16([2, 9, 22, 30],requires_grad=False).to(device)



            #Training starts here
            decoder.train()
            transformer.train()
            vgg.train()

            for e in tqdm(range(args.epochs)):

                agg_content_loss = 0.
                agg_style_loss = 0.
                count = 0
                #print(len(next(iter(style_loader))[0]))
                #for batch_id, ((x, _), (style, _)) in enumerate(zip(train_loader, style_loader)):

                x, _ = next(train_loader)
                style, _ = next(style_loader)

                adjust_learning_rate(optimizer, iteration_count=e, lr=args.lr)
                n_batch = len(x)

                if n_batch < args.batch_size:
                    break  # skip to next epoch when no enough images left in the last batch of current epoch

                #count += n_batch


                x_features, x_g = vgg(x.to(device))
                style_features, style_g = vgg(style.to(device))

                # stylized_img, t = transformer(x_features.content_lyr, style_features.content_lyr)#x_g, style_g)
                #
                # stylized_features, _ = vgg(stylized_img)
                #
                # content_loss = calc_content_loss(stylized_features.content_lyr, t)
                # style_loss = 0.
                # for stylized_f, style_f in zip(stylized_features, style_features):
                #     style_loss+= calc_style_loss(stylized_f, style_f)
                # style_loss *= 1e-3#args.style_weight
                # #print(style_loss)

                content_loss, style_loss = transformer(x_features, style_features)
                total_loss = content_loss * args.content_weight + style_loss * args.style_weight

                optimizer.zero_grad()  # initialize with zero gradients
                total_loss.backward()
                optimizer.step()

                agg_content_loss += content_loss.item()
                agg_style_loss += style_loss.item()

                if (e + 1) % args.log_interval == 0:
                    mesg = "{}\tEpoch {}:\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                        time.ctime(), len(train_classds),
                                      agg_content_loss / (e + 1),
                                      agg_style_loss / (e + 1),
                                      (agg_content_loss + agg_style_loss) / (e + 1)
                    )
                    print(mesg)
                state_dict = transformer.decoder.state_dict()
                if args.checkpoint_model_dir is not None and (e + 1) % args.checkpoint_interval == 0:
                    transformer.eval().cpu()
                    ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(e + 1) + ".pth"
                    ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
                    torch.save(transformer.state_dict(), ckpt_model_path)
                    transformer.to(device).train()
            else:
                #process the entire training dataset and generate the style counterparts
                print('Making the style transfer counterparts now...')
                transformer.eval()
                decoder.eval()
                vgg.eval()

                def test_transform(size, crop):
                    transform_list = []
                    if size != 0:
                        transform_list.append(transforms.Resize(size))
                    if crop:
                        transform_list.append(transforms.CenterCrop(size))
                    transform_list.append(transforms.ToTensor())
                    transform = transforms.Compose(transform_list)
                    return transform

                decoder.load_state_dict(state_dict)

                os.makedirs(f'fst-augmentations-kmeans/{clusterstyle}/{label_domain}/{label}', exist_ok= True)
                save_path = f'fst-augmentations-kmeans/{clusterstyle}/{label_domain}/{label}/'
                count = 0
                img_count = 0 #change this for class balanced version

                content_tf = test_transform(args.image_size, False)
                style_tf = test_transform(args.style_size, False)

                _, test_ds = get_domain_dl(label_domain, content_tf)
                indices = list(
                    filter(lambda idx: test_ds.fileclasses[idx] == f'{label_idx}', range(len(test_ds.fileclasses))))
                test_classds = Subset(test_ds, indices)
                style_ds = ImageFolder(f'{style_dataset_root}/{label_domain}/{clusterstyle}/',
                                       transform=style_tf)
                style_indices = list(
                    filter(lambda idx: style_ds[idx][1] == label_idx - 1,
                           range(len(style_ds)))
                )
                style_testds = Subset(style_ds, style_indices)

                test_loader = DataLoader(
                    test_classds, batch_size=args.batch_size,

                )
                style_loader = iter(DataLoader(
                style_testds, batch_size=args.batch_size,
                sampler=InfiniteSamplerWrapper(style_testds),
                ))

                for batch_id, ((x, _), (style, _)) in enumerate(zip(test_loader,style_loader)):
                    n_batch = len(x)

                    if n_batch < args.batch_size:
                        break  # skip to next epoch when no enough images left in the last batch of current epoch

                    count += n_batch
                    #print(x.shape, style.shape)
                    # style_features, style_g = vgg(style.to(device))
                    # x_features, x_g = vgg(x.to(device))
                    #styled, _ = transformer(x_features.content_lyr, style_features.content_lyr)#(x_g, style_g)
                    styled = style_transfer(vgg, decoder,
                                         x.to(device),
                                         style.to(device)
                                         )

                    for img in styled:

                        #print(style.shape)
                        #for img in styled:
                        #print(img.max(), img.min())
                        #img = (img - img.min())/(img.max()-img.min())
                        save_image(img.detach().cpu(), save_path + str(img_count) + '.jpg')
                        img_count += 1

            # save model
            transformer.eval().cpu()
            os.makedirs(f'{args.save_model_dir}/{label_domain}/kmeans_{clusterstyle}/{label}/', exist_ok=True)
            save_model_filename = f'{label_domain}/kmeans_{clusterstyle}/{label}/' + "cd_epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':',
                                                                                                                  '') + "_" + str(
                int(
                    args.content_weight)) + "_" + str(int(args.style_weight)) + ".model"
            save_model_path = os.path.join(args.save_model_dir, save_model_filename)
            torch.save(transformer.state_dict(), save_model_path)

            print("\nDone, trained model saved at", save_model_path)
            break
        break

def stylize(args, model):
    device = torch.device("cuda" if args.cuda else "cpu")

    content_image = load_image(args.content_image, scale=args.content_scale)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
        style_model = Net(decoder)#TransformerNet(style_num=args.style_num)
        state_dict = torch.load(args.model)
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        output = style_model(content_image, style_id=[args.style_id]).cpu()

    save_image(output[0], 'output/' + args.output_image + '_style' + str(args.style_id) + '.jpg')

def main():

    main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--epochs", type=int, default=2,
                                  help="number of training epochs, default is 2")
    train_arg_parser.add_argument("--batch-size", type=int, default=4,
                                  help="batch size for training, default is 4")
    train_arg_parser.add_argument("--save-model-dir", type=str, required=True,
                                  help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--checkpoint-model-dir", type=str, default=None,
                                  help="path to folder where checkpoints of trained models will be saved")
    train_arg_parser.add_argument("--image-size", type=int, default=256,
                                  help="size of training images, default is 256 X 256")
    train_arg_parser.add_argument("--style-size", type=int, default=256,
                                  help="size of style-image, default is the original size of style image")
    train_arg_parser.add_argument("--cuda", type=int, required=True,
                                  help="set it to 1 for running on GPU, 0 for CPU")
    train_arg_parser.add_argument("--seed", type=int, default=42,
                                  help="random seed for training")
    train_arg_parser.add_argument("--content-weight", type=float, default=1.0,
                                  help="weight for content-loss, default is 1e5")
    train_arg_parser.add_argument("--style-weight", type=float, default=10.0,
                                  help="weight for style-loss, default is 1e10")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 1e-3")
    train_arg_parser.add_argument("--log-interval", type=int, default=100,
                                  help="number of images after which the training loss is logged, default is 250")
    train_arg_parser.add_argument("--checkpoint-interval", type=int, default=1000,
                                  help="number of batches after which a checkpoint of the trained model will be created")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--content", type=str, required=True,
                                 help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--content-scale", type=float, default=None,
                                 help="factor for scaling down the content image")
    eval_arg_parser.add_argument("--output-image", type=str, required=True,
                                 help="path for saving the output image")
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for stylizing the image. If file ends in .pth - PyTorch path is used, if in .onnx - Caffe2 path")
    eval_arg_parser.add_argument("--cuda", type=int, required=True,
                                 help="set it to 1 for running on GPU, 0 for CPU")
    eval_arg_parser.add_argument("--style-id", type=int, required=True,
                                 help="style number id corresponding to the order in training")
    eval_arg_parser.add_argument("--batch-size", type=int, default=4,
                                 help="batch size for testing, default is 4")
    eval_arg_parser.add_argument("--style-num", type=int, default=4,
                                 help="number of styles used in training, default is 4")

    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)  # 1 for all other types of error besides syntax
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if args.subcommand == "train":
        check_paths(args)
        train(args)
    else:
        stylize(args)


if __name__ == "__main__":
    main()