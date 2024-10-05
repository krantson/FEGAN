import argparse
import torch
import random
import os
from pathlib import Path
from utils import *
from FE import FE


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()

    """ environment setting"""
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--n_cpu', type=int, default=2, help='number of workers')
    parser.add_argument('--patch_size', type=int, default=128, help='patch size')
    parser.add_argument('--noisy_src_path', type=str, default='data/train/train_noisy_100', help='path for noisy images folder')
    parser.add_argument('--clean_src_path', type=str, default='data/train/train_clean_100', help='path for clean images folder')
    parser.add_argument('--datarootN_val', type=str, default='data/test_mcaptcha', help='path for validation folder')


    """ train setting """
    parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')

    parser.add_argument('--lrG', type=float, default=1e-4, help='learning rate of generator')
    parser.add_argument('--lrD', type=float, default=1e-4, help='learning rate of discriminator')
    parser.add_argument('--decay_step', default="[100]", help='learning rate decay step')

    parser.add_argument('--n_res_gen', type=int, default=2, help='number of resblock in C2N generator')
    parser.add_argument('--n_conv_dis', type=int, default=3, help='number of layers in N2C discriminator')
    parser.add_argument('--ch_genn2c', type=int, default=64, help='channel size in N2C generator')
    parser.add_argument('--ch_genc2n', type=int, default=32, help='channel size in C2N generator')
    parser.add_argument('--ch_dis', type=int, default=64, help='channel size in discriminator')
    
    parser.add_argument('--adv_w', type=float, default=15.0, help='weight of clean adversarial loss')
    parser.add_argument('--texture_w', type=float, default=10.0, help='weight of texture adversarial loss')
    parser.add_argument('--vgg_w', type=float, default=2.0, help='weight of vgg19 loss')
    parser.add_argument('--tv_w', type=float, default=20.0, help='weight of total variance loss')
    parser.add_argument('--time_w', type=float, default=10.0, help='weight of cycle loss')
    parser.add_argument('--freq_w', type=float, default=5.0, help='weight of freqeuncy recon loss')
    

    """ test setting """
    parser.add_argument('--test', type=int, default=2, help='1: train, 1:val, 2:test')
    parser.add_argument('--only_test_time', type=str2bool, default=False, help='test denoise time (test=2)')
    parser.add_argument('--testmodel_n2c', default='checkpoints\last-mcaptcha2.pth', help='path for n2c test model')

    args = parser.parse_args()

    return args


def prepare(args):
    if args.seed == 0:
        args.seed = random.randint(1, 10000)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.logpath = "./exp"
    args.testimagepath = os.path.join(args.logpath, "testimages")
    args.modelsavepath = os.path.join(args.logpath, "saved_models")
    
    os.makedirs(args.testimagepath, exist_ok=True)
    os.makedirs(args.modelsavepath, exist_ok=True)


def main():
    args = parse_args()

    prepare(args)
    GOOGLE_DRIVE_ROOT = "/content/drive/MyDrive/"
    COLAB_ROOT = "/content"
    noisy_path = Path(args.noisy_src_path.replace("\\", "/"))
    clean_path = Path(args.clean_src_path.replace("\\", "/"))
    val_path = Path(args.datarootN_val.replace("\\", "/"))
    test_img_path = Path(args.testimagepath.replace("\\", "/"))
    model_save_path = Path(args.modelsavepath.replace("\\", "/"))

    if os.path.exists(COLAB_ROOT):
        args.noisy_src_path = os.path.join(COLAB_ROOT, noisy_path.parts[-1])
        args.clean_src_path = os.path.join(COLAB_ROOT, clean_path.parts[-1])
        args.datarootN_val = os.path.join(COLAB_ROOT, val_path.parts[-1])
        args.testimagepath = os.path.join(GOOGLE_DRIVE_ROOT, test_img_path.parts[-1])
        args.modelsavepath = os.path.join(GOOGLE_DRIVE_ROOT, model_save_path.parts[-1])
        os.makedirs(args.testimagepath, exist_ok=True)
        os.makedirs(args.modelsavepath, exist_ok=True)
    print(args.noisy_src_path, args, args.clean_src_path, args.datarootN_val)

    fe = FE(args)

    fe.build_model(args)

    if args.test:
        print("Test Started")
        fe.test(args)

    else:
        print("Train Started")
        fe.train(args)


if __name__=="__main__":
    main()