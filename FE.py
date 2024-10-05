import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from gen_dataset_synthetic import gen_clean_noisy_dataset
from tqdm import tqdm
from models import *
from utils import *
from dataloader import *

import time
import itertools
import sys
import datetime
import os


class FE(object):
    def __init__(self, args):
        print(args)

    def build_model(self, args):
        """ Transform & Dataloader """
        if not args.test: # train mode
            noisy_dst_path, clean_dst_path = gen_clean_noisy_dataset(noisy_src_path=args.noisy_src_path, clean_src_path=args.clean_src_path)
            args.datarootN, args.datarootC = str(noisy_dst_path), str(clean_dst_path)

        if args.test:
            test_transform = transforms.Compose([transforms.ToTensor()])

            self.test_dataset = real_Dataset_test(
                args.datarootN_val, transform=test_transform
            )


            self.val_loader = DataLoader(
                self.test_dataset, batch_size=1, shuffle=False, num_workers=args.n_cpu
            )
            
        else:
            train_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
            valid_transform = transforms.Compose([transforms.ToTensor()])

            self.train_dataset = realDataset_from_h5(
                args.datarootN,
                args.datarootC,
                transform=train_transform,
            )

            self.val_dataset = real_Dataset_test(
                args.datarootN_val,  
                transform=valid_transform,
            )


            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.n_cpu,
            )

            self.val_loader = DataLoader(
                self.val_dataset, batch_size=1, shuffle=False, num_workers=args.n_cpu
            )

        """ Define Generator & Discriminator """
        self.genN2C = Generator_N2C(
            input_channel=3,
            output_channel=3,
            middle_channel=args.ch_genn2c
        )

        self.disC = Discriminator(
            input_nc=3, ndf=args.ch_dis, n_layers=args.n_conv_dis
        )
        self.disT = Discriminator(
            input_nc=1, ndf=args.ch_dis, n_layers=args.n_conv_dis
        )

        self.genN2C = nn.DataParallel(self.genN2C)
        self.disC = nn.DataParallel(self.disC)
        self.disT = nn.DataParallel(self.disT)

        """ Device """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True

        """ Define Loss """
        self.MSE_Loss = nn.MSELoss().to(self.device)  # LSGAN loss
        self.L1_Loss = nn.L1Loss().to(self.device)
        self.TV_Loss = TVLoss().to(self.device)
        self.SSIM_Loss = SSIM_loss().to(self.device)

        vgg = vgg_19()
        if torch.cuda.is_available():
            vgg.to(self.device)

        vgg.eval()

        for param in vgg.parameters():
            param.requires_grad = False

        self.VGG_Loss = VGG_loss(vgg).to(self.device)
        self.ColorShift = ColorShift(device=self.device)
        self.Freq_Recon_loss = Freq_Recon_loss()

        """ Optimizer """
        self.G_optim = optim.Adam(
            itertools.chain(
            self.genN2C.parameters()
            ),
            lr=args.lrG,
        )
        self.D_optim = optim.Adam(
            itertools.chain(
                self.disC.parameters(), self.disT.parameters()
            ),
            lr=args.lrD,
            betas=(0.5, 0.999),
            weight_decay=0.0001,
        )

    def train(self, args):
        self.genN2C.train().to(self.device)
        self.disC.train().to(self.device), self.disT.train().to(self.device)

        """ CheckPoint """
        ckpt_path = os.path.join(args.modelsavepath, "last.pth")
        last_epoch = 0
        best_epoch = 0
        best_PSNR = 0.0
        best_SSIM = 0.0

        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=self.device)
            self.genN2C.load_state_dict(ckpt["genN2C"])
            self.disC.load_state_dict(ckpt["disL"])
            self.disT.load_state_dict(ckpt["disH"])            
            self.G_optim.load_state_dict(ckpt["G_optimizer"])
            self.D_optim.load_state_dict(ckpt["D_optimizer"])
            last_epoch = ckpt["epoch"] + 1
            best_PSNR = ckpt["best_PSNR"]
            best_SSIM = ckpt["best_SSIM"]
            args.seed = ckpt["seed"]
            print("Last checkpoint is loaded. start_epoch:", last_epoch)
        else:
            print("No checkpoint is found.")

        milestones = [epoch - last_epoch for epoch in eval(args.decay_step)]
        decay_epoch = milestones[0]
        G_schedular = torch.optim.lr_scheduler.LambdaLR(
            self.G_optim, lr_lambda=LambdaLR(args.n_epochs, 0, decay_epoch).step
        )
        D_schedular = torch.optim.lr_scheduler.LambdaLR(
            self.D_optim, lr_lambda=LambdaLR(args.n_epochs, 0, decay_epoch).step
        )

        """ Training Loop """

        prev_time = time.time()
        loop = tqdm(range(last_epoch, args.n_epochs))
        for epoch in loop:
            
            self.genN2C.train()
            self.disC.train(), self.disT.train()
     
            for G_param_group, D_param_group in zip(
                self.G_optim.param_groups, self.D_optim.param_groups
            ):
                print(
                    " ***SEED: %d  LRG: %f  LRD : %f***"
                    % (args.seed, G_param_group["lr"], D_param_group["lr"])
                )
                f = open("%s/log.txt" % args.logpath, "at", encoding="utf-8")
                f.write(
                    "\n ***SEED: %d  LRG: %f  LRD : %f***"
                    % (args.seed, G_param_group["lr"], D_param_group["lr"])
                )
            accu = Accumulator()
            for i, (Noisy, Clean) in enumerate(self.train_loader):
                accu.add(n_iter=1)
                Noisy = Noisy.to(self.device)
                Clean = Clean.to(self.device)


                self.D_optim.zero_grad(set_to_none=True)
                fake_N2C = self.genN2C(Noisy)
                real_img_logit = self.disC(Clean)
                fake_img_logit = self.disC(fake_N2C.detach())
                clean_texture, fake_n2c_texture = self.ColorShift(Clean, fake_N2C)

                real_texture_logit = self.disT(clean_texture.detach())
                fake_texture_logit = self.disT(fake_n2c_texture.detach())
                
                del clean_texture

                D_ad_loss_img = self.MSE_Loss(
                    real_img_logit, torch.ones_like(real_img_logit)
                ) + self.MSE_Loss(
                    fake_img_logit, torch.zeros_like(fake_img_logit)
                )

                D_ad_loss_texture = self.MSE_Loss(
                    real_texture_logit, torch.ones_like(real_texture_logit)
                ) + self.MSE_Loss(
                    fake_texture_logit, torch.zeros_like(fake_texture_logit)
                )
                accu.add(
                    D_ad_img=args.adv_w * D_ad_loss_img.item(),
                    D_ad_texture=args.texture_w * D_ad_loss_texture.item()
                )
       
                (args.adv_w * D_ad_loss_img).backward()
                (args.texture_w * D_ad_loss_texture).backward()
                
                    
                self.D_optim.step()

                # Update G
                self.G_optim.zero_grad(set_to_none=True)
                fake_img_logit = self.disC(fake_N2C)
                fake_texture_logit = self.disT(fake_n2c_texture)
                G_ad_loss_img = self.MSE_Loss(fake_img_logit, torch.ones_like(fake_img_logit))
                G_ad_loss_texture = self.MSE_Loss(fake_texture_logit, torch.ones_like(fake_texture_logit))
                G_ad_loss = args.adv_w * G_ad_loss_img + args.texture_w * G_ad_loss_texture
        
                G_cycle_loss = self.L1_Loss(fake_N2C, Clean)
                G_freq_loss = self.Freq_Recon_loss(fake_N2C, Clean)
                G_vgg_loss = self.VGG_Loss(fake_N2C, Clean)
                G_TV_loss = self.TV_Loss(fake_N2C)

                accu.add(
                    G_ad_img=args.adv_w * G_ad_loss_img.item(),
                    G_ad_texture=args.texture_w * G_ad_loss_texture.item(),
                    G_cycle=args.time_w * G_cycle_loss.item(), 
                    G_freq=args.freq_w * G_freq_loss.item(), 
                    G_vgg=args.vgg_w * G_vgg_loss.item(),
                    G_TV=args.tv_w * G_TV_loss.item(), 
                )   

                del fake_N2C
                # del fake_N2C2N
                
                Generator_loss = (
                    G_ad_loss
                    + args.vgg_w * G_vgg_loss
                    + args.time_w * G_cycle_loss
                    + args.freq_w * G_freq_loss
                    + args.tv_w * G_TV_loss
                )

                Generator_loss.backward()
                self.G_optim.step()

                """ evaluation & checkpoint save & image write """
                batches_done = epoch * len(self.train_loader) + i
                batches_left = args.n_epochs * len(self.train_loader) - batches_done
                time_left = datetime.timedelta(
                    seconds=(batches_left) * (time.time() - prev_time)
                )
                prev_time = time.time()
                
                sys.stdout.write(
                    "\r[Epoch: {}/{}] [Batch: {}/{}] [ETA: {}]".format(
                        epoch,
                        args.n_epochs,
                        i,
                        len(self.train_loader),
                        time_left,
                    )
                )

            val_psnr, val_ssim = self.test(args)

            G_schedular.step()
            D_schedular.step()

            ckpt = {
                "genN2C": self.genN2C.state_dict(),
                "disL": self.disC.state_dict(),
                "disH": self.disT.state_dict(),
                "G_optimizer": self.G_optim.state_dict(),
                "D_optimizer": self.D_optim.state_dict(),
                "epoch": epoch,
                "best_PSNR": best_PSNR,
                "best_SSIM": best_SSIM,
                "seed": args.seed,
            }

            torch.save(ckpt, ckpt_path)

            if best_PSNR < val_psnr:
                best_PSNR = val_psnr
                best_SSIM = val_ssim
                best_epoch = epoch
                torch.save(
                    self.genN2C.state_dict(),
                    "{}/best_G_N2C.pth".format(args.modelsavepath),
                )
                
            print(
                "\n=== Best PSNR: %.4f, SSIM: %.4f at epoch %d === "
                % (best_PSNR, best_SSIM, best_epoch)
            )
            f = open("%s/log.txt" % args.logpath, "at", encoding="utf-8")
            f.write(
                "\n === Best PSNR: %.4f, SSIM: %.4f at epoch %d ==="
                % (best_PSNR, best_SSIM, best_epoch)
            )
            loop.set_postfix(**accu.postfix)
            
    def test(self, args):
        if args.test:
            print("test model loading")
            testmodel_n2c = args.testmodel_n2c
            try:
                self.genN2C.load_state_dict(torch.load(testmodel_n2c, map_location=self.device))
            except:
                self.genN2C.load_state_dict(torch.load(testmodel_n2c, map_location=self.device)["genN2C"])
                print("reload state dict success.")


        self.genN2C.eval().to(self.device)

        cumulative_psnr = 0
        cumulative_ssim = 0
        # resize = transforms.Resize([64, 192]) 
        dataset_len = len(self.val_loader.dataset)
        dataloader_len = len(self.val_loader)
        val_psnr, val_ssim = None, None
        print('dataset_len:', dataset_len)
        with torch.no_grad():

            now = None
            now_str = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
            os.makedirs(args.testimagepath + now_str, exist_ok=True)
            for i, (Clean, Noisy, label) in enumerate(self.val_loader, 1):
                if not now: 
                    now = datetime.datetime.now()

                _, _, h, w = Noisy.size()
     
                Clean = None if args.test == 2 else Clean.to(self.device) 
                Noisy = Noisy.to(self.device)

                output_n2c = self.genN2C(Noisy, instance_norm=True)

                output_n2c = torch.clamp(output_n2c, 0, 1) #截断

                # if args.test:
                if args.test == 2 and args.only_test_time:
                    continue

                save_image(
                    output_n2c[0],
                    "{}/{}.png".format(args.testimagepath + now_str, label[0]),
                    nrow=1,
                    normalize=True, 
                )
                print(f"saved {i+1} denoised images.") if (i+1) % 10 == 0 else None
                if args.test == 2: # denoise mode
  
                    continue  
                cur_psnr = calc_psnr(Clean, output_n2c)
                output_n2c_gray, Clean_gray = RGB2gray(output_n2c, keepdim=True), RGB2gray(Clean, keepdim=True)
                cur_ssim = calc_ssim(output_n2c_gray[0], Clean_gray[0])
                cumulative_psnr += cur_psnr
                cumulative_ssim += cur_ssim

            end = datetime.datetime.now()
            total_ms = (end - now).total_seconds()*1000
            if args.test != 2:
                val_psnr = cumulative_psnr / dataloader_len
                val_ssim = cumulative_ssim / dataloader_len
        
        print("Inference Time Per image (ms):", total_ms / dataset_len)
        print("PSNR : {}, SSIM : {}".format(val_psnr, val_ssim)) if args.test != 2 else None

        return val_psnr, val_ssim
    