import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.strategies import DDPStrategy
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
import argparse
import pyopenpose as op
from torchvision import transforms
import os
import sys
import numpy as np
import torchvision

from data.dataset import PersonInWifiDataset
from models import PIF
from utils import MWL2

def argparser():
    parser = argparse.ArgumentParser(description='Wi-Fi CSI Human Segmentation')
    parser.add_argument('--model_name', default='test', help='Name of the model')
    parser.add_argument('--data_root', type=str, default='/root/CSI_Dataset', help='Root directory of the dataset')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--max_epochs', type=int, default=10, help='Maximum number of training epochs')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use for training')
    # size
    parser.add_argument('--crop_size', type=tuple, default=(192, 256), help='Size of the cropped image')
    # openpose
    parser.add_argument('--skeleton_model', type=str, default='/root/openpose/models', help='Path to the openpose library')
    # metric
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary segmentation')
    # resume or test
    parser.add_argument('--ckpt_path', help='Path to the checkpoint file to resume training or testing', required='--resume' in sys.argv or '--test' in sys.argv)
    parser.add_argument('--resume', action='store_true', help='Resume training')
    parser.add_argument('--test', action='store_true', help='Test the model')
    args = parser.parse_args()
    return args

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Skeleton
        params = {
            "model_folder": args.skeleton_model,
            "heatmaps_add_parts": True,
            "heatmaps_add_PAFs": True,
            "heatmaps_add_bkg": True,
            "heatmaps_scale": 2,
        }

        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()

        self.model = PIF.Model()

        self.BItransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(args.crop_size, antialias=True),
        ])

        # loss
        # binary cross entropy loss for segmentation
        self.bce = nn.BCELoss()
        # Matthew Weight L2 loss for keypoint detection
        self.mwl2_jhm = MWL2(k=1, b=1)
        self.mwl2_paf = MWL2(k=1, b=0.3)
        # mean squared error for keypoint detection
        # self.mse = nn.MSELoss()

        # metric
        self.iou = BinaryJaccardIndex(threshold=args.threshold)
        self.ssim = SSIM(data_range=(0., 1.))
        self.log_iou = None
        self.log_ssim = None
        self.log_dice = None

    def generate_skeleton(self, img):
        # generate JHM and PAF of skeleton
        img_gt_np = img.cpu().detach().numpy()
        jhm_gt, paf_gt = None, None
        for img in img_gt_np:
            datum = op.Datum()
            datum.cvInputData = img
            self.opWrapper.emplaceAndPop(op.VectorDatum([datum]))
            heatmaps = datum.poseHeatMaps.copy()
            heatmaps = (heatmaps).astype(dtype='uint8')

            jhm = self.BItransform(heatmaps[:25].transpose(1, 2, 0)).unsqueeze(0)
            paf = self.BItransform(heatmaps[26:].transpose(1, 2, 0)).unsqueeze(0)
            if jhm_gt is None and paf_gt is None:
                jhm_gt = jhm
                paf_gt = paf
            else:
                jhm_gt = torch.cat((jhm_gt, jhm), 0)
                paf_gt = torch.cat((paf_gt, paf), 0)

        return jhm_gt, paf_gt



    def training_step(self, batch, batch_idx):
        amp, pha, mask_gt, img_gt = batch

        # generate JHM and PAF of skeleton
        jhm_gt, paf_gt = self.generate_skeleton(img_gt)
        jhm_gt = jhm_gt.type_as(mask_gt)
        paf_gt = paf_gt.type_as(mask_gt)

        # model
        sm, jhm, paf = self.model(amp)

        # loss
        bce = self.bce(sm, mask_gt)
        mwl2_jhm = self.mwl2_jhm(jhm, jhm_gt)
        mwl2_paf = self.mwl2_paf(paf, paf_gt)
        loss = 0.1*bce + mwl2_jhm + mwl2_paf

        # log
        self.log_dict({
            'Loss': loss,
            'BCE': bce,
            'MSE_JHM': mwl2_jhm,
            'MSE_PAF': mwl2_paf
        }, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        # log image
        self.logger.experiment.add_images('image', img_gt.permute(0, 3, 1, 2)[0:1], batch_idx)
        self.logger.experiment.add_images('mask-gt', mask_gt[0:1], batch_idx)
        self.logger.experiment.add_images('mask', sm[0:1], batch_idx)
        self.logger.experiment.add_images('jhm-gt', jhm_gt[0, :10].unsqueeze(1), batch_idx)
        self.logger.experiment.add_images('jhm', jhm[0, :10].unsqueeze(1), batch_idx)
        self.logger.experiment.add_images('paf-gt', paf_gt[0, :10].unsqueeze(1), batch_idx)
        self.logger.experiment.add_images('paf', paf[0, :10].unsqueeze(1), batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        amp, pha, mask_gt, img_gt = batch

        # generate JHM and PAF of skeleton
        jhm_gt, paf_gt = self.generate_skeleton(img_gt)
        jhm_gt = jhm_gt.type_as(mask_gt)
        paf_gt = paf_gt.type_as(mask_gt)

        # model
        sm, jhm, paf = self.model(amp)

        # metric
        # IoU bewtween mask_gt and sm
        iou = self.iou(sm, mask_gt.long())
        self.log('val_iou', iou)


    def test_step(self, batch, batch_idx):
        amp, pha, mask_gt, img_gt = batch

        # generate JHM and PAF of skeleton
        jhm_gt, paf_gt = self.generate_skeleton(img_gt)
        jhm_gt = jhm_gt.type_as(mask_gt)
        paf_gt = paf_gt.type_as(mask_gt)

        # model
        sm, jhm, paf = self.model(amp)
        from utils import dice_loss
        # metric
        # IoU bewtween mask_gt and sm
        iou = self.iou(sm, mask_gt.long())
        dice = dice_loss(sm.squeeze(1), mask_gt.squeeze(1))
        ssim = self.ssim(sm.float(), mask_gt.float())
        
        self.log_dict({
            'test_iou': iou,
            'test_dice': dice,
            'test_ssim': ssim,
        }, on_step=True, prog_bar=True, logger=True, sync_dist=True)

        if self.log_iou is None:
            self.log_iou = iou.cpu().detach().numpy().reshape(1)
            self.log_ssim = ssim.cpu().detach().numpy().reshape(1)
            self.log_dice = dice.cpu().detach().numpy().reshape(1)
        else:
            self.log_iou = np.concatenate((self.log_iou, iou.cpu().detach().numpy().reshape(1)))
            self.log_ssim = np.concatenate((self.log_ssim, ssim.cpu().detach().numpy().reshape(1)))
            self.log_dice = np.concatenate((self.log_dice, dice.cpu().detach().numpy().reshape(1)))

        # if batch_idx % 76 == 0:
        #     sm = sm.cpu().detach()
        #     threshed = sm.clone()
        #     threshed[threshed > args.threshold] = 1
        #     threshed[threshed <= args.threshold] = 0
        #     test_img = torch.cat((mask_gt.cpu(), sm, threshed), dim=2)
        #     torchvision.utils.save_image(test_img, f'./out/val_img_{batch_idx}.png')
        


        # log image
        self.logger.experiment.add_images('test-mask-gt', mask_gt[0:1], batch_idx)
        self.logger.experiment.add_images('test-mask', sm[0:1], batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)

# Parse arguments
args = argparser()
# logger
tb_logger = pl_loggers.TensorBoardLogger(
    save_dir=os.getcwd(),
    version=args.model_name,
    name='lightning_logs'
)
# Data

model = LitModel()

# trainer = pl.Trainer(fast_dev_run=True, accelerator='gpu', devices=[1])
trainer = pl.Trainer(max_epochs=10,
                     val_check_interval=10000, limit_val_batches=0.25,
                     strategy=DDPStrategy(find_unused_parameters=True),
                     callbacks=pl.callbacks.ModelCheckpoint(every_n_train_steps=10000, save_top_k=-1),
                     logger=tb_logger,
                     devices=[1])

if args.test:
    # Testing
    split = 'test'
    test_dataset = PersonInWifiDataset(data_root=args.data_root, split=split)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)
    trainer.test(model, test_loader, ckpt_path=args.ckpt_path)
    print(f'Average IoU: {model.log_iou.mean()}')
    print(f'Average SSIM: {model.log_ssim.mean()}')
    print(f'Average Dice: {model.log_dice.mean()}')
    np.save(f'./out/{split}_iou.npy', model.log_iou)
    np.save(f'./out/{split}_ssim.npy', model.log_ssim)
    np.save(f'./out/{split}_dice.npy', model.log_dice)
else:
    # Training
    # Create datasets
    train_dataset = PersonInWifiDataset(data_root=args.data_root, split='train_1')
    val_dataset = PersonInWifiDataset(data_root=args.data_root, split='val')
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=8)

    if not args.resume:
        args.ckpt_path = None

    trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt_path)