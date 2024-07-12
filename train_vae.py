from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from models.optimizer import Lion
from Dataset import Maks_Dataset
from utils import Training_Log
from models.VAE import VAE
from Loss import dice_loss
from tqdm import tqdm
from torch import nn
import torch

torch.set_float32_matmul_precision('medium')

def main():
    log = Training_Log(model_name='VAE', weight_start=1)
    DATA_ROOT = '../Data_Disk/CSI_Dataset/'
    torch.cuda.set_device(log.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Training on: device", torch.cuda.current_device(),
        torch.cuda.get_device_name(torch.cuda.current_device()))

    if log.compile:
        if log.resume is not None:
            net = torch.compile(VAE(activation='leakyrelu').to(device))
            net.load_state_dict(torch.load(log.resume))
            print("[INFO] Model loaded from checkpoint:", log.resume)
            print('[INFO] Model compile set to [True]')
        else:
            net = torch.compile(VAE(activation='leakyrelu').to(device))
            print('[INFO] Model trained from scratch.')
            print('[INFO] Model compile set to [False]')

    else:
        if log.resume is not None:
            net = VAE(activation='leakyrelu').to(device)
            net.load_state_dict(torch.load(log.resume))
            print("[INFO] Model loaded from checkpoint:", log.resume)
            print('[INFO] Model compile set to [True]')
        else:
            net = VAE(activation='leakyrelu').to(device)
            print('[INFO] Model trained from scratch.')
            print('[INFO] Model compile set to [False]')

    
    trainset = Maks_Dataset(data_root=DATA_ROOT, split='train_1')
    trainloader = DataLoader(trainset, batch_size=log.batch, shuffle=True, num_workers=log.num_workers)
    testset = Maks_Dataset(data_root=DATA_ROOT, split='test')
    testloader = DataLoader(testset, batch_size=24, shuffle=True, num_workers=log.num_workers)

    if log.optimizer == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=log.lr)
    elif log.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(net.parameters(), lr=log.lr)
    elif log.optimizer == 'radam':
        optimizer = torch.optim.RAdam(net.parameters(), lr=log.lr)
    elif log.optimizer == 'lion':
        optimizer = Lion(net.parameters(), lr=log.lr, betas=(0.9, 0.99))
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=log.lr, momentum=0.9, weight_decay=1e-5)

    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90, 120, 250], gamma=0.7)

    mse = nn.MSELoss(reduction='mean')
    bce = nn.BCEWithLogitsLoss(reduction='mean')
    ssim = SSIM(data_range=(0., 1.)).to(device)

    def kl_div(mu:torch.Tensor, logvar:torch.Tensor):
        return (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
    
    def train_loss(out: torch.Tensor, label: torch.Tensor, mu=None, logvar=None, test=False):
        mse_loss:torch.Tensor = mse(torch.sigmoid(out), label) * 100
        bce_loss:torch.Tensor = bce(out, label) * 100
        ssim_loss:torch.Tensor =  -ssim(torch.sigmoid(out).float(), label.float())
        dice = dice_loss(torch.sigmoid(out).squeeze(1), label.squeeze(1)) * 10
        if test:
            loss = mse_loss + bce_loss + ssim_loss + dice
            return loss, mse_loss, bce_loss, ssim_loss, dice
        
        else:
            kl_loss:torch.Tensor = kl_div(mu, logvar) * 100
            kl_loss = torch.clip(kl_loss, 0, 10)
            loss:torch.Tensor = mse_loss + kl_loss + bce_loss + ssim_loss + dice
            return loss, mse_loss, kl_loss, bce_loss, ssim_loss, dice

    losses_name = ["total_loss", "mse", "kl", "bce", "ssim", "dice"]
    test_losses_name = ["total_loss", "mse", "bce", "ssim", "dice"]
    log.init_loss(train_losses=losses_name, test_losses=test_losses_name)

    print('[INFO] Start training...')
    for eps in range(log.total_epochs):
        # ---------------------------training---------------------------#
        net.train()
        i_bar = tqdm(trainloader, unit='iter', desc=f"epoch {eps+1}", ncols=140)
        for mask in i_bar:
            label = mask.to(device)
            optimizer.zero_grad()

            if log.auto_cast:
                with autocast():
                    out, mu, logvar = net(label)
                    loss, mse_loss, kl_loss, bce_loss, ssim_loss, dice = train_loss(out, label, mu, logvar)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            else:
                out, mu, logvar = net(label)
                loss, mse_loss, kl_loss, bce_loss, ssim_loss, dice = train_loss(out, label, mu, logvar)
                loss.backward()
                optimizer.step()

            log.train_loss.push_loss([
                loss.item(),
                mse_loss.item(),
                kl_loss.item(),
                bce_loss.item(),
                -ssim_loss.item(),
                dice.item()
            ])
            loss_str = log.train_loss.avg_loss()
            i_bar.set_postfix_str(str(f"{loss_str}"))
        scheduler.step()

        # ---------------------------testing---------------------------#
        net.eval()
        with torch.no_grad():
            t_bar = tqdm(testloader, unit='iter', desc=f"epoch {eps+1}", ncols=140)
            for mask in t_bar:
                label = mask.to(device)
                out, mu, logvar = net(label)
                loss, mse_loss, bce_loss, ssim_loss, dice = train_loss(out, label, test=True)
                
                log.test_loss.push_loss([
                    loss.item(),
                    mse_loss.item(),
                    bce_loss.item(),
                    -ssim_loss.item(),
                    dice.item()
                ])
                loss_str = log.test_loss.avg_loss()
                t_bar.set_postfix_str(
                    str(f"{loss_str} "))
                
                save_img = torch.cat((label.cpu(), out.cpu().detach()), dim=2)
        log.step(test_img=save_img, net_weight=net.state_dict())

if __name__ == '__main__':
    main()