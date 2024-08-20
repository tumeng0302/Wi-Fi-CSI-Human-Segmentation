from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.classification import BinaryJaccardIndex
from torch.cuda.amp import autocast, GradScaler
from Dataset import CSI_Dataset_Finetune
from torch.utils.data import DataLoader
from models.FullModel import FullModel
from models.optimizer import Lion
from models.VAE import Decoder
from utils import Training_Log
from Loss import dice_loss
from tqdm import tqdm
from torch import nn
import torch
import yaml

torch.set_float32_matmul_precision('medium')

def main():
    log = Training_Log(model_name='FullModel_finetune', step_mode='step', weight_start=1)
    DATA_ROOT = '../Data_Disk/CSI_Dataset/'
    torch.cuda.set_device(log.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Training on: device", torch.cuda.current_device(),
        torch.cuda.get_device_name(torch.cuda.current_device()))

    # <--------------------------------------Create VAE Decoder-------------------------------------->
    decoder = Decoder()

    # <--------------------------------------Create transformer model-------------------------------------->
    with open('./model_config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.CLoader)
    net = FullModel(config, decoder).to(device)
    net.encoder.return_channel_stream = False
    net.encoder.encoder.return_channel_stream = False
    if log.compile:
        net = torch.compile(net).to(device)

    if log.resume is not None:
        print(f'[INFO] Resume traing from check-point: {log.resume}')
        net_weight = torch.load(log.resume)
        for name, param in net.named_parameters():
            if name in net_weight:
                param.data = net_weight[name]
            
            else:
                print(f'[WARNING] {name} not found in checkpoint!')

        print('[INFO] Model loaded successfully!')

    net.encoder.requires_grad_(False)
    print(f'[INFO] Encoder parameters has been freezed!')
    # net.decoder.requires_grad_(False)
    # print(f'[INFO] Decoder parameters has been freezed!')
    print(f'[INFO] Total parameters: {sum(p.numel() for p in net.parameters())}')
    print(f'[INFO] Trainable parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad)}')
    # <--------------------------------------Load Dataset-------------------------------------->
    trainset = CSI_Dataset_Finetune(data_root=DATA_ROOT, split='train_1', npy_num=log.npy_num)
    trainloader = DataLoader(trainset, batch_size=log.batch, shuffle=True, num_workers=log.num_workers)
    testset = CSI_Dataset_Finetune(data_root=DATA_ROOT, split='test', npy_num=log.npy_num)
    testloader = DataLoader(testset, batch_size=24, shuffle=True, num_workers=log.num_workers)
    valset = CSI_Dataset_Finetune(data_root=DATA_ROOT, split='val', npy_num=log.npy_num)
    valloader = DataLoader(valset, batch_size=23, shuffle=True, num_workers=log.num_workers)

    # <--------------------------------------Optimizer-------------------------------------->
    if log.optimizer == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=log.lr)
    elif log.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(net.parameters(), lr=log.lr)
    elif log.optimizer == 'radam':
        optimizer = torch.optim.RAdam(net.parameters(), lr=log.lr)
    elif log.optimizer == 'lion':
        optimizer = Lion(net.parameters(), lr=log.lr, betas=(0.9, 0.99))
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=log.lr, momentum=0.75, weight_decay=1e-5)

    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4,6,8], gamma=0.5)

    # <--------------------------------------Loss Function-------------------------------------->
    def kl_div(mu:torch.Tensor, logvar:torch.Tensor):
        return (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
    bce = nn.BCEWithLogitsLoss()
    ssim = SSIM(data_range=(0., 1.)).to(device)
    IoU = BinaryJaccardIndex(threshold=0.3).to(device)
    mse = nn.MSELoss()
    
    Wbce = 5
    WIoU = 3
    Wdice = 5
    Wmse =  40

    train_losses = ["total_loss", "dice", "bce", "mse", "ssim", "kl"]
    test_losses = ["total_loss", "dice", "bce", "ssim", "IoU_L"]
    metrics = ["total_loss", "dice", "ssim", "IoU"]
    log.init_loss(train_losses = train_losses, test_losses = test_losses, metrics = metrics)

    def train_loss(out: torch.Tensor, mask: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor)->torch.Tensor:

        dice_ = dice_loss(torch.sigmoid(out).squeeze(1), mask.squeeze(1)) * Wdice
        bce_loss = bce(out, mask) * Wbce
        ssim_loss = -ssim(torch.sigmoid(out).float(), mask.float())
        kl_loss = kl_div(mu, logvar) * 10
        mse_loss = mse(torch.sigmoid(out), mask) * Wmse
        loss: torch.Tensor = dice_ + bce_loss + ssim_loss + kl_loss + mse_loss

        return loss, dice_, bce_loss, ssim_loss, kl_loss, mse_loss

    def test_loss(out: torch.Tensor, mask: torch.Tensor)->torch.Tensor:
        dice_ = dice_loss(torch.sigmoid(out).squeeze(1), mask.squeeze(1)) * Wdice
        bce_loss = bce(out, mask) * Wbce
        ssim_loss = -ssim(torch.sigmoid(out).float(), mask.float())
        IoU_loss = (1 - IoU(torch.sigmoid(out).float(), mask.long())) * WIoU
        loss: torch.Tensor = dice_ + bce_loss + ssim_loss + IoU_loss
        return loss, dice_, bce_loss, ssim_loss, IoU_loss
    
    def metrics_loss(out, mask)->torch.Tensor:
        dice_ = dice_loss(torch.sigmoid(out).squeeze(1), mask.squeeze(1)) * Wdice
        ssim_loss = ssim(torch.sigmoid(out).float(), mask.float())
        IoU_loss = IoU(torch.sigmoid(out).float(), mask.long())
        loss = dice_ - ssim_loss - IoU_loss
        return loss, dice_, ssim_loss, IoU_loss
    

    # <--------------------------------------Other-------------------------------------->
    def random_src_mask(shape: list, ratio: float = 0.):
        mask = torch.rand(shape)
        mask = mask < ratio
        return mask

    print('[INFO] Start training...')
    for eps in range(log.total_epochs):
        # ---------------------------training---------------------------#
        net.train()
        i_bar = tqdm(trainloader, unit='iter', desc=f"epoch {eps+1}", ncols=140)
        for steps, [amplitude, phase, mask] in enumerate(i_bar):
            amplitude, phase, mask = amplitude.to(device), phase.to(device), mask.to(device)

            if log.auto_cast:
                with autocast():
                    out, mu, logvar = net(amplitude, phase, )
                    total_loss, dice_, bce_loss, ssim_loss, kl_loss, mse_loss = train_loss(out, mask, mu, logvar)
                    total_loss = total_loss / log.grad_accum
                scaler.scale(total_loss).backward()

                if steps % log.grad_accum == 0:
                    scaler.step(optimizer)
                    optimizer.zero_grad()
                    scaler.update()
                    
            else:
                out, mu, logvar = net(amplitude, phase, )
                total_loss, dice_, bce_loss, ssim_loss, kl_loss, mse_loss = train_loss(out, mask, mu, logvar)
                total_loss = total_loss / log.grad_accum
                total_loss.backward()

                if steps % log.grad_accum == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            log.train_loss.push_loss([
                total_loss.item(),
                dice_.item(),
                bce_loss.item(),
                mse_loss.item(),
                -ssim_loss.item(),
                kl_loss.item(),

            ])
            loss_str = log.train_loss.avg_loss()
            i_bar.set_postfix_str(str(f"{loss_str}"))
            
            if steps % 500 == 0 and steps != 0:
                train_img = torch.cat((mask.cpu(), torch.sigmoid(out).cpu().detach()), dim=2)
                # ---------------------------testing---------------------------#
                net.eval()
                with torch.no_grad():
                    t_bar = tqdm(testloader, unit='iter', desc=f"epoch {eps+1}", ncols=140)
                    for amplitude, phase, mask in t_bar:
                        amplitude, phase, mask = amplitude.to(device), phase.to(device), mask.to(device)
                        out, _, _ = net(amplitude, phase)
                        loss, dice_, bce_loss, ssim_loss, IoU_loss = test_loss(out, mask)
                        
                        log.test_loss.push_loss([
                            loss.item(),
                            dice_.item(),
                            bce_loss.item(),
                            -ssim_loss.item(),
                            IoU_loss.item(),
                        ])
                        loss_str = log.test_loss.avg_loss()
                        t_bar.set_postfix_str(str(f"{loss_str} "))
                        
                    test_img = torch.cat((mask.cpu(), torch.sigmoid(out).cpu().detach()), dim=2)

                # ---------------------------validation---------------------------#
                with torch.no_grad():
                    v_bar = tqdm(valloader, unit='iter', desc=f"epoch {eps+1}", ncols=140)
                    for amplitude, phase, mask in v_bar:
                        amplitude, phase, mask = amplitude.to(device), phase.to(device), mask.to(device)
                        out, _, _ = net(amplitude, phase)
                        loss, dice_, ssim_loss, IoU_loss = metrics_loss(out, mask)
                        
                        log.metrics.push_loss([
                            loss.item(),
                            dice_.item(),
                            ssim_loss.item(),
                            IoU_loss.item(),
                        ])
                        loss_str = log.metrics.avg_loss()
                        v_bar.set_postfix_str(str(f"{loss_str} "))
                    
                    val_img = torch.cat((mask.cpu(), torch.sigmoid(out).cpu().detach()), dim=2)
                log.step(eps, net.state_dict(), train_img=train_img[:8], test_img=test_img[:10], val_img=val_img)
                net.train()

        scheduler.step()
        

if __name__ == '__main__':
    main()
