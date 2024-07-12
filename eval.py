from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.classification import BinaryJaccardIndex
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from models.FullModel import FullModel_Finetune
from models.optimizer import Lion
from Dataset import CSI_Dataset
from models.VAE import Decoder
from utils import Training_Log
from torchvision import utils
# from Loss import dice_loss
from tqdm import tqdm
from torch import nn
import torch
import yaml

torch.set_float32_matmul_precision('medium')

def main():
    log = Training_Log(model_name='FullModel_Eval', save_weight=False, step_mode='step')
    DATA_ROOT = '../Data_Disk/CSI_Dataset/'
    print(log.gpu)
    torch.cuda.set_device(log.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Training on: device", torch.cuda.current_device(),
        torch.cuda.get_device_name(torch.cuda.current_device()))

    # <--------------------------------------Create transformer model-------------------------------------->
    decoder = Decoder()
    with open('./model_config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.CLoader)
    if log.compile:
        net = torch.compile(FullModel_Finetune(config, decoder).to(device))
    else:
        net = FullModel_Finetune(config, decoder).to(device)
    
    if log.resume is not None:
        print(f'[INFO] Resume traing from check-point: {log.resume}')
        net.load_state_dict(torch.load(log.resume))
    
    print(f'[INFO] Total parameters: {sum(p.numel() for p in net.parameters())}')
    print(f'[INFO] Trainable parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad)}')
    
    # <--------------------------------------Load Dataset-------------------------------------->
    testset = CSI_Dataset(data_root=DATA_ROOT, split='test')
    testloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=log.num_workers)
    valset = CSI_Dataset(data_root=DATA_ROOT, split='val')
    valloader = DataLoader(valset, batch_size=1, shuffle=True, num_workers=log.num_workers)

    # <--------------------------------------Loss Function-------------------------------------->
    THRESHOLD = 0.3
    ssim = SSIM(data_range=(0., 1.)).to(device)
    mse = nn.MSELoss()
    IoU = BinaryJaccardIndex(threshold=THRESHOLD).to(device)
    train_losses = ["Dummy"]
    test_losses = ["IoU", "mse", "ssim", ]
    metrics = ["IoU", "mse", "ssim", ]
    log.init_loss(train_losses = train_losses, test_losses = test_losses, metrics = metrics)
    def metrics_loss(out, mask)->torch.Tensor:
        mse_loss = mse(torch.sigmoid(out), mask)
        ssim_loss = ssim(torch.sigmoid(out).float(), mask.float())
        IoU_loss = IoU(torch.sigmoid(out).float(), mask.long())
        loss = mse_loss - ssim_loss - IoU_loss
        return loss, mse_loss, ssim_loss, IoU_loss

    print('[INFO] Start training...')
    net.eval()
    log.train_loss.push_loss([0.0])
    loss_str = log.train_loss.avg_loss()
    print(f"{'train':<5} - {loss_str}")
    # ---------------------------testing---------------------------#
    with torch.no_grad():
        t_bar = tqdm(testloader, unit='iter', desc=f"epoch {1}", ncols=140)
        for steps, (amplitude, phase, mask) in enumerate(t_bar):
            amplitude, phase, mask = amplitude.to(device), phase.to(device), mask.to(device)
            out, _, _, _, _ = net(amplitude, phase)
            _, mse_loss, ssim_loss, IoU_loss = metrics_loss(out, mask)
            
            log.test_loss.push_loss([
                IoU_loss.item(),
                mse_loss.item(),
                ssim_loss.item(),
            ])
            loss_str = log.test_loss.avg_loss()
            t_bar.set_postfix_str(str(f"{loss_str} "))
            if steps % 100 == 0:
                out = torch.sigmoid(out).cpu().detach()
                out[out > THRESHOLD] = 1
                out[out <= THRESHOLD] = 0
                test_img = torch.cat((mask.cpu(), out), dim=2)
                utils.save_image(test_img, f'{log.project_name}/out/test_img_{steps}.png')
    # ---------------------------validation---------------------------#
    with torch.no_grad():
        v_bar = tqdm(valloader, unit='iter', desc=f"epoch {1}", ncols=140)
        for steps, (amplitude, phase, mask) in enumerate(v_bar):
            amplitude, phase, mask = amplitude.to(device), phase.to(device), mask.to(device)
            out, _, _, _, _ = net(amplitude, phase)
            _, mse_loss, ssim_loss, IoU_loss = metrics_loss(out, mask)
            
            log.metrics.push_loss([
                IoU_loss.item(),
                mse_loss.item(),
                ssim_loss.item(),
            ])
            loss_str = log.metrics.avg_loss()
            v_bar.set_postfix_str(str(f"{loss_str} "))
            if steps % 100 == 0:
                out = torch.sigmoid(out).cpu().detach()
                out[out > THRESHOLD] = 1
                out[out <= THRESHOLD] = 0
                val_img = torch.cat((mask.cpu(), out), dim=2)
                utils.save_image(val_img, f'{log.project_name}/out/val_img_{steps}.png')
    log.step(1)

if __name__ == '__main__':
    main()
