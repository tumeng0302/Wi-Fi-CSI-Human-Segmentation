from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from models.FullModel import FullModel
from models.optimizer import Lion
from Dataset import CSI_Dataset
from models.VAE import Decoder
from utils import Training_Log
# from Loss import dice_loss
from tqdm import tqdm
from torch import nn
import torch
import yaml

torch.set_float32_matmul_precision('medium')

def main():
    log = Training_Log(model_name='FullModel', step_mode='step', weight_start=1)
    DATA_ROOT = '../Data_Disk/CSI_Dataset/'
    torch.cuda.set_device(log.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Training on: device", torch.cuda.current_device(),
        torch.cuda.get_device_name(torch.cuda.current_device()))

    # <--------------------------------------Load pre-trained VAE Decoder-------------------------------------->
    decoder = Decoder()
    print(f'[INFO] Load pre-trained VAE Decoder from {log.decoder_weight}')
    decoder_weight = torch.load(log.decoder_weight)
    for name, param in decoder.named_parameters():
        if 'compiled' in log.decoder_weight:
            name = '_orig_mod.decoder.' + name
        if name in decoder_weight:
            param.data = decoder_weight[name]
            param.requires_grad = False
        else:
            print(f'[Warning] \"{name}\"<- parameter not in pre-trained weight!')
    print('[INFO] VAE Decoder loaded successfully!')
    decoder_weight = None

    # <--------------------------------------Create transformer model-------------------------------------->
    with open('./model_config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.CLoader)
    if log.compile:
        net = torch.compile(FullModel(config, decoder).to(device))
    else:
        net = FullModel(config, decoder).to(device)
    
    if log.resume is not None:
        print(f'[INFO] Resume traing from check-point: {log.resume}')
        net.load_state_dict(torch.load(log.resume))
        print('[INFO] Model loaded successfully!')
    
    # <--------------------------------------Load Dataset-------------------------------------->
    trainset = CSI_Dataset(data_root=DATA_ROOT, split='train', interpolation=0.5)
    trainloader = DataLoader(trainset, batch_size=log.batch, shuffle=True, num_workers=log.num_workers)
    testset = CSI_Dataset(data_root=DATA_ROOT, split='test')
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
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15], gamma=0.7)

    def kl_div(mu:torch.Tensor, logvar:torch.Tensor):
        return (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
    bce = nn.BCEWithLogitsLoss()
    ssim = SSIM(data_range=(0., 1.)).to(device)
    mse = nn.MSELoss()
    cosine = nn.CosineEmbeddingLoss(margin=0.1)

    train_losses = ["total_loss", "cos", "mse", "bce", "ssim", "kl"]
    test_losses = ["total_loss", "mse", "bce", "ssim"]
    log.init_loss(train_losses = train_losses, test_losses = test_losses)

    def train_loss(out: torch.Tensor, mask: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, eps: int)->torch.Tensor:
        # def ratio(x): return 1/(1+torch.exp(-torch.tensor(0.03*x-1.6)))

        mse_loss = mse(torch.sigmoid(out), mask) * 100
        bce_loss = bce(out, mask) * 20
        ssim_loss = -ssim(torch.sigmoid(out).float(), mask.float())
        kl_loss = kl_div(mu, logvar) * 100
        loss = mse_loss + bce_loss + ssim_loss + kl_loss

        return loss, mse_loss, bce_loss, ssim_loss, kl_loss
    
    def feature_loss(features, label)->torch.Tensor:
        cos_loss = 0
        amp1, pha1 = features[0]
        amp2, pha2 = features[1]
        amp1, pha1, amp2, pha2 = amp1.view(amp1.shape[0], -1), pha1.view(pha1.shape[0], -1), amp2.view(amp2.shape[0], -1), pha2.view(pha2.shape[0], -1)
        cos_loss += (cosine(amp1, amp2, label) + cosine(pha1, pha2, label))/2
        return cos_loss/len(features[0])

    def test_loss(out, mask, eps)->torch.Tensor:
        mse_loss = mse(torch.sigmoid(out), mask) * 100
        bce_loss = bce(out, mask) * 20
        ssim_loss = -ssim(torch.sigmoid(out).float(), mask.float())
        loss = mse_loss + bce_loss + ssim_loss
        return loss, mse_loss, bce_loss, ssim_loss
    
    def random_src_mask(shape: list, ratio: float = 0.):
        mask = torch.rand(shape)
        mask = mask < ratio
        return mask

    print('[INFO] Start training...')
    for eps in range(log.total_epochs):
        # ---------------------------training---------------------------#
        net.train()
        i_bar = tqdm(trainloader, unit='iter', desc=f"epoch {eps+1}", ncols=140)
        for steps, [data, cos_label] in enumerate(i_bar):
            cos_label = cos_label.to(device)
            
            features= []
            total_loss: torch.Tensor = 0
            
            for amplitude, phase, mask in data:
                src_mask = random_src_mask([phase.shape[1], phase.shape[1]], 0.1).to(device)
                amplitude, phase, mask = amplitude.to(device), phase.to(device), mask.to(device)

                if log.auto_cast:
                    with autocast():
                        out, mu, logvar, amp_channel, pha_channel = net(amplitude, phase, src_mask)
                        loss, mse_loss, bce_loss, ssim_loss, kl_loss = train_loss(out, mask, mu, logvar, eps)
                        total_loss = total_loss + loss
                        
                else:
                    out, mu, logvar, amp_channel, pha_channel = net(amplitude, phase, src_mask)
                    loss, mse_loss, bce_loss, ssim_loss, kl_loss = train_loss(out, mask, mu, logvar, eps)
                    total_loss = total_loss + loss

                amplitude, phase, src_mask = None, None, None
                features.append([amp_channel, pha_channel])
            
            if log.auto_cast:
                with autocast():
                    cos_loss = feature_loss(features, cos_label)
                    total_loss = (total_loss / 2 + cos_loss) / log.grad_accum
                scaler.scale(total_loss).backward()

                if steps % log.grad_accum == 0:
                    scaler.step(optimizer)
                    optimizer.zero_grad()
                    scaler.update()

            else:
                cos_loss = feature_loss(features, cos_label)
                total_loss = (total_loss / 2 + cos_loss) / log.grad_accum
                total_loss.backward()

                if steps % log.grad_accum == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            features = None
            log.train_loss.push_loss([
                total_loss.item(),
                cos_loss.item(),
                mse_loss.item(),
                bce_loss.item(),
                -ssim_loss.item(),
                kl_loss.item(),
            ])
            loss_str = log.train_loss.avg_loss()
            i_bar.set_postfix_str(str(f"{loss_str}"))
            
            if steps % 3000 == 0 and steps != 0:
                # ---------------------------testing---------------------------#
                net.eval()
                with torch.no_grad():
                    t_bar = tqdm(testloader, unit='iter', desc=f"epoch {eps+1}", ncols=140)
                    for amplitude, phase, mask in t_bar:
                        amplitude, phase, mask = amplitude.to(device), phase.to(device), mask.to(device)
                        out, _, _, _, _ = net(amplitude, phase)
                        loss, mse_loss, bce_loss, ssim_loss = test_loss(out, mask, eps)
                        
                        log.test_loss.push_loss([
                            loss.item(),
                            mse_loss.item(),
                            bce_loss.item(),
                            -ssim_loss.item(),
                        ])
                        loss_str = log.test_loss.avg_loss()
                        t_bar.set_postfix_str(str(f"{loss_str} "))
                        
                        save_img = torch.cat((mask.cpu(), torch.sigmoid(out).cpu().detach()), dim=2)
                log.step(eps, save_img, net.state_dict())
                net.train()

        scheduler.step()
        

if __name__ == '__main__':
    main()