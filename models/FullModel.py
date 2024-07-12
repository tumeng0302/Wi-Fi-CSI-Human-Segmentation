import torch
from torch import nn
from models.VAE import Decoder
from models.Encoder import ERC_Transformer
from models.Modules import AggregationBlock, CrossAggregationBlock, Adapter_Block
from functools import reduce

class FullModel(nn.Module):
    def __init__(self, CONFIG:dict, decoder:Decoder):
        super(FullModel, self).__init__()
        """
        Full Model for CSI Reconstruction
        args:
            CONFIG: dict, configuration for ERC_Transformer and FullModel
            decoder: nn.Module, VAE Decoder
            vae_latent_shape : [C, H, W]
        """
        print("[MODEL] Create Full Model")
        self.encoder = ERC_Transformer(**CONFIG['ERC_Transformer'])
        self.decoder = decoder
        self.lat_shape = CONFIG['FullModel']['vae_latent_shape']
        latent_dim = reduce(lambda x, y: x*y, self.lat_shape)
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_var = nn.Linear(latent_dim, latent_dim)
        self.fc = nn.Linear(latent_dim, latent_dim)
        if CONFIG['FullModel']['aggregation'] == 'cross':
            self.aggr = CrossAggregationBlock(**CONFIG['AggregationBlock'])
        else:
            self.aggr = AggregationBlock(**CONFIG['AggregationBlock'])
        # self.adapter_1 = Adapter_Block(**CONFIG['Adapter_Block'])
        # self.adapter_2 = Adapter_Block(**CONFIG['Adapter_Block'])

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, amp, pha, srcmask=None):
        if self.encoder.return_channel_stream:
            amp, pha, amp_channel, pha_channel = self.encoder(amp, pha, srcmask)
        else:
            amp, pha = self.encoder(amp, pha, srcmask)
        out = self.aggr(amp, pha)
        mu = self.fc_mu(out)
        log_var = self.fc_var(out)
        z = self.reparameterize(mu, log_var)
        z = self.fc(z)
        z = z.reshape(-1, self.lat_shape[0], self.lat_shape[1]*self.lat_shape[2]).transpose(1, 2)
        # z = self.adapter_1(z, amp, pha)
        # z = self.adapter_2(z, amp, pha)
        z = z.transpose(1, 2).reshape(-1, self.lat_shape[0], self.lat_shape[1], self.lat_shape[2])

        out = self.decoder(z)
        
        if self.encoder.return_channel_stream:
            return out, mu, log_var, amp_channel, pha_channel
        else:
            return out, mu, log_var

class FullModel_Finetune(nn.Module):
    def __init__(self, CONFIG:dict, decoder:Decoder):
        super(FullModel_Finetune, self).__init__()
        """
        Full Model for CSI Reconstruction
        args:
            CONFIG: dict, configuration for ERC_Transformer and FullModel
            decoder: nn.Module, VAE Decoder
            vae_latent_shape : [C, H, W]
        """
        print("[MODEL] Create Full Model")
        self.encoder = ERC_Transformer(**CONFIG['ERC_Transformer'])
        self.decoder = decoder
        self.lat_shape = CONFIG['FullModel']['vae_latent_shape']
        latent_dim = reduce(lambda x, y: x*y, self.lat_shape)
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_var = nn.Linear(latent_dim, latent_dim)
        self.fc = nn.Linear(latent_dim, latent_dim)
        if CONFIG['FullModel']['aggregation'] == 'cross':
            self.aggr = CrossAggregationBlock(**CONFIG['AggregationBlock'])
        else:
            self.aggr = AggregationBlock(**CONFIG['AggregationBlock'])
        self.adapter_1 = Adapter_Block(**CONFIG['Adapter_Block'])

        self.factor = 16
        self.pix_unshuf = nn.PixelUnshuffle(self.factor)
        self.out_adapter = Adapter_Block(**CONFIG['Out_Adapter_Block'])
        self.pix_shuf = nn.PixelShuffle(self.factor)
        self.out_conv = nn.Sequential(
            nn.Conv2d(1, 4, 5, stride=1, padding=2),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(),
            nn.Conv2d(4, 1, 3, stride=1, padding=1),
        )

    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def refine(self, out, amp, pha):
        _, c, h, w = out.shape
        out = self.pix_unshuf(out)
        out = out.reshape(-1, self.factor**2, (h//self.factor) * (w//self.factor)).transpose(1, 2)
        out = self.out_adapter(out, amp, pha)
        out = out.transpose(1, 2).reshape(-1, self.factor**2, h//self.factor, w//self.factor)
        out = self.pix_shuf(out)
        out = self.out_conv(out)
        return out

    def forward(self, amp, pha, srcmask=None):
        if self.encoder.return_channel_stream:
            amp, pha, amp_channel, pha_channel = self.encoder(amp, pha, srcmask)
        else:
            amp, pha = self.encoder(amp, pha, srcmask)
        out = self.aggr(amp, pha)
        mu = self.fc_mu(out)
        log_var = self.fc_var(out)
        z = self.reparameterize(mu, log_var)
        z = self.fc(z)
        z = z.reshape(-1, self.lat_shape[0], self.lat_shape[1], self.lat_shape[2])
        z = z.reshape(-1, self.lat_shape[0], self.lat_shape[1]*self.lat_shape[2]).transpose(1, 2)
        z = self.adapter_1(z, amp, pha)
        z = z.transpose(1, 2).reshape(-1, self.lat_shape[0], self.lat_shape[1], self.lat_shape[2])
        out = self.decoder(z)
        residual = out.clone()
        out = self.refine(out, amp, pha) + residual
        
        if self.encoder.return_channel_stream:
            return out, mu, log_var, amp_channel, pha_channel
        else:
            return out, mu, log_var