""" Pytorch implementation of TRNet, a neural network for multi-frame super resolution (MFSR) by recursive fusion. """

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
import pdb

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5 # 1/sqrt(64)=0.125
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None, maps = None, K=64):
        b, n, _, h = *x.shape, self.heads # b:batch_size, n:17, _:64, heads:heads as an example
        qkv = self.to_qkv(x).chunk(3, dim = -1) # self.to_qkv(x) to generate [b=batch_size, n=17, hd=192]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv) # q, k, v [b=batch_size, heads=heads, n=17, d=depth]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale # [b=batch_size, heads=heads, 17, 17]

        mask_value = -torch.finfo(dots.dtype).max # A big negative number

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True) # [b=batch_size, 17]
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions' # mask [4, 17], dots [4, 8, 17, 17]
            assert len(mask.shape) == 2
            dots = dots.view(-1, K*K, dots.shape[1], dots.shape[2], dots.shape[3])
            mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            dots = dots * mask + mask_value * (1 - mask)
            dots = dots.view(-1, dots.shape[2], dots.shape[3], dots.shape[4])
            del mask

        if maps is not None:
            # maps [16384, 16] -> [16384, 17] , dots [16384, 8, 17, 17]
            maps = F.pad(maps.flatten(1), (1, 0), value = 1.)
            maps = maps.unsqueeze(1).unsqueeze(2)
            dots.masked_fill_(~maps.bool(), mask_value)
            del maps

        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None, maps = None, K=64):
        for attn, ff in self.layers:
            x = attn(x, mask = mask, maps = maps, K = K)
            x = ff(x)
        return x

class SuperResTransformer(nn.Module):
    def __init__(self, *, dim, depth, heads, mlp_dim, dropout = 0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_cls_token = nn.Identity()

    def forward(self, img, mask = None, maps=None, K = 64):
        b, n, _ = img.shape
        # No need to add position code, just add token
        features_token = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((features_token, img), dim=1)
        x = self.transformer(x, mask, maps, K)
        x = self.to_cls_token(x[:, 0])

        return x

class ResidualBlock(nn.Module):
    def __init__(self, channel_size=64, kernel_size=3):
        '''
        Args:
            channel_size : int, number of hidden channels
            kernel_size : int, shape of a 2D kernel
        '''
        super(ResidualBlock, self).__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=kernel_size, padding=padding),
            nn.PReLU(),
            nn.Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=kernel_size, padding=padding),
            nn.PReLU()
        )

    def forward(self, x):
        '''
        Args:
            x : tensor (B, C, W, H), hidden state
        Returns:
            x + residual: tensor (B, C, W, H), new hidden state
        '''
        residual = self.block(x)
        return x + residual

class Encoder(nn.Module):
    def __init__(self, config):
        '''
        Args:
            config : dict, configuration file
        '''
        super(Encoder, self).__init__()

        in_channels = config["in_channels"]
        num_layers = config["num_layers"]
        kernel_size = config["kernel_size"]
        channel_size = config["channel_size"]
        padding = kernel_size // 2

        self.init_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=channel_size, kernel_size=kernel_size, padding=padding),
            nn.PReLU())

        res_layers = [ResidualBlock(channel_size, kernel_size) for _ in range(num_layers)]
        self.res_layers = nn.Sequential(*res_layers)

        self.final = nn.Sequential(
            nn.Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=kernel_size, padding=padding)
        )

    def forward(self, x):
        '''
        Encodes an input tensor x.
        Args:
            x : tensor (B, C_in, W, H), input images
        Returns:
            out: tensor (B, C, W, H), hidden states
        '''
        x = self.init_layer(x)
        x = self.res_layers(x)
        x = self.final(x)
        return x

class Decoder(nn.Module):
    def __init__(self, config):
        '''
        Args:
            config : dict, configuration file
        '''
        super(Decoder, self).__init__()

        self.final = nn.Sequential(nn.Conv2d(in_channels=config["final"]["in_channels"],
                                             out_channels=9,
                                             kernel_size=config["final"]["kernel_size"],
                                             padding=config["final"]["kernel_size"] // 2),
                     nn.PReLU())

        self.pixelshuffle = nn.PixelShuffle(3)

    def forward(self, x):

        x = self.final(x)
        x = self.pixelshuffle(x)

        return x

class TRNet(nn.Module):
    ''' TRNet, a neural network for multi-frame super resolution (MFSR) by recursive fusion. '''

    def __init__(self, config):
        '''
        Args:
            config : dict, configuration file
        '''

        super(TRNet, self).__init__()
        self.encode = Encoder(config["encoder"])
        self.superres = SuperResTransformer(dim=config["transformer"]["dim"],
                                            depth=config["transformer"]["depth"],
                                            heads=config["transformer"]["heads"],
                                            mlp_dim=config["transformer"]["mlp_dim"],
                                            dropout=config["transformer"]["dropout"])
        self.decode = Decoder(config["decoder"])

    def forward(self, lrs, alphas, maps, K):
        '''
        Super resolves a batch of low-resolution images.
        Args:
            lrs : tensor (B, L, W, H), low-resolution images
            alphas : tensor (B, L), boolean indicator (0 if padded low-res view, 1 otherwise)
        Returns:
            srs: tensor (B, C_out, W, H), super-resolved images
        '''
        batch_size, seq_len, heigth, width = lrs.shape
        lrs = lrs.view(-1, seq_len, 1, heigth, width)

        if str(self.encode.init_layer[0])[7] == '2': # The number of channels of the first layer is 2
            refs, _ = torch.median(lrs[:, :9], 1, keepdim=True)  # reference image aka anchor, shared across multiple views
            refs = refs.repeat(1, seq_len, 1, 1, 1)
            stacked_input = torch.cat([lrs, refs], 2) # tensor (B, L, 2*C_in, W, H)
            stacked_input = stacked_input.view(batch_size * seq_len, 2, width, heigth)
            layer1 = self.encode(stacked_input) # encode input tensor
        elif str(self.encode.init_layer[0])[7] == '1': # The number of channels of the first layer is 1
            lrs = lrs.view(batch_size * seq_len, 1, width, heigth)
            layer1 = self.encode(lrs)

        ####################### encode ######################
        layer1 = layer1.view(batch_size, seq_len, -1, width, heigth) # tensor (B, L, C, W, H)

        ####################### fuse ######################
        img = layer1.permute(0, 3, 4, 1, 2).reshape(-1, layer1.shape[1], layer1.shape[2])  # .contiguous().view == .reshape()
        if maps is not None:
            maps = maps.permute(0, 2, 3, 1).reshape(-1, maps.shape[1])
        preds = self.superres(img, mask=alphas, maps=maps, K=K)
        preds = preds.view(-1, K, K, preds.shape[-1]).permute(0, 3, 1, 2)

        ####################### decode ######################
        srs = self.decode(preds)  # decode final hidden state (B, C_out, 3*W, 3*H)
        
        return srs
