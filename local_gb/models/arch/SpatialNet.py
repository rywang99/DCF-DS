from typing import *

import torch
import torch.nn as nn
from models.arch.base.norm import *
from models.arch.base.non_linear import *
from models.arch.base.linear_group import LinearGroup
from torch import Tensor
from torch.nn import MultiheadAttention
import pdb


from models.io.norm import Norm
from models.io.stft import STFT


paras_16k = {
    'n_fft': 512,
    'n_hop': 256,
    'win_len': 512,
}

paras_8k = {
    'n_fft': 256,
    'n_hop': 128,
    'win_len': 256,
}

class SpatialNetLayer(nn.Module):

    def __init__(
            self,
            dim_hidden: int,
            dim_ffn: int,
            dim_squeeze: int,
            num_freqs: int,
            num_heads: int,
            dropout: Tuple[float, float, float] = (0, 0, 0),
            kernel_size: Tuple[int, int] = (5, 3),
            conv_groups: Tuple[int, int] = (8, 8),
            norms: List[str] = ("LN", "LN", "GN", "LN", "LN", "LN"),
            padding: str = 'zeros',
            full: nn.Module = None,
    ) -> None:
        super().__init__()
        f_conv_groups = conv_groups[0]
        t_conv_groups = conv_groups[1]
        f_kernel_size = kernel_size[0]
        t_kernel_size = kernel_size[1]

        # cross-band block
        # frequency-convolutional module
        self.fconv1 = nn.ModuleList([
            new_norm(norms[3], dim_hidden, seq_last=True, group_size=None, num_groups=f_conv_groups),
            nn.Conv1d(in_channels=dim_hidden, out_channels=dim_hidden, kernel_size=f_kernel_size, groups=f_conv_groups, padding='same', padding_mode=padding),
            nn.PReLU(dim_hidden),
        ])
        # full-band linear module
        self.norm_full = new_norm(norms[5], dim_hidden, seq_last=False, group_size=None, num_groups=f_conv_groups)
        self.full_share = False if full == None else True
        self.squeeze = nn.Sequential(nn.Conv1d(in_channels=dim_hidden, out_channels=dim_squeeze, kernel_size=1), nn.SiLU())
        self.dropout_full = nn.Dropout2d(dropout[2]) if dropout[2] > 0 else None
        self.full = LinearGroup(num_freqs, num_freqs, num_groups=dim_squeeze) if full == None else full
        self.unsqueeze = nn.Sequential(nn.Conv1d(in_channels=dim_squeeze, out_channels=dim_hidden, kernel_size=1), nn.SiLU())
        # frequency-convolutional module
        self.fconv2 = nn.ModuleList([
            new_norm(norms[4], dim_hidden, seq_last=True, group_size=None, num_groups=f_conv_groups),
            nn.Conv1d(in_channels=dim_hidden, out_channels=dim_hidden, kernel_size=f_kernel_size, groups=f_conv_groups, padding='same', padding_mode=padding),
            nn.PReLU(dim_hidden),
        ])

        # narrow-band block
        # MHSA module
        self.norm_mhsa = new_norm(norms[0], dim_hidden, seq_last=False, group_size=None, num_groups=t_conv_groups)
        self.mhsa = MultiheadAttention(embed_dim=dim_hidden, num_heads=num_heads, batch_first=True)
        self.dropout_mhsa = nn.Dropout(dropout[0])
        # T-ConvFFN module
        self.tconvffn = nn.ModuleList([
            new_norm(norms[1], dim_hidden, seq_last=True, group_size=None, num_groups=t_conv_groups),
            nn.Conv1d(in_channels=dim_hidden, out_channels=dim_ffn, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(in_channels=dim_ffn, out_channels=dim_ffn, kernel_size=t_kernel_size, padding='same', groups=t_conv_groups),
            nn.SiLU(),
            nn.Conv1d(in_channels=dim_ffn, out_channels=dim_ffn, kernel_size=t_kernel_size, padding='same', groups=t_conv_groups),
            new_norm(norms[2], dim_ffn, seq_last=True, group_size=None, num_groups=t_conv_groups),
            nn.SiLU(),
            nn.Conv1d(in_channels=dim_ffn, out_channels=dim_ffn, kernel_size=t_kernel_size, padding='same', groups=t_conv_groups),
            nn.SiLU(),
            nn.Conv1d(in_channels=dim_ffn, out_channels=dim_hidden, kernel_size=1),
        ])
        self.dropout_tconvffn = nn.Dropout(dropout[1])
        
        self.embedding_input_proj = nn.Linear(512, dim_hidden)
        self.norm_embedding = new_norm(norms[0], dim_hidden, seq_last=False, group_size=None, num_groups=t_conv_groups)
        # self.attn_spkemb_norm_weights = nn.functional.softmax(nn.Parameter(torch.ones(6) / 6), dim=-1)
        self.dec_spkemb_weights_1 = nn.Parameter(torch.ones(6) / 6)
        self.mhsa1 = MultiheadAttention(embed_dim=dim_hidden, num_heads=num_heads, batch_first=True)
        self.dropout_mhsa1 = nn.Dropout(dropout[0])
        self.norm_mhsa1 = new_norm(norms[0], dim_hidden, seq_last=False, group_size=None, num_groups=t_conv_groups)
        self.norm_mhsa2 = new_norm(norms[0], dim_hidden, seq_last=False, group_size=None, num_groups=t_conv_groups)
        

    def forward(self, x: Tensor, embedding_input: Tensor, att_mask: Optional[Tensor] = None) -> Tensor:
        r"""
        Args:
            x: shape [B, F, T, H]
            att_mask: the mask for attention along T. shape [B, T, T]

        Shape:
            out: shape [B, F, T, H]
        """
        # pdb.set_trace()
        x = x + self._fconv(self.fconv1, x)
        x = x + self._full(x)
        x = x + self._fconv(self.fconv2, x) # torch.Size([4, 257, 800, 96])
        embedding_input = self.embedding_input_proj(embedding_input)
        # pdb.set_trace()
        
        x_, attn = self._tsa(x, embedding_input, att_mask)
        x = x + x_
        x = x + self._tconvffn(x)
        return x, attn

    def _tsa(self, x: Tensor, embedding_input: Tensor, attn_mask: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        B, F, T, H = x.shape 
        # pdb.set_trace()
        x = self.norm_mhsa(x) # B, F, T, H
        x = x.reshape(B * F, T, H)
        
        x_ = self.norm_mhsa2(x)
        x_ = x_.reshape(B * F, T, H) # B*F, T, H
        
        embedding_input = self.norm_embedding(embedding_input) # [B, 6, N, H]
        attn_spkemb_norm_weights = nn.functional.softmax(self.dec_spkemb_weights_1, dim=-1) # [6]
        embedding_input_weighted = (embedding_input * attn_spkemb_norm_weights.view(-1, 1, 1)).sum(dim=1) # [B, 6, N, H] * [6, 1, 1] -> [B, N, H]
        B, N, H1 = embedding_input_weighted.shape
        embedding_input_weighted = embedding_input_weighted[None,...].expand(F, B, N, H1).reshape(B * F, N, H1) # [B*F, N, H]
        # pdb.set_trace()
        
        need_weights = False if hasattr(self, "need_weights") else self.need_weights
        x1, attn = self.mhsa.forward(embedding_input_weighted, x, x, need_weights=need_weights, attn_mask=attn_mask) # [B*F, N, H]
        x1 = self.dropout_mhsa(x1)
        
        # pdb.set_trace()
        x1 = self.norm_mhsa1(x1)
        x2, attn = self.mhsa1.forward(x_, x1, x1, need_weights=need_weights, attn_mask=attn_mask) # [B*F, T, H]
        
        x2 = x2.reshape(B, F, T, H)
        x2 = self.dropout_mhsa1(x2)
        return x2, attn

    def _tconvffn(self, x: Tensor) -> Tensor:
        B, F, T, H0 = x.shape
        # T-Conv
        x = x.transpose(-1, -2)  # [B,F,H,T]
        x = x.reshape(B * F, H0, T)
        for m in self.tconvffn:
            if type(m) == GroupBatchNorm:
                x = m(x, group_size=F)
            else:
                x = m(x)
        x = x.reshape(B, F, H0, T)
        x = x.transpose(-1, -2)  # [B,F,T,H]
        return self.dropout_tconvffn(x)

    def _fconv(self, ml: nn.ModuleList, x: Tensor) -> Tensor:
        B, F, T, H = x.shape
        x = x.permute(0, 2, 3, 1)  # [B,T,H,F]
        x = x.reshape(B * T, H, F)
        for m in ml:
            if type(m) == GroupBatchNorm:
                x = m(x, group_size=T)
            else:
                x = m(x)
        x = x.reshape(B, T, H, F)
        x = x.permute(0, 3, 1, 2)  # [B,F,T,H]
        return x

    def _full(self, x: Tensor) -> Tensor:
        B, F, T, H = x.shape
        x = self.norm_full(x)
        x = x.permute(0, 2, 3, 1)  # [B,T,H,F]
        x = x.reshape(B * T, H, F)
        x = self.squeeze(x)  # [B*T,H',F]
        if self.dropout_full:
            x = x.reshape(B, T, -1, F)
            x = x.transpose(1, 3)  # [B,F,H',T]
            x = self.dropout_full(x)  # dropout some frequencies in one utterance
            x = x.transpose(1, 3)  # [B,T,H',F]
            x = x.reshape(B * T, -1, F)

        x = self.full(x)  # [B*T,H',F]
        x = self.unsqueeze(x)  # [B*T,H,F]
        x = x.reshape(B, T, H, F)
        x = x.permute(0, 3, 1, 2)  # [B,F,T,H]
        return x

    def extra_repr(self) -> str:
        return f"full_share={self.full_share}"


class SpatialNet(nn.Module):

    def __init__(
            self,
            dim_input: int,  # the input dim for each time-frequency point
            dim_output: int,  # the output dim for each time-frequency point
            dim_squeeze: int,
            num_layers: int,
            num_freqs: int,
            encoder_kernel_size: int = 5,
            dim_hidden: int = 192,
            dim_ffn: int = 384,
            num_heads: int = 2,
            dropout: Tuple[float, float, float] = (0, 0, 0),
            kernel_size: Tuple[int, int] = (5, 3),
            conv_groups: Tuple[int, int] = (8, 8),
            norms: List[str] = ("LN", "LN", "GN", "LN", "LN", "LN"),
            padding: str = 'zeros',
            full_share: int = 0,  # share from layer 0
    ):
        super().__init__()

        # encoder
        self.encoder = nn.Conv1d(in_channels=dim_input, out_channels=dim_hidden, kernel_size=encoder_kernel_size, stride=1, padding="same")

        # spatialnet layers
        full = None
        layers = []
        for l in range(num_layers):
            layer = SpatialNetLayer(
                dim_hidden=dim_hidden,
                dim_ffn=dim_ffn,
                dim_squeeze=dim_squeeze,
                num_freqs=num_freqs,
                num_heads=num_heads,
                dropout=dropout,
                kernel_size=kernel_size,
                conv_groups=conv_groups,
                norms=norms,
                padding=padding,
                full=full if l > full_share else None,
            )
            if hasattr(layer, 'full'):
                full = layer.full
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        # decoder
        # self.decoder = nn.Linear(in_features=dim_hidden, out_features=dim_output)
        self.stft = STFT(**paras_16k)
        self.norm = Norm(mode='utterance')
        self.dec_spkemb_weights_2 = nn.Parameter(torch.ones(6) / 6)
        self.embedding_input_proj1 = nn.Linear(512, dim_hidden)
        self.proj = nn.Linear(in_features=8, out_features=16)

    def forward(self, x: Tensor, embedding_outlist, return_attn_score: bool = False) -> Tensor:
        # x: [Batch, Freq, Time, Feature]
        # x: [Batch, T]

        # input_wav = x[:]
        X, stft_paras = self.stft.stft(x) # [Batch, T] -> [Batch, Fre, Time]
        x = torch.unsqueeze(X, dim=1) # [Batch, Ch, Fre, Time]
        B, C, F, T = x.shape
        x, norm_paras = self.norm.norm(x, ref_channel=0) # [Batch, Ch, Fre, Time]
        x = x.permute(0, 2, 3, 1)  # B,F,T,C; complex
        x = torch.view_as_real(x).reshape(B, F, T, -1)  # B,F,T,2C
        # pdb.set_trace()
        
        # Yr_hat = self.norm.inorm(x, norm_paras)
        # yr_hat = self.stft.istft(Yr_hat, stft_paras)
        # print(yr_hat[:,0,:].shape, input_wav.shape)
        # (yr_hat[:,0,:] - input_wav).mean() tensor(-5.7843e-11, device='cuda:0')
        # (yr_hat[:,0,:] - input_wav).max() tensor(2.1197e-06, device='cuda:0')
        # print(torch.allclose(yr_hat[:,0,:], input_wav, atol=1e-7, rtol=1e-1))
        # pdb.set_trace()
        
        B, F, T, H0 = x.shape
        x = self.encoder(x.reshape(B * F, T, H0).permute(0, 2, 1)).permute(0, 2, 1) # [B,F,T,C]
        H = x.shape[2]

        attns = [] if return_attn_score else None
        x = x.reshape(B, F, T, H)
        
        embedding_input = torch.stack(embedding_outlist, dim =1) # [B, 6, SPK, DIM]
        # pdb.set_trace()
        for m in self.layers:
            setattr(m, "need_weights", return_attn_score)
            x, attn = m(x, embedding_input)
            if return_attn_score:
                attns.append(attn)
        # pdb.set_trace()
        embedding_input = self.embedding_input_proj1(embedding_input)
        attn_spkemb_norm_weights = nn.functional.softmax(self.dec_spkemb_weights_2, dim=-1) # [6]
        embedding_input_weighted1 = (embedding_input * attn_spkemb_norm_weights.view(-1, 1, 1)).sum(dim=1) # [B, 6, N, H] * [6, 1, 1] -> [B, N, H]
        B, N, H1 = embedding_input_weighted1.shape
        # pdb.set_trace()
        
        # embedding_input_weighted1 = embedding_input_weighted1.reshape(-1, N, embedding_input_weighted.shape[-1]) # [B, N, H] -> [B*N, H]
        x1 = x.reshape(B, F*T, -1) # [B, F, T, H] -> [B, F*T, H]
        x2 = torch.matmul(x1, embedding_input_weighted1.permute(0,2,1)) # [B, F*T, H] * [B, H, N] -> [B, F*T, N]
        x3 = x2.reshape(B, F, T, N) # [B, F, T, N]
        x3 = self.proj(x3) # [B, F, T, 2N]
        if not torch.is_complex(x3):
            x4 = torch.view_as_complex(x3.float().reshape(B, F, T, -1, 2))  # [B,F,T,Spk]
        # pdb.set_trace()
        x5 = x4.permute(0, 3, 1, 2)  # [B,Spk,F,T]
        # pdb.set_trace()
        
        Yr_hat = self.norm.inorm(x5, norm_paras)
        yr_hat = self.stft.istft(Yr_hat, stft_paras)
        # pdb.set_trace()
        # y = self.decoder(x)
        if return_attn_score:
            return yr_hat.contiguous(), attns
        else:
            return yr_hat.contiguous()


if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES=7, python -m models.arch.SpatialNet
    x = torch.randn((1, 129, 251, 12))  #.cuda() # 251 = 4 second; 129 = 8 kHz; 257 = 16 kHz
    spatialnet_small = SpatialNet(
        dim_input=12,
        dim_output=4,
        num_layers=8,
        dim_hidden=96,
        dim_ffn=192,
        kernel_size=(5, 3),
        conv_groups=(8, 8),
        norms=("LN", "LN", "GN", "LN", "LN", "LN"),
        dim_squeeze=8,
        num_freqs=129,
        full_share=0,
    )  #.cuda()
    # from packaging.version import Version
    # if Version(torch.__version__) >= Version('2.0.0'):
    #     SSFNet_small = torch.compile(SSFNet_small)
    # torch.cuda.synchronize(7)
    import time
    ts = time.time()
    y = spatialnet_small(x)
    # torch.cuda.synchronize(7)
    te = time.time()
    print(spatialnet_small)
    print(y.shape)
    print(te - ts)

    spatialnet_small = spatialnet_small.to('meta')
    x = x.to('meta')
    from torch.utils.flop_counter import FlopCounterMode # requires torch>=2.1.0
    with FlopCounterMode(spatialnet_small, display=False) as fcm:
        y = spatialnet_small(x)
        flops_forward_eval = fcm.get_total_flops()
        res = y.sum()
        res.backward()
        flops_backward_eval = fcm.get_total_flops() - flops_forward_eval

    params_eval = sum(param.numel() for param in spatialnet_small.parameters())
    print(f"flops_forward={flops_forward_eval/1e9:.2f}G, flops_back={flops_backward_eval/1e9:.2f}G, params={params_eval/1e6:.2f} M")
