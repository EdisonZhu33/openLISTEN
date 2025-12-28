# Adopted from https://github.com/ddlBoJack/SLAM-LLM/blob/main/src/slam_llm/models/projector.py


import torch
import torch.nn as nn
from math import sqrt
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import copy

class MultiHeadSelfAttention(nn.Module):
    dim_in: int  # input dimension
    dim_k: int   # key and query dimension
    dim_v: int   # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, dim_in, dim_k, dim_v, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)

    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head

        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n

        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v
        return att

class EncoderProjectorConcat(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.speech_encoder_ds_rate
        self.encoder_dim = config.speech_encoder_hidden_size
        self.llm_dim = config.hidden_size
        self.linear1 = nn.Linear(self.encoder_dim * self.k, 4096) #2048
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(4096, config.hidden_size)

    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        num_frames_to_discard = seq_len % self.k
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)
        
        x = x.contiguous()
        x = x.view(batch_size, seq_len // self.k, dim * self.k)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class EncoderProjectorConcat_CNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.speech_encoder_ds_rate
        self.encoder_dim = 128
        self.llm_dim = config.hidden_size

        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(self.encoder_dim *(self.k-1), config.hidden_size)

        self.CNN1 = nn.Conv1d(1280,512,5,stride=5)
        self.bn1 = nn.BatchNorm1d(self.encoder_dim *(self.k-1), eps=1e-3, momentum=0.99)

        self.atten = MultiHeadSelfAttention(dim_in=self.encoder_dim *(self.k-1), dim_k=self.encoder_dim *(self.k-1), dim_v=self.encoder_dim *(self.k-1), num_heads=8)
    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        num_frames_to_discard = seq_len % self.k
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        x = x.permute(0, 2, 1)
        x = x.contiguous()
        #CNN下采样
        # print(f"输入的x:{x.shape}")
        x = self.CNN1(x)
        # print(f"X_CNN:{x.shape}")
        x = self.bn1(x)
        x = self.relu(x)

        #attention
        x = x.permute(0, 2, 1)
        x = self.atten(x)
        # print(f"attention_x:{x.shape}")
        x = self.relu(x)
        x = self.linear1(x)
        # print(f"X_linear1:{x.shape}")

        return x

class CNNAdapter(torch.nn.Module):
    def __init__(
            self,
            enc_out_dim: int = 512,
            llm_embed_dim: int = 4096,
            kernel_size: int = 5,
    ):
        super().__init__()

        self.left_padding1 = nn.ConstantPad1d((kernel_size - 1, 0), 0.0)
        self.left_padding2 = nn.ConstantPad1d((kernel_size - 1, 0), 0.0)

        self.conv1d1 = nn.Conv1d(enc_out_dim, 2 * enc_out_dim, kernel_size, 1, 0)
        self.conv1d2 = nn.Conv1d(2 * enc_out_dim, 4 * enc_out_dim, kernel_size, 1, 0)

        self.bn1 = nn.BatchNorm1d(2 * enc_out_dim, eps=1e-3, momentum=0.99)
        self.bn2 = nn.BatchNorm1d(4 * enc_out_dim, eps=1e-3, momentum=0.99)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.project = nn.Linear(4 * enc_out_dim, llm_embed_dim)

    def forward(self, x, mask_pad):
        """
            x: B, T, enc_out_dim
            mask: (B, T) or (B, 1, T)
        """
        x = x.transpose(1, 2)  # B, channels, T

        # mask batch padding
        if mask_pad.size(2) > 0:  # time > 0
            x.masked_fill_(~mask_pad, 0.0)

        x = self.left_padding1(x)
        x = self.conv1d1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.left_padding2(x)
        x = self.conv1d2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = x.transpose(1, 2)
        x = self.project(x)

        return x, mask_pad




class EncoderProjectorConcat2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.speech_encoder_ds_rate
        self.encoder_dim = config.speech_encoder_hidden_size
        self.llm_dim = config.hidden_size
        self.linear1 = nn.Linear(self.encoder_dim * self.k, 4096)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(4096, config.hidden_size)
        self.n_layers = 2

        #定义decoderlayer
        n_layers, n_dims, n_heads, n_inter_dims = list(map(int, config.ctc_decoder_config[1:-1].split(",")))
        n_layers = self.n_layers
        _config = copy.deepcopy(config)
        _config.hidden_size = n_dims
        _config.num_hidden_layers = n_layers
        _config.num_attention_heads = n_heads
        _config.num_key_value_heads = n_heads
        _config.intermediate_size = n_inter_dims
        self.DecoderLayer = nn.ModuleList(
            [LlamaDecoderLayer(_config, layer_idx) for layer_idx in range(n_layers )]
        )

    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        num_frames_to_discard = seq_len % self.k  # 
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)

        x = x.contiguous()
        x = x.view(batch_size, seq_len // self.k, dim * self.k)
        x = self.linear1(x)
        x = self.relu(x)
        batch_size, seq_len, dim = x.size()


        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0)#.repeat(batch_size,-1)  # 假设每个序列的长度都是425
        # attention_mask = torch.ones_like(x,dtype=torch.long, device=x.device)
        for layer in self.DecoderLayer:
            layer_outputs = layer(
                x,
                # attention_mask=attention_mask,
                position_ids=position_ids,
            )
            x = layer_outputs[0]
        x = self.linear2(x)
        return x