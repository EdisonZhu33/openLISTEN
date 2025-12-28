# Adopted from https://github.com/haotian-liu/LLaVA. We modify the code to support speech input. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from abc import ABC, abstractmethod

import torch
from einops import rearrange, repeat
from torch import einsum, nn
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.patches import Ellipse

from .speech_encoder.builder import build_speech_encoder
from .speech_projector.builder import build_speech_projector
IGNORE_INDEX = -100
SPEECH_TOKEN_INDEX = -200
DEFAULT_SPEECH_TOKEN = "<speech>"

import torch
import matplotlib.pyplot as plt

from matplotlib import transforms  # 需要显式导入transforms模块
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch
import numpy as np




def confidence_ellipse(x, y, ax, n_std=2.0, **kwargs):
    """绘制置信椭圆 [[9]]"""
    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(np.mean(x), np.mean(y))
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def find_nearest_point(point, points):
    """查找最近邻点"""
    return points[np.argmin(np.linalg.norm(points - point, axis=1))]



class CrossAttention(nn.Module):
    def __init__(self, text_dim: int, audio_dim: int, num_heads: int = 8):

        super().__init__()

        self.text_dim = text_dim  # 新增属性保存维度
        self.num_heads = num_heads
        self.head_dim = text_dim // num_heads
        assert self.head_dim * num_heads == text_dim, "text_dim必须能被num_heads整除"
        self.q_projs = nn.Linear(text_dim, text_dim//1)
        self.k_projs = nn.Linear(audio_dim, text_dim//1)
        self.v_projs = nn.Linear(audio_dim, text_dim//1)
        self.scale = num_heads**-0.5
        # 输出层与归一化
        self.out_proj = nn.Linear(text_dim//1, text_dim)
        self.norm = nn.LayerNorm(text_dim)

    def forward(self, text_emb: torch.Tensor, audio_emb: torch.Tensor) -> torch.Tensor:
        batch_size = text_emb.size(0)


        Qs = self.q_projs(text_emb)  # [B, text_len, text_dim]
        Ks = self.k_projs(audio_emb)  # [B, audio_len, text_dim]
        Vs = self.v_projs(audio_emb)  # [B, audio_len, text_dim]

        h = self.num_heads
        Qs, Ks, Vs = rearrange(Qs, "b n (h d) -> b h n d", h=h), rearrange(Ks, "b n (h d) -> b h n d", h=h), rearrange(Vs,"b n (h d) -> b h n d",h=h)
        Qs = Qs * self.scale

   
        sim = einsum("... i d, ... j d  -> ... i j", Qs, Ks)


        attn = sim.softmax(dim=-1)
        out = einsum("... i j, ... j d -> ... i d", attn, Vs)

        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        context  = out


        output = self.out_proj(context)
        output = self.norm(output + text_emb)  
        return output



class CustomModel(nn.Module):
    def __init__(self,inputs,outputs):
        super(CustomModel, self).__init__()
        self.linear = nn.Linear(in_features=inputs, out_features=outputs, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)

        return x

class OmniSpeechMetaModel:

    def __init__(self, config):
        super(OmniSpeechMetaModel, self).__init__(config)

        if hasattr(config, "speech_encoder"):
            self.speech_encoder = build_speech_encoder(config)
            self.speech_projector = build_speech_projector(config)


        # 新增模块初始化
        self.cross_attn = CrossAttention(config.hidden_size, config.hidden_size, 8)
        self.gate_projs = CustomModel(2 * config.hidden_size, config.hidden_size)



    def get_speech_encoder(self):
        speech_encoder = getattr(self, 'speech_encoder', None)
        if type(speech_encoder) is list:
            speech_encoder = speech_encoder[0]
        return speech_encoder

    def initialize_speech_modules(self, model_args, fsdp=None):
        self.config.speech_encoder = getattr(model_args, "speech_encoder", None)
        self.config.speech_encoder_type = getattr(model_args, "speech_encoder_type", None)
        self.config.speech_projector_type = getattr(model_args, 'speech_projector_type', 'linear')
        self.config.speech_encoder_ds_rate = getattr(model_args, 'speech_encoder_ds_rate', 5)
        self.config.speech_encoder_hidden_size = getattr(model_args, 'speech_encoder_hidden_size', 1280)

        if self.get_speech_encoder() is None:
            speech_encoder = build_speech_encoder(self.config)

            if fsdp is not None and len(fsdp) > 0:
                self.speech_encoder = [speech_encoder]
            else:
                self.speech_encoder = speech_encoder#.to(dtype=torch.bfloat16)

        if getattr(self, 'speech_projector', None) is None:
            self.speech_projector = build_speech_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.speech_projector.parameters():
                p.requires_grad = True

        if model_args.pretrain_speech_projector is not None:
            pretrain_speech_projector_weights = torch.load(model_args.pretrain_speech_projector, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.speech_projector.load_state_dict(get_w(pretrain_speech_projector_weights, 'speech_projector'))


class OmniSpeechMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_speech_encoder(self):
        return self.get_model().get_speech_encoder()
    
    def get_speech_projector(self):
        return self.get_model().speech_projector

    def get_cross_attn(self):
        return self.get_model().cross_attn

    def get_gates(self):
        return self.get_model().gate_projs

    def encode_speech(self, speech, speech_lengths):
        speech_encoder_type = self.config.speech_encoder_type
        speech_encoder = self.get_speech_encoder()
        if "whisper" in speech_encoder_type.lower():
            speech = speech.to(dtype=torch.bfloat16)
            encoder_outs = speech_encoder(speech.permute(0, 2, 1))
            speech_lengths = (speech_lengths + 1) // 2
        else:
            raise ValueError(f'Unknown speech encoder: {speech_encoder}')
        speech_projector_type = self.config.speech_projector_type
        speech_projector = self.get_speech_projector()
        if speech_projector_type == "linear":
            encoder_outs = speech_projector(encoder_outs)
            speech_lengths = speech_lengths // speech_projector.k
        else:
            raise ValueError(f'Unknown speech projector: {speech_projector_type}')
        speech_features = [encoder_outs[i, :speech_lengths[i]] for i in range(len(encoder_outs))]
        return speech_features


    def fuse_modalities(self, text_emb, audio_emb):


        cross_attns = self.get_cross_attn()
        gates_projs  = self.get_gates()

        attn_text  = cross_attns(text_emb,audio_emb)

        audio_context = audio_emb.mean(dim=1, keepdim=True)  # [B, 1, D]
        combined = torch.cat([attn_text, audio_context.expand_as(attn_text)], dim=-1)

        gate = gates_projs(combined)

        fused_text = gate * attn_text + (1 - gate) * audio_context.expand_as(attn_text)


        fused_out = fused_text

        return fused_out


      

    def calculate_lable_length(self,tensor):
        if tensor is None:
            return 0
        last_index = 0
        for i in range(len(tensor[0])):
            if tensor[0][i]!=IGNORE_INDEX:

                last_index = i
                break

        return len(tensor[0]) - last_index



    def plot_embedding_abs_means(self,text_emb_list, audio_emb, final_embs):

        text_cat = torch.cat(text_emb_list)
        text_line = torch.abs(text_cat.float()).mean(dim=-1).cpu().numpy()
        audio_line = torch.abs(audio_emb.float()).mean(dim=-1).cpu().numpy()
        final_line = torch.abs(final_embs.float()).mean(dim=-1).cpu().numpy()


        plt.figure(figsize=(8, 6))
        plt.plot(text_line, label='Text Embedding', color='blue', marker='o')
        plt.plot(audio_line, label='Audio Embedding', color='green', marker='o')
        plt.plot(final_line, label='Final Embedding', color='red', marker='o')


        plt.xlabel('Index')
        plt.ylabel('Average Absolute Value')
        plt.title('Average Absolute Values of Embeddings')
        plt.legend(loc='upper right')  # Legend in the upper right corner
        plt.grid(True)
        plt.show()

    def plot_embedding_clusters5(self, text_emb_list, audio_emb, final_embs):

        text_cat = torch.cat([t.float().cpu() for t in text_emb_list]).detach().numpy()
        audio_data = audio_emb.float().cpu().detach().numpy()
        final_data = final_embs.float().cpu().detach().numpy()

        text_mean = np.mean(text_cat, axis=-1)
        audio_mean = np.mean(audio_data, axis=-1)
        final_mean = np.mean(final_data, axis=-1)


        all_means = np.concatenate([text_mean, audio_mean, final_mean])
        scaler = StandardScaler()
        scaled_means = scaler.fit_transform(all_means.reshape(-1, 1)).flatten()


        split_idx = [len(text_mean), len(audio_mean), len(final_mean)]
        text_scaled = scaled_means[:split_idx[0]]
        audio_scaled = scaled_means[split_idx[0]:split_idx[0] + split_idx[1]]
        final_scaled = scaled_means[split_idx[0] + split_idx[1]:]


        n_text = len(text_scaled)
        n_audio = len(audio_scaled)
        n_final = len(final_scaled)
        max_n = max(n_text, n_audio, n_final)


        plt.figure(figsize=(6, 4))


        x_jitter = lambda n: np.random.normal(0, 0.5, n)

        plt.scatter(np.arange(1, n_text + 1) + x_jitter(n_text),
                    text_scaled, c='red',alpha=0.5, label=f'Text Embedding')#, s=100)
        plt.scatter(np.arange(1, n_audio + 1) + x_jitter(n_audio),
                    audio_scaled, c='blue',alpha=0.5, label=f'Audio Embedding')#, s=100)
        plt.scatter(np.arange(1, n_final + 1) + x_jitter(n_final),
                    final_scaled, c='green',alpha=0.5, label=f'Fused Embedding')#, s=100)


        plt.title('Feature Mean Distribution by Sample Quantity')#, fontsize=20)
        plt.xlabel('Sample Index within Modality')#, fontsize=20)
        plt.ylabel('Standardized Mean Value')#, fontsize=20)
        plt.xlim(0, max_n * 1.05)
        plt.legend()#(fontsize=20)
        plt.grid( linestyle='--', alpha=0.5)
        plt.tight_layout()

        output_path = ""  

        base_filename = ""

        formats = ['svg', 'pdf', 'png'] 

        for fmt in formats:
            output_path = f"{base_filename}.{fmt}"

            if fmt == 'png': 
                plt.savefig(output_path,
                            format=fmt,
                            dpi=600,  # 
                            bbox_inches='tight')
            else: 
                plt.savefig(output_path,
                            format=fmt,
                            bbox_inches='tight')

        plt.show()

    def plot_embedding_clusters6(self, text_emb_list, audio_emb, final_embs):

        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
        from sklearn.preprocessing import StandardScaler


        text_cat = torch.cat([t.float().cpu() for t in text_emb_list]).detach().numpy()
        audio_data = audio_emb.float().cpu().detach().numpy()
        final_data = final_embs.float().cpu().detach().numpy()

        text_mean = np.mean(text_cat, axis=-1)
        audio_mean = np.mean(audio_data, axis=-1)
        final_mean = np.mean(final_data, axis=-1)


        all_means = np.concatenate([text_mean, audio_mean, final_mean])
        scaler = StandardScaler()
        scaled_means = scaler.fit_transform(all_means.reshape(-1, 1)).flatten()


        split_idx = [len(text_mean), len(audio_mean), len(final_mean)]
        text_scaled = scaled_means[:split_idx[0]]
        audio_scaled = scaled_means[split_idx[0]:split_idx[0] + split_idx[1]]
        final_scaled = scaled_means[split_idx[0] + split_idx[1]:]

        n_text = len(text_scaled)
        n_audio = len(audio_scaled)
        n_final = len(final_scaled)

        x = np.arange(1, n_text + 1)
        y = np.zeros(n_text)
        z = text_scaled

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')


        sc = ax.scatter(x, y, z, c=z, cmap='coolwarm', alpha=0.7, label=f'Text (n={n_text})')


        x_audio = np.arange(1, n_audio + 1) + n_text * 1.1
        y_audio = np.ones(n_audio)
        z_audio = audio_scaled
        sc = ax.scatter(x_audio, y_audio, z_audio, c=z_audio, cmap='coolwarm', alpha=0.7, label=f'Audio (n={n_audio})')


        x_final = np.arange(1, n_final + 1) + (n_text + n_audio) * 1.1
        y_final = np.ones(n_final) * 2
        z_final = final_scaled
        sc = ax.scatter(x_final, y_final, z_final, c=z_final, cmap='coolwarm', alpha=0.7, label=f'Final (n={n_final})')


        ax.set_xlabel('Sample Index', fontsize=12)
        ax.set_ylabel('Feature Type', fontsize=12)
        ax.set_zlabel('Standardized Mean Value', fontsize=12)
        ax.set_title('3D Feature Mean Distribution Visualization', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True)
        fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10)

        plt.tight_layout()
        plt.show()

    def prepare_inputs_labels_for_speech_and_text(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        speech, speech_lengths
    ):
        speech_encoder = self.get_speech_encoder()
        if speech_encoder is None or speech is None or input_ids.shape[1] == 1:

            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        speech_features = self.encode_speech(speech, speech_lengths)


        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []

        text_emb_list = []
        cur_speech_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_speech = (cur_input_ids == SPEECH_TOKEN_INDEX).sum()
            if num_speech == 0:
                cur_speech_features = speech_features[cur_speech_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)



                text_emb_list.append(cur_input_embeds_1)

                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_speech_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_speech_idx += 1
                continue

            speech_token_indices = [-1] + torch.where(cur_input_ids == SPEECH_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_nospeech = []
            cur_labels = labels[batch_idx]
            cur_labels_nospeech = []
            for i in range(len(speech_token_indices) - 1):
                cur_input_ids_nospeech.append(cur_input_ids[speech_token_indices[i]+1:speech_token_indices[i+1]])
                cur_labels_nospeech.append(cur_labels[speech_token_indices[i]+1:speech_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_nospeech]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_nospeech))
            cur_input_embeds_no_speech = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []


            for i in range(num_speech + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_speech[i])
                cur_new_labels.append(cur_labels_nospeech[i])


                text_emb_list.append(cur_input_embeds_no_speech[i])

                if i < num_speech:
                    cur_speech_features = speech_features[cur_speech_idx]
                    cur_speech_idx += 1
                    cur_new_input_embeds.append(cur_speech_features)
                    cur_new_labels.append(torch.full((cur_speech_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        text_emb_list_1 = []
        text_emb_list_2 = []
        for i in text_emb_list:
            t = i.clone()
            j = i.clone()
            # print(f"t {t.shape}")
            text_emb_list_1.append(t)
            text_emb_list_2.append(j)


        if labels !=None:
            labels_lens = self.calculate_lable_length(new_labels)
        else:
            labels_lens = 0


        text_emb_list_p = torch.cat(text_emb_list_1)[:labels_lens,:]

        if new_labels !=None:

            only_lables_embs = torch.cat(text_emb_list_2)[text_emb_list_p.shape[0]:,:]
            only_lables_embs = only_lables_embs.unsqueeze(0)


        text_emb = text_emb_list_p

        audio_emb = torch.cat(speech_features)  

        final_emb = self.fuse_modalities(
            audio_emb.unsqueeze(0),
            text_emb.unsqueeze(0)
        )

        audio_emb = audio_emb.unsqueeze(0)
        if labels !=None:
            final_embs =  torch.cat([text_emb.unsqueeze(0),final_emb,only_lables_embs], dim=1)
        else:

            final_embs =  torch.cat([text_emb.unsqueeze(0),final_emb], dim=1)

        final_embs = final_embs.squeeze(0)

        new_input_embeds = []
        new_input_embeds.append(final_embs)

        




        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]


        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]

            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None


        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels



    '''
    def prepare_inputs_labels_for_speech_and_text(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        speech, speech_lengths
    ):
        speech_encoder = self.get_speech_encoder()
        if speech_encoder is None or speech is None or input_ids.shape[1] == 1:
            # print(f"speech is None ? {speech is None} ro input_ids.shape[1] {input_ids.shape[1] == 1} or  if speech_encoder is None {speech_encoder is None}")
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        speech_features = self.encode_speech(speech, speech_lengths)

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []

        text_emb_list = []
        cur_speech_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_speech = (cur_input_ids == SPEECH_TOKEN_INDEX).sum()
            if num_speech == 0:
                cur_speech_features = speech_features[cur_speech_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)


                #单独获取text_tokens_embed--------新加(跨注意力)
                text_emb_list.append(cur_input_embeds_1)

                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_speech_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_speech_idx += 1
                continue

            speech_token_indices = [-1] + torch.where(cur_input_ids == SPEECH_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_nospeech = []
            cur_labels = labels[batch_idx]
            cur_labels_nospeech = []
            for i in range(len(speech_token_indices) - 1):
                cur_input_ids_nospeech.append(cur_input_ids[speech_token_indices[i]+1:speech_token_indices[i+1]])
                cur_labels_nospeech.append(cur_labels[speech_token_indices[i]+1:speech_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_nospeech]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_nospeech))
            cur_input_embeds_no_speech = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []


            for i in range(num_speech + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_speech[i])
                cur_new_labels.append(cur_labels_nospeech[i])

                #单独获取text_tokens_embed--------新加(跨注意力)
                text_emb_list.append(cur_input_embeds_no_speech[i])

                if i < num_speech:
                    cur_speech_features = speech_features[cur_speech_idx]
                    cur_speech_idx += 1
                    cur_new_input_embeds.append(cur_speech_features)
                    cur_new_labels.append(torch.full((cur_speech_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
        text_emb_list_1 = []
        text_emb_list_2 = []
        for i in text_emb_list:
            t = i.clone()
            j = i.clone()
            # print(f"t {t.shape}")
            text_emb_list_1.append(t)
            text_emb_list_2.append(j)

        # print(f" text_emb_list_2 { text_emb_list_2[0].shape}")
        # print(f"new_lables {new_labels} and {torch.cat(new_labels).shape}")
        if labels !=None:
            labels_lens = self.calculate_lable_length(new_labels)
        else:
            labels_lens = 0
        # print(f"lables_lens: {labels_lens}")
        # print(f"之前的text_emb_list {torch.cat(text_emb_list).shape}")

        text_emb_list_p = torch.cat(text_emb_list_1)[:labels_lens,:]
        # print(f"去掉lables {text_emb_list_p.shape}")
        if new_labels !=None:
            # print(f"text_emb_list_2[0] {torch.cat(text_emb_list_2).shape}")
            # print(f"text_emb_list_p.shape[0] {text_emb_list_p.shape[0]}")
            only_lables_embs = torch.cat(text_emb_list_2)[text_emb_list_p.shape[0]:,:]
            only_lables_embs = only_lables_embs.unsqueeze(0)
            # print(f"only_lables: {only_lables_embs.shape}")

        text_emb = text_emb_list_p #torch.cat(text_emb_list)  # .permute(0,2,1)  # [1, T, D]
        # print(f"text_emb {text_emb.shape}")
        audio_emb = torch.cat(speech_features)  # .permute(0,2,1)  # [1, S, D]
        # print(f"audio_emb {audio_emb.shape}")

        # print(f"audio_emb {audio_emb} and {audio_emb.dtype}")
        # print(f"text_emb {text_emb} and {text_emb.dtype}")
        # 确保数据类型一致
        # print(f"纯文本的均值 {text_emb.mean(dim=-1)}")
        # print(f"audio特征的均值 : {audio_emb.mean(dim=-1)}")
        # 4.3 交叉注意力（文本作为Query）和门控
        final_emb = self.fuse_modalities(
            text_emb.unsqueeze(0),
            audio_emb.unsqueeze(0)
        )
        # print(f"final_emb {final_emb.shape}")
        # print(f"audio_emb {audio_emb.shape}")
        audio_emb = audio_emb.unsqueeze(0)
        #画图
        # self.plot_embedding_abs_means(text_emb_list,audio_emb,final_emb)

        if labels !=None:
            print(f"only_lables_embs {only_lables_embs} and {only_lables_embs.shape}")
            final_embs =  torch.cat([final_emb, audio_emb,only_lables_embs], dim=1)
        else:
            print(f"推理使用******************")
            final_embs =  torch.cat([final_emb, audio_emb], dim=1)
        self.plot_embedding_abs_means(text_emb_list,audio_emb.squeeze(0),final_emb.squeeze(0))
        final_embs = final_embs.squeeze(0)

        # print(f"门限互注意力后的均值 {final_emb.mean(dim=-1)}")
        new_input_embeds = []
        new_input_embeds.append(final_embs)
        # print(f"new_input_embeds {torch.cat(new_input_embeds).shape}")
        # print(f"new_labels {torch.cat(new_labels).shape}")

        # Truncate sequences to max length as speech features can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            # print(f"cur_len {cur_len}")
            # print(f"cur_new_lables {cur_new_labels.shape}")
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        #
        # print(f"final_input :{new_input_embeds.shape}")
        # print(f"new_lables: {new_labels.shape} and {new_labels}")

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
    '''




    def prepare_inputs_labels_for_speech_and_text2(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        speech, speech_lengths
    ):
        speech_encoder = self.get_speech_encoder()
        if speech_encoder is None or speech is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        speech_features = self.encode_speech(speech, speech_lengths)

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_speech_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_speech = (cur_input_ids == SPEECH_TOKEN_INDEX).sum()
            if num_speech == 0:
                cur_speech_features = speech_features[cur_speech_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_speech_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_speech_idx += 1
                continue

            speech_token_indices = [-1] + torch.where(cur_input_ids == SPEECH_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_nospeech = []
            cur_labels = labels[batch_idx]
            cur_labels_nospeech = []
            for i in range(len(speech_token_indices) - 1):
                cur_input_ids_nospeech.append(cur_input_ids[speech_token_indices[i]+1:speech_token_indices[i+1]])
                cur_labels_nospeech.append(cur_labels[speech_token_indices[i]+1:speech_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_nospeech]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_nospeech))
            cur_input_embeds_no_speech = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_speech + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_speech[i])
                cur_new_labels.append(cur_labels_nospeech[i])
                if i < num_speech:
                    cur_speech_features = speech_features[cur_speech_idx]
                    cur_speech_idx += 1
                    cur_new_input_embeds.append(cur_speech_features)
                    cur_new_labels.append(torch.full((cur_speech_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as speech features can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels