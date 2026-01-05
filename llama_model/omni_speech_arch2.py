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

from .speech_encoder.builder import build_speech_encoder
from .speech_projector.builder import build_speech_projector
IGNORE_INDEX = -100
SPEECH_TOKEN_INDEX = -200
DEFAULT_SPEECH_TOKEN = "<speech>"

import torch


from matplotlib import transforms  # 需要显式导入transforms模块


import torch
import numpy as np


# 示例用法（请确保text_emb_list, audio_emb, final_embs在您的代码中已正确计算）：
# plot_embedding_abs_means(text_emb_list, audio_emb, final_embs)










class OmniSpeechMetaModel:

    def __init__(self, config):
        super(OmniSpeechMetaModel, self).__init__(config)

        if hasattr(config, "speech_encoder"):
            self.speech_encoder = build_speech_encoder(config)
            self.speech_projector = build_speech_projector(config)




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
            #speech_encoder  = speech_encoder.to(dtype=torch.bfloat16)
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

    # def get_cross_attn(self):
    #     return self.get_model().cross_attn

    # def get_gates(self):
    #     return self.get_model().gate_projs

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




      

    def calculate_lable_length(self,tensor):
        if tensor is None:
            return 0
        last_index = 0
        for i in range(len(tensor[0])):
            if tensor[0][i]!=IGNORE_INDEX:
                # print(f"tensor : {tensor[0][i]} and {i}")
                last_index = i
                break

        return len(tensor[0]) - last_index



   
    #对audio_features 做为主特征进行交叉注意力机制
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
        # print(f"text_emb_list: {len(text_emb_list)}  and {text_emb_list[0].shape} and {text_emb_list[1].shape}")
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


        
        # #如果要使用Cro-Atention + Gate机制融合多模态特征，请取消注释以下代码块
        # #交叉注意力（文本作为Query）和门控

        # final_emb = self.fuse_modalities(
        #     audio_emb.unsqueeze(0),
        #     text_emb.unsqueeze(0)
        # )
        # # print(f"final_emb {final_emb.shape}")
        # # print(f"audio_emb {audio_emb.shape}")
        # audio_emb = audio_emb.unsqueeze(0)
        # if labels !=None:
        #     final_embs =  torch.cat([text_emb.unsqueeze(0),final_emb,only_lables_embs], dim=1)
        # else:
        #     print(f"推理路径~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        #     final_embs =  torch.cat([text_emb.unsqueeze(0),final_emb], dim=1)
        # # self.plot_embedding_abs_means(text_emb_list,audio_emb.squeeze(0),final_emb.squeeze(0))
        # final_embs = final_embs.squeeze(0)

        # # print(f"门限互注意力后的均值 {final_emb.mean(dim=-1)}")
        # new_input_embeds = []
        # new_input_embeds.append(final_embs)
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