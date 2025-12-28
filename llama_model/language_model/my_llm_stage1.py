
from typing import List, Optional, Tuple, Union


from model_file.constants import IGNORE_INDEX, SPEECH_TOKEN_INDEX

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput


from model_file.speech_encoder import WhisperWrappedEncoder,build_speech_encoder
from llama_model.speech_projector.speech_projector import EncoderProjectorConcat,EncoderProjectorConcat_CNN,EncoderProjectorConcat2

class OmniSpeechConfig(LlamaConfig):
    model_type = "omni_speech_llama"


class LlamaomniConfig:
    def __init__(self, config_path):
        # 读取JSON文件
        with open(config_path, 'r') as f:
            config_data = json.load(f)

        # 设置属性
        for key, value in config_data.items():
            setattr(self, key, value)



class SpeechForCausalLM(LlamaForCausalLM):
    config_class = OmniSpeechConfig

    def __init__(self, config):
        super(SpeechForCausalLM,self).__init__(config)


        self.config = config

        self.speech_encode =  WhisperWrappedEncoder.load(self.config)
        self.speech_projector = EncoderProjectorConcat(self.config)
        # self.speech_projector = EncoderProjectorConcat2(self.config)

        self.input_emds = super().get_input_embeddings()

        self.pretraining_tp =  self.config.pretraining_tp
        self.vocab_size =  self.config.vocab_size
        self.lm_head = nn.Linear( self.config.hidden_size,  self.config.vocab_size, bias=False)


        self.post_init()


        self.prompt_finetune = True
        self.add_prompt_before = True
        self.prompt_num = 10
        self.llm_embed_dim = config.hidden_size
        if self.prompt_finetune:
            self.prompt_embeddings = nn.Embedding(self.prompt_num, self.llm_embed_dim)
            self.prompt_ids = torch.Tensor([i for i in range(self.prompt_num)]).long()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        speech: Optional[torch.FloatTensor] = None,
        speech_lengths: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # print("inut_ids", input_ids)
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_speech_and_text(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                speech,
                speech_lengths
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        speech: Optional[torch.Tensor] = None,
        speech_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if speech is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_speech_and_text(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                speech,
                speech_lengths
            )
        else:
            inputs_embeds = self.input_emds(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )


    #在推理的时候对输入进行是否有speech的处理
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        speech = kwargs.pop("speech", None)
        speech_lengths = kwargs.pop("speech_lengths", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if speech is not None:
            inputs['speech'] = speech
            inputs['speech_lengths'] = speech_lengths
        return inputs


    def prepare_inputs_labels_for_speech_and_text(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        speech, speech_lengths
    ):

        if self.speech_encode  is None or speech is None or input_ids.shape[1] == 1:
            # print("speech is NONE!!!!")
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
                cur_input_embeds_1 = self.input_emds(cur_input_ids)
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
            cur_input_embeds = self.input_emds(torch.cat(cur_input_ids_nospeech))
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

        # # 添加prompt embedding
        if self.prompt_finetune:
            # 获取inputs_embeds的第一个维度
            batch_size = new_input_embeds.size(0)

            prompt_ids = self.prompt_ids.repeat(batch_size, 1).to(new_input_embeds.device)  # .to(inputs_embeds.device)
            prompt_embeds = self.prompt_embeddings(
                prompt_ids)  # B, 5, D
            if new_labels != None:
                prompt_label = torch.full(prompt_ids.shape, IGNORE_INDEX,device=new_labels.device)
                prompt_label = prompt_label.long()
            if self.add_prompt_before:
                new_input_embeds = torch.cat((prompt_embeds, new_input_embeds), 1)  # B, (T+5), D
                if new_labels != None:
                    new_labels_padded = torch.cat((prompt_label, new_labels), 1)
                    new_labels = new_labels_padded
        #print(f"之前的lables的是什么？？？？？？？？？？？？？？？？？？？？？？？？？？{new_labels}")
        return None, position_ids, attention_mask, past_key_values, new_input_embeds,new_labels


    #使用whisper编码器对语音进行编码
    def encode_speech(self, speech, speech_lengths):
        speech_encoder_type = self.config.speech_encoder_type

        if "whisper" in speech_encoder_type.lower():
            encoder_outs = self.speech_encode(speech.permute(0, 2, 1))
            speech_lengths = (speech_lengths + 1) // 2
            # print(f"speech_lengths {speech_lengths} **************************")
        else:
            raise ValueError(f'Unknown speech encoder: {self.speech_encode}')
        speech_projector_type = self.config.speech_projector_type

        if speech_projector_type == "linear":
            encoder_outs = self.speech_projector(encoder_outs)
            # print(f"encoder_outs {encoder_outs}")
            speech_lengths = speech_lengths // self.speech_projector.k
        else:
            raise ValueError(f'Unknown speech projector: {speech_projector_type}')
        speech_features = [encoder_outs[i, :speech_lengths[i]] for i in range(len(encoder_outs))]
        return speech_features

    def move_suffix_to_prefix(self,tensor, value):
        """
        Move the suffix after the last occurrence of `value` in the tensor to the prefix.

        Parameters:
        tensor (torch.Tensor): The input tensor.
        value (int): The value to search for.

        Returns:
        torch.Tensor: A new tensor with the suffix moved to the prefix.
        """
        # 找到所有-100的索引
        indices = torch.where(tensor == value)[0]

        # 如果没有找到-100或者-100是最后一个元素，则不需要移动
        if len(indices) == 0 or indices[-1].item() == len(tensor) - 1:
            return tensor


        last_index = indices[-1].item()


        new_tensor = torch.cat((tensor[last_index + 1:], tensor[:last_index + 1]))

        return new_tensor



AutoConfig.register("omni_speech_llama", OmniSpeechConfig)
AutoModelForCausalLM.register(OmniSpeechConfig, SpeechForCausalLM)


if __name__=="__main__":

    import json
    def load_data(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data

    base_modle_path = ""
    model_config_path =""
    config =  OmniSpeechConfig.from_pretrained(model_config_path )

    model = SpeechForCausalLM.from_pretrained( base_modle_path,config=config,
            low_cpu_mem_usage=False)
    print(model)