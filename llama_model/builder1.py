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

import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llama_model import *
from llama_model.speech_encoder.builder import build_speech_encoder
from llama_model.language_model.my_llm_stage1 import SpeechForCausalLM
from llama_model.language_model.omni_speech_llama import OmniSpeechConfig

def load_pretrained_model(model_path, model_base,is_lora=False, s2s=False, load_8bit=False, load_4bit=False, device="cuda", use_flash_attn=False, **kwargs):
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    if s2s:
        model_cls = OmniSpeech2SLlamaForCausalLM
    elif "unified" in model_path.lower() or kwargs.get("model_type") == "omni_speech_llama_unified":
        from .language_model.omni_speech_llama_unified import OmniSpeechLlamaForCausalLM as model_cls
    else:
        model_cls = OmniSpeechLlamaForCausalLM

    # Load OmniSpeech model
    if is_lora:
        assert model_base is not None, "model_base is required for LoRA models."

        lora_cfg_pretrained = OmniSpeechConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        print('Loading OmniSpeech from base model...')
        model = model_cls.from_pretrained(model_base, low_cpu_mem_usage=True, ignore_mismatched_sizes=True, config=lora_cfg_pretrained, **kwargs)
        print('Loading additional OmniSpeech weights...')
        if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
            non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)

        from peft import PeftModel
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)
        print('Merging LoRA weights...')
        model = model.merge_and_unload()
        print('Model is loaded...')
    elif model_base is not None:
        print('Loading OmniSpeech from base model...')
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        cfg_pretrained = AutoConfig.from_pretrained(model_path)
        model = model_cls.from_pretrained(model_base, low_cpu_mem_usage=False, config=cfg_pretrained, **kwargs)
        
        speech_projector_weights = torch.load(os.path.join(model_path, 'speech_projector.bin'), map_location='cpu')
        speech_projector_weights = {k: v.to(torch.float16) for k, v in speech_projector_weights.items()}
        model.load_state_dict(speech_projector_weights, strict=False)
        model = model.to(device=device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = model_cls.from_pretrained(
            model_path,
            low_cpu_mem_usage=False,
            **kwargs
        )
        model = model.to(device=device)

    model.get_model().speech_encoder = build_speech_encoder(model.config)
    model.get_model().speech_encoder.to(device=device, dtype=torch.float16)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, context_len


def create_model(model_path, model_base, use_my=True,is_lora=False, s2s=False, load_8bit=False, load_4bit=False,
                 use_flash_attn=False, **kwargs):
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.bfloat16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    from llama_model.language_model.omni_speech_llama import OmniSpeechConfig
    config = OmniSpeechConfig.from_pretrained("/home/chengz/Code/sft_chatglm/model_config/config.json")

    if use_my:
        # model_cls = SpeechForCausalLM(config)
        model_cls = SpeechForCausalLM
    else:
        # model_cls = OmniSpeech2SLlamaForCausalLM(config) if s2s else OmniSpeechLlamaForCausalLM(config)
        model_cls = OmniSpeech2SLlamaForCausalLM if s2s else OmniSpeechLlamaForCausalLM

    if is_lora:
        if model_base is not None:
            print("已经存在lora路径，融合到模型本体*****************************************************")
            from llama_model.language_model.omni_speech_llama import OmniSpeechConfig
            config = OmniSpeechConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            print('Loading OmniSpeech from base model...')
            base_model = model_cls.from_pretrained(model_path, low_cpu_mem_usage=False, config=config, **kwargs)
            print('Loading additional OmniSpeech weights...')

            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
                non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            base_model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(
                base_model,
                model_base,
                torch_dtype=torch.bfloat16,
            )
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        else:
            print(f"创建新的lora重新学习*******************************************************************")
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            base_model = model_cls.from_pretrained(
                model_path,
                low_cpu_mem_usage=False,
                **kwargs,)
            lora_config = LoraConfig(
                r=64,
                lora_alpha=16,
                target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "up_proj",
                "gate_proj",
                "down_proj",
            ],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            model = get_peft_model(base_model, lora_config)
            model.print_trainable_parameters()


    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right", padding=True, use_fast=False)
        model = model_cls.from_pretrained(
            model_path,
            low_cpu_mem_usage=False,
            **kwargs,

        )
    if s2s:
        model.initialize_speech_generator(model.config)

    if use_my:
        model = model.to(dtype=torch.bfloat16)  # .to(device=device,dtype=torch.bfloat16)
    else:
        model = model.to(dtype=torch.bfloat16)  # .to(device=device,dtype=torch.bfloat16)
        model.get_model().speech_encoder = build_speech_encoder(model.config)
        model.get_model().speech_encoder.to(dtype=torch.bfloat16)  # .to(device=device, dtype=torch.bfloat16)

    for name, param in model.named_parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if 'speech_projector' in name:
            param.requires_grad = True
        if 'prompt_embeddings' in name:
            param.requires_grad = True
        # if 'speech_encode' in name:
        #     param.requires_grad = True

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, context_len
