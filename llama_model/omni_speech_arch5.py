# Q-Former Ablation Experiment Version
# Adopted from omni_speech_arch.py - Replacing GCA with Q-Former for ablation study
# 
# Key differences from GCA version:
# - Uses Blip2QFormerModel from transformers for cross-modal fusion
# - Fixed output length K=32 (bottleneck compression)
# - 1-layer Q-Former with qformer_dim=768 for fair comparison with 1-layer GCA
#
# Copyright 2023 Haotian Liu (Original LLaVA)
# Modified for Q-Former ablation experiment

from abc import ABC, abstractmethod

import torch
from torch import nn
import torch.nn.functional as F

from transformers import Blip2QFormerConfig, Blip2QFormerModel

from .speech_encoder.builder import build_speech_encoder
from .speech_projector.builder import build_speech_projector

IGNORE_INDEX = -100
SPEECH_TOKEN_INDEX = -200
DEFAULT_SPEECH_TOKEN = "<speech>"


class SpeechQFormerHF(nn.Module):
    """
    Q-Former module for speech-to-LLM feature bridging.
    Uses HuggingFace's Blip2QFormerModel as the backbone.
    
    Key design choices for ablation:
    - num_query_tokens=32 (BLIP-2 default)
    - qformer_dim=768 (lightweight, saves VRAM)
    - num_layers=1 (fair comparison with 1-layer GCA)
    """
    def __init__(
        self,
        llm_dim: int,             # LLM hidden_size (e.g., 4096)
        speech_dim: int,          # speech_projector output dim (usually = llm_dim)
        num_query_tokens: int = 32,
        num_layers: int = 1,
        num_heads: int = 8,
        qformer_dim: int = 768,   # Smaller dim for efficiency
        dropout: float = 0.0,
        cross_attention_frequency: int = 1,
    ):
        super().__init__()
        self.num_query_tokens = num_query_tokens
        self.qformer_dim = qformer_dim
        self.llm_dim = llm_dim
        
        # Project speech features to qformer_dim if dimensions don't match
        self.speech_in = nn.Linear(speech_dim, qformer_dim, bias=False) if speech_dim != qformer_dim else nn.Identity()
        
        # Q-Former configuration
        cfg = Blip2QFormerConfig(
            hidden_size=qformer_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=4 * qformer_dim,
            encoder_hidden_size=qformer_dim,  # KV dimension for cross-attention
            cross_attention_frequency=cross_attention_frequency,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            use_qformer_text_input=False,  # Don't use BERT token embedding
        )
        self.qformer = Blip2QFormerModel(cfg)
        
        # Learnable query tokens: [1, K, qformer_dim]
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, qformer_dim) * 0.02)
        
        # Project qformer output back to llm_dim
        self.out_proj = nn.Linear(qformer_dim, llm_dim, bias=False) if qformer_dim != llm_dim else nn.Identity()
        
        # LayerNorm to stabilize output scale (match LLM embedding scale)
        self.out_norm = nn.LayerNorm(llm_dim)
        
        # ========== Fix for DeepSpeed gradient issue ==========
        # 1. Ensure all Q-Former attention/FFN parameters are trainable
        for param in self.qformer.parameters():
            param.requires_grad = True
        
        # 2. Freeze unused internal embeddings (we pass query_embeds directly, 
        #    so word_embeddings/position_embeddings are not used in forward pass)
        #    This prevents DeepSpeed from expecting gradients for unused params
        if hasattr(self.qformer, 'embeddings'):
            for param in self.qformer.embeddings.parameters():
                param.requires_grad = False
        
        # 3. Initialize out_proj with small weights to avoid exploding outputs
        if isinstance(self.out_proj, nn.Linear):
            nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.01)
        # ======================================================
    
    def forward(self, speech_tokens: torch.Tensor, speech_attention_mask: torch.Tensor = None):
        """
        Args:
            speech_tokens: [B, S, speech_dim] - Speech features from projector
            speech_attention_mask: [B, S] - Optional attention mask (1=valid, 0=pad)
        
        Returns:
            [B, K, llm_dim] - Compressed speech representation (K=32 tokens)
        """
        B = speech_tokens.size(0)
        
        # Project speech to qformer_dim
        enc = self.speech_in(speech_tokens)  # [B, S, qformer_dim]
        
        # Expand query tokens for batch
        q = self.query_tokens.expand(B, -1, -1)  # [B, K, qformer_dim]
        
        # Q-Former forward: queries attend to speech features
        out = self.qformer(
            query_embeds=q,
            encoder_hidden_states=enc,
            encoder_attention_mask=speech_attention_mask,
            return_dict=True,
        ).last_hidden_state  # [B, K, qformer_dim]
        
        # Project back to LLM dimension and normalize
        out = self.out_proj(out)  # [B, K, llm_dim]
        out = self.out_norm(out)  # Normalize to match LLM embedding scale
        return out


class OmniSpeechMetaModel:
    """
    Meta model class for OmniSpeech with Q-Former fusion.
    Replaces GCA (CrossAttention + Gate) with Q-Former for ablation study.
    """
    
    def __init__(self, config):
        super(OmniSpeechMetaModel, self).__init__(config)
        
        if hasattr(config, "speech_encoder"):
            self.speech_encoder = build_speech_encoder(config)
            self.speech_projector = build_speech_projector(config)
        
        # Q-Former module (replaces cross_attn + gate_projs)
        self.qformer_fusion = SpeechQFormerHF(
            llm_dim=config.hidden_size,
            speech_dim=config.hidden_size,
            num_query_tokens=32,
            num_layers=1,       # 1-layer for fair comparison with 1-layer GCA
            num_heads=8,
            qformer_dim=768,    
            dropout=0.0,
        )
    
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
                self.speech_encoder = speech_encoder
        
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
    """
    Abstract class for OmniSpeech causal language model with Q-Former fusion.
    """
    
    @abstractmethod
    def get_model(self):
        pass
    
    def get_speech_encoder(self):
        return self.get_model().get_speech_encoder()
    
    def get_speech_projector(self):
        return self.get_model().speech_projector
    
    def get_qformer_fusion(self):
        """Get the Q-Former module (replaces get_cross_attn and get_gates)"""
        return self.get_model().qformer_fusion
    
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
    
    def fuse_modalities_qformer(self, speech_emb: torch.Tensor) -> torch.Tensor:
        """
        Q-Former based fusion (replaces fuse_modalities).
        
        Args:
            speech_emb: [B, S, D] - Speech embeddings from encoder/projector
        
        Returns:
            [B, K, D] - Compressed speech representation (K=32 fixed)
        
        Note: Unlike GCA which outputs [B, S, D], Q-Former compresses to fixed K tokens.
        """
        qformer = self.get_qformer_fusion()
        return qformer(speech_emb)
    
    def calculate_lable_length(self, tensor):
        if tensor is None:
            return 0
        last_index = 0
        for i in range(len(tensor[0])):
            if tensor[0][i] != IGNORE_INDEX:
                last_index = i
                break
        return len(tensor[0]) - last_index
    
    def prepare_inputs_labels_for_speech_and_text(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        speech, speech_lengths
    ):
        """
        Prepare inputs for LLM with Q-Former fusion.
        
        Key difference from GCA version:
        - GCA: final_embs = [text_emb, fused_audio(S tokens), labels_emb]
        - Q-Former: final_embs = [text_emb, qformer_output(32 tokens), labels_emb]
        
        This results in shorter sequences, which may benefit efficiency
        but could lose fine-grained speech information.
        """
        speech_encoder = self.get_speech_encoder()
        if speech_encoder is None or speech is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        
        speech_features = self.encode_speech(speech, speech_lengths)
        
        # Handle None cases
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
        
        # Remove padding using attention_mask
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
                    # Q-Former compresses variable-length speech features [S, D] -> fixed [K, D]
                    qformer_out = self.fuse_modalities_qformer(cur_speech_features.unsqueeze(0)).squeeze(0)
                    cur_new_input_embeds.append(qformer_out)
                    cur_new_labels.append(
                        torch.full(
                            (qformer_out.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )
            
            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
        
        # Truncate sequences to max length
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            if _labels is not None:
                new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
        
        # Combine and pad
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)
        
        new_input_embeds_padded = []
        new_labels_padded = None
        if _labels is not None:
            new_labels_padded = torch.full(
                (batch_size, max_len),
                IGNORE_INDEX,
                dtype=new_labels[0].dtype,
                device=new_labels[0].device,
            )
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        
        for i, cur_new_embed in enumerate(new_input_embeds):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    if _labels is not None:
                        cur_new_labels = new_labels[i]
                        if cur_new_labels.shape[0] != cur_len:
                            raise RuntimeError(
                                f"Length mismatch after Q-Former fusion: embeds={cur_len}, labels={cur_new_labels.shape[0]}"
                            )
                        new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    if _labels is not None:
                        cur_new_labels = new_labels[i]
                        if cur_new_labels.shape[0] != cur_len:
                            raise RuntimeError(
                                f"Length mismatch after Q-Former fusion: embeds={cur_len}, labels={cur_new_labels.shape[0]}"
                            )
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
