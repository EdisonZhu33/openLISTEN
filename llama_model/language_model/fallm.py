
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
    LlamaConfig

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput


from  FAcode_llms.build_llm import FAcodeSpeechLlamaForCausalLM,FAcodeSpeechConfig
from llama_model.speech_generator.builder import build_speech_generator
from llama_model.speech_generator.generation import GenerationWithCTC

IGNORE_INDEX = -100
class FAcodeSpeech2SConfig(LlamaConfig):
    model_type = "omni_speech2s_llama"


class FAcode2SLlamaForCausalLM(FAcodeSpeechLlamaForCausalLM, GenerationWithCTC):
    config_class = FAcodeSpeech2SConfig

    def __init__(self, config):
        super().__init__(config)


        # self.sex_num = 5
        # self.llm_embed_dim = config.hidden_size
        # self.sex_embeddings = nn.Embedding(self.sex_num, self.llm_embed_dim)


        # Initialize weights and apply final processing
        self.post_init()
        if hasattr(config, "speech_generator_type"):
            self.speech_generator = build_speech_generator(config)

    def initialize_speech_generator(self, model_args):
        self.config.speech_generator_type = getattr(model_args, 'speech_generator_type', 'ctc')
        self.config.ctc_decoder_config = getattr(model_args, 'ctc_decoder_config', '(4,4096,32,11008)')
        self.config.ctc_upsample_factor = getattr(model_args, 'ctc_upsample_factor', 1)
        self.config.ctc_loss_weight = getattr(model_args, 'ctc_loss_weight', 1.0)
        self.config.unit_vocab_size = getattr(model_args, 'unit_vocab_size', 1000)
        self.tune_speech_generator_only = getattr(model_args, 'tune_speech_generator_only', True)
        if getattr(self, "speech_generator", None) is None:
            self.speech_generator = build_speech_generator(self.config)

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
            tgt_units: Optional[torch.LongTensor] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,

            sex = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

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
            # print(labels)
        if self.training:
            if self.tune_speech_generator_only:
                with torch.no_grad():
                    llama_output = super(FAcodeSpeechLlamaForCausalLM, self).forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds,
                        labels=labels,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        output_hidden_states=True,
                        return_dict=return_dict
                    )

                # if sex == 0:
                #     sex_ids = torch.Tensor([i for i in range(self.prompt_num)]).long()
                #
                # else:
                #     sex_ids = torch.Tensor([i+1 for i in range(self.prompt_num)]).long()



                # speech_g = llama_output['hidden_states'][-1]
                # batch_sizes = speech_g.size(0)
                # print("batch_sizes",batch_sizes)
                # # sex_ids = sex_ids.repeat(1,1).to(speech_g.device)
                # sex_ids = sex_ids.unsqueeze(0).to(speech_g.device)
                # sex_emb = self.sex_embeddings(sex_ids)
                # sex_emb = sex_emb.to(speech_g.device)
                # speech_g = torch.cat((sex_emb,speech_g), dim=1)
                # if labels != None:
                #     # sex_label = torch.full_like(input=sex_ids, fill_value=IGNORE_INDEX, device=labels.device,dtype=torch.long)
                #     sex_label = torch.Tensor([IGNORE_INDEX for i in range(self.sex_num)]).long()
                #     sex_label = sex_label.unsqueeze(0).to(speech_g.device)
                #
                #     print(f"sex_label {sex_label.shape}")
                #     print(f"sex_label = {sex_label} and labels = {sex_label.shape}")
                #
                #     labels = torch.cat([sex_label,labels],dim=1)
                #     labels = labels.long()
                # print("**************************输入到sppech_generator的输入lables", labels)
                # loss = self.speech_generator(speech_g, labels, tgt_units)
                loss = self.speech_generator(llama_output['hidden_states'][-1], labels, tgt_units)

            else:
                llama_output = super(FAcodeSpeechLlamaForCausalLM, self).forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=True,
                    return_dict=return_dict
                )
                lm_loss = llama_output.loss

    
                # if sex == 0:
                #     sex_ids = torch.Tensor([i for i in range(self.sex_num)]).long()
                #
                # else:
                #     sex_ids = torch.Tensor([i + 1 for i in range(self.sex_num)]).long()

                # 将sex编码填充进去
                # speech_g = llama_output['hidden_states'][-1]
                # batch_sizes = speech_g.size(0)
                # print("batch_sizes",batch_sizes)
                # sex_ids = sex_ids.repeat(batch_sizes, 1).to(speech_g.device)
                # sex_emb = self.sex_embeddings(sex_ids)
                # speech_g = torch.cat((sex_emb, speech_g), dim=1)
                # if labels != None:
                #     sex_label = torch.full_like(input=sex_ids, fill_value=IGNORE_INDEX, device=labels.device)
                #     print(f"sex_label = {sex_label} and labels = {sex_label.shape}")
                #     sex_label = sex_label.long()
                #     labels = torch.cat([sex_label, labels], dim=1)
                #     labels = labels.long()
                #     print("**************************输入到sppech_generator的输入lables",labels)
                # ctc_loss = self.speech_generator(speech_g, labels, tgt_units)
                ctc_loss = self.speech_generator(llama_output['hidden_states'][-1], labels, tgt_units)
                loss = lm_loss + ctc_loss * self.config.ctc_loss_weight
        else:
            llama_output = super(FAcodeSpeechLlamaForCausalLM, self).forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict
            )
            loss = llama_output.loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=llama_output.logits,
            past_key_values=llama_output.past_key_values,
            hidden_states=llama_output.hidden_states,
            attentions=llama_output.attentions
        )

    @torch.no_grad()
    def generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            speech: Optional[torch.Tensor] = None,
            speech_lengths: Optional[torch.Tensor] = None,
            streaming_unit_gen=False,
            sex = None,
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
            inputs_embeds = self.get_model().embed_tokens(inputs)
            # inputs_embeds = self.input_emds(inputs)

        outputs = GenerationWithCTC.generate(
            self,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            return_dict_in_generate=True,
            streaming_unit_gen=streaming_unit_gen,
            **kwargs
        )

        hidden_states = outputs['hidden_states']
        hidden_states = torch.cat(
            [hidden_states[0][-1][:, -1:, :]] + [hidden_states[i][-1] for i in range(1, len(hidden_states))], dim=1)
        ctc_pred = self.speech_generator.predict(hidden_states.squeeze(0))
        # ctc_pred = self.speech_generator.predict_sex(hidden_states.squeeze(0))

        return outputs.sequences, ctc_pred

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


AutoConfig.register("omni_speech2s_llama", FAcodeSpeech2SConfig)
AutoModelForCausalLM.register(FAcodeSpeech2SConfig, FAcode2SLlamaForCausalLM)
