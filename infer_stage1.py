import argparse
import torch
import os
import json
import whisper

import sys
sys.path.append('Github_openlisten')

from llama_model.language_model.omni_speech_llama import OmniSpeechLlamaForCausalLM,OmniSpeechConfig

from llama_model.speech_encoder.builder import build_speech_encoder
from llama_model.language_model.my_llm_stage1 import SpeechForCausalLM
from llama_model.language_model.fallm import FAcodeSpeechLlamaForCausalLM,FAcodeSpeechConfig,FAcode2SLlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from librosa.filters import mel as librosa_mel_fn
from infer_unstuding import load_custom_state_dict
from llama_model.utils import disable_torch_init

IGNORE_INDEX = -100
SPEECH_TOKEN_INDEX = -200
DEFAULT_SPEECH_TOKEN = "<speech>"


mel_basis = {}
hann_window = {}
def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window  # pylint: disable=global-statement
    if f"{str(fmax)}_{str(y.device)}" not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax) + "_" + str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect"
    )
    y = y.squeeze(1)

    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[str(y.device)],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec



def ctc_postprocess2(tokens, blank):
    _toks = tokens.squeeze(0).tolist()
    deduplicated_toks = [v for i, v in enumerate(_toks) ]
    hyp = [v for v in deduplicated_toks if v != blank]
    hyp = " ".join(list(map(str, hyp)))
    return hyp



def tokenizer_speech_token(prompt, tokenizer, speech_token_index=SPEECH_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<speech>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [speech_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids



def bulid_infer_model(args,device,cro_attn=False):


    model_config_path =""
    config = OmniSpeechConfig.from_pretrained(model_config_path)


    model_cls = OmniSpeech2SLlamaForCausalLM if args.s2s else OmniSpeechLlamaForCausalLM


    print(f"model_cls: {model_cls}")
    model = model_cls.from_pretrained(
        args.model_path,
        config=config,
        low_cpu_mem_usage=False,
        torch_dtype=torch.bfloat16,
        ignore_mismatched_sizes=True
    )
    if args.s2s:
        model.initialize_speech_generator(model.config)

    from safetensors.torch import load_file, save_file
    save_path = f"{args.lora_path}/sg_state.safetensors"
    spech_dict = load_file(save_path)
    for name, param in model.named_parameters():
        if "speech_generator" in name:
            p_m = "base_model.model." + name
            if p_m in spech_dict:
                print(f"{p_m}")
                print(spech_dict[p_m].data.shape)
                param.data = spech_dict[p_m].data
    model.load_state_dict(spech_dict, strict=False)

    if cro_attn:
        cro_path = args.cro_path

        model = load_custom_state_dict(
            model=model,
            state_dict_path=f"{cro_path}/save_state.safetensors",
            target_layers=['speech_projector', 'cross_attn', 'gate_projs'],
            verbose=True
        )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, padding_side="right", padding=True, use_fast=False)


    print(model)
    model.get_model().speech_encoder = build_speech_encoder(model.config)

    model = model.to(device=device,dtype=torch.bfloat16)
    return model,tokenizer


def infer(args):
    import torch
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    disable_torch_init()

    model, tokenizer = bulid_infer_model(args,device,cro_attn=True)
    system = "You are a helpful language and speech assistant. You are able to understand the speech content that the user provides, and assist the user with a variety of tasks using natural language."
    prompt = "<speech>\n Please directly answer the questions in the user's speech."
 
    system_p = f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user:<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant:<|end_header_id|>"

    speech = whisper.load_audio(args.speech_path)
    speech = whisper.pad_or_trim(speech)
    speech_tensor = whisper.log_mel_spectrogram(speech, n_mels=128).permute(1, 0)
    speech_length = torch.LongTensor([speech_tensor.shape[0]])

    prompt_ids = tokenizer_speech_token(system_p,tokenizer, return_tensors='pt')
    prompt_ids = prompt_ids.unsqueeze(0).to(device=device, non_blocking=True)
    speech_tensor = speech_tensor.unsqueeze(0).to(dtype=torch.bfloat16, device=device, non_blocking=True)

    print(f"prompt_ids: {prompt_ids.shape}")
    with torch.inference_mode():



        outputs = model.generate(
            prompt_ids,
            speech=speech_tensor,
            speech_lengths=speech_length,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            pad_token_id=128004,
        )
        output_ids= outputs
        outputs_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print(f"output:\t{outputs_text}")

    # if  args.s2s:
    #     if args.use_flow_matching:
    #         print(f"******************************使用flow-matching进行TTS*********************************")
    #         from llama_model.acoustical_model.ac_flow import MaskedDiffWithXvec
    #         from safetensors.torch import load_file, save_file
    #         from Preprocess.speech2unit import Speech2Unit
    #         from llama_model.flow_model.modules import MelSpec
    #         import torchaudio.compliance.kaldi as kaldi
    #         import onnxruntime
    #         import torchaudio

    #         #加载flow_modle预训练模型
    #         flow_model = MaskedDiffWithXvec()
    #         # save_path = f"{args.flow_path}/sg_state.safetensors"
    #         save_path = f"{args.flow_path}/model.safetensors"
    #         flow_dict = load_file(save_path)
    #         flow_model.load_state_dict(flow_dict, strict=True)

    #         # 使用flow_model进行推理
    #         #准备prompt_speech and speak embedding
    #         target_sample_rate = 22050
    #         ckpt_dir = "/home/chengz/Code/sft_chatglm/models/speech2unit"
    #         s2u = Speech2Unit(
    #             ckpt_dir=ckpt_dir
    #         )
    #         get_mel = MelSpec(
    #                 n_fft=1024,
    #                 hop_length=256,
    #                 win_length=1024,
    #                 n_mel_channels=80,
    #                 target_sample_rate=target_sample_rate,
    #                 mel_spec_type="vocos",
    #             )

    #         prompt_token = s2u(args.prompt_speech)
    #         prompt_tokens = torch.tensor([int(item) for item in prompt_token.split(' ')])
    #         prompt_tokens = prompt_tokens.unsqueeze(0)
    #         print(f"prompt_tokens :{prompt_tokens} and {prompt_tokens.shape}")

    #         # 配置 ONNX 运行时
    #         option = onnxruntime.SessionOptions()
    #         option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    #         option.intra_op_num_threads = 1
    #         providers = ["CPUExecutionProvider"]
    #         ort_session = onnxruntime.InferenceSession(args.onnx_path, sess_options=option, providers=providers)

    #         def single_job2(utt2wav):
    #             # 读取音频文件
    #             audio, sample_rate = torchaudio.load(utt2wav)
    #             if sample_rate != 16000:
    #                 audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)

    #             # 提取特征
    #             feat = kaldi.fbank(audio,
    #                                num_mel_bins=80,
    #                                dither=0,
    #                                sample_frequency=16000)
    #             feat = feat - feat.mean(dim=0, keepdim=True)

    #             # 使用 ONNX 模型生成嵌入
    #             embedding = \
    #             ort_session.run(None, {ort_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[
    #                 0].flatten().tolist()
    #             return embedding

    #         prompt_embedding  = single_job2(args.prompt_speech)
    #         prompt_embedding = torch.tensor(prompt_embedding).unsqueeze(0)
    #         print(f"prompt_embedding {prompt_embedding.shape}")

    #         prompt_audio, source_sample_rate = torchaudio.load(args.prompt_speech)
    #         if source_sample_rate != target_sample_rate:
    #             resampler = torchaudio.transforms.Resample(source_sample_rate, target_sample_rate)
    #             prompt_audio = resampler(prompt_audio)
    #         # prompt_mel = get_mel(prompt_audio)
    #         prompt_mel = mel_spectrogram(y=prompt_audio, n_fft=1024, num_mels=80, sampling_rate=22050, hop_size=256, win_size=1024, fmin=0, fmax=8000, center=False)
    #         prompt_mel = prompt_mel.permute(0,2,1)
    #         prompt_mel = torch.zeros_like(prompt_mel)
    #         # print(f"prompt_mel {prompt_mel.shape}")
    #         llm_tokens = list(map(int, output_units.split()))
    #         # 转换为 PyTorch 的 long 类型张量
    #         llm_tokens = torch.LongTensor(llm_tokens).unsqueeze(0)
    #         # print(f"llm_tokens :{llm_tokens} and {llm_tokens.shape}")


    #         # prompt_tokens = torch.zeros(1, 0, dtype=torch.int32)
    #         # prompt_mel = torch.zeros(1, 0, 80)
    #         # print(f"prompt_mel {prompt_mel.shape}")
    #         flow_mel, flow_cache = flow_model.inference(
    #             token=llm_tokens,
    #             token_len=torch.LongTensor([llm_tokens.shape[-1]]),
    #             prompt_token=prompt_tokens,
    #             prompt_token_len=torch.LongTensor([prompt_tokens.shape[-1]]),
    #             # prompt_token_len=torch.LongTensor([0]),
    #             prompt_feat=prompt_mel,
    #             prompt_feat_len=prompt_mel.shape[1],
    #             embedding = prompt_embedding,
    #             flow_cache=torch.zeros(1, 80, 0, 2)
    #         )
    #         # print(f"flow_mel {flow_mel.shape} and {flow_cache}")

    #         #加载HIFIGAN
    #         from cosyvoice.hifigan.generator import HiFTGenerator
    #         from cosyvoice.hifigan.f0_predictor import ConvRNNF0Predictor
    #         import os
    #         hift = HiFTGenerator(
    #             in_channels=80,
    #             base_channels=512,
    #             nb_harmonics=8,
    #             sampling_rate=22050,
    #             nsf_alpha=0.1,
    #             nsf_sigma=0.003,
    #             nsf_voiced_threshold=10,
    #             upsample_rates=[8, 5,3],
    #             upsample_kernel_sizes=[16, 11,7],
    #             resblock_kernel_sizes=[3, 7, 11],
    #             resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    #             source_resblock_kernel_sizes=[7, 7, 11],
    #             source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    #             lrelu_slope=0.1,
    #             audio_limit=0.99,
    #             f0_predictor=ConvRNNF0Predictor(
    #                 num_class=1,
    #                 in_channels=80,
    #                 cond_channels=512)

    #         )
    #         hift_model_path = "/home/chengz/Code/sft_chatglm/models/CosyVoice2-0.5B/hift.pt"
    #         hift_state_dict = torch.load(hift_model_path)
    #         hift.load_state_dict(hift_state_dict, strict=True)
    #         print(f"prompt_mel {prompt_mel.shape} and flow_mel {flow_mel.shape}")
    #         # tts_speech, tts_source = hift.inference(speech_feat=prompt_mel.permute(0,2,1))
    #         tts_speech, tts_source = hift.inference(speech_feat=flow_mel)
    #         tts_fn = os.path.join(args.flow_wav, 'llm_flow.wav')
    #         torchaudio.save(tts_fn, tts_speech, sample_rate=22050)


    #     else:
  
    #         import os
    #         import random
    #         import torch
    #         import soundfile as sf
    #         import json
    #         from tqdm import tqdm
    #         from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder
    #         from pathlib import Path

    #         def generate_waveform_from_code(args, device,data):


   
    #             with open(args.vocoder_cfg) as f:
    #                 vocoder_cfg = json.load(f)

  
    #             vocoder = CodeHiFiGANVocoder(args.vocoder, vocoder_cfg)

    #             multispkr = vocoder.model.multispkr
    #             if multispkr:

    #                 print("multi-speaker vocoder")
    #                 num_speakers = vocoder_cfg.get("num_speakers", 200)
    #                 assert (
    #                         args.speaker_id < num_speakers
    #                 ), f"invalid --speaker-id ({args.speaker_id}) with total #speakers = {num_speakers}"



    #             Path(args.results_path).mkdir(exist_ok=True, parents=True)

    #             output_units_list = list(map(int, data.split()))

    #             output_units = torch.LongTensor(output_units_list)
    #             print(f"output_units_int :{output_units}")
    #             data = output_units
    #             x = {
    #                 "code": output_units,
    #             }
    #             out_path = "/home/chengz/Code/sft_chatglm/text_json/out.wav"
    #             wav = vocoder(x, True)
    #             sf.write(
    #                 out_path,
    #                 wav.detach().cpu().numpy(),
    #                 16000,
    #             )
    #         generate_waveform_from_code(args, device,output_units)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str,
                        default="")
    parser.add_argument("--lora_path", type=str,
                        default="")
    parser.add_argument("--cro_path", type=str,
                        default="")

    parser.add_argument("--flow_path", type=str, default="")
    parser.add_argument("--tokenizer_path", type=str,default="")
    parser.add_argument("--speech_path", type=str,default="")
    parser.add_argument("--prompt_speech", type=str,default="")
    parser.add_argument("--wavtokenizer_config", type=str,default="")
    parser.add_argument("--wavtokenizer_model_path", type=str,default="")
    parser.add_argument("--onnx_path", type=str, default="")
    parser.add_argument("--llm_out_path", type=str,default="")
    parser.add_argument("--flow_wav", type=str,default="")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_p", type=float, default=0.1)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--input_type", type=str, default="mel")
    parser.add_argument("--mel_size", type=int, default=128)
    parser.add_argument("--s2s", type=bool, default=True)
    parser.add_argument("--is_lora", action="store_true", default=False)
    parser.add_argument("--have_speech", type=bool,default=True)
    parser.add_argument("--use_my", type=bool,default=False)
    parser.add_argument("--use_fa", type=bool,default=False)
    parser.add_argument("--use_flow_matching", type=bool,default=False)


    parser.add_argument('--vocoder', type=str, default="", help="Path to vocoder checkpoint")
    parser.add_argument('--vocoder-cfg', type=str, default="", help="Path to vocoder config file")
    parser.add_argument('--results-path', type=str, default="", help="Path to save generated waveforms")


    args = parser.parse_args()
    infer(args)