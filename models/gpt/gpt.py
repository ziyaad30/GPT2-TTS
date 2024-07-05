import functools
import json
import os
import random
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2Model, GPT2Config, LogitsProcessorList
from common.xtts_dataset import XTTSDataset
from models.arch_util import AttentionBlock, TorchCodeMelSpectrogram
from models.diffusion.diff_utils.diffusion import SpacedDiffusion, space_timesteps, get_named_beta_schedule
from models.diffusion.model import TTS_diffusion, normalize_tacotron_mel, denormalize_tacotron_mel
from models.dvae.dvae import DiscreteVAE
from models.gpt.gpt_inference_model import GPT2InferenceModel
from models.typical_sampling import TypicalLogitsWarper
from text.voice_tokenizer import VoiceBpeTokenizer
from utils import latest_checkpoint_path, oldest_checkpoint_path, plot_spectrogram_to_numpy, summarize
from vocoder2.vocos import Vocos


def null_position_embeddings(range, dim):
    return torch.zeros((range.shape[0], range.shape[1], dim), device=range.device)


def load_discrete_vocoder_diffuser(trained_diffusion_steps=1000, desired_diffusion_steps=50, cond_free=True,
                                   cond_free_k=2.):
    return SpacedDiffusion(use_timesteps=space_timesteps(trained_diffusion_steps, [desired_diffusion_steps]),
                           model_mean_type='epsilon',
                           model_var_type='learned_range', loss_type='mse',
                           betas=get_named_beta_schedule('linear', trained_diffusion_steps),
                           conditioning_free=cond_free, conditioning_free_k=cond_free_k,
                           sampler='dpm++2m')


def do_spectrogram_diffusion(diffusion_model, diffuser, latents, refer, temperature=1., verbose=True):
    """
    Uses the specified diffusion model to convert discrete codes into a spectrogram.
    """
    with torch.no_grad():
        output_seq_len = latents.shape[2] * 4  # This converts from 22kHz spec codes to a 24kHz spec signal.
        output_shape = (latents.shape[0], 100, output_seq_len)

        noise = torch.randn(output_shape, device=latents.device) * temperature
        mel = diffuser.p_sample_loop(diffusion_model, output_shape, noise=noise,
                                     model_kwargs={
                                         "latent": latents,
                                         "refer": refer
                                     },
                                     progress=verbose)
        return denormalize_tacotron_mel(mel)[:, :, :output_seq_len]


class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim, init=.02, relative=False):
        super().__init__()
        # nn.Embedding
        self.emb = nn.Embedding(seq_len, model_dim)
        # Initializing this way is standard for GPT-2
        self.emb.weight.data.normal_(mean=0.0, std=init)
        self.relative = relative
        self.seq_len = seq_len

    def forward(self, x):
        sl = x.shape[1]
        if self.relative:
            start = random.randint(sl, self.seq_len) - sl
            return self.emb(torch.arange(start, start + sl, device=x.device))
        else:
            return self.emb(torch.arange(0, sl, device=x.device))

    def get_fixed_embedding(self, ind, dev):
        return self.emb(torch.tensor([ind], device=dev)).unsqueeze(0)


class ConditioningEncoder(nn.Module):
    def __init__(self,
                 spec_dim,
                 embedding_dim,
                 attn_blocks=6,
                 num_attn_heads=4,
                 do_checkpointing=False,
                 mean=False):
        super().__init__()
        attn = []
        self.init = nn.Conv1d(spec_dim, embedding_dim, kernel_size=1)
        for a in range(attn_blocks):
            attn.append(AttentionBlock(embedding_dim, num_attn_heads, do_checkpoint=do_checkpointing))
        self.attn = nn.Sequential(*attn)
        self.dim = embedding_dim
        self.do_checkpointing = do_checkpointing
        self.mean = mean

    def forward(self, x):
        h = self.init(x)
        h = self.attn(h)
        if self.mean:
            return h.mean(dim=2)
        else:
            return h[:, :, 0]


class TTSModel(nn.Module):
    def __init__(self, vocab_size=255, model_dim=1024, layers=15, heads=16, mel_bin=100, types=1, checkpointing=False):
        super(TTSModel, self).__init__()
        max_mel_seq_len = 608
        max_text_seq_len = 404

        self.max_mel_tokens = 604
        self.max_text_tokens = 402
        self.layers = layers
        self.heads = heads

        self.number_text_tokens = vocab_size
        self.start_text_token = self.number_text_tokens
        self.stop_text_token = 0
        self.start_mel_token = 8192
        self.stop_mel_token = 8193
        self.number_mel_codes = 8194

        self.model_dim = model_dim

        gpt_config = GPT2Config(vocab_size=vocab_size + 1,  # Unused.
                                n_positions=max_mel_seq_len + max_text_seq_len,
                                n_ctx=max_mel_seq_len + max_text_seq_len,
                                n_embd=model_dim,
                                n_layer=layers,
                                n_head=heads,
                                gradient_checkpointing=checkpointing,
                                use_cache=not checkpointing)
        self.gpt = GPT2Model(gpt_config)

        self.diffusion = TTS_diffusion(in_latent_channels=model_dim)

        self.diffuser = SpacedDiffusion(
            use_timesteps=space_timesteps(1000, [1000]),
            model_mean_type='epsilon',
            model_var_type='learned_range', loss_type='mse',
            betas=get_named_beta_schedule('linear', 1000),
            conditioning_free=False, conditioning_free_k=2.)

        self.mel_embedding = nn.Embedding(self.number_mel_codes, model_dim)
        self.mel_pos_embedding = LearnedPositionEmbeddings(max_mel_seq_len, model_dim)
        self.text_pos_embedding = LearnedPositionEmbeddings(max_text_seq_len, model_dim)
        self.mel_length_compression = 1024
        self.conditioning_encoder = ConditioningEncoder(mel_bin, model_dim, num_attn_heads=heads)
        self.text_embedding = nn.Embedding(self.number_text_tokens * types + 1, model_dim)
        self.final_norm = nn.LayerNorm(model_dim)
        self.text_head = nn.Linear(model_dim, self.number_text_tokens * types + 1)
        self.mel_head = nn.Linear(model_dim, self.number_mel_codes)

        self.init_model()

    def init_model(self):
        print("Model init...")
        # Override the built-in positional embeddings
        del self.gpt.wpe
        self.gpt.wpe = functools.partial(null_position_embeddings, dim=self.model_dim)
        # Built-in token embeddings are unused.
        del self.gpt.wte
        # Initialize the embeddings per the GPT-2 scheme
        embeddings = [self.text_embedding]
        for module in embeddings:
            module.weight.data.normal_(mean=0.0, std=.02)

    def build_aligned_inputs_and_targets(self, input, start_token, stop_token):
        inp = F.pad(input, (1, 0), value=start_token)
        tar = F.pad(input, (0, 1), value=stop_token)
        return inp, tar

    @property
    def device(self):
        return next(self.parameters()).device

    def set_mel_padding(self, mel_input_tokens, wav_lengths):
        """
        Given mel tokens that are derived from a padded audio clip and the actual lengths of each batch element in
        that audio clip, reformats the tokens with STOP_MEL_TOKEN in place of the zero padding. This is required
        preformatting to create a working TTS model.
        """
        # Set padding areas within MEL (currently it is coded with the MEL code for <zero>).
        mel_lengths = wav_lengths // self.mel_length_compression
        for b in range(len(mel_lengths)):
            actual_end = mel_lengths[b] + 1
            # Due to the convolutional nature of how these tokens are generated, it would be best if the model predicts
            # a token past the actual last token.
            if actual_end < mel_input_tokens.shape[-1]:
                mel_input_tokens[b, actual_end:] = self.stop_mel_token
        return mel_input_tokens

    def get_conditioning(self, speech_conditioning_input):
        speech_conditioning_input = speech_conditioning_input.unsqueeze(1) if len(
            speech_conditioning_input.shape) == 3 else speech_conditioning_input
        conds = []
        for j in range(speech_conditioning_input.shape[1]):
            conds.append(self.conditioning_encoder(speech_conditioning_input[:, j]))
        conds = torch.stack(conds, dim=1)
        conds = conds.mean(dim=1)
        return conds

    def get_logits(self, conds, text_emb, text_head, mel_emb, mel_head):
        emb = torch.cat([conds, text_emb, mel_emb], dim=1)
        gpt_out = self.gpt(inputs_embeds=emb, return_dict=True, output_attentions=False)

        enc = gpt_out.last_hidden_state[:, 1:]  # The first logit is tied to the speech_conditioning_input
        enc = self.final_norm(enc)

        latent_logits = enc[:, -mel_emb.shape[1]:]

        text_logits = enc[:, :text_emb.shape[1]]
        text_logits = text_head(text_logits)
        text_logits = text_logits.permute(0, 2, 1)

        mel_logits = enc[:, -mel_emb.shape[1]:]
        mel_logits = mel_head(mel_logits)
        mel_logits = mel_logits.permute(0, 2, 1)

        return text_logits, mel_logits, latent_logits

    def forward(self, speech_conditioning_latent, text_inputs, text_lengths, mel_codes, wav_lengths,
                return_latent=False):

        max_text_len = text_lengths.max()
        text_inputs = text_inputs[:, :max_text_len]
        text_inputs = F.pad(text_inputs, (0, 1), value=self.stop_text_token)

        max_mel_len = wav_lengths.max() // self.mel_length_compression
        mel_codes = mel_codes[:, :max_mel_len]
        mel_codes = self.set_mel_padding(mel_codes, wav_lengths)
        mel_codes = F.pad(mel_codes, (0, 1), value=self.stop_mel_token)

        speech_conditioning_latent = self.get_conditioning(speech_conditioning_latent)
        conds = speech_conditioning_latent.unsqueeze(1)

        text_inputs, text_targets = self.build_aligned_inputs_and_targets(text_inputs, self.start_text_token,
                                                                          self.stop_text_token)
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)
        mel_codes, mel_targets = self.build_aligned_inputs_and_targets(mel_codes, self.start_mel_token,
                                                                       self.stop_mel_token)
        mel_inp = mel_codes
        mel_emb = self.mel_embedding(mel_inp)
        mel_emb = mel_emb + self.mel_pos_embedding(mel_codes)

        text_logits, mel_logits, latent_logits = self.get_logits(conds, text_emb, self.text_head, mel_emb,
                                                                 self.mel_head)
        sub = -2

        loss_text = F.cross_entropy(text_logits, text_targets.long())
        loss_mel = F.cross_entropy(mel_logits, mel_targets.long())

        latent = latent_logits[:, :sub]
        if return_latent:
            return latent

        return loss_text.mean(), loss_mel.mean(), mel_logits, latent

    def post_init_gpt2_config(self, use_deepspeed=False, kv_cache=False, half=False):
        seq_length = self.max_mel_tokens + self.max_text_tokens + 2
        gpt_config = GPT2Config(
            vocab_size=self.max_mel_tokens,
            n_positions=seq_length,
            n_ctx=seq_length,
            n_embd=self.model_dim,
            n_layer=self.layers,
            n_head=self.heads,
            gradient_checkpointing=False,
            use_cache=True,
        )
        self.inference_model = GPT2InferenceModel(
            gpt_config,
            self.gpt,
            self.mel_pos_embedding,
            self.mel_embedding,
            self.final_norm,
            self.mel_head,
            kv_cache=kv_cache,
        )
        if use_deepspeed and half and torch.cuda.is_available():
            import deepspeed
            self.ds_engine = deepspeed.init_inference(model=self.inference_model,
                                                      mp_size=1,
                                                      replace_with_kernel_inject=True,
                                                      dtype=torch.float16)
            self.inference_model = self.ds_engine.module.eval()
        elif use_deepspeed and torch.cuda.is_available():
            import deepspeed
            self.ds_engine = deepspeed.init_inference(model=self.inference_model,
                                                      mp_size=1,
                                                      replace_with_kernel_inject=True,
                                                      dtype=torch.float32)
            self.inference_model = self.ds_engine.module.eval()
        else:
            self.inference_model = self.inference_model.eval()

        self.gpt.wte = self.mel_embedding

    def inference_speech(self, speech_conditioning_latent, text_inputs, input_tokens=None, num_return_sequences=1,
                         max_generate_length=None, typical_sampling=False, typical_mass=.9, **hf_generate_kwargs):

        text_inputs = F.pad(text_inputs, (0, 1), value=self.stop_text_token)
        text_inputs, _ = self.build_aligned_inputs_and_targets(text_inputs, self.start_text_token, self.stop_text_token)
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)

        speech_conditioning_latent = self.get_conditioning(speech_conditioning_latent)
        conds = speech_conditioning_latent.unsqueeze(1)

        emb = torch.cat([conds, text_emb], dim=1)
        self.inference_model.store_mel_emb(emb)

        fake_inputs = torch.full((emb.shape[0], conds.shape[1] + emb.shape[1],), fill_value=1, dtype=torch.long,
                                 device=text_inputs.device)
        fake_inputs[:, -1] = self.start_mel_token
        trunc_index = fake_inputs.shape[1]
        if input_tokens is None:
            inputs = fake_inputs
        else:
            assert num_return_sequences % input_tokens.shape[
                0] == 0, "The number of return sequences must be divisible by the number of input sequences"
            fake_inputs = fake_inputs.repeat(num_return_sequences, 1)
            input_tokens = input_tokens.repeat(num_return_sequences // input_tokens.shape[0], 1)
            inputs = torch.cat([fake_inputs, input_tokens], dim=1)

        logits_processor = LogitsProcessorList(
            [TypicalLogitsWarper(mass=typical_mass)]) if typical_sampling else LogitsProcessorList()
        max_length = trunc_index + self.max_mel_tokens - 1 if max_generate_length is None else trunc_index + max_generate_length
        gen = self.inference_model.generate(inputs, bos_token_id=self.start_mel_token, pad_token_id=self.stop_mel_token,
                                            eos_token_id=self.stop_mel_token,
                                            max_length=max_length, logits_processor=logits_processor,
                                            num_return_sequences=num_return_sequences, **hf_generate_kwargs)
        return gen[:, trunc_index:]
