import copy
import json
import os
from pathlib import Path

import torch
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from common.xtts_dataset import XTTSDataset
from models.arch_util import TorchCodeMelSpectrogram
from models.diffusion.aa_model import denormalize_tacotron_mel, AA_diffusion, normalize_tacotron_mel
from models.dvae.dvae import DiscreteVAE

from models.diffusion.utils.diffusion import SpacedDiffusion, space_timesteps, get_named_beta_schedule
from models.gpt.gpt import TTSModel
from text.voice_tokenizer import VoiceBpeTokenizer
from utils import latest_checkpoint_path, summarize, plot_spectrogram_to_numpy, oldest_checkpoint_path
from vocoder2.feature_extractors import MelSpectrogramFeatures
from vocoder2.vocos import Vocos


def load_discrete_vocoder_diffuser(trained_diffusion_steps=1000, desired_diffusion_steps=50, cond_free=True,
                                   cond_free_k=2.):
    return SpacedDiffusion(use_timesteps=space_timesteps(trained_diffusion_steps, [desired_diffusion_steps]),
                           model_mean_type='epsilon',
                           model_var_type='learned_range', loss_type='mse',
                           betas=get_named_beta_schedule('linear', trained_diffusion_steps),
                           conditioning_free=cond_free, conditioning_free_k=cond_free_k,
                           sampler='dpm++2m')


def do_spectrogram_diffusion(diffusion_model, diffuser, latents, conditioning_latents, temperature=1., verbose=True):
    """
    Uses the specified diffusion model to convert discrete codes into a spectrogram.
    """
    with torch.no_grad():
        output_seq_len = latents.shape[2] * 4  # This diff model converts from 22kHz spec codes to a 24kHz spec signal.
        output_shape = (latents.shape[0], 100, output_seq_len)

        noise = torch.randn(output_shape, device=latents.device) * temperature
        mel = diffuser.p_sample_loop(diffusion_model, output_shape, noise=noise,
                                     model_kwargs={
                                         "hint": latents,
                                         "refer": conditioning_latents
                                     },
                                     progress=verbose)
        return denormalize_tacotron_mel(mel)[:, :, :output_seq_len]


def get_grad_norm(model):
    total_norm = 0
    for name, p in model.named_parameters():
        try:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        except (Exception,):
            print(name)
            pass
    total_norm = total_norm ** (1. / 2)
    return total_norm


def warmup(step):
    if step < 1:
        return float(step / 1)
    else:
        return 1


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


torch_mel_spectrogram_vocos = TorchCodeMelSpectrogram(n_mel_channels=100, sampling_rate=24000)
mel_extractor = MelSpectrogramFeatures()


@torch.no_grad()  # torch no grad to avoid gradients from the pre-processing and DVAE codes extraction
def format_batch_on_device(dvae, batch):
    """Compute spectrograms on the device."""
    batch["text_lengths"] = batch["text_lengths"]
    batch["wav_lengths"] = batch["wav_lengths"]
    batch["text_inputs"] = batch["padded_text"]
    # compute conditioning mel specs
    # transform waves from torch.Size([B, num_cond_samples, 1, T] to torch.Size([B * num_cond_samples, 1, T]
    # because if is faster than iterating the tensor
    B, num_cond_samples, C, T = batch["conditioning"].size()
    conditioning_reshaped = batch["conditioning"].view(B * num_cond_samples, C, T)
    mel_refer = mel_extractor(conditioning_reshaped).squeeze(1)
    batch["mel_refer"] = mel_refer

    # compute codes using DVAE
    dvae_wav = batch["wav"]
    dvae_mel_spec = mel_extractor(dvae_wav).squeeze(1)
    codes = dvae.get_codebook_indices(dvae_mel_spec)
    batch["audio_codes"] = codes
    batch["padded_mel"] = dvae_mel_spec

    # delete useless batch tensors
    del batch["padded_text"]
    del batch["conditioning"]
    return batch


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def _get_target_encoder(model):
    target_encoder = copy.deepcopy(model)
    set_requires_grad(target_encoder, False)
    for p in target_encoder.parameters():
        p.DO_NOT_TRAIN = True
    return target_encoder


class Trainer(object):
    def __init__(self, cfg_path='configs/tts_config.json'):
        self.tokenizer = VoiceBpeTokenizer()
        self.vocos = Vocos.from_pretrained('pretrained/pytorch_model.bin',
                                           'vocoder2/config.yaml')
        self.cfg = json.load(open(cfg_path))

        self.trained_diffusion_steps = 1000
        self.desired_diffusion_steps = 1000
        cond_free_k = 2.

        self.sample_rate = self.cfg['vae_train']['sample_rate']
        self.n_mels = self.cfg['vae_train']['n_mels']

        self.dvae = DiscreteVAE(channels=self.n_mels,
                                num_tokens=8192,
                                hidden_dim=512,
                                num_resnet_blocks=3,
                                codebook_dim=512,
                                num_layers=2,
                                positional_dims=1,
                                kernel_size=3,
                                use_transposed_convs=False).eval()
        dvae_path = latest_checkpoint_path(self.cfg['vae_train']['logs_dir'], f"dvae_[0-9]*")
        dvae_checkpoint = torch.load(dvae_path, map_location=torch.device("cpu"))
        self.dvae.load_state_dict(dvae_checkpoint['model'], strict=True)
        print(">> DVAE weights restored from:", dvae_path)

        self.tts = TTSModel(vocab_size=self.tokenizer.vocab_size())
        gpt_model_path = latest_checkpoint_path(self.cfg['gpt_train']['logs_dir'], f"GPT_[0-9]*")
        gpt_checkpoint = torch.load(gpt_model_path, map_location="cpu")
        if 'model' in gpt_checkpoint:
            gpt_checkpoint = gpt_checkpoint['model']
        self.tts.load_state_dict(gpt_checkpoint, strict=False)
        print(">> GPT weights restored from:", gpt_model_path)
        self.tts.eval().cuda()

        self.mel_length_compression = self.tts.mel_length_compression

        self.diffuser = SpacedDiffusion(
            use_timesteps=space_timesteps(self.trained_diffusion_steps, [self.desired_diffusion_steps]),
            model_mean_type='epsilon',
            model_var_type='learned_range', loss_type='mse',
            betas=get_named_beta_schedule('linear', self.trained_diffusion_steps),
            conditioning_free=False, conditioning_free_k=cond_free_k)

        self.diffusion = AA_diffusion().cuda()
        print("model params:", count_parameters(self.diffusion))

        self.dataset = XTTSDataset(self.cfg, self.tokenizer, is_eval=False)
        self.eval_dataset = XTTSDataset(self.cfg, self.tokenizer, is_eval=True)

        self.dataloader = DataLoader(dataset=self.dataset, batch_size=self.cfg['diff_dataloader']['batch_size'],
                                     collate_fn=self.dataset.collate_fn,
                                     drop_last=self.cfg['diff_dataloader']['drop_last'],
                                     num_workers=self.cfg['diff_dataloader']['num_workers'],
                                     pin_memory=self.cfg['diff_dataloader']['pin_memory'],
                                     shuffle=self.cfg['diff_dataloader']['shuffle'])

        self.eval_dataloader = DataLoader(self.eval_dataset, batch_size=1, shuffle=False, num_workers=0,
                                          pin_memory=False, collate_fn=self.eval_dataset.collate_fn)

        self.total_epochs = self.cfg['diff_train']['train_epochs']
        self.val_freq = self.cfg['diff_train']['val_freq']
        self.save_freq = self.cfg['diff_train']['save_freq']

        self.logs_folder = Path(self.cfg['diff_train']['logs_dir'])
        self.logs_folder.mkdir(exist_ok=True, parents=True)

        self.optimizer = AdamW(self.diffusion.parameters(), lr=self.cfg['diff_train']['lr'], betas=(0.9, 0.96),
                               weight_decay=0.01)
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=warmup)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=1, T_mult=1, eta_min=0)

        self.ema_model = _get_target_encoder(self.diffusion).to("cuda")

        self.epoch = 0
        self.step = 1

        self.load()

        self.writer = SummaryWriter(log_dir=os.path.join(self.logs_folder))

    def load(self):
        try:
            model_path = latest_checkpoint_path(f"{self.logs_folder}", f"DIFF_[0-9]*")
            checkpoint = torch.load(model_path, map_location="cpu")
            if 'step' in checkpoint:
                self.step = checkpoint['step'] + 1
            if 'epoch' in checkpoint:
                self.epoch = checkpoint['epoch']
            if 'model' in checkpoint:
                checkpoint = checkpoint['model']
            self.diffusion.load_state_dict(checkpoint, strict=False)
            print(f">> Diffusion weights restored from: {model_path} at step: {self.step}")

        except Exception as e:
            print(e)
            self.ema_model = _get_target_encoder(self.diffusion).to("cuda")

    def save(self):
        data = {
            'step': self.step,
            'epoch': self.epoch,
            'model': self.diffusion.state_dict(),
        }
        torch.save(data, f'{self.logs_folder}/DIFF_{self.step}.pth')
        print(f'Saved Diffusion model to {self.logs_folder}/DIFF_{self.step}.pth')
        keep_ckpts = self.cfg['diff_train']['keep_ckpts']
        old_ckpt = oldest_checkpoint_path(f"{self.logs_folder}", f"DIFF_[0-9]*", preserved=keep_ckpts)
        if os.path.exists(old_ckpt):
            print(f"Removed old GPT model {old_ckpt}")
            os.remove(old_ckpt)

    def train(self):
        self.ema_model.train()

        for self.epoch in range(self.epoch, self.total_epochs + 1):
            for idx, batch in enumerate(self.dataloader):
                total_loss = 0.

                batch = format_batch_on_device(self.dvae, batch)
                mel_refer = batch["mel_refer"].cuda()
                padded_mel = batch["padded_mel"].cuda()
                text_input = batch["text_inputs"].cuda()
                padded_quant_mel = batch["audio_codes"].cuda()

                with torch.no_grad():
                    latent = self.tts(mel_refer, text_input,
                                      torch.tensor([text_input.shape[-1]], device="cuda"),
                                      padded_quant_mel,
                                      torch.tensor(
                                          [padded_quant_mel.shape[-1] * self.mel_length_compression],
                                          device="cuda"),
                                      return_latent=True).transpose(1, 2)
                x_start = normalize_tacotron_mel(padded_mel)
                aligned_conditioning = latent
                conditioning_latent = normalize_tacotron_mel(mel_refer)
                t = torch.randint(0, self.desired_diffusion_steps, (x_start.shape[0],), device="cuda").long().to("cuda")

                loss = self.diffuser.training_losses(
                    model=self.diffusion,
                    x_start=x_start,
                    t=t,
                    model_kwargs={
                        "hint": aligned_conditioning,
                        "refer": conditioning_latent
                    },
                )["loss"].mean()

                unused_params = []
                model = self.diffusion
                unused_params.extend(list(model.refer_model.blocks.parameters()))
                unused_params.extend(list(model.refer_model.out.parameters()))
                unused_params.extend(list(model.refer_model.hint_converter.parameters()))
                unused_params.extend(list(model.refer_enc.visual.proj))
                extraneous_addition = 0
                for p in unused_params:
                    extraneous_addition = extraneous_addition + p.mean()
                loss = loss + 0 * extraneous_addition
                total_loss += loss.item()
                loss.backward()

                grad_norm = get_grad_norm(self.diffusion)
                torch.nn.utils.clip_grad_norm_(self.diffusion.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                lr = self.scheduler.get_last_lr()[0]

                if self.step % 5 == 0:
                    print(f'[Epoch: {self.epoch}, '
                          f'Iteration: {idx + 1}/{len(self.dataloader)} - '
                          f'{100. * (idx + 1) / len(self.dataloader):.2f}%]')

                    print(f"Step: {self.step}, loss: {total_loss}, "
                          f"loss/grad: {grad_norm}, lr: {lr}")
                    scalar_dict = {"loss": total_loss, "loss/grad": grad_norm, "lr": self.scheduler.get_last_lr()[0]}
                    summarize(
                        writer=self.writer,
                        global_step=self.step,
                        scalars=scalar_dict
                    )

                if self.step % self.save_freq == 0:
                    self.ema_model.eval()
                    for idx_, batch_ in enumerate(self.eval_dataloader):
                        if idx_ == 2:
                            break
                        batch = format_batch_on_device(self.dvae, batch_)
                        mel_refer = batch["mel_refer"].cuda()
                        padded_mel = batch["padded_mel"].cuda()
                        text_input = batch["text_inputs"].cuda()
                        padded_quant_mel = batch["audio_codes"].cuda()
                        with torch.no_grad():
                            latent = self.tts(mel_refer, text_input,
                                              torch.tensor([text_input.shape[-1]], device="cuda"),
                                              padded_quant_mel,
                                              torch.tensor(
                                                  [padded_quant_mel.shape[-1] * self.mel_length_compression],
                                                  device="cuda"),
                                              return_latent=True).transpose(1, 2)
                        refer_padded = normalize_tacotron_mel(mel_refer)
                        with torch.no_grad():
                            diffuser = load_discrete_vocoder_diffuser(desired_diffusion_steps=60)
                            mel = do_spectrogram_diffusion(self.diffusion, diffuser,
                                                           latent, refer_padded, temperature=1.0)
                            mel = mel.detach().cpu()
                            gen = self.vocos.decode(mel)
                            audio_dict = {
                                f"gen/audio_{idx_}": gen,
                            }
                            image_dict = {
                                f"gt/mel_{idx_}": plot_spectrogram_to_numpy(
                                    padded_mel[0, :, :].detach().unsqueeze(-1).cpu()),
                                f"gen/mel_{idx_}": plot_spectrogram_to_numpy(mel[0, :, :].detach().unsqueeze(-1).cpu()),
                            }
                            summarize(
                                writer=self.writer,
                                audios=audio_dict,
                                global_step=self.step,
                                images=image_dict,
                            )
                    self.save()
                    self.ema_model.train()

                self.step += 1
                self.scheduler.step(self.epoch + idx / len(self.dataloader))
            self.epoch += 1


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
