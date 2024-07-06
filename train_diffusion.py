import json
import os
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from common.xtts_dataset import XTTSDataset
from models.arch_util import TorchCodeMelSpectrogram
from models.diffusion.model import Diffusion_Tts, denormalize_tacotron_mel, normalize_tacotron_mel
from models.diffusion.utils.diffusion import SpacedDiffusion, space_timesteps, get_named_beta_schedule
from models.dvae.dvae import DiscreteVAE
from models.gpt.gpt import TTSModel
from text.voice_tokenizer import VoiceBpeTokenizer
from utils import plot_spectrogram_to_numpy, summarize, latest_checkpoint_path, oldest_checkpoint_path
from vocoder2.vocos import Vocos


def load_discrete_vocoder_diffuser(trained_diffusion_steps=1000, desired_diffusion_steps=50, cond_free=True,
                                   cond_free_k=2.):
    return SpacedDiffusion(use_timesteps=space_timesteps(trained_diffusion_steps, [desired_diffusion_steps]),
                           model_mean_type='epsilon',
                           model_var_type='learned_range', loss_type='mse',
                           betas=get_named_beta_schedule('linear', trained_diffusion_steps),
                           conditioning_free=cond_free, conditioning_free_k=cond_free_k, sampler='dpm++2m')


def do_spectrogram_diffusion(diffusion_model, diffuser, latents, refer, temperature=1., verbose=True):
    """
    Uses the specified diffusion model to convert discrete codes into a spectrogram.
    """
    with torch.no_grad():
        output_seq_len = latents.shape[
                             2] * 4  # This diffusion model converts from 22kHz spectrogram codes to a 24kHz spectrogram signal.
        output_shape = (latents.shape[0], 100, output_seq_len)

        noise = torch.randn(output_shape, device=latents.device) * temperature
        mel = diffuser.p_sample_loop(diffusion_model, output_shape, noise=noise,
                                     model_kwargs={
                                         "latent": latents,
                                         "refer": refer
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
            pass
    total_norm = total_norm ** (1. / 2)
    return total_norm


torch_mel_spectrogram_vocos = TorchCodeMelSpectrogram(n_mel_channels=100, sampling_rate=24000)


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
    mel_refer = torch_mel_spectrogram_vocos(conditioning_reshaped)
    batch["mel_refer"] = mel_refer

    # compute codes using DVAE
    dvae_wav = batch["wav"]
    dvae_mel_spec = torch_mel_spectrogram_vocos(dvae_wav)
    batch["padded_mel"] = dvae_mel_spec
    codes = dvae.get_codebook_indices(dvae_mel_spec)
    batch["audio_codes"] = codes

    # delete useless batch tensors
    del batch["padded_text"]
    del batch["conditioning"]
    return batch


def train(cfg_path='configs/tts_config.json'):
    print("Training...")
    cfg = json.load(open(cfg_path))
    tokenizer = VoiceBpeTokenizer()

    vocos = Vocos.from_pretrained('pretrained/pytorch_model.bin',
                                  'vocoder2/config.yaml')

    tts = TTSModel(vocab_size=tokenizer.vocab_size(), **cfg["gpt"])
    gpt_model_path = latest_checkpoint_path(cfg['gpt_train']['logs_dir'], f"GPT_[0-9]*")
    gpt_checkpoint = torch.load(gpt_model_path, map_location="cpu")
    tts.load_state_dict(gpt_checkpoint, strict=False)
    tts.eval().cuda()
    print(f">> GPT weights restored from: {gpt_model_path}")

    dvae = DiscreteVAE(channels=cfg["gpt"]["mel_bin"],
                       num_tokens=8192,
                       hidden_dim=512,
                       num_resnet_blocks=3,
                       codebook_dim=512,
                       num_layers=2,
                       positional_dims=1,
                       kernel_size=3,
                       use_transposed_convs=False).eval()
    dvae_path = latest_checkpoint_path(cfg['vae_train']['logs_dir'], f"dvae_[0-9]*")
    dvae_checkpoint = torch.load(dvae_path, map_location=torch.device("cpu"))
    dvae.load_state_dict(dvae_checkpoint['model'], strict=True)

    diffusion = Diffusion_Tts(num_layers=6).cuda()

    trained_diffusion_steps = 1000
    desired_diffusion_steps = 1000

    diffuser = SpacedDiffusion(
        use_timesteps=space_timesteps(trained_diffusion_steps, [desired_diffusion_steps]),
        model_mean_type='epsilon',
        model_var_type='learned_range', loss_type='mse',
        betas=get_named_beta_schedule('linear', trained_diffusion_steps),
        conditioning_free=False, conditioning_free_k=2.)

    dataset = XTTSDataset(cfg, tokenizer, is_eval=False)
    eval_dataset = XTTSDataset(cfg, tokenizer, is_eval=True)
    dataloader = DataLoader(dataset=dataset, **cfg['gpt_dataloader'], collate_fn=dataset.collate_fn)

    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False,
                                 collate_fn=eval_dataset.collate_fn)

    total_epochs = cfg['gpt_train']['train_epochs']
    print_freq = cfg['gpt_train']['print_freq']
    save_freq = cfg['gpt_train']['save_freq']
    scalar_freq = cfg['gpt_train']['scalar_freq']

    logs_folder = Path('logs/diffusion')
    logs_folder.mkdir(exist_ok=True, parents=True)

    optimizer = AdamW(diffusion.parameters(), lr=5e-05, betas=(0.9, 0.96), weight_decay=0.01)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=1, eta_min=0)

    epoch = 0
    step = 1
    train_diffusion_steps = 30

    try:
        model_path = latest_checkpoint_path(f"{logs_folder}", f"DIFF_[0-9]*")
        checkpoint = torch.load(model_path, map_location="cpu")
        if 'step' in checkpoint:
            step = checkpoint['step'] + 1
        if 'epoch' in checkpoint:
            epoch = checkpoint['epoch']
        if 'model' in checkpoint:
            checkpoint = checkpoint['model']
        diffusion.load_state_dict(checkpoint, strict=True)
        print(f">> Diffusion weights restored from: {model_path} at step: {step}")
    except Exception as e:
        print(e)

    # ema_model = _get_target_encoder(diffusion).to("cuda")

    writer = SummaryWriter(log_dir=os.path.join(logs_folder))
    diffusion.train()

    for epoch in range(epoch, total_epochs + 1):
        for idx, batch in enumerate(dataloader):
            diffusion.zero_grad()

            batch = format_batch_on_device(dvae, batch)

            padded_mel = batch["padded_mel"].cuda()
            text_input = batch["text_inputs"].cuda()
            padded_quant_mel = batch["audio_codes"].cuda()
            mel_refer = batch["mel_refer"].cuda()

            latent = tts(mel_refer,
                         text_input,
                         torch.tensor([text_input.shape[-1]], device=text_input.device),
                         padded_quant_mel,
                         torch.tensor([padded_quant_mel.shape[-1] * tts.mel_length_compression]),
                         return_latent=True).transpose(1, 2)

            x_start = normalize_tacotron_mel(padded_mel)
            aligned_conditioning = latent
            conditioning_latent = normalize_tacotron_mel(mel_refer)
            t = torch.randint(0, desired_diffusion_steps, (x_start.shape[0],), device=text_input.device).long().to(
                "cuda")

            loss = diffuser.training_losses(
                model=diffusion,
                x_start=x_start,
                t=t,
                model_kwargs={
                    "latent": aligned_conditioning,
                    "refer": conditioning_latent
                },
            )["loss"].mean()

            loss += loss.item()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=1)
            # grad_norm = get_grad_norm(diffusion)
            optimizer.step()
            lr = scheduler.get_last_lr()[0]

            if step % print_freq == 0:
                print(f'[Epoch: {epoch}, '
                      f'Iteration: {idx + 1}/{len(dataloader)} - '
                      f'{100. * (idx + 1) / len(dataloader):.2f}%]')

            print(f"Step: {step}, Total loss: {loss :.5f}, Grad norm: {grad_norm :.5f}, Lr: {lr}")

            if step % scalar_freq == 0:
                scalar_dict = {"loss": loss, "loss/grad": grad_norm, "lr": lr}
                summarize(
                    writer=writer,
                    global_step=step,
                    scalars=scalar_dict
                )

            if step % save_freq == 0:
                print("Saving...")
                data = {
                    'step': step,
                    'epoch': epoch,
                    'model': diffusion.state_dict(),
                }
                torch.save(data, f'{logs_folder}/DIFF_{step}.pth')
                print(f'Saved Diffusion model to {logs_folder}/DIFF_{step}.pth')
                keep_ckpts = cfg['gpt_train']['keep_ckpts']
                old_ckpt = oldest_checkpoint_path(f"{logs_folder}", f"DIFF_[0-9]*", preserved=keep_ckpts)
                if os.path.exists(old_ckpt):
                    print(f"Removed old Diffusion model {old_ckpt}")
                    os.remove(old_ckpt)

                diffusion.eval()
                print("Evaluating...")
                for idx_, batch_ in enumerate(eval_dataloader):
                    if idx_ == 1:
                        break
                    batch = format_batch_on_device(dvae, batch_)
                    mel_refer = batch["mel_refer"].cuda()
                    padded_mel = batch["padded_mel"].cuda()
                    text_tokens = batch["text_inputs"].cuda()
                    codes = batch["audio_codes"].cuda()
                    gt_wav = batch["wav"]

                    with torch.no_grad():
                        latent = tts(mel_refer,
                                     text_tokens,
                                     torch.tensor([text_tokens.shape[-1]], device=text_tokens.device),
                                     codes,
                                     torch.tensor([codes.shape[-1] * tts.mel_length_compression]),
                                     return_latent=True).transpose(1, 2)
                    refer_padded = normalize_tacotron_mel(mel_refer)
                    with torch.no_grad():
                        eval_diffuser = load_discrete_vocoder_diffuser(desired_diffusion_steps=train_diffusion_steps)
                        mel = do_spectrogram_diffusion(diffusion, eval_diffuser, latent, refer_padded, temperature=0.8)
                        mel = mel.detach().cpu()

                        gen = vocos.decode(mel)

                        audio_dict = {
                            f"train/gen_audio_{idx_}": gen,
                            f"train/gt_audio_{idx_}": gt_wav,
                        }

                        image_dict = {
                            f"train/gt_mel_{idx_}": plot_spectrogram_to_numpy(
                                padded_mel[0, :, :].detach().unsqueeze(-1).cpu()),
                            f"train/gen_mel_{idx_}": plot_spectrogram_to_numpy(
                                mel[0, :, :].detach().unsqueeze(-1).cpu()),
                        }
                        summarize(
                            writer=writer,
                            audios=audio_dict,
                            global_step=step,
                            images=image_dict,
                        )
                diffusion.train()
                print("Training...")

            step += 1
            # scheduler.step()
            scheduler.step(epoch + idx / len(dataloader))
        epoch += 1


if __name__ == '__main__':
    train()
