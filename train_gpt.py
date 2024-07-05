import json
import os
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from common.xtts_dataset import XTTSDataset
from models.arch_util import TorchCodeMelSpectrogram
from models.diffusion.model import normalize_tacotron_mel
from models.dvae.dvae import DiscreteVAE
from models.gpt.gpt import TTSModel, load_discrete_vocoder_diffuser, do_spectrogram_diffusion
from text.voice_tokenizer import VoiceBpeTokenizer
from utils import plot_spectrogram_to_numpy, summarize, latest_checkpoint_path, oldest_checkpoint_path
from vocoder2.vocos import Vocos


def warmup(step):
    return float(1)


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
                                  'vocoder2/config.yaml').cuda()

    tts = TTSModel(vocab_size=tokenizer.vocab_size(), **cfg["gpt"])

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

    dataset = XTTSDataset(cfg, tokenizer, is_eval=False)
    eval_dataset = XTTSDataset(cfg, tokenizer, is_eval=True)
    dataloader = DataLoader(dataset=dataset, **cfg['gpt_dataloader'], collate_fn=dataset.collate_fn)

    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False,
                                 collate_fn=eval_dataset.collate_fn)

    total_epochs = cfg['gpt_train']['train_epochs']
    val_freq = cfg['gpt_train']['val_freq']
    save_freq = cfg['gpt_train']['save_freq']

    logs_folder = Path(cfg['gpt_train']['logs_dir'])
    logs_folder.mkdir(exist_ok=True, parents=True)

    optimizer = AdamW(tts.parameters(), lr=cfg['gpt_train']['lr'], betas=(0.9, 0.96), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup)

    epoch = 0
    step = 1
    train_diffusion_steps = 30

    try:
        gpt_model_path = latest_checkpoint_path(f"{logs_folder}", f"TEST_[0-9]*")
        gpt_checkpoint = torch.load(gpt_model_path, map_location="cpu")
        if 'step' in gpt_checkpoint:
            step = gpt_checkpoint['step'] + 1
        if 'epoch' in gpt_checkpoint:
            epoch = gpt_checkpoint['epoch']
        if 'model' in gpt_checkpoint:
            gpt_checkpoint = gpt_checkpoint['model']
        tts.load_state_dict(gpt_checkpoint, strict=False)
        print(f">> GPT weights restored from: {gpt_model_path} at step: {step}")
    except Exception as e:
        print(e)
        try:
            gpt_model_path = "C:\\Users\\User\\PycharmProjects\\new_tortoise\\.models\\autoregressive.pth"
            gpt_checkpoint = torch.load(gpt_model_path, map_location="cpu")
            if 'step' in gpt_checkpoint:
                step = gpt_checkpoint['step'] + 1
            if 'epoch' in gpt_checkpoint:
                epoch = gpt_checkpoint['epoch']
            if 'model' in gpt_checkpoint:
                gpt_checkpoint = gpt_checkpoint['model']
            tts.load_state_dict(gpt_checkpoint, strict=False)
            print(f">> GPT weights restored from: {gpt_model_path} at step: {step}")
        except Exception as e:
            print(e)

    writer = SummaryWriter(log_dir=os.path.join(logs_folder))
    tts.cuda()
    tts.train()

    for epoch in range(epoch, total_epochs + 1):
        for idx, batch in enumerate(dataloader):
            tts.zero_grad()

            batch = format_batch_on_device(dvae, batch)

            padded_mel = batch["padded_mel"].cuda()
            text_input = batch["text_inputs"].cuda()
            text_lens = batch['text_lengths'].cuda()
            padded_quant_mel = batch["audio_codes"].cuda()
            wav_lens = batch["wav_lengths"].cuda()
            mel_refer = batch["mel_refer"].cuda()

            loss_text, loss_mel, mel_logits, latents = tts(padded_mel,
                                                           text_input,
                                                           text_lens,
                                                           padded_quant_mel,
                                                           wav_lens)

            x_start = normalize_tacotron_mel(padded_mel)
            refer = normalize_tacotron_mel(mel_refer)
            t = torch.randint(0, 1000, (x_start.shape[0],), device=tts.device).long().to(tts.device)

            diffusion_loss = tts.diffuser.training_losses(
                model=tts.diffusion,
                x_start=x_start,
                t=t,
                model_kwargs={
                    "latent": latents.transpose(1, 2),
                    "refer": refer
                },
            )["loss"].mean()

            unused_params = []
            extraneous_addition = 0
            for p in unused_params:
                extraneous_addition = extraneous_addition + p.mean()
            diffusion_loss = diffusion_loss + 0 * extraneous_addition

            loss = loss_text * 0.01 + loss_mel * 1.0 + diffusion_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(tts.parameters(), max_norm=1)
            grad_norm = get_grad_norm(tts)
            optimizer.step()
            lr = scheduler.get_last_lr()[0]

            if step % 5 == 0:
                print(f'[Epoch: {epoch}, '
                      f'Iteration: {idx + 1}/{len(dataloader)} - '
                      f'{100. * (idx + 1) / len(dataloader):.2f}%]')

            print(f"Step: {step}, Total loss: {loss :.5f}, "
                  f"Diffusion loss: {diffusion_loss.item() :.5f}, "
                  f"Text loss: {loss_text.item() * 0.01 :.5f}, "
                  f"Mel loss: {loss_mel.item() * 1.0 :.5f}, Grad norm: {grad_norm :.5f}, Lr: {lr}")

            if step % save_freq == 0:
                print("Saving...")
                data = {
                    'step': step,
                    'epoch': epoch,
                    'model': tts.state_dict(),
                }
                torch.save(data, f'{logs_folder}/TEST_{step}.pth')
                print(f'Saved GPT model to {logs_folder}/TEST_{step}.pth')
                keep_ckpts = cfg['gpt_train']['keep_ckpts']
                old_ckpt = oldest_checkpoint_path(f"{logs_folder}", f"TEST_[0-9]*", preserved=keep_ckpts)
                if os.path.exists(old_ckpt):
                    print(f"Removed old GPT model {old_ckpt}")
                    os.remove(old_ckpt)

                tts.eval()
                print("Evaluating...")
                for idx_, batch_ in enumerate(eval_dataloader):
                    if idx_ == 1:
                        break
                    batch = format_batch_on_device(dvae, batch_)
                    mel_refer = batch["mel_refer"].cuda()
                    padded_mel = batch["padded_mel"].cuda()
                    text_tokens = batch["text_inputs"].cuda()

                    tts.post_init_gpt2_config(kv_cache=False, use_deepspeed=False)

                    refer = normalize_tacotron_mel(mel_refer)

                    with torch.no_grad():
                        codes = tts.inference_speech(mel_refer, text_tokens,
                                                     top_k=50,
                                                     top_p=.5,
                                                     temperature=.5,
                                                     do_sample=True,
                                                     num_beams=1,
                                                     num_return_sequences=1,
                                                     length_penalty=1.0,
                                                     repetition_penalty=2.0,
                                                     output_attentions=False,
                                                     output_hidden_states=True)

                        dvae_mel = dvae.decode(codes[:, :-1].to("cpu"))[0]
                        dvae_audio = vocos.decode(dvae_mel.cuda())

                        text_len = torch.tensor([text_tokens.shape[-1]], device=text_tokens.device)
                        expected_output_len = torch.tensor(
                            [codes.shape[-1] * tts.mel_length_compression], device=text_tokens.device)

                        latent = tts(mel_refer,
                                     text_tokens,
                                     text_len,
                                     codes,
                                     expected_output_len, return_latent=True).transpose(1, 2)

                        diffuser = load_discrete_vocoder_diffuser(desired_diffusion_steps=train_diffusion_steps)
                        mel = do_spectrogram_diffusion(tts.diffusion, diffuser,
                                                       latent, refer, temperature=1.0)
                        gen = vocos.decode(mel)

                        audio_dict = {
                            f"train/gen_audio_{idx_}": gen,
                            f"train/gen_dvae_audio_{idx_}": dvae_audio,
                        }

                        image_dict = {
                            f"train/gt_mel_{idx_}": plot_spectrogram_to_numpy(
                                padded_mel[0, :, :].detach().unsqueeze(-1).cpu()),
                            f"train/gen_mel_{idx_}": plot_spectrogram_to_numpy(
                                mel[0, :, :].detach().unsqueeze(-1).cpu()),
                            f"train/gen_dave_mel_{idx_}": plot_spectrogram_to_numpy(
                                dvae_mel[0, :, :].detach().unsqueeze(-1).cpu()),
                        }
                        summarize(
                            writer=writer,
                            audios=audio_dict,
                            global_step=step,
                            images=image_dict,
                        )
                try:
                    del tts.gpt.wte
                except Exception as e:
                    print(f'1. ---> {e}')
                try:
                    del tts.inference_model
                except Exception as e:
                    print(f'2. ---> {e}')
                tts.train()
                print("Training...")

            step += 1
            scheduler.step()
        epoch += 1


if __name__ == '__main__':
    train()
