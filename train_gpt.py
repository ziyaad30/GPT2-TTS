import copy
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
from models.dvae.xtts_dvae import DiscreteVAE
# from models.dvae.dvae import DiscreteVAE
from models.gpt.gpt import TTSModel
from text.voice_tokenizer import VoiceBpeTokenizer
from utils import plot_spectrogram_to_numpy, summarize, latest_checkpoint_path, oldest_checkpoint_path
from vocoder2.feature_extractors import MelSpectrogramFeatures
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


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def _get_target_encoder(model):
    target_encoder = copy.deepcopy(model)
    set_requires_grad(target_encoder, False)
    for p in target_encoder.parameters():
        p.DO_NOT_TRAIN = True
    return target_encoder


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
    batch["padded_mel"] = dvae_mel_spec
    codes = dvae.get_codebook_indices(dvae_mel_spec)
    batch["audio_codes"] = codes

    # delete useless batch tensors
    del batch["padded_text"]
    del batch["conditioning"]
    return batch


class Trainer(object):
    def __init__(self, cfg_path='configs/tts_config.json'):
        print("Training...")
        self.cfg = json.load(open(cfg_path))
        tokenizer = VoiceBpeTokenizer()

        self.vocos = Vocos.from_pretrained('pretrained/pytorch_model.bin',
                                           'vocoder2/config.yaml').cuda()

        self.tts = TTSModel(vocab_size=tokenizer.vocab_size(), **self.cfg["gpt"]).cuda()

        self.dvae = DiscreteVAE(channels=self.cfg["gpt"]["mel_bin"],
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

        dataset = XTTSDataset(self.cfg, tokenizer, is_eval=False)
        eval_dataset = XTTSDataset(self.cfg, tokenizer, is_eval=True)
        self.dataloader = DataLoader(dataset=dataset, **self.cfg['gpt_dataloader'], collate_fn=dataset.collate_fn)

        self.eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False,
                                          collate_fn=eval_dataset.collate_fn)

        self.total_epochs = self.cfg['gpt_train']['train_epochs']
        self.print_freq = self.cfg['gpt_train']['print_freq']
        self.save_freq = self.cfg['gpt_train']['save_freq']
        self.scalar_freq = self.cfg['gpt_train']['scalar_freq']

        self.logs_folder = Path(self.cfg['gpt_train']['logs_dir'])
        self.logs_folder.mkdir(exist_ok=True, parents=True)

        lr = self.cfg['gpt_train']['lr']

        self.optimizer = AdamW(self.tts.parameters(), lr=lr, betas=(0.9, 0.96), weight_decay=0.01)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=1, T_mult=1, eta_min=0)

        self.epoch = 0
        self.step = 1

        try:
            gpt_model_path = latest_checkpoint_path(f"{self.logs_folder}", f"GPT_[0-9]*")
            gpt_checkpoint = torch.load(gpt_model_path, map_location="cpu")
            if 'step' in gpt_checkpoint:
                self.step = gpt_checkpoint['step'] + 1
            if 'epoch' in gpt_checkpoint:
                self.epoch = gpt_checkpoint['epoch']
            if 'model' in gpt_checkpoint:
                gpt_checkpoint = gpt_checkpoint['model']
            self.tts.load_state_dict(gpt_checkpoint, strict=False)
            print(f">> GPT weights restored from: {gpt_model_path} at step: {self.step}")
        except Exception as e:
            print(e)
            try:
                gpt_model_path = "C:\\Users\\User\\PycharmProjects\\new_tortoise\\.models\\autoregressive.pth"
                gpt_checkpoint = torch.load(gpt_model_path, map_location="cpu")
                if 'step' in gpt_checkpoint:
                    self.step = gpt_checkpoint['step'] + 1
                if 'epoch' in gpt_checkpoint:
                    self.epoch = gpt_checkpoint['epoch']
                if 'model' in gpt_checkpoint:
                    gpt_checkpoint = gpt_checkpoint['model']
                self.tts.load_state_dict(gpt_checkpoint, strict=False)
                print(f">> GPT weights restored from: {gpt_model_path} at step: {self.step}")
            except Exception as e:
                print(e)

        self.ema_model = _get_target_encoder(self.tts)

        self.writer = SummaryWriter(log_dir=os.path.join(self.logs_folder))
        self.ema_model.cuda()
        self.ema_model.train()

    def train(self):
        for self.epoch in range(self.epoch, self.total_epochs + 1):
            for idx, batch in enumerate(self.dataloader):
                batch = format_batch_on_device(self.dvae, batch)

                padded_mel = batch["padded_mel"].cuda()
                text_input = batch["text_inputs"].cuda()
                text_lens = batch['text_lengths'].cuda()
                padded_quant_mel = batch["audio_codes"].cuda()
                wav_lens = batch["wav_lengths"].cuda()

                loss_text, loss_mel, mel_logits, latents = self.tts(padded_mel,
                                                                    text_input,
                                                                    text_lens,
                                                                    padded_quant_mel,
                                                                    wav_lens)

                loss = loss_text * 0.01 + loss_mel * 1.0  # + diffusion_loss
                loss.backward()

                grad_norm = get_grad_norm(self.tts.gpt)
                torch.nn.utils.clip_grad_norm_(self.tts.parameters(), max_norm=1)

                self.optimizer.step()
                self.optimizer.zero_grad()
                lr = self.scheduler.get_last_lr()[0]

                if self.step % self.print_freq == 0:
                    print(f'[Epoch: {self.epoch}, '
                          f'Iteration: {idx + 1}/{len(self.dataloader)} - '
                          f'{100. * (idx + 1) / len(self.dataloader):.2f}%]')

                print(f"Step: {self.step}, Total loss: {loss :.5f}, "
                      f"Text loss: {loss_text.item() * 0.01 :.5f}, "
                      f"Mel loss: {loss_mel.item() * 1.0 :.5f}, Grad norm: {grad_norm :.5f}, Lr: {lr}")

                if self.step % self.scalar_freq == 0:
                    scalar_dict = {
                        "train/loss_mel": loss_mel * 1.0,
                        "train/loss_text": loss_text * 0.01,
                        "train/total_loss": loss,
                        "train/grad_norm": grad_norm,
                        "train/lr": self.scheduler.get_last_lr()[0]
                    }
                    summarize(
                        writer=self.writer,
                        global_step=self.step,
                        scalars=scalar_dict
                    )

                if self.step % self.save_freq == 0:
                    print("Saving...")
                    data = {
                        'step': self.step,
                        'epoch': self.epoch,
                        'model': self.tts.state_dict(),
                    }
                    torch.save(data, f'{self.logs_folder}/GPT_{self.step}.pth')
                    print(f'Saved GPT model to {self.logs_folder}/GPT_{self.step}.pth')
                    keep_ckpts = self.cfg['gpt_train']['keep_ckpts']
                    old_ckpt = oldest_checkpoint_path(f"{self.logs_folder}", f"GPT_[0-9]*", preserved=keep_ckpts)
                    if os.path.exists(old_ckpt):
                        print(f"Removed old GPT model {old_ckpt}")
                        os.remove(old_ckpt)

                    self.ema_model.eval()
                    print("Evaluating...")
                    for idx_, batch_ in enumerate(self.eval_dataloader):
                        if idx_ == 1:
                            break
                        batch = format_batch_on_device(self.dvae, batch_)
                        mel_refer = batch["mel_refer"].cuda()
                        padded_mel = batch["padded_mel"].cuda()
                        text_tokens = batch["text_inputs"].cuda()
                        gt_wav = batch["wav"]

                        self.tts.post_init_gpt2_config(kv_cache=False, use_deepspeed=False)

                        with torch.no_grad():
                            codes = self.tts.inference_speech(mel_refer, text_tokens,
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

                            dvae_mel = self.dvae.decode(codes[:, :-1].to("cpu"))[0]
                            dvae_audio = self.vocos.decode(dvae_mel.cuda())

                            audio_dict = {
                                f"train/gen_dvae_audio_{idx_}": dvae_audio,
                            }

                            image_dict = {
                                f"train/gt_mel_{idx_}": plot_spectrogram_to_numpy(
                                    padded_mel[0, :, :].detach().unsqueeze(-1).cpu()),
                                f"train/gen_dave_mel_{idx_}": plot_spectrogram_to_numpy(
                                    dvae_mel[0, :, :].detach().unsqueeze(-1).cpu()),
                            }
                            summarize(
                                writer=self.writer,
                                audios=audio_dict,
                                global_step=self.step,
                                images=image_dict,
                            )

                            audio_dict_gt = {
                                f"train/gt_audio_{idx_}": gt_wav,
                            }
                            summarize(
                                writer=self.writer,
                                audios=audio_dict_gt,
                                global_step=self.step,
                                audio_sampling_rate=24000
                            )
                    try:
                        del self.tts.gpt.wte
                    except Exception as e:
                        print(f'1. ---> {e}')
                    try:
                        del self.tts.inference_model
                    except Exception as e:
                        print(f'2. ---> {e}')
                    self.ema_model.train()
                    print("Training...")

                self.step += 1
                # scheduler.step()
                self.scheduler.step(self.epoch + idx / len(self.dataloader))
            self.epoch += 1


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
