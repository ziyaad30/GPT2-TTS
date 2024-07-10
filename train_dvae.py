import json
import os
from pathlib import Path

import torch
from torch.optim import AdamW
# from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from common.dvae_dataset import DvaeMelDataset
# from models.dvae.xtts_dvae import DiscreteVAE
from models.dvae.dvae import DiscreteVAE
from utils import plot_spectrogram_to_numpy, summarize, oldest_checkpoint_path, latest_checkpoint_path


def get_grad_norm(model):
    total_norm = 0
    for name, p in model.named_parameters():
        try:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        except:
            print(name)
    total_norm = total_norm ** (1. / 2)
    return total_norm


def warmup(step):
    return float(1)


class Trainer(object):
    def __init__(self, cfg_path='configs/tts_config.json'):
        self.cfg = json.load(open(cfg_path))

        self.logs_folder = Path(self.cfg['vae_train']['logs_dir'])
        self.logs_folder.mkdir(exist_ok=True, parents=True)

        self.dvae = DiscreteVAE(channels=self.cfg["gpt"]["mel_bin"],
                                num_tokens=8192,
                                hidden_dim=512,
                                num_resnet_blocks=3,
                                codebook_dim=512,
                                num_layers=2,
                                positional_dims=1,
                                kernel_size=3,
                                use_transposed_convs=False)

        self.dataset = DvaeMelDataset(self.cfg)
        self.dataloader = DataLoader(self.dataset, **self.cfg['vae_dataloader'])
        self.lr = self.cfg['vae_train']['lr']
        self.optimizer = AdamW(self.dvae.parameters(), lr=self.lr, betas=(0.9, 0.9999), weight_decay=0.01)
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=warmup)
        # self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=1, T_mult=1, eta_min=0)
        self.train_epochs = self.cfg['vae_train']['train_epochs']
        self.eval_interval = self.cfg['vae_train']['eval_interval']
        self.save_freq = self.cfg['vae_train']['save_freq']
        self.log_interval = self.cfg['vae_train']['log_interval']
        self.step = 1
        self.epoch = 0
        self.writer = SummaryWriter(log_dir=os.path.join(self.logs_folder))

        try:
            dvae_model_path = latest_checkpoint_path(f"{self.logs_folder}", f"dvae_[0-9]*")
            # dvae_model_path = "C:\\Users\\User\\PycharmProjects\\new_tortoise\\.models\\dvae.pth"
            dvae_checkpoint = torch.load(dvae_model_path, map_location="cpu")
            if 'step' in dvae_checkpoint:
                self.step = dvae_checkpoint['step'] + 1
            if 'epoch' in dvae_checkpoint:
                self.epoch = dvae_checkpoint['epoch']
            if 'model' in dvae_checkpoint:
                dvae_checkpoint = dvae_checkpoint['model']
            self.dvae.load_state_dict(dvae_checkpoint, strict=False)
            print(f">> DVAE weights restored from: {dvae_model_path} at step: {self.step}")
        except:
            pass

    def train(self):
        self.dvae.cuda()
        self.dvae.train()

        for self.epoch in range(self.epoch, self.train_epochs + 1):
            for idx, batch in enumerate(self.dataloader):
                total_loss = 0.
                self.dvae.zero_grad()
                mel = batch
                mel = mel.to("cuda").squeeze(1)

                recon_loss, commitment_loss, mel_recon = self.dvae(mel)
                recon_loss = torch.mean(recon_loss)
                loss = recon_loss + 0.25 * commitment_loss
                total_loss += loss.item()

                loss.backward()
                grad_norm = get_grad_norm(self.dvae)

                self.optimizer.step()
                # lr = self.scheduler.get_last_lr()[0]
                # self.optimizer.zero_grad()

                if self.step % self.log_interval == 0:
                    print(f'[Epoch: {self.epoch}, '
                          f'Iteration: {idx + 1}/{len(self.dataloader)} - {100. * (idx + 1) / len(self.dataloader):.2f}%]')
                    print(f"step: {self.step}, total_loss: {total_loss}, "
                          f"loss_mel: {recon_loss}, "
                          f"loss_commitment: {commitment_loss}, "
                          f"grad_norm: {grad_norm}"
                          # f"lr: {lr}"
                          )

                if self.step % self.eval_interval == 0:
                    self.dvae.eval()
                    with torch.no_grad():
                        mel_recon_ema = self.dvae.infer(mel)[0]
                    scalar_dict = {"loss": total_loss, "loss_mel": recon_loss, "loss_commitment": commitment_loss,
                                   "loss/grad": grad_norm}
                    image_dict = {
                        "all/spec": plot_spectrogram_to_numpy(mel[0, :, :].detach().unsqueeze(-1).cpu()),
                        "all/spec_pred": plot_spectrogram_to_numpy(mel_recon[0, :, :].detach().unsqueeze(-1).cpu()),
                        "all/spec_pred_ema": plot_spectrogram_to_numpy(
                            mel_recon_ema[0, :, :].detach().unsqueeze(-1).cpu()),
                    }
                    summarize(
                        writer=self.writer,
                        global_step=self.step,
                        images=image_dict,
                        scalars=scalar_dict
                    )
                    self.dvae.train()

                if self.step % self.save_freq == 0:
                    print("Saving...")
                    keep_ckpts = self.cfg['vae_train']['keep_ckpts']

                    data = {
                        'step': self.step,
                        'epoch': self.epoch,
                        'model': self.dvae.state_dict(),
                    }
                    torch.save(data, f'{self.logs_folder}/dvae_{self.step}.pth')

                    old_ckpt = oldest_checkpoint_path(f"{self.logs_folder}", f"dvae_[0-9]*", preserved=keep_ckpts)
                    if os.path.exists(old_ckpt):
                        print(f"Removed old GPT model {old_ckpt}")
                        os.remove(old_ckpt)

                self.step += 1
                # self.scheduler.step()
                # self.scheduler.step(self.epoch + idx / len(self.dataloader))


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
