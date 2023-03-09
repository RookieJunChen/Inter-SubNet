import matplotlib.pyplot as plt
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from joblib import Parallel, delayed
import numpy as np

import audio_zen.metrics as metrics
from audio_zen.acoustics.feature import mag_phase, drop_band
from audio_zen.acoustics.mask import build_complex_ideal_ratio_mask, decompress_cIRM, build_ideal_ratio_mask
from audio_zen.trainer.base_trainer import BaseTrainer
from audio_zen.utils import ExecutionTime
from utils.logger import log
from audio_zen.acoustics.utils import transform_pesq_range

plt.switch_backend('agg')


class Trainer(BaseTrainer):
    def __init__(self, dist, rank, config, resume, only_validation, model, loss_function, optimizer, train_dataloader,
                 validation_dataloader):
        super().__init__(dist, rank, config, resume, only_validation, model, loss_function, optimizer)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = validation_dataloader

    def _train_epoch(self, epoch):

        loss_total = 0.0
        progress_bar = None

        if self.rank == 0:
            progress_bar = tqdm(total=len(self.train_dataloader), desc=f"Training")

        for noisy, clean in self.train_dataloader:
            self.optimizer.zero_grad()

            noisy = noisy.to(self.rank)
            clean = clean.to(self.rank)

            noisy_complex = self.torch_stft(noisy)
            clean_complex = self.torch_stft(clean)

            noisy_mag, _ = mag_phase(noisy_complex)
            ground_truth_cIRM = build_complex_ideal_ratio_mask(noisy_complex, clean_complex)  # [B, F, T, 2]
            ground_truth_cIRM = drop_band(
                ground_truth_cIRM.permute(0, 3, 1, 2),  # [B, 2, F ,T]
                self.model.module.num_groups_in_drop_band
            ).permute(0, 2, 3, 1)

            with autocast(enabled=self.use_amp):
                # [B, F, T] => [B, 1, F, T] => model => [B, 2, F, T] => [B, F, T, 2]
                noisy_mag = noisy_mag.unsqueeze(1)
                cRM = self.model(noisy_mag)
                cRM = cRM.permute(0, 2, 3, 1)
                loss = self.loss_function(ground_truth_cIRM, cRM)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_total += loss.item()

            if self.rank == 0:
                progress_bar.update(1)
                progress_bar.refresh()

        if self.rank == 0:
            log(f"[Train] Epoch {epoch}, Loss {loss_total / len(self.train_dataloader)}")
            self.writer.add_scalar(f"Loss/Train", loss_total / len(self.train_dataloader), epoch)

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        progress_bar = None
        if self.rank == 0:
            progress_bar = tqdm(total=len(self.valid_dataloader), desc=f"Validation")

        visualization_n_samples = self.visualization_config["n_samples"]
        visualization_num_workers = self.visualization_config["num_workers"]
        visualization_metrics = self.visualization_config["metrics"]

        loss_total = 0.0
        loss_list = {"With_reverb": 0.0, "No_reverb": 0.0, }
        item_idx_list = {"With_reverb": 0, "No_reverb": 0, }
        noisy_y_list = {"With_reverb": [], "No_reverb": [], }
        clean_y_list = {"With_reverb": [], "No_reverb": [], }
        enhanced_y_list = {"With_reverb": [], "No_reverb": [], }
        validation_score_list = {"With_reverb": 0.0, "No_reverb": 0.0}

        # speech_type in ("with_reverb", "no_reverb")
        for i, (noisy, clean, name, speech_type) in enumerate(self.valid_dataloader):
            assert len(name) == 1, "The batch size for the validation stage must be one."
            name = name[0]
            speech_type = speech_type[0]

            noisy = noisy.to(self.rank)
            clean = clean.to(self.rank)

            noisy_complex = self.torch_stft(noisy)
            clean_complex = self.torch_stft(clean)

            noisy_mag, _ = mag_phase(noisy_complex)
            cIRM = build_complex_ideal_ratio_mask(noisy_complex, clean_complex)  # [B, F, T, 2]

            noisy_mag = noisy_mag.unsqueeze(1)
            cRM = self.model(noisy_mag)
            cRM = cRM.permute(0, 2, 3, 1)

            loss = self.loss_function(cIRM, cRM)

            cRM = decompress_cIRM(cRM)

            enhanced_real = cRM[..., 0] * noisy_complex.real - cRM[..., 1] * noisy_complex.imag
            enhanced_imag = cRM[..., 1] * noisy_complex.real + cRM[..., 0] * noisy_complex.imag
            enhanced_complex = torch.stack((enhanced_real, enhanced_imag), dim=-1)
            enhanced = self.torch_istft(enhanced_complex, length=noisy.size(-1))

            noisy = noisy.detach().squeeze(0).cpu().numpy()
            clean = clean.detach().squeeze(0).cpu().numpy()
            enhanced = enhanced.detach().squeeze(0).cpu().numpy()

            assert len(noisy) == len(clean) == len(enhanced)
            loss_total += loss

            # Separated loss
            loss_list[speech_type] += loss
            item_idx_list[speech_type] += 1

            # if item_idx_list[speech_type] <= visualization_n_samples:
            #     self.spec_audio_visualization(noisy, enhanced, clean, name, epoch, mark=speech_type)

            noisy_y_list[speech_type].append(noisy)
            clean_y_list[speech_type].append(clean)
            enhanced_y_list[speech_type].append(enhanced)

            if self.rank == 0:
                progress_bar.update(1)

        log(f"[Test] Epoch {epoch}, Loss {loss_total / len(self.valid_dataloader)}")
        self.writer.add_scalar(f"Loss/Validation_Total", loss_total / len(self.valid_dataloader), epoch)

        for speech_type in ("With_reverb", "No_reverb"):
            log(f"[Test] Epoch {epoch}, {speech_type}, Loss {loss_list[speech_type] / len(self.valid_dataloader)}")
            self.writer.add_scalar(f"Loss/{speech_type}", loss_list[speech_type] / len(self.valid_dataloader), epoch)

            validation_score_list[speech_type] = self.metrics_visualization(
                noisy_y_list[speech_type], clean_y_list[speech_type], enhanced_y_list[speech_type],
                visualization_metrics, epoch, visualization_num_workers, mark=speech_type
            )

        return validation_score_list["No_reverb"]


class New_Judge_Trainer(BaseTrainer):
    def __init__(self, dist, rank, config, resume, only_validation, model, loss_function, optimizer, train_dataloader,
                 validation_dataloader):
        super().__init__(dist, rank, config, resume, only_validation, model, loss_function, optimizer)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = validation_dataloader

    def metrics_visualization(self, noisy_list, clean_list, enhanced_list, metrics_list, epoch, num_workers=10,
                              mark=""):
        """
        Get metrics on validation dataset by paralleling.

        Notes:
            1. You can register other metrics, but STOI and WB_PESQ metrics must be existence. These two metrics are
             used for checking if the current epoch is a "best epoch."
            2. If you want to use a new metric, you must register it in "util.metrics" file.
        """
        assert "STOI" in metrics_list and "WB_PESQ" in metrics_list, "'STOI' and 'WB_PESQ' must be existence."

        # Check if the metric is registered in "util.metrics" file.
        for i in metrics_list:
            assert i in metrics.REGISTERED_METRICS.keys(), f"{i} is not registered, please check 'util.metrics' file."

        stoi_mean = 0.0
        wb_pesq_mean = 0.0
        for metric_name in metrics_list:
            score_on_noisy = Parallel(n_jobs=num_workers)(
                delayed(metrics.REGISTERED_METRICS[metric_name])(ref, est, sr=self.acoustic_config["sr"]) for
                ref, est
                in zip(clean_list, noisy_list)
            )
            score_on_enhanced = Parallel(n_jobs=num_workers)(
                delayed(metrics.REGISTERED_METRICS[metric_name])(ref, est, sr=self.acoustic_config["sr"]) for
                ref, est
                in zip(clean_list, enhanced_list)
            )

            # Add the mean value of the metric to tensorboard
            mean_score_on_noisy = np.mean(score_on_noisy)
            mean_score_on_enhanced = np.mean(score_on_enhanced)
            self.writer.add_scalars(f"{mark}_Validation/{metric_name}", {
                "Noisy": mean_score_on_noisy,
                "Enhanced": mean_score_on_enhanced
            }, epoch)
            log(f"{mark}_Validation/{metric_name}  " + str(mean_score_on_enhanced))
            if metric_name == "STOI":
                stoi_mean = mean_score_on_enhanced

            if metric_name == "WB_PESQ":
                wb_pesq_mean = transform_pesq_range(mean_score_on_enhanced)

            if metric_name == "SI_SDR":
                si_snr_mean = mean_score_on_enhanced / 6

        return (stoi_mean + wb_pesq_mean + si_snr_mean) / 3

    def _train_epoch(self, epoch):
        loss_total = 0.0
        progress_bar = None

        if self.rank == 0:
            progress_bar = tqdm(total=len(self.train_dataloader), desc=f"Training")

        for noisy, clean in self.train_dataloader:
            self.optimizer.zero_grad()

            noisy = noisy.to(self.rank)
            clean = clean.to(self.rank)

            noisy_complex = self.torch_stft(noisy)
            clean_complex = self.torch_stft(clean)

            noisy_mag, _ = mag_phase(noisy_complex)
            ground_truth_cIRM = build_complex_ideal_ratio_mask(noisy_complex, clean_complex)  # [B, F, T, 2]
            ground_truth_cIRM = drop_band(
                ground_truth_cIRM.permute(0, 3, 1, 2),  # [B, 2, F ,T]
                self.model.module.num_groups_in_drop_band
            ).permute(0, 2, 3, 1)

            with autocast(enabled=self.use_amp):
                # [B, F, T] => [B, 1, F, T] => model => [B, 2, F, T] => [B, F, T, 2]
                noisy_mag = noisy_mag.unsqueeze(1)
                cRM = self.model(noisy_mag)
                cRM = cRM.permute(0, 2, 3, 1)
                loss = self.loss_function(ground_truth_cIRM, cRM)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_total += loss.item()

            if self.rank == 0:
                progress_bar.update(1)
                progress_bar.refresh()

        if self.rank == 0:
            log(f"[Train] Epoch {epoch}, Loss {loss_total / len(self.train_dataloader)}")
            self.writer.add_scalar(f"Loss/Train", loss_total / len(self.train_dataloader), epoch)

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        progress_bar = None
        if self.rank == 0:
            progress_bar = tqdm(total=len(self.valid_dataloader), desc=f"Validation")

        visualization_n_samples = self.visualization_config["n_samples"]
        visualization_num_workers = self.visualization_config["num_workers"]
        visualization_metrics = self.visualization_config["metrics"]

        loss_total = 0.0
        loss_list = {"With_reverb": 0.0, "No_reverb": 0.0, }
        item_idx_list = {"With_reverb": 0, "No_reverb": 0, }
        noisy_y_list = {"With_reverb": [], "No_reverb": [], }
        clean_y_list = {"With_reverb": [], "No_reverb": [], }
        enhanced_y_list = {"With_reverb": [], "No_reverb": [], }
        validation_score_list = {"With_reverb": 0.0, "No_reverb": 0.0}

        # speech_type in ("with_reverb", "no_reverb")
        for i, (noisy, clean, name, speech_type) in enumerate(self.valid_dataloader):
            assert len(name) == 1, "The batch size for the validation stage must be one."
            name = name[0]
            speech_type = speech_type[0]

            noisy = noisy.to(self.rank)
            clean = clean.to(self.rank)

            noisy_complex = self.torch_stft(noisy)
            clean_complex = self.torch_stft(clean)

            noisy_mag, _ = mag_phase(noisy_complex)
            cIRM = build_complex_ideal_ratio_mask(noisy_complex, clean_complex)  # [B, F, T, 2]

            noisy_mag = noisy_mag.unsqueeze(1)
            cRM = self.model(noisy_mag)
            cRM = cRM.permute(0, 2, 3, 1)

            loss = self.loss_function(cIRM, cRM)

            cRM = decompress_cIRM(cRM)

            enhanced_real = cRM[..., 0] * noisy_complex.real - cRM[..., 1] * noisy_complex.imag
            enhanced_imag = cRM[..., 1] * noisy_complex.real + cRM[..., 0] * noisy_complex.imag
            enhanced_complex = torch.stack((enhanced_real, enhanced_imag), dim=-1)
            enhanced = self.torch_istft(enhanced_complex, length=noisy.size(-1))

            noisy = noisy.detach().squeeze(0).cpu().numpy()
            clean = clean.detach().squeeze(0).cpu().numpy()
            enhanced = enhanced.detach().squeeze(0).cpu().numpy()

            assert len(noisy) == len(clean) == len(enhanced)
            loss_total += loss

            # Separated loss
            loss_list[speech_type] += loss
            item_idx_list[speech_type] += 1

            # if item_idx_list[speech_type] <= visualization_n_samples:
            #     self.spec_audio_visualization(noisy, enhanced, clean, name, epoch, mark=speech_type)

            noisy_y_list[speech_type].append(noisy)
            clean_y_list[speech_type].append(clean)
            enhanced_y_list[speech_type].append(enhanced)

            if self.rank == 0:
                progress_bar.update(1)

        log(f"[Test] Epoch {epoch}, Loss {loss_total / len(self.valid_dataloader)}")
        self.writer.add_scalar(f"Loss/Validation_Total", loss_total / len(self.valid_dataloader), epoch)

        for speech_type in ("With_reverb", "No_reverb"):
            log(f"[Test] Epoch {epoch}, {speech_type}, Loss {loss_list[speech_type] / len(self.valid_dataloader)}")
            self.writer.add_scalar(f"Loss/{speech_type}", loss_list[speech_type] / len(self.valid_dataloader), epoch)

            validation_score_list[speech_type] = self.metrics_visualization(
                noisy_y_list[speech_type], clean_y_list[speech_type], enhanced_y_list[speech_type],
                visualization_metrics, epoch, visualization_num_workers, mark=speech_type
            )

        return  (validation_score_list["No_reverb"] + validation_score_list["With_reverb"]) / 2


class Complex_New_Judge_Trainer(BaseTrainer):
    def __init__(self, dist, rank, config, resume, only_validation, model, loss_function, optimizer, train_dataloader,
                 validation_dataloader):
        super().__init__(dist, rank, config, resume, only_validation, model, loss_function, optimizer)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = validation_dataloader

    def metrics_visualization(self, noisy_list, clean_list, enhanced_list, metrics_list, epoch, num_workers=10,
                              mark=""):
        """
        Get metrics on validation dataset by paralleling.

        Notes:
            1. You can register other metrics, but STOI and WB_PESQ metrics must be existence. These two metrics are
             used for checking if the current epoch is a "best epoch."
            2. If you want to use a new metric, you must register it in "util.metrics" file.
        """
        assert "STOI" in metrics_list and "WB_PESQ" in metrics_list, "'STOI' and 'WB_PESQ' must be existence."

        # Check if the metric is registered in "util.metrics" file.
        for i in metrics_list:
            assert i in metrics.REGISTERED_METRICS.keys(), f"{i} is not registered, please check 'util.metrics' file."

        stoi_mean = 0.0
        wb_pesq_mean = 0.0
        for metric_name in metrics_list:
            score_on_noisy = Parallel(n_jobs=num_workers)(
                delayed(metrics.REGISTERED_METRICS[metric_name])(ref, est, sr=self.acoustic_config["sr"]) for
                ref, est
                in zip(clean_list, noisy_list)
            )
            score_on_enhanced = Parallel(n_jobs=num_workers)(
                delayed(metrics.REGISTERED_METRICS[metric_name])(ref, est, sr=self.acoustic_config["sr"]) for
                ref, est
                in zip(clean_list, enhanced_list)
            )

            # Add the mean value of the metric to tensorboard
            mean_score_on_noisy = np.mean(score_on_noisy)
            mean_score_on_enhanced = np.mean(score_on_enhanced)
            self.writer.add_scalars(f"{mark}_Validation/{metric_name}", {
                "Noisy": mean_score_on_noisy,
                "Enhanced": mean_score_on_enhanced
            }, epoch)
            log(f"{mark}_Validation/{metric_name}  " + str(mean_score_on_enhanced))
            if metric_name == "STOI":
                stoi_mean = mean_score_on_enhanced

            if metric_name == "WB_PESQ":
                wb_pesq_mean = transform_pesq_range(mean_score_on_enhanced)

            if metric_name == "SI_SDR":
                si_snr_mean = mean_score_on_enhanced / 6

        return (stoi_mean + wb_pesq_mean + si_snr_mean) / 3

    def _train_epoch(self, epoch):
        loss_total = 0.0
        progress_bar = None

        if self.rank == 0:
            progress_bar = tqdm(total=len(self.train_dataloader), desc=f"Training")

        for noisy, clean in self.train_dataloader:
            self.optimizer.zero_grad()

            noisy = noisy.to(self.rank)
            clean = clean.to(self.rank)

            noisy_complex = self.torch_stft(noisy)
            clean_complex = self.torch_stft(clean)

            noisy_mag, _ = mag_phase(noisy_complex)
            ground_truth_cIRM = build_complex_ideal_ratio_mask(noisy_complex, clean_complex)  # [B, F, T, 2]
            ground_truth_cIRM = drop_band(
                ground_truth_cIRM.permute(0, 3, 1, 2),  # [B, 2, F ,T]
                self.model.module.num_groups_in_drop_band
            ).permute(0, 2, 3, 1)

            with autocast(enabled=self.use_amp):
                # [B, F, T] => [B, 1, F, T] => model => [B, 2, F, T] => [B, F, T, 2]
                noisy_mag = noisy_mag.unsqueeze(1)
                noisy_real = (noisy_complex.real).unsqueeze(1)
                noisy_imag = (noisy_complex.imag).unsqueeze(1)
                complex_input = torch.cat([noisy_mag, noisy_real, noisy_imag], dim=1)   # [B, 3, F, T]
                cRM = self.model(complex_input)
                cRM = cRM.permute(0, 2, 3, 1)
                loss = self.loss_function(ground_truth_cIRM, cRM)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_total += loss.item()

            if self.rank == 0:
                progress_bar.update(1)
                progress_bar.refresh()

        if self.rank == 0:
            log(f"[Train] Epoch {epoch}, Loss {loss_total / len(self.train_dataloader)}")
            self.writer.add_scalar(f"Loss/Train", loss_total / len(self.train_dataloader), epoch)

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        progress_bar = None
        if self.rank == 0:
            progress_bar = tqdm(total=len(self.valid_dataloader), desc=f"Validation")

        visualization_n_samples = self.visualization_config["n_samples"]
        visualization_num_workers = self.visualization_config["num_workers"]
        visualization_metrics = self.visualization_config["metrics"]

        loss_total = 0.0
        loss_list = {"With_reverb": 0.0, "No_reverb": 0.0, }
        item_idx_list = {"With_reverb": 0, "No_reverb": 0, }
        noisy_y_list = {"With_reverb": [], "No_reverb": [], }
        clean_y_list = {"With_reverb": [], "No_reverb": [], }
        enhanced_y_list = {"With_reverb": [], "No_reverb": [], }
        validation_score_list = {"With_reverb": 0.0, "No_reverb": 0.0}

        # speech_type in ("with_reverb", "no_reverb")
        for i, (noisy, clean, name, speech_type) in enumerate(self.valid_dataloader):
            assert len(name) == 1, "The batch size for the validation stage must be one."
            name = name[0]
            speech_type = speech_type[0]

            noisy = noisy.to(self.rank)
            clean = clean.to(self.rank)

            noisy_complex = self.torch_stft(noisy)
            clean_complex = self.torch_stft(clean)

            noisy_mag, _ = mag_phase(noisy_complex)
            cIRM = build_complex_ideal_ratio_mask(noisy_complex, clean_complex)  # [B, F, T, 2]

            noisy_mag = noisy_mag.unsqueeze(1)
            noisy_real = (noisy_complex.real).unsqueeze(1)
            noisy_imag = (noisy_complex.imag).unsqueeze(1)
            complex_input = torch.cat([noisy_mag, noisy_real, noisy_imag], dim=1)  # [B, 3, F, T]
            cRM = self.model(complex_input)
            cRM = cRM.permute(0, 2, 3, 1)

            loss = self.loss_function(cIRM, cRM)

            cRM = decompress_cIRM(cRM)

            enhanced_real = cRM[..., 0] * noisy_complex.real - cRM[..., 1] * noisy_complex.imag
            enhanced_imag = cRM[..., 1] * noisy_complex.real + cRM[..., 0] * noisy_complex.imag
            enhanced_complex = torch.stack((enhanced_real, enhanced_imag), dim=-1)
            enhanced = self.torch_istft(enhanced_complex, length=noisy.size(-1))

            noisy = noisy.detach().squeeze(0).cpu().numpy()
            clean = clean.detach().squeeze(0).cpu().numpy()
            enhanced = enhanced.detach().squeeze(0).cpu().numpy()

            assert len(noisy) == len(clean) == len(enhanced)
            loss_total += loss

            # Separated loss
            loss_list[speech_type] += loss
            item_idx_list[speech_type] += 1

            # if item_idx_list[speech_type] <= visualization_n_samples:
            #     self.spec_audio_visualization(noisy, enhanced, clean, name, epoch, mark=speech_type)

            noisy_y_list[speech_type].append(noisy)
            clean_y_list[speech_type].append(clean)
            enhanced_y_list[speech_type].append(enhanced)

            if self.rank == 0:
                progress_bar.update(1)

        log(f"[Test] Epoch {epoch}, Loss {loss_total / len(self.valid_dataloader)}")
        self.writer.add_scalar(f"Loss/Validation_Total", loss_total / len(self.valid_dataloader), epoch)

        for speech_type in ("With_reverb", "No_reverb"):
            log(f"[Test] Epoch {epoch}, {speech_type}, Loss {loss_list[speech_type] / len(self.valid_dataloader)}")
            self.writer.add_scalar(f"Loss/{speech_type}", loss_list[speech_type] / len(self.valid_dataloader), epoch)

            validation_score_list[speech_type] = self.metrics_visualization(
                noisy_y_list[speech_type], clean_y_list[speech_type], enhanced_y_list[speech_type],
                visualization_metrics, epoch, visualization_num_workers, mark=speech_type
            )

        return  (validation_score_list["No_reverb"] + validation_score_list["With_reverb"]) / 2
