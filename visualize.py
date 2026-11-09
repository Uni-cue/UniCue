import csv
from pathlib import Path
from typing import List, Dict, Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['agg.path.chunksize'] = 10000  # 比如 10000，或者 1000 / 5000 等



class MetricsTracker:
    def __init__(self) -> None:
        self.iterations: List[int] = []
        self.train_loss: List[float] = []
        self.epochs: List[int] = []
        self.val_loss: List[float] = []
        self.val_si_snr: List[float] = []

    def log_train_iter(self, iteration: int, loss_value: float) -> None:
        self.iterations.append(iteration)
        self.train_loss.append(loss_value)

    def log_val_epoch(self, epoch: int, val_loss_value: float, val_si_snr_value: float) -> None:
        self.epochs.append(epoch)
        self.val_loss.append(val_loss_value)
        self.val_si_snr.append(val_si_snr_value)

    def save_csv(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_fp = output_dir / 'metrics.csv'
        with csv_fp.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'train_loss'])
            for i, l in zip(self.iterations, self.train_loss):
                writer.writerow([i, l])
            writer.writerow([])
            writer.writerow(['epoch', 'val_loss', 'val_si_snr'])
            for idx in range(len(self.epochs)):
                e = self.epochs[idx]
                l = self.val_loss[idx] if idx < len(self.val_loss) else ''
                s = self.val_si_snr[idx] if idx < len(self.val_si_snr) else ''
                writer.writerow([e, l, s])

    def plot(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        if len(self.iterations) > 0:
            plt.figure(figsize=(8, 4))
            plt.plot(self.iterations, self.train_loss, label='train si-snr loss')
            plt.xlabel('iteration')
            plt.ylabel('loss')
            plt.title('Training Loss')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig((output_dir / 'train_loss.png').as_posix())
            plt.close()

        if len(self.epochs) > 0:
            plt.figure(figsize=(8, 4))
            plt.plot(self.epochs, self.val_loss, marker='o', label='val loss (mean)')
            if len(self.val_si_snr) == len(self.epochs):
                ax2 = plt.twinx()
                ax2.plot(self.epochs, self.val_si_snr, color='orange', marker='s', label='val SI-SNR (dB)')
                ax2.set_ylabel('SI-SNR (dB)')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.title('Validation: Loss and SI-SNR by Epoch')
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig((output_dir / 'val_loss.png').as_posix())
            plt.close()


