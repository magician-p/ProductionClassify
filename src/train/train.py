import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer

from src.configuration import *
from src.dataprocess import get_dataloader
from src.train._engine import create_model, train_epoch

__all__ = ['train']

@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-4
    model_path: str = '/models'
    log_dir: Path = Path('/logs')
    save_steps: int = SAVE_STEPS
    score: Literal['loss', 'accuracy', 'precision', 'recall', 'f1'] = 'loss'
    patience: int = 3
    cast_enable: bool = True
    checkpoint_path: Path = LAST_MODEL_DIR / 'checkpoint.pt'

class Trainer:
    def __init__(self, model, device, tokenizer, trainconfig) -> None:
        self.trainconfig = trainconfig
        self.device = device
        self.model = model.to(device)
        self.train_dataloader = get_dataloader(
            'train', tokenizer, self.trainconfig.batch_size
        )
        self.valid_dataloader = get_dataloader(
            'valid', tokenizer, self.trainconfig.batch_size
        )
        self.optim = torch.optim.Adam(self.model.parameters(), self.trainconfig.lr)
        self.step = 1
        self.min_loss = float('inf')
        self.write = SummaryWriter(self.trainconfig.log_dir / 'train' / time.strftime('%Y-%m-%d_%H-%M-%S'))
        self.patience_counter = 0
        self.scalar = torch.amp.GradScaler(self.device.type, enabled=self.trainconfig.cast_enable)

    def train(self):
        self._load_checkpoint()
        for epoch in tqdm(range(EPOCHS), desc='Training...', leave=True, position=0, ncols=100):
            self.step, self.min_loss, self.patience_counter = train_epoch(
                self.model,
                self.train_dataloader,
                self.valid_dataloader,
                self.optim,
                self.device,
                epoch,
                self.min_loss,
                self.write,
                self.step,
                self.trainconfig.save_steps,
                self.trainconfig.score,
                self.trainconfig.patience,
                self.patience_counter,
                self.trainconfig.cast_enable,
                self.scalar,
                self._save_checkpoint
            )
        self.write.close()

    def _save_checkpoint(self, model, optim, scalar, step, min_loss, patience_counter):
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'scalar_state_dict': scalar.state_dict() if hasattr(scalar, 'state_dict') else None,
            'step': step,
            'min_loss': min_loss,
            'patience_counter': patience_counter
        }
        torch.save(checkpoint, self.trainconfig.checkpoint_path)

    def _load_checkpoint(self):
        if Path(self.trainconfig.checkpoint_path).exists():
            tqdm.write(f"Loading checkpoint from {self.trainconfig.checkpoint_path}")
            checkpoint = torch.load(self.trainconfig.checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
            if checkpoint['scalar_state_dict'] is not None:
                self.scalar.load_state_dict(checkpoint['scalar_state_dict'])
            self.step = checkpoint['step']
            self.min_loss = checkpoint['min_loss']
            self.patience_counter = checkpoint['patience_counter']
        else:
            tqdm.write(f"No checkpoint found at {self.trainconfig.checkpoint_path}. Starting from scratch.")

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_DIR)
    
    model = create_model(PRETRAINED_MODEL_DIR)
    model.to(device)

    traincofig = TrainConfig(
        EPOCHS, 
        BATCH_SIZE, 
        LR, 
        str(BEST_MODEL_DIR), 
        LOG_DIR
    )
    trainer = Trainer(model, device, tokenizer, traincofig)

    trainer.train()

if __name__ == '__main__':
    train()