import json

import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
from transformers.utils import logging

from src.configuration.config import *

logging.disable_progress_bar()


def create_model(model_path):
    with open(PROCESSED_DATA_DIR / 'id2label.json', 'r', encoding='utf-8') as f:
        id2label = json.load(f)
    id2label = {int(k): v for k,v in id2label.items()}
    label2id = {v:k for k,v in id2label.items()}
    net = AutoModelForSequenceClassification.from_pretrained(
        model_path, 
        num_labels=len(id2label),
        id2label = id2label,
        label2id = label2id
    )
    return net

def train_epoch(net, train_dataloader, valid_dataloader, optim, device, epoch, 
                min_loss, write, step, save_steps, 
                score, patience, patience_counter, cast_enable, scalar,
                save_checkpoint):
    loss_ = 0
    net.train()
    for batch in tqdm(train_dataloader, desc=f'Epoch {epoch}/{EPOCHS} train: ', leave=False, position=1, ncols=100):
        with torch.autocast(device_type = device.type, dtype=torch.float16, enabled=cast_enable):
            inputs = {k: v.to(device) for k,v in batch.items()}

            loss = net(**inputs).loss

        optim.zero_grad()
        scalar.scale(loss).backward()
        scalar.step(optim)
        scalar.update()

        loss_ = loss.item()

        if step % save_steps == 0:
            tqdm.write(f'Train loss {loss_}')
            write.add_scalar('Train loss', loss_, step/save_steps)

            valid_scores = val(net, valid_dataloader, device)
            valid_score = valid_scores[score] if score == 'loss' else -valid_scores[score]

            if valid_score < min_loss:
                min_loss = valid_score
                patience_counter = 0
                net.save_pretrained(BEST_MODEL_DIR)
                save_checkpoint(net, optim, scalar, step, min_loss, patience_counter)
                tqdm.write(f'Save model to {BEST_MODEL_DIR}')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    tqdm.write(f'Early stopping at epoch {epoch} step {step}')
                    patience_counter = 0
                    return step, min_loss, patience_counter
        step += 1

    return step, min_loss, patience_counter


def val(net, dataloader, device):
    loss_ = 0
    loss_num = 0
    pred_labels = []
    true_labels = []
    net.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Val: ', leave=False, position=1, ncols=100):
            inputs = {k: v.to(device) for k,v in batch.items()}
            outputs = net(**inputs)
            loss = outputs.loss
            logits = outputs.logits
            loss_ += loss.item() * inputs['input_ids'].shape[0]
            loss_num += inputs['input_ids'].shape[0]
            pred_labels.append(torch.argmax(logits, dim=-1).cpu())
            true_labels.append(inputs['labels'].cpu())
        pred_labels = torch.cat(pred_labels).numpy()
        true_labels = torch.cat(true_labels).numpy()
        return {
            'loss': loss_ / loss_num,
            'accuracy': accuracy_score(true_labels, pred_labels),
            'precision': precision_score(true_labels, pred_labels, average='weighted', zero_division=0),
            'recall': recall_score(true_labels, pred_labels, average='weighted', zero_division=0),
            'f1': f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
        }


def batch_predict(net, dataloader, device):
    net.eval()
    with torch.no_grad():
        pred_labels = []
        true_labels = []
        for batch in tqdm(dataloader, desc='Evaluate: ', leave=False, position=0, ncols=100):
            inputs = {k: v.to(device) for k,v in batch.items()}
            logits = net(**inputs).logits
            pred_labels.append(torch.argmax(logits, dim=-1).cpu())
            true_labels.append(inputs['labels'].cpu())
        return (
            torch.cat(pred_labels).numpy(),
            torch.cat(true_labels).numpy()
        )


def pred(net, inputs):
    net.eval()
    with torch.no_grad():
        pred_labels = net(**inputs).logits
        return torch.argmax(pred_labels, dim=-1).cpu()