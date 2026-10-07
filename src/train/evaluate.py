import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import AutoTokenizer

from src.configuration import *
from src.dataprocess import get_dataloader
from src.train._engine import batch_predict, create_model


def evaluate():
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_DIR)
    dataset = get_dataloader(
        'test',
        tokenizer,
        Batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False
    )

    model = create_model(BEST_MODEL_DIR)
    model.to(device)

    pred_labels, true_labels = batch_predict(model, dataset, device)

    print("Accuracy:", accuracy_score(true_labels, pred_labels))
    print("Precision:", precision_score(true_labels, pred_labels, average="macro", zero_division=0))
    print("Recall:", recall_score(true_labels, pred_labels, average="macro", zero_division=0))
    print("F1:", f1_score(true_labels, pred_labels, average="macro", zero_division=0))

    print(classification_report(true_labels, pred_labels, zero_division=0))
    print(confusion_matrix(true_labels, pred_labels))

__all__ = ['evaluate']