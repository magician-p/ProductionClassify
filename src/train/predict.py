import json

import torch
from transformers import AutoTokenizer

from src.configuration.config import *
from src.train._engine import create_model, pred


class Predictor:
    def __init__(self):
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_DIR)
        self.model = create_model(BEST_MODEL_DIR)
        self.model.to(self.device)
        with open(PROCESSED_DATA_DIR / 'id2label.json', 'r', encoding='utf-8') as f:
            self.id2label = json.load(f)

    def predict(self, text):
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=TOKEN_LENGTH, return_tensors='pt')
        inputs = {
            k: v.to(self.device) for k,v in inputs.items()
        }
        label = pred(self.model, inputs)
        return {'label': label.item(), 'category': self.id2label[str(label.item())]}

def run_predict(text):
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_DIR)
    model = create_model(BEST_MODEL_DIR)
    model.to(device)
    with open(PROCESSED_DATA_DIR / 'id2label.json', 'r', encoding='utf-8') as f:
        id2label = json.load(f)

    inputs = tokenizer(text, padding='max_length', truncation=True, max_length=TOKEN_LENGTH, return_tensors='pt')
    inputs = {
        k: v.to(device) for k,v in inputs.items()
    }
    label = pred(model, inputs)
    print(label.item())
    print(f'类别: {id2label[str(label.item())]}')
    return label.item(), id2label[str(label.item())]


if __name__ == '__main__':
    text = input('输入文本：')
    run_predict(text)