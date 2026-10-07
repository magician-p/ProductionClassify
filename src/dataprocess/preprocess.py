__all__ = ['preprocess']

import json

from datasets import ClassLabel, Features, Value, load_dataset
from transformers import AutoTokenizer

from src.configuration import *


def preprocess():

    dataset_dict = load_dataset(
        'csv',
        data_files={
            'train': str(RAW_TRAIN_DATA),
            'test': str(RAW_TEST_DATA),
            'valid': str(RAW_VALID_DATA)
        },
        delimiter='\t',
        features=Features({
            'label': Value('string'),
            'text_a': Value('string')
        })
    )
    all_labels = sorted({
        label
        for item in dataset_dict.values()
        for label in item['label']
    })
    dataset_dict = dataset_dict.cast_column('label', ClassLabel(names=all_labels))
    print(dataset_dict['train'][0:3])
    id2label = {idx:label for idx, label in enumerate(all_labels)}
    with open(PROCESSED_DATA_DIR / 'id2label.json', 'w', encoding='utf-8') as f:
        json.dump(id2label, f, ensure_ascii=False, indent=4)
   
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_DIR)
    def process(x):
        inputs = tokenizer(text = x['text_a'], padding='max_length', truncation=True, max_length=TOKEN_LENGTH)
        inputs['labels'] = x['label']
        return inputs
    dataset_dict = dataset_dict.map(process, batched=True, remove_columns=['label', 'text_a'])

    dataset_dict.save_to_disk(PROCESSED_DATA_DIR)

if __name__ == "__main__":
    preprocess()