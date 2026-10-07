__all__ = ['get_dataloader', 'get_dataset']


from typing import Any, Literal

from datasets import DatasetDict, load_from_disk
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as PTDataset
from transformers import AutoTokenizer, DataCollatorWithPadding

from src.configuration.config import *


class HFDataset(PTDataset):
    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> Any:
        return self.dataset[index]

def get_dataset(datatype:Literal['train', 'test', 'valid']):
    dataset = load_from_disk(PROCESSED_DATA_DIR / datatype)
    dataset = (dataset[datatype] if isinstance(dataset, DatasetDict) else dataset)
    dataset.set_format('torch')
    dataset = HFDataset(dataset)
    return dataset

def get_dataloader(datatype:Literal['train', 'test', 'valid'], tokenizer, Batch_size=16, shuffle=True, drop_last=False):
    # dataset = load_from_disk(PROCESSED_DATA_DIR / datatype)
    # dataset = (dataset[datatype] if isinstance(dataset, DatasetDict) else dataset)
    # dataset.set_format('torch')
    # dataset = HFDataset(dataset)
    dataset = get_dataset(datatype)
    collate_fn = DataCollatorWithPadding(tokenizer, padding=True)
    dataset = DataLoader(
        dataset, 
        batch_size=Batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=collate_fn
    )
    return dataset

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_DIR)
    loader = get_dataloader('train', tokenizer)
    it = iter(loader)
    batch = next(it)
    for k,v in batch.items():
        print(k,v)