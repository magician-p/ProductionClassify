from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

# 数据路径
DATA_DIR = PROJECT_ROOT / 'data'
# 原始数据
RAW_DATA_DIR = DATA_DIR / 'raw'
RAW_TRAIN_DATA = RAW_DATA_DIR / 'train.txt'
RAW_TEST_DATA = RAW_DATA_DIR / 'test.txt'
RAW_VALID_DATA = RAW_DATA_DIR / 'valid.txt'
# 处理后的数据路径
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
PROCESSED_TRAIN_DATA = PROCESSED_DATA_DIR / 'train'
PROCESSED_TEST_DATA = PROCESSED_DATA_DIR / 'test'
PROCESSED_VALID_DATA = PROCESSED_DATA_DIR / 'valid'

# 训练模型保存目录
CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoint'
BEST_MODEL_DIR = CHECKPOINT_DIR / 'best'
LAST_MODEL_DIR = CHECKPOINT_DIR / 'last'

# 预训练模型位置
PRETRAINED_MODEL_DIR = PROJECT_ROOT / 'pretrained'
# 预训练模型名称
PRETRAINED_MODEL_NAME = 'google-bert/bert-base-chinese'

# 日志目录
LOG_DIR = PROJECT_ROOT / 'logs'

# 分词超参
TOKEN_LENGTH = 128

# 训练超参
BATCH_SIZE = 16
EPOCHS = 10
LR = 1E-4
SAVE_STEPS = 100