import sys
from argparse import ArgumentParser
from pathlib import Path

if __name__ == '__main__':
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    parser = ArgumentParser()
    parser.add_argument('action', choices=['preprocess', 'train', 'predict', 'evaluate', 'serve'])

    args = parser.parse_args()
    arg = args.action

    match arg:
        case 'preprocess':
            from src.dataprocess.preprocess import preprocess
            preprocess()
        case 'train':
            from src.train.train import train
            train()
        case 'predict':
            from src.train.predict import run_predict
            text = input('输入文本：')
            run_predict(text)
        case 'evaluate':
            from src.train.evaluate import evaluate
            evaluate()
        case 'serve':
            from src.web.app import serve
            serve()