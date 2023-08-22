import argparse
import json
import os.path
import re
from datetime import datetime
import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoTokenizer, AutoModel, AutoConfig

from config.cdr_config import CDRConfig
from corpus.cdr_corpus import CDRCorpus
from dataset.collator import Collator, collate_fn
from torch.utils.data import DataLoader
import logging

from dataset.utils import read_cdr
from model.trainer import Trainer
import re
from datetime import datetime
import os
import json


# TODO: valid norm layer
# List architecture: gcn, gate gcn, res gcn, dense gcn, jump gcn, gcnii, incep gcn, jknet, sgc,


def seed_all(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)


def get_logger(filename: str):
    logger = logging.getLogger("TRAIN")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(name)s %(filename)s %(funcName)s %(levelname)s %(module)s %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(filename=filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def setup_experiment_dir(config: CDRConfig):
    experiment_name = re.sub(r"\s", "_", config.experiment)
    experiment_subdir = f"{experiment_name}_{datetime.today().strftime('%Y_%m_%d_%H_%M_%S')}"
    experiment_dir = os.path.join(config.experiment_dir, experiment_subdir)
    os.makedirs(experiment_dir)
    with open(f"{experiment_dir}/config.json", "w") as outfile:
        json.dump(json.loads(str(config)), outfile, indent=4, ensure_ascii=False)
    return experiment_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config.json")
    parser.add_argument("--seed", default=22, type=int)
    parser.add_argument("--concat", action="store_true")

    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    args = parser.parse_args()
    # device = "cuda"
    pretrained_name = 'allenai/scibert_scivocab_cased'
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    device = "cpu"
    config = CDRConfig.from_json(args.config)
    experiment_dir = setup_experiment_dir(config)
    logger = get_logger(f"{experiment_dir}/log.txt")
    logger.info("_" * 50)
    train_features = read_cdr(config.data.train_file_path, tokenizer)
    dev_features = read_cdr(config.data.dev_file_path, tokenizer)
    test_features = read_cdr(config.data.test_file_path, tokenizer)
    train_loader = DataLoader(train_features,
                              batch_size=config.train.batch_size,
                              shuffle=True, collate_fn=collate_fn)
    if dev_features is not None:
        dev_loader = DataLoader(dev_features, batch_size=config.train.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_features, batch_size=config.train.batch_size, collate_fn=collate_fn)
    bert_model = AutoModel.from_pretrained(pretrained_name)
    bert_config = AutoConfig.from_pretrained(pretrained_name)

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type

    train_features.


    # if args.concat:
    #     train_features.extend(dev_features)
    #     collator = Collator(corpus)
    #     train_dataloader = DataLoader(train_dataset,
    #                                   batch_size=config.train.batch_size,
    #                                   shuffle=True,
    #                                   collate_fn=collator.collate)
    #     test_dataloader = DataLoader(test_dataset,
    #                                  batch_size=config.train.batch_size,
    #                                  shuffle=True,
    #                                  collate_fn=collator.collate)
    #     logger.info(config)
    #     logger.info("_" * 50)
    #     trainer = Trainer(config, logger, None, device)
    #     trainer.train(train_dataloader)
    #     trainer.evaluate(test_dataloader)
    # else:
    #     collator = Collator(corpus)
    #     train_dataloader = DataLoader(train_dataset,
    #                                   batch_size=config.train.batch_size,
    #                                   shuffle=True,
    #                                   collate_fn=collator.collate)
    #     dev_dataloader = DataLoader(dev_dataset,
    #                                 batch_size=config.train.batch_size,
    #                                 shuffle=True,
    #                                 collate_fn=collator.collate)
    #     logger.info(config)
    #     logger.info("_" * 50)
    #     trainer = Trainer(config, logger, None, device)
    #     trainer.train(train_dataloader, dev_dataloader)
