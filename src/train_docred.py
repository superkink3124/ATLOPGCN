import argparse
import logging
import os
import re
from datetime import datetime

import numpy as np
import torch
# from apex import amp
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, set_seed
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from config.run_config import RunConfig
from model.model import ATLOPGCN
from dataset.collator import collate_fn
from dataset.utils import read_docred
from evaluation import to_official, official_evaluate
# import wandb


def train(args, model, train_features, dev_features, test_features, experiment_dir, logger):
    def finetune(features, optimizer, num_epoch, num_steps):
        best_score = -1
        train_dataloader = DataLoader(features,
                                      batch_size=args.train_batch_size,
                                      shuffle=True,
                                      collate_fn=collate_fn,
                                      drop_last=True)
        train_iterator = range(int(num_epoch))
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)
        logger.info("Total steps: {}".format(total_steps))
        logger.info("Warmup steps: {}".format(warmup_steps))
        for epoch in train_iterator:
            model.zero_grad()
            for step, batch in tqdm(enumerate(train_dataloader)):
                model.train()
                (
                    input_ids, input_mask,
                    entity_pos, sent_pos,
                    graph, num_mention, num_entity, num_sent,
                    labels, hts
                ) = batch
                inputs = {'input_ids': input_ids.to(args.device),
                          'attention_mask': input_mask.to(args.device),
                          'entity_pos': entity_pos,
                          'sent_pos': sent_pos,
                          'graph': graph.to(args.device),
                          'num_mention': num_mention,
                          'num_entity': num_entity,
                          'num_sent': num_sent,
                          'labels': labels,
                          'hts': hts,
                          }
                outputs = model(**inputs)
                loss = outputs[0] / args.gradient_accumulation_steps
                loss.backward()
                # with amp.scale_loss(loss, optimizer) as scaled_loss:
                #     scaled_loss.backward()
                if step % args.gradient_accumulation_steps == 0:
                    # if args.max_grad_norm > 0:
                    #     torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    num_steps += 1
                # wandb.log({"loss": loss.item()}, step=num_steps)
                if step % 100 == 0:
                    logger.info(loss)
                if (step + 1) == len(train_dataloader) - 1 or (args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
                    dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
                    logger.info(f'Dev score: {dev_score}')
                    test_score, test_output = evaluate(args, model, test_features, 'test')
                    logger.info(f'Test output: {test_output}')
                #     # wandb.log(dev_output, step=num_steps)
                #     logger.info (dev_output)
                #     if dev_score > best_score:
                #         best_score = dev_score
                #         pred = report(args, model, test_features)
                #         with open("result.json", "w") as fh:
                #             json.dump(pred, fh)
                #         if args.save_path != "":
                #             torch.save(model.state_dict(), args.save_path)
        os.makedirs(os.path.join(experiment_dir, 'model'))
        torch.save(model.state_dict(), os.path.join(experiment_dir, 'model', 'model.pt'))
        return num_steps

    new_layer = ["extractor", "bilinear", "gnn"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": 1e-4},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    num_steps = 0
    set_seed(args.seed)
    model.zero_grad()
    finetune(train_features, optimizer, args.num_train_epochs, num_steps)
    dev_score, dev_output = evaluate(args, model, dev_features, 'dev')
    logger.info(f'Dev score: {dev_score}')
    logger.info(f'Dev output: {dev_output}')
    test_score, test_output = evaluate(args, model, test_features, 'test')
    logger.info(f'Test score: {test_score}')
    logger.info(f'Test output: {test_output}')


def evaluate(args, model, features, tag="dev"):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    for batch in tqdm(dataloader):
        model.eval()

        (
            input_ids, input_mask,
            entity_pos, sent_pos,
            graph, num_mention, num_entity, num_sent,
            labels, hts
        ) = batch

        inputs = {'input_ids': input_ids.to(args.device),
                  'attention_mask': input_mask.to(args.device),
                  'entity_pos': entity_pos,
                  'sent_pos': sent_pos,
                  'graph': graph.to(args.device),
                  'num_mention': num_mention,
                  'num_entity': num_entity,
                  'num_sent': num_sent,
                  'hts': hts,
                  }
        # print(input_ids.shape)
        with torch.no_grad():
            pred, *_ = model(**inputs)
            # print(torch.mean(pred))
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            # print(pred.shape)
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    ans = to_official(preds, features)
    if len(ans) > 0:
        best_f1, _, best_f1_ign, _ = official_evaluate(ans, args.data_dir)
    output = {
        tag + "_F1": best_f1 * 100,
        tag + "_F1_ign": best_f1_ign * 100,
    }
    return best_f1, output


def report(args, model, features):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    for batch in tqdm(dataloader):
        model.eval()

        (
            input_ids, input_mask,
            entity_pos, sent_pos,
            graph, num_mention, num_entity, num_sent,
            labels, hts
        ) = batch
        inputs = {'input_ids': input_ids.to(args.device),
                  'attention_mask': input_mask.to(args.device),
                  'entity_pos': entity_pos,
                  'sent_pos': sent_pos,
                  'graph': graph.to(args.device),
                  'num_mention': num_mention,
                  'num_entity': num_entity,
                  'num_sent': num_sent,
                  'hts': hts,
                  }

        with torch.no_grad():
            pred, *_ = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    preds = to_official(preds, features)
    return preds


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


def setup_experiment_dir(config, tokenizer, bert_model):
    experiment_name = re.sub(r"\s", "_", config.experiment)
    experiment_subdir = f"{experiment_name}_{datetime.today().strftime('%Y_%m_%d_%H_%M_%S')}"
    experiment_dir = os.path.join(config.experiment_dir, experiment_subdir)
    os.makedirs(experiment_dir)
    with open(f"{experiment_dir}/config.json", "w") as outfile:
        outfile.write(str(config))
    tokenizer.save_pretrained(f'{experiment_dir}')
    bert_model.save_pretrained(f'{experiment_dir}')
    return experiment_dir


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="../dataset/docred", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)

    parser.add_argument("--train_file", default="train_annotated.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)
    parser.add_argument("--save_path", default="", type=str)
    parser.add_argument("--load_path", default="", type=str)

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=8, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_labels", default=4, type=int,
                        help="Max number of labels in prediction.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=97,
                        help="Number of relation types in dataset.")
    parser.add_argument('--config_path', type=str, default='config_file/docred_config.json')
    args = parser.parse_args()
    # wandb.init(project="DocRED")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # args.n_gpu = torch.cuda.device_count()
    args.device = device

    bert_config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=args.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
    )
    tokenizer.add_special_tokens({
        'additional_special_tokens': [
            '[ENTITY]',
            '[SENT]',
            '[/ENTITY]',
            '[/SENT]'
        ]
    })
    bert_model = AutoModel.from_pretrained(
        args.model_name_or_path,
        config=bert_config,
    )
    bert_model.resize_token_embeddings(len(tokenizer))
    read = read_docred

    train_file = os.path.join(args.data_dir, args.train_file)
    dev_file = os.path.join(args.data_dir, args.dev_file)
    test_file = os.path.join(args.data_dir, args.test_file)
    train_features = read(train_file, tokenizer, max_seq_length=args.max_seq_length)
    dev_features = read(dev_file, tokenizer, max_seq_length=args.max_seq_length)
    test_features = read(test_file, tokenizer, max_seq_length=args.max_seq_length)
    bert_config.cls_token_id = tokenizer.cls_token_id
    bert_config.sep_token_id = tokenizer.sep_token_id
    bert_config.transformer_type = args.transformer_type

    config_path = args.config_path
    config = RunConfig.from_json(config_path)
    model = ATLOPGCN(config.model, bert_model, device)
    model = model.to(device)
    experiment_dir = setup_experiment_dir(config, tokenizer, bert_model)
    logger = get_logger(os.path.join(experiment_dir, 'log.txt'))
    # train_features.extend(dev_features)

    if args.load_path == "":  # Training
        train(args, model, train_features, dev_features, test_features, experiment_dir, logger)
    else:  # Testing
        # model = amp.initialize(model, opt_level="O1", verbosity=0)
        model.load_state_dict(torch.load(args.load_path))
        dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
        print(dev_output)
        pred = report(args, model, test_features)
        with open("result.json", "w") as fh:
            json.dump(pred, fh)


if __name__ == "__main__":
    main()
