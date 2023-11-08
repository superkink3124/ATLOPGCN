import itertools

import dgl
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from config.cdr_config import RunConfig
from logging import Logger
from torch.utils.tensorboard import SummaryWriter

from corpus.cdr_corpus import CDRCorpus
from model.model import Model
from utils.loss import get_loss_from_config
from utils.metrics import f1_score, decode_ner, compute_NER_f1_macro
from utils.optimizers import get_optimizer_from_config
from tqdm import tqdm


class Trainer:
    def __init__(self, config: RunConfig, logger: Logger, summary_writer: SummaryWriter, device: str,
                 experiment_dir: str):
        super().__init__()
        self.logger = logger
        self.summary_writer = summary_writer
        self.device = device
        self.corpus = CDRCorpus(config)
        self.corpus.load_all_vocabs(config.data.saved_data_path)
        self.re_loss_fn = get_loss_from_config(config, device)
        self.ner_loss_fn = nn.CrossEntropyLoss()
        self.model = Model(config.model, len(self.corpus.pos_vocab), len(self.corpus.char_vocab), device)
        num_param = sum([param.numel() for param in self.model.parameters()])
        num_trainable_param = sum([param.numel() for param in self.model.parameters() if param.requires_grad == True])
        print(f"Num param: {num_param}")
        print(f"Num trainable param: {num_trainable_param}")
        if device == "cuda":
            self.model = self.model.cuda()
        self.optimizer = get_optimizer_from_config(config.train.optimizer, self.model)
        self.num_epochs = config.train.num_epochs
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.5)
        self.lr_decay_step = config.train.lr_decay_step
        # self.lr_scheduler = None
        self.log_interval = config.train.log_interval
        self.eval_interval = config.train.eval_interval
        self.experiment_dir = experiment_dir

        self.current_step = 0
        self.current_epoch = 0
        self.config = config

    def get_logits(self, inputs, training: bool):
        if training:
            self.model.train()
            re_logits, ner_logits = self.model(inputs)
            return re_logits, ner_logits
        else:
            self.model.eval()
            with torch.no_grad():
                re_logits, ner_logits = self.model(inputs)
            return re_logits, ner_logits

    def get_loss(self, inputs, ner_labels, labels, label_mask, training: bool):
        re_logits, ner_logits = self.get_logits(inputs, training)
        re_loss = self.re_loss_fn(re_logits, labels, label_mask)
        ner_loss = self.ner_loss_fn(torch.permute(ner_logits, (0, 3, 1, 2)), ner_labels)
        if self.config.model.use_ner:
            loss = ner_loss + re_loss
        else:
            loss = re_loss
        # print(loss)
        # raise Exception("Stop")
        return loss, re_loss, ner_loss, re_logits, ner_logits

    def train(self, train_dataloader, dev_dataloader=None):
        if self.logger is not None:
            self.logger.info("Start training")
        while self.current_epoch < self.num_epochs:
            interval_re_loss_hist = []
            interval_ner_loss_hist = []
            epoch_re_loss_hist = []
            epoch_ner_loss_hist = []
            if self.logger is not None:
                self.logger.info(f"Current epoch: {self.current_epoch} Num iteration {len(train_dataloader)}")
            for batch in tqdm(train_dataloader):
                self.optimizer.zero_grad()
                # noinspection DuplicatedCode
                if self.device == "cuda":
                    batch = [elem.to("cuda:0") if isinstance(elem, Tensor) or isinstance(elem, dgl.DGLGraph)
                             else elem for elem in batch]
                inputs = batch[:-3]
                ner_labels = batch[-3]
                labels = batch[-2]
                labels_mask = batch[-1]
                (
                    loss, re_loss, ner_loss,
                    re_logits, ner_logits
                ) = self.get_loss(inputs, ner_labels, labels, labels_mask, training=True)
                # print("_"*40)
                # print(loss)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                nn.utils.clip_grad_norm(self.model.parameters(), self.config.train.gradient_clipping)
                interval_re_loss_hist.append(re_loss.item())
                interval_ner_loss_hist.append(ner_loss.item())
                epoch_re_loss_hist.append(re_loss.item())
                epoch_ner_loss_hist.append(ner_loss.item())
                # loss = self.get_loss(inputs, labels, training=True)
                # print(loss)
                # print("_"*40)
                self.current_step += 1
                self.summary_writer.add_scalar("train_loss", loss.item(), self.current_step)
                if self.lr_scheduler is not None:
                    current_lr = self.lr_scheduler.get_last_lr()[0]
                    self.summary_writer.add_scalar("lr", current_lr, self.current_step)
                if self.current_step % self.log_interval == 0:
                    average_re_loss = np.mean(interval_re_loss_hist)
                    average_ner_loss = np.mean(interval_ner_loss_hist)
                    if self.lr_scheduler is not None:
                        self.logger.info(f"")
                    self.logger.info(f"["
                                     f"{self.current_step % len(train_dataloader)}"
                                     f"/ {len(train_dataloader)}]\n"
                                     f"Interval re loss: {average_re_loss}\n"
                                     f"Interval ner loss: {average_ner_loss}")
                    print("_" * 30)
                    print(f"Pre loss: re: {re_loss.item()} ner: {ner_loss.item()}")
                    (
                        loss, re_loss, ner_loss,
                        re_logits, ner_logits
                    ) = self.get_loss(inputs, ner_labels, labels, labels_mask, training=True)
                    print(f"Current loss: re: {re_loss.item()} ner: {ner_loss.item()}")
                    self.optimizer.zero_grad()
                    print("_" * 30)
                    interval_re_loss_hist = []
                    interval_ner_loss_hist = []

                if self.current_step % self.lr_decay_step == 0:
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
            if self.logger is not None:
                self.logger.info(f"Epoch {self.current_epoch}\n"
                                 f"Train re loss: {np.mean(epoch_re_loss_hist)}\n"
                                 f"Train ner loss: {np.mean(epoch_ner_loss_hist)}")
            epoch_re_loss_hist = []
            epoch_ner_loss_hist = []
            self.current_epoch += 1
            if dev_dataloader is not None:
                self.evaluate(dev_dataloader)
        self.save_model()

    def evaluate(self, dataloader):
        if self.logger is not None:
            self.logger.info("Start evaluate on devset")
        list_re_labels = []
        list_re_predicts = []

        list_ner_labels = []
        list_ner_predicts = []

        list_re_loss = []
        list_ner_loss = []
        for batch in tqdm(dataloader):
            if self.device == "cuda":
                batch = [elem.to("cuda:0") if isinstance(elem, Tensor) or isinstance(elem, dgl.DGLGraph) else elem for
                         elem in batch]
            inputs = batch[:-3]
            ner_labels = batch[-3]
            labels = batch[-2]
            labels_mask = batch[-1]
            (
                loss, re_loss, ner_loss,
                re_logits, ner_logits
            ) = self.get_loss(inputs, ner_labels, labels, labels_mask, training=False)
            batch_size = int(labels.shape[0])
            labels_per_batch = int(labels.shape[1])
            for batch_id in range(batch_size):
                for label_id in range(labels_per_batch):
                    if labels_mask[batch_id][label_id].item() == 0:
                        continue
                    list_re_labels.append(labels[batch_id][label_id].item())
                    list_re_predicts.append(re_logits[batch_id][label_id])
            list_re_loss.append(re_loss.item())
            list_ner_loss.append(ner_loss.item())

            ner_pred_classes = torch.argmax(ner_logits, dim=-1).cpu().reshape(batch_size, -1).data.numpy().tolist()
            ner_labels_classes = ner_labels.cpu().reshape(batch_size, -1).data.numpy().tolist()

            ner_pred_classes = decode_ner(ner_pred_classes)
            ner_labels_classes = decode_ner(ner_labels_classes)
            list_ner_predicts.extend(ner_pred_classes)
            list_ner_labels.extend(ner_labels_classes)

        ner_f1 = compute_NER_f1_macro(list_ner_predicts, list_ner_labels)
        re_f1, re_precision, re_recall = f1_score(list_re_labels, list_re_predicts, self.logger)
        if self.logger is not None:
            self.logger.info(f"Re loss: {np.mean(list_re_loss)}")
            self.logger.info(f"NER loss: {np.mean(list_ner_loss)}")

            self.logger.info(f"NER")
            self.logger.info(f"NER F1: {ner_f1}")

            self.logger.info(f"RE")
            self.logger.info(f"Precision: {re_precision}")
            self.logger.info(f"Recall: {re_recall}")
            self.logger.info(f"F1 Score: {re_f1}")
        return re_f1

    def save_model(self):
        ckpt = {
            "model": self.model.state_dict()
        }
        save_path = f"{self.experiment_dir}/model.pth"
        torch.save(ckpt, f=save_path)
