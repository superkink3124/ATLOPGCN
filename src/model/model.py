import torch
from torch import nn
from config.model_config import EncoderConfig, ModelConfig, ClassifierConfig, NERClassifierConfig
from allennlp.modules.elmo import Elmo
from utils.constansts import ELMO_OPTIONS_FILE, ELMO_WEIGHT_FILE
from model.GNN import GNN


class ATLOPGCN(nn.Module):
    def __init__(self, config, bert_model, emb_size=768, block_size=64, num_labels=2):
        super().__init__()
        self.config = config
        self.bert_model = bert_model
        self.hidden_size = config.hidden_size

        self.head_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.tail_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels


    def forward(self, input_ids, attention_mask):
