import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn
from transformers import AutoModel, PreTrainedModel
import math

from config.model_config import ModelConfig
from model.utils import process_long_input
from model.GNN import GNN
from opt_einsum import contract
from model.losses import ATLoss


def focal_loss(pred, target, mask, alpha=0.25, gamma=2.0):
    """
    Compute focal loss for binary classification.

    :param pred: Predicted probabilities, shape (b, n, n)
    :param target: Ground truth labels, shape (b, n, n)
    :param mask: Mask for valid pairs, shape (b, n, n)
    :param alpha: Weighting factor for class 1
    :param gamma: Focusing parameter
    :return: Scalar focal loss value
    """
    # Apply mask to predictions and targets
    pred = torch.sigmoid(pred)
    pred = pred * mask
    target = target * mask

    # Calculate the binary cross-entropy loss
    bce_loss = F.binary_cross_entropy(pred, target, reduction='none')

    # Calculate focal loss components
    p_t = target * pred + (1 - target) * (1 - pred)
    alpha_t = target * alpha + (1 - target) * (1 - alpha)
    focal_loss = alpha_t * (1 - p_t) ** gamma * bce_loss

    # Apply mask to focal loss and sum
    focal_loss = focal_loss * mask
    loss = focal_loss.sum() / mask.sum()

    return loss


class PairwiseBilinear(nn.Module):
    def __init__(self, d):
        super(PairwiseBilinear, self).__init__()
        self.W = nn.Parameter(torch.Tensor(d, d))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize the weights and bias similar to nn.Bilinear
        """
        init.kaiming_uniform_(self.W, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.W)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        Compute the pairwise bilinear form with bias.

        :param x: Input tensor of shape (b, n, d)
        :return: Pairwise bilinear tensor of shape (b, n, n)
        """
        bilinear_form = torch.einsum('bnd,de,bme->bmn', x, self.W, x) + self.bias
        return bilinear_form


class ATLOPGCN(nn.Module):
    def __init__(self, config: ModelConfig,
                 bert_model: PreTrainedModel, device: torch.device,
                 emb_size=768, block_size=64, use_ner: bool = False, use_entity_classify: bool = True):
        super().__init__()
        self.use_entity_classify = use_entity_classify
        self.config = config
        self.use_ner = use_ner
        self.device = device
        bert_config = bert_model.config
        self.bert_config = bert_config
        self.bert_model = bert_model
        self.hidden_size = bert_config.hidden_size

        self.cr_bilinear = PairwiseBilinear(bert_config.hidden_size + config.gnn.node_type_embedding)

        self.head_extractor = nn.Linear(2 * bert_config.hidden_size + config.gnn.node_type_embedding, emb_size)
        self.tail_extractor = nn.Linear(2 * bert_config.hidden_size + config.gnn.node_type_embedding, emb_size)

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = config.classifier.num_classes

        self.bilinear = nn.Linear(emb_size * block_size, self.num_labels)
        self.offset = 0
        self.gnn = GNN(config.gnn, bert_config.hidden_size + config.gnn.node_type_embedding, device)

        # if self.use_ner:
        #     self.ner_hidden_layer = nn.Linear(bert_config.hidden_size, config.ner_classifier.hidden_dim)
        #     self.ner_activation = nn.LeakyReLU()
        #     self.ner_classifier = nn.Linear(config.ner_classifier.hidden_dim, config.ner_classifier.ner_classes)
        #     self.ner_loss_func = nn.CrossEntropyLoss()

        if self.use_entity_classify:
            self.entity_classifier = nn.Linear(bert_config.hidden_size + config.gnn.node_type_embedding, 2)
            self.entity_classify_loss_func = nn.CrossEntropyLoss(reduction='none')

        self.loss_fnt = ATLoss()

    def encode(self, input_ids, attention_mask):
        config = self.bert_config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        else:
            raise NotImplementedError()
        sequence_output, attention = process_long_input(self.bert_model, input_ids,
                                                        attention_mask, start_tokens, end_tokens)
        return sequence_output, attention

    def get_sent_embed(self, sequence_output, batch_sent_pos, num_sent):
        batch_size, _, embed_dim = sequence_output.shape
        sent_embed = torch.zeros((batch_size, num_sent, embed_dim)).to(self.device)
        for batch_id, sent_pos in enumerate(batch_sent_pos):
            for sent_id, pos in enumerate(sent_pos):
                sent_embed[batch_id, sent_id] = sequence_output[batch_id, pos[0] + self.offset]
        return sent_embed

    def get_mention_embed(self, sequence_output, batch_entity_pos, num_mention):
        batch_size, _, embed_dim = sequence_output.shape
        mention_embed = torch.zeros((batch_size, num_mention, embed_dim)).to(self.device)
        for batch_id, entity_pos in enumerate(batch_entity_pos):
            mention_id = 0
            for ent_pos in entity_pos:
                for mention_pos in ent_pos:
                    mention_embed[batch_id, mention_id] = sequence_output[batch_id, mention_pos[0] + self.offset]
                    mention_id += 1
        return mention_embed

    def get_entity_embed(self, sequence_output, batch_entity_pos, num_entity):
        batch_size, _, embed_dim = sequence_output.shape
        entity_embed = torch.zeros((batch_size, num_entity, embed_dim)).to(self.device)
        for batch_id, entity_pos in enumerate(batch_entity_pos):
            for entity_id, ent_pos in enumerate(entity_pos):
                embeds = []
                for mention_pos in ent_pos:
                    embeds.append(sequence_output[batch_id, mention_pos[0] + self.offset])
                entity_embed[batch_id, entity_id] = torch.logsumexp(torch.stack(embeds, dim=0), dim=0)
        return entity_embed

    def get_rss(self, sequence_output, attention, entity_pos, hts):
        offset = 1 if self.bert_config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        for i in range(len(entity_pos)):
            entity_atts = []
            for e in entity_pos[i]:
                if len(e) > 1:
                    e_att = []
                    for start, end in e:
                        if start + offset < c:
                            e_att.append(attention[i, :, start + offset])
                    if len(e_att) > 0:
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_att = attention[i, :, start + offset]
                    else:
                        e_att = torch.zeros(h, c).to(attention)
                entity_atts.append(e_att)
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)
            rss.append(rs)
        rss = torch.cat(rss, dim=0)
        # print(rss.shape)
        return rss

    def get_pair_entity_embed(self, entity_hidden_state, hts):
        s_embed, t_embed = [], []
        for batch_id, ht in enumerate(hts):
            for pair in ht:
                s_embed.append(entity_hidden_state[batch_id, pair[0]])
                t_embed.append(entity_hidden_state[batch_id, pair[1]])
        s_embed = torch.stack(s_embed, dim=0)
        t_embed = torch.stack(t_embed, dim=0)
        return s_embed, t_embed

    def get_ner_logits(self, hidden_state):
        return self.ner_classifier(self.ner_activation(self.ner_hidden_layer(hidden_state)))

    def forward(self, input_ids, attention_mask,
                entity_pos, sent_pos,
                cr_matrix, cr_mask,
                graph, num_mention, num_entity, num_sent,
                labels=None, ner_labels=None,
                entity_type=None, entity_mask=None,
                hts=None):
        sequence_output, attention = self.encode(input_ids, attention_mask)
        mention_embed = self.get_mention_embed(sequence_output, entity_pos, num_mention)
        entity_embed = self.get_entity_embed(sequence_output, entity_pos, num_entity)
        sent_embed = self.get_sent_embed(sequence_output, sent_pos, num_sent)
        entity_hidden_state, mention_hidden_state = self.gnn([mention_embed, entity_embed, sent_embed, graph])

        cr_preds = self.cr_bilinear(mention_hidden_state)
        cr_loss = focal_loss(cr_preds, cr_matrix, cr_mask)

        local_context = self.get_rss(sequence_output, attention, entity_pos, hts)
        s_embed, t_embed = self.get_pair_entity_embed(entity_hidden_state, hts)

        s_embed = torch.tanh(self.head_extractor(torch.cat([s_embed, local_context], dim=1)))
        t_embed = torch.tanh(self.tail_extractor(torch.cat([t_embed, local_context], dim=1)))

        b1 = s_embed.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = t_embed.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)
        output = {
            "label": self.loss_fnt.get_label(logits, num_labels=self.num_labels),
            "cr_loss": cr_loss
        }
        if self.use_entity_classify:
            entity_logits = self.entity_classifier(entity_hidden_state)
            entity_logits = entity_logits.view(-1, 2)
            entity_type = entity_type.view(-1)
            entity_mask = entity_mask.view(-1)
            entity_classify_loss = self.entity_classify_loss_func(entity_logits, entity_type) * entity_mask
            entity_classify_loss = entity_classify_loss.sum() / entity_mask.sum()
            output["ec_loss"] = entity_classify_loss
        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)
            loss = self.loss_fnt(logits.float(), labels.float())
            output["loss"] = loss.to(sequence_output)
            if self.use_ner:
                ner_logits = self.get_ner_logits(sequence_output)
                ner_logits = ner_logits.view(-1, self.config.ner_classifier.ner_classes)
                ner_labels = ner_labels.view(-1)
                ner_loss = self.ner_loss_func(ner_logits, ner_labels)
                output["ner_loss"] = ner_loss
        return output
