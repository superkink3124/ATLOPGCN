import torch
from .graph_builder import GraphBuilder


graph_builder = GraphBuilder()


ner_vocab = {
    'O': 0,
    'B_Chemical': 1,
    'I_Chemical': 2,
    'B_Disease': 3,
    'I_Disease': 4,
}


def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0 for _ in range(max_len - len(f["input_ids"]))] for f in batch]
    ner_labels = [f['ner_labels'] + ['O' for _ in range(max_len - len(f["input_ids"]))] for f in batch]
    ner_labels = [[ner_vocab[elem] for elem in ner_label] for ner_label in ner_labels]
    ner_labels = torch.tensor(ner_labels, dtype=torch.long)
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    entity_pos = [f["entity_pos"] for f in batch]
    sent_pos = [f['sent_pos'] for f in batch]
    hts = [f["hts"] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    ner_labels = torch.tensor(ner_labels)
    graph, num_mention, num_entity, num_sent = graph_builder.create_graph(entity_pos, sent_pos)
    output = (input_ids, input_mask,
              entity_pos, sent_pos,
              graph, num_mention, num_entity, num_sent,
              labels, ner_labels, hts)
    return output
