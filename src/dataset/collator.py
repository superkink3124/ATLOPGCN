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

entity_type_vocab = {
    'Chemical': 0,
    'Disease': 1
}


def gen_cr_matrix(batch_entity_pos, num_mention):
    batch_size = len(batch_entity_pos)
    cr_matrix = torch.zeros((batch_size, num_mention, num_mention), dtype=torch.long)
    cr_mask = torch.zeros((batch_size, num_mention, num_mention), dtype=torch.float)
    for batch_idx, entity_pos_arr in enumerate(batch_entity_pos):
        current_idx = 0
        for entity_idx, mention_pos_arr in enumerate(entity_pos_arr):
            for i in range(current_idx, current_idx + len(mention_pos_arr)):
                for j in range(current_idx, current_idx + len(mention_pos_arr)):
                    cr_matrix[batch_idx, i, j] = 1
            current_idx += len(mention_pos_arr)
        current_num_mention = sum([len(mention_pos_arr) for mention_pos_arr in entity_pos_arr])
        for i in range(current_num_mention):
            for j in range(current_num_mention):
                if i == j:
                    continue
                cr_mask[batch_idx, i, j] = 1
    return cr_matrix, cr_mask


def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    # Bert input
    input_ids = [f["input_ids"] + [0 for _ in range(max_len - len(f["input_ids"]))] for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)

    # NER label
    ner_labels = [f['ner_labels'] + ['O' for _ in range(max_len - len(f["input_ids"]))] for f in batch]
    ner_labels = [[ner_vocab[elem] for elem in ner_label] for ner_label in ner_labels]
    ner_labels = torch.tensor(ner_labels, dtype=torch.long)
    ner_labels = torch.tensor(ner_labels)

    labels = [f["labels"] for f in batch]
    batch_entity_pos = [f["entity_pos"] for f in batch]
    batch_sent_pos = [f['sent_pos'] for f in batch]
    hts = [f["hts"] for f in batch]

    graph, num_mention, num_entity, num_sent = graph_builder.create_graph(batch_entity_pos,
                                                                          batch_sent_pos)
    cr_matrix, cr_mask = gen_cr_matrix(batch_entity_pos, num_mention)
    entity_type = [[entity_type_vocab[t] for t in f["entity_type"]] +
                   [0 for _ in range(num_entity - len(f["entity_type"]))] for f in batch]
    entity_type = torch.tensor(entity_type)
    entity_mask = torch.tensor([[1 for _ in f["entity_type"]] +
                                [0 for _ in range(num_entity - len(f["entity_type"]))] for f in batch])
    output = (input_ids, input_mask,
              batch_entity_pos, batch_sent_pos,
              cr_matrix, cr_mask,
              graph, num_mention, num_entity, num_sent,
              labels, ner_labels,
              entity_type, entity_mask, hts)
    return output
