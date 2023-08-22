import torch
from typing import Tuple, List, Dict
from torch import Tensor


def get_sentence_to_sentence_edges(num_sent: int,
                                   list_sent_list: Tuple[List[Tuple[int, int]]]
                                   ) -> Tuple[List[int], List[int]]:
    u = []
    v = []
    for batch_id, sent_list in enumerate(list_sent_list):
        for sent_1_id in range(len(sent_list)):
            for sent_2_id in range(len(sent_list)):
                if sent_1_id == sent_2_id:
                    continue
                u.append(get_id(num_sent, batch_id, sent_1_id))
                v.append(get_id(num_sent, batch_id, sent_2_id))
    return u, v


def get_mention_to_mention_edges(num_mention: int,
                                 list_mention_idx: List[Dict[Tuple[str, int, Tuple[int, ...]], int]]
                                 ) -> Tuple[List[int], List[int]]:
    u = []
    v = []

    for batch_id, mention_idx in enumerate(list_mention_idx):
        for mention1 in mention_idx:
            for mention2 in mention_idx:
                if mention1 == mention2:
                    continue
                if mention1[1] == mention2[1]:
                    u.append(get_id(num_mention, batch_id, mention_idx[mention1]))
                    v.append(get_id(num_mention, batch_id, mention_idx[mention2]))
    return u, v


def get_mention_to_entity_edges(num_mention: int, num_entity: int,
                                list_mention_idx: List[Dict[Tuple[str, int, Tuple[int, ...]], int]],
                                list_entity_idx: List[Dict[str, int]]) -> Tuple[List[int], List[int]]:
    u = []
    v = []
    for batch_id, entity_idx in enumerate(list_entity_idx):
        mention_idx = list_mention_idx[batch_id]
        for mention in mention_idx:
            entity = mention[0]
            u.append(get_id(num_mention, batch_id, mention_idx[mention]))
            v.append(get_id(num_entity, batch_id, entity_idx[entity]))
    return u, v


def get_mention_to_sentence_edges(num_mention: int,
                                  num_sent: int,
                                  list_mention_idx: List[Dict[Tuple[str, int, Tuple[int, ...]], int]]) -> \
        Tuple[List[int], List[int]]:
    u = []
    v = []
    for batch_id, mention_idx in enumerate(list_mention_idx):
        for mention, mention_id in mention_idx.items():
            sent_id = mention[1]
            u.append(get_id(num_mention, batch_id, mention_idx[mention]))
            v.append(get_id(num_sent, batch_id, sent_id))
    return u, v


def get_entity_to_sentence_edges(num_entity: int,
                                 num_sent: int,
                                 list_mention_idx: List[Dict[Tuple[str, int, Tuple[int, ...]], int]],
                                 list_entity_idx: List[Dict[str, int]]) -> \
        Tuple[List[int], List[int]]:
    u = []
    v = []
    for batch_id, mention_idx in enumerate(list_mention_idx):
        entity_idx = list_entity_idx[batch_id]
        for mention in mention_idx:
            u.append(get_id(num_entity, batch_id, entity_idx[mention[0]]))
            v.append(get_id(num_sent, batch_id, mention[1]))
    return u, v


def get_id(num_col: int, row_idx: int, col_idx: int) -> int:
    return num_col * row_idx + col_idx
