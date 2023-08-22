import dgl
from .graph_builder_utils import get_mention_to_sentence_edges, get_entity_to_sentence_edges, \
    get_mention_to_entity_edges, get_sentence_to_sentence_edges, \
    get_mention_to_mention_edges
import torch
from typing import Dict, List, Tuple

class GraphBuilder:
    def __init__(self,
                 create_undirected_edges: bool = True,
                 add_self_edge: bool = True):
        self.create_undirected_edges = create_undirected_edges
        self.add_self_edge = add_self_edge

    def create_graph(self, num_mention, num_entity, num_sent,
                     list_mention_idx: List[Dict[Tuple[str, int, Tuple[int, ...]], int]],
                     list_entity_idx: List[Dict[str, int]],
                     list_sent_list: Tuple[List[Tuple[int, int]], ...]):
        mention_to_mention_edges = get_mention_to_mention_edges(num_mention, list_mention_idx)
        sentence_to_sentence_edges = get_sentence_to_sentence_edges(num_sent, list_sent_list)
        mention_to_sentence_edges = get_mention_to_sentence_edges(num_mention, num_sent,
                                                                  list_mention_idx)
        mention_to_entity_edges = get_mention_to_entity_edges(num_mention, num_entity, list_mention_idx,
                                                              list_entity_idx)
        entity_to_sentence_edges = get_entity_to_sentence_edges(num_entity, num_sent,
                                                                list_mention_idx, list_entity_idx)
        u = []
        v = []
        batch_size = len(list_sent_list)

        def get_new_entity_id(origin_entity_id):
            return num_mention * batch_size + origin_entity_id

        def get_new_sent_id(origin_sent_id):
            return num_mention * batch_size + num_entity * batch_size + origin_sent_id

        edge_u, edge_v = mention_to_mention_edges
        for edge_id in range(len(edge_u)):
            u.append(edge_u[edge_id])
            v.append(edge_v[edge_id])

        edge_u, edge_v = sentence_to_sentence_edges
        for edge_id in range(len(edge_u)):
            u.append(get_new_sent_id(edge_u[edge_id]))
            v.append(get_new_sent_id(edge_v[edge_id]))

        edge_u, edge_v = mention_to_sentence_edges
        for edge_id in range(len(edge_u)):
            u.append(edge_u[edge_id])
            v.append(get_new_sent_id(edge_v[edge_id]))
            if self.create_undirected_edges:
                v.append(edge_u[edge_id])
                u.append(get_new_sent_id(edge_v[edge_id]))

        edge_u, edge_v = mention_to_entity_edges
        for edge_id in range(len(edge_u)):
            u.append(edge_u[edge_id])
            v.append(get_new_entity_id(edge_v[edge_id]))
            if self.create_undirected_edges:
                u.append(edge_u[edge_id])
                v.append(get_new_entity_id(edge_v[edge_id]))

        edge_u, edge_v = entity_to_sentence_edges
        for edge_id in range(len(edge_u)):
            u.append(get_new_entity_id(edge_u[edge_id]))
            v.append(get_new_sent_id(edge_v[edge_id]))
            if self.create_undirected_edges:
                u.append(get_new_entity_id(edge_u[edge_id]))
                v.append(get_new_sent_id(edge_v[edge_id]))

        for edge_id in range(len(u)):
            assert u[edge_id] != v[edge_id], f"Exist self edge {u[edge_id]} to {v[edge_id]}"

        graph = dgl.graph((torch.tensor(u), torch.tensor(v)), num_nodes=(num_mention * batch_size +
                                                                         num_entity * batch_size +
                                                                         num_sent * batch_size))
        if self.add_self_edge:
            graph = dgl.add_self_loop(graph)
        return graph
