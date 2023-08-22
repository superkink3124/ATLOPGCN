from typing import Dict, List, Tuple, Any
from torch.utils.data import Dataset
from utils.constansts import ner_vocab, CHEMICAL_STRING, DISEASE_STRING


class CDRDataset(Dataset):
    def __init__(
            self,
            dict_token_ids: Dict[str, List[List[int]]],
            dict_pos_ids: Dict[str, List[List[int]]],
            dict_char_ids: Dict[str, List[List[List[int]]]],
            dict_sent_list: Dict[str, List[Tuple[int, int]]],
            dict_entity_mapping: Dict[str, Dict[Tuple[str, str], List[Tuple[int, List[int]]]]],
            ner_labels: Dict[str, List[List[int]]],
            labels: List[Tuple[str, str, str, str]],
    ):
        super(CDRDataset, self).__init__()

        self.ner_labels = ner_labels
        self.dict_entity_mapping = dict_entity_mapping
        self.dict_sent_list = dict_sent_list
        self.dict_char_ids = dict_char_ids
        self.dict_pos_ids = dict_pos_ids
        self.dict_token_ids = dict_token_ids
        self.labels = labels

        self.ner_vocab = ner_vocab
        self.pudid_list: List[str] = list(self.dict_token_ids.keys())
        self.pudid_rels: Dict[str, List[Tuple[str, str, str]]] = dict()
        for label in labels:
            pud_id, c_id, d_id, rel = label
            if pud_id not in self.pudid_rels:
                self.pudid_rels[pud_id] = []
            self.pudid_rels[pud_id].append((c_id, d_id, rel))

        # Ignore doc that does not have any cid relation
        # self.pudid_list = [pud_id for p_id in self.pudid_rels if len([rel for rel in self.pudid_rels[pud_id]
        #                                                               if rel[2] == "CID"]) > 0]

    def __len__(self):
        return len(self.pudid_list)

    def __getitem__(self, idx):
        pud_id = self.pudid_list[idx]
        rels = self.pudid_rels[pud_id]
        token_ids = self.dict_token_ids[pud_id]
        pos_ids = self.dict_pos_ids[pud_id]
        char_ids = self.dict_char_ids[pud_id]
        sent_list = self.dict_sent_list[pud_id]
        assert len(char_ids) == len(pos_ids) == len(token_ids)
        ner_label_ids = self.ner_labels[pud_id]
        chemical_list = list(set([entity[1] for entity in self.dict_entity_mapping[pud_id]
                                  if entity[0] == CHEMICAL_STRING]))
        disease_list = list(set([entity[1] for entity in self.dict_entity_mapping[pud_id]
                                 if entity[0] == DISEASE_STRING]))
        return (
            token_ids,
            pos_ids,
            char_ids,
            sent_list,
            self.dict_entity_mapping[pud_id],
            chemical_list,
            disease_list,
            ner_label_ids,
            rels,
        )
