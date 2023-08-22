"""This module describe how we prepare the training, development and testing dataset from Biocreative CDR5 corpus."""
import codecs
import itertools
import json
import os
import pickle
import random
from collections import defaultdict
from itertools import groupby
from typing import Any, Dict, List, Tuple
import numpy as np
import torch

from config.cdr_config import CDRConfig
from utils.spacy_nlp import nlp
from utils.constansts import CHEMICAL_STRING, DISEASE_STRING, UNK, PAD

ner_vocab = {"O": 0, "B_Chemical": 1, "I_Chemical": 2, "B_Disease": 3, "I_Disease": 4}
ner_idx2label = {0: "O", 1: "B_Chemical", 2: "I_Chemical", 3: "B_Disease", 4: "I_Disease"}
# idx2word = {k: v for v, k in word_vocab.items()}
ADJACENCY_REL = "node"
ROOT_REL = "root"
SELF_REL = "self"

LabelAnnotationType = Dict[Tuple[str, str, str], str]
DocAnnotationType = Dict[str, Tuple[str, str, List[Tuple[int, int, str, str, str]]]]
Doc = Any
Token = Any


# TODO: unique entity mention mapping (just 1 case)
class CDRCorpus:
    def __init__(self, config: CDRConfig):
        """[summary]

        Args:
            config ([type]): [description]
        """
        self.config = config
        self.word_vocab: Dict[str, int] = {}
        self.pos_vocab: Dict[str, int] = {}
        self.char_vocab: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.list_feature_names = [
            "dict_token_ids.pkl",
            "dict_pos_ids.pkl",
            "dict_char_ids.pkl",
            "dict_sent_list.pkl",
            "dict_entity_mapping.pkl",
            # "dict_entity_annotation.pkl",
            "ner_labels.pkl",
            "labels.pkl"
        ]
        self.list_vocab_names = ["word_vocab.json", "pos_vocab.json", "char_vocab.json"]

    def load_vocab(self, file_path) -> Dict[str, int]:
        with open(file_path) as f:
            vocab = json.load(f)
            return vocab

    def save_vocab(self, vocab, file_path):
        with open(file_path, "w") as f:
            json.dump(vocab, f)

    def load_tensor(self, tensor_file_path) -> torch.Tensor:
        with open(tensor_file_path, "rb") as f:
            tensor = pickle.load(f)
            return tensor

    def load_numpy(self, numpy_file_path) -> np.ndarray:
        with open(numpy_file_path, "rb") as f:
            matrix = np.load(f)
            return matrix

    def save_feature(self, feature, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(feature, f)

    def load_feature(self, file_path):
        with open(file_path, "rb") as f:
            feature = pickle.load(f)
        return feature

    def load_all_vocabs(self, saved_folder_path):
        if os.path.exists(os.path.join(saved_folder_path, self.list_vocab_names[0])):
            self.word_vocab = self.load_vocab(os.path.join(saved_folder_path, self.list_vocab_names[0]))
            self.pos_vocab = self.load_vocab(os.path.join(saved_folder_path, self.list_vocab_names[1]))
            self.char_vocab = self.load_vocab(os.path.join(saved_folder_path, self.list_vocab_names[2]))
            self.id_to_word = {v: k for k, v in self.word_vocab.items()}
        else:
            raise Exception(
                "You have not prepared the vocabs. Please prepare them and the features by running the build_data scipt"
            )

    def load_all_features_for_one_dataset(self, saved_folder_path: str, data_type: str) -> List[Any]:
        list_features = []
        for feature_name in self.list_feature_names:
            feature = self.load_feature(os.path.join(saved_folder_path, data_type, feature_name))
            list_features.append(feature)
        return list_features

    def prepare_features_for_one_dataset(self, data_file_path: str, saved_folder_path: str, data_type: str):
        if not os.path.exists(os.path.join(saved_folder_path, data_type)):
            os.mkdir(os.path.join(saved_folder_path, data_type))
        self.load_all_vocabs(saved_folder_path)
        entity_mapping_dict, labels, doc_dict, entity_annotation_dict = self.process_dataset(data_file_path)
        features = self.convert_examples_to_features(
            doc_dict,
            entity_mapping_dict,
            entity_annotation_dict,
        )
        features = list(features)
        features.append(labels)

        print("Saving generated features .......")

        for feature_name, feature in list(zip(self.list_feature_names, features)):
            self.save_feature(feature, os.path.join(saved_folder_path, data_type, feature_name))

    def prepare_all_vocabs(self, saved_folder_path) -> None:
        """[summary]

        Returns:
            [type]: [description]
        """
        (
            train_entity_mapping_dict,
            train_all_labels,
            train_doc_dict,
            _
        ) = self.process_dataset(self.config.data.train_file_path)
        (
            dev_entity_mapping_dict,
            dev_all_labels,
            dev_doc_dict,
            _
        ) = self.process_dataset(self.config.data.dev_file_path)
        (
            test_entity_mapping_dict,
            test_all_labels,
            test_doc_dict,
            _
        ) = self.process_dataset(self.config.data.test_file_path)

        print("Saving vocabs .......")
        vocabs = self.create_vocabs([train_doc_dict, dev_doc_dict, test_doc_dict])
        for vocab_name, vocab in list(zip(self.list_vocab_names, vocabs)):
            self.save_vocab(vocab, os.path.join(saved_folder_path, vocab_name))

    def make_pairs(self, entity_annotations: List[Tuple[int, int, str, str, str]]) -> List[Tuple[str, str]]:
        """[summary]

        Args:
            entity_annotations (List[Tuple[int, int, str, str, str]]): [description]

        Returns:
            List[Tuple[Any, Any]]: [description]
        """
        chem_entity_ids = [anno[-1] for anno in entity_annotations if anno[-2] == CHEMICAL_STRING]
        dis_entity_ids = [anno[-1] for anno in entity_annotations if anno[-2] == DISEASE_STRING]

        chem_entity_ids = list(set(chem_entity_ids))
        dis_entity_ids = list(set(dis_entity_ids))

        chem_dis_pair_ids = list(itertools.product(chem_entity_ids, dis_entity_ids))

        return chem_dis_pair_ids

    def get_valid_entity_mentions(
            self, entity_mentions_annotations: List[Tuple[int, int, str, str, str]], invalid_id: str = "-1"
    ) -> List[Tuple[Any, Any, Any, Any, Any]]:
        """Remove all entity which has unknown id.

        Args:
            entity_mentions_annotations (List[Tuple[int, int, str, str, str]]): list of entity mention annotations,
            whose each element is a tuple of (start_offset, end_offset, text, entity type, mesh_id).
            invalid_id (int, optional): The unknown entity id from CDR5. Defaults to '-1'.

        Returns:
            [type]: [description]
        """

        # remove entity anno in document's title and entity with id = -1
        return [mention_anno for mention_anno in entity_mentions_annotations if mention_anno[-1] != invalid_id]

    def remove_entity_mention_in_title(
            self, entity_mentions_annotations: List[Tuple[Any, Any, Any, Any, Any]], title
    ) -> List[Tuple[Any, Any, Any, Any, Any]]:
        """[summary]

        Args:
            entity_mentions_annotations (List[Tuple[Any, Any, Any, Any, Any]]): [description]
            title ([type]): [description]

        Returns:
            List[Tuple[Any, Any, Any, Any, Any]]: [description]
        """
        return [mention_anno for mention_anno in entity_mentions_annotations if int(mention_anno[1]) >= len(title)]

    def read_raw_dataset(self, file_path: str) -> Tuple[LabelAnnotationType, DocAnnotationType]:
        """Read the raw biocreative CDR5 dataset

        Args:
            file_path (str): path to the dataset

        Returns:
            Tuple[Label_annotation_type, Doc_annotation_type]: A tuple of two dictionary, the label
            annotation whose each key is a tuple of (chemical_mesh_id, disease_mesh_id, document_id)
            and its value is the relation (eg: CID or None)
            the document annotation contains key, values pairs with key is the document id
            and value is a list whose elements are the document title, abstract and
            list of entity mention annotations respectively.
        """

        with open(file_path) as f_raw:
            lines = f_raw.read().split("\n")

            raw_doc_annotations = [list(group) for k, group in groupby(lines, lambda x: x == "") if not k]
            label_annotations = {}
            doc_annotations = {}

            for doc_annos in raw_doc_annotations:

                title = None
                abstract = None
                current_annotations = []

                for anno in doc_annos:

                    if "|t|" in anno:
                        pud_id, title = anno.strip().split("|t|")
                    elif "|a|" in anno:
                        _, abstract = anno.strip().split("|a|")
                    else:
                        splits = anno.strip().split("\t")
                        if len(splits) == 4:
                            _, rel, e1_id, e2_id = splits
                            label_annotations[(e1_id, e2_id, pud_id)] = rel
                        elif len(splits) == 6:
                            _, start, end, mention, label, kg_ids = splits
                            for kg_id in kg_ids.split("|"):
                                current_annotations.append((int(start), int(end), mention, label, kg_id))
                        elif len(splits) == 7:
                            _, start, end, mention, label, kg_ids, split_mentions = splits
                            for kg_id in kg_ids.split("|"):
                                current_annotations.append((int(start), int(end), mention, label, kg_id))

                assert title is not None and abstract is not None
                doc_annotations[pud_id] = (title, abstract, current_annotations)
            return label_annotations, doc_annotations

    def create_features_one_doc(self, pud_id: str, abstract: str,
                                entity_annotations: List[Tuple[int, int, str, str, str]]) \
            -> Tuple[Dict[Tuple[int, int, str, str, str], Token], Doc]:
        # if pud_id == "17854040":
        #     print(entity_annotations)
        #     print("_" * 30)
        #     print(abstract)
        # sentence tokenize
        doc: Doc = nlp(abstract)
        entity_mapping = {}
        for en_anno in entity_annotations:
            start, end, mention, label, kg_id = en_anno
            key = (start, end, mention, label, kg_id)
            entity_mapping[key] = []
            for token in doc:
                token_start = token.idx
                token_end = token_start + len(token)
                if token_start >= start and token_end <= end:
                    entity_mapping[key].append(token)
                # some annotations which form abc-#mention-abcxyz, so we extra token spans to a pre-defined threshhold.
                # elif (token_start >= start - offset_span and token_end <= end + offset_span) and mention in token.text:
                #     entity_mapping[key].append(token)
                # # hard code for some specific mention
                # elif token.text in SOME_SPECIFIC_MENTIONS:
                #     entity_mapping[key].append(token)
            if len(entity_mapping[key]) == 0:
                for token in doc:
                    token_start = token.idx
                    token_end = token_start + len(token)
                    if (token_start <= start and token_end >= end) and \
                            mention in token.text:
                        entity_mapping[key].append(token)
                        break
            # if len(entity_mapping[key]) == 0:
            #     for token in doc:
            #         if token.text in SOME_SPECIFIC_MENTIONS and mention in token.text:
            #             entity_mapping[key].append(token)
            #             break
            try:
                assert entity_mapping[key] != []
            except:
                print(en_anno)
                print(pud_id)
                print(abstract[start - 50: end + 50])
                raise Exception("Cannot map entity")
        # if pud_id == "12617329":
        #     print("_"*40)
        #     print(entity_mapping)
        #     print("_"*40)
        #     print(doc)
        #     print("_"*40)
        #     raise Exception("")
        return entity_mapping, doc

    def preprocess_one_doc(self, pud_id: str, title: str, abstract: str,
                           entity_annotations: List[Tuple[int, int, str, str, str]]) \
            -> Tuple[List[Tuple[str, str]], Dict[Tuple[int, int, str, str, str], Token], Doc]:
        # remove all annotations of invalid enity (i.e entity id equals -1)
        entity_annotations = self.get_valid_entity_mentions(entity_annotations)
        if not self.config.data.use_title:
            # subtract title offset plus one space
            for en_anno in entity_annotations:
                en_anno[0] -= len(title) + 1
                en_anno[1] -= len(title) + 1
            # remove entity mention in the title
            entity_annotations = self.remove_entity_mention_in_title(entity_annotations, title)

        # make all pairs chemical disease entities
        chem_dis_pair_ids = self.make_pairs(entity_annotations)
        # create doc_adjacency_dict and entity_to_tokens_mapping
        entity_mapping, doc = self.create_features_one_doc(
            pud_id, title + " " + abstract if self.config.data.use_title else abstract, entity_annotations
        )

        # print(chem_dis_pair_ids)
        return chem_dis_pair_ids, entity_mapping, doc

    def process_dataset(self, file_path: str) -> \
            [Dict[str, Dict[Tuple[int, int, str, str, str], Token]],
             List[Tuple[str, str, str, str]], Dict[str, Doc], Any]:
        """[summary]

        Args:
            file_path ([type]): [description]

        Returns:
            [type]: [description]
        """
        label_annotations, doc_annotations = self.read_raw_dataset(file_path)
        label_docs = defaultdict(list)
        entity_mapping_dict = {}
        doc_dict = {}
        entity_annotation_dict = dict()
        # process all document
        for pud_id, doc_anno in doc_annotations.items():
            title, abstract, entity_annotations = doc_anno
            (
                chem_dis_pair_ids,
                entity_mapping,
                doc,
            ) = self.preprocess_one_doc(pud_id, title, abstract, entity_annotations)
            doc_dict[pud_id] = doc
            label_docs[pud_id] = chem_dis_pair_ids
            entity_mapping_dict[pud_id] = entity_mapping
            entity_annotation_dict[pud_id] = entity_annotations
        # gather positive examples and negative examples
        pos_doc_examples = defaultdict(list)
        neg_doc_examples = defaultdict(list)
        for pud_id in doc_annotations.keys():
            for c_e, d_e in label_docs[pud_id]:
                if (c_e, d_e, pud_id) in label_annotations:
                    pos_doc_examples[pud_id].append((c_e, d_e))
                else:
                    neg_doc_examples[pud_id].append((c_e, d_e))
        if self.config.data.mesh_filtering:
            ent_tree_map = defaultdict(list)
            with codecs.open(self.config.data.mesh_path, "r", encoding="utf-16-le") as f:
                lines = [l.rstrip().split("\t") for i, l in enumerate(f) if i > 0]
                [ent_tree_map[l[1]].append(l[0]) for l in lines]
                neg_doc_examples, n_filtered_samples = self.filter_with_mesh_vocab(
                    ent_tree_map, pos_doc_examples, neg_doc_examples
                )

            print("number of negative examples are filtered:", n_filtered_samples)

        all_labels = []
        for pud_id, value in pos_doc_examples.items():
            for c_id, d_id in value:
                key = (pud_id, c_id, d_id, "CID")
                all_labels.append(key)

        for pud_id, value in neg_doc_examples.items():
            for c_id, d_id in value:
                key = (pud_id, c_id, d_id, "NULL")
                all_labels.append(key)

        random.shuffle(all_labels)
        print("total samples: ", len(all_labels))
        return entity_mapping_dict, all_labels, doc_dict, entity_annotation_dict

    def filter_with_mesh_vocab(self, mesh_tree, pos_doc_examples, neg_doc_examples):
        """[summary]

        Args:
            mesh_tree ([type]): [description]
            pos_doc_examples ([type]): [description]
            neg_doc_examples ([type]): [description]

        Returns:
            [type]: [description]
        """
        neg_filterd_exampled = defaultdict(list)
        n_filterd_samples = 0
        negative_count = 0
        hypo_count = 0
        # i borrowed this code from https://github.com/patverga/bran/blob/master/src/processing/utils/filter_hypernyms.py
        for doc_id in neg_doc_examples.keys():
            # get nodes for all the positive diseases
            pos_e2_examples = [(pos_node, pe) for pe in pos_doc_examples[doc_id] for pos_node in mesh_tree[pe[1]]]
            # chemical
            pos_e1_examples = [(pos_node, pe) for pe in pos_doc_examples[doc_id] for pos_node in mesh_tree[pe[0]]]

            for ne in neg_doc_examples[doc_id]:
                neg_e1 = ne[0]
                neg_e2 = ne[1]
                example_hyponyms = 0
                for neg_node in mesh_tree[ne[1]]:
                    hyponyms = [
                        pos_node for pos_node, pe in pos_e2_examples if neg_node in pos_node and neg_e1 == pe[0]
                    ]
                    example_hyponyms += len(hyponyms)
                if example_hyponyms == 0:
                    negative_count += 1
                    neg_filterd_exampled[doc_id].append((neg_e1, neg_e2))
                else:
                    hypo_count += example_hyponyms
                    n_filterd_samples += 1

        return neg_filterd_exampled, n_filterd_samples

    def create_vocabs(self, list_doc_dict: List[Dict[str, Doc]]) -> \
            Tuple[Dict[str, int],
            Dict[str, int],
            Dict[str, int]]:

        list_words = []
        list_poses = []
        list_chars = []

        for doc_dict in list_doc_dict:
            for pud_id in doc_dict.keys():
                for token in doc_dict[pud_id]:
                    list_words.append(token.text)
                    list_poses.append(token.tag_)
                    list_chars.extend(list(token.text))

        word_vocab = list(set(list_words))
        word_vocab.append(UNK)
        word_vocab.append(PAD)

        word_vocab = {value: key for key, value in enumerate(word_vocab)}

        pos_vocab = list(set(list_poses))
        pos_vocab.append(PAD)
        pos_vocab = {value: key for key, value in enumerate(pos_vocab)}

        char_vocab = list(set(list_chars))
        char_vocab.append(PAD)
        char_vocab = {value: key for key, value in enumerate(char_vocab)}

        print(f"word vocab: {len(word_vocab)} unique words")
        print(f"char vocab: {len(char_vocab)} unique characters")
        print(f"pos vocab: {len(pos_vocab)} unique POS tags")

        return word_vocab, pos_vocab, char_vocab

    def convert_examples_to_features(
            self,
            doc_dict: Dict[str, Doc],
            entity_mapping_dicts: Dict[str, Dict[Tuple[int, int, str, str, str], Any]],
            dict_entity_annotation: Dict[str, Any]
    ):
        dict_token_ids: Dict[str, List[List[int]]] = {}
        dict_pos_ids: Dict[str, List[List[int]]] = {}
        dict_char_ids: Dict[str, List[List[List[int]]]] = {}
        dict_entity_mapping: Dict[str, Dict[Tuple[str, str], List[Tuple[int, List[int]]]]] = {}
        dict_sent_list: Dict[str, List[Tuple[int, int]]] = {}
        max_char_length = -1
        max_entity_span = -1
        max_mentions = -1
        dict_ner_labels: Dict[str, List[List[int]]] = {}

        for pud_id, doc in doc_dict.items():
            token_ids = [[self.word_vocab[tok.text] for tok in sent] for sent in doc.sents]
            pos_ids = [[self.pos_vocab[tok.tag_] for tok in sent] for sent in doc.sents]
            char_ids = [[[self.char_vocab[char] for char in tok.text] for tok in sent] for sent in doc.sents]
            max_char_length = max(max_char_length, max([len(tok.text) for tok in doc]))
            sent_list = []
            num_tokens = 0
            for sent in doc.sents:
                sent_list.append((num_tokens, num_tokens + len(sent) - 1))
                num_tokens += len(sent)
            dict_token_ids[pud_id] = token_ids
            dict_pos_ids[pud_id] = pos_ids
            dict_char_ids[pud_id] = char_ids
            dict_sent_list[pud_id] = sent_list

        for pud_id, doc in doc_dict.items():
            sent_start_ids = []
            for sent in doc.sents:
                sent_start_ids.append(sent[0].i)

            def find_token_position(token_id):
                for sent_id, start_token_id in enumerate(sent_start_ids):
                    if start_token_id > token_id:
                        return sent_id - 1, sent_start_ids[sent_id - 1]
                return len(sent_start_ids) - 1, sent_start_ids[-1]

            entity_mapping: Dict[Tuple[str, str], List[Tuple[int, List[int]]]] = defaultdict(list)
            for mention in entity_mapping_dicts[pud_id]:
                _, _, _, entity_type, entity_id = mention
                mention_tokens = entity_mapping_dicts[pud_id][mention]
                sent_id, start_token_id = find_token_position(mention_tokens[0].i)
                positions = []
                max_entity_span = max(max_entity_span, len(mention_tokens))
                for token in mention_tokens:
                    positions.append(token.i - start_token_id)
                entity_mapping[(entity_type, entity_id)].append((sent_id, positions))
                max_entity_span = max(max_entity_span, len(positions))
                max_mentions = max(max_mentions, len(entity_mapping[(entity_type, entity_id)]))
            dict_entity_mapping[pud_id] = entity_mapping
        dict_entity_mapping = dict(dict_entity_mapping)

        for pud_id, doc in doc_dict.items():
            doc_ner_labels = [[ner_vocab["O"] for tok in sent] for sent in doc.sents]
            for key, value in dict_entity_mapping[pud_id].items():
                en_type, en_id = key
                for mention in value:
                    sent_id, positions = mention
                    doc_ner_labels[sent_id][positions[0]] = ner_vocab[f"B_{en_type}"]
                    for position in positions[1:]:
                        doc_ner_labels[sent_id][position] = ner_vocab[f"I_{en_type}"]
            dict_ner_labels[pud_id] = doc_ner_labels

        print("max entity spans: ", max_entity_span)
        print("max entity mentions: ", max_mentions)
        print("max characters length: ", max_char_length)
        return (
            dict_token_ids,
            dict_pos_ids,
            dict_char_ids,
            dict_sent_list,
            dict_entity_mapping,
            dict_ner_labels,
        )
