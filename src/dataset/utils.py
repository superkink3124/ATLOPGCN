from tqdm import tqdm


cdr_rel2id = {'1:NR:2': 0, '1:CID:2': 1}


def chunks(l, n):
    res = []
    for i in range(0, len(l), n):
        assert len(l[i:i + n]) == n
        res += [l[i:i + n]]
    return res


def read_cdr(file_in, tokenizer, max_seq_length=1024) -> List[Any]:
    pmids = set()
    features = []
    maxlen = 0
    with open(file_in, 'r') as infile:
        lines = infile.readlines()
        for i_l, line in enumerate(tqdm(lines)):
            line = line.rstrip().split('\t')
            pmid = line[0]

            if pmid not in pmids:
                pmids.add(pmid)
                text = line[1]
                prs = chunks(line[2:], 17)

                ent2idx = {}
                train_triples = {}

                entity_pos = set()
                for p in prs:
                    es = list(map(int, p[8].split(':')))
                    ed = list(map(int, p[9].split(':')))
                    tpy = p[7]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                    es = list(map(int, p[14].split(':')))
                    ed = list(map(int, p[15].split(':')))
                    tpy = p[13]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                sents = [t.split(' ') for t in text.split('|')]
                new_sents = []
                sent_map = {}
                i_t = 0
                for sent in sents:
                    for token in sent:
                        tokens_wordpiece = tokenizer.tokenize(token)
                        for start, end, tpy in list(entity_pos):
                            if i_t == start:
                                tokens_wordpiece = ["*"] + tokens_wordpiece
                            if i_t + 1 == end:
                                tokens_wordpiece = tokens_wordpiece + ["*"]
                        sent_map[i_t] = len(new_sents)
                        new_sents.extend(tokens_wordpiece)
                        i_t += 1
                    sent_map[i_t] = len(new_sents)
                sents = new_sents

                entity_pos = []

                for p in prs:
                    if p[0] == "not_include":
                        continue
                    if p[1] == "L2R":
                        h_id, t_id = p[5], p[11]
                        h_start, t_start = p[8], p[14]
                        h_end, t_end = p[9], p[15]
                    else:
                        t_id, h_id = p[5], p[11]
                        t_start, h_start = p[8], p[14]
                        t_end, h_end = p[9], p[15]
                    h_start = map(int, h_start.split(':'))
                    h_end = map(int, h_end.split(':'))
                    t_start = map(int, t_start.split(':'))
                    t_end = map(int, t_end.split(':'))
                    h_start = [sent_map[idx] for idx in h_start]
                    h_end = [sent_map[idx] for idx in h_end]
                    t_start = [sent_map[idx] for idx in t_start]
                    t_end = [sent_map[idx] for idx in t_end]
                    if h_id not in ent2idx:
                        ent2idx[h_id] = len(ent2idx)
                        entity_pos.append(list(zip(h_start, h_end)))
                    if t_id not in ent2idx:
                        ent2idx[t_id] = len(ent2idx)
                        entity_pos.append(list(zip(t_start, t_end)))
                    h_id, t_id = ent2idx[h_id], ent2idx[t_id]

                    r = cdr_rel2id[p[0]]
                    if (h_id, t_id) not in train_triples:
                        train_triples[(h_id, t_id)] = [{'relation': r}]
                    else:
                        train_triples[(h_id, t_id)].append({'relation': r})

                relations, hts = [], []
                for h, t in train_triples.keys():
                    relation = [0] * len(cdr_rel2id)
                    for mention in train_triples[h, t]:
                        relation[mention["relation"]] = 1
                    relations.append(relation)
                    hts.append([h, t])

            maxlen = max(maxlen, len(sents))
            sents = sents[:max_seq_length - 2]
            input_ids = tokenizer.convert_tokens_to_ids(sents)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

            if len(hts) > 0:
                feature = {'input_ids': input_ids,
                           'entity_pos': entity_pos,
                           'labels': relations,
                           'hts': hts,
                           'title': pmid,
                           }
                features.append(feature)
    print("Number of documents: {}.".format(len(features)))
    print("Max document length: {}.".format(maxlen))
    return features


# def pad_sequences(list_token_ids: Tuple[List[List[int]], ...],
#                   max_num_sents,
#                   max_sent_length, pad_idx) -> Tuple[Tensor, Tensor]:
#     batch_size = len(list_token_ids)
#     padded_seq_mask: List[List[List[int]]] = [[[0 for _ in range(max_sent_length)]
#                                                for _ in range(max_num_sents)]
#                                               for _ in range(batch_size)]
#     padded_token_ids: List[List[List[int]]] = [[] for _ in range(batch_size)]
#
#     for batch_id, list_sent in enumerate(list_token_ids):
#         for sent_id, sent in enumerate(list_sent):
#             for i in range(len(sent)):
#                 padded_seq_mask[batch_id][sent_id][i] = 1
#             padded_token_ids[batch_id].append([token for token in sent])
#             padded_token_ids[batch_id][-1].extend([pad_idx
#                                                         for _ in range(max_sent_length - len(sent))])
#
#         if len(list_sent) < max_num_sents:
#             padded_token_ids[batch_id].extend([[pad_idx for _ in range(max_sent_length)]
#                                                     for _ in range(max_num_sents - len(list_sent))])
#
#     padded_token_ids: Tensor = torch.tensor(padded_token_ids)
#     padded_seq_mask: Tensor = torch.tensor(padded_seq_mask)
#     return padded_token_ids, padded_seq_mask
#
#
# def pad_characters(list_char_ids: Tuple[List[List[List[int]]], ...],
#                    max_num_sents,
#                    max_sent_length,
#                    max_char_length,
#                    pad_idx):
#     raise NotImplementedError()
#
#
# def get_cdr_dataset(corpus: CDRCorpus, saved_folder_path: str, data_type: str):
#     (
#         dict_token_ids,
#         dict_pos_ids,
#         dict_char_ids,
#         dict_sent_list,
#         dict_entity_mapping,
#         ner_labels,
#         labels
#     ) = corpus.load_all_features_for_one_dataset(saved_folder_path, data_type)
#     dataset = CDRDataset(dict_token_ids,
#                          dict_pos_ids,
#                          dict_char_ids,
#                          dict_sent_list,
#                          dict_entity_mapping,
#                          ner_labels,
#                          labels)
#     return dataset
#
#
# def concat_dataset(datasets: List[CDRDataset]):
#     res = CDRDataset(
#         {k: v for dataset in datasets for k, v in dataset.dict_token_ids.items()},
#         {k: v for dataset in datasets for k, v in dataset.dict_pos_ids.items()},
#         {k: v for dataset in datasets for k, v in dataset.dict_char_ids.items()},
#         {k: v for dataset in datasets for k, v in dataset.dict_sent_list.items()},
#         {k: v for dataset in datasets for k, v in dataset.dict_entity_mapping.items()},
#         {k: v for dataset in datasets for k, v in dataset.ner_labels.items()},
#         [label for dataset in datasets for label in dataset.labels],
#     )
#     return res
