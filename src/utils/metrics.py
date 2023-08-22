from typing import List
from torch import Tensor
from logging import Logger

from utils.constansts import ner_idx2label


def compute_NER_f1_macro(label_pred, label_correct):

    f1_macro = 0.0
    labels = ["B_Chemical", "B_Disease"]

    f1_per_labels = {}

    for label in labels:
        prec = compute_NER_precision_label(label_pred, label_correct, label)
        rec = compute_NER_precision_label(label_correct, label_pred, label)

        f1 = 0
        if (rec + prec) > 0:
            f1 = 2.0 * prec * rec / (prec + rec)
        f1_macro += f1

        f1_per_labels[label] = f1

    return f1_macro / len(labels), f1_per_labels


def compute_NER_precision_label(guessed_sentences, correct_sentences, label):

    assert len(guessed_sentences) == len(correct_sentences)
    correctCount = 0
    count = 0

    for sentenceIdx in range(len(guessed_sentences)):

        guessed = guessed_sentences[sentenceIdx]
        correct = correct_sentences[sentenceIdx]

        assert len(guessed) == len(correct)
        idx = 0

        while idx < len(guessed):

            if guessed[idx] == label:

                count += 1

                if guessed[idx] == correct[idx]:
                    idx += 1
                    correctlyFound = True

                    while idx < len(guessed) and guessed[idx][0] == "I":  # Scan until it no longer starts with I
                        if guessed[idx] != correct[idx]:
                            correctlyFound = False
                        idx += 1

                    if idx < len(guessed):
                        if correct[idx][0] == "I":  # The chunk in correct was longer
                            correctlyFound = False

                    if correctlyFound:
                        correctCount += 1
                else:
                    idx += 1
            else:
                idx += 1

    precision = 0
    if count > 0:
        precision = float(correctCount) / count

    return precision


def decode_ner(list_ner_tokens):

    list_seq = []
    for ner_tokens in list_ner_tokens:
        ner_labels = [ner_idx2label[x] for x in ner_tokens]
        list_seq.append(ner_labels)

    return list_seq


def f1_score(labels: List[int], predicts: List[Tensor], logger: Logger = None, threshold: float = 0.5,):
    tp, fp, fn, tn = 0, 0, 0, 0
    for idx, label in enumerate(labels):
        if label == 1 and predicts[idx][1].item() >= threshold:
            tp += 1
        elif label == 1 and predicts[idx][1].item() < threshold:
            fn += 1
        elif predicts[idx][1].item() >= threshold:
            fp += 1
        else:
            tn += 1
    if logger is not None:
        logger.info(f"TP: {tp}\nFP: {fp}\nFN: {fn}\nTN: {tn}")
    if (tp + fp) == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    if (tp + fn) == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall
