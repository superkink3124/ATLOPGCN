PAD = "<PAD>"
UNK = "<UNK>"
CHEMICAL_STRING = "Chemical"
DISEASE_STRING = "Disease"
ADJACENCY_REL = "node"
ROOT_REL = "root"

# NER
NER_O = "O"
ner_vocab = {"O": 0, "B_Chemical": 1, "I_Chemical": 2, "B_Disease": 3, "I_Disease": 4}
ner_idx2label = {0: "O", 1: "B_Chemical", 2: "I_Chemical", 3: "B_Disease", 4: "I_Disease"}

# ELMO PATH
ELMO_OPTIONS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pubmed/elmo_2x4096_512_2048cnn_2xhighway_options.json"
ELMO_WEIGHT_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pubmed/elmo_2x4096_512_2048cnn_2xhighway_weights_PubMed_only.hdf5"