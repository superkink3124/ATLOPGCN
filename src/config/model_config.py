from typing import Dict, Any

from config.base_config import BaseConfig


class EncoderConfig(BaseConfig):
    required_arguments = {"elmo_dim",
                          "use_char", "char_embedding_dim", "kernel_size", "n_filters",
                          "use_pos", "pos_embedding_dim",
                          "lstm_hidden_dim", "lstm_num_layers",
                          "drop_out"}

    def __init__(self, elmo_dim,
                 use_char, char_embedding_dim, kernel_size, n_filters,
                 use_pos, pos_embedding_dim,
                 lstm_hidden_dim, lstm_num_layers,
                 drop_out):
        self.elmo_dim = elmo_dim
        self.use_char = use_char
        self.char_embedding_dim = char_embedding_dim
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.use_pos = use_pos
        self.pos_embedding_dim = pos_embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.drop_out = drop_out

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EncoderConfig":
        cls.check_required(d)
        return EncoderConfig(d['elmo_dim'],
                             d['use_char'], d['char_embedding_dim'], d['kernel_size'], d['n_filters'],
                             d['use_pos'], d['pos_embedding_dim'],
                             d['lstm_hidden_dim'], d['lstm_num_layers'], d['drop_out'])


class GNNConfig(BaseConfig):
    required_arguments = {"gnn_type", "node_type_embedding",
                          "args"}

    def __init__(self, gnn_type, node_type_embedding, args):
        self.gnn_type = gnn_type
        self.node_type_embedding = node_type_embedding
        self.args = args

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GNNConfig":
        cls.check_required(d)
        return GNNConfig(d['gnn_type'],
                         d['node_type_embedding'],
                         d['args'])


class ClassifierConfig(BaseConfig):
    required_arguments = {
        "use_hidden_layer",
        "hidden_layer_dim",
        "num_classes",
        "drop_out"
    }
    def __init__(self, use_hidden_layer, hidden_layer_dim, num_classes, drop_out):
        self.use_hidden_layer = use_hidden_layer
        self.hidden_layer_dim = hidden_layer_dim
        self.num_classes = num_classes
        self.drop_out = drop_out

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ClassifierConfig":
        cls.check_required(d)
        return ClassifierConfig(d['use_hidden_layer'],
                                d['hidden_layer_dim'],
                                d['num_classes'],
                                d['drop_out'])


class NERClassifierConfig(BaseConfig):
    required_arguments = {
        "hidden_dim", "ner_classes"
    }

    def __init__(self, hidden_dim, ner_classes):
        self.hidden_dim = hidden_dim
        self.ner_classes = ner_classes

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NERClassifierConfig":
        cls.check_required(d)
        return NERClassifierConfig(d['hidden_dim'], d['ner_classes'])


class ModelConfig(BaseConfig):
    required_arguments = {"encoder",
                          "gnn",
                          "classifier",
                          "use_ner",
                          "ner_classifier"}

    def __init__(self,
                 encoder: EncoderConfig,
                 gnn: GNNConfig,
                 ner_classifier: NERClassifierConfig,
                 classifier: ClassifierConfig,
                 use_ner: bool):
        self.encoder = encoder
        self.gnn = gnn
        self.ner_classifier = ner_classifier
        self.classifier = classifier
        self.use_ner = use_ner

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        cls.check_required(d)
        return ModelConfig(EncoderConfig.from_dict(d['encoder']),
                           GNNConfig.from_dict(d['gnn']),
                           NERClassifierConfig.from_dict(d['ner_classifier']),
                           ClassifierConfig.from_dict(d['classifier']),
                           d['use_ner'])
