from typing import Dict, Any

from config.base_config import BaseConfig


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


class ModelConfig(BaseConfig):
    required_arguments = {"gnn",
                          "classifier",
                          "ner_classifier"}

    def __init__(self,
                 gnn: GNNConfig,
                 classifier: ClassifierConfig):
        self.gnn = gnn
        self.classifier = classifier

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        cls.check_required(d)
        return ModelConfig(GNNConfig.from_dict(d['gnn']),
                           ClassifierConfig.from_dict(d['classifier']))
