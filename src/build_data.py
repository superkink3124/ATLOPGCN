import os.path

from config.cdr_config import CDRConfig
from corpus.cdr_corpus import CDRCorpus

if __name__ == "__main__":
    config_file_path = "gate_config.json"
    config = CDRConfig.from_json(config_file_path)
    # print(config)
    corpus = CDRCorpus(config)
    corpus.prepare_all_vocabs(config.data.saved_data_path)
    if not os.path.exists(config.data.saved_data_path):
        os.mkdirs(config.data.saved_data_path)
    corpus.prepare_features_for_one_dataset(
        config.data.train_file_path, config.data.saved_data_path, "train"
    )
    corpus.prepare_features_for_one_dataset(
        config.data.dev_file_path, config.data.saved_data_path, "dev"
    )
    corpus.prepare_features_for_one_dataset(
        config.data.test_file_path, config.data.saved_data_path, "test"
    )
