from torch import nn
import torch
from config.model_config import GNNConfig
from .gnn.gnn_factory import GNNFactory


class GNN(nn.Module):
    def __init__(self, config: GNNConfig, in_feat_dim: int, device: str):
        super().__init__()
        self.node_type_embedding = nn.Embedding(3, config.node_type_embedding)
        self.gnn = GNNFactory.from_config(config, in_feat_dim)
        self.layer_norm = nn.LayerNorm(in_feat_dim)
        self.device = device

    def forward(self, inputs):
        (
            token_hidden_state,
            token_mask,
            entity_mention_converter,
            mention_token_converter,
            graph,
        ) = inputs
        batch_size, num_sent, sent_length, _ = token_hidden_state.shape

        convert_token_hidden_state = torch.reshape(token_hidden_state, (batch_size, num_sent * sent_length, -1))
        mention_hidden_state = torch.matmul(mention_token_converter,
                                            convert_token_hidden_state)
        entity_hidden_state = torch.matmul(entity_mention_converter,
                                           mention_hidden_state)

        masked_token_hidden_state = token_hidden_state * token_mask.view(batch_size, num_sent, sent_length, 1)
        sent_hidden_state = torch.sum(masked_token_hidden_state, dim=2)
        sent_real_length = torch.sum(token_mask, dim=-1, keepdim=True)
        sent_real_length[sent_real_length < 1] = 1  # Avoid divide zero
        sent_hidden_state /= sent_real_length
        # sent_hidden_state = torch.mean(token_hidden_state, dim=2)
        if self.device == "cuda":
            mention_type_embedding = self.node_type_embedding(torch.tensor(0).cuda()).view(1, 1, -1)
            entity_type_embedding = self.node_type_embedding(torch.tensor(1).cuda()).view(1, 1, -1)
            sent_type_embedding = self.node_type_embedding(torch.tensor(2).cuda()).view(1, 1, -1)
        else:
            mention_type_embedding = self.node_type_embedding(torch.tensor(0)).view(1, 1, -1)
            entity_type_embedding = self.node_type_embedding(torch.tensor(1)).view(1, 1, -1)
            sent_type_embedding = self.node_type_embedding(torch.tensor(2)).view(1, 1, -1)

        num_mention_per_batch = int(mention_token_converter.shape[1])
        num_entity_per_batch = int(entity_mention_converter.shape[1])
        num_sent_per_batch = num_sent

        mention_type_embedding = torch.broadcast_to(mention_type_embedding, (batch_size, num_mention_per_batch, -1))
        entity_type_embedding = torch.broadcast_to(entity_type_embedding, (batch_size, num_entity_per_batch, -1))
        sent_type_embedding = torch.broadcast_to(sent_type_embedding, (batch_size, num_sent_per_batch, -1))

        mention_hidden_state = torch.concat((mention_hidden_state, mention_type_embedding), dim=2)
        entity_hidden_state = torch.concat((entity_hidden_state, entity_type_embedding), dim=2)
        sent_hidden_state = torch.concat((sent_hidden_state, sent_type_embedding), dim=2)

        node_hidden_state = torch.concat((mention_hidden_state, entity_hidden_state, sent_hidden_state), dim=1)

        node_hidden_state = self.layer_norm(node_hidden_state)  # Try out

        num_node_per_batch = int(node_hidden_state.shape[1])
        node_hidden_state = torch.reshape(node_hidden_state, (batch_size * num_node_per_batch, -1))

        # GNN
        output_node_hidden_state = self.gnn(graph, node_hidden_state)

        output_node_hidden_state = torch.reshape(output_node_hidden_state, (batch_size, num_node_per_batch, -1))
        entity_hidden_state = output_node_hidden_state[:, num_mention_per_batch:num_mention_per_batch +
                                                                                num_entity_per_batch]
        return entity_hidden_state
