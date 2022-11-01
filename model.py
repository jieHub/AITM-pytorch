import torch
from torch import nn


class Tower(nn.Module):

    def __init__(self,
                 input_dim: int,
                 dims=[128, 64, 32],
                 drop_probs=[0.1, 0.3, 0.3]):
        super().__init__()
        self.dims = dims
        self.drop_probs = drop_probs
        self.layers = nn.Sequential(nn.Linear(input_dim, dims[0]), nn.ReLU(),
                                    nn.Dropout(drop_probs[0]),
                                    nn.Linear(dims[0], dims[1]), nn.ReLU(),
                                    nn.Dropout(drop_probs[1]),
                                    nn.Linear(dims[1], dims[2]), nn.ReLU(),
                                    nn.Dropout(drop_probs[2]))

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.layers(x)
        return x


class Attention(nn.Module):

    def __int__(self, dim=32):
        super().__init__()
        self.dim = dim
        self.q_layer = nn.Linear(dim, dim, bias=False)
        self.k_layer = nn.Linear(dim, dim, bias=False)
        self.v_layer = nn.Linear(dim, dim, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        q = self.q_layer(x)
        k = self.k_layer(x)
        v = self.v_layer(x)
        atten = torch.sum(torch.mul(q, k), -1) / torch.sqrt(torch.tensor(self.dim))
        atten = self.softmax(atten)
        outputs = torch.sum(torch.mul(torch.unsqueeze(a, -1), v), dim=1)
        return outputs


class AITM(nn.Module):

    def __init__(self, 
                 feature_vocabulary: dict[str, int],
                 embedding_size: int,
                 tower_dims=[128, 64, 32],
                 drop_probs=[0.1, 0.3, 0.3]):
        super().__init__()
        self.feature_vocabulary = feature_vocabulary
        self.feature_names = sorted(feature_vocabulary.keys())
        self.embedding_size = embedding_size
        self._init_embedding_dict()

        self.tower_input_size = len(feature_vocabulary) * embedding_size
        self.click_tower = Tower(self.tower_input_size, tower_dims, drop_probs)
        self.conversion_tower = Tower(self.tower_input_size, tower_dims, drop_probs)
        self.atten_layer = Attention(tower_dims[-1])

        self.info_layer = nn.Sequential(nn.Linear(tower_dims[-1], 32), nn.ReLU(),
                                        nn.Dropout(drop_probs[-1]))
        self.click_layer = nn.Sequential(nn.Linear(tower_dims[-1], 1),
                                         nn.Sigmoid())
        self.conversion_layer = nn.Sequential(nn.Linear(tower_dims[-1], 1), 
                                        nn.Sigmoid())
       
    def _init_embedding_dict(self):
        self.embedding_dict = nn.ModuleDict()
        for name, dim in self.feature_vocabulary.items():
            emb = nn.Embedding(size, self.embedding_size)
            nn.init.normal_(emb.weight, mean=0.0, std=0.01)
            self.embedding_dict[name] = emb

    def forward(self, x):
        feature_embedding = []
        for name in self.feature_names:
            embed = self.embedding_dict[name](x[name])
            feature_embedding.append(embed)
        feature_embedding = torch.cat(feature_embedding, 1)
        
        tower_click = self.click_tower(feature_embedding)
        tower_conversion = self.tower_conversion(feature_embedding)
        info = self.info_layer(tower_click)
        ait = self.atten_layer(torch.stack([tower_conversion, info], 1))

        click = torch.squeeze(self.click_layer(tower_click), dim=1)
        conversion = torch.squeeze(self.conversion_layer(ait), dim=1)

        return click, conversion



