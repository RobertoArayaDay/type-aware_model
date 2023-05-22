import torch
import torch.nn as nn
import numpy as np


class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)
    
    
class ComplexNetwork(nn.Module):
    def __init__(self, in_features, out_features):
        # dos fully connected
        # condicionada por el tipo de prenda
        # positional encoding para informar acerca de la condicion, sobre el condicion
        # 2 fully-connected
        # entrenar de a pares
        
        super(ComplexNetwork, self).__init__()
        #self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        #self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features, 128)
        self.fc2 = nn.Linear(128, out_features)

    def forward(self, x):
        #x = x.unsqueeze(1)
        #x = self.conv1(x)
        #x = self.relu(x)
        #x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        #x = self.relu(x)
        x = self.fc2(x)
        return x

    
# Visual transformer: evaluar uso
# Autoencoder
# modelos no-supervisados que obtienen mejores embeddings/resultados, mejores resultados en ImageNet
class TypeSpecificNetAttention(nn.Module):
    def __init__(self, args, embeddingnet, n_conditions):
        """ args: Input arguments from the main script
            embeddingnet: The network that projects the inputs into an embedding of embedding_size
            embedding_size: Number of dimensions of the embedding output from the embeddingnet
            n_conditions: Integer defining number of different similarity notions
        """
        super(TypeSpecificNetAttention, self).__init__()
        # Boolean indicating whether masks are learned or fixed
        learnedmask = args.learned
        
        self.embeddingnet = embeddingnet

        # When true we l2 normalize the output type specific embeddings
        self.l2_norm = args.l2_embed
        
        # define attention weights
        masks = []
        
        # cambiar de ComplexNetwork a LInear si se quiere ocupar una red fully connected
        for i in range(n_conditions):
        #    masks.append(nn.Linear(args.dim_embed, args.dim_embed))
            masks.append(ComplexNetwork(args.dim_embed, args.dim_embed))
        self.masks = ListModule(*masks)
        
        
        #self.attention_heads = nn.ModuleList([
        #    nn.MultiheadAttention(embed_dim=self.dim_embed, num_heads=args.num_heads)
        #    for _ in range(n_conditions)
        #])

           
    def forward(self, x, c = None):
        """ x: input image data
            c: type specific embedding to compute for the images, returns all embeddings
               when None including the general embedding concatenated onto the end
        """
        
        embedded_x = self.embeddingnet(x)
        # (1, 64)
       
        if c is None:
            # used during testing, wants all type specific embeddings returned for an image
            masked_embedding = []
            for mask in self.masks:
                masked_embedding.append(mask(embedded_x).unsqueeze(1))

            masked_embedding = torch.cat(masked_embedding, 1)
            embedded_x = embedded_x.unsqueeze(1)

            if self.l2_norm:
                norm = torch.norm(masked_embedding, p=2, dim=2) + 1e-10
                masked_embedding = masked_embedding / norm.unsqueeze(-1).expand_as(masked_embedding)

            return torch.cat((masked_embedding, embedded_x), 1)
        
        mask_norm = 0.
        masked_embedding = []
        for embed, condition in zip(embedded_x, c):
             mask = self.masks[condition]
             masked_embedding.append(mask(embed.unsqueeze(0)))
             #mask_norm += mask.weight.norm(1)

        masked_embedding = torch.cat(masked_embedding)

        embed_norm = embedded_x.norm(2)
        if self.l2_norm:
            norm = torch.norm(masked_embedding, p=2, dim=1) + 1e-10
            masked_embedding = masked_embedding / norm.unsqueeze(-1).expand_as(masked_embedding)

        return masked_embedding, mask_norm, embed_norm, embedded_x