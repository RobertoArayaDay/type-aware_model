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
        self.n_conditions = n_conditions
        self.dim_embed = args.dim_embed

        # When true we l2 normalize the output type specific embeddings
        self.l2_norm = args.l2_embed
        
        # define attention weights
        
        attention_heads = []
        for i in range(n_conditions):
            attention_heads.append(nn.MultiheadAttention(embed_dim=self.dim_embed, num_heads=args.num_heads))
        self.attention_heads = ListModule(*attention_heads)

           
    def forward(self, x, c = None):
        """ x: input image data
            c: type specific embedding to compute for the images, returns all embeddings
               when None including the general embedding concatenated onto the end
        """
        
        embedded_x = self.embeddingnet(x)
        
        if c is None:
            # used during testing, wants all type specific embeddings returned for an image
            masked_embedding = []
            for mask in self.attention_heads:
                attn_output, _ = mask(embedded_x.permute(1, 0, 2).contiguous(), embedded_x.permute(1, 0, 2).contiguous(), embedded_x.permute(1, 0, 2).contiguous())
                masked_embedding.append(attn_output.permute(1, 0, 2).unsqueeze(1))
            

            masked_embedding = torch.cat(masked_embedding, 1)
            embedded_x = embedded_x.unsqueeze(1)

            if self.l2_norm:
                norm = torch.norm(masked_embedding, p=2, dim=2) + 1e-10
                masked_embedding = masked_embedding / norm.unsqueeze(-1).expand_as(masked_embedding)

            return torch.cat((masked_embedding, embedded_x), 1)
        
        mask_norm = 0.
        masked_embedding = []
        for embed, condition in zip(embedded_x, c):
            mask = self.attention_heads[condition]
            attn_output, _ = mask(embed.permute(1, 0, 2).contiguous(), embed.permute(1, 0, 2).contiguous(), embed.permute(1, 0, 2).contiguous())
            
            masked_embedding.append(attn_output.permute(1, 0, 2).unsqueeze(1))
            mask_norm += mask.weight.norm(1)

        masked_embedding = torch.cat(masked_embedding)
        embed_norm = embedded_x.norm(2)
            
            
            
            
        attn_output, _ = self.attention_heads(c)(embedded_x.permute(1, 0, 2), embedded_x.permute(1, 0, 2), embedded_x.permute(1, 0, 2))
        masked_embedding = attn_output.permute(1, 0, 2).unsqueeze(1)
        mask_norm = self.attention_heads(c).norm(1)
        
        
        

        embed_norm = embedded_x.norm(2)
        if self.l2_norm:
            norm = torch.norm(masked_embedding, p=2, dim=1) + 1e-10
            masked_embedding = masked_embedding / norm.unsqueeze(-1).expand_as(masked_embedding)

        return masked_embedding, mask_norm, embed_norm, embedded_x