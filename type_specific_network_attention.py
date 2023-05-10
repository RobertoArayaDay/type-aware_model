import torch
import torch.nn as nn
import numpy as np

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
        
        # Boolean indicating whether masks are initialized in equally sized disjoint
        # sections or random otherwise
        prein = args.prein

        # Indicates that there isn't a 1:1 relationship between type specific spaces
        # and pairs of items categories
        if args.rand_typespaces:
            n_conditions = int(np.ceil(n_conditions / float(args.num_rand_embed)))

        self.learnedmask = learnedmask
        self.embeddingnet = embeddingnet
        self.attention = nn.MultiheadAttention(embed_dim=args.dim_embed, num_heads=4)
        self.attention_linear = nn.Linear(args.dim_embed, args.dim_embed)
        self.dropout = nn.Dropout(p=0.2)

        # When true we l2 normalize the output type specific embeddings
        self.l2_norm = args.l2_embed

        # create the mask
        if prein:
            # define masks
            self.masks = torch.nn.Embedding(n_conditions, args.dim_embed)
            # initialize masks
            mask_array = np.zeros([n_conditions, args.dim_embed])
            mask_array.fill(0.1)
            mask_len = int(args.dim_embed / n_conditions)
            for i in range(n_conditions):
                mask_array[i, i*mask_len:(i+1)*mask_len] = 1
            # no gradients for the masks
            self.masks.weight = torch.nn.Parameter(torch.Tensor(mask_array), requires_grad=True)
        else:
            # define masks with gradients
            self.masks = torch.nn.Embedding(n_conditions, args.dim_embed)
            # initialize weights
            self.masks.weight.data.normal_(0.9, 0.7) # 0.1, 0.005


    def compute_masked_embedding(self, embedded_x, c):
        if self.fc_masks:
            mask_norm = 0.
            masked_embedding = []
            for embed, condition in zip(embedded_x, c):
                mask = self.masks[condition]
                masked_embed = self.dropout(self.attention_linear(self.attention(embed.unsqueeze(0).transpose(0, 1), mask.unsqueeze(0), mask.unsqueeze(0)).squeeze(0)))
                masked_embedding.append(masked_embed)
                mask_norm += torch.norm(mask)
            masked_embedding = torch.stack(masked_embedding)
            if self.l2_norm:
                masked_embedding = nn.functional.normalize(masked_embedding, p=2, dim=1)
            return masked_embedding, mask_norm
        else:
            mask = self.masks[c]
            masked_embed = self.dropout(self.attention_linear(self.attention(embedded_x.unsqueeze(0).transpose(0, 1), mask.unsqueeze(0), mask.unsqueeze(0)).squeeze(0)))
            if self.l2_norm:
                masked_embed = nn.functional.normalize(masked_embed, p=2, dim=1)
            return masked_embed
        
    def forward(self, x, c):
        # x is the input tensor of shape (batch_size, input_size)
        # c is the condition tensor of shape (batch_size,)

        # project the input into the embedding space
        embedded_x = self.embeddingnet(x)

        # apply the attention mechanism to the embedded input
        attn_output, _ = self.attention(embedded_x.transpose(0, 1), embedded_x.transpose(0, 1), embedded_x.transpose(0, 1))

        # apply a linear transformation to the attention output
        linear_output = self.attention_linear(attn_output)

        # apply dropout to the linear output
        dropped_output = self.dropout(linear_output)

        if self.fc_masks:
            # apply the learned fully connected layer to transform the general embedding
            # to the type specific embedding
            masked_embedding = []
            for embed, condition in zip(dropped_output, c):
                mask = self.masks[condition]
                masked_embedding.append(mask(embed.unsqueeze(0)).squeeze(0))
            masked_embedding = torch.stack(masked_embedding)
        else:
            # apply the fixed mask to get the type specific embedding
            masked_embedding = self.masks(c) * dropped_output

        if self.l2_norm:
            # L2 normalize the output type specific embeddings
            masked_embedding = nn.functional.normalize(masked_embedding, p=2, dim=1)

        return masked_embedding

    
    

