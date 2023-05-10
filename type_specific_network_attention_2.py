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
        
        self.attention = nn.MultiheadAttention(embed_dim=args.dim_embed*4, num_heads=4)
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


           
    def forward(self, x, c = None):
        """ x: input image data
            c: type specific embedding to compute for the images, returns all embeddings
               when None including the general embedding concatenated onto the end
        """
        
        embedded_x = self.embeddingnet(x)
        if c is None:
            # used during testing, wants all type specific embeddings returned for an image
            
            masks = Variable(self.masks.weight.data)
            masks = masks.unsqueeze(0).repeat(embedded_x.size(0), 1, 1)
            embedded_x = embedded_x.unsqueeze(1)
            masked_embedding = embedded_x.expand_as(masks) * masks

            if self.l2_norm:
                norm = torch.norm(masked_embedding, p=2, dim=2) + 1e-10
                masked_embedding = masked_embedding / norm.unsqueeze(-1).expand_as(masked_embedding)
                
            
            query = self.attention_linear(masked_embedding)
            query = self.dropout(query)
            out, _ = self.attention(query.transpose(0, 1), embedded_x.transpose(0, 1), embedded_x.transpose(0, 1))
            out = out.transpose(0, 1)
            
            return torch.cat((out, embedded_x), 1)
            
        self.mask = self.masks(c)
        self.mask = torch.nn.functional.relu(self.mask)

        masked_embedding = embedded_x * self.mask
        mask_norm = self.mask.norm(1)

        embed_norm = embedded_x.norm(2)
        if self.l2_norm:
            norm = torch.norm(masked_embedding, p=2, dim=1) + 1e-10
            masked_embedding = masked_embedding / norm.unsqueeze(-1).expand_as(masked_embedding)
        
        
        # apply attention to the masked embedding
        query = self.attention_linear(masked_embedding)
        query = self.dropout(query)
        
        print(query.transpose(0, 1).shape, embedded_x.transpose(0, 1).shape)
        out, _ = self.attention(query.transpose(0, 1), embedded_x.transpose(0, 1), embedded_x.transpose(0, 1))
        out = out.transpose(0, 1)
        
        return out, mask_norm, embed_norm, embedded_x

    
    

