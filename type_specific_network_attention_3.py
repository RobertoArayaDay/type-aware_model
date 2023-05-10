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
        

        # When true we l2 normalize the output type specific embeddings
        self.l2_norm = args.l2_embed
        self.dim_embed = args.dim_embed

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
        
        batch_size = x.shape[0]
        n_conditions = self.masks.num_embeddings
        dim_embed = self.dim_embed
        if c is None:
            
            # used during testing, wants all type specific embeddings returned for an image
            masks = self.masks.weight.unsqueeze(0).repeat(batch_size, 1, 1)
            
            masked_embedding = torch.bmm(masks, embedded_x.unsqueeze(-1)).squeeze(-1)
            if self.l2_norm:
                norm = torch.norm(masked_embedding, p=2, dim=1) + 1e-10
                masked_embedding = masked_embedding / norm.unsqueeze(-1)

            return torch.cat((masked_embedding, embedded_x), 1)

        attention_weights = self.masks(c)
        attention_weights = torch.nn.functional.softmax(attention_weights)
        
        print()
        print(attention_weights.shape)
        print(embedded_x.shape)
        print(embedded_x.unsqueeze(-1).shape)
        masked_embedding = torch.bmm(attention_weights, embedded_x.unsqueeze(-1)).squeeze(-1)

        if self.l2_norm:
            norm = torch.norm(masked_embedding, p=2, dim=1) + 1e-10
            masked_embedding = masked_embedding / norm.unsqueeze(-1)

        mask_norm = attention_weights.norm(1, dim=(1, 2))
        embed_norm = embedded_x.norm(2, dim=1)

        return masked_embedding, mask_norm, embed_norm, embedded_x

    
    

