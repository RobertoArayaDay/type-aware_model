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

        # Indicates that there isn't a 1:1 relationship between type specific spaces
        # and pairs of items categories

        self.embeddingnet = embeddingnet

        # When true we l2 normalize the output type specific embeddings
        self.l2_norm = args.l2_embed
        
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
            
            return torch.cat((masked_embedding, embedded_x), 1)
            
        self.mask = self.masks(c)
        self.mask = torch.nn.functional.relu(self.mask)

        masked_embedding = embedded_x * self.mask
        mask_norm = self.mask.norm(1)

        embed_norm = embedded_x.norm(2)
        if self.l2_norm:
            norm = torch.norm(masked_embedding, p=2, dim=1) + 1e-10
            masked_embedding = masked_embedding / norm.unsqueeze(-1).expand_as(masked_embedding)
        
        
        return masked_embedding, mask_norm, embed_norm, embedded_x

    
    

