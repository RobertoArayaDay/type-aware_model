import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from torch.autograd import Variable
from itertools import combinations


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

def create_mapping(length, integer):
    if length <= 2 or integer < 0 or integer >= length:
        return None

    mapping = [0] * length
    mapping[integer] = 1

    # Find the next available index
    index = (integer + 1) % length
    while mapping[index] == 1:
        index = (index + 1) % length

    mapping[index] = 1

    return mapping

def generate_index_pairs(num_conditions):
    indices = list(range(num_conditions))
    index_pairs = list(combinations(indices, 2))
    return index_pairs
    

class ConditionalSimNet(nn.Module):
    def __init__(self, args, embeddingnet, n_conditions, num_category):
        """ args: Input arguments from the main script
            embeddingnet: The network that projects the inputs into an embedding of embedding_size
            embedding_size: Number of dimensions of the embedding output from the embeddingnet
            n_conditions: Integer defining number of different similarity notions
        """
        super(ConditionalSimNet, self).__init__()
        # Boolean indicating whether masks are learned or fixed
        self.learnedmask = args.learned
        
        # Boolean indicating whether masks are initialized in equally sized disjoint
        # sections or random otherwise
        self.prein = args.prein
        
        self.num_conditions = n_conditions
        self.num_category = num_category
        self.embeddingnet = embeddingnet
        
        # category network
        self.cate_net = list()
        self.cate_net.append(nn.Linear(self.num_category * 2, self.num_conditions))
        self.cate_net.append(nn.ReLU(inplace=True))
        self.cate_net.append(nn.Linear(self.num_conditions, self.num_conditions))
        self.cate_net.append(nn.Softmax(dim=1))
        self.cate_net = nn.Sequential(*self.cate_net)
        

        # When true a fully connected layer is learned to transform the general
        # embedding to the type specific embedding
        self.dim_embed = args.dim_embed
        
        list_condition_pairs = generate_index_pairs(num_category)
        self.list_condition_pairs = list_condition_pairs
        self.conditions_dict = dict(zip(range(len(list_condition_pairs)), list_condition_pairs))


        # create the mask
        if self.learnedmask:
            if self.prein:
                # define masks
                self.masks = torch.nn.Embedding(self.num_conditions, self.dim_embed)
                # initialize masks
                mask_array = np.zeros([self.num_conditions, self.dim_embed])
                mask_array.fill(0.1)
                mask_len = int(self.dim_embed / self.num_conditions)
                for i in range(self.num_conditions):
                    mask_array[i, i*mask_len:(i+1)*mask_len] = 1
                # no gradients for the masks
                self.masks.weight = torch.nn.Parameter(torch.Tensor(mask_array), requires_grad=True)
            else:
                # define masks with gradients
                self.masks = torch.nn.Embedding(self.num_conditions, self.dim_embed)
                # initialize weights
                self.masks.weight.data.normal_(0.9, 0.7) # 0.1, 0.005
        else:
            # define masks
            self.masks = torch.nn.Embedding(self.num_conditions, self.dim_embed)
            # initialize masks
            mask_array = np.zeros([self.num_conditions, self.dim_embed])
            mask_len = int(self.dim_embed / self.num_conditions)
            for i in range(self.num_conditions):
                mask_array[i, i*mask_len:(i+1)*mask_len] = 1
            # no gradients for the masks
            self.masks.weight = torch.nn.Parameter(torch.Tensor(mask_array), requires_grad=False)

    
    def forward(self, image, c):
        # image (B,C,H,W)
        # image_category (B, NUM_CATE)
        # concat_categories (B, NUM_CATE * @)

        embedded_x = self.embeddingnet(image) # Batch, embedding_dims
        feature_x = embedded_x.unsqueeze(dim=1) # Batch, 1, embedding_dims
        b, _, _ = feature_x.size()
        feature_x = feature_x.expand((b, self.num_conditions, self.dim_embed))  # Batch, num_conditions, embedding_dims

        index = Variable(torch.LongTensor(range(self.num_conditions)))

        if image.is_cuda:
            index = index.cuda() # num_conditions
        index = index.unsqueeze(dim=0) # 1, num_conditions
        index = index.expand((b, self.num_conditions)) # batch_size, num_conditions

        embed = self.masks(index) # batch_size, num_conditions, embedding_dims
        embed_feature = embed * feature_x # batch_size, num_conditions, embedding_dims
        
        for condition_pair_key in list(self.conditions_dict.keys()):
            print(condition_pair)
        
        print(c)
        if c is None:
            final_feature = []
            for condition_pair in self.list_condition_pairs:
                c_array = create_mapping(self.num_category, condition_pair)
                
                attention_weight = self.cate_net(c_array) # batch_size, num_conditions
                attention_weight = attention_weight.unsqueeze(dim=2)
                attention_weight = attention_weight.expand((b, self.num_conditions, self.embedding_size)) # batch_size, num_conditions, embedding_dims
                
                weighted_feature = embed_feature * attention_weight # batch_size, num_conditions, embedding_dims

                condition_feature = torch.sum(weighted_feature, dim=1) # batch_size, embedding_dims
                final_feature.append(final_feature)
                
            final_feature = torch.cat(final_feature, 1)
            return torch.cat((masked_embedding, embedded_x), 1)
            
        else:
            print(c)
            c_array = create_mapping(self.num_category, c) # se crea array de condiciones
            print(c_array)
            print()
            
            attention_weight = self.cate_net(c_array) # batch_size, num_conditions
            attention_weight = attention_weight.unsqueeze(dim=2)
            attention_weight = attention_weight.expand((b, self.num_conditions, self.embedding_size)) # batch_size, num_conditions, embedding_dims

            weighted_feature = embed_feature * attention_weight # batch_size, num_conditions, embedding_dims

            final_feature = torch.sum(weighted_feature, dim=1) # batch_size, embedding_dims

            embed_norm = embedded_x.norm(2)
            mask_norm = 0.
            
            for mask in self.masks:
                mask_norm += self.mask.norm(1)

            return final_feature, mask_norm, embed_norm, embedded_x
