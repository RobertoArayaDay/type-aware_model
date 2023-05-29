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
    

def one_hot_encoding(items, unique_items):
    big_array = []
    for lista_items in items:
        encoding = [0] * len(unique_items) * 2

        index0 = unique_items.index(lista_items[0])
        index1 = unique_items.index(lista_items[1]) + len(unique_items)

        encoding[index0] = 1
        encoding[index1] = 1

        #return encoding
        big_array.append(encoding)
    return big_array



def get_unique_items(dictionary):
    unique_items = []

    for pair in dictionary:
        item1, item2 = pair
        if item1 not in unique_items: unique_items.append(item1)
        if item2 not in unique_items: unique_items.append(item2)

    return unique_items
    

class ConditionalSimNet(nn.Module):
    def __init__(self, args, embeddingnet, n_conditions, typespaces):
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
        self.embeddingnet = embeddingnet
        
        # categories attributes
        self.typespaces = typespaces
        self.unique_items = list(get_unique_items(typespaces)) # from pairs of categories in typespace obtains unique categories
        self.num_category = len(self.unique_items) # integer of unique categories in typespaces
        
        #self.unique_dict = dict(zip(unique_items, range(len(unique_items))))
        #pairs_items = list(combinations(unique_items, 2)) 
        
        
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

    
    def forward(self, image, c=None):
        # image (B,C,H,W)
        # image_category (B, NUM_CATE)
        # concat_categories (B, NUM_CATE * @)

        embedded_x = self.embeddingnet(image) # Batch, embedding_dims
        feature_x = embedded_x.unsqueeze(dim=1)
        b, _, _ = feature_x.size()
        feature_x = feature_x.expand((b, self.num_conditions, self.dim_embed))
        # Batch, num_conditions, embedding_dims

        index = torch.arange(self.num_conditions, device=image.device).unsqueeze(dim=0).expand(b, self.num_conditions) # batch_size, num_conditions
        
        embed = self.masks(index) # batch_size, num_conditions, embedding_dims
        embed_feature = embed * feature_x # batch_size, num_conditions, embedding_dims
        
        if c is None:
            final_feature = []
            
            categories_encoding = list(self.typespaces.keys()) # total_pair_categories, 2 - se obtienen todas los pares de categorias - las posibles combinaciones
            
            categories_encoding = torch.Tensor(one_hot_encoding(categories_encoding, self.unique_items)).cuda()                 # (total_pair_categories, num_categoriesx2)
            categories_encoding = categories_encoding.unsqueeze(dim = 0)
            total_pair_categories = categories_encoding.shape[1]
            
            categories_encoding.expand((b, total_pair_categories, self.num_category*2))  # (batch_size, total_pair_categories num_categoriesx2)
            
            #for single_embed in embed_feature:
            attention_weight = self.cate_net(categories_encoding) # batch_size, total_pair_categories, num_conditions
            attention_weight = attention_weight.unsqueeze(dim=3)  # batch_size, total_pair_categories, num_conditions, 1
            attention_weight = attention_weight.expand((b, total_pair_categories, self.num_conditions, self.dim_embed)) # batch_size, total_pair_categories, num_conditions, embedding_dims
            
            embed_feature = embed_feature.unsqueeze(dim=1)
            embed_feature = embed_feature.expand((b, total_pair_categories, self.num_conditions, self.dim_embed))
            weighted_feature = embed_feature * attention_weight # batch_size, total_pair_categories, num_conditions, embedding_dims

            condition_feature = torch.sum(weighted_feature, dim=2) # batch_size, total_pair_categories, embedding_dims
            #final_feature.append(condition_feature)
                
            #final_feature = torch.cat(final_feature, 1)
            return torch.cat((condition_feature, feature_x), 1)
            
        else:
            c_array = [list(self.typespaces.keys())[list(self.typespaces.values()).index(i)] for i in c] # creates a list of all pair of categories for all conditions in c 
            
            c_array = torch.Tensor(one_hot_encoding(c_array, self.unique_items)).cuda() # returns the one-hot encoding for all conditions in c_array given the unique items - (batch_size, num_category * 2)
            
            
            attention_weight = self.cate_net(c_array) # batch_size, num_conditions
            attention_weight = attention_weight.unsqueeze(dim=2)
            attention_weight = attention_weight.expand((-1, -1, self.dim_embed)) # batch_size, num_conditions, embedding_dims

            weighted_feature = embed_feature * attention_weight # batch_size, num_conditions, embedding_dims

            final_feature = torch.sum(weighted_feature, dim=1) # batch_size, embedding_dims

            embed_norm = embedded_x.norm(2)
            mask_norm = 0. # cambiar esto

            return final_feature, mask_norm, embed_norm, embedded_x
