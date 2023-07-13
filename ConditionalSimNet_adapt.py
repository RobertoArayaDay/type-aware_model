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
        # c tiene que ser un par (categoria1, categoria2)
        # image (B,C,H,W)
        # image_category (B, NUM_CATE)
        # concat_categories (B, NUM_CATE * @)
        
        device = next(self.parameters()).device
        embedded_x = self.embeddingnet(image) # Batch, embedding_dims
        feature_x = embedded_x.unsqueeze(dim=1)
        b, _, _ = feature_x.size()
        feature_x = feature_x.expand((b, self.num_conditions, self.dim_embed)) # Batch, num_conditions, embedding_dims

        index = torch.arange(self.num_conditions, device=image.device).unsqueeze(dim=0).expand(b, self.num_conditions) # batch_size, num_conditions
        
        embed = self.masks(index) # batch_size, num_conditions, embedding_dims
        embed_feature = embed * feature_x # batch_size, num_conditions, embedding_dims
        if c is None:
            
            categories_encoding = list(self.typespaces.keys()) # total_pair_categories, 2 - se obtienen todas los pares de categorias - las posibles combinaciones
            
            categories_encoding = torch.Tensor(one_hot_encoding(categories_encoding, self.unique_items)).to(device)                 # (total_pair_categories, num_categoriesx2)
            categories_encoding = categories_encoding.unsqueeze(dim = 0)
            total_pair_categories = categories_encoding.shape[1]
            
            categories_encoding = categories_encoding.expand((b, total_pair_categories, self.num_category*2))  # (batch_size, total_pair_categories num_categoriesx2)
            
            #for single_embed in embed_feature:
            #attention_weight = torch.tensor([0, 0, 1, 0, 0])
            #attention_weight = attention_weight.unsqueeze(0).unsqueeze(0).cuda()
            
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
            
            c_array = torch.Tensor(one_hot_encoding(c_array, self.unique_items)).to(device)   # returns the one-hot encoding for all conditions in c_array given the unique items - (batch_size, num_category * 2)
            
            attention_weight = self.cate_net(c_array) # batch_size, num_conditions
            attention_weight = attention_weight.unsqueeze(dim=2)
            attention_weight = attention_weight.expand((-1, -1, self.dim_embed)) # batch_size, num_conditions, embedding_dims

            weighted_feature = embed_feature * attention_weight # batch_size, num_conditions, embedding_dims

            final_feature = torch.sum(weighted_feature, dim=1) # batch_size, embedding_dims

            embed_norm = embedded_x.norm(2)
            
            mask_norm = 0. # cambiar esto
            #for mask in 

            return final_feature, mask_norm, embed_norm, embedded_x

        
        
        
        
class ColorNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ColorNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

        
class ConditionalSimNetColor(nn.Module):
    def __init__(self, args, embeddingnet, n_conditions, typespaces):
        """ args: Input arguments from the main script
            embeddingnet: The network that projects the inputs into an embedding of embedding_size
            embedding_size: Number of dimensions of the embedding output from the embeddingnet
            n_conditions: Integer defining number of different similarity notions
        """
        super(ConditionalSimNetColor, self).__init__()
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
        self.cate_net.append(nn.Linear(self.num_category * 2 + 6, self.num_conditions))
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
             

    
    def forward(self, image, color, c=None):
        # c tiene que ser un par (categoria1, categoria2)
        # image (B,C,H,W)
        # image_category (B, NUM_CATE)
        # concat_categories (B, NUM_CATE * @)

        embedded_x = self.embeddingnet(image) # Batch, embedding_dims
        feature_x = embedded_x.unsqueeze(dim=1)
        b, _, _ = feature_x.size()
        feature_x = feature_x.expand((b, self.num_conditions, self.dim_embed)) # Batch, num_conditions, embedding_dims

        index = torch.arange(self.num_conditions, device=image.device).unsqueeze(dim=0).expand(b, self.num_conditions) # batch_size, num_conditions
        
        embed = self.masks(index) # batch_size, num_conditions, embedding_dims
        embed_feature = embed * feature_x # batch_size, num_conditions, embedding_dims
        if c is None:
            
            categories_encoding = list(self.typespaces.keys()) # total_pair_categories, 2 - se obtienen todas los pares de categorias - las posibles combinaciones
            
            categories_encoding = torch.Tensor(one_hot_encoding(categories_encoding, self.unique_items)).cuda() 
            # (total_pair_categories, num_categoriesx2)
            
            categories_encoding = categories_encoding.unsqueeze(dim = 0)
            total_pair_categories = categories_encoding.shape[1]
            
            categories_encoding = categories_encoding.expand((b, total_pair_categories, self.num_category*2))  # (batch_size, total_pair_categories num_categoriesx2)
            
            color = color.unsqueeze(dim=1).expand((b, total_pair_categories, 3))
            categories_encoding = torch.cat((categories_encoding, color), 2)
            
            
            #for single_embed in embed_feature:
            #attention_weight = torch.tensor([0, 0, 1, 0, 0])
            #attention_weight = attention_weight.unsqueeze(0).unsqueeze(0).cuda()
            
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

            c_array = torch.cat((c_array, color), 1)
            
            attention_weight = self.cate_net(c_array) # batch_size, num_conditions
            attention_weight = attention_weight.unsqueeze(dim=2)
            attention_weight = attention_weight.expand((-1, -1, self.dim_embed)) # batch_size, num_conditions, embedding_dims

            weighted_feature = embed_feature * attention_weight # batch_size, num_conditions, embedding_dims

            final_feature = torch.sum(weighted_feature, dim=1) # batch_size, embedding_dims

            embed_norm = embedded_x.norm(2)
            
            mask_norm = 0. # cambiar esto
            #for mask in 

            return final_feature, mask_norm, embed_norm, embedded_x
        
        
class ConditionalSimNetColor2(nn.Module):
    def __init__(self, args, embeddingnet, n_conditions, typespaces):
        """ args: Input arguments from the main script
            embeddingnet: The network that projects the inputs into an embedding of embedding_size
            embedding_size: Number of dimensions of the embedding output from the embeddingnet
            n_conditions: Integer defining number of different similarity notions
        """
        super(ConditionalSimNetColor2, self).__init__()
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
        
        #self.unique_dict = dict(zip(unique_items, range(len(unique_items))))
        #pairs_items = list(combinations(unique_items, 2)) 
        color_output_dim = 5
        self.general_linear = nn.Linear(self.dim_embed + color_output_dim, self.dim_embed)
        self.colornet = ColorNet(input_dim=3, hidden_dim=10, output_dim=color_output_dim)  # example dimensions
        

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
             

    
    def forward(self, image, color, c=None):
        # c tiene que ser un par (categoria1, categoria2)
        # image (B,C,H,W)
        # image_category (B, NUM_CATE)
        # concat_categories (B, NUM_CATE * @)

        embedded_x = self.embeddingnet(image) # Batch, embedding_dims
        embedded_color = self.colornet(color.view(color.shape[0], -1))
        combined_embedding = torch.cat((embedded_x, embedded_color), dim=1)
        
        combined_embedding = self.general_linear(combined_embedding)
        
        feature_x = combined_embedding.unsqueeze(dim=1)
        b, _, combined_dim = feature_x.size()
        feature_x = feature_x.expand((b, self.num_conditions, self.dim_embed)) # Batch, num_conditions, embedding_dims

        index = torch.arange(self.num_conditions, device=image.device).unsqueeze(dim=0).expand(b, self.num_conditions) # batch_size, num_conditions
        
        embed = self.masks(index) # batch_size, num_conditions, embedding_dims
        embed_feature = embed * feature_x # batch_size, num_conditions, embedding_dims
        if c is None:
            
            categories_encoding = list(self.typespaces.keys()) # total_pair_categories, 2 - se obtienen todas los pares de categorias - las posibles combinaciones
            
            categories_encoding = torch.Tensor(one_hot_encoding(categories_encoding, self.unique_items)).cuda()                 # (total_pair_categories, num_categoriesx2)
            categories_encoding = categories_encoding.unsqueeze(dim = 0)
            total_pair_categories = categories_encoding.shape[1]
            
            categories_encoding = categories_encoding.expand((b, total_pair_categories, self.num_category*2))  # (batch_size, total_pair_categories num_categoriesx2)
            
            #for single_embed in embed_feature:
            #attention_weight = torch.tensor([0, 0, 1, 0, 0])
            #attention_weight = attention_weight.unsqueeze(0).unsqueeze(0).cuda()
            
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
            #for mask in 

            return final_feature, mask_norm, embed_norm, embedded_x
        
        

        

class ConditionalSimNetRobert(nn.Module):
    def __init__(self, args, embeddingnet, n_conditions, typespaces, premodel):
        """ args: Input arguments from the main script
            embeddingnet: The network that projects the inputs into an embedding of embedding_size
            embedding_size: Number of dimensions of the embedding output from the embeddingnet
            n_conditions: Integer defining number of different similarity notions
        """
        super(ConditionalSimNetRobert, self).__init__()
        # Boolean indicating whether masks are learned or fixed
        self.learnedmask = args.learned
        
        self.premodel = premodel
        
        # Boolean indicating whether masks are initialized in equally sized disjoint
        # sections or random otherwise
        self.prein = args.prein
        
        self.num_conditions = n_conditions
        self.embeddingnet = embeddingnet
        
        # categories attributes
        self.typespaces = typespaces
        self.unique_items = list(get_unique_items(typespaces)) # from pairs of categories in typespace obtains unique categories
        
        self.unique_items_embs = torch.tensor(self.premodel.encode(self.unique_items))
        
        
        #self.unique_dict = dict(zip(unique_items, range(len(unique_items))))
        #pairs_items = list(combinations(unique_items, 2)) 
        
        
        # category network
        self.cate_net = list()
        self.cate_net.append(nn.Linear(768*2, self.num_conditions))
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
        # c tiene que ser un par (categoria1, categoria2)
        # image (B,C,H,W)
        # image_category (B, NUM_CATE)
        # concat_categories (B, NUM_CATE * @)

        embedded_x = self.embeddingnet(image) # Batch, embedding_dims
        feature_x = embedded_x.unsqueeze(dim=1)
        b, _, _ = feature_x.size()
        feature_x = feature_x.expand((b, self.num_conditions, self.dim_embed)) # Batch, num_conditions, embedding_dims

        index = torch.arange(self.num_conditions, device=image.device).unsqueeze(dim=0).expand(b, self.num_conditions) # batch_size, num_conditions
        
        embed = self.masks(index) # batch_size, num_conditions, embedding_dims
        embed_feature = embed * feature_x # batch_size, num_conditions, embedding_dims
        
        #return torch.sum(embed_feature, dim=1)
        if c is None:
            
            categories_encoding = list(self.typespaces.keys()) # total_pair_categories, 2 - se obtienen todas los pares de categorias - las posibles combinaciones
            
            categories_encoding_emb = []
            for cats in self.typespaces:
                cat1, cat2 = cats
                concat_cat = torch.cat((self.unique_items_embs[self.unique_items.index(cat1)], self.unique_items_embs[self.unique_items.index(cat2)])).unsqueeze(0)
                categories_encoding_emb.append(concat_cat)
            categories_encoding = torch.cat(categories_encoding_emb, axis=0).cuda()    # (total_pair_categories, 2x768)
            categories_encoding = categories_encoding.unsqueeze(dim = 0)
            total_pair_categories = categories_encoding.shape[1]
            
            categories_encoding = categories_encoding.expand((b, total_pair_categories, 768*2))  # (batch_size, total_pair_categories num_categoriesx2)
            
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
            
            categories_encoding_emb = []
            for cats in c_array:
                cat1, cat2 = cats
                concat_cat = torch.cat((self.unique_items_embs[self.unique_items.index(cat1)], self.unique_items_embs[self.unique_items.index(cat2)])).unsqueeze(0)
                categories_encoding_emb.append(concat_cat)
            c_array = torch.cat(categories_encoding_emb, axis=0).cuda()
            
            
            
            #c_array = torch.Tensor(one_hot_encoding(c_array, self.unique_items)).cuda() # returns the one-hot encoding for all conditions in c_array given the unique items - (batch_size, num_category * 2)
            
            attention_weight = self.cate_net(c_array) # batch_size, num_conditions
            attention_weight = attention_weight.unsqueeze(dim=2)
            attention_weight = attention_weight.expand((-1, -1, self.dim_embed)) # batch_size, num_conditions, embedding_dims

            weighted_feature = embed_feature * attention_weight # batch_size, num_conditions, embedding_dims

            final_feature = torch.sum(weighted_feature, dim=1) # batch_size, embedding_dims

            embed_norm = embedded_x.norm(2)
            
            mask_norm = 0. # cambiar esto
            #for mask in 

            return final_feature, mask_norm, embed_norm, embedded_x
        

        
        
        
class ConditionalSimNetRobertSum(nn.Module):
    def __init__(self, args, embeddingnet, n_conditions, typespaces, premodel):
        """ args: Input arguments from the main script
            embeddingnet: The network that projects the inputs into an embedding of embedding_size
            embedding_size: Number of dimensions of the embedding output from the embeddingnet
            n_conditions: Integer defining number of different similarity notions
        """
        super(ConditionalSimNetRobertSum, self).__init__()
        # Boolean indicating whether masks are learned or fixed
        self.learnedmask = args.learned
        
        self.premodel = premodel
        
        # Boolean indicating whether masks are initialized in equally sized disjoint
        # sections or random otherwise
        self.prein = args.prein
        
        self.num_conditions = n_conditions
        self.embeddingnet = embeddingnet
        
        # categories attributes
        self.typespaces = typespaces
        self.unique_items = list(get_unique_items(typespaces)) # from pairs of categories in typespace obtains unique categories
        
        self.unique_items_embs = torch.tensor(self.premodel.encode(self.unique_items))
        
        
        #self.unique_dict = dict(zip(unique_items, range(len(unique_items))))
        #pairs_items = list(combinations(unique_items, 2)) 
        
        
        # category network
        self.cate_net = list()
        self.cate_net.append(nn.Linear(768, self.num_conditions))
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
        # c tiene que ser un par (categoria1, categoria2)
        # image (B,C,H,W)
        # image_category (B, NUM_CATE)
        # concat_categories (B, NUM_CATE * @)

        embedded_x = self.embeddingnet(image) # Batch, embedding_dims
        feature_x = embedded_x.unsqueeze(dim=1)
        b, _, _ = feature_x.size()
        feature_x = feature_x.expand((b, self.num_conditions, self.dim_embed)) # Batch, num_conditions, embedding_dims

        index = torch.arange(self.num_conditions, device=image.device).unsqueeze(dim=0).expand(b, self.num_conditions) # batch_size, num_conditions
        
        embed = self.masks(index) # batch_size, num_conditions, embedding_dims
        embed_feature = embed * feature_x # batch_size, num_conditions, embedding_dims
        
        #return torch.sum(embed_feature, dim=1)
        if c is None:
            
            categories_encoding = list(self.typespaces.keys()) # total_pair_categories, 2 - se obtienen todas los pares de categorias - las posibles combinaciones
            
            categories_encoding_emb = []
            for cats in self.typespaces:
                cat1, cat2 = cats
                concat_cat =(self.unique_items_embs[self.unique_items.index(cat1)]+ self.unique_items_embs[self.unique_items.index(cat2)]).unsqueeze(0)
                
                categories_encoding_emb.append(concat_cat)
            categories_encoding = torch.cat(categories_encoding_emb, axis=0).cuda()    # (total_pair_categories, 2x768)
            categories_encoding = categories_encoding.unsqueeze(dim = 0)
            total_pair_categories = categories_encoding.shape[1]
            
            categories_encoding = categories_encoding.expand((b, total_pair_categories, 768))  # (batch_size, total_pair_categories num_categoriesx2)
            
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
            
            categories_encoding_emb = []
            for cats in c_array:
                cat1, cat2 = cats
                concat_cat =(self.unique_items_embs[self.unique_items.index(cat1)]+ self.unique_items_embs[self.unique_items.index(cat2)]).unsqueeze(0)
                
                categories_encoding_emb.append(concat_cat)
            c_array = torch.cat(categories_encoding_emb, axis=0).cuda()
            
            
            
            #c_array = torch.Tensor(one_hot_encoding(c_array, self.unique_items)).cuda() # returns the one-hot encoding for all conditions in c_array given the unique items - (batch_size, num_category * 2)
            
            attention_weight = self.cate_net(c_array) # batch_size, num_conditions
            attention_weight = attention_weight.unsqueeze(dim=2)
            attention_weight = attention_weight.expand((-1, -1, self.dim_embed)) # batch_size, num_conditions, embedding_dims

            weighted_feature = embed_feature * attention_weight # batch_size, num_conditions, embedding_dims

            final_feature = torch.sum(weighted_feature, dim=1) # batch_size, embedding_dims

            embed_norm = embedded_x.norm(2)
            
            mask_norm = 0. # cambiar esto
            #for mask in 

            return final_feature, mask_norm, embed_norm, embedded_x
        
        

        
        
        
        

class ConditionalSimNet2(nn.Module):
    def __init__(self, args, embeddingnet, n_conditions, typespaces, rep_embed=64):
        """ args: Input arguments from the main script
            embeddingnet: The network that projects the inputs into an embedding of embedding_size
            embedding_size: Number of dimensions of the embedding output from the embeddingnet
            n_conditions: Integer defining number of different similarity notions
        """
        super(ConditionalSimNet2, self).__init__()
        # Boolean indicating whether masks are learned or fixed
        self.learnedmask = args.learned
        
        # Boolean indicating whether masks are initialized in equally sized disjoint
        # sections or random otherwise
        self.prein = args.prein
        
        self.num_conditions = n_conditions
        self.embeddingnet = embeddingnet
        self.rep_embed = rep_embed
        
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
        
        # learn a fully connected layer rather than a mask to project the general embedding
        # into the type specific space
        masks_rep = []
        for i in range(n_conditions):
            masks_rep.append(nn.Linear(args.dim_embed, rep_embed))
        self.masks_rep = ListModule(*masks_rep)
        
        

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
        feature_x = embedded_x.unsqueeze(dim=1) # Batch, 1, embedding_dims
        b, _, _ = feature_x.size()
        feature_x = feature_x.expand((b, self.num_conditions, self.dim_embed))
        # Batch, num_conditions, embedding_dims
        
        
        feature_x2 = torch.permute(feature_x, (1, 0, 2))
        
        masked_rep_embedding = []
        for embed, condition in zip(feature_x2, range(self.num_conditions)):
            mask = self.masks_rep[condition]
            masked_embed = mask(embed.unsqueeze(0))
            masked_rep_embedding.append(masked_embed)

        masked_rep_embedding = torch.cat(masked_rep_embedding, 0)
        masked_rep_embedding = torch.permute(masked_rep_embedding, (1, 0, 2))
        
        

        index = torch.arange(self.num_conditions, device=image.device).unsqueeze(dim=0).expand(b, self.num_conditions) # batch_size, num_conditions
        
        embed = self.masks(index) # batch_size, num_conditions, embedding_dims
        embed_feature = embed * masked_rep_embedding # batch_size, num_conditions, embedding_dims
        
        if c is None:
            final_feature = []
            
            categories_encoding = list(self.typespaces.keys()) # total_pair_categories, 2 - se obtienen todas los pares de categorias - las posibles combinaciones
            
            categories_encoding = torch.Tensor(one_hot_encoding(categories_encoding, self.unique_items)).cuda()                 # (total_pair_categories, num_categoriesx2)
            categories_encoding = categories_encoding.unsqueeze(dim = 0)
            total_pair_categories = categories_encoding.shape[1]
            
            categories_encoding = categories_encoding.expand((b, total_pair_categories, self.num_category*2))  # (batch_size, total_pair_categories num_categoriesx2)
            
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
        
        
        
class ConditionalSimNetNew(nn.Module):
    def __init__(self, args, embeddingnet, n_conditions, typespaces, rep_embed=64):
        """ args: Input arguments from the main script
            embeddingnet: The network that projects the inputs into an embedding of embedding_size
            embedding_size: Number of dimensions of the embedding output from the embeddingnet
            n_conditions: Integer defining number of different similarity notions
        """
        super(ConditionalSimNetNew, self).__init__()
        # Boolean indicating whether masks are learned or fixed
        self.learnedmask = args.learned
        
        # Boolean indicating whether masks are initialized in equally sized disjoint
        # sections or random otherwise
        self.prein = args.prein
        
        self.num_conditions = n_conditions
        self.embeddingnet = embeddingnet
        self.rep_embed = rep_embed
        self.l2_norm = args.l2_embed
        
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
        self.rep_embed = rep_embed
        
        # learn a fully connected layer rather than a mask to project the general embedding
        # into the type specific space
        masks_rep = []
        for i in range(n_conditions):
            masks_rep.append(nn.Linear(args.dim_embed, rep_embed))
        self.masks_rep = ListModule(*masks_rep)
        
        # create the mask
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

    
    def forward(self, image, c=None):
        # image (B,C,H,W)
        # image_category (B, NUM_CATE)
        # concat_categories (B, NUM_CATE * @)

        embedded_x = self.embeddingnet(image) # Batch, embedding_dims
        feature_x = embedded_x.unsqueeze(dim=1)
        b, _, _ = feature_x.size()
        feature_x = feature_x.expand((b, self.num_conditions, self.dim_embed))
        # Batch, num_conditions, embedding_dims
        
        feature_x2 = torch.permute(feature_x, (1, 0, 2))
        
        masked_rep_embedding = []
        for embed, condition in zip(feature_x2, range(self.num_conditions)):
            mask = self.masks_rep[condition]
            masked_embed = mask(embed.unsqueeze(0))
            masked_rep_embedding.append(masked_embed)

        masked_rep_embedding = torch.cat(masked_rep_embedding, 0)
        masked_rep_embedding = torch.permute(masked_rep_embedding, (1, 0, 2))
        

        index = torch.arange(self.num_conditions, device=image.device).unsqueeze(dim=0).expand(b, self.num_conditions) # batch_size, num_conditions
        
        embed = self.masks(index) # batch_size, num_conditions, embedding_dims
        embed_feature = embed * masked_rep_embedding # batch_size, num_conditions, embedding_dims
        
        if c is None:
            final_feature = []
            
            categories_encoding = list(self.typespaces.keys()) # total_pair_categories, 2 - se obtienen todas los pares de categorias - las posibles combinaciones
            
            categories_encoding = torch.Tensor(one_hot_encoding(categories_encoding, self.unique_items)).cuda()                 # (total_pair_categories, num_categoriesx2)
            categories_encoding = categories_encoding.unsqueeze(dim = 0)
            total_pair_categories = categories_encoding.shape[1]
            
            categories_encoding = categories_encoding.expand((b, total_pair_categories, self.num_category*2))  # (batch_size, total_pair_categories num_categoriesx2)
            
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
            return torch.cat((condition_feature, masked_rep_embedding), 1)
            
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
            
            # se implementa regularizacion l2
            #if self.l2_norm:
            #    norm = torch.norm(final_feature, p=2, dim=1) + 1e-10
            #    final_feature = final_feature / norm.unsqueeze(-1).expand_as(final_feature)
            

            return final_feature, mask_norm, embed_norm, embedded_x
        
