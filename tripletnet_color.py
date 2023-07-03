import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def make_fc_1d(f_in, f_out):
    return nn.Sequential(nn.Linear(f_in, f_out),
                         nn.BatchNorm1d(f_out,eps=0.001,momentum=0.01),
                         nn.ReLU(inplace=True))

def selective_margin_loss(pos_samples, neg_samples, margin, has_sample):
    """ pos_samples: Distance between positive pair
        neg_samples: Distance between negative pair
        margin: minimum desired margin between pos and neg samples
        has_sample: Indicates if the sample should be used to calcuate the loss
    """
    margin_diff = torch.clamp((pos_samples - neg_samples) + margin, min=0, max=1e6)
    num_sample = max(torch.sum(has_sample), 1)
    return torch.sum(margin_diff * has_sample) / num_sample

def accuracy(pos_samples, neg_samples):
    """ pos_samples: Distance between positive pair
        neg_samples: Distance between negative pair
    """
    is_cuda = pos_samples.is_cuda
    margin = 0
    pred = (pos_samples - neg_samples - margin).cpu().data
    acc = (pred > 0).sum()*1.0/pos_samples.size()[0]
    acc = torch.from_numpy(np.array([acc], np.float32))
    if is_cuda:
        acc = acc.cuda()
    
    return Variable(acc)

class EmbedBranch(nn.Module):
    def __init__(self, feat_dim, embedding_dim):
        super(EmbedBranch, self).__init__()
        self.fc1 = make_fc_1d(feat_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        
        # L2 normalize each feature vector
        norm = torch.norm(x, p=2, dim=1) + 1e-10
        
        x = x / norm.unsqueeze(-1).expand_as(x)
        return x

class Tripletnet(nn.Module):
    def __init__(self, args, embeddingnet, text_dim, criterion):
        super(Tripletnet, self).__init__()
        self.embeddingnet = embeddingnet
        self.text_branch = EmbedBranch(text_dim, args.dim_embed)
        self.metric_branch = None
        if args.learned_metric:
            self.metric_branch = nn.Linear(args.dim_embed, 1, bias=False)

            # initilize as having an even weighting across all dimensions
            weight = torch.zeros(1,args.dim_embed)/float(args.dim_embed)
            self.metric_branch.weight = nn.Parameter(weight)

        self.criterion = criterion
        self.margin = args.margin



    def forward(self, x, y, z):
        """ x: Anchor data
            y: Distant (negative) data
            z: Close (positive) data
        """
        # conditions only available on the anchor sample
        c = x.conditions
        embedded_x, masknorm_norm_x, embed_norm_x, general_x = self.embeddingnet(x.images, x.color, c)
        embedded_y, masknorm_norm_y, embed_norm_y, general_y = self.embeddingnet(y.images, y.color, c)
        embedded_z, masknorm_norm_z, embed_norm_z, general_z = self.embeddingnet(z.images, z.color, c)
        
        
        dist_a = F.pairwise_distance(embedded_x, embedded_y, 2)
        dist_b = F.pairwise_distance(embedded_x, embedded_z, 2)
        
        target = torch.FloatTensor(dist_a.size()).fill_(1)
        if dist_a.is_cuda:
            target = target.cuda()
        target = Variable(target)

        # type specific triplet loss
        loss_triplet = self.criterion(dist_a, dist_b, target)
        acc = accuracy(dist_a, dist_b)

        return acc, loss_triplet