# path de las imágenes de polyvore
import os
import itertools
import argparse
import os
import sys
import shutil
import json
import numpy as np
import torch

import torch.nn as nn
from torch.nn import MSELoss

import torch.nn.functional as F
import torch.optim as optim
# el mismo
from torchvision import transforms
import torchvision.transforms as T

from torch.autograd import Variable
import torch.backends.cudnn as cudnn

# important libraries for ML models
from torchvision import models
from torchvision.models import resnet50, ResNet50_Weights
from polyvore_dataset_loader import DoubletImageLoader


from bimodal_byol_shoes.data.custom_transforms import BatchTransform, ListToTensor, PadToSquare, SelectFromTuple
from bimodal_byol_shoes.models.BYOL2_model import BYOL2




# Training settings
parser = argparse.ArgumentParser(description='Fashion Compatibility Example')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='number of start epoch (default: 1)')
parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                    help='learning rate (default: 5e-5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--log-interval', type=int, default=250, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='Type_Specific_Fashion_Compatibility', type=str,
                    help='name of experiment')
parser.add_argument('--polyvore_split', default='nondisjoint', type=str,
                    help='specifies the split of the polyvore data (either disjoint or nondisjoint)')
parser.add_argument('--datadir', default='data', type=str,
                    help='directory of the polyvore outfits dataset (default: data)')
parser.add_argument('--test', dest='test', action='store_true', default=False,
                    help='To only run inference on test set')
parser.add_argument('--dim_embed', type=int, default=64, metavar='N',
                    help='how many dimensions in embedding (default: 64)')
parser.add_argument('--use_fc', action='store_true', default=False,
                    help='Use a fully connected layer to learn type specific embeddings.')
parser.add_argument('--learned', dest='learned', action='store_true', default=False,
                    help='To learn masks from random initialization')
parser.add_argument('--prein', dest='prein', action='store_true', default=False,
                    help='To initialize masks to be disjoint')
parser.add_argument('--rand_typespaces', action='store_true', default=False,
                    help='randomly assigns comparisons to type-specific embeddings where #comparisons < #embeddings')
parser.add_argument('--num_rand_embed', type=int, default=4, metavar='N',
                    help='number of random embeddings when rand_typespaces=True')
parser.add_argument('--l2_embed', dest='l2_embed', action='store_true', default=False,
                    help='L2 normalize the output of the type specific embeddings')
parser.add_argument('--learned_metric', dest='learned_metric', action='store_true', default=False,
                    help='Learn a distance metric rather than euclidean distance')
parser.add_argument('--margin', type=float, default=0.3, metavar='M',
                    help='margin for triplet loss (default: 0.2)')
parser.add_argument('--embed_loss', type=float, default=5e-4, metavar='M',
                    help='parameter for loss for embedding norm')
parser.add_argument('--mask_loss', type=float, default=5e-4, metavar='M',
                    help='parameter for loss for mask norm')
parser.add_argument('--vse_loss', type=float, default=5e-3, metavar='M',
                    help='parameter for loss for the visual-semantic embedding')
parser.add_argument('--sim_t_loss', type=float, default=5e-5, metavar='M',
                    help='parameter for loss for text-text similarity')
parser.add_argument('--sim_i_loss', type=float, default=5e-5, metavar='M',
                    help='parameter for loss for image-image similarity')


# path to important folders
polyvore_dataset = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'Datasets', 'Polyvore')
polyvore_images = os.path.join(polyvore_dataset, 'Polyvore_images', 'images')
polyvore_info = os.path.join(polyvore_dataset, 'Polyvore_info')

polyvore_train = os.path.join(polyvore_info, 'train_no_dup')
polyvore_val = os.path.join(polyvore_info, 'valid_no_dup')
polyvore_test = os.path.join(polyvore_info, 'test_no_dup')

meta_data_train = json.load(open(polyvore_train, 'r'))
meta_data_val = json.load(open(polyvore_val, 'r'))
meta_data_test = json.load(open(polyvore_test, 'r'))

def main():
    torch.set_printoptions(linewidth=200)
    
    global args
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # las transformaciones a los datos, revisar bien que transformaciones aplicar
    # transformaciones a las imágenes 1 e imágenes 2
    #transforms_1 = T.Compose([
    #    BatchTransform(SelectFromTuple(0)),
        #BatchTransform(PadToSquare(255)),
        #BatchTransform(T.Resize((224,224))),
        #ListToTensor('cuda', torch.float),
    #])
    #transforms_2 = T.Compose([
    #    BatchTransform(SelectFromTuple(1)),
        #BatchTransform(PadToSquare(255)),
        #BatchTransform(T.Resize((224,224))),
        #ListToTensor('cuda', torch.float),
    #])

    transforms_1 = BatchTransform(SelectFromTuple(0))
    transforms_2 = BatchTransform(SelectFromTuple(1))

    # encoder for getting initial embeddings
    encoder = models.resnet50(weights='DEFAULT')

    #encoder.load_state_dict(torch.load('../checkpoints/resnet50_byol_quickdraw_128_1000_v3.pt'))
    empty_transform = T.Compose([])
    byol_learner = BYOL2(
        encoder,
        image_size=224,
        hidden_layer='avgpool',
        augment_fn=empty_transform,
        cosine_ema_steps=args.epochs*args.epoch_size
    )

    if args.cuda:
        byol_learner.cuda()

    # parámetros pre-definidos (revisar que se encuentren bien ajustados para la tarea )
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    # asegurarse que la carpeta exista
    models_folder = '/checkpoint_models'
    
    text_feature_dim = 6000
    kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
    
    # se define el DataLoader de los datos de testeo
    test_loader = torch.utils.data.DataLoader(
        DoubletImageLoader(args, 'test', polyvore_images, polyvore_info,
                           transform=transforms.Compose([
                               transforms.Resize((224, 224)),
                               transforms.CenterCrop(112),
                               transforms.ToTensor(),
                               normalize,
                           ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)


    # dataloaders for training and validation da
    train_loader = torch.utils.data.DataLoader(
        DoubletImageLoader(args, 'train', polyvore_images, polyvore_info, text_dim=text_feature_dim,
                           transform=transforms.Compose([
                               transforms.Resize((224, 224)),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               normalize,
                           ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    
    val_loader = torch.utils.data.DataLoader(
        DoubletImageLoader(args, 'valid', polyvore_images, polyvore_info,
                           transform=transforms.Compose([
                               transforms.Resize((224, 224)),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               normalize,
                           ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    best_acc = 0

    # VER DESPUES
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_prec1']
            #dnet.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True    
    if args.test:
        test_acc = test(byol_learner, test_loader)
        sys.exit()
    
    # se obtienen los parámetros del modelo de aprendizaje
    parameters = byol_learner.parameters()
    n_parameters = sum([p.data.nelement() for p in parameters])
    
    # se define el optimizador
    optimizer = torch.optim.Adam(byol_learner.parameters(), lr=args.lr)
    
    print("Entra a la iteración")
    for epoch in range(args.start_epoch, args.epochs + 1):
        print("Número de épocas: ", epoch)
        
        # update learning rate
        adjust_learning_rate(args.lr, optimizer, epoch)
        
        # train for one epoch
        train(byol_learner, train_loader, optimizer, epoch, args.log_interval)
        # evaluate on validation set
        val_loss = valid_loss(byol_learner, val_loader)

        # remember best acc and save checkpoint
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        save_checkpoint(models_folder, {
            'epoch': epoch + 1,
            'state_dict': byol_learner.state_dict(),
            'best_prec1': best_val_loss,
        }, is_best)

    checkpoint = torch.load(os.path.join(models_folder, 'model_best.pth.tar'))
    byol_learner.load_state_dict(checkpoint['state_dict'])
    test_acc = test(byol_learner, test_loader)
    print(test_acc)


def calculate_loss(criteria, img1embed, img2embed):
    return F.pairwise_distance(img1embed, img2embed)


def train(learner, train_loader, opt, epoch, log_interval):
    losses = AverageMeter()

    # switch to train mode
    print()
    for batch_idx, images in enumerate(train_loader):
        loss = learner(images)
        # entrenamiento del modelo
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        learner.update_moving_average() # update moving average of target encoder
        
        num_items = images[0].shape[0]
                            
        # measure accuracy and record loss
        losses.update(loss, num_items)
        #emb_norms.update(embedding)
            
        # compute gradient and do optimizer step
        opt.zero_grad()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f} ({:.4f}) \t'.format(
                epoch, batch_idx * num_items, len(train_loader.dataset),
                losses.val, losses.avg))

            
def valid_loss(learner, test_loader):
    losses = AverageMeter()

    # switch to train mode
    print()
    for batch_idx, (img1, img2) in enumerate(test_loader):
        
        # entrenamiento del modelo
        _ , img1embed = learner(img1, return_embedding = True)
        _ , img2embed = learner(img2, return_embedding = True)
        
        # definir función de perdida (OTRA MAS)
        loss = torch.sum(F.pairwise_distance(img1embed, img2embed))
        
        num_items = len(img1)
                            
        # measure accuracy and record loss
        losses.update(loss, num_items)
        #emb_norms.update(embedding)
        
        print("Validation Loss Total: {}\t".format(losses.sum))
        print()
        return losses.sum
    
    

def test(learner, test_loader):
    global cuda
    # switch to evaluation mode
    embeddings_list = []
    # for test/val data we get images only from the data loader
    contador = 0
    for batch_idx, images in enumerate(test_loader):
        print(batch_idx, len(images))
        #if cuda:
        #    images = images.cuda()
        projections, embeddings = learner(images, return_embedding = True)
        embeddings_list.append(embeddings)
        print(embeddings.shape)
        print()
        contador +=1
        if contador == 10: break
        
    embeddings_list = torch.cat(embeddings_list)
    metric = roc_auc_score
    auc = test_loader.dataset.test_compatibility(embeddings_list, metric)
    acc = test_loader.dataset.test_fitb(embeddings_list, metric)
    total = auc + acc
    print('\n{} set: Compat AUC: {:.2f} FITB: {:.1f}\n'.format(
        test_loader.dataset.split,
        round(auc, 2), round(acc * 100, 1)))
    
    return total

def save_checkpoint(models_folder, state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = models_folder
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(directory, 'model_best.pth.tar'))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * ((1 - 0.015) ** epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()    
