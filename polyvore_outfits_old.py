from PIL import Image
import os
import os.path
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import json
import torch
import pickle
import h5py
import json
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
import itertools

def default_image_loader(path):
    return Image.open(path).convert('RGB')

def parse_iminfo(question, gt = None):
    """ Maps the questions from the FITB and compatibility tasks back to
        their index in the precomputed matrix of features

        question: List of images to measure compatibility between
        im2index: Dictionary mapping an image name to its location in a
                  precomputed matrix of features
        gt: optional, the ground truth outfit set this item belongs to
    """
    questions = []
    is_correct = np.zeros(len(question), np.bool)
    for index, im_id in enumerate(question):
        # el index es la enumeracion de las preguntas, im_id es el id de la imagen: outfit_id, index
        set_id = im_id.split('_')[0]
        
        #image_index = im_id.split('_')[1]
        if gt is None:
            gt = set_id

        questions.append(im_id)
        is_correct[index] = set_id == gt

    return questions, is_correct, gt



def load_typespaces(rootdir, rand_typespaces, num_rand_embed):
    """ loads a mapping of pairs of types to the embedding used to
        compare them
        rand_typespaces: Boolean indicator of randomly assigning type
                         specific spaces to their embedding
        num_rand_embed: number of embeddings to use when
                        rand_typespaces is true
    """
    typespace_fn = os.path.join(rootdir, 'typespaces.p')
    
    typespaces = None
    with open(typespace_fn) as f: typespaces = [line for line in f]
    
    # typespaces = pickle.load(open(typespace_fn,'rb'))
    if not rand_typespaces:
        ts = {}
        for index, t in enumerate(typespaces):
            t = t.replace('\n', '').replace("(", "").replace(")", "").replace(",", " ").split()
            t = list(map(int, t))
            t = tuple(t)
            
            ts[t] = index

        typespaces = ts
        return typespaces

    # load a previously created random typespace or create one
    # if none exist
    width = 0
    fn = os.path.join(rootdir, 'typespaces_rand_%i.p') % num_rand_embed
    if os.path.isfile(fn):
        typespaces = pickle.load(open(fn, 'rb'))
    else:
        spaces = np.random.permutation(len(typespaces))
        width = np.ceil(len(spaces) / float(num_rand_embed))
        ts = {}
        for index, t in enumerate(spaces):
            ts[typespaces[t]] = int(np.floor(index / width))

        typespaces = ts
        pickle.dump(typespaces, open(fn, 'wb'))

    return typespaces




def load_compatibility_questions(fn, image_path, outfit_data):
    """ Returns the list of compatibility questions for the
        split """
    # filtrar para considerar datos de validación y testeo

    with open(fn, 'r') as f:
        lines = f.readlines()

    compatibility_questions = []
    for line in lines:
        data = line.strip().split()
        compat_question = data[1:]
        compatibility_questions.append((compat_question, int(data[0])))
    
    # se devuelve una lista con outfitid_clothesid: [123456_1, 126653_2, 198572_3]
    #print(compatibility_questions)
    return compatibility_questions

def load_fitb_questions(fn, image_path, outfit_data):
    """ Returns the list of fill in the blank questions for the
        split """
    # filtrar para considerar datos de validación y testeo
    
    data = json.load(open(fn, 'r'))
    questions = []
    for item in data:
        question = item['question']
        q_index, _, gt = parse_iminfo(question)
        answer = item['answers']
        a_index, is_correct, _ = parse_iminfo(answer, gt)
        # debería retornar una lista de preguntas, una lista de respuestas, y un diccionario indicando si es la respuesta correcta o no.
        questions.append((q_index, a_index, is_correct))
    #print(questions)
    return questions


class TripletImageLoader(torch.utils.data.Dataset):
    def __init__(self, split, image_path, rootdir, meta_data, text_dim, rand_typespaces, num_rand_embed, transform=None, loader=default_image_loader, return_image_path=False):
        
        # path de carpetas de imágenes, de información de metadatos, y si el image loader es de entrenamiento, validacion o testeo
        self.impath = image_path
        self.infopath = meta_data
        self.is_train = split == 'train'
        self.return_image_path = return_image_path
        
        # se carga el metadata de entenamiento, validación o testeo, dependiendo de que se especifique, y se guardan
        # el metadata contiene información de likes, precio, categoría de las prendas, descripción y nombre
        data_json = os.path.join(self.infopath, '%s_no_dup.json' % split)
        outfit_data = json.load(open(data_json, 'r'))
        self.data = outfit_data
        
        # otros datos importantes
        self.transform = transform
        self.loader = loader
        self.split = split
        self.text_feat_dim = text_dim
        catlist = [3, 7, 11, 17, 19, 21, 24, 25, 27, 28, 29, 41, 51, 55]
           
        
        # imname contiene la lista de todos los ítemes de todos los outfits en orden de lectura (como se muestran en el archivo)
        # si se quiere guardar otra cosa, aparte de categoria, se debe seguir lo mismo que se hizo para categoria
        imnames = []
        imcategory = []
        category2ims = {}
        im2desc = {}
         
        #imnames = set()
        #imcategory = set()
        for outfit in outfit_data:
            items = outfit['items']
            outfit_id = outfit['set_id']
            
            for item in items:
                item_name = '%s_%s' % (outfit_id, item['index'])
                category = int(item['categoryid'])
                desc = item['name']
                
                desc = desc.replace('\n','').encode('ascii', 'ignore').strip().lower()
                im2desc[item_name] = desc
                
                imnames.append(item_name)
                imcategory.append(category)
                
                if category not in category2ims:
                    category2ims[category] = {}

                if outfit_id not in category2ims[category]:
                    category2ims[category][outfit_id] = []

                category2ims[category][outfit_id].append(item_name)
                #print(item_name, item['categoryid'])
        #self.imnames = list(imnames)
        #self.imcategory = list(imcategory)
        
        
        
        self.imnames = imnames
        self.imcategory = imcategory
        self.category2ims = category2ims
        self.im2desc = im2desc
        self.typespaces = load_typespaces(rootdir, rand_typespaces, num_rand_embed)
        
        print(self.typespaces)
        
        # si es un imageLoader de entrenamiento
        if self.is_train or split == 'valid':

            # At train time we pull the list of outfits and enumerate the pairwise
            # comparisons between them to train with. Originally, also negative pairs were pulled by the
            # __get_item__ function, but now we dont need it. Still, it will be implemented
            
            # se generan pares positivos de ejemplos y se guardan en la variable pos_pairs
            pos_pairs = []
            
            # compute the metadata of the postive pairs
            pos_pairs_metadata=[]
            
            max_items = 0
            # lo guardamos de la forma 'outfit_id__clothes_index'
            for outfit in outfit_data:
                items = outfit['items']
                cnt = len(items)
                max_items = max(cnt, max_items)
                outfit_id = outfit['set_id']
                
                
                outfit_clothes = []
                #for item in items:
                    #itemtype = item['categoryid']
                    
                    #if itemtype in catlist:
                    #    outfit_clothes.append('%s_%s' % (outfit_id, item['index']))
                #print(len(outfit_clothes), outfit_clothes)
                    
                    
                outfit_clothes = [ '%s_%s' % (outfit_id, s['index']) for s in items] # CAMBIAR ACA SI SE QUIEREN 
                                                                                     # RESTRINGIR LAS ROPAS POR CATEGORIA
                
                pairs = list(itertools.combinations(outfit_clothes, 2))
                pos_pairs.extend(pairs)
                #outfit_folder = os.path.join(polyvore_images, outfit_id)
                #if os.path.exists(outfit_folder):
                    
                    #outfit_clothes = [ os.path.join(outfit_folder, s) for s in os.listdir(outfit_folder)]
                    #pairs = list(itertools.combinations(outfit_clothes, 2))
                    #pos_pairs.append(pairs)

            self.pos_pairs = pos_pairs
            self.max_items = max_items
        else:
            # si no es de entrenamiento, se 
            # pull the two task's questions for test and val splits
            fn = os.path.join(meta_data, 'fill_in_blank_test.json')
            self.fitb_questions = load_fitb_questions(fn, image_path, outfit_data)
            fn = os.path.join(meta_data, 'fashion_compatibility_prediction.txt')
            self.compatibility_questions = load_compatibility_questions(fn, image_path, outfit_data)

    def load_train_item(self, clothes_id):
        """ Returns a single item in the doublet and its data
        """
        outfit_id, image_index_id = clothes_id.split('_')
        imfn = os.path.join(self.impath, outfit_id, '%s.jpg' % image_index_id)
        img = self.loader(imfn)
        
        # ver si sacar self.transform
        if self.transform is not None:
            img = self.transform(img)
        
        text_features = np.zeros(self.text_dim, np.float32)
        if clothes_id in self.im2desc:
            text = self.im2desc[clothes_id]
            has_text = 1
        
        else:
            has_text = 0
        
        has_text = np.float32(has_text)
        img_index = self.imnames.index(clothes_id)
        img_category = self.imcategory[img_index]
        
        return img, text_features, has_text, img_category

    
    # hay que haber computado los embeddings de antemano para que funcione
    def test_compatibility(self, embeds, metric):
        """ Returns the area under a roc curve for the compatibility
            task

            embeds: precomputed embedding features used to score
                    each compatibility question
            metric: a function used to score the elementwise product
                    of a pair of embeddings, if None euclidean
                    distance is used
        """
        
        scores = []
        # se crea una lista de ceros
        labels = np.zeros(len(self.compatibility_questions), np.int32)
        # para cada elemento en self.compatibility_question, se obtiene el index
        for index, (outfit, label) in enumerate(self.compatibility_questions):
            # se define que el valor guardado en index es el valor de compatibilidad
            # se decir, se guardan en orden si los outfits son compatibles o no
            labels[index] = label
            n_items = len(outfit)
            outfit_score = 0.0
            num_comparisons = 0.0
            for i in range(n_items-1):
                # un outfit de compatibility_question esta compuesto por el puntaje de compatibilidad, puntaje de compatibilidad
                # y la imágen de la prenda, o id
                # SACAR SOLAMENTE EL ITEM I, Y BUSCAR EN DICCIONARIO EMBEDS POR EL VALOR DEL EMBEDDING
                
                #item1, img1 = outfit[i]
                # se busca el índice en la lista de indices donde se encuentra el item1, ver si es igual a pasarle el indice cualquiera
                item1 = outfit[i]
                item1index = self.imnames.index(item1)
                type1 = self.imcategory[item1index]
                
                for j in range(i+1, n_items):
                    # se obtiene el puntaje de compatibilidad e imágen de la prenda para las siguientes prendas
                    item2 = outfit[j]
                    item2index = self.imnames.index(item2)
                    type2 = self.imcategory[item2index]
                    
                    condition = self.get_typespace(type1, type2)
                    
                    # ver como sacar los embeds, por la forma en que están escritos, deberían poder recuperarse con 
                    embed1 = embeds[item1index][condition].unsqueeze(0)
                    embed2 = embeds[item2index][condition].unsqueeze(0)
                    
                    if metric is None:
                        outfit_score += torch.nn.functional.pairwise_distance(embed1, embed2, 2)
                    else:
                        outfit_score += metric(Variable(embed1 * embed2)).data

                    num_comparisons += 1.
                
            outfit_score /= num_comparisons
            scores.append(outfit_score)
            
        scores = torch.cat(scores).squeeze().cpu().numpy()
        #scores = np.load('feats.npy')
        #print(scores)
        #assert(False)
        #np.save('feats.npy', scores)
        
        print(labels)
        print(1 - scores)
        auc = roc_auc_score(labels, 1 - scores, multi_class='ovr')
        return auc
    
    
    def sample_negative(self, outfit_id, item_id, item_type):
        """ Returns a randomly sampled item from a different set
            than the outfit at data_index, but of the same type as
            item_type
        
            data_index: index in self.data where the positive pair
                        of items was pulled from
            item_type: the coarse type of the item that the item
                       that was paired with the anchor
        """
        
        candidate_sets = self.category2ims[item_type].keys()
        attempts = 0
        item_out = item_id
        while item_out == item_id and attempts < 100:
            choice = np.random.choice(list(candidate_sets))
            items = self.category2ims[item_type][choice]
            item_index = np.random.choice(range(len(items)))
            item_out = items[item_index]
            attempts += 1
            
        return item_out
    def get_typespace(self, anchor, pair):
        """ Returns the index of the type specific embedding
            for the pair of item types provided as input
        """
        query = (anchor, pair)
        if query not in self.typespaces:
            query = (pair, anchor)

        return self.typespaces[query]
    

    def test_fitb(self, embeds, metric):
        """ Returns the accuracy of the fill in the blank task

            embeds: precomputed embedding features used to score
                    each compatibility question
            metric: a function used to score the elementwise product
                    of a pair of embeddings, if None euclidean
                    distance is used
        """
        correct = 0.
        n_questions = 0.
        for q_index, (questions, answers, is_correct) in enumerate(self.fitb_questions):
            answer_score = np.zeros(len(answers), dtype=np.float32)
            for index, answer in enumerate(answers):
                
                # se busca el índice en la lista de indices donde se encuentra el item1, ver si es igual a pasarle el indice cualquiera
                answerindex = self.imnames.index(answer)
                answertype = self.imcategory[answerindex]
                
                score = 0.0
                for question in questions:
                    # se busca el índice en la lista de indices donde se encuentra el item1, ver si es igual a pasarle el indice cualquiera
                    questionindex = self.imnames.index(question)
                    questiontype = self.imcategory[questionindex]
                    
                    condition = self.get_typespace(answertype, questiontype)
                    
                    embed1 = embeds[questionindex][condition].unsqueeze(0)
                    embed2 = embeds[answerindex][condition].unsqueeze(0)
                    
                    if metric is None:
                        score += torch.nn.functional.pairwise_distance(embed1, embed2, 2)
                    else:
                        score += metric(Variable(embed1 * embed2)).data

                answer_score[index] = score.squeeze().cpu().numpy()
            
            correct += is_correct[np.argmin(answer_score)]
            n_questions += 1
                        
        # scores are based on distances so need to convert them so higher is better
        acc = correct / n_questions
        return acc

    def __getitem__(self, index):
        if self.is_train or self.split == 'valid':
            anchor_im, pos_im = self.pos_pairs[index]
            outfit_id = anchor_im.split('_')[0]
            
            # cambiar aca la cantidad de argumentos si se quieren recibir mas datos de las imagenes (texto de descripcion por               # ejemplo)
            
            img1, desc1, has_text1, img1cat = self.load_train_item(anchor_im)
            img2, desc2, has_text2, img2cat = self.load_train_item(pos_im)
            
            ## sample negative examples
            neg_im = self.sample_negative(outfit_id, pos_im, img2category)
            img3, desc3, has_text3, img3cat = self.load_train_item(neg_im)
            
            condition = self.get_typespace(anchor_type, item_type)
            
            if self.return_image_path: #return img1, img1category, anchor_im, img2, img2category, pos_im
                return img1, img1category, anchor_im, img2, img2category, pos_im, img3, img3category, neg_im
            
            #return img1, img1category, img2, img2category
            return img1, desc1, has_text1, img2, desc2, has_text2, img3, desc3, has_text3, condition
        
        
        outfit_id, clothes_index = self.imnames[index].split('_')
        img_category = self.imcategory[index]
        img_path = os.path.join(self.impath, outfit_id, '%s.jpg' % clothes_index)
        img1 = self.loader(img_path)
        if self.transform is not None:
            img1 = self.transform(img1)
        
        if self.return_image_path: return img1, img_category, img_path
        return img1, img_category
    


    def shuffle(self):
        np.random.shuffle(self.pos_pairs)
        
    def __len__(self):
        if self.is_train:
            return len(self.pos_pairs)
        return len(self.imnames)
        
        
        