import torch
import random
import numpy as np
import torch.nn as nn
from pathlib import Path
import logging
import json
from plotnine import *
import pandas as pd
import ursa.similarity as S
import ursa.util as U
import pickle
DECILES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

def analyze(output_root_dir, layers, nb_samples=70000):
    output_dir = Path(output_root_dir) / 'attn'
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Attention pooling; global diagnostic")
    config = dict(directory=output_root_dir,
                  output=output_dir,
                  attention='linear',
                  standardize=True,
                  hidden_size=None,
                  attention_hidden_size=None,
                  epochs=500,
                  test_size=1/2,
                  layers=layers,
                  device='cuda',
                  runs=3)
    global_diagnostic(config)

    logging.info("Attention pooling; global RSA")
    config = dict(directory=output_root_dir,
                  output=output_dir,
                  attention='linear',
                  standardize=True,
                  attention_hidden_size=None,
                  epochs=60,
                  test_size=1/2,
                  layers=layers,
                  device='cuda',
                  runs=3)
    global_rsa(config)


    output_dir = Path(output_root_dir) / 'mean'
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Mean pooling; global diagnostic")
    config = dict(directory=output_root_dir,
                  output=output_dir,
                  attention='mean',
                  hidden_size=None,
                  attention_hidden_size=None,
                  epochs=500,
                  test_size=1/2,
                  layers=layers,
                  device='cuda',
                  runs=3)
    global_diagnostic(config)

    logging.info("Mean pooling; global RSA")
    config = dict(directory=output_root_dir,
                  output=output_dir,
                  attention='mean',
                  epochs=60,
                  test_size=1/2,
                  layers=layers,
                  device='cpu',
                  runs=1)
    global_rsa(config)


    output_dir = Path(output_root_dir) / 'local'
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Local diagnostic")
    config = dict(directory=output_root_dir,
                  output=output_dir,
                  hidden=None,
                  epochs=40,
                  layers=layers,
                  runs=3)
    local_diagnostic(config)

    logging.info("Local RSA")
    config = dict(directory=output_root_dir,
                  output=output_dir,
                  size=nb_samples // 2,
                  layers=layers,
                  matrix=False,
                  runs=1)
    local_rsa(config)


## Models
### Local

def local_diagnostic(config):
    directory = config['directory']
    out = config["output"] / "local_diagnostic.json"
    del config['output']
    runs = range(1, config['runs']+1)
    del config['runs']
    output = []
    for run in runs:
        logging.info("Starting run {}".format(run))
        data_mfcc = pickle.load(open('{}/local_input.pkl'.format(directory), 'rb'))
        for mode in ['trained', 'random']:
            if 'mfcc' in config['layers']:
                logging.info("Fitting local classifier for mfcc")
                result  = local_classifier(data_mfcc['features'], data_mfcc['labels'], epochs=config['epochs'], device='cuda:0', hidden=config['hidden'])
                logging.info("Result for {}, {} = {}".format(mode, 'mfcc', result['acc']))
                result['model'] = mode
                result['layer'] = 'mfcc'
                result['run'] = run
                output.append(result)
            for layer in set(config['layers'])-set(['mfcc']):
                data = pickle.load(open('{}/local_{}_{}.pkl'.format(directory, mode, layer), 'rb'))
                logging.info("Fitting local classifier for {}, {}".format(mode, layer))
                result = local_classifier(data[layer]['features'], data[layer]['labels'], epochs=config['epochs'], device='cuda:0', hidden=config['hidden'])
                logging.info("Result for {}, {} = {}".format(mode, layer, result['acc']))
                result['model'] = mode
                result['layer'] = layer
                result['run'] = run
                output.append(result)
    logging.info("Writing {}".format(out))
    json.dump(output, open(out, "w"), indent=True)

def local_rsa(config):
    out = config['output'] / "local_rsa.json"
    del config['output']
    del config['runs']
    if config['matrix']:
        raise NotImplementedError
        #result = framewise_RSA_matrix(directory, layers=config['layers'], size=config['size'])
    else:
        del config['matrix']
        result = framewise_RSA(**config)
    logging.info("Writing {}".format(out))
    json.dump(result, open(out, 'w'), indent=2)

### Global

def global_rsa(config):
    out = config['output'] / 'global_rsa.json'
    del config['output']
    result = []
    runs = range(1, config['runs']+1)
    del config['runs']
    for run in runs:
        result += inject(weighted_average_RSA(**config), {'run': run})
    logging.info("Writing {}".format(out))
    json.dump(result, open(out, 'w'), indent=2)

def global_rsa_partial(config):
    out = config['output'] / 'global_rsa_partial.json'
    del config['output']
    del config['runs']
    result = weighted_average_RSA_partial(**config)
    json.dump(result, open(out, 'w'), indent=2)

def global_diagnostic(config):
    out = config['output'] / 'global_diagnostic.json'
    del config['output']
    result = []
    runs = range(1, config['runs']+1)
    del config['runs']
    for run in runs:
        logging.info("Starting run {}".format(run))
        result += inject(weighted_average_diagnostic(**config), {'run':run})
    logging.info("Writing {}".format(out))
    json.dump(result, open(out, 'w') , indent=2)


def plot_pooled_feature_std():
    path = 'data/out/rnn-vgs/'
    layers=['conv', 'rnn0', 'rnn1', 'rnn2', 'rnn3']
    data = pd.DataFrame()
    for model in ['trained', 'random']:
        layer = 'mfcc'
        act = pickle.load(open("{}/global_input.pkl".format(path), "rb"))['audio']
        act_avg = np.stack([x.mean(axis=0) for x in act])
        rows=pd.DataFrame(data=dict(std=act_avg.std(axis=1), model=np.repeat(model, len(act_avg)), layer=np.repeat(layer, len(act_avg))))
        data = data.append(rows)
        for layer in layers:
            act = pickle.load(open("{}/global_{}_{}.pkl".format(path, model, layer), "rb"))[layer]
            act_avg = np.stack([x.mean(axis=0) for x in act])
            rows=pd.DataFrame(data=dict(std=act_avg.std(axis=1), model=np.repeat(model, len(act_avg)), layer=np.repeat(layer, len(act_avg))))
            data = data.append(rows)
    order = list(data['layer'].unique())
    data['layer_id'] = [ order.index(x) for x in data['layer'] ]
    # Only plot RNN layers
    data = data[data['layer'].str.startswith('rnn')]
    z = data.groupby(['layer_id', 'model']).mean().reset_index()
    g = ggplot(data, aes(x='layer_id', y='std', color='model')) + \
                            geom_point(alpha=0.1, size=2, position='jitter', fill='white') +  \
                            geom_line(data=z, size=2, alpha=1.0) + \
                            ylab("Standard deviation") +\
                            xlab("layer id")
    ggsave(g, 'fig/rnn-vgs/pooled_feature_std.png')


def majority_binary(y):
    return (y.mean(dim=0) >= 0.5).float()

def majority_multiclass(y):
    labels, counts = np.unique(y, return_counts=True)
    return labels[counts.argmax()]

def local_classifier(features, labels, test_size=1/2, epochs=1, device='cpu', hidden=None, weight_decay=0.0):
    """Fit classifier on part of features and labels and return accuracy on the other part."""
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    # remove sil
    sil = labels != 'sil'
    features = features[sil]
    labels   = labels[sil]

    splitseed = random.randint(0, 1024)

    X, X_val, y, y_val = train_test_split(features, labels, test_size=test_size, random_state=splitseed)

    le = LabelEncoder()
    y = torch.tensor(le.fit_transform(y)).long()
    y_val = torch.tensor(le.transform(y_val)).long()

    scaler = StandardScaler()
    X = torch.tensor(scaler.fit_transform(X)).float()
    X_val  = torch.tensor(scaler.transform(X_val)).float()
    logging.info("Setting up model on {}".format(device))
    if hidden is None:
        model = SoftmaxClassifier(X.size(1), y.max().item()+1, weight_decay=weight_decay).to(device)
    else:
        model = MLP(X.size(1), y.max().item()+1, hidden_size=hidden).to(device)
    result = train_classifier(model, X, y, X_val, y_val, epochs=epochs, majority=majority_multiclass)
    return result


def weight_variance():
    kinds = ['rnn0', 'rnn1', 'rnn2', 'rnn3']
    w = []
    layer = []
    trained = []
    for kind in kinds:
        rand = np.load("logreg_w_random_{}.npy".format(kind)).flatten()
        train = np.load("logreg_w_trained_{}.npy".format(kind)).flatten()
        w.append(rand)
        w.append(train)
        for _ in rand:
            layer.append(kind)
            trained.append('random')
        for _ in train:
            layer.append(kind)
            trained.append('trained')
        print(kind, "random", np.var(rand))
        print(kind, "trained", np.var(train))
    data = pd.DataFrame(dict(w = np.concatenate(w), layer=np.array(layer), trained=np.array(trained)))
    #g = ggplot(data, aes(y='w', x='layer')) + geom_violin() + facet_wrap('~trained', nrow=2, scales="free_y")
    g = ggplot(data, aes(y='w', x='layer')) + geom_point(position='jitter', alpha=0.1) + facet_wrap('~trained', nrow=2, scales="free_y")
    ggsave(g, 'weight_variance.png')


def framewise_RSA(directory='.', layers=[], size=70000):
    result = []
    mfcc_done = False
    data = pickle.load(open("{}/local_input.pkl".format(directory), "rb"))
    for mode in ["trained", "random"]:
        mfcc_cor = [ item['cor']  for item in result if item['layer'] == 'mfcc']
        if len(mfcc_cor) > 0:
            logging.info("Result for MFCC computed previously")
            result.append(dict(model=mode, layer='mfcc', cor=mfcc_cor[0]))
        else:
            cor = correlation_score(data['features'], data['labels'], size=size)
            logging.info("Point biserial correlation for {}, mfcc: {}".format(mode, cor))
            result.append(dict(model=mode, layer='mfcc', cor=cor))
        for layer in layers:
            logging.info("Loading phoneme data for {} {}".format(mode, layer))
            data = pickle.load(open("{}/local_{}_{}.pkl".format(directory, mode, layer), "rb"))
            cor = correlation_score(data[layer]['features'], data[layer]['labels'], size=size)
            logging.info("Point biserial correlation for {}, {}: {}".format(mode, layer, cor))
            result.append(dict(model=mode, layer=layer, cor=cor))
    return result

def correlation_score(features, labels, size):
    from sklearn.metrics.pairwise import paired_cosine_distances
    from scipy.stats import pearsonr
    logging.info("Sampling 2x{} stimuli from a total of {}".format(size, len(labels)))
    indices = np.array(random.sample(range(len(labels)), size*2))
    y = labels[indices]
    x = features[indices]
    y_sim = y[: size] == y[size :]
    x_sim = 1 - paired_cosine_distances(x[: size], x[size :])
    return pearsonr(x_sim, y_sim)[0]

def ed_rsa(directory='.', layers=[], test_size=1/2, quantiles=DECILES):
    from sklearn.model_selection import train_test_split
    from nltk.tokenize import word_tokenize
    from sklearn.preprocessing import LabelEncoder
    def flatten(xss):
        return [ x for xs in xss for x in xs ] 
    device = 'cpu'
    result = []
    logging.info("Loading transcription data")
    data_in = pickle.load(open("{}/global_input.pkl".format(directory), "rb"))
    trans = data_in['ipa']
    aid = data_in['audio_id']
    words  = [ word_tokenize(x) for x in data_in['text'] ]
    le = LabelEncoder()
    le.fit(flatten(words))
    text = [ le.transform(s) for s in words ]
    splitseed = random.randint(0, 1024)
    _, aid, _, trans, _, text = train_test_split(aid, trans, text, test_size=test_size, random_state=splitseed)
    logging.info("Converting word IDs to Unicode characters before computing edit distance")
    text = [ ''.join(chr(i) for i in s) for s in text ] 
    logging.info("Computing phoneme edit distances for transcriptions")
    trans_sim = torch.tensor(U.pairwise(S.stringsim, trans)).float().to(device)
    logging.info("Computing word edit distance for text")
    text_sim = torch.tensor(U.pairwise(S.stringsim, text)).float().to(device)
    logging.info("Saving metadata and similarity matrices for transcriptions.")
    torch.save(aid, "{}/aid.pt".format(directory))
    torch.save(trans_sim, "{}/trans_sim.pt".format(directory))
    torch.save(text_sim, "{}/text_sim.pt".format(directory))
    logging.info("Computing RSA correlation between phoneme strings and word sequences")
    cor =  S.pearson_r(S.triu(trans_sim), S.triu(text_sim))
    logging.info("RSA for phonemes vs words: {}".format(cor))
    for mode in ['trained', 'random']:
        for layer in layers:
            logging.info("Loading activations for {} {}".format(mode, layer))
            data = pickle.load(open("{}/global_{}_{}.pkl".format(directory, mode, layer), "rb"))
            act = data[layer] 
            logging.info("Sampling data for {} {}".format(mode, layer))
            _, act = train_test_split(act, test_size=test_size, random_state=splitseed)
            logging.info("Converting to symbols and collapsing runs")
            codes = [ ''.join(collapse_runs([ chr(x) for x in item.argmax(axis=1)])) for item in act ]
            logging.info("Computing edit distances for codes")
            codes_sim = torch.tensor(U.pairwise(S.stringsim, codes)).float().to(device)
            logging.info("Saving similarity matrix for {} {}".format(mode, layer))
            torch.save(codes_sim, "{}/codes_sim_{}_{}.pt".format(directory, mode, layer))
            logging.info("Computing RSA correlation with phoneme strings")
            cor_phoneme = S.pearson_r(S.triu(trans_sim), S.triu(codes_sim))
            result.append({'cor': cor_phoneme.item(), 'model': mode, 'layer': layer, 'reference': 'phoneme', 'by_size': False})
            logging.info("Computing RSA correlation with word sequences")
            cor_word = S.pearson_r(S.triu(text_sim), S.triu(codes_sim))
            result.append({'cor': cor_word.item(), 'model': mode, 'layer': layer, 'reference': 'word', 'by_size': False})
            for record in ed_rsa_by_size(data_in, aid, trans_sim, text_sim, codes_sim, quantiles=quantiles):
                record['model'] = mode
                record['layer'] = layer
                result.append(record)
    return result

def ed_rsa_by_size(data, aid, trans_sim, text_sim, codes_sim, quantiles=DECILES):
    size = np.array([len(data['audio'][i]) for i in range(len(data['audio'])) if data['audio_id'][i] in aid ])
    qs = np.quantile(size, quantiles)
    for i in range(1, len(qs)):
        j = (size > qs[i-1]) & (size <= qs[i])
        tr = trans_sim[j, :][:, j]
        te = text_sim[j, :][:, j]
        co = codes_sim[j, :][:, j]
        cor_phoneme = S.pearson_r(S.triu(tr), S.triu(co))
        cor_word  = S.pearson_r(S.triu(te), S.triu(co))
        yield dict(quantile=quantiles[i], gt=qs[i-1], leq=qs[i], cor=cor_phoneme.item(), reference='phoneme', by_size=True)
        yield dict(quantile=quantiles[i], gt=qs[i-1], leq=qs[i], cor=cor_word.item(),  reference='word', by_size=True)

def weighted_average_RSA(directory='.', layers=[], attention='linear', test_size=1/2,  attention_hidden_size=None, standardize=False, epochs=1, device='cpu'):
    from sklearn.model_selection import train_test_split
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    splitseed = random.randint(0, 1024)
    result = []
    logging.info("Loading transcription data")
    data = pickle.load(open("{}/global_input.pkl".format(directory), "rb"))
    trans = data['ipa']
    act = [ torch.tensor([item[:, :]]).float().to(device) for item in data['audio'] ]

    trans, trans_val, act, act_val = train_test_split(trans, act, test_size=test_size, random_state=splitseed)
    if standardize:
        logging.info("Standardizing data")
        act, act_val = normalize(act, act_val)
    logging.info("Computing edit distances")
    edit_sim = torch.tensor(U.pairwise(S.stringsim, trans)).float().to(device)
    edit_sim_val = torch.tensor(U.pairwise(S.stringsim, trans_val)).float().to(device)
    logging.info("Training for input features")
    this = train_wa(edit_sim, edit_sim_val, act, act_val, attention=attention, attention_hidden_size=None, epochs=epochs, device=device)
    result.append({**this, 'model': 'random', 'layer': 'mfcc'})
    result.append({**this, 'model': 'trained', 'layer': 'mfcc'})
    del act, act_val
    logging.info("Maximum correlation on val: {} at epoch {}".format(result[-1]['cor'], result[-1]['epoch']))
    for mode in ["trained", "random"]:
        for layer in layers:
            logging.info("Loading activations for {} {}".format(mode, layer))
            data = pickle.load(open("{}/global_{}_{}.pkl".format(directory, mode, layer), "rb"))
            logging.info("Training for {} {}".format(mode, layer))
            act = [ torch.tensor([item[:, :]]).float().to(device) for item in data[layer] ]
            act, act_val = train_test_split(act, test_size=test_size, random_state=splitseed)
            if standardize:
                logging.info("Standardizing data")
                act, act_val = normalize(act, act_val)
            this = train_wa(edit_sim, edit_sim_val, act, act_val, attention=attention, attention_hidden_size=None, epochs=epochs, device=device)
            result.append({**this, 'model': mode, 'layer': layer})
            del act, act_val
            logging.info("Maximum correlation on val: {} at epoch {}".format(result[-1]['cor'], result[-1]['epoch']))
    return result


def collapse_runs(seq):
    """Collapse runs of the same symbol in a sequence to a single symbol."""
    current = None
    out = []
    for x in seq:
        if x != current:
            out.append(x)
            current = x
    return out



def train_wa(edit_sim, edit_sim_val, stack, stack_val, attention='scalar', attention_hidden_size=None, epochs=1, device='cpu'):
    if attention == 'scalar':
        wa = platalea.attention.ScalarAttention(stack[0].size(2), hidden_size).to(device)
    elif attention == 'linear':
        wa = platalea.attention.LinearAttention(stack[0].size(2)).to(device)
    elif attention == 'mean':
        wa = platalea.attention.MeanPool().to(device)
        avg_pool_val = torch.cat([ wa(item) for item in stack_val])
        avg_pool_sim_val = S.cosine_matrix(avg_pool_val, avg_pool_val)
        cor_val = S.pearson_r(S.triu(avg_pool_sim_val), S.triu(edit_sim_val))
        return {'epoch': None, 'cor': cor_val.item() }

    else:
        wa = platalea.attention.Attention(stack[0].size(2), attention_hidden_size).to(device)

    optim = torch.optim.Adam(wa.parameters())
    minloss = 0; minepoch = None
    logging.info("Optimizing for {} epochs".format(epochs))
    for epoch in range(1, 1+epochs):
        avg_pool = torch.cat([ wa(item) for item in stack])
        avg_pool_sim = S.cosine_matrix(avg_pool, avg_pool)
        loss = -S.pearson_r(S.triu(avg_pool_sim), S.triu(edit_sim))
        with torch.no_grad():
            avg_pool_val = torch.cat([ wa(item) for item in stack_val])
            avg_pool_sim_val = S.cosine_matrix(avg_pool_val, avg_pool_val)
            loss_val = -S.pearson_r(S.triu(avg_pool_sim_val), S.triu(edit_sim_val))
        logging.info("{} {} {}".format(epoch, -loss.item(), -loss_val.item()))
        if loss_val.item() <= minloss:
            minloss = loss_val.item()
            minepoch = epoch
        optim.zero_grad()
        loss.backward()
        optim.step()
        # Release CUDA-allocated tensors
        del loss, loss_val,  avg_pool, avg_pool_sim, avg_pool_val, avg_pool_sim_val
    del wa, optim
    return {'epoch': minepoch, 'cor': -minloss}

def weighted_average_RSA_partial(directory='.', layers=[], test_size=1/2,  standardize=False, epochs=1, device='cpu'):
    from sklearn.model_selection import train_test_split
    from platalea.dataset import Flickr8KData
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    splitseed = random.randint(0, 1024)
    result = []
    logging.info("Loading transcription data")
    data = pickle.load(open("{}/global_input.pkl".format(directory), "rb"))
    trans = data['ipa']
    act = [ torch.tensor([item[:, :]]).float().to(device) for item in data['audio'] ]
    val = Flickr8KData(root='/roaming/gchrupal/datasets/flickr8k/', split='val')
    image_map = { item['audio_id']: item['image'] for item in val }
    image = np.stack([ image_map[item] for item in data['audio_id'] ])

    trans, trans_val, act, act_val, image, image_val = train_test_split(trans, act, image, test_size=test_size, random_state=splitseed)
    if standardize:
        logging.info("Standardizing data")
        act, act_val = normalize(act, act_val)
    logging.info("Computing edit distances")
    edit_sim = torch.tensor(U.pairwise(S.stringsim, trans)).float().to(device)
    edit_sim_val = torch.tensor(U.pairwise(S.stringsim, trans_val)).float().to(device)
    logging.info("Computing image similarities")
    image = torch.tensor(image).float()
    image_val = torch.tensor(image_val).float()
    sim_image = S.cosine_matrix(image, image)
    sim_image_val = S.cosine_matrix(image_val, image_val)

    logging.info("Computing partial correlation for input features (mean pooling)")
    wa = platalea.attention.MeanPool().to(device)
    avg_pool = torch.cat([ wa(item) for item in act])
    avg_pool_sim = S.cosine_matrix(avg_pool, avg_pool)
    avg_pool_val = torch.cat([ wa(item) for item in act_val])
    avg_pool_sim_val = S.cosine_matrix(avg_pool_val, avg_pool_val)
    # Training data
    #  Edit ~ Act + Image
    Edit  = S.triu(edit_sim).cpu().numpy()
    Image = S.triu(sim_image).cpu().numpy()
    Act   = S.triu(avg_pool_sim).cpu().numpy()
    # Val data
    Edit_val  = S.triu(edit_sim_val).cpu().numpy()
    Image_val = S.triu(sim_image_val).cpu().numpy()
    Act_val   = S.triu(avg_pool_sim_val).cpu().numpy()
    e_full, e_base, e_mean = partial_r2(Edit, Act, Image, Edit_val, Act_val, Image_val)
    logging.info("Full, base, mean error: {} {}".format(e_full, e_base, e_mean))
    r2 =  (e_base - e_full)/e_base
    this =  {'epoch': None, 'error': e_full, 'baseline': e_base, 'error_mean': e_mean, 'r2': r2  }

    result.append({**this, 'model': 'random', 'layer': 'mfcc'})
    result.append({**this, 'model': 'trained', 'layer': 'mfcc'})
    del act, act_val
    logging.info("Partial r2 on val: {} at epoch {}".format(result[-1]['r2'], result[-1]['epoch']))
    for mode in ["trained", "random"]:
        for layer in layers:
            logging.info("Loading activations for {} {}".format(mode, layer))
            data = pickle.load(open("{}/global_{}_{}.pkl".format(directory, mode, layer), "rb"))
            logging.info("Training for {} {}".format(mode, layer))
            act = [ torch.tensor([item[:, :]]).float().to(device) for item in data[layer] ]
            act, act_val = train_test_split(act, test_size=test_size, random_state=splitseed)
            if standardize:
                logging.info("Standardizing data")
                act, act_val = normalize(act, act_val)
            avg_pool = torch.cat([ wa(item) for item in act])
            avg_pool_sim = S.cosine_matrix(avg_pool, avg_pool)
            avg_pool_val = torch.cat([ wa(item) for item in act_val])
            avg_pool_sim_val = S.cosine_matrix(avg_pool_val, avg_pool_val)
            Act   = S.triu(avg_pool_sim).cpu().numpy()
            Act_val   = S.triu(avg_pool_sim_val).cpu().numpy()
            e_full, e_base, e_mean = partial_r2(Edit, Act, Image, Edit_val, Act_val, Image_val)
            logging.info("Full, base, mean error: {} {}".format(e_full, e_base, e_mean))
            r2 =  (e_base - e_full)/e_base
            this =  {'epoch': None, 'error': e_full, 'baseline': e_base, 'error_mean': e_mean, 'r2': r2  }
            pickle.dump(dict(Edit=Edit, Act=Act, Image=Image, Edit_val=Edit_val, Act_val=Act_val, Image_val=Image_val), open("fufi_{}_{}.pkl".format(mode, layer), "wb"), protocol=4)
            result.append({**this, 'model': mode, 'layer': layer})
            del act, act_val
            logging.info("Partial R2 on val: {} at epoch {}".format(result[-1]['r2'], result[-1]['epoch']))
    return result

def partial_r2(Y, X, Z, y, x, z):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    from scipy.stats import pearsonr
    full = LinearRegression()
    full.fit(np.stack([X, Z], axis=1), Y)
    base = LinearRegression()
    base.fit(Z.reshape((-1, 1)), Y)
    y_full = full.predict(np.stack([x, z], axis=1))
    y_base = base.predict(z.reshape((-1, 1)))
    e_full = mean_squared_error(y, y_full)
    e_base = mean_squared_error(y, y_base)
    e_mean = mean_squared_error(y, np.repeat(Y.mean().item(), len(y)))
    r_full = pearsonr(y, y_full)[0]
    r_base = pearsonr(y, y_base)[0]
    logging.info("Pearson's r full : {}".format(r_full))
    logging.info("Pearson's r base : {}".format(r_base))
    logging.info("Pearson's partial: {}".format(rer(r_full, r_base)))
    return e_full.item(), e_base.item(), e_mean.item()

def normalize(X, X_val):
    device = X[0].device
    X = [x.cpu() for x in X ]
    X_val = [x.cpu() for x in X_val]
    d = X[0].shape[-1]
    flat = torch.cat([ x.view(-1, d) for x in X])
    mu = flat.mean(dim=0)
    sigma = flat.std(dim=0)
    X_norm = [ (item - mu) / sigma for item in X]
    X_val_norm = [ (item - mu) /sigma for item in X_val ]
    return [x.to(device) for x in X_norm], [x.to(device) for x in X_val_norm ]


def weighted_average_diagnostic(directory='.', layers=[], attention='scalar', test_size=1/2, attention_hidden_size=None, hidden_size=None, standardize=False, epochs=1, factor=0.1, device='cpu'):
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    splitseed = random.randint(0, 1024)
    result = []
    logging.info("Loading transcription data")
    data = pickle.load(open("{}/global_input.pkl".format(directory), "rb"))
    trans = data['ipa']
    act = [ torch.tensor(item[:, :]).float().to(device) for item in data['audio'] ]

    trans, trans_val, X, X_val = train_test_split(trans, act, test_size=test_size, random_state=splitseed)
    if standardize:
        logging.info("Standardizing data")
        X, X_val = normalize(X, X_val)
    logging.info("Computing targets")
    vec = CountVectorizer(lowercase=False, analyzer='char')
    # Binary instead of counts
    y = torch.tensor(vec.fit_transform(trans).toarray()).float().clamp(min=0, max=1)
    y_val = torch.tensor(vec.transform(trans_val).toarray()).float().clamp(min=0, max=1)
    logging.info("Training for input features")
    model = PooledClassifier(input_size=X[0].shape[1],  output_size=y[0].shape[0],
                             hidden_size=hidden_size, attention_hidden_size=attention_hidden_size, attention=attention).to(device)
    this = train_classifier(model, X, y, X_val, y_val, epochs=epochs, factor=factor)
    result.append({**this, 'model': 'random', 'layer': 'mfcc'})
    result.append({**this, 'model': 'trained', 'layer': 'mfcc'})
    del X, X_val
    logging.info("Maximum accuracy on val: {} at epoch {}".format(result[-1]['acc'], result[-1]['epoch']))
    for mode in ["trained", "random"]:
        for layer in layers:
            logging.info("Loading activations for {} {}".format(mode, layer))
            data = pickle.load(open("{}/global_{}_{}.pkl".format(directory, mode, layer), "rb"))
            logging.info("Training for {} {}".format(mode, layer))
            act = [ torch.tensor(item[:, :]).float() for item in data[layer] ]
            X, X_val = train_test_split(act, test_size=test_size, random_state=splitseed)
            if standardize:
                logging.info("Standardizing data")
                X, X_val = normalize(X, X_val)
            model = PooledClassifier(input_size=X[0].shape[1], output_size=y[0].shape[0],
                                     hidden_size=hidden_size, attention_hidden_size=attention_hidden_size, attention=attention).to(device)
            this = train_classifier(model, X, y, X_val, y_val, epochs=epochs, factor=factor)
            result.append({**this, 'model': mode, 'layer': layer})
            del X, X_val
            logging.info("Maximum accuracy on val: {} at epoch {}".format(result[-1]['acc'], result[-1]['epoch']))
    return result

class PooledClassifier(nn.Module):

    def __init__(self, input_size, output_size, attention_hidden_size=1024, hidden_size=None, attention='scalar', weight_decay=0.0):
        super(PooledClassifier, self).__init__()
        if attention == 'scalar':
            self.wa = platalea.attention.ScalarAttention(input_size, attention_hidden_size)
        elif attention == 'linear':
            self.wa = platalea.attention.LinearAttention(input_size)
        elif attention == 'mean':
            self.wa = platalea.attention.MeanPool()
        else:
            self.wa = platalea.attention.Attention(input_size, attention_hidden_size)
        if hidden_size is None:
            self.project = nn.Linear(in_features=input_size, out_features=output_size)
        else:
            self.project = MLP(input_size, output_size, hidden_size=hidden_size)
        self.loss = nn.functional.binary_cross_entropy_with_logits
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.project(self.wa(x))

    def predict(self, x):
        logit = self.project(self.wa(x))
        return (logit >= 0.0).float()


class SoftmaxClassifier(nn.Module):

    def __init__(self, input_size, output_size, weight_decay=0.0):
        super(SoftmaxClassifier, self).__init__()
        self.project = nn.Linear(in_features=input_size, out_features=output_size)
        self.loss = nn.functional.cross_entropy
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.project(x)

    def predict(self, x):
        return self.project(x).argmax(dim=1)

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=500, weight_decay=0.0):
        super(MLP, self).__init__()
        self.i2h = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.Dropout = nn.Dropout(p=0.5)
        self.h2o = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.loss = nn.functional.cross_entropy
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.h2o(torch.relu(self.Dropout(self.i2h(x))))

    def predict(self, x):
        return self.forward(x).argmax(dim=1)


def collate(items):
    x, y = zip(*items)
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0)
    y = torch.stack(y)
    return x, y

def tuple_stack(xy):
    x, y = zip(*xy)
    return torch.stack(x), torch.stack(y)

def rer(hi, lo):
    return ((1-lo) - (1-hi))/(1-lo)

def train_classifier(model, X, y, X_val, y_val, epochs=1, patience=50, factor=0.1, majority=majority_binary):
    device = list(model.parameters())[0].device
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=model.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', factor=factor, patience=10)
    data = torch.utils.data.DataLoader(list(zip(X, y)), batch_size=64, shuffle=True, collate_fn=collate)
    data_val = torch.utils.data.DataLoader(list(zip(X_val, y_val)), batch_size=64, shuffle=False, collate_fn=collate)
    logging.info("Optimizing for {} epochs".format(epochs))
    scores = []
    with torch.no_grad():
        model.eval()
        maj = majority(y)
        baseline = np.mean([ (maj == y_i).cpu().numpy() for y_i in y_val ])
        logging.info("Baseline accuracy: {}".format(baseline))
    for epoch in range(1, 1+epochs):
        model.train()
        epoch_loss = []
        for x, y in data:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = model.loss(y_pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss.append(loss.item())
        with torch.no_grad():
            model.eval()
            loss_val = np.mean( [model.loss(model(x.to(device)), y.to(device)).item() for x,y in data_val])
            accuracy_val = np.concatenate([ model.predict(x.to(device)).cpu().numpy() == y.cpu().numpy() for x, y in data_val]).mean()
            #scheduler.step(loss_val)
            scheduler.step(1-accuracy_val)
            logging.info("{} {} {} {} {}".format(epoch, optim.state_dict()['param_groups'][0]['lr'], np.mean(epoch_loss), loss_val, accuracy_val))
        scores.append(dict(epoch=epoch, train_loss=np.mean(epoch_loss), acc=accuracy_val, loss=loss_val, baseline=baseline))
        minepoch = max(scores, key=lambda a: a['acc'])['epoch']
        if epoch - minepoch >= patience:
            logging.info("No improvement for {} epochs, stopping.".format(patience))
            break
        # Release CUDA-allocated tensors
        del x, y, loss, loss_val,  y_pred
    del model, optim
    return max(scores, key=lambda a: a['acc'])

# PLOTTING
import pandas as pd
from plotnine import *
from plotnine.options import figure_size



def plot(path, output):
    ld = pd.read_json("{}/local/local_diagnostic.json".format(path), orient="records");   ld['scope'] = 'local';      ld['method'] = 'diagnostic'
    lr = pd.read_json("{}/local/local_rsa.json".format(path), orient="records");          lr['scope'] = 'local';      lr['method'] = 'rsa'
    gd = pd.read_json("{}/mean/global_diagnostic.json".format(path), orient="records");   gd['scope'] = 'mean pool';  gd['method'] = 'diagnostic'
    gr = pd.read_json("{}/mean/global_rsa.json".format(path), orient="records");          gr['scope'] = 'mean pool';  gr['method'] = 'rsa'
    ad = pd.read_json("{}/attn/global_diagnostic.json".format(path), orient="records");   ad['scope'] = 'attn pool';  ad['method'] = 'diagnostic'
    ar = pd.read_json("{}/attn/global_rsa.json".format(path), orient="records");          ar['scope'] = 'attn pool';  ar['method'] = 'rsa'
    data = pd.concat([ld, lr, gd, gr, ad, ar], sort=False)

    data['rer'] = rer(data['acc'], data['baseline'])
    data['score'] = data['rer'].fillna(0.0) + data['cor'].fillna(0.0)

    order = list(data['layer'].unique())
    data['layer id'] = [ order.index(x) for x in data['layer'] ]
    # Reorder scope
    data['scope'] = pd.Categorical(data['scope'], categories=['local', 'mean pool', 'attn pool'])
    # Reorder model
    data['model'] = pd.Categorical(data['model'], categories=['trained', 'random'])
    # Make variable to group model x run interaction for plotting multiple runs.
    data['modelxrun'] = data.apply(lambda x: "{} {}".format(x['model'], x['run']), axis=1)

    g = ggplot(data, aes(x='layer id', y='score', color='model', linetype='model', shape='model')) + geom_point() +  geom_line(aes(group='modelxrun')) + \
                            facet_wrap('~ method + scope') + \
                            theme(figure_size=(figure_size[0]*1.5, figure_size[1]*1.5))
    ggsave(g, '{}/plot.png'.format(output))

def rer(hi, lo):
    return ((1-lo) - (1-hi))/(1-lo)

def plot_r2_partial():
    path = 'data/out/rnn-vgs/mean/'
    data = pd.read_json("{}/global_rsa_partial.json".format(path), orient="records")
    order = list(data['layer'].unique())
    data['layer id'] = [ order.index(x) for x in data['layer'] ]
    data['partial R²'] = (data['baseline'] - data['error']) / data['baseline']
    data['cor'] = abs(data['partial R²'])**0.5
    # Reorder model
    data['model'] = pd.Categorical(data['model'], categories=['trained', 'random'])
    g = ggplot(data, aes(x='layer id', y='cor', color='model', linetype='model', shape='model')) + geom_point() + geom_line() +\
                            ylab("√R² (partial)")
    ggsave(g, 'fig/rnn-vgs/r2_partial.png')

def partialing(path='.'):
    data = pd.read_json("{}/global_rsa_partial.json".format(path), orient="records")
    order = list(data['layer'].unique())
    data['layer id'] = [ order.index(x) for x in data['layer'] ]
    data['partial R²'] = (data['baseline'] - data['error']) / data['baseline']
    # Reorder model
    data['model'] = pd.Categorical(data['model'], categories=['trained', 'random'])
    #
    pass

    ggsave(g, 'partialing.png')


def inject(x, e):
    return [ {**xi, **e} for xi in x ]

