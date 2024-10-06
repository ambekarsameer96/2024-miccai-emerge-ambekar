

import argparse
from argparse import Namespace
import collections
import json
import os
import random
import sys
import time
import uuid
from itertools import chain
import itertools
import copy

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader, DataParallelPassthrough
from domainbed import model_selection
from domainbed.lib.query import Q
from domainbed import adapt_algorithms
import itertools
import domainbed.memo_aug as memo_aug



class Dataset:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
































def data_augment_function(data):
    
    augmented_data1 = []
    augmented_data2 = []
    augmented_data3 = []
    for i in range(len(data)):
        
        

        aug_img1 = memo_aug.augment_m(data[i])
        
        
    
    
    
        augmented_data1.append(aug_img1)
    
    
    
    augmented_data1 = torch.stack(augmented_data1)
    
    
    
    


    return augmented_data1

def generate_featurelized_loader(loader, network, classifier, batch_size=32, adapt=True):
    
    z_list = []
    y_list = []
    p_list = []
    z_aug =  []
    z_aug2 = []
    z_aug3 = []
    
    network.eval()
    classifier.eval()
    for x, y in loader:
        x = x.to(device)
        z = network(x)
        p = classifier(z)
        if adapt:
            new_aug = data_augment_function(x)
            
            
            new_aug = new_aug.to(device)
            z1 = network(new_aug)
            z1 = z1.to(device)
            z2 = network(new_aug)
            z2 = z2.to(device)
            z3 = network(new_aug)
            z3 = z3.to(device)
            
            
            
            
            
            
            
        
        z_list.append(z.detach().cpu())
        z_aug.append(z1.detach().cpu())
        z_aug2.append(z2.detach().cpu())
        z_aug3.append(z3.detach().cpu())
        
        
        y_list.append(y.detach().cpu())
        p_list.append(p.detach().cpu())
        
    network.train()
    classifier.train()
    z = torch.cat(z_list)
    z1 = torch.cat(z_aug)
    z2 = torch.cat(z_aug2)
    z3 = torch.cat(z_aug3)
    
    y = torch.cat(y_list)
    p = torch.cat(p_list)
    ent = softmax_entropy(p)
    py = p.argmax(1).float().cpu().detach()
    dataset1, dataset2, dataset3, dataset4, dataset5  = Dataset(z, y), Dataset(z, py), Dataset(z1, py), Dataset(z2, py), Dataset(z3, py)
    loader1 = torch.utils.data.DataLoader(dataset1, batch_size=batch_size, shuffle=False, drop_last=True)
    loader2 = torch.utils.data.DataLoader(dataset2, batch_size=batch_size, shuffle=False, drop_last=True)
    loader3 = torch.utils.data.DataLoader(dataset3, batch_size=batch_size, shuffle=False, drop_last=True)
    loader4 = torch.utils.data.DataLoader(dataset4, batch_size=batch_size, shuffle=False, drop_last=True)
    loader5 = torch.utils.data.DataLoader(dataset5, batch_size=batch_size, shuffle=False, drop_last=True)
    
    
    return loader1, loader2, loader3, loader4, loader5, ent


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def accuracy_ent(network, loader, weights, device, adapt=False):
    correct = 0
    total = 0
    weights_offset = 0
    ent = 0

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            if adapt is None:
                p = network(x)
            else:
                
                p = network(x, adapt)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset: weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
            ent += softmax_entropy(p).sum().item()
    network.train()

    return correct / total, ent / total




def accuracy_ent_aug(network, loader, loader_aug, loader_aug2, loader_aug3, weights, device, adapt=True):
    correct = 0
    total = 0
    weights_offset = 0
    ent = 0

    network.eval()
    with torch.no_grad():
        
        
        for (x, y), (x_aug, y_aug), (x_aug2, y_aug2), (x_aug3, y_aug3) in zip(loader, loader_aug, loader_aug2, loader_aug3):
            x = x.to(device)
            y = y.to(device)
            del y_aug, y_aug2, y_aug3
            x_aug = x_aug.to(device)
            x_aug2 = x_aug2.to(device)
            x_aug3 = x_aug3.to(device)
            
            if adapt is None:
                p = network(x)
            else:
                
                p = network(x, x_aug, x_aug2, x_aug3, adapt)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset: weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
            ent += softmax_entropy(p).sum().item()
    network.train()

    return correct / total, ent / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    
    
    
    parser = argparse.ArgumentParser(description='Domain generalization')
    
    parser.add_argument('--input_dir', type=str, default = './Fetal_ds_train_output_adanpc_erm')

    parser.add_argument('--adapt_algorithm', type=str, default="TAST")
    parser.add_argument('--test_valid', type=str, default='0')
    
    parser.add_argument('--hparams_seed', type=int, default=0,)
    
    

    parser.add_argument('--dataset', type=str, default="Messidor")
    parser.add_argument('--data_dir', type=str, default='data/FETAL_PLANES_ZENODO/data_files')
    
    
    parser. add_argument('--output_dir', type=str)
    args_in = parser.parse_args()

    epochs_path = os.path.join(args_in.input_dir, 'results.jsonl')
    records = []
    with open(epochs_path, 'r') as f:
        for line in f:
            records.append(json.loads(line[:-1]))
    records = Q(records)
    r = records[0]
    args = Namespace(**r['args'])
    args.input_dir = args_in.input_dir

    if '-' in args_in.adapt_algorithm:
        args.adapt_algorithm, test_batch_size = args_in.adapt_algorithm.split('-')
        args.test_batch_size = int(test_batch_size)
    else:
        args.adapt_algorithm = args_in.adapt_algorithm
        args.test_batch_size = 32  

    args.output_dir = args.input_dir
    
    alg_name = args_in.adapt_algorithm

    if args.adapt_algorithm in['T3A', 'TentPreBN', 'TentClf', 'PLClf', 'TAST']:
        use_featurer_cache = True
    else:
        use_featurer_cache = False
    if os.path.exists(os.path.join(args.output_dir, 'done_{}'.format(alg_name))):
        print("{} has already excecuted".format(alg_name))

    
    
    algorithm_dict = None
    
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out_{}.txt'.format(alg_name)))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err_{}.txt'.format(alg_name)))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))
        
    
    args.dataset='Fetal_8'
    args.data_dir ="data/FETAL_PLANES_ZENODO/data_files"

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    assert os.path.exists(os.path.join(args.output_dir, 'done'))
    assert os.path.exists(os.path.join(args.output_dir, 'IID_best.pkl'))  

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError

    
    
    
    
    
    
    
    
    
    
    
    
    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        uda = []
        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_)*args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    
    train_loaders = [FastDataLoader(
        dataset=env,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(out_splits)
        if i in args.test_envs]
    
    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)
        if i in args.test_envs]

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=args.test_batch_size,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
        for i in range(len(uda_splits))]

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)
    if hasattr(algorithm, 'network'):
        algorithm.network = DataParallelPassthrough(algorithm.network)
    else:
        for m in algorithm.children():
            m = DataParallelPassthrough(m)

    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    
    ckpt = torch.load(os.path.join(args.output_dir, 'IID_best.pkl'))
    algorithm_dict = ckpt['model_dict']
    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    
    print("Base model's results")
    results = {}
    evals = zip(eval_loader_names, eval_loaders, eval_weights)
    for name, loader, weights in evals:
        acc, ent = accuracy_ent(algorithm, loader, weights, device, adapt=None)
        results[name+'_acc'] = acc
        results[name+'_ent'] = ent
    results_keys = sorted(results.keys())
    misc.print_row(results_keys, colwidth=12)
    misc.print_row([results[key] for key in results_keys], colwidth=12)

    print("\nAfter {}".format(alg_name))
    
    if use_featurer_cache:
        original_evals = zip(eval_loader_names, eval_loaders, eval_weights)
        loaders = []
        for name, loader, weights in original_evals:
            
            loader1, loader2, loader3, loader4, loader5, ent = generate_featurelized_loader(loader, network=algorithm.featurizer, classifier=algorithm.classifier, batch_size=32, adapt=True)
            loaders.append((name, loader1, loader3, loader4, loader5, weights))
    else:
        loaders = zip(eval_loader_names, eval_loaders, eval_weights)
    
    evals = []
    for name, loader,  loader2, loader3, loader4, weights in loaders:
        if name in ['env{}_in'.format(i) for i in args.test_envs]:
            train_loader = (name, loader,  loader2, loader3, loader4,  weights)
        else:
            evals.append((name, loader, loader2, loader3, loader4,  weights))

    last_results_keys = None
    adapt_algorithm_class = adapt_algorithms.get_algorithm_class(
        args.adapt_algorithm)
    
    if args.adapt_algorithm in ['T3A']:
        adapt_hparams_dict = {
            'filter_K': [1, 5, 20, 50, 100, -1], 
        }
    elif args.adapt_algorithm in ['TentFull', 'TentPreBN', 'TentClf', 'TentNorm']:
        adapt_hparams_dict = {
            'alpha': [0.1, 1.0, 10.0],
            'gamma': [1, 3]
        }
    elif args.adapt_algorithm in ['PseudoLabel', 'PLClf']:
        adapt_hparams_dict = {
            'alpha': [0.1, 1.0, 10.0],
            'gamma': [1, 3], 
            'beta': [0.9]
        }
    elif args.adapt_algorithm in ['SHOT', 'SHOTIM']:
        adapt_hparams_dict = {
            'alpha': [0.1, 1.0, 10.0],
            'gamma': [1, 3], 
            'beta': [0.9], 
            'theta': [0.1], 
        }
    elif args.adapt_algorithm in ['TAST_BN']:
        adapt_hparams_dict = {
            'filter_K': [1, 5, 20],
            'gamma': [1, 3],
            'lr': [1e-3],
            'tau': [10],
            'k': [1, 2, 4, 8],
        }
    elif args.adapt_algorithm in ['TAST']:
        adapt_hparams_dict = {
            'num_ensemble': [1, 5, 10, 20],
            'filter_K': [1, 5, 20, 50, 100, -1],
            'gamma': [1, 3],
            'lr': [1e-3],
            'tau': [10],
            'k': [1, 2, 4, 8],
            'init_mode': ['kaiming_normal']
        }
    else:
        raise Exception("Not Implemented Error")
    product = [x for x in itertools.product(*adapt_hparams_dict.values())]
    adapt_hparams_list = [dict(zip(adapt_hparams_dict.keys(), r)) for r in product]

    for adapt_hparams in adapt_hparams_list:
        adapt_hparams['cached_loader'] = use_featurer_cache
        adapted_algorithm = adapt_algorithm_class(dataset.input_shape, dataset.num_classes,
            len(dataset) - len(args.test_envs), adapt_hparams, algorithm
        )
        
        adapted_algorithm.to(device)
        
        results = adapt_hparams

        for key, val in checkpoint_vals.items():
            results[key] = np.mean(val)

        
        for name, loader, loader2, loader3, loader4, weights in evals:
            
            acc, ent = accuracy_ent_aug(adapted_algorithm, loader, loader2, loader3, loader4, weights, device, adapt=True)
            results[name+'_acc'] = acc
            results[name+'_ent'] = ent
            adapted_algorithm.reset()

        name, loader, loader_aug,  loader_aug2, loader_aug3, weights = train_loader
        acc, ent = accuracy_ent_aug(adapted_algorithm, loader, loader2, loader3, loader4,weights, device, adapt=True)
        results[name+'_acc'] = acc
        results[name+'_ent'] = ent

        del adapt_hparams['cached_loader']
        results_keys = sorted(results.keys())

        if results_keys != last_results_keys:
            misc.print_row(results_keys, colwidth=12)
            last_results_keys = results_keys
        misc.print_row([results[key] for key in results_keys],
            colwidth=12)

        results.update({
            'hparams': hparams,
            'args': vars(args)    
        })
        
        epochs_path = os.path.join(args.output_dir, 'results_{}.jsonl'.format(alg_name))
        with open(epochs_path, 'a') as f:
            f.write(json.dumps(results, sort_keys=True) + "\n")

    
    with open(os.path.join(args.output_dir, 'done_{}'.format(alg_name)), 'w') as f:
        f.write('done')

        