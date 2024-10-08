



import argparse
import copy
import getpass
import hashlib
import json
import os
import random
import shutil
import time
import uuid

import numpy as np
import torch

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed import command_launchers

import tqdm
import shlex

class Job:
    NOT_LAUNCHED = 'Not launched'
    INCOMPLETE = 'Incomplete'
    DONE = 'Done'

    def __init__(self, train_args, sweep_output_dir):
        args_str = json.dumps(train_args, sort_keys=True)
        args_hash = hashlib.md5(args_str.encode('utf-8')).hexdigest()
        self.output_dir = os.path.join(sweep_output_dir, args_hash)

        self.train_args = copy.deepcopy(train_args)
        self.train_args['output_dir'] = self.output_dir
        command = ['OMP_NUM_THREADS=1', 'python', '-m', 'domainbed.scripts.train']
        for k, v in sorted(self.train_args.items()):
            if isinstance(v, list):
                v = ' '.join([str(v_) for v_ in v])
            elif isinstance(v, str):
                v = shlex.quote(v)
            command.append(f'--{k} {v}')
        self.command_str = ' '.join(command)

        if os.path.exists(os.path.join(self.output_dir, 'done')):
            self.state = Job.DONE
        elif os.path.exists(self.output_dir):
            self.state = Job.INCOMPLETE
        else:
            self.state = Job.NOT_LAUNCHED

    def __str__(self):
        job_info = (self.train_args['dataset'],
            self.train_args['algorithm'],
            self.train_args['test_envs'],
            self.train_args['hparams_seed'])
        return '{}: {} {}'.format(
            self.state,
            self.output_dir,
            job_info)

    @staticmethod
    def launch(jobs, launcher_fn):
        print('Launching...')
        jobs = jobs.copy()
        np.random.shuffle(jobs)
        print('Making job directories:')
        for job in tqdm.tqdm(jobs, leave=False):
            os.makedirs(job.output_dir, exist_ok=True)
        commands = [job.command_str for job in jobs]
        launcher_fn(commands)
        print(f'Launched {len(jobs)} jobs!')

    @staticmethod
    def delete(jobs):
        print('Deleting...')
        for job in jobs:
            shutil.rmtree(job.output_dir)
        print(f'Deleted {len(jobs)} jobs!')


class SAJob:
    NOT_LAUNCHED = 'Not launched'
    INCOMPLETE = 'Incomplete'
    PRETRAINED = 'Pretrained'
    DONE = 'Done'

    def __init__(self, train_args, sweep_output_dir, ft_mode):
        args_str = json.dumps(train_args, sort_keys=True)
        args_hash = hashlib.md5(args_str.encode('utf-8')).hexdigest()
        self.output_dir = os.path.join(sweep_output_dir, args_hash)
        self.ft_mode = ft_mode
        self.train_args = copy.deepcopy(train_args)
        self.train_args['output_dir'] = self.output_dir
        command = [
            'python', '-m', 'domainbed.scripts.supervised_adaptation',
            '--input_dir', self.train_args['output_dir'], 
            '--ft_mode', ft_mode
        ]
        self.command_str = ' '.join(command)

        if os.path.exists(os.path.join(self.output_dir, 'done')):
            if os.path.exists(os.path.join(self.output_dir, 'done_{}'.format(ft_mode))):
                self.state = SAJob.DONE
            else:
                self.state = SAJob.PRETRAINED
        elif os.path.exists(os.path.join(self.output_dir, 'results_{}.jsonl'.format(ft_mode))):
            self.state = SAJob.INCOMPLETE
        else:
            self.state = SAJob.NOT_LAUNCHED

    def __str__(self):
        job_info = (self.train_args['dataset'],
            self.train_args['algorithm'],
            self.train_args['test_envs'],
            self.train_args['hparams_seed'], self.ft_mode)
        return '{}: {} {}'.format(
            self.state,
            self.output_dir,
            job_info)

    @staticmethod
    def launch(jobs, launcher_fn):
        print('Launching...')
        jobs = jobs.copy()
        np.random.shuffle(jobs)
        print('Making job directories:')
        for job in tqdm.tqdm(jobs, leave=False):
            os.makedirs(job.output_dir, exist_ok=True)
        commands = [job.command_str for job in jobs]
        launcher_fn(commands)
        print(f'Launched {len(jobs)} jobs!')
        print('Launching...')
        jobs = jobs.copy()
        np.random.shuffle(jobs)
        print('Making job directories:')
        for job in tqdm.tqdm(jobs, leave=False):
            os.makedirs(job.output_dir, exist_ok=True)
        commands = [job.command_str for job in jobs]
        launcher_fn(commands)
        print(f'Launched {len(jobs)} jobs!')


class UAJob:
    NOT_LAUNCHED = 'Not launched'
    INCOMPLETE = 'Incomplete'
    PRETRAINED = 'Pretrained'
    DONE = 'Done'

    def __init__(self, train_args, sweep_output_dir, adapt_algorithm, test_valid=0):
        valid = "test" if test_valid else "train"
        args_str = json.dumps(train_args, sort_keys=True)
        args_hash = hashlib.md5(args_str.encode('utf-8')).hexdigest()
        self.output_dir = os.path.join(sweep_output_dir, args_hash)
        self.adapt_algorithm = adapt_algorithm
        self.train_args = copy.deepcopy(train_args)
        self.train_args['output_dir'] = self.output_dir
        command = [
            'python', '-m', 'domainbed.scripts.unsupervised_adaptation',
            '--input_dir', self.train_args['output_dir'], 
            '--adapt_algorithm', adapt_algorithm,
            '--test_valid', str(test_valid),
        ]
        self.command_str = ' '.join(command)

        if os.path.exists(os.path.join(self.output_dir, 'done')):
            if os.path.exists(os.path.join(self.output_dir, 'done_{}_{}'.format(adapt_algorithm, valid))):
                self.state = UAJob.DONE
            else:
                self.state = UAJob.PRETRAINED
        elif os.path.exists(os.path.join(self.output_dir, 'results_{}_{}.jsonl'.format(adapt_algorithm, valid))):
            self.state = UAJob.INCOMPLETE
        else:
            self.state = UAJob.NOT_LAUNCHED

    def __str__(self):
        job_info = (self.train_args['dataset'],
            self.train_args['algorithm'],
            self.train_args['test_envs'],
            self.train_args['hparams_seed'], self.adapt_algorithm)
        return '{}: {} {}'.format(
            self.state,
            self.output_dir,
            job_info)

    @staticmethod
    def launch(jobs, launcher_fn):
        print('Launching...')
        jobs = jobs.copy()
        np.random.shuffle(jobs)
        print('Making job directories:')
        for job in tqdm.tqdm(jobs, leave=False):
            os.makedirs(job.output_dir, exist_ok=True)
        commands = [job.command_str for job in jobs]
        launcher_fn(commands)
        print(f'Launched {len(jobs)} jobs!')
        print('Launching...')
        jobs = jobs.copy()
        np.random.shuffle(jobs)
        print('Making job directories:')
        for job in tqdm.tqdm(jobs, leave=False):
            os.makedirs(job.output_dir, exist_ok=True)
        commands = [job.command_str for job in jobs]
        launcher_fn(commands)
        print(f'Launched {len(jobs)} jobs!')


def all_test_env_combinations(n):
    
    assert(n >= 3)
    for i in range(n):
        yield [i]
        for j in range(i+1, n):
            yield [i, j]

def make_args_list(n_trials_from, n_trials, dataset_names, algorithms, n_hparams_from, n_hparams, steps,
    data_dir, task, holdout_fraction, single_test_envs, hparams):
    args_list = []
    for trial_seed in range(n_trials_from, n_trials_from+n_trials):
        for dataset in dataset_names:
            for algorithm in algorithms:
                if single_test_envs:
                    all_test_envs = [
                        [i] for i in range(datasets.num_environments(dataset))]
                else:
                    all_test_envs = all_test_env_combinations(
                        datasets.num_environments(dataset))
                for test_envs in all_test_envs:
                    for hparams_seed in range(n_hparams_from, n_hparams):
                        train_args = {}
                        train_args['dataset'] = dataset
                        train_args['algorithm'] = algorithm
                        train_args['test_envs'] = test_envs
                        train_args['holdout_fraction'] = holdout_fraction
                        train_args['hparams_seed'] = hparams_seed
                        train_args['data_dir'] = data_dir
                        train_args['task'] = task
                        train_args['trial_seed'] = trial_seed
                        train_args['seed'] = misc.seed_hash(dataset,
                            algorithm, test_envs, hparams_seed, trial_seed)
                        if steps is not None:
                            train_args['steps'] = steps
                        if hparams is not None:
                            train_args['hparams'] = hparams
                        args_list.append(train_args)
    return args_list

def ask_for_confirmation():
    response = input('Are you sure? (y/n) ')
    if not response.lower().strip()[:1] == "y":
        print('Nevermind!')
        exit(0)

DATASETS = [d for d in datasets.DATASETS if "Debug" not in d]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a sweep')
    parser.add_argument('command', choices=[
        'launch', 'delete_incomplete', 'supervised_adaptation', 
        'unsupervised_adaptation', 'unsup_adapt'])
    parser.add_argument('--datasets', nargs='+', type=str, default=DATASETS)
    parser.add_argument('--algorithms', nargs='+', type=str)
    parser.add_argument('--task', type=str, default="domain_generalization")
    parser.add_argument('--n_hparams_from', type=int, default=0)
    parser.add_argument('--n_hparams', type=int, default=20)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_trials_from', type=int, default=3)
    parser.add_argument('--n_trials', type=int, default=3)
    parser.add_argument('--command_launcher', type=str, required=True)
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--test_valid', type=int, default=1)
    parser.add_argument('--hparams', type=str, default=None)
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--single_test_envs', action='store_true')
    parser.add_argument('--skip_confirmation', action='store_true')
    args = parser.parse_args()

    args_list = make_args_list(
        n_trials_from=args.n_trials_from, 
        n_trials=args.n_trials,
        dataset_names=args.datasets,
        algorithms=args.algorithms,
        n_hparams_from=args.n_hparams_from,
        n_hparams=args.n_hparams,
        steps=args.steps,
        data_dir=args.data_dir,
        task=args.task,
        holdout_fraction=args.holdout_fraction,
        single_test_envs=args.single_test_envs,
        hparams=args.hparams
    )

    if args.command in ['launch', 'delete_incomplete']:
        jobs = [Job(train_args, args.output_dir) for train_args in args_list]

        for job in jobs:
            print(job)
        print("{} jobs: {} done, {} incomplete, {} not launched.".format(
            len(jobs),
            len([j for j in jobs if j.state == Job.DONE]),
            len([j for j in jobs if j.state == Job.INCOMPLETE]),
            len([j for j in jobs if j.state == Job.NOT_LAUNCHED]))
        )

        if args.command == 'launch':
            to_launch = [j for j in jobs if j.state == Job.NOT_LAUNCHED]
            print(f'About to launch {len(to_launch)} jobs.')
            if not args.skip_confirmation:
                ask_for_confirmation()
            launcher_fn = command_launchers.REGISTRY[args.command_launcher]
            Job.launch(to_launch, launcher_fn)

        elif args.command == 'delete_incomplete':
            to_delete = [j for j in jobs if j.state == Job.INCOMPLETE]
            print(f'About to delete {len(to_delete)} jobs.')
            if not args.skip_confirmation:
                ask_for_confirmation()
            Job.delete(to_delete)

    elif args.command == 'supervised_adaptation':
        jobs = [SAJob(train_args, args.output_dir, ft_mode='clf') for train_args in args_list]
        jobs += [SAJob(train_args, args.output_dir, ft_mode='token') for train_args in args_list]
        jobs += [SAJob(train_args, args.output_dir, ft_mode='transformer') for train_args in args_list]
        jobs += [SAJob(train_args, args.output_dir, ft_mode='all') for train_args in args_list]

        for job in jobs:
            print(job)
        print("{} jobs: {} done, {} pretrained, {} incomplete, {} not launched.".format(
            len(jobs),
            len([j for j in jobs if j.state == SAJob.DONE]),
            len([j for j in jobs if j.state == SAJob.PRETRAINED]),
            len([j for j in jobs if j.state == SAJob.INCOMPLETE]),
            len([j for j in jobs if j.state == SAJob.NOT_LAUNCHED]))
        )

        to_launch = [j for j in jobs if j.state == SAJob.PRETRAINED]
        print(f'About to launch {len(to_launch)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation() 
        launcher_fn = command_launchers.REGISTRY[args.command_launcher]
        Job.launch(to_launch, launcher_fn)

    elif args.command in ['unsupervised_adaptation', 'unsup_adapt']:
        methods = [
            'AdaNPC', "AdaNPCBN"
            ]
        jobs = []
        for method in methods:
            jobs += [UAJob(
                train_args, args.output_dir,
                adapt_algorithm=method, test_valid=args.test_valid) for train_args in args_list]
            
            
            
            
            
            
            
            

        for job in jobs:
            print(job)
        print("{} jobs: {} done, {} pretrained, {} incomplete, {} not launched.".format(
            len(jobs),
            len([j for j in jobs if j.state == UAJob.DONE]),
            len([j for j in jobs if j.state == UAJob.PRETRAINED]),
            len([j for j in jobs if j.state == UAJob.INCOMPLETE]),
            len([j for j in jobs if j.state == UAJob.NOT_LAUNCHED]))
        )

        to_launch = [j for j in jobs if j.state == UAJob.PRETRAINED]
        print(f'About to launch {len(to_launch)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation() 
        launcher_fn = command_launchers.REGISTRY[args.command_launcher]
        Job.launch(to_launch, launcher_fn)
