import os
import time
import json
import yaml
import shutil
import random
import datetime
from pathlib import Path

import numpy as np

from typing import Dict, List, Tuple, Union, Optional, Literal

import torch
import optuna

from models import MLP, FTTransformer, AutoInt, DCNv2, NODE
from models.abstract import TabModel, check_dir
from data.utils import Dataset
from data.processor import DataProcessor

MODEL_CARDS = {
    'xgboost': None, 'catboost': None, 'lightgbm': None,
    'mlp': MLP, 'autoint': AutoInt, 'dcnv2': DCNv2, 'node': NODE,
    'ft-transformer': FTTransformer, 'saint': None, 
    't2g-former': None, 'excel-former': None,
}
HPOLib = Literal['optuna', 'hyperopt'] # TODO: add 'hyperopt' support

def get_model_cards():
    return {
        'ready': sorted(list([key for key, value in MODEL_CARDS.items() if value])),
        'comming soon': sorted(list([key for key, value in MODEL_CARDS.items() if not value]))
    }

def seed_everything(seed=42):
    '''
    Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.
    '''
    random.seed(seed)
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_config_from_file(file):
    file = str(file)
    if file.endswith('.yaml'):
        with open(file, 'r') as f:
            cfg = yaml.safe_load(f)
    elif file.endswith('.json'):
        with open(file, 'r') as f:
            cfg = json.load(f)
    else:
        raise AssertionError('Config files only support yaml or json format now.')
    return cfg

def extract_config(model_config: dict, is_large_data: bool = False):
    """selection of different search spaces"""
    used_cfgs = {"model": {}, "training": {}, 'meta': model_config.get('meta', {})}
    for field in ['model', 'training']:
        for k in model_config[field]:
            cfgs = model_config[field][k]
            if 'type2' not in cfgs:
                used_cfg = cfgs
            else:
                if not is_large_data:
                    used_cfg = {k: v for k, v in cfgs.items() if not k.endswith('2')}
                else:
                    used_cfg = {k[:-1]: v for k, v in cfgs.items() if k.endswith('2')}
            used_cfgs[field][k] = used_cfg
    return used_cfgs

def make_baseline(
    model_name, 
    model_config: Union[dict, str], 
    n_num: int, 
    cat_card: Optional[List[int]],
    n_labels: int,
    sparsity_scheme: Optional[str] = None,
    device: Union[str, torch.device] = 'cuda',
) -> TabModel:
    """Process Model Configs and Call Specific Model APIs"""
    assert model_name in MODEL_CARDS, f"unrecognized `{model_name}` model name, choose one of valid models in {MODEL_CARDS}"
    if isinstance(model_config, str):
        model_config = load_config_from_file(model_config)['model']
    if MODEL_CARDS[model_name] is None:
        raise NotImplementedError("Please add corresponding model implementation to `models` module")
    if sparsity_scheme is not None:
        assert 'mlp' in model_name
        return MODEL_CARDS[model_name](
            model_config=model_config, 
            n_num_features=n_num, categories=cat_card, n_labels=n_labels, 
            sparsity_scheme=sparsity_scheme)
    return MODEL_CARDS[model_name](
        model_config=model_config, 
        n_num_features=n_num, categories=cat_card, n_labels=n_labels)

def tune(
    model_name: str = None,
    search_config: Union[dict, str] = None, 
    dataset: Dataset = None,
    batch_size: int = 64,
    patience: int = 8, # a small patience for quick tune
    n_iterations: int = 50,
    framework: HPOLib = 'optuna',
    device: Union[str, torch.device] = 'cuda',
    output_dir: Optional[str] = None,
) -> 'TabModel':
    # assert framework in HPOLib, f"hyper tune only support the following frameworks '{HPOLib}'"

    # device
    device = torch.device(device)

    # task params
    n_num_features = dataset.n_num_features
    categories = dataset.get_category_sizes('train')
    if len(categories) == 0:
        categories = None
    n_labels = dataset.n_classes or 1
    y_std = dataset.y_info.get('std') # for regression
    # preprocess
    datas = DataProcessor.prepare(dataset, device=device)
    # hpo search space
    if isinstance(search_config, str):
        search_spaces = load_config_from_file(search_config)
    else:
        search_spaces = search_config
    search_spaces = extract_config(search_spaces) # for multi-choice spaces
    # meta args
    if output_dir is None:
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        output_dir = f"results/{model_name}-{dataset.name}-{now}"
    search_spaces['meta'] = {'save_path': Path(output_dir) / 'tunning'} # save tuning results
    tuned_dir = Path(output_dir) / 'tuned'
    # global variable
    running_time = 0.

    def get_configs(trial: optuna.Trial): # sample configs
        config = {}
        for field in ['model', 'training']:
            config[field] = {}
            for k, space in search_spaces[field].items():
                if space['type'] in ['int', 'float', 'uniform', 'loguniform']:
                    config[field][k] = eval(f"trial.suggest_{space['type']}")(k, low=space['min'], high=space['max'])
                elif space['type'] == 'categorical':
                    config[field][k] = trial.suggest_categorical(k, choices=space['choices'])
                elif space['type'] == 'const':
                    config[field][k] = space['value']
                else:
                    raise TypeError(f"Unsupport suggest type {space['type']} for framework: {framework}")
        config['meta'] = search_spaces['meta']
        config['training'].setdefault('batch_size', batch_size)
        return config
    
    def objective(trial: optuna.Trial):
        configs = get_configs(trial)
        model = make_baseline(
            model_name, configs['model'], 
            n_num=n_num_features, 
            cat_card=categories, 
            n_labels=n_labels, 
            device=device)
        nonlocal running_time
        start = time.time()
        model.fit(
            X_num=datas['train'][0], X_cat=datas['train'][1], ys=datas['train'][2], y_std=y_std,
            eval_set=(datas['val'],),
            patience=patience,
            task=dataset.task_type.value,
            training_args=configs['training'],
            meta_args=configs['meta']) # save best model and configs
        running_time += time.time() - start
        val_metric = (
            model.history['val']['best_metric']
            if dataset.task_type.value != 'regression'
            else -model.history['val']['best_metric']
        )
        return val_metric
    
    def save_per_iter(study: optuna.Study, trail: optuna.Trial):
        # current tuning infos
        tunning_infos = {
            'model_name': model_name,
            'dataset': dataset.name,
            'cur_trail': trail.number,
            'best_trial': study.best_trial.number,
            'best_val_metric': study.best_value,
            'scores': [t.value for t in study.trials],
            'used_time (s)': running_time,
        }
        with open(Path(search_spaces['meta']['save_path']) / 'tunning.json', 'w') as f:
            json.dump(tunning_infos, f, indent=4)
        # only copy the best tuning result
        if study.best_trial.number == trail.number:
            src_dir = search_spaces['meta']['save_path']
            dst_dir = tuned_dir
            print(f'copy best configs and results: {str(src_dir)} -> {str(dst_dir)}')
            print(f'best val metric: ', np.round(study.best_value, 4))
            if os.path.exists(dst_dir):
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir)

    study = optuna.create_study(direction='maximize')
    study.optimize(func=objective, n_trials=n_iterations, callbacks=[save_per_iter])

    # load best model
    config_file = Path(tuned_dir) / 'configs.yaml'
    configs = load_config_from_file(config_file)
    model = make_baseline(
        model_name, configs['model'],
        n_num=n_num_features, cat_card=categories, n_labels=n_labels,
        device=device)
    model.load_best_dnn(tuned_dir)
    # prediction
    predictions, results = model.predict(
        X_num=datas['val'][0], X_cat=datas['val'][1], ys=datas['val'][2], y_std=y_std,
        task=dataset.task_type.value,
        return_probs=True, return_metric=True,
        meta_args={'save_path': output_dir})
    print(results)

    return model