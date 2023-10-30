import os
import json
import shutil
from pathlib import Path
import numpy as np

DATA = Path('data/datasets')
CUSTOM_DATA = Path('data/custom_datasets') # user custom datasets

BENCHMARKS = {
    'ft-transformer': {
        'path': 'data/benchmarks/ft-transformer',
        # The dataset list of the benchmark
        # if not appears in DATASETS or CUSTOM_DATASETS
        # it should be added to the 'path'
        'datasets': [], # priority: DATASETS > CUSTOM_DATASETS > 'path'
    },
    't2g-former': {
        'path': 'data/benchmarks/t2g-former',
        'datasets': ['california', 'eye', 'helena', 'otto']
    }
}

# available single datasets and specific DNN processing methods
# default: `normalization: quantile`
DATASETS = {
    'adult': {'path': DATA / 'adult'},
    'california': {'path': DATA / 'california'},
    'churn': {'path': DATA / 'churn'},
    'eye': {'path': DATA / 'eye', 'normalization': 'standard'},
    'gesture': {'path': DATA / 'gesture'},
    'helena': {'path': DATA / 'helena', 'normalization': 'standard'},
    'higgs-small': {'path': DATA / 'higgs-small'},
    'house': {'path': DATA / 'house'},
    'jannis': {'path': DATA / 'jannis'},
    'otto': {'path': DATA / 'otto', 'normalization': None},
    'fb-comments': {'path': DATA / 'fb-comments'},
    'year': {'path': DATA / 'year'},
}

CUSTOM_DATASETS = {}

def read_custom_infos():
    with open(CUSTOM_DATA / 'infos.json', 'r') as f:
        custom_infos = json.load(f)
    return custom_infos
# read the `infos.json` to load
def reload_custom_infos():
    custom_infos = read_custom_infos()
    global CUSTOM_DATASETS
    CUSTOM_DATASETS = {
        info['name']: {
            'path': CUSTOM_DATA / info['name'], 
            'task_type': info['task_type'],
            'normalization': info.get('normalization', 'quantile')
        } for info in custom_infos['data_list']
    }
reload_custom_infos()

def write_custom_infos(infos):
    with open(CUSTOM_DATA / 'infos.json', 'w') as f:
        json.dump(infos, f, indent=4)
    reload_custom_infos()

def push_custom_datasets(
    X_num, X_cat, ys, idx,
    info # TODO: add normalization field to info
):
    data_dir = CUSTOM_DATA / info['name']
    os.makedirs(data_dir)
    try:
        for spl in ['train', 'val', 'test']:
            np.save(data_dir / f'idx_{spl}.npy', idx[spl])
            if X_num is not None:
                np.save(data_dir / f'X_num_{spl}.npy', X_num[spl])
            if X_cat is not None:
                np.save(data_dir / f'X_cat_{spl}.npy', X_cat[spl])
            np.save(data_dir / f'y_{spl}.npy', ys[spl])
        with open(data_dir / 'info.json', 'w') as f:
            json.dump(info, f, indent=4)
    except:
        print('failed to add custom dataset: ', info['name'])
        shutil.rmtree(data_dir)
        return
    custom_infos = read_custom_infos()
    custom_infos['data_list'].append({'name': info['name'], 'task_type': info['task_type']})
    custom_infos[info['task_type']] += 1
    custom_infos['n_datasets'] += 1
    write_custom_infos(custom_infos)
    print(f"push dataset: '{info['name']}' done")

def available_datasets():
    return sorted(list(DATASETS.keys()) + list(CUSTOM_DATASETS.keys()))