# finetune a baseline with given configs
import os
import sys
sys.path.append(os.getcwd())

import torch
from data import available_datasets
from data.processor import DataProcessor
from utils.model import seed_everything, get_model_cards, make_baseline, load_config_from_file


if __name__ == '__main__':
    seed_everything(42)
    device = torch.device('cuda')
    # model infos
    print('model cards: ', get_model_cards())
    base_model = 'mlp'
    # dataset infos
    print('available datasets: ', available_datasets())
    dataset_name = 'adult'
    # config files
    default_config_file = f'configs/default/{base_model}.yaml' # path to your config file
    output_dir = f"results/{base_model}/{dataset_name}" # path to save results
    # load configs
    configs = load_config_from_file(default_config_file) # or you can direcly pass config file to `make_baseline`
    # some necessary configs
    configs['training']['max_epochs'] = 100 # training args: max training epochs
    configs['training']['batch_size'] = 128 # training args: batch_size
    configs['meta'] = {'save_path': output_dir} # meta args: result dir

    # load dataset (processing upon model type)
    dataset = DataProcessor.load_preproc_default(output_dir, base_model, dataset_name, seed=0)
    # build model
    n_num_features = dataset.n_num_features
    categories = dataset.get_category_sizes('train')
    if len(categories) == 0:
        categories = None
    n_labels = dataset.n_classes or 1 # regression n_classes is None
    y_std = dataset.y_info.get('std') # for regression

    model = make_baseline(
        base_model, configs['model'], 
        n_num=n_num_features, 
        cat_card=categories, 
        n_labels=n_labels,
        device=device
    )
    # convert to tensor
    datas = DataProcessor.prepare(dataset, model)

    # training (automatically load best model at the end)
    model.fit(
        X_num=datas['train'][0], X_cat=datas['train'][1], ys=datas['train'][2], y_std=y_std, 
        eval_set=(datas['val'],), # similar as sk-learn
        patience=8, # for early stop, <= 0 no early stop
        task=dataset.task_type.value,
        training_args=configs['training'], # training args
        meta_args=configs['meta'], # meta args: other infos, e.g. result dir, experiment name / id
    )

    # prediction (best metric checkpoint)
    # model.load_best_dnn(output_dir, file='best') # or you can load manually
    predictions, results = model.predict(
        X_num=datas['test'][0], X_cat=datas['test'][1], ys=datas['test'][2], y_std=y_std,
        task=dataset.task_type.value,
        return_probs=True, return_metric=True, return_loss=True,
    )
    model.save_prediction(output_dir, results) # save results
    print("=== Prediction (best metric) ===")
    print(results)

    # prediction (best logloss checkpoint)
    if dataset.task_type.value != 'regression':
        model.load_best_dnn(output_dir, file='best-logloss')
        predictions, results = model.predict(
            X_num=datas['test'][0], X_cat=datas['test'][1], ys=datas['test'][2], y_std=y_std,
            task=dataset.task_type.value,
            return_probs=True, return_metric=True, return_loss=True,
        )
        model.save_prediction(output_dir, results, file='prediction_logloss')
        print("=== Prediction (best logloss) ===")
        print(results)