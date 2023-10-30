# tune then finetune in one function
import os
import sys
sys.path.append(os.getcwd())

import torch
from data.processor import DataProcessor
from utils.model import load_config_from_file, seed_everything, get_model_cards, tune, make_baseline

if __name__ == '__main__':
    seed_everything(42)
    device = torch.device('cuda')
    print('available model infos: ', get_model_cards())
    base_model = 'mlp'
    dataset_name = 'adult'
    # model, training args
    search_space_file = f'configs/{base_model}.yaml' # refer to sample search space config file and build yours
    output_dir = f"results-tuned/{base_model}/{dataset_name}" # output dir for tuned configs & checkpoints
    # load dataset
    dataset = DataProcessor.load_preproc_default(output_dir, base_model, dataset_name, seed=0)

    # tune (will load the checkpoint of the best config to predict at the end)
    model = tune(
        model_name=base_model, 
        search_config=search_space_file,
        dataset=dataset, 
        batch_size=128,
        patience=3, # a small patience for fast tune
        n_iterations=5, # tune interations
        device=device,
        output_dir=output_dir)
    print('done')

    # if you want to use the best tuned config
    # but a different training args (e.g. patience, batch size)
    # you should manually load the best config and finetune
    best_config_file = f'{output_dir}/tuned/configs.yaml'
    best_configs = load_config_from_file(best_config_file)
    # data args
    n_num_features = dataset.n_num_features
    categories = dataset.get_category_sizes('train')
    if len(categories) == 0:
        categories = None
    n_labels = dataset.n_classes or 1 # regression n_classes is None
    y_std = dataset.y_info.get('std') # for regression
    # build model from the given config
    model = make_baseline(
        model_name=base_model, 
        # you can directly pass config file, but this can not modify training args explicitly
        # model_config=best_config_file, 
        model_config=best_configs['model'],
        n_num=n_num_features, 
        cat_card=categories, 
        n_labels=n_labels,
        device=device
    )
    # here you can modify the training args (if read config file above)
    best_configs['training']['batch_size'] = 256
    best_configs['training']['lr'] = 5e-5
    output_dir2 = 'final_output_dir'
    best_configs['meta']['save_path'] = output_dir2 # save new results with tuned configs
    # prepare tensor data
    datas = DataProcessor.prepare(dataset, model)
    # finetune
    model.fit(
        X_num=datas['train'][0], X_cat=datas['train'][1], ys=datas['train'][2], y_std=y_std, 
        eval_set=(datas['val'],),
        patience=8, # can use a differnet patience
        task=dataset.task_type.value,
        training_args=best_configs['training'], # training args
        meta_args=best_configs['meta'], # meta args: other infos, e.g. result dir, experiment name / id
    )
    # prediction
    predictions, results = model.predict(
        X_num=datas['test'][0], X_cat=datas['test'][1], ys=datas['test'][2], y_std=y_std,
        task=dataset.task_type.value,
        return_probs=True, return_metric=True, return_loss=True,
    )
    model.save_prediction(output_dir2, results) # save results
    print("=== Prediction (best metric) ===")
    print(results)