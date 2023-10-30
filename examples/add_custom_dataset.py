# add your custom datasets from csv files
import os
import sys
sys.path.append(os.getcwd())
from data import available_datasets
from data.processor import DataProcessor

if __name__ == '__main__':
    # my_csv_file = 'examples/[kaggle]Assay of serum free light chain.csv' # binclass
    # print('available datasets: ', available_datasets())
    # DataProcessor.add_custom_dataset(my_csv_file) # add a dataset
    # print('available datasets: ', available_datasets())
    # dataset = DataProcessor.load_preproc_default('result/test', 'ft-transformer', '[kaggle]Assay of serum free light chai')
    # DataProcessor.del_custom_dataset("[kaggle]Assay of serum free light chain") # remove a dataset
    print('available datasets: ', available_datasets())
    my_csv_file = 'examples/[openml]bodyfat.csv' # regression
    DataProcessor.add_custom_dataset(my_csv_file) # add
    print('available datasets: ', available_datasets())
    dataset = DataProcessor.load_preproc_default('result/test', 'ft-transformer', '[openml]bodyfat') # load
    pass