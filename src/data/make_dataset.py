# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn.functional as F
from dotenv import find_dotenv, load_dotenv
from numpy import load
from torch.utils.data import TensorDataset


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    folder = '/Users/maria/example_cookiecutter_project/data/raw/'
    train0 = load(folder + 'train_0.npz')
    train1 = load(folder + 'train_1.npz')
    train2 = load(folder + 'train_2.npz')
    train3 = load(folder + 'train_3.npz')
    train4 = load(folder + 'train_4.npz')
    test = load(folder + 'test.npz')

    train_images = np.concatenate((train0['images'], train1['images'], train2['images'], train3['images'], train4['images']))
    train_labels = np.concatenate((train0['labels'], train1['labels'], train2['labels'], train3['labels'], train4['labels']))
    
    train_images_tensor = F.normalize(torch.Tensor(train_images))
    train_labels_tensor = torch.Tensor(train_labels).type(torch.LongTensor)
    
    # Save train data to TensorDataset
    train_data = TensorDataset(train_images_tensor,train_labels_tensor)

    test_images_tensor = F.normalize(torch.Tensor(test['images']))
    test_labels_tensor = torch.Tensor(test['labels']).type(torch.LongTensor)
    
    # Save train data to TensorDataset
    test_data = TensorDataset(test_images_tensor,test_labels_tensor)

    torch.save(train_data, '/Users/maria/example_cookiecutter_project/data/processed/traindata.pt')
    torch.save(test_data, '/Users/maria/example_cookiecutter_project/data/processed/testdata.pt')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
