import os
import copy
import logging
import random
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.linear_model import LinearRegression
from .utils import split_data, create_lr_scheduler
from .models import EmbedPhenoDataset, model_loader

logger = logging.getLogger(__name__)
pd.set_option("mode.copy_on_write", True)

class NAIAD:
    """
    A class for the NAIAD deep learning model.

    Args:
        data (pandas.DataFrame): DataFrame containing phenotype measurements
        n_train (int or float, optional): number of data points to use for training. If n_train > 1 and is an `int`, then treat it 
            as the absolute number of training examples. If 0 < n_train < 1, then treat it as a fraction of the dataset to use for training.
        n_val (int or float, optional): number of data points to use for validation. If n_val > 1 and is an `int`, then treat it 
            as the absolute number of validation examples. If 0 < n_train < 1, then treat it as a fraction of the dataset to use for validation.
        n_test (int or float, optional): number of data points to use for testing. If n_test > 1 and is an `int`, then treat it 
            as the absolute number of test examples. If 0 < n_train < 1, then treat it as a fraction of the dataset to use for testing.
        batch_size (int): size of batch to use for model training
        seed (float): default 0. Seed to use for random sampling and parameter initialization
            in numpy and torch
        
        Data spliting examples:
        Example 1: Using absolute counts
            n_train = 2000
            n_val = 1000
            n_test = 1000

        Example 2: Using fractions of the dataset
            n_train = 0.5
            n_val = 0.2
            n_test = 0.3

        Example 3: Combining absolute counts with fractions
            n_train = 0.5
            n_val = 0.1
            n_test = 1000
    """
    
    # map number of times each gene is seen to embedding size for model
    n_gene_seen = {0: 2,
                   2: 2,
                   4: 4,
                   10: 16,
                   20: 16,
                   30: 32,
                   40: 64,
                   60: 64,
                   80: 128,
                   100: 128}
    
    def __init__(self, 
                 data,
                 n_train,
                 n_test,
                 n_val = None,
                 batch_size = 1024, 
                 add_training_rank = False,
                 ranking_bins = 10,
                 seed = None):
                 
        self.original_data = data
        self.seed = seed
        self.batch_size = batch_size
        self.add_training_rank = add_training_rank
        self.ranking_bins = ranking_bins

        gene_cols = [col for col in self.original_data.columns if 'gene' in col]
        genes = [set(self.original_data[col]) for col in gene_cols]
        self.genes = sorted(list(set.union(*genes)))

        # set up number of training points selected
        if n_train > 1:
            if not isinstance(n_train, int):
                raise RuntimeWarning('n_train is greater than 1 but not an int, so casting n_train as an int')
            self.n_train = n_train
        elif 0 < n_train < 1:
            self.n_train = int(self.original_data.shape[0] * n_train)
        else:
            raise RuntimeError('n_train must be greater than 0')

        # set up number of testing points selected
        if n_test > 1:
            if not isinstance(n_test, int):
                raise RuntimeWarning('n_test is greater than 1 but not an int, so casting n_test as an int')
            self.n_test = n_test
        elif 0 < n_test < 1:
            self.n_test = int(self.original_data.shape[0] * n_test)
        else:
            raise RuntimeError('n_test must be greater than 0')
        
        # set up number of val points selected
        # NOTE: if `n_val` is not specified, then we assume all data points not used for training and testing are used for validation
        if n_val is None:
            self.n_val = int(self.original_data.shape[0] - self.n_test - self.n_train)
        elif n_val > 1:
            if not isinstance(n_val, int):
                raise RuntimeWarning('n_val is greater than 1 but not an int, so casting n_val as an int')
            self.n_val = n_val
        elif 0 < n_val < 1:
            self.n_val = int(self.original_data.shape[0] * n_val)
        else:
            raise RuntimeError('n_val must be greater than 0')
        
        if self.n_train + self.n_val + self.n_test > self.original_data.shape[0]:
            raise RuntimeError('Number of data points assigned to train, val, and test is larger than total number \
                               of available data points.')

    def set_seed(self, seed, set_torch=True):
        """
        Set seed for all random number generators used in Numpy and Pytorch, and update internal object seed to `seed`.
        """
        self.np_rng = np.random.default_rng(seed)
        np.random.seed(seed)
        random.seed(seed)

        if set_torch:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            # self.torch_rng = torch.Generator().manual_seed(seed)
        
        self.seed = seed
        self.new_seed_old_shuffle = True # did we update the seed without reshuffling the data?
    
    def prepare_dataloaders(self):
        """
        Create dataloaders for each split within `self.data`. Also create a dataloader for the overall dataset. Store dataloaders in 
        `self.dataloaders` attribute.
        """
        if self.data is None:
            raise ValueError('Must populate `self.data` field before calling `prepare_dataloaders()`.')
        
        datasets = {split: EmbedPhenoDataset(self.data[split], self.genes, split) for split in self.data}
        dataloaders = {split: DataLoader(datasets[split], shuffle=False, batch_size=self.batch_size) for split in datasets}
        self.dataloaders = dataloaders

    def prepare_data(self):
        """
        Shuffle rows within `self.original_data` based on `self.seed`, split the data, and and create dataloaders for each data split.
        """
        if self.seed is None:
            raise RuntimeError('Must run `set_seed()` function before initializing data (since seed used for data splits)')
        
        logger.info(f'Shuffling data with seed {self.seed}...')
        self.data = self.original_data.sample(frac=1, random_state=self.np_rng) # shuffle rows of data



        self.data = split_data(self.data, self.n_train, self.n_val, self.n_test)
        # add rank to training data
        if self.add_training_rank:
            self.data['train']['training_rank'], bins = pd.qcut(self.data['train']['comb_score'],
                                                                q=self.ranking_bins, retbins=True, labels=False)
            self.data['val']['training_rank'] = self.data['val']['comb_score'].apply(lambda score: self.assign_rank(score, bins))
            self.data['test']['training_rank'] = self.data['test']['comb_score'].apply(lambda score: self.assign_rank(score, bins))
        # self.dataloaders = prepare_dataloaders(self.data, EmbedPhenoDataset, self.batch_size, genes=self.genes)
        self.prepare_dataloaders()

    def assign_rank(self, score, bins):
        """Assign a bin and rank to a comb_score based on bins."""
        rank_idx = np.digitize(score, bins, right=True) - 1
        # if the value is smaller than the smallest bin, then assign it to  -1. 
        rank_idx = np.clip(rank_idx, -1, len(bins) - 2)  # Ensure bin index is within bounds
        return rank_idx

    def initialize_model(self, model_args=None, device='cpu'):
        """
        Initialize NAIAD model used for training and inference of genetic perturbation response. Use self.model_args to
        initialize model, and populate it with default values for `d_embed` and `d_pheno_hid` if they aren't already
        specified by user.
        
        Args:
            model_args (dict, optional): a dictionary containing arguments to use for NAIAD model. If it is not provided, then
                a default dictionary of model arguments will be used. The dictionary should have the following entries:
                    - model_type (str): model type corresponding to 'embed', 'pheno', 'both'
                    - d_embed (int): dimension of embedding and its submodel hidden layers
                    - d_pheno_hid (int): hidden layer dimension of pheno submodel
                    - p_dropout (float): dropout probability for network during training
            device (str, optional): device for running model, e.g. 'cpu' or 'cuda'. Default: 'cpu'
        """
        if self.seed is None:
            raise RuntimeWarning('Initializing model without setting a seed via `set_seed()`')
        
        self.model_args = model_args
        self.device = device

        if model_args is None:
            self.model_args = {}

        if 'model_type' not in self.model_args:
            self.model_args['model_type'] = 'both'
            
        if 'p_dropout' not in self.model_args:
            self.model_args['p_dropout'] = 0.1

        if 'd_embed' not in self.model_args:
            logger.info('`d_embed` is not set for model, so assigning a value based on training dataset size and number of genes in dataset')
        
            n_gene = len(self.genes)
            n_gene_seen_in_train = self.n_train / n_gene
            n_gene_seen_floor = [x for x in self.n_gene_seen if x < n_gene_seen_in_train][-1] # if want to make this more efficient, do a binary search of the sorted list of keys
            self.model_args['d_embed'] = self.n_gene_seen[n_gene_seen_floor]

        if 'd_pheno_hid' not in self.model_args:
            logger.info('`d_pheno_hid` is not set for model, so using default value of `256`')
            self.model_args['d_pheno_hid'] = 256

        if 'var_pred' in self.model_args:
            raise ValueError('var_pred is not currently supported by NAIAD')
        
        self.model = model_loader(len(self.genes), **self.model_args).to(self.device)

    def setup_trainer(self, n_epoch, model_optimizer_settings=None, loss_fn=nn.MSELoss(reduction='sum')):
        """
        Create torch optimizer and LR annealer for given `self.model`, using the specified `self.model_type`.

        Args:
            model_optimizer_settings (dict, optional): a dictionary or a list of dictionaries for settings of model
                optimizer. If set as None, then will be populated with default values. Should be either a dictionary 
                with the following entries:
                    - pheno_lr (optional: float): learning rate for pheno submodel, 
                        present if model_type is 'pheno' or 'both'
                    - embed_lr (optional: float): learning rate for embedding submodel, 
                        present if model_type is 'embed' or 'both'
                    - weight_decay (float): weight decay for model
            n_epoch (int): number of epochs to train the model
            loss_fn (function or nn.Module, optional): a loss function for the model (can be any differentiable function or an nn.Module).
                By default loss is torch.nn.MSELoss()
        """
        if self.model is None:
            raise RuntimeError('Must call `initialize_model()` before calling `setup_optimzer()`.')

        if model_optimizer_settings is None:
            self.optimizer_settings = {'pheno_lr': 1e-2, 'embed_lr': 1e-2, 'weight_decay': 0}
        else:
            self.optimizer_settings = model_optimizer_settings

        self.n_epoch = n_epoch
        self.loss_fn = loss_fn
        
        model_type = self.model_args['model_type']
        weight_decay = self.optimizer_settings['weight_decay']
        optimizer_dicts = []
        if model_type == 'pheno' or model_type == 'both':
            optimizer_dicts.append({'params': self.model.pheno_ffn.parameters(), 'lr': self.optimizer_settings['pheno_lr'], 'weight_decay': weight_decay})
        if model_type == 'embed' or model_type == 'both':
            embedding_params = list(self.model.embedding_ffn.parameters()) + list(self.model.embedding.parameters()) + list(self.model.embedding_comb.parameters())
            optimizer_dicts.append({'params': embedding_params, 'lr': self.optimizer_settings['embed_lr'], 'weight_decay': weight_decay})

        optimizer = torch.optim.Adam(optimizer_dicts)
        self.optimizer = optimizer

        n_train_steps = self.n_epoch * len(self.dataloaders['train'])
        self.lr_scheduler = create_lr_scheduler(self.optimizer, n_train_steps / 10, n_train_steps)

    def train_model(self, ranking_model=False):
        """
        Run training loop for `self.model` for `self.n_epoch` using the training data provided in `self.dataloaders`, the optimizer
        provided by `self.optimizer`, and the scheduler provided by `self.lr_scheduler`. Loss statistics are stored in `self.training_metrics`.
        The final model from the end of training is stored as `self.model`, and the best model from training (based on validation loss) 
        is stored as `self.best_model`.
        """
        if self.model is None or self.optimizer is None or self.lr_scheduler is None:
            raise RuntimeError('Must call `initialize_model()` before calling `train_model()`.')
        if self.optimizer is None:
            raise RuntimeError('Must call `setup_trainer()` before calling `train_model()`.')
        
        if ranking_model:
            rank_predictor_optimizer =  torch.optim.Adam(self.model.rank_ffn.parameters(), lr=1e-3)


       
        all_loss = {split: [] for split in self.dataloaders}
        min_val_loss = np.inf
        best_model = self.model
        for _ in range(self.n_epoch):
            train_loss = 0
            train_rank_loss = 0
            self.model.train()
            for  train_loader in self.dataloaders['train']:
                if ranking_model:
                    genes, targets, phenos, rank = train_loader
                    rank = rank.to(self.device)
                else:
                    genes, targets, phenos = train_loader
                genes = genes.to(self.device)
                targets = targets.to(self.device)
                phenos = phenos.to(self.device)

                preds = self.model(genes, phenos)
                loss = self.loss_fn(targets, preds)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                train_loss += loss


                if ranking_model:
                    rank_pred = self.model.forward_rank_predictor(genes, phenos)
#                    rank_loss = F.mse_loss(rank.squeeze(), rank_pred.squeeze())
                    rank_loss = ((rank.squeeze() -  rank_pred.squeeze())**2).mean()
                    train_rank_loss += rank_loss

                    rank_predictor_optimizer.zero_grad()
                    rank_loss.backward()
                    rank_predictor_optimizer.step()
                 

            all_loss['train'].append(train_loss.detach().cpu().numpy().item() / self.dataloaders['train'].dataset.data.shape[0])

            self.model.eval()
            with torch.no_grad():
                for split in self.dataloaders:
                    split_loss = 0

                    for loader in self.dataloaders[split]:
                        if ranking_model:
                            genes, targets, phenos, rank = loader
                            rank = rank.to(self.device)
                        else: 
                            genes, targets, phenos = loader
                
                        genes = genes.to(self.device)
                        targets = targets.to(self.device)
                        phenos = phenos.to(self.device)

                        preds = self.model(genes, phenos)
                        loss = self.loss_fn(targets, preds)

                        split_loss += loss
                    
                    split_loss = split_loss.detach().cpu().numpy().item() / self.dataloaders[split].dataset.data.shape[0]

                    if split == 'val' and split_loss < min_val_loss:
                        min_val_loss = split_loss
                        best_model = copy.deepcopy(self.model)

                    all_loss[split].append(split_loss)

        self.best_model = best_model
        self.training_metrics = {f'{split}_loss': all_loss[split] for split in all_loss}

    def generate_attentions(self, use_best=False):
        all_attentions = {split: [] for split in self.dataloaders}
        self.model.eval()
        with torch.no_grad():
            if use_best:
                model = self.best_model
            else:
                model = self.model
            for split in self.dataloaders:
                for loader in self.dataloaders[split]:
                    genes, targets, phenos = loader
                    genes = genes.to(self.device)
                    targets = targets.to(self.device)
                    phenos = phenos.to(self.device)
                    attention_weights = model.get_attention_weights(genes).detach().cpu()
                    # attention_weights dims: (number_atten, batch_size, target seq length, source seq length)
                    all_attentions[split].append(attention_weights)
                all_attentions[split] = torch.cat(all_attentions[split], dim=1)
        self.all_attentions = all_attentions

   
    def generate_preds(self, use_best=False):
        """
        Generate predictions for each data split using the specified model given by `use_best`.

        Args:
            use_best (bool): should the 'best_model' (from early stopping) or final 'model' (from last epoch of training) be used to 
                generate predictions? if True: use best_model; if False, use final model

        Returns:
            data (dict): dictionary of pandas DataFrames containing genes, pheno values, and model predictions
        """
        if self.best_model is None and use_best:
            raise RuntimeError('Cannot set `use_best=True` to generate predictions before model has been trained.')
        
        data = {split: self.data[split] for split in self.data}
        for split in self.dataloaders:
            split_preds = []
            for split_loader in self.dataloaders[split]:
                if self.add_training_rank:
                    genes, targets, phenos, rank = split_loader
                    rank = rank.to(self.device)
                else:
                    genes, targets, phenos = split_loader

                genes = genes.to(self.device)
                targets = targets.to(self.device)
                phenos = phenos.to(self.device)

                if use_best:
                    model = self.best_model
                else:
                    model = self.model
                preds = model(genes, phenos)

                preds = preds.detach().cpu().numpy().tolist()
                split_preds.extend(preds)

            data[split].loc[:, 'preds'] = split_preds

        self.preds = data

    def generate_intermediate_results(self, use_best=False):
        """
        Generate intermediate results for each data split using the specified model given by `use_best`.

        Args:
            use_best (bool): should the 'best_model' (from early stopping) or final 'model' (from last epoch of training) be used to 
                generate predictions? if True: use best_model; if False, use final model

        Returns:
            data (dict): dictionary of pandas DataFrames containing genes, pheno values, and model predictions
        """
        if self.best_model is None and use_best:
            raise RuntimeError('Cannot set `use_best=True` to generate predictions before model has been trained.')
        
        data = {split: self.data[split] for split in self.data}
        self.model.eval()
        with torch.no_grad():
            for split in self.dataloaders:
                split_preds = []
                split_gene_term = []
                split_additive_term = []

                for genes, targets, phenos in self.dataloaders[split]:
                    genes = genes.to(self.device)
                    targets = targets.to(self.device)
                    phenos = phenos.to(self.device)

                    if use_best:
                        model = self.best_model
                    else:
                        model = self.model
                    preds, gene_term, additive_term = model(genes, phenos, return_intermediate=True)

                    split_preds.extend(preds.detach().cpu().numpy().tolist())
                    split_gene_term.extend(gene_term.detach().cpu().numpy().tolist())
                    split_additive_term.extend(additive_term.detach().cpu().numpy().tolist())

                data[split].loc[:, 'preds'] = split_preds
                data[split].loc[:, 'gene_term'] = split_gene_term
                data[split].loc[:, 'additive_term'] = split_additive_term
    
        self.intermediate_results = data

    def run_linear_regression(self):
        """
        Run linear regression on the embeddings to predict the targets. Store the linear regression model in `self.lr_model`.
        """
       # model.fit(naiad_data[['g1_score', 'g2_score']], naiad_data['comb_score'])

        self.lr_model = LinearRegression()
        self.lr_model.fit(self.data['train'][['g1_score', 'g2_score']], self.data['train']['comb_score'])
        for split in self.data:
            self.data[split]['linear_predicted'] = self.lr_model.predict(self.data[split][['g1_score', 'g2_score']])
            self.data[split]['linear_residuals'] = self.data[split]['comb_score'] - self.data[split]['linear_predicted']
 


    def plot_loss_curves(self, log=False, ax=None):
        """
        Plot the training and validation loss curves. 
        
        Args:
            ax (matplotlib.axes.Axes, Optional): an Axes object on which to plot the results
        
        Returns:
            ax (matplotlib.axes.Axes): Axes object containing requested plot
        """

        if len(self.training_metrics) == 0:
            raise RuntimeError('Need to call train_model() function before plotting results.')

        data = self.training_metrics
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        if log: 
            sns.lineplot(np.log(data['train_loss']), label = 'Train Loss', ax = ax)
            sns.lineplot(np.log(data['val_loss']), label = 'Val Loss', ax = ax)
            ax.set_ylabel('log(Loss)')
        else:
            sns.lineplot(data['train_loss'], label = 'Train Loss', ax = ax)
            sns.lineplot(data['val_loss'], label = 'Val Loss', ax = ax)
            ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
        ax.set_title(f'Training and Val Loss')
        
        return ax
    
    def plot_preds(self, split, ax=None):
        """
        Plot targets vs predictions for the model `ensemble` from `sampling_type` in `round` and data `split`.

        Args:
            split (str): which data split to use for plots (options are 'train', 'val', 'test', 'overall')
            ax (matplotlib.axes.Axes, Optional): an Axes object on which to plot the results

        Returns:
            ax (matplotlib.axes.Axes): Axes object containing requested plot
        """

        if len(self.preds) == 0:
            raise RuntimeError('Need to call train_model() function before plotting results.')

        data = self.preds[split]
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        sns.scatterplot(data, x='comb_score', y=f'preds', ax=ax)
        ax.set_xlabel('Targets')
        ax.set_ylabel('Preds')
        ax.set_title(f'Targets vs Preds, Split: {split}')
        
        return ax
    
    def save_results(self, save_dir, file_prefix=''):
        """
        Save the `results`, `training_metrics`, and `aggregate_metrics` from the ActiveLearner instance as
        a pickle file.
        
        Args:
            save_dir (str): directory for storing results
            file_prefix (str, optional): prefix for file name to store data
        """
        # in theory these are same condition, so if one of them fails but not the other then there is a bug somewhere
        if len(self.training_metrics) == 0: 
            raise RuntimeError('Must call `train_model()` function to generate results before saving them to file.')
        
        if not os.path.isdir(save_dir):
            raise ValueError('Must supply a valid directory to save results.')
        
        combined_results = {'preds': self.preds,
                            'training_metrics': self.training_metrics}
        
        file_name = f'NAIAD_{self.data_source}_method{self.method}_seed{self.seed}.pkl'
        if file_prefix == '':
            save_path = os.path.join(save_dir, file_name)
        else: 
            save_path = os.path.join(save_dir, f'{file_prefix}_{file_name}')

        with open(save_path, 'wb') as f:
            pickle.dump(combined_results, f)