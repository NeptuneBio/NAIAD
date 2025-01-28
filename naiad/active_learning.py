import os
import copy
import logging
import random
import re
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression

import torch
import torch.multiprocessing as mp

from .naiad import NAIAD
from .utils import split_data, find_top_n_perturbations

logger = logging.getLogger(__name__)
pd.set_option("mode.copy_on_write", True)

class ActiveLearner:
    """
    A class for active learning of combinatorial perturbation data.

    Args:
        n_round (int): number of rounds to perform active learning. This includes 
            the first round of training on randomly sampled `start_frac` of the data.
        data (pandas.DataFrame): DataFrame containing phenotype measurements
        n_ensemble (int): number of models to initialize in the ensemble for each round
        n_epoch (int): number of epochs to train the model
        device (str): device for running model, e.g. 'cpu' or 'cuda' or 'mps'
        model (torch.nn.Module, optional): class of model to use for active learning. Default: NAIAD
        model_args (dict or list[dict], optional): a dictionary or list of dictionaries containing
            arguments to use for `model`, default None. Should be either a single dictionary (all 
            models for each active learning round are initialized with the same parameters), or a 
            list of length `n_round` containing parameter dictionaries for `model` (the models in each 
            round are initialized with the corresponding index in the list). 
            
            If None, then model will use inferred values for `model_args`.

            Entries within each dictionary should include:
            - model_type (str): model type corresponding to 'embed', 'pheno', 'both'
            - d_in (int): embedding dimension of model
            - d_embed_hid (int): hidden layer dimension of embedding submodel
            - d_pheno_hid (int): hidden layer dimension of phenotype submodel
            - p_dropout (float): dropout probability for network during training

        model_optimizer_settings (dict or list[dict], optional): a dictionary or a list of 
            dictionaries for settings of model optimizer default None. Should be either a single 
            dictionary (all models for each active learning round are initialized with the same 
            parameters), or a list of length `n_rounds` containing parameter dictionaries for `model` 
            (the models in each round are initialized with the corresponding index in the list). 

            If None, then model will use default values for model_optimizer_settings.

            Entries within each dictionary should include:
            - pheno_lr (optional: float): learning rate for pheno submodel, 
                present if model_type is 'pheno' or 'both'
            - embed_lr (optional: float): learning rate for embedding submodel, 
                present if model_type is 'embed' or 'both'
            - weight_decay (float): weight decay for model

        seed (float): default 0. Seed to use for random sampling and parameter initialization
            in numpy and torch
        start_frac (float, optional): fraction of data to initially use for training. Must either specify `start_frac` and `inc_frac` or `n_sample`
        inc_frac (float, optional): fraction of data to increase per round of sampling. Must either specify `start_frac` and `inc_frac` or `n_sample`
        test_frac (float, optional): fraction of data to use for test evaluation. Must either specify `test_frac` or `n_test`
        n_sample (list[int], optional): number of data points to use for training in each round. Must either specify `n_sample` or `start_frac` and `inc_frac`
        n_test (int, optional): number of data points to use for testing across all rounds. Must either specify `n_test` or `test_frac`
        method (str): selection method of new points in active learning. Options are 'mean',
            'std', 'mean+std', 'residual', 'residual+std', 'leverage'
        batch_size (int): size of batch to use for model training
        method_min (bool): should the selection `method` be minimized for selecting
            new points during active learning?
        early_stop (bool): should an early stopping criteria be used to select the best model during training? If yes, then use the validation set to find
            the best model for early stopping
    """
    def __init__(self, 
                 n_round, 
                 data,
                 n_ensemble, n_epoch,  
                 device, 
                 model = NAIAD, 
                 model_args = None, model_optimizer_settings = None,
                 n_sample = None, start_frac = None, inc_frac = None,
                 n_test = None, test_frac = None,
                 seed = None,
                 batch_size = 1024,
                 method = None, method_min = None,
                 early_stop = False):
        self.n_round = n_round
        self.original_data = data
        self.model = model
        self.model_args = model_args
        self.model_optimizer_settings = model_optimizer_settings
        self.n_ensemble = n_ensemble
        self.n_epoch = n_epoch
        self.device = device
        self.seed = seed
        self.batch_size = batch_size
        self.early_stop = early_stop

        treatment_cols = [col for col in self.original_data.columns if re.match(r'^id\d+$', col)]
        treatments = [set(self.original_data[col]) for col in treatment_cols]
        self.treatments = sorted(list(set.union(*treatments)))

        # check argument validity    
        if not (model_optimizer_settings is None or isinstance(model_optimizer_settings, dict) or \
                (isinstance(model_optimizer_settings, list) and len(model_optimizer_settings) != n_round)):
            raise ValueError('model_optimizer_settings should be None (settings will be set to default \
                             values), a single dictionary (all model settings for each active learning round \
                             are initialized with the same parameters), or a list of length `n_rounds` \
                             containing parameter dictionaries for `model` (the models in each round are \
                             initialized with the corresponding index in the list)')
        
        if not (model_args is None or isinstance(model_args, dict) or \
                (isinstance(model_args, list) and len(model_args) == n_round)):
            raise ValueError('model_args should be None (settings will be set to default values), \
                             a single dictionary (all model settings for each active learning round are \
                             initialized with the same parameters), or a list of length `n_rounds` containing \
                             parameter dictionaries for `model` (the models in each round are initialized \
                             with the corresponding index in the list)')
        
        if n_round < 1:
            raise ValueError('Must have at least one round of training') 

        # if not specified during initialization, user must set method separately
        if (method is None) ^ (method_min is None):
            raise ValueError('During initialization, `method` and `method_min` should be either both specified \
                              or both None.')
        if method is not None and method_min is not None:
            self.update_method(method, method_min)
        else:
            self.method = None
            self.method_min = None

        # set up sampling schedule
        if (n_sample is None) and (start_frac is None or inc_frac is None):
            raise ValueError('Must specify either `n_sample`, or both `start_frac` and `inc_frac`')
        if (n_sample is not None) and (start_frac is not None or inc_frac is not None):
            raise ValueError('Cannot specify both `n_sample` and (`start_frac` and `inc_frac`)')
        if n_sample is not None:
            if not isinstance(n_sample, list) or len(n_sample) < self.n_round:
                raise ValueError('`n_sample` must be a list of ints of length `n_round`')
            self.n_sample = n_sample
        else:
            self.n_sample = [self.original_data.shape[0]*(start_frac + (round*inc_frac)) for round in range(self.n_round)]

        # set up number of testing points selected
        if (n_test is None) and (test_frac is None):
            raise ValueError('Must specify either `n_test` or `test_frac`')
        if (n_test is not None) and (test_frac is not None):
            raise ValueError('Cannot specify both `n_test` and `test_frac`')
        if n_test is not None:
            self.n_test = n_test
        else:
            self.n_test = int(self.original_data.shape[0] * test_frac)

        # initialize RNGs and data
        if seed is not None:
            self.seed = seed

        # set up model and optimizer initialization lists
        if isinstance(self.model_args, dict) or self.model_args is None:
            self.model_args = [self.model_args for _ in range(self.n_round)]
        if isinstance(self.model_optimizer_settings, dict) or self.model_optimizer_settings is None:
            self.model_optimizer_settings = [self.model_optimizer_settings for _ in range(self.n_round)]

        # initialize dicts holding results and analysis
        self.preds = {} # will be used to store preds of each model
        self.training_metrics = {} # metrics about training (e.g. train loss, val loss)
        self.aggregate_metrics = {} # metrics for aggregated results across ensembles

        # did we update seed without reshuffling data?
        self.new_seed_old_shuffle = False

    def update_method(self, method, method_min):
        """
        Set metric for which ensemble statistics are used for training active learning. Options are 'mean', 'std', 'leverage'

        Args:
            method (str): selection method of new points in active learning. Options are 'mean', 'std', 'mean+std', 'residual,
                'residual+std', 'leverage'
            method_min (bool): should the selection `method` be minimized for selecting new points during active learning?        
        """
        if method not in ['leverage', 'mean', 'std', 'mean+std', 'residual', 'residual+std']:
            raise ValueError('Ensemble method must be "leverage", "mean", "std", "mean+std", "residual", or "residual+std"')
        
        if method == 'leverage':
            raise ValueError('leverage score sampling not yet implemented')
        
        if method_min not in [True, False]:
            raise ValueError('method_min must be either True or False')

        self.method = method
        self.method_min = method_min

    def set_seed(self, seed):
        """
        Set seed for all random number generators used in Numpy and Pytorch, and update internal object seed to `seed`.
        """
        # self.torch_rng = torch.Generator().manual_seed(seed)
        self.np_rng = np.random.default_rng(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.seed = seed
        self.new_seed_old_shuffle = True

    def shuffle_data(self):
        """
        Load data and shuffle rows
        """
        if self.seed is None:
            raise RuntimeError('Must run `set_seed()` function before initializing data (since seed used for data splits)')
        logger.info(f'Shuffling data with seed {self.seed}...')

        self.data = self.original_data.sample(frac=1, random_state=self.np_rng) # shuffle rows of data
        self.new_seed_old_shuffle = False

    # TODO: should this be refactored as a static method?
    def _sample_round_data(self, round, sample_type):
        """
        Sample next round of training data based on results from previous rounds. Can sample either randomly or via active selection.
        For round = 0, only random selection can be used. For subsequent rounds, either random or active sampling can be used.

        Args:
            round (int): which round of active learning are we sampling data for?
            sample_type (str): how to sample training data, options are 'random', 'active'.
        """
        if sample_type == 'random':
            n_train = self.n_sample[round]
            n_val = self.data.shape[0] - n_train - self.n_test
            data = split_data(self.data, n_train, n_val, self.n_test)

        elif sample_type == 'active':
            if (round-1) not in self.preds:
                raise ValueError(f"Can't perform active learning for round {round} before results from round {round-1} have been collected")
            if round == 1: # if first round of active selection, then choose best values from randomly selected data of round 0
                prev_results = self.preds[round-1]['random']
            else:
                prev_results = self.preds[round-1]['active']

            n_select = self.n_sample[round] - self.n_sample[round-1]
            if self.method == 'leverage':
                select_preds = self.np_rng.choice(len(prev_results['val']), size=n_select, replace=False, p=prev_results['val']['leverage'].values)
                select_preds = prev_results['val'].iloc[select_preds, :]
            else:
                select_preds = prev_results['val'].sort_values(by=self.method, ascending=self.method_min).reset_index(drop=True)[:n_select]

            new_train_data = pd.concat([prev_results['train'], select_preds], axis=0).reset_index(drop=True)
            new_val_data = prev_results['val'].merge(new_train_data, how='outer', indicator=True)
            new_val_data = new_val_data[new_val_data['_merge'] == 'left_only'].drop(columns=['_merge']).reset_index(drop=True)

            # use only the columns from the original data, i.e. drop columns of analysis from previous round
            new_train_data = new_train_data[self.data.columns]
            new_val_data = new_val_data[self.data.columns]
            new_test_data = prev_results['test'][self.data.columns]
            data = {'train': new_train_data, 'val': new_val_data, 'test': new_test_data}

        # add overall data to evaluate model on overall performance
        data['overall'] = copy.deepcopy(self.data)
        return data
    
    @staticmethod
    def _calculate_pred_stats(df, categories):
        """
        Given a dataframe `df`, calculate the requested statistics listed in `categories`. 

        Args:
            df (pd.DataFrame): dataframe containing data to collect summary statistics over
            categories (str or list[str]): type of summary statistic to calculate from `df`. Either a single 
                type, or a list of types. Possible options include 'mean', 'std', 'mean+std', 'residual', 'residual+std'

        Returns:
            result_df (pd.DataFrame): dataframe containing summary statistics from `df` requested in 
                `categories`
        """

        if len(set(categories)) < len(categories):
            raise ValueError('Cannot have duplicate entries within categories for calculating prediction statistics')
        
        result_df = pd.DataFrame()
        if isinstance(categories, str):
            categories = [categories]

        model_pred_cols = [col for col in df.columns if re.search(r'(random\d+|active\d+)', col)]
        df_model_preds = df.loc[:, model_pred_cols]

        for cat in categories:
            if cat == 'mean':
                result_df.loc[:, 'mean'] = df_model_preds.mean(axis=1).values
            if cat == 'std':
                result_df.loc[:, 'std'] = df_model_preds.std(axis=1).values
            if cat == 'mean+std':
                result_df.loc[:, 'mean+std'] = np.abs(df_model_preds.mean(axis=1).values) + df_model_preds.std(axis=1).values
            if cat == 'residual':
                result_df.loc[:, 'residual'] = np.abs(df.loc[:, 'linear'].values - df_model_preds.mean(axis=1).values) ## TODO: check which columns are chosen in this calculation
            if cat == 'residual+std':
                result_df.loc[:, 'residual+std'] = df_model_preds.std(axis=1).values + np.abs(df.loc[:, 'linear'].values - df_model_preds.mean(axis=1).values)
            
        return result_df
    
    """
    @staticmethod
    def _calculate_leverage_score(model, genes, gene_comb):
        # TODO: test this function; extend it to handle more than pairwise combinations
        if gene_comb.shape[1] > 2:
            raise ValueError('Cannot calculate leverage score if testing more than 2 genes per combination')
        n_genes = len(genes)
    
        all_pair_genes = []
        for i in range(n_genes):
            pair_genes = [[genes[i], g] if g > genes[i] else [g, genes[i]] for g in genes[i:]]
            all_pair_genes.extend(pair_genes)
        all_pair_g1, all_pair_g2 = zip(*all_pair_genes)
        all_pair_genes = pd.DataFrame({'gene1': all_pair_g1, 'gene2': all_pair_g2})
        
        keep_genes = pd.merge(all_pair_genes, gene_comb, how='left', on=['gene1', 'gene2'], indicator=True)
        keep_genes = (keep_genes['_merge'] == 'both')
        all_pair_genes_filtered = all_pair_genes[keep_genes]

        emb = model.embedding.weight.detach()

        all_pair_emb = []
        for i in range(n_genes):
            pair_emb = emb[i:, :] + emb[i, :]
            all_pair_emb.append(pair_emb)

        all_pair_emb = torch.concat(all_pair_emb, dim=0).cpu().numpy()

        all_pair_emb = (all_pair_emb - np.mean(all_pair_emb, axis=1, keepdims=True)) / np.var(all_pair_emb, axis=1, keepdims=True)
        all_pair_emb = all_pair_emb[keep_genes]
        u, s, vh = np.linalg.svd(all_pair_emb, full_matrices=False) # if don't use full_matrices = False, then U matrix is padded to have
                                                                    # unit length for each row, which doesn't allow us to calculate unique
                                                                    # leverage scores per feature (i.e. row)
        l = np.sum(u**2, axis=1)
        l = l / np.sum(l)

        all_pair_genes_filtered[f'leverage'] = l

        return all_pair_genes_filtered
        """
    
    @staticmethod
    def _overwrite_data_with_measured(pred_data, measured_data):
        """
        Overwrite predictions in `pred_data` with any matching measurements from `measured_data` data. Useful for calculating TPR of finding strongest treatments.
        
        Args:
            pred_data (pd.DataFrame): Dataframe containing columns for `id1`, `id2`, ... `idN` and `mean`
            measured_data (pd.DataFrame): Dataframe containing columns for `id1`, `id2`, ... `idN` and `comb_score`

        Returns:
            combined_data (pd.DataFrame): updated dataframe with new column `measured+pred` for the combination of predicted and measured treatments.
        """

        treatment_cols = [col for col in pred_data.columns if 'id' in col]
        matches = pd.merge(left=pred_data, right=measured_data, on=treatment_cols, how='left', indicator=True)
        match_rows = matches[matches['_merge'] == 'both'].index

        combined_data = copy.deepcopy(pred_data)
        combined_data['measured+pred'] = pred_data['mean']
        combined_data.loc[match_rows, 'measured+pred'] = combined_data.loc[match_rows, 'comb_score']

        return combined_data
        
    def run_active_learning(self):
        """
        Run active learning pipeline for `n_round` rounds. If `n_sample` is specified, all sampling rounds follow the schedule specified in `n_sample`.
        If `start_frac` and `inc_frac` were specified, the first round is simply randomly sampling `start_frac` of the data, and each subsequent 
        round increases the sampling by `inc_frac`. 
        """

        if self.seed is None or self.data is None:
            raise RuntimeError('Need to run set_seed() and shuffle_data() before running active learning.')

        if self.method is None:
            raise RuntimeError('Need to specify method and method_min via update_method() before running active learning.')

        if self.new_seed_old_shuffle:
            raise RuntimeWarning('Seed was updated but data was not reshuffled. This may be an issue if you expect new data splits \
                                 when using new seed.')

        for round in range(self.n_round):
            self.preds[round] = {}
            self.training_metrics[round] = {}            
            round_data = {}
            
            rand_data = self._sample_round_data(round, 'random')
            round_data['random'] = rand_data
            n_train = round_data['random']['train'].shape[0]
            n_val = round_data['random']['val'].shape[0]
            n_test = round_data['random']['test'].shape[0]
            ensemble_models = [self.model(self.original_data, n_train, n_test, n_val, self.batch_size) for _ in range(self.n_ensemble)]

            if (round > 0): # only perform active learning on rounds after the first
                active_data = self._sample_round_data(round, 'active')
                round_data['active'] = active_data

            round_models = {'random': ensemble_models}
            if len(round_data) > 1:
                round_models['active'] = copy.deepcopy(ensemble_models) # duplicate initialized ensemble for both random and active learning

            round_args = self.model_args[round]
            round_optimizer_settings = self.model_optimizer_settings[round]
            
            for sampling_type in round_models:    
                logger.info(f'Starting ensemble training for {sampling_type} selection in round {round}...')
                metrics = {}
                new_cols = []
                for i, model in enumerate(round_models[sampling_type]):
                    model.set_seed(self.seed, set_torch=False) # the model shouldn't actually use the seed for anything, but
                                                               # set the seed anyway to avoid some warnings
                    model.data = round_data[sampling_type]
                    model.prepare_dataloaders()
                    model.initialize_model(round_args, self.device)
                    model.setup_trainer(self.n_epoch, round_optimizer_settings)
                    model.train_model()
                    model.generate_preds(use_best=self.early_stop)
                    metrics[i] = {k: model.training_metrics[k] for k in ['train_loss', 'val_loss', 'test_loss']}
                    result_preds = copy.deepcopy(model.preds)
                    new_col = f'{sampling_type}{i}'
                    new_cols.append(new_col)
                    for split in result_preds:
                        result_preds[split].rename(columns={'preds': new_col}, inplace=True)
                        round_data[sampling_type][split] = pd.merge(round_data[sampling_type][split], result_preds[split])

                self.training_metrics[round][sampling_type] = metrics
                
                # generate predictions from linear model to benchmark model predictions
                linear_model = LinearRegression()
                feature_cols = [col for col in round_data[sampling_type]['train'].columns if re.search(r'id\d+_score', col)]
                train_features = round_data[sampling_type]['train'][feature_cols]
                linear_model.fit(train_features, round_data[sampling_type]['train']['comb_score'])
                for split in round_data[sampling_type]:
                    split_linear_preds = linear_model.predict(round_data[sampling_type][split][feature_cols])
                    round_data[sampling_type][split].loc[:, 'linear'] = split_linear_preds
                new_cols.append('linear')

                for split in round_data[sampling_type]:
                    pred_stats = ActiveLearner._calculate_pred_stats(round_data[sampling_type][split][new_cols], categories=['mean', 'std', 'mean+std', 'residual', 'residual+std'])
                    round_data[sampling_type][split] = pd.concat([round_data[sampling_type][split], pred_stats], axis=1)
                    if split == 'overall':
                        round_data[sampling_type]['overall'] = ActiveLearner._overwrite_data_with_measured(round_data[sampling_type]['overall'], round_data[sampling_type]['train'])
                    
                logger.info(f'Finished ensemble training for {sampling_type} selection in round {round}')

            self.preds[round] = round_data

    def plot_loss_curves(self, round, sampling_type, ensemble, ax=None):
        """
        Plot the training and validation loss curves from the given `round`, `sampling_type`, `split`, and `ensemble`. 
        
        TODO: If any of these are `None`, then plot results from all categories within the given `None` field.

        Args:
            round (int): round of sampling to plot results for (must be less than `n_round`)
            sampling_type (str): which sampling type to use for plots (options are 'random', 'active')
            ensemble (int): which value in ensemble to use for plots (must be less than `n_ensemble`)
            ax (matplotlib.axes.Axes): 
        
        Returns:
            plt (matplotlib.pyplot or list[matplotlib.pyplot]): plot object containing requested plot
        """

        """
        if round == None:
            round = range(self.n_round)

        if sampling_type == None:
            sampling_type = ['random', 'active']

        if split == None:
            split = ['train', 'val', 'test']

        if ensemble == None:
            ensemble = range(self.n_ensemble)
        """

        if len(self.preds) == 0:
            raise RuntimeError('Need to call run_active_learning() function before plotting results.')

        data = self.training_metrics[round][sampling_type][ensemble]
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        sns.lineplot(np.log(data['train_loss']), label = 'Train Loss', ax = ax)
        sns.lineplot(np.log(data['val_loss']), label = 'Val Loss', ax = ax)
        ax.set_ylabel('log(Loss)')
        ax.set_xlabel('Epoch')
        ax.set_title(f'Round: {round}, Sampling: {sampling_type}, Ensemble: {ensemble}')
        
        return fig
    
    def plot_preds(self, round, sampling_type, split, ensemble, ax=None):
        """
        Plot targets vs predictions for the model `ensemble` from `sampling_type` in `round` and data `split`.

        Args:
            round (int): round of sampling to plot results for (must be less than `n_round`)
            sampling_type (str): which sampling type to use for plots (options are 'random', 'active')
            split (str): which data split to use for plots (options are 'train', 'val', 'test')
            ensemble (int): which value in ensemble to use for plots (must be less than `n_ensemble`)
            ax (matplotlib.axes.Axes): 
        """

        if len(self.preds) == 0:
            raise RuntimeError('Need to call run_active_learning() function before plotting results.')

        data = self.preds[round][sampling_type][split]
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        sns.scatterplot(data, x='comb_score', y=f'{sampling_type}{ensemble}', ax=ax)
        ax.set_xlabel('Targets')
        ax.set_ylabel('Preds')
        ax.set_title(f'Targets vs Preds, Round: {round}, Sampling: {sampling_type}, Split: {split}, Ensemble: {ensemble}')
        
        return fig
    
    def calculate_aggregate_metrics(self, return_value=False):
        """
        Calculate aggregated metrics across ensemble members.

        Args:
            return_value (bool): should aggregated results be returned? Default: False

        Returns:
            results (dict): dictionary of metrics including mean square error (mse) of val and test data
                at each round, as well as TPR of val and test data at each round
        """

        if len(self.preds) == 0:
            raise RuntimeError('Need to call run_active_learning() function between calculating aggregate metrics.')
    
        aggregate_metrics = {}
        for round in range(self.n_round):
            round_metrics = {}
            for sampling_type in self.preds[round]:
                sampling_metrics = {}
                split_data = self.preds[round][sampling_type]
                for split in split_data:
                    data = split_data[split]
                    mse = np.sum((data['mean'] - data['comb_score'])**2) / data.shape[0]
                    if split == 'overall':
                        tpr = find_top_n_perturbations(df = data, 
                                                       pred_keys = 'measured+pred', 
                                                       pheno_key = 'comb_score', 
                                                       min = 10, max = 200, by = 5, 
                                                       ascending = True,
                                                       plot = False)
                        tpr = tpr.rename(columns={'measured+pred': 'tpr'})
                    else:
                        tpr = find_top_n_perturbations(df = data, 
                                                       pred_keys = 'mean', 
                                                       pheno_key = 'comb_score', 
                                                       min = 10, max = 200, by = 5, 
                                                       ascending = True,
                                                       plot = False)
                        tpr = tpr.rename(columns={'mean': 'tpr'})
                    split_metrics = {'mse': mse, 'tpr': tpr}
                    sampling_metrics[split] = split_metrics
                
                """
                # find mse and tpr for overall data set
                split_data_join = pd.concat([split_data[split] for split in split_data], axis=0)
                mse = np.sum((data['mean'] - data['comb_score'])**2) / data.shape[0]
                tpr = find_top_n_perturbations(df = split_data_join, 
                                               pred_keys = 'mean',
                                               pheno_key = 'comb_score', 
                                               min = 10, max = 200, by = 5, 
                                               ascending = True,
                                               plot = False)
                sampling_metrics['overall'] = {'mse': mse, 'tpr': tpr}
                """
                round_metrics[sampling_type] = sampling_metrics
            aggregate_metrics[round] = round_metrics

        self.aggregate_metrics = aggregate_metrics

        if return_value:
            return self.aggregate_metrics
        return None
    
    def save_results(self, save_dir, file_prefix=''):
        """
        Save the `results`, `training_metrics`, and `aggregate_metrics` from the ActiveLearner instance as
        a pickle file.
        
        Args:
            save_dir (str): directory for storing results
            file_prefix (str, optional): prefix for file name to store data
        """

        # in theory these are same condition, so if one of them fails but not the other then there is a bug somewhere
        if len(self.preds) == 0 or len(self.training_metrics) == 0: 
            raise RuntimeError('Must call `run_active_learning()` function to generate results before saving them to file.')
        
        if len(self.aggregate_metrics) == 0:
            raise RuntimeError('Must call `calculate_aggregate_metrics()` function to aggregate metrics before saving them to file.')
        
        if not os.path.isdir(save_dir):
            raise ValueError('Must supply a valid directory to save results.')
        
        combined_results = {'results': self.preds,
                            'training_metrics': self.training_metrics,
                            'aggregate_metrics': self.aggregate_metrics}
        
        file_name = f'active_learner_method{self.method}_seed{self.seed}_ens{self.n_ensemble}_nround{self.n_round}.pkl'
        if file_prefix == '':
            save_path = os.path.join(save_dir, file_name)
        else: 
            save_path = os.path.join(save_dir, f'{file_prefix}_{file_name}')

        with open(save_path, 'wb') as f:
            pickle.dump(combined_results, f)
        
    

class ActiveLearnerReplicates:
    """
    Class for generating multiple Active Learning instances, based on a reference `active_learner` object. 
    Warning: this class will overwrite internal state of `active_learner` object passed as initialization argument.

    Args:
        n_rep (int): number of active learning replicates to generate
        overall_seed (int): seed to use to initialize other seeds for active learning
        active_learner (ActiveLearner): pre-initialized active learning object used for active learning. This class
            generates replicates of the provided active_learner object, changing seeds for each run
        save_prefix (str, optional): prefix to store ActiveLearner results
        save_dir (str, optional): directory to store ActiveLearner results
    """
    def __init__(self, n_rep, overall_seed, active_learner, save_prefix = None, save_dir = None):
        self.n_rep = n_rep
        self.overall_seed = overall_seed
        self.seeds = np.random.default_rng(seed=self.overall_seed).integers(low=0, high=1000, size=self.n_rep)
        self.active_learner = copy.deepcopy(active_learner)
        self.results = {}
        self.aggregated_metrics_across_seeds = {}

        if bool(save_prefix) ^ bool(save_dir):
            raise ValueError('Both `save_prefix` and `save_dir` must be specified, or neither should be specified.')
        
        if save_prefix and save_dir:
            self.save_prefix = save_prefix
            self.save_dir = save_dir

    def set_method(self, method, method_min):
        """
        Set method for which ensemble statistics are used for training active learning. Options are 'mean', 'std', 
        'mean+std', 'residual', 'residual+std', 'leverage'

        Args:
            method (str): ensemble method to use for active learning, options are 'mean', 'std', 'mean+std', 
                'residual', 'residual+std', 'leverage'
            method_min (bool): should the metric chosen in `method` be minimized?
        """
        self.method = method
        self.method_min = method_min
        self.active_learner.update_method(self.method, self.method_min)

    def _run_active_learning_for_seed(self, active_learner, seed):
        """
        Run ActiveLearner `active_learner` for `seed`. Shuffle data, run active learning, and calculate 
        aggregated metrics. If `self.save_dir` and `self.save_prefix` specified, save the results of active learning.

        Args:
            active_learner (ActiveLearner): instance of ActiveLearner class
            seed (int): seed for shuffling data

        Returns:
            result_dict (dict): dictionary containing active_learner.preds, active_learner.training_metrics, and
                active_learner.aggregate_metrics as 'preds', 'training_metrics', and 'aggregate_metrics' keys.
        """
        logger.info(f'Running active learning module for seed {seed}...')
        active_learner.set_seed(seed)
        active_learner.shuffle_data()
        active_learner.run_active_learning()
        active_learner.calculate_aggregate_metrics()
        
        if self.save_dir:
            active_learner.save_results(self.save_dir, self.save_prefix)

        result_dict = {'preds': active_learner.preds,
                       'training_metrics': active_learner.training_metrics,
                       'aggregate_metrics': active_learner.aggregate_metrics}
        return result_dict
    
    def run_replicates(self, parallel=False, return_value=False):
        """
        Run ActiveLearner class for `self.n_rep`, with different seeds.

        Args:
            parallel (bool): Should torch.multiprocess be used to parallelize running the models? Default: False
            return_value (bool): Should the results of the run be returned? Default: False

        Returns: 
            results (dict): dictionary of results, containing information from `results`, `training_metrics`, and
                `aggregate_metrics` for each model
        """
        method_results = {}
        if parallel:
            mp.set_start_method('spawn', force=True)
            active_learners = [copy.deepcopy(self.active_learner) for _ in self.seeds]
            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = [pool.apply_async(self._run_active_learning_for_seed, args=(active_learner, seed)) for active_learner, seed in zip(active_learners, self.seeds)]
                pool.close()
                pool.join()

            for seed, result in zip(self.seeds, results):
                method_results[seed] = result.get()
        
        else:
            for seed in self.seeds:
                active_learner = self.active_learner
                method_results[seed] = self._run_active_learning_for_seed(active_learner, seed)
                
        self.results[self.method] = method_results

        if return_value:
            return self.results
        return None
        
        
    def aggregate_replicate_metrics(self, return_value=False):
        """
        Aggregate metrics across different seeds used in replicates of `active_learner`. Collect metrics across all rounds, 
        for splits of interest.

        Args:
            # metrics (str or list(str)): metrics to collect, e.g. mean square error and true positive rate. Default: ['mse', 'tpr']
            # splits (str or list(str)): data splits to collect metrics for. Default: ['val', 'test']
            return_value (bool): should the data aggregated by returned?

        Returns:
            aggregated_results (dict): dictionary of data frames for metrics of interest. Primary key of dict are the entries of `metrics`. Each primary key
                holds additional dicts, with subkeys from `splits`. Each entry of these subdictionaries are data frames, with columns for 'Round', 'Score'
                based on each `metric` of interest.
        """
        if len(self.results) == 0:
            raise RuntimeError('Need to call `run_active_learner()` function before aggregating results across seeds.')

        """
        before this series of loops, data is in format:
        self.results
            - seed
                - 'aggregate_metrics'
                    - round
                        - sampling_type
                            - split
                                - metric
        
        we want to reorganize data into this format:
        results
            - metric
                - split
                    dataframes containing averaged metrics across seeds per round for each sampling type
        """

        if len(self.aggregated_metrics_across_seeds) > 0 and self.method in self.aggregated_metrics_across_seeds['mse']['train']['Method'].values:
            raise RuntimeError(f'Results for method "{self.method}" have already been aggregated.')
        
        sampling_types = ['random', 'active']
        metrics = ['mse', 'tpr']
        splits = ['train', 'val', 'test', 'overall']
        aggregate_metrics = {}
        for metric in metrics:
            metric_results = {}
            for split in splits:
                split_results = {}
                for sampling_type in sampling_types:
                    sampling_results = {}    
                    for round in range(self.active_learner.n_round):
                        round_results = {}
                        for seed in self.seeds:
                            sampling_data = copy.deepcopy(self.results[self.method][seed]['aggregate_metrics'][round])
                            if sampling_type in sampling_data: # need to check this since in round 0, only 'random' selected is used
                                round_result = sampling_data[sampling_type][split][metric]
                            else: # fill Round 0 active learning results with NAs in shape of desired result
                                round_result = sampling_data[list(sampling_data.keys())[0]][split][metric]
                                if metric == 'tpr':
                                    round_result.loc[:, 'tpr'] = np.nan
                                else:
                                    round_result = np.nan

                            if metric == 'tpr':
                                round_result.rename(columns={'tpr': seed}, inplace=True)

                            # first, keep results as dictionary
                            round_results[seed] = round_result 
                        
                        # next, convert dictionary of results into data frame (concat by column if 'tpr', stack by row if anything else, e.g. 'mse')
                        if metric == 'tpr':
                            round_results = pd.concat(round_results.values(), axis=1)
                            round_results = round_results.loc[:, ~round_results.columns.duplicated()]
                        else:
                            round_results = pd.DataFrame({'seed': round_results.keys(), metric: round_results.values()})
                        round_results.loc[:, 'Round'] = round

                        # filter any NAs, e.g. from Round 0 active learning
                        round_results = round_results.dropna()

                        sampling_results[round] = round_results

                    # stack dataframes from each round
                    sampling_results = pd.concat(sampling_results.values(), axis=0)
                    sampling_results.loc[:, 'Sampling'] = sampling_type

                    split_results[sampling_type] = sampling_results

                # stack dataframes from each sampling type
                split_results = pd.concat(split_results.values(), axis=0)
                split_results.loc[:, 'Method'] = self.method # add label about which ensemble method was used for training these models
                metric_results[split] = split_results
            aggregate_metrics[metric] = metric_results
        
        if len(self.aggregated_metrics_across_seeds) == 0:
            self.aggregated_metrics_across_seeds = aggregate_metrics
        else: 
            # if self.aggregated_metrics_across_seeds is already populated, then we are adding new results
            # from different ensembling method
            for metric in self.aggregated_metrics_across_seeds:
                for split in self.aggregated_metrics_across_seeds[metric]:
                    prev_split_results = self.aggregated_metrics_across_seeds[metric][split]
                    new_split_results = aggregate_metrics[metric][split]
                    split_results = pd.concat([prev_split_results, new_split_results], axis=0)
                    self.aggregated_metrics_across_seeds[metric][split] = split_results

        if return_value:
            return self.aggregated_metrics_across_seeds
        
        return None

    def save_aggregated_results(self, name, dir = None):
        """
        Save results from `self.aggregated_metrics_across_seeds` to `dir` with prefix `name` as a pickle file.
        If `dir` is not specified, check if `self.save_dir` is specified, and use it if so.

        Args:
            name (str): prefix for name of file to save
            dir (str): directory to store pickle file
        """
        if dir is not None and self.save_dir is not None:
            raise RuntimeWarning('Since `dir` is specified, we will not save to `self.save_dir` specified during ActiveLearningReplicates initialization.')
        
        if dir is None:
            dir = self.save_dir

        save_name = f'{name}_aggmetrics_start.pkl'
        if self.save_prefix:
            save_name = f'{self.save_prefix}_{save_name}'
        
        if not os.path.isdir(dir):
            raise ValueError('`dir` must be a valid directory to store results')
        with open(os.path.join(dir, save_name), 'wb') as f:
            pickle.dump(self.aggregated_metrics_across_seeds, f)

        return None

    def plot_aggregated_results(self, metrics, splits, methods, return_value=False, max_round=None, orientation = 'vertical', max_tpr=None, label_map=None):
        """
        Generate plots for requested `metrics` from data trained using `methods` in each of the requested data 
        `splits` collected across replicates.

        Args:
            metrics (str or list(str)): which metrics should be plotted? Options are 'mse', 'tpr'
            splits (str or list(str)): which splits should be plotted? Options are 'train', 'val', 'test', 'overall'
            methods (str or list(str)): which methods should be plotted? Options are 'mean', 'std', 'mean+std', 
                'residual', 'residual+std', 'leverage'
            return_value (bool): should plots generated be returned?
            max_round (int, optional): maximum round to plot results up to
            orientation (str, optional): which way should TPR plots be oriented? Default: 'vertical'. Options are ['vertical', 'horizontal']
            max_tpr (int, optional): should there be a x-axis threshold for the TPR plots?
            label_map (dict, optional): dictionary for mapping method labels to other (e.g. human-readable) text
        """

        if isinstance(metrics, str):
            metrics = [metrics]

        if isinstance(splits, str):
            splits = [splits]

        if isinstance(methods, str):
            methods = [methods]

        figs = []

        if max_tpr is not None and 'tpr' not in metrics:
            raise ValueError('Cannot specify `max_tpr` if "tpr" is not one of the `metrics` to plot')

        if orientation not in ['vertical', 'horizontal']:
            raise ValueError('`orientation` must be either in ["vertical", "horizontal"]')
        
        n_round = self.active_learner.n_round
        if max_round is not None: 
            if (max_round < 0 or max_round > n_round):
                raise ValueError('max_round must be > 0 and < n_round within ActiveLearner')
            else:
                n_round = max_round

        method_colors = { # colors for plotting measuerments of different `methods`
            'random': '#1f77b4',  # blue
            'mean': '#2ca02c',     # green
            'std': '#d62728',         # red
            'mean+std': '#ff7f0e',     # orange
            'residual': '#9467bd',       # purple
            'residual+std': '#8c564b'          # brown
        }

        for metric in metrics:
            for split in splits:
                if metric == 'mse':
                    data = copy.deepcopy(self.aggregated_metrics_across_seeds[metric][split])
                    data = data.loc[data['Round'] < n_round, :]
                    mse_plot_data = []
                    for method in methods:
                        plot_data = data[['mse', 'Round']].loc[(data['Sampling'] == 'active') & (data['Method'] == method), :]
                        plot_data.loc[:, 'Method'] = method
                        mse_plot_data.append(plot_data)

                    plot_data = data[['mse', 'Round']].loc[(data['Sampling'] == 'random') & (data['Method'] == method), :]
                    plot_data.loc[:, 'Method'] = 'random'
                    mse_plot_data.append(plot_data)
                    mse_plot_data = pd.concat(mse_plot_data, axis=0)

                    fig, ax = plt.subplots(1, 1)
                    sns.boxplot(mse_plot_data, x='Round', y=metric, hue='Method', palette=method_colors, fill=False, ax=ax)
                    ax.set_ylabel('MSE')
                    ax.legend(fontsize=9, frameon=False)
                    fig.suptitle(f'Active Learning Error, Split: {split}')
                    figs.append(fig)
                    if label_map:
                        for text in ax.get_legend().get_texts():
                            old_label = text.get_text()
                            if old_label in label_map:
                                text.set_text(label_map[old_label])
                    if not return_value:
                        plt.show()
                elif metric == 'tpr':
                    min_ylim = np.inf
                    max_ylim = -np.inf
                    data = copy.deepcopy(self.aggregated_metrics_across_seeds[metric][split])
                    data = data.loc[data['Round'] < n_round, :]
                    if max_tpr:
                        data = data.loc[data['n_preds'] <= max_tpr, :]
                    data_stacked = pd.melt(data, id_vars=['Round', 'n_preds', 'Method', 'Sampling'], value_vars=self.seeds, var_name='seed', value_name='tpr')

                    if orientation == 'vertical':
                        fig, axs = plt.subplots(n_round, 1, figsize=(3, (n_round*2.25)+2))
                    elif orientation == 'horizontal':
                        fig, axs = plt.subplots(1, n_round, figsize=((n_round*3)+2, 4))

                    for idx in range(n_round):
                        # random data should be the same regardless of method, so choose random data from last method to plot
                        data_plot = data_stacked[(data_stacked['Round'] == idx) & (data_stacked['Method'] == methods[0]) & (data_stacked['Sampling'] == 'random')]
                        sns.lineplot(data_plot, x='n_preds', y='tpr', errorbar='se', label='random', color=method_colors['random'], ax=axs[idx])
                        
                        if idx > 0:
                            for method in methods:
                                if method == 'overall':
                                    data_plot = copy.deepcopy(data_stacked)
                                    data_plot[[(data_stacked['Round'] == idx) & (data_stacked['Method'] == method) & (data_stacked['Sampling'] == 'active')]] \
                                        = data_plot[[(data_stacked['Round'] == idx) & (data_stacked['Method'] == method) & (data_stacked['Sampling'] == 'active')]]
                                data_plot = data_stacked[(data_stacked['Round'] == idx) & (data_stacked['Method'] == method) & (data_stacked['Sampling'] == 'active')]
                                sns.lineplot(data_plot, x='n_preds', y='tpr', errorbar='se', label=method, color=method_colors[method], ax=axs[idx])

                        ylims = axs[idx].get_ylim()
                        if ylims[0] < min_ylim:
                            min_ylim = ylims[0]
                        if ylims[1] > max_ylim:
                            max_ylim = ylims[1]

                        if orientation == 'horizontal':
                            axs[idx].set_title(f'Round {idx}')
                            axs[idx].legend().remove()
                            axs[idx].set_xlabel('Top N Perturbations')
                            axs[idx].set_ylabel('')
                        elif orientation == 'vertical':
                            axs[idx].set_title(f'')
                            axs[idx].legend().remove()
                            axs[idx].set_xlabel('')
                            axs[idx].set_ylabel('Fraction Discovered')
                            
                    if orientation == 'horizontal':
                        axs[0].set_ylabel('Fraction Discovered')
                    elif orientation == 'vertical':
                        axs[-1].set_xlabel('Top N Perturbations')
                        
                    for ax in axs:
                        ax.set_ylim(min_ylim, max_ylim)
                        if max_tpr is not None:
                            ax.set_xticks([(i+1)*int(max_tpr/4) for i in range(4)])

                    axs[-1].legend(fontsize=9, frameon=False)
                    if label_map:
                        for text in axs[-1].get_legend().get_texts():
                            old_label = text.get_text()
                            if old_label in label_map:
                                text.set_text(label_map[old_label])

                    fig.suptitle(f'Active Learning True Positive Rate, Split: {split}')
                    fig.tight_layout()
                    figs.append(fig)
                    if not return_value:
                        plt.show()
        
        if return_value:
            return figs
        else:
            return None