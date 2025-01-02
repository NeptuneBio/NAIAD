import warnings
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import importlib
import sys
import tqdm


def load_naiad_data(data_path, control_treatment_name='negative', shuffle_treatments=True):
    """
    Generate a Pandas DataFrame from a combinatorial treatment (perturbation or stimulation) dataset. 
    Resulting data frame will list each combination of treatments tested in the combinatorial screen, and
    have information about phenotype score for each combinatorial perturbation in the provided dataset. 
    Also include columns for the phenotypic measurements from the single-treatment experiment for each 
    treatment within a combination.

    Args:
        data_path (str): path to data file containing phenotypic readouts of 
            combinatorial perturbation / stimulation experiments
        reorganize_data (bool): whether to reorganize the data to include single-gene effects in
            new columns (default is True)
        control_treatment_name (str, Optional): name of control ID used in the experiment (default is 'negative').
            Only necessary if `reorganize_data` is True.
        shuffle_treatments (bool): whether to shuffle ID and corresponding score columns for each row in the DataFrame.
    
    Returns:
        score_data (pd.DataFrame): data frame of genetic perturbation data containing
            the following columns:
              'id1': first ID (by alphabetical order) in each combinatorial perturbation
              'id2': second ID (by alphabetical order) in each combinatorial perturbation
              ...
              'idN': Nth ID (by alphabetical order) in each combinatorial perturbation
              'comb_score': phenotype score for each combinatorial perturbation
              'id1_score': phenotype score for corresponding single-id in `id1` column
              'id2_score': phenotype score for corresponding single-id in `id2` column
              ...
              'idN_score': phenotype score for corresponding single-id in `idN` column
        Note: data will be returned as numpy.float32 values, rather than default numpy.float64.
            This is done to match the default format of PyTorch parameters.
    """

    phenotype_df = load_phenotype_df(data_path)
    naiad_data = reorganize_single_treatment_effects(phenotype_df, control_id=control_treatment_name)
    if shuffle_treatments:
        naiad_data = shuffle_ids(naiad_data)
    return naiad_data
    
def load_phenotype_df(data_path):
    """
    Create a Pandas DataFrame for a bulk perturbation or stimulation experiment of interest. 

    We perform the following operations on the data: 
    1) within each row of the data (corresponding to each combination), sort the treatment conditions in 
        alphabetical order
    2) average all combinations containing the same set of treatments

    Args:
        data_path (str): path to data file containing phenotypic readouts of 
            combinatorial treatments
        
    Returns:
        comb_data (pd.DataFrame): data frame of genetic perturbation / chemical stimulation treatment data containing
            the following columns:
              'id1': first ID (by alphabetical order) in each combinatorial treatment
              'id2': second ID (by alphabetical order) in each combinatorial treatment
              ...
              'idN': Nth ID (by alphabetical order) in each combinatorial treatment
              'score': phenotype score for each combinatorial perturbation
        Note: data will be returned as numpy.float32 values, rather than default numpy.float64.
            This is done to match the default format of PyTorch parameters.
    """

    comb_data = pd.read_csv(data_path)
    
    id_cols = [col for col in comb_data.columns if 'id' in col]
    all_cols = id_cols + ['score']
    id_names = comb_data[id_cols]
    # sort ID names in each row
    id_names = id_names.apply(lambda row: pd.Series(sorted(row)), axis=1)
    comb_data.loc[:, id_cols] = id_names.values
    comb_data = comb_data[all_cols]

    comb_data = comb_data.groupby(id_cols) \
                         .mean() \
                         .reset_index()
    comb_data['score'] = comb_data['score'].astype(np.float32)

    return comb_data

def reorganize_single_treatment_effects(comb_data, control_id='negative'):
    """
    Filter unprocessed data frame of combinatorial phenotype assay data. 
    Perform the following opertions:
    - remove all single-treatment perturbations from dataset
    - add single-treatment phenotypes as new columns in combinatorial perturbation data frame
        For genetic perturbations, this corresponds to the condition of double-guides targeting the 
        same gene

    Args:
        comb_data (pd.DataFrame): data frame containing combinatorial pheno results.
            Must contain columns for `id1` and `id2` names for each
            combinatorial treatment, as well as phenotype results for each
            treatment
        control_id (str): name of control treatment ID used in the experiment 
            (default is 'negative')

    Returns:
        comb_data (pd.DataFrame): data frame containing combinatorial pheno
            results. Contains all columns as `comb_data`, as well as additional columns
            for single-treatment perturbation phenotypes for the individual effect of
            the treatments in each row.
    """
    id_col_names = [x for x in comb_data.columns if 'id' in x]
    id_cols = comb_data[id_col_names]
    single_id_row_filter = (id_cols !=  control_id).sum(axis=1) == 1
    single_id_rows = id_cols[single_id_row_filter]
    single_ids = single_id_rows[single_id_rows != control_id].values.flatten()
    single_ids = [x for x in single_ids if isinstance(x, str)]
    single_id_scores = comb_data.loc[single_id_row_filter, 'score'].values

    comb_data_single = pd.DataFrame({'id': single_ids, 
                                     'score': single_id_scores})

    # filter single-pert examples from data
    comb_data = comb_data[~single_id_row_filter]
    
    # append negative guide score
    all_neg_row_filter = (comb_data == control_id).sum(axis=1) == len(id_col_names)
    neg_score = comb_data.loc[all_neg_row_filter, 'score'].values
    neg_data = pd.DataFrame({'id': control_id, 'score': neg_score})
    comb_data_single = pd.concat([comb_data_single, neg_data], ignore_index=True)

    # filter rows of exclusively non-targeting
    comb_data = comb_data[~all_neg_row_filter]
    
    comb_data = comb_data.rename(columns={'score': 'comb_score'})
    id_score_cols = []
    for i, col in enumerate(id_col_names):
        comb_data = pd \
            .merge(comb_data, comb_data_single, left_on=col, right_on='id', how='left') \
            .drop(columns='id') \
            .rename(columns={'score': f'id{i+1}_score'})
        id_score_cols.append(f'id{i+1}_score') 

    comb_data = comb_data[id_col_names + ['comb_score'] + id_score_cols]

    # filter rows with missing single-treatment effects
    comb_data = comb_data.dropna()

    return comb_data

def shuffle_ids(data, id_prefix='id', score_suffix='_score'):
    """
    Shuffle ID and corresponding score columns for each row in a DataFrame.

    Parameters:
        data (pd.DataFrame): Input DataFrame containing ID and score columns.
        id_prefix (str): Prefix for ID columns (default is 'id').
        score_suffix (str): Suffix for score columns corresponding to ID columns.

    Returns:
        pd.DataFrame: DataFrame with shuffled ID and score columns.
    """
    # Identify gene and score columns
    id_cols = [col for col in data.columns if re.match(rf"^{id_prefix}\d+$", col)]
    score_cols = [col + score_suffix for col in id_cols]
    ids_array = data[id_cols].values
    scores_array = data[score_cols].values

    # Generate shuffled indices for each row
    shuffled_indices = np.array([np.random.permutation(len(id_cols)) for _ in range(len(data))])

    # Shuffle the arrays using the generated indices
    shuffled_genes = np.take_along_axis(ids_array, shuffled_indices, axis=1)
    shuffled_scores = np.take_along_axis(scores_array, shuffled_indices, axis=1)
    data[id_cols] = shuffled_genes
    data[score_cols] = shuffled_scores

    return data

def split_data(treatment_data, n_train, n_val, n_test):
    """
    Split data into train, val, test sets based on train_frac, val_frac, and test_frac.

    Args:
        pert_data (pd.DataFrame): data frame containing combinatorial perturbation data
        n_train (Union[int, float]): either number of data points to sample for training split, or if `n_train` < 1, the fraction of data to assign to training set
        n_val (Union[int, float]): either number of data points to sample for validation split, or if `n_val` < 1, the fraction of data to assign to validation set
        n_test (Union[int, float]): either number of data points to sample for test split, or if `n_test` < 1, the fraction of data to assign to test set
    """
    if n_train < 1:
        n_train = int(n_train * treatment_data.shape[0])
    if n_val < 1:
        n_val = int(n_val * treatment_data.shape[0])
    if n_test < 1:
        n_test = int(n_test * treatment_data.shape[0])

    if (n_train + n_val + n_test) > treatment_data.shape[0]:
        raise ValueError('Cannot specify `n_train`, `n_val`, and `n_test` that sum to larger than dataset size.')
    
    train_treatment_data = treatment_data.iloc[:n_train, :]
    val_treatment_data = treatment_data.iloc[-(n_val+n_test):-n_test, :]
    test_treatment_data = treatment_data.iloc[-n_test:, :]

    return {'train': train_treatment_data,
            'val': val_treatment_data,
            'test': test_treatment_data} 

def create_lr_scheduler(optimizer, warmup_steps, total_steps):
    """
    Create linear learning rate scheduler for training. Linearly increase LR ratio
    from 0 to 1 over `warmup_steps`, then decrease LR ratio to 0 over the remaining
    steps to `total_steps`.

    Args:
        optimizer (torch.optim.Optimizer): optimizer for training model
        warmup_steps (int): number of steps to use for warmup
        total_steps (int): total number of steps for training

    Returns:
        scheduler (torch.optim.lr_scheduler.LRScheduler): learning rate scheduler
            for `optimizer`
    """
    if (int(warmup_steps) != warmup_steps):
        warnings.warn('warmup_steps is not an int. Coercing into an int now...', UserWarning)
        warmup_steps = int(warmup_steps)

    if (int(total_steps) != total_steps):
        warnings.warn('total_steps is not an int. Coercing into an int now...', UserWarning)
        total_steps = int(total_steps)

    lr_lambda = lambda step: (step + 1) / warmup_steps if step < warmup_steps else \
                              0.1 + 0.9*((total_steps - step) / (total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler

def find_top_n_perturbations(df, pred_keys, pheno_key, min=10, max=200, by=10, ascending=True, plot=False, ax=None):
    """
    Identify how many of the strongest N perturbations are predicted by model, 
    as the value of N is varied.
    
    More specifically, we find the top N predictions for each
    model and check how many of these predictions are present within the top N strongest
    perturbations (as measured in the experimental data).

    Args:
        df (pd.DataFrame): data frame containing predictions from models of interest, along 
            with experimental measurements
        pred_keys (str or list(str)): column names from `df` corresponding to model predicted
            phenotype values
        pheno_key (str): column name from `df` corresponding to experimentally measured 
            phenotype value
        min (int): minimum number of top perturbations to consider
        max (int): maximum number of top perturbations to consider
        by (int): increment from `min` to `max` number of top perturbations to consider
        ascending (bool): whether strongest perturbations correspond to maximum or minimum values
            in the `pred_keys` columns of `df`
        plot (bool): should we make a plot of the top N perturbations as a function of N?
        ax (matplotlib.pyplot.axes): Axes object to embed plot within, if `plot` is True

    Returns:
        roc_matches (dict): number of hits per top N perturbations
        p (matplotlib.pyplot): pyplot object show fraction of top N strongest perturbations
            predicted by each model (y-axis), as a function of N (x-axis)
    """
    if isinstance(pred_keys, str):
        pred_keys = [pred_keys]

    roc_matches = {k: [] for k in pred_keys}
    n_preds = list(range(min, max, by))

    df = df.sort_values(pheno_key, ascending=ascending)
    if plot and ax is None:
        fig, ax = plt.subplots()

    for k in pred_keys:
        for label_cutoff in n_preds:
            strongest_pheno_labels = np.zeros(df.shape[0])
            strongest_pheno_labels[:label_cutoff] = 1
            strongest_pheno_labels[label_cutoff:] = 0
            df['Gamma Labels'] = strongest_pheno_labels
            strongest_preds_pheno = df.sort_values(k, ascending=ascending)
            n_pos = np.sum(strongest_preds_pheno['Gamma Labels'][:label_cutoff])
            roc_matches[k].append(n_pos / label_cutoff)

        if plot:
            sns.lineplot(x=n_preds, y=roc_matches[k], label=k, errorbar=None, ax=ax)

    roc_matches = pd.DataFrame(roc_matches)
    roc_matches.loc[:, 'n_preds'] = n_preds

    if plot:
        ax.legend(fontsize='8')
        ax.set_title(f'Predicting Strongest N Perturbations')
        ax.set_xlabel('Number of Perturbations and Predictions')
        ax.set_ylabel('Fraction of Perturbations Correctly Predicted')
        return roc_matches, ax
    else:
        return roc_matches


def RunModelsReplicates(model, n_ensemble, output_type = 'loss', model_args=None, device=None, n_epoch=100, seed=1442, verbose=False):
    rep_models = [] 
    rep_output = []
    loop = tqdm.tqdm(range(n_ensemble)) if verbose else range(n_ensemble)
    for i in loop:
        model.set_seed(seed=seed + i)
        model.prepare_data()
        model.initialize_model(device=device, model_args=model_args)
        model.setup_trainer(n_epoch=n_epoch)
        model.train_model()
        if output_type == 'loss':
            losses = model.training_metrics  
            rep_output.append(losses)
        elif output_type == 'intermediate':
            model.run_linear_regression()
            model.generate_intermediate_results(use_best=True) 
            results = model.intermediate_results
            rep_output.append(results)

        rep_models.append(model)

    return rep_models, rep_output


def reload_module(module_name):
    try:
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
            print('reloading module', module_name )
        else:

            importlib.import_module(module_name)
            print('loading module', module_name )
    except ModuleNotFoundError:
        print(f"Module {module_name} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
