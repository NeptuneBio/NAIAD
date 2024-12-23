import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import torch

def load_naiad_data(data_path, control_gene_name='negative', shuffle_gene=True):
    """
    Generate a Pandas DataFrame from a bulk perturbation screen dataset. Resulting
    data frame will list each combination of genes tested in the combinatorial screen, and
    have information about phenotype score for each combinatorial perturbation in the 
    provided dataset. Also include columns for the phenotypic measurements from the 
    single-gene perturbations for each gene within a combination.

    Args:
        data_path (str): path to data file containing phenotypic readouts of 
            combinatorial perturbation experiments
    
    Returns:
        score_data (pd.DataFrame): data frame of genetic perturbation data containing
            the following columns:
              'gene1': first gene (by alphabetical order) in each combinatorial perturbation
              'gene2': second gene (by alphabetical order) in each combinatorial perturbation
              ...
              'geneN': Nth gene (by alphabetical order) in each combinatorial perturbation
              'comb_score': phenotype score for each combinatorial perturbation
              'g1_score': phenotype score for corresponding single-gene in `gene1` column
              'g2_score': phenotype score for corresponding single-gene in `gene2` column
              ...
              'gN_score': phenotype score for corresponding single-gene in `geneN` column
        Note: data will be returned as numpy.float32 values, rather than default numpy.float64.
            This is done to match the default format of PyTorch parameters.
    """
    phenotype_df = load_phenotype_df(data_path)
    naiad_data = reorganize_single_gene_effects(phenotype_df, control_gene_name=control_gene_name)
    if shuffle_gene:
        naiad_data = naiad_data.apply(shuffle_genes, axis=1)
    return naiad_data

def shuffle_genes(row):
    # Identify gene name and score columns
    gene_cols = [col for col in row.index if col.startswith('gene')]
    score_cols = [f"{col}_score" for col in gene_cols]
    score_cols = [s.replace('gene', 'g') for s in score_cols]
    
    genes_and_scores = list(zip(row[gene_cols], row[score_cols]))
    
    # shuffle the genes and scores together
    random.shuffle(genes_and_scores)
    for i, (gene, score) in enumerate(genes_and_scores):
        row[gene_cols[i]] = gene
        row[score_cols[i]] = score
    
    return row
    
def load_phenotype_df(data_path):
    """
    Create a Pandas DataFrame for a bulk perturbation experiment of interest. 

    We perform the following operations on the data: 
    1) within each row of the data (corresponding to each combination), sort the genes in 
        alphabetical order
    2) average all combinations containing the same set of genes

    Args:
        data_path (str): path to data file containing phenotypic readouts of 
            combinatorial perturbation experiments
        
    Returns:
        comb_data (pd.DataFrame): data frame of genetic perturbation data containing
            the following columns:
              'gene1': first gene (by alphabetical order) in each combinatorial perturbation
              'gene2': second gene (by alphabetical order) in each combinatorial perturbation
              ...
              'geneN': Nth gene (by alphabetical order) in each combinatorial perturbation
              'score': phenotype score for each combinatorial perturbation
        Note: data will be returned as numpy.float32 values, rather than default numpy.float64.
            This is done to match the default format of PyTorch parameters.
    """

    comb_data = pd.read_csv(data_path)
    
    gene_cols = [col for col in comb_data.columns if 'gene' in col]
    all_cols = gene_cols + ['score']
    gene_names = comb_data[gene_cols]
    # sort gene names in each row
    gene_names = gene_names.apply(lambda row: pd.Series(sorted(row)), axis=1)
    comb_data.loc[:, gene_cols] = gene_names.values
    comb_data = comb_data[all_cols]

    comb_data = comb_data.groupby(gene_cols) \
                         .mean() \
                         .reset_index()
    comb_data['score'] = comb_data['score'].astype(np.float32)

    return comb_data

def reorganize_single_gene_effects(comb_data, control_gene_name='negative'):
    """
    Filter unprocessed data frame of combinatorial phenotype assay data. 
    Perform the following opertions:
    - remove all single-gene perturbations from dataset
    - add single-gene perturbation (via double-guide targeting single-gene measurements)
        phenotypes as new columns in combinatorial perturbation data frame

    Args:
        comb_data (pd.DataFrame): data frame containing combinatorial pheno results.
            Must contain columns for `gene1` and `gene2` names for each
            combinatorial perturbation, as well as phenotype results for each
            perturbation

    Returns:
        comb_data (pd.DataFrame): data frame containing combinatorial pheno
            results. Contains all columns as `comb_data`, as well as two columns
            for double-guide single-gene perturbation phenotypes for the
            `gene1` and `gene2` for each row (i.e. each combinatorial perturbation).
    """
    gene_col_names = [x for x in comb_data.columns if 'gene' in x]
    gene_cols = comb_data[gene_col_names]
    single_gene_row_filter = (gene_cols !=  control_gene_name).sum(axis=1) == 1
    single_gene_rows = gene_cols[single_gene_row_filter]
    single_genes = single_gene_rows[single_gene_rows != control_gene_name].values.flatten()
    single_genes = [x for x in single_genes if isinstance(x, str)]
    single_gene_scores = comb_data.loc[single_gene_row_filter, 'score'].values

    comb_data_single = pd.DataFrame({'gene': single_genes, 
                                     'score': single_gene_scores})

    # filter single-pert examples from data
    comb_data = comb_data[~single_gene_row_filter]
    
    # append negative guide score
    all_neg_row_filter = (comb_data == control_gene_name).sum(axis=1) == len(gene_col_names)
    neg_score = comb_data.loc[all_neg_row_filter, 'score'].values
    neg_data = pd.DataFrame({'gene': control_gene_name, 'score': neg_score})
    comb_data_single = pd.concat([comb_data_single, neg_data], ignore_index=True)

    # filter rows of exclusively non-targeting
    comb_data = comb_data[~all_neg_row_filter]
    
    comb_data = comb_data.rename(columns={'score': 'comb_score'})
    gene_score_cols = []
    for i, col in enumerate(gene_col_names):
        comb_data = pd \
            .merge(comb_data, comb_data_single, left_on=col, right_on='gene', how='left') \
            .drop(columns='gene') \
            .rename(columns={'score': f'g{i+1}_score'})
        gene_score_cols.append(f'g{i+1}_score') 

    comb_data = comb_data[gene_col_names + ['comb_score'] + gene_score_cols]

    # filter rows with missing single-gene effects
    comb_data = comb_data.dropna()

    return comb_data

def split_data(pert_data, n_train, n_val, n_test):
    """
    Split data into train, val, test sets based on train_frac, val_frac, and test_frac.

    Args:
        pert_data (pd.DataFrame): data frame containing combinatorial perturbation data
        n_train (Union[int, float]): either number of data points to sample for training split, or if `n_train` < 1, the fraction of data to assign to training set
        n_val (Union[int, float]): either number of data points to sample for validation split, or if `n_val` < 1, the fraction of data to assign to validation set
        n_test (Union[int, float]): either number of data points to sample for test split, or if `n_test` < 1, the fraction of data to assign to test set
    """
    if n_train < 1:
        n_train = int(n_train * pert_data.shape[0])
    if n_val < 1:
        n_val = int(n_val * pert_data.shape[0])
    if n_test < 1:
        n_test = int(n_test * pert_data.shape[0])

    if (n_train + n_val + n_test) > pert_data.shape[0]:
        raise ValueError('Cannot specify `n_train`, `n_val`, and `n_test` that sum to larger than dataset size.')
    
    train_pert_data = pert_data.iloc[:n_train, :]
    val_pert_data = pert_data.iloc[-(n_val+n_test):-n_test, :]
    test_pert_data = pert_data.iloc[-n_test:, :]

    return {'train': train_pert_data,
            'val': val_pert_data,
            'test': test_pert_data} 

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