# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 20:37:29 2023

@author: User
"""

import torch
import pandas as pd
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import scale, minmax_scale
from collections import defaultdict
from itertools import chain
from .data import make_dir, split_data

metrics = ['spearmanr', 'ndcg', 'topk_pr']
group_names = ['single_local', 'single_cross', 'single_rest', '2_sites', '3_sites', '4_sites',
               'multi_combined', 'multi_cross', 'multi_rest', 'all_rest']

def pairwise_ranking_loss(input1, input2, label1, label2, fn='hinge', margin=1.0):
    target = torch.where(label1 > label2, 1.0, -1.0)
    
    if fn == 'hinge':
        loss = F.margin_ranking_loss(input1, input2, target, margin=margin)
    elif fn == 'exp':
        loss = torch.exp(- target * (input1 - input2)).mean()
    elif fn == 'log':
        loss = torch.log(1 + torch.exp(- target * (input1 - input2))).mean()
    else:
        raise ValueError('Unknown pairwise ranking function: ' + fn)
    return loss

def listwise_ranking_loss(predicts, targets):
    ''' ListMLE loss '''
    indices = targets.sort(descending=True, dim=-1).indices
    predicts = torch.gather(predicts, dim=1, index=indices)

    cumsums = predicts.exp().flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
    loss = torch.log(cumsums + 1e-10) - predicts
    return loss.sum(dim=1).mean()

def compute_scores(predicts, targets, labels, k=30):
    report = dict(size=len(predicts))
    report['spearmanr'] = spearmanr(predicts, targets).statistic
    # std_tgts = scale([targets], axis=1)
    std_tgts = minmax_scale([targets], (0, 5), axis=1)
    report['ndcg'] = ndcg_score(std_tgts, [predicts])
    k = min(len(predicts), k)
    predicts, labels = torch.tensor(predicts), torch.tensor(labels)
    indices = predicts.topk(k).indices
    report['topk_pr'] = torch.count_nonzero(labels[indices]).item() / k
    return report

def group_scores(train_df, pred_df, test_df, k=30):
    train_sites = set(chain(*train_df.index.str.split(':')))
    train_pos = set(chain(*train_df['positions']))
    
    groups = defaultdict(list)
    for mutant, row in test_df.iterrows():
        n_sites = len(row['positions'])
        if n_sites == 1:
            groups['single_rest'].append(mutant)
            if row['positions'][0] in train_pos:
                groups['single_local'].append(mutant)
            else:
                groups['single_cross'].append(mutant)
        else:
            groups['multi_rest'].append(mutant)
            if n_sites <= 4:
                groups[f'{n_sites}_sites'].append(mutant)
            sites = set(mutant.split(':'))
            if sites.issubset(train_sites):
                groups['multi_combined'].append(mutant)
            elif sites.isdisjoint(train_sites):
                groups['multi_cross'].append(mutant)
    
    names, report = [], []
    for name in group_names:
        indices = groups.get(name)
        if not indices or len(indices) < 3:
            continue
        names.append(name)
        report.append(compute_scores(pred_df.loc[indices].to_list(),
                                     test_df.loc[indices, 'DMS_score'].to_list(),
                                     test_df.loc[indices, 'DMS_score_bin'].to_list(),
                                     k))
    report.append(compute_scores(pred_df.loc[test_df.index].to_list(),
                                 test_df['DMS_score'].to_list(),
                                 test_df['DMS_score_bin'].to_list(),
                                 k))
    report = pd.DataFrame(report, index=names + ['all_rest'])
    return report, groups

def summarize_scores(score_groups, save_path=None):
    summary = {}
    for metric in metrics:
        reports = [{name: row[metric] for name, row in groups.iterrows()} \
                       for groups in score_groups.values()]
        reports = pd.DataFrame(reports, index=score_groups.keys())
        reports.loc['average'] = reports.mean()
        summary[metric] = reports[[name for name in group_names if name in reports.columns]]
    
    if save_path is not None:
        make_dir(save_path)
        torch.save(summary, save_path)
    return summary

def get_best_augment(proteins, raw_dir, train_size, n_sites, candidates, metric='spearmanr'):
    results = {}
    for protein in proteins:
        if len(protein['df']) < train_size:
            continue
        train, _ = split_data(protein, train_size, n_sites=n_sites)
        raw_data = pd.read_csv(f'{raw_dir}/{protein["name"]}.csv', index_col='mutant')
        
        best = float('-inf')
        for name in candidates:
            predicts = raw_data.loc[train.index, name]
            score = compute_scores(predicts['df'].to_list(),
                                   train['df']['DMS_score'].to_list(),
                                   train['df']['DMS_score_bin'].to_list())[metric]
            if score > best:
                selected = name
                best = score
        results[protein['name']] = selected
    return results
