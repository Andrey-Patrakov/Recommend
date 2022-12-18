import numpy as np
import pandas as pd

def hit_rate(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(recommended_list, bought_list)
    hit_rate = int(flags.sum() > 0)  
    
    return hit_rate


def hit_rate_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(recommended_list[:k], bought_list,)   
    hit_rate = int(flags.sum() > 0)
    
    return hit_rate

def precision(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    flags = np.isin(bought_list, recommended_list)
    precision = flags.sum() / len(recommended_list)

    return precision


def precision_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    bought_list = bought_list  # Тут нет [:k] !!
    recommended_list = recommended_list[:k]

    flags = np.isin(bought_list, recommended_list)
    precision = flags.sum() / len(recommended_list)

    return precision


def money_precision_at_k(recommended_list, bought_list, prices_recommended, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    prices_recommended = np.array(prices_recommended)

    bought_list = bought_list  # Тут нет [:k] !!
    recommended_list = recommended_list[:k]
    prices_recommended = prices_recommended[:k]

    flags = np.isin(bought_list, recommended_list)
    precision = (flags*prices_recommended).sum() / prices_recommended.sum()

    return precision

def recall(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(bought_list, recommended_list)
    recall = flags.sum() / len(bought_list)

    return recall

def recall_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]

    flags = np.isin(bought_list, recommended_list)  
    return flags.sum() / len(bought_list)


def money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]
    prices_recommended = np.array(prices_recommended)[:k]
    prices_bought = np.array(prices_bought)

    flags = np.isin(recommended_list, bought_list)    
    return (flags * prices_recommended).sum() / prices_bought.sum()

def ap_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(recommended_list, bought_list)
    if sum(flags) == 0:
        return 0

    sum_ = 0
    for i in range(k):
        if flags[i]:
            p_k = precision_at_k(recommended_list, bought_list, k=i+1)
            sum_ += p_k

    return sum_ / k

def reciprocal_rank_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    recommended_list = recommended_list[:k]
    flags = np.isin(recommended_list, bought_list)
    flag_id = np.nonzero(flags)[0]

    if not len(flag_id):
        return 0

    return 1 / (flag_id[0] + 1)

def MRR(recommended_list, bought_list, k=5):
    result = []
    for i in range(len(recommended_list)):
        result.append(reciprocal_rank_at_k(recommended_list[i], bought_list[i], k))

    return np.mean(result)

def ndcg_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]

    flags_dcg = np.isin(recommended_list, bought_list)

    flags_i_dcg = np.ones(len(bought_list))
    flags_i_dcg = np.hstack((flags_i_dcg, np.zeros(len(flags_dcg)-len(flags_i_dcg))))

    j = np.array([i + 1 for i in range(len(flags_dcg))])

    dcg = sum(flags_dcg / np.log2(j + 1))
    i_dcg = sum(flags_i_dcg / np.log2(j + 1))

    return dcg / i_dcg
