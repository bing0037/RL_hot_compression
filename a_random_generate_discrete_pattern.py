import copy
import random
import numpy as np


#combination based on length, if length equals 2, the result is [12,13,14,15,16,17,.......23,24,25,....]
import numpy as np
import torch


def combination(length,block_size):
    location_number = block_size * block_size
    data = [i for i in range(1, location_number + 1)]
    if length > location_number // 2:
        new_length = location_number - length
    else:
        new_length = length
    result = []
    temp = [0] * new_length
    l = len(data)
    def next_num(li = 0,ni = 0):
        if ni == new_length:
            result.append(copy.copy(temp))
            return
        for lj in range(li, l):
            temp[ni] = data[lj]
            next_num(lj+1,ni+1)
    next_num()
    if length > location_number // 2:
        final_result = []
        for e in result:
            reverse_temp = []
            for e1 in data:
                if e1 not in e:
                    reverse_temp.append(e1)
            final_result.append(reverse_temp)
        return final_result
    return result


#random sample pruning location
def random_choose_pruning_location(pruning_number,block_size,random_number):
    location = block_size * block_size
    data = [i for i in range(location)]
    combi_list = []
    for j in range(random_number):
        pattern_number = random.sample(data, pruning_number)
        combi_list.append(pattern_number)
    return combi_list


#build pattern space based on pruning number
def partial_pattern_space(combi_list, block_size):
    mask = []
    for k in range(len(combi_list)):
        temp = torch.ones(block_size,block_size,dtype=torch.int)
        for location in combi_list[k]:
            row = location // block_size
            column = location % block_size
            temp[row][column] = 0
        mask.append(temp)
    return mask


def generate_mask(pruning_number,block_size,random_number):
    combi_list = random_choose_pruning_location(pruning_number,block_size,random_number)
    mask = partial_pattern_space(combi_list,block_size)
    return mask


def random_generate_pattern_dict(model,prune_ratios,pruning_number,block_size):
    para_dict = {}
    for name, weight in model.named_parameters():
        if ('in_proj_weight' in name) or ('out_proj.weight' in name) \
                or ('linear1.weight' in name) or ('linear2.weight' in name) \
                or ('encoder.weight' in name) or ('decoder.weight' in name):
            if prune_ratios[name] == 0.0:
                continue
            else:
                para_dict[name] = generate_mask(pruning_number, block_size, 4)
    return para_dict

