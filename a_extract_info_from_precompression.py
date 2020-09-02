import random
import torch
from b_pattern_pruning_CPU import weight_padding, weight_depadding


#extract original precompression model's whole pattern for every layer
def extract_original_layers_whole_pattern(model,device):
    original_whole_pattern = {}
    for name,weight in model.named_parameters():
        if ('in_proj_weight' in name) or ('out_proj.weight' in name) \
                or ('linear1.weight' in name) or ('linear2.weight' in name) \
                or ('encoder.weight' in name) or ('decoder.weight' in name):
            weight_replica = weight.detach().cpu()
            whole_pattern = torch.ones(weight_replica.shape[0],weight_replica.shape[1], dtype=torch.int)
            zero_location = (weight_replica == 0.)
            whole_pattern[zero_location] = 0
            whole_pattern.to(device)
            original_whole_pattern[name] = whole_pattern
    return original_whole_pattern



#cut weight to block
def cut_weight_to_block(weight,block_size):
    block_weight = []
    row_block_num = weight.shape[0] // block_size
    column_block_num = weight.shape[1] // block_size
    for i in range(row_block_num):
        for j in range(column_block_num):
            temp_block = weight[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            block_weight.append(temp_block)
    return block_weight


#extract top-10 important weight matrix for every layer
def compute_importance_weight(model,block_size,extract_m,extract_k,extract_p,prune_ratios):
    importance_weight_dict = {}
    for name,weight in model.named_parameters():
        if ('in_proj_weight' in name) or ('out_proj.weight' in name) \
                or ('linear1.weight' in name) or ('linear2.weight' in name) \
                or ('encoder.weight' in name) or ('decoder.weight' in name):
            if prune_ratios[name] == 0.0:
                continue
            else:
                weight_replica = weight.detach().cpu()
                padding_weight, original_row, original_column = weight_padding(weight_replica, block_size)
                block_list = cut_weight_to_block(padding_weight,block_size)
                temp_importance_weight_list = []
                for i in range(extract_k):
                    sample_block_list = random.sample(block_list,extract_m)
                    importance_weight = torch.zeros(block_size,block_size)
                    for sample_block in sample_block_list:
                        importance_weight = torch.add(importance_weight,sample_block)
                    importance_weight_sum = torch.sum(importance_weight)
                    temp_importance_weight_list.append((importance_weight_sum,importance_weight))
                importance_sorted = sorted(temp_importance_weight_list,key=lambda t:t[0])
                max_p_weight = importance_sorted[0:extract_p]
                importance_weight_dict[name] = max_p_weight #tuple
    return importance_weight_dict


#for every layer, for every important weight, for every sparsity, extract 10 patterns
def generate_layer_pattern(importance_weight_dict,block_size,pruning_number_list,repeat_number = 4):
    extract_pattern = []
    for name in importance_weight_dict:
        max_p_weight_list = importance_weight_dict[name]
        for pruning_number in pruning_number_list:
            pruning_rate_pattern_list = []
            for tuple in max_p_weight_list:
                sequence_matrix = tuple[1].view(-1)
                sort, idx = torch.sort(sequence_matrix,descending=True)
                pattern = torch.ones(block_size, block_size,dtype=torch.int)
                importance_matrix = idx[0:pruning_number]
                for location in importance_matrix:
                    row = location // block_size
                    column = location % block_size
                    pattern[row][column] = 0
                pruning_rate_pattern_list.append(pattern)
            for i in range(repeat_number):
                extract_pattern.append(pruning_rate_pattern_list)
    return extract_pattern


def generate_test_layer_pattern(importance_weight_dict,block_size,pruning_number):
    extract_pattern_dict = {}
    for name in importance_weight_dict:
        max_p_weight_list = importance_weight_dict[name]
        temp_pattern_list = []
        for tuple in max_p_weight_list:
            sequence_matrix = tuple[1].view(-1)
            sort, idx = torch.sort(sequence_matrix,descending=True)
            pattern = torch.ones(block_size, block_size,dtype=torch.int)
            importance_matrix = idx[0:pruning_number]
            for location in importance_matrix:
                row = location // block_size
                column = location % block_size
                pattern[row][column] = 0
            temp_pattern_list.append(pattern)
        extract_pattern_dict[name] = temp_pattern_list #{name:[[],[],[],[]]}
    return extract_pattern_dict