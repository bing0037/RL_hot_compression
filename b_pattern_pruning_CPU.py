import torch
import numpy as np


#zero padding
def weight_padding(weight,block_size):
    weight_shape = weight.shape
    block_row = weight_shape[0]
    block_column = weight_shape[1]
    block_new_row = int((block_row + block_size -1) // block_size) * block_size
    block_new_column = int((block_column + block_size - 1) // block_size) * block_size
    zero_right = torch.zeros(block_row,block_new_column-block_column)
    weight = torch.cat((weight,zero_right),1)#column
    zero_bottom = torch.zeros(block_new_row-block_row,block_new_column)
    weight = torch.cat((weight,zero_bottom),0)#row
    return weight,block_row,block_column


#zero de-padding
def weight_depadding(weight,block_row,block_column):
    return weight[:block_row,:block_column]


#recovery block to normal weight shape
def recovery_block_to_weight(block_weight,block_size,original_row,original_column):
    block_row = original_row // block_size
    block_column = original_column // block_size
    row_dimension = torch.tensor([],dtype=torch.int)
    for j in range(block_row):
        column_dimension = block_weight[j*block_column]
        for i in range(1,block_column):
            column_dimension = torch.cat((column_dimension,block_weight[j*block_column+i]),1)
        row_dimension = torch.cat((row_dimension,column_dimension),0)
    return row_dimension


#a block weight chooses pattern from mask based on euclidean distance
def pattern_choose_Euclidean(block_weight,mask):
    min = np.array(9999)
    min_pattern = []
    block_weight_replica = block_weight.detach().cpu()  # convert tensor to numpy
    for temp_pattern in mask:
        tensor_pattern = torch.tensor(temp_pattern)
        weight_replica_pattern = block_weight_replica.__mul__(tensor_pattern)  # weight multiple pattern
        sub_original_pruning = block_weight_replica - weight_replica_pattern  # sub between weight and pruning weight
        euc_distance = np.linalg.norm(sub_original_pruning)  # compute euclidean distance
        if euc_distance < min:
            min = euc_distance
            min_pattern = tensor_pattern
    result_pattern = min_pattern
    return result_pattern


#choose whole pattern for pattern pruning
def build_whole_pattern(model,mask_dict,block_size,device):
    # print("Build whole pattern based on Euclidean distance!")
    dict_pattern = {}
    for name,weight in model.named_parameters():
        if ('in_proj_weight' in name) or ('out_proj.weight' in name) \
                or ('linear1.weight' in name) or ('linear2.weight' in name) \
                or ('encoder.weight' in name) or ('decoder.weight' in name):
            if name in mask_dict:
                mask = mask_dict[name]
                temp_whole_pattern = []
                weight_replica = weight.detach().cpu()
                weight_replica, original_row, original_column = weight_padding(weight_replica,block_size)
                new_row = weight_replica.shape[0]
                new_column = weight_replica.shape[1]
                row_block_num = new_row // block_size
                column_block_num = new_column // block_size
                for i in range(row_block_num):
                    for j in range(column_block_num):
                        block_weight = weight_replica[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
                        result_pattern = pattern_choose_Euclidean(block_weight,mask)
                        result_pattern = [e.numpy() for e in result_pattern]
                        temp_whole_pattern.append(result_pattern)
                result_whole_pattern = torch.tensor(temp_whole_pattern,dtype=torch.int)
                result_whole_pattern = recovery_block_to_weight(result_whole_pattern,block_size,new_row,new_column)
                result_whole_pattern = weight_depadding(result_whole_pattern,original_row,original_column)
                result_whole_pattern.to(device)
                dict_pattern[name] = result_whole_pattern
            else:
                row = weight.shape[0]
                column = weight.shape[1]
                result_whole_pattern = torch.ones(row,column,dtype=torch.int).to(device)
                dict_pattern[name] = result_whole_pattern
    return dict_pattern


#pattern pruning based on whole pattern which will change the original model
def changed_pattern_pruning(model,dict_pattern,device):
    for name, weight in model.named_parameters():
        if ('in_proj_weight' in name) or ('out_proj.weight' in name) \
                or ('linear1.weight' in name) or ('linear2.weight' in name) \
                or ('encoder.weight' in name) or ('decoder.weight' in name):
            weight.data = weight.data.mul_(dict_pattern[name].to(device))





