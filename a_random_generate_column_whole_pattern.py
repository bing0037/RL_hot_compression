import random

import torch
from b_pattern_pruning_CPU import weight_padding, weight_depadding



def random_generate_column_whole_pattern(model,prune_ratios,block_size):
    column_pattern_dict = {}
    for name, weight in model.named_parameters():
        if ('in_proj_weight' in name) or ('out_proj.weight' in name) \
                or ('linear1.weight' in name) or ('linear2.weight' in name) \
                or ('encoder.weight' in name) or ('decoder.weight' in name):
            if prune_ratios[name] == 0.0:
                whole_column_pattern = torch.ones(weight.shape[0], weight.shape[1], dtype=torch.int)
                column_pattern_dict[name] = whole_column_pattern
            else:
                weight_replica = weight.detach().cpu()
                padding_weight, original_row, original_column = weight_padding(weight_replica, block_size)
                whole_column_pattern = torch.ones(padding_weight.shape[0], padding_weight.shape[1], dtype=torch.int)
                rectangle_row = padding_weight.shape[0] // block_size
                rectangle_column = padding_weight.shape[1]
                pruning_location = prune_ratios[name] * rectangle_row * rectangle_column
                data = [i for i in range(rectangle_row * rectangle_column)]
                choose_location = random.sample(data, int(pruning_location))
                for location in choose_location:
                    location_row = location // rectangle_column
                    location_column = location % rectangle_column
                    whole_column_pattern[location_row * block_size:(location_row + 1) * block_size,location_column:location_column+1] = 0
                whole_column_pattern = weight_depadding(whole_column_pattern,original_row,original_column)
                column_pattern_dict[name] = whole_column_pattern
    return column_pattern_dict
