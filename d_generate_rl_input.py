from a_extract_info_from_precompression import compute_importance_weight, generate_layer_pattern
from a_random_generate_discrete_pattern import generate_mask


def random_generate_rl_input(prune_ratios,pruning_number_list,block_size,random_number=10,repeat_number=4):
    all_layer = []
    for name in prune_ratios:
        if prune_ratios[name] == 0.0:
            continue
        else:
            for pruning_number in pruning_number_list:
                mask = generate_mask(pruning_number, block_size, random_number)
                for r in range(repeat_number):
                    all_layer.append(mask)
    return all_layer#[[],[],[],......]

def extract_generate_rl_input(model,block_size,prune_ratios,pruning_number_list):
    importance_weight_dict = compute_importance_weight(model, block_size, 10, 16, 10, prune_ratios)
    extract_pattern = generate_layer_pattern(importance_weight_dict,block_size,pruning_number_list)
    return extract_pattern
