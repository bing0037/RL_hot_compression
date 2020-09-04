import copy

from b_pattern_pruning_CPU import changed_pattern_pruning


def sparsity_ratio(model,print_enable):
    nonzero_param, total_param = 0, 0
    prune_ratios_dict = {}
    for name, weight in model.named_parameters():
        if ('in_proj_weight' in name) or ('out_proj.weight' in name) \
                or ('linear1.weight' in name) or ('linear2.weight' in name) \
                or ('encoder.weight' in name) or ('decoder.weight' in name):
            prune_ratios_dict[name] = (abs(weight) == 0).sum().item() / weight.numel()
            if print_enable:
                print("[at weight {}]".format(name))
                print("percentage of pruned: {:.4f}%".format(100 * (abs(weight) == 0).sum().item() / weight.numel()))#sparsity ratio in every weight
                print("nonzero parameters after pruning: {} / {}\n".format((weight != 0).sum().item(), weight.numel()))#nonzero value in every weight
        total_param += weight.numel()
        nonzero_param += (weight != 0).sum().item()
    pruning_rate = (total_param - nonzero_param) / total_param
    print("total pruning rate: {} / {} ({:.4f}%)".
          format(nonzero_param, total_param, 100 * pruning_rate))
    print('-'*89)
    return prune_ratios_dict

def whole_sparsity_ratio(model,mask_dict_set,every_mask_whole_pattern,device):
    before_prune_ratios_list = []
    for mask_j in range(len(mask_dict_set)):
        whole_weight_pattern = every_mask_whole_pattern[mask_j]
        # cpoy a submodel to evaluate
        sub_model = copy.deepcopy(model)
        changed_pattern_pruning(sub_model,whole_weight_pattern,device)
        before_prune_ratios = sparsity_ratio(sub_model,print_enable=True)
        before_prune_ratios_list.append(before_prune_ratios)
    return before_prune_ratios_list