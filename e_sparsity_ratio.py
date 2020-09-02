
def sparsity_ratio(model):
    nonzero_param, total_param = 0, 0
    for name, weight in model.named_parameters():
        if ('in_proj_weight' in name) or ('out_proj.weight' in name) \
                or ('linear1.weight' in name) or ('linear2.weight' in name) \
                or ('encoder.weight' in name) or ('decoder.weight' in name):
            print("[at weight {}]".format(name))
            print(
                "percentage of pruned: {:.4f}%".format(100 * (abs(weight) == 0).sum().item() / weight.numel()))#sparsity ratio in every weight
            print(
                "nonzero parameters after pruning: {} / {}\n".format((weight != 0).sum().item(), weight.numel()))#nonzero value in every weight
        total_param += weight.numel()
        nonzero_param += (weight != 0).sum().item()
    pruning_rate = (total_param - nonzero_param) / total_param
    print("total pruning rate: {} / {} ({:.4f}%)".
          format(nonzero_param, total_param, 100 * pruning_rate))