import torch
from d_generate_rl_input import random_generate_rl_input, extract_generate_rl_input
from e_load_onfig_file import load_config_file
from c_precompression_extract_joint_training import model


block_size = 100
pruning_number_list = [100,1300,2500,3700,5000,6200,7500,8700,9300,9900]

ours = True
random_pattern = False
if ours:#True
    print('#' * 89)
    print('A.pattern pruning from precompression model')
    print('B.extract important pattern from precompression model')
    print('C.training(pruning number={})'.format(pruning_number_list))
    print('#' * 89)

    config_file = './config_file/prune_ratio_v6.yaml'
    prune_ratios = load_config_file(config_file)
    model.load_state_dict(torch.load('./model/random_column_pruning_bingbing_pattern.pt'))
    para_set = extract_generate_rl_input(model,block_size,prune_ratios,pruning_number_list)

else:
    if random_pattern:#False True
        print('#' * 89)
        print('A.pattern pruning from precompression model')
        print('B.random generate pattern for every layer')
        print('C.training(pruning number={})'.format(pruning_number_list))
        print('#' * 89)

        config_file = './config_file/prune_ratio_v6.yaml'
        prune_ratios = load_config_file(config_file)
        model.load_state_dict(torch.load('./model/random_column_pruning_bingbing_pattern.pt'))
        para_set = random_generate_rl_input(prune_ratios,pruning_number_list,block_size)
    else:#False False
        print('#' * 89)
        print('A.pattern pruning from random column pruning model(10epochs)')
        print('B.extract pattern from random column pruning model')
        print('C.training(pruning number={})'.format(pruning_number_list))
        print('#' * 89)

        config_file = './config_file/prune_ratio_v1.yaml'
        prune_ratios = load_config_file(config_file)
        model.load_state_dict(torch.load('./model/random_column_pruning_average.pt'))
        para_set = extract_generate_rl_input(model,block_size,prune_ratios,pruning_number_list)


controller_params = {
    "model": model,
    "sw_space":(para_set),
    "level_space":([[[0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 1, 5], [0, 1, 6], [0, 1, 7], [0, 1, 8], [0, 1, 9], [0, 2, 3], [0, 2, 4], [0, 2, 5], [0, 2, 6], [0, 2, 7], [0, 2, 8], [0, 2, 9], [0, 3, 4], [0, 3, 5], [0, 3, 6], [0, 3, 7], [0, 3, 8], [0, 3, 9], [0, 4, 5], [0, 4, 6], [0, 4, 7], [0, 4, 8], [0, 4, 9], [0, 5, 6], [0, 5, 7], [0, 5, 8], [0, 5, 9], [0, 6, 7], [0, 6, 8], [0, 6, 9], [0, 7, 8], [0, 7, 9], [0, 8, 9], [1, 2, 3], [1, 2, 4], [1, 2, 5], [1, 2, 6], [1, 2, 7], [1, 2, 8], [1, 2, 9], [1, 3, 4], [1, 3, 5], [1, 3, 6], [1, 3, 7], [1, 3, 8], [1, 3, 9], [1, 4, 5], [1, 4, 6], [1, 4, 7], [1, 4, 8], [1, 4, 9], [1, 5, 6], [1, 5, 7], [1, 5, 8], [1, 5, 9], [1, 6, 7], [1, 6, 8], [1, 6, 9], [1, 7, 8], [1, 7, 9], [1, 8, 9], [2, 3, 4], [2, 3, 5], [2, 3, 6], [2, 3, 7], [2, 3, 8], [2, 3, 9], [2, 4, 5], [2, 4, 6], [2, 4, 7], [2, 4, 8], [2, 4, 9], [2, 5, 6], [2, 5, 7], [2, 5, 8], [2, 5, 9], [2, 6, 7], [2, 6, 8], [2, 6, 9], [2, 7, 8], [2, 7, 9], [2, 8, 9], [3, 4, 5], [3, 4, 6], [3, 4, 7], [3, 4, 8], [3, 4, 9], [3, 5, 6], [3, 5, 7], [3, 5, 8], [3, 5, 9], [3, 6, 7], [3, 6, 8], [3, 6, 9], [3, 7, 8], [3, 7, 9], [3, 8, 9], [4, 5, 6], [4, 5, 7], [4, 5, 8], [4, 5, 9], [4, 6, 7], [4, 6, 8], [4, 6, 9], [4, 7, 8], [4, 7, 9], [4, 8, 9], [5, 6, 7], [5, 6, 8], [5, 6, 9], [5, 7, 8], [5, 7, 9], [5, 8, 9], [6, 7, 8], [6, 7, 9], [6, 8, 9], [7, 8, 9]]]),
    "num_children_per_episode": 1,
    'hidden_units': 35,
    'max_episodes': 300,
    'epochs':1,
    "timing_constraint":115#115 for high, 104 for middle,94 for low
}