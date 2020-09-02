import torch
from d_generate_rl_input import random_generate_rl_input, extract_generate_rl_input
from e_load_onfig_file import load_config_file
from c_precompression_extract_joint_training import model


block_size = 100
pruning_number_list = [100, 2500, 5000, 7500, 9900]

ours = False
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
    "level_space":([[[0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 2, 3], [0, 2, 4], [0, 3, 4], [1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]]),
    "num_children_per_episode": 1,
    'hidden_units': 35,
    'max_episodes': 300,
    'epochs':1
}