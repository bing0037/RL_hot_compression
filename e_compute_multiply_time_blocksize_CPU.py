import os
import random
import time
import torch


#generate weight block(each block size = block_size * block_size)
def generate_weight_block(pruning_number_list,block_size):
    length = block_size * block_size
    data = [i for i in range(length)]
    weight_block = []
    for pruning_number in pruning_number_list:
        init_pattern = [[1 for i in range(block_size)] for j in range(block_size)]
        pattern_number = random.sample(data, pruning_number)
        for location in pattern_number:
            row = location // block_size
            column = location % block_size
            init_pattern[row][column] = 0
        weight_block.append(init_pattern)
    return weight_block


def compute_time(pruning_number_list,block_size):
    #write file
    file_name = "./computation_time_result/computation_time_blocksize_{}_CPU.out".format(block_size)
    if os.path.exists(file_name):
        os.remove(file_name)
    file_path = open(file_name,'a')

    #input data
    input_row = 350
    input_column = block_size
    input = torch.randn(input_row,input_column)

    #weight block
    weight_block = generate_weight_block(pruning_number_list,block_size)

    #result
    result = torch.zeros(input_row,block_size)

    average_time_list = []
    pruning_number_time_dict = {}
    # 10 iterations for average computation time
    for block_j in range(len(weight_block)):
        total_time = 0
        for epoch in range(1,11):
            start = time.time()
            # multiply
            for k in range(input_row):#input row
                for p in range(block_size):
                    for q in range(block_size):#block column
                        if(weight_block[block_j][q][p] == 0):
                            continue
                        else:
                            result[k][p] += input[k][q] * weight_block[block_j][q][p]
            end = time.time()
            duration = (end - start) * 1000
            total_time += duration
        print('-' * 89, file=file_path)
        print("pruning number = ", pruning_number_list[block_j], file=file_path)
        average_time = total_time / epoch
        print("average time:", average_time, "ms", file=file_path)
        print('-' * 89, file=file_path)
        average_time_list.append(average_time)
        pruning_number_time_dict[block_j] = average_time
    return average_time_list,pruning_number_time_dict



import matplotlib.pyplot as plt

def plt_show(pruning_number_list,average_list,block_size):
    plt.figure(figsize=(8,4))
    plt.plot(pruning_number_list,average_list,"b",marker='*',linewidth=1)
    plt.xlabel("Pruning number")
    plt.ylabel("Time(ms)")
    plt.title("Computation Time(CPU,block size={})".format(block_size))
    plt.savefig("./computation_time_result/CPU_blocksize_{}_time.png".format(block_size))
    plt.show()


# def compute_time_ratio(pruning_number_time_dict):
#     max_time = pruning_number_time_dict[0]
#     time_ratio_dict = {}
#     for no in pruning_number_time_dict:
#         time_ratio = pruning_number_time_dict[no] / max_time
#         time_ratio_dict[no] = 1 - time_ratio
#     return time_ratio_dict


def frequency_time(level,frequency,pruning_number_time_dict):
    max_frequency = 1400
    latency = (float(max_frequency) / frequency) * pruning_number_time_dict[level+1]
    return latency


def energy_power(frequency_list,voltage_list):
    usage_time_level = []
    for i in range(3):
        if i == 0:
            energy = 0.6
        if i == 1:
            energy = 0.35
        if i == 2:
            energy = 0.05
        power = frequency_list[i] * pow(voltage_list[i], 2)
        usage_time = (energy * pow(10, 6)) / power
        usage_time_level.append(usage_time)
    return usage_time_level


def normalization(array):
    result = []
    min = 0
    max = 7.1
    k= 1.0 / (max - min)
    for number in array:
        norY = 0 + k * (number - min)
        result.append(norY)
    return result


#just need to modify these two parameters
block_size = 10
pruning_number_list = [0,1,25,50,75,99]
frequency_list = [1400,1000,600]
voltage_list = [1.240,1.06625,0.9175]

average_time_list, pruning_number_time_dict = compute_time(pruning_number_list,block_size)
# plt_show(pruning_number_list,average_time_list,block_size)
print("time dict:",pruning_number_time_dict)
usage_time_level = energy_power(frequency_list,voltage_list)
print("usage time level:",usage_time_level)



#test inference times
# level = [[0,2,3],[0,2,4],[0,3,4],[1,2,3],[1,2,4],[1,3,4],[2,3,4]]
# for times in level:
#     times_list = []
#     for i in range(3):
#         latency = frequency_time(times[i],frequency_list[i],pruning_number_time_dict)
#         once = usage_time_level[i] / latency
#         times_list.append(once)
#     print(times_list)
#     nor_list = normalization(times_list)
#     print(sum(times_list))
#     print(nor_list)
#     print(sum(nor_list))
#     print('-'*29)
