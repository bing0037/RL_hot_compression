from c_precompression_extract_joint_training import model

#compute the all mulpitly operations of non-pruned model
def all_multiply_operation_times():
    multiply_operation_dict = {}
    for name, weight in model.named_parameters():
        if ('in_proj_weight' in name) or ('out_proj.weight' in name) \
                or ('linear1.weight' in name) or ('linear2.weight' in name) \
                or ('encoder.weight' in name) or ('decoder.weight' in name):
            multiply_operation_dict[name] = int(weight.shape[0] * weight.shape[1])
    return multiply_operation_dict


#compute all addition operations of non-pruned model
def all_add_operation_times():
    add_operation_dict = {}
    for name, weight in model.named_parameters():
        if ('in_proj_weight' in name) or ('out_proj.weight' in name) or ('encoder.weight' in name):
            pixels = weight.shape[0] * weight.shape[1]
            add_operation_dict[name] = int(pixels - weight.shape[1])
        elif ('linear1.weight' in name) or ('linear2.weight' in name) or ('decoder.weight' in name):#linear
            pixels = weight.shape[0] * weight.shape[1]
            add_operation_dict[name] = int(pixels - weight.shape[0])
    return add_operation_dict


def pruned_multiple_operation(prune_ratios, multiply_operation_dict,add_operation_dict):
    #compute total multiply and add operation number
    sum_multiply_operation = 0
    sum_add_operation = 0
    for name in prune_ratios:
        sum_multiply_operation += multiply_operation_dict[name] * (1-prune_ratios[name])
        sum_add_operation += add_operation_dict[name] - multiply_operation_dict[name] * prune_ratios[name]
    multiply_operation = sum_multiply_operation / 10000
    add_operation = sum_add_operation / 10000
    return multiply_operation,add_operation


def compute_total_latency(frequency,multiply_operation,add_operation,multiply_10000,add_10000):
    #under confirmed frequency, compute the total latency of add and multiply operation
    m_time = (3.0 / frequency) * multiply_10000
    multiply_latency = multiply_operation * m_time

    a_time = (3.0 / frequency) * add_10000
    add_latency = add_operation * a_time

    total_latency = multiply_latency + add_latency
    return total_latency


def generate_latency_list(prune_ratio_list,multiply_operation_dict,add_operation_dict,frequency_list,multiply_10000,add_10000):
    latency_list = []
    for i in range(len(frequency_list)):
        prune_ratios = prune_ratio_list[i]
        multiply_operation, add_operation = pruned_multiple_operation(prune_ratios, multiply_operation_dict,add_operation_dict)
        sub_latency = compute_total_latency(frequency_list[i], multiply_operation, add_operation,multiply_10000,add_10000)
        latency_list.append(sub_latency)
    print("latency_list:",latency_list)
    return latency_list


def energy_power(frequency_list,voltage_list):
    usage_time_level = []
    full_energy = 37800
    for i in range(len(frequency_list)):
        if i == 0:
            energy = 0.5 * full_energy
        if i == 1:
            energy = 0.35 * full_energy
        if i == 2:
            energy = 0.15 * full_energy
        power = frequency_list[i] * pow(10,9) * pow(voltage_list[i], 2) * pow(10,-10)
        # print("power:",power)
        usage_time = (energy / power) * 1000
        usage_time_level.append(usage_time)
    # print("usage_time_level:{}ms".format(usage_time_level))
    return usage_time_level


def compute_runs_number(usage_time_level,latency_list):
    runs_number = []
    for i in range(len(usage_time_level)):
        every_runs = usage_time_level[i] // latency_list[i]
        runs_number.append(every_runs)
    print("runs_number:",runs_number)
    return runs_number


def normalization(number):
    min = 2300000
    max = 4500000
    k= 1.0 / (max - min)
    norY = 0 + k * (number - min)
    return norY


def nom_runs_number(frequency_list,voltage_list,latency_list):
    usage_time_level = energy_power(frequency_list,voltage_list)
    runs_number = compute_runs_number(usage_time_level,latency_list)
    sum_runs_number = sum(runs_number)
    nom_runs = normalization(sum_runs_number)
    return nom_runs


def times_reward(prune_ratio_list):
    #CPU 3GHZ
    multiply_10000 = 0.014370
    add_10000 = 0.014575
    frequency_list = [1.400, 1.000, 0.800]
    voltage_list = [1.240, 1.06625, 0.9925]
    multiply_operation_dict = all_multiply_operation_times()
    add_operation_dict = all_add_operation_times()
    latency_list = generate_latency_list(prune_ratio_list,multiply_operation_dict,add_operation_dict,frequency_list,multiply_10000,add_10000)
    nom_runs = nom_runs_number(frequency_list,voltage_list,latency_list)
    return latency_list, nom_runs