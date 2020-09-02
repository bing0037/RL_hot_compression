# RL_hot_compression

##1. experiment 1 
    modify d_rl_input.py
    **ours = True
    random_pattern = False**
    CUDA_VISIBLE_DEVICES=XXX nohup python -u d_rl_controller.py > ./rl_result/ours_300episodes_1epochs

##2. experiment 2
  modify d_rl_input.py
  **ours = False
    random_pattern = True**
  CUDA_VISIBLE_DEVICES=XXX nohup python -u d_rl_controller.py > ./rl_result/contrast1_300episodes_1epochs
  
##3. experiment 3
  modify d_rl_input.py
  **ours = False
    random_pattern = False**
  CUDA_VISIBLE_DEVICES=XXX nohup python -u d_rl_controller.py > ./rl_result/contrast2_300episodes_1epochs
  
