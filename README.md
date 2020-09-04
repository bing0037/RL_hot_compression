# RL_hot_compression

## Ours
modify d_rl_input.py
**"timing_constraint":115#115 for high, 104 for middle,94 for low**

CUDA_VISIBLE_DEVICES=XXX nohup python -u d_rl_controller.py > ./rl_result/high_timing_ours
CUDA_VISIBLE_DEVICES=XXX nohup python -u d_rl_controller.py > ./rl_result/middle_timing_ours
CUDA_VISIBLE_DEVICES=XXX nohup python -u d_rl_controller.py > ./rl_result/low_timing_ours

