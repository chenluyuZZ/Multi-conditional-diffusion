import subprocess

command = "mpiexec -n 4 python guided-diffusion/scripts/classifier_train.py"

# mpiexec -n 8 python scripts/classifier_sample.py --attention_resolutions 32,16,8 --class_cond True 
# --diffusion_steps 1000 --image_size 64 --learn_sigma True --noise_schedule linear --num_channels 256 
# --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True
# --classifier_scale 1.0 --classifier_path "models/diffusion_models/64x64_classifier.pt" 
# --model_path "models/diffusion_models/64x64_diffusion.pt" --batch_size 1 --num_samples 4 --timestep_respacing 250


#mpiexec -n N python train.py --data_dir path/to/data --num_gpus N

# 将命令中的占位符替换为实际的值
# command = command.replace("N", "4")  # 替换为所需的进程数
# command = command.replace("$TRAIN_FLAGS", "--train_flags=value")  # 替换为所需的训练标志
# command = command.replace("$CLASSIFIER_FLAGS", "--classifier_flags=value")  # 替换为所需的分类器标志

# 执行命令
subprocess.run(command, shell=True)
