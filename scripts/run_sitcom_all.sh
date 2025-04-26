Xvfb :1 -screen 0 1024x768x24 &

export VK_LAYER_PATH=/usr/share/vulkan/explicit_layer.d
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json

export DISPLAY=:1
model_name=sitcom
tasks=(
  bridge.sh
)
ckpts=(
    /home/rishisha/SimplerEnv-SITCOM/openvla_finetuned/openvla-7b+simpler_rlds+b6+lr-0.0005+lora-r16+dropout-0.0--image_aug
)
unnorm_key=simpler_rlds

action_ensemble_temp=-0.8
temperature=1.0
# Define num_candidates values to iterate through
num_candidates_values=(10 15)

for ckpt_path in ${ckpts[@]}; do
  base_dir=$(dirname $ckpt_path)

  # Loop through each num_candidates value
  for num_candidates in ${num_candidates_values[@]}; do
    # Updated logging directory to include temperature and num_candidates
    logging_dir=results/$(basename $ckpt_path)${action_ensemble_temp}_temp${temperature}_candidates${num_candidates}
    mkdir -p $logging_dir
    for i in ${!tasks[@]}; do
      task=${tasks[$i]}
      echo "🚀 running $task with num_candidates=${num_candidates}..."
      device=0,1,2,3,4,5,6,7
      session_name=CUDA${device}-$(basename $logging_dir)-${task}
      # Pass temperature and num_candidates as parameters
      bash scripts/$task $ckpt_path $model_name $action_ensemble_temp $logging_dir $device $unnorm_key $temperature $num_candidates
    done

    # statistics evalution results
    echo "🚀 all tasks DONE for num_candidates=${num_candidates}! Calculating metrics..."
    python tools/calc_metrics_evaluation_videos.py \
      --log-dir-root $logging_dir \
      >>$logging_dir/total.metrics
  done
done