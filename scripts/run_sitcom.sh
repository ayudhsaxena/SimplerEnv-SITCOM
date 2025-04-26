Xvfb :1 -screen 0 1024x768x24 &
export DISPLAY=:1
model_name=sitcom
tasks=(
  bridge.sh
)
ckpts=(
  /data/user_data/ayudhs/random/multimodal/openvla-7b+simpler_rlds+b6+lr-0.0005+lora-r16+dropout-0.0--image_aug
)
unnorm_key=simpler_rlds
traj_len=10
run_name=oracle_reward
num_rollouts=25
window_size=5

action_ensemble_temp=-0.8
for ckpt_path in ${ckpts[@]}; do
  base_dir=$(dirname $ckpt_path)

  # evaluation in simulator
  # logging_dir=$base_dir/simpler_env/$(basename $ckpt_path)${action_ensemble_temp}
  logging_dir=results/$(basename $ckpt_path)${action_ensemble_temp}_traj_len_${traj_len}_num_rollouts_${num_rollouts}_window_sz_${window_size}_${run_name}
  mkdir -p $logging_dir
  for i in ${!tasks[@]}; do
    task=${tasks[$i]}
    echo "🚀 running $task ..."
    device=5
    session_name=CUDA${device}-$(basename $logging_dir)-${task}
    bash scripts/$task $ckpt_path $model_name $action_ensemble_temp $logging_dir $device $unnorm_key $traj_len $num_rollouts $window_size
  done

  # statistics evalution results
  echo "🚀 all tasks DONE! Calculating metrics..."
  python tools/calc_metrics_evaluation_videos.py \
    --log-dir-root $logging_dir \
    >>$logging_dir/total.metrics
done
