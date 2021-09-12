task=$1
part=$2
seed=$3
python train/run_crf.py --seed $seed --sf seed${seed} --lab $part --task $task
python train/run_self_train --model ${task}_crf_label${part}unl50_seed${seed} --lab $part --seed $seed --sf seed${seed} --task $task
python train/run_trf_semi --model ${task}_crf_label${part}unl50_seed${seed} --lab $part --seed $seed --sf seed${seed} --task $task

