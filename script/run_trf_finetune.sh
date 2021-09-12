task=$1
lab=$2
unl=$3
seed=$4
python train/run_trf_pretrain.py --seed $seed --lab $lab --unl $unl --task $task --sf seed${seed} 
python train/run_crf.py --seed $seed --model ${task}_trf_pretrain_label${lab}unl${unl}_seed${seed} --lab $lab --unl $unl --task $task
python train/run_trf_semi.py --seed $seed --model ${task}_trf_pretrain_label${lab}unl${unl}_seed${seed} --lab $lab --unl $unl --task $task
