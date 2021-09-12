seed=$1
python train/run_trf_pretrain.py --seed $seed --sf seed${seed} --nbest 1
python train/run_trf_semi.py --seed $seed --sf seed${seed} --nbest 1 --opt adam --lr 1e-3

