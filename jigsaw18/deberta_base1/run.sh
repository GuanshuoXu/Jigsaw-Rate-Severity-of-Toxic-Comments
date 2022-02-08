cd warmup
python -m torch.distributed.launch --nproc_per_node=3 train.py > train.txt
cd ..
cd train
python -m torch.distributed.launch --nproc_per_node=3 train.py > train.txt
cd ..
cd valid
python valid.py > valid.txt
cd ..
