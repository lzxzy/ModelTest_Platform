python -m torch.distributed.launch --use_env --nnodes=1 --nproc_per_node=2 --master_addr localhost --master_port 29400  elastic_ddp.py