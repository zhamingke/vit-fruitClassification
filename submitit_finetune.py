# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# A script to run multinode training with submitit.
# --------------------------------------------------------

import argparse
import os
import uuid
from pathlib import Path
import main_finetune as classification
import submitit
os.environ['PYTHONPATH'] = "/home/jovyan/mae-main:" + os.environ.get('PYTHONPATH', '')

def parse_args():
   
    classification_parser = classification.get_args_parser()
    parser = argparse.ArgumentParser("Submitit for MAE finetune", parents=[classification_parser])

    parser.add_argument("--ngpus", default=8, type=int, help="Number of GPUs to request on each node")
    parser.add_argument("--nodes", default=2, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=4320, type=int, help="Duration of the job in minutes")
    parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic")

    parser.add_argument("--partition", default="learnfair", type=str, help="Partition where to submit")
    parser.add_argument("--use_volta32", action='store_true', help="Request 32G V100 GPUs")
    parser.add_argument('--comment', default="", type=str, help="Comment to pass to scheduler")

    return parser.parse_args()


def get_shared_folder() -> Path:
    
    user = os.getenv("USER", "zmk")  
    if Path("finetune/checkpoint/").is_dir():  
        p = Path(f"finetune/checkpoint/{user}/experiments")
        p.mkdir(exist_ok=True) 
        return p
    raise RuntimeError("No shared folder available")  


def get_init_file():
    """
    获取初始化文件路径，并确保文件的父目录存在
    初始化文件必须不存在，如果存在则删除
    """
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"  
    if init_file.exists():
        os.remove(str(init_file))  
    return init_file.resolve()  


class Trainer(object):
    """
    Trainer类用于封装训练过程。包含训练时所需的各种参数和方法。
    """

    def __init__(self, args):
        self.args = args  

    def __call__(self):
        
        self._setup_gpu_args()  
        classification.main(self.args)  

    def checkpoint(self):
        
        self.args.dist_url = get_init_file().as_uri() 
        checkpoint_file = os.path.join(self.args.output_dir, "checkpoint.pth")  
        if os.path.exists(checkpoint_file):
            self.args.resume = checkpoint_file  
        print("Requeuing ", self.args)  
        empty_trainer = type(self)(self.args) 
        return submitit.helpers.DelayedSubmission(empty_trainer) 

    def _setup_gpu_args(self):
        
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment() 
        self.args.output_dir = Path(str(self.args.output_dir).replace("%j", str(job_env.job_id)))  
        self.args.log_dir = self.args.output_dir  
        self.args.gpu = job_env.local_rank  
        self.args.rank = job_env.global_rank  
        self.args.world_size = job_env.num_tasks  
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")  


def main():
   
    args = parse_args()  
    if args.job_dir == "":
        args.job_dir = get_shared_folder() / "%j"  

    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout

    partition = args.partition  
    kwargs = {}  
    if args.use_volta32:
        kwargs['slurm_constraint'] = 'volta32gb' 
    if args.comment:
        kwargs['slurm_comment'] = args.comment  

   
    executor.update_parameters(
        mem_gb=40 * num_gpus_per_node,  
        gpus_per_node=num_gpus_per_node,  
        tasks_per_node=num_gpus_per_node,  
        cpus_per_task=10,  
        nodes=nodes,  
        timeout_min=timeout_min, 
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        **kwargs  
    )

    executor.update_parameters(name="mae")  

    args.dist_url = get_init_file().as_uri()
    args.output_dir = args.job_dir

    trainer = Trainer(args)  
    job = executor.submit(trainer)   

    print(job.job_id)


if __name__ == "__main__":
    main()  