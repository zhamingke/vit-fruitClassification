import argparse
import os
import uuid
from pathlib import Path

import main_pretrain as trainer
import submitit
os.environ['PYTHONPATH'] = "/home/jovyan/mae-main:" + os.environ.get('PYTHONPATH', '')

def parse_args():
    trainer_parser = trainer.get_args_parser()
    parser = argparse.ArgumentParser("Submitit for MAE pretrain", parents=[trainer_parser])
    parser.add_argument("--ngpus", default=8, type=int, help="Number of GPUs to request on each node")
    parser.add_argument("--nodes", default=2, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=4320, type=int, help="Duration of the job in minutes")
    parser.add_argument("--job_dir", default="", type=str, help="Job directory. Leave empty for automatic creation.")
    parser.add_argument("--partition", default="learnfair", type=str, help="Cluster partition where the job will be submitted")
    parser.add_argument("--use_volta32", action='store_true', help="Request 32GB V100 GPUs if available")
    parser.add_argument('--comment', default="", type=str, help="Additional comment to pass to the scheduler")
    return parser.parse_args()

def get_shared_folder() -> Path:

    user = os.getenv("USER", "zmk") 
    if Path("mae-main/pretrain/").is_dir():
        p = Path(f"mae-main/pretrain/zmk/experiments").resolve()
        p.mkdir(parents=True, exist_ok=True)  
        return p
    raise RuntimeError("No shared folder available")

def get_init_file():
    
    shared_folder = get_shared_folder().resolve()
    os.makedirs(str(shared_folder), exist_ok=True)  
    init_file = shared_folder / f"{uuid.uuid4().hex}_init"  
    if init_file.exists():
        os.remove(str(init_file))  
    return init_file.resolve()

class Trainer(object):
    
    def __init__(self, args):
        self.args = args  

    def __call__(self):
        
        import main_pretrain as trainer

        self._setup_gpu_args()  
        trainer.main(self.args)  

    def checkpoint(self):
       
        import os
        import submitit

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

    args.ngpus 
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
    print(f"Job ID: {job.job_id}, State: {job.state}")  

if __name__ == "__main__":
    main()
