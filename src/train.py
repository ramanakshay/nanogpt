from data import TextData
from model import GPTModel
from algorithm import Trainer

import os
import torch
from torch.distributed import init_process_group, destroy_process_group
from contextlib import nullcontext

import hydra
from omegaconf import DictConfig, OmegaConf

def setup(config):
    # ddp init
    is_ddp = int(os.environ.get('RANK', -1)) != -1
    if is_ddp:
        init_process_group(backend=config.system.backend)
        global_rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{local_rank}'
        torch.cuda.set_device(device)
        is_master_process = global_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = global_rank # each process gets a different seed
    else:
        global_rank = local_rank = -1
        world_size = 1
        device = config.system.device
        is_master_process = True
        seed_offset = 0

    device_type = 'cuda' if 'cuda' in device else 'cpu'

    state = OmegaConf.create({"is_ddp": is_ddp,
                              "device": device,
                              "device_type": device_type,
                              "is_master_process": is_master_process,
                              "local_rank": local_rank,
                              "global_rank": global_rank,
                              "world_size": world_size})

    # expected tokens periter
    tokens_per_iter = config.algorithm.grad_accum * world_size * config.data.batch_size * config.data.block_size
    if state.is_master_process:
        print(f"tokens per iteration will be: {tokens_per_iter:,}")

    # set random seed
    torch.manual_seed(1337 + seed_offset)

    # floating point optimizations
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    dtype = config.system.dtype
    if dtype == 'float16': assert torch.cuda.is_available()
    if dtype == 'bfloat16': assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config.system.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    return state, ctx


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(config: DictConfig) -> None:
    state, ctx = setup(config)

    ## DATA ##
    data = TextData(config, state)
    if state.is_master_process:
        print('Data loaded.')

    ## MODEL ##
    model = GPTModel(config, state)
    if state.is_master_process:
        print('Model created.')

    ## ALGORITHM ##
    algorithm = Trainer(data, model, config, state, ctx)
    algorithm.run()
    if state.is_master_process:
        print('Done!')

    if state.is_ddp:
        destroy_process_group()

if __name__ == "__main__":
    main()
