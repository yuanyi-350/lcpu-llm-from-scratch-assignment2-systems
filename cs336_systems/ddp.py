import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Optimizer
import torch._utils as tutils
from typing import Type, Any



class DDPIndividualParameters(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.handles = []
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        if dist.is_initialized():
            with torch.no_grad():
                for param in self.module.parameters():
                    dist.broadcast(param.data, src=0)

        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._get_hook())

    def _get_hook(self):
        def hook(param: torch.Tensor):
            if not dist.is_initialized() or param.grad is None:
                return

            handle = dist.all_reduce(
                param.grad.data, 
                op=dist.ReduceOp.AVG, 
                async_op=True
            )
            
            self.handles.append(handle)

        return hook

    def forward(self, *inputs, **kwargs):
        self.handles = []
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
            
        self.handles = []



class DDPBucketed(nn.Module):
    def __init__(self, module: nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        
        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size()
            with torch.no_grad():
                for param in self.module.parameters():
                    dist.broadcast(param.data, src=0)
        else:
            self.world_size = 1
            
        self.buckets = []
        current_bucket = []
        current_size_bytes = 0
        limit_bytes = bucket_size_mb * 1024 * 1024
        
        params = [p for p in self.module.parameters() if p.requires_grad]
        params.reverse() 
        
        for p in params:
            p_size = p.numel() * p.element_size()
            if current_size_bytes + p_size > limit_bytes and len(current_bucket) > 0:
                self.buckets.append(current_bucket)
                current_bucket = []
                current_size_bytes = 0
            
            current_bucket.append(p)
            current_size_bytes += p_size
            
        if current_bucket:
            self.buckets.append(current_bucket)
            
        self.bucket_ready_counts = [0] * len(self.buckets)
        self.bucket_works = [None] * len(self.buckets)     
        self.bucket_flat_grads = [None] * len(self.buckets)

        for idx, bucket in enumerate(self.buckets):
            for p in bucket:
                p.register_post_accumulate_grad_hook(self._get_hook(idx))

    def _get_hook(self, bucket_idx):
        def hook(param):
            self.bucket_ready_counts[bucket_idx] += 1
            
            if self.bucket_ready_counts[bucket_idx] == len(self.buckets[bucket_idx]):
                self._fire_bucket(bucket_idx)
        return hook

    def _fire_bucket(self, bucket_idx):
        grads = [p.grad for p in self.buckets[bucket_idx]]
        flat_grad = tutils._flatten_dense_tensors(grads)
        
        flat_grad.div_(self.world_size)
        
        work = dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM, async_op=True)
        
        self.bucket_works[bucket_idx] = work
        self.bucket_flat_grads[bucket_idx] = flat_grad

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for idx in range(len(self.buckets)):
            if self.bucket_works[idx] is not None:
                self.bucket_works[idx].wait()
                
                grads = [p.grad for p in self.buckets[idx]]
                unflattened_grads = tutils._unflatten_dense_tensors(self.bucket_flat_grads[idx], grads)
                
                for orig_grad, new_grad in zip(grads, unflattened_grads):
                    orig_grad.copy_(new_grad)
                    
                self.bucket_works[idx] = None
                self.bucket_flat_grads[idx] = None
                self.bucket_ready_counts[idx] = 0




class ShardedOptimizer(Optimizer):
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
            
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs
        
        self.param_to_rank = {}
        self.param_index = 0
        
        self._inner_param_groups = []
        self.inner_optimizer = None

        super().__init__(params, defaults=kwargs)

        self.inner_optimizer = self.optimizer_cls(self._inner_param_groups, **self.optimizer_kwargs)

    def add_param_group(self, param_group: dict[str, Any]):
        super().add_param_group(param_group)

        sharded_params = []
        
        for p in param_group['params']:
            owner_rank = self.param_index % self.world_size
            self.param_to_rank[p] = owner_rank
            
            if owner_rank == self.rank:
                sharded_params.append(p)
                
            self.param_index += 1

        inner_group = {k: v for k, v in param_group.items() if k != 'params'}
        inner_group['params'] = sharded_params

        if self.inner_optimizer is None:
            self._inner_param_groups.append(inner_group)
        else:
            self.inner_optimizer.add_param_group(inner_group)

    def step(self, closure=None, **kwargs):
        loss = self.inner_optimizer.step(closure=closure, **kwargs)

        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    owner_rank = self.param_to_rank[p]
                    dist.broadcast(p.data, src=owner_rank)
        return loss