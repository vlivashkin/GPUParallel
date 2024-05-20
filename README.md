[![Build Status](https://travis-ci.com/vlivashkin/GPUParallel.svg?branch=main)](https://travis-ci.com/vlivashkin/gpuparallel)
[![Documentation Status](https://readthedocs.org/projects/gpuparallel/badge/?version=latest)](https://gpuparallel.readthedocs.io/en/latest/?badge=latest)
[![pypi](https://pypip.in/v/gpuparallel/badge.svg)](https://pypi.python.org/pypi/gpuparallel/)
# GPUParallel
Joblib-like interface for parallel GPU computations (e.g. data preprocessing).

```python
import torch
from gpuparallel import GPUParallel, delayed

def perform(idx, gpu_id, **kwargs):
    tensor = torch.Tensor([idx]).to(gpu_id)
    return (tensor * tensor).item()

result = GPUParallel(n_gpu=2)(delayed(perform)(idx) for idx in range(5))
print(sorted(result))  # [0.0, 1.0, 4.0, 9.0, 16.0]
```

Features:
* [Initialize networks on worker init](#initialize-networks-on-worker-init)
* [Reuse initialized workers](#reuse-initialized-workers)
* [Simple logging from workers](#simple-logging-from-workers)
* Sync mode for tasks debug (use `n_gpu = 0`)
* Progressbar with [tqdm](https://github.com/tqdm/tqdm): `progressbar` flag
* Optional ignoring task errors: `ignore_errors` flag

## Install
```bash
python3 -m pip install gpuparallel
# or
python3 -m pip install git+git://github.com/vlivashkin/gpuparallel.git
```

## Examples
### Initialize networks on worker init
Function `init_fn` is called on init of every worker. All common resources (e.g. networks) can be initialized here.

```python
from gpuparallel import GPUParallel, delayed

def init(gpu_id=None, **kwargs):
    global model
    model = load_model().to(gpu_id)

def perform(img, gpu_id=None, **kwargs):
    global model
    return model(img.to(gpu_id))
    
gp = GPUParallel(n_gpu=16, n_workers_per_gpu=2, init_fn=init)
results = gp(delayed(perform)(img) for img in fnames)
```

### Reuse initialized workers
Once workers are initialized, they keep live until `GPUParallel` object exist.
You can perform several queues of tasks without reinitializing worker resources:

```python
gp = GPUParallel(n_gpu=16, n_workers_per_gpu=2, init_fn=init)
overall_results = []
for folder_images in folders:
    folder_results = gp(delayed(perform)(img) for img in folder_images)
    overall_results.extend(folder_results)
del gp  # this will close process pool to free memory
```

### Simple logging from workers
Inside workers you won't see output of `print()`, but 
Use `log_to_stderr()` call to init logging, and `log.info(message)` to log info from workers
```python
from gpuparallel import GPUParallel, delayed, log_to_stderr, log

log_to_stderr()

def perform(idx, worker_id=None, gpu_id=None):
    hi = f'Hello world #{idx} from worker #{worker_id} with GPU#{gpu_id}!'
    log.info(hi)

GPUParallel(n_gpu=2)(delayed(perform)(idx) for idx in range(2))
```
It will return:
```
[INFO/Worker-1(GPU1)]:Hello world #1 from worker #1 with GPU#1!
[INFO/Worker-0(GPU0)]:Hello world #0 from worker #0 with GPU#0!
```