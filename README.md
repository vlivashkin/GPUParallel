[![Build Status](https://travis-ci.com/vlivashkin/GPUParallel.svg?branch=main)](https://travis-ci.com/vlivashkin/gpuparallel)
[![Documentation Status](https://readthedocs.org/projects/gpuparallel/badge/?version=latest)](https://gpuparallel.readthedocs.io/en/latest/?badge=latest)
[![pypi](https://pypip.in/v/gpuparallel/badge.svg)](https://pypi.python.org/pypi/gpuparallel/)
# GPUParallel
Joblib-like interface for parallel GPU computations (e.g. data preprocessing).

```python
from gpuparallel import GPUParallel, delayed

def perform(idx, **kwargs):
    return idx * idx

result = GPUParallel(n_gpu=2)(delayed(perform)(idx) for idx in range(5))
print(sorted(result))  # [0, 1, 4, 9, 16]
```

## Initialize networks on worker init
Function `init` is called on init of every worker. All common resources (e.g. networks) can be initialized here.

```python
from gpuparallel import GPUParallel, delayed

def init(gpu_id=None, **kwargs):
    global model
    
    model = TheModel()
    model.load_state_dict(torch.load(PATH))
    model.to(gpu_id)
    model.eval()

def perform(img, gpu_id=None, **kwargs):
    global model
    
    img = img.to(gpu_id)
    result = model(img)
    return result
    
gp = GPUParallel(n_gpu=16, n_workers_per_gpu=2, init_fn=init)
overall_results = []
for folder_images in folders:
    folder_results = gp(delayed(perform)(img) for img in folder_images)
    overall_results.extend(folder_results)
```

## Simple logging from workers
```python
from gpuparallel import GPUParallel, delayed, log_to_stderr, log

log_to_stderr()

def perform(idx, worker_id=None, gpu_id=None):
    hi = f'Hello world #{idx} from worker #{worker_id} with GPU#{gpu_id}!'
    log.info(hi)

GPUParallel(n_gpu=2)(delayed(perform)(idx) for idx in range(2))
```
will return:
```
[INFO/Worker-1(GPU1)]:Hello world #1 from worker #1 with GPU#1!
[INFO/Worker-0(GPU0)]:Hello world #0 from worker #0 with GPU#0!
```

## Install
```bash
python3 -m pip install gpuparallel
# or
python3 -m pip install git+git://github.com/vlivashkin/gpuparallel.git
```