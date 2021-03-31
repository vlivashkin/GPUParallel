[![Build Status](https://travis-ci.com/vlivashkin/GPUParallel.svg?branch=main)](https://travis-ci.com/vlivashkin/gpuparallel)
[![Documentation Status](https://readthedocs.org/projects/gpuparallel/badge/?version=latest)](https://gpuparallel.readthedocs.io/en/latest/?badge=latest)
# GPUParallel
Joblib-like interface for parallel GPU computations (e.g. data preprocessing).


## Install
```bash
python3 -m pip install gpuparallel
# or
python3 -m pip install git+git://github.com/vlivashkin/gpuparallel.git
```

## Basic usage
```python
from gpuparallel import GPUParallel
from functools import partial
import multiprocessing as mp

mp.log_to_stderr()
mp.get_logger().setLevel('INFO')

def perform(idx, worker_id=None, gpu_id=None):
    """
    Function to be performed on worker. Variables `worker_id` and `gpu_id` will be
    filled automatically with actual values of a current worker.
    """
    hi = f'Hello world #{idx} from worker #{worker_id} with GPU#{gpu_id}!'
    mp.get_logger().info(hi)

GPUParallel(n_gpu=2)(partial(perform, idx) for idx in range(100))
```

## Advanced usage
```python
from gpuparallel import GPUParallel
from functools import partial

def init(gpu_id=None, **kwargs):
    """
    This function will be called on init of every worker.
    All common resources (e.g. networks) can be initialized here.
    """
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
    folder_results = gp(partial(perform, img) for img in folder_images)
    overall_results.extend(folder_results)
```
