# GPUParallel
Joblib-like interface for parallel GPU computations (e.g. data preprocessing).

## Install
```bash
git clone https://github.com/vlivashkin/gpu_parallel.git
cd gpu_parallel
python3 -m pip install .
```

## Basic usage
```python
from gpuparallel import GPUParallel
from functools import partial

def perform(idx, worker_id=None, gpu_id=None):
    """
    Function to be performed on worker. Variables `worker_id` and `gpu_id` will be filled 
    automatically with actual values of a current worker.
    """
    print(f'Hello world #{idx} from worker #{worker_id} with GPU#{gpu_id}!')
    
GPUParallel(n_gpu=2)(partial(perform, idx) for idx in range(100))
```

## Advanced usage
```python
from gpuparallel import GPUParallel
from functools import partial

def init(worker_id=None, gpu_id=None):
    """
    This function will be called on init of every worker.
    All common resources (e.g. networks) can be initialized here.
    """
    global model
    
    model = TheModelClass()
    model.load_state_dict(torch.load(PATH))
    model.to(gpu_id)
    model.eval()

def perform(img, worker_id=None, gpu_id=None):
    global model
    
    img = img.to(gpu_id)
    result = model(img)
    return result
    
gp = GPUParallel(n_gpu=16, n_workers_per_gpu=2, init_fn=init)
results = gp(partial(perform, idx) for img in images_dataset)
```
