# gpu_parallel
Joblib-like interface for parallel GPU computations (e.g. data preprocessing).


## Basic usage
```python
from gpuparallel import GPUParallel
from functools import partial

def perform(idx, worker_id=None, gpu_id=None):
    """
    Function to be performed on worker.
    Variables `worker_id` and `gpu_id` are available globally inside the worker
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
    
    model = TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load(PATH))
    model.to(gpu_id)
    model.eval()

def perform(img, worker_id=None, gpu_id=None):
    global model
    
    img = img.to(gpu_id)
    result = model(img)
    return result
    
parallel = GPUParallel(n_gpu=2, n_workers_per_gpu=2, init_fn=init)
results = parallel(partial(perform, idx) for img in images_dataset)
```
