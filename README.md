# gpu_parallel
Joblib-like interface for parallel GPU computations (e.g. data preprocessing).


## Basic usage
```python
from gpuparallel import GPUParallel
from functools import partial

def perform(idx):
    """
    Function to be performed on worker.
    Variables `worker_id` and `gpu_id` are available globally inside the worker
    """
    global worker_id, gpu_id
    print(f'Hello world #{idx} from worker #{worker_id} with GPU#{gpu_id}!')
    
GPUParallel(n_gpu=2)(partial(perform, idx) for idx in range(100))
```

## Advanced usage
```python
from gpuparallel import GPUParallel
from functools import partial

def init():
    """
    This function will be called on init of every worker.
    All common resources (e.g. networks) can be initialized here.
    """
    global gpu_id, model
    
    model = TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load(PATH))
    model.to(gpu_id)
    model.eval()

def perform(img):
    global gpu_id
    img = img.to(gpu_id)
    result = model(img)
    return result
    
results = GPUParallel(n_gpu=2, n_workers_per_gpu=2, init_fn=init)(partial(perform, idx) for img in images_dataset)
```
