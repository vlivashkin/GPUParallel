[![Build Status](https://travis-ci.com/vlivashkin/GPUParallel.svg?branch=main)](https://travis-ci.com/vlivashkin/gpuparallel)
[![codecov](https://codecov.io/gh/vlivashkin/GPUParallel/branch/main/graph/badge.svg?token=eo2uyiDmj1)](https://codecov.io/gh/vlivashkin/GPUParallel)
[![Documentation Status](https://readthedocs.org/projects/gpuparallel/badge/?version=latest)](https://gpuparallel.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/gpuparallel.svg)](https://badge.fury.io/py/gpuparallel)
# GPUParallel
Joblib-like interface for parallel GPU computations (e.g. data preprocessing).

```python
import torch
from gpuparallel import GPUParallel, delayed

def perform(idx, device_id, **kwargs):
    tensor = torch.Tensor([idx]).to(device_id)
    return (tensor * tensor).item()

result = GPUParallel(n_gpu=2)(delayed(perform)(idx) for idx in range(5))
print(list(result))  # result: [0.0, 1.0, 4.0, 9.0, 16.0], ordered in accordance with input parameters
```

Features:
* [Initialize networks once on worker init](#initialize-networks-once-on-worker-init)
* [Reuse initialized workers](#reuse-initialized-workers)
* Preserve sample order: `preserve_order` flag, turned on by default
* [Result is a generator](#result-is-a-generator)
* [Auto batching](#auto-batching)
* [Simple logging from workers](#simple-logging-from-workers)
* Main process inference mode for tasks debug (use `debug = True`)
* Progressbar with [tqdm](https://github.com/tqdm/tqdm): `progressbar` flag
* Optional ignoring task errors: `ignore_errors` flag

## Install
```bash
python3 -m pip install gpuparallel
# or
python3 -m pip install git+git://github.com/vlivashkin/gpuparallel.git
```

## Examples
### Initialize networks once on worker init
Function `init_fn` is called on init of every worker. All common resources (e.g. networks) can be initialized here.

```python
from gpuparallel import GPUParallel, delayed

def init(device_id=None, **kwargs):
    global model
    model = load_model().to(device_id)

def perform(img, device_id=None, **kwargs):
    global model
    return model(img.to(device_id))
    
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

### Result is a generator
GPUParallel call returns a generator to use results during caclulations (e.g. for sequential saving ordered results)

```python
import h5py

gp = GPUParallel(n_gpu=16, n_workers_per_gpu=2, preserve_order=True)
result = gp(delayed(perform)(img) for img in images)

with h5py.File('output.h5') as f:
    result_dataset = f.create_dataset('result', shape=(300, 224, 224, 3))

    for idx, result in enumerate(result):
        result_dataset[idx] = result
```

### Auto batching
Use class `BatchGPUParallel` for auto spliting tensor to workers.
`flat_result` flag de-batches results (works only if single array/tensor returned)

```python
arr = np.zeros((102, 103))
bgpup = BatchGPUParallel(task_fn=task, batch_size=3, flat_result=True, n_gpu=2)
flat_results = np.array(list(bgpup(arr)))
```

### Simple logging from workers
`print()` inside a worker won't be seen in the main process, but you still can use logging to stderr of the main process.
Use `log_to_stderr()` call to init logging, and `log.info(message)` to log info from workers
```python
from gpuparallel import GPUParallel, delayed, log_to_stderr, log

log_to_stderr('INFO')

def perform(idx, worker_id=None, device_id=None):
    hi = f'Hello world #{idx} from worker #{worker_id} with {device_id}!'
    log.info(hi)

GPUParallel(n_gpu=2)(delayed(perform)(idx) for idx in range(2))
```
It will return:
```
[INFO/Worker-1(cuda:1)]:Hello world #1 from worker #1 with cuda:1!
[INFO/Worker-0(cuda:0)]:Hello world #0 from worker #0 with cuda:0!
```
