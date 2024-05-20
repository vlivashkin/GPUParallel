.. GPUParallel documentation master file, created by
   sphinx-quickstart on Wed Mar 31 15:18:32 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

GPUParallel
===========

Release v\ |version|. (:ref:`Installation <install>`)

Joblib-like interface for parallel GPU computations (e.g. data preprocessing)::

   import torch
   from gpuparallel import GPUParallel, delayed

   def perform(idx, gpu_id, **kwargs):
       tensor = torch.Tensor([idx]).to(gpu_id)
       return (tensor * tensor).item()

   result = GPUParallel(n_gpu=2)(delayed(perform)(idx) for idx in range(5))
   print(sorted(result))  # [0.0, 1.0, 4.0, 9.0, 16.0]

Features
--------

- :ref:`Initialize networks on worker init`
- :ref:`Reuse initialized workers`
- :ref:`Simple logging from workers`
- Sync mode for tasks debug (use ``n_gpu = 0``)
- Progressbar with tqdm_: ``progressbar=True``
- Optional ignoring task errors: ``ignore_errors=True``

See :ref:`Quickstart` and :ref:`API Reference` for details.

.. _tqdm: https://github.com/tqdm/tqdm

User Guide
----------

.. toctree::
   :maxdepth: 2

   install
   quickstart
   api
