# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright (c) 2023 by Ace, All Rights Reserved.  

from mmengine.registry import Registry
from mmengine.registry import LOOPS as MMENGINE_LOOPS
from mmengine.registry import HOOKS as MMENGINE_HOOKS
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import METRICS as MMENGINE_METRICS
from mmengine.registry import RUNNERS as MMENGINE_RUNNERS
from mmengine.registry import DATASETS as MMENGINE_DATASETS
from mmengine.registry import EVALUATOR as MMENGINE_EVALUATOR
from mmengine.registry import OPTIMIZERS as MMENGINE_OPTIMIZERS
from mmengine.registry import TASK_UTILS as MMENGINE_TASK_UTILS
from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS
from mmengine.registry import VISBACKENDS as MMENGINE_VISBACKENDS
from mmengine.registry import VISUALIZERS as MMENGINE_VISUALIZERS
from mmengine.registry import DATA_SAMPLERS as MMENGINE_DATA_SAMPLERS
from mmengine.registry import MODEL_WRAPPERS as MMENGINE_MODEL_WRAPPERS
from mmengine.registry import OPTIM_WRAPPERS as MMENGINE_OPTIM_WRAPPERS
from mmengine.registry import LOG_PROCESSORS as MMENGINE_LOG_PROCESSORS
from mmengine.registry import PARAM_SCHEDULERS as MMENGINE_PARAM_SCHEDULERS
from mmengine.registry import RUNNER_CONSTRUCTORS as MMENGINE_RUNNER_CONSTRUCTORS
from mmengine.registry import WEIGHT_INITIALIZERS as MMENGINE_WEIGHT_INITIALIZERS
from mmengine.registry import OPTIM_WRAPPER_CONSTRUCTORS as MMENGINE_OPTIM_WRAPPER_CONSTRUCTORS


SCOPE = 'alchemy'

# manage all kinds of runners like `EpochBasedRunner` and `IterBasedRunner`
RUNNERS = Registry(
    'runner', 
    scope=SCOPE, 
    parent=MMENGINE_RUNNERS, 
    locations=['mmdet.engine.runner', 'mmdet3d.engine']
    )

# manage runner constructors that define how to initialize runners
RUNNER_CONSTRUCTORS = Registry(
    'runner constructor', 
    scope=SCOPE,
    parent=MMENGINE_RUNNER_CONSTRUCTORS, 
    locations=['mmdet.engine.runner', 'mmdet3d.engine']
    )

# manage all kinds of loops like `EpochBasedTrainLoop`
LOOPS = Registry(
    'loop', 
    scope=SCOPE, 
    parent=MMENGINE_LOOPS, 
    locations=['mmdet.engine.runner', 'mmdet3d.engine']
    )

# manage all kinds of hooks like `CheckpointHook`
HOOKS = Registry(
    'hook', 
    scope=SCOPE, 
    parent=MMENGINE_HOOKS, 
    locations=['mmdet.engine.hooks', 'mmdet3d.engine.hooks'])

# manage data-related modules
DATASETS = Registry(
    'dataset', 
    scope=SCOPE,
    parent=MMENGINE_DATASETS, 
    locations=['mmdet.datasets', 'mmdet3d.datasets']
    )

DATA_SAMPLERS = Registry(
    'data sampler', 
    scope=SCOPE, 
    parent=MMENGINE_DATA_SAMPLERS, 
    locations=['mmdet.datasets.samplers', 'mmdet3d.datasets']
    )

TRANSFORMS = Registry(
    'transform', 
    scope=SCOPE, 
    parent=MMENGINE_TRANSFORMS, 
    locations=['mmdet.datasets.transforms', 'mmdet3d.datasets.transforms']
    )

# manage all kinds of modules inheriting `nn.Module`
MODELS = Registry(
    'model',
    scope=SCOPE,
    parent=MMENGINE_MODELS, 
    locations=['mmdet.models', 'alchemy.models', 'mmdet3d.models']
    )

# manage all kinds of model wrappers like 'MMDistributedDataParallel'
MODEL_WRAPPERS = Registry(
    'model_wrapper', 
    scope=SCOPE, 
    parent=MMENGINE_MODEL_WRAPPERS, 
    locations=['mmdet.models', 'mmdet3d.models']
    )

# manage all kinds of weight initialization modules like `Uniform`
WEIGHT_INITIALIZERS = Registry(
    'weight initializer', 
    scope=SCOPE, 
    parent=MMENGINE_WEIGHT_INITIALIZERS, 
    locations=['mmdet.models', 'mmdet3d.models']
    )

# manage all kinds of optimizers like `SGD` and `Adam`
OPTIMIZERS = Registry(
    'optimizer', 
    scope=SCOPE, 
    parent=MMENGINE_OPTIMIZERS, 
    locations=['mmdet.engine.optimizers', 'mmdet3d.engine']
    )

# manage optimizer wrapper
OPTIM_WRAPPERS = Registry(
    'optim_wrapper', 
    scope=SCOPE, 
    parent=MMENGINE_OPTIM_WRAPPERS, 
    locations=['mmdet.engine.optimizers', 'mmdet3d.engine']
    )

# manage constructors that customize the optimization hyperparameters.
OPTIM_WRAPPER_CONSTRUCTORS = Registry(
    'optimizer constructor', 
    scope=SCOPE,
    parent=MMENGINE_OPTIM_WRAPPER_CONSTRUCTORS,
    locations=['mmdet.engine.optimizers', 'mmdet3d.engine']
    )

# manage all kinds of parameter schedulers like `MultiStepLR`
PARAM_SCHEDULERS = Registry(
    'parameter scheduler',
    scope=SCOPE,
    parent=MMENGINE_PARAM_SCHEDULERS,
    locations=['mmdet.engine.schedulers', 'mmdet3d.engine']
    )

# manage all kinds of metrics
METRICS = Registry(
    'metric', 
    scope=SCOPE, 
    parent=MMENGINE_METRICS, 
    locations=['mmdet.evaluation', 'alchemy.metrics', 'mmdet3d.evaluation']
    )

# manage evaluator
EVALUATOR = Registry(
    'evaluator', 
    scope=SCOPE, 
    parent=MMENGINE_EVALUATOR, 
    locations=['mmdet.evaluation', 'mmdet3d.evaluation']
    )

# manage task-specific modules like anchor generators and box coders
TASK_UTILS = Registry(
    'task util', 
    scope=SCOPE, 
    parent=MMENGINE_TASK_UTILS, 
    locations=['mmdet.models', 'mmdet3d.models', 'alchemy.models']
    )

# manage visualizer
VISUALIZERS = Registry(
    'visualizer',
    scope=SCOPE,
    parent=MMENGINE_VISUALIZERS,
    locations=['mmdet.visualization', 'mmdet3d.visualization']
    )

# manage visualizer backend
VISBACKENDS = Registry(
    'vis_backend',
    scope=SCOPE,
    parent=MMENGINE_VISBACKENDS,
    locations=['mmdet.visualization', 'mmdet3d.visualization'])

# manage logprocessor
LOG_PROCESSORS = Registry(
    'log_processor',
    parent=MMENGINE_LOG_PROCESSORS,
    scope=SCOPE,
    locations=['mmdet.engine', 'mmdet3d.engine']
    )
