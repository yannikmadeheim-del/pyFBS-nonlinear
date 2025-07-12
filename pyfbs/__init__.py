__version__ = '1.0.0'
import importlib as _importlib

submodules = [
    'data',
    'display',
    'dsp',
    'expansion',
    'fbs',
    'interface',
    'io',
    'mck',
    'modal_id',
    'tpa',
    'utility',
]

__all__ = submodules + [
    '__version__',
]


def __dir__():
    return __all__


def __getattr__(name):
    if name in submodules:
        return _importlib.import_module(f'pyfbs.{name}')
    else:
        raise AttributeError(f"Module 'pyfbs' has no attribute '{name}'")
