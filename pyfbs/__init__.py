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


__new_structure_dict = {
    'view3D': 'display.View3D',
    'SEMM': 'expansion.semm',
    'SEREP': 'expansion.serep',
    'M_SEMM': 'expansion.m_semm',
    'VPT': 'interface.VPT',
    'SVT': 'interface.SVT',
    'MK_model': 'mck.Model',
    'OSI': 'tpa',
}


def __attribute_error__(name):
    if name in __new_structure_dict:
        new_name = __new_structure_dict[name]
        if name == 'MK_model':
            raise AttributeError(
                f"'pyFBS.{name}' has been replaced by 'pyfbs.{new_name}'. "
                "To instantiate a model based on Ansys results, use "
                "'pyfbs.mck.Model.from_ansys()' instead."
            )
        raise AttributeError(
            f"'pyFBS.{name}' has been replaced by 'pyfbs.{new_name}'. "
        )
    else:
        raise AttributeError(f"Module 'pyfbs' has no attribute '{name}'")


def __dir__():
    return __all__


def __getattr__(name):
    if name in submodules:
        return _importlib.import_module(f'pyfbs.{name}')
    else:
        __attribute_error__(name)
