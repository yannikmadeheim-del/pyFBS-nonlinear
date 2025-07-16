__version__ = '1.0.0'
import importlib as _importlib

_submodules = [
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

_utility_functions = [
    'mcf',
    'mac',
    'coh',
    'lac',
    'cmif',
    'tsvd',
    'tpinv',
    'reciprocity',
]

__all__ = (
    _submodules
    + _utility_functions
    + [
        '__version__',
    ]
)


_new_structure_dict = {
    'view3D': 'display.View3D',
    'SEMM': 'expansion.semm',
    'SEREP': 'expansion.serep',
    'M_SEMM': 'expansion.m_semm',
    'VPT': 'interface.VPT',
    'SVT': 'interface.SVT',
    'MK_model': 'mck.Model',
    'OSI': 'tpa',
}

_new_utility_dict = {
    'modeshape_sync_lstsq': 'utility.modeshape_sync_lstsq',
    'modeshape_scaling_DP': 'utility.modeshape_scaling_dp',
    'MCF': 'mcf',
    'flatten_FRFs': 'utility.flatten_frfs',
    'unflatten_modes': 'utility.unflatten_modes',
    'mode_animation': 'utility.mode_animation',
    'MAC': 'mac',
    'coh_frf': 'utility.coh_frf',
    'dict_animation': 'utility.dict_animation',
    'CMIF': 'cmif',
    'TSVD': 'tsvd',
    'M': 'utility.rotation_matrix',
    'angle': 'utility.angle',
    'rotation_matrix_from_vectors': 'utility.rotation_matrix_from_vectors',
    'unit_vector': 'utility.unit_vector',
    'angle_between': 'utility.angle_between',
    'generate_channels_from_sensors': 'utility.generate_channels_from_sensors',
    'generate_sensors_from_channels': 'utility.generate_sensors_from_channels',
    'generate_VP_from_position': 'utility.generate_vp_from_position',
    'coh_on_FRF': 'reciprocity',
    'orient_in_global': 'utility.orient_in_global',
    'orient_in_global_2': 'utility.orient_in_global_2',
    'MCC': 'utility.mcc',
    'MPC': 'utility.mpc',
    'auralization': 'utility.auralization',
    'SSA_filter': 'utility.ssa_filter',
    'SSA_evaluate': 'utility.ssa_evaluate',
    'PRF': 'utility.prf',
    'ODS_FRF': 'utility.ods_frf',
    'ODS_FRF_averaging': 'utility.ods_frf_averaging',
}

_new_display_dict = {
    'barchart': 'barchart',
    'imshow': 'imshow',
    'complex_plot': 'complex_plot',
    'complex_plot_3D': 'complex_plot_3d',
    'plot_FRF': 'plot_frf',
    'plot_frequency_response': 'plot_frequency_response',
    'comparison_plot': 'comparison_plot',
    'plot_comparison_multiple': 'plot_comparison_multiple',
    'plot_coh': 'plot_coh',
    'plot_coh_group': 'plot_coh_group',
    'tranfer_path': 'tranfer_path',
    'contour_plot': 'contour_plot',
    'runup_data': 'runup_data',
}


def _attribute_error(name):
    if name in _new_structure_dict:
        new_name = _new_structure_dict[name]
    elif name in _new_utility_dict:
        new_name = f'{_new_utility_dict[name]}'
    elif name in _new_display_dict:
        new_name = f'display.{_new_display_dict[name]}'
    else:
        raise AttributeError(f"Module 'pyfbs' has no attribute '{name}'")

    if name == 'MK_model':
        raise AttributeError(
            f"'pyFBS.{name}' has been replaced by 'pyfbs.{new_name}'. "
            "To instantiate a model based on Ansys results, use "
            "'pyfbs.mck.Model.from_ansys()' instead."
        )
    raise AttributeError(
        f"'pyFBS.{name}' has been replaced by 'pyfbs.{new_name}'. "
    )


def __dir__():
    return __all__


def __getattr__(name):
    if name in _submodules:
        return _importlib.import_module(f'pyfbs.{name}')
    elif name in _utility_functions:
        pyfbs_utility = _importlib.import_module('pyfbs.utility')
        return getattr(pyfbs_utility, name)
    else:
        _attribute_error(name)
