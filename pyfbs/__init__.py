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

__new_utility_dict = {
    'modeshape_sync_lstsq': 'modeshape_sync_lstsq',
    'modeshape_scaling_DP': 'modeshape_scaling_dp',
    'MCF': 'mcf',
    'flatten_FRFs': 'flatten_frfs',
    'unflatten_modes': 'unflatten_modes',
    'mode_animation': 'mode_animation',
    'MAC': 'mac',
    'coh_frf': 'coh_frf',
    'dict_animation': 'dict_animation',
    'CMIF': 'cmif',
    'TSVD': 'tsvd',
    'M': 'rotation_matrix',
    'angle': 'angle',
    'rotation_matrix_from_vectors': 'rotation_matrix_from_vectors',
    'unit_vector': 'unit_vector',
    'angle_between': 'angle_between',
    'generate_channels_from_sensors': 'generate_channels_from_sensors',
    'generate_sensors_from_channels': 'generate_sensors_from_channels',
    'generate_VP_from_position': 'generate_vp_from_position',
    'coh_on_FRF': 'reciprocity',
    'orient_in_global': 'orient_in_global',
    'orient_in_global_2': 'orient_in_global_2',
    'MCC': 'mcc',
    'MPC': 'mpc',
    'auralization': 'auralization',
    'SSA_filter': 'ssa_filter',
    'SSA_evaluate': 'ssa_evaluate',
    'PRF': 'prf',
    'ODS_FRF': 'ods_frf',
    'ODS_FRF_averaging': 'ods_frf_averaging',
}

__new_display_dict = {
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


def __attribute_error__(name):
    if name in __new_structure_dict:
        new_name = __new_structure_dict[name]
    elif name in __new_utility_dict:
        new_name = f'utility.{__new_utility_dict[name]}'
    elif name in __new_display_dict:
        new_name = f'display.{__new_display_dict[name]}'
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
    if name in submodules:
        return _importlib.import_module(f'pyfbs.{name}')
    else:
        __attribute_error__(name)
