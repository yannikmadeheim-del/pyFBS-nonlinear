from ..utility import MAC
from .tools import find_locations_in_data_frames
from .serep import serep
import numpy as np
import scipy as sp

def m_semm(eig_val_num, damping_num, eig_vec_num, eig_val_exp, damping_exp, eig_vec_exp, df_chn_num, df_chn_exp, SEMM_type = 'extended', tsvd_rcond = 0, ns_rcond = 1.e-12):
    """
    This function performs M-SEMM, System Equivalent Model Mixing in modal domain. 
    Hybrid model is generated from a numerical and an experimental models using a substructuring approach. 

    :param eig_val_num: Eigenvalues of the numerical model.
    :type eig_val_num: array(float)
    :param damping_num: Modal damping ratios of the numerical model.
    :type damping_num: array(float)
    :param eig_vec_num: Mass normalized eigenvectors of the numerical model.
    :type eig_vec_num: array(float)
    
    :param eig_val_exp: Eigenvalues of the experimental model.
    :type eig_val_exp: array(float)
    :param damping_exp: Modal damping ratios of the experimental model.
    :type damping_exp: array(float)
    :param eig_vec_exp: Mass normalized eigenvectors of the experimental model.
    :type eig_vec_exp: array(float)
    
    :param df_chn_num: Response locations and directions for the numerical model.
    :type df_chn_num: pandas.DataFrame
    :param df_chn_exp: Response locations and directions for the experimental model.
    :type df_chn_exp: pandas.DataFrame

    :param SEMM_type: Selection of the applied decoupling constraints (compatibility and equilibrium).
    :type SEMM_type: str('basic', 'extended')
    :param tsvd_rcond: Relative condition number for small singular value trucation tp weaken the extended decoupling constraints.
    :type tsvd_rcond: float
    :param ns_rcond: Relative condition number for the null-space hybrid eigenvector evaluation.
    :type ns_rcond: float

    :return: hybrid modal parameters (eigenvalues, damping ratios, eigenvectors)
    :rtype: array(float)
    """

    # Initialization data
    eig_val_num = np.asarray(np.copy(eig_val_num)).ravel().astype(float)
    damping_num = np.asarray(np.copy(damping_num)).ravel().astype(float)
    eig_val_exp = np.asarray(np.copy(eig_val_exp)).ravel().astype(float)
    damping_exp = np.asarray(np.copy(damping_exp)).ravel().astype(float)
    
    eig_vec_num = np.asarray(np.copy(eig_vec_num)).astype(float)
    eig_vec_exp = np.asarray(np.copy(eig_vec_exp)).astype(float)

    # Input data validation
    if df_chn_num.shape[0] != eig_vec_num.shape[0]:
        raise Exception('Numerical model - Incompatible channel data and eigenvector shape.')
    if df_chn_exp.shape[0] != eig_vec_exp.shape[0]:
        raise Exception('Experimental model - Incompatible channel data and eigenvector shape.')
    if not eig_val_num.shape[0] == damping_num.shape[0] == eig_vec_num.shape[1]:
        raise Exception('Numerical model - Number of modes at the given modal parameters does not match.')
    if not eig_val_exp.shape[0] == damping_exp.shape[0] == eig_vec_exp.shape[1]:
        raise Exception('Numerical model - Number of modes at the given modal parameters does not match.')

    # Internal and boundary DoF partition
    maching_locations_chn = find_locations_in_data_frames(df_chn_num, df_chn_exp)
    if maching_locations_chn.shape[0] != df_chn_exp.shape[0]:
        raise Exception('Not all experimental channel locations have a matching location in the numerical channel dataframe.')

    b_dof_ind_num = np.copy(maching_locations_chn[:,0])
    b_dof_ind_exp = np.copy(maching_locations_chn[:,1])
    i_dof_ind_num = np.setdiff1d(np.arange(eig_vec_num.shape[0]),b_dof_ind_num)

    n_i = i_dof_ind_num.shape[0]
    n_b = b_dof_ind_num.shape[0]
    n_g = n_i + n_b

    # Parent and overlay model generation 
    # (removed model not needed due to duplicity with the parent model)
    eig_val_par = np.copy(eig_val_num)
    xi_par = np.copy(damping_num)
    eig_vec_par = np.copy(eig_vec_num)[np.hstack([i_dof_ind_num,b_dof_ind_num]),:]

    eig_val_ov = np.copy(eig_val_exp)
    xi_ov = np.copy(damping_exp)
    eig_vec_ov = np.copy(eig_vec_exp)[b_dof_ind_exp,:]

    m_par = eig_val_par.shape[0]     
    m_ov = eig_val_ov.shape[0]

    # (Uncoupled) modal matrices - assume proportional viscous damping
    M_m_par = np.eye(m_par)
    C_m_par = np.diag(2*xi_par*eig_val_par**0.5)
    K_m_par = np.diag(eig_val_par)

    M_m_ov = np.eye(m_ov)
    C_m_ov = np.diag(2*xi_ov*eig_val_ov**0.5)
    K_m_ov = np.diag(eig_val_ov)

    if n_b >= m_par:
        print('Redirected to SEREP due to the mathematical equivalence.')
        eig_vec_serep = serep(eig_vec_num, eig_vec_exp, df_chn_num, df_chn_exp)
        return eig_val_ov.real, xi_ov.real, eig_vec_serep.real
        
    else:
        # Number of spurious modes
        n_s = m_par - n_b 
        
        # Coupling step
        P_1 = eig_vec_par[-n_b:]
        P_2 = eig_vec_ov

        P_12 = np.linalg.pinv(P_1) @ P_2
        N_1 = sp.linalg.null_space(P_1)
        B_c = np.block([np.zeros([n_b,m_par]) , eig_vec_par[-n_b:] , -eig_vec_ov])

        # Decoupling step
        def generate_N3(eigval_ov):
            if SEMM_type == 'basic':
                P3 = np.copy(P_1)
                N3 = np.copy(N_1)

            elif SEMM_type == 'extended':
                denom = eig_val_par - eigval_ov

                if np.isclose(denom,0).any():
                    u = eig_vec_par[:,np.where(np.isclose(denom,0))[0]]
                    W = u.T
                else:
                    inp = np.einsum('kij,k->ij',np.einsum("ik,jk->kij",eig_vec_par,eig_vec_par[-n_b:]),1/denom)
                    u = sp.linalg.orth(inp, tsvd_rcond)
                    W = u.T

                P3 = W @ eig_vec_par
                N3 = sp.linalg.null_space(P3)

            return N3
        
        def generate_Dsemm(Dov,Dpar,N3):
            return np.block([[Dov,                 np.zeros([m_ov,n_s]),                 P_12.T @ Dpar @ N3],
                            [np.zeros([n_s,m_ov]), np.diag(np.random.random(n_s)*1e-20), N_1.T @ Dpar @ N3],
                            [N3.T @ Dpar @ P_12,   N3.T @ Dpar @ N_1,                    N3.T @ Dpar @ N3]])
        
        def generate_Lpar(N3):
            return np.block([P_12, N_1, N3])
    
        # Output data initialization
        eig_val_semm = np.empty(0)
        xi_semm = np.empty(0)
        eig_vec_semm = np.empty((n_g,0))

        # Copy just to be on the safe side
        _eig_val_ov = np.copy(eig_val_ov)
        _eig_vec_ov = np.copy(eig_vec_ov)

        # Solve for hybrid modal parameters
        for k, (lam_k, phi_ov_k) in enumerate(zip(_eig_val_ov, _eig_vec_ov.T)):
            # One evaluation for 'basic' and "frequency" dependent evaluations for 'extended' formulation 
            if SEMM_type == 'basic' and k == 0 or SEMM_type == 'extended':
                N_3 = generate_N3(lam_k)
                L_par = generate_Lpar(N_3)            

                M_semm = generate_Dsemm(M_m_ov, M_m_par, N_3)
                C_semm = generate_Dsemm(C_m_ov, C_m_par, N_3)
                K_semm = generate_Dsemm(K_m_ov, K_m_par, N_3)

            try:
                _phi_k = sp.linalg.null_space(K_semm - lam_k * M_semm, rcond = ns_rcond)
                np.argmax(_phi_k)
            except:
                # Just in case the null-space evaluation fails
                print('Eigenvalue solver employed.')
                lam_k, _phi_k = sp.sparse.linalg.eigs(K_semm, k=1, M=M_semm, sigma = lam_k)
            
            # mass normalization              
            _phi_k /= np.diagonal(_phi_k.T @ M_semm @ _phi_k)**0.5
            phi_k = eig_vec_par @ L_par @ _phi_k

            if phi_k.shape[1] > 1:
                #print('Spurious solution detected for mode ' + str(k))
                true_ind = [np.argmax(MAC(phi_k[-n_b:],phi_ov_k, output_type = 'diagonal').real)]
                phi_k = phi_k[:,true_ind]
       
            xi_k = np.diagonal(_phi_k.T @ C_semm @ _phi_k) / 2 / lam_k**0.5
             
            eig_val_semm = np.append(eig_val_semm, lam_k.ravel())
            xi_semm = np.append(xi_semm, xi_k.ravel())
            eig_vec_semm = np.append(eig_vec_semm, phi_k, axis = 1)

        # Reordering to input dof order
        reord_ind = np.argsort(np.hstack([i_dof_ind_num,b_dof_ind_num]))
        eig_vec_semm = eig_vec_semm[reord_ind,:]

        return eig_val_semm.real, xi_semm.real, eig_vec_semm.real