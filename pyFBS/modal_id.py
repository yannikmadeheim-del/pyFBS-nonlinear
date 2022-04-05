import numpy as np
from scipy import linalg
from pyFBS import MAC
from app_stability import MyWindow
from PyQt5 import QtGui, QtCore, QtWidgets

class modal_id(object):
    """
    Poly-reference frequency domain identification for modal parameter estimation as a combination of 
    poly-reference Least-Squares Complex Frequency (pLSCF) and Least-Squares Frequency Domain (LSFD) methods.

    :param freq: Frequency range.
    :type ch: array (float)
    :param FRF: Frequency response function matrix in the form of [frequency points, ouputs, inputs].
    :type refch: array (complex)

    References:     
    -----------
    [1] Guillaume, Patrick, et al. "A poly-reference implementation of the least-squares complex 
        frequency-domain estimator." Proceedings of IMAC. Vol. 21. Kissimmee, FL: A Conference & Exposition 
        on Structural Dynamics, Society for Experimental Mechanics, 2003.
        
    """
    
    def __init__(self, freq, FRF):
        self.FRF = FRF
        self.freq = freq
        self.No = FRF.shape[1]
        self.Ni = FRF.shape[2]
        try:
            from IPython import get_ipython
            get_ipython().magic('gui qt')
        except BaseException as e:
            # issued if code runs in bare Python
            print('Could not enable IPython gui support: %s.' % e)

        # get QApplication instance
        self.app = QtCore.QCoreApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication(['app'])
            self.app.references = set()

    def stabilization(self):  
        self.win = MyWindow(self)
        self.app.references.add(self.win)
        
    def pLSCF(self, max_order, step_order=2, stab_f=0.01, stab_damp=0.05, stab_mpf=0.05):
        """
        Poly-reference Least-Squares Complex Frequency (pLSCF) method for system's poles and modal
        participation factors estimation.

        :param max_order: Highest order of the polynomial basis functions.
        :type pos: int
        :param step_order: Step between two consecutive model orders.
        :type pos: int, optional
        :param stab_f: variation over consecutive model orders of the natural frequency.
        :type pos: float, optional
        :param stab_damp: variation over consecutive model orders of the damping ratio.
        :type pos: float, optional
        :param stab_mpf: variation over consecutive model orders of the modal participation factor.
        :type pos: float, optional
        
        Note: Treat stabilization of modal participation factors with care for models with low number of
        inputs (consider increasing variation criterion).
        """
        
        stab_plot = np.asarray([[0,1e-12,1e-12,0]])
        p = [[np.array([0])]]
        L = [[np.zeros((self.Ni,1))]]
        
        d = self.freq[-1] - self.freq[0]
        r = np.arange(max_order)

        exp_term_ = np.exp(1.j*np.pi/d*np.einsum('ij,k->kij',r[:,np.newaxis],self.freq))
        exp_term = exp_term_@np.conj(exp_term_).transpose(0,2,1)

        R = np.einsum("ijk->jk",exp_term).real
        S = -np.einsum("ijk, ilm", self.FRF, exp_term).transpose(0,2,3,1).reshape(self.No,(max_order),(max_order)*self.Ni).real
        T = np.einsum("ijk, ilm", np.conj(self.FRF).transpose(0,2,1) @ self.FRF,
                      exp_term).transpose(2,0,3,1).reshape((max_order)*self.Ni,(max_order)*self.Ni).real
        
        M = (T - np.sum(S.transpose(0,2,1) @ np.linalg.pinv(R) @ S, axis=0))

        for n_p in np.arange(1,max_order,step_order):
            # alpha coefficients
            alpha = np.block([[np.linalg.solve(-M[:n_p*self.Ni,:n_p*self.Ni],M[:n_p*self.Ni,n_p*self.Ni:(n_p+1)*self.Ni])],
                      [np.eye(self.Ni)]])[:-self.Ni]

            # companion matrix
            A = np.block([[-alpha.reshape(n_p,self.Ni,self.Ni)[::-1].reshape(alpha.shape).T],
                  [np.eye(self.Ni*(n_p-1),self.Ni*n_p)]])

            # poles
            eigval, eigvec = linalg.eig(A)
            eigval_t = -1*np.log(eigval)*2*d
            f_np, damp_np, p_np, L_np = self.transform_poles(eigval_t, eigvec, self.Ni) 

            # append results
            if p_np.size == 0:
                stab_plot = np.vstack((stab_plot, [n_p+1,1e-12,1e-12,0]))
                p.append([np.array([0])])
                L.append([np.zeros((self.Ni,1))])
            else:
                p.append([p_np])
                L.append([L_np])
                stab_plot = np.vstack((stab_plot,
                    self.select_stable_poles(stab_plot,L,n_p,f_np,damp_np,stab_f,stab_damp,stab_mpf)))
        
        self.stab_plot = stab_plot[1:]
        self.poles = p[1:]
        self.mpf = L[1:]

    def pLSFD(self, reconstruction = True, lower_residuals = True, upper_residuals = True):
        """
        With poles are available, the residues can be estimated with a Least-Squares Frequency Domain (LSFD)
        method.

        :param reconstruction: Reconstruct FRF from estimated modal parameters.
        :type reconstruction: bool, optional
        :param lower_residuals: Compute lower residuals.
        :type lower_residuals: bool, optional
        :param upper_residuals: Compute upper residuals.
        :type upper_residuals: bool, optional
        """

        # prepare input data
        s = self.selected_poles[np.newaxis]
        L = self.selected_mpf[np.newaxis]
        w = 2*np.pi*self.freq[:,np.newaxis,np.newaxis]

        # generate P
        p11 = (-s.real*L.real+(w-s.imag)*L.imag) / (s.real**2+(w-s.imag)**2)+\
              (-s.real*L.real+(w+s.imag)*L.imag) / (s.real**2+(w+s.imag)**2)

        p12 = ( s.real*L.imag+(w-s.imag)*L.real) / (s.real**2+(w-s.imag)**2)+\
              (-s.real*L.imag-(w+s.imag)*L.real) / (s.real**2+(w+s.imag)**2)

        p21 = (-s.real*L.imag-(w-s.imag)*L.real) / (s.real**2+(w-s.imag)**2)+\
              (-s.real*L.imag-(w+s.imag)*L.real) / (s.real**2+(w+s.imag)**2)

        p22 = (-s.real*L.real+(w-s.imag)*L.imag) / (s.real**2+(w-s.imag)**2)+\
              ( s.real*L.real-(w+s.imag)*L.imag) / (s.real**2+(w+s.imag)**2)

        P = np.block([[[p11,p12]],[[p21,p22]]])
        
        # lower and upper residuals
        if lower_residuals == True:
            p13_L = np.kron(np.kron(np.eye(self.Ni),np.array([1,0]))[::-1] , -1/w**2)
            p23_L = np.kron(np.kron(np.eye(self.Ni),np.array([0,1]))[::-1] , -1/w**2)
        if upper_residuals == True:
            p13_U = np.kron(np.kron(np.eye(self.Ni),np.array([1,0]))[::-1] ,
                            np.ones(self.freq.shape[0])[:,np.newaxis,np.newaxis])
            p23_U = np.kron(np.kron(np.eye(self.Ni),np.array([0,1]))[::-1] ,
                            np.ones(self.freq.shape[0])[:,np.newaxis,np.newaxis]) 

        if lower_residuals == True and upper_residuals == False:
            P = np.block([[[p11,p12,p13_L]],[[p21,p22,p23_L]]])

        elif lower_residuals == False and upper_residuals == True:
            P = np.block([[[p11,p12,p13_U]],[[p21,p22,p23_U]]])

        elif lower_residuals == True and upper_residuals == True:
            P = np.block([[[p11,p12,p13_L,p13_U]],[[p21,p22,p23_L,p23_U]]])

        else: 
            P = np.block([[[p11,p12]],[[p21,p22]]])

        # get A
        Y_ = np.block([[[self.FRF.real]],[[self.FRF.imag]]])
        A_ = np.linalg.pinv(P.transpose(1,0,2).reshape(-1, P.shape[-1]))@\
             Y_.transpose(2,0,1).reshape(-1, Y_.shape[-2])
        Ar, Ai = np.split(A_, 2)
        A = (Ar + 1.j*Ai).T

        if reconstruction == False:
            self.A = A
        else:
            # frf reconstruction
            Y_rec_ = np.einsum("fip,op->foi", P , A_.T)
            Y_rec_r, Y_rec_i = np.split(Y_rec_, 2)
            Y_rec = (Y_rec_r + 1.j*Y_rec_i)
            self.A = A
            self.FRF_rec = Y_rec
        
    @staticmethod
    def transform_poles(poles, mpf, Ni):
        """
        Obtain natural frequencies, damping ratios, corresponding poles and modal participation factors from all poles. 

        :param poles: Position of the channel.
        :param Ni: Number of inputs.
        :param L: Modal participation factors.
        
        :returns f: Natural frequencies.
        :type f: array(float)
        :returns damp: Damping ratios.
        :type damp: array(float)
        """

        f = np.abs(poles)*np.sign(np.imag(poles))/2/np.pi
        damp = -np.real(poles)/np.abs(poles)
        
        # select physically meaningful poles and modal participation factors
        ind = np.logical_and(f > 0, damp > 0)
        f_pos = f[ind]
        damp_pos = damp[ind]*100
        p = poles[ind]
        L = mpf[-Ni:,ind]

        return f_pos, damp_pos, p, L
    
    @staticmethod
    def select_stable_poles(stab_plot, L, order, freq_n1, damp_n1, stab_f, stab_damp, stab_mpf):
        """
        Find stable poles and prepare plotting data for stabilization plot.

        :param stab_plot: Array of natural frequencies and damping ratios for each polynomial order.
        :type pos: array(float)
        :param L: Mode participation factors for each polynomial order.
        :type pos: list
        :param order: Current polynomial order.
        :type pos: int
        :param freq_n1: Natural frequencies at current polynomial order.
        :type pos: array
        :param damp_n1: Damping ratios at current polynomial order.
        :type pos: int
        :param stab_f: variation over consecutive model orders of the natural frequency.
        :type pos: float
        :param stab_damp: variation over consecutive model orders of the damping ratio.
        :type pos: float
        :param stab_mpf: variation over consecutive model orders of the modal participation factor.
        :type pos: float
        
        :returns new_stab_plot: Array of natural frequencies and damping ratios with last model order added. 
        """

        # prepare datasets
        poles_n = stab_plot[np.where(stab_plot[:,0] == np.amax(stab_plot[:,0]))]
        freq_n = poles_n[:,1]
        damp_n = poles_n[:,2]

        # each nat. freq. from n+1 is compared to all nat. freq. from n order
        freq_n = np.transpose([freq_n]*freq_n1.shape[0])
        # calculate relative error between nat. freq. from n+1 and n for all possible combinations, check where error bellow limit
        ind = np.argmin(np.abs(freq_n1 - freq_n)/freq_n, axis=1)
        _ind = (np.abs(freq_n1 - freq_n)/freq_n)[np.arange(freq_n.shape[0]),ind] < stab_f
        ind_n1 = ind[_ind]
        ind_n = np.arange(freq_n.shape[0])[_ind]

        # all poles are new at first
        pole_type = np.zeros(freq_n1.shape[0])
        # indices where frequency is stable, add 1
        pole_type[ind_n1] += 1
        
        # indices where damping is stable, check only combinations from frequency limit, add 1
        _ind = np.abs(damp_n1[ind_n1] - damp_n[ind_n])/damp_n[ind_n] < stab_damp
        ind_damp_n1 = ind_n1[_ind]
        ind_damp_n = ind_n[_ind]
        pole_type[ind_damp_n1] += 1
        
        # modal participation factor check, check only poles stable in freq and damp, add 1
        ind_L = ind_damp_n1[1 - np.diag(MAC(L[-1][0][:,ind_damp_n1],L[-2][0][:,ind_damp_n]).real) < stab_mpf]
        pole_type[ind_L] += 1
        
        # save poles as list: model order / nat_freq / damp_rat / pole_type (3-stable, 2-stable freq&damp, 1-stable freq, 0-unstable)
        new_stab_plot = np.array([np.repeat(order+1, freq_n1.shape[0]), freq_n1, damp_n1, pole_type]).T.tolist()

        return new_stab_plot
    
    def pL_from_index(self, index):
        """
        Select pole and mode participation factor based on selected pole from stabilization plot.

        :param index: Index of selected pole from stabilization plot data.
        :type index: int
        
        :returns p_sel: Selected pole.
        :type complex:
        :returns L_sel: Selected mode participation factor.
        :type complex:
        """
        i_order = self.stab_plot[index,0]
        n_all = np.argwhere(self.stab_plot[:,0] == i_order)
        p_L_i = np.argwhere(np.unique(self.stab_plot[:,0]) == i_order).ravel()[0]
        n_i = np.argwhere(n_all.flatten() == index).ravel()[0]

        p_sel = self.poles[p_L_i][0][n_i]
        L_sel = self.mpf[p_L_i][0][:,n_i]

        return p_sel, L_sel
