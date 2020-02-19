from qutip import *
import numpy as np
from scipy.optimize import minimize
from scipy.stats import unitary_group
import random

class Opt_Param():

    def __init__(self, init_state, suc_sqg, suc_tqg, incoherent_noise, noise_prob, depth, method):
        """
        Class is instantiated with the following parameters :

        init_state : The initial state on which the gates act. If set to 'rand', a random normalized
                     vector is generated.
        suc_sqg    : The success probabilities of single qubit gates, we have four single qubit gates,
                     an array of values between 0. and 1. determine the success of the four gates,
                     with 1. mapping the gate to be perfect and 0. mapping the gate to completely
                     imperfect.
        suc_tqg    : The success probability of the only two qubit gate, CNOT. As above, 1. implies
                     perfect gate, 0. implies imperfect gate.

        Both `suc_sqg` and `suc_tqg` are used to capture coherent noise. If 1., there is no noise added
        else a random unitary acts along with the gate.

        incoherent_noise : Currently, only two qubit dephasing and amplitude dampening channels are supported.
                           Only one of the dephasing and amplitude dampening channel can be applied for CNOT.
                           WIP : Single qubit channels inducing incoherent noise are not supported.

        noise_prob : The noise for the two qubit channel, an array with two channel probabilities, see `Kraus_dephase`,
                     `Kraus_amp_damp` below.

        depth : The number of times the sequence of gates should be applied, i.e., number of layers of similar unitary
                blocks.

        method : a. 'overlap' computes the absolute of the overlap with target state, i.e., the bell state.
                 b. 'expect' computes the expectation value with respect to the operator optimizes the measure of the bell state.
        """
        if init_state == 'rand':
            rc = [random.random() for j in range(4)]
            init_state_1 = (rc[0]*ket("0") + rc[1]*ket("1")).unit()
            init_state_2 = (rc[2]*ket("0") + rc[3]*ket("1")).unit()
            self._init_state = tensor(init_state_1, init_state_2)
        else:
            self._init_state = init_state
        self._suc_sqg = suc_sqg
        self._suc_tqg = suc_tqg
        self._incoherent_noise = incoherent_noise
        self._noise_prob = noise_prob
        self._depth = depth
        self._method = method

    def R_x(self, p):
        """
        R_x gate

        Params
        ======

        p  : The angle which generates the unitary.
        """
        return Qobj([[np.cos(p/2.), 0.-1j*np.sin(p/2.)], [0.-1j*np.sin(p/2.), np.cos(p/2)]], dims=[[2], [2]])

    def R_y(self, p):
        """
        R_y gate

        Params
        ======

        p  : The angle which generates the unitary.
        """
        return Qobj([[np.cos(p/2.), -np.sin(p/2.)], [np.sin(p/2.), np.cos(p/2)]], dims=[[2], [2]])

    def CNOT(self):
        """
        CNOT
        """
        return Qobj([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dims=[[2, 2], [2, 2]])

    def Kraus_dephase(self, p):
        """
        Single qubit dephasing Kraus operator

        Params
        ======

        p  : The probability of dephasing.
        """
        return np.sqrt(1-p)*qeye(2), np.sqrt(p)*sigmaz()

    def Kraus_amp_damp(self, p):
        """
        Single qubit amplitude dampening Kraus operator

        Params
        ======

        p  : The probability that a qubit decays from |1> to |0>
        """
        return Qobj([[1, 0], [0, np.sqrt(1-p)]], dims=[[2], [2]]), Qobj([[0, np.sqrt(p)], [0, 0]], dims=[[2], [2]])

    def generate_noise_mat(self):
        """
        Generate random unitaries depending on the success probability of the respective gates.
        Returns 5 random unitaries with the first four being single qubit unitaries while the
        5th is a 2 qubit unitary acting on the CNOT
        """
        rand_compare_arr_sqg = [random.random() for i in range(len(self._suc_sqg))]
        nmat = [qeye(2) for j in range(len(self._suc_sqg))]
        for j, r in enumerate(rand_compare_arr_sqg):
            if self._suc_sqg[j] < r:
                nmat[j] = Qobj(unitary_group.rvs(2), dims=[[2], [2]])
        rand_compare_arr_tqg = random.random()
        if self._suc_tqg[0] > rand_compare_arr_tqg:
            nmat.append(tensor(qeye(2), qeye(2)))
        else:
            nmat.append(Qobj(unitary_group.rvs(4), dims=[[2, 2], [2, 2]]))
        return nmat

    def overlap_gate_noise(self, phi_l):
        """
        Returns the overlap of the final state, generated after the action of unitaries with respect to the bell state.

        Params
        ======

        phi_l : The parameterized angles
        """
        nmat =  self.generate_noise_mat()
        param_state = tensor(nmat[2]*self.R_x(phi_l[2]), nmat[3]*self.R_y(phi_l[3]))*nmat[4]*self.CNOT()*tensor(nmat[0]*self.R_y(phi_l[0]), nmat[1]*self.R_x(phi_l[1]))*self._init_state
        ghz_2 = (ket("00") + ket("11")).unit()
        olap = -abs((param_state.dag()*ghz_2).data.todense()[0, 0])**2
        return olap

    def expect_gate_noise(self, phi_l):
        """
        Returns the expectation value of an operator which is equivalent to a Hadamard upto a complex factor.

        Params
        ======

        phi_l : The parameterized angles
        """
        nmat =  self.generate_noise_mat()
        param_state = tensor(nmat[2]*self.R_x(phi_l[2]), nmat[3]*self.R_y(phi_l[3]))*nmat[4]*self.CNOT()*tensor(nmat[0]*self.R_y(phi_l[0]), nmat[1]*self.R_x(phi_l[1]))*self._init_state
        h_gate = self.R_y(np.pi/2.)*self.R_x(-np.pi)*self.R_y(np.pi)
        return  np.real((param_state.dag()*tensor(h_gate, h_gate)*param_state).data.todense()[0,0])

    def noisy_state_noisy_gates_depth(self):
        """
        Returns the optimized parameterized angles, starting with a noisy state and coherent noise for unitary gates.
        """
        print(" init state : ")
        print(self._init_state)
        for d in range(self._depth):
            print(" depth : ", d + 1)
            if self._method == 'overlap':
                res = minimize(self.overlap_gate_noise, [0., 0., 0., 0.], method='Nelder-Mead', tol=1e-6)
                print(" Overlap : ", abs(res.fun))
                print(" phi's : ", res.x)
                print(" no of evaluations of the obj func : ", res.nfev)
                print("################")
                if abs(res.fun) > 0.9:
                    return abs(res.fun), res.x
                else:
                    self._init_state = tensor(self.R_x(res.x[2]), self.R_y(res.x[3]))*self.CNOT()*tensor(self.R_y(res.x[0]), self.R_x(res.x[1]))*self._init_state
            elif self._method == 'expect':
                res = minimize(self.expect_gate_noise, [0., 0., 0., 0.], method='Nelder-Mead', tol=1e-6)
                print(" Expectation : ", res.fun)
                print(" phi's : ", res.x)
                print(" no of evaluations of the obj func : ", res.nfev)
                print("################")
                if abs(res.fun) > 0.9:
                    return res.fun, res.x
                else:
                    self._init_state = tensor(self.R_x(res.x[2]), self.R_y(res.x[3]))*self.CNOT()*tensor(self.R_y(res.x[0]), self.R_x(res.x[1]))*self._init_state

    def gen_incoherent_noise_mc(self, phi_l, ntraj):
        """
        Returns the density matrix after the application of two qubit incoherent noise, i.e., either amplitude dampening
        or dephasing.

        Params
        ======
        phi_l : The parameterized angles

        ntraj : The number of trajectories used in the monte carlo solver to generate the density matrix
        """
        nmat =  self.generate_noise_mat()

        if self._incoherent_noise == 'amp_damp':
            kops1 = self.Kraus_amp_damp(self._noise_prob[0])
            kops2 = self.Kraus_amp_damp(self._noise_prob[1])
        elif self._incoherent_noise == 'dephase':
            kops1 = self.Kraus_dephase(self._noise_prob[0])
            kops2 = self.Kraus_dephase(self._noise_prob[1])

        kcombs = [tensor(i, j) for i in kops1 for j in kops2]
        U1 = tensor(nmat[0]*self.R_y(phi_l[0]), nmat[1]*self.R_x(phi_l[1]))
        state = self.CNOT()*U1*self._init_state

        # MC traj
        kraus_prob = [np.real((state.dag()*k_op.dag()*k_op*state).data.todense()[0, 0]) for k_op in kcombs]
        prob_sum = np.sum(kraus_prob)
        rho = 0*bra("00")*ket("00")
        for traj in range(ntraj):
            rand = random.random()
            prob_jump = 0.
            for j in range(4):
                prob_jump = prob_jump + kraus_prob[j]/prob_sum
                if prob_jump >= rand:
                    rho = rho + (kcombs[j]*state*state.dag()*kcombs[j].dag())/np.real(kraus_prob[j])
                    break

        rho = rho/float(ntraj)
        U2 = tensor(nmat[2]*self.R_x(phi_l[2]), nmat[3]*self.R_y(phi_l[3]))
        rho = U2*rho*U2.dag()
        # print(" trace : ", rho.tr())
        return rho

    def overlap_gen_noise_mc(self, phi_l, ntraj):
        """
        Returns the fidelity of the final state with respect to the Bell state. The final state generated
        considers the single and two qubit coherent noise along with two qubit incoherent noise (either
        dephasing or amplitude dampening).

        Params
        ======

        phi_l : The parameterized angles

        ntraj : Number of trajectories used in the monte carlo solver to generate rho

        """
        rho = self.gen_incoherent_noise_mc(phi_l, ntraj)
        ghz = (ket("00") + ket("11")).unit()
        ghz_rho = ghz.dag()*ghz
        ghz_rho_sq = ghz_rho.sqrtm()
        fid = -abs(((ghz_rho_sq*rho*ghz_rho_sq).sqrtm()).tr())**2
        return fid

    def expect_gen_noise_mc(self, phi_l, ntraj):
        """
        Returns the expectation value of an operator equivalent to a Hadamard upto a complex factor with resect to the final state.
        The final state generated considers the single and two qubit coherent noise along with two qubit incoherent noise (either
        dephasing or amplitude dampening).

        Params
        ======

        phi_l : The parameterized angles

        ntraj : Number of trajectories used in the monte carlo solver to generate rho

        """
        rho = self.gen_incoherent_noise_mc(phi_l, ntraj)
        h_gate = self.R_y(np.pi/2.)*self.R_x(-np.pi)*self.R_y(np.pi)
        return np.real(expect(tensor(h_gate, h_gate), rho))

    def gen_noisy_gates_mc(self, ntraj):
        """
        Returns the optimized parameterized angles using either the `overlap` or `expect` approaches
        """
        if self._method == 'overlap':
            res = minimize(self.overlap_gen_noise_mc, [0., 0., 0., 0.], args=(ntraj), method='Nelder-Mead', tol=1e-6)
            print(" Fidelity : ", abs(res.fun))
            print(" phi's : ", res.x)
            print("################")
            return abs(res.fun), res.x
        elif self._method == 'expect':
            res = minimize(self.expect_gen_noise_mc, [0., 0., 0., 0.], args=(ntraj), method='Nelder-Mead', tol=1e-6)
            print(" Expectation : ", res.fun)
            print(" phi's : ", res.x)
            print("################")
            return res.fun, res.x

class ResOpt:
    """
    Optimized Result class
    """
    def __init__(self):
        pass

    def overlap_no_noise(self):
        """
        Optimized angles via overlap method with no coherent and incoherent noise (perfect gates).
        """
        return Opt_Param(ket("00"), [1., 1., 1., 1.], [1.], None, None, 1, 'overlap').noisy_state_noisy_gates_depth()

    def expect_no_noise(self):
        """
        Optimized angles via expect method with no coherent and incoherent noise (perfect gates).
        """
        return Opt_Param(ket("00"), [1., 1., 1., 1.], [1.], None, None, 1, 'expect').noisy_state_noisy_gates_depth()

    def overlap_noisy_gate(self, single_qubit_gate_success_prob, two_qubit_gate_success_prob):
        """
        Optimized angles via overlap method with coherent noise and no incoherent noise with a single layer (depth = 1).

        Params
        ======

        single_qubit_gate_success_prob : success probabilities of the single qubit gates (Array with four values between 0. and 1.)

        two_qubit_gate_success_prob : success probability of two qubit gate (Array with one value between 0. and 1.)

        """
        return Opt_Param(ket("00"), single_qubit_gate_success_prob, two_qubit_gate_success_prob, None, None, 1, 'overlap').noisy_state_noisy_gates_depth()

    def expect_noisy_gate(self, single_qubit_gate_success_prob, two_qubit_gate_success_prob):
        """
        Optimized angles via expectation method with coherent noise and no incoherent noise with a single layer (depth = 1).

        Params
        ======

        single_qubit_gate_success_prob : success probabilities of the single qubit gates (Array with four values between 0. and 1.)

        two_qubit_gate_success_prob : success probability of two qubit gate (Array with one value between 0. and 1.)

        """
        return Opt_Param(ket("00"), single_qubit_gate_success_prob, two_qubit_gate_success_prob, None, None, 1, 'expect').noisy_state_noisy_gates_depth()

    def overlap_noisy_initial_state(self, init_state='rand', depth=5):
        """
        Optimized angles via overlap method with no coherent and no incoherent noise with a noisy initialization and depth set to 5.

        Params
        ======

        init_state : Random initial state, can be changed using the keyword `init_state`

        depth : Number of layers is set to 5, can we varied using the keyword `depth`

        """
        return Opt_Param(init_state, [1., 1., 1., 1.], [1.], None, None, depth, 'overlap').noisy_state_noisy_gates_depth()

    def expect_noisy_initial_state(self, init_state='rand', depth=5):
        """
        Optimized angles via expectation method with no coherent and no incoherent noise with a noisy initialization and depth set to 5.

        Params
        ======

        init_state : Random initial state, can be changed using the keyword `init_state`

        depth : Number of layers is set to 5, can we varied using the keyword `depth`

        """
        return Opt_Param(init_state, [1., 1., 1., 1.], [1.], None, None, depth, 'expect').noisy_state_noisy_gates_depth()

    def overlap_noisy_state_noisy_gates(self, single_qubit_gate_success_prob, two_qubit_gate_success_prob, init_state='rand', depth=5):
        """
        Optimized angles via overlap method with coherent noise and no incoherent noise with a noisy initialization and depth set to 5.

        Params
        ======

        single_qubit_gate_success_prob : success probabilities of the single qubit gates (Array with four values between 0. and 1.)

        two_qubit_gate_success_prob : success probability of two qubit gate (Array with one value between 0. and 1.)

        init_state : Random initial state, can be changed using the keyword `init_state`

        depth : Number of layers is set to 5, can we varied using the keyword `depth`

        """
        return Opt_Param(init_state, single_qubit_gate_success_prob, two_qubit_gate_success_prob, None, None, depth, 'overlap').noisy_state_noisy_gates_depth()

    def expect_noisy_state_noisy_gates(self, single_qubit_gate_success_prob, two_qubit_gate_success_prob, init_state='rand', depth=5):
        """
        Optimized angles via expectation method with coherent noise and no incoherent noise with a noisy initialization and depth set to 5.

        Params
        ======

        single_qubit_gate_success_prob : success probabilities of the single qubit gates (Array with four values between 0. and 1.)

        two_qubit_gate_success_prob : success probability of two qubit gate (Array with one value between 0. and 1.)

        init_state : Random initial state, can be changed using the keyword `init_state`

        depth : Number of layers is set to 5, can we varied using the keyword `depth`

        """
        return Opt_Param(init_state, single_qubit_gate_success_prob, two_qubit_gate_success_prob, None, None, depth, 'expect').noisy_state_noisy_gates_depth()

    ##################################
    def overlap_dephasing_noisy_gate_mc(self, single_qubit_gate_success_prob, two_qubit_gate_success_prob, dephase_probs, ntraj=100):
        """
        Optimized angles via overlap method with coherent and incoherent dephasing noise with a |00> initialization and depth 1.

        Params
        ======

        single_qubit_gate_success_prob : success probabilities of the single qubit gates (Array with four values between 0. and 1.)

        two_qubit_gate_success_prob : success probability of two qubit gate (Array with one value between 0. and 1.)

        dephase_probs : probability of dephasing in channel 1 and channel 2 (Array with two values between 0. and 1.)

        ntraj : Number of trajectories to generate rho, default is set to 100, can we varied with the keyword `ntraj`

        """
        return Opt_Param(ket("00"), single_qubit_gate_success_prob, two_qubit_gate_success_prob, 'dephase', dephase_probs, None, 'overlap').gen_noisy_gates_mc(ntraj)

    def expect_dephasing_noisy_gate_mc(self, single_qubit_gate_success_prob, two_qubit_gate_success_prob, dephase_probs, ntraj=100):
        """
        Optimized angles via expectation method with coherent and incoherent noise with a |00> initialization and depth 1.

        Params
        ======

        single_qubit_gate_success_prob : success probabilities of the single qubit gates (Array with four values between 0. and 1.)

        two_qubit_gate_success_prob : success probability of two qubit gate (Array with one value between 0. and 1.)

        dephase_probs : probability of dephasing in channel 1 and channel 2 (Array with two values between 0. and 1.)

        ntraj : Number of trajectories to generate rho, default is set to 100, can we varied with the keyword `ntraj`

        """
        return Opt_Param(ket("00"), single_qubit_gate_success_prob, two_qubit_gate_success_prob, 'dephase', dephase_probs, None, 'expect').gen_noisy_gates_mc(ntraj)

    def overlap_amp_damp_noisy_gate_mc(self, single_qubit_gate_success_prob, two_qubit_gate_success_prob, amp_damp_probs, ntraj=100):
        """
        Optimized angles via overlap method with coherent and incoherent amplitude dampening noise with a |00> initialization and depth 1.

        Params
        ======

        single_qubit_gate_success_prob : success probabilities of the single qubit gates (Array with four values between 0. and 1.)

        two_qubit_gate_success_prob : success probability of two qubit gate (Array with one value between 0. and 1.)

        dephase_probs : probability of decay in channel 1 and channel 2 (Array with two values  between 0. and 1., one for each channel)

        ntraj : Number of trajectories to generate rho, default is set to 100, can we varied with the keyword `ntraj`

        """
        return Opt_Param(ket("00"), single_qubit_gate_success_prob, two_qubit_gate_success_prob, 'amp_damp', amp_damp_probs, None, 'overlap').gen_noisy_gates_mc(ntraj)

    def expect_amp_damp_noisy_gate_mc(self, single_qubit_gate_success_prob, two_qubit_gate_success_prob, amp_damp_probs, ntraj=100):
        """
        Optimized angles via expectation method with coherent and incoherent amplitude dampening noise with a |00> initialization and depth 1.

        Params
        ======

        single_qubit_gate_success_prob : success probabilities of the single qubit gates (Array with four values between 0. and 1.)

        two_qubit_gate_success_prob : success probability of two qubit gate (Array with one value between 0. and 1.)

        dephase_probs : probability of decay in channel 1 and channel 2 (Array with two values  between 0. and 1., one for each channel)

        ntraj : Number of trajectories to generate rho, default is set to 100, can we varied with the keyword `ntraj`

        """
        return Opt_Param(ket("00"), single_qubit_gate_success_prob, two_qubit_gate_success_prob, 'amp_damp', amp_damp_probs, None, 'expect').gen_noisy_gates_mc(ntraj)
