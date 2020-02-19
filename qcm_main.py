from qutip import *
import numpy as np
from scipy.optimize import minimize
from scipy.stats import unitary_group
import random

class Opt_Param():

    def __init__(self, init_state, suc_sqg, suc_tqg, incoherent_noise, noise_prob, depth, method):
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
        return Qobj([[np.cos(p/2.), 0.-1j*np.sin(p/2.)], [0.-1j*np.sin(p/2.), np.cos(p/2)]], dims=[[2], [2]])

    def R_y(self, p):
        return Qobj([[np.cos(p/2.), -np.sin(p/2.)], [np.sin(p/2.), np.cos(p/2)]], dims=[[2], [2]])

    def CNOT(self):
        return Qobj([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dims=[[2, 2], [2, 2]])

    def Kraus_dephase(self, p):
        return np.sqrt(1-p)*qeye(2), np.sqrt(p)*sigmaz()

    def Kraus_amp_damp(self, p):
        return Qobj([[1, 0], [0, np.sqrt(1-p)]], dims=[[2], [2]]), Qobj([[0, np.sqrt(p)], [0, 0]], dims=[[2], [2]])

    # def gen_ghz_gate_noise(phi_l, init_state, rand_fixed_arr_sqg, rand_fixed_arr_tqg):
    def generate_noise_mat(self):
        rand_compare_arr_sqg = [random.random() for i in range(len(self._suc_sqg))]
        nmat = [qeye(2) for j in range(len(self._suc_sqg))]
        for j, r in enumerate(rand_compare_arr_sqg):
            # if self._suc_sqg[j] >= r:
                # nmat.append(qeye(2))
            if self._suc_sqg[j] < r:
                # nmat.append(Qobj(unitary_group.rvs(2), dims=[[2], [2]]))
                nmat[j] = Qobj(unitary_group.rvs(2), dims=[[2], [2]])
        rand_compare_arr_tqg = random.random()
        if self._suc_tqg[0] > rand_compare_arr_tqg:
            nmat.append(tensor(qeye(2), qeye(2)))
        else:
            nmat.append(Qobj(unitary_group.rvs(4), dims=[[2, 2], [2, 2]]))
            # nmat[4] = Qobj(unitary_group.rvs(4), dims=[[2, 2], [2, 2]])
        return nmat

    def overlap_gate_noise(self, phi_l):
        nmat =  self.generate_noise_mat()
        param_state = tensor(nmat[2]*self.R_x(phi_l[2]), nmat[3]*self.R_y(phi_l[3]))*nmat[4]*self.CNOT()*tensor(nmat[0]*self.R_y(phi_l[0]), nmat[1]*self.R_x(phi_l[1]))*self._init_state
        ghz_2 = (ket("00") + ket("11")).unit()
        olap = -abs((param_state.dag()*ghz_2).data.todense()[0, 0])**2
        # print("phi's : ", phi_l, " overlap : ", olap)
        return olap

    def expect_gate_noise(self, phi_l):
        nmat =  self.generate_noise_mat()
        param_state = tensor(nmat[2]*self.R_x(phi_l[2]), nmat[3]*self.R_y(phi_l[3]))*nmat[4]*self.CNOT()*tensor(nmat[0]*self.R_y(phi_l[0]), nmat[1]*self.R_x(phi_l[1]))*self._init_state
        h_gate = self.R_y(np.pi/2.)*self.R_x(-np.pi)*self.R_y(np.pi)
        # return  -np.real((param_state.dag()*tensor(hadamard_transform(), hadamard_transform())*param_state).data.todense()[0,0])
        return  np.real((param_state.dag()*tensor(h_gate, h_gate)*param_state).data.todense()[0,0])

    def noisy_state_noisy_gates_depth(self):
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
        nmat =  self.generate_noise_mat()

        if self._incoherent_noise == 'amp_damp':
            kops1 = self.Kraus_amp_damp(self._noise_prob[0])
            kops2 = self.Kraus_amp_damp(self._noise_prob[1])
        elif self._incoherent_noise == 'dephase':
            kops1 = self.Kraus_amp_damp(self._noise_prob[0])
            kops2 = self.Kraus_amp_damp(self._noise_prob[1])

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
                    # print(kraus_prob, prob_jump, rand, j)
                    rho = rho + (kcombs[j]*state*state.dag()*kcombs[j].dag())/np.real(kraus_prob[j])
                    break

        rho = rho/float(ntraj)
        U2 = tensor(nmat[2]*self.R_x(phi_l[2]), nmat[3]*self.R_y(phi_l[3]))
        rho = U2*rho*U2.dag()
        print(" trace : ", rho.tr())
        return rho

    def overlap_gen_noise_mc(self, phi_l, ntraj):
        rho = self.gen_incoherent_noise_mc(phi_l, ntraj)
        # print(" trace : ", rho.tr())
        ghz = (ket("00") + ket("11")).unit()
        ghz_rho = ghz.dag()*ghz
        ghz_rho_sq = ghz_rho.sqrtm()
        fid = -abs(((ghz_rho_sq*rho*ghz_rho_sq).sqrtm()).tr())**2
        # print("phi's : ", phi_l, " fidelity : ", fid)
        return fid

    def expect_gen_noise_mc(self, phi_l, ntraj):
        rho = self.gen_incoherent_noise_mc(phi_l, ntraj)
        h_gate = self.R_y(np.pi/2.)*self.R_x(-np.pi)*self.R_y(np.pi)
        # print("phi's : ", phi_l, " overlap : ", olap)
        return np.real(expect(tensor(h_gate, h_gate), rho))

    def gen_noisy_gates_mc(self):
        if self._method == 'overlap':
            res = minimize(self.overlap_gen_noise_mc, [0., 0., 0., 0.], args=(500), method='Nelder-Mead', tol=1e-6)
            print(" Fidelity : ", abs(res.fun))
            print(" phi's : ", res.x)
            print("################")
            return abs(res.fun), res.x
        elif self._method == 'expect':
            res = minimize(self.expect_gen_noise_mc, [0., 0., 0., 0.], args=(500), method='Nelder-Mead', tol=1e-6)
            print(" Expectation : ", res.fun)
            print(" phi's : ", res.x)
            print("################")
            return res.fun, res.x

def overlap_no_noise():
    return Opt_Param(ket("00"), [1., 1., 1., 1.], [1.], None, 1, 'overlap').noisy_state_noisy_gates_depth()

def expect_no_noise():
    return Opt_Param(ket("00"), [1., 1., 1., 1.], [1.], None, 1, 'expect').noisy_state_noisy_gates_depth()

def overlap_noisy_gate(single_qubit_gate_success_prob, two_qubit_gate_success_prob):
    return Opt_Param(ket("00"), single_qubit_gate_success_prob, two_qubit_gate_success_prob, None, 1, 'overlap').noisy_state_noisy_gates_depth()

def expect_noisy_gate(single_qubit_gate_success_prob, two_qubit_gate_success_prob):
    return Opt_Param(ket("00"), single_qubit_gate_success_prob, two_qubit_gate_success_prob, None, 1, 'expect').noisy_state_noisy_gates_depth()

def overlap_noisy_initial_state(init_state='rand', depth=5):
    return Opt_Param(init_state, [1., 1., 1., 1.], [1.], None, depth, 'overlap').noisy_state_noisy_gates_depth()

def expect_noisy_initial_state(init_state='rand', depth=5):
    return Opt_Param(init_state, [1., 1., 1., 1.], [1.], None, depth, 'expect').noisy_state_noisy_gates_depth()

def overlap_noisy_state_noisy_gates(single_qubit_gate_success_prob, two_qubit_gate_success_prob, init_state='rand', depth=5):
    return Opt_Param(init_state, single_qubit_gate_success_prob, two_qubit_gate_success_prob, None, depth, 'overlap').noisy_state_noisy_gates_depth()

def expect_noisy_state_noisy_gates(single_qubit_gate_success_prob, two_qubit_gate_success_prob, init_state='rand', depth=5):
    return Opt_Param(init_state, single_qubit_gate_success_prob, two_qubit_gate_success_prob, None, depth, 'expect').noisy_state_noisy_gates_depth()


def overlap_dephasing_noisy_gate_mc(single_qubit_gate_success_prob, two_qubit_gate_success_prob, dephase_probs):
    return Opt_Param(ket("00"), single_qubit_gate_success_prob, two_qubit_gate_success_prob, 'dephase', dephase_probs, None, 'overlap').gen_noisy_gates_mc()

# print(overlap_dephasing_noisy_gate_mc([0.8, .7, .6, .5], [0.5], [0.2, 0.88]))
# print(overlap_dephasing_noisy_gate_mc([1., 1., 1., 1.], [1.], [1., 0.]))

def expect_dephasing_noisy_gate_mc(single_qubit_gate_success_prob, two_qubit_gate_success_prob, dephase_probs):
    return Opt_Param(ket("00"), single_qubit_gate_success_prob, two_qubit_gate_success_prob, 'dephase', dephase_probs, None, 'expect').gen_noisy_gates_mc()

# print(overlap_dephasing_noisy_gate_mc([0.8, .7, .6, .5], [0.5], [0.2, 0.88]))
# print(overlap_dephasing_noisy_gate_mc([1., 1., 1., 1.], [1.], [1., 0.]))

def overlap_amp_damp_noisy_gate_mc(single_qubit_gate_success_prob, two_qubit_gate_success_prob, amp_damp_probs):
    return Opt_Param(ket("00"), single_qubit_gate_success_prob, two_qubit_gate_success_prob, 'amp_damp', amp_damp_probs, None, 'overlap').gen_noisy_gates_mc()

# print(overlap_dephasing_noisy_gate_mc([0.8, .7, .6, .5], [0.5], [0.2, 0.88]))
# print(overlap_amp_damp_noisy_gate_mc([1., 1., 1., 1.], [1.], [1., 0.]))

def expect_amp_damp_noisy_gate_mc(single_qubit_gate_success_prob, two_qubit_gate_success_prob, amp_damp_probs):
    return Opt_Param(ket("00"), single_qubit_gate_success_prob, two_qubit_gate_success_prob, 'amp_damp', amp_damp_probs, None, 'expect').gen_noisy_gates_mc()

## print(expect_amp_damp_noisy_gate_mc([0.8, .7, .6, .5], [0.5], [0.2, 0.88]))
