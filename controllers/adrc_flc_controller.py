import numpy as np
from observers.eso import ESO
from .controller import Controller
from models.manipulator_model import ManiuplatorModel


class ADRFLController(Controller):
    def __init__(self, Tp, initial_state, Kp, Kd, poles):
        # Utwórz model wewnętrzny
        self.dynamic_model = ManiuplatorModel(Tp)

        # Przechowaj macierze sprzężenia zwrotnego
        self.Kp_matrix = Kp
        self.Kd_matrix = Kd

        # Zbuduj macierz wzmocnień obserwatora
        L_gain = np.zeros((6, 2))
        for axis in range(2):
            p = poles[axis]
            L_gain[0 + axis, axis] = 3 * p
            L_gain[2 + axis, axis] = 3 * p ** 2
            L_gain[4 + axis, axis] = p ** 3

        # Zainicjalizuj macierze systemu obserwatora
        A_obs = np.zeros((6, 6))
        for idx in range(2):
            A_obs[idx, idx + 2] = 1
            A_obs[idx + 2, idx + 4] = 1

        B_obs = np.zeros((6, 2))
        W_obs = np.zeros((2, 6))
        W_obs[0, 0] = 1
        W_obs[1, 1] = 1

        # Inicjalizacja obserwatora
        self.observer = ESO(A_obs, B_obs, W_obs, L_gain, initial_state, Tp)
        self._recalculate_dynamics(initial_state[:2], initial_state[2:])

    def _recalculate_dynamics(self, q, dq):
        # Oblicz macierze dynamiczne w nowym punkcie pracy
        full_state = np.concatenate((q, dq))
        M_mat = self.dynamic_model.M(full_state)
        C_mat = self.dynamic_model.C(full_state)

        # Tworzenie nowej postaci A i B
        new_A = np.zeros((6, 6))
        new_B = np.zeros((6, 2))

        new_A[0:2, 2:4] = np.eye(2)
        new_A[2:4, 4:6] = np.eye(2)
        new_A[2:4, 2:4] = -np.linalg.solve(M_mat, C_mat)

        new_B[2:4, :] = np.linalg.inv(M_mat)

        self.observer.A = new_A
        self.observer.B = new_B

    def calculate_control(self, system_state, desired_pos, desired_vel, desired_acc):
        # Uzyskaj aktualne dane z modelu
        M = self.dynamic_model.M(system_state)
        C = self.dynamic_model.C(system_state)

        q = system_state[:2]

        # Wydziel dane z obserwatora
        state_vector = self.observer.get_state()
        est_pos = state_vector[:2]
        est_vel = state_vector[2:4]
        dist = state_vector[4:]

        # Oblicz v według reguły sterowania
        v_ctrl = desired_acc + self.Kd_matrix @ (desired_vel - est_vel) + self.Kp_matrix @ (desired_pos - q)

        # Sterowanie końcowe
        u = M @ (v_ctrl - dist) + C @ est_vel

        # Aktualizacja obserwatora
        self._recalculate_dynamics(est_pos, est_vel)
        self.observer.update(q[:, None], u[:, None])

        return u
