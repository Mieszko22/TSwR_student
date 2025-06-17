

import numpy as np
from observers.eso import ESO
from .controller import Controller


class ADRCJointController(Controller):
    def __init__(self, b_estimate, kp_gain, kd_gain, observer_pole, initial_state, Tp):
        self.b = b_estimate
        self.kp = kp_gain
        self.kd = kd_gain
        self.u = 0

        # Macierze obserwatora stanu rozszerzonego (ESO)
        A_matrix = np.zeros((3, 3))
        A_matrix[0, 1] = 1
        A_matrix[1, 2] = 1

        B_matrix = np.zeros((3, 1))
        B_matrix[1, 0] = self.b

        observer_gain = np.array([
            [3 * observer_pole],
            [3 * observer_pole ** 2],
            [observer_pole ** 3]
        ])

        output_matrix = np.array([[1, 0, 0]])

        # Inicjalizacja obserwatora
        self.eso = ESO(A_matrix, B_matrix, output_matrix, observer_gain, initial_state, Tp)

    def set_b(self, new_b):
        self.b = new_b
        updated_B = np.zeros((3, 1))
        updated_B[1, 0] = new_b
        self.eso.set_B(updated_B)

    def calculate_control(self, joint_state, reference_pos, reference_vel, reference_acc):
        position = joint_state[0]
        velocity = joint_state[1]

        # Aktualny stan obserwowany
        estimate = self.eso.get_state()
        estimated_pos = estimate[0]
        estimated_vel = estimate[1]
        disturbance = estimate[2]

        # Aktualizacja obserwatora
        self.eso.update(position, self.u)

        # Obliczenie sygnału pomocniczego v
        error_pos = reference_pos - position
        error_vel = reference_vel - estimated_vel
        virtual_control = reference_acc + self.kd * error_vel + self.kp * error_pos

        # Obliczenie sterowania
        u = (virtual_control - disturbance) / self.b
        self.u = u
        return u
