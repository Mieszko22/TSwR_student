import numpy as np
from .controller import Controller
from models.manipulator_model import ManiuplatorModel

class MMAController(Controller):
    def __init__(self, Tp):
        # Tworzenie trzech wariantów modelu z różnymi parametrami końcówki
        M1 = ManiuplatorModel(Tp)
        M1.m3 = 0.1
        M1.r3 = 0.05
        M1.I_3 = 2. / 5 * M1.m3 * M1.r3 ** 2

        M2 = ManiuplatorModel(Tp)
        M2.m3 = 0.01
        M2.r3 = 0.01
        M2.I_3 = 2. / 5 * M2.m3 * M2.r3 ** 2

        M3 = ManiuplatorModel(Tp)
        M3.m3 = 1.0
        M3.r3 = 0.3
        M3.I_3 = 2. / 5 * M3.m3 * M3.r3 ** 2

        self.models = [M1, M2, M3]
        self.i = 0
        self.prev_u = np.zeros((2, 1))  
        self.Tp = Tp



    def choose_model(self, state, previous_tau):
        joint_pos = state[:2]
        joint_vel = state[2:]
        candidate_errors = []

        for candidate in self.models:
            mass_matrix = candidate.M(state)
            coriolis_matrix = candidate.C(state)
            tau_input = previous_tau.reshape(-1)

            accel_estimate = np.linalg.solve(mass_matrix, tau_input - coriolis_matrix @ joint_vel)
            pos_estimate = joint_pos + candidate.Tp * joint_vel
            vel_estimate = joint_vel + candidate.Tp * accel_estimate
            predicted_state = np.hstack((pos_estimate, vel_estimate))

            deviation = np.linalg.norm(state - predicted_state)
            candidate_errors.append(deviation)

        best_index = np.argmin(candidate_errors)
        self.i = int(best_index)

    def calculate_control(self, current_state, desired_pos, desired_vel, desired_acc):
        P_gain = 25.5
        D_gain = 15.0

        self.choose_model(current_state, self.prev_u)

        current_pos = current_state[:2]
        current_vel = current_state[2:]

        position_error = desired_pos - current_pos
        velocity_error = desired_vel - current_vel
        reference_acc = desired_acc + D_gain * velocity_error + P_gain * position_error

        selected_model = self.models[self.i]
        inertia_matrix = selected_model.M(current_state)
        coriolis_effects = selected_model.C(current_state)

        u = inertia_matrix @ reference_acc[:, np.newaxis] + coriolis_effects @ current_vel[:, np.newaxis]
        self.prev_u = u
        return u
