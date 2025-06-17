import numpy as np
from models.manipulator_model import ManiuplatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        super().__init__()
        self.model = ManiuplatorModel(Tp)

        # Użytkownik może ustawić własne, -1 oznacza domyślne
        self.Kp = -1
        self.Kd = -1


    # def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        

    #     mass_matrix = self.model.M(x)
    #     coriolis_matrix = self.model.C(x)

    #     v = mass_matrix @ q_r_ddot + coriolis_matrix @ q_r_dot
    #     return v

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):

        q_vec = np.array(x[:2])
        q_dot_vec = np.array(x[2:])
        mass_matrix = self.model.M(x)
        coriolis_matrix = self.model.C(x)
        position_error = q_vec - q_r
        velocity_error = q_dot_vec - q_r_dot
        dynamics_comp = mass_matrix @ q_r_ddot + coriolis_matrix @ q_r_dot
        feedback_pd = self.Kd * velocity_error + self.Kp * position_error

        v = dynamics_comp + feedback_pd
        return v
