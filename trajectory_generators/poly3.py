import numpy as np
from trajectory_generators.trajectory_generator import TrajectoryGenerator


class Poly3(TrajectoryGenerator):
    def __init__(self, start_q, desired_q, T):
        self.T = T
        self.q_0 = start_q
        self.q_k = desired_q
        """
        Please implement the formulas for a_0 till a_3 using self.q_0 and self.q_k
        Assume that the velocities at start and end are zero.
        """
        self.a_0 = self.q_0
        self.a_1 = 3 * self.q_0 + start_q
        self.a_2 = 3 * self.q_k  - desired_q
        self.a_3 = self.q_k

    def generate(self, t):
        """
        Implement trajectory generator for your manipulator.
        Positional trajectory should be a 3rd degree polynomial going from an initial state q_0 to desired state q_k.
        Remember to derive the first and second derivative of it also.
        Use following formula for the polynomial from the instruction.
        """
        normalized_t = t / self.T
        one_minus_t = 1.0 - normalized_t

        # Pozycja
        q = (self.a_0 * one_minus_t**3 +
            self.a_1 * normalized_t * one_minus_t**2 +
            self.a_2 * normalized_t**2 * one_minus_t +
            self.a_3 * normalized_t**3)

        # Prędkość
        q_dot = (3 * self.a_0 * (-one_minus_t**2) +
                self.a_1 * (3 * normalized_t**2 - 4 * normalized_t + 1) +
                self.a_2 * (2 * normalized_t - 3 * normalized_t**2) +
                3 * self.a_3 * normalized_t**2)

        # Przyspieszenie
        q_ddot = (6 * self.a_0 * one_minus_t +
                self.a_1 * (6 * normalized_t - 4) +
                self.a_2 * (2 - 6 * normalized_t) +
                6 * self.a_3 * normalized_t)

        return q, q_dot / self.T, q_ddot / self.T**2

